"""Gold-aware evaluation. Run only after frozen deterministic gate execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIGS = ("L1", "L2", "L3-Final")
FIELDS = ("time_scenario", "hazard_scenario", "p", "objective")


def clean(value: Any) -> Any:
    if pd.isna(value) or value == "":
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            number = float(value)
            return int(number) if number.is_integer() else number
        except ValueError:
            return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def truth(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def payload_from_row(row: pd.Series, prefix: str = "predicted_") -> dict[str, Any]:
    issues = []
    try:
        issues = json.loads(str(row.get(f"{prefix}issues", "[]")))
    except json.JSONDecodeError:
        pass
    return {
        "status": clean(row.get(f"{prefix}status")),
        "time_scenario": clean(row.get(f"{prefix}time_scenario")),
        "hazard_scenario": clean(row.get(f"{prefix}hazard_scenario")),
        "p": clean(row.get(f"{prefix}p")),
        "objective": clean(row.get(f"{prefix}objective")),
        "issues": issues,
    }


def gold_payload(row: pd.Series) -> dict[str, Any]:
    issue = clean(row["gold_issue"])
    return {
        "status": row["gold_status"],
        "time_scenario": clean(row["gold_time_scenario"]),
        "hazard_scenario": clean(row["gold_hazard_scenario"]),
        "p": clean(row["gold_p"]),
        "objective": clean(row["gold_objective"]),
        "issues": [] if issue is None else [issue],
    }


def bootstrap(values, seed: int, repetitions: int = 1000) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(repetitions, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def category(row: pd.Series) -> str:
    if row["gold_status"] != "valid" and row["solver_called"]:
        return "INVALID_REQUEST_ACCEPTED"
    if row["gold_status"] == "valid" and not row["solver_called"]:
        return "VALID_REQUEST_BLOCKED"
    issues = json.loads(row["validator_issues"]) if isinstance(row["validator_issues"], str) else []
    order = [
        "SCHEMA_ERROR", "TIME_CONFLICT", "TIME_AMBIGUOUS", "TIME_PREDICTION_CONFLICT",
        "HAZARD_CONFLICT", "HAZARD_AMBIGUOUS", "HAZARD_PREDICTION_CONFLICT",
        "P_MISSING", "P_INVALID", "P_PREDICTION_CONFLICT", "UNSUPPORTED_OBJECTIVE",
        "DIRECT_LLM_LOCATION_REQUEST",
    ]
    return next((name for name in order if name in issues), "OTHER")


def evaluate(workspace_root: Path) -> Path:
    final_root = workspace_root / "llm_experiment_final"
    old_root = workspace_root / "results" / "llm_experiment_E5"
    results_dir = final_root / "results"
    figures_dir = final_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Gold is loaded here, after the gate result already exists and its validator is frozen.
    gold = pd.read_csv(old_root / "benchmark" / "request_gold.csv", dtype={"gold_p": "Int64"})
    old_predictions = pd.read_csv(old_root / "results" / "llm_predictions.csv")
    gate = pd.read_csv(results_dir / "E5_final_predictions.csv")
    sys.path.insert(0, str(workspace_root / "code"))
    from llm_for_shelter.llm_experiment.src.optimizer_adapter import OptimizerAdapter
    adapter = OptimizerAdapter(workspace_root)

    records: list[dict[str, Any]] = []
    for config in ("L1", "L2"):
        subset = old_predictions[old_predictions["configuration"].eq(config)]
        for row in subset.to_dict("records"):
            payload = payload_from_row(pd.Series(row))
            allowed = truth(row["final_schema_valid"]) and truth(row["final_valid"]) and payload["status"] == "valid"
            records.append({
                "request_id": row["request_id"], "request_type": row["request_type"], "configuration": config,
                "predicted_status": payload["status"], "predicted_time_scenario": payload["time_scenario"],
                "predicted_hazard_scenario": payload["hazard_scenario"], "predicted_p": payload["p"],
                "predicted_objective": payload["objective"], "schema_valid": truth(row["final_schema_valid"]),
                "solver_allowed": allowed, "validator_issues": "[]", "solver_block_reason": "" if allowed else "PARSER_OR_SCHEMA_BLOCK",
            })
    for row in gate.to_dict("records"):
        records.append({
            "request_id": row["request_id"], "request_type": row["request_type"], "configuration": "L3-Final",
            "predicted_status": row["final_status"], "predicted_time_scenario": clean(row["L2_time"]),
            "predicted_hazard_scenario": clean(row["L2_hazard"]), "predicted_p": clean(row["L2_p"]),
            "predicted_objective": clean(row["L2_objective"]), "schema_valid": truth(row["schema_valid"]),
            "solver_allowed": truth(row["solver_allowed"]), "validator_issues": row["validator_issues"],
            "solver_block_reason": clean(row["solver_block_reason"]) or "",
        })
    frame = pd.DataFrame(records).merge(gold, on=["request_id", "request_type"], validate="many_to_one")
    for field in FIELDS:
        frame[f"{field}_correct"] = [
            clean(pred) == clean(expected)
            for pred, expected in zip(frame[f"predicted_{field}"], frame[f"gold_{field}"])
        ]
    frame["parameter_match"] = frame[[f"{f}_correct" for f in FIELDS]].all(axis=1)
    frame["gold_valid"] = frame["gold_status"].eq("valid")
    frame["exact_match"] = frame["gold_valid"] & frame["solver_allowed"] & frame["predicted_status"].eq("valid") & frame["parameter_match"]
    frame["detection_correct"] = frame["predicted_status"].eq(frame["gold_status"])

    solver_called, objectives, shelter_sets = [], [], []
    for row in frame.to_dict("records"):
        payload = {
            "status": "valid", "time_scenario": clean(row["predicted_time_scenario"]),
            "hazard_scenario": clean(row["predicted_hazard_scenario"]), "p": clean(row["predicted_p"]),
            "objective": clean(row["predicted_objective"]), "issues": [],
        }
        result = adapter.execute(payload) if row["solver_allowed"] else {"solver_called": False, "solution": None}
        solution = result["solution"]
        solver_called.append(bool(result["solver_called"]))
        objectives.append(np.nan if solution is None else solution["objective"])
        shelter_sets.append("" if solution is None else ";".join(solution["selected_shelters"]))
    frame["solver_called"] = solver_called
    frame["objective_if_executed"] = objectives
    frame["selected_shelters_if_executed"] = shelter_sets

    gold_solution_cache: dict[tuple[str, str, int], dict[str, Any]] = {}
    end_rows = []
    for row in frame[frame["gold_valid"]].to_dict("records"):
        key = (row["gold_time_scenario"], row["gold_hazard_scenario"], int(row["gold_p"]))
        if key not in gold_solution_cache:
            gold_solution_cache[key] = adapter.execute(gold_payload(pd.Series(row)))["solution"]
        gold_solution = gold_solution_cache[key]
        objective_match = row["solver_called"] and abs(row["objective_if_executed"] - gold_solution["objective"]) <= 1e-7
        set_match = row["solver_called"] and set(str(row["selected_shelters_if_executed"]).split(";")) == set(gold_solution["selected_shelters"])
        end_rows.append({
            "request_id": row["request_id"], "request_type": row["request_type"], "configuration": row["configuration"],
            "parameter_match": bool(row["parameter_match"]), "solver_called": bool(row["solver_called"]),
            "gold_objective": gold_solution["objective"], "predicted_objective": row["objective_if_executed"],
            "objective_match": bool(objective_match), "shelter_set_match": bool(set_match),
            "alternative_optimum_possible": bool(objective_match and not set_match),
            "end_to_end_solution_match": bool(row["parameter_match"] and objective_match and set_match),
        })
    end = pd.DataFrame(end_rows)

    metric_rows = []
    for config, group in frame.groupby("configuration", sort=False):
        valid = group[group["gold_valid"]]
        nonvalid = group[~group["gold_valid"]]
        e2e = end[end["configuration"].eq(config)]
        arrays = {
            "schema_valid_rate": group["schema_valid"],
            "exact_match_rate": valid["exact_match"],
            "valid_executable_rate": valid["solver_called"],
            "detection_accuracy": nonvalid["detection_correct"],
            "false_acceptance_rate": nonvalid["solver_called"],
            "parameter_match_rate": valid["parameter_match"],
            "objective_match_rate": e2e["objective_match"],
            "shelter_set_match_rate": e2e["shelter_set_match"],
            "end_to_end_solution_match_rate": e2e["end_to_end_solution_match"],
        }
        row = {"configuration": config, "n_requests": len(group)}
        for index, (metric, values) in enumerate(arrays.items()):
            row[metric] = values.mean()
            if metric in {"exact_match_rate", "valid_executable_rate", "detection_accuracy", "false_acceptance_rate", "end_to_end_solution_match_rate"}:
                low, high = bootstrap(values, 20260812 + index)
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)

    l2 = frame[frame["configuration"].eq("L2")].set_index("request_id")
    l3 = frame[frame["configuration"].eq("L3-Final")].set_index("request_id")
    correct_l2_valid = l2["gold_valid"] & l2["exact_match"]
    retained = correct_l2_valid & l3["solver_called"]
    valid_retention = retained.sum() / correct_l2_valid.sum() if correct_l2_valid.sum() else np.nan
    unnecessary_block = (l3["gold_valid"] & ~l3["solver_called"]).sum() / l3["gold_valid"].sum()
    vr_low, vr_high = bootstrap(l3.loc[correct_l2_valid, "solver_called"], 20260830)
    ub_low, ub_high = bootstrap(~l3.loc[l3["gold_valid"], "solver_called"], 20260831)
    metrics["valid_retention_rate"] = np.nan
    metrics["valid_retention_rate_ci_low"] = np.nan
    metrics["valid_retention_rate_ci_high"] = np.nan
    metrics["unnecessary_block_rate"] = np.nan
    metrics["unnecessary_block_rate_ci_low"] = np.nan
    metrics["unnecessary_block_rate_ci_high"] = np.nan
    for config in ("L2", "L3-Final"):
        metrics.loc[metrics["configuration"].eq(config), ["valid_retention_rate", "valid_retention_rate_ci_low", "valid_retention_rate_ci_high"]] = [valid_retention, vr_low, vr_high]
    metrics.loc[metrics["configuration"].eq("L3-Final"), ["unnecessary_block_rate", "unnecessary_block_rate_ci_low", "unnecessary_block_rate_ci_high"]] = [unnecessary_block, ub_low, ub_high]
    far_l2 = float(metrics.loc[metrics["configuration"].eq("L2"), "false_acceptance_rate"].iloc[0])
    far_l3 = float(metrics.loc[metrics["configuration"].eq("L3-Final"), "false_acceptance_rate"].iloc[0])
    far_analysis = pd.DataFrame([{
        "FAR_L2": far_l2, "FAR_L3_Final": far_l3, "absolute_reduction": far_l2 - far_l3,
        "relative_reduction": (far_l2 - far_l3) / far_l2 if far_l2 else np.nan,
    }])

    by_type_rows = []
    for (config, request_type), group in frame.groupby(["configuration", "request_type"], sort=False):
        valid = group[group["gold_valid"]]
        nonvalid = group[~group["gold_valid"]]
        e2e = end[(end["configuration"].eq(config)) & (end["request_type"].eq(request_type))]
        by_type_rows.append({
            "configuration": config, "request_type": request_type, "n": len(group),
            "schema_valid_rate": group["schema_valid"].mean(),
            "exact_match_rate": valid["exact_match"].mean() if len(valid) else np.nan,
            "valid_executable_rate": valid["solver_called"].mean() if len(valid) else np.nan,
            "detection_accuracy": nonvalid["detection_correct"].mean() if len(nonvalid) else np.nan,
            "false_acceptance_rate": nonvalid["solver_called"].mean() if len(nonvalid) else np.nan,
            "end_to_end_solution_match_rate": e2e["end_to_end_solution_match"].mean() if len(e2e) else np.nan,
        })
    by_type = pd.DataFrame(by_type_rows)

    paired = pd.DataFrame({
        "request_id": l2.index,
        "L2_correct": l2["exact_match"].values,
        "L3_correct": l3.loc[l2.index, "exact_match"].values,
        "L2_solver_allowed": l2["solver_called"].values,
        "L3_solver_allowed": l3.loc[l2.index, "solver_called"].values,
    })
    gate_eval = gate.merge(gold[["request_id", "gold_status"]], on="request_id", validate="one_to_one")
    gate_eval["solver_called"] = frame[frame["configuration"].eq("L3-Final")].set_index("request_id").loc[gate_eval["request_id"], "solver_called"].values
    gate_eval["error_category"] = gate_eval.apply(category, axis=1)
    errors = gate_eval[(gate_eval["gold_status"].eq("valid") & ~gate_eval["solver_called"]) | (~gate_eval["gold_status"].eq("valid") & gate_eval["solver_called"]) | gate_eval["validator_issues"].ne("[]")].copy()

    l3_solution = frame[frame["configuration"].eq("L3-Final")].set_index("request_id")
    gate["solver_called"] = l3_solution.loc[gate["request_id"], "solver_called"].values
    gate["objective_if_executed"] = l3_solution.loc[gate["request_id"], "objective_if_executed"].values
    e3 = end[end["configuration"].eq("L3-Final")].set_index("request_id")
    gate["selected_solution_match"] = [e3.loc[r, "shelter_set_match"] if r in e3.index else np.nan for r in gate["request_id"]]
    gate.to_csv(results_dir / "E5_final_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(results_dir / "E5_final_metrics.csv", index=False, encoding="utf-8-sig")
    by_type.to_csv(results_dir / "metrics_by_type.csv", index=False, encoding="utf-8-sig")
    gate_eval.to_csv(results_dir / "validation_analysis.csv", index=False, encoding="utf-8-sig")
    end.to_csv(results_dir / "end_to_end_results.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(results_dir / "paired_L2_L3.csv", index=False, encoding="utf-8-sig")
    errors.to_csv(results_dir / "E5_final_error_cases.csv", index=False, encoding="utf-8-sig")
    far_analysis.to_csv(results_dir / "false_acceptance_reduction.csv", index=False, encoding="utf-8-sig")

    plot = metrics[["configuration", "schema_valid_rate", "exact_match_rate", "valid_executable_rate", "detection_accuracy", "false_acceptance_rate", "end_to_end_solution_match_rate", "valid_retention_rate", "unnecessary_block_rate"]]
    plot.to_csv(figures_dir / "plot_E5_final_metrics.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    chart = plot.set_index("configuration")[["exact_match_rate", "valid_executable_rate", "false_acceptance_rate", "end_to_end_solution_match_rate"]]
    chart.plot(kind="bar", ax=ax, color=["#555555", "#777777", "#999999", "#BBBBBB"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_xlabel("Configuration")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "E5_final_main_metrics.png", dpi=200)
    plt.close(fig)
    return results_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", default=".")
    args = parser.parse_args()
    print(evaluate(Path(args.workspace_root).resolve()))
