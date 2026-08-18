"""CSV, report, QA, and simple figure exports for MHA-PM experiments."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_loader import HAZARD_ORDER, TIME_ORDER, ExperimentData
from .experiments import ExperimentOutputs


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def export_llm_experiment_framework(results_root: str | Path) -> None:
    source = Path(__file__).resolve().parent / "llm_experiment"
    destination = Path(results_root) / "llm_experiment"
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "request_schema.json",
        "request_gold_template.csv",
        "validate_request.py",
        "compare_prediction_to_gold.py",
        "run_optimizer_from_request.py",
    ):
        shutil.copy2(source / name, destination / name)


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_data_qa_report(data: ExperimentData, output_path: Path) -> None:
    lines = [
        "# Data QA Report",
        "",
        "## Files actually read",
        "",
        *[f"- {time}: `{path}`" for time, path in data.demand_files.items()],
        f"- Shelter candidates: `{data.candidate_file}`",
        f"- OD network-distance matrix: `{data.od_file}`",
        "",
        "## Dimensions",
        "",
        f"- Demand locations per time scenario: {', '.join(f'{time}={len(data.demands[time])}' for time in TIME_ORDER)}",
        f"- Candidate shelter sites: {len(data.candidates)}",
        f"- OD pairs: {len(data.od)} (expected {len(data.demands[TIME_ORDER[0]]) * len(data.candidates)})",
        f"- Feasible candidates: baseline={int(data.candidates['baseline'].sum())}, moderate={int(data.candidates['moderate'].sum())}, strict={int(data.candidates['strict'].sum())}",
        "",
        "## QA checks",
        "",
        "| Check | Status | Severity | Detail |",
        "|---|---:|---|---|",
    ]
    for check in data.qa_checks:
        lines.append(f"| {check['check']} | {'PASS' if check['passed'] else 'FAIL'} | {check['severity']} | {check['detail']} |")
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- The demand weight is a normalized mobility activity proxy, not actual population.",
            "- Hazard fields are used only for candidate feasibility `a_jh`; they are not interpreted as road damage.",
            "- OD reachability is used only as `r_ij` in the normative optimization model.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_optimization_qa_report(data: ExperimentData, outputs: ExperimentOutputs, output_path: Path) -> None:
    failed_data = [item for item in data.qa_checks if not item["passed"] and item["severity"] == "error"]
    failed_solution = outputs.solution_qa.loc[~outputs.solution_qa["passed"]]
    lines = [
        "# Optimization QA Report",
        "",
        f"- Data QA errors: {len(failed_data)}",
        f"- Solved scenarios: {len(outputs.summaries)}",
        f"- Scenario QA checks: {len(outputs.solution_qa)}",
        f"- Failed scenario QA checks: {len(failed_solution)}",
        f"- All solver statuses optimal: {bool((outputs.summaries['solver_status'] == 'OPTIMAL').all())}",
        f"- Maximum objective recomputation difference: {outputs.summaries['objective_recalculation_difference'].max():.3e}",
        f"- All E4 diagonal checks passed: {bool(outputs.e4_cross_time.loc[outputs.e4_cross_time['optimized_time'] == outputs.e4_cross_time['evaluation_time'], 'diagonal_matches_optimum'].all())}",
        "",
        "## Failed checks",
        "",
    ]
    if failed_solution.empty and not failed_data:
        lines.append("None.")
    else:
        for item in failed_data:
            lines.append(f"- DATA: {item['check']} — {item['detail']}")
        for row in failed_solution.itertuples():
            lines.append(f"- {row.scenario_id}: {row.check} — {row.detail}")
    lines.extend(
        [
            "",
            "## Determinism",
            "",
            "All instances use Gurobi with `Threads=1`, `Seed=0`, and `ConcurrentMIP=1`. No stochastic search, GA, RL, HRL, or LLM-based site selection is used.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _simple_figures(outputs: ExperimentOutputs, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(6, 4))
    e1 = outputs.e1_temporal.set_index("time_scenario").loc[list(TIME_ORDER)]
    ax.bar(e1.index, e1["weighted_mean_distance"])
    ax.set_ylabel("Weighted mean network distance (m)")
    ax.set_title("E1 Temporal Distance")
    fig.tight_layout()
    fig.savefig(figure_dir / "E1_temporal_distance.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for time in TIME_ORDER:
        subset = outputs.e2_hcc.loc[outputs.e2_hcc["time_scenario"] == time].set_index("hazard_scenario").loc[list(HAZARD_ORDER)]
        ax.plot(list(HAZARD_ORDER), subset["HCC_percent"], marker="o", label=time)
    ax.set_ylabel("HCC (%)")
    ax.set_title("E2 Hazard Constraint Cost")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "E2_HCC.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for (time, hazard), subset in outputs.e3_p_sensitivity.groupby(["time_scenario", "hazard_scenario"], sort=False):
        subset = subset.sort_values("p")
        ax.plot(subset["p"], subset["weighted_mean_distance"], marker="o", label=f"{time}-{hazard}")
    ax.set_xlabel("p")
    ax.set_ylabel("Weighted mean network distance (m)")
    ax.set_title("E3 Facility Number Sensitivity")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(figure_dir / "E3_p_sensitivity.png", dpi=200)
    plt.close(fig)

    baseline = outputs.e4_tmp.loc[outputs.e4_tmp["hazard_scenario"] == "baseline"]
    matrix = baseline.pivot(index="optimized_time", columns="evaluation_time", values="TMP_percent").loc[list(TIME_ORDER), list(TIME_ORDER)]
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix.to_numpy(), aspect="auto")
    ax.set_xticks(range(3), matrix.columns)
    ax.set_yticks(range(3), matrix.index)
    ax.set_xlabel("Evaluation time")
    ax.set_ylabel("Optimized time")
    ax.set_title("E4 TMP Heatmap — Baseline (%)")
    for row in range(3):
        for col in range(3):
            ax.text(col, row, f"{matrix.iloc[row, col]:.1f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="TMP (%)")
    fig.tight_layout()
    fig.savefig(figure_dir / "E4_TMP_heatmap.png", dpi=200)
    plt.close(fig)


def export_results(data: ExperimentData, outputs: ExperimentOutputs, results_root: str | Path) -> None:
    root = Path(results_root)
    e1 = root / "E1_temporal"
    e2 = root / "E2_hazard"
    e3 = root / "E3_p_sensitivity"
    e4 = root / "E4_cross_time"
    figures = root / "figures"
    qa = root / "qa"
    verification = root / "solver_verification"

    _write_csv(outputs.e1_temporal, e1 / "experiment_E1_temporal.csv")
    _write_csv(outputs.e1_selected, e1 / "E1_selected_shelters.csv")
    _write_csv(outputs.e1_assignments, e1 / "E1_assignments.csv")
    _write_csv(outputs.e1_tcc_matrix, e1 / "E1_TCC_matrix.csv")
    _write_csv(outputs.e1_overlap, figures / "plot_shelter_overlap.csv")
    _write_csv(outputs.e1_temporal[["time_scenario", "objective", "weighted_mean_distance", "weighted_p90_distance", "max_distance"]], figures / "plot_temporal_objective.csv")

    _write_csv(outputs.e2_hazard, e2 / "experiment_E2_hazard.csv")
    _write_csv(outputs.e2_selected, e2 / "E2_selected_shelters.csv")
    _write_csv(outputs.e2_hcc, e2 / "E2_HCC.csv")
    _write_csv(outputs.e2_hcc, figures / "plot_hazard_cost.csv")

    _write_csv(outputs.e3_p_sensitivity, e3 / "experiment_E3_p_sensitivity.csv")
    _write_csv(outputs.e3_p_sensitivity[["time_scenario", "hazard_scenario", "p", "weighted_mean_distance", "weighted_p90_distance", "max_distance", "runtime_seconds"]], figures / "plot_p_sensitivity.csv")

    _write_csv(outputs.e4_cross_time, e4 / "E4_cross_time_cost_matrix.csv")
    _write_csv(outputs.e4_tmp, e4 / "E4_TMP_matrix.csv")
    _write_csv(outputs.e4_tmp, figures / "plot_TMP_heatmap.csv")

    _write_csv(outputs.summaries, root / "all_scenario_summaries.csv")
    _write_csv(outputs.selected_shelters, root / "all_selected_shelters.csv")
    _write_csv(outputs.assignments, root / "all_assignments.csv")
    _write_csv(outputs.solver_verification, verification / "solver_verification.csv")
    _write_csv(outputs.solution_qa, qa / "scenario_QA_checks.csv")
    (root / "schema_mapping.json").write_text(json.dumps(data.schema_mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    write_data_qa_report(data, root / "data_QA_report.md")
    write_optimization_qa_report(data, outputs, qa / "optimization_QA_report.md")
    _simple_figures(outputs, figures)
    export_llm_experiment_framework(root)


def write_experiment_report(data: ExperimentData, outputs: ExperimentOutputs, output_path: Path) -> None:
    e1 = outputs.e1_temporal[["time_scenario", "objective", "weighted_p90_distance", "max_distance", "solver_status"]]
    e2 = outputs.e2_hazard[["time_scenario", "hazard_scenario", "objective", "feasible_candidate_count", "HCC_percent"]]
    e3_mb = outputs.e3_p_sensitivity.loc[outputs.e3_p_sensitivity["p"].isin([10, 15]), ["time_scenario", "hazard_scenario", "p", "MB_5_10", "MB_10_15"]]
    lines = [
        "# LLMforShelter MHA-PM Experiment Report",
        "",
        "This report records experiment outputs and verifiable checks only. It does not provide manuscript conclusions.",
        "",
        "## Files actually read",
        "",
        *[f"- {time}: `{path}`" for time, path in data.demand_files.items()],
        f"- Candidates: `{data.candidate_file}`",
        f"- OD network distance: `{data.od_file}`",
        "",
        "## Data dimensions and mathematical mapping",
        "",
        f"- Demand locations: {len(data.demands['Morning'])} per time scenario.",
        f"- Candidate shelter sites: {len(data.candidates)}.",
        f"- OD pairs: {len(data.od)}.",
        "- Demand activity intensity is normalized within each time scenario to obtain `w_it`, with sum exactly 1 within numerical tolerance.",
        "- OD road-network distance maps to `d_ij`; OD reachability maps to `r_ij`.",
        "- Candidate feasibility is baseline=all candidates, moderate=`safe_z12`, strict=`safe_z123`, mapping only to `a_jh`.",
        "- Full original-column mapping is stored in `schema_mapping.json`; model descriptions use mathematical variables rather than database fields.",
        "",
        "## Experiment execution status",
        "",
        f"- E1 completed: {len(outputs.e1_temporal)} instances.",
        f"- E2 completed: {len(outputs.e2_hazard)} instances.",
        f"- E3 completed: {len(outputs.e3_p_sensitivity)} instances.",
        f"- E4 completed: {len(outputs.e4_cross_time)} fixed-set evaluations.",
        f"- Gurobi statuses: {outputs.summaries['solver_status'].value_counts().to_dict()}.",
        "- HiGHS verification was not run because the current environment does not provide highspy or SciPy/HiGHS.",
        "",
        "## E1 principal result table",
        "",
        _markdown_table(e1),
        "",
        "## E2 principal result table",
        "",
        _markdown_table(e2),
        "",
        "## E3 marginal accessibility benefit table",
        "",
        _markdown_table(e3_mb),
        "",
        "## QA results",
        "",
        f"- Data QA errors: {sum(not item['passed'] and item['severity'] == 'error' for item in data.qa_checks)}.",
        f"- Scenario QA failures: {int((~outputs.solution_qa['passed']).sum())}.",
        f"- Maximum model-vs-recalculated objective difference: {outputs.summaries['objective_recalculation_difference'].max():.3e}.",
        f"- E4 diagonal equality checks all passed: {bool(outputs.e4_cross_time.loc[outputs.e4_cross_time['optimized_time'] == outputs.e4_cross_time['evaluation_time'], 'diagonal_matches_optimum'].all())}.",
        "",
        "## Anomalies and items requiring human confirmation",
        "",
        f"- Unreachable OD pairs: {int((data.od['reachable'] == 0).sum())}.",
        "- Confirm that `safe_z12` and `safe_z123` are the intended moderate and strict candidate-exclusion scenarios; this mapping follows field contents and nesting but remains a study-design interpretation requiring author confirmation.",
        "- Confirm that `mobility_i` is the intended mobility activity proxy for `w_it`; it is not interpreted as actual population.",
        "- HiGHS cross-solver verification remains pending until a supported HiGHS environment is available.",
        "- Assignment outputs are normative model assignments, not observed evacuee behavior.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
