"""Verify the minimal release without re-solving any optimization model."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest() -> int:
    manifest = ROOT / "FILE_MANIFEST_SHA256.csv"
    require(manifest.exists(), "FILE_MANIFEST_SHA256.csv is missing")
    rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig", newline="")))
    for row in rows:
        path = ROOT / row["path"]
        require(path.is_file(), f"Manifest file missing: {row['path']}")
        require(path.stat().st_size == int(row["bytes"]), f"Size mismatch: {row['path']}")
        require(sha256(path) == row["sha256"], f"SHA-256 mismatch: {row['path']}")
    return len(rows)


def verify_data() -> dict[str, int]:
    data = ROOT / "data" / "processed"
    demand_paths = [
        data / "02_Demand_Points_Morning_Peak.csv",
        data / "03_Demand_Points_Evening_Peak.csv",
        data / "04_Demand_Points_Night.csv",
    ]
    demands = [pd.read_csv(path, dtype={"demand_id": str}) for path in demand_paths]
    for frame in demands:
        require(len(frame) == 61, "Each demand table must contain 61 rows")
        require(frame["demand_id"].is_unique, "Demand IDs must be unique within time")
        require(frame["mobility_i"].notna().all(), "Mobility activity contains missing values")
        require((frame["mobility_i"] >= 0).all(), "Mobility activity must be nonnegative")
        weights = frame["mobility_i"] / frame["mobility_i"].sum()
        require(math.isclose(float(weights.sum()), 1.0, abs_tol=1e-12), "Weights do not sum to one")
    id_sets = [set(frame["demand_id"]) for frame in demands]
    require(id_sets[0] == id_sets[1] == id_sets[2], "Demand ID sets differ across time")

    candidates = pd.read_csv(data / "01_Shelter_Candidates_Hazard_438.csv", dtype={"candidate_id": str})
    require(len(candidates) == 438 and candidates["candidate_id"].is_unique, "Candidate QA failed")
    require(candidates[["safe_z12", "safe_z123"]].isin([0, 1]).all().all(), "Feasibility is not binary")
    require((candidates["safe_z123"] <= candidates["safe_z12"]).all(), "Feasibility sets are not nested")
    require(int(candidates["safe_z12"].sum()) == 394, "Moderate feasible count must be 394")
    require(int(candidates["safe_z123"].sum()) == 379, "Strict feasible count must be 379")

    od = pd.read_csv(data / "od" / "OD.csv", dtype={"demand_id": str, "candidate_id": str})
    require(len(od) == 61 * 438, "OD table must contain 26,718 rows")
    require(not od.duplicated(["demand_id", "candidate_id"]).any(), "Duplicate OD pair")
    require((od["network_distance_m"] >= 0).all(), "Negative network distance")
    require(od["reachable"].eq(1).all(), "Released OD table contains an unreachable pair")
    require(set(od["demand_id"]) == id_sets[0], "OD demand IDs mismatch")
    require(set(od["candidate_id"]) == set(candidates["candidate_id"]), "OD candidate IDs mismatch")
    return {"demand_locations": 61, "candidates": 438, "od_pairs": len(od)}


def verify_results() -> dict[str, float]:
    results = ROOT / "results"
    summaries = pd.read_csv(results / "all_scenario_summaries.csv")
    selected = pd.read_csv(results / "all_selected_shelters.csv", dtype={"selected_shelter_id": str})
    assignments = pd.read_csv(results / "all_assignments.csv", dtype={"demand_id": str, "selected_shelter_id": str})
    require(len(summaries) == 27, "Expected 27 scenario summaries")
    require(summaries["scenario_id"].is_unique, "Scenario IDs are not unique")
    require(summaries["solver_status"].eq("OPTIMAL").all(), "A cached scenario is not OPTIMAL")

    selected_counts = selected.groupby("scenario_id")["selected_shelter_id"].nunique().to_dict()
    assignment_groups = assignments.groupby("scenario_id", sort=False)
    maximum_difference = 0.0
    for row in summaries.itertuples(index=False):
        require(selected_counts[row.scenario_id] == int(row.p), f"Selected count mismatch: {row.scenario_id}")
        group = assignment_groups.get_group(row.scenario_id)
        require(len(group) == 61 and group["demand_id"].nunique() == 61, f"Assignment QA failed: {row.scenario_id}")
        chosen = set(selected.loc[selected["scenario_id"] == row.scenario_id, "selected_shelter_id"])
        require(set(group["selected_shelter_id"]).issubset(chosen), f"Assignment to unselected site: {row.scenario_id}")
        recalculated = float((group["weight"] * group["assigned_distance"]).sum())
        difference = abs(recalculated - float(row.objective))
        maximum_difference = max(maximum_difference, difference)
        require(difference <= 1e-9, f"Objective mismatch: {row.scenario_id} ({difference})")
    return {"scenarios": len(summaries), "max_objective_difference": maximum_difference}


def read_key_values(path: Path) -> dict[str, str]:
    return dict(line.strip().split("=", 1) for line in path.read_text(encoding="utf-8").splitlines() if "=" in line)


def verify_llm_experiment() -> dict[str, int]:
    benchmark_path = ROOT / "results" / "llm_experiment_E5" / "benchmark" / "request_gold.csv"
    benchmark = pd.read_csv(benchmark_path, dtype={"request_id": str})
    require(len(benchmark) == 60 and benchmark["request_id"].is_unique, "Benchmark must contain 60 unique requests")
    require(Counter(benchmark["request_type"]) == Counter({"simple": 20, "composite": 20, "ambiguous_invalid": 20}), "Request-type composition mismatch")
    require(Counter(benchmark["gold_status"]) == Counter({"valid": 40, "needs_clarification": 10, "invalid": 10}), "Gold-status composition mismatch")

    final_root = ROOT / "llm_experiment_final"
    frozen_l2 = final_root / "inputs" / "L2_predictions_frozen.csv"
    frozen = pd.read_csv(frozen_l2)
    require(len(frozen) == 60, "Frozen L2 table must contain 60 rows")
    require(not any(column.startswith("gold_") for column in frozen.columns), "Frozen L2 input contains gold columns")
    ids = (final_root / "inputs" / "frozen_request_ids.txt").read_text(encoding="utf-8").splitlines()
    require(ids == frozen["request_id"].tolist(), "Frozen request-ID order mismatch")
    require(set(ids) == set(benchmark["request_id"]), "Frozen request-ID set mismatch")
    input_hashes = read_key_values(final_root / "inputs" / "input_hashes.txt")
    require(sha256(frozen_l2) == input_hashes["frozen_L2_predictions_sha256"], "Frozen L2 hash mismatch")

    validator_dir = final_root / "validator"
    validator_hashes = read_key_values(validator_dir / "validator_rules_hash.txt")
    require(sha256(validator_dir / "validator_rules.yaml") == validator_hashes["rules_sha256"], "Validator rule hash mismatch")
    require(sha256(validator_dir / "validator.py") == validator_hashes["validator_code_sha256"], "Validator code hash mismatch")
    return {"benchmark_requests": len(benchmark), "frozen_l2_predictions": len(frozen)}


def main() -> int:
    report = {
        "manifest_files": verify_manifest(),
        "data": verify_data(),
        "results": verify_results(),
        "llm_experiment": verify_llm_experiment(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("PASS: minimal release integrity and numerical QA checks completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
