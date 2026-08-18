"""Validated parsed JSON -> deterministic MHA-PM -> selected shelters and metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from ..data_loader import load_experiment_data
    from ..model import solve_mha_pm
    from .validate_request import validate_request
except ImportError:  # Standalone copy under results/llm_experiment.
    workspace = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(workspace / "code"))
    from llm_for_shelter.data_loader import load_experiment_data
    from llm_for_shelter.model import solve_mha_pm
    from validate_request import validate_request


def run_optimizer_from_request(request: dict[str, Any], workspace_root: str | Path) -> dict[str, Any]:
    validation = validate_request(request)
    if not validation["valid"]:
        return {"executed": False, "validation": validation, "solution": None}
    parsed = validation["normalized_request"]
    data = load_experiment_data(workspace_root)
    solution = solve_mha_pm(data, parsed["time_scenario"], parsed["hazard_scenario"], parsed["p"])
    return {
        "executed": True,
        "validation": validation,
        "solution": {
            **solution.summary_dict(),
            "selected_shelters": solution.selected_shelters,
        },
    }


def solution_match(predicted: dict[str, Any], gold: dict[str, Any], workspace_root: str | Path, tolerance: float = 1e-7) -> dict[str, Any]:
    predicted_result = run_optimizer_from_request(predicted, workspace_root)
    gold_result = run_optimizer_from_request(gold, workspace_root)
    if not predicted_result["executed"] or not gold_result["executed"]:
        return {"solution_match": False, "reason": "one or both requests are not executable"}
    predicted_solution = predicted_result["solution"]
    gold_solution = gold_result["solution"]
    objective_match = abs(predicted_solution["objective"] - gold_solution["objective"]) <= tolerance
    set_match = set(predicted_solution["selected_shelters"]) == set(gold_solution["selected_shelters"])
    return {
        "solution_match": objective_match and set_match,
        "objective_match": objective_match,
        "selected_set_match": set_match,
        "objective_difference": abs(predicted_solution["objective"] - gold_solution["objective"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("request_json")
    parser.add_argument("--workspace-root", default=".")
    args = parser.parse_args()
    request = json.loads(Path(args.request_json).read_text(encoding="utf-8"))
    print(json.dumps(run_optimizer_from_request(request, args.workspace_root), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
