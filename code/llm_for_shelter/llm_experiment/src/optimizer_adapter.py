"""Safety-gated adapter from validated E5 parameters to the existing MHA-PM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ...data_loader import load_experiment_data
from ...model import solve_mha_pm
from .validator import validate_output

TIME_TO_MODEL = {"morning": "Morning", "evening": "Evening", "night": "Night"}


class OptimizerAdapter:
    def __init__(self, workspace_root: str | Path):
        self.workspace_root = Path(workspace_root).resolve()
        self.summary_cache: dict[tuple[str, str, int], dict[str, Any]] = {}
        self.set_cache: dict[tuple[str, str, int], list[str]] = {}
        self._data = None
        self._load_existing_results()

    def _load_existing_results(self) -> None:
        result_dir = self.workspace_root / "results"
        summary_path = result_dir / "all_scenario_summaries.csv"
        shelters_path = result_dir / "all_selected_shelters.csv"
        if summary_path.exists():
            for row in pd.read_csv(summary_path).to_dict("records"):
                key = (str(row["time_scenario"]), str(row["hazard_scenario"]), int(row["p"]))
                self.summary_cache[key] = row
        if shelters_path.exists():
            selected = pd.read_csv(shelters_path)
            for key_values, group in selected.groupby(["time_scenario", "hazard_scenario", "p"], sort=False):
                key = (str(key_values[0]), str(key_values[1]), int(key_values[2]))
                self.set_cache[key] = sorted(group["selected_shelter_id"].astype(str).tolist())

    def _solve_or_cached(self, params: dict[str, Any]) -> dict[str, Any]:
        model_time = TIME_TO_MODEL[params["time_scenario"]]
        key = (model_time, params["hazard_scenario"], int(params["p"]))
        if key in self.summary_cache and key in self.set_cache:
            row = self.summary_cache[key]
            return {
                "source": "verified_E1_E4_result_cache",
                "solver_status": row["solver_status"],
                "objective": float(row["objective"]),
                "selected_shelters": self.set_cache[key],
            }
        if self._data is None:
            self._data = load_experiment_data(self.workspace_root)
        solution = solve_mha_pm(self._data, model_time, params["hazard_scenario"], int(params["p"]))
        return {
            "source": "live_existing_mha_pm_call",
            "solver_status": solution.solver_status,
            "objective": solution.objective,
            "selected_shelters": solution.selected_shelters,
        }

    def execute(self, parsed_output: Any) -> dict[str, Any]:
        validation = validate_output(parsed_output)
        if not validation["executable"]:
            return {"solver_called": False, "validation": validation, "solution": None}
        solution = self._solve_or_cached(validation["normalized_request"])
        return {"solver_called": True, "validation": validation, "solution": solution}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
