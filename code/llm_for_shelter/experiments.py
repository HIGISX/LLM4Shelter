"""E1-E4 experiment orchestration using one MHA-PM solver entry point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import HAZARD_ORDER, TIME_ORDER, ExperimentData
from .metrics import shelter_set_metrics
from .model import ScenarioSolution, evaluate_fixed_shelters, solve_mha_pm
from .qa import validate_solution

P_VALUES = (5, 10, 15)


@dataclass
class ExperimentOutputs:
    solutions: dict[tuple[str, str, int], ScenarioSolution]
    summaries: pd.DataFrame
    selected_shelters: pd.DataFrame
    assignments: pd.DataFrame
    e1_temporal: pd.DataFrame
    e1_selected: pd.DataFrame
    e1_assignments: pd.DataFrame
    e1_overlap: pd.DataFrame
    e1_tcc_matrix: pd.DataFrame
    e2_hazard: pd.DataFrame
    e2_selected: pd.DataFrame
    e2_hcc: pd.DataFrame
    e3_p_sensitivity: pd.DataFrame
    e4_cross_time: pd.DataFrame
    e4_tmp: pd.DataFrame
    solution_qa: pd.DataFrame
    solver_verification: pd.DataFrame


def _selected_rows(solution: ScenarioSolution) -> list[dict[str, Any]]:
    return [
        {
            "scenario_id": solution.scenario_id,
            "time_scenario": solution.time_scenario,
            "hazard_scenario": solution.hazard_scenario,
            "p": solution.p,
            "selected_shelter_id": shelter_id,
        }
        for shelter_id in solution.selected_shelters
    ]


def run_all_experiments(data: ExperimentData) -> ExperimentOutputs:
    solutions: dict[tuple[str, str, int], ScenarioSolution] = {}
    qa_rows: list[dict[str, Any]] = []
    for time in TIME_ORDER:
        for hazard in HAZARD_ORDER:
            for p in P_VALUES:
                solution = solve_mha_pm(data, time, hazard, p)
                solutions[(time, hazard, p)] = solution
                for check in validate_solution(data, solution):
                    qa_rows.append({"scenario_id": solution.scenario_id, **check})

    summaries = pd.DataFrame([solution.summary_dict() for solution in solutions.values()])
    selected = pd.DataFrame([row for solution in solutions.values() for row in _selected_rows(solution)])
    assignments = pd.concat([solution.assignments for solution in solutions.values()], ignore_index=True)

    # E1: temporal effect at baseline, p=10.
    e1_solutions = [solutions[(time, "baseline", 10)] for time in TIME_ORDER]
    e1_temporal = pd.DataFrame([solution.summary_dict() for solution in e1_solutions])
    e1_selected = pd.DataFrame([row for solution in e1_solutions for row in _selected_rows(solution)])
    e1_assignments = pd.concat([solution.assignments for solution in e1_solutions], ignore_index=True)
    overlap_rows = []
    tcc = pd.DataFrame(index=TIME_ORDER, columns=TIME_ORDER, dtype=float)
    for first in TIME_ORDER:
        for second in TIME_ORDER:
            metrics = shelter_set_metrics(
                set(solutions[(first, "baseline", 10)].selected_shelters),
                set(solutions[(second, "baseline", 10)].selected_shelters),
            )
            overlap_rows.append({"time_1": first, "time_2": second, **metrics})
            tcc.loc[first, second] = metrics["temporal_configuration_change"]
    e1_overlap = pd.DataFrame(overlap_rows)
    e1_tcc_matrix = tcc.rename_axis("time_scenario").reset_index()

    # E2: hazard constraints at p=10.
    e2_rows: list[dict[str, Any]] = []
    e2_hcc_rows: list[dict[str, Any]] = []
    e2_selected_rows: list[dict[str, Any]] = []
    for time in TIME_ORDER:
        baseline = solutions[(time, "baseline", 10)]
        baseline_set = set(baseline.selected_shelters)
        for hazard in HAZARD_ORDER:
            solution = solutions[(time, hazard, 10)]
            selected_set = set(solution.selected_shelters)
            feasibility = data.candidates.set_index("candidate_id")[hazard].astype(int)
            invalidated = sum(int(feasibility.loc[shelter]) == 0 for shelter in baseline_set)
            replaced = len(baseline_set - selected_set)
            hcc = (solution.objective - baseline.objective) / baseline.objective * 100.0
            row = {
                **solution.summary_dict(),
                "selected_shelter_changes": len(baseline_set.symmetric_difference(selected_set)),
                "baseline_shelters_invalidated": invalidated,
                "baseline_shelters_replaced": replaced,
                "HCC_percent": hcc,
            }
            e2_rows.append(row)
            e2_hcc_rows.append(
                {
                    "time_scenario": time,
                    "hazard_scenario": hazard,
                    "baseline_objective": baseline.objective,
                    "hazard_objective": solution.objective,
                    "HCC_percent": hcc,
                }
            )
            e2_selected_rows.extend(_selected_rows(solution))
    e2_hazard = pd.DataFrame(e2_rows)
    e2_hcc = pd.DataFrame(e2_hcc_rows)
    e2_selected = pd.DataFrame(e2_selected_rows)

    # E3: p sensitivity, with marginal benefits recorded on p=10 and p=15 rows.
    e3_rows: list[dict[str, Any]] = []
    for time in TIME_ORDER:
        for hazard in HAZARD_ORDER:
            z = {p: solutions[(time, hazard, p)].objective for p in P_VALUES}
            mb_5_10 = (z[5] - z[10]) / z[5]
            mb_10_15 = (z[10] - z[15]) / z[10]
            for p in P_VALUES:
                solution = solutions[(time, hazard, p)]
                e3_rows.append(
                    {
                        **solution.summary_dict(),
                        "selected_shelters": ";".join(solution.selected_shelters),
                        "MB_5_10": mb_5_10 if p == 10 else np.nan,
                        "MB_10_15": mb_10_15 if p == 15 else np.nan,
                    }
                )
    e3 = pd.DataFrame(e3_rows)

    # E4: fixed-set cross-time evaluation, p=10.
    cross_rows: list[dict[str, Any]] = []
    for hazard in HAZARD_ORDER:
        for optimized_time in TIME_ORDER:
            selected_set = solutions[(optimized_time, hazard, 10)].selected_shelters
            for evaluation_time in TIME_ORDER:
                row = evaluate_fixed_shelters(
                    data,
                    optimized_time,
                    evaluation_time,
                    hazard,
                    10,
                    selected_set,
                )
                optimum = solutions[(evaluation_time, hazard, 10)].objective
                row["evaluation_time_optimum"] = optimum
                row["TMP_percent"] = (float(row["cross_time_cost"]) - optimum) / optimum * 100.0
                row["diagonal_matches_optimum"] = (
                    bool(np.isclose(float(row["cross_time_cost"]), optimum, atol=1e-7, rtol=1e-7))
                    if optimized_time == evaluation_time
                    else np.nan
                )
                cross_rows.append(row)
    e4_cross = pd.DataFrame(cross_rows)
    e4_tmp = e4_cross[
        ["hazard_scenario", "p", "optimized_time", "evaluation_time", "cross_time_cost", "evaluation_time_optimum", "TMP_percent", "diagonal_matches_optimum"]
    ].copy()

    verification = pd.DataFrame(
        [
            {
                "verification_solver": "HiGHS",
                "scenario_id": scenario,
                "status": "NOT_RUN",
                "reason": "highspy and SciPy/HiGHS are not installed in the current environment",
                "gurobi_status": solutions[key].solver_status,
                "gurobi_objective": solutions[key].objective,
                "objective_difference": np.nan,
                "selected_set_match": np.nan,
            }
            for scenario, key in [
                ("Morning_baseline_p10", ("Morning", "baseline", 10)),
                ("Evening_moderate_p10", ("Evening", "moderate", 10)),
                ("Night_strict_p10", ("Night", "strict", 10)),
            ]
        ]
    )
    return ExperimentOutputs(
        solutions=solutions,
        summaries=summaries,
        selected_shelters=selected,
        assignments=assignments,
        e1_temporal=e1_temporal,
        e1_selected=e1_selected,
        e1_assignments=e1_assignments,
        e1_overlap=e1_overlap,
        e1_tcc_matrix=e1_tcc_matrix,
        e2_hazard=e2_hazard,
        e2_selected=e2_selected,
        e2_hcc=e2_hcc,
        e3_p_sensitivity=e3,
        e4_cross_time=e4_cross,
        e4_tmp=e4_tmp,
        solution_qa=pd.DataFrame(qa_rows),
        solver_verification=verification,
    )
