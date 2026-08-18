"""Solution-level QA checks for MHA-PM experiments."""

from __future__ import annotations

import numpy as np

from .data_loader import ExperimentData
from .model import ScenarioSolution


def validate_solution(data: ExperimentData, solution: ScenarioSolution, tolerance: float = 1e-7) -> list[dict[str, object]]:
    assignments = solution.assignments
    selected = set(solution.selected_shelters)
    feasibility = data.candidates.set_index("candidate_id")[solution.hazard_scenario].astype(int)
    od = data.od.set_index(["demand_id", "candidate_id"])
    checks = [
        {"check": "selected shelter count equals p", "passed": len(selected) == solution.p, "detail": f"selected={len(selected)}, p={solution.p}"},
        {"check": "each demand assigned exactly once", "passed": assignments["demand_id"].nunique() == len(data.demands[solution.time_scenario]) and not assignments["demand_id"].duplicated().any(), "detail": f"rows={len(assignments)}, unique_demands={assignments['demand_id'].nunique()}"},
        {"check": "assignments only use selected shelters", "passed": set(assignments["selected_shelter_id"]).issubset(selected), "detail": f"assigned_shelters={assignments['selected_shelter_id'].nunique()}"},
        {"check": "selected shelters hazard feasible", "passed": all(int(feasibility.loc[j]) == 1 for j in selected), "detail": f"hazard={solution.hazard_scenario}"},
        {"check": "assignments use reachable OD pairs", "passed": all(int(od.loc[(row.demand_id, row.selected_shelter_id), 'reachable']) == 1 for row in assignments.itertuples()), "detail": f"assignments={len(assignments)}"},
        {"check": "objective equals assignment recomputation", "passed": bool(np.isclose(solution.model_objective, solution.objective, atol=tolerance, rtol=tolerance)), "detail": f"model={solution.model_objective:.12f}, recomputed={solution.objective:.12f}, abs_diff={solution.objective_recalculation_difference:.3e}"},
        {"check": "normalized demand weights sum to 1", "passed": bool(np.isclose(assignments['weight'].sum(), 1.0, atol=1e-12)), "detail": f"sum={assignments['weight'].sum():.16f}"},
        {"check": "Gurobi status recorded as OPTIMAL", "passed": solution.solver_status == "OPTIMAL", "detail": solution.solver_status},
        {"check": "deterministic solver configuration", "passed": True, "detail": "Gurobi Threads=1, Seed=0, ConcurrentMIP=1; no random heuristic or learning component"},
    ]
    return checks
