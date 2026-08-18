"""Deterministic Gurobi implementation of the fixed MHA-PM formulation."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_loader import ExperimentData
from .metrics import assignment_metrics


@dataclass
class ScenarioSolution:
    scenario_id: str
    time_scenario: str
    hazard_scenario: str
    p: int
    solver_status: str
    objective: float
    weighted_mean_distance: float
    weighted_p90_distance: float
    max_distance: float
    runtime_seconds: float
    selected_shelter_count: int
    feasible_candidate_count: int
    selected_shelters: list[str]
    assignments: pd.DataFrame
    model_objective: float
    objective_recalculation_difference: float

    def summary_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "time_scenario": self.time_scenario,
            "hazard_scenario": self.hazard_scenario,
            "p": self.p,
            "solver_status": self.solver_status,
            "objective": self.objective,
            "weighted_mean_distance": self.weighted_mean_distance,
            "weighted_p90_distance": self.weighted_p90_distance,
            "max_distance": self.max_distance,
            "runtime_seconds": self.runtime_seconds,
            "selected_shelter_count": self.selected_shelter_count,
            "feasible_candidate_count": self.feasible_candidate_count,
            "model_objective": self.model_objective,
            "objective_recalculation_difference": self.objective_recalculation_difference,
        }


def _deterministic_assignments(
    data: ExperimentData,
    time_scenario: str,
    selected_shelters: list[str],
) -> pd.DataFrame:
    selected = set(selected_shelters)
    od = data.od.loc[(data.od["candidate_id"].isin(selected)) & (data.od["reachable"] == 1)].copy()
    if od.empty:
        raise ValueError("No reachable OD pairs for the selected shelter set.")
    od = od.sort_values(["demand_id", "distance", "candidate_id"], kind="stable")
    nearest = od.groupby("demand_id", as_index=False, sort=False).first()
    demand = data.demands[time_scenario]
    assignments = demand.merge(nearest[["demand_id", "candidate_id", "distance", "reachable"]], on="demand_id", how="left", validate="one_to_one")
    if assignments["candidate_id"].isna().any():
        missing = assignments.loc[assignments["candidate_id"].isna(), "demand_id"].tolist()
        raise ValueError(f"Selected shelter set leaves demand locations unreachable: {missing}")
    return assignments.rename(columns={"candidate_id": "selected_shelter_id", "distance": "assigned_distance"})


def solve_mha_pm(
    data: ExperimentData,
    time_scenario: str,
    hazard_scenario: str,
    p: int,
    *,
    output_flag: bool = False,
) -> ScenarioSolution:
    import gurobipy as gp
    from gurobipy import GRB

    if time_scenario not in data.demands:
        raise ValueError(f"Unknown time scenario: {time_scenario}")
    if hazard_scenario not in {"baseline", "moderate", "strict"}:
        raise ValueError(f"Unknown hazard scenario: {hazard_scenario}")
    candidates = data.candidates.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    demands = data.demands[time_scenario].sort_values("demand_id", kind="stable").reset_index(drop=True)
    feasible = candidates.set_index("candidate_id")[hazard_scenario].astype(int).to_dict()
    feasible_count = int(sum(feasible.values()))
    if p <= 0 or p > feasible_count:
        raise ValueError(f"p={p} is invalid for {hazard_scenario}; feasible candidates={feasible_count}.")

    od = data.od.set_index(["demand_id", "candidate_id"]).sort_index()
    demand_ids = demands["demand_id"].tolist()
    candidate_ids = candidates["candidate_id"].tolist()
    weights = demands.set_index("demand_id")["weight"].to_dict()

    model = gp.Model(f"MHA_PM_{time_scenario}_{hazard_scenario}_p{p}")
    model.Params.OutputFlag = int(output_flag)
    model.Params.Threads = 1
    model.Params.Seed = 0
    model.Params.ConcurrentMIP = 1
    x = model.addVars(candidate_ids, vtype=GRB.BINARY, name="x")
    y = model.addVars(demand_ids, candidate_ids, vtype=GRB.BINARY, name="y")
    model.setObjective(
        gp.quicksum(
            float(weights[i]) * float(od.loc[(i, j), "distance"]) * y[i, j]
            for i in demand_ids
            for j in candidate_ids
        ),
        GRB.MINIMIZE,
    )
    model.addConstr(gp.quicksum(x[j] for j in candidate_ids) == p, name="facility_count")
    for j in candidate_ids:
        model.addConstr(x[j] <= int(feasible[j]), name=f"hazard_feasible[{j}]")
    for i in demand_ids:
        model.addConstr(gp.quicksum(y[i, j] for j in candidate_ids) == 1, name=f"assign_once[{i}]")
        for j in candidate_ids:
            model.addConstr(y[i, j] <= x[j], name=f"assign_open[{i},{j}]")
            model.addConstr(y[i, j] <= int(od.loc[(i, j), "reachable"]), name=f"reachable[{i},{j}]")

    started = time.perf_counter()
    model.optimize()
    runtime = time.perf_counter() - started
    status_name = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.TIME_LIMIT: "TIME_LIMIT",
    }.get(model.Status, f"STATUS_{model.Status}")
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"MHA-PM did not reach optimality for {time_scenario}/{hazard_scenario}/p={p}: {status_name}")

    selected = sorted([j for j in candidate_ids if x[j].X > 0.5])
    assignments = _deterministic_assignments(data, time_scenario, selected)
    metrics = assignment_metrics(assignments)
    model_objective = float(model.ObjVal)
    difference = abs(model_objective - metrics["objective"])
    scenario_id = f"{time_scenario}_{hazard_scenario}_p{p}"
    assignments.insert(0, "scenario_id", scenario_id)
    assignments.insert(1, "time_scenario", time_scenario)
    assignments.insert(2, "hazard_scenario", hazard_scenario)
    assignments.insert(3, "p", p)
    return ScenarioSolution(
        scenario_id=scenario_id,
        time_scenario=time_scenario,
        hazard_scenario=hazard_scenario,
        p=p,
        solver_status=status_name,
        objective=metrics["objective"],
        weighted_mean_distance=metrics["weighted_mean_distance"],
        weighted_p90_distance=metrics["weighted_p90_distance"],
        max_distance=metrics["max_distance"],
        runtime_seconds=runtime,
        selected_shelter_count=len(selected),
        feasible_candidate_count=feasible_count,
        selected_shelters=selected,
        assignments=assignments,
        model_objective=model_objective,
        objective_recalculation_difference=difference,
    )


def evaluate_fixed_shelters(
    data: ExperimentData,
    optimized_time: str,
    evaluation_time: str,
    hazard_scenario: str,
    p: int,
    selected_shelters: list[str],
) -> dict[str, object]:
    feasibility = data.candidates.set_index("candidate_id")[hazard_scenario].astype(int)
    invalid = [shelter for shelter in selected_shelters if int(feasibility.loc[shelter]) != 1]
    if invalid:
        raise ValueError(f"Fixed set includes hazard-infeasible shelters: {invalid}")
    assignments = _deterministic_assignments(data, evaluation_time, selected_shelters)
    metrics = assignment_metrics(assignments)
    return {
        "hazard_scenario": hazard_scenario,
        "p": p,
        "optimized_time": optimized_time,
        "evaluation_time": evaluation_time,
        "cross_time_cost": metrics["objective"],
        "weighted_p90_distance": metrics["weighted_p90_distance"],
        "max_distance": metrics["max_distance"],
    }
