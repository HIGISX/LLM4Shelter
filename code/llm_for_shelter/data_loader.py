"""Automatic input discovery, schema mapping, normalization, and data QA."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TIME_ORDER = ("Morning", "Evening", "Night")
HAZARD_ORDER = ("baseline", "moderate", "strict")


@dataclass
class ExperimentData:
    workspace_root: Path
    input_dir: Path
    demand_files: dict[str, Path]
    candidate_file: Path
    od_file: Path
    demands: dict[str, pd.DataFrame]
    candidates: pd.DataFrame
    od: pd.DataFrame
    schema_mapping: dict[str, Any]
    qa_checks: list[dict[str, Any]]


def _read_csv(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Could not decode CSV {path}: {last_error}")


def _find_column(columns: list[str], aliases: tuple[str, ...], *, required: bool = True) -> str | None:
    lookup = {str(column).strip().lower(): str(column) for column in columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    for alias in aliases:
        for normalized, original in lookup.items():
            if alias.lower() in normalized:
                return original
    if required:
        raise ValueError(f"Could not identify required field from aliases {aliases}; columns={columns}")
    return None


def _infer_time(path: Path, df: pd.DataFrame) -> str | None:
    text = path.stem.lower()
    if "morning" in text:
        return "Morning"
    if "evening" in text:
        return "Evening"
    if "night" in text:
        return "Night"
    for column in df.columns:
        if "time" in column.lower():
            values = " ".join(df[column].dropna().astype(str).str.lower().unique())
            if "morning" in values:
                return "Morning"
            if "evening" in values:
                return "Evening"
            if "night" in values:
                return "Night"
    return None


def discover_input_files(workspace_root: str | Path) -> tuple[dict[str, Path], Path, Path, Path]:
    root = Path(workspace_root).resolve()
    csv_files = [path for path in root.rglob("*.csv") if "results" not in {p.lower() for p in path.parts}]
    inspected: list[tuple[Path, pd.DataFrame]] = []
    for path in csv_files:
        try:
            sample = pd.read_csv(path, nrows=10)
        except Exception:
            continue
        inspected.append((path, sample))

    od_candidates = [
        (path, sample)
        for path, sample in inspected
        if {"demand_id", "candidate_id"}.issubset({str(c).lower() for c in sample.columns})
        and any("distance" in str(c).lower() for c in sample.columns)
    ]
    if not od_candidates:
        raise FileNotFoundError("Could not identify the OD network-distance CSV.")
    od_file = max(od_candidates, key=lambda item: item[0].stat().st_size)[0]

    candidate_options = [
        (path, sample)
        for path, sample in inspected
        if "candidate_id" in {str(c).lower() for c in sample.columns}
        and any("safe_z12" == str(c).lower() for c in sample.columns)
        and path != od_file
    ]
    if not candidate_options:
        raise FileNotFoundError("Could not identify the shelter-candidate hazard CSV.")
    exact_438 = [item for item in candidate_options if len(_read_csv(item[0])) == 438]
    candidate_file = (exact_438 or candidate_options)[0][0]

    demand_files: dict[str, Path] = {}
    for path, sample in inspected:
        lower = {str(c).lower() for c in sample.columns}
        if "demand_id" not in lower or not any("mobility" in c for c in lower):
            continue
        full = _read_csv(path)
        time = _infer_time(path, full)
        if time and time not in demand_files:
            demand_files[time] = path
    missing = [time for time in TIME_ORDER if time not in demand_files]
    if missing:
        raise FileNotFoundError(f"Could not identify demand files for: {missing}")
    input_dir = Path(os.path.commonpath([str(path.parent) for path in [*demand_files.values(), candidate_file, od_file]]))
    return demand_files, candidate_file, od_file, input_dir


def _check(name: str, passed: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "severity": severity, "detail": detail}


def load_experiment_data(workspace_root: str | Path, mapping_output: str | Path | None = None) -> ExperimentData:
    root = Path(workspace_root).resolve()
    demand_files, candidate_file, od_file, input_dir = discover_input_files(root)
    mapping: dict[str, Any] = {"files": {}, "hazard_scenarios": {}}
    demands: dict[str, pd.DataFrame] = {}
    checks: list[dict[str, Any]] = []

    for time in TIME_ORDER:
        path = demand_files[time]
        raw = _read_csv(path)
        id_col = _find_column(list(raw.columns), ("demand_id", "demand_id_", "locationid"))
        weight_col = _find_column(list(raw.columns), ("mobility_i", "mobility_index", "activity_weight"))
        intensity = pd.to_numeric(raw[weight_col], errors="coerce")
        total = float(intensity.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"Demand intensity sum must be positive for {time}; got {total}.")
        standardized = pd.DataFrame(
            {
                "demand_id": raw[id_col].astype(str),
                "activity_intensity": intensity,
                "weight": intensity / total,
            }
        )
        demands[time] = standardized
        mapping["files"][str(path.relative_to(root))] = {
            "role": f"{time} demand",
            "columns": {id_col: "i (demand location identifier)", weight_col: "raw mobility activity intensity; normalized to w_it"},
        }
        checks.extend(
            [
                _check(f"{time}: one record per demand location", not standardized["demand_id"].duplicated().any(), f"rows={len(standardized)}, unique_ids={standardized['demand_id'].nunique()}"),
                _check(f"{time}: demand intensity complete", standardized["activity_intensity"].notna().all(), f"missing={int(standardized['activity_intensity'].isna().sum())}"),
                _check(f"{time}: nonnegative demand intensity", bool((standardized["activity_intensity"] >= 0).all()), f"minimum={standardized['activity_intensity'].min():.12g}"),
                _check(f"{time}: normalized weights sum to 1", bool(np.isclose(standardized["weight"].sum(), 1.0, atol=1e-12)), f"sum={standardized['weight'].sum():.16f}"),
            ]
        )

    demand_sets = [set(demands[time]["demand_id"]) for time in TIME_ORDER]
    checks.append(_check("Demand ID sets identical across time scenarios", demand_sets[0] == demand_sets[1] == demand_sets[2], f"set_sizes={[len(s) for s in demand_sets]}"))

    candidate_raw = _read_csv(candidate_file)
    candidate_id_col = _find_column(list(candidate_raw.columns), ("candidate_id", "site_id", "shelter_id"))
    moderate_col = _find_column(list(candidate_raw.columns), ("safe_z12",))
    strict_col = _find_column(list(candidate_raw.columns), ("safe_z123",))
    candidates = pd.DataFrame(
        {
            "candidate_id": candidate_raw[candidate_id_col].astype(str),
            "baseline": 1,
            "moderate": pd.to_numeric(candidate_raw[moderate_col], errors="coerce"),
            "strict": pd.to_numeric(candidate_raw[strict_col], errors="coerce"),
        }
    )
    for optional, aliases in {
        "candidate_name": ("candidate_name", "name"),
        "longitude": ("longitude", "lon"),
        "latitude": ("latitude", "lat"),
    }.items():
        column = _find_column(list(candidate_raw.columns), aliases, required=False)
        if column:
            candidates[optional] = candidate_raw[column].values
    mapping["files"][str(candidate_file.relative_to(root))] = {
        "role": "shelter candidates and hazard feasibility",
        "columns": {
            candidate_id_col: "j (candidate shelter identifier)",
            moderate_col: "a_jh for moderate hazard constraint",
            strict_col: "a_jh for strict hazard constraint",
        },
        "derived": {"baseline": "a_jh = 1 for every candidate"},
    }
    mapping["hazard_scenarios"] = {
        "baseline": "all candidates feasible",
        "moderate": f"candidate feasibility from {moderate_col}",
        "strict": f"candidate feasibility from {strict_col}",
    }
    checks.extend(
        [
            _check("Candidate shelter IDs unique", not candidates["candidate_id"].duplicated().any(), f"rows={len(candidates)}, unique_ids={candidates['candidate_id'].nunique()}"),
            _check("Hazard feasibility complete", candidates[["baseline", "moderate", "strict"]].notna().all().all(), f"missing={candidates[['baseline','moderate','strict']].isna().sum().to_dict()}"),
            _check("Hazard feasibility binary", bool(candidates[["baseline", "moderate", "strict"]].isin([0, 1]).all().all()), "Expected a_jh in {0,1}"),
            _check("Strict feasible set nested in moderate", bool((candidates["strict"] <= candidates["moderate"]).all()), f"moderate={int(candidates['moderate'].sum())}, strict={int(candidates['strict'].sum())}"),
        ]
    )

    od_raw = _read_csv(od_file)
    od_demand_col = _find_column(list(od_raw.columns), ("demand_id", "origin_id"))
    od_candidate_col = _find_column(list(od_raw.columns), ("candidate_id", "site_id", "destination_id"))
    distance_col = _find_column(list(od_raw.columns), ("network_distance_m", "network_distance", "shortest_path_m", "distance_m"))
    reachable_col = _find_column(list(od_raw.columns), ("reachable", "is_reachable", "reachability"))
    od = pd.DataFrame(
        {
            "demand_id": od_raw[od_demand_col].astype(str),
            "candidate_id": od_raw[od_candidate_col].astype(str),
            "distance": pd.to_numeric(od_raw[distance_col], errors="coerce"),
            "reachable": pd.to_numeric(od_raw[reachable_col], errors="coerce"),
        }
    )
    mapping["files"][str(od_file.relative_to(root))] = {
        "role": "OD road-network distances and reachability",
        "columns": {
            od_demand_col: "i (demand location identifier)",
            od_candidate_col: "j (candidate shelter identifier)",
            distance_col: "d_ij (road-network distance, metres)",
            reachable_col: "r_ij (network reachability indicator)",
        },
    }
    expected_pairs = len(demand_sets[0]) * len(candidates)
    actual_pairs = len(od)
    checks.extend(
        [
            _check("OD pair uniqueness", not od.duplicated(["demand_id", "candidate_id"]).any(), f"rows={actual_pairs}, unique_pairs={len(od.drop_duplicates(['demand_id','candidate_id']))}"),
            _check("OD pair completeness", actual_pairs == expected_pairs, f"actual={actual_pairs}, expected={expected_pairs}"),
            _check("Network distances complete", od["distance"].notna().all(), f"missing={int(od['distance'].isna().sum())}"),
            _check("Network distances nonnegative", bool((od["distance"] >= 0).all()), f"minimum={od['distance'].min():.12g}"),
            _check("Reachability indicator complete and binary", bool(od["reachable"].notna().all() and od["reachable"].isin([0, 1]).all()), f"counts={od['reachable'].value_counts(dropna=False).to_dict()}"),
            _check("OD demand IDs match demand tables", set(od["demand_id"]) == demand_sets[0], f"od_ids={od['demand_id'].nunique()}, demand_ids={len(demand_sets[0])}"),
            _check("OD candidate IDs match candidate table", set(od["candidate_id"]) == set(candidates["candidate_id"]), f"od_ids={od['candidate_id'].nunique()}, candidate_ids={len(candidates)}"),
        ]
    )
    unreachable = int((od["reachable"] == 0).sum())
    checks.append(_check("Unreachable OD pairs documented", True, f"unreachable_pairs={unreachable}", severity="info"))

    if mapping_output is not None:
        output = Path(mapping_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    return ExperimentData(
        workspace_root=root,
        input_dir=input_dir,
        demand_files=demand_files,
        candidate_file=candidate_file,
        od_file=od_file,
        demands=demands,
        candidates=candidates,
        od=od,
        schema_mapping=mapping,
        qa_checks=checks,
    )
