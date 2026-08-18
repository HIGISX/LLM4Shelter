"""Frozen deterministic L3-Final gate. This module must not access gold labels."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator


FORBIDDEN_COLUMNS = {
    "gold_status", "gold_time_scenario", "gold_hazard_scenario", "gold_p",
    "gold_objective", "gold_issue",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rules(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def assert_no_gold(record: dict[str, Any]) -> None:
    overlap = FORBIDDEN_COLUMNS.intersection(record)
    if overlap:
        raise RuntimeError(f"Validator received forbidden gold fields: {sorted(overlap)}")


def _contains_pattern(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _labels(text: str, mapping: dict[str, list[str]]) -> set[str]:
    return {label for label, patterns in mapping.items() if any(_contains_pattern(text, p) for p in patterns)}


def _chinese_integer(token: str) -> int | None:
    if token.isdigit():
        return int(token)
    digits = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if token == "十":
        return 10
    if "十" in token:
        left, right = token.split("十", 1)
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        return tens * 10 + ones
    return digits.get(token)


def extract_p_values(text: str) -> list[float]:
    number = r"(?:\d+(?:\.\d+)?|[一二两三四五六七八九十]+)"
    patterns = [
        rf"(?P<n>{number})\s*个半\s*(?:避难所|避难设施|避难点|设施)",
        rf"(?P<n>{number})\s*(?:个|处|座)?\s*(?:候选)?(?:避难所|避难设施|避难点|设施)",
        rf"(?:设施数|设施数量|避难设施数量|p)\s*(?:固定为|设为|设置为|设成|取|=|等于)?\s*(?P<n>{number})",
        rf"(?:开放|选择|选取|设置|配置|规划)\s*(?P<n>{number})\s*(?:个|处|座)",
    ]
    values: list[float] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            token = match.group("n")
            if "个半" in match.group(0):
                base = float(_chinese_integer(token) if not token[0].isdigit() else token)
                value = base + 0.5
            elif re.fullmatch(r"\d+(?:\.\d+)?", token):
                value = float(token)
            else:
                parsed = _chinese_integer(token)
                if parsed is None:
                    continue
                value = float(parsed)
            if value not in values:
                values.append(value)
    return values


def request_schema(rules: dict[str, Any]) -> dict[str, Any]:
    domains = rules["domains"]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["status", "time_scenario", "hazard_scenario", "p", "objective", "issues"],
        "properties": {
            "status": {"type": "string", "enum": domains["status"]},
            "time_scenario": {"type": ["string", "null"], "enum": domains["time_scenario"] + [None]},
            "hazard_scenario": {"type": ["string", "null"], "enum": domains["hazard_scenario"] + [None]},
            "p": {"type": ["integer", "null"], "minimum": domains["p_min"], "maximum": domains["p_max"]},
            "objective": {"type": ["string", "null"], "enum": [domains["objective"], None]},
            "issues": {"type": "array", "items": {"type": "string"}},
        },
    }


def _parse_prediction(record: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        payload = json.loads(str(record.get("final_output", "")))
    except (json.JSONDecodeError, TypeError):
        return None, ["SCHEMA_ERROR"]
    if not isinstance(payload, dict):
        return None, ["SCHEMA_ERROR"]
    return payload, []


def validate_record(record: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    assert_no_gold(record)
    text = str(record.get("raw_request", ""))
    prediction, issues = _parse_prediction(record)
    schema_valid = False
    if prediction is not None:
        errors = list(Draft202012Validator(request_schema(rules)).iter_errors(prediction))
        schema_valid = not errors
        if errors:
            issues.append("SCHEMA_ERROR")
    l2_status = prediction.get("status") if prediction else None
    l2_time = prediction.get("time_scenario") if prediction else None
    l2_hazard = prediction.get("hazard_scenario") if prediction else None
    l2_p = prediction.get("p") if prediction else None
    l2_objective = prediction.get("objective") if prediction else None

    time_labels = _labels(text, rules["time_patterns"])
    hazard_labels = _labels(text, rules["hazard_patterns"])
    p_values = extract_p_values(text)
    ambiguous_time = not time_labels and any(_contains_pattern(text, p) for p in rules["ambiguous_time_patterns"])
    ambiguous_hazard = not hazard_labels and any(_contains_pattern(text, p) for p in rules["ambiguous_hazard_patterns"])
    indefinite_p = not p_values and any(_contains_pattern(text, p) for p in rules["indefinite_p_patterns"])
    unsupported_objective = any(_contains_pattern(text, p) for p in rules["unsupported_objective_patterns"])
    direct_location = any(_contains_pattern(text, p) for p in rules["direct_llm_location_patterns"])

    invalid_issues: list[str] = []
    clarify_issues: list[str] = []
    if len(time_labels) > 1:
        invalid_issues.append("TIME_CONFLICT")
    elif ambiguous_time or not time_labels:
        clarify_issues.append("TIME_AMBIGUOUS")
    elif l2_status == "valid" and l2_time not in time_labels:
        clarify_issues.append("TIME_PREDICTION_CONFLICT")

    if len(hazard_labels) > 1:
        invalid_issues.append("HAZARD_CONFLICT")
    elif ambiguous_hazard or not hazard_labels:
        clarify_issues.append("HAZARD_AMBIGUOUS")
    elif l2_status == "valid" and l2_hazard not in hazard_labels:
        clarify_issues.append("HAZARD_PREDICTION_CONFLICT")

    if not p_values:
        clarify_issues.append("P_MISSING")
    elif len(p_values) > 1 or any(value != int(value) or value < rules["domains"]["p_min"] or value > rules["domains"]["p_max"] for value in p_values):
        invalid_issues.append("P_INVALID")
    elif l2_status == "valid" and (not isinstance(l2_p, int) or l2_p != int(p_values[0])):
        clarify_issues.append("P_PREDICTION_CONFLICT")

    if unsupported_objective:
        invalid_issues.append("UNSUPPORTED_OBJECTIVE")
    if direct_location:
        invalid_issues.append("DIRECT_LLM_LOCATION_REQUEST")

    # The gate may downgrade but never upgrade the L2 status.
    if l2_status == "invalid":
        final_status = "invalid"
    elif l2_status == "needs_clarification":
        final_status = "needs_clarification"
    elif not schema_valid:
        final_status = "needs_clarification"
    elif invalid_issues:
        final_status = "invalid"
    elif clarify_issues:
        final_status = "needs_clarification"
    else:
        final_status = "valid"
    all_issues = list(dict.fromkeys(issues + invalid_issues + clarify_issues))
    validator_pass = schema_valid and not invalid_issues and not clarify_issues
    solver_allowed = l2_status == "valid" and schema_valid and validator_pass and final_status == "valid"
    if solver_allowed:
        block_reason = ""
    elif not schema_valid:
        block_reason = "SCHEMA_ERROR"
    elif l2_status != "valid":
        block_reason = f"L2_STATUS_{str(l2_status).upper()}"
    else:
        block_reason = ";".join(all_issues) or "VALIDATION_BLOCKED"
    return {
        "request_id": record.get("request_id"),
        "request_type": record.get("request_type"),
        "request_text": text,
        "L2_status": l2_status,
        "L2_time": l2_time,
        "L2_hazard": l2_hazard,
        "L2_p": l2_p,
        "L2_objective": l2_objective,
        "schema_valid": schema_valid,
        "validator_pass": validator_pass,
        "validator_issues": json.dumps(all_issues, ensure_ascii=False),
        "final_status": final_status,
        "solver_allowed": solver_allowed,
        "solver_called": False,
        "solver_block_reason": block_reason,
        "objective_if_executed": None,
        "selected_solution_match": None,
    }
