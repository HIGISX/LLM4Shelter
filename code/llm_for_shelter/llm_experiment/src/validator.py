"""JSON-schema and logical validation for the fixed E5 request contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

CORE_FIELDS = ("time_scenario", "hazard_scenario", "p", "objective")
ALLOWED_ISSUES = {
    "missing_parameter",
    "ambiguous_time",
    "ambiguous_hazard",
    "invalid_p",
    "conflicting_time",
    "conflicting_hazard",
    "unsupported_objective",
    "other",
}


def schema_path() -> Path:
    return Path(__file__).resolve().parents[1] / "schema" / "request_schema.json"


def load_schema() -> dict[str, Any]:
    return json.loads(schema_path().read_text(encoding="utf-8"))


def validate_output(payload: Any) -> dict[str, Any]:
    schema_errors: list[str] = []
    if payload is None:
        schema_errors = ["No parseable JSON object."]
    else:
        validator = Draft202012Validator(load_schema())
        schema_errors = [error.message for error in sorted(validator.iter_errors(payload), key=lambda e: list(e.path))]
    schema_valid = not schema_errors
    logical_errors: list[str] = []
    if schema_valid:
        status = payload["status"]
        issues = payload["issues"]
        if status == "valid":
            missing = [field for field in CORE_FIELDS if payload[field] is None]
            if missing:
                logical_errors.append(f"valid status has null core fields: {missing}")
            if issues:
                logical_errors.append("valid status requires an empty issues array")
        else:
            if not issues:
                logical_errors.append("non-valid status requires at least one issue")
        if any(issue not in ALLOWED_ISSUES for issue in issues):
            logical_errors.append("issues contains an unsupported label")
    logical_valid = schema_valid and not logical_errors
    executable = logical_valid and payload.get("status") == "valid"
    normalized = None
    if executable:
        normalized = {field: payload[field] for field in CORE_FIELDS}
    return {
        "schema_valid": schema_valid,
        "logical_valid": logical_valid,
        "final_valid": logical_valid,
        "executable": executable,
        "schema_errors": schema_errors,
        "logical_errors": logical_errors,
        "normalized_request": normalized,
    }
