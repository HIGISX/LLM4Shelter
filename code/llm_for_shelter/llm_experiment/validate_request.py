"""Validate parsed JSON without repairing or expanding the fixed research model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ALLOWED_TIMES = {"Morning", "Evening", "Night"}
ALLOWED_HAZARDS = {"baseline", "moderate", "strict"}
ALLOWED_KEYS = {"time_scenario", "hazard_scenario", "p", "objective"}
REQUIRED_KEYS = ALLOWED_KEYS


def validate_request(request: Any) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(request, dict):
        return {"valid": False, "errors": ["Request must be a JSON object."], "normalized_request": None}
    missing = sorted(REQUIRED_KEYS - set(request))
    extra = sorted(set(request) - ALLOWED_KEYS)
    if missing:
        errors.append(f"Missing required fields: {missing}")
    if extra:
        errors.append(f"Unexpected fields: {extra}")
    if request.get("time_scenario") not in ALLOWED_TIMES:
        errors.append("time_scenario must be Morning, Evening, or Night.")
    if request.get("hazard_scenario") not in ALLOWED_HAZARDS:
        errors.append("hazard_scenario must be baseline, moderate, or strict.")
    p = request.get("p")
    if isinstance(p, bool) or not isinstance(p, int) or p < 1:
        errors.append("p must be a positive integer.")
    if request.get("objective") != "weighted_p_median":
        errors.append("objective currently only allows weighted_p_median.")
    normalized = {key: request.get(key) for key in ("time_scenario", "hazard_scenario", "p", "objective")} if not errors else None
    return {"valid": not errors, "errors": errors, "normalized_request": normalized}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("request_json")
    args = parser.parse_args()
    payload = json.loads(Path(args.request_json).read_text(encoding="utf-8"))
    result = validate_request(payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
