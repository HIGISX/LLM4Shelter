"""Compare parsed request predictions with completed gold annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .validate_request import validate_request
except ImportError:  # Standalone copy under results/llm_experiment.
    from validate_request import validate_request

FIELDS = ("time_scenario", "hazard_scenario", "p", "objective")


def compare_prediction_to_gold(predictions: list[dict[str, Any]], gold: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    prediction_by_id = {str(item["request_id"]): item for item in predictions}
    rows: list[dict[str, Any]] = []
    for gold_row in gold.itertuples(index=False):
        if str(gold_row.annotation_status).lower() != "complete":
            continue
        request_id = str(gold_row.request_id)
        prediction = prediction_by_id.get(request_id, {})
        parsed = prediction.get("parsed_request")
        json_valid = isinstance(parsed, dict)
        validation = validate_request(parsed) if json_valid else {"valid": False, "normalized_request": None}
        gold_valid = str(gold_row.gold_is_valid).strip().lower() in {"true", "1", "yes"}
        field_matches = {}
        for field in FIELDS:
            expected = getattr(gold_row, f"gold_{field}")
            actual = parsed.get(field) if isinstance(parsed, dict) else None
            if field == "p" and pd.notna(expected):
                expected = int(expected)
            field_matches[field] = actual == expected
        exact = all(field_matches.values()) if gold_valid else (not validation["valid"])
        rows.append(
            {
                "request_id": request_id,
                "json_valid": json_valid,
                "schema_valid": bool(validation["valid"]),
                **{f"{field}_correct": match for field, match in field_matches.items()},
                "exact_match": exact,
                "invalid_request_detection": (not validation["valid"]) == (not gold_valid),
                "executable_request": bool(validation["valid"]),
                "repair_success": bool(prediction.get("repair_success", False)),
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, {"evaluated_requests": 0.0}
    metrics = {"evaluated_requests": float(len(detail))}
    for column in detail.columns:
        if column != "request_id" and detail[column].dtype == bool:
            metrics[column + "_rate"] = float(detail[column].mean())
    return detail, metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_json")
    parser.add_argument("gold_csv")
    parser.add_argument("--output", default="prediction_evaluation.csv")
    args = parser.parse_args()
    predictions = json.loads(Path(args.predictions_json).read_text(encoding="utf-8"))
    gold = pd.read_csv(args.gold_csv)
    detail, metrics = compare_prediction_to_gold(predictions, gold)
    detail.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
