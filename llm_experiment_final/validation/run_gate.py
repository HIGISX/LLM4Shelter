"""Run frozen L3-Final gate using only request text and frozen L2 predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _hash_values(path: Path) -> dict[str, str]:
    return dict(line.strip().split("=", 1) for line in path.read_text(encoding="utf-8").splitlines() if "=" in line)


def run(workspace_root: Path) -> Path:
    final_root = workspace_root / "llm_experiment_final"
    validator_dir = final_root / "validator"
    sys.path.insert(0, str(validator_dir))
    from validator import file_sha256, load_rules, validate_record

    frozen = _hash_values(validator_dir / "validator_rules_hash.txt")
    if file_sha256(validator_dir / "validator_rules.yaml") != frozen["rules_sha256"]:
        raise RuntimeError("Frozen validator rules hash mismatch.")
    if file_sha256(validator_dir / "validator.py") != frozen["validator_code_sha256"]:
        raise RuntimeError("Frozen validator code hash mismatch.")
    input_path = final_root / "inputs" / "L2_predictions_frozen.csv"
    frame = pd.read_csv(input_path).where(pd.notna, None)
    if any(column.startswith("gold_") for column in frame.columns):
        raise RuntimeError("Gold columns are forbidden during gate execution.")
    rules = load_rules(validator_dir / "validator_rules.yaml")
    output = pd.DataFrame([validate_record(row, rules) for row in frame.to_dict("records")])
    if len(output) != 60:
        raise RuntimeError("Expected 60 L3-Final gate records.")
    results_dir = final_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "E5_final_predictions.csv"
    output.to_csv(path, index=False, encoding="utf-8-sig")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", default=".")
    args = parser.parse_args()
    print(run(Path(args.workspace_root).resolve()))
