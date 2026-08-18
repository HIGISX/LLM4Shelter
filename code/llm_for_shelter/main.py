"""Command-line runner for the complete deterministic MHA-PM experiment suite."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data_loader import load_experiment_data
from .experiments import run_all_experiments
from .export import export_results, write_experiment_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LLMforShelter MHA-PM experiments.")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--results-dir", default="results")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = Path(args.results_dir).resolve()
    results.mkdir(parents=True, exist_ok=True)
    data = load_experiment_data(args.workspace_root, results / "schema_mapping.json")
    outputs = run_all_experiments(data)
    export_results(data, outputs, results)
    write_experiment_report(data, outputs, results / "EXPERIMENT_REPORT.md")
    print(f"Completed {len(outputs.summaries)} optimal MHA-PM scenarios. Results: {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
