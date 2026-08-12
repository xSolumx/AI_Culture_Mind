"""Validate and aggregate independent hierarchical-retrieval seed artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hierarchical_matched_retrieval import summarize


def aggregate(paths: list[Path], expected_seeds: list[int]) -> dict[str, object]:
    if len(paths) != len(expected_seeds):
        raise ValueError("one independent artifact is required per expected seed")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    observed = [int(report["seeds"][0]) for report in reports]
    if sorted(observed) != sorted(expected_seeds) or len(set(observed)) != len(observed):
        raise ValueError(f"seed mismatch: expected {expected_seeds}, observed {observed}")
    if any(len(report["results"]) != 1 for report in reports):
        raise ValueError("every input must come from one independent seed process")
    if any(
        report["protocol"] != "HIERARCHICAL_MATCHED_RETRIEVAL_PREREGISTRATION.md"
        for report in reports
    ):
        raise ValueError("protocol mismatch")
    if any(not report["summary"]["implementation_gate_passed"] for report in reports):
        raise ValueError("an input implementation gate failed")
    rows = [report["results"][0] for report in reports]
    first_grid = reports[0]["grid"]
    if any(report["grid"] != first_grid for report in reports[1:]):
        raise ValueError("grid mismatch")
    return {
        "experiment": reports[0]["experiment"],
        "protocol": reports[0]["protocol"],
        "aggregation": "validated independent one-seed processes",
        "source_artifacts": [str(path).replace("\\", "/") for path in paths],
        "seeds": sorted(observed),
        "grid": first_grid,
        "results": sorted(rows, key=lambda row: int(row["seed"])),
        "summary": summarize(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--expected-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(args.inputs, args.expected_seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
