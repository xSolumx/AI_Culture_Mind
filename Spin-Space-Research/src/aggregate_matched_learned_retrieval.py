"""Validate and aggregate per-seed matched-retrieval artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matched_learned_retrieval import summarize


def aggregate(paths: list[Path]) -> dict[str, object]:
    if not paths:
        raise ValueError("at least one input artifact is required")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    reference = reports[0]
    for path, report in zip(paths[1:], reports[1:], strict=True):
        for key in ("experiment", "protocol", "device", "dtype", "grid"):
            if report[key] != reference[key]:
                raise ValueError(f"{path} has a mismatched {key}")
    rows = [row for report in reports for row in report["results"]]
    seeds = [int(row["seed"]) for row in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError("input artifacts contain duplicate seeds")
    order = sorted(range(len(rows)), key=lambda index: seeds[index])
    rows = [rows[index] for index in order]
    seeds = [int(row["seed"]) for row in rows]
    return {
        "experiment": reference["experiment"],
        "protocol": reference["protocol"],
        "device": reference["device"],
        "dtype": reference["dtype"],
        "seeds": seeds,
        "grid": reference["grid"],
        "results": rows,
        "summary": summarize(rows),
        "aggregation": {
            "validated_fields": [
                "experiment",
                "protocol",
                "device",
                "dtype",
                "grid",
            ],
            "source_artifacts": [str(path) for path in paths],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
