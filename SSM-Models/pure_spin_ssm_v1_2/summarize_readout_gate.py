"""Validate and decide the preregistered three-seed triality-readout gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

EXPECTED_SEEDS = (71, 73, 79)
EXPECTED_READOUTS = ("direction", "triality_invariants")
MINIMUM_WINS = 2
MINIMUM_MEAN_IMPROVEMENT = 0.0100
MAXIMUM_SINGLE_SEED_REGRESSION = 0.0500


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compatibility_key(report: dict[str, Any]) -> dict[str, Any]:
    config = {key: value for key, value in report["config"].items() if key != "seed"}
    return {
        "config_excluding_seed": config,
        "readouts": report["readouts"],
        "parameter_counts": report["parameter_counts"],
        "dataset": report["dataset"],
        "environment": report["environment"],
        "implementation_sha256": report["implementation_sha256"],
    }


def summarize(paths: list[Path]) -> dict[str, Any]:
    reports = [json.loads(path.read_text()) for path in paths]
    seeds = tuple(report["config"]["seed"] for report in reports)
    if tuple(sorted(seeds)) != EXPECTED_SEEDS or len(set(seeds)) != len(seeds):
        raise ValueError(f"expected exactly seeds {EXPECTED_SEEDS}, received {seeds}")
    expected = _compatibility_key(reports[0])
    if tuple(expected["readouts"]) != EXPECTED_READOUTS:
        raise ValueError(f"unexpected readout order: {expected['readouts']}")
    for path, report in zip(paths[1:], reports[1:]):
        if _compatibility_key(report) != expected:
            raise ValueError(f"incompatible readout-gate artifact: {path}")

    rows = []
    improvements = []
    for path, report in sorted(
        zip(paths, reports), key=lambda pair: pair[1]["config"]["seed"]
    ):
        by_readout = {row["readout"]: row for row in report["results"]}
        if set(by_readout) != set(EXPECTED_READOUTS):
            raise ValueError(f"unexpected result set in {path}: {sorted(by_readout)}")
        baseline = by_readout["direction"]["final_bits_per_byte"]
        candidate = by_readout["triality_invariants"]["final_bits_per_byte"]
        improvement = baseline - candidate
        improvements.append(improvement)
        rows.append(
            {
                "seed": report["config"]["seed"],
                "artifact": path.as_posix(),
                "sha256": sha256(path),
                "direction_bits_per_byte": baseline,
                "triality_invariants_bits_per_byte": candidate,
                "improvement_bits_per_byte": improvement,
            }
        )

    wins = sum(value > 0.0 for value in improvements)
    mean_improvement = statistics.fmean(improvements)
    criteria = {
        "minimum_two_of_three_wins": wins >= MINIMUM_WINS,
        "mean_improvement_at_least_0_0100": (
            mean_improvement >= MINIMUM_MEAN_IMPROVEMENT
        ),
        "no_regression_worse_than_0_0500": (
            min(improvements) >= -MAXIMUM_SINGLE_SEED_REGRESSION
        ),
        "finite_results": all(math.isfinite(value) for value in improvements),
    }
    passed = all(criteria.values())
    return {
        "schema_version": 1,
        "summary_implementation_sha256": sha256(Path(__file__)),
        "claim_scope": (
            "prospective internal v1.2 readout gate; not a Mamba comparison "
            "or speed claim"
        ),
        "compatibility": expected,
        "runs": rows,
        "decision": {
            "candidate": "triality_invariants",
            "baseline": "direction",
            "wins": wins,
            "improvements_bits_per_byte": improvements,
            "mean_improvement_bits_per_byte": mean_improvement,
            "criteria": criteria,
            "passed": passed,
            "verdict": (
                "promote_triality_invariants" if passed else "retain_direction_default"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.artifacts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
