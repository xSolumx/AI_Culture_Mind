"""Audit and summarize the frozen Spin-Delta quality gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

EXPECTED_SEEDS = [353, 359, 367]
EXPECTED_VARIANTS = ["independent_v1_2", "spin_delta"]
MAXIMUM_INITIAL_LOGIT_DIFFERENCE = 1.0e-6


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(paths: list[Path]) -> dict[str, object]:
    if len(paths) != len(EXPECTED_SEEDS):
        raise ValueError("the frozen gate requires exactly three artifacts")
    rows = []
    reference_hashes = None
    reference_dataset = None
    for path in paths:
        payload = json.loads(path.read_text())
        if payload["stage"] != "spin_delta":
            raise ValueError(f"not a Spin-Delta artifact: {path}")
        if payload["variant_order"] != EXPECTED_VARIANTS:
            raise ValueError(f"unexpected variant order: {path}")
        pairing = payload["initial_pairing"]
        if not pairing["common_parameters_bitwise_equal"]:
            raise ValueError(f"common parameter mismatch: {path}")
        if (
            pairing["maximum_absolute_logit_difference"]
            > MAXIMUM_INITIAL_LOGIT_DIFFERENCE
        ):
            raise ValueError(f"initial logit pairing bound exceeded: {path}")
        if reference_hashes is None:
            reference_hashes = payload["implementation_sha256"]
            reference_dataset = payload["dataset"]
        if payload["implementation_sha256"] != reference_hashes:
            raise ValueError("implementation hashes differ across artifacts")
        if payload["dataset"] != reference_dataset:
            raise ValueError("dataset identity differs across artifacts")
        baseline, candidate = payload["results"]
        values = (
            baseline["final_bits_per_byte"],
            candidate["final_bits_per_byte"],
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"nonfinite result: {path}")
        improvement = values[0] - values[1]
        rows.append(
            {
                "seed": payload["config"]["seed"],
                "independent_v1_2_bits_per_byte": values[0],
                "spin_delta_bits_per_byte": values[1],
                "improvement_bits_per_byte": improvement,
                "candidate_wins": improvement > 0.0,
                "source": path.as_posix(),
                "source_sha256": sha256(path),
            }
        )
    rows.sort(key=lambda row: row["seed"])
    if [row["seed"] for row in rows] != EXPECTED_SEEDS:
        raise ValueError(f"expected frozen seeds {EXPECTED_SEEDS}")
    improvements = [row["improvement_bits_per_byte"] for row in rows]
    wins = sum(row["candidate_wins"] for row in rows)
    mean = sum(improvements) / len(improvements)
    thresholds = {
        "minimum_wins": 2,
        "minimum_mean_improvement_bits_per_byte": 0.01,
        "maximum_single_seed_regression_bits_per_byte": 0.05,
        "maximum_initial_logit_difference": MAXIMUM_INITIAL_LOGIT_DIFFERENCE,
    }
    checks = {
        "wins": wins >= thresholds["minimum_wins"],
        "mean_improvement": mean
        >= thresholds["minimum_mean_improvement_bits_per_byte"],
        "worst_regression": min(improvements)
        >= -thresholds["maximum_single_seed_regression_bits_per_byte"],
        "finite_compatible_and_paired": True,
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "claim_scope": "frozen Spin-Delta quality decision",
        "seeds": EXPECTED_SEEDS,
        "rows": rows,
        "wins": wins,
        "mean_improvement_bits_per_byte": mean,
        "worst_improvement_bits_per_byte": min(improvements),
        "best_improvement_bits_per_byte": max(improvements),
        "thresholds": thresholds,
        "checks": checks,
        "quality_pass": passed,
        "speed_gate_authorized": passed,
        "implementation_sha256": reference_hashes,
        "dataset": reference_dataset,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
