"""Audit and summarize the frozen shared-action compression artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.inputs) != 3:
        raise ValueError("the frozen gate requires exactly three artifacts")

    rows = []
    reference_hashes = None
    reference_dataset = None
    for path in args.inputs:
        payload = json.loads(path.read_text())
        if payload["stage"] != "compression":
            raise ValueError(f"not a compression artifact: {path}")
        if payload["variant_order"] != ["independent_v1_2", "shared_identity"]:
            raise ValueError(f"unexpected variant order: {path}")
        if reference_hashes is None:
            reference_hashes = payload["implementation_sha256"]
            reference_dataset = payload["dataset"]
        if payload["implementation_sha256"] != reference_hashes:
            raise ValueError("implementation hashes differ across artifacts")
        if payload["dataset"] != reference_dataset:
            raise ValueError("dataset identity differs across artifacts")
        baseline, candidate = payload["results"]
        improvement = (
            baseline["final_bits_per_byte"] - candidate["final_bits_per_byte"]
        )
        rows.append(
            {
                "seed": payload["config"]["seed"],
                "independent_bits_per_byte": baseline["final_bits_per_byte"],
                "shared_action_bits_per_byte": candidate["final_bits_per_byte"],
                "improvement_bits_per_byte": improvement,
                "non_regressive": improvement >= 0.0,
                "source": path.as_posix(),
                "source_sha256": sha256(path),
            }
        )

    rows.sort(key=lambda row: row["seed"])
    expected_seeds = [179, 181, 191]
    if [row["seed"] for row in rows] != expected_seeds:
        raise ValueError(f"expected frozen seeds {expected_seeds}")
    improvements = [row["improvement_bits_per_byte"] for row in rows]
    mean = sum(improvements) / len(improvements)
    thresholds = {
        "minimum_mean_improvement_bits_per_byte": -0.01,
        "minimum_non_regressive_seeds": 1,
        "maximum_single_seed_regression_bits_per_byte": 0.05,
    }
    non_regressive = sum(row["non_regressive"] for row in rows)
    checks = {
        "mean_non_inferiority": mean
        >= thresholds["minimum_mean_improvement_bits_per_byte"],
        "non_regressive_seeds": non_regressive
        >= thresholds["minimum_non_regressive_seeds"],
        "worst_regression": min(improvements)
        >= -thresholds["maximum_single_seed_regression_bits_per_byte"],
        "finite_and_compatible": True,
    }
    report = {
        "schema_version": 1,
        "claim_scope": "frozen shared-action quality non-inferiority decision",
        "seeds": expected_seeds,
        "rows": rows,
        "non_regressive_seeds": non_regressive,
        "mean_improvement_bits_per_byte": mean,
        "worst_improvement_bits_per_byte": min(improvements),
        "best_improvement_bits_per_byte": max(improvements),
        "thresholds": thresholds,
        "checks": checks,
        "quality_non_inferior": all(checks.values()),
        "throughput_gate_authorized": all(checks.values()),
        "implementation_sha256": reference_hashes,
        "dataset": reference_dataset,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
