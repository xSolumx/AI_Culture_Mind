"""Audit and summarize frozen coupled-isotypic paired gate artifacts."""

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
        raise ValueError("the frozen gate requires exactly three seed artifacts")

    rows = []
    reference_hashes = None
    reference_dataset = None
    for path in args.inputs:
        payload = json.loads(path.read_text())
        if payload["stage"] != "multiplicity":
            raise ValueError(f"not a Stage A artifact: {path}")
        if payload["variant_order"] != ["shared_identity", "shared_orthogonal"]:
            raise ValueError(f"unexpected variant order: {path}")
        if reference_hashes is None:
            reference_hashes = payload["implementation_sha256"]
            reference_dataset = payload["dataset"]
        if payload["implementation_sha256"] != reference_hashes:
            raise ValueError("implementation hashes differ across artifacts")
        if payload["dataset"] != reference_dataset:
            raise ValueError("dataset identity differs across artifacts")
        baseline, candidate = payload["results"]
        if baseline["initial_nats_per_byte"] != candidate["initial_nats_per_byte"]:
            raise ValueError(f"initial losses are not exactly paired: {path}")
        improvement = (
            baseline["final_bits_per_byte"] - candidate["final_bits_per_byte"]
        )
        rows.append(
            {
                "seed": payload["config"]["seed"],
                "shared_identity_bits_per_byte": baseline["final_bits_per_byte"],
                "shared_orthogonal_bits_per_byte": candidate["final_bits_per_byte"],
                "improvement_bits_per_byte": improvement,
                "candidate_wins": improvement > 0.0,
                "source": path.as_posix(),
                "source_sha256": sha256(path),
            }
        )

    rows.sort(key=lambda row: row["seed"])
    expected_seeds = [149, 151, 157]
    if [row["seed"] for row in rows] != expected_seeds:
        raise ValueError(f"expected frozen seeds {expected_seeds}")
    improvements = [row["improvement_bits_per_byte"] for row in rows]
    wins = sum(row["candidate_wins"] for row in rows)
    mean = sum(improvements) / len(improvements)
    thresholds = {
        "minimum_wins": 2,
        "minimum_mean_improvement_bits_per_byte": 0.01,
        "maximum_single_seed_regression_bits_per_byte": 0.05,
    }
    checks = {
        "wins": wins >= thresholds["minimum_wins"],
        "mean_improvement": mean
        >= thresholds["minimum_mean_improvement_bits_per_byte"],
        "worst_regression": min(improvements)
        >= -thresholds["maximum_single_seed_regression_bits_per_byte"],
        "finite_and_compatible": True,
    }
    report = {
        "schema_version": 1,
        "claim_scope": "frozen Stage A recurrent multiplicity promotion decision",
        "seeds": expected_seeds,
        "rows": rows,
        "wins": wins,
        "mean_improvement_bits_per_byte": mean,
        "worst_improvement_bits_per_byte": min(improvements),
        "best_improvement_bits_per_byte": max(improvements),
        "thresholds": thresholds,
        "checks": checks,
        "promote": all(checks.values()),
        "conditional_stage_b_authorized": all(checks.values()),
        "implementation_sha256": reference_hashes,
        "dataset": reference_dataset,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
