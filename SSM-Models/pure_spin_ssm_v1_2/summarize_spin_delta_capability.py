"""Validate the frozen Spin-Delta overwrite capability cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

EXPECTED_SEEDS = [401, 409, 419]
EXPECTED_VARIANTS = ["independent_v1_2", "spin_delta"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(paths: list[Path]) -> dict[str, object]:
    if len(paths) != 3:
        raise ValueError("the frozen capability gate requires three artifacts")
    rows = []
    hashes = None
    task = None
    for path in paths:
        payload = json.loads(path.read_text())
        if payload["stage"] != "spin_delta_overwrite_capability":
            raise ValueError(f"wrong capability stage: {path}")
        if payload["variant_order"] != EXPECTED_VARIANTS:
            raise ValueError(f"wrong variant order: {path}")
        pairing = payload["pairing"]
        if not pairing["common_parameters_bitwise_equal"]:
            raise ValueError(f"common parameter mismatch: {path}")
        if pairing["maximum_absolute_logit_difference"] > 2.0e-6:
            raise ValueError(f"pairing residual exceeds bound: {path}")
        if hashes is None:
            hashes = payload["implementation_sha256"]
            task = payload["task"]
        if payload["implementation_sha256"] != hashes or payload["task"] != task:
            raise ValueError("implementation or task identity differs")
        baseline, candidate = payload["results"]
        metrics = {}
        for writes in (8, 16, 32):
            base_accuracy = baseline["final"][str(writes)]["accuracy"]
            delta_accuracy = candidate["final"][str(writes)]["accuracy"]
            if not all(math.isfinite(x) for x in (base_accuracy, delta_accuracy)):
                raise ValueError(f"nonfinite accuracy: {path}")
            metrics[str(writes)] = {
                "independent_v1_2_accuracy": base_accuracy,
                "spin_delta_accuracy": delta_accuracy,
                "improvement": delta_accuracy - base_accuracy,
            }
        rows.append(
            {
                "seed": payload["config"]["seed"],
                "metrics": metrics,
                "source": path.as_posix(),
                "source_sha256": sha256(path),
            }
        )
    rows.sort(key=lambda row: row["seed"])
    if [row["seed"] for row in rows] != EXPECTED_SEEDS:
        raise ValueError(f"expected frozen seeds {EXPECTED_SEEDS}")
    delta_id = [row["metrics"]["8"]["spin_delta_accuracy"] for row in rows]
    delta_16 = [row["metrics"]["16"]["spin_delta_accuracy"] for row in rows]
    improvements_16 = [row["metrics"]["16"]["improvement"] for row in rows]
    checks = {
        "candidate_id_accuracy": min(delta_id) >= 0.90,
        "candidate_length16_accuracy": min(delta_16) >= 0.75,
        "length16_differential_wins": sum(x >= 0.05 for x in improvements_16) >= 2,
        "length16_mean_improvement": (
            sum(improvements_16) / len(improvements_16) >= 0.05
        ),
        "finite_compatible_and_paired": True,
    }
    return {
        "schema_version": 1,
        "claim_scope": "frozen Spin-Delta overwrite capability decision",
        "seeds": EXPECTED_SEEDS,
        "rows": rows,
        "checks": checks,
        "candidate_capability_pass": (
            checks["candidate_id_accuracy"]
            and checks["candidate_length16_accuracy"]
        ),
        "differential_advantage_pass": all(checks.values()),
        "implementation_sha256": hashes,
        "task": task,
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
