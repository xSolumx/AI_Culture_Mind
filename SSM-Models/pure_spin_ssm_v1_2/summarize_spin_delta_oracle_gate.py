"""Validate the frozen Spin-Delta oracle-address intervention."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

EXPECTED_SEEDS = [431, 433, 439]
EXPECTED_VARIANTS = ["learned_addresses", "oracle_addresses"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(paths: list[Path]) -> dict[str, object]:
    if len(paths) != 3:
        raise ValueError("the oracle intervention requires three artifacts")
    rows = []
    hashes = None
    intervention = None
    for path in paths:
        payload = json.loads(path.read_text())
        if payload["stage"] != "spin_delta_oracle_address_intervention":
            raise ValueError(f"wrong stage: {path}")
        if payload["variant_order"] != EXPECTED_VARIANTS:
            raise ValueError(f"wrong variant order: {path}")
        if not payload["pairing"]["all_parameters_bitwise_equal"]:
            raise ValueError(f"parameter mismatch: {path}")
        if hashes is None:
            hashes = payload["implementation_sha256"]
            intervention = payload["intervention"]
        if (
            payload["implementation_sha256"] != hashes
            or payload["intervention"] != intervention
        ):
            raise ValueError("implementation or intervention differs")
        learned, oracle = payload["results"]
        metrics = {}
        for writes in (8, 16, 32):
            learned_accuracy = learned["final"][str(writes)]["accuracy"]
            oracle_accuracy = oracle["final"][str(writes)]["accuracy"]
            if not all(math.isfinite(x) for x in (learned_accuracy, oracle_accuracy)):
                raise ValueError(f"nonfinite accuracy: {path}")
            metrics[str(writes)] = {
                "learned_accuracy": learned_accuracy,
                "oracle_accuracy": oracle_accuracy,
                "oracle_improvement": oracle_accuracy - learned_accuracy,
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
    oracle_8 = [row["metrics"]["8"]["oracle_accuracy"] for row in rows]
    oracle_16 = [row["metrics"]["16"]["oracle_accuracy"] for row in rows]
    improvements = [
        row["metrics"]["16"]["oracle_improvement"] for row in rows
    ]
    checks = {
        "oracle_id_accuracy": min(oracle_8) >= 0.95,
        "oracle_length16_accuracy": min(oracle_16) >= 0.95,
        "rescue_wins": sum(value >= 0.05 for value in improvements) >= 2,
        "mean_rescue": sum(improvements) / len(improvements) >= 0.05,
        "finite_compatible_and_paired": True,
    }
    return {
        "schema_version": 1,
        "claim_scope": "frozen oracle-address causal intervention decision",
        "seeds": EXPECTED_SEEDS,
        "rows": rows,
        "checks": checks,
        "oracle_capacity_pass": (
            checks["oracle_id_accuracy"] and checks["oracle_length16_accuracy"]
        ),
        "address_inference_bottleneck_pass": all(checks.values()),
        "implementation_sha256": hashes,
        "intervention": intervention,
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
