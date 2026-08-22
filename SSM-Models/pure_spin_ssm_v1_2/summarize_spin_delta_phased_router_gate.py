"""Validate the frozen Spin-Delta phase-separated router gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

EXPECTED_SEEDS = [467, 479, 487]
EXPECTED_VARIANTS = ["joint_schedule", "phase_separated_schedule"]
WRITES = (8, 16, 32)
ROUTER_METRICS = (
    "write_event_f1",
    "query_event_f1",
    "write_slot_accuracy",
    "query_slot_accuracy",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(paths: list[Path]) -> dict[str, object]:
    if len(paths) != 3:
        raise ValueError("the phased-router gate requires three artifacts")
    rows = []
    hashes = None
    intervention = None
    for path in paths:
        payload = json.loads(path.read_text())
        if payload["stage"] != "spin_delta_phased_router_gate":
            raise ValueError(f"wrong stage: {path}")
        if payload["variant_order"] != EXPECTED_VARIANTS:
            raise ValueError(f"wrong variant order: {path}")
        if not payload["pairing"]["all_initial_tensors_bitwise_equal"]:
            raise ValueError(f"initial tensor mismatch: {path}")
        if hashes is None:
            hashes = payload["implementation_sha256"]
            intervention = payload["intervention"]
        if (
            payload["implementation_sha256"] != hashes
            or payload["intervention"] != intervention
        ):
            raise ValueError("implementation or intervention differs")
        joint, phased = payload["results"]
        if (
            phased["phase_a_core_untouched"] is not True
            or phased["router_frozen_during_phase_b"] is not True
            or phased["autonomous_evaluation"] is not True
        ):
            raise ValueError(f"phase separation contract failed: {path}")
        metrics = {}
        readiness = {}
        for writes in WRITES:
            joint_accuracy = joint["final"][str(writes)]["accuracy"]
            phased_accuracy = phased["final"][str(writes)]["accuracy"]
            router = phased["phase_a_router"][str(writes)]["router"]
            values = (joint_accuracy, phased_accuracy) + tuple(
                router[name] for name in ROUTER_METRICS
            )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"nonfinite metric: {path}")
            metrics[str(writes)] = {
                "joint_accuracy": joint_accuracy,
                "phased_accuracy": phased_accuracy,
                "phased_improvement": phased_accuracy - joint_accuracy,
            }
            readiness[str(writes)] = {
                name: router[name] for name in ROUTER_METRICS
            }
        rows.append(
            {
                "seed": payload["config"]["seed"],
                "metrics": metrics,
                "phase_a_router": readiness,
                "source": path.as_posix(),
                "source_sha256": sha256(path),
            }
        )
    rows.sort(key=lambda row: row["seed"])
    if [row["seed"] for row in rows] != EXPECTED_SEEDS:
        raise ValueError(f"expected frozen seeds {EXPECTED_SEEDS}")
    phased_8 = [row["metrics"]["8"]["phased_accuracy"] for row in rows]
    phased_16 = [row["metrics"]["16"]["phased_accuracy"] for row in rows]
    phased_32 = [row["metrics"]["32"]["phased_accuracy"] for row in rows]
    improvements = [
        row["metrics"]["16"]["phased_improvement"] for row in rows
    ]
    router_values = [
        row["phase_a_router"][str(writes)][metric]
        for row in rows
        for writes in WRITES
        for metric in ROUTER_METRICS
    ]
    checks = {
        "phased_8_accuracy": min(phased_8) >= 0.95,
        "phased_16_accuracy": min(phased_16) >= 0.95,
        "phased_32_accuracy": min(phased_32) >= 0.93,
        "router_readiness_all_lengths": min(router_values) >= 0.99,
        "phased_rescue_wins": sum(value >= 0.05 for value in improvements) >= 2,
        "mean_phased_rescue": sum(improvements) / len(improvements) >= 0.05,
        "finite_matched_and_phase_contract": True,
    }
    return {
        "schema_version": 1,
        "claim_scope": "frozen joint-versus-phase-separated router decision",
        "seeds": EXPECTED_SEEDS,
        "rows": rows,
        "checks": checks,
        "phase_separated_capacity_pass": all(
            checks[name]
            for name in (
                "phased_8_accuracy",
                "phased_16_accuracy",
                "phased_32_accuracy",
            )
        ),
        "router_readiness_pass": checks["router_readiness_all_lengths"],
        "coadaptation_bottleneck_pass": (
            checks["phased_rescue_wins"] and checks["mean_phased_rescue"]
        ),
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
