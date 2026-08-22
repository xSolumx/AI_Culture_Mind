"""Validate the frozen Spin-Delta causal-router gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

EXPECTED_SEEDS = [449, 457, 461]
EXPECTED_VARIANTS = ["learned_continuous", "causal_discrete_aux"]
WRITES = (8, 16, 32)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(paths: list[Path]) -> dict[str, object]:
    if len(paths) != 3:
        raise ValueError("the causal-router gate requires three artifacts")
    rows = []
    hashes = None
    intervention = None
    candidate_extra_parameters = None
    for path in paths:
        payload = json.loads(path.read_text())
        if payload["stage"] != "spin_delta_causal_router_gate":
            raise ValueError(f"wrong stage: {path}")
        if payload["variant_order"] != EXPECTED_VARIANTS:
            raise ValueError(f"wrong variant order: {path}")
        pairing = payload["pairing"]
        if not pairing["all_core_tensors_bitwise_equal"]:
            raise ValueError(f"core parameter mismatch: {path}")
        if hashes is None:
            hashes = payload["implementation_sha256"]
            intervention = payload["intervention"]
            candidate_extra_parameters = pairing["candidate_extra_parameters"]
        if (
            payload["implementation_sha256"] != hashes
            or payload["intervention"] != intervention
            or pairing["candidate_extra_parameters"] != candidate_extra_parameters
        ):
            raise ValueError("implementation, intervention, or parameter count differs")
        baseline, candidate = payload["results"]
        if candidate["autonomous_evaluation"] is not True:
            raise ValueError(f"candidate evaluation is not autonomous: {path}")
        metrics = {}
        for writes in WRITES:
            baseline_accuracy = baseline["final"][str(writes)]["accuracy"]
            candidate_row = candidate["final"][str(writes)]
            candidate_accuracy = candidate_row["accuracy"]
            router = candidate_row["router"]
            values = (
                baseline_accuracy,
                candidate_accuracy,
                router["write_event_f1"],
                router["query_event_f1"],
                router["write_slot_accuracy"],
                router["query_slot_accuracy"],
            )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"nonfinite metric: {path}")
            metrics[str(writes)] = {
                "baseline_accuracy": baseline_accuracy,
                "candidate_accuracy": candidate_accuracy,
                "candidate_improvement": candidate_accuracy - baseline_accuracy,
                "router": {
                    name: router[name]
                    for name in (
                        "write_event_f1",
                        "query_event_f1",
                        "write_slot_accuracy",
                        "query_slot_accuracy",
                    )
                },
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
    candidate_8 = [row["metrics"]["8"]["candidate_accuracy"] for row in rows]
    candidate_16 = [row["metrics"]["16"]["candidate_accuracy"] for row in rows]
    candidate_32 = [row["metrics"]["32"]["candidate_accuracy"] for row in rows]
    improvements = [
        row["metrics"]["16"]["candidate_improvement"] for row in rows
    ]
    router_values = [
        row["metrics"][str(writes)]["router"][metric]
        for row in rows
        for writes in WRITES
        for metric in (
            "write_event_f1",
            "query_event_f1",
            "write_slot_accuracy",
            "query_slot_accuracy",
        )
    ]
    checks = {
        "candidate_8_accuracy": min(candidate_8) >= 0.95,
        "candidate_16_accuracy": min(candidate_16) >= 0.95,
        "candidate_32_accuracy": min(candidate_32) >= 0.93,
        "router_identification_all_lengths": min(router_values) >= 0.99,
        "rescue_wins": sum(value >= 0.05 for value in improvements) >= 2,
        "mean_rescue": sum(improvements) / len(improvements) >= 0.05,
        "finite_compatible_and_paired": True,
    }
    return {
        "schema_version": 1,
        "claim_scope": "frozen autonomous causal-router decision",
        "seeds": EXPECTED_SEEDS,
        "rows": rows,
        "checks": checks,
        "autonomous_retrieval_capacity_pass": all(
            checks[name]
            for name in (
                "candidate_8_accuracy",
                "candidate_16_accuracy",
                "candidate_32_accuracy",
            )
        ),
        "router_identification_pass": checks[
            "router_identification_all_lengths"
        ],
        "robust_rescue_pass": (
            checks["rescue_wins"] and checks["mean_rescue"]
        ),
        "candidate_extra_parameters": candidate_extra_parameters,
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
