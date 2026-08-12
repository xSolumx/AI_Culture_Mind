"""Aggregate and verify prospective Task-B paired-action replication rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from task_b_paired_action_replication import verify_retained_parameters


def aggregate(
    rows: list[dict[str, object]],
    *,
    expected_seeds: list[int],
    verify_parameters: bool = True,
) -> dict[str, object]:
    seeds = [int(row["seed"]) for row in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seed artifact")
    if sorted(seeds) != sorted(expected_seeds):
        raise ValueError(f"expected seeds {sorted(expected_seeds)}, received {sorted(seeds)}")
    rows = sorted(rows, key=lambda row: int(row["seed"]))
    if any(
        row["experiment"] != "Task-B prospective paired-action replication"
        for row in rows
    ):
        raise ValueError("unexpected experiment type")
    if any(not bool(row["dense"]) for row in rows):
        raise ValueError("frozen aggregate requires dense length sweeps")
    verification = (
        [verify_retained_parameters(row) for row in rows]
        if verify_parameters
        else [row.get("retained_parameter_verification", {"passed": True}) for row in rows]
    )
    if any(not bool(item["passed"]) for item in verification):
        raise ValueError("retained-parameter verification failed")
    implementation_passes = sum(
        bool(row["decision"]["implementation_passed"]) for row in rows
    )
    prior_wins = sum(
        bool(row["decision"]["representation_prior_win"]) for row in rows
    )
    required_wins = min(8, len(rows))
    shared = [
        float(row["decision"]["shared_delta_length2048_mean_cosine"])
        for row in rows
    ]
    independent = [
        float(row["decision"]["independent_delta_length2048_mean_cosine"])
        for row in rows
    ]
    matched = [
        float(
            row["decision"][
                "routing_matched_independent_delta_length2048_mean_cosine"
            ]
        )
        for row in rows
    ]
    return {
        "experiment": "Task-B prospective paired-action replication aggregate",
        "protocol": "TASK_B_PAIRED_ACTION_REPLICATION_PREREGISTRATION.md",
        "expected_seeds": sorted(expected_seeds),
        "rows": rows,
        "retained_parameter_verification": verification,
        "summary": {
            "seeds": len(rows),
            "implementation_passes": implementation_passes,
            "representation_prior_wins": prior_wins,
            "required_representation_prior_wins": required_wins,
            "task_b_decision_rule_fully_empirically_closed": (
                implementation_passes == len(rows) and prior_wins >= required_wins
            ),
            "shared_delta_length2048_mean_across_seeds": sum(shared) / len(shared),
            "shared_delta_length2048_worst_seed": min(shared),
            "independent_delta_length2048_mean_across_seeds": (
                sum(independent) / len(independent)
            ),
            "independent_delta_length2048_best_seed": max(independent),
            "routing_matched_independent_delta_length2048_mean": (
                sum(matched) / len(matched)
            ),
            "routing_matched_independent_delta_length2048_best_seed": max(matched),
            "maximum_direct_delta_error": max(
                float(row["decision"]["maximum_direct_delta_error"]) for row in rows
            ),
            "maximum_retained_parameter_replay_difference": max(
                float(item["maximum_metric_difference"]) for item in verification
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--expected-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    report = aggregate(rows, expected_seeds=args.expected_seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
