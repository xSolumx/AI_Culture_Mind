"""Validate and aggregate independent Task-B delta-action replay artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def aggregate(
    rows: list[dict[str, object]], *, expected_seeds: list[int]
) -> dict[str, object]:
    seeds = [int(row["seed"]) for row in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seed artifact")
    if sorted(seeds) != sorted(expected_seeds):
        raise ValueError(f"expected seeds {sorted(expected_seeds)}, received {sorted(seeds)}")
    rows = sorted(rows, key=lambda row: int(row["seed"]))
    if any(row["experiment"] != "Task-B independent-action delta replay" for row in rows):
        raise ValueError("unexpected experiment type")
    if any(not bool(row["dense"]) for row in rows):
        raise ValueError("frozen aggregate requires dense length sweeps")
    implementation_passes = sum(
        bool(row["decision"]["implementation_passed"]) for row in rows
    )
    representation_prior_wins = sum(
        bool(row["decision"]["representation_prior_win"]) for row in rows
    )
    required_wins = min(8, len(rows))
    independent_long = [
        float(row["decision"]["independent_delta_length2048_mean_cosine"])
        for row in rows
    ]
    shared_long = [
        float(row["decision"]["shared_direct_length2048_mean_cosine"])
        for row in rows
    ]
    return {
        "experiment": "Task-B independent-action delta replay aggregate",
        "protocol": "TASK_B_DELTA_ACTION_REPLAY_PREREGISTRATION.md",
        "expected_seeds": sorted(expected_seeds),
        "rows": rows,
        "summary": {
            "seeds": len(rows),
            "implementation_passes": implementation_passes,
            "representation_prior_wins": representation_prior_wins,
            "required_representation_prior_wins": required_wins,
            "task_b_decision_rule_fully_empirically_closed": (
                implementation_passes == len(rows)
                and representation_prior_wins >= required_wins
            ),
            "independent_delta_length2048_mean_across_seeds": (
                sum(independent_long) / len(independent_long)
            ),
            "independent_delta_length2048_worst_seed": min(independent_long),
            "independent_delta_length2048_best_seed": max(independent_long),
            "shared_direct_length2048_mean_across_seeds": (
                sum(shared_long) / len(shared_long)
            ),
            "maximum_direct_delta_state_error": max(
                float(row["decision"]["maximum_direct_delta_state_error"])
                for row in rows
            ),
            "maximum_direct_delta_prediction_error": max(
                float(row["decision"]["maximum_direct_delta_prediction_error"])
                for row in rows
            ),
            "minimum_write_route_agreement": min(
                float(row["decision"]["minimum_write_route_agreement"])
                for row in rows
            ),
            "minimum_query_route_agreement": min(
                float(row["decision"]["minimum_query_route_agreement"])
                for row in rows
            ),
            "minimum_oracle_delta_query_cosine": min(
                float(row["decision"]["minimum_oracle_delta_query_cosine"])
                for row in rows
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
