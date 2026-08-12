"""Aggregate and verify the large-slot semantic hierarchy cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from large_slot_semantic_hierarchy import verify_retained_parameters


def aggregate(
    rows: list[dict[str, object]],
    *,
    expected_seeds: list[int],
    verify_parameters: bool = True,
) -> dict[str, object]:
    seeds = [int(row["seed"]) for row in rows]
    if len(seeds) != len(set(seeds)) or sorted(seeds) != sorted(expected_seeds):
        raise ValueError(
            f"seed mismatch: expected {sorted(expected_seeds)}, got {sorted(seeds)}"
        )
    if any(
        row["experiment"] != "large-slot overlapping-semantic hierarchy"
        for row in rows
    ):
        raise ValueError("unexpected experiment identifier")
    verification = (
        [verify_retained_parameters(row) for row in rows]
        if verify_parameters
        else [
            row.get("retained_parameter_verification", {"passed": True})
            for row in rows
        ]
    )
    if any(not bool(item["passed"]) for item in verification):
        raise ValueError("retained-parameter verification failed")
    implementation_passes = sum(
        bool(row["decision"]["implementation_passed"]) for row in rows
    )
    shared_router_passes = sum(
        bool(row["decision"]["shared_router_completion_passed"]) for row in rows
    )
    hierarchy_passes = sum(
        bool(row["decision"]["hierarchical_routing_passed"]) for row in rows
    )

    def decision_values(name: str) -> list[float]:
        return [float(row["decision"][name]) for row in rows]

    shared_accuracies = decision_values("shared_heldout_hard_accuracy")
    independent_heldout = decision_values("independent_heldout_hard_accuracy")
    independent_observed = decision_values("independent_observed_hard_accuracy")
    direct_improvements = [
        float(row["decision"]["mean_block_improvement_over_dense"]["direct"])
        for row in rows
    ]
    delta_improvements = [
        float(row["decision"]["mean_block_improvement_over_dense"]["delta"])
        for row in rows
    ]
    required = 8 if len(rows) == 10 else max(1, (4 * len(rows) + 4) // 5)
    summary = {
        "seeds": len(rows),
        "implementation_passes": implementation_passes,
        "shared_router_completion_passes": shared_router_passes,
        "hierarchical_routing_passes": hierarchy_passes,
        "required_scientific_passes": required,
        "shared_router_completion_supported": (
            implementation_passes == len(rows) and shared_router_passes >= required
        ),
        "hierarchical_routing_supported": (
            implementation_passes == len(rows) and hierarchy_passes >= required
        ),
        "shared_heldout_hard_accuracy_mean": sum(shared_accuracies) / len(rows),
        "shared_heldout_hard_accuracy_range": [
            min(shared_accuracies),
            max(shared_accuracies),
        ],
        "independent_heldout_hard_accuracy_mean": (
            sum(independent_heldout) / len(rows)
        ),
        "independent_observed_hard_accuracy_mean": (
            sum(independent_observed) / len(rows)
        ),
        "direct_block_improvement_mean": sum(direct_improvements) / len(rows),
        "direct_block_improvement_range": [
            min(direct_improvements),
            max(direct_improvements),
        ],
        "delta_block_improvement_mean": sum(delta_improvements) / len(rows),
        "delta_block_improvement_range": [
            min(delta_improvements),
            max(delta_improvements),
        ],
        "maximum_canonicalization_error": max(
            float(row["decision"]["maximum_canonicalization_error"])
            for row in rows
        ),
        "maximum_hard_direct_delta_state_error": max(
            float(row["decision"]["maximum_hard_direct_delta_state_error"])
            for row in rows
        ),
        "maximum_hard_direct_delta_prediction_error": max(
            float(row["decision"]["maximum_hard_direct_delta_prediction_error"])
            for row in rows
        ),
        "maximum_retained_parameter_replay_difference": max(
            float(item["maximum_difference"]) for item in verification
        ),
        "claim_boundary": {
            "shared_router_result_is_storage_capacity": False,
            "hierarchy_result_is_triality_specific": False,
            "learned_action_discovery_established": False,
            "model_level_quality_established": False,
        },
    }
    return {
        "experiment": "large-slot overlapping-semantic hierarchy aggregate",
        "protocol": "LARGE_SLOT_SEMANTIC_HIERARCHY_PREREGISTRATION.md",
        "seeds": sorted(seeds),
        "summary": summary,
        "results": sorted(rows, key=lambda row: int(row["seed"])),
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
