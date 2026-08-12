"""Audit endpoint-only coverage, query budgets, gauges, and failure controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_recurrences import GROUPS, parse_input_elements
from endpoint_group_discovery import (
    GroupEndpointOracle,
    endpoint_transition_cover,
    infer_four_token_inverse_matching,
    passive_representatives,
    recover_from_endpoint_queries,
)


INPUT_LABELS = ("23145", "31245", "23451", "51234")


def exact_isomorphism(recovered, true_table: np.ndarray, unmap) -> bool:
    labels = np.asarray([unmap(int(label)) for label in recovered.element_to_state])
    return bool(
        np.array_equal(
            labels[recovered.group.table],
            true_table[labels[:, None], labels[None, :]],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=1_000)
    parser.add_argument("--gauge-seeds", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    group = GROUPS["a5"]
    inputs = parse_input_elements(INPUT_LABELS, group)

    cover_times = []
    exact_recoveries = 0
    inverse_recoveries = 0
    query_counts = []
    for seed in range(args.seeds):
        oracle = GroupEndpointOracle(group, inputs)
        recovered, report = recover_from_endpoint_queries(
            oracle.query,
            state_count=group.order,
            token_count=len(inputs),
            passive_samples=1_024,
            passive_word_length=16,
            seed=50_000 + seed,
        )
        cover_times.append(report.samples_until_all_states)
        inverse_recoveries += int(report.inferred_inverse_tokens == (1, 0, 3, 2))
        exact_recoveries += int(
            exact_isomorphism(recovered, group.table, lambda label: label)
        )
        query_counts.append(oracle.query_count)

    gauge_passes = 0
    distinct_identity_labels = set()
    for seed in range(args.gauge_seeds):
        permutation = np.random.default_rng(70_000 + seed).permutation(group.order)
        inverse_permutation = np.empty(group.order, dtype=np.int64)
        inverse_permutation[permutation] = np.arange(group.order)
        oracle = GroupEndpointOracle(group, inputs)

        def remapped_query(word):
            return int(permutation[oracle.query(word)])

        recovered, report = recover_from_endpoint_queries(
            remapped_query,
            state_count=group.order,
            token_count=len(inputs),
            passive_samples=1_024,
            passive_word_length=16,
            seed=80_000 + seed,
        )
        distinct_identity_labels.add(report.identity_label)
        gauge_passes += int(
            exact_isomorphism(
                recovered,
                group.table,
                lambda label: int(inverse_permutation[label]),
            )
        )

    missing_extension_refusals = 0
    for seed in range(100):
        oracle = GroupEndpointOracle(group, inputs)
        identity = oracle.query(())
        representatives, _ = passive_representatives(
            oracle.query,
            token_count=4,
            state_count=group.order,
            samples=1_024,
            word_length=16,
            seed=90_000 + seed,
        )
        inverse = infer_four_token_inverse_matching(oracle.query, identity)
        queried_tokens = (0, 2)
        omitted_label = seed % group.order
        omitted_token = queried_tokens[(seed // group.order) % 2]
        try:
            endpoint_transition_cover(
                oracle.query,
                representatives,
                inverse,
                state_count=group.order,
                omit_extension=(omitted_label, omitted_token),
            )
        except ValueError:
            missing_extension_refusals += 1

    report = {
        "experiment": "endpoint-only finite-action supervision audit",
        "passive_protocol": {
            "samples": 1_024,
            "word_length": 16,
            "endpoint_labels_only": True,
            "seeds": args.seeds,
            "all_state_coverage_passes": len(cover_times),
            "samples_until_all_states": {
                "minimum": min(cover_times),
                "mean": float(np.mean(cover_times)),
                "maximum": max(cover_times),
            },
        },
        "active_protocol": {
            "identity_queries": 1,
            "inverse_pair_queries": 3,
            "extension_queries": 120,
            "total_active_queries": 124,
            "total_endpoint_labels_including_passive": 1_148,
            "unique_query_counts": sorted(set(query_counts)),
        },
        "exact_inverse_recoveries": inverse_recoveries,
        "exact_action_recoveries": exact_recoveries,
        "gauge_audit": {
            "label_permutations": args.gauge_seeds,
            "exact_isomorphism_passes": gauge_passes,
            "distinct_identity_labels": len(distinct_identity_labels),
        },
        "missing_one_extension_control": {
            "cases": 100,
            "safe_refusals": missing_extension_refusals,
        },
        "label_budget_comparison": {
            "endpoint_compiler_labels": 1_148,
            "two_dense_prefix_batches_at_256x16": 8_192,
            "compiler_label_reduction_factor": 8_192 / 1_148,
            "endpoint_training_labels_at_2000x256": 512_000,
            "dense_prefix_training_labels_at_2000x256x16": 8_192_000,
            "training_label_reduction_factor": 16.0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
