"""Audit inverse-cover identifiability, equal-budget controls, and gauge robustness."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from compare_recurrences import GROUPS, make_group_batches, parse_input_elements
from latent_group_discovery import (
    TransitionEvidence,
    exact_inverse_tokens,
    inverse_cover_partial_evidence,
    random_partial_evidence,
)
from representation_retraction import regular_irrep_candidates


INPUT_LABELS = ("23145", "31245", "23451", "51234")


def complete_evidence() -> TransitionEvidence:
    group = GROUPS["a5"]
    inputs = parse_input_elements(INPUT_LABELS, group)
    evidence = TransitionEvidence(group.order, len(inputs))
    for tokens, targets in make_group_batches(
        group, 4, 512, 16, 18_901, input_elements=inputs
    ):
        evidence.observe(tokens, targets)
    if not evidence.complete:
        raise RuntimeError("audit source evidence is unexpectedly incomplete")
    return evidence


def recover_mask(
    full: TransitionEvidence, calibration_fraction: float, seed: int
) -> tuple[bool, dict[str, object], str | None]:
    partial, mask = inverse_cover_partial_evidence(
        full, calibration_fraction=calibration_fraction, seed=seed
    )
    try:
        inferred = partial.infer_inverse_pairs_and_complete()
        recovered = partial.recover()
    except ValueError as error:
        return False, mask, str(error)
    exact = inferred == exact_inverse_tokens(full.next_states)
    replay = np.array_equal(partial.next_states, full.next_states)
    return bool(exact and replay and recovered.group.order == full.state_count), mask, None


def is_exact_isomorphism(recovered, true_table: np.ndarray) -> bool:
    labels = recovered.element_to_state
    # A non-identity base state identifies recovered element ``g`` with the
    # observed state ``base*g``. Remove that left coset before testing the
    # identity-preserving group isomorphism.
    base_label = int(labels[0])
    inverse_candidates = np.flatnonzero(true_table[base_label] == 0)
    if len(inverse_candidates) != 1:
        return False
    normalized = true_table[int(inverse_candidates[0]), labels]
    transported = normalized[recovered.group.table]
    expected = true_table[normalized[:, None], normalized[None, :]]
    return bool(np.array_equal(transported, expected))


def run_audit(mask_seeds: int) -> dict[str, object]:
    group = GROUPS["a5"]
    full = complete_evidence()
    calibration_levels = (0.0, 1 / 60, 2 / 60, 3 / 60, 0.10)
    curve = []
    for fraction in calibration_levels:
        successes = 0
        failures: dict[str, int] = {}
        last_mask = None
        for seed in range(mask_seeds):
            success, mask, error = recover_mask(full, fraction, seed)
            last_mask = mask
            successes += int(success)
            if error is not None:
                failures[error] = failures.get(error, 0) + 1
        curve.append(
            {
                "calibration_fraction_per_inverse_family": fraction,
                "calibration_pairs": last_mask["calibration_pairs"],
                "observed_edges": last_mask["observed_edges"],
                "total_edges": last_mask["total_edges"],
                "observed_fraction": last_mask["observed_fraction"],
                "successful_exact_recoveries": successes,
                "mask_seeds": mask_seeds,
                "failures": failures,
            }
        )

    global_calibration_curve = []
    for calibration_pairs in (0, 1, 2):
        successes = 0
        family_allocations: dict[str, int] = {}
        last_mask = None
        for seed in range(mask_seeds):
            partial, mask = inverse_cover_partial_evidence(
                full,
                calibration_pairs_total=calibration_pairs,
                seed=20_000 + seed,
            )
            last_mask = mask
            allocation = json.dumps(
                mask["calibration_pairs_by_inverse_family"], sort_keys=True
            )
            family_allocations[allocation] = family_allocations.get(allocation, 0) + 1
            try:
                inferred = partial.infer_inverse_pairs_and_complete()
                recovered = partial.recover()
                successes += int(
                    inferred == exact_inverse_tokens(full.next_states)
                    and np.array_equal(partial.next_states, full.next_states)
                    and recovered.group.order == full.state_count
                )
            except ValueError:
                pass
        global_calibration_curve.append(
            {
                "global_calibration_pairs": calibration_pairs,
                "observed_edges": last_mask["observed_edges"],
                "observed_fraction": last_mask["observed_fraction"],
                "successful_exact_recoveries": successes,
                "mask_seeds": mask_seeds,
                "calibration_family_allocations": family_allocations,
            }
        )

    undercovered_successes = 0
    for seed in range(mask_seeds):
        partial, _ = inverse_cover_partial_evidence(
            full, calibration_pairs_total=0, seed=30_000 + seed
        )
        observed = np.argwhere(partial.next_states >= 0)
        source, token = observed[seed % len(observed)]
        partial.next_states[source, token] = -1
        partial.counts[source, token] = 0
        try:
            partial.infer_inverse_pairs_and_complete()
            partial.recover()
            undercovered_successes += 1
        except ValueError:
            pass

    random_controls = []
    for observed_edges in (120, 121, 122):
        random_successes = 0
        random_missing_after_propagation = []
        random_errors: dict[str, int] = {}
        for seed in range(mask_seeds):
            partial = random_partial_evidence(
                full, observed_edges=observed_edges, seed=seed
            )
            # Diagnostic only: even granting the true inverse pairing, count
            # the directions that remain unknowable because both members of a
            # reverse-edge pair were hidden.
            audited = partial.next_states.copy()
            true_inverses = exact_inverse_tokens(full.next_states)
            for token, inverse in enumerate(true_inverses):
                for source in range(full.state_count):
                    target = audited[source, token]
                    if target >= 0 and audited[target, inverse] < 0:
                        audited[target, inverse] = source
            random_missing_after_propagation.append(int(np.sum(audited < 0)))
            try:
                partial.infer_inverse_pairs_and_complete()
                partial.recover()
                random_successes += 1
            except ValueError as error:
                message = str(error)
                family = message.split(":", 1)[0]
                random_errors[family] = random_errors.get(family, 0) + 1
        random_controls.append(
            {
                "observed_edges": observed_edges,
                "mask_seeds": mask_seeds,
                "successful_exact_recoveries": random_successes,
                "failure_families": random_errors,
                "missing_edges_after_propagation": {
                    "minimum": min(random_missing_after_propagation),
                    "mean": float(np.mean(random_missing_after_propagation)),
                    "maximum": max(random_missing_after_propagation),
                },
            }
        )

    target_fraction = 1 / 60
    completed, _ = inverse_cover_partial_evidence(
        full, calibration_fraction=target_fraction, seed=91
    )
    completed.infer_inverse_pairs_and_complete()
    generator_orders = list(itertools.permutations(range(full.token_count)))
    base_states = (0, 1, 7, 13, 29, 47)
    gauges = set()
    isomorphism_passes = 0
    compiler_passes = 0
    compiler_max_invariance_rms = 0.0
    compiler_max_homomorphism_rms = 0.0
    for order in generator_orders:
        for base_state in base_states:
            recovered = completed.recover(
                base_state=base_state, generator_order=order
            )
            gauges.add(tuple(int(x) for x in recovered.state_to_element))
            isomorphism_passes += int(
                is_exact_isomorphism(recovered, group.table)
            )
        representative = completed.recover(base_state=0, generator_order=order)
        candidates = regular_irrep_candidates(representative.group, 3, seed=7_301)
        exact = len(candidates) == 2
        if candidates:
            compiler_max_invariance_rms = max(
                compiler_max_invariance_rms,
                max(candidate.invariance_rms for candidate in candidates),
            )
            compiler_max_homomorphism_rms = max(
                compiler_max_homomorphism_rms,
                max(candidate.homomorphism_rms for candidate in candidates),
            )
            exact = exact and all(
                candidate.invariance_rms < 1e-10
                and candidate.homomorphism_rms < 1e-10
                for candidate in candidates
            )
        compiler_passes += int(exact)

    return {
        "experiment": "partial Cayley inverse-cover supervision and gauge audit",
        "source_edges": int(full.next_states.size),
        "true_inverse_tokens_used_by_masker_only": list(
            exact_inverse_tokens(full.next_states)
        ),
        "supervision_curve": curve,
        "joint_matching_minimum_curve": global_calibration_curve,
        "reverse_cover_minus_one_control": {
            "observed_edges": 119,
            "successful_exact_recoveries": undercovered_successes,
            "mask_seeds": mask_seeds,
        },
        "equal_budget_random_mask_controls": random_controls,
        "gauge_robustness": {
            "base_states": list(base_states),
            "generator_orders": len(generator_orders),
            "variants": len(base_states) * len(generator_orders),
            "unique_recovered_gauges": len(gauges),
            "exact_posthoc_isomorphism_passes": isomorphism_passes,
            "compiler_order_variants": len(generator_orders),
            "compiler_exact_passes": compiler_passes,
            "maximum_compiler_invariance_rms": compiler_max_invariance_rms,
            "maximum_compiler_homomorphism_rms": compiler_max_homomorphism_rms,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-seeds", type=int, default=1_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(args.mask_seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
