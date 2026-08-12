"""Adversarial post-hoc audit of latent Cayley recovery and long horizons."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from changed_generator_transfer import select_changed_generators
from compare_recurrences import GROUPS, make_group_batches, parse_input_elements
from latent_group_discovery import (
    TransitionEvidence,
    inverse_cover_partial_evidence,
)
from mechanistic_group_actions import PureGroupActionModel
from train_self_compiling_retraction import INPUT_LABELS, evaluate_anchor


AUDIT_LENGTHS = tuple(range(4_096, 16_385, 1_024))


def conjugacy_classes(table: np.ndarray) -> list[list[int]]:
    order = len(table)
    inverses = np.asarray([np.flatnonzero(table[element] == 0)[0] for element in range(order)])
    remaining = set(range(order))
    classes = []
    while remaining:
        element = min(remaining)
        conjugates = {
            int(table[table[change, element], inverses[change]])
            for change in range(order)
        }
        classes.append(sorted(conjugates))
        remaining -= conjugates
    return classes


def is_simple_from_conjugacy_classes(table: np.ndarray, classes: list[list[int]]) -> bool:
    nonidentity = [group_class for group_class in classes if 0 not in group_class]
    order = len(table)
    for mask in range(1, 1 << len(nonidentity)):
        subset = {0}
        for index, group_class in enumerate(nonidentity):
            if mask & (1 << index):
                subset.update(group_class)
        if len(subset) == order:
            continue
        if all(int(table[left, right]) in subset for left in subset for right in subset):
            return False
    return True


def recover_seed(
    seed: int,
    *,
    generator_class_index: int,
    inverse_cover_calibration: float | None,
    inverse_cover_calibration_pairs_total: int | None,
) -> tuple[TransitionEvidence, object, dict[str, object]]:
    true_group = GROUPS["a5"]
    inputs = parse_input_elements(INPUT_LABELS, true_group)
    batches = make_group_batches(
        true_group,
        2,
        256,
        16,
        seed + 1_000,
        input_elements=inputs,
        held_out_pairs=((0, 2),),
    )
    evidence = TransitionEvidence(true_group.order, len(inputs))
    first_coverage = None
    recovery_step = None
    recovery_minimum_edge_count = None
    recovered = None
    for step, (tokens, targets) in enumerate(batches, start=1):
        evidence.observe(tokens, targets)
        if step == 1:
            first_coverage = evidence.coverage
        if recovered is None and evidence.complete:
            recovered = evidence.recover(base_state=0)
            recovery_step = step
            recovery_minimum_edge_count = int(evidence.counts.min())
    if recovered is None:
        raise RuntimeError(f"seed {seed} did not recover in two batches")

    partial_mask = None
    if (
        inverse_cover_calibration is not None
        or inverse_cover_calibration_pairs_total is not None
    ):
        evidence, partial_mask = inverse_cover_partial_evidence(
            evidence,
            calibration_fraction=inverse_cover_calibration or 0.0,
            calibration_pairs_total=inverse_cover_calibration_pairs_total,
            seed=910_001 + seed,
        )
        inferred = evidence.infer_inverse_pairs_and_complete()
        partial_mask["inferred_inverse_tokens"] = list(inferred)
        recovered = evidence.recover(base_state=0)

    table = recovered.group.table
    classes = conjugacy_classes(table)
    class_sizes = sorted(map(len, classes))
    noncommutative = bool(np.any(table != table.T))
    simple = is_simple_from_conjugacy_classes(table, classes)
    mapping = recovered.element_to_state
    mapped_products = mapping[table]
    true_products = true_group.table[mapping[:, None], mapping[None, :]]
    exact_isomorphism = bool(np.array_equal(mapped_products, true_products))

    changed = select_changed_generators(
        true_group, inputs, selection_index=generator_class_index
    )
    changed_permutations = np.stack(
        [true_group.table[:, element] for element in changed]
    )
    overlap = sum(
        np.array_equal(original, changed)
        for original, changed in itertools.product(
            recovered.token_permutations, changed_permutations
        )
    )

    incomplete = TransitionEvidence(evidence.state_count, evidence.token_count)
    incomplete.next_states[:] = evidence.next_states
    incomplete.counts[:] = evidence.counts
    incomplete.next_states[0, 0] = -1
    incomplete.counts[0, 0] = 0
    refused_missing_edge = False
    try:
        incomplete.recover(base_state=0)
    except ValueError:
        refused_missing_edge = True

    return evidence, recovered, {
        "first_batch_coverage": first_coverage,
        "recovery_step": recovery_step,
        "minimum_edge_count_at_recovery": recovery_minimum_edge_count,
        "generated_group_order": recovered.group.order,
        "noncommutative": noncommutative,
        "conjugacy_class_sizes": class_sizes,
        "simple": simple,
        "exact_isomorphism_to_a5_posthoc": exact_isomorphism,
        "gauge_fixed_points": int(np.sum(recovered.state_to_element == np.arange(60))),
        "state_to_element": recovered.state_to_element.tolist(),
        "changed_generator_elements_disjoint": not bool(set(inputs) & set(changed)),
        "token_permutation_overlap_with_changed_generators": int(overlap),
        "missing_single_edge_refused": refused_missing_edge,
        "partial_inverse_cover": partial_mask,
    }


def evaluate_seed(
    seed: int,
    checkpoint_directory: Path,
    device: torch.device,
    *,
    generator_class_index: int,
    inverse_cover_calibration: float | None,
    inverse_cover_calibration_pairs_total: int | None,
) -> dict[str, object]:
    evidence, recovered, algebra = recover_seed(
        seed,
        generator_class_index=generator_class_index,
        inverse_cover_calibration=inverse_cover_calibration,
        inverse_cover_calibration_pairs_total=(
            inverse_cover_calibration_pairs_total
        ),
    )
    path = checkpoint_directory / f"self_compiling_retraction_seed{seed}.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not checkpoint["config"].get("table_blind"):
        raise ValueError(f"checkpoint {path} is not marked table-blind")
    if checkpoint["config"].get("inverse_cover_calibration") != inverse_cover_calibration:
        raise ValueError(f"checkpoint {path} has the wrong partial-evidence protocol")
    if (
        checkpoint["config"].get("inverse_cover_calibration_pairs_total")
        != inverse_cover_calibration_pairs_total
    ):
        raise ValueError(f"checkpoint {path} has the wrong calibration-pair count")
    true_group = GROUPS["a5"]
    inputs = tuple(checkpoint["input_elements"])
    model = PureGroupActionModel(
        len(inputs),
        true_group.order,
        family=checkpoint["family"],
        channels=checkpoint["config"]["channels"],
        max_rotor_angle=checkpoint["config"]["max_rotor_angle"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    dense_long = evaluate_anchor(
        model,
        true_group,
        inputs,
        checkpoint["config"]["anchor_channel"],
        generator_class=generator_class_index,
        lengths=AUDIT_LENGTHS,
        batches=1,
        batch_size=256,
        seed_base=1_710_000,
        device=device,
    )
    return {
        "training_seed": seed,
        "algebra_audit": algebra,
        "dense_L4096_L16384": dense_long,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator-class-index", type=int, default=33)
    parser.add_argument("--inverse-cover-calibration", type=float, default=None)
    parser.add_argument(
        "--inverse-cover-calibration-pairs-total", type=int, default=None
    )
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    results = [
        evaluate_seed(
            seed,
            args.checkpoint_directory,
            device,
            generator_class_index=args.generator_class_index,
            inverse_cover_calibration=args.inverse_cover_calibration,
            inverse_cover_calibration_pairs_total=(
                args.inverse_cover_calibration_pairs_total
            ),
        )
        for seed in args.seeds
    ]
    report = {
        "experiment": "latent Cayley adversarial audit",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "lengths": list(AUDIT_LENGTHS),
        "generator_class_index": args.generator_class_index,
        "inverse_cover_calibration": args.inverse_cover_calibration,
        "inverse_cover_calibration_pairs_total": (
            args.inverse_cover_calibration_pairs_total
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "seeds": args.seeds}))


if __name__ == "__main__":
    main()
