"""Validate endpoint-manifold reconstruction on already exact checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compare_recurrences import GROUPS, make_group_batches, parse_input_elements
from endpoint_representation_discovery import recover_endpoint_manifold
from mechanistic_group_actions import PureGroupActionModel
from representation_retraction import compile_nearest_representation
from train_self_compiling_retraction import INPUT_LABELS, token_commutator_max


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    group = GROUPS["a5"]
    inputs = parse_input_elements(INPUT_LABELS, group)
    results = []
    for seed in args.seeds:
        checkpoint = torch.load(
            args.checkpoint_directory / f"self_compiling_retraction_seed{seed}.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = PureGroupActionModel(
            len(inputs),
            group.order,
            family=checkpoint["family"],
            channels=checkpoint["config"]["channels"],
            max_rotor_angle=checkpoint["config"]["max_rotor_angle"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        rotations = (
            model.action_matrices().detach().numpy()[:, :, 1:4, 1:4]
        )
        batches = make_group_batches(
            group,
            64,
            256,
            8,
            seed + 31_000,
            input_elements=inputs,
            held_out_pairs=((0, 2),),
        )
        words = np.concatenate([tokens.numpy() for tokens, _ in batches])
        labels = np.concatenate(
            [targets[:, -1].numpy() for _, targets in batches]
        )
        channels = []
        for channel in range(model.channels):
            try:
                recovery = recover_endpoint_manifold(
                    rotations[:, channel], words, labels, state_count=group.order
                )
                mapping = recovery.label_to_element
                exact_isomorphism = bool(
                    np.array_equal(
                        recovery.group.table[mapping[:, None], mapping[None, :]],
                        mapping[group.table],
                    )
                )
                compiled = compile_nearest_representation(
                    rotations[:, channel],
                    recovery.group,
                    recovery.input_elements,
                    seed=83_010 + 1_003 * seed + channel,
                )
                commutator = float(token_commutator_max(rotations[:, channel]))
                threshold_pass = bool(
                    compiled.alignment_rms <= 0.08
                    and compiled.runner_up_rms - compiled.alignment_rms >= 0.20
                    and commutator >= 0.50
                    and recovery.minimum_assignment_gap >= 0.10
                    and recovery.multiplication_max <= 0.20
                )
                channels.append({
                    "channel": channel,
                    "accepted": True,
                    "exact_isomorphism_to_a5_posthoc": exact_isomorphism,
                    "class_consistency_rms": recovery.class_consistency_rms,
                    "multiplication_max": recovery.multiplication_max,
                    "minimum_assignment_gap": recovery.minimum_assignment_gap,
                    "alignment_rms": float(compiled.alignment_rms),
                    "runner_up_rms": float(compiled.runner_up_rms),
                    "commutator": commutator,
                    "threshold_pass": threshold_pass,
                })
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                channels.append({
                    "channel": channel,
                    "accepted": False,
                    "reason": str(error),
                })
        accepted = [item["channel"] for item in channels if item["accepted"]]
        threshold_candidates = [
            item for item in channels if item.get("threshold_pass", False)
        ]
        threshold_candidates.sort(key=lambda item: item["alignment_rms"])
        selected = (
            threshold_candidates[0]["channel"] if threshold_candidates else None
        )
        anchor = checkpoint["config"]["anchor_channel"]
        results.append({
            "training_seed": seed,
            "checkpoint_anchor": anchor,
            "accepted_channels": accepted,
            "threshold_channels": [
                item["channel"] for item in threshold_candidates
            ],
            "selected_channel": selected,
            "correct_anchor_selection": selected == anchor,
            "channels": channels,
        })
    report = {
        "experiment": "post-retraction endpoint-manifold compiler validation",
        "causal_limitation": (
            "checkpoints were already exact-retracted; this validates the "
            "reconstruction pipeline, not blind pre-retraction discovery"
        ),
        "results": results,
        "all_correct_anchor_selection": all(
            result["correct_anchor_selection"] for result in results
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "all_correct_anchor_selection": report["all_correct_anchor_selection"],
    }))


if __name__ == "__main__":
    main()
