"""Exact Q8 endpoint-mixing and identity-gradient audit for the center gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from compare_recurrences import GROUPS, make_group_batches
from endpoint_credit_assignment_audit import (
    information_metrics,
    token_endpoint_joint,
    transition_tensor,
)
from mechanistic_group_actions import PureGroupActionModel


INPUT_ELEMENTS = (1, 5, 2, 6)  # i, -i, j, -j
HELD_OUT_PAIR = (0, 2)
LENGTHS = (1, 2, 4, 8, 16, 32)
FAMILIES = (
    ("pure_quaternion_spinor", 2.0 * math.pi),
    ("pure_ga_rotor", 2.0 * math.pi),
    ("pure_ga_rotor", 2.2),
)


def gradient_metrics(
    family: str,
    max_angle: float,
    length: int,
    *,
    batches: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    group = GROUPS["q8"]
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = PureGroupActionModel(
        len(INPUT_ELEMENTS),
        group.order,
        family=family,
        channels=4,
        max_rotor_angle=max_angle,
    ).to(device)
    generated = make_group_batches(
        group,
        batches,
        batch_size,
        length,
        seed + 10_000 + length,
        input_elements=INPUT_ELEMENTS,
        held_out_pairs=(HELD_OUT_PAIR,) if length >= 2 else (),
    )
    gradients = []
    contrast_gradients = []
    common_gradients = []
    losses = []
    accuracies = []
    for tokens, targets in generated:
        tokens = tokens.to(device)
        endpoint = targets[:, -1].to(device)
        model.zero_grad(set_to_none=True)
        logits = model(tokens)[:, -1]
        loss = nn.functional.cross_entropy(logits, endpoint)
        loss.backward()
        gradient = model.action_parameters.grad.detach().cpu()
        common = gradient.mean(dim=0, keepdim=True)
        gradients.append(gradient.flatten())
        contrast_gradients.append((gradient - common).flatten())
        common_gradients.append(common.flatten())
        losses.append(float(loss.detach()))
        accuracies.append(float((logits.argmax(-1) == endpoint).float().mean()))
    matrix = torch.stack(gradients).double()
    mean_gradient = matrix.mean(dim=0)
    rms_batch_norm = torch.sqrt(matrix.square().sum(dim=1).mean())
    centered_rms = torch.sqrt(
        (matrix - mean_gradient).square().sum(dim=1).mean()
    )
    mean_norm = mean_gradient.norm()
    cosine_to_mean = (
        (matrix @ mean_gradient)
        / (matrix.norm(dim=1) * mean_norm).clamp_min(1e-30)
    )
    contrast_matrix = torch.stack(contrast_gradients).double()
    contrast_mean = contrast_matrix.mean(dim=0)
    contrast_rms = torch.sqrt(contrast_matrix.square().sum(dim=1).mean())
    contrast_mean_norm = contrast_mean.norm()
    contrast_cosine = (
        (contrast_matrix @ contrast_mean)
        / (
            contrast_matrix.norm(dim=1) * contrast_mean_norm
        ).clamp_min(1e-30)
    )
    common_matrix = torch.stack(common_gradients).double()
    common_mean = common_matrix.mean(dim=0)
    common_rms = torch.sqrt(common_matrix.square().sum(dim=1).mean())
    return {
        "family": family,
        "max_angle": max_angle,
        "length": length,
        "mean_loss": float(np.mean(losses)),
        "mean_accuracy": float(np.mean(accuracies)),
        "mean_action_gradient_norm": float(mean_norm),
        "rms_batch_action_gradient_norm": float(rms_batch_norm),
        "centered_gradient_rms": float(centered_rms),
        "gradient_signal_to_rms_ratio": float(mean_norm / rms_batch_norm),
        "mean_cosine_to_mean_gradient": float(cosine_to_mean.mean()),
        "token_contrast_mean_gradient_norm": float(contrast_mean_norm),
        "token_contrast_rms_batch_gradient_norm": float(contrast_rms),
        "token_contrast_signal_to_rms_ratio": float(
            contrast_mean_norm / contrast_rms
        ),
        "token_contrast_mean_cosine_to_mean": float(contrast_cosine.mean()),
        "common_mode_mean_gradient_norm": float(common_mean.norm()),
        "common_mode_rms_batch_gradient_norm": float(common_rms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    group = GROUPS["q8"]
    kernel = transition_tensor(group.table, INPUT_ELEMENTS)
    eigenvalues = np.sort(np.abs(np.linalg.eigvals(kernel)))[::-1]
    exact = {}
    for length in LENGTHS:
        per_position = [
            information_metrics(
                token_endpoint_joint(
                    group.table, INPUT_ELEMENTS, length, position
                )
            )
            for position in range(length)
        ]
        exact[str(length)] = {
            "mutual_information_bits": {
                "minimum": min(x["mutual_information_bits"] for x in per_position),
                "mean": float(np.mean([x["mutual_information_bits"] for x in per_position])),
                "maximum": max(x["mutual_information_bits"] for x in per_position),
            },
            "endpoint_entropy_bits": {
                "minimum": min(x["endpoint_entropy_bits"] for x in per_position),
                "maximum": max(x["endpoint_entropy_bits"] for x in per_position),
            },
        }
    gradients = [
        gradient_metrics(
            family,
            max_angle,
            length,
            batches=args.batches,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
        )
        for family, max_angle in FAMILIES
        for length in LENGTHS
    ]
    report = {
        "experiment": "Q8 endpoint credit-assignment audit",
        "group": "q8",
        "input_elements": list(INPUT_ELEMENTS),
        "input_labels": [group.elements[index] for index in INPUT_ELEMENTS],
        "held_out_token_pair": list(HELD_OUT_PAIR),
        "augmented_markov_spectral_audit": {
            "state_count": int(group.order * len(INPUT_ELEMENTS)),
            "largest_eigenvalue_modulus": float(eigenvalues[0]),
            "second_largest_eigenvalue_modulus": float(eigenvalues[1]),
        },
        "exact_sampler_information": exact,
        "identity_initialization_gradient_audit": gradients,
        "interpretation_contract": {
            "information_selects_curriculum_not_family_winner": True,
            "chart_scaling_changes_gradient_magnitude": True,
            "the_balanced_center_falsifier_remains_required": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
