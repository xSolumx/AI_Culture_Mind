"""Audit first-order endpoint information and gradient cancellation by word length."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import torch
from torch import nn

from compare_recurrences import GROUPS, make_group_batches, parse_input_elements
from mechanistic_group_actions import PureGroupActionModel
from train_self_compiling_retraction import INPUT_LABELS


LENGTHS = (1, 2, 4, 8, 16)
HELD_OUT_PAIR = (0, 2)


def transition_tensor(
    table: np.ndarray, input_elements: tuple[int, ...]
) -> np.ndarray:
    """Return the Markov kernel over (group state, previous token)."""
    group_order = table.shape[0]
    token_count = len(input_elements)
    kernel = np.zeros(
        (group_order, token_count, group_order, token_count), dtype=np.float64
    )
    for state in range(group_order):
        for previous in range(token_count):
            allowed = [
                token
                for token in range(token_count)
                if (previous, token) != HELD_OUT_PAIR
            ]
            probability = 1.0 / len(allowed)
            for token in allowed:
                target = int(table[state, input_elements[token]])
                kernel[state, previous, target, token] += probability
    return kernel.reshape(group_order * token_count, group_order * token_count)


def initial_distribution(
    table: np.ndarray, input_elements: tuple[int, ...]
) -> np.ndarray:
    group_order = table.shape[0]
    token_count = len(input_elements)
    distribution = np.zeros((group_order, token_count), dtype=np.float64)
    for token, element in enumerate(input_elements):
        distribution[int(table[0, element]), token] = 1.0 / token_count
    return distribution


def token_endpoint_joint(
    table: np.ndarray,
    input_elements: tuple[int, ...],
    length: int,
    position: int,
) -> np.ndarray:
    """Exact P(token_at_position, final_group) under the training sampler."""
    if not 0 <= position < length:
        raise ValueError("position is outside the word")
    group_order = table.shape[0]
    token_count = len(input_elements)
    kernel = transition_tensor(table, input_elements)
    initial = initial_distribution(table, input_elements)

    selected = np.zeros(
        (token_count, group_order, token_count), dtype=np.float64
    )
    if position == 0:
        for token, element in enumerate(input_elements):
            state = int(table[0, element])
            selected[token, state, token] = 1.0 / token_count
    else:
        prefix = initial.reshape(-1)
        for _ in range(position - 1):
            prefix = prefix @ kernel
        prefix = prefix.reshape(group_order, token_count)
        for state in range(group_order):
            for previous in range(token_count):
                allowed = [
                    token
                    for token in range(token_count)
                    if (previous, token) != HELD_OUT_PAIR
                ]
                probability = prefix[state, previous] / len(allowed)
                for token in allowed:
                    target = int(table[state, input_elements[token]])
                    selected[token, target, token] += probability

    flattened = selected.reshape(token_count, -1)
    for _ in range(length - position - 1):
        flattened = flattened @ kernel
    return flattened.reshape(token_count, group_order, token_count).sum(axis=2)


def information_metrics(joint: np.ndarray) -> dict[str, float]:
    token_marginal = joint.sum(axis=1, keepdims=True)
    endpoint_marginal = joint.sum(axis=0, keepdims=True)
    product = token_marginal @ endpoint_marginal
    mask = joint > 0.0
    mutual_information = float(
        np.sum(joint[mask] * np.log2(joint[mask] / product[mask]))
    )
    conditional = np.divide(
        joint,
        token_marginal,
        out=np.zeros_like(joint),
        where=token_marginal > 0.0,
    )
    total_variation = 0.5 * np.abs(conditional - endpoint_marginal).sum(axis=1)
    endpoint_entropy = float(
        -np.sum(endpoint_marginal[endpoint_marginal > 0.0]
                * np.log2(endpoint_marginal[endpoint_marginal > 0.0]))
    )
    return {
        "mutual_information_bits": mutual_information,
        "mean_conditional_total_variation": float(
            np.sum(token_marginal[:, 0] * total_variation)
        ),
        "maximum_conditional_total_variation": float(total_variation.max()),
        "endpoint_entropy_bits": endpoint_entropy,
    }


def gradient_metrics(
    *, length: int, batches: int, batch_size: int, seed: int, device: torch.device
) -> dict[str, object]:
    group = GROUPS["a5"]
    input_elements = parse_input_elements(INPUT_LABELS, group)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = PureGroupActionModel(
        len(input_elements),
        group.order,
        family="pure_ga_rotor",
        channels=4,
        max_rotor_angle=2.2,
    ).to(device)
    generated = make_group_batches(
        group,
        batches,
        batch_size,
        length,
        seed + 10_000 + length,
        input_elements=input_elements,
        held_out_pairs=(HELD_OUT_PAIR,) if length >= 2 else (),
    )
    gradients = []
    losses = []
    accuracies = []
    for tokens, targets in generated:
        tokens = tokens.to(device)
        endpoint = targets[:, -1].to(device)
        model.zero_grad(set_to_none=True)
        logits = model(tokens)[:, -1]
        loss = nn.functional.cross_entropy(logits, endpoint)
        loss.backward()
        gradients.append(model.action_parameters.grad.detach().flatten().cpu())
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
    return {
        "length": length,
        "batches": batches,
        "batch_size": batch_size,
        "mean_loss": float(np.mean(losses)),
        "mean_accuracy": float(np.mean(accuracies)),
        "mean_action_gradient_norm": float(mean_norm),
        "rms_batch_action_gradient_norm": float(rms_batch_norm),
        "centered_gradient_rms": float(centered_rms),
        "gradient_signal_to_rms_ratio": float(mean_norm / rms_batch_norm),
        "mean_cosine_to_mean_gradient": float(cosine_to_mean.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    group = GROUPS["a5"]
    input_elements = parse_input_elements(INPUT_LABELS, group)
    kernel_eigenvalue_magnitudes = np.sort(
        np.abs(np.linalg.eigvals(transition_tensor(group.table, input_elements)))
    )[::-1]
    exact = {}
    for length in LENGTHS:
        per_position = [
            information_metrics(
                token_endpoint_joint(
                    group.table, input_elements, length, position
                )
            )
            for position in range(length)
        ]
        exact[str(length)] = {
            "positions": per_position,
            "mutual_information_bits": {
                "minimum": min(x["mutual_information_bits"] for x in per_position),
                "mean": float(np.mean([x["mutual_information_bits"] for x in per_position])),
                "maximum": max(x["mutual_information_bits"] for x in per_position),
            },
            "conditional_total_variation": {
                "minimum": min(x["mean_conditional_total_variation"] for x in per_position),
                "mean": float(np.mean([x["mean_conditional_total_variation"] for x in per_position])),
                "maximum": max(x["mean_conditional_total_variation"] for x in per_position),
            },
        }
    empirical = [
        gradient_metrics(
            length=length,
            batches=args.batches,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
        )
        for length in LENGTHS
    ]
    report = {
        "experiment": "endpoint credit-assignment information audit",
        "group": "a5",
        "input_elements": list(input_elements),
        "held_out_pair": list(HELD_OUT_PAIR),
        "augmented_markov_spectral_audit": {
            "state_count": int(group.order * len(input_elements)),
            "largest_eigenvalue_modulus": float(kernel_eigenvalue_magnitudes[0]),
            "second_largest_eigenvalue_modulus": float(
                kernel_eigenvalue_magnitudes[1]
            ),
        },
        "exact_sampler_information": exact,
        "identity_initialization_gradient_audit": empirical,
        "interpretation_contract": {
            "nonzero_gradient_norm_is_not_directional_signal": True,
            "mutual_information_is_a_first_order_diagnostic_not_a_training_proof": True,
            "curriculum_changes_short_word_exposure_and_ordering": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "empirical": empirical}, indent=2))


if __name__ == "__main__":
    main()
