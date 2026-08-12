"""Train write-free Q8 actions and test central-sign fidelity prospectively."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn

from compare_recurrences import GROUPS, make_group_batches
from mechanistic_group_actions import (
    PureGroupActionModel,
    representation_diagnostics,
    streaming_equivalence,
)


INPUT_ELEMENTS = (1, 5, 2, 6)  # i, -i, j, -j
HELD_OUT_PAIR = (0, 2)
CENTRAL_ELEMENT = 4  # -1
CURRICULUM = (
    ((1,), 250),
    ((2,), 250),
    ((3, 4), 250),
    ((7, 8), 250),
    ((15, 16), 1_000),
)
PRIMARY_FAMILIES = (
    ("pure_quaternion_spinor", 2.0 * math.pi),
    ("pure_ga_rotor", 2.0 * math.pi),
    ("pure_ga_rotor", 2.2),
    ("pure_householder4_shared", math.pi),
    ("pure_householder", math.pi),
)


def curriculum_batches(seed: int, batch_size: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build the frozen parity-complete short-to-long batch schedule."""

    group = GROUPS["q8"]
    result: list[tuple[torch.Tensor, torch.Tensor]] = []
    for stage, (lengths, total) in enumerate(CURRICULUM):
        if total % len(lengths):
            raise ValueError("curriculum stage does not divide across lengths")
        per_length = total // len(lengths)
        blocks = [
            make_group_batches(
                group,
                per_length,
                batch_size,
                length,
                seed + 1_000 + 10_000 * stage + 1_000_000 * offset,
                input_elements=INPUT_ELEMENTS,
                held_out_pairs=(HELD_OUT_PAIR,) if length >= 2 else (),
            )
            for offset, length in enumerate(lengths)
        ]
        for index in range(per_length):
            for block in blocks:
                result.append(block[index])
    if len(result) != 2_000:
        raise AssertionError("frozen Q8 curriculum must contain 2,000 batches")
    return result


def final_state(model: PureGroupActionModel, tokens: torch.Tensor) -> torch.Tensor:
    """Evaluate a word with O(1) recurrent memory."""

    state = model.initial_state(tokens.shape[0])
    actions = model.action_matrices()
    for position in range(tokens.shape[1]):
        selected = actions[tokens[:, position]]
        state = torch.einsum("bcij,bcj->bci", selected, state)
    return state


@torch.no_grad()
def central_pair_evaluation(
    model: PureGroupActionModel,
    *,
    base_lengths: tuple[int, ...],
    batches: int,
    batch_size: int,
    seed_base: int,
    device: torch.device,
) -> dict[str, object]:
    group = GROUPS["q8"]
    negative_identity_tokens = torch.tensor((0, 0), device=device)
    by_length: dict[str, object] = {}
    minimum_member_accuracy = 1.0
    minimum_joint_accuracy = 1.0
    for length in base_lengths:
        correct_members = 0
        correct_pairs = 0
        examples = 0
        separation_squared = 0.0
        separation_max = 0.0
        generated = make_group_batches(
            group,
            batches,
            batch_size,
            length,
            seed_base + length,
            input_elements=INPUT_ELEMENTS,
        )
        for tokens, targets in generated:
            tokens = tokens.to(device)
            first_targets = targets[:, -1].to(device)
            second_targets = torch.as_tensor(
                group.table[first_targets.cpu().numpy(), CENTRAL_ELEMENT],
                dtype=torch.long,
                device=device,
            )
            first_state = final_state(model, tokens)
            second_state = first_state
            for token in negative_identity_tokens:
                second_state = model.step(token.expand(tokens.shape[0]), second_state)
            first_predictions = model.decode(first_state[:, None])[:, 0].argmax(-1)
            second_predictions = model.decode(second_state[:, None])[:, 0].argmax(-1)
            first_correct = first_predictions == first_targets
            second_correct = second_predictions == second_targets
            correct_members += int(first_correct.sum() + second_correct.sum())
            correct_pairs += int((first_correct & second_correct).sum())
            examples += tokens.shape[0]
            separation = (second_state - first_state).norm(dim=-1).mean(dim=-1)
            separation_squared += float(separation.square().sum())
            separation_max = max(separation_max, float(separation.max()))
        member_accuracy = correct_members / (2 * examples)
        joint_accuracy = correct_pairs / examples
        minimum_member_accuracy = min(minimum_member_accuracy, member_accuracy)
        minimum_joint_accuracy = min(minimum_joint_accuracy, joint_accuracy)
        by_length[str(length)] = {
            "base_parity": "odd" if length % 2 else "even",
            "pair_member_accuracy": member_accuracy,
            "both_members_correct_accuracy": joint_accuracy,
            "central_state_separation_rms": math.sqrt(
                separation_squared / examples
            ),
            "central_state_separation_max": separation_max,
        }
    return {
        "by_base_length": by_length,
        "minimum_pair_member_accuracy": minimum_member_accuracy,
        "minimum_both_members_correct_accuracy": minimum_joint_accuracy,
        "gate_pass": minimum_member_accuracy >= 0.99 and minimum_joint_accuracy >= 0.99,
    }


def train_one(
    family: str,
    max_angle: float,
    seed: int,
    *,
    channels: int,
    batch_size: int,
    device: torch.device,
    checkpoint_directory: Path,
    base_lengths: tuple[int, ...],
) -> dict[str, object]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    group = GROUPS["q8"]
    model = PureGroupActionModel(
        len(INPUT_ELEMENTS),
        group.order,
        family=family,
        channels=channels,
        max_rotor_angle=max_angle,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    batches = curriculum_batches(seed, batch_size)
    trajectory: dict[str, object] = {}
    start = time.perf_counter()
    model.train()
    for step, (tokens, targets) in enumerate(batches, start=1):
        tokens = tokens.to(device)
        endpoint = targets[:, -1].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)[:, -1]
        loss = nn.functional.cross_entropy(logits, endpoint)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        if step == 1 or step % 50 == 0:
            trajectory[str(step)] = {
                "length": int(tokens.shape[1]),
                "loss": float(loss.detach()),
                "accuracy": float((logits.argmax(-1) == endpoint).float().mean()),
                "gradient_norm_before_clip": gradient_norm,
            }
            print(
                f"family={family} angle={max_angle:.6g} seed={seed} "
                f"step={step}/2000 L={tokens.shape[1]} "
                f"loss={float(loss.detach()):.5f}",
                flush=True,
            )
    elapsed = time.perf_counter() - start
    model.eval()
    central = central_pair_evaluation(
        model,
        base_lengths=base_lengths,
        batches=2,
        batch_size=512,
        seed_base=2_300_000 + 10_000 * seed,
        device=device,
    )
    probe = torch.arange(96, device=device).reshape(3, 32) % len(INPUT_ELEMENTS)
    parity = streaming_equivalence(model, probe)
    diagnostics = representation_diagnostics(model, group, INPUT_ELEMENTS)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_directory / (
        f"q8_{family}_angle{max_angle:.9f}_seed{seed}.pt"
    )
    torch.save(
        {
            "family": family,
            "group": "q8",
            "input_elements": INPUT_ELEMENTS,
            "config": {
                "seed": seed,
                "channels": channels,
                "max_angle": max_angle,
                "steps": 2_000,
                "batch_size": batch_size,
                "curriculum": CURRICULUM,
            },
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
        },
        checkpoint,
    )
    return {
        "family": family,
        "max_angle": max_angle,
        "seed": seed,
        "channels": channels,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "action_parameters": model.action_parameters.numel(),
        "trajectory": trajectory,
        "central_pair_evaluation": central,
        "streaming_equivalence": parity,
        "representation_diagnostics": diagnostics,
        "checkpoint": str(checkpoint),
        "elapsed_seconds": elapsed,
    }


def parse_family(specification: str) -> tuple[str, float]:
    if "@" in specification:
        family, angle = specification.split("@", 1)
        return family, float(angle)
    matches = [item for item in PRIMARY_FAMILIES if item[0] == specification]
    if len(matches) != 1:
        raise ValueError(
            f"family {specification!r} needs an explicit @max_angle"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--families",
        nargs="+",
        default=[f"{family}@{angle}" for family, angle in PRIMARY_FAMILIES],
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--base-lengths",
        type=int,
        nargs="+",
        default=[15, 16, 31, 32, 63, 64, 127, 128, 255, 256],
    )
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    torch.use_deterministic_algorithms(True)
    families = [parse_family(item) for item in args.families]
    results = [
        train_one(
            family,
            max_angle,
            seed,
            channels=args.channels,
            batch_size=args.batch_size,
            device=device,
            checkpoint_directory=args.checkpoint_directory,
            base_lengths=tuple(args.base_lengths),
        )
        for family, max_angle in families
        for seed in args.seeds
    ]
    report = {
        "experiment": "write-free Q8 spinor center-fidelity gate",
        "device": str(device),
        "input_elements": list(INPUT_ELEMENTS),
        "input_labels": [GROUPS["q8"].elements[i] for i in INPUT_ELEMENTS],
        "curriculum": [
            {"lengths": list(lengths), "batches": total}
            for lengths, total in CURRICULUM
        ],
        "balanced_pair_definition": "(w, w*i*i)",
        "exact_oracle_contract": {
            "spinor_pair_member_accuracy": 1.0,
            "spinor_both_members_correct_accuracy": 1.0,
            "sandwich_pair_member_ceiling": 0.5,
            "sandwich_both_members_correct_ceiling": 0.0,
            "regular_permutation_pair_member_accuracy": 1.0,
            "regular_permutation_both_members_correct_accuracy": 1.0,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
