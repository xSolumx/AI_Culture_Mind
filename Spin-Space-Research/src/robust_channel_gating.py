"""Fit bounded channel gates on frozen GA checkpoints and test unseen macros."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F

from changed_generator_transfer import (
    select_changed_generators,
    strongest_commutator_channel,
)
from compare_recurrences import GROUPS, FiniteGroup, make_group_batches
from mechanistic_group_actions import PureGroupActionModel, _compose_word, canonical_group_words


FIT_LENGTHS = (16, 32, 48, 64, 80, 128, 192, 256)
DENSE_LENGTHS = tuple(range(16, 257, 16))


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def macro_alphabet(
    group: FiniteGroup,
    original_inputs: tuple[int, ...],
    base_actions: torch.Tensor,
    generator_set_index: int,
) -> tuple[tuple[int, ...], torch.Tensor]:
    elements = select_changed_generators(
        group, original_inputs, selection_index=generator_set_index
    )
    words = canonical_group_words(group, original_inputs)
    actions = torch.stack(
        [_compose_word(base_actions, words[element]) for element in elements]
    )
    return elements, actions


@torch.no_grad()
def final_states(
    model: PureGroupActionModel,
    actions: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    state = model.initial_state(len(tokens))
    for position in range(tokens.shape[1]):
        selected = actions[tokens[:, position]]
        state = torch.einsum("bcij,bcj->bci", selected, state)
    return state


@torch.no_grad()
def channel_logit_parts(
    model: PureGroupActionModel, state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = model.logit_scale.exp().clamp(max=100.0)
    weights = model.output_head.weight.reshape(
        model.output_head.out_features, model.channels, -1
    )
    channel_logits = scale * torch.einsum("bcd,ocd->bco", state, weights)
    bias = scale * model.output_head.bias
    return channel_logits.detach(), bias.detach()


@torch.no_grad()
def collect_strata(
    model: PureGroupActionModel,
    group: FiniteGroup,
    actions: torch.Tensor,
    input_elements: tuple[int, ...],
    lengths: tuple[int, ...],
    *,
    batches: int,
    batch_size: int,
    seed_base: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor | int]]:
    strata = []
    for length in lengths:
        generated = make_group_batches(
            group,
            batches,
            batch_size,
            length,
            seed_base + length,
            input_elements=input_elements,
        )
        states = []
        targets = []
        for tokens, sequence_targets in generated:
            tokens = tokens.to(device)
            states.append(final_states(model, actions, tokens))
            targets.append(sequence_targets[:, -1].to(device))
        state = torch.cat(states)
        channel_logits, bias = channel_logit_parts(model, state)
        strata.append(
            {
                "length": length,
                "channel_logits": channel_logits,
                "bias": bias,
                "targets": torch.cat(targets),
            }
        )
    return strata


def gate_vector(
    raw_auxiliary: torch.Tensor, anchor: int, channels: int
) -> torch.Tensor:
    gates = torch.ones(channels, device=raw_auxiliary.device)
    auxiliary = [channel for channel in range(channels) if channel != anchor]
    gates[auxiliary] = raw_auxiliary.sigmoid()
    return gates


def logits_for_gates(
    stratum: dict[str, torch.Tensor | int], gates: torch.Tensor
) -> torch.Tensor:
    channel_logits = stratum["channel_logits"]
    bias = stratum["bias"]
    assert isinstance(channel_logits, torch.Tensor)
    assert isinstance(bias, torch.Tensor)
    return bias + (channel_logits * gates[None, :, None]).sum(dim=1)


def robust_margin_objective(
    strata: list[dict[str, torch.Tensor | int]], gates: torch.Tensor
) -> torch.Tensor:
    losses = []
    for stratum in strata:
        targets = stratum["targets"]
        assert isinstance(targets, torch.Tensor)
        logits = logits_for_gates(stratum, gates)
        true_logits = logits.gather(1, targets[:, None]).squeeze(1)
        target_mask = F.one_hot(targets, num_classes=logits.shape[-1]).bool()
        false_logits = logits.masked_fill(target_mask, -torch.inf).amax(dim=-1)
        losses.append(F.softplus(1.0 - (true_logits - false_logits)).mean())
    stacked = torch.stack(losses)
    return 0.1 * torch.logsumexp(stacked / 0.1, dim=0)


@torch.no_grad()
def accuracy_summary(
    strata: list[dict[str, torch.Tensor | int]], gates: torch.Tensor
) -> dict[str, object]:
    by_length = {}
    for stratum in strata:
        targets = stratum["targets"]
        assert isinstance(targets, torch.Tensor)
        accuracy = float((logits_for_gates(stratum, gates).argmax(-1) == targets).float().mean())
        by_length[str(stratum["length"])] = accuracy
    values = list(by_length.values())
    return {
        "by_length": by_length,
        "dense_minimum_accuracy": min(values),
        "dense_mean_accuracy": sum(values) / len(values),
        "dense_gate_pass": min(values) >= 0.9,
    }


def fixed_gate(channels: int, active: list[int], device: torch.device) -> torch.Tensor:
    gates = torch.zeros(channels, device=device)
    gates[active] = 1.0
    return gates


def fit_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    steps: int,
    learning_rate: float,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    group = GROUPS[checkpoint["group"]]
    original_inputs = tuple(checkpoint["input_elements"])
    model = PureGroupActionModel(
        len(original_inputs),
        group.order,
        family=checkpoint["family"],
        channels=config["channels"],
        max_rotor_angle=config["max_rotor_angle"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    base_actions = model.action_matrices().detach()
    anchor = strongest_commutator_channel(base_actions)
    macro_elements = {}
    macro_actions = {}
    for index in (0, 1, 2):
        macro_elements[index], macro_actions[index] = macro_alphabet(
            group, original_inputs, base_actions, index
        )

    fit_original = collect_strata(
        model,
        group,
        base_actions,
        original_inputs,
        FIT_LENGTHS,
        batches=1,
        batch_size=256,
        seed_base=410_000,
        device=device,
    )
    fit_pair_zero = collect_strata(
        model,
        group,
        macro_actions[0],
        macro_elements[0],
        FIT_LENGTHS,
        batches=1,
        batch_size=256,
        seed_base=420_000,
        device=device,
    )
    selection_pair_one = collect_strata(
        model,
        group,
        macro_actions[1],
        macro_elements[1],
        DENSE_LENGTHS,
        batches=1,
        batch_size=512,
        seed_base=510_000,
        device=device,
    )

    raw_auxiliary = torch.nn.Parameter(torch.zeros(model.channels - 1, device=device))
    optimizer = torch.optim.Adam([raw_auxiliary], lr=learning_rate)
    best_key = (-math.inf, -math.inf, 0)
    best_step = 0
    best_gates = gate_vector(raw_auxiliary, anchor, model.channels).detach().clone()
    trajectory = []
    fit_strata = fit_original + fit_pair_zero
    for step in range(steps + 1):
        if step % 10 == 0:
            gates = gate_vector(raw_auxiliary, anchor, model.channels).detach()
            selection = accuracy_summary(selection_pair_one, gates)
            key = (
                selection["dense_minimum_accuracy"],
                selection["dense_mean_accuracy"],
                -step,
            )
            trajectory.append(
                {
                    "step": step,
                    "gates": [float(value) for value in gates],
                    "selection_dense_minimum_accuracy": selection[
                        "dense_minimum_accuracy"
                    ],
                    "selection_dense_mean_accuracy": selection[
                        "dense_mean_accuracy"
                    ],
                }
            )
            if key > best_key:
                best_key = key
                best_step = step
                best_gates = gates.clone()
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        gates = gate_vector(raw_auxiliary, anchor, model.channels)
        loss = robust_margin_objective(fit_strata, gates)
        loss.backward()
        optimizer.step()

    evaluation = {}
    alphabets = {
        "original": (original_inputs, base_actions),
        "pair_0": (macro_elements[0], macro_actions[0]),
        "pair_1": (macro_elements[1], macro_actions[1]),
        "pair_2_untouched": (macro_elements[2], macro_actions[2]),
    }
    anchor_gate = fixed_gate(model.channels, [anchor], device)
    full_gate = torch.ones(model.channels, device=device)
    for offset, (name, (elements, actions)) in enumerate(alphabets.items()):
        strata = collect_strata(
            model,
            group,
            actions,
            elements,
            DENSE_LENGTHS,
            batches=2,
            batch_size=512,
            seed_base=610_000 + 20_000 * offset,
            device=device,
        )
        evaluation[name] = {
            "anchor_only": accuracy_summary(strata, anchor_gate),
            "all_channels": accuracy_summary(strata, full_gate),
            "learned_gates": accuracy_summary(strata, best_gates),
        }

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "training_seed": config["seed"],
        "anchor_channel": anchor,
        "macro_elements": {
            str(index): list(elements) for index, elements in macro_elements.items()
        },
        "macro_element_names": {
            str(index): [group.elements[element] for element in elements]
            for index, elements in macro_elements.items()
        },
        "selected_step": best_step,
        "selected_gates": [float(value) for value in best_gates],
        "selection_trajectory": trajectory,
        "evaluation": evaluation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    results = [
        fit_checkpoint(path, device, args.steps, args.learning_rate)
        for path in args.checkpoints
    ]
    report = {
        "experiment": "frozen-action robust channel gating",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "torch_version": torch.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "fit_lengths": list(FIT_LENGTHS),
        "dense_lengths": list(DENSE_LENGTHS),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
