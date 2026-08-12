"""Zero-shot transfer to compiled unseen A5 generators from frozen checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from compare_recurrences import GROUPS, FiniteGroup, make_group_batches
from mechanistic_group_actions import (
    PureGroupActionModel,
    _compose_word,
    canonical_group_words,
)
from rotor_ssm_torch import GA_DIM


def element_order(group: FiniteGroup, element: int) -> int:
    product = 0
    for order in range(1, group.order + 1):
        product = int(group.table[product, element])
        if product == 0:
            return order
    raise RuntimeError(f"element {element} has no finite order")


def inverse_element(group: FiniteGroup, element: int) -> int:
    candidates = np.flatnonzero(group.table[element] == 0)
    if len(candidates) != 1:
        raise RuntimeError(f"element {element} has no unique inverse")
    return int(candidates[0])


def generated_subgroup(
    group: FiniteGroup, generators: tuple[int, ...]
) -> set[int]:
    seen = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for generator in generators:
            product = int(group.table[current, generator])
            if product not in seen:
                seen.add(product)
                frontier.append(product)
    return seen


def select_changed_generators(
    group: FiniteGroup,
    original_inputs: tuple[int, ...],
    selection_index: int = 0,
) -> tuple[int, int, int, int]:
    """Select a ranked macro alphabet, quotienting inverse re-labellings."""

    if selection_index < 0:
        raise ValueError("selection_index must be non-negative")
    excluded = set(original_inputs)
    order_three = [
        element
        for element in range(1, group.order)
        if element not in excluded and element_order(group, element) == 3
    ]
    order_five = [
        element
        for element in range(1, group.order)
        if element not in excluded and element_order(group, element) == 5
    ]
    seen_equivalence_classes: set[
        tuple[tuple[int, int], tuple[int, int]]
    ] = set()
    qualifying: list[tuple[int, int, int, int]] = []
    for first in order_three:
        first_inverse = inverse_element(group, first)
        for second in order_five:
            second_inverse = inverse_element(group, second)
            macros = (first, first_inverse, second, second_inverse)
            if len(set(macros)) != 4 or set(macros) & excluded:
                continue
            equivalence_class = (
                tuple(sorted((first, first_inverse))),
                tuple(sorted((second, second_inverse))),
            )
            if equivalence_class in seen_equivalence_classes:
                continue
            if len(generated_subgroup(group, (first, second))) != group.order:
                continue
            seen_equivalence_classes.add(equivalence_class)
            qualifying.append(macros)
            if len(qualifying) > selection_index:
                return qualifying[selection_index]
    raise RuntimeError(
        f"changed generator-set index {selection_index} is unavailable; "
        f"found {len(qualifying)} qualifying equivalence classes"
    )


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def strongest_commutator_channel(actions: torch.Tensor) -> int:
    values = torch.zeros(actions.shape[1], device=actions.device)
    for left in range(actions.shape[0]):
        for right in range(left + 1, actions.shape[0]):
            difference = actions[right] @ actions[left] - actions[left] @ actions[right]
            values = torch.maximum(
                values,
                difference.square().sum(dim=(-2, -1)).sqrt() / math.sqrt(GA_DIM),
            )
    return int(values.argmax())


@torch.no_grad()
def evaluate_macro_actions(
    model: PureGroupActionModel,
    macro_actions: torch.Tensor,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    strong_channel: int,
    device: torch.device,
    canonical_prototypes: torch.Tensor | None = None,
    custom_gates: torch.Tensor | None = None,
) -> dict[str, object]:
    full_correct = strong_correct = examples = 0
    custom_gate_correct = 0
    only_channel_correct = [0] * model.channels
    leave_one_out_correct = [0] * model.channels
    channel_subsets = [
        subset
        for size in range(1, model.channels + 1)
        for subset in itertools.combinations(range(model.channels), size)
    ]
    subset_correct = {"+".join(map(str, subset)): 0 for subset in channel_subsets}
    subset_repairs = {"+".join(map(str, subset)): 0 for subset in channel_subsets}
    subset_damages = {"+".join(map(str, subset)): 0 for subset in channel_subsets}
    subset_both_correct = dict.fromkeys(subset_correct, 0)
    subset_both_wrong_same = dict.fromkeys(subset_correct, 0)
    subset_both_wrong_different = dict.fromkeys(subset_correct, 0)
    repaired_anchor_margin_sum = dict.fromkeys(subset_correct, 0.0)
    repaired_residual_margin_sum = dict.fromkeys(subset_correct, 0.0)
    defect_correlation_sums = {
        subset: {
            "count": 0,
            "x": 0.0,
            "y": 0.0,
            "xx": 0.0,
            "yy": 0.0,
            "xy": 0.0,
            "positive_margin": 0,
            "cosine_sum": 0.0,
            "positive_cosine": 0,
        }
        for subset in subset_correct
    }
    canonical_strong_correct = 0
    maximum_norm_error = 0.0
    for tokens, targets in batches:
        tokens, targets = tokens.to(device), targets.to(device)
        state = model.initial_state(len(tokens))
        for position in range(tokens.shape[1]):
            selected = macro_actions[tokens[:, position]]
            state = torch.einsum("bcij,bcj->bci", selected, state)
        target = targets[:, -1]
        full_predictions = model.decode(state).argmax(-1)
        if custom_gates is not None:
            custom_predictions = model.decode(
                state * custom_gates[None, :, None]
            ).argmax(-1)
            custom_gate_correct += int((custom_predictions == target).sum())
        strong_state = torch.zeros_like(state)
        strong_state[:, strong_channel] = state[:, strong_channel]
        strong_logits = model.decode(strong_state)
        strong_predictions = strong_logits.argmax(-1)
        strong_is_correct = strong_predictions == target
        target_mask = torch.nn.functional.one_hot(
            target, num_classes=strong_logits.shape[-1]
        ).bool()
        strong_competitor = strong_logits.masked_fill(target_mask, -torch.inf).argmax(-1)
        strong_true_logits = strong_logits.gather(1, target[:, None]).squeeze(1)
        strong_competitor_logits = strong_logits.gather(
            1, strong_competitor[:, None]
        ).squeeze(1)
        strong_true_margin = strong_true_logits - strong_competitor_logits
        defect_against_truth = None
        canonical_strong_is_correct = None
        if canonical_prototypes is not None:
            canonical_strong_state = torch.zeros_like(state)
            canonical_strong_state[:, strong_channel] = canonical_prototypes[
                target, strong_channel
            ]
            canonical_strong_logits = model.decode(canonical_strong_state)
            canonical_strong_is_correct = canonical_strong_logits.argmax(-1) == target
            canonical_strong_correct += int(
                canonical_strong_is_correct.sum()
            )
            defect_logits = strong_logits - canonical_strong_logits
            defect_against_truth = (
                defect_logits.gather(1, strong_competitor[:, None]).squeeze(1)
                - defect_logits.gather(1, target[:, None]).squeeze(1)
            )
        for channel in range(model.channels):
            only_channel = torch.zeros_like(state)
            only_channel[:, channel] = state[:, channel]
            without_channel = state.clone()
            without_channel[:, channel] = 0.0
            only_channel_correct[channel] += int(
                (model.decode(only_channel).argmax(-1) == targets[:, -1]).sum()
            )
            leave_one_out_correct[channel] += int(
                (model.decode(without_channel).argmax(-1) == targets[:, -1]).sum()
            )
        for subset in channel_subsets:
            subset_state = torch.zeros_like(state)
            subset_state[:, subset] = state[:, subset]
            subset_name = "+".join(map(str, subset))
            subset_logits = model.decode(subset_state)
            subset_predictions = subset_logits.argmax(-1)
            subset_is_correct = subset_predictions == target
            repaired = (~strong_is_correct) & subset_is_correct
            damaged = strong_is_correct & (~subset_is_correct)
            both_correct = strong_is_correct & subset_is_correct
            both_wrong = (~strong_is_correct) & (~subset_is_correct)
            both_wrong_same = both_wrong & (strong_predictions == subset_predictions)
            both_wrong_different = both_wrong & (
                strong_predictions != subset_predictions
            )
            subset_correct[subset_name] += int(subset_is_correct.sum())
            subset_repairs[subset_name] += int(repaired.sum())
            subset_damages[subset_name] += int(damaged.sum())
            subset_both_correct[subset_name] += int(both_correct.sum())
            subset_both_wrong_same[subset_name] += int(both_wrong_same.sum())
            subset_both_wrong_different[subset_name] += int(
                both_wrong_different.sum()
            )
            residual_logits = subset_logits - strong_logits
            residual_compensating_margin = (
                residual_logits.gather(1, target[:, None]).squeeze(1)
                - residual_logits.gather(1, strong_competitor[:, None]).squeeze(1)
            )
            repaired_anchor_margin_sum[subset_name] += float(
                strong_true_margin[repaired].sum()
            )
            repaired_residual_margin_sum[subset_name] += float(
                residual_compensating_margin[repaired].sum()
            )
            if defect_against_truth is not None:
                correlation_mask = (~strong_is_correct) & canonical_strong_is_correct
                x = defect_against_truth[correlation_mask].double()
                y = residual_compensating_margin[correlation_mask].double()
                sums = defect_correlation_sums[subset_name]
                sums["count"] += len(x)
                sums["x"] += float(x.sum())
                sums["y"] += float(y.sum())
                sums["xx"] += float(x.square().sum())
                sums["yy"] += float(y.square().sum())
                sums["xy"] += float((x * y).sum())
                sums["positive_margin"] += int((y > 0.0).sum())
                centered_defect = defect_logits[correlation_mask].double()
                centered_defect -= centered_defect.mean(dim=-1, keepdim=True)
                centered_residual = residual_logits[correlation_mask].double()
                centered_residual -= centered_residual.mean(dim=-1, keepdim=True)
                cosine = torch.nn.functional.cosine_similarity(
                    -centered_defect, centered_residual, dim=-1, eps=1e-12
                )
                sums["cosine_sum"] += float(cosine.sum())
                sums["positive_cosine"] += int((cosine > 0.0).sum())
        full_correct += int((full_predictions == target).sum())
        strong_correct += int((strong_predictions == target).sum())
        examples += len(tokens)
        maximum_norm_error = max(
            maximum_norm_error,
            float((state.norm(dim=-1) - 1.0).abs().max()),
        )
    defect_correlations = {}
    corrective_margin_direction_rates = {}
    defect_residual_cosine_means = {}
    defect_residual_positive_cosine_rates = {}
    for subset, sums in defect_correlation_sums.items():
        count = sums["count"]
        covariance = count * sums["xy"] - sums["x"] * sums["y"]
        x_variance = count * sums["xx"] - sums["x"] ** 2
        y_variance = count * sums["yy"] - sums["y"] ** 2
        denominator = math.sqrt(max(0.0, x_variance * y_variance))
        defect_correlations[subset] = (
            covariance / denominator if count >= 2 and denominator > 0.0 else None
        )
        corrective_margin_direction_rates[subset] = (
            sums["positive_margin"] / count if count else None
        )
        defect_residual_cosine_means[subset] = (
            sums["cosine_sum"] / count if count else None
        )
        defect_residual_positive_cosine_rates[subset] = (
            sums["positive_cosine"] / count if count else None
        )
    return {
        "full_final_position_accuracy": full_correct / examples,
        "strong_channel_only_final_position_accuracy": strong_correct / examples,
        "custom_gate_final_position_accuracy": (
            custom_gate_correct / examples if custom_gates is not None else None
        ),
        "only_channel_final_position_accuracy": [
            correct / examples for correct in only_channel_correct
        ],
        "leave_one_channel_out_final_position_accuracy": [
            correct / examples for correct in leave_one_out_correct
        ],
        "channel_subset_final_position_accuracy": {
            subset: correct / examples for subset, correct in subset_correct.items()
        },
        "channel_subset_repairs_over_strong_rate": {
            subset: repaired / examples for subset, repaired in subset_repairs.items()
        },
        "channel_subset_damages_over_strong_rate": {
            subset: damaged / examples for subset, damaged in subset_damages.items()
        },
        "channel_subset_both_correct_rate": {
            subset: count / examples for subset, count in subset_both_correct.items()
        },
        "channel_subset_both_wrong_same_prediction_rate": {
            subset: count / examples
            for subset, count in subset_both_wrong_same.items()
        },
        "channel_subset_both_wrong_different_prediction_rate": {
            subset: count / examples
            for subset, count in subset_both_wrong_different.items()
        },
        "channel_subset_repaired_anchor_margin_mean": {
            subset: (
                repaired_anchor_margin_sum[subset] / subset_repairs[subset]
                if subset_repairs[subset]
                else None
            )
            for subset in subset_correct
        },
        "channel_subset_repaired_residual_compensating_margin_mean": {
            subset: (
                repaired_residual_margin_sum[subset] / subset_repairs[subset]
                if subset_repairs[subset]
                else None
            )
            for subset in subset_correct
        },
        "channel_subset_anchor_defect_residual_pearson_on_canonical_correct": (
            defect_correlations
        ),
        "channel_subset_corrective_margin_direction_rate_on_canonical_correct": (
            corrective_margin_direction_rates
        ),
        "channel_subset_defect_residual_cosine_mean_on_canonical_correct": (
            defect_residual_cosine_means
        ),
        "channel_subset_defect_residual_positive_cosine_rate_on_canonical_correct": (
            defect_residual_positive_cosine_rates
        ),
        "strong_canonical_prototype_accuracy": (
            canonical_strong_correct / examples
            if canonical_prototypes is not None
            else None
        ),
        "maximum_channel_norm_error": maximum_norm_error,
    }


def run_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    validation_batches: int,
    validation_batch_size: int,
    generator_set_index: int,
    custom_gates: list[float] | None = None,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    group = GROUPS[checkpoint["group"]]
    original_inputs = tuple(checkpoint["input_elements"])
    config = checkpoint["config"]
    model = PureGroupActionModel(
        len(original_inputs),
        group.order,
        family=checkpoint["family"],
        channels=config["channels"],
        max_rotor_angle=config["max_rotor_angle"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    base_actions = model.action_matrices()
    words = canonical_group_words(group, original_inputs)
    macros = select_changed_generators(
        group, original_inputs, selection_index=generator_set_index
    )
    macro_words = tuple(words[element] for element in macros)
    macro_actions = torch.stack(
        [_compose_word(base_actions, word) for word in macro_words]
    )
    canonical_actions = torch.stack(
        [_compose_word(base_actions, words[element]) for element in range(group.order)]
    )
    canonical_prototypes = torch.einsum(
        "gcij,cj->gci", canonical_actions, model.initial_state(1)[0]
    )
    strong_channel = strongest_commutator_channel(base_actions)
    custom_gate_tensor = (
        torch.tensor(custom_gates, device=device, dtype=base_actions.dtype)
        if custom_gates is not None
        else None
    )
    if custom_gate_tensor is not None and len(custom_gate_tensor) != model.channels:
        raise ValueError(
            f"expected {model.channels} custom gates, got {len(custom_gate_tensor)}"
        )
    lengths = [2, 4, 8, *range(16, 257, 16)]
    length_results = {}
    for length in lengths:
        batches = make_group_batches(
            group,
            validation_batches,
            validation_batch_size,
            length,
            271_000 + length,
            input_elements=macros,
        )
        length_results[str(length)] = evaluate_macro_actions(
            model,
            macro_actions,
            batches,
            strong_channel,
            device,
            canonical_prototypes,
            custom_gate_tensor,
        )
    dense_full = [
        length_results[str(length)]["full_final_position_accuracy"]
        for length in range(16, 257, 16)
    ]
    dense_strong = [
        length_results[str(length)]["strong_channel_only_final_position_accuracy"]
        for length in range(16, 257, 16)
    ]
    dense_custom = (
        [
            length_results[str(length)]["custom_gate_final_position_accuracy"]
            for length in range(16, 257, 16)
        ]
        if custom_gates is not None
        else None
    )
    dense_only_by_channel = [
        [
            length_results[str(length)]["only_channel_final_position_accuracy"][channel]
            for length in range(16, 257, 16)
        ]
        for channel in range(model.channels)
    ]
    dense_without_by_channel = [
        [
            length_results[str(length)]["leave_one_channel_out_final_position_accuracy"][channel]
            for length in range(16, 257, 16)
        ]
        for channel in range(model.channels)
    ]
    subset_names = length_results["16"][
        "channel_subset_final_position_accuracy"
    ].keys()
    dense_by_subset = {
        subset: [
            length_results[str(length)][
                "channel_subset_final_position_accuracy"
            ][subset]
            for length in range(16, 257, 16)
        ]
        for subset in subset_names
    }
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "family": checkpoint["family"],
        "training_seed": config["seed"],
        "generator_set_index": generator_set_index,
        "original_input_elements": list(original_inputs),
        "changed_macro_elements": list(macros),
        "changed_macro_element_names": [group.elements[element] for element in macros],
        "changed_macro_orders": [element_order(group, element) for element in macros],
        "compiled_original_token_words": [list(word) for word in macro_words],
        "compiled_word_lengths": [len(word) for word in macro_words],
        "generated_subgroup_order": len(generated_subgroup(group, (macros[0], macros[2]))),
        "strongest_original_commutator_channel": strong_channel,
        "length_generalization": length_results,
        "dense_full_minimum_accuracy": min(dense_full),
        "dense_strong_channel_minimum_accuracy": min(dense_strong),
        "custom_gates": custom_gates,
        "dense_custom_gate_minimum_accuracy": (
            min(dense_custom) if dense_custom is not None else None
        ),
        "dense_only_channel_minimum_accuracy": [
            min(values) for values in dense_only_by_channel
        ],
        "dense_leave_one_channel_out_minimum_accuracy": [
            min(values) for values in dense_without_by_channel
        ],
        "dense_channel_subset_minimum_accuracy": {
            subset: min(values) for subset, values in dense_by_subset.items()
        },
        "dense_full_gate_pass": min(dense_full) >= 0.9,
        "dense_strong_channel_gate_pass": min(dense_strong) >= 0.9,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--validation-batches", type=int, default=2)
    parser.add_argument("--validation-batch-size", type=int, default=512)
    parser.add_argument("--generator-set-index", type=int, default=0)
    parser.add_argument(
        "--gate-report",
        type=Path,
        help="optional robust_channel_gating report for paired custom-gate scoring",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    gates_by_hash = {}
    if args.gate_report:
        gate_report = json.loads(args.gate_report.read_text(encoding="utf-8"))
        gates_by_hash = {
            result["checkpoint_sha256"]: result["selected_gates"]
            for result in gate_report["results"]
        }
    results = []
    for path in args.checkpoints:
        digest = checkpoint_sha256(path)
        if args.gate_report and digest not in gates_by_hash:
            raise KeyError(f"gate report has no entry for checkpoint {digest}")
        results.append(
            run_checkpoint(
                path,
                device,
                args.validation_batches,
                args.validation_batch_size,
                args.generator_set_index,
                gates_by_hash.get(digest),
            )
        )
    report = {
        "experiment": "zero-shot compiled changed-generator A5 transfer",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "torch_version": torch.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "selection_rule": (
            "ranked disjoint order-3/order-5 generating-pair equivalence class "
            "in repository order; inverse re-labellings are quotiented"
        ),
        "generator_set_index": args.generator_set_index,
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
