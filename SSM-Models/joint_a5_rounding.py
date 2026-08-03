"""Compare learned, independently rounded, and jointly exact A5 anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from a5_anchor_representation_audit import align_representation, nearest_rotation
from changed_generator_transfer import select_changed_generators, strongest_commutator_channel
from compare_recurrences import GROUPS, FiniteGroup, make_group_batches
from mechanistic_group_actions import (
    PureGroupActionModel,
    _compose_word,
    _element_inverses,
    _element_orders,
    _rotation_matrix_to_bivector,
    a5_orthogonal_irrep,
    canonical_group_words,
)
from robust_channel_gating import final_states
from rotor_ssm_torch import GA_DIM, rotor_from_bivector, rotor_sandwich


DENSE_LENGTHS = tuple(range(16, 257, 16))
VARIANTS = ("learned", "independent_angle", "joint_exact")


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def full_rotor_action_matrices(
    rotations: np.ndarray,
    max_rotor_angle: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    parameters = np.stack(
        [_rotation_matrix_to_bivector(matrix, max_rotor_angle) for matrix in rotations]
    )
    bivectors = torch.as_tensor(parameters, dtype=dtype, device=device)
    rotors = rotor_from_bivector(bivectors, max_rotor_angle)
    basis = torch.eye(GA_DIM, dtype=dtype, device=device)
    transformed = rotor_sandwich(rotors[:, None, :], basis[None, :, :])
    matrices = transformed.transpose(-1, -2)
    expected = torch.as_tensor(rotations, dtype=dtype, device=device)
    error = (matrices[:, 1:4, 1:4] - expected).abs().max()
    if float(error) > 2e-5:
        raise RuntimeError(f"rotor conversion vector-block error {float(error):.3e}")
    return matrices


def independent_angle_rounding(
    learned: np.ndarray, oracle: np.ndarray
) -> np.ndarray:
    rounded = []
    for learned_matrix, oracle_matrix in zip(learned, oracle):
        learned_log = Rotation.from_matrix(nearest_rotation(learned_matrix)).as_rotvec()
        learned_angle = float(np.linalg.norm(learned_log))
        if learned_angle <= 1e-10:
            oracle_log = Rotation.from_matrix(oracle_matrix).as_rotvec()
            axis = oracle_log / np.linalg.norm(oracle_log)
        else:
            axis = learned_log / learned_angle
        exact_angle = float(Rotation.from_matrix(oracle_matrix).magnitude())
        rounded.append(Rotation.from_rotvec(axis * exact_angle).as_matrix())
    return np.stack(rounded)


def anchor_mechanism_diagnostics(
    actions: torch.Tensor,
    group: FiniteGroup,
    input_elements: tuple[int, ...],
) -> dict[str, object]:
    actions = actions.double()
    words = canonical_group_words(group, input_elements)
    operators = torch.stack(
        [_compose_word(actions[:, None], word)[0] for word in words]
    )
    products = torch.as_tensor(group.table, dtype=torch.long, device=actions.device)
    composed = operators.unsqueeze(0) @ operators.unsqueeze(1)
    difference = composed - operators[products]
    vector_difference = difference[..., 1:4, 1:4]
    pair_rms = vector_difference.square().mean(dim=(-2, -1)).sqrt()
    vector_identity = torch.eye(3, dtype=actions.dtype, device=actions.device)
    vector_actions = actions[:, 1:4, 1:4]
    orthogonality = vector_actions.transpose(-1, -2) @ vector_actions - vector_identity
    orders = _element_orders(group)
    relators = []
    for token, element in enumerate(input_elements):
        residual = torch.linalg.matrix_power(vector_actions[token], int(orders[element])) - vector_identity
        relators.append(
            {
                "tokens": [token],
                "rms": float(residual.square().mean().sqrt()),
            }
        )
    for left in range(len(input_elements)):
        for right in range(left + 1, len(input_elements)):
            product = int(group.table[input_elements[left], input_elements[right]])
            product_action = vector_actions[right] @ vector_actions[left]
            residual = torch.linalg.matrix_power(product_action, int(orders[product])) - vector_identity
            relators.append(
                {
                    "tokens": [left, right],
                    "rms": float(residual.square().mean().sqrt()),
                }
            )
    cyclic = [entry["rms"] for entry in relators if len(entry["tokens"]) == 1]
    mixed = [entry["rms"] for entry in relators if len(entry["tokens"]) == 2]
    return {
        "vector_homomorphism_rms": float(pair_rms.square().mean().sqrt()),
        "vector_homomorphism_max": float(pair_rms.max()),
        "vector_orthogonality_rms": float(orthogonality.square().mean().sqrt()),
        "cyclic_relator_rms_max": max(cyclic),
        "mixed_relator_rms_max": max(mixed),
        "relators": relators,
    }


def compiled_actions(
    actions: torch.Tensor,
    group: FiniteGroup,
    original_inputs: tuple[int, ...],
    generator_set_index: int | None,
) -> tuple[tuple[int, ...], torch.Tensor]:
    if generator_set_index is None:
        return original_inputs, actions
    elements = select_changed_generators(
        group, original_inputs, selection_index=generator_set_index
    )
    words = canonical_group_words(group, original_inputs)
    return elements, torch.stack(
        [_compose_word(actions, words[element]) for element in elements]
    )


@torch.no_grad()
def evaluate_variants(
    model: PureGroupActionModel,
    group: FiniteGroup,
    original_inputs: tuple[int, ...],
    variant_actions: dict[str, torch.Tensor],
    gates: torch.Tensor,
    generator_set_index: int | None,
    lengths: tuple[int, ...],
    batches: int,
    batch_size: int,
    seed_base: int,
    device: torch.device,
) -> dict[str, object]:
    compiled = {
        variant: compiled_actions(
            actions, group, original_inputs, generator_set_index
        )
        for variant, actions in variant_actions.items()
    }
    input_elements = next(iter(compiled.values()))[0]
    by_variant = {variant: {} for variant in VARIANTS}
    norm_error = {variant: 0.0 for variant in VARIANTS}
    for length in lengths:
        generated = make_group_batches(
            group,
            batches,
            batch_size,
            length,
            seed_base + length,
            input_elements=input_elements,
        )
        correct = {variant: 0 for variant in VARIANTS}
        examples = 0
        for tokens, targets in generated:
            tokens = tokens.to(device)
            target = targets[:, -1].to(device)
            examples += len(tokens)
            for variant in VARIANTS:
                state = final_states(model, compiled[variant][1], tokens)
                predictions = model.decode(state * gates[None, :, None]).argmax(-1)
                correct[variant] += int((predictions == target).sum())
                norm_error[variant] = max(
                    norm_error[variant],
                    float((state.norm(dim=-1) - 1.0).abs().max()),
                )
        for variant in VARIANTS:
            by_variant[variant][str(length)] = correct[variant] / examples
    return {
        variant: {
            "by_length": values,
            "minimum_accuracy": min(values.values()),
            "mean_accuracy": sum(values.values()) / len(values),
            "gate_pass": min(values.values()) >= 0.9,
            "maximum_state_norm_error": norm_error[variant],
        }
        for variant, values in by_variant.items()
    }


def run_checkpoint(
    path: Path,
    selected_gates: list[float],
    device: torch.device,
) -> dict[str, object]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    group = GROUPS[checkpoint["group"]]
    input_elements = tuple(checkpoint["input_elements"])
    model = PureGroupActionModel(
        len(input_elements),
        group.order,
        family=checkpoint["family"],
        channels=config["channels"],
        max_rotor_angle=config["max_rotor_angle"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    learned_actions = model.action_matrices().detach()
    anchor = strongest_commutator_channel(learned_actions)
    learned_vector = learned_actions[:, anchor, 1:4, 1:4].cpu().double().numpy()
    representation = a5_orthogonal_irrep(group, branch=0)
    inverses = _element_inverses(group)
    oracle = np.stack(
        [representation[inverses[element]] for element in input_elements]
    )
    alignment, alignment_rms = align_representation(
        learned_vector, oracle, seed=20_000 + config["seed"]
    )
    exact_vector = alignment[None] @ oracle @ alignment.T[None]
    independent_vector = independent_angle_rounding(learned_vector, oracle)
    exact_full = full_rotor_action_matrices(
        exact_vector,
        config["max_rotor_angle"],
        learned_actions.dtype,
        device,
    )
    independent_full = full_rotor_action_matrices(
        independent_vector,
        config["max_rotor_angle"],
        learned_actions.dtype,
        device,
    )
    variant_actions = {"learned": learned_actions}
    for name, replacement in (
        ("independent_angle", independent_full),
        ("joint_exact", exact_full),
    ):
        actions = learned_actions.clone()
        actions[:, anchor] = replacement
        variant_actions[name] = actions
    gates = torch.tensor(selected_gates, dtype=learned_actions.dtype, device=device)

    evaluations = {}
    alphabet_indices: tuple[tuple[str, int | None], ...] = (
        ("original", None),
        ("class_0", 0),
        ("class_1", 1),
        ("class_2", 2),
        ("class_11_new_order3", 11),
    )
    for offset, (name, index) in enumerate(alphabet_indices):
        evaluations[name] = evaluate_variants(
            model,
            group,
            input_elements,
            variant_actions,
            gates,
            index,
            DENSE_LENGTHS,
            batches=2,
            batch_size=512,
            seed_base=810_000 + 20_000 * offset,
            device=device,
        )
    long_stress = {}
    for offset, (name, index) in enumerate(
        (("original", None), ("class_11_new_order3", 11))
    ):
        long_stress[name] = evaluate_variants(
            model,
            group,
            input_elements,
            variant_actions,
            gates,
            index,
            (4096,),
            batches=1,
            batch_size=512,
            seed_base=990_000 + 20_000 * offset,
            device=device,
        )
    diagnostics = {
        variant: anchor_mechanism_diagnostics(
            actions[:, anchor], group, input_elements
        )
        for variant, actions in variant_actions.items()
    }
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": checkpoint_sha256(path),
        "training_seed": config["seed"],
        "anchor_channel": anchor,
        "selected_gates": selected_gates,
        "joint_alignment_rms": alignment_rms,
        "mechanism_diagnostics": diagnostics,
        "dense_evaluation": evaluations,
        "long_stress": long_stress,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    gate_report = json.loads(args.gate_report.read_text(encoding="utf-8"))
    gates_by_hash = {
        result["checkpoint_sha256"]: result["selected_gates"]
        for result in gate_report["results"]
    }
    results = []
    for path in args.checkpoints:
        digest = checkpoint_sha256(path)
        if digest not in gates_by_hash:
            raise KeyError(f"gate report has no checkpoint {digest}")
        results.append(run_checkpoint(path, gates_by_hash[digest], device))
    report = {
        "experiment": "joint exact A5 anchor rounding",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "torch_version": torch.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "dense_lengths": list(DENSE_LENGTHS),
        "untouched_generator_class_index": 11,
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
