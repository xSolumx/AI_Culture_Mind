"""Compile recovered anonymous finite actions into quaternion/Householder charts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from latent_group_discovery import exact_inverse_tokens
from mechanistic_group_actions import (
    PureGroupActionModel,
    canonical_group_words,
    representation_diagnostics,
    streaming_equivalence,
)
from q8_spinor_joint_retraction import target_parameters
from representation_retraction import element_inverses, regular_irrep_candidates
from spin8_table_blind_compiler import (
    _negative_controls,
    discover_blind_section,
    posthoc_q8_score,
)


SUPPORTED_FAMILIES = ("pure_quaternion_spinor", "pure_householder4_shared")
ORBIT_RANK_TOLERANCE = 1e-5


def nearest_orthogonal(matrix: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(matrix)
    return left @ right


def align_orthogonal_representation(
    learned: np.ndarray, candidate: np.ndarray, *, seed: int
) -> tuple[np.ndarray, float]:
    """Find one simultaneous O(n) conjugacy from the intertwiner nullspace."""

    dimension = learned.shape[-1]
    identity = np.eye(dimension)
    system = np.concatenate(
        [
            np.kron(identity, left) - np.kron(right.T, identity)
            for left, right in zip(learned, candidate)
        ]
    )
    _, singular, right_vectors = np.linalg.svd(system)
    tolerance = max(system.shape) * np.finfo(np.float64).eps * singular[0]
    nullity = max(1, int(np.sum(singular <= max(tolerance, 1e-8))))
    basis = right_vectors[-nullity:]
    coefficients = [np.eye(nullity)[index] for index in range(nullity)]
    rng = np.random.default_rng(seed)
    coefficients.extend(rng.normal(size=(64, nullity)))
    best_change = np.eye(dimension)
    best_rms = np.inf
    for weights in coefficients:
        raw = (weights @ basis).reshape(dimension, dimension, order="F")
        if np.linalg.norm(raw) <= 1e-12:
            continue
        change = nearest_orthogonal(raw)
        residual = learned - change[None] @ candidate @ change.T[None]
        rms = float(np.sqrt(np.mean(np.square(residual))))
        if rms < best_rms:
            best_rms = rms
            best_change = change
    return best_change, best_rms


def matrix_to_householders(matrix: np.ndarray) -> np.ndarray:
    """Factor an SO(n) matrix into n reflections in application order."""

    dimension = matrix.shape[0]
    current = np.asarray(matrix, dtype=np.float64).copy()
    reductions: list[np.ndarray] = []
    for column in range(dimension):
        source = current[column:, column]
        target = np.zeros_like(source)
        target[0] = 1.0
        difference = source - target
        if np.linalg.norm(difference) <= 1e-10:
            vector = np.zeros(dimension)
            reflection = np.eye(dimension)
        else:
            vector = np.zeros(dimension)
            vector[column:] = difference / np.linalg.norm(difference)
            reflection = np.eye(dimension) - 2.0 * np.outer(vector, vector)
        current = reflection @ current
        reductions.append(vector)
    if not np.allclose(current, np.eye(dimension), atol=2e-7, rtol=2e-7):
        raise ValueError("orthogonal matrix did not reduce to identity")
    # Model application produces H(last) ... H(first), hence reverse the
    # reductions because matrix = H0 ... H(n-1).
    parameters = np.stack(reductions[::-1])
    reconstructed = np.eye(dimension)
    for vector in parameters:
        norm = np.linalg.norm(vector)
        if norm > 1e-12:
            unit = vector / norm
            reflection = np.eye(dimension) - 2.0 * np.outer(unit, unit)
            reconstructed = reflection @ reconstructed
    if not np.allclose(reconstructed, matrix, atol=2e-7, rtol=2e-7):
        raise RuntimeError("Householder reconstruction failed")
    return parameters


def quaternion_targets(
    model: PureGroupActionModel, next_states: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    inverse_tokens = exact_inverse_tokens(next_states)
    first = 0
    first_inverse = inverse_tokens[first]
    second = min(
        token for token in range(model.vocab_size)
        if token not in (first, first_inverse)
    )
    second_inverse = inverse_tokens[second]
    before = (
        model.token_actions(torch.arange(model.vocab_size, device=model.action_parameters.device))
        .detach().cpu().double().numpy()
    )
    targets = np.zeros_like(before)
    projection = []
    for channel in range(model.channels):
        axis_a = before[first, channel, 1:] - before[first_inverse, channel, 1:]
        axis_b = before[second, channel, 1:] - before[second_inverse, channel, 1:]
        axis_a /= np.linalg.norm(axis_a).clip(min=1e-12)
        axis_b -= np.dot(axis_b, axis_a) * axis_a
        axis_b /= np.linalg.norm(axis_b).clip(min=1e-12)
        targets[first, channel, 1:] = axis_a
        targets[first_inverse, channel, 1:] = -axis_a
        targets[second, channel, 1:] = axis_b
        targets[second_inverse, channel, 1:] = -axis_b
        projection.append(float(np.sqrt(np.mean(np.square(targets[:, channel] - before[:, channel])))))
    return targets, {
        "inferred_inverse_tokens": list(inverse_tokens),
        "generator_pair": [first, second],
        "per_channel_action_projection_rms": projection,
    }


def householder_targets(
    model: PureGroupActionModel, recovered, *, seed: int
) -> tuple[np.ndarray, dict[str, object]]:
    candidates = regular_irrep_candidates(
        recovered.group, 4, seed=91_000 + seed
    )
    if len(candidates) != 1:
        raise ValueError(
            f"recovered table exposed {len(candidates)} inequivalent 4D irreps"
        )
    inverses = element_inverses(recovered.group)
    candidate_tokens = np.stack(
        [candidates[0].actions[inverses[element]] for element in recovered.input_elements]
    )
    learned = model.action_matrices().detach().cpu().double().numpy()[..., :4, :4]
    parameters = np.zeros_like(model.action_parameters.detach().cpu().numpy())
    alignment_rms = []
    reconstruction = []
    for channel in range(model.channels):
        change, rms = align_orthogonal_representation(
            learned[:, channel], candidate_tokens, seed=seed + 101 * channel
        )
        exact = change[None] @ candidate_tokens @ change.T[None]
        alignment_rms.append(rms)
        for token in range(model.vocab_size):
            parameters[token, channel] = matrix_to_householders(exact[token])
            vectors = parameters[token, channel]
            rebuilt = np.eye(4)
            for vector in vectors:
                norm = np.linalg.norm(vector)
                if norm > 1e-12:
                    unit = vector / norm
                    rebuilt = (np.eye(4) - 2 * np.outer(unit, unit)) @ rebuilt
            reconstruction.append(float(np.max(np.abs(rebuilt - exact[token]))))
    return parameters, {
        "recovered_4d_irrep_count": len(candidates),
        "per_channel_simultaneous_alignment_rms": alignment_rms,
        "factorization_reconstruction_max_abs": max(reconstruction),
    }


@torch.no_grad()
def exact_model_orbit(model: PureGroupActionModel, recovered) -> np.ndarray:
    words = canonical_group_words(recovered.group, recovered.input_elements)
    columns = []
    for word in words:
        state = model.initial_state(1)
        for token in word:
            state = model.step(
                torch.tensor([token], device=state.device), state
            )
        columns.append(state[0].detach().cpu().double().numpy())
    return np.stack(columns, axis=-1)


def compile_family_checkpoint(
    source: Path, destination: Path, *, device: torch.device
):
    checkpoint = torch.load(source, map_location=device, weights_only=False)
    family = checkpoint["family"]
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"unsupported family {family!r}")
    config = checkpoint["config"]
    model = PureGroupActionModel(
        4, 8, family=family, channels=int(config["channels"]),
        max_rotor_angle=float(config["max_angle"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    seed = int(config["seed"])
    section = discover_blind_section(
        model, state_count=8, token_count=4,
        seed_base=12_500_000 + 10_000 * seed, device=device,
    )
    recovered = section.recovered
    next_states = section.transition_votes.argmax(axis=-1)
    old_weight = model.output_head.weight.detach().cpu().double().numpy()
    old_flat = section.orbit.reshape(-1, 8)
    desired_logits = old_weight @ old_flat
    if family == "pure_quaternion_spinor":
        targets, family_metrics = quaternion_targets(model, next_states)
        parameters = target_parameters(targets, float(config["max_angle"]))
    else:
        parameters, family_metrics = householder_targets(
            model, recovered, seed=seed
        )
    with torch.no_grad():
        model.action_parameters.copy_(torch.as_tensor(
            parameters, dtype=model.action_parameters.dtype, device=device
        ))
    exact_orbit = exact_model_orbit(model, recovered)
    exact_flat = exact_orbit.reshape(-1, 8)
    singular_values = np.linalg.svd(exact_flat, compute_uv=False)
    exact_rank = int(np.sum(singular_values > ORBIT_RANK_TOLERANCE))
    nonzero_condition = float(
        singular_values[0] / singular_values[exact_rank - 1]
    ) if exact_rank else float("inf")
    section_pinv = np.linalg.pinv(exact_flat, rcond=ORBIT_RANK_TOLERANCE)
    row_projector = section_pinv @ exact_flat
    realizable_logits = desired_logits @ row_projector
    discarded_logit_rms = float(
        np.sqrt(np.mean(np.square(realizable_logits - desired_logits)))
    )
    anonymous_targets = recovered.element_to_state
    projected_predictions = realizable_logits.argmax(axis=0)
    projected_accuracy = float(np.mean(projected_predictions == anonymous_targets))
    correct_scores = realizable_logits[anonymous_targets, np.arange(8)]
    masked = realizable_logits.copy()
    masked[anonymous_targets, np.arange(8)] = -np.inf
    projected_margin = float(np.min(correct_scores - masked.max(axis=0)))
    transported = old_weight + (
        realizable_logits - old_weight @ exact_flat
    ) @ section_pinv
    with torch.no_grad():
        model.output_head.weight.copy_(torch.as_tensor(
            transported, dtype=model.output_head.weight.dtype, device=device
        ))
    actual_weight = model.output_head.weight.detach().cpu().double().numpy()
    logit_transport = float(
        np.max(np.abs(actual_weight @ exact_flat - realizable_logits))
    )
    diagnostics = representation_diagnostics(
        model, recovered.group, recovered.input_elements
    )
    streaming = streaming_equivalence(
        model, torch.arange(128, device=device).reshape(4, 32) % 4
    )
    gates = {
        "transition_winner_fraction": float(section.winner_fractions.min()) >= 0.99,
        "transition_vote_gap": float(section.vote_gaps.min()) >= 0.98,
        "recovered_homomorphism": diagnostics["linear_homomorphism_rms"] <= 1e-5,
        "theoretical_section_rank": exact_rank == 4,
        "projected_anonymous_accuracy": projected_accuracy == 1.0,
        "projected_minimum_margin": projected_margin > 0.0,
        "centroid_logit_transport": logit_transport <= 1e-5,
        "streaming_state": max(
            streaming["chunked_state_max_abs_error"],
            streaming["streaming_state_max_abs_error"],
        ) <= 1e-5,
        "streaming_logits": max(
            streaming["chunked_logit_max_abs_error"],
            streaming["streaming_logit_max_abs_error"],
        ) <= 1e-4,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    compiled = dict(checkpoint)
    compiled["config"] = {
        **config,
        "table_blind_family_compiler": True,
        "recovered_group_table": recovered.group.table.tolist(),
        "recovered_input_elements": list(recovered.input_elements),
        "recovered_element_to_anonymous_class": recovered.element_to_state.tolist(),
    }
    compiled["state_dict"] = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    torch.save(compiled, destination)
    result = {
        "source": str(source), "destination": str(destination), "seed": seed,
        "family": family, "hidden_table_used_by_compiler": False,
        "target_labels_used_by_compiler": False,
        "gradient_steps_after_compilation": 0,
        "anonymous_class_counts": section.class_counts.tolist(),
        "transition_winner_fraction_min": float(section.winner_fractions.min()),
        "transition_vote_gap_min": float(section.vote_gaps.min()),
        "recovered_next_states": next_states.tolist(),
        "recovered_group_table": recovered.group.table.tolist(),
        "recovered_input_elements": list(recovered.input_elements),
        "family_metrics": family_metrics,
        "exact_section_rank": exact_rank,
        "exact_section_singular_values": singular_values.tolist(),
        "exact_section_nonzero_condition_number": nonzero_condition,
        "discarded_teacher_logit_rms": discarded_logit_rms,
        "projected_anonymous_accuracy": projected_accuracy,
        "projected_minimum_margin": projected_margin,
        "centroid_logit_transport_max_abs": logit_transport,
        "representation_diagnostics_recovered_table": diagnostics,
        "streaming_equivalence": streaming,
        "negative_controls": _negative_controls(next_states, seed=seed + 72_000),
        "compiler_gates": gates,
        "compiler_passed": all(gates.values()),
    }
    return result, recovered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    torch.use_deterministic_algorithms(True)
    result, recovered = compile_family_checkpoint(
        args.source, args.destination, device=device
    )
    result = posthoc_q8_score(result, recovered, args.destination, device=device)
    report = {"experiment": "table-blind family baseline", "result": result,
              "passed": result["passed"]}
    rendered = json.dumps(report, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
