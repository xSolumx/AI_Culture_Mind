"""Recover and compile a finite Spin(8) action without a supplied group table."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from latent_group_discovery import RecoveredPermutationGroup, TransitionEvidence
from mechanistic_group_actions import (
    PureGroupActionModel,
    representation_diagnostics,
    streaming_equivalence,
)
from spin8_q8_joint_retraction import positive_spin8_parameters
from spin8_q8_path_section_compiler import minimum_change_observer
from spin8_q8_regular_orbit_retraction import (
    regular_ambient_actions,
    regular_orbit_projection,
)


CALIBRATION_LENGTHS = (15, 16)
CALIBRATION_BATCHES = 32
CALIBRATION_BATCH_SIZE = 512
MINIMUM_EDGE_VOTES = 128
MINIMUM_WINNER_FRACTION = 0.99
MINIMUM_VOTE_GAP = 0.98


@dataclass(frozen=True)
class BlindSection:
    orbit: np.ndarray
    recovered: RecoveredPermutationGroup
    class_counts: np.ndarray
    transition_votes: np.ndarray
    winner_fractions: np.ndarray
    vote_gaps: np.ndarray


def _final_state(model: PureGroupActionModel, tokens: torch.Tensor) -> torch.Tensor:
    state = model.initial_state(tokens.shape[0])
    actions = model.action_matrices()
    for position in range(tokens.shape[1]):
        state = torch.einsum(
            "bcij,bcj->bci", actions[tokens[:, position]], state
        )
    return state


@torch.no_grad()
def discover_blind_section(
    model: PureGroupActionModel,
    *,
    state_count: int,
    token_count: int,
    seed_base: int,
    device: torch.device,
) -> BlindSection:
    """Infer anonymous endpoint transitions from model predictions only."""

    if model.output_head.out_features != state_count:
        raise ValueError("declared state count does not match model observer")
    if model.vocab_size != token_count:
        raise ValueError("declared token count does not match model")
    sums = torch.zeros(
        state_count, model.channels, 8, dtype=torch.float64, device=device
    )
    counts = torch.zeros(state_count, dtype=torch.int64, device=device)
    votes = torch.zeros(
        state_count, token_count, state_count, dtype=torch.int64, device=device
    )
    actions = model.action_matrices()
    for length in CALIBRATION_LENGTHS:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed_base + length)
        for _ in range(CALIBRATION_BATCHES):
            tokens = torch.randint(
                token_count,
                (CALIBRATION_BATCH_SIZE, length),
                generator=generator,
                dtype=torch.long,
            ).to(device)
            states = _final_state(model, tokens)
            labels = model.decode(states[:, None])[:, 0].argmax(dim=-1)
            sums.index_add_(0, labels, states.to(torch.float64))
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.int64))
            for token in range(token_count):
                selected = actions[token].expand(states.shape[0], -1, -1, -1)
                successors = torch.einsum("bcij,bcj->bci", selected, states)
                next_labels = model.decode(successors[:, None])[:, 0].argmax(dim=-1)
                flat = labels * state_count + next_labels
                histogram = torch.bincount(flat, minlength=state_count * state_count)
                votes[:, token] += histogram.reshape(state_count, state_count)

    if bool((counts == 0).any()):
        raise ValueError(f"anonymous endpoint coverage failed: {counts.tolist()}")
    ordered, _ = votes.sort(dim=-1, descending=True)
    totals = votes.sum(dim=-1)
    if bool((totals < MINIMUM_EDGE_VOTES).any()):
        raise ValueError("one or more anonymous transition edges lack vote support")
    winner_fraction = ordered[..., 0].double() / totals.double()
    vote_gap = (ordered[..., 0] - ordered[..., 1]).double() / totals.double()
    if float(winner_fraction.min()) < MINIMUM_WINNER_FRACTION:
        raise ValueError(
            f"transition winner fraction {float(winner_fraction.min()):.6f} "
            f"is below {MINIMUM_WINNER_FRACTION}"
        )
    if float(vote_gap.min()) < MINIMUM_VOTE_GAP:
        raise ValueError(
            f"transition vote gap {float(vote_gap.min()):.6f} "
            f"is below {MINIMUM_VOTE_GAP}"
        )
    next_states = votes.argmax(dim=-1).cpu().numpy()
    evidence = TransitionEvidence(state_count, token_count)
    evidence.next_states[:] = next_states
    evidence.counts[:] = totals.cpu().numpy()
    recovered = evidence.recover(base_state=0)

    centroids = sums / counts[:, None, None]
    # Recovered element order is a gauge over anonymous decoder classes.
    ordered_centroids = centroids[
        torch.as_tensor(recovered.element_to_state, device=device)
    ]
    orbit = ordered_centroids.permute(1, 2, 0).cpu().numpy()
    return BlindSection(
        orbit=orbit,
        recovered=recovered,
        class_counts=counts.cpu().numpy(),
        transition_votes=votes.cpu().numpy(),
        winner_fractions=winner_fraction.cpu().numpy(),
        vote_gaps=vote_gap.cpu().numpy(),
    )


def _table_is_q8_posthoc(recovered: RecoveredPermutationGroup) -> bool:
    """Score isomorphism only; this function is never called by compilation."""

    from compare_recurrences import GROUPS

    true_table = GROUPS["q8"].table
    labels = recovered.element_to_state
    base_label = int(labels[0])
    inverse_candidates = np.flatnonzero(true_table[base_label] == 0)
    if len(inverse_candidates) != 1:
        return False
    normalized = true_table[int(inverse_candidates[0]), labels]
    transported = normalized[recovered.group.table]
    expected = true_table[normalized[:, None], normalized[None, :]]
    return bool(np.array_equal(transported, expected))


def _negative_controls(next_states: np.ndarray, *, seed: int) -> dict[str, object]:
    collapsed = next_states.copy()
    collapsed[collapsed == collapsed.shape[0] - 1] = collapsed.shape[0] - 2
    collapsed_permutations = all(
        len(np.unique(collapsed[:, token])) == collapsed.shape[0]
        for token in range(collapsed.shape[1])
    )
    rng = np.random.default_rng(seed)
    scrambled = next_states.copy()
    for source in range(scrambled.shape[0]):
        for token in range(scrambled.shape[1]):
            scrambled[source, token] = int(rng.integers(scrambled.shape[0]))
    scrambled_evidence = TransitionEvidence(*scrambled.shape)
    scrambled_evidence.next_states[:] = scrambled
    scrambled_evidence.counts[:] = 1
    try:
        scrambled_recovered = scrambled_evidence.recover(base_state=0)
        scrambled_outcome = {
            "regular_closure": True,
            "same_transition_action": bool(np.array_equal(scrambled, next_states)),
            "recovered_order": scrambled_recovered.group.order,
        }
    except ValueError as error:
        scrambled_outcome = {"regular_closure": False, "reason": str(error)}
    return {
        "collapsed_label_permutation_gate": collapsed_permutations,
        "collapsed_label_rejected": not collapsed_permutations,
        "independent_successor_scramble": scrambled_outcome,
    }


def compile_checkpoint_blind(
    source: Path,
    destination: Path,
    *,
    device: torch.device,
    state_count: int = 8,
    token_count: int = 4,
) -> tuple[dict[str, object], RecoveredPermutationGroup]:
    checkpoint = torch.load(source, map_location=device, weights_only=False)
    if checkpoint["family"] != "pure_spin8_positive":
        raise ValueError("table-blind compiler requires a positive-chiral checkpoint")
    config = checkpoint["config"]
    model = PureGroupActionModel(
        token_count,
        state_count,
        family=checkpoint["family"],
        channels=int(config["channels"]),
        max_rotor_angle=float(config["max_angle"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    seed = int(config["seed"])
    section = discover_blind_section(
        model,
        state_count=state_count,
        token_count=token_count,
        seed_base=11_500_000 + 10_000 * seed,
        device=device,
    )
    recovered = section.recovered
    old_weight = model.output_head.weight.detach().cpu().double().numpy()
    old_flat = section.orbit.reshape(-1, state_count)
    desired_logits = old_weight @ old_flat
    targets, conjugations, projection_rms, gram_eigenvalues, commutant = (
        regular_orbit_projection(section.orbit, recovered.group)
    )
    target_actions = regular_ambient_actions(
        conjugations, recovered.group, recovered.input_elements
    )
    parameters, imaginary_max, tangent_projection_max = positive_spin8_parameters(
        target_actions
    )
    initial = targets[:, :, 0]
    initial /= np.linalg.norm(initial, axis=1, keepdims=True).clip(min=1e-12)
    with torch.no_grad():
        model.action_parameters.copy_(
            torch.as_tensor(parameters, dtype=model.action_parameters.dtype, device=device)
        )
        model.initial_orbit_state.copy_(
            torch.as_tensor(initial, dtype=model.initial_orbit_state.dtype, device=device)
        )

    # Build the exact section without consulting any hidden table.
    # ``targets`` is already ordered in the recovered element gauge.
    exact_orbit = targets
    exact_flat = exact_orbit.reshape(-1, state_count)
    exact_rank = int(np.linalg.matrix_rank(exact_flat, tol=1e-7))
    exact_condition = float(np.linalg.cond(exact_flat))
    transported = minimum_change_observer(old_weight, old_flat, exact_flat)
    with torch.no_grad():
        model.output_head.weight.copy_(
            torch.as_tensor(
                transported, dtype=model.output_head.weight.dtype, device=device
            )
        )
    actual_weight = model.output_head.weight.detach().cpu().double().numpy()
    logit_transport_max = float(
        np.max(np.abs(actual_weight @ exact_flat - desired_logits))
    )
    reconstructed = model.action_matrices().detach().cpu().double().numpy()
    action_reconstruction_max = float(np.max(np.abs(reconstructed - target_actions)))
    diagnostics = representation_diagnostics(
        model, recovered.group, recovered.input_elements
    )
    probe = torch.arange(128, device=device).reshape(4, 32) % token_count
    streaming = streaming_equivalence(model, probe)
    gates = {
        "anonymous_class_coverage": bool(np.all(section.class_counts > 0)),
        "minimum_edge_votes": int(section.transition_votes.sum(-1).min())
        >= MINIMUM_EDGE_VOTES,
        "transition_winner_fraction": float(section.winner_fractions.min())
        >= MINIMUM_WINNER_FRACTION,
        "transition_vote_gap": float(section.vote_gaps.min()) >= MINIMUM_VOTE_GAP,
        "centroid_projection": float(projection_rms.max()) <= 0.03,
        "commutant": float(commutant.max()) <= 1e-10,
        "spin8_action_reconstruction": action_reconstruction_max <= 1e-5,
        "recovered_homomorphism": diagnostics["linear_homomorphism_rms"] <= 1e-5,
        "full_section_rank": exact_rank == state_count,
        "centroid_logit_transport": logit_transport_max <= 1e-5,
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
        "spin8_table_blind_compiler": True,
        "spin8_table_blind_source": str(source),
        "recovered_group_table": recovered.group.table.tolist(),
        "recovered_input_elements": list(recovered.input_elements),
        "recovered_element_to_anonymous_class": recovered.element_to_state.tolist(),
    }
    compiled["state_dict"] = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    torch.save(compiled, destination)
    next_states = section.transition_votes.argmax(axis=-1)
    result = {
        "source": str(source),
        "destination": str(destination),
        "seed": seed,
        "method": "anonymous transition recovery plus shared regular Spin8 retraction",
        "hidden_table_used_by_compiler": False,
        "target_labels_used_by_compiler": False,
        "model_predictions_used_as_anonymous_labels": True,
        "group_aware_calibration_sampler_used": False,
        "gradient_steps_after_compilation": 0,
        "rank_threshold_used": False,
        "independent_token_normalization": False,
        "anonymous_class_counts": section.class_counts.tolist(),
        "transition_votes": section.transition_votes.tolist(),
        "recovered_next_states": next_states.tolist(),
        "transition_winner_fraction_min": float(section.winner_fractions.min()),
        "transition_vote_gap_min": float(section.vote_gaps.min()),
        "recovered_group_table": recovered.group.table.tolist(),
        "recovered_input_elements": list(recovered.input_elements),
        "recovered_element_to_anonymous_class": recovered.element_to_state.tolist(),
        "per_channel_centroid_projection_rms": projection_rms.tolist(),
        "per_channel_projected_gram_eigenvalues": gram_eigenvalues.tolist(),
        "per_channel_commutant_max_abs": commutant.tolist(),
        "exact_section_rank": exact_rank,
        "exact_section_condition_number": exact_condition,
        "centroid_logit_transport_max_abs": logit_transport_max,
        "matrix_log_imaginary_max_abs": imaginary_max,
        "lie_tangent_projection_max_abs": tangent_projection_max,
        "spin8_action_reconstruction_max_abs": action_reconstruction_max,
        "representation_diagnostics_recovered_table": diagnostics,
        "streaming_equivalence": streaming,
        "negative_controls": _negative_controls(next_states, seed=seed + 71_000),
        "compiler_gates": gates,
        "compiler_passed": all(gates.values()),
    }
    return result, recovered


def posthoc_q8_score(
    result: dict[str, object],
    recovered: RecoveredPermutationGroup,
    destination: Path,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Evaluate hidden Q8 only after the blind checkpoint has been persisted."""

    from q8_spinor_center_experiment import central_pair_evaluation
    from q8_spinor_center_long_audit import LONG_BASE_LENGTHS
    from q8_spinor_joint_retraction import SMOKE_BASE_LENGTHS

    checkpoint = torch.load(destination, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = PureGroupActionModel(
        4,
        8,
        family=checkpoint["family"],
        channels=int(config["channels"]),
        max_rotor_angle=float(config["max_angle"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    seed = int(config["seed"])
    dense = central_pair_evaluation(
        model,
        base_lengths=SMOKE_BASE_LENGTHS,
        batches=2,
        batch_size=512,
        seed_base=11_700_000 + 10_000 * seed,
        device=device,
    )
    long = central_pair_evaluation(
        model,
        base_lengths=LONG_BASE_LENGTHS,
        batches=1,
        batch_size=128,
        seed_base=11_800_000 + 10_000 * seed,
        device=device,
    )
    q8_isomorphic = _table_is_q8_posthoc(recovered)
    result = dict(result)
    result["posthoc_q8_isomorphic"] = q8_isomorphic
    result["posthoc_dense_central_pair_evaluation"] = dense
    result["posthoc_long_central_pair_evaluation"] = long
    result["posthoc_gates"] = {
        "q8_isomorphic": q8_isomorphic,
        "dense_central_pair": bool(dense["gate_pass"]),
        "long_central_pair": bool(long["gate_pass"]),
    }
    result["passed"] = bool(
        result["compiler_passed"] and all(result["posthoc_gates"].values())
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    torch.use_deterministic_algorithms(True)
    result, recovered = compile_checkpoint_blind(
        args.source, args.destination, device=device
    )
    result = posthoc_q8_score(result, recovered, args.destination, device=device)
    report = {
        "experiment": "Spin(8) table-blind decoder-labeled compiler",
        "result": result,
        "passed": result["passed"],
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
