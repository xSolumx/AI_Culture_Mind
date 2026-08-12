"""Recover and compile a finite Spin(8) action from recurrent states alone."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
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
from spin8_table_blind_compiler import _negative_controls, posthoc_q8_score


CALIBRATION_LENGTHS = (15, 16)
CALIBRATION_BATCHES = 32
CALIBRATION_BATCH_SIZE = 512
KMEANS_RESTARTS = 8
KMEANS_MAX_ITERATIONS = 100
KMEANS_TOLERANCE = 1e-10
MINIMUM_CLUSTER_COUNT = 256
MINIMUM_SEPARATION_RATIO = 2.0
MINIMUM_EDGE_VOTES = 256
MINIMUM_WINNER_FRACTION = 0.99
MINIMUM_VOTE_GAP = 0.98


@dataclass(frozen=True)
class KMeansResult:
    centers: np.ndarray
    labels: np.ndarray
    inertia: float
    iterations: int
    restart: int
    runner_up_inertia: float


@dataclass(frozen=True)
class StateOnlySection:
    centroids: np.ndarray
    recovered: RecoveredPermutationGroup
    class_counts: np.ndarray
    transition_votes: np.ndarray
    winner_fractions: np.ndarray
    vote_gaps: np.ndarray
    within_cluster_rms: float
    minimum_centroid_separation: float
    separation_ratio: float
    identity_cluster: int
    origin_winner_fraction: float
    origin_vote_gap: float
    initial_nearest_cluster: int
    identity_distance: float
    initial_nearest_distance: float
    kmeans: KMeansResult


def squared_distances(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = (
        np.sum(np.square(points), axis=1, keepdims=True)
        + np.sum(np.square(centers), axis=1)[None, :]
        - 2.0 * points @ centers.T
    )
    return np.maximum(distances, 0.0)


def kmeans_plus_plus(
    points: np.ndarray, clusters: int, *, rng: np.random.Generator
) -> np.ndarray:
    centers = [points[int(rng.integers(len(points)))]]
    minimum = squared_distances(points, np.asarray(centers))[:, 0]
    for _ in range(1, clusters):
        total = float(minimum.sum())
        if total <= 0.0:
            raise ValueError("k-means++ found fewer distinct points than clusters")
        centers.append(points[int(rng.choice(len(points), p=minimum / total))])
        candidate = squared_distances(points, np.asarray(centers[-1:]))[:, 0]
        minimum = np.minimum(minimum, candidate)
    return np.asarray(centers, dtype=np.float64)


def deterministic_kmeans(
    points: np.ndarray,
    clusters: int,
    *,
    seed: int,
    restarts: int = KMEANS_RESTARTS,
    max_iterations: int = KMEANS_MAX_ITERATIONS,
    tolerance: float = KMEANS_TOLERANCE,
) -> KMeansResult:
    """Deterministic multi-restart Euclidean k-means with k-means++ seeds."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or not 1 <= clusters <= len(values):
        raise ValueError("points must be a nonempty matrix with a valid k")
    outcomes = []
    for restart in range(restarts):
        rng = np.random.default_rng(seed + 104_729 * restart)
        centers = kmeans_plus_plus(values, clusters, rng=rng)
        previous_inertia = np.inf
        valid = True
        for iteration in range(1, max_iterations + 1):
            distances = squared_distances(values, centers)
            labels = distances.argmin(axis=1)
            counts = np.bincount(labels, minlength=clusters)
            if np.any(counts == 0):
                valid = False
                break
            revised = np.stack(
                [values[labels == cluster].mean(axis=0) for cluster in range(clusters)]
            )
            inertia = float(
                np.sum(np.square(values - revised[labels]), dtype=np.float64)
            )
            centers = revised
            if abs(previous_inertia - inertia) <= tolerance * max(1.0, inertia):
                break
            previous_inertia = inertia
        if not valid:
            continue
        outcomes.append((inertia, restart, iteration, centers, labels))
    if not outcomes:
        raise ValueError("all k-means restarts produced an empty cluster")
    outcomes.sort(key=lambda item: (item[0], item[1]))
    inertia, restart, iterations, centers, labels = outcomes[0]
    runner_up = outcomes[1][0] if len(outcomes) > 1 else np.inf
    return KMeansResult(
        centers=centers,
        labels=labels,
        inertia=inertia,
        iterations=iterations,
        restart=restart,
        runner_up_inertia=float(runner_up),
    )


@torch.no_grad()
def collect_state_paths(
    model: PureGroupActionModel,
    *,
    seed_base: int,
    token_count: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Collect states and one-token successors without touching the decoder."""

    states_out = []
    successors_out = []
    token_batches = []
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
            token_batches.append(tokens.cpu().numpy())
            state = model.initial_state(tokens.shape[0])
            for position in range(length):
                state = torch.einsum(
                    "bcij,bcj->bci", actions[tokens[:, position]], state
                )
            successors = []
            for token in range(token_count):
                selected = actions[token].expand(state.shape[0], -1, -1, -1)
                successors.append(
                    torch.einsum("bcij,bcj->bci", selected, state)
                )
            states_out.append(state.flatten(1).cpu().double().numpy())
            successors_out.append(
                torch.stack(successors, dim=1).flatten(2).cpu().double().numpy()
            )
    return (
        np.concatenate(states_out),
        np.concatenate(successors_out),
        token_batches,
    )


def discover_state_only_section(
    model: PureGroupActionModel,
    *,
    state_count: int,
    token_count: int,
    seed_base: int,
    device: torch.device,
    minimum_separation_ratio: float | None = MINIMUM_SEPARATION_RATIO,
) -> StateOnlySection:
    points, successors, token_batches = collect_state_paths(
        model, seed_base=seed_base, token_count=token_count, device=device
    )
    clustered = deterministic_kmeans(
        points, state_count, seed=seed_base + 8_000_003
    )
    labels = clustered.labels
    counts = np.bincount(labels, minlength=state_count)
    if int(counts.min()) < MINIMUM_CLUSTER_COUNT:
        raise ValueError(
            f"minimum cluster count {int(counts.min())} is below "
            f"{MINIMUM_CLUSTER_COUNT}"
        )
    pairwise = np.sqrt(squared_distances(clustered.centers, clustered.centers))
    np.fill_diagonal(pairwise, np.inf)
    minimum_separation = float(pairwise.min())
    within_rms = float(np.sqrt(clustered.inertia / len(points)))
    ratio = minimum_separation / max(within_rms, 1e-15)
    if minimum_separation_ratio is not None and ratio < minimum_separation_ratio:
        raise ValueError(
            f"centroid separation ratio {ratio:.6f} is below "
            f"{minimum_separation_ratio}"
        )

    votes = np.zeros((state_count, token_count, state_count), dtype=np.int64)
    for token in range(token_count):
        successor_labels = squared_distances(
            successors[:, token], clustered.centers
        ).argmin(axis=1)
        np.add.at(votes[:, token], (labels, successor_labels), 1)
    ordered = np.sort(votes, axis=-1)[..., ::-1]
    totals = votes.sum(axis=-1)
    if int(totals.min()) < MINIMUM_EDGE_VOTES:
        raise ValueError("one or more state-only edges lack vote support")
    winner_fraction = ordered[..., 0] / totals
    vote_gap = (ordered[..., 0] - ordered[..., 1]) / totals
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
    next_states = votes.argmax(axis=-1)
    origin_scores = np.zeros(state_count, dtype=np.int64)
    offset = 0
    for token_batch in token_batches:
        batch_labels = labels[offset: offset + len(token_batch)]
        offset += len(token_batch)
        for base in range(state_count):
            predicted = np.full(len(token_batch), base, dtype=np.int64)
            for position in range(token_batch.shape[1]):
                predicted = next_states[predicted, token_batch[:, position]]
            origin_scores[base] += int(np.sum(predicted == batch_labels))
    if offset != len(labels):
        raise RuntimeError("calibration token/state alignment was lost")
    origin_order = np.argsort(origin_scores)[::-1]
    origin_fractions = origin_scores / len(labels)
    origin_winner_fraction = float(origin_fractions[origin_order[0]])
    origin_vote_gap = float(
        origin_fractions[origin_order[0]] - origin_fractions[origin_order[1]]
    )
    if origin_winner_fraction < MINIMUM_WINNER_FRACTION:
        raise ValueError(
            f"torsor-origin winner fraction {origin_winner_fraction:.6f} is "
            f"below {MINIMUM_WINNER_FRACTION}"
        )
    if origin_vote_gap < MINIMUM_VOTE_GAP:
        raise ValueError(
            f"torsor-origin vote gap {origin_vote_gap:.6f} is below "
            f"{MINIMUM_VOTE_GAP}"
        )
    identity_cluster = int(origin_order[0])
    initial = model.initial_state(1).flatten(1).detach().cpu().double().numpy()
    identity_distances = np.sqrt(squared_distances(initial, clustered.centers)[0])
    identity_order = np.argsort(identity_distances)
    evidence = TransitionEvidence(state_count, token_count)
    evidence.next_states[:] = next_states
    evidence.counts[:] = totals
    recovered = evidence.recover(base_state=identity_cluster)
    centroids = clustered.centers.reshape(state_count, model.channels, 8)
    return StateOnlySection(
        centroids=centroids,
        recovered=recovered,
        class_counts=counts,
        transition_votes=votes,
        winner_fractions=winner_fraction,
        vote_gaps=vote_gap,
        within_cluster_rms=within_rms,
        minimum_centroid_separation=minimum_separation,
        separation_ratio=ratio,
        identity_cluster=identity_cluster,
        origin_winner_fraction=origin_winner_fraction,
        origin_vote_gap=origin_vote_gap,
        initial_nearest_cluster=int(identity_order[0]),
        identity_distance=float(identity_distances[identity_cluster]),
        initial_nearest_distance=float(identity_distances[identity_order[0]]),
        kmeans=clustered,
    )


def align_audit_section(
    primary: StateOnlySection, audit: StateOnlySection
) -> dict[str, object]:
    first = primary.centroids.reshape(len(primary.centroids), -1)
    second = audit.centroids.reshape(len(audit.centroids), -1)
    costs = np.sqrt(squared_distances(second, first))
    audit_labels, primary_labels = linear_sum_assignment(costs)
    audit_to_primary = np.empty(len(first), dtype=np.int64)
    audit_to_primary[audit_labels] = primary_labels
    primary_to_audit = np.empty(len(first), dtype=np.int64)
    primary_to_audit[primary_labels] = audit_labels
    primary_next = primary.transition_votes.argmax(axis=-1)
    audit_next = audit.transition_votes.argmax(axis=-1)
    aligned = np.empty_like(primary_next)
    for source in range(len(first)):
        audit_source = primary_to_audit[source]
        aligned[source] = audit_to_primary[audit_next[audit_source]]
    agreement = float(np.mean(aligned == primary_next))
    aligned_audit_identity = int(audit_to_primary[audit.identity_cluster])
    return {
        "audit_to_primary_cluster": audit_to_primary.tolist(),
        "transition_agreement": agreement,
        "all_transitions_agree": bool(np.array_equal(aligned, primary_next)),
        "aligned_audit_identity_cluster": aligned_audit_identity,
        "identity_clusters_agree": aligned_audit_identity == primary.identity_cluster,
        "maximum_matched_centroid_distance": float(
            costs[audit_labels, primary_labels].max()
        ),
        "mean_matched_centroid_distance": float(
            costs[audit_labels, primary_labels].mean()
        ),
        "aligned_audit_next_states": aligned.tolist(),
    }


def abstract_group_isomorphic(left: np.ndarray, right: np.ndarray) -> bool:
    """Exhaustively test identity-preserving isomorphism for small tables."""

    if left.shape != right.shape or left.ndim != 2 or left.shape[0] != left.shape[1]:
        return False
    order = left.shape[0]
    for tail in itertools.permutations(range(1, order)):
        mapping = np.asarray((0,) + tail, dtype=np.int64)
        if np.array_equal(
            mapping[left], right[mapping[:, None], mapping[None, :]]
        ):
            return True
    return False


def _q8_isomorphic_posthoc(recovered: RecoveredPermutationGroup) -> bool:
    from compare_recurrences import GROUPS

    return abstract_group_isomorphic(recovered.group.table, GROUPS["q8"].table)


def compile_state_only_checkpoint(
    source: Path,
    destination: Path,
    *,
    device: torch.device,
    state_count: int = 8,
    minimum_separation_ratio: float | None = MINIMUM_SEPARATION_RATIO,
    method: str = "state-only replicated clustering plus shared regular Spin8 retraction",
    state_cardinality_supplied: bool = True,
    compiler_config_key: str = "spin8_state_only_compiler",
    extra_result: dict[str, object] | None = None,
    extra_gates: dict[str, bool] | None = None,
) -> tuple[dict[str, object], RecoveredPermutationGroup]:
    checkpoint = torch.load(source, map_location=device, weights_only=False)
    if checkpoint["family"] != "pure_spin8_positive":
        raise ValueError("state-only compiler requires a positive-chiral checkpoint")
    config = checkpoint["config"]
    model = PureGroupActionModel(
        4, 8, family=checkpoint["family"], channels=int(config["channels"]),
        max_rotor_angle=float(config["max_angle"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    seed = int(config["seed"])
    primary = discover_state_only_section(
        model, state_count=state_count, token_count=4,
        seed_base=13_500_000 + 20_000 * seed, device=device,
        minimum_separation_ratio=minimum_separation_ratio,
    )
    audit = discover_state_only_section(
        model, state_count=state_count, token_count=4,
        seed_base=13_510_000 + 20_000 * seed, device=device,
        minimum_separation_ratio=minimum_separation_ratio,
    )
    replication = align_audit_section(primary, audit)
    if not replication["all_transitions_agree"]:
        raise ValueError(
            f"independent state-only transition agreement is "
            f"{replication['transition_agreement']:.6f}"
        )
    if not replication["identity_clusters_agree"]:
        raise ValueError("independent state-only identity clusters disagree")
    recovered = primary.recovered
    ordered = primary.centroids[recovered.element_to_state]
    orbit = ordered.transpose(1, 2, 0)

    # Discovery is complete. The observer first becomes accessible here.
    old_weight = model.output_head.weight.detach().cpu().double().numpy()
    old_flat = orbit.reshape(-1, 8)
    desired_logits = old_weight @ old_flat
    targets, conjugations, projection_rms, gram_eigenvalues, commutant = (
        regular_orbit_projection(orbit, recovered.group)
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
        model.action_parameters.copy_(torch.as_tensor(
            parameters, dtype=model.action_parameters.dtype, device=device
        ))
        model.initial_orbit_state.copy_(torch.as_tensor(
            initial, dtype=model.initial_orbit_state.dtype, device=device
        ))
    exact_flat = targets.reshape(-1, 8)
    exact_rank = int(np.linalg.matrix_rank(exact_flat, tol=1e-7))
    transported = minimum_change_observer(old_weight, old_flat, exact_flat)
    with torch.no_grad():
        model.output_head.weight.copy_(torch.as_tensor(
            transported, dtype=model.output_head.weight.dtype, device=device
        ))
    actual_weight = model.output_head.weight.detach().cpu().double().numpy()
    logit_transport = float(
        np.max(np.abs(actual_weight @ exact_flat - desired_logits))
    )
    reconstructed = model.action_matrices().detach().cpu().double().numpy()
    action_reconstruction = float(np.max(np.abs(reconstructed - target_actions)))
    diagnostics = representation_diagnostics(
        model, recovered.group, recovered.input_elements
    )
    streaming = streaming_equivalence(
        model, torch.arange(128, device=device).reshape(4, 32) % 4
    )
    gates = {
        "primary_cluster_count": int(primary.class_counts.min()) >= MINIMUM_CLUSTER_COUNT,
        "primary_transition_winner": float(primary.winner_fractions.min()) >= MINIMUM_WINNER_FRACTION,
        "primary_transition_gap": float(primary.vote_gaps.min()) >= MINIMUM_VOTE_GAP,
        "primary_origin_winner": primary.origin_winner_fraction >= MINIMUM_WINNER_FRACTION,
        "primary_origin_gap": primary.origin_vote_gap >= MINIMUM_VOTE_GAP,
        "audit_cluster_count": int(audit.class_counts.min()) >= MINIMUM_CLUSTER_COUNT,
        "audit_transition_winner": float(audit.winner_fractions.min()) >= MINIMUM_WINNER_FRACTION,
        "audit_transition_gap": float(audit.vote_gaps.min()) >= MINIMUM_VOTE_GAP,
        "audit_origin_winner": audit.origin_winner_fraction >= MINIMUM_WINNER_FRACTION,
        "audit_origin_gap": audit.origin_vote_gap >= MINIMUM_VOTE_GAP,
        "independent_transition_replication": bool(replication["all_transitions_agree"]),
        "independent_identity_replication": bool(replication["identity_clusters_agree"]),
        "centroid_projection": float(projection_rms.max()) <= 0.03,
        "commutant": float(commutant.max()) <= 1e-10,
        "spin8_action_reconstruction": action_reconstruction <= 1e-5,
        "recovered_homomorphism": diagnostics["linear_homomorphism_rms"] <= 1e-5,
        "full_section_rank": exact_rank == 8,
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
    if minimum_separation_ratio is not None:
        gates.update({
            "primary_separation": (
                primary.separation_ratio >= minimum_separation_ratio
            ),
            "audit_separation": audit.separation_ratio >= minimum_separation_ratio,
        })
    if extra_gates:
        gates.update(extra_gates)
    destination.parent.mkdir(parents=True, exist_ok=True)
    compiled = dict(checkpoint)
    compiled["config"] = {
        **config, compiler_config_key: True,
        "recovered_group_table": recovered.group.table.tolist(),
        "recovered_input_elements": list(recovered.input_elements),
    }
    compiled["state_dict"] = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    torch.save(compiled, destination)
    next_states = primary.transition_votes.argmax(axis=-1)
    result = {
        "source": str(source), "destination": str(destination), "seed": seed,
        "method": method,
        "decoder_access_during_discovery": False,
        "hidden_table_used_by_compiler": False,
        "target_labels_used_by_compiler": False,
        "state_cardinality_supplied": state_count if state_cardinality_supplied else None,
        "state_cardinality_selected": state_count,
        "gradient_steps_after_compilation": 0,
        "primary": {
            "cluster_counts": primary.class_counts.tolist(),
            "within_cluster_rms": primary.within_cluster_rms,
            "minimum_centroid_separation": primary.minimum_centroid_separation,
            "separation_ratio": primary.separation_ratio,
            "identity_cluster": primary.identity_cluster,
            "origin_winner_fraction": primary.origin_winner_fraction,
            "origin_vote_gap": primary.origin_vote_gap,
            "initial_nearest_cluster": primary.initial_nearest_cluster,
            "identity_distance": primary.identity_distance,
            "initial_nearest_distance": primary.initial_nearest_distance,
            "kmeans_inertia": primary.kmeans.inertia,
            "kmeans_runner_up_inertia": primary.kmeans.runner_up_inertia,
            "kmeans_selected_restart": primary.kmeans.restart,
            "kmeans_iterations": primary.kmeans.iterations,
            "transition_winner_fraction_min": float(primary.winner_fractions.min()),
            "transition_vote_gap_min": float(primary.vote_gaps.min()),
            "transition_votes": primary.transition_votes.tolist(),
            "recovered_next_states": next_states.tolist(),
        },
        "audit": {
            "cluster_counts": audit.class_counts.tolist(),
            "within_cluster_rms": audit.within_cluster_rms,
            "minimum_centroid_separation": audit.minimum_centroid_separation,
            "separation_ratio": audit.separation_ratio,
            "identity_cluster": audit.identity_cluster,
            "origin_winner_fraction": audit.origin_winner_fraction,
            "origin_vote_gap": audit.origin_vote_gap,
            "initial_nearest_cluster": audit.initial_nearest_cluster,
            "identity_distance": audit.identity_distance,
            "initial_nearest_distance": audit.initial_nearest_distance,
            "transition_winner_fraction_min": float(audit.winner_fractions.min()),
            "transition_vote_gap_min": float(audit.vote_gaps.min()),
        },
        "independent_replication": replication,
        "recovered_group_table": recovered.group.table.tolist(),
        "recovered_input_elements": list(recovered.input_elements),
        "per_channel_centroid_projection_rms": projection_rms.tolist(),
        "per_channel_projected_gram_eigenvalues": gram_eigenvalues.tolist(),
        "per_channel_commutant_max_abs": commutant.tolist(),
        "exact_section_rank": exact_rank,
        "exact_section_condition_number": float(np.linalg.cond(exact_flat)),
        "centroid_logit_transport_max_abs": logit_transport,
        "matrix_log_imaginary_max_abs": imaginary_max,
        "lie_tangent_projection_max_abs": tangent_projection_max,
        "spin8_action_reconstruction_max_abs": action_reconstruction,
        "representation_diagnostics_recovered_table": diagnostics,
        "streaming_equivalence": streaming,
        "negative_controls": _negative_controls(next_states, seed=seed + 73_000),
        "compiler_gates": gates,
        "compiler_passed": all(gates.values()),
    }
    if extra_result:
        result.update(extra_result)
    return result, recovered


def posthoc_state_only_score(
    result: dict[str, object], recovered: RecoveredPermutationGroup,
    destination: Path, *, device: torch.device
) -> dict[str, object]:
    scored = posthoc_q8_score(result, recovered, destination, device=device)
    # Replace the decoder-label-dependent isomorphism scorer with abstract table
    # isomorphism. The behavioral evaluations from ``posthoc_q8_score`` remain.
    abstract = _q8_isomorphic_posthoc(recovered)
    scored["posthoc_q8_isomorphic"] = abstract
    scored["posthoc_gates"]["q8_isomorphic"] = abstract
    scored["passed"] = bool(
        scored["compiler_passed"] and all(scored["posthoc_gates"].values())
    )
    return scored


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
    result, recovered = compile_state_only_checkpoint(
        args.source, args.destination, device=device
    )
    result = posthoc_state_only_score(
        result, recovered, args.destination, device=device
    )
    report = {"experiment": "Spin8 state-only finite-action compiler",
              "result": result, "passed": result["passed"]}
    rendered = json.dumps(report, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
