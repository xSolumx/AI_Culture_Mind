"""Audit which state cardinalities yield replicated regular token actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from latent_group_discovery import TransitionEvidence
from mechanistic_group_actions import PureGroupActionModel
from spin8_state_only_compiler import (
    KMEANS_RESTARTS,
    collect_state_paths,
    deterministic_kmeans,
    squared_distances,
)


def candidate_action(
    points: np.ndarray,
    successors: np.ndarray,
    token_batches: list[np.ndarray],
    *,
    clusters: int,
    seed: int,
) -> tuple[dict[str, object], np.ndarray | None, np.ndarray | None, int | None]:
    try:
        fit = deterministic_kmeans(
            points, clusters, seed=seed, restarts=KMEANS_RESTARTS
        )
    except ValueError as error:
        return {"k": clusters, "kmeans_failure": str(error), "viable": False}, None, None, None
    labels = fit.labels
    counts = np.bincount(labels, minlength=clusters)
    pairwise = np.sqrt(squared_distances(fit.centers, fit.centers))
    np.fill_diagonal(pairwise, np.inf)
    separation = float(pairwise.min())
    within = float(np.sqrt(fit.inertia / len(points)))
    votes = np.zeros((clusters, successors.shape[1], clusters), dtype=np.int64)
    for token in range(successors.shape[1]):
        following = squared_distances(
            successors[:, token], fit.centers
        ).argmin(axis=1)
        np.add.at(votes[:, token], (labels, following), 1)
    ordered = np.sort(votes, axis=-1)[..., ::-1]
    totals = votes.sum(axis=-1)
    winner = float(np.min(ordered[..., 0] / np.maximum(totals, 1)))
    gap = float(np.min(
        (ordered[..., 0] - ordered[..., 1]) / np.maximum(totals, 1)
    ))
    next_states = votes.argmax(axis=-1)
    expected = np.arange(clusters)
    permutations = bool(all(
        np.array_equal(np.sort(next_states[:, token]), expected)
        for token in range(next_states.shape[1])
    ))

    origin_scores = np.zeros(clusters, dtype=np.int64)
    offset = 0
    for token_batch in token_batches:
        batch_labels = labels[offset: offset + len(token_batch)]
        offset += len(token_batch)
        for base in range(clusters):
            predicted = np.full(len(token_batch), base, dtype=np.int64)
            for position in range(token_batch.shape[1]):
                predicted = next_states[predicted, token_batch[:, position]]
            origin_scores[base] += int(np.sum(predicted == batch_labels))
    origin_order = np.argsort(origin_scores)[::-1]
    origin_fractions = origin_scores / len(labels)
    origin = int(origin_order[0])
    origin_winner = float(origin_fractions[origin_order[0]])
    origin_gap = float(
        origin_fractions[origin_order[0]] - origin_fractions[origin_order[1]]
    ) if clusters > 1 else origin_winner
    regular = False
    recovery_error = None
    if permutations:
        evidence = TransitionEvidence(clusters, next_states.shape[1])
        evidence.next_states[:] = next_states
        evidence.counts[:] = totals
        try:
            recovered = evidence.recover(base_state=origin)
            regular = recovered.group.order == clusters
        except ValueError as error:
            recovery_error = str(error)
    local = bool(
        winner >= 0.99 and gap >= 0.98 and permutations and regular
        and origin_winner >= 0.99 and origin_gap >= 0.98
    )
    return {
        "k": clusters,
        "minimum_cluster_count": int(counts.min()),
        "within_cluster_rms": within,
        "minimum_centroid_separation": separation,
        "separation_ratio": separation / max(within, 1e-15),
        "transition_winner_fraction_min": winner,
        "transition_vote_gap_min": gap,
        "token_actions_are_permutations": permutations,
        "regular_group_closure": regular,
        "regular_recovery_error": recovery_error,
        "origin_cluster": origin,
        "origin_winner_fraction": origin_winner,
        "origin_vote_gap": origin_gap,
        "local_viability": local,
        "kmeans_inertia": fit.inertia,
        "kmeans_runner_up_inertia": fit.runner_up_inertia,
        "kmeans_selected_restart": fit.restart,
        "viable": False,
    }, fit.centers, next_states, origin


def replicated_candidate(
    primary, audit, *, clusters: int
) -> dict[str, object]:
    first, first_centers, first_next, first_origin = primary
    second, second_centers, second_next, second_origin = audit
    result = {"k": clusters, "primary": first, "audit": second, "viable": False}
    if first_centers is None or second_centers is None:
        return result
    costs = np.sqrt(squared_distances(second_centers, first_centers))
    audit_labels, primary_labels = linear_sum_assignment(costs)
    audit_to_primary = np.empty(clusters, dtype=np.int64)
    audit_to_primary[audit_labels] = primary_labels
    primary_to_audit = np.empty(clusters, dtype=np.int64)
    primary_to_audit[primary_labels] = audit_labels
    aligned = np.empty_like(first_next)
    for source in range(clusters):
        aligned[source] = audit_to_primary[second_next[primary_to_audit[source]]]
    transition_agreement = float(np.mean(aligned == first_next))
    origin_agreement = int(audit_to_primary[second_origin]) == first_origin
    viable = bool(
        first["local_viability"] and second["local_viability"]
        and transition_agreement == 1.0 and origin_agreement
    )
    result.update({
        "transition_agreement": transition_agreement,
        "origin_agreement": origin_agreement,
        "maximum_matched_centroid_distance": float(
            costs[audit_labels, primary_labels].max()
        ),
        "viable": viable,
    })
    return result


def audit_checkpoint(path: Path, *, device: torch.device) -> dict[str, object]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = PureGroupActionModel(
        4, 8, family=checkpoint["family"], channels=int(config["channels"]),
        max_rotor_angle=float(config["max_angle"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    seed = int(config["seed"])
    primary_data = collect_state_paths(
        model, seed_base=13_500_000 + 20_000 * seed,
        token_count=4, device=device,
    )
    audit_data = collect_state_paths(
        model, seed_base=13_510_000 + 20_000 * seed,
        token_count=4, device=device,
    )
    candidates = []
    for clusters in range(2, 13):
        primary = candidate_action(
            *primary_data, clusters=clusters,
            seed=13_500_000 + 20_000 * seed + 8_000_003,
        )
        audit = candidate_action(
            *audit_data, clusters=clusters,
            seed=13_510_000 + 20_000 * seed + 8_000_003,
        )
        candidate = replicated_candidate(primary, audit, clusters=clusters)
        candidates.append(candidate)
        print(
            f"seed={seed} k={clusters} viable={candidate['viable']} "
            f"winner={candidate['primary'].get('transition_winner_fraction_min')}",
            flush=True,
        )
    viable = [candidate["k"] for candidate in candidates if candidate["viable"]]
    return {
        "seed": seed, "source": str(path), "viable_cardinalities": viable,
        "unique_k8": viable == [8], "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    torch.use_deterministic_algorithms(True)
    results = [audit_checkpoint(path, device=device) for path in args.sources]
    report = {
        "experiment": "Spin8 state-only cardinality and closure audit",
        "frozen_state_only_gate_unchanged": True,
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
