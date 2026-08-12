"""Classify learned Cl(3) channels against both A5 irreps and audit defects."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import torch

from changed_generator_transfer import strongest_commutator_channel
from compare_recurrences import GROUPS
from mechanistic_group_actions import (
    PureGroupActionModel,
    _element_inverses,
    _element_orders,
    a5_orthogonal_irrep,
)


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(matrix)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return rotation


def align_representation(
    learned: np.ndarray, oracle: np.ndarray, seed: int
) -> tuple[np.ndarray, float]:
    """Find one global SO(3) conjugacy aligning all token actions."""

    identity = np.eye(3)
    intertwiner = np.concatenate(
        [np.kron(identity, left) - np.kron(right.T, identity)
         for left, right in zip(learned, oracle)],
        axis=0,
    )
    _, _, right_singular = np.linalg.svd(intertwiner)
    linear_guess = right_singular[-1].reshape(3, 3, order="F")
    guesses = [nearest_rotation(linear_guess), identity]
    generator = np.random.default_rng(seed)
    guesses.extend(Rotation.random(8, random_state=generator).as_matrix())

    def objective(rotvec: np.ndarray) -> float:
        change = Rotation.from_rotvec(rotvec).as_matrix()
        residual = learned - change[None] @ oracle @ change.T[None]
        return float(np.mean(np.square(residual)))

    best_matrix = identity
    best_value = math.inf
    for guess in guesses:
        result = minimize(
            objective,
            Rotation.from_matrix(guess).as_rotvec(),
            method="BFGS",
            options={"maxiter": 1000, "gtol": 1e-12},
        )
        if result.fun < best_value:
            best_value = float(result.fun)
            best_matrix = Rotation.from_rotvec(result.x).as_matrix()
    return best_matrix, math.sqrt(best_value)


def channel_commutator(actions: np.ndarray) -> float:
    values = []
    for left in range(len(actions)):
        for right in range(left + 1, len(actions)):
            difference = actions[right] @ actions[left] - actions[left] @ actions[right]
            values.append(np.linalg.norm(difference) / math.sqrt(3.0))
    return max(values)


def channel_relator_mean(
    actions: np.ndarray,
    group_table: np.ndarray,
    input_elements: tuple[int, ...],
    orders: np.ndarray,
) -> float:
    residuals = []
    identity = np.eye(3)
    for token, element in enumerate(input_elements):
        residuals.append(
            np.linalg.norm(np.linalg.matrix_power(actions[token], int(orders[element])) - identity)
            / math.sqrt(3.0)
        )
    for left in range(len(input_elements)):
        for right in range(left + 1, len(input_elements)):
            product = int(group_table[input_elements[left], input_elements[right]])
            product_action = actions[right] @ actions[left]
            residuals.append(
                np.linalg.norm(np.linalg.matrix_power(product_action, int(orders[product])) - identity)
                / math.sqrt(3.0)
            )
    return float(np.mean(residuals))


def rational_angle_diagnostic(angle: float) -> dict[str, float | int]:
    turns = angle / (2.0 * math.pi)
    approximation = Fraction(float(turns)).limit_denominator(1000)
    return {
        "angle_radians": angle,
        "turns": turns,
        "nearest_numerator_denominator_at_most_1000": approximation.numerator,
        "nearest_denominator_at_most_1000": approximation.denominator,
        "turns_absolute_error": abs(turns - float(approximation)),
    }


def defect_lie_audit(
    learned: np.ndarray, oracle: np.ndarray, alignment: np.ndarray
) -> dict[str, object]:
    aligned = alignment[None] @ oracle @ alignment.T[None]
    defects = np.stack(
        [nearest_rotation(left) @ nearest_rotation(right).T
         for left, right in zip(learned, aligned)]
    )
    logs = Rotation.from_matrix(defects).as_rotvec()
    defect_commutator = defects[2] @ defects[0] - defects[0] @ defects[2]
    first, second = logs[0], logs[2]
    bracket = np.cross(first, second)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    axis_dot = (
        float(np.dot(first, second) / (first_norm * second_norm))
        if first_norm > 1e-12 and second_norm > 1e-12
        else None
    )
    if first_norm <= 1e-8 and second_norm <= 1e-8:
        rank = 0
        normalized_singular_values = [0.0, 0.0, 0.0]
    elif first_norm <= 1e-8 or second_norm <= 1e-8:
        rank = 1
        normalized_singular_values = [1.0, 0.0, 0.0]
    else:
        first_axis = first / first_norm
        second_axis = second / second_norm
        cross_norm = float(np.linalg.norm(np.cross(first_axis, second_axis)))
        if cross_norm <= 1e-6:
            rank = 1
            normalized_singular_values = [math.sqrt(2.0), 0.0, 0.0]
        else:
            bracket_axis = np.cross(first_axis, second_axis) / cross_norm
            normalized_closure = np.stack(
                (first_axis, second_axis, bracket_axis), axis=1
            )
            normalized_singular_values = np.linalg.svd(
                normalized_closure, compute_uv=False
            ).tolist()
            rank = 3
    return {
        "residual_convention": "learned @ aligned_oracle.T",
        "per_token_log_vectors": logs.tolist(),
        "independent_generator_log_vectors": [first.tolist(), second.tolist()],
        "independent_generator_axis_dot": axis_dot,
        "raw_log_bracket_norm": float(np.linalg.norm(bracket)),
        "defect_generator_commutator_frobenius": float(
            np.linalg.norm(defect_commutator)
        ),
        "normalized_lie_closure_singular_values": normalized_singular_values,
        "parallel_axis_sine_tolerance": 1e-6,
        "lie_closure_rank": rank,
        "first_generator_rational_angle": rational_angle_diagnostic(first_norm),
        "second_generator_rational_angle": rational_angle_diagnostic(second_norm),
        "interpretation": (
            "rank_three_local/generic dense-SO(3) risk; numerical diagnostic, not proof of irrationality"
            if rank == 3
            else "rank-one circle-like defect" if rank == 1
            else "numerically degenerate defect"
        ),
    }


def audit_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint["family"] != "pure_ga_rotor" or checkpoint["group"] != "a5":
        raise ValueError("audit requires a pure_ga_rotor A5 checkpoint")
    config = checkpoint["config"]
    input_elements = tuple(checkpoint["input_elements"])
    group = GROUPS["a5"]
    model = PureGroupActionModel(
        len(input_elements),
        group.order,
        family=checkpoint["family"],
        channels=config["channels"],
        max_rotor_angle=config["max_rotor_angle"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    full_actions = model.action_matrices().detach().cpu().double()
    anchor = strongest_commutator_channel(full_actions.to(device))
    learned = full_actions[..., 1:4, 1:4].numpy().transpose(1, 0, 2, 3)
    inverses = _element_inverses(group)
    orders = _element_orders(group)
    oracle_tokens = []
    for branch in (0, 1):
        representation = a5_orthogonal_irrep(group, branch=branch)
        oracle_tokens.append(
            np.stack([representation[inverses[element]] for element in input_elements])
        )

    channels = []
    for channel, channel_actions in enumerate(learned):
        alignments = []
        for branch in (0, 1):
            change, rms = align_representation(
                channel_actions, oracle_tokens[branch], seed=1000 * config["seed"] + 10 * channel + branch
            )
            alignments.append({"branch": branch, "alignment": change, "rms": rms})
        best = min(alignments, key=lambda value: value["rms"])
        commutator = channel_commutator(channel_actions)
        relator = channel_relator_mean(
            channel_actions, group.table, input_elements, orders
        )
        trace = float(np.trace(channel_actions[2]))
        branch_trace_distances = [
            abs(trace - float(np.trace(oracle_tokens[branch][2])))
            for branch in (0, 1)
        ]
        representation_like = (
            best["rms"] <= 0.05 and commutator >= 0.5 and relator <= 0.1
        )
        channel_report: dict[str, object] = {
            "channel": channel,
            "is_anchor": channel == anchor,
            "best_irrep_branch": best["branch"],
            "branch_alignment_rms": [value["rms"] for value in alignments],
            "token_2_five_cycle_trace": trace,
            "token_2_trace_distance_by_branch": branch_trace_distances,
            "maximum_generator_commutator": commutator,
            "mean_cyclic_and_pair_relator_rms": relator,
            "representation_like_threshold_pass": representation_like,
        }
        if channel == anchor:
            channel_report["defect_lie_audit"] = defect_lie_audit(
                channel_actions,
                oracle_tokens[int(best["branch"])],
                np.asarray(best["alignment"]),
            )
        channels.append(channel_report)

    return {
        "checkpoint": str(path),
        "checkpoint_sha256": checkpoint_sha256(path),
        "training_seed": config["seed"],
        "anchor_channel": anchor,
        "max_rotor_angle": config["max_rotor_angle"],
        "input_elements": list(input_elements),
        "oracle_token_2_trace_by_branch": [
            float(np.trace(oracle_tokens[branch][2])) for branch in (0, 1)
        ],
        "oracle_token_2_rotation_angle_by_branch": [
            float(Rotation.from_matrix(oracle_tokens[branch][2]).magnitude())
            for branch in (0, 1)
        ],
        "channels": channels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    report = {
        "experiment": "A5 irrep-branch and anchor defect Lie-closure audit",
        "device": str(device),
        "representation_like_thresholds": {
            "alignment_rms_at_most": 0.05,
            "commutator_at_least": 0.5,
            "mean_relator_rms_at_most": 0.1,
        },
        "results": [audit_checkpoint(path, device) for path in args.checkpoints],
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
