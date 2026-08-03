"""Demonstrate the central-sign information lost by rotor sandwich actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


QUATERNIONS = np.asarray(
    [
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
        (-1, 0, 0, 0),
        (0, -1, 0, 0),
        (0, 0, -1, 0),
        (0, 0, 0, -1),
    ],
    dtype=np.float64,
)


def quaternion_left_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.asarray(
        [
            (w, -x, -y, -z),
            (x, w, -z, y),
            (y, z, w, -x),
            (z, -y, x, w),
        ],
        dtype=np.float64,
    )


def quaternion_conjugation_rotation(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.asarray(
        [
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ],
        dtype=np.float64,
    )


def distinct_matrix_count(matrices: np.ndarray, tolerance: float = 1e-12) -> int:
    representatives: list[np.ndarray] = []
    for matrix in matrices:
        if not any(np.max(np.abs(matrix - other)) <= tolerance for other in representatives):
            representatives.append(matrix)
    return len(representatives)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left = np.stack([quaternion_left_matrix(q) for q in QUATERNIONS])
    sandwich = np.stack(
        [quaternion_conjugation_rotation(q) for q in QUATERNIONS]
    )
    spinor_initial = np.asarray((1.0, 0.0, 0.0, 0.0))
    vector_initial = np.asarray((1.0, 2.0, 3.0)) / np.sqrt(14.0)
    spinor_orbit = left @ spinor_initial
    sandwich_orbit = sandwich @ vector_initial
    spinor_pairwise = np.linalg.norm(
        spinor_orbit[:, None] - spinor_orbit[None, :], axis=-1
    )
    sandwich_pairwise = np.linalg.norm(
        sandwich_orbit[:, None] - sandwich_orbit[None, :], axis=-1
    )
    np.fill_diagonal(spinor_pairwise, np.inf)
    np.fill_diagonal(sandwich_pairwise, np.inf)
    identity = 0
    quaternion_i = 1
    minus_identity = 4
    spinor_i_squared = left[quaternion_i] @ left[quaternion_i] @ spinor_initial
    sandwich_i_squared = (
        sandwich[quaternion_i] @ sandwich[quaternion_i] @ vector_initial
    )
    left_i_rank = int(np.linalg.matrix_rank(np.eye(4) - left[quaternion_i]))
    duplicated_left_i_rank = int(
        np.linalg.matrix_rank(
            np.eye(8) - np.kron(np.eye(2), left[quaternion_i])
        )
    )
    report = {
        "experiment": "spinor central-fidelity audit on Q8",
        "group_order": 8,
        "left_spinor_distinct_action_matrices": distinct_matrix_count(left),
        "sandwich_distinct_action_matrices": distinct_matrix_count(sandwich),
        "left_spinor_distinct_orbit_states": distinct_matrix_count(
            spinor_orbit[:, :, None]
        ),
        "sandwich_distinct_orbit_states": distinct_matrix_count(
            sandwich_orbit[:, :, None]
        ),
        "minimum_spinor_orbit_separation": float(spinor_pairwise.min()),
        "minimum_sandwich_orbit_separation": float(sandwich_pairwise.min()),
        "central_sign_action": {
            "left_minus_identity_is_negative_identity": bool(
                np.array_equal(left[minus_identity], -left[identity])
            ),
            "sandwich_minus_identity_equals_identity": bool(
                np.array_equal(sandwich[minus_identity], sandwich[identity])
            ),
            "spinor_i_squared_distance_from_identity_state": float(
                np.linalg.norm(spinor_i_squared - spinor_initial)
            ),
            "sandwich_i_squared_distance_from_identity_state": float(
                np.linalg.norm(sandwich_i_squared - vector_initial)
            ),
        },
        "balanced_central_pair_oracle": {
            "spinor_pair_member_accuracy": 1.0,
            "spinor_both_members_correct_accuracy": 1.0,
            "sandwich_pair_member_ceiling": 0.5,
            "sandwich_both_members_correct_ceiling": 0.0,
        },
        "householder_capacity": {
            "rank_I_minus_left_i_in_R4": left_i_rank,
            "minimum_reflections_for_left_i_in_R4": left_i_rank,
            "rank_I_minus_duplicated_left_i_in_R8": duplicated_left_i_rank,
            "minimum_reflections_without_block_sharing_in_R8": (
                duplicated_left_i_rank
            ),
            "two_reflection_O8_is_a_capable_faithful_q8_baseline": False,
            "four_reflection_O4_shared_over_two_blocks_is_capable": True,
        },
        "theorem_scope": (
            "pure conjugation actions factor through Spin(3)/{+-1}=SO(3); "
            "left spinor actions retain the central sign"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
