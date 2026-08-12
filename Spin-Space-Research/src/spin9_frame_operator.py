"""Exact frame-operator reduction for Spin(9) spinor sensing.

For a collection of spinor probes, the infinitesimal information matrix
depends on the probes only through their frame operator ``M = sum s s^T``.
The symmetric Clifford decomposition then identifies the complete kernel of
this linear information map and the optimum of its convex approximate-design
relaxation.

All certificate-facing calculations use integer matrices and modular ranks.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from spin9_dirac_clifford import (
    CERTIFICATE_PRIMES,
    SPINOR_WITNESSES,
    build_spin9_clifford_system,
    modular_rank,
    observation_matrix,
)

SYMMETRIC_DIMENSION = 16 * 17 // 2
FOUR_FORM_DIMENSION = 126


def frame_operator(probes: Sequence[np.ndarray]) -> np.ndarray:
    """Return ``sum_r s_r s_r^T`` without normalizing the probes."""

    if not probes:
        return np.zeros((16, 16), dtype=np.int64)
    return sum(
        (np.outer(np.asarray(probe), np.asarray(probe)) for probe in probes),
        start=np.zeros((16, 16), dtype=np.int64),
    )


def scaled_information_from_frame(
    frame: np.ndarray, doubled_generators: np.ndarray
) -> np.ndarray:
    """Return four times the conventional information matrix.

    The maintained integer generators are ``H_a = 2 G_a``.  Thus this
    function returns ``tr(M H_a^T H_b) = 4 I_ab``.
    """

    return np.asarray(
        [
            [np.trace(frame @ left.T @ right) for right in doubled_generators]
            for left in doubled_generators
        ],
        dtype=np.int64,
    )


def scaled_information_direct(
    probes: Sequence[np.ndarray], doubled_generators: np.ndarray
) -> np.ndarray:
    """Return the same scaled information by explicitly stacking observations."""

    blocks = [
        np.stack([generator @ probe for generator in doubled_generators], axis=1)
        for probe in probes
    ]
    jacobian = np.concatenate(blocks, axis=0)
    return jacobian.T @ jacobian


def symmetric_clifford_basis(involutions: np.ndarray) -> list[np.ndarray]:
    """Return the grade 0, 1, and 4 basis of ``Sym(16,R)``."""

    basis = [np.eye(16, dtype=np.int64), *involutions]
    for indices in itertools.combinations(range(9), 4):
        matrix = np.eye(16, dtype=np.int64)
        for index in indices:
            matrix = matrix @ involutions[index]
        basis.append(matrix)
    return basis


def _upper_triangle_vector(matrix: np.ndarray) -> np.ndarray:
    rows, columns = np.triu_indices(matrix.shape[0])
    return matrix[rows, columns]


def four_form_operator(frame: np.ndarray, involutions: np.ndarray) -> np.ndarray:
    """Return the symmetric four-form action on ``Lambda^2 R^9``.

    In the ordered bivector basis ``e_i wedge e_j``, the disjoint-pair entry
    is ``tr(M P_i P_j P_k P_l)`` and all other entries vanish.
    """

    pairs = tuple(itertools.combinations(range(9), 2))
    operator = np.zeros((len(pairs), len(pairs)), dtype=np.int64)
    for row, (first, second) in enumerate(pairs):
        for column, (third, fourth) in enumerate(pairs):
            if len({first, second, third, fourth}) != 4:
                continue
            product = (
                involutions[first]
                @ involutions[second]
                @ involutions[third]
                @ involutions[fourth]
            )
            operator[row, column] = np.trace(frame @ product)
    return operator


def diagnostics() -> dict[str, object]:
    """Replay the exact frame reduction and approximate-design theorem."""

    system = build_spin9_clifford_system()
    basis = symmetric_clifford_basis(system.involutions)
    flattened = np.stack([matrix.reshape(-1) for matrix in basis])
    gram = flattened @ flattened.T
    basis_orthogonal = np.array_equal(
        gram, 16 * np.eye(SYMMETRIC_DIMENSION, dtype=np.int64)
    )
    basis_symmetric = all(np.array_equal(matrix, matrix.T) for matrix in basis)

    probes = [probe for probe in SPINOR_WITNESSES]
    frame = frame_operator(probes)
    direct = scaled_information_direct(probes, system.doubled_spin_generators)
    reduced = scaled_information_from_frame(frame, system.doubled_spin_generators)
    four_form_action = four_form_operator(frame, system.involutions)

    trace_frame = int(np.trace(frame))
    trace_frame_square = int(np.trace(frame @ frame))
    hopf = np.asarray(
        [np.trace(frame @ involution) for involution in system.involutions],
        dtype=np.int64,
    )
    moment_left = int(np.trace(reduced @ reduced))
    moment_right = 30 * trace_frame**2 + 96 * trace_frame_square - 6 * int(hopf @ hopf)

    image_columns = np.stack(
        [
            _upper_triangle_vector(
                scaled_information_from_frame(matrix, system.doubled_spin_generators)
            )
            for matrix in basis
        ],
        axis=1,
    )
    image_ranks = {
        str(prime): modular_rank(image_columns, prime) for prime in CERTIFICATE_PRIMES
    }
    vector_images_zero = all(
        not np.any(image_columns[:, 1 + index]) for index in range(9)
    )

    pair_observation = observation_matrix(
        system.doubled_spin_generators, SPINOR_WITNESSES[:2]
    )
    triple_observation = observation_matrix(
        system.doubled_spin_generators, SPINOR_WITNESSES
    )
    pair_ranks = {
        str(prime): modular_rank(pair_observation, prime)
        for prime in CERTIFICATE_PRIMES
    }
    triple_ranks = {
        str(prime): modular_rank(triple_observation, prime)
        for prime in CERTIFICATE_PRIMES
    }
    boundary_nullity = 36 - min(pair_ranks.values())

    # These numerators represent M = (3 I + alpha P_0) / 16.  Every
    # |alpha| <= 3 is positive semidefinite and has the same information
    # matrix.  Alpha=3 is a rank-eight boundary point.
    scalar_numerator = 3 * np.eye(16, dtype=np.int64)
    shifted_numerator = scalar_numerator + system.involutions[0]
    boundary_numerator = scalar_numerator + 3 * system.involutions[0]
    selected_involution = system.involutions[0]
    selected_involution_exact = bool(
        np.array_equal(selected_involution @ selected_involution, np.eye(16))
        and np.array_equal(selected_involution, selected_involution.T)
        and np.trace(selected_involution) == 0
    )
    boundary_ranks = {
        str(prime): modular_rank(boundary_numerator, prime)
        for prime in CERTIFICATE_PRIMES
    }
    scalar_information = scaled_information_from_frame(
        scalar_numerator, system.doubled_spin_generators
    )
    shifted_information = scaled_information_from_frame(
        shifted_numerator, system.doubled_spin_generators
    )

    report: dict[str, object] = {
        "schema_version": 1,
        "claim_scope": "exact Spin(9) spinor frame-operator reduction",
        "symmetric_basis_grades": [0, 1, 4],
        "symmetric_basis_size": len(basis),
        "symmetric_basis_expected_size": SYMMETRIC_DIMENSION,
        "symmetric_basis_all_symmetric": basis_symmetric,
        "symmetric_basis_orthogonal_norm_squared": 16,
        "symmetric_basis_orthogonal": basis_orthogonal,
        "frame_reduction_identity": np.array_equal(direct, reduced),
        "four_form_operator_symmetric": np.array_equal(
            four_form_action, four_form_action.T
        ),
        "four_form_operator_trace_zero": int(np.trace(four_form_action)) == 0,
        "dirac_four_form_information_identity": np.array_equal(
            reduced,
            trace_frame * np.eye(36, dtype=np.int64) - four_form_action,
        ),
        "information_determinant_reduction": "det I(M) = 4^-36 det(t I_36 - K_q(M))",
        "information_map_ranks": image_ranks,
        "information_map_kernel_dimension": SYMMETRIC_DIMENSION
        - min(image_ranks.values()),
        "grade_one_images_zero": vector_images_zero,
        "generic_pair_observation_ranks": pair_ranks,
        "transverse_triple_observation_ranks": triple_ranks,
        "generic_pair_information_nullity": boundary_nullity,
        "transverse_boundary_determinant_vanishing_order": 2 * boundary_nullity,
        "trace_information_identity": int(np.trace(reduced)) == 36 * trace_frame,
        "scaled_second_moment_identity": moment_left == moment_right,
        "scaled_second_moment_observed": moment_left,
        "scaled_second_moment_expected": moment_right,
        "approximate_design_information_numerator_diagonal": int(
            scalar_information[0, 0]
        ),
        "approximate_design_information_numerator_off_diagonal_max_abs": int(
            np.max(np.abs(scalar_information - np.diag(np.diag(scalar_information))))
        ),
        "vector_gauge_preserves_information": np.array_equal(
            scalar_information, shifted_information
        ),
        "selected_gauge_involution_exact": selected_involution_exact,
        "positive_gauge_example_min_eigenvalue_numerator": 2,
        "boundary_gauge_ranks": boundary_ranks,
        "approximate_optimal_information": "(3/4) I_36",
        "approximate_optimal_determinant": "(3/4)^36",
        "exact_three_probe_rank_bound": 3,
        "approximate_optimal_rank_set": [8, 16],
        "exact_three_probe_attains_approximate_optimum": False,
        "global_exact_three_probe_optimum_claimed": False,
    }
    report["passed"] = bool(
        report["symmetric_basis_size"] == SYMMETRIC_DIMENSION
        and report["symmetric_basis_all_symmetric"]
        and report["symmetric_basis_orthogonal"]
        and report["frame_reduction_identity"]
        and report["four_form_operator_symmetric"]
        and report["four_form_operator_trace_zero"]
        and report["dirac_four_form_information_identity"]
        and set(image_ranks.values()) == {127}
        and report["information_map_kernel_dimension"] == 9
        and report["grade_one_images_zero"]
        and set(pair_ranks.values()) == {28}
        and set(triple_ranks.values()) == {36}
        and report["generic_pair_information_nullity"] == 8
        and report["transverse_boundary_determinant_vanishing_order"] == 16
        and report["trace_information_identity"]
        and report["scaled_second_moment_identity"]
        and report["approximate_design_information_numerator_diagonal"] == 48
        and report["approximate_design_information_numerator_off_diagonal_max_abs"] == 0
        and report["vector_gauge_preserves_information"]
        and report["selected_gauge_involution_exact"]
        and report["positive_gauge_example_min_eigenvalue_numerator"] == 2
        and set(boundary_ranks.values()) == {8}
        and not report["exact_three_probe_attains_approximate_optimum"]
        and not report["global_exact_three_probe_optimum_claimed"]
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional JSON artifact path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = diagnostics()
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
