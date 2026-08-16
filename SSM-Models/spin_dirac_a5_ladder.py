"""Exact Clifford ladder and binary-icosahedral subgroup diagnostics.

This module extends the maintained real Spin(8) gamma system through Spin(12)
without pretending that Spin(8) triality persists in higher dimensions.  It
also restricts every spinor module to the same embedded Spin(3), where A5 acts
on vectors and its binary lift 2.A5 acts on spinors.

The Clifford and branching checks use Gaussian-integer matrices.  The
icosahedral quaternion construction uses float64 values in a fixed algebraic
formula; group enumeration and residuals are therefore numerical certificates,
not symbolic proofs of the binary-icosahedral presentation.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from spin8_triality import build_spin8_triality_algebra

LADDER_DIMENSIONS = (3, 8, 9, 10, 11, 12)
PRESENTATION_A = (1, 3, 2, 5, 4)
PRESENTATION_B = (2, 4, 3, 1, 5)
QUATERNION_TOLERANCE = 5e-11


@dataclass(frozen=True)
class CliffordStage:
    """One complex Euclidean Clifford representation."""

    vector_dimension: int
    gamma: np.ndarray
    construction: str

    @property
    def spinor_dimension(self) -> int:
        return int(self.gamma.shape[-1])


def _clean_complex(matrix: np.ndarray, tolerance: float = 1e-13) -> np.ndarray:
    """Remove floating zero noise while preserving Gaussian-integer entries."""

    result = np.asarray(matrix, dtype=np.complex128).copy()
    real = result.real
    imaginary = result.imag
    real[np.abs(real) < tolerance] = 0.0
    imaginary[np.abs(imaginary) < tolerance] = 0.0
    real = np.where(np.abs(real - np.rint(real)) < tolerance, np.rint(real), real)
    imaginary = np.where(
        np.abs(imaginary - np.rint(imaginary)) < tolerance,
        np.rint(imaginary),
        imaginary,
    )
    return real + 1j * imaginary


def chirality_operator(gamma: np.ndarray) -> np.ndarray:
    """Return the Hermitian chirality involution for an even Clifford system."""

    vector_dimension, spinor_dimension, _ = gamma.shape
    if vector_dimension % 2:
        raise ValueError("chirality is defined here only in even dimension")
    product = np.eye(spinor_dimension, dtype=np.complex128)
    for matrix in gamma:
        product = product @ matrix
    return _clean_complex((1j ** (vector_dimension // 2)) * product)


def _even_to_next_odd(stage: CliffordStage) -> CliffordStage:
    chirality = chirality_operator(stage.gamma)
    return CliffordStage(
        vector_dimension=stage.vector_dimension + 1,
        gamma=np.concatenate((stage.gamma, chirality[None]), axis=0),
        construction=f"Cl({stage.vector_dimension}) plus its chirality involution",
    )


def _odd_to_next_even(stage: CliffordStage) -> CliffordStage:
    pauli_x = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
    pauli_y = np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128)
    identity = np.eye(stage.spinor_dimension, dtype=np.complex128)
    lifted = [np.kron(pauli_x, matrix) for matrix in stage.gamma]
    lifted.append(np.kron(pauli_y, identity))
    return CliffordStage(
        vector_dimension=stage.vector_dimension + 1,
        gamma=_clean_complex(np.stack(lifted)),
        construction=(f"graded Pauli doubling of Cl({stage.vector_dimension})"),
    )


def build_clifford_ladder() -> dict[int, CliffordStage]:
    """Build Spin(3), then the maintained Spin(8) system and its 9--12 ladder."""

    pauli_x = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
    pauli_y = np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128)
    pauli_z = np.asarray([[1, 0], [0, -1]], dtype=np.complex128)
    spin3 = CliffordStage(
        vector_dimension=3,
        gamma=np.stack((pauli_x, pauli_y, pauli_z)),
        construction="Pauli matrices",
    )

    maintained = build_spin8_triality_algebra()
    spin8 = CliffordStage(
        vector_dimension=8,
        gamma=np.asarray(maintained.gamma, dtype=np.complex128),
        construction="maintained octonionic real Spin(8) gamma system",
    )
    spin9 = _even_to_next_odd(spin8)
    spin10 = _odd_to_next_even(spin9)
    spin11 = _even_to_next_odd(spin10)
    spin12 = _odd_to_next_even(spin11)
    return {
        stage.vector_dimension: stage
        for stage in (spin3, spin8, spin9, spin10, spin11, spin12)
    }


def generator_pairs(vector_dimension: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (left, right)
        for left in range(vector_dimension)
        for right in range(left + 1, vector_dimension)
    )


def spin_generators(
    stage: CliffordStage,
) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    """Return conventional generators T_ij = gamma_i gamma_j / 2."""

    pairs = generator_pairs(stage.vector_dimension)
    generators = np.stack(
        [0.5 * stage.gamma[left] @ stage.gamma[right] for left, right in pairs]
    )
    return pairs, generators


def _max_abs(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(matrix))) if matrix.size else 0.0


def clifford_stage_diagnostics(stage: CliffordStage) -> dict[str, object]:
    """Check Clifford, Hermiticity, chirality, and Lie-vector covariance laws."""

    gamma = stage.gamma
    vector_dimension = stage.vector_dimension
    identity = np.eye(stage.spinor_dimension, dtype=np.complex128)
    clifford_residual = 0.0
    for left in range(vector_dimension):
        for right in range(vector_dimension):
            target = 2.0 * identity if left == right else np.zeros_like(identity)
            clifford_residual = max(
                clifford_residual,
                _max_abs(
                    gamma[left] @ gamma[right] + gamma[right] @ gamma[left] - target
                ),
            )
    hermitian_residual = _max_abs(gamma - gamma.conj().transpose(0, 2, 1))

    pairs, generators = spin_generators(stage)
    skew_hermitian_residual = _max_abs(
        generators + generators.conj().transpose(0, 2, 1)
    )
    covariance_residual = 0.0
    for generator, (left, right) in zip(generators, pairs, strict=True):
        for vector_index in range(vector_dimension):
            target = np.zeros_like(identity)
            if vector_index == right:
                target += gamma[left]
            if vector_index == left:
                target -= gamma[right]
            covariance_residual = max(
                covariance_residual,
                _max_abs(
                    generator @ gamma[vector_index]
                    - gamma[vector_index] @ generator
                    - target
                ),
            )

    chirality: dict[str, object] | None = None
    if vector_dimension % 2 == 0:
        operator = chirality_operator(gamma)
        eigenvalues = np.linalg.eigvalsh(operator)
        chirality = {
            "hermitian_max_abs": _max_abs(operator - operator.conj().T),
            "square_max_abs": _max_abs(operator @ operator - identity),
            "gamma_anticommutator_max_abs": max(
                _max_abs(operator @ matrix + matrix @ operator) for matrix in gamma
            ),
            "positive_multiplicity": int(np.sum(eigenvalues > 0.5)),
            "negative_multiplicity": int(np.sum(eigenvalues < -0.5)),
        }

    passed = (
        clifford_residual == 0.0
        and hermitian_residual == 0.0
        and skew_hermitian_residual == 0.0
        and covariance_residual == 0.0
        and (
            chirality is None
            or (
                chirality["hermitian_max_abs"] == 0.0
                and chirality["square_max_abs"] == 0.0
                and chirality["gamma_anticommutator_max_abs"] == 0.0
                and chirality["positive_multiplicity"] == stage.spinor_dimension // 2
                and chirality["negative_multiplicity"] == stage.spinor_dimension // 2
            )
        )
    )
    return {
        "vector_dimension": vector_dimension,
        "spinor_dimension": stage.spinor_dimension,
        "construction": stage.construction,
        "gamma_shape": list(gamma.shape),
        "spin_lie_dimension": len(pairs),
        "clifford_max_abs": clifford_residual,
        "gamma_hermitian_max_abs": hermitian_residual,
        "spin_generator_skew_hermitian_max_abs": skew_hermitian_residual,
        "spin_vector_covariance_max_abs": covariance_residual,
        "chirality": chirality,
        "passed": bool(passed),
    }


def _pair_generator(stage: CliffordStage, left: int, right: int) -> np.ndarray:
    return 0.5 * stage.gamma[left] @ stage.gamma[right]


def branching_diagnostics(stages: dict[int, CliffordStage]) -> dict[str, object]:
    """Check the literal recursive embeddings used by the 8--12 ladder."""

    spin8, spin9 = stages[8], stages[9]
    spin10, spin11, spin12 = stages[10], stages[11], stages[12]
    identity2 = np.eye(2, dtype=np.complex128)

    spin8_to_9 = max(
        _max_abs(spin9.gamma[:8] - spin8.gamma),
        _max_abs(spin9.gamma[8] - chirality_operator(spin8.gamma)),
    )
    spin9_to_10 = max(
        _max_abs(
            _pair_generator(spin10, left, right)
            - np.kron(identity2, _pair_generator(spin9, left, right))
        )
        for left, right in generator_pairs(9)
    )
    spin10_to_11 = max(
        _max_abs(spin11.gamma[:10] - spin10.gamma),
        _max_abs(spin11.gamma[10] - chirality_operator(spin10.gamma)),
    )
    spin11_to_12 = max(
        _max_abs(
            _pair_generator(spin12, left, right)
            - np.kron(identity2, _pair_generator(spin11, left, right))
        )
        for left, right in generator_pairs(11)
    )
    residuals = {
        "spin8_to_spin9_chirality_extension_max_abs": spin8_to_9,
        "spin9_to_spin10_spin_action_doubling_max_abs": spin9_to_10,
        "spin10_to_spin11_chirality_extension_max_abs": spin10_to_11,
        "spin11_to_spin12_spin_action_doubling_max_abs": spin11_to_12,
    }
    return {**residuals, "passed": max(residuals.values()) == 0.0}


def permutation_compose(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    """Compose permutations as functions, returning left after right."""

    return tuple(left[value - 1] for value in right)


def permutation_power(permutation: Sequence[int], exponent: int) -> tuple[int, ...]:
    value = tuple(range(1, len(permutation) + 1))
    for _ in range(exponent):
        value = permutation_compose(value, permutation)
    return value


def permutation_is_even(permutation: Sequence[int]) -> bool:
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return inversions % 2 == 0


def enumerate_permutation_group(
    generators: Iterable[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    generators = tuple(tuple(generator) for generator in generators)
    identity = tuple(range(1, len(generators[0]) + 1))
    queue = deque([identity])
    reached = {identity}
    while queue:
        current = queue.popleft()
        for generator in generators:
            candidate = permutation_compose(current, generator)
            if candidate not in reached:
                reached.add(candidate)
                queue.append(candidate)
    return tuple(sorted(reached))


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for quaternions stored as (scalar, x, y, z)."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    scalar = left[0] * right[0] - float(np.dot(left[1:], right[1:]))
    vector = left[0] * right[1:] + right[0] * left[1:] + np.cross(left[1:], right[1:])
    return np.concatenate(([scalar], vector))


def quaternion_inverse(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    return np.concatenate(([quaternion[0]], -quaternion[1:])) / float(
        np.dot(quaternion, quaternion)
    )


def quaternion_power(quaternion: np.ndarray, exponent: int) -> np.ndarray:
    value = np.asarray([1.0, 0.0, 0.0, 0.0])
    for _ in range(exponent):
        value = quaternion_multiply(value, quaternion)
    return value


def icosahedral_spin_generators() -> tuple[np.ndarray, np.ndarray]:
    """Return lifts of the (2,3,5) generators in unit quaternions."""

    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    a = np.asarray([0.0, 0.0, 0.0, 1.0])
    b = 0.5 * np.asarray([1.0, math.sqrt(2.0 - golden_ratio), 0.0, -golden_ratio])
    return a, b


def _quaternion_key(
    quaternion: np.ndarray, *, projective: bool = False, digits: int = 10
) -> tuple[float, ...]:
    value = np.asarray(quaternion, dtype=np.float64)
    value = value / np.linalg.norm(value)
    value[np.abs(value) < 10.0 ** (-digits)] = 0.0
    if projective:
        nonzero = np.flatnonzero(np.abs(value) > 10.0 ** (-digits))
        if nonzero.size and value[int(nonzero[0])] < 0.0:
            value = -value
    return tuple(float(item) for item in np.round(value, digits))


def enumerate_quaternion_group(
    generators: Iterable[np.ndarray], *, maximum_order: int = 256
) -> tuple[np.ndarray, ...]:
    """Numerically enumerate the group generated by fixed unit quaternions."""

    generators = tuple(
        np.asarray(generator, dtype=np.float64) for generator in generators
    )
    steps = generators + tuple(
        quaternion_inverse(generator) for generator in generators
    )
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    queue = deque([identity])
    reached = {_quaternion_key(identity): identity}
    while queue:
        current = queue.popleft()
        for step in steps:
            candidate = quaternion_multiply(current, step)
            candidate /= np.linalg.norm(candidate)
            key = _quaternion_key(candidate)
            if key not in reached:
                reached[key] = candidate
                queue.append(candidate)
                if len(reached) > maximum_order:
                    raise RuntimeError("quaternion enumeration exceeded its safety cap")
    return tuple(reached[key] for key in sorted(reached))


def quaternion_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Return the active SO(3) rotation represented by a unit quaternion."""

    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def embedded_vector_rotation(quaternion: np.ndarray, dimension: int) -> np.ndarray:
    result = np.eye(dimension)
    result[:3, :3] = quaternion_rotation_matrix(quaternion)
    return result


def spin_matrix_from_quaternion(
    stage: CliffordStage, quaternion: np.ndarray
) -> np.ndarray:
    """Represent a Spin(3) quaternion in the first three Clifford directions."""

    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    gamma = stage.gamma
    quaternion_i = -gamma[1] @ gamma[2]
    quaternion_j = -gamma[2] @ gamma[0]
    quaternion_k = -gamma[0] @ gamma[1]
    return _clean_complex(
        w * np.eye(stage.spinor_dimension)
        + x * quaternion_i
        + y * quaternion_j
        + z * quaternion_k
    )


def _matrix_power(matrix: np.ndarray, exponent: int) -> np.ndarray:
    return np.linalg.matrix_power(matrix, exponent)


def _spin_vector_covariance_residual(
    stage: CliffordStage, quaternion: np.ndarray
) -> float:
    spin = spin_matrix_from_quaternion(stage, quaternion)
    rotation = embedded_vector_rotation(quaternion, stage.vector_dimension)
    residual = 0.0
    for source in range(stage.vector_dimension):
        observed = spin @ stage.gamma[source] @ spin.conj().T
        expected = sum(
            rotation[target, source] * stage.gamma[target]
            for target in range(stage.vector_dimension)
        )
        residual = max(residual, _max_abs(observed - expected))
    return residual


def a5_spin_stage_diagnostics(
    stage: CliffordStage,
    group: Sequence[np.ndarray],
    a: np.ndarray,
    b: np.ndarray,
) -> dict[str, object]:
    """Audit the same binary-icosahedral generators on one spinor module."""

    identity = np.eye(stage.spinor_dimension, dtype=np.complex128)
    spin_a = spin_matrix_from_quaternion(stage, a)
    spin_b = spin_matrix_from_quaternion(stage, b)
    spin_ab = spin_a @ spin_b
    relator_residuals = {
        "a_squared_to_minus_identity_max_abs": _max_abs(spin_a @ spin_a + identity),
        "b_cubed_to_minus_identity_max_abs": _max_abs(
            _matrix_power(spin_b, 3) + identity
        ),
        "ab_fifth_to_minus_identity_max_abs": _max_abs(
            _matrix_power(spin_ab, 5) + identity
        ),
    }
    unitary_residual = 0.0
    central_sign_residual = 0.0
    character_residual = 0.0
    chiral_character_residual = 0.0
    chirality_commutator = 0.0
    chirality = (
        chirality_operator(stage.gamma) if stage.vector_dimension % 2 == 0 else None
    )
    for quaternion in group:
        spin = spin_matrix_from_quaternion(stage, quaternion)
        unitary_residual = max(
            unitary_residual, _max_abs(spin.conj().T @ spin - identity)
        )
        central_sign_residual = max(
            central_sign_residual,
            _max_abs(spin_matrix_from_quaternion(stage, -quaternion) + spin),
        )
        expected_character = stage.spinor_dimension * float(quaternion[0])
        character_residual = max(
            character_residual, abs(complex(np.trace(spin)) - expected_character)
        )
        if chirality is not None:
            chirality_commutator = max(
                chirality_commutator, _max_abs(chirality @ spin - spin @ chirality)
            )
            for sign in (-1.0, 1.0):
                projector = 0.5 * (identity + sign * chirality)
                expected_chiral = 0.5 * stage.spinor_dimension * float(quaternion[0])
                chiral_character_residual = max(
                    chiral_character_residual,
                    abs(complex(np.trace(projector @ spin)) - expected_chiral),
                )

    covariance_residual = max(
        _spin_vector_covariance_residual(stage, generator) for generator in (a, b)
    )
    tolerance = QUATERNION_TOLERANCE
    passed = (
        max(relator_residuals.values()) <= tolerance
        and unitary_residual <= tolerance
        and central_sign_residual <= tolerance
        and character_residual <= tolerance
        and chiral_character_residual <= tolerance
        and chirality_commutator <= tolerance
        and covariance_residual <= tolerance
    )
    return {
        "vector_dimension": stage.vector_dimension,
        "spinor_dimension": stage.spinor_dimension,
        "spin3_fundamental_multiplicity": stage.spinor_dimension // 2,
        "chiral_spinor_dimension": (
            stage.spinor_dimension // 2 if stage.vector_dimension % 2 == 0 else None
        ),
        "spin3_fundamental_multiplicity_per_chirality": (
            stage.spinor_dimension // 4 if stage.vector_dimension % 2 == 0 else None
        ),
        "triality_dimension_match": (
            stage.vector_dimension == 8
            and stage.spinor_dimension // 2 == stage.vector_dimension
        ),
        "lifted_presentation_residuals": relator_residuals,
        "unitarity_max_abs": unitary_residual,
        "central_minus_one_visibility_max_abs": central_sign_residual,
        "spin3_isotypic_character_max_abs": float(character_residual),
        "chiral_spin3_isotypic_character_max_abs": float(chiral_character_residual),
        "chirality_commutator_max_abs": chirality_commutator,
        "spin_to_vector_covariance_max_abs": covariance_residual,
        "passed": bool(passed),
    }


def vector_centralizer_diagnostics(
    a: np.ndarray, b: np.ndarray, dimensions: Iterable[int]
) -> dict[str, object]:
    """Measure the Lie centralizer of the embedded icosahedral vector action.

    The expected centralizer is the untouched ``so(n-3)`` block.  This is the
    local stabilizer calculation needed before treating the embeddings as
    points of a representation variety or quotient stack.
    """

    reports: dict[str, object] = {}
    for dimension in dimensions:
        pairs = generator_pairs(dimension)
        basis = []
        for left, right in pairs:
            matrix = np.zeros((dimension, dimension))
            matrix[left, right] = 1.0
            matrix[right, left] = -1.0
            basis.append(matrix)
        rotations = (
            embedded_vector_rotation(a, dimension),
            embedded_vector_rotation(b, dimension),
        )
        constraints = np.concatenate(
            [
                np.stack(
                    [
                        (matrix @ rotation - rotation @ matrix).reshape(-1)
                        for matrix in basis
                    ],
                    axis=1,
                )
                for rotation in rotations
            ],
            axis=0,
        )
        singular_values = np.linalg.svd(constraints, compute_uv=False)
        tolerance = 1e-10 * float(singular_values[0])
        rank = int(np.sum(singular_values > tolerance))
        centralizer_dimension = len(pairs) - rank
        expected = (dimension - 3) * (dimension - 4) // 2
        reports[str(dimension)] = {
            "ambient_spin_lie_dimension": len(pairs),
            "centralizer_dimension_numerical": centralizer_dimension,
            "expected_so_n_minus_3_dimension": expected,
            "conjugacy_orbit_dimension": len(pairs) - centralizer_dimension,
            "smallest_singular_value_above_rank_cutoff": (
                float(singular_values[rank - 1]) if rank else None
            ),
            "largest_singular_value_below_rank_cutoff": (
                float(singular_values[rank]) if rank < len(singular_values) else None
            ),
            "passed": centralizer_dimension == expected,
        }
    return {
        "interpretation": (
            "the only infinitesimal vector stabilizer is the untouched so(n-3) block"
        ),
        "stages": reports,
        "passed": all(report["passed"] for report in reports.values()),
    }


def a5_diagnostics(stages: dict[int, CliffordStage]) -> dict[str, object]:
    """Check exact permutation A5 and the numerical binary lift separately."""

    permutation_identity = (1, 2, 3, 4, 5)
    permutation_ab = permutation_compose(PRESENTATION_A, PRESENTATION_B)
    permutation_group = enumerate_permutation_group((PRESENTATION_A, PRESENTATION_B))
    permutation_checks = {
        "a_squared_is_identity": permutation_power(PRESENTATION_A, 2)
        == permutation_identity,
        "b_cubed_is_identity": permutation_power(PRESENTATION_B, 3)
        == permutation_identity,
        "ab_fifth_is_identity": permutation_power(permutation_ab, 5)
        == permutation_identity,
        "generated_group_order": len(permutation_group),
        "all_generated_permutations_are_even": all(
            permutation_is_even(element) for element in permutation_group
        ),
    }

    a, b = icosahedral_spin_generators()
    ab = quaternion_multiply(a, b)
    quaternion_group = enumerate_quaternion_group((a, b))
    projective_order = len(
        {_quaternion_key(value, projective=True) for value in quaternion_group}
    )
    minus_one = np.asarray([-1.0, 0.0, 0.0, 0.0])
    quaternion_relators = {
        "a_squared_to_minus_one_max_abs": _max_abs(quaternion_power(a, 2) - minus_one),
        "b_cubed_to_minus_one_max_abs": _max_abs(quaternion_power(b, 3) - minus_one),
        "ab_fifth_to_minus_one_max_abs": _max_abs(quaternion_power(ab, 5) - minus_one),
    }
    vector_a = quaternion_rotation_matrix(a)
    vector_b = quaternion_rotation_matrix(b)
    vector_relators = {
        "a_squared_to_identity_max_abs": _max_abs(
            _matrix_power(vector_a, 2) - np.eye(3)
        ),
        "b_cubed_to_identity_max_abs": _max_abs(_matrix_power(vector_b, 3) - np.eye(3)),
        "ab_fifth_to_identity_max_abs": _max_abs(
            _matrix_power(vector_a @ vector_b, 5) - np.eye(3)
        ),
    }
    stage_reports = {
        str(dimension): a5_spin_stage_diagnostics(
            stages[dimension], quaternion_group, a, b
        )
        for dimension in LADDER_DIMENSIONS
    }
    centralizers = vector_centralizer_diagnostics(a, b, LADDER_DIMENSIONS)
    passed = (
        permutation_checks["a_squared_is_identity"]
        and permutation_checks["b_cubed_is_identity"]
        and permutation_checks["ab_fifth_is_identity"]
        and permutation_checks["generated_group_order"] == 60
        and permutation_checks["all_generated_permutations_are_even"]
        and len(quaternion_group) == 120
        and projective_order == 60
        and max(quaternion_relators.values()) <= QUATERNION_TOLERANCE
        and max(vector_relators.values()) <= QUATERNION_TOLERANCE
        and all(report["passed"] for report in stage_reports.values())
        and centralizers["passed"]
    )
    return {
        "abstract_a5_permutation_certificate": permutation_checks,
        "binary_lift": {
            "construction": (
                "a=(0,0,0,1), b=(1,sqrt(2-phi),0,-phi)/2, phi=(1+sqrt(5))/2"
            ),
            "analytic_relation": "a^2=b^3=(ab)^5=-1 before projection",
            "generated_group_order_numerical": len(quaternion_group),
            "projective_group_order_numerical": projective_order,
            "quaternion_relator_max_abs": quaternion_relators,
            "projected_vector_relator_max_abs": vector_relators,
        },
        "spinor_restrictions": stage_reports,
        "vector_centralizers": centralizers,
        "tolerance": QUATERNION_TOLERANCE,
        "passed": bool(passed),
    }


def diagnostics() -> dict[str, object]:
    """Run the complete Clifford, branching, and A5/2.A5 audit."""

    stages = build_clifford_ladder()
    stage_reports = {
        str(dimension): clifford_stage_diagnostics(stages[dimension])
        for dimension in LADDER_DIMENSIONS
    }
    branching = branching_diagnostics(stages)
    a5 = a5_diagnostics(stages)
    passed = (
        all(report["passed"] for report in stage_reports.values())
        and branching["passed"]
        and a5["passed"]
    )
    return {
        "schema_version": 1,
        "experiment": "Spin(3)-to-Spin(12) Dirac ladder with A5 binary lift",
        "ladder": list(LADDER_DIMENSIONS),
        "claim_scope": {
            "exact_matrix_checks": [
                "Clifford relations for Gaussian-integer gamma systems",
                "Hermitian gamma and skew-Hermitian spin generators",
                "spin-vector covariance",
                "chirality and recursive 8-to-12 branching",
                "abstract A5 permutation presentation and order",
            ],
            "float64_algebraic_checks": [
                "binary-icosahedral quaternion enumeration",
                "2.A5 lifted relators and A5 vector relators",
                "Spin(3)-isotypic character and central-sign residuals",
            ],
            "not_claimed": [
                "triality beyond Spin(8)",
                "a trained sequence model",
                "an SSM performance advantage",
                "a derived-algebraic or representation-scheme theorem",
            ],
        },
        "stages": stage_reports,
        "branching": branching,
        "a5": a5,
        "passed": bool(passed),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
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
