"""Exact Spin(9) Clifford diagnostics derived from the maintained Spin(8) core.

The eight gamma matrices already used to construct the two chiral Spin(8)
representations, together with their chirality product, form a symmetric
Clifford system on R^16.  This module makes that extension explicit without
changing the Spin(8) implementation or its claim boundary.

All theorem-facing algebra and rank witnesses use integer arithmetic.  The
floating-point helpers at the end are intended for later model experiments;
they are not part of the exact certificate.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spin8_triality import build_spin8_triality_algebra

SPIN9_VECTOR_DIM = 9
SPIN9_SPINOR_DIM = 16
SPIN9_LIE_DIM = 36
CERTIFICATE_PRIMES = (1_000_003, 1_000_033, 1_000_037)

# Frozen rational witnesses.  Scaling a nonzero probe does not change the
# stabilizer Jacobian rank, so unit normalization is unnecessary here.
SPINOR_WITNESSES = np.asarray(
    [
        [2, -2, 0, -2, -3, -2, 0, 1, -3, 0, 0, 0, -1, 3, -2, 1],
        [3, -2, -2, 1, -3, 0, 0, -3, -1, -1, 1, 1, 2, -1, 0, -1],
        [1, 1, 2, 1, 0, 2, 3, 1, -2, 1, 2, 3, 1, -1, 2, 0],
    ],
    dtype=np.int64,
)

VECTOR_WITNESSES = np.asarray(
    [
        [1, -3, -3, 3, 1, -1, 0, 2, 1],
        [0, 0, 3, 2, -2, 0, -1, 3, -3],
        [1, 2, 1, 0, 3, -1, 2, 2, 0],
        [2, -1, 3, -3, -3, 1, 3, 0, 0],
        [3, 2, 2, 0, 1, -1, -1, 3, 3],
        [3, -1, -1, 3, 3, 0, -1, -2, 2],
        [-3, -1, 1, -2, -1, -2, -3, 2, -3],
        [2, 3, -2, -3, -1, -2, -2, 2, -1],
    ],
    dtype=np.int64,
)


@dataclass(frozen=True)
class Spin9CliffordSystem:
    """The nine involutions and 36 integer-scaled spin generators."""

    involutions: np.ndarray
    generator_pairs: tuple[tuple[int, int], ...]
    doubled_spin_generators: np.ndarray
    vector_generators: np.ndarray


def _as_exact_integer(array: np.ndarray) -> np.ndarray:
    rounded = np.rint(array).astype(np.int64)
    if not np.array_equal(array, rounded):
        raise ValueError("the maintained Spin(8) construction is not integral")
    return rounded


def build_spin9_clifford_system() -> Spin9CliffordSystem:
    """Extend the maintained eight gamma matrices by chirality.

    ``doubled_spin_generators[i]`` is ``P_a P_b``.  It is twice the
    conventional infinitesimal generator ``(1/2) P_a P_b``; the integer scale
    is retained so that every certificate can be replayed without fractions.
    """

    spin8 = build_spin8_triality_algebra()
    involutions = np.concatenate(
        [_as_exact_integer(spin8.gamma), _as_exact_integer(spin8.chirality)[None]],
        axis=0,
    )
    pairs = tuple(
        (left, right)
        for left in range(SPIN9_VECTOR_DIM)
        for right in range(left + 1, SPIN9_VECTOR_DIM)
    )
    doubled = np.stack(
        [involutions[left] @ involutions[right] for left, right in pairs]
    )

    vector = []
    for left, right in pairs:
        matrix = np.zeros((SPIN9_VECTOR_DIM, SPIN9_VECTOR_DIM), dtype=np.int64)
        matrix[left, right] = 1
        matrix[right, left] = -1
        vector.append(matrix)

    return Spin9CliffordSystem(
        involutions=involutions,
        generator_pairs=pairs,
        doubled_spin_generators=doubled,
        vector_generators=np.stack(vector),
    )


def hurwitz_radon_number(dimension: int) -> int:
    """Return rho(n) for the Hurwitz--Radon square-composition theorem."""

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    exponent = 0
    odd_part = dimension
    while odd_part % 2 == 0:
        exponent += 1
        odd_part //= 2
    quotient, remainder = divmod(exponent, 4)
    return 8 * quotient + 2**remainder


def _modular_determinant(matrix: np.ndarray, prime: int) -> int:
    """Compute a square determinant over F_p by exact elimination."""

    work = np.remainder(np.asarray(matrix, dtype=np.int64), prime).copy()
    rows, columns = work.shape
    if rows != columns:
        raise ValueError("determinant requires a square matrix")
    determinant = 1
    for column in range(columns):
        candidates = np.flatnonzero(work[column:, column])
        if candidates.size == 0:
            return 0
        pivot = column + int(candidates[0])
        if pivot != column:
            work[[column, pivot]] = work[[pivot, column]]
            determinant = -determinant
        pivot_value = int(work[column, column])
        determinant = (determinant * pivot_value) % prime
        inverse = pow(pivot_value, -1, prime)
        work[column] = np.remainder(work[column] * inverse, prime)
        active_rows = np.flatnonzero(work[column + 1 :, column]) + column + 1
        if active_rows.size:
            factors = work[active_rows, column, None]
            work[active_rows] = np.remainder(
                work[active_rows] - factors * work[column], prime
            )
    return determinant % prime


def modular_pivot_certificate(matrix: np.ndarray, prime: int) -> dict[str, object]:
    """Return rank and an explicitly nonzero pivot minor over F_p."""

    if prime <= 2:
        raise ValueError("an odd prime is required")
    work = np.remainder(np.asarray(matrix, dtype=np.int64), prime).copy()
    rows, columns = work.shape
    row_labels = np.arange(rows)
    pivot_rows: list[int] = []
    pivot_columns: list[int] = []
    rank = 0
    for column in range(columns):
        candidates = np.flatnonzero(work[rank:, column])
        if candidates.size == 0:
            continue
        pivot = rank + int(candidates[0])
        if pivot != rank:
            work[[rank, pivot]] = work[[pivot, rank]]
            row_labels[[rank, pivot]] = row_labels[[pivot, rank]]
        pivot_rows.append(int(row_labels[rank]))
        pivot_columns.append(column)
        inverse = pow(int(work[rank, column]), -1, prime)
        work[rank] = np.remainder(work[rank] * inverse, prime)
        mask = np.arange(rows) != rank
        active_rows = np.flatnonzero(mask & (work[:, column] != 0))
        if active_rows.size:
            factors = work[active_rows, column, None]
            work[active_rows] = np.remainder(
                work[active_rows] - factors * work[rank], prime
            )
        rank += 1
        if rank == rows:
            break
    minor = np.asarray(matrix)[np.ix_(pivot_rows, pivot_columns)]
    determinant = _modular_determinant(minor, prime)
    if rank and determinant == 0:
        raise AssertionError("elimination returned a singular pivot minor")
    return {
        "rank": rank,
        "pivot_rows": pivot_rows,
        "pivot_columns": pivot_columns,
        "pivot_minor_determinant_mod_prime": determinant,
    }


def modular_rank(matrix: np.ndarray, prime: int) -> int:
    """Compute matrix rank exactly over the prime field F_p."""

    return int(modular_pivot_certificate(matrix, prime)["rank"])


def observation_matrix(
    generators: np.ndarray, probes: Sequence[np.ndarray]
) -> np.ndarray:
    """Stack the infinitesimal observations ``G_j x`` for each probe."""

    blocks = [
        np.stack([generator @ probe for generator in generators], axis=1)
        for probe in probes
    ]
    if not blocks:
        return np.zeros((0, generators.shape[0]), dtype=np.int64)
    return np.concatenate(blocks, axis=0)


def certified_ranks(matrix: np.ndarray) -> dict[str, int]:
    """Return exact lower-bound ranks over three independent prime fields."""

    return {str(prime): modular_rank(matrix, prime) for prime in CERTIFICATE_PRIMES}


def _quadratic_coefficients(matrix: np.ndarray) -> dict[tuple[int, ...], int]:
    """Coefficient map of x^T M x for a symmetric integer matrix."""

    dimension = matrix.shape[0]
    coefficients: dict[tuple[int, ...], int] = {}
    for left in range(dimension):
        diagonal = int(matrix[left, left])
        if diagonal:
            exponent = [0] * dimension
            exponent[left] = 2
            coefficients[tuple(exponent)] = diagonal
        for right in range(left + 1, dimension):
            value = int(matrix[left, right] + matrix[right, left])
            if value:
                exponent = [0] * dimension
                exponent[left] = exponent[right] = 1
                coefficients[tuple(exponent)] = value
    return coefficients


def _square_polynomial(
    coefficients: dict[tuple[int, ...], int],
) -> dict[tuple[int, ...], int]:
    result: defaultdict[tuple[int, ...], int] = defaultdict(int)
    items = tuple(coefficients.items())
    for left_exp, left_value in items:
        for right_exp, right_value in items:
            exponent = tuple(a + b for a, b in zip(left_exp, right_exp, strict=True))
            result[exponent] += left_value * right_value
    return dict(result)


def exact_hopf_residual(involutions: np.ndarray) -> dict[tuple[int, ...], int]:
    """Return coefficients of sum_i (s^T P_i s)^2 - (s^T s)^2."""

    residual: defaultdict[tuple[int, ...], int] = defaultdict(int)
    for involution in involutions:
        for exponent, value in _square_polynomial(
            _quadratic_coefficients(involution)
        ).items():
            residual[exponent] += value

    dimension = involutions.shape[-1]
    for left in range(dimension):
        exponent = [0] * dimension
        exponent[left] = 4
        residual[tuple(exponent)] -= 1
        for right in range(left + 1, dimension):
            exponent = [0] * dimension
            exponent[left] = exponent[right] = 2
            residual[tuple(exponent)] -= 2
    return {key: value for key, value in residual.items() if value}


def dirac_operator(address: np.ndarray, *, dtype=np.float64) -> np.ndarray:
    """Return D(a)=sum_i a_i P_i in the real spin representation."""

    system = build_spin9_clifford_system()
    address = np.asarray(address, dtype=dtype)
    if address.shape != (SPIN9_VECTOR_DIM,):
        raise ValueError(f"address must have shape ({SPIN9_VECTOR_DIM},)")
    return np.tensordot(address, system.involutions.astype(dtype), axes=1)


def even_spin_transition(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the even Clifford product D(second) D(first).

    Unit inputs give an orthogonal Spin(9) action.  A single ``D(a)`` is an odd
    Pin(9) element; pairing the factors is therefore a mathematical contract,
    not merely an implementation convenience.
    """

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm == 0.0 or second_norm == 0.0:
        raise ValueError("transition addresses must be nonzero")
    return dirac_operator(second / second_norm) @ dirac_operator(first / first_norm)


def diagnostics() -> dict[str, object]:
    """Replay the exact algebra, Hopf identity, and frozen rank witnesses."""

    system = build_spin9_clifford_system()
    identity = np.eye(SPIN9_SPINOR_DIM, dtype=np.int64)
    zero = np.zeros_like(identity)

    clifford_ok = True
    for left in range(SPIN9_VECTOR_DIM):
        for right in range(SPIN9_VECTOR_DIM):
            expected = 2 * identity if left == right else zero
            observed = (
                system.involutions[left] @ system.involutions[right]
                + system.involutions[right] @ system.involutions[left]
            )
            clifford_ok &= np.array_equal(observed, expected)

    commutator_ok = True
    for generator, (left, right) in zip(
        system.doubled_spin_generators, system.generator_pairs, strict=True
    ):
        for vector_index, involution in enumerate(system.involutions):
            expected = np.zeros_like(identity)
            if vector_index == right:
                expected += 2 * system.involutions[left]
            if vector_index == left:
                expected -= 2 * system.involutions[right]
            observed = generator @ involution - involution @ generator
            commutator_ok &= np.array_equal(observed, expected)

    spin8 = build_spin8_triality_algebra()
    positive = _as_exact_integer(2 * spin8.positive_generators)
    negative = _as_exact_integer(2 * spin8.negative_generators)
    spin8_indices = [
        index for index, (_, right) in enumerate(system.generator_pairs) if right < 8
    ]
    spin8_restriction_ok = np.array_equal(
        system.doubled_spin_generators[spin8_indices, :8, :8], positive
    ) and np.array_equal(
        system.doubled_spin_generators[spin8_indices, 8:, 8:], negative
    )

    spinor_rank_rows = []
    three_spinor_matrix = None
    for count in range(1, 4):
        matrix = observation_matrix(
            system.doubled_spin_generators, SPINOR_WITNESSES[:count]
        )
        if count == 3:
            three_spinor_matrix = matrix
        spinor_rank_rows.append(
            {"probe_count": count, "ranks_mod_prime": certified_ranks(matrix)}
        )

    vector_rank_rows = []
    for count in range(1, 9):
        matrix = observation_matrix(system.vector_generators, VECTOR_WITNESSES[:count])
        vector_rank_rows.append(
            {"probe_count": count, "ranks_mod_prime": certified_ranks(matrix)}
        )

    degenerate_third = SPINOR_WITNESSES[0] + 2 * SPINOR_WITNESSES[1]
    degenerate_ranks = certified_ranks(
        observation_matrix(
            system.doubled_spin_generators,
            [SPINOR_WITNESSES[0], SPINOR_WITNESSES[1], degenerate_third],
        )
    )

    expected_spinor = (15, 28, 36)
    spinor_ranks_ok = all(
        tuple(row["ranks_mod_prime"].values()) == (expected,) * 3
        for row, expected in zip(spinor_rank_rows, expected_spinor, strict=True)
    )
    expected_vector = (8, 15, 21, 26, 30, 33, 35, 36)
    vector_ranks_ok = all(
        tuple(row["ranks_mod_prime"].values()) == (expected,) * 3
        for row, expected in zip(vector_rank_rows, expected_vector, strict=True)
    )

    report: dict[str, object] = {
        "schema_version": 1,
        "claim_scope": {
            "exact": [
                "nine symmetric Clifford involutions on R^16",
                "Spin(8) chiral restriction",
                "Spin(9)-vector commutator law",
                "quadratic Hopf norm identity",
                "frozen rational probe ranks over three prime fields",
            ],
            "external_theorem": [
                "Hurwitz-Radon maximality rho(16)=9",
                "one-spinor stabilizer Spin(7)",
                "generic pair stabilizer SU(3)",
                "generic triple stabilizer is trivial",
            ],
            "not_claimed": [
                "mechanical global-stabilizer certificate for the frozen triple",
                "optimal conditioning of three probes",
                "machine-learning superiority",
            ],
        },
        "involution_shape": list(system.involutions.shape),
        "generator_shape": list(system.doubled_spin_generators.shape),
        "symmetric_involutions": bool(
            np.array_equal(system.involutions, system.involutions.transpose(0, 2, 1))
        ),
        "clifford_relations": bool(clifford_ok),
        "skew_generators": bool(
            np.array_equal(
                system.doubled_spin_generators,
                -system.doubled_spin_generators.transpose(0, 2, 1),
            )
        ),
        "vector_commutator_law": bool(commutator_ok),
        "spin8_chiral_restriction": bool(spin8_restriction_ok),
        "hopf_polynomial_nonzero_coefficients": len(
            exact_hopf_residual(system.involutions)
        ),
        "hurwitz_radon": {
            "rho_8": hurwitz_radon_number(8),
            "rho_16": hurwitz_radon_number(16),
        },
        "spinor_probe_ranks": spinor_rank_rows,
        "three_spinor_pivot_certificates": {
            str(prime): modular_pivot_certificate(three_spinor_matrix, prime)
            for prime in CERTIFICATE_PRIMES
        },
        "vector_probe_ranks": vector_rank_rows,
        "degenerate_third_spinor_ranks": degenerate_ranks,
    }
    report["passed"] = bool(
        report["symmetric_involutions"]
        and report["clifford_relations"]
        and report["skew_generators"]
        and report["vector_commutator_law"]
        and report["spin8_chiral_restriction"]
        and report["hopf_polynomial_nonzero_coefficients"] == 0
        and report["hurwitz_radon"] == {"rho_8": 8, "rho_16": 9}
        and spinor_ranks_ok
        and vector_ranks_ok
        and tuple(degenerate_ranks.values()) == (28, 28, 28)
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
