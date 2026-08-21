"""Bounded exact probe for the unresolved monomial--golden Spin(8) closure.

The repository already certifies two finite systems in a fixed eight-dimensional
carrier: the 21,504-element signed-monomial normalizer of the octonion left
operators, and a golden-field binary-icosahedral Spin(3) embedding.  This
module investigates their *joint* closure.  A bounded exact word search finds
a candidate and a separate exact cyclotomic-factor gate can prove that the
joint closure is infinite; neither step attempts an abstract classification.

All state multiplication and equality in the bounded scan use Q(sqrt(5))
exactly.  Numerical eigenvalues are only a triage screen, and every reported
order is verified by exact matrix multiplication.  The default generator
system is deliberately small: the nine normalizer generators together with
the vector a,b pair.  It is the first fixed-basis gate before mixing all three
triality views.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

from octonion_operator_groups import (
    _automorphism_group_certificate,
    _normalizer_certificate,
    _operator_group_certificate,
    matrix_from_monomial,
)
from spin8_triality_2a5_closure import FIELD, _spin8_action_matrices

Pair = tuple[Fraction, Fraction]
Matrix = tuple[tuple[Pair, ...], ...]
Vector = tuple[Pair, ...]

SQRT5 = sp.sqrt(5)
ZERO: Pair = (Fraction(0), Fraction(0))
ONE: Pair = (Fraction(1), Fraction(0))


def _json_default(value: object) -> object:
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _pair(value: Any) -> Pair:
    """Convert an exact Q(sqrt(5)) value to a canonical rational pair."""

    if isinstance(value, (int, np.integer)):
        return (Fraction(int(value)), Fraction(0))
    polynomial = sp.Poly(sp.expand(FIELD.to_sympy(value)), SQRT5, domain=sp.QQ)
    if polynomial.degree() > 1:
        raise ValueError("value is not in Q(sqrt(5))")
    constant = sp.Rational(polynomial.nth(0))
    radical = sp.Rational(polynomial.nth(1))
    return (
        Fraction(int(constant.p), int(constant.q)),
        Fraction(int(radical.p), int(radical.q)),
    )


def _add(left: Pair, right: Pair) -> Pair:
    return (left[0] + right[0], left[1] + right[1])


def _subtract(left: Pair, right: Pair) -> Pair:
    return (left[0] - right[0], left[1] - right[1])


def _multiply(left: Pair, right: Pair) -> Pair:
    return (
        left[0] * right[0] + 5 * left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _divide(left: Pair, right: Pair) -> Pair:
    denominator = right[0] * right[0] - 5 * right[1] * right[1]
    if denominator == 0:
        raise ZeroDivisionError("zero Q(sqrt(5)) pivot")
    return (
        (left[0] * right[0] - 5 * left[1] * right[1]) / denominator,
        (left[1] * right[0] - left[0] * right[1]) / denominator,
    )


def _identity(dimension: int = 8) -> Matrix:
    return tuple(
        tuple(ONE if row == column else ZERO for column in range(dimension))
        for row in range(dimension)
    )


def _matrix(values: Sequence[Sequence[Any]]) -> Matrix:
    return tuple(tuple(_pair(value) for value in row) for row in values)


def _product(left: Matrix, right: Matrix) -> Matrix:
    dimension = len(left)
    if dimension != len(right) or any(len(row) != dimension for row in (*left, *right)):
        raise ValueError("matrix product requires equal square matrices")
    return tuple(
        tuple(
            _sum_products(
                (left[row][index], right[index][column])
                for index in range(dimension)
            )
            for column in range(dimension)
        )
        for row in range(dimension)
    )


def _sum_products(products: Sequence[Pair] | Any) -> Pair:
    total = ZERO
    for left, right in products:
        total = _add(total, _multiply(left, right))
    return total


def _apply(matrix: Matrix, vector: Vector) -> Vector:
    return tuple(
        _sum_products((matrix[row][column], vector[column]) for column in range(len(vector)))
        for row in range(len(matrix))
    )


def _matrix_to_float(matrix: Matrix) -> np.ndarray:
    radical = float(SQRT5.evalf(30))
    return np.asarray(
        [[float(value[0]) + radical * float(value[1]) for value in row] for row in matrix],
        dtype=np.float64,
    )


def _sympy_matrix(matrix: Matrix) -> sp.Matrix:
    return sp.Matrix(
        [
            [sp.Rational(value[0].numerator, value[0].denominator) + sp.Rational(value[1].numerator, value[1].denominator) * SQRT5 for value in row]
            for row in matrix
        ]
    )


def _cyclotomic_factor_gate(matrix: Matrix) -> dict[str, object]:
    """Exactly decide whether a selected 8D characteristic polynomial is cyclotomic.

    A root of this polynomial has degree at most eight over Q(sqrt(5)), hence
    degree at most sixteen over Q.  A root of unity with that degree must occur
    among the listed cyclotomic orders (the complete phi(n) <= 16 catalogue).
    """

    variable = sp.Symbol("x")
    characteristic = sp.Poly(
        _sympy_matrix(matrix).charpoly(variable).as_expr(),
        variable,
        extension=SQRT5,
    )
    factors = sp.factor_list(characteristic.as_expr(), variable, extension=SQRT5)[1]
    allowed_orders = (
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        20, 21, 22, 24, 26, 28, 30, 32, 34, 36, 40, 42, 48, 60,
    )
    cyclotomic = {
        order: sp.Poly(
            sp.cyclotomic_poly(order, variable), variable, extension=SQRT5
        )
        for order in allowed_orders
    }
    factor_payload: list[dict[str, object]] = []
    non_cyclotomic: list[str] = []
    for factor, multiplicity in factors:
        factor_poly = sp.Poly(factor, variable, extension=SQRT5)
        matching_orders = [
            order
            for order, polynomial in cyclotomic.items()
            if polynomial.rem(factor_poly).is_zero
        ]
        rendered = str(sp.factor(factor, extension=SQRT5))
        factor_payload.append(
            {
                "factor": rendered,
                "multiplicity": int(multiplicity),
                "root_of_unity_orders_if_any": matching_orders,
            }
        )
        if not matching_orders:
            non_cyclotomic.append(rendered)
    return {
        "method": "exact characteristic-polynomial factorization over Q(sqrt(5)) against every cyclotomic factor with phi(n) <= 16",
        "characteristic_polynomial": str(sp.factor(characteristic.as_expr(), extension=SQRT5)),
        "factors": factor_payload,
        "has_non_cyclotomic_factor": bool(non_cyclotomic),
        "non_cyclotomic_factors": non_cyclotomic,
    }


def _exact_order(matrix: Matrix, maximum_order: int) -> int | None:
    power = _identity(len(matrix))
    identity = power
    for order in range(1, maximum_order + 1):
        power = _product(power, matrix)
        if power == identity:
            return order
    return None


def _numerical_root_of_unity_screen(
    matrix: Matrix, *, order_cap: int, tolerance: float
) -> dict[str, object]:
    """A diagnostic screen only; it is never used as a proof of finite order."""

    eigenvalues = np.linalg.eigvals(_matrix_to_float(matrix))
    radius_error = max(abs(abs(value) - 1.0) for value in eigenvalues)
    powers = [
        max(abs(value**order - 1.0) for value in eigenvalues)
        for order in range(1, order_cap + 1)
    ]
    best_order = min(range(1, order_cap + 1), key=lambda order: powers[order - 1])
    return {
        "max_modulus_error": radius_error,
        "best_order_at_or_below_cap": best_order,
        "best_max_eigenvalue_power_residual": powers[best_order - 1],
        "screen_passes": radius_error <= tolerance and powers[best_order - 1] <= tolerance,
    }


def _vector_key(vector: Vector) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        (value[0].numerator, value[0].denominator, value[1].numerator, value[1].denominator)
        for value in vector
    )


def _rank_over_quadratic_field(rows: Sequence[Sequence[Pair]]) -> int:
    """Exact row rank over Q(sqrt(5)), using fraction-pair elimination."""

    work = [list(row) for row in rows]
    if not work:
        return 0
    pivot_row = 0
    for column in range(len(work[0])):
        pivot = next(
            (row for row in range(pivot_row, len(work)) if work[row][column] != ZERO),
            None,
        )
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        pivot_value = work[pivot_row][column]
        work[pivot_row] = [_divide(value, pivot_value) for value in work[pivot_row]]
        for row in range(len(work)):
            if row == pivot_row or work[row][column] == ZERO:
                continue
            factor = work[row][column]
            work[row] = [
                _subtract(value, _multiply(factor, pivot_value))
                for value, pivot_value in zip(work[row], work[pivot_row], strict=True)
            ]
        pivot_row += 1
        if pivot_row == len(work):
            break
    return pivot_row


def _commutant_dimension(generators: Sequence[Matrix]) -> int:
    """Dimension of {X : XG = GX for every stated generator} over Q(sqrt(5))."""

    dimension = len(generators[0])
    rows: list[list[Pair]] = []
    for generator in generators:
        for row in range(dimension):
            for column in range(dimension):
                equation = [ZERO for _ in range(dimension * dimension)]
                for inner in range(dimension):
                    equation[row * dimension + inner] = _add(
                        equation[row * dimension + inner], generator[inner][column]
                    )
                    equation[inner * dimension + column] = _subtract(
                        equation[inner * dimension + column], generator[row][inner]
                    )
                rows.append(equation)
    return dimension * dimension - _rank_over_quadratic_field(rows)


def _bounded_orbit(
    generators: Sequence[Matrix], seed: Vector, maximum_size: int
) -> dict[str, object]:
    reached = {_vector_key(seed): seed}
    frontier = [seed]
    cursor = 0
    while cursor < len(frontier) and len(frontier) < maximum_size:
        current = frontier[cursor]
        cursor += 1
        for generator in generators:
            candidate = _apply(generator, current)
            key = _vector_key(candidate)
            if key in reached:
                continue
            reached[key] = candidate
            frontier.append(candidate)
            if len(frontier) >= maximum_size:
                break
    return {
        "reached": len(frontier),
        "cap": maximum_size,
        "closed_before_cap": cursor == len(frontier),
    }


def _mixed_generators() -> tuple[dict[str, object], tuple[str, ...], tuple[Matrix, ...]]:
    operator_report, operator_group, operator_generators = _operator_group_certificate()
    automorphism_report, automorphism_group, automorphism_generators = (
        _automorphism_group_certificate()
    )
    normalizer_report = _normalizer_certificate(
        operator_report,
        operator_group,
        operator_generators,
        automorphism_group,
        automorphism_generators,
    )
    golden = _spin8_action_matrices()
    monomial_names = tuple(
        [f"left_operator_e{index}" for index in range(1, 8)]
        + ["fano_automorphism_a", "fano_automorphism_b"]
    )
    names = (*monomial_names, "golden_vector_a", "golden_vector_b")
    matrices = tuple(
        _matrix(matrix_from_monomial(generator))
        for generator in (*operator_generators, *automorphism_generators)
    ) + (_matrix(golden[0]), _matrix(golden[1]))
    return (
        {
            "monomial_normalizer_order": normalizer_report["order"],
            "monomial_normalizer_structure": normalizer_report["structure"],
            "monomial_normalizer_passed": normalizer_report["passed"],
            "binary_icosahedral_field": "Q(sqrt(5))",
            "golden_pair_relations": "a^2=b^3=(ab)^5=1 in vector view",
        },
        names,
        matrices,
    )


def certificate(
    *, max_word_length: int = 3, order_cap: int = 1000, orbit_cap: int = 2048
) -> dict[str, object]:
    """Return a reproducible, intentionally bounded mixed-closure report."""

    if max_word_length < 1 or order_cap < 1 or orbit_cap < 2:
        raise ValueError("all bounds must be positive, and orbit_cap must exceed one")
    components, names, generators = _mixed_generators()
    left_operator_commutant_dimension = _commutant_dimension(generators[:7])
    mixed_commutant_dimension = _commutant_dimension(generators)
    identity = _identity()
    reached = {identity}
    word_matrices = {"I": identity}
    frontier: list[tuple[Matrix, str]] = [(identity, "I")]
    reached_by_depth: list[int] = []
    screen_failures: list[dict[str, object]] = []
    maximum_radius_error = 0.0
    for depth in range(1, max_word_length + 1):
        next_frontier: list[tuple[Matrix, str]] = []
        for current, current_word in frontier:
            for generator_index, generator in enumerate(generators):
                candidate = _product(generator, current)
                if candidate in reached:
                    continue
                reached.add(candidate)
                word = names[generator_index] if current_word == "I" else f"{names[generator_index]}*({current_word})"
                word_matrices[word] = candidate
                next_frontier.append((candidate, word))
                screen = _numerical_root_of_unity_screen(
                    candidate, order_cap=order_cap, tolerance=1e-8
                )
                maximum_radius_error = max(
                    maximum_radius_error, float(screen["max_modulus_error"])
                )
                if not bool(screen["screen_passes"]) and len(screen_failures) < 16:
                    screen_failures.append(
                        {
                            "depth": depth,
                            "word": word,
                            "left_generator": names[generator_index],
                            **screen,
                        }
                    )
        frontier = next_frontier
        reached_by_depth.append(len(reached))

    mixed_pair_orders: list[dict[str, object]] = []
    for monomial_name, monomial in zip(names[:9], generators[:9], strict=True):
        for golden_name, golden in zip(names[9:], generators[9:], strict=True):
            product = _product(monomial, golden)
            mixed_pair_orders.append(
                {
                    "word": f"{monomial_name}*{golden_name}",
                    "exact_order_if_at_most_cap": _exact_order(product, order_cap),
                }
            )

    exact_cyclotomic_gate: dict[str, object] | None = None
    if screen_failures:
        first_word = str(screen_failures[0]["word"])
        # Recover the stored exact matrix without relying on a floating-point
        # eigensystem.  The word is generated deterministically above.
        candidate = word_matrices[first_word]
        exact_cyclotomic_gate = {"word": first_word, **_cyclotomic_factor_gate(candidate)}

    e0: Vector = (ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO)
    all_ones: Vector = (ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE)
    payload: dict[str, object] = {
        "schema_version": 1,
        "experiment": "bounded exact mixed monomial--golden Spin(8) closure probe",
        "generator_system": {
            "names": list(names),
            "word_semantics": "left-positive words in the stated 11 generators; inverses are not separately enumerated",
            "dimension": 8,
            "coefficient_field": "Q(sqrt(5))",
            "components": components,
        },
        "bounds": {
            "max_positive_word_length": max_word_length,
            "exact_mixed_pair_order_cap": order_cap,
            "candidate_orbit_cap": orbit_cap,
        },
        "exact_bounded_enumeration": {
            "unique_matrices_total": len(reached),
            "unique_matrices_total_by_depth": reached_by_depth,
        },
        "exact_representation_diagnostics": {
            "left_operator_commutant_dimension_over_qsqrt5": left_operator_commutant_dimension,
            "mixed_generator_commutant_dimension_over_qsqrt5": mixed_commutant_dimension,
            "interpretation": "dimension one means that only scalar endomorphisms commute with every stated generator over Q(sqrt(5))",
        },
        "numerical_spectral_triage": {
            "purpose": "diagnostic only; neither passing nor failing certifies finite or infinite closure",
            "screened_nonidentity_matrices": len(reached) - 1,
            "maximum_modulus_error": maximum_radius_error,
            "failures_retained_at_most": 16,
            "failures": screen_failures,
        },
        "exact_cyclotomic_factor_gate": exact_cyclotomic_gate,
        "exact_mixed_pair_orders": mixed_pair_orders,
        "candidate_orbits": {
            "e0": _bounded_orbit(generators, e0, orbit_cap),
            "all_ones": _bounded_orbit(generators, all_ones, orbit_cap),
        },
        "claim_scope": {
            "proved_exactly": [
                "every listed bounded word, equality comparison, reported exact pair order, and bounded orbit transition uses exact Q(sqrt(5)) arithmetic",
                "the two component systems are reconstructed through their existing exact certificates before the mixed probe runs",
                "the stated mixed representation has scalar commutant over Q(sqrt(5)), so it is irreducible over that field",
            ],
            "empirical_or_numerical": [
                "the eigenvalue screen is floating-point triage and is retained only to prioritize exact follow-up words",
            ],
            "open": [
                "the abstract isomorphism type, presentation, and Lie/topological closure of the infinite mixed subgroup",
                "which proper closed subgroup of SO(8), if any, contains the mixed subgroup",
                "a useful finite quotient or finite spanning configuration for a faithful permutation calculation",
            ],
            "not_claimed": [
                "a new group, a classification of the infinite subgroup, or density in SO(8)",
                "an ML performance or systems-kernel advantage",
            ],
        },
        "passed": bool(components["monomial_normalizer_passed"]),
    }
    if exact_cyclotomic_gate is not None and bool(
        exact_cyclotomic_gate["has_non_cyclotomic_factor"]
    ):
        payload["proved_conclusion"] = {
            "statement": "the subgroup generated by the stated monomial normalizer generators and golden vector a,b is infinite",
            "witness_word": exact_cyclotomic_gate["word"],
            "reason": "a non-cyclotomic characteristic-polynomial factor gives an eigenvalue that is not a root of unity; a finite-order matrix has only root-of-unity eigenvalues",
            "inheritance": "any enlarged mixed system containing these eleven generators, including the all-three-triality-view system, is also infinite",
        }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=_json_default
    )
    payload["certificate_sha256_without_self_hash"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-word-length", type=int, default=3)
    parser.add_argument("--order-cap", type=int, default=1000)
    parser.add_argument("--orbit-cap", type=int, default=2048)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    report = certificate(
        max_word_length=arguments.max_word_length,
        order_cap=arguments.order_cap,
        orbit_cap=arguments.orbit_cap,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
