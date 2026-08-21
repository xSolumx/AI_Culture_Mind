"""Exact infinitude and SO(8)-density of the mixed monomial/golden groups.

The maintained octonion-operator normalizer is a finite signed-monomial
subgroup of SO(8).  The maintained binary-icosahedral embedding supplies a
golden-field generator in each of the vector and two half-spin views.  This
module places those matrices in the same fixed basis and proves that every one
of the three mixed groups is topologically dense in SO(8).

The proof is deliberately arithmetic.  If a matrix has finite order, its
eigenvalues are roots of unity, so every coefficient of its characteristic
polynomial is an algebraic integer.  The three-letter word ``FanoA FanoB b``
has a characteristic coefficient outside the ring of integers
Z[(1 + sqrt(5))/2] in all three views.  Its rational characteristic norm also
has nonintegral coefficients and therefore cannot be a product of cyclotomic
polynomials.

Density follows from an exact adjoint-representation certificate.  Under the
monomial group, so(8) splits as irreducible dimensions 7 and 21.  The first is
not a Lie subalgebra, the second is the spin(7) Lie algebra, and the golden
generator fails to normalize the latter in every view.  Infinitude makes the
identity component nonzero, leaving all of so(8) as the only possible Lie
algebra of the compact closure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TypeVar

import sympy as sp
from sympy import ZZ
from sympy.polys.matrices import DomainMatrix

import octonion_operator_groups as monomial
import spin8_triality_2a5_closure as golden
from spin8_triality import octonion_left_multiplication

VIEW_GENERATOR_INDICES = {
    "vector": (0, 1),
    "positive_half_spin": (2, 3),
    "negative_half_spin": (4, 5),
}
WITNESS_GOLDEN_LETTER = "b"
MAXIMUM_LENGTH_TWO_ORDER = 120
T = TypeVar("T")


def _unique(values: Iterable[T]) -> tuple[T, ...]:
    return tuple(dict.fromkeys(values))


def _field_matrix_from_monomial(element: monomial.Monomial) -> golden.Matrix8:
    matrix = monomial.matrix_from_monomial(element)
    return tuple(
        tuple(golden._field(int(value)) for value in row) for row in matrix
    )


def _sympy_matrix(matrix: golden.Matrix8) -> sp.Matrix:
    return sp.Matrix(
        [
            [golden.FIELD.to_sympy(value) for value in row]
            for row in matrix
        ]
    )


def _matrix_order(matrix: golden.Matrix8, maximum: int) -> int | None:
    identity = golden._identity_matrix(8)
    power = identity
    for order in range(1, maximum + 1):
        power = golden._matrix_product(power, matrix)
        if power == identity:
            return order
    return None


def _matrix_closure_order(
    generators: Sequence[golden.Matrix8],
    maximum: int,
) -> int:
    identity = golden._identity_matrix(8)
    seen = {identity}
    frontier = [identity]
    while frontier:
        current = frontier.pop()
        for generator in generators:
            candidate = golden._matrix_product(current, generator)
            if candidate in seen:
                continue
            seen.add(candidate)
            frontier.append(candidate)
            if len(seen) > maximum:
                raise RuntimeError("golden source subgroup exceeded its exact cap")
    return len(seen)


def _quadratic_coefficients(value: sp.Expr) -> tuple[sp.Rational, sp.Rational]:
    """Return ``a, b`` for ``value = a + b*sqrt(5)``."""

    expanded = sp.expand(value)
    b = sp.Rational(expanded.coeff(sp.sqrt(5)))
    a = sp.Rational(sp.expand(expanded - b * sp.sqrt(5)))
    return a, b


def _is_quadratic_integer(value: sp.Expr) -> bool:
    """Test membership in the full integer ring of Q(sqrt(5))."""

    a, b = _quadratic_coefficients(value)
    twice_a = 2 * a
    twice_b = 2 * b
    return bool(
        twice_a.q == 1
        and twice_b.q == 1
        and (int(twice_a) - int(twice_b)) % 2 == 0
    )


def _characteristic_certificate(matrix: golden.Matrix8) -> dict[str, object]:
    variable = sp.symbols("x")
    expression = sp.expand(_sympy_matrix(matrix).charpoly(variable).as_expr())
    polynomial = sp.Poly(expression, variable, extension=sp.sqrt(5))
    coefficients = tuple(sp.simplify(value) for value in polynomial.all_coeffs())
    nonintegral_indices = [
        index
        for index, coefficient in enumerate(coefficients)
        if not _is_quadratic_integer(coefficient)
    ]

    conjugate = expression.xreplace({sp.sqrt(5): -sp.sqrt(5)})
    norm = sp.Poly(
        sp.expand(expression * conjugate),
        variable,
        domain=sp.QQ,
    ).monic()
    norm_coefficients = tuple(sp.Rational(value) for value in norm.all_coeffs())
    norm_denominators = sorted({int(value.q) for value in norm_coefficients})

    return {
        "characteristic_coefficients_descending": [
            str(value) for value in coefficients
        ],
        "nonintegral_coefficient_indices": nonintegral_indices,
        "first_nonintegral_coefficient": (
            str(coefficients[nonintegral_indices[0]])
            if nonintegral_indices
            else None
        ),
        "characteristic_norm_coefficients_descending": [
            str(value) for value in norm_coefficients
        ],
        "characteristic_norm_coefficient_denominators": norm_denominators,
        "checks": {
            "characteristic_polynomial_is_monic": polynomial.LC() == 1,
            "characteristic_polynomial_is_reciprocal": (
                coefficients == tuple(reversed(coefficients))
            ),
            "has_nonintegral_quadratic_field_coefficient": bool(
                nonintegral_indices
            ),
            "rational_characteristic_norm_is_not_integral": any(
                value != 1 for value in norm_denominators
            ),
        },
    }


def _symmetric_monomial_steps(
    generators: Sequence[monomial.Monomial],
) -> tuple[monomial.Monomial, ...]:
    return _unique(
        step
        for generator in generators
        for step in (generator, monomial.inverse(generator))
    )


def _symmetric_golden_steps(
    generators: Sequence[golden.Matrix8],
) -> tuple[golden.Matrix8, ...]:
    return _unique(
        step
        for generator in generators
        for step in (generator, golden._transpose(generator))
    )


def _length_two_certificate(
    monomial_steps: Sequence[monomial.Monomial],
    golden_steps: Sequence[golden.Matrix8],
) -> dict[str, object]:
    distribution: Counter[int | None] = Counter()
    for monomial_step in monomial_steps:
        left = _field_matrix_from_monomial(monomial_step)
        for golden_step in golden_steps:
            product = golden._matrix_product(left, golden_step)
            distribution[_matrix_order(product, MAXIMUM_LENGTH_TWO_ORDER)] += 1

    failures = int(distribution.pop(None, 0))
    return {
        "monomial_symmetric_generator_count": len(monomial_steps),
        "golden_symmetric_generator_count": len(golden_steps),
        "mixed_products_checked": len(monomial_steps) * len(golden_steps),
        "exact_order_distribution": {
            str(order): count for order, count in sorted(distribution.items())
        },
        "products_without_order_at_most_120": failures,
        "checks": {
            "every_symmetric_length_two_mixed_word_has_exact_finite_order": (
                failures == 0
            ),
        },
    }


def _orthogonal(matrix: golden.Matrix8) -> bool:
    return golden._matrix_product(golden._transpose(matrix), matrix) == (
        golden._identity_matrix(8)
    )


def _centralizer_certificate(matrices: Sequence[sp.Matrix]) -> dict[str, int]:
    dimension = matrices[0].rows
    identity = sp.eye(dimension)
    equations = sp.Matrix.vstack(
        *(
            sp.kronecker_product(identity, matrix)
            - sp.kronecker_product(matrix.T, identity)
            for matrix in matrices
        )
    )
    rank = DomainMatrix.from_Matrix(equations).convert_to(ZZ).rank()
    return {
        "representation_dimension": dimension,
        "equation_rows": equations.rows,
        "equation_columns": equations.cols,
        "equation_rank": rank,
        "centralizer_dimension": dimension * dimension - rank,
    }


def _clifford_adjoint_certificate(
    normalizer_generators: Sequence[monomial.Monomial],
    triality_matrices: Sequence[golden.Matrix8],
) -> dict[str, object]:
    rho = [
        sp.Matrix(matrix.astype(int).tolist())
        for matrix in octonion_left_multiplication()
    ]
    grade_one = rho[1:]
    grade_two_labels = [
        (left + 1, right + 1)
        for left in range(7)
        for right in range(left + 1, 7)
    ]
    grade_two = [
        grade_one[left - 1] * grade_one[right - 1]
        for left, right in grade_two_labels
    ]
    basis = [*grade_one, *grade_two]

    grade_one_rank = sp.Matrix.hstack(
        *(matrix.reshape(64, 1) for matrix in grade_one)
    ).rank()
    grade_two_rank = sp.Matrix.hstack(
        *(matrix.reshape(64, 1) for matrix in grade_two)
    ).rank()
    union_rank = sp.Matrix.hstack(
        *(matrix.reshape(64, 1) for matrix in basis)
    ).rank()
    summands_are_orthogonal = all(
        (left.T * right).trace() == 0
        for left in grade_one
        for right in grade_two
    )

    def coordinates(matrix: sp.Matrix) -> list[int]:
        result: list[int] = []
        for target in basis:
            coefficient = sp.Rational((target.T * matrix).trace(), 8)
            if coefficient.q != 1:
                raise AssertionError("adjoint coordinate is not integral")
            result.append(int(coefficient))
        return result

    adjoint_generators: list[sp.Matrix] = []
    splitting_is_invariant = True
    for element in normalizer_generators:
        matrix = sp.Matrix(monomial.matrix_from_monomial(element))
        columns = [coordinates(matrix * source * matrix.T) for source in basis]
        adjoint = sp.Matrix.hstack(
            *(sp.Matrix(column) for column in columns)
        )
        splitting_is_invariant &= (
            adjoint[:7, 7:] == sp.zeros(7, 21)
            and adjoint[7:, :7] == sp.zeros(21, 7)
        )
        adjoint_generators.append(adjoint)

    grade_one_adjoint = [matrix[:7, :7] for matrix in adjoint_generators]
    grade_two_adjoint = [matrix[7:, 7:] for matrix in adjoint_generators]
    grade_one_centralizer = _centralizer_certificate(grade_one_adjoint)
    grade_two_centralizer = _centralizer_certificate(grade_two_adjoint)
    full_centralizer = _centralizer_certificate(adjoint_generators)

    grade_one_brackets = [
        left * right - right * left
        for index, left in enumerate(grade_one)
        for right in grade_one[index + 1 :]
    ]
    grade_one_bracket_rank = sp.Matrix.hstack(
        *(matrix.reshape(64, 1) for matrix in grade_one_brackets)
    ).rank()
    grade_two_is_lie_subalgebra = all(
        (target.T * (left * right - right * left)).trace() == 0
        for left in grade_two
        for right in grade_two
        for target in grade_one
    )

    view_leaks: dict[str, object] = {}
    for view, indices in VIEW_GENERATOR_INDICES.items():
        golden_b = _sympy_matrix(triality_matrices[indices[1]])
        leaks: list[tuple[tuple[int, int], int, sp.Expr]] = []
        for label, source in zip(grade_two_labels, grade_two, strict=True):
            conjugate = golden_b * source * golden_b.T
            for target_index, target in enumerate(grade_one, start=1):
                coefficient = sp.simplify((target.T * conjugate).trace() / 8)
                if coefficient != 0:
                    leaks.append((label, target_index, coefficient))
        first = leaks[0] if leaks else None
        view_leaks[view] = {
            "nonzero_grade_two_to_grade_one_coefficients": len(leaks),
            "first_leak": (
                {
                    "source_grade_two_pair": list(first[0]),
                    "target_grade_one_index": first[1],
                    "coefficient": str(first[2]),
                }
                if first is not None
                else None
            ),
            "golden_b_does_not_normalize_grade_two": bool(leaks),
        }

    checks = {
        "grade_one_rank_is_7": grade_one_rank == 7,
        "grade_two_rank_is_21": grade_two_rank == 21,
        "union_spans_so8": union_rank == 28,
        "summands_are_frobenius_orthogonal": summands_are_orthogonal,
        "monomial_adjoint_preserves_7_plus_21": splitting_is_invariant,
        "grade_one_is_real_irreducible": (
            grade_one_centralizer["centralizer_dimension"] == 1
        ),
        "grade_two_is_real_irreducible": (
            grade_two_centralizer["centralizer_dimension"] == 1
        ),
        "full_adjoint_commutant_dimension_is_2": (
            full_centralizer["centralizer_dimension"] == 2
        ),
        "grade_one_brackets_span_grade_two": grade_one_bracket_rank == 21,
        "grade_one_is_not_a_lie_subalgebra": (
            grade_one_bracket_rank == 21 and summands_are_orthogonal
        ),
        "grade_two_is_a_lie_subalgebra": grade_two_is_lie_subalgebra,
        "golden_b_breaks_grade_two_in_every_view": all(
            report["golden_b_does_not_normalize_grade_two"]
            for report in view_leaks.values()
        ),
    }
    return {
        "decomposition": "so(8) = V7 + V21",
        "grade_one_dimension": grade_one_rank,
        "grade_two_dimension": grade_two_rank,
        "union_dimension": union_rank,
        "grade_one_centralizer": grade_one_centralizer,
        "grade_two_centralizer": grade_two_centralizer,
        "full_centralizer": full_centralizer,
        "grade_one_bracket_span_dimension": grade_one_bracket_rank,
        "view_normalizer_failures": view_leaks,
        "checks": checks,
        "passed": all(checks.values()),
    }


def certificate() -> dict[str, object]:
    """Return the exact infinitude and SO(8)-density certificate."""

    operator_report, _, operator_generators = (
        monomial._operator_group_certificate()
    )
    automorphism_report, _, automorphism_generators = (
        monomial._automorphism_group_certificate()
    )
    normalizer_generators = (*operator_generators, *automorphism_generators)
    normalizer = monomial.closure(
        normalizer_generators,
        maximum_order=21_504,
    )
    triality_matrices = golden._spin8_action_matrices()
    adjoint_report = _clifford_adjoint_certificate(
        normalizer_generators,
        triality_matrices,
    )

    if len(automorphism_generators) != 2:
        raise AssertionError("the maintained signed-Fano pair is not binary")
    bridge = monomial.compose(
        automorphism_generators[0], automorphism_generators[1]
    )
    bridge_matrix = _field_matrix_from_monomial(bridge)
    monomial_steps = _symmetric_monomial_steps(normalizer_generators)

    view_reports: dict[str, object] = {}
    for view, indices in VIEW_GENERATOR_INDICES.items():
        pair = tuple(triality_matrices[index] for index in indices)
        expected_source_order = 60 if view == "vector" else 120
        source_order = _matrix_closure_order(pair, expected_source_order)
        symmetric_golden_steps = _symmetric_golden_steps(pair)
        length_two = _length_two_certificate(
            monomial_steps,
            symmetric_golden_steps,
        )
        witness = golden._matrix_product(bridge_matrix, pair[1])
        characteristic = _characteristic_certificate(witness)
        witness_checks = {
            "witness_is_exactly_orthogonal": _orthogonal(witness),
            "witness_has_determinant_one": golden._determinant_is_one(witness),
            "nonintegral_characteristic_coefficient_proves_infinite_order": (
                bool(
                    characteristic["checks"][
                        "has_nonintegral_quadratic_field_coefficient"
                    ]
                )
            ),
            "noncyclotomic_characteristic_norm_proves_infinite_order": (
                bool(
                    characteristic["checks"][
                        "rational_characteristic_norm_is_not_integral"
                    ]
                )
            ),
        }
        dense_in_so8 = bool(
            all(witness_checks.values())
            and adjoint_report["passed"]
            and adjoint_report["view_normalizer_failures"][view][
                "golden_b_does_not_normalize_grade_two"
            ]
        )
        view_reports[view] = {
            "source_subgroup": "A5" if view == "vector" else "2.A5",
            "source_subgroup_order": source_order,
            "source_subgroup_order_is_expected": (
                source_order == expected_source_order
            ),
            "length_two_screen": length_two,
            "infinite_order_witness": {
                "word": ["FanoA", "FanoB", WITNESS_GOLDEN_LETTER],
                "word_length": 3,
                "monomial_prefix": list(bridge),
                "characteristic": characteristic,
                "checks": witness_checks,
            },
            "mixed_closure_is_infinite": all(witness_checks.values()),
            "topological_closure": "SO(8)" if dense_in_so8 else "unresolved",
            "mixed_group_is_topologically_dense_in_SO8": dense_in_so8,
        }

    all_length_two_pass = all(
        report["length_two_screen"]["checks"][
            "every_symmetric_length_two_mixed_word_has_exact_finite_order"
        ]
        for report in view_reports.values()
    )
    every_view_infinite = all(
        report["mixed_closure_is_infinite"] for report in view_reports.values()
    )
    every_view_dense = all(
        report["mixed_group_is_topologically_dense_in_SO8"]
        for report in view_reports.values()
    )
    source_orders_pass = all(
        report["source_subgroup_order_is_expected"]
        for report in view_reports.values()
    )
    checks = {
        "operator_group_input_passed": bool(operator_report["passed"]),
        "signed_fano_input_passed": bool(automorphism_report["passed"]),
        "monomial_normalizer_order_is_21504": len(normalizer) == 21_504,
        "normalizer_forward_generator_count_is_nine": (
            len(normalizer_generators) == 9
        ),
        "normalizer_symmetric_generator_count_is_seventeen": (
            len(monomial_steps) == 17
        ),
        "same_three_letter_word_is_infinite_in_every_triality_view": (
            every_view_infinite
        ),
        "clifford_adjoint_density_gate_passed": bool(adjoint_report["passed"]),
        "all_three_mixed_groups_are_dense_in_SO8": every_view_dense,
        "source_A5_and_2A5_orders_are_reconstructed_exactly": source_orders_pass,
        "length_three_is_minimal_in_the_symmetric_generator_alphabet": (
            all_length_two_pass and every_view_infinite
        ),
    }
    passed = all(checks.values())
    payload: dict[str, object] = {
        "schema_version": 2,
        "experiment": "mixed monomial/golden Spin(8) infinitude and density",
        "field": "Q(sqrt(5))",
        "integer_ring": "Z[(1+sqrt(5))/2]",
        "monomial_group": {
            "order": len(normalizer),
            "structure": "2_+^(1+6):PSL(2,7)",
            "forward_generators": [
                *[f"L{index}" for index in range(1, 8)],
                "FanoA",
                "FanoB",
            ],
            "signed_fano_generators": [
                list(value) for value in automorphism_generators
            ],
            "witness_monomial_prefix": list(bridge),
        },
        "proof": {
            "finite_order_implication": (
                "A finite-order characteristic-zero matrix has root-of-unity "
                "eigenvalues, hence algebraic-integer characteristic "
                "coefficients."
            ),
            "norm_implication": (
                "Its Q(sqrt(5))/Q characteristic norm must consequently be "
                "a monic integer product of cyclotomic polynomials."
            ),
            "witness_word": ["FanoA", "FanoB", "b"],
            "witness_word_length": 3,
            "density_implication": (
                "The closure of an infinite subgroup of compact SO(8) has "
                "nonzero identity Lie algebra. Monomial invariance restricts "
                "that algebra to V7, V21, or so(8); V7 is not bracket closed "
                "and golden b does not normalize V21, leaving so(8)."
            ),
        },
        "clifford_adjoint_density": adjoint_report,
        "views": view_reports,
        "checks": checks,
        "claim_scope": {
            "proved": [
                "the fixed monomial/vector-A5 mixed closure is infinite",
                "the fixed monomial/positive-2.A5 mixed closure is infinite",
                "the fixed monomial/negative-2.A5 mixed closure is infinite",
                "the displayed length-three witness is minimal in the maintained symmetric generator alphabet",
                "all three fixed mixed groups are topologically dense in SO(8)",
            ],
            "not_claimed": [
                "a quantitative spectral gap or equidistribution rate",
                "classification of every relative embedding of these abstract finite groups",
                "a new abstract finite or infinite group",
                "an ML-quality or kernel-speed advantage",
            ],
        },
        "passed": passed,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["certificate_sha256_without_self_hash"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    report = certificate()
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
