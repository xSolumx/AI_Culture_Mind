"""Exact low-degree mixing bounds for the dense monomial/golden groups.

The companion closure certificate proves that three fixed mixed subgroups are
dense in SO(8), but density alone supplies no rate.  This module takes the
uniform symmetric measure on the maintained generators and their inverses and
certifies strict contraction in three low-degree representations:

* the defining representation on R^8;
* the adjoint representation on exterior^2(R^8), dimension 28;
* the traceless symmetric-square representation, dimension 35.

All acceptance checks use exact arithmetic over Q(sqrt(5)).  The 8-dimensional
bounds use exact characteristic-polynomial root counts.  The 28- and
35-dimensional bounds use exact positive-definite LDL^T certificates for
``c I +/- M`` or, in the non-orthonormal traceless basis, ``c G +/- G M``.

These finite-representation bounds are not a proof of a spectral gap on the
full mean-zero L2(SO(8)), a total-variation mixing bound, or an ML advantage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar

import sympy as sp

import mixed_monomial_golden_closure as closure
import octonion_operator_groups as monomial
import spin8_triality_2a5_closure as golden

FieldElement = Any
FieldMatrix = tuple[tuple[FieldElement, ...], ...]
T = TypeVar("T")

VECTOR_RADIUS_BOUND = sp.Rational(7, 25)
HALF_SPIN_RADIUS = sp.Rational(4, 21)
ADJOINT_RADIUS_BOUNDS = {
    "vector": sp.Rational(5, 8),
    "positive_half_spin": sp.Rational(3, 4),
    "negative_half_spin": sp.Rational(3, 4),
}
SYMMETRIC_RADIUS_BOUNDS = {
    "vector": sp.Rational(1, 3),
    "positive_half_spin": sp.Rational(3, 8),
    "negative_half_spin": sp.Rational(3, 8),
}


def _unique(values: Iterable[T]) -> tuple[T, ...]:
    return tuple(dict.fromkeys(values))


def _field(value: sp.Expr | int) -> FieldElement:
    return golden.FIELD.from_sympy(sp.expand(value))


def _field_matrix_to_sympy(matrix: FieldMatrix) -> sp.Matrix:
    return sp.Matrix(
        [
            [golden.FIELD.to_sympy(value) for value in row]
            for row in matrix
        ]
    )


def _field_string(value: FieldElement) -> str:
    return str(sp.factor(golden.FIELD.to_sympy(value)))


def _matrix_strings(matrix: FieldMatrix) -> list[list[str]]:
    return [[_field_string(value) for value in row] for row in matrix]


def _matrix_hash(matrix: FieldMatrix) -> str:
    canonical = json.dumps(
        _matrix_strings(matrix), separators=(",", ":"), sort_keys=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _mean_matrix(matrices: Sequence[FieldMatrix]) -> FieldMatrix:
    if not matrices:
        raise ValueError("cannot average an empty matrix family")
    dimension = len(matrices[0])
    if any(
        len(matrix) != dimension
        or any(len(row) != dimension for row in matrix)
        for matrix in matrices
    ):
        raise ValueError("all averaged matrices must have one square dimension")
    inverse_count = _field(sp.Rational(1, len(matrices)))
    return tuple(
        tuple(
            sum(
                (matrix[row][column] for matrix in matrices),
                golden.FIELD.zero,
            )
            * inverse_count
            for column in range(dimension)
        )
        for row in range(dimension)
    )


def _wedge_square(matrix: FieldMatrix) -> FieldMatrix:
    """Return exterior-square coordinates in the increasing-pair basis."""

    dimension = len(matrix)
    pairs = tuple(
        (left, right)
        for left in range(dimension)
        for right in range(left + 1, dimension)
    )
    return tuple(
        tuple(
            matrix[target_left][source_left]
            * matrix[target_right][source_right]
            - matrix[target_left][source_right]
            * matrix[target_right][source_left]
            for source_left, source_right in pairs
        )
        for target_left, target_right in pairs
    )


def _traceless_symmetric_square(matrix: FieldMatrix) -> FieldMatrix:
    """Return Sym^2_0 coordinates in a fixed rational 35-vector basis.

    The first seven basis vectors are ``E_ii - E_77``.  The remaining 28 are
    ``E_ij + E_ji`` for increasing pairs.  For ``Y = Q X Q^T``, its first
    seven diagonal entries and upper-triangular off-diagonal entries are the
    coordinates in this basis.
    """

    dimension = len(matrix)
    if dimension != 8:
        raise ValueError("the maintained traceless-square basis expects R^8")
    pairs = tuple(
        (left, right)
        for left in range(dimension)
        for right in range(left + 1, dimension)
    )

    columns: list[tuple[FieldElement, ...]] = []
    for source in range(7):
        diagonal = tuple(
            matrix[target][source] * matrix[target][source]
            - matrix[target][7] * matrix[target][7]
            for target in range(7)
        )
        off_diagonal = tuple(
            matrix[left][source] * matrix[right][source]
            - matrix[left][7] * matrix[right][7]
            for left, right in pairs
        )
        columns.append((*diagonal, *off_diagonal))

    for source_left, source_right in pairs:
        diagonal = tuple(
            _field(2)
            * matrix[target][source_left]
            * matrix[target][source_right]
            for target in range(7)
        )
        off_diagonal = tuple(
            matrix[left][source_left] * matrix[right][source_right]
            + matrix[left][source_right] * matrix[right][source_left]
            for left, right in pairs
        )
        columns.append((*diagonal, *off_diagonal))

    return tuple(tuple(column[row] for column in columns) for row in range(35))


def _traceless_gram() -> FieldMatrix:
    """Return the Frobenius Gram matrix of the maintained 35-vector basis."""

    return tuple(
        tuple(
            _field(
                2
                if row == column
                else 1
                if row < 7 and column < 7
                else 0
            )
            for column in range(35)
        )
        for row in range(35)
    )


def _matrix_product(left: FieldMatrix, right: FieldMatrix) -> FieldMatrix:
    rows = len(left)
    inner = len(left[0])
    columns = len(right[0])
    if len(right) != inner:
        raise ValueError("matrix dimensions do not align")
    return tuple(
        tuple(
            sum(
                (
                    left[row][offset] * right[offset][column]
                    for offset in range(inner)
                ),
                golden.FIELD.zero,
            )
            for column in range(columns)
        )
        for row in range(rows)
    )


def _transpose(matrix: FieldMatrix) -> FieldMatrix:
    return tuple(tuple(value) for value in zip(*matrix, strict=True))


def _bounded_form(
    mean: FieldMatrix,
    bound: sp.Rational,
    gram: FieldMatrix | None = None,
    sign: int = 1,
) -> FieldMatrix:
    """Return ``cG + sign*G*M``, taking G=I when it is omitted."""

    dimension = len(mean)
    metric = gram or golden._identity_matrix(dimension)
    metric_mean = _matrix_product(metric, mean)
    field_bound = _field(bound)
    return tuple(
        tuple(
            field_bound * metric[row][column]
            + sign * metric_mean[row][column]
            for column in range(dimension)
        )
        for row in range(dimension)
    )


def _quadratic_parts(value: FieldElement) -> tuple[Any, Any]:
    """Return rational ``a, b`` with ``value = a + b*sqrt(5)``."""

    coefficients = list(value.rep)
    if not coefficients:
        return golden.FIELD.dom.zero, golden.FIELD.dom.zero
    if len(coefficients) == 1:
        return coefficients[0], golden.FIELD.dom.zero
    if len(coefficients) == 2:
        return coefficients[1], coefficients[0]
    raise AssertionError("unexpected degree in the quadratic field")


def _field_sign(value: FieldElement) -> int:
    """Return the exact real sign in the fixed positive sqrt(5) embedding."""

    a, b = _quadratic_parts(value)
    zero = golden.FIELD.dom.zero
    if b == zero:
        return (a > zero) - (a < zero)
    if a == zero:
        return (b > zero) - (b < zero)
    if a > zero and b > zero:
        return 1
    if a < zero and b < zero:
        return -1
    comparison = a * a - golden.FIELD.dom.convert(5) * b * b
    if comparison == zero:
        raise AssertionError("nonzero rational ratio cannot equal sqrt(5)")
    if a > zero:
        return 1 if comparison > zero else -1
    return -1 if comparison > zero else 1


def _pivot_hash(pivots: Sequence[FieldElement]) -> str:
    canonical = json.dumps(
        [_field_string(value) for value in pivots],
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _ldlt_positive_definite(matrix: FieldMatrix) -> dict[str, object]:
    """Certify positive definiteness by exact unpivoted LDL^T."""

    dimension = len(matrix)
    symmetric = matrix == _transpose(matrix)
    if not symmetric:
        return {
            "dimension": dimension,
            "matrix_is_symmetric": False,
            "pivot_count": 0,
            "all_pivots_strictly_positive": False,
            "pivot_sequence_sha256": None,
            "passed": False,
        }

    lower = [
        [golden.FIELD.zero for _ in range(dimension)]
        for _ in range(dimension)
    ]
    pivots: list[FieldElement] = []
    for column in range(dimension):
        pivot = matrix[column][column] - sum(
            (
                lower[column][offset]
                * lower[column][offset]
                * pivots[offset]
                for offset in range(column)
            ),
            golden.FIELD.zero,
        )
        if _field_sign(pivot) <= 0:
            return {
                "dimension": dimension,
                "matrix_is_symmetric": True,
                "pivot_count": len(pivots) + 1,
                "all_pivots_strictly_positive": False,
                "first_nonpositive_pivot_index": column,
                "first_nonpositive_pivot": _field_string(pivot),
                "pivot_sequence_sha256": None,
                "passed": False,
            }
        pivots.append(pivot)
        lower[column][column] = golden.FIELD.one
        for row in range(column + 1, dimension):
            numerator = matrix[row][column] - sum(
                (
                    lower[row][offset]
                    * lower[column][offset]
                    * pivots[offset]
                    for offset in range(column)
                ),
                golden.FIELD.zero,
            )
            lower[row][column] = numerator / pivot

    return {
        "dimension": dimension,
        "matrix_is_symmetric": True,
        "pivot_count": len(pivots),
        "all_pivots_strictly_positive": True,
        "pivot_sequence_sha256": _pivot_hash(pivots),
        "passed": True,
    }


def _ldlt_radius_certificate(
    mean: FieldMatrix,
    bound: sp.Rational,
    gram: FieldMatrix | None = None,
) -> dict[str, object]:
    metric = gram or golden._identity_matrix(len(mean))
    metric_mean = _matrix_product(metric, mean)
    self_adjoint = metric_mean == _transpose(metric_mean)
    plus = _ldlt_positive_definite(
        _bounded_form(mean, bound, gram=metric, sign=1)
    )
    minus = _ldlt_positive_definite(
        _bounded_form(mean, bound, gram=metric, sign=-1)
    )
    passed = bool(self_adjoint and plus["passed"] and minus["passed"])
    return {
        "dimension": len(mean),
        "radius_upper_bound": str(bound),
        "gap_lower_bound": str(1 - bound),
        "inequalities_are_strict": True,
        "metric": "identity" if gram is None else "traceless-basis Gram",
        "mean_matrix_sha256": _matrix_hash(mean),
        "mean_is_self_adjoint_in_metric": self_adjoint,
        "positive_definite_cG_plus_GM": plus,
        "positive_definite_cG_minus_GM": minus,
        "passed": passed,
    }


def _vector_defining_certificate(mean: FieldMatrix) -> dict[str, object]:
    variable = sp.symbols("x")
    expression = sp.expand(
        _field_matrix_to_sympy(mean).charpoly(variable).as_expr()
    )
    conjugate = expression.xreplace({sp.sqrt(5): -sp.sqrt(5)})
    norm = sp.Poly(sp.expand(expression * conjugate), variable, domain=sp.QQ)
    left = -VECTOR_RADIUS_BOUND
    right = VECTOR_RADIUS_BOUND
    roots_inside = int(sp.count_roots(norm, left, right))
    endpoints_nonzero = norm.eval(left) != 0 and norm.eval(right) != 0
    passed = bool(norm.degree() == 16 and roots_inside == 16 and endpoints_nonzero)
    return {
        "dimension": 8,
        "exact_mean_matrix": _matrix_strings(mean),
        "mean_matrix_sha256": _matrix_hash(mean),
        "characteristic_polynomial": str(sp.factor(expression)),
        "rational_characteristic_norm_coefficients_descending": [
            str(value) for value in norm.all_coeffs()
        ],
        "root_count_interval": [str(left), str(right)],
        "rational_norm_degree": norm.degree(),
        "rational_norm_roots_strictly_inside_interval": roots_inside,
        "interval_endpoints_are_not_roots": endpoints_nonzero,
        "radius_upper_bound": str(VECTOR_RADIUS_BOUND),
        "gap_lower_bound": str(1 - VECTOR_RADIUS_BOUND),
        "inequalities_are_strict": True,
        "passed": passed,
    }


def _half_spin_defining_certificate(mean: FieldMatrix) -> dict[str, object]:
    variable = sp.symbols("x")
    quintic = sp.Poly(
        variable**5
        - variable**4 / 7
        - variable**3 / 63
        + 5 * variable**2 / 3087
        + 13 * variable / 194481
        - sp.Rational(5, 1361367),
        variable,
        domain=sp.QQ,
    )
    expected = sp.Poly(
        variable
        * (variable - sp.Rational(4, 21))
        * (variable - sp.Rational(1, 21))
        * quintic.as_expr(),
        variable,
        domain=sp.QQ,
    )
    characteristic = sp.Poly(
        _field_matrix_to_sympy(mean).charpoly(variable).as_expr(),
        variable,
        domain=sp.QQ,
    )
    left = -HALF_SPIN_RADIUS
    right = HALF_SPIN_RADIUS
    quintic_roots_inside = int(sp.count_roots(quintic, left, right))
    endpoints_nonzero = quintic.eval(left) != 0 and quintic.eval(right) != 0
    factorization_matches = characteristic == expected
    passed = bool(
        factorization_matches
        and quintic_roots_inside == 5
        and endpoints_nonzero
    )
    return {
        "dimension": 8,
        "exact_mean_matrix": _matrix_strings(mean),
        "mean_matrix_sha256": _matrix_hash(mean),
        "characteristic_factorization": (
            "x*(x - 4/21)*(x - 1/21)*(" + str(quintic.as_expr()) + ")"
        ),
        "characteristic_factorization_matches": factorization_matches,
        "quintic_root_count_interval": [str(left), str(right)],
        "quintic_roots_strictly_inside_interval": quintic_roots_inside,
        "interval_endpoints_are_not_quintic_roots": endpoints_nonzero,
        "radius": str(HALF_SPIN_RADIUS),
        "gap": str(1 - HALF_SPIN_RADIUS),
        "radius_is_attained_by_linear_factor": True,
        "passed": passed,
    }


def _symmetric_alphabet(
    normalizer_generators: Sequence[monomial.Monomial],
    golden_pair: Sequence[golden.Matrix8],
) -> tuple[FieldMatrix, ...]:
    """Return the symmetric labelled alphabet, retaining cross-source repeats.

    A matrix that appears once in the monomial alphabet and once in the golden
    alphabet receives twice the probability mass.  This is the random walk
    obtained by choosing uniformly from the concatenated maintained generator
    lists, not uniformly from the set-theoretic union of their matrix values.
    """

    monomial_steps = closure._symmetric_monomial_steps(normalizer_generators)
    monomial_matrices = tuple(
        closure._field_matrix_from_monomial(step) for step in monomial_steps
    )
    golden_steps = closure._symmetric_golden_steps(golden_pair)
    return (*monomial_matrices, *golden_steps)


def certificate() -> dict[str, object]:
    """Return the exact low-degree contraction certificate."""

    operator_report, _, operator_generators = (
        monomial._operator_group_certificate()
    )
    automorphism_report, _, automorphism_generators = (
        monomial._automorphism_group_certificate()
    )
    normalizer_generators = (*operator_generators, *automorphism_generators)
    triality_matrices = golden._spin8_action_matrices()
    gram = _traceless_gram()

    view_reports: dict[str, object] = {}
    defining_means: dict[str, FieldMatrix] = {}
    for view, indices in closure.VIEW_GENERATOR_INDICES.items():
        pair = tuple(triality_matrices[index] for index in indices)
        alphabet = _symmetric_alphabet(normalizer_generators, pair)
        distinct_matrices = len(_unique(alphabet))
        defining_mean = _mean_matrix(alphabet)
        adjoint_mean = _mean_matrix(
            tuple(_wedge_square(step) for step in alphabet)
        )
        symmetric_mean = _mean_matrix(
            tuple(_traceless_symmetric_square(step) for step in alphabet)
        )
        defining_means[view] = defining_mean

        defining = (
            _vector_defining_certificate(defining_mean)
            if view == "vector"
            else _half_spin_defining_certificate(defining_mean)
        )
        adjoint = _ldlt_radius_certificate(
            adjoint_mean,
            ADJOINT_RADIUS_BOUNDS[view],
        )
        traceless_symmetric = _ldlt_radius_certificate(
            symmetric_mean,
            SYMMETRIC_RADIUS_BOUNDS[view],
            gram=gram,
        )
        checks = {
            "alphabet_has_expected_size": len(alphabet)
            == (20 if view == "vector" else 21),
            "defining_representation_bound_passed": bool(defining["passed"]),
            "adjoint_representation_bound_passed": bool(adjoint["passed"]),
            "traceless_symmetric_representation_bound_passed": bool(
                traceless_symmetric["passed"]
            ),
        }
        view_reports[view] = {
            "symmetric_alphabet_size": len(alphabet),
            "distinct_matrix_count": distinct_matrices,
            "uniform_label_weight": f"1/{len(alphabet)}",
            "defining_8": defining,
            "adjoint_28": adjoint,
            "traceless_symmetric_35": traceless_symmetric,
            "checks": checks,
            "passed": all(checks.values()),
        }

    half_means_equal = (
        defining_means["positive_half_spin"]
        == defining_means["negative_half_spin"]
    )
    checks = {
        "operator_group_input_passed": bool(operator_report["passed"]),
        "signed_fano_input_passed": bool(automorphism_report["passed"]),
        "normalizer_forward_generator_count_is_nine": (
            len(normalizer_generators) == 9
        ),
        "positive_and_negative_defining_means_are_equal": half_means_equal,
        "all_view_certificates_passed": all(
            report["passed"] for report in view_reports.values()
        ),
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "experiment": "mixed monomial/golden exact low-degree mixing bounds",
        "field": "Q(sqrt(5))",
        "measure": {
            "definition": (
                "uniform probability measure on the concatenated symmetric "
                "monomial and view-specific golden labelled alphabets; "
                "cross-source duplicate matrices retain multiplicity"
            ),
            "monomial_forward_generators": 9,
            "monomial_symmetric_steps": 17,
            "measure_is_symmetric": True,
        },
        "proof_method": {
            "defining_8": (
                "exact characteristic polynomial and Sturm root counts"
            ),
            "adjoint_28": (
                "exact LDL^T positivity of cI+M and cI-M"
            ),
            "traceless_symmetric_35": (
                "exact LDL^T positivity of cG+GM and cG-GM"
            ),
            "quadratic_field_signs": (
                "exact rational comparison of a^2 and 5b^2 for a+b*sqrt(5)"
            ),
        },
        "views": view_reports,
        "checks": checks,
        "claim_scope": {
            "proved": [
                "strict contraction in each displayed finite-dimensional representation for each fixed symmetric measure",
                "an exact defining-representation radius of 4/21 in both half-spin views",
                "the displayed rational lower bounds on the corresponding low-degree contraction gaps",
            ],
            "not_claimed": [
                "a spectral gap on the full mean-zero L2(SO(8))",
                "a total-variation or Wasserstein mixing-time theorem",
                "optimality of the maintained generator weights",
                "an SSM accuracy, efficiency, or hardware-speed advantage",
            ],
        },
        "passed": all(checks.values()),
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
