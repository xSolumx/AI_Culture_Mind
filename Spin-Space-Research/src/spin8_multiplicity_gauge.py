"""Exact multiplicity-space gauge certificate for repeated triality views.

If several probes use the same representation, their information contribution
depends on their row covariance, not on a chosen list of probe vectors.  An
orthogonal change of basis across that list is therefore an exact gauge
symmetry.  This module checks the maintained Spin(8) realization and records a
non-orthogonal two-probe example in exact arithmetic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp

from spin8_cayley_spectrum import (
    CAYLEY_TERMS,
    symbolic_query_projector,
    symbolic_triality_generators,
)


def _basis() -> list[list[sp.Integer]]:
    return [[sp.Integer(row == column) for column in range(8)] for row in range(8)]


def _linear_combination(
    left_scale: sp.Expr,
    left: list[sp.Expr],
    right_scale: sp.Expr,
    right: list[sp.Expr],
) -> list[sp.Expr]:
    return [
        sp.factor(left_scale * left[index] + right_scale * right[index])
        for index in range(8)
    ]


def _exact_cayley(frame: list[list[sp.Expr]]) -> sp.Expr:
    value = sp.Integer(0)
    matrix = sp.Matrix(frame)
    for columns, coefficient in CAYLEY_TERMS.items():
        value += coefficient * matrix[:, list(columns)].det()
    return sp.factor(value)


def exact_multiplicity_gauge_certificate() -> dict[str, object]:
    generators = symbolic_triality_generators()
    basis = _basis()

    # Two unit vectors with a nonzero rational correlation.
    x = _linear_combination(sp.Rational(3, 5), basis[0], sp.Rational(4, 5), basis[2])
    y = _linear_combination(sp.Rational(5, 13), basis[0], sp.Rational(12, 13), basis[3])
    correlation = sp.factor(sp.Matrix(x).dot(sp.Matrix(y)))

    # A nontrivial rational SO(2) multiplicity-basis change.
    alpha, beta = sp.Rational(3, 5), sp.Rational(4, 5)
    mixed_x = _linear_combination(alpha, x, beta, y)
    mixed_y = _linear_combination(-beta, x, alpha, y)

    projector_residuals = []
    for view in range(3):
        before = symbolic_query_projector(view, x, generators)
        before += symbolic_query_projector(view, y, generators)
        after = symbolic_query_projector(view, mixed_x, generators)
        after += symbolic_query_projector(view, mixed_y, generators)
        projector_residuals.append(after - before)

    rows_before = sp.Matrix([x, y])
    rows_after = sp.Matrix([mixed_x, mixed_y])
    covariance_residual = rows_after.T * rows_after - rows_before.T * rows_before

    # Put the pair into a four-frame to verify both invariants used by the
    # Dirac--Gram objective. The SO(2) change has determinant +1.
    frame_before = [basis[1], basis[4], x, y]
    frame_after = [basis[1], basis[4], mixed_x, mixed_y]
    gram_before = sp.Matrix(frame_before) * sp.Matrix(frame_before).T
    gram_after = sp.Matrix(frame_after) * sp.Matrix(frame_after).T
    cayley_before = _exact_cayley(frame_before)
    cayley_after = _exact_cayley(frame_after)

    # The canonical 45-degree gauge diagonalizes a two-vector Gram matrix.
    root_two = sp.sqrt(2)
    orthogonal_x = _linear_combination(1 / root_two, x, 1 / root_two, y)
    orthogonal_y = _linear_combination(1 / root_two, x, -1 / root_two, y)
    orthogonal_inner_product = sp.factor(
        sp.Matrix(orthogonal_x).dot(sp.Matrix(orthogonal_y))
    )
    orthogonal_norm_squares = [
        sp.factor(sp.Matrix(vector).dot(sp.Matrix(vector)))
        for vector in (orthogonal_x, orthogonal_y)
    ]
    diagonalized_projector_residuals = []
    for view in range(3):
        before = symbolic_query_projector(view, x, generators)
        before += symbolic_query_projector(view, y, generators)
        after = symbolic_query_projector(view, orthogonal_x, generators)
        after += symbolic_query_projector(view, orthogonal_y, generators)
        diagonalized_projector_residuals.append(after - before)

    def zero_matrix(matrix: sp.Matrix) -> bool:
        return all(sp.factor(entry) == 0 for entry in matrix)

    passed = bool(
        correlation != 0
        and all(zero_matrix(matrix) for matrix in projector_residuals)
        and zero_matrix(covariance_residual)
        and sp.factor(gram_after.det() - gram_before.det()) == 0
        and sp.factor(cayley_after - cayley_before) == 0
        and orthogonal_inner_product == 0
        and orthogonal_norm_squares == [1 + correlation, 1 - correlation]
        and all(zero_matrix(matrix) for matrix in diagonalized_projector_residuals)
    )
    return {
        "experiment": "exact repeated-view multiplicity gauge certificate",
        "same_view_projector_identity_verified": [
            zero_matrix(matrix) for matrix in projector_residuals
        ],
        "row_covariance_identity_verified": zero_matrix(covariance_residual),
        "gram_determinant_identity_verified": sp.factor(
            gram_after.det() - gram_before.det()
        )
        == 0,
        "cayley_orientation_identity_verified": sp.factor(cayley_after - cayley_before)
        == 0,
        "example_pair_correlation": str(correlation),
        "orthogonalized_inner_product": str(orthogonal_inner_product),
        "orthogonalized_norm_squares": [
            str(value) for value in orthogonal_norm_squares
        ],
        "expected_norm_squares": [str(1 + correlation), str(1 - correlation)],
        "diagonalized_projector_identity_verified": [
            zero_matrix(matrix) for matrix in diagonalized_projector_residuals
        ],
        "theorem": (
            "For probes X in one triality representation, the summed Fisher "
            "block is a linear function of X^T X and is invariant under "
            "X -> U X for every orthogonal multiplicity action U."
        ),
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = exact_multiplicity_gauge_certificate()
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("multiplicity gauge certificate failed")


if __name__ == "__main__":
    main()
