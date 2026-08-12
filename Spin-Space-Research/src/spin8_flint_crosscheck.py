"""Independent FLINT arithmetic checks for central SymPy certificates.

SymPy is still used to construct the maintained triality projectors.  They are
then serialized coefficient-by-coefficient into python-flint matrices.  FLINT
independently recomputes ranks, determinants, characteristic polynomials, and
the 28th-degree fixed-support weight determinant.  This catches arithmetic
backend errors; it does not independently validate the geometric model used to
construct the matrices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp

try:
    from flint import ctx, fmpq, fmpq_mat
except ImportError:  # pragma: no cover - exercised only in the minimal install
    ctx = fmpq = fmpq_mat = None

FLINT_AVAILABLE = fmpq_mat is not None

from spin8_cayley_spectrum import symbolic_query_projector, symbolic_triality_generators


def _basis() -> list[list[sp.Integer]]:
    return [[sp.Integer(row == column) for column in range(8)] for row in range(8)]


def _flint_rational(value: sp.Expr) -> fmpq:
    rational = sp.Rational(value)
    return fmpq(int(rational.p), int(rational.q))


def _flint_matrix(matrix: sp.Matrix) -> fmpq_mat:
    return fmpq_mat(
        matrix.rows,
        matrix.cols,
        [_flint_rational(value) for value in matrix],
    )


def _sympy_rational(value: object) -> sp.Rational:
    return sp.Rational(str(value))


def _charpoly_coefficients(polynomial: object) -> list[sp.Rational]:
    coefficients = [_sympy_rational(value) for value in polynomial.coeffs()]
    return list(reversed(coefficients))


def _interpolate_with_flint(
    points: list[fmpq], values: list[fmpq]
) -> list[sp.Rational]:
    vandermonde = fmpq_mat(
        len(points),
        len(points),
        [point**power for point in points for power in range(len(points))],
    )
    column = fmpq_mat(len(points), 1, values)
    coefficients = vandermonde.inv() * column
    return [_sympy_rational(coefficients[index, 0]) for index in range(len(points))]


def run(*, flint_threads: int = 6) -> dict[str, object]:
    if not FLINT_AVAILABLE:
        raise RuntimeError("python-flint is required; install the 'exact' extra")
    if not 1 <= flint_threads <= 7:
        raise ValueError("flint_threads must leave at least one CPU core free")
    ctx.threads = flint_threads
    generators = symbolic_triality_generators()
    basis = _basis()
    query_rows = (
        (0, basis[0]),
        (1, basis[0]),
        (1, basis[1]),
        (2, basis[2]),
        (2, basis[4]),
    )
    projectors_sympy = [
        symbolic_query_projector(view, vector, generators)
        for view, vector in query_rows
    ]
    projectors_flint = [_flint_matrix(matrix) for matrix in projectors_sympy]
    information_sympy = sum(projectors_sympy, sp.zeros(28))
    information_flint = sum(projectors_flint[1:], projectors_flint[0])

    sympy_charpoly = sp.Poly(information_sympy.charpoly().as_expr()).all_coeffs()
    flint_charpoly = _charpoly_coefficients(information_flint.charpoly())
    matrix_agreement = {
        "sympy_rank": int(information_sympy.rank()),
        "flint_rank": int(information_flint.rank()),
        "sympy_determinant": str(information_sympy.det()),
        "flint_determinant": str(information_flint.det()),
        "characteristic_coefficients_match": sympy_charpoly == flint_charpoly,
    }

    alpha = sp.symbols("alpha")
    expected = sp.Poly(
        -(alpha**3) * (alpha - 5) ** 21 * (7 * alpha + 5) ** 4 / 2**60,
        alpha,
    )
    expected_coefficients = list(reversed(expected.all_coeffs()))
    points = [fmpq(index, 7) for index in range(29)]
    values = []
    for point in points:
        beta = (fmpq(5) - point) / 4
        weighted = point * projectors_flint[0]
        for projector in projectors_flint[1:]:
            weighted += beta * projector
        values.append(weighted.det())
    reconstructed = _interpolate_with_flint(points, values)
    polynomial_agreement = {
        "degree": len(reconstructed) - 1,
        "sample_count": len(points),
        "coefficients_match_sympy_factorization": reconstructed
        == expected_coefficients,
        "flint_boundary_rank_alpha_zero": int(
            sum(projectors_flint[1:], projectors_flint[1] * 0).rank()
        ),
        "flint_boundary_rank_alpha_five": int(projectors_flint[0].rank()),
    }

    moved = list(basis[3])
    replacement = symbolic_query_projector(2, moved, generators)
    boundary_sympy = information_sympy - projectors_sympy[-1] + replacement
    boundary_flint = _flint_matrix(boundary_sympy)
    boundary_agreement = {
        "sympy_rank": int(boundary_sympy.rank()),
        "flint_rank": int(boundary_flint.rank()),
        "sympy_determinant": str(boundary_sympy.det()),
        "flint_determinant": str(boundary_flint.det()),
    }

    passed = bool(
        matrix_agreement["sympy_rank"] == matrix_agreement["flint_rank"] == 28
        and matrix_agreement["sympy_determinant"]
        == matrix_agreement["flint_determinant"]
        == "81/1024"
        and matrix_agreement["characteristic_coefficients_match"]
        and polynomial_agreement["coefficients_match_sympy_factorization"]
        and polynomial_agreement["flint_boundary_rank_alpha_zero"] == 25
        and polynomial_agreement["flint_boundary_rank_alpha_five"] == 7
        and boundary_agreement["sympy_rank"] == boundary_agreement["flint_rank"] == 25
        and boundary_agreement["sympy_determinant"]
        == boundary_agreement["flint_determinant"]
        == "0"
    )
    return {
        "experiment": "independent SymPy and python-flint arithmetic cross-check",
        "flint_threads": flint_threads,
        "matrix_certificate": matrix_agreement,
        "weight_polynomial_certificate": polynomial_agreement,
        "rank_boundary_certificate": boundary_agreement,
        "scope_boundary": (
            "FLINT independently checks exact arithmetic after the maintained "
            "projectors are constructed. It does not independently derive the "
            "Spin(8) representation matrices."
        ),
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run(flint_threads=arguments.threads)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("FLINT cross-verification failed")


if __name__ == "__main__":
    main()
