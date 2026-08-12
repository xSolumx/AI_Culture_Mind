"""Exact counterexample to coordinatewise Cholesky decorrelation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import sympy as sp

from spin8_cayley_spectrum import (
    symbolic_query_projector,
    symbolic_triality_generators,
)
from spin8_dirac_star import rational_circle

RATIONAL_CIRCLE_COORDINATES = (
    -sp.Rational(14, 19),
    sp.Rational(39, 50),
    sp.Rational(5, 31),
    -sp.Rational(39, 50),
    -sp.Rational(11, 64),
    sp.Rational(46, 59),
    -sp.Rational(7, 9),
)
PARAMETER_NAMES = ("a", "d", "e", "g", "h", "i", "c")
REMOVED_RESIDUALS = ("e", "h", "i")


def _frame(
    pairs: list[tuple[sp.Expr, sp.Expr]],
    basis: list[list[sp.Integer]],
) -> list[list[sp.Expr]]:
    (a, aa), (d, dd), (e, ee), (g, gg), (h, hh), (i, ii), (c, ss) = pairs
    return [
        basis[0],
        [a * basis[0][k] + aa * basis[1][k] for k in range(8)],
        [d * basis[0][k] + dd * (e * basis[1][k] + ee * basis[2][k]) for k in range(8)],
        [
            g * basis[0][k]
            + gg
            * (
                h * basis[1][k]
                + hh * (i * basis[2][k] + ii * (c * basis[3][k] + ss * basis[4][k]))
            )
            for k in range(8)
        ],
    ]


def _exact_audit(
    pairs: list[tuple[sp.Expr, sp.Expr]],
    generators: list[list[list[list[sp.Rational]]]],
    basis: list[list[sp.Integer]],
    fixed: sp.Matrix,
) -> dict[str, object]:
    frame = _frame(pairs, basis)
    information = (
        fixed
        + symbolic_query_projector(1, frame[1], generators)
        + symbolic_query_projector(2, frame[2], generators)
        + symbolic_query_projector(2, frame[3], generators)
    )
    determinant = sp.factor(information.det(method="domain-ge"))
    gram_determinant = sp.factor(sp.prod(complement**2 for _, complement in pairs[:6]))
    normalized_determinant = sp.factor(determinant / gram_determinant**3)
    gram = sp.Matrix(frame) * sp.Matrix(frame).T
    gram_float = np.asarray(gram, dtype=np.float64)
    leading_principal_minors = [
        sp.factor(gram[:size, :size].det()) for size in range(1, 5)
    ]
    return {
        "determinant_exact": str(determinant),
        "determinant_float": float(determinant),
        "gram_determinant_exact": str(gram_determinant),
        "gram_determinant_float": float(gram_determinant),
        "gram_matrix_exact": [
            [str(gram[row, column]) for column in range(4)] for row in range(4)
        ],
        "gram_matrix_float": gram_float.tolist(),
        "gram_eigenvalues_float": np.linalg.eigvalsh(gram_float).tolist(),
        "gram_leading_principal_minors_exact": [
            str(value) for value in leading_principal_minors
        ],
        "gram_positive_definite_exact": all(
            value > 0 for value in leading_principal_minors
        ),
        "normalized_determinant_exact": str(normalized_determinant),
        "normalized_determinant_float": float(normalized_determinant),
        "_normalized": normalized_determinant,
    }


def run() -> dict[str, object]:
    generators = symbolic_triality_generators()
    basis = [[sp.Integer(row == column) for column in range(8)] for row in range(8)]
    fixed = symbolic_query_projector(
        0, basis[0], generators
    ) + symbolic_query_projector(1, basis[0], generators)
    pairs = [rational_circle(value) for value in RATIONAL_CIRCLE_COORDINATES]
    general = _exact_audit(pairs, generators, basis, fixed)
    star_pairs = pairs.copy()
    for index in (2, 4, 5):
        star_pairs[index] = (sp.Integer(0), sp.Integer(1))
    star = _exact_audit(star_pairs, generators, basis, fixed)
    ratio = sp.factor(general.pop("_normalized") / star.pop("_normalized"))
    normalized_difference = sp.factor(
        sp.Rational(general["normalized_determinant_exact"])
        - sp.Rational(star["normalized_determinant_exact"])
    )
    normalized_cayley = pairs[6][0]
    cayley_form = sp.factor(
        normalized_cayley * sp.prod(complement for _, complement in pairs[:6])
    )
    passed = bool(ratio > 1)
    return {
        "experiment": "exact conditional-decorrelation counterexample",
        "rational_circle_coordinates": {
            name: str(value)
            for name, value in zip(
                PARAMETER_NAMES, RATIONAL_CIRCLE_COORDINATES, strict=True
            )
        },
        "cholesky_partial_correlations": {
            name: str(pair[0])
            for name, pair in zip(PARAMETER_NAMES, pairs, strict=True)
        },
        "positive_cholesky_diagonals": {
            name: str(pair[1])
            for name, pair in zip(PARAMETER_NAMES, pairs, strict=True)
        },
        "normalized_cayley_exact": str(normalized_cayley),
        "normalized_cayley_float": float(normalized_cayley),
        "cayley_form_exact": str(cayley_form),
        "general_frame": general,
        "decorrelated_star_frame": star,
        "normalized_determinant_ratio_exact": str(ratio),
        "normalized_determinant_difference_exact": str(normalized_difference),
        "normalized_determinant_difference_float": float(normalized_difference),
        "normalized_determinant_ratio_float": float(ratio),
        "normalized_log_gain_float": math.log(float(ratio)),
        "removed_residual_partial_correlations": list(REMOVED_RESIDUALS),
        "falsified_statement": (
            "At fixed (a,d,g,c), setting the residual Cholesky partial "
            "correlations (e,h,i) to zero cannot decrease det(I)/det(G)^3."
        ),
        "not_falsified": (
            "The global Dirac--Gram inequality and other invariant-preserving "
            "deformation paths remain open."
        ),
        "exact_reversal_verified": passed,
        "exact_positive_definiteness_verified": bool(
            general["gram_positive_definite_exact"]
            and star["gram_positive_definite_exact"]
        ),
        "passed": passed,
    }


def verify_report(report: dict[str, object]) -> bool:
    ratio = sp.Rational(report["normalized_determinant_ratio_exact"])
    difference = sp.Rational(report["normalized_determinant_difference_exact"])
    return bool(
        ratio > 1
        and difference > 0
        and report["exact_reversal_verified"]
        and report["exact_positive_definiteness_verified"]
        and report["passed"]
        and report["removed_residual_partial_correlations"] == ["e", "h", "i"]
    )


def verify_artifact(path: Path) -> bool:
    return verify_report(json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "normalized_determinant_ratio_exact": report[
                    "normalized_determinant_ratio_exact"
                ],
                "normalized_determinant_ratio_float": report[
                    "normalized_determinant_ratio_float"
                ],
                "normalized_determinant_difference_float": report[
                    "normalized_determinant_difference_float"
                ],
                "normalized_log_gain_float": report["normalized_log_gain_float"],
                "exact_reversal_verified": report["exact_reversal_verified"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
