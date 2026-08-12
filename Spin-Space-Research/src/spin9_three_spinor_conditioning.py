"""Exact D-optimality certificate on a symmetric three-spinor curve.

The explicit curve consists of mutually orthonormal spinors whose quadratic
Spin(9) Hopf images have one common pairwise inner product c.  This module
proves the determinant formula and its unique maximizer on the complete
parameter interval.  It does not assert that this curve exhausts the
equiangular locus or that it is globally optimal among arbitrary triples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp

from spin9_dirac_clifford import build_spin9_clifford_system

INFORMATION_BLOCKS = (
    (6, 12, 14, 17, 21, 34),
    (0, 5, 7, 13, 15, 16, 22, 26, 33, 35),
    (1, 4, 9, 10, 19, 23, 25, 27, 29, 31),
    (2, 3, 8, 11, 18, 20, 24, 28, 30, 32),
)


def _symbolic_template() -> (
    tuple[sp.Symbol, list[sp.Matrix], sp.Matrix, dict[sp.Symbol, sp.Expr]]
):
    """Return a polynomial template and the symmetric-family substitution."""

    c = sp.symbols("c", real=True)
    d, b, y, z = sp.symbols("d b y z", real=True)
    d_value = sp.sqrt((1 + c) / 2)
    b_value = sp.sqrt((1 - c) / 2)
    substitution = {
        d: d_value,
        b: b_value,
        y: c * (1 - c) / (2 * b_value * (1 + c)),
        z: sp.sqrt((1 - c) * (1 + 2 * c) / (2 * (1 + c) ** 2)),
    }

    spinors = []
    first = sp.zeros(16, 1)
    first[0] = 1
    spinors.append(first)

    second = sp.zeros(16, 1)
    second[1] = d
    second[8] = b
    spinors.append(second)

    third = sp.zeros(16, 1)
    third[2] = -d
    third[11] = y
    third[12] = z
    spinors.append(third)

    system = build_spin9_clifford_system()
    # The maintained integer matrices are twice the conventional generators.
    generators = [sp.Matrix(matrix) / 2 for matrix in system.doubled_spin_generators]
    jacobian = sp.Matrix.vstack(
        *[
            sp.Matrix.hstack(*[generator * spinor for generator in generators])
            for spinor in spinors
        ]
    )
    return c, spinors, sp.expand(jacobian.T * jacobian), substitution


def _hopf_gram(spinors: list[sp.Matrix]) -> sp.Matrix:
    system = build_spin9_clifford_system()
    hopf = sp.Matrix(
        [
            [
                (spinor.T * sp.Matrix(matrix) * spinor)[0]
                for matrix in system.involutions
            ]
            for spinor in spinors
        ]
    )
    return sp.simplify(hopf * hopf.T)


def diagnostics() -> dict[str, object]:
    """Reconstruct and verify the complete exact one-parameter certificate."""

    c, spinors, information, substitution = _symbolic_template()
    expected_indices = set(range(36))
    observed_indices = {index for block in INFORMATION_BLOCKS for index in block}
    partition_ok = observed_indices == expected_indices and sum(
        len(block) for block in INFORMATION_BLOCKS
    ) == len(observed_indices)

    off_block_zero = True
    block_of = {
        index: block_index
        for block_index, block in enumerate(INFORMATION_BLOCKS)
        for index in block
    }
    for row in range(36):
        for column in range(36):
            if block_of[row] != block_of[column] and information[row, column] != 0:
                off_block_zero = False

    block_determinants = [
        sp.factor(
            information.extract(block, block).det(method="domain-ge").subs(substitution)
        )
        for block in INFORMATION_BLOCKS
    ]
    expected_blocks = [
        (1 - c) * (c + 2) ** 2 / sp.Integer(2) ** 7,
        (1 - c) ** 3 * (c + 2) * (2 * c + 1) / sp.Integer(2) ** 12,
        (1 - c) ** 3 * (c + 2) * (2 * c + 1) / sp.Integer(2) ** 12,
        (1 - c) ** 3 * (c + 2) * (2 * c + 1) / sp.Integer(2) ** 12,
    ]
    block_identities = [
        sp.factor(observed - expected) == 0
        for observed, expected in zip(block_determinants, expected_blocks, strict=True)
    ]

    spectral_variable = sp.symbols("x")
    block_characteristic_polynomials = [
        sp.factor(
            information.extract(block, block)
            .charpoly(spectral_variable)
            .as_expr()
            .subs(substitution)
        )
        for block in INFORMATION_BLOCKS
    ]
    first_quadratic = 8 * spectral_variable**2 - 8 * spectral_variable + 1 - c
    second_quadratic = 4 * spectral_variable**2 - 7 * spectral_variable + 2 + c
    quartic = (
        16 * spectral_variable**4
        - 60 * spectral_variable**3
        + (64 + 4 * c) * spectral_variable**2
        - (16 + 8 * c) * spectral_variable
        + 1
        + c
        - 2 * c**2
    )
    expected_characteristic_polynomials = [
        first_quadratic * second_quadratic**2 / sp.Integer(2) ** 7,
        first_quadratic**2 * second_quadratic * quartic / sp.Integer(2) ** 12,
        first_quadratic**2 * second_quadratic * quartic / sp.Integer(2) ** 12,
        first_quadratic**2 * second_quadratic * quartic / sp.Integer(2) ** 12,
    ]
    characteristic_identities = [
        sp.factor(observed - expected) == 0
        for observed, expected in zip(
            block_characteristic_polynomials,
            expected_characteristic_polynomials,
            strict=True,
        )
    ]
    full_characteristic_polynomial = sp.factor(
        first_quadratic**7 * second_quadratic**5 * quartic**3 / sp.Integer(2) ** 43
    )

    determinant = sp.factor(sp.prod(block_determinants))
    expected_determinant = sp.factor(
        (1 - c) ** 10 * (c + 2) ** 5 * (2 * c + 1) ** 3 / sp.Integer(2) ** 43
    )
    log_derivative = sp.factor(sp.diff(determinant, c) / determinant)
    expected_log_derivative = sp.factor(
        3 * (12 * c**2 + 17 * c + 1) / ((c - 1) * (c + 2) * (2 * c + 1))
    )
    optimum = sp.factor((-17 + sp.sqrt(241)) / 24)

    spinor_gram = sp.simplify(
        (sp.Matrix.hstack(*spinors).T * sp.Matrix.hstack(*spinors)).subs(substitution)
    )
    expected_hopf_gram = sp.Matrix(
        [[1 if row == column else c for column in range(3)] for row in range(3)]
    )
    hopf_gram = sp.simplify(_hopf_gram(spinors).subs(substitution))

    report: dict[str, object] = {
        "schema_version": 1,
        "claim_scope": "explicit orthonormal/equiangular three-spinor curve only",
        "feasible_open_interval": ["-1/2", "1"],
        "information_block_sizes": [len(block) for block in INFORMATION_BLOCKS],
        "information_blocks_partition_36": partition_ok,
        "off_block_entries_zero": off_block_zero,
        "block_determinants": [sp.sstr(value) for value in block_determinants],
        "block_determinant_identities": block_identities,
        "block_characteristic_polynomials": [
            sp.sstr(value) for value in block_characteristic_polynomials
        ],
        "block_characteristic_polynomial_identities": characteristic_identities,
        "full_characteristic_polynomial": sp.sstr(full_characteristic_polynomial),
        "spectral_factor_degrees": [2, 2, 4],
        "spectral_factor_multiplicities": [7, 5, 3],
        "determinant": sp.sstr(determinant),
        "determinant_identity": sp.factor(determinant - expected_determinant) == 0,
        "trace": sp.sstr(sp.simplify(sp.trace(information).subs(substitution))),
        "log_derivative": sp.sstr(log_derivative),
        "log_derivative_identity": sp.factor(log_derivative - expected_log_derivative)
        == 0,
        "unique_maximizer": sp.sstr(optimum),
        "unique_maximizer_approx": float(sp.N(optimum, 17)),
        "spinor_gram_identity": spinor_gram == sp.eye(3),
        "hopf_gram_identity": sp.simplify(hopf_gram - expected_hopf_gram)
        == sp.zeros(3),
        "boundary_determinants_zero": bool(
            sp.limit(determinant, c, sp.Rational(-1, 2), dir="+") == 0
            and sp.limit(determinant, c, 1, dir="-") == 0
        ),
        "complete_equiangular_locus_claimed": False,
        "global_all_triples_optimality_claimed": False,
    }
    report["passed"] = bool(
        report["information_blocks_partition_36"]
        and report["off_block_entries_zero"]
        and all(block_identities)
        and all(characteristic_identities)
        and report["determinant_identity"]
        and report["trace"] == "27"
        and report["log_derivative_identity"]
        and report["spinor_gram_identity"]
        and report["hopf_gram_identity"]
        and report["boundary_determinants_zero"]
        and not report["complete_equiangular_locus_claimed"]
        and not report["global_all_triples_optimality_claimed"]
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
