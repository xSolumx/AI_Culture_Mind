"""Raw characteristic-zero identities on the two coupled boundary blow-ups.

The uniform projective compactification of the coupled Spin(9) V1+V5 graph
slice loses rank at two points.  Retaining the finite third spinor gives two
explicit three-column boundary frames.  This module constructs their raw
36-dimensional information matrices over QQ(sqrt(2))[d,w] and verifies the
factor formulas used by :mod:`spin9_v1_v5_blowup` without using the modular
18,600-coefficient reconstruction.

After clearing the positive Gram denominator, every information entry has
total degree at most two.  The determinant difference therefore has total
degree at most 72.  Exact agreement on the lower Newton grid
``{(i,j): i,j >= 0, i+j <= 72}`` proves the polynomial identity in
characteristic zero.  This is a boundary theorem only; it is not the raw
identity on the finite-radius coupled three-variable slice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import sympy as sp

from spin9_dirac_clifford import build_spin9_clifford_system
from spin9_v1_v5_blowup import ROOT, expected_factorization
from spin9_v1_v5_reconstruction import COUPLED_INFORMATION_BLOCKS

SQRT2 = sp.sqrt(2)
D, W = sp.symbols("d w", real=True)
IDENTITY_TOTAL_DEGREE = 72
NEWTON_GRID = tuple(
    (left, total - left)
    for total in range(IDENTITY_TOTAL_DEGREE + 1)
    for left in range(total + 1)
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "spin9_v1_v5_boundary_char0_20260811.json"


def _base_and_variations() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix]:
    base = sp.zeros(16, 3)
    base[0, 0] = 1
    base[1, 1] = 1 / SQRT2
    base[8, 1] = 1 / SQRT2
    base[2, 2] = -1 / SQRT2
    base[12, 2] = 1 / SQRT2

    scalar = sp.zeros(16, 3)
    for row, column, value in (
        (1, 1, 1),
        (2, 2, -1),
        (5, 2, 1),
        (6, 1, 1),
        (8, 1, -1),
        (9, 0, -SQRT2),
        (11, 2, 1),
        (12, 2, -1),
        (14, 0, -SQRT2),
        (15, 1, -1),
    ):
        scalar[row, column] = value

    axis = sp.zeros(16, 3)
    for row, column, value in (
        (1, 1, 2),
        (2, 2, 1),
        (5, 2, -1),
        (6, 1, -1),
        (8, 1, -2),
        (9, 0, -2 * SQRT2),
        (11, 2, -1),
        (12, 2, 1),
        (14, 0, SQRT2),
        (15, 1, 1),
    ):
        axis[row, column] = value

    transverse = sp.zeros(16, 3)
    for row, column, value in (
        (4, 2, -SQRT2),
        (6, 0, -1),
        (10, 2, SQRT2),
        (14, 1, SQRT2),
        (15, 0, 1),
    ):
        transverse[row, column] = value
    return base, scalar, axis, transverse


def _observation_outer(
    column: sp.Matrix,
    generators: tuple[sp.Matrix, ...],
) -> sp.Matrix:
    observations = sp.Matrix.hstack(*(generator * column for generator in generators)).T
    return observations * observations.T


@lru_cache(maxsize=2)
def _boundary_data(
    family: str,
) -> tuple[tuple[sp.Matrix, sp.Matrix], sp.Expr, int, dict[str, bool]]:
    base, scalar, axis, transverse = _base_and_variations()
    generators = tuple(
        sp.Matrix(matrix)
        for matrix in build_spin9_clifford_system().doubled_spin_generators
    )
    if family == "A":
        leading = scalar + axis
        finite = base + D / SQRT2 * scalar + W / SQRT2 * transverse
        columns = (leading[:, 0], leading[:, 1], finite[:, 2])
        norm = 1 + 2 * D**2 + 2 * W**2
        denominator_scale = 18
        expected_gram = sp.diag(18, 18, norm)
        cleared_information = (
            norm
            * (
                _observation_outer(columns[0], generators)
                + _observation_outer(columns[1], generators)
            )
            + 18 * _observation_outer(columns[2], generators)
        )
        limit_checks = {
            "leading_third_column_zero": leading[:, 2] == sp.zeros(16, 1),
        }
    elif family == "B":
        leading = -2 * scalar + axis + 3 * transverse
        finite = base + D / SQRT2 * scalar - 3 * W / SQRT2 * transverse
        columns = (
            leading[:, 0],
            leading[:, 2],
            finite[:, 0] - finite[:, 1],
        )
        norm = 1 + 2 * D**2 - 6 * D * W + 9 * W**2
        denominator_scale = 72
        expected_gram = sp.diag(36, 72, 2 * norm)
        cleared_information = (
            2 * norm * _observation_outer(columns[0], generators)
            + norm * _observation_outer(columns[1], generators)
            + 36 * _observation_outer(columns[2], generators)
        )
        limit_checks = {
            "leading_first_two_columns_equal": leading[:, 0] == leading[:, 1],
        }
    else:
        raise ValueError("family must be 'A' or 'B'")

    boundary_frame = sp.Matrix.hstack(*columns)
    cleared_information = cleared_information.applyfunc(sp.expand)
    blocks = tuple(
        cleared_information.extract(block, block)
        for block in COUPLED_INFORMATION_BLOCKS
    )
    cross_block = cleared_information.extract(
        COUPLED_INFORMATION_BLOCKS[0],
        COUPLED_INFORMATION_BLOCKS[1],
    )
    checks = {
        **limit_checks,
        "boundary_gram_identity": sp.simplify(boundary_frame.T * boundary_frame)
        == expected_gram,
        "information_blocks_exact": cross_block == sp.zeros(16, 20),
        "cleared_entry_degree_at_most_two": all(
            sp.Poly(entry, D, W, extension=SQRT2).total_degree() <= 2
            for entry in cleared_information
        ),
    }
    return blocks, norm, denominator_scale, checks


@lru_cache(maxsize=1)
def _base_block_determinants() -> tuple[int, int]:
    base, _, _, _ = _base_and_variations()
    generators = tuple(
        sp.Matrix(matrix)
        for matrix in build_spin9_clifford_system().doubled_spin_generators
    )
    information = sum(
        (_observation_outer(base[:, index], generators) for index in range(3)),
        sp.zeros(36),
    )
    return tuple(
        int(information.extract(block, block).det(method="domain-ge"))
        for block in COUPLED_INFORMATION_BLOCKS
    )


def _grid_row(task: tuple[str, int, int]) -> tuple[str, str, str] | None:
    family, d_value, w_value = task
    blocks, norm, denominator_scale, _ = _boundary_data(family)
    substitution = {D: d_value, W: w_value}
    determinants = tuple(
        sp.expand(block.subs(substitution).det(method="domain-ge"))
        for block in blocks
    )
    normalized_numerator, _, _ = expected_factorization(family)
    base_product = sp.prod(_base_block_determinants())
    expected = sp.expand(
        base_product
        * denominator_scale**36
        * norm.subs(substitution) ** 22
        * normalized_numerator.subs(substitution)
    )
    observed = sp.expand(determinants[0] * determinants[1])
    if observed != expected:
        return None
    return str(determinants[0]), str(determinants[1]), str(observed)


def _family_certificate(
    family: str,
    *,
    workers: int,
) -> dict[str, object]:
    _, _, denominator_scale, structural_checks = _boundary_data(family)
    tasks = ((family, left, right) for left, right in NEWTON_GRID)
    if workers == 1:
        rows = [_grid_row(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_grid_row, tasks, chunksize=8))
    all_nodes_match = all(row is not None for row in rows)
    digest = hashlib.sha256()
    for (left, right), row in zip(NEWTON_GRID, rows, strict=True):
        digest.update(f"{family},{left},{right},{row}\n".encode())
    return {
        "family": family,
        "cleared_information_denominator": f"{denominator_scale}*n_{family}",
        "entry_total_degree_bound": 2,
        "determinant_identity_total_degree_bound": IDENTITY_TOTAL_DEGREE,
        "newton_grid_node_count": len(NEWTON_GRID),
        "newton_grid_definition": "i>=0, j>=0, i+j<=72",
        "structural_checks": structural_checks,
        "all_newton_grid_nodes_match": all_nodes_match,
        "raw_block_determinant_rows_sha256": digest.hexdigest(),
        "passed": all(structural_checks.values()) and all_nodes_match,
    }


def certificate(*, workers: int = 1) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers must be positive")
    base_determinants = _base_block_determinants()
    families = [
        _family_certificate(family, workers=workers) for family in ("A", "B")
    ]
    passed = base_determinants == (65_536, 262_144) and all(
        row["passed"] for row in families
    )
    return {
        "schema_version": 1,
        "claim_scope": (
            "raw characteristic-zero determinant identities on the two "
            "exceptional coupled boundary planes"
        ),
        "arithmetic_field": "QQ(sqrt(2))",
        "identity_method": (
            "degree-72 polynomial identity on the 2701-node lower Newton grid"
        ),
        "base_information_block_determinants": list(base_determinants),
        "families": families,
        "modular_reconstruction_used": False,
        "finite_radius_coupled_identity_certified": False,
        "global_coupled_determinant_theorem_claimed": False,
        "passed": passed,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = certificate(workers=args.workers)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
