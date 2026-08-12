"""Exact invariant-block explanation of the balanced Cayley spectrum.

The full characteristic polynomial in :mod:`spin8_cayley_spectrum` is exact,
but its factorization was originally obtained as one 28 by 28 calculation.
This module exposes a smaller structural certificate: in the maintained
bivector basis the entire one-parameter information family has four constant
invariant coordinate blocks of dimensions 8, 8, 8, and 4.  Their determinants
explain every factor in the recurring D-optimal value ``81/1024``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp

from spin8_cayley_spectrum import (
    symbolic_query_projector,
    symbolic_triality_generators,
)
from spin8_triality import SPIN8_PAIRS


def _circle_reduce(expression: sp.Expr, sine: sp.Symbol, cayley: sp.Symbol) -> sp.Expr:
    """Reduce exactly in Q[c,s]/(s^2 + c^2 - 1)."""

    return sp.rem(
        sp.Poly(sp.expand(expression), sine),
        sp.Poly(sine**2 - (1 - cayley**2), sine),
    ).as_expr()


def exact_cayley_information_family() -> tuple[sp.Matrix, sp.Symbol, sp.Symbol]:
    """Construct the exact balanced ``(2, 2, 1)`` information family."""

    generators = symbolic_triality_generators()
    basis = [[sp.Integer(row == column) for column in range(8)] for row in range(8)]
    cayley, sine = sp.symbols("c s")
    final_negative = [
        cayley * basis[3][column] + sine * basis[4][column] for column in range(8)
    ]
    information = (
        symbolic_query_projector(0, basis[0], generators)
        + symbolic_query_projector(1, basis[0], generators)
        + symbolic_query_projector(1, basis[1], generators)
        + symbolic_query_projector(2, basis[2], generators)
        + symbolic_query_projector(2, final_negative, generators)
    )
    return information, cayley, sine


def _pair_basis_invariance_certificate(view: int) -> bool:
    """Check exact O(2) basis invariance of a same-view query pair."""

    generators = symbolic_triality_generators()
    basis = [[sp.Integer(row == column) for column in range(8)] for row in range(8)]
    cosine, sine = sp.symbols("r t")
    first = basis[2 * (view - 1)]
    second = basis[2 * (view - 1) + 1]
    rotated_first = [cosine * first[index] + sine * second[index] for index in range(8)]
    rotated_second = [
        -sine * first[index] + cosine * second[index] for index in range(8)
    ]
    reference = symbolic_query_projector(view, first, generators) + (
        symbolic_query_projector(view, second, generators)
    )
    rotated = symbolic_query_projector(view, rotated_first, generators) + (
        symbolic_query_projector(view, rotated_second, generators)
    )
    return sp.expand(rotated - (cosine**2 + sine**2) * reference) == sp.zeros(28)


def exact_support_components(matrix: sp.Matrix) -> list[list[int]]:
    """Return connected components of the exact off-diagonal support graph."""

    adjacency = {index: set() for index in range(matrix.rows)}
    for row in range(matrix.rows):
        for column in range(row + 1, matrix.cols):
            if matrix[row, column] != 0:
                adjacency[row].add(column)
                adjacency[column].add(row)

    components: list[list[int]] = []
    unseen = set(range(matrix.rows))
    while unseen:
        start = min(unseen)
        stack = [start]
        unseen.remove(start)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            neighbours = sorted(adjacency[current] & unseen, reverse=True)
            for neighbour in neighbours:
                unseen.remove(neighbour)
                stack.append(neighbour)
        components.append(sorted(component))
    return components


def _expected_block_characteristics(
    cayley: sp.Symbol, eigenvalue: sp.Symbol
) -> list[sp.Expr]:
    first = (
        -sp.Rational(1, 4)
        * (eigenvalue - 1) ** 2
        * (
            2 * cayley * eigenvalue
            - cayley
            - 2 * eigenvalue**3
            + 8 * eigenvalue**2
            - 6 * eigenvalue
            + 1
        )
        * (
            2 * cayley * eigenvalue
            - cayley
            + 2 * eigenvalue**3
            - 8 * eigenvalue**2
            + 6 * eigenvalue
            - 1
        )
    )
    twin = (
        sp.Rational(1, 16)
        * (cayley - 2 * eigenvalue**2 + 4 * eigenvalue - 1)
        * (cayley - 2 * eigenvalue**2 + 6 * eigenvalue - 3)
        * (cayley + 2 * eigenvalue**2 - 6 * eigenvalue + 3)
        * (cayley + 2 * eigenvalue**2 - 4 * eigenvalue + 1)
    )
    constant = (eigenvalue - 1) ** 2 * (eigenvalue**2 - 3 * eigenvalue + 1)
    return [first, twin, twin, constant]


def expected_block_characteristics(
    cayley: sp.Symbol, eigenvalue: sp.Symbol
) -> list[sp.Expr]:
    """Return the four exact characteristic laws in block order."""

    return _expected_block_characteristics(cayley, eigenvalue)


def _twin_intertwiner() -> sp.Matrix:
    """The exact signed permutation conjugating the two twin 8-blocks."""

    return sp.Matrix(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ]
    )


def exact_cayley_block_certificate() -> dict[str, object]:
    """Prove the constant block split and its determinant consequences."""

    information, cayley, sine = exact_cayley_information_family()
    eigenvalue = sp.symbols("lambda")
    components = exact_support_components(information)
    expected_characteristics = _expected_block_characteristics(cayley, eigenvalue)

    block_rows: list[dict[str, object]] = []
    reduced_characteristics: list[sp.Expr] = []
    determinants: list[sp.Expr] = []
    blocks: list[sp.Matrix] = []
    for component, expected in zip(components, expected_characteristics, strict=True):
        block = information.extract(component, component)
        characteristic = _circle_reduce(
            block.charpoly(eigenvalue).as_expr(), sine, cayley
        )
        determinant = sp.factor(characteristic.subs(eigenvalue, 0))
        reduced_characteristics.append(characteristic)
        determinants.append(determinant)
        blocks.append(block)
        block_rows.append(
            {
                "dimension": len(component),
                "bivector_planes": [list(SPIN8_PAIRS[index]) for index in component],
                "characteristic_polynomial": str(sp.factor(characteristic)),
                "determinant": str(determinant),
                "expected_characteristic_identity": sp.expand(characteristic - expected)
                == 0,
            }
        )

    intertwiner = _twin_intertwiner()
    twin_conjugacy = sp.zeros(8)
    for row in range(8):
        for column in range(8):
            twin_conjugacy[row, column] = _circle_reduce(
                (intertwiner * blocks[1] - blocks[2] * intertwiner)[row, column],
                sine,
                cayley,
            )

    # The first block has a constant unit-eigenvector: the difference of its
    # fifth and sixth local coordinates.  This certifies a genuine 1 + 7
    # invariant refinement rather than merely a factored polynomial.
    fixed_vector = sp.Matrix([0, 0, 0, 0, 1, -1, 0, 0])
    first_fixed_residual = (blocks[0] - sp.eye(8)) * fixed_vector

    characteristic_product = sp.prod(reduced_characteristics)
    full_characteristic = _circle_reduce(
        information.charpoly(eigenvalue).as_expr(), sine, cayley
    )
    determinant_product = sp.factor(sp.prod(determinants))
    expected_determinant = (
        (1 - cayley**2) ** 3 * (9 - cayley**2) ** 2 / sp.Integer(1024)
    )
    balanced_determinants = [sp.factor(value.subs(cayley, 0)) for value in determinants]
    pair_basis_invariance = {
        "positive_chiral": _pair_basis_invariance_certificate(1),
        "negative_chiral": _pair_basis_invariance_certificate(2),
    }

    passed = (
        [len(component) for component in components] == [8, 8, 8, 4]
        and all(row["expected_characteristic_identity"] for row in block_rows)
        and intertwiner.T * intertwiner == sp.eye(8)
        and twin_conjugacy == sp.zeros(8)
        and first_fixed_residual == sp.zeros(8, 1)
        and sp.expand(characteristic_product - full_characteristic) == 0
        and sp.expand(determinant_product - expected_determinant) == 0
        and balanced_determinants
        == [sp.Rational(1, 4), sp.Rational(9, 16), sp.Rational(9, 16), 1]
        and all(pair_basis_invariance.values())
    )
    return {
        "theorem": "exact constant invariant blocks for the balanced Cayley family",
        "quotient_relation": "s^2 = 1 - c^2",
        "block_dimensions": [len(component) for component in components],
        "blocks": block_rows,
        "twin_blocks_exactly_orthogonally_conjugate": twin_conjugacy == sp.zeros(8),
        "twin_intertwiner": [list(map(int, row)) for row in intertwiner.tolist()],
        "first_block_constant_unit_eigenvector": list(map(int, fixed_vector)),
        "first_block_constant_unit_eigenvector_verified": first_fixed_residual
        == sp.zeros(8, 1),
        "balanced_block_determinants": [str(value) for value in balanced_determinants],
        "global_determinant": str(determinant_product),
        "balanced_global_determinant": str(
            sp.factor(determinant_product.subs(cayley, 0))
        ),
        "same_view_pair_basis_invariance": pair_basis_invariance,
        "characteristic_product_identity": sp.expand(
            characteristic_product - full_characteristic
        )
        == 0,
        "determinant_product_identity": sp.expand(
            determinant_product - expected_determinant
        )
        == 0,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = exact_cayley_block_certificate()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("exact Cayley block certificate failed")


if __name__ == "__main__":
    main()
