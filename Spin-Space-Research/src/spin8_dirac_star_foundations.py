"""Exact structural foundations for the signed-star determinant certificate.

The large interpolation and Bernstein certificate in :mod:`spin8_dirac_star`
is conclusive only after four finite-dimensional facts are established:

* every star-boundary Jacobian has rank at most 25;
* the resulting sixth-power divisibilities combine in the circle quotient;
* signed triality symmetries leave exactly the trivial and orientation sectors;
* the interpolation grids span the resulting conservative degree spaces.

This module records those facts as exact rational/symbolic proof objects.
"""

from __future__ import annotations

import argparse
import itertools
import json
from math import prod
from pathlib import Path

import sympy as sp

from spin8_cayley_spectrum import (
    CAYLEY_TERMS,
    symbolic_query_projector,
    symbolic_triality_generators,
)
from spin8_dirac_star import (
    NODE_SETS,
    U,
    V,
    W,
    Z,
    polynomial_from_records,
    rational_circle,
)

VARIABLE_NAMES = ("a", "A", "d", "D", "g", "G", "c", "s")
TRIVIAL_PARITY = (0, 0, 0, 0, 0, 0, 0, 0)
ORIENTATION_PARITY = (1, 1, 1, 1, 1, 1, 1, 0)
SOURCE_ARTIFACT = (
    Path(__file__).resolve().parents[1] / "artifacts" / "spin8_dirac_star_20260804.json"
)


def _observation_block(
    view: int,
    state: list[sp.Expr],
    generators: list[list[list[list[sp.Rational]]]],
) -> sp.Matrix:
    return sp.Matrix(
        [
            [
                sum(
                    generators[view][generator][row][column] * state[column]
                    for column in range(8)
                )
                for generator in range(28)
            ]
            for row in range(8)
        ]
    )


def _character(signs: tuple[int, ...], mask: tuple[int, ...]) -> int:
    return prod(sign**exponent for sign, exponent in zip(signs, mask, strict=True))


def _group_closure(generators: list[tuple[int, ...]]) -> set[tuple[int, ...]]:
    identity = (1,) * len(generators[0])
    group = {identity}
    changed = True
    while changed:
        changed = False
        for left in tuple(group):
            for right in generators:
                product_signs = tuple(
                    first * second for first, second in zip(left, right, strict=True)
                )
                if product_signs not in group:
                    group.add(product_signs)
                    changed = True
    return group


def exact_star_parity_certificate() -> dict[str, object]:
    """Prove the trivial-plus-orientation invariant-module decomposition."""

    generators = symbolic_triality_generators()
    ambient_signs = [
        signs
        for signs in itertools.product((1, -1), repeat=8)
        if signs[0] == 1
        and all(
            prod(signs[index] for index in indices) == 1 for indices in CAYLEY_TERMS
        )
    ]

    # Common signed-diagonal Spin(7) actions on the star coordinates.
    parameter_generators = [
        (1, signs[1], 1, signs[2], 1, 1, signs[3], signs[4]) for signs in ambient_signs
    ]
    # Projective invariance of the three moving query blocks.
    parameter_generators.extend(
        (
            (-1, -1, 1, 1, 1, 1, 1, 1),
            (1, 1, -1, -1, 1, 1, 1, 1),
            (1, 1, 1, 1, -1, -1, 1, 1),
            # (G,c,s)->(-G,-c,-s) leaves G(c e3+s e4) unchanged.
            (1, 1, 1, 1, 1, -1, -1, -1),
        )
    )
    sign_group = _group_closure(parameter_generators)
    annihilator = {
        mask
        for mask in itertools.product((0, 1), repeat=8)
        if all(_character(signs, mask) == 1 for signs in sign_group)
    }

    def conjugation_character(
        matrix: list[list[sp.Rational]], signs: tuple[int, ...]
    ) -> int | None:
        characters = {
            signs[row] * signs[column]
            for row in range(8)
            for column in range(8)
            if matrix[row][column] != 0
        }
        return int(next(iter(characters))) if len(characters) == 1 else None

    common_adjoint_conjugacy = True
    for signs in ambient_signs:
        adjoint = tuple(
            conjugation_character(generators[0][index], signs) for index in range(28)
        )
        common_adjoint_conjugacy &= None not in adjoint and all(
            conjugation_character(generators[view][index], signs) == adjoint[index]
            for view in range(3)
            for index in range(28)
        )

    generic_state = list(sp.symbols("x0:8"))
    projective_query_invariance = all(
        symbolic_query_projector(view, generic_state, generators)
        == symbolic_query_projector(
            view, [-coordinate for coordinate in generic_state], generators
        )
        for view in range(3)
    )
    passed = (
        len(ambient_signs) == 8
        and len(sign_group) == 128
        and annihilator == {TRIVIAL_PARITY, ORIENTATION_PARITY}
        and common_adjoint_conjugacy
        and projective_query_invariance
    )
    return {
        "coordinate_order": list(VARIABLE_NAMES),
        "fixed_e0_cayley_diagonal_action_count": len(ambient_signs),
        "fixed_e0_cayley_diagonal_actions": [
            list(signs) for signs in sorted(ambient_signs)
        ],
        "parameter_sign_generators": [list(signs) for signs in parameter_generators],
        "generated_parameter_sign_group_order": len(sign_group),
        "invariant_parity_masks": [list(mask) for mask in sorted(annihilator)],
        "orientation_monomial": "a*A*d*D*g*G*c",
        "common_adjoint_conjugacy_verified": common_adjoint_conjugacy,
        "projective_query_invariance_verified": projective_query_invariance,
        "module_conclusion": (
            "In the circle quotient, every invariant polynomial is an even "
            "polynomial in u,v,w,z plus a*A*d*D*g*G*c times another even "
            "polynomial; no other Walsh character survives."
        ),
        "passed": passed,
    }


def exact_star_boundary_certificate() -> dict[str, object]:
    """Certify rank loss, quotient divisibility, and exact cubic order."""

    generators = symbolic_triality_generators()
    basis = [[sp.Integer(row == column) for column in range(8)] for row in range(8)]
    a, diagonal_a, d, diagonal_d, g, diagonal_g, cayley, sine = sp.symbols(
        "a A d D g G c s"
    )

    def state(*terms: tuple[sp.Expr, int]) -> list[sp.Expr]:
        return [
            sum(coefficient * basis[index][column] for coefficient, index in terms)
            for column in range(8)
        ]

    def jacobian(
        positive: list[sp.Expr],
        first_negative: list[sp.Expr],
        second_negative: list[sp.Expr],
    ) -> sp.Matrix:
        return sp.Matrix.vstack(
            _observation_block(0, basis[0], generators),
            _observation_block(1, basis[0], generators),
            _observation_block(1, positive, generators),
            _observation_block(2, first_negative, generators),
            _observation_block(2, second_negative, generators),
        )

    positive = state((a, 0), (diagonal_a, 1))
    first_negative = state((d, 0), (diagonal_d, 2))
    second_negative = state((g, 0), (diagonal_g * cayley, 3), (diagonal_g * sine, 4))
    negative_e0 = [-coordinate for coordinate in basis[0]]
    boundary_matrices = {
        "A=0": {
            "+": jacobian(basis[0], first_negative, second_negative),
            "-": jacobian(negative_e0, first_negative, second_negative),
        },
        "D=0": {
            "+": jacobian(positive, basis[0], second_negative),
            "-": jacobian(positive, negative_e0, second_negative),
        },
        "G=0": {
            "+": jacobian(positive, first_negative, basis[0]),
            "-": jacobian(positive, first_negative, negative_e0),
        },
    }

    branch_rows: list[dict[str, object]] = []
    for boundary, branches in boundary_matrices.items():
        for branch, matrix in branches.items():
            nullspace = matrix.nullspace()
            residual_zero = all(
                all(sp.factor(entry) == 0 for entry in matrix * vector)
                for vector in nullspace
            )
            branch_rows.append(
                {
                    "boundary": boundary,
                    "circle_branch": branch,
                    "symbolic_generic_rank": 28 - len(nullspace),
                    "universal_rank_upper_bound": 28 - len(nullspace),
                    "symbolic_nullity": len(nullspace),
                    "nullspace_residual_exact_zero": residual_zero,
                    "nullspace_vectors": [
                        [str(sp.factor(entry)) for entry in vector]
                        for vector in nullspace
                    ],
                }
            )

    circle_relations = (
        a**2 + diagonal_a**2 - 1,
        d**2 + diagonal_d**2 - 1,
        g**2 + diagonal_g**2 - 1,
        cayley**2 + sine**2 - 1,
    )
    circle_groebner = sp.groebner(
        circle_relations,
        a,
        d,
        g,
        cayley,
        diagonal_a,
        diagonal_d,
        diagonal_g,
        sine,
        order="lex",
    )
    leading_monomials = [
        list(polynomial.LM(order="lex").exponents)
        for polynomial in circle_groebner.polys
    ]
    expected_leading_monomials = [
        [2, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0],
    ]

    source = json.loads(SOURCE_ARTIFACT.read_text(encoding="utf-8"))
    normalized_even = polynomial_from_records(
        source["confirmation_node_set"]["even_coefficients"]
    ).as_expr()
    exact_order_witnesses = {
        "u=1,v=w=z=0": str(normalized_even.subs({U: 1, V: 0, W: 0, Z: 0})),
        "v=1,u=w=z=0": str(normalized_even.subs({U: 0, V: 1, W: 0, Z: 0})),
        "w=1,u=v=z=0": str(normalized_even.subs({U: 0, V: 0, W: 1, Z: 0})),
    }
    expected_witnesses = {
        "u=1,v=w=z=0": "25/2",
        "v=1,u=w=z=0": "75/2",
        "w=1,u=v=z=0": "75/2",
    }
    all_generators_skew = all(
        sp.Matrix(generators[view][plane]) + sp.Matrix(generators[view][plane]).T
        == sp.zeros(8)
        for view in range(3)
        for plane in range(28)
    )
    passed = (
        all(
            row["symbolic_generic_rank"] == 25
            and row["symbolic_nullity"] == 3
            and row["nullspace_residual_exact_zero"]
            for row in branch_rows
        )
        and leading_monomials == expected_leading_monomials
        and exact_order_witnesses == expected_witnesses
        and all_generators_skew
    )
    return {
        "query_view_order": ["vector", "positive", "positive", "negative", "negative"],
        "boundary_branches": branch_rows,
        "maximal_minor_lemma": (
            "If rank J(0)<=n-r for a polynomial m-by-n matrix J(t), every "
            "n-by-n minor is divisible by t^r: terms using fewer than r "
            "positive-order columns contain more than n-r columns from J(0) "
            "and vanish by multilinearity. Cauchy-Binet then makes det(J^T J) "
            "divisible by t^(2r)."
        ),
        "circle_quotient": {
            "leading_monomial_exponents": leading_monomials,
            "free_module_rank_over_diagonal_ring": 16,
            "both_branch_and_product_argument": (
                "Write each circle normal form as p(A)+a*q(A). The A=0 "
                "branches a=+1 and a=-1 give sixth-order vanishing of their "
                "sum and difference, hence A^6 divides p and q. Repeat for "
                "D and G. The distinct diagonal variables are coprime, so "
                "A^6*D^6*G^6=Delta^3 divides every coefficient."
            ),
        },
        "all_84_generators_exactly_skew": all_generators_skew,
        "raw_coordinate_pair_degree_upper_bound": 14,
        "post_division_pair_degree_upper_bound": 8,
        "even_squared_multidegree_upper_bound": [4, 4, 4, 7],
        "orientation_squared_multidegree_upper_bound": [3, 3, 3, 6],
        "exact_cubic_order_witnesses_for_1024_det_over_delta_cubed": (
            exact_order_witnesses
        ),
        "passed": passed,
    }


def exact_interpolation_contract() -> dict[str, object]:
    """Record the exact nodes and uniqueness conditions of both reconstructions."""

    rows: dict[str, object] = {}
    for name, nodes in NODE_SETS.items():
        spatial_parameters = list(nodes["u"])
        cayley_parameters = list(nodes["z"])
        spatial_pairs = [rational_circle(value) for value in spatial_parameters]
        cayley_pairs = [rational_circle(value) for value in cayley_parameters]
        spatial_squares = [sp.factor(pair[0] ** 2) for pair in spatial_pairs]
        cayley_squares = [sp.factor(pair[0] ** 2) for pair in cayley_pairs]
        rows[name] = {
            "spatial_circle_parameters": [str(value) for value in spatial_parameters],
            "cayley_circle_parameters": [str(value) for value in cayley_parameters],
            "spatial_squared_nodes": [str(value) for value in spatial_squares],
            "cayley_squared_nodes": [str(value) for value in cayley_squares],
            "spatial_nodes_distinct": len(set(spatial_squares)) == 5,
            "cayley_nodes_distinct": len(set(cayley_squares)) == 8,
            "all_gram_diagonals_nonzero": all(pair[1] != 0 for pair in spatial_pairs),
            "odd_subgrid_orientation_denominator_nonzero": all(
                pair[0] != 0 and pair[1] != 0 for pair in spatial_pairs[1:]
            )
            and all(pair[0] != 0 for pair in cayley_pairs[1:]),
        }

    discovery = NODE_SETS["discovery"]
    confirmation = NODE_SETS["confirmation"]
    shared_spatial = set(discovery["u"]) & set(confirmation["u"])
    shared_cayley = set(discovery["z"]) & set(confirmation["z"])
    passed = (
        all(
            row["spatial_nodes_distinct"]
            and row["cayley_nodes_distinct"]
            and row["all_gram_diagonals_nonzero"]
            and row["odd_subgrid_orientation_denominator_nonzero"]
            for row in rows.values()
        )
        and shared_spatial == {0}
        and shared_cayley == {0}
    )
    return {
        "node_sets": rows,
        "even_monomial_basis": "u^i v^j w^k z^l, 0<=i,j,k<=4, 0<=l<=7",
        "even_basis_dimension": 5**3 * 8,
        "odd_monomial_basis": "u^i v^j w^k z^l, 0<=i,j,k<=3, 0<=l<=6",
        "odd_basis_dimension": 4**3 * 7,
        "uniqueness": (
            "Distinct nodes make each one-dimensional Vandermonde matrix "
            "invertible; the tensor product therefore determines one unique "
            "polynomial in each conservative multidegree space."
        ),
        "grid_relation": (
            "The grids are independently chosen and nonidentical, but not "
            "strictly disjoint: each coordinate list shares its zero anchor, "
            "so the full tensor grids share exactly the all-zero node."
        ),
        "passed": passed,
    }


def run() -> dict[str, object]:
    parity = exact_star_parity_certificate()
    boundary = exact_star_boundary_certificate()
    interpolation = exact_interpolation_contract()
    return {
        "theorem_role": "signed-star finite-dimensional structural certificate",
        "parity": parity,
        "boundary_and_degree": boundary,
        "interpolation": interpolation,
        "passed": parity["passed"] and boundary["passed"] and interpolation["passed"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("signed-star structural foundation certificate failed")


if __name__ == "__main__":
    main()
