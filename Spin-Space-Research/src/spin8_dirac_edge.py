"""Exact Cayley-null edge-family certificate for the Dirac--Gram program."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import sympy as sp

from spin8_cayley_spectrum import (
    CAYLEY_TERMS,
    symbolic_query_projector,
    symbolic_triality_generators,
)
from spin8_dirac_star import rational_circle

VARIABLES = sp.symbols("u v r w")
U, V, R, W = VARIABLES

NODE_SETS = {
    "discovery": (
        sp.Rational(1, 10),
        sp.Rational(1, 5),
        sp.Rational(2, 5),
        sp.Rational(3, 5),
        sp.Rational(4, 5),
    ),
    "confirmation": (
        sp.Rational(1, 12),
        sp.Rational(1, 6),
        sp.Rational(1, 3),
        sp.Rational(1, 2),
        sp.Rational(2, 3),
    ),
}

HOLDOUT_PARAMETERS = (sp.Rational(1, 11), sp.Rational(3, 11))
SIGNS = tuple(itertools.product((1, -1), repeat=4))
TRIVIAL_CHARACTER = (0, 0, 0, 0)
ORIENTATION_CHARACTER = (1, 1, 1, 0)


def exact_walsh_symmetry_certificate() -> dict[str, object]:
    """Certify the diagonal Spin(7) symmetries forcing two Walsh sectors.

    A Cayley-preserving diagonal action fixing e0 has freely selectable signs
    (t1,t2,t4) on (e1,e2,e4).  Restoring positive Cholesky diagonals with the
    projective sign invariance P(x)=P(-x) transforms partial-correlation signs
    by (a,d,e,g) -> (t1*a,t2*d,t1*t2*e,t4*g).  The annihilator of this action is
    exactly the trivial character and ade.
    """

    generators = symbolic_triality_generators()
    ambient_signs = []
    for signs in itertools.product((1, -1), repeat=8):
        if signs[0] != 1:
            continue
        if all(
            signs[i] * signs[j] * signs[k] * signs[l] == 1
            for i, j, k, l in CAYLEY_TERMS
        ):
            ambient_signs.append(signs)
    projected = {(signs[1], signs[2], signs[4]) for signs in ambient_signs}
    induced = {(t1, t2, t1 * t2, t4) for t1, t2, t4 in projected}
    annihilator = {
        mask
        for mask in itertools.product((0, 1), repeat=4)
        if all(_character(signs, mask) == 1 for signs in induced)
    }
    expected_projected = set(itertools.product((1, -1), repeat=3))
    expected_annihilator = {TRIVIAL_CHARACTER, ORIENTATION_CHARACTER}

    def conjugation_character(
        matrix: list[list[sp.Rational]], signs: tuple[int, ...]
    ) -> int | None:
        characters = {
            signs[row] * signs[column]
            for row in range(8)
            for column in range(8)
            if matrix[row][column] != 0
        }
        if len(characters) != 1:
            return None
        return int(next(iter(characters)))

    triality_actions = []
    common_adjoint_conjugacy_verified = True
    for signs in ambient_signs:
        adjoint_signs = tuple(
            conjugation_character(generators[0][index], signs) for index in range(28)
        )
        action_verified = None not in adjoint_signs and all(
            conjugation_character(generators[view][index], signs)
            == adjoint_signs[index]
            for view in range(3)
            for index in range(28)
        )
        common_adjoint_conjugacy_verified &= action_verified
        triality_actions.append(
            {
                "vector_signs": list(signs),
                "positive_spinor_signs": list(signs),
                "negative_spinor_signs": list(signs),
                "adjoint_generator_signs": [int(value) for value in adjoint_signs],
                "common_adjoint_conjugacy_verified": action_verified,
            }
        )

    passed = (
        projected == expected_projected
        and annihilator == expected_annihilator
        and common_adjoint_conjugacy_verified
    )
    return {
        "fixed_e0_diagonal_cayley_symmetry_count": len(ambient_signs),
        "projected_sign_triples": [list(value) for value in sorted(projected)],
        "induced_partial_sign_group": [list(value) for value in sorted(induced)],
        "walsh_annihilator": [list(value) for value in sorted(annihilator)],
        "triality_representation_actions": triality_actions,
        "common_adjoint_conjugacy_verified": common_adjoint_conjugacy_verified,
        "passed": passed,
    }


def _symbolic_observation_block(
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


def exact_boundary_rank_certificate() -> dict[str, object]:
    """Prove the circle-constrained boundary defects forcing Delta^3."""

    generators = symbolic_triality_generators()
    basis = [[sp.Integer(row == column) for column in range(8)] for row in range(8)]
    a, diagonal_a, d, diagonal_d, e, diagonal_e, g, diagonal_g = sp.symbols(
        "a A d D e E g C"
    )

    def state(
        coefficients: tuple[sp.Expr, ...], indices: tuple[int, ...]
    ) -> list[sp.Expr]:
        return _vector(coefficients, indices, basis)

    def jacobian(
        positive: list[sp.Expr],
        first_negative: list[sp.Expr],
        second_negative: list[sp.Expr],
    ) -> sp.Matrix:
        return sp.Matrix.vstack(
            _symbolic_observation_block(0, basis[0], generators),
            _symbolic_observation_block(1, basis[0], generators),
            _symbolic_observation_block(1, positive, generators),
            _symbolic_observation_block(2, first_negative, generators),
            _symbolic_observation_block(2, second_negative, generators),
        )

    general_positive = state((a, diagonal_a), (0, 1))
    general_first_negative = state(
        (d, diagonal_d * e, diagonal_d * diagonal_e), (0, 1, 2)
    )
    general_second_negative = state((g, diagonal_g), (0, 4))
    negative_basis_zero = [-entry for entry in basis[0]]
    boundary_jacobians = {
        "A=0": {
            "+": jacobian(basis[0], general_first_negative, general_second_negative),
            "-": jacobian(
                negative_basis_zero, general_first_negative, general_second_negative
            ),
        },
        "D=0": {
            "+": jacobian(general_positive, basis[0], general_second_negative),
            "-": jacobian(
                general_positive, negative_basis_zero, general_second_negative
            ),
        },
        "E=0": {
            "+": jacobian(
                general_positive,
                state((d, diagonal_d), (0, 1)),
                general_second_negative,
            ),
            "-": jacobian(
                general_positive,
                state((d, -diagonal_d), (0, 1)),
                general_second_negative,
            ),
        },
        "C=0": {
            "+": jacobian(general_positive, general_first_negative, basis[0]),
            "-": jacobian(
                general_positive, general_first_negative, negative_basis_zero
            ),
        },
    }
    branch_rows = []
    for boundary, branches in boundary_jacobians.items():
        for branch, matrix in branches.items():
            nullspace = matrix.nullspace()
            residuals = [matrix * vector for vector in nullspace]
            branch_rows.append(
                {
                    "boundary": boundary,
                    "circle_branch": branch,
                    "symbolic_rank": 28 - len(nullspace),
                    "symbolic_nullity": len(nullspace),
                    "nullspace_residual_exact_zero": all(
                        all(sp.factor(entry) == 0 for entry in residual)
                        for residual in residuals
                    ),
                    "nullspace_vectors": [
                        [str(sp.factor(entry)) for entry in vector]
                        for vector in nullspace
                    ],
                    "guaranteed_minor_vanishing_order_lower_bound": len(nullspace),
                    "guaranteed_gram_determinant_vanishing_order_lower_bound": (
                        2 * len(nullspace)
                    ),
                }
            )
    rows = []
    for boundary in boundary_jacobians:
        branches = [row for row in branch_rows if row["boundary"] == boundary]
        rows.append(
            {
                "boundary": boundary,
                "circle_branches_verified": [row["circle_branch"] for row in branches],
                "symbolic_rank": min(row["symbolic_rank"] for row in branches),
                "symbolic_nullity": max(row["symbolic_nullity"] for row in branches),
                "nullspace_residual_exact_zero": all(
                    row["nullspace_residual_exact_zero"] for row in branches
                ),
                "guaranteed_minor_vanishing_order_lower_bound": min(
                    row["guaranteed_minor_vanishing_order_lower_bound"]
                    for row in branches
                ),
                "guaranteed_gram_determinant_vanishing_order_lower_bound": min(
                    row["guaranteed_gram_determinant_vanishing_order_lower_bound"]
                    for row in branches
                ),
            }
        )
    passed = all(
        row["symbolic_rank"] == 25
        and row["symbolic_nullity"] == 3
        and row["nullspace_residual_exact_zero"]
        for row in branch_rows
    )
    circle_relations = (
        a**2 + diagonal_a**2 - 1,
        d**2 + diagonal_d**2 - 1,
        e**2 + diagonal_e**2 - 1,
        g**2 + diagonal_g**2 - 1,
    )
    circle_groebner = sp.groebner(
        circle_relations,
        a,
        d,
        e,
        g,
        diagonal_a,
        diagonal_d,
        diagonal_e,
        diagonal_g,
        order="lex",
    )
    leading_monomial_exponents = [
        [
            int(exponent)
            for exponent in polynomial.LM(order=circle_groebner.order).exponents
        ]
        for polynomial in circle_groebner.polys
    ]
    expected_leading_monomial_exponents = [
        [2, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0],
    ]
    normal_form_verified = (
        leading_monomial_exponents == expected_leading_monomial_exponents
    )
    module_basis = [
        "".join(
            symbol
            for symbol, exponent in zip(("a", "d", "e", "g"), exponents, strict=True)
            if exponent
        )
        or "1"
        for exponents in itertools.product((0, 1), repeat=4)
    ]
    varying_query_ranks = [
        int(_symbolic_observation_block(view, basis[0], generators).rank())
        for view in (1, 2, 2)
    ]
    exact_skew_symmetry_verified = all(
        sp.Matrix(generators[view][plane]) + sp.Matrix(generators[view][plane]).T
        == sp.zeros(8)
        for view in range(3)
        for plane in range(28)
    )
    raw_degree_bound = 2 * max(varying_query_ranks)
    post_division_degree_bound = raw_degree_bound - 6
    degree_certificate = {
        "varying_query_block_ranks": varying_query_ranks,
        "maintained_generator_count": 84,
        "all_generators_exactly_skew_symmetric": exact_skew_symmetry_verified,
        "universal_query_rank_upper_bound": 7,
        "raw_coordinate_pair_degree_upper_bound": raw_degree_bound,
        "post_diagonal_sixth_power_degree_upper_bound": post_division_degree_bound,
        "even_sector_squared_multidegree_upper_bound": [4, 4, 4, 4],
        "adeAD_sector_squared_multidegree_upper_bound": [3, 3, 3, 4],
        "derivation": (
            "A varying query projector J(x)^T J(x) is quadratic in its row "
            "coordinates and has rank at most seven, so a determinant term "
            "uses it at most seven times: pair degree <=14. Division by the "
            "corresponding diagonal sixth power leaves pair degree <=8. The "
            "even sector therefore has squared degree <=4; removing aA, dD, "
            "and e from the adeAD sector leaves squared degrees <=3,<=3,<=3, "
            "while g remains <=4."
        ),
        "passed": (varying_query_ranks == [7, 7, 7] and exact_skew_symmetry_verified),
    }
    passed = bool(passed and normal_form_verified and degree_certificate["passed"])
    return {
        "coordinate_ring": (
            "Q[a,A,d,D,e,E,g,C]/" "(a^2+A^2-1,d^2+D^2-1,e^2+E^2-1,g^2+C^2-1)"
        ),
        "analytic_boundary_argument": (
            "Near A=0 use a=sigma*sqrt(1-A^2) on both sigma=+1 and sigma=-1 "
            "branches, and analogously for D,E,C. "
            "The state/Jacobian perturbation is O(A), rank(J(0))<=25, so every "
            "28-row minor is O(A^3) and det(J^T J) is O(A^6). Because the "
            "restricted determinant is polynomial/analytic in the quotient "
            "coordinate, A^6=(1-u)^3 divides it. Repeat independently at all "
            "four circle boundaries."
        ),
        "circle_normal_form": {
            "groebner_variable_order": ["a", "d", "e", "g", "A", "D", "E", "C"],
            "groebner_leading_monomial_exponents": leading_monomial_exponents,
            "free_module_basis_over_Q_A_D_E_C": module_basis,
            "free_module_rank": len(module_basis),
            "both_branch_lemma": (
                "Write P=p(A)+a*q(A) in the free basis {1,a}. On the branches "
                "a=+sqrt(1-A^2) and a=-sqrt(1-A^2), order >=6 for both P "
                "values implies order >=6 separately for p and q by sum and "
                "difference; sqrt(1-A^2) is a unit at A=0. Thus A^6 divides "
                "both normal-form coefficients."
            ),
            "product_divisibility": (
                "The 16 normal-form coefficients lie in Q[A,D,E,C]. Applying "
                "the two-branch lemma in each circle pair makes every "
                "coefficient divisible by A^6,D^6,E^6,C^6. These are distinct "
                "polynomial variables, hence pairwise coprime, so their "
                "product divides every coefficient and therefore P in the "
                "circle quotient."
            ),
            "passed": normal_form_verified and len(module_basis) == 16,
        },
        "boundary_rows": rows,
        "boundary_branch_rows": branch_rows,
        "guaranteed_minor_vanishing_order_lower_bound": 3,
        "guaranteed_determinant_vanishing_order_lower_bound": 6,
        "divisibility": "A^6 D^6 E^6 C^6 = Delta^3",
        "degree_certificate": degree_certificate,
        "conservative_even_squared_multidegree": degree_certificate[
            "even_sector_squared_multidegree_upper_bound"
        ],
        "conservative_odd_quotient_multidegree": degree_certificate[
            "adeAD_sector_squared_multidegree_upper_bound"
        ],
        "passed": passed,
    }


def _vector(
    coefficients: tuple[sp.Expr, ...],
    indices: tuple[int, ...],
    basis: list[list[sp.Integer]],
) -> list[sp.Expr]:
    return [
        sum(
            coefficient * basis[index][column]
            for coefficient, index in zip(coefficients, indices, strict=True)
        )
        for column in range(8)
    ]


def _character(signs: tuple[int, ...], mask: tuple[int, ...]) -> int:
    return int(sp.prod(sign**power for sign, power in zip(signs, mask, strict=True)))


def _tensor_interpolate(
    values: dict[tuple[int, int, int, int], sp.Expr],
    nodes: list[sp.Expr],
) -> sp.Poly:
    count = len(nodes)
    level_three = {
        (left, middle, residual): sp.interpolate(
            [
                (nodes[index], values[left, middle, residual, index])
                for index in range(count)
            ],
            W,
        )
        for left in range(count)
        for middle in range(count)
        for residual in range(count)
    }
    level_two = {
        (left, middle): sp.interpolate(
            [
                (nodes[index], level_three[left, middle, index])
                for index in range(count)
            ],
            R,
        )
        for left in range(count)
        for middle in range(count)
    }
    level_one = {
        left: sp.interpolate(
            [(nodes[index], level_two[left, index]) for index in range(count)],
            V,
        )
        for left in range(count)
    }
    return sp.Poly(
        sp.interpolate([(nodes[index], level_one[index]) for index in range(count)], U),
        *VARIABLES,
    )


def polynomial_records(polynomial: sp.Poly) -> list[dict[str, object]]:
    return [
        {"powers": list(powers), "coefficient": str(coefficient)}
        for powers, coefficient in polynomial.terms()
    ]


def polynomial_from_records(records: list[dict[str, object]]) -> sp.Poly:
    expression = sp.Integer(0)
    for record in records:
        monomial = sp.prod(
            variable ** int(power)
            for variable, power in zip(VARIABLES, record["powers"], strict=True)
        )
        expression += sp.Rational(record["coefficient"]) * monomial
    return sp.Poly(expression, *VARIABLES)


def records_hash(records: list[dict[str, object]]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def bernstein_records(polynomial: sp.Poly) -> tuple[tuple[int, ...], list[str]]:
    degrees = tuple(int(polynomial.degree(variable)) for variable in VARIABLES)
    coefficients = np.empty(tuple(degree + 1 for degree in degrees), dtype=object)
    coefficients.fill(sp.Integer(0))
    for powers, coefficient in polynomial.terms():
        coefficients[powers] = coefficient

    for axis, degree in enumerate(degrees):
        transform = [
            [
                (
                    sp.Rational(sp.binomial(row, column), sp.binomial(degree, column))
                    if column <= row
                    else sp.Integer(0)
                )
                for column in range(degree + 1)
            ]
            for row in range(degree + 1)
        ]
        coefficients = np.moveaxis(coefficients, axis, 0)
        shape = coefficients.shape
        flattened = coefficients.reshape((degree + 1, -1))
        transformed = np.empty_like(flattened)
        for row in range(degree + 1):
            for column in range(flattened.shape[1]):
                transformed[row, column] = sum(
                    transform[row][source] * flattened[source, column]
                    for source in range(degree + 1)
                )
        coefficients = np.moveaxis(transformed.reshape(shape), 0, axis)
    return degrees, [str(sp.factor(value)) for value in coefficients.flat]


def _normalized_determinants(
    pairs: tuple[tuple[sp.Expr, sp.Expr], ...],
    generators: list[list[list[list[sp.Rational]]]],
    basis: list[list[sp.Integer]],
    fixed: sp.Matrix,
    signs_to_evaluate: tuple[tuple[int, ...], ...] = SIGNS,
) -> dict[tuple[int, ...], sp.Expr]:
    (a, diagonal_a), (d, diagonal_d), (e, diagonal_e), (g, diagonal_g) = pairs
    delta = diagonal_a**2 * diagonal_d**2 * diagonal_e**2 * diagonal_g**2
    results: dict[tuple[int, ...], sp.Expr] = {}
    for signs in signs_to_evaluate:
        signed_a, signed_d, signed_e, signed_g = (
            sign * value for sign, value in zip(signs, (a, d, e, g), strict=True)
        )
        positive = symbolic_query_projector(
            1,
            _vector((signed_a, diagonal_a), (0, 1), basis),
            generators,
        )
        first_negative = symbolic_query_projector(
            2,
            _vector(
                (
                    signed_d,
                    diagonal_d * signed_e,
                    diagonal_d * diagonal_e,
                ),
                (0, 1, 2),
                basis,
            ),
            generators,
        )
        second_negative = symbolic_query_projector(
            2,
            _vector((signed_g, diagonal_g), (0, 4), basis),
            generators,
        )
        information = fixed + positive + first_negative + second_negative
        results[signs] = sp.factor(
            1024 * information.det(method="domain-ge") / delta**3
        )
    return results


def _walsh_coefficients(
    determinants: dict[tuple[int, ...], sp.Expr],
) -> dict[tuple[int, ...], sp.Expr]:
    return {
        mask: sp.factor(
            sum(
                determinant * _character(signs, mask)
                for signs, determinant in determinants.items()
            )
            / 16
        )
        for mask in itertools.product((0, 1), repeat=4)
    }


def reconstruct_edge_polynomials(
    node_set: str,
) -> tuple[sp.Poly, sp.Poly, dict[str, object]]:
    parameters = NODE_SETS[node_set]
    pairs = [rational_circle(value) for value in parameters]
    squares = [sp.factor(value**2) for value, _ in pairs]
    generators = symbolic_triality_generators()
    basis = [[sp.Integer(row == column) for column in range(8)] for row in range(8)]
    fixed = symbolic_query_projector(
        0, basis[0], generators
    ) + symbolic_query_projector(1, basis[0], generators)
    even_values: dict[tuple[int, int, int, int], sp.Expr] = {}
    odd_values: dict[tuple[int, int, int, int], sp.Expr] = {}
    unexpected: set[tuple[int, ...]] = set()

    count = len(pairs)
    reconstruction_signs = (SIGNS[0], (-1, 1, 1, 1))
    for indices in itertools.product(range(count), repeat=4):
        selected = tuple(pairs[index] for index in indices)
        determinants = _normalized_determinants(
            selected, generators, basis, fixed, reconstruction_signs
        )
        (a, diagonal_a), (d, diagonal_d), (e, _), _ = selected
        positive = determinants[reconstruction_signs[0]]
        negative = determinants[reconstruction_signs[1]]
        even_values[indices] = sp.factor((positive + negative) / 2)
        odd_values[indices] = sp.factor(
            ((positive - negative) / 2) / (a * d * e * diagonal_a * diagonal_d)
        )

    anchor_walsh = _walsh_coefficients(
        _normalized_determinants(
            tuple(pairs[index] for index in range(4)), generators, basis, fixed
        )
    )
    unexpected.update(
        mask
        for mask, coefficient in anchor_walsh.items()
        if mask not in (TRIVIAL_CHARACTER, ORIENTATION_CHARACTER) and coefficient != 0
    )

    even = _tensor_interpolate(even_values, squares)
    odd = _tensor_interpolate(odd_values, squares)
    return (
        even,
        odd,
        {
            "evaluated_sign_orientations_per_node": 2,
            "complete_sign_anchor_orientations": 16,
            "interpolation_nodes": count**4,
            "unexpected_nonzero_walsh_characters": [
                list(mask) for mask in sorted(unexpected)
            ],
        },
    )


def certificate_from_polynomials(even: sp.Poly, odd: sp.Poly) -> dict[str, object]:
    margin = sp.Poly(sp.expand(81 - even.as_expr()), *VARIABLES)
    discriminant = sp.Poly(
        sp.expand(
            margin.as_expr() ** 2 - U * V * R * (1 - U) * (1 - V) * odd.as_expr() ** 2
        ),
        *VARIABLES,
    )
    margin_degrees, margin_bernstein = bernstein_records(margin)
    discriminant_degrees, discriminant_bernstein = bernstein_records(discriminant)
    margin_shape = tuple(degree + 1 for degree in margin_degrees)
    discriminant_shape = tuple(degree + 1 for degree in discriminant_degrees)
    margin_array = np.asarray(margin_bernstein, dtype=object).reshape(margin_shape)
    discriminant_array = np.asarray(discriminant_bernstein, dtype=object).reshape(
        discriminant_shape
    )

    def zero_indices(coefficients: np.ndarray) -> list[list[int]]:
        return [
            [int(index) for index in indices]
            for indices in np.argwhere(coefficients == "0")
        ]

    def minimum_positive(coefficients: list[str]) -> str:
        positive = [
            sp.Rational(value) for value in coefficients if sp.Rational(value) > 0
        ]
        return str(min(positive))

    even_records = polynomial_records(even)
    odd_records = polynomial_records(odd)
    return {
        "even_degrees": [int(even.degree(variable)) for variable in VARIABLES],
        "odd_degrees": [int(odd.degree(variable)) for variable in VARIABLES],
        "even_term_count": len(even.terms()),
        "odd_term_count": len(odd.terms()),
        "even_coefficients": even_records,
        "odd_coefficients": odd_records,
        "even_coefficients_sha256": records_hash(even_records),
        "odd_coefficients_sha256": records_hash(odd_records),
        "margin_bernstein_degrees": list(margin_degrees),
        "margin_bernstein_coefficients": margin_bernstein,
        "margin_bernstein_negative_count": sum(
            1 for value in margin_bernstein if sp.Rational(value) < 0
        ),
        "margin_bernstein_zero_count": margin_bernstein.count("0"),
        "margin_bernstein_zero_indices": zero_indices(margin_array),
        "margin_bernstein_minimum_positive": minimum_positive(margin_bernstein),
        "orientation_discriminant_bernstein_degrees": list(discriminant_degrees),
        "orientation_discriminant_bernstein_coefficients": discriminant_bernstein,
        "orientation_discriminant_bernstein_negative_count": sum(
            1 for value in discriminant_bernstein if sp.Rational(value) < 0
        ),
        "orientation_discriminant_bernstein_zero_count": discriminant_bernstein.count(
            "0"
        ),
        "orientation_discriminant_bernstein_zero_indices": zero_indices(
            discriminant_array
        ),
        "orientation_discriminant_bernstein_minimum_positive": minimum_positive(
            discriminant_bernstein
        ),
    }


def exact_holdout_certificate(even: sp.Poly, odd: sp.Poly) -> dict[str, object]:
    generators = symbolic_triality_generators()
    basis = [[sp.Integer(row == column) for column in range(8)] for row in range(8)]
    fixed = symbolic_query_projector(
        0, basis[0], generators
    ) + symbolic_query_projector(1, basis[0], generators)
    comparisons = 0
    maximum_error = sp.Integer(0)
    for parameters in itertools.product(HOLDOUT_PARAMETERS, repeat=4):
        pairs = tuple(rational_circle(value) for value in parameters)
        determinants = _normalized_determinants(pairs, generators, basis, fixed)
        substitutions = {
            variable: pair[0] ** 2
            for variable, pair in zip(VARIABLES, pairs, strict=True)
        }
        (a, diagonal_a), (d, diagonal_d), (e, _), _ = pairs
        amplitude = a * d * e * diagonal_a * diagonal_d
        for signs, observed in determinants.items():
            predicted = sp.factor(
                even.as_expr().subs(substitutions)
                + _character(signs, ORIENTATION_CHARACTER)
                * amplitude
                * odd.as_expr().subs(substitutions)
            )
            error = sp.factor(observed - predicted)
            if error != 0:
                maximum_error = max(maximum_error, abs(error))
            comparisons += 1
    return {
        "magnitude_frames": 16,
        "sign_orientations_per_frame": 16,
        "exact_determinant_comparisons": comparisons,
        "maximum_exact_error": str(maximum_error),
        "passed": comparisons == 256 and maximum_error == 0,
    }


def verify_report(report: dict[str, object]) -> bool:
    discovery = report["discovery_node_set"]
    confirmation = report["confirmation_node_set"]
    for certificate in (discovery, confirmation):
        if (
            records_hash(certificate["even_coefficients"])
            != certificate["even_coefficients_sha256"]
        ):
            return False
        if (
            records_hash(certificate["odd_coefficients"])
            != certificate["odd_coefficients_sha256"]
        ):
            return False
        if certificate["parity_audit"]["unexpected_nonzero_walsh_characters"]:
            return False
        reconstructed = certificate_from_polynomials(
            polynomial_from_records(certificate["even_coefficients"]),
            polynomial_from_records(certificate["odd_coefficients"]),
        )
        for key, value in reconstructed.items():
            if certificate.get(key) != value:
                return False
        for prefix in ("margin", "orientation_discriminant"):
            coefficients = certificate[f"{prefix}_bernstein_coefficients"]
            if any(sp.Rational(value) < 0 for value in coefficients):
                return False
            degrees = certificate[f"{prefix}_bernstein_degrees"]
            shape = tuple(int(degree) + 1 for degree in degrees)
            coefficient_array = np.asarray(coefficients, dtype=object).reshape(shape)
            expected_zero_indices = [
                [int(index) for index in indices]
                for indices in np.argwhere(coefficient_array == "0")
            ]
            if certificate[f"{prefix}_bernstein_zero_indices"] != expected_zero_indices:
                return False
            positive = [
                sp.Rational(value) for value in coefficients if sp.Rational(value) > 0
            ]
            if certificate[f"{prefix}_bernstein_minimum_positive"] != str(
                min(positive)
            ):
                return False
    fresh_symmetry = exact_walsh_symmetry_certificate()
    fresh_degree_divisibility = exact_boundary_rank_certificate()
    maps_match = bool(
        discovery["even_coefficients"] == confirmation["even_coefficients"]
        and discovery["odd_coefficients"] == confirmation["odd_coefficients"]
        and discovery["even_coefficients_sha256"]
        == confirmation["even_coefficients_sha256"]
        and discovery["odd_coefficients_sha256"]
        == confirmation["odd_coefficients_sha256"]
    )
    derived_pass = bool(
        fresh_symmetry["passed"]
        and fresh_degree_divisibility["passed"]
        and maps_match
        and confirmation["even_degrees"] == [3, 3, 3, 3]
        and confirmation["odd_degrees"] == [2, 2, 2, 3]
        and report["off_grid_exact_holdouts"]["passed"]
    )
    return bool(
        report["exact_walsh_symmetry"] == fresh_symmetry
        and report["exact_degree_divisibility"] == fresh_degree_divisibility
        and report["coefficient_maps_match"] == maps_match
        and derived_pass
        and report["edge_family_theorem_proved"] == derived_pass
        and report["passed"] == derived_pass
        and not report["global_dirac_gram_theorem_proved"]
    )


def verify_artifact(path: Path) -> bool:
    if not path.is_file():
        return False
    report = json.loads(path.read_text(encoding="utf-8"))
    if not verify_report(report):
        return False
    confirmation = report["confirmation_node_set"]
    even = polynomial_from_records(confirmation["even_coefficients"])
    odd = polynomial_from_records(confirmation["odd_coefficients"])
    return exact_holdout_certificate(even, odd) == report["off_grid_exact_holdouts"]


def run() -> dict[str, object]:
    symmetry = exact_walsh_symmetry_certificate()
    degree_divisibility = exact_boundary_rank_certificate()
    discovery_even, discovery_odd, discovery_parity = reconstruct_edge_polynomials(
        "discovery"
    )
    confirmation_even, confirmation_odd, confirmation_parity = (
        reconstruct_edge_polynomials("confirmation")
    )
    discovery = certificate_from_polynomials(discovery_even, discovery_odd)
    confirmation = certificate_from_polynomials(confirmation_even, confirmation_odd)
    discovery["parity_audit"] = discovery_parity
    confirmation["parity_audit"] = confirmation_parity
    coefficient_maps_match = (
        discovery["even_coefficients"] == confirmation["even_coefficients"]
        and discovery["odd_coefficients"] == confirmation["odd_coefficients"]
        and discovery["even_coefficients_sha256"]
        == confirmation["even_coefficients_sha256"]
        and discovery["odd_coefficients_sha256"]
        == confirmation["odd_coefficients_sha256"]
    )
    holdouts = exact_holdout_certificate(confirmation_even, confirmation_odd)
    passed = bool(
        symmetry["passed"]
        and degree_divisibility["passed"]
        and coefficient_maps_match
        and discovery_parity["unexpected_nonzero_walsh_characters"] == []
        and confirmation_parity["unexpected_nonzero_walsh_characters"] == []
        and confirmation["even_degrees"] == [3, 3, 3, 3]
        and confirmation["odd_degrees"] == [2, 2, 2, 3]
        and confirmation["margin_bernstein_negative_count"] == 0
        and confirmation["orientation_discriminant_bernstein_negative_count"] == 0
        and holdouts["passed"]
    )
    return {
        "experiment": "Spin8 exact Cayley-null edge-family Dirac--Gram theorem",
        "exact_walsh_symmetry": symmetry,
        "exact_degree_divisibility": degree_divisibility,
        "discovery_node_set": discovery,
        "confirmation_node_set": confirmation,
        "coefficient_maps_match": coefficient_maps_match,
        "off_grid_exact_holdouts": holdouts,
        "edge_family_theorem_proved": passed,
        "global_dirac_gram_theorem_proved": False,
        "passed": passed,
    }


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
                "coefficient_maps_match": report["coefficient_maps_match"],
                "off_grid_exact_holdouts": report["off_grid_exact_holdouts"],
                "edge_family_theorem_proved": report["edge_family_theorem_proved"],
                "global_dirac_gram_theorem_proved": False,
                "passed": report["passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
