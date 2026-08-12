"""Final-residual bridge for the unrestricted Spin(8) Dirac--Gram gate.

The complete lower-triangular chart has seven signed partial-correlation
coordinates ``(a,d,e,g,h,i,c)``.  The maintained two-edge theorem fixes
``h=0``.  This module adds two deliberately separate layers:

* an exact theorem on the full ``h``-extension of the former orthonormal
  equality slice ``a=d=e=g=i=0``;
* GPU/CPU counterexample searches on the complete seven-coordinate interior.

The numerical layer can falsify the unrestricted conjecture.  It cannot
promote it to a theorem.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import psutil
import sympy as sp
import torch
from torch import nn

from spin8_cayley_spectrum import (
    CAYLEY_TERMS,
    balanced_frame_information,
    symbolic_query_projector,
    symbolic_triality_generators,
)
from spin8_dirac_edge import (
    _character,
    _symbolic_observation_block,
    exact_walsh_symmetry_certificate,
)
from spin8_dirac_gram import _bernstein_coefficients
from spin8_dirac_one_edge import _symbolic_vector
from spin8_dirac_star import rational_circle
from spin8_resource_limits import constrain_current_process
from spin8_triality import torch_triality_generators

PARAMETER_NAMES = ("a", "d", "e", "g", "h", "i", "c")
FULL_CHART_PARAMETER_ORDER = (
    "a",
    "A",
    "d",
    "D",
    "e",
    "E",
    "g",
    "G",
    "h",
    "H",
    "i",
    "I",
    "c",
    "s",
)
FULL_LOWER_INDICES = tuple(range(0, 14, 2))
FULL_COMPLEMENT_INDICES = tuple(range(1, 14, 2))
GRAM_PARAMETER_COUNT = 6


def _symbolic_basis() -> list[list[sp.Integer]]:
    return [[sp.Integer(row == column) for column in range(8)] for row in range(8)]


def exact_final_residual_equality_slice_certificate() -> dict[str, object]:
    """Prove the full final-residual slice, not merely its quadratic jet.

    The slice is ``a=d=e=g=i=0``.  Put ``r=h**2`` and ``z=c**2``.  Exact
    determinant elimination gives

        target - 1024 det(I)/Delta**3 = r (1-z)**3 q(r,z),

    where every tensor-product Bernstein coefficient of ``q`` on the unit
    square is strictly positive.
    """

    generators = symbolic_triality_generators()
    basis = _symbolic_basis()
    h, h_complement, cayley, sine = sp.symbols("h H c s", real=True)
    final_probe = [
        h * basis[1][column]
        + h_complement * (cayley * basis[3][column] + sine * basis[4][column])
        for column in range(8)
    ]
    information = (
        symbolic_query_projector(0, basis[0], generators)
        + symbolic_query_projector(1, basis[0], generators)
        + symbolic_query_projector(1, basis[1], generators)
        + symbolic_query_projector(2, basis[2], generators)
        + symbolic_query_projector(2, final_probe, generators)
    )
    determinant = sp.factor(information.det(method="domain-ge"))

    first_factor = (
        10 * h_complement**4 * cayley**4
        + 22 * h_complement**4 * cayley**2 * sine**2
        + 12 * h_complement**4 * sine**4
        + 15 * h_complement**2 * cayley**2 * h**2
        + 16 * h_complement**2 * cayley**2
        + 17 * h_complement**2 * h**2 * sine**2
        + 18 * h_complement**2 * sine**2
        + 6 * h**4
        + 13 * h**2
        + 6
    )
    second_factor = (
        15 * h_complement**4 * cayley**4
        + 33 * h_complement**4 * cayley**2 * sine**2
        + 18 * h_complement**4 * sine**4
        + 25 * h_complement**2 * cayley**2 * h**2
        + 14 * h_complement**2 * cayley**2
        + 28 * h_complement**2 * h**2 * sine**2
        + 15 * h_complement**2 * sine**2
        + 10 * h**4
        + 11 * h**2
        + 3
    )
    expected_determinant = (
        h_complement**6 * sine**6 * first_factor * second_factor / 16384
    )
    determinant_identity = sp.expand(determinant - expected_determinant) == 0

    r, z = sp.symbols("r z", nonnegative=True)
    reduced_first = sp.factor(
        first_factor.subs(
            {
                h**2: r,
                h_complement**2: 1 - r,
                cayley**2: z,
                sine**2: 1 - z,
            },
            simultaneous=True,
        )
    )
    reduced_second = sp.factor(
        second_factor.subs(
            {
                h**2: r,
                h_complement**2: 1 - r,
                cayley**2: z,
                sine**2: 1 - z,
            },
            simultaneous=True,
        )
    )
    expected_first = r**2 + 4 * r * z - 12 * r - 4 * z + 36
    expected_second = 4 * (r * z - 3 * r - z + 9)
    factor_reduction_identity = bool(
        sp.expand(reduced_first - expected_first) == 0
        and sp.expand(reduced_second - expected_second) == 0
    )

    normalized_determinant = sp.factor(
        (1 - z) ** 3 * reduced_first * reduced_second / 16
    )
    target = (1 - z) ** 3 * (9 - z) ** 2
    q = sp.factor(
        (
            -(r**2) * z
            + 3 * r**2
            - 4 * r * z**2
            + 25 * r * z
            - 45 * r
            + 8 * z**2
            - 96 * z
            + 216
        )
        / 4
    )
    gap = sp.factor(target - normalized_determinant)
    gap_identity = sp.expand(gap - r * (1 - z) ** 3 * q) == 0
    bernstein = _bernstein_coefficients(q, (r, z))
    flat_bernstein = [value for row in bernstein for value in row]
    minimum_bernstein = min(flat_bernstein)
    passed = bool(
        determinant_identity
        and factor_reduction_identity
        and gap_identity
        and minimum_bernstein > 0
    )
    return {
        "experiment": "exact final-residual equality-slice theorem",
        "slice": "a=d=e=g=i=0; r=h^2; z=c^2",
        "determinant_factorization": str(expected_determinant),
        "determinant_identity_verified": determinant_identity,
        "reduced_first_factor": str(reduced_first),
        "reduced_second_factor": str(reduced_second),
        "factor_reduction_identity_verified": factor_reduction_identity,
        "normalized_gap": str(gap),
        "strict_residual_polynomial": str(q),
        "gap_identity_verified": gap_identity,
        "bernstein_degree": [2, 2],
        "bernstein_coefficients": [[str(value) for value in row] for row in bernstein],
        "minimum_bernstein_coefficient": str(minimum_bernstein),
        "equality_set": "h=0 or c^2=1",
        "scope_boundary": (
            "This proves every physical h on the former equality slice. It does "
            "not control interactions with a,d,e,g,i."
        ),
        "passed": passed,
    }


def exact_chart_invariants_certificate() -> dict[str, object]:
    """Check the complete Cholesky Gram and Cayley identities symbolically."""

    basis = _symbolic_basis()
    pairs = [
        sp.symbols(f"{name} {name.upper()}", real=True) for name in PARAMETER_NAMES
    ]
    (a, A), (d, D), (e, E), (g, G), (h, H), (i, I), (c, S) = pairs
    frame = sp.Matrix(
        [
            basis[0],
            [a * basis[0][k] + A * basis[1][k] for k in range(8)],
            [
                d * basis[0][k] + D * (e * basis[1][k] + E * basis[2][k])
                for k in range(8)
            ],
            [
                g * basis[0][k]
                + G
                * (
                    h * basis[1][k]
                    + H * (i * basis[2][k] + I * (c * basis[3][k] + S * basis[4][k]))
                )
                for k in range(8)
            ],
        ]
    )
    gram_determinant = sp.factor((frame * frame.T).det())
    expected_gram = A**2 * D**2 * E**2 * G**2 * H**2 * I**2
    gram_circle_remainder = sp.factor(
        gram_determinant.subs(S**2, 1 - c**2) - expected_gram
    )
    cayley = sp.factor(
        sum(
            coefficient * frame[:, list(columns)].det()
            for columns, coefficient in CAYLEY_TERMS.items()
        )
    )
    expected_cayley = A * D * E * G * H * I * c
    return {
        "parameter_order": list(PARAMETER_NAMES),
        "gram_determinant": str(gram_determinant),
        "expected_gram_determinant": str(expected_gram),
        "gram_circle_remainder": str(gram_circle_remainder),
        "gram_identity_verified": gram_circle_remainder == 0,
        "cayley_value": str(cayley),
        "expected_cayley_value": str(expected_cayley),
        "cayley_identity_verified": sp.expand(cayley - expected_cayley) == 0,
        "normalized_cayley_coordinate": "c on every full-rank chart point",
        "passed": bool(
            gram_circle_remainder == 0 and sp.expand(cayley - expected_cayley) == 0
        ),
    }


@lru_cache(maxsize=1)
def exact_full_chart_sign_certificate() -> dict[str, object]:
    """Derive the complete seven-circle parity quotient from triality.

    The calculation includes independent signs for every lower Cholesky
    coordinate and its circle complement.  It solves the induced sign action
    in the three triality representations, including the harmless projective
    sign of each probe, and then computes the annihilator exactly.
    """

    base = exact_walsh_symmetry_certificate()
    induced: set[tuple[int, ...]] = set()
    for action in base["triality_representation_actions"]:
        positive_signs = action["positive_spinor_signs"]
        negative_signs = action["negative_spinor_signs"]
        for (
            positive_gauge,
            first_negative_gauge,
            second_negative_gauge,
        ) in itertools.product((1, -1), repeat=3):
            for signs in itertools.product((1, -1), repeat=14):
                a, A, d, D, e, E, g, G, h, H, i, I, c, sine = signs
                if a != positive_gauge * positive_signs[0]:
                    continue
                if A != positive_gauge * positive_signs[1]:
                    continue
                if d != first_negative_gauge * negative_signs[0]:
                    continue
                if D * e != first_negative_gauge * negative_signs[1]:
                    continue
                if D * E != first_negative_gauge * negative_signs[2]:
                    continue
                if g != second_negative_gauge * negative_signs[0]:
                    continue
                if G * h != second_negative_gauge * negative_signs[1]:
                    continue
                if G * H * i != second_negative_gauge * negative_signs[2]:
                    continue
                if G * H * I * c != second_negative_gauge * negative_signs[3]:
                    continue
                if G * H * I * sine != second_negative_gauge * negative_signs[4]:
                    continue
                induced.add(signs)

    annihilator = {
        mask
        for mask in itertools.product((0, 1), repeat=14)
        if all(_character(signs, mask) == 1 for signs in induced)
    }
    chart_characters = []
    lower_masks = set()
    for mask in sorted(annihilator):
        lower = tuple(mask[index] for index in FULL_LOWER_INDICES)
        complement = tuple(mask[index] for index in FULL_COMPLEMENT_INDICES)
        lower_masks.add(lower)
        chart_characters.append(
            {
                "lower_mask": list(lower),
                "complement_mask": list(complement),
                "forced_monomial": " ".join(
                    name
                    for name, bit in zip(FULL_CHART_PARAMETER_ORDER, mask, strict=True)
                    if bit
                )
                or "1",
            }
        )

    passed = bool(
        base["passed"]
        and base["common_adjoint_conjugacy_verified"]
        and len(induced) == 1024
        and len(annihilator) == 16
        and len(lower_masks) == 16
    )
    return {
        "chart_parameter_order": list(FULL_CHART_PARAMETER_ORDER),
        "induced_chart_sign_group_order": len(induced),
        "annihilator_order": len(annihilator),
        "one_complement_character_per_lower_character": len(lower_masks)
        == len(annihilator),
        "chart_characters": chart_characters,
        "global_sector_ansatz": (
            "Each of the sixteen sectors equals its forced chart monomial "
            "times a polynomial in a^2,d^2,e^2,g^2,h^2,i^2,c^2 after the "
            "seven circle relations."
        ),
        "passed": passed,
    }


@lru_cache(maxsize=1)
def exact_full_boundary_divisibility_certificate() -> dict[str, object]:
    """Prove the common ``Delta**3 * s**6`` divisor on the full chart.

    Each of the seven circle complements has two boundary branches.  On every
    branch the observation Jacobian has nullity three.  The two-branch normal
    form in the circle quotient therefore makes the information determinant
    divisible by the sixth power of that complement.  Coprimality combines
    the seven separate divisors.
    """

    generators = symbolic_triality_generators()
    basis = _symbolic_basis()
    a, A, d, D, e, E, g, G, h, H, i, I, c, sine = sp.symbols(
        "a A d D e E g G h H i I c s"
    )
    positive = _symbolic_vector((a, A), (0, 1), basis)
    first_negative = _symbolic_vector((d, D * e, D * E), (0, 1, 2), basis)
    second_negative = _symbolic_vector(
        (g, G * h, G * H * i, G * H * I * c, G * H * I * sine),
        (0, 1, 2, 3, 4),
        basis,
    )

    fixed_blocks = (
        _symbolic_observation_block(0, basis[0], generators),
        _symbolic_observation_block(1, basis[0], generators),
    )

    def jacobian(
        second: list[sp.Expr],
        third: list[sp.Expr],
        fourth: list[sp.Expr],
    ) -> sp.Matrix:
        return sp.Matrix.vstack(
            *fixed_blocks,
            _symbolic_observation_block(1, second, generators),
            _symbolic_observation_block(2, third, generators),
            _symbolic_observation_block(2, fourth, generators),
        )

    probes = (positive, first_negative, second_negative)
    circle_pairs = (
        ("A", a, A),
        ("D", d, D),
        ("E", e, E),
        ("G", g, G),
        ("H", h, H),
        ("I", i, I),
        ("s", c, sine),
    )
    branch_rows = []
    for name, lower, complement in circle_pairs:
        for branch in (1, -1):
            substitutions = {lower: branch, complement: 0}
            matrix = jacobian(
                *(
                    [sp.factor(entry.subs(substitutions)) for entry in probe]
                    for probe in probes
                )
            )
            nullspace = matrix.nullspace()
            branch_rows.append(
                {
                    "boundary_branch": f"{name}=0:{'+' if branch == 1 else '-'}",
                    "symbolic_rank": 28 - len(nullspace),
                    "symbolic_nullity": len(nullspace),
                    "nullspace_residual_exact_zero": all(
                        all(sp.factor(entry) == 0 for entry in matrix * vector)
                        for vector in nullspace
                    ),
                    "guaranteed_information_determinant_order": 2 * len(nullspace),
                }
            )

    lower = (a, d, e, g, h, i, c)
    complement = (A, D, E, G, H, I, sine)
    relations = tuple(
        left**2 + right**2 - 1 for left, right in zip(lower, complement, strict=True)
    )
    groebner = sp.groebner(relations, *lower, *complement, order="lex")
    leading_monomials = [
        list(polynomial.LM(order=groebner.order).exponents)
        for polynomial in groebner.polys
    ]
    expected_leading = [
        [int(index == position) * 2 for index in range(14)] for position in range(7)
    ]
    branches_pass = all(
        row["symbolic_rank"] == 25
        and row["symbolic_nullity"] == 3
        and row["nullspace_residual_exact_zero"]
        for row in branch_rows
    )
    exact_skew_symmetry = all(
        sp.Matrix(generators[view][plane]) + sp.Matrix(generators[view][plane]).T
        == sp.zeros(8)
        for view in range(3)
        for plane in range(28)
    )
    passed = bool(
        branches_pass and leading_monomials == expected_leading and exact_skew_symmetry
    )
    return {
        "coordinate_ring": (
            "Q[a,A,d,D,e,E,g,G,h,H,i,I,c,s]/"
            "(a^2+A^2-1,d^2+D^2-1,e^2+E^2-1,"
            "g^2+G^2-1,h^2+H^2-1,i^2+I^2-1,c^2+s^2-1)"
        ),
        "boundary_branch_rows": branch_rows,
        "all_fourteen_branches_rank_25": branches_pass,
        "groebner_leading_monomials": leading_monomials,
        "free_module_rank_over_complement_ring": 128,
        "both_branch_and_product_argument": (
            "On both branches of each circle boundary, nullity three makes "
            "every maximal Jacobian minor vanish to order at least three and "
            "det(J^T J) to order at least six. The rank-128 circle normal "
            "form separates the two branches. Each coefficient is divisible "
            "by the corresponding sixth power; the seven distinct complement "
            "variables are coprime, so their product divides every coefficient."
        ),
        "proved_divisor": "A^6 D^6 E^6 G^6 H^6 I^6 s^6 = Delta^3 s^6",
        "all_84_generators_exactly_skew_symmetric": exact_skew_symmetry,
        "universal_query_rank_upper_bound": 7,
        "raw_coordinate_pair_degree_upper_bound": 14,
        "post_division_coordinate_pair_degree_upper_bound": 8,
        "passed": passed,
    }


@lru_cache(maxsize=1)
def exact_full_multidegree_certificate() -> dict[str, object]:
    """Combine full-chart symmetry, boundary divisibility, and degree bounds."""

    signs = exact_full_chart_sign_certificate()
    boundary = exact_full_boundary_divisibility_certificate()
    rows = []
    for character in signs["chart_characters"]:
        lower_mask = character["lower_mask"]
        complement_mask = character["complement_mask"]
        degrees = [
            (8 - int(lower_bit) - int(complement_bit)) // 2
            for lower_bit, complement_bit in zip(
                lower_mask, complement_mask, strict=True
            )
        ]
        rows.append(
            {
                "lower_mask": lower_mask,
                "complement_mask": complement_mask,
                "residual_polynomial_multidegree_upper_bound": degrees,
                "tensor_grid_point_count": math.prod(degree + 1 for degree in degrees),
            }
        )
    passed = bool(signs["passed"] and boundary["passed"] and len(rows) == 16)
    return {
        "experiment": "unrestricted exact sign, divisor, and multidegree reduction",
        "variable_order": [f"{name}^2" for name in PARAMETER_NAMES],
        "full_chart_sign_certificate": signs,
        "full_boundary_divisibility_certificate": boundary,
        "sector_rows": rows,
        "separate_sector_grid_point_total": sum(
            row["tensor_grid_point_count"] for row in rows
        ),
        "two_disjoint_grid_point_total": 2
        * sum(row["tensor_grid_point_count"] for row in rows),
        "interpretation": (
            "The unrestricted inequality is reduced to sixteen explicitly "
            "bounded seven-variable polynomial sectors. This certificate "
            "does not establish their signs."
        ),
        "passed": passed,
    }


@lru_cache(maxsize=1)
def _exact_context():
    generators = symbolic_triality_generators()
    basis = _symbolic_basis()
    fixed = symbolic_query_projector(
        0, basis[0], generators
    ) + symbolic_query_projector(1, basis[0], generators)
    return generators, basis, fixed


def exact_normalized_determinant_from_half_angles(
    half_angles: tuple[sp.Rational, ...], signs: tuple[int, ...]
) -> sp.Expr:
    """Evaluate the complete chart exactly at rational half-angle data."""

    if len(half_angles) != 7 or len(signs) != 7:
        raise ValueError("the complete chart has seven coordinates")
    pairs = [rational_circle(value) for value in half_angles]
    pairs = [
        (sign * lower, complement)
        for sign, (lower, complement) in zip(signs, pairs, strict=True)
    ]
    generators, basis, fixed = _exact_context()
    (a, A), (d, D), (e, E), (g, G), (h, H), (i, I), (c, S) = pairs
    x2 = [a * basis[0][k] + A * basis[1][k] for k in range(8)]
    x3 = [d * basis[0][k] + D * (e * basis[1][k] + E * basis[2][k]) for k in range(8)]
    x4 = [
        g * basis[0][k]
        + G
        * (
            h * basis[1][k]
            + H * (i * basis[2][k] + I * (c * basis[3][k] + S * basis[4][k]))
        )
        for k in range(8)
    ]
    information = (
        fixed
        + symbolic_query_projector(1, x2, generators)
        + symbolic_query_projector(2, x3, generators)
        + symbolic_query_projector(2, x4, generators)
    )
    delta = A**2 * D**2 * E**2 * G**2 * H**2 * I**2
    return sp.cancel(information.det(method="domain-ge") / delta**3)


def exact_final_residual_structure_certificate() -> dict[str, object]:
    """Certify the 16-sector sign quotient and final-axis degree ceiling.

    The degree ceiling is structural.  The last query is a rank-seven update,
    so Cauchy--Binet gives degree at most fourteen in its vector coordinates.
    The already-proved boundary divisibility removes ``H**6``.  After the
    forced physical factor ``h**m H**n`` is removed, the remaining polynomial
    in ``r=h**2`` has degree at most ``floor((8-m-n)/2)``.
    """

    base_symmetry = exact_walsh_symmetry_certificate()
    induced = set()
    for action in base_symmetry["triality_representation_actions"]:
        t1, t2, t3, t4 = action["vector_signs"][1:5]
        induced.add((t1, t2, t1 * t2, t4, t1 * t4, t2 * t4, t3 * t4))
    masks = sorted(
        mask
        for mask in itertools.product((0, 1), repeat=7)
        if all(_character(signs, mask) == 1 for signs in induced)
    )

    all_signs = tuple(itertools.product((1, -1), repeat=7))
    unused = set(all_signs)
    representatives = []
    while unused:
        representative = min(unused)
        coset = {
            tuple(
                left * right for left, right in zip(representative, group, strict=True)
            )
            for group in induced
        }
        representatives.append(representative)
        unused -= coset
    representatives.sort()
    hadamard = sp.Matrix(
        [[_character(signs, mask) for mask in masks] for signs in representatives]
    )
    hadamard_identity = hadamard.T * hadamard == 16 * sp.eye(16)

    base = (
        sp.Rational(1, 7),
        sp.Rational(2, 9),
        sp.Rational(3, 11),
        sp.Rational(2, 13),
        sp.Rational(3, 14),
        sp.Rational(4, 15),
    )
    h_nodes = (
        sp.Rational(1, 10),
        sp.Rational(1, 5),
        sp.Rational(1, 3),
        sp.Rational(1, 2),
        sp.Rational(2, 3),
        sp.Rational(3, 4),
    )
    variable = sp.symbols("r")
    sector_samples = {mask: [] for mask in masks}
    for h_node in h_nodes:
        half_angles = base[:4] + (h_node,) + base[4:]
        determinants = sp.Matrix(
            [
                exact_normalized_determinant_from_half_angles(half_angles, signs)
                for signs in representatives
            ]
        )
        sectors = hadamard.T * determinants / 16
        h_value, h_complement = rational_circle(h_node)
        squared = h_value**2
        for mask, sector in zip(masks, sectors, strict=True):
            h_lower = mask[4]
            h_upper = mask[5] ^ mask[6]
            forced = h_value**h_lower * h_complement**h_upper
            sector_samples[mask].append((squared, sp.factor(sector / forced)))

    sector_rows = []
    all_holdouts_match = True
    all_degrees_within_bound = True
    for mask in masks:
        h_lower = mask[4]
        h_upper = mask[5] ^ mask[6]
        ceiling = (8 - h_lower - h_upper) // 2
        interpolation_count = ceiling + 1
        polynomial = sp.Poly(
            sp.interpolate(sector_samples[mask][:interpolation_count], variable),
            variable,
        )
        holdouts_match = all(
            sp.factor(polynomial.eval(point) - value) == 0
            for point, value in sector_samples[mask][interpolation_count:]
        )
        within_bound = polynomial.degree() <= ceiling
        all_holdouts_match &= holdouts_match
        all_degrees_within_bound &= within_bound
        sector_rows.append(
            {
                "mask": list(mask),
                "forced_h_power": h_lower,
                "forced_H_power": h_upper,
                "degree_ceiling_in_h_squared": ceiling,
                "observed_anchor_degree": int(polynomial.degree()),
                "independent_holdouts_match": holdouts_match,
            }
        )

    passed = bool(
        base_symmetry["passed"]
        and len(induced) == 8
        and len(masks) == 16
        and len(representatives) == 16
        and hadamard_identity
        and all_holdouts_match
        and all_degrees_within_bound
    )
    return {
        "experiment": "exact final-residual sign and degree reduction",
        "parameter_order": list(PARAMETER_NAMES),
        "induced_sign_group": [list(row) for row in sorted(induced)],
        "sector_masks": [list(mask) for mask in masks],
        "sector_count": len(masks),
        "coset_representatives": [list(row) for row in representatives],
        "hadamard_identity_verified": hadamard_identity,
        "forced_final_complement_law": "H exponent = i-sign bit XOR c-sign bit",
        "degree_argument": (
            "rank-seven last-query update: degree <=14; quotient boundary factor "
            "H^6; remaining r=h^2 degree <=floor((8-m-n)/2)"
        ),
        "anchor_half_angles_except_h": [str(value) for value in base],
        "h_half_angle_nodes": [str(value) for value in h_nodes],
        "sector_rows": sector_rows,
        "all_independent_holdouts_match": all_holdouts_match,
        "scope_boundary": (
            "The sign quotient and degree ceiling are exact structural reductions. "
            "The anchor interpolation checks implementation and sharpness; it is "
            "not a reconstruction of the other six variables or a sign proof."
        ),
        "passed": passed,
    }


def _rational_half_angle(value: float, maximum_denominator: int) -> sp.Rational:
    """Rationalize a correlation through its signed half-angle coordinate."""

    value = max(-1.0, min(1.0, float(value)))
    half_angle = value / (1 + math.sqrt(max(0.0, 1 - value * value)))
    fraction = Fraction(half_angle).limit_denominator(maximum_denominator)
    if fraction == 0 and value != 0:
        fraction = Fraction(1 if value > 0 else -1, maximum_denominator)
    if abs(fraction) == 1:
        fraction = Fraction(
            maximum_denominator - 1 if fraction > 0 else 1 - maximum_denominator,
            maximum_denominator,
        )
    return sp.Rational(fraction.numerator, fraction.denominator)


def exact_replay_correlations(
    correlations: torch.Tensor | list[float], *, maximum_denominator: int = 512
) -> dict[str, object]:
    """Rationalize and replay one numerical candidate in exact arithmetic."""

    values = (
        correlations.detach().cpu().tolist()
        if isinstance(correlations, torch.Tensor)
        else correlations
    )
    half_angles = [_rational_half_angle(value, maximum_denominator) for value in values]
    pairs = [rational_circle(value) for value in half_angles]
    normalized = sp.factor(
        exact_normalized_determinant_from_half_angles(
            tuple(half_angles), (1, 1, 1, 1, 1, 1, 1)
        )
    )
    c = pairs[-1][0]
    target = sp.factor((1 - c**2) ** 3 * (9 - c**2) ** 2 / 1024)
    gap = sp.factor(target - normalized)
    ratio = sp.factor(normalized / target)
    return {
        "half_angle_coordinates": [str(value) for value in half_angles],
        "rationalized_partial_correlations": [str(pair[0]) for pair in pairs],
        "normalized_determinant_exact": str(normalized),
        "target_exact": str(target),
        "target_minus_normalized_exact": str(gap),
        "ratio_exact": str(ratio),
        "ratio_float": float(ratio),
        "exact_violation": bool(gap < 0),
        "exact_equality": bool(gap == 0),
    }


def cholesky_frames(correlations: torch.Tensor) -> torch.Tensor:
    """Map signed partial correlations to four unit rows in ``R^8``."""

    if correlations.shape[-1] != len(PARAMETER_NAMES):
        raise ValueError(f"expected final axis {len(PARAMETER_NAMES)}")
    correlations = correlations.to(dtype=torch.float64)
    complements = torch.sqrt((1 - correlations.square()).clamp_min(0))
    a, d, e, g, h, i, c = correlations.unbind(dim=-1)
    A, D, E, G, H, I, S = complements.unbind(dim=-1)
    frame = torch.zeros(
        correlations.shape[:-1] + (4, 8),
        dtype=correlations.dtype,
        device=correlations.device,
    )
    frame[..., 0, 0] = 1
    frame[..., 1, 0] = a
    frame[..., 1, 1] = A
    frame[..., 2, 0] = d
    frame[..., 2, 1] = D * e
    frame[..., 2, 2] = D * E
    frame[..., 3, 0] = g
    frame[..., 3, 1] = G * h
    frame[..., 3, 2] = G * H * i
    frame[..., 3, 3] = G * H * I * c
    frame[..., 3, 4] = G * H * I * S
    return frame


def log_ratios(
    correlations: torch.Tensor, generators: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return log conjecture ratios and valid-positive-determinant mask."""

    frame = cholesky_frames(correlations)
    information = balanced_frame_information(frame, generators)
    sign, raw_logdet = torch.linalg.slogdet(information)
    gram_logdet = torch.log1p(-correlations[..., :GRAM_PARAMETER_COUNT].square()).sum(
        dim=-1
    )
    cayley_square = correlations[..., -1].square()
    target_logdet = (
        3 * torch.log1p(-cayley_square)
        + 2 * torch.log(9 - cayley_square)
        - math.log(1024)
    )
    ratio = raw_logdet - 3 * gram_logdet - target_logdet
    valid = sign > 0
    return torch.where(valid, ratio, -torch.inf), valid


def _sample_correlations(
    count: int,
    random: torch.Generator,
    *,
    boundary_fraction: float = 0.5,
    maximum: float = 1 - 2**-20,
) -> torch.Tensor:
    boundary_count = round(count * boundary_fraction)
    uniform_count = count - boundary_count
    uniform = (2 * torch.rand(uniform_count, 7, generator=random) - 1) * maximum
    concentration = torch.full((boundary_count, 7), 0.25)
    left = torch._standard_gamma(concentration, generator=random)
    right = torch._standard_gamma(concentration, generator=random)
    magnitude = (left / (left + right)).clamp_max(maximum)
    signs = torch.where(
        torch.rand(boundary_count, 7, generator=random) < 0.5,
        -torch.ones_like(magnitude),
        torch.ones_like(magnitude),
    )
    return torch.cat((uniform, signs * magnitude), dim=0).to(torch.float64)


@torch.no_grad()
def numerical_counterexample_screen(
    generators: torch.Tensor,
    *,
    seed: int,
    samples: int,
    batch_size: int,
) -> dict[str, object]:
    random = torch.Generator(device="cpu").manual_seed(seed)
    best_log_ratio = -math.inf
    best = None
    valid_count = 0
    floating_candidate_count = 0
    candidate_limit = 512
    candidates: list[tuple[float, torch.Tensor]] = []
    candidate_overflow = False
    completed = 0
    while completed < samples:
        count = min(batch_size, samples - completed)
        points = _sample_correlations(count, random).to(generators.device)
        ratios, valid = log_ratios(points, generators)
        valid_count += int(valid.sum())
        candidate_indices = torch.where(ratios > 1e-9)[0]
        floating_candidate_count += int(candidate_indices.numel())
        for index in candidate_indices.tolist():
            if len(candidates) < candidate_limit:
                candidates.append((float(ratios[index]), points[index].detach().cpu()))
            else:
                candidate_overflow = True
        value, index = ratios.max(dim=0)
        if float(value) > best_log_ratio:
            best_log_ratio = float(value)
            best = points[int(index)].detach().cpu()
        completed += count
    assert best is not None
    exact_replays = [
        {
            "floating_log_ratio": value,
            **exact_replay_correlations(point),
        }
        for value, point in sorted(candidates, key=lambda row: row[0], reverse=True)
    ]
    exact_violations = sum(row["exact_violation"] for row in exact_replays)
    return {
        "seed": seed,
        "sample_count": completed,
        "valid_count": valid_count,
        "maximum_log_ratio": best_log_ratio,
        "maximum_ratio": math.exp(best_log_ratio),
        "maximizer": {
            name: float(value)
            for name, value in zip(PARAMETER_NAMES, best, strict=True)
        },
        "floating_candidate_count_at_1e_minus_9": floating_candidate_count,
        "candidate_replay_limit": candidate_limit,
        "candidate_overflow": candidate_overflow,
        "exact_candidate_replays": exact_replays,
        "exact_violation_count": exact_violations,
        "status": "floating-point falsifier only",
        "passed_screen": exact_violations == 0 and not candidate_overflow,
    }


def adversarial_counterexample_screen(
    generators: torch.Tensor,
    *,
    seed: int,
    restarts: int,
    steps: int,
    learning_rate: float,
) -> dict[str, object]:
    random = torch.Generator(device="cpu").manual_seed(seed)
    initial = 1.75 * torch.randn(restarts, 7, generator=random, dtype=torch.float64)
    logits = nn.Parameter(initial.to(generators.device))
    optimizer = torch.optim.Adam((logits,), lr=learning_rate)
    trajectory = []
    checkpoints = {0, 49, 199, 499, steps - 1}
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        correlations = torch.tanh(logits) * (1 - 2**-20)
        ratios, valid = log_ratios(correlations, generators)
        objective = torch.where(valid, ratios, torch.full_like(ratios, -1e6))
        (-objective.sum()).backward()
        torch.nn.utils.clip_grad_norm_((logits,), 100.0)
        optimizer.step()
        if step in checkpoints:
            trajectory.append(
                {
                    "step": step + 1,
                    "maximum_log_ratio": float(objective.detach().max()),
                    "median_log_ratio": float(objective.detach().median()),
                }
            )
    with torch.no_grad():
        correlations = torch.tanh(logits) * (1 - 2**-20)
        ratios, _valid = log_ratios(correlations, generators)
        value, index = ratios.max(dim=0)
        best = correlations[int(index)].detach().cpu()
    exact_replay = exact_replay_correlations(best) if float(value) > 1e-9 else None
    return {
        "seed": seed,
        "restarts": restarts,
        "steps": steps,
        "learning_rate": learning_rate,
        "maximum_log_ratio": float(value),
        "maximum_ratio": math.exp(float(value)),
        "maximizer": {
            name: float(component)
            for name, component in zip(PARAMETER_NAMES, best, strict=True)
        },
        "trajectory": trajectory,
        "exact_replay_if_floating_positive": exact_replay,
        "status": "floating-point falsifier only",
        "passed_screen": bool(
            float(value) <= 1e-9
            or (exact_replay is not None and not exact_replay["exact_violation"])
        ),
    }


def run(
    *,
    device: torch.device,
    random_samples: int,
    batch_size: int,
    restarts: int,
    steps: int,
    workers: int,
) -> dict[str, object]:
    resource = constrain_current_process(workers=workers)
    generators = torch_triality_generators(dtype=torch.float64, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    chart = exact_chart_invariants_certificate()
    equality_slice = exact_final_residual_equality_slice_certificate()
    structure = exact_final_residual_structure_certificate()
    random_screen = numerical_counterexample_screen(
        generators,
        seed=2026080701,
        samples=random_samples,
        batch_size=batch_size,
    )
    adversarial = adversarial_counterexample_screen(
        generators,
        seed=2026080702,
        restarts=restarts,
        steps=steps,
        learning_rate=1e-2,
    )
    passed = bool(
        chart["passed"]
        and equality_slice["passed"]
        and structure["passed"]
        and random_screen["passed_screen"]
        and adversarial["passed_screen"]
    )
    return {
        "experiment": "Spin8 final Cholesky residual gate",
        "parameter_order": list(PARAMETER_NAMES),
        "exact_chart_invariants": chart,
        "exact_equality_slice": equality_slice,
        "exact_sign_and_degree_reduction": structure,
        "random_screen": random_screen,
        "adversarial_screen": adversarial,
        "hardware": {
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            "resource_contract": resource,
        },
        "scope_boundary": (
            "The equality-slice result is exact. The complete seven-coordinate "
            "campaign is a falsifier and does not prove the unrestricted theorem."
        ),
        "unrestricted_theorem_proved": False,
        "passed_all_declared_nonpromotion_gates": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--random-samples", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--restarts", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/spin8_dirac_final_residual_20260807.json"),
    )
    arguments = parser.parse_args()
    report = run(
        device=torch.device(arguments.device),
        random_samples=arguments.random_samples,
        batch_size=arguments.batch_size,
        restarts=arguments.restarts,
        steps=arguments.steps,
        workers=arguments.workers,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "exact_equality_slice_passed": report["exact_equality_slice"]["passed"],
                "random_maximum_log_ratio": report["random_screen"][
                    "maximum_log_ratio"
                ],
                "adversarial_maximum_log_ratio": report["adversarial_screen"][
                    "maximum_log_ratio"
                ],
                "unrestricted_theorem_proved": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
