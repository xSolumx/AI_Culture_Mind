"""Exact all-parameter certificates on two Spin(9) Grassmann V5 rays.

The symmetric Spin(9) three-plane curve passes through a Cayley-null plane.
Its Grassmann normal slice is ``V1 + V5``, where ``V5`` is the spin-two
module ``Sym_0(3)``.  This module constructs two explicit graph-chart rays in
that V5 summand:

* a zero-cubic ray; and
* an axisymmetric ray fixed by a one-dimensional stabilizer.

For both rays the information determinant is reconstructed exactly from the
rank-one observation matrices.  Polynomial determinants are evaluated over
``QQ(sqrt(2))[t]``.  Sturm root counts then certify a global inequality on
each complete real ray.  The result is not a certificate on the full V5
shape interval or on the global Grassmann quotient.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sympy as sp
from sympy.polys.matrices import DomainMatrix

from spin9_dirac_clifford import build_spin9_clifford_system
from spin9_grassmann_slice import _horizontal_constraint
from spin9_three_spinor_conditioning import _symbolic_template
from spin9_three_spinor_symmetry import _canonical_integer_vector

T = sp.symbols("t", real=True)
SQRT2 = sp.sqrt(2)


def _cayley_null_frame() -> sp.Matrix:
    """Return the maintained orthonormal frame at symmetric parameter c=0."""

    c, spinors, _, substitution = _symbolic_template()
    return sp.Matrix.hstack(*spinors).subs(substitution).subs(c, 0)


def _normal_slice_data() -> tuple[
    sp.Matrix,
    sp.Matrix,
    list[sp.Matrix],
    sp.Matrix,
]:
    """Reconstruct the normal basis, metric, stabilizer actions, and Casimir."""

    frame = _cayley_null_frame()
    projector = frame * frame.T
    system = build_spin9_clifford_system()
    generators = [sp.Matrix(matrix) / 2 for matrix in system.doubled_spin_generators]
    orbit = sp.Matrix.hstack(
        *[
            ((sp.eye(16) - projector) * generator * frame).reshape(48, 1)
            for generator in generators
        ]
    )
    horizontal = _horizontal_constraint(frame)
    normal = sp.Matrix.hstack(*sp.Matrix.vstack(orbit.T, horizontal).nullspace())
    metric = sp.simplify(normal.T * normal)

    stabilizer_coordinates = [
        _canonical_integer_vector(vector) for vector in orbit.nullspace()
    ]
    stabilizer = [
        sum(
            (
                coordinates[index] * generators[index]
                for index in range(len(generators))
            ),
            sp.zeros(16),
        )
        for coordinates in stabilizer_coordinates
    ]

    actions: list[sp.Matrix] = []
    for element in stabilizer:
        plane_action = frame.T * element * frame
        columns = []
        for index in range(normal.cols):
            variation = sp.Matrix(16, 3, list(normal[:, index]))
            transformed = element * variation - variation * plane_action
            columns.append(
                sp.simplify(metric.inv() * normal.T * transformed.reshape(48, 1))
            )
        actions.append(sp.Matrix.hstack(*columns))

    casimir = sp.simplify(
        -(actions[0] ** 2 + 2 * actions[1] ** 2 + actions[2] ** 2) / 2
    )
    return normal, metric, actions, casimir


def _v5_coordinates() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """Return a V5 basis and the two selected vectors in normal coordinates."""

    # These coordinates are relative to the canonical nullspace basis produced
    # by ``_normal_slice_data``.  The identities below are replayed rather than
    # inferred from the stored coordinates.
    v5_basis = sp.Matrix(
        [
            [0, 0, -1, 0, -SQRT2],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    )
    zero_cubic = v5_basis * sp.Matrix([1, 0, 0, 0, 0])
    axisymmetric = v5_basis * sp.Matrix([0, 0, SQRT2, 0, 1])
    return v5_basis, zero_cubic, axisymmetric


def _v5_cubic(coordinates: sp.Matrix) -> sp.Expr:
    """Evaluate the canonical cubic invariant in the selected V5 basis."""

    x0, x1, x2, x3, x4 = coordinates
    return sp.factor(
        -(
            2 * x0**2 * x4
            - 4 * x0 * x1 * x3
            + SQRT2 * x1**2 * x2
            + 2 * x2**2 * x4
            - 2 * SQRT2 * x2 * x3**2
            + 2 * SQRT2 * x2 * x4**2
            - 4 * x3**2 * x4
        )
        / 4
    )


def _rank_one_coefficients(
    base: sp.Matrix,
    variation: sp.Matrix,
    generators: list[sp.Matrix],
) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """Return coefficients of the observation Gram for ``base+t*variation``."""

    base_observations = sp.Matrix.hstack(
        *[generator * base for generator in generators]
    )
    variation_observations = sp.Matrix.hstack(
        *[generator * variation for generator in generators]
    )
    constant = base_observations.T * base_observations
    linear = (
        base_observations.T * variation_observations
        + variation_observations.T * base_observations
    )
    quadratic = variation_observations.T * variation_observations
    return constant, linear, quadratic


def _information_polynomial(
    frame: sp.Matrix,
    variation: sp.Matrix,
) -> tuple[list[sp.Matrix], sp.Expr, list[list[int]]]:
    """Return a common-denominator information polynomial and its blocks."""

    if frame.T * variation != sp.zeros(3):
        raise ValueError("the graph variation must be horizontal")
    variation_gram = sp.simplify(variation.T * variation)
    if variation_gram != sp.diag(*variation_gram.diagonal()):
        raise ValueError("the selected graph columns must remain orthogonal")

    norms = [sp.Rational(variation_gram[index, index]) for index in range(3)]
    distinct_norms = sorted(set(norms))
    generators = [
        sp.Matrix(matrix)
        for matrix in build_spin9_clifford_system().doubled_spin_generators
    ]
    column_coefficients = [
        _rank_one_coefficients(frame[:, index], variation[:, index], generators)
        for index in range(3)
    ]

    denominator = sp.prod(1 + norm * T**2 for norm in distinct_norms)
    coefficient_matrices = [sp.zeros(36) for _ in range(2 * len(distinct_norms) + 1)]
    polynomial_matrix = sp.zeros(36)
    for norm in distinct_norms:
        group = [index for index, value in enumerate(norms) if value == norm]
        group_polynomial = sp.zeros(36)
        for index in group:
            constant, linear, quadratic = column_coefficients[index]
            group_polynomial += constant + T * linear + T**2 * quadratic
        multiplier = sp.div(sp.Poly(denominator, T), sp.Poly(1 + norm * T**2, T))[
            0
        ].as_expr()
        polynomial_matrix += multiplier * group_polynomial

    polynomial_matrix = polynomial_matrix.applyfunc(sp.expand)
    for degree in range(len(coefficient_matrices)):
        coefficient_matrices[degree] = polynomial_matrix.applyfunc(
            lambda value, degree=degree: sp.expand(value).coeff(T, degree)
        )

    pattern = np.zeros((36, 36), dtype=bool)
    for matrix in coefficient_matrices:
        pattern |= np.asarray(
            [[entry != 0 for entry in row] for row in matrix.tolist()],
            dtype=bool,
        )
    adjacency = [set(np.flatnonzero(pattern[index]).tolist()) for index in range(36)]
    seen: set[int] = set()
    blocks: list[list[int]] = []
    for start in range(36):
        if start in seen:
            continue
        pending = [start]
        block: list[int] = []
        while pending:
            index = pending.pop()
            if index in seen:
                continue
            seen.add(index)
            block.append(index)
            pending.extend(adjacency[index] - seen)
        blocks.append(sorted(block))
    blocks.sort(key=lambda block: (len(block), block))
    return coefficient_matrices, sp.expand(denominator), blocks


def _block_determinants(
    coefficients: list[sp.Matrix],
    blocks: list[list[int]],
) -> list[sp.Expr]:
    """Compute exact block determinants over QQ(sqrt(2))[t]."""

    field = sp.QQ.algebraic_field(SQRT2)
    polynomial_ring = field.poly_ring(T)
    determinants: list[sp.Expr] = []
    for block in blocks:
        rows = [
            [
                polynomial_ring.from_sympy(
                    sum(
                        (
                            coefficients[degree][row, column] * T**degree
                            for degree in range(len(coefficients))
                        ),
                        sp.S.Zero,
                    )
                )
                for column in block
            ]
            for row in block
        ]
        domain_matrix = DomainMatrix(rows, (len(block), len(block)), polynomial_ring)
        determinants.append(domain_matrix.det().as_expr())
    return determinants


def _ray_determinant(
    frame: sp.Matrix,
    variation: sp.Matrix,
) -> dict[str, object]:
    """Reconstruct the determinant ratio for one complete graph ray."""

    coefficients, denominator, blocks = _information_polynomial(frame, variation)
    block_determinants = _block_determinants(coefficients, blocks)
    numerator_raw = sp.expand(sp.prod(block_determinants))
    base_value = numerator_raw.subs(T, 0)
    determinant_denominator = sp.expand(denominator**36)
    reduced_ratio = sp.cancel(
        numerator_raw / (base_value * determinant_denominator),
        extension=SQRT2,
    )
    numerator, determinant_denominator = sp.fraction(reduced_ratio)
    origin_scale = determinant_denominator.subs(T, 0)
    numerator = sp.expand(numerator / origin_scale)
    determinant_denominator = sp.expand(determinant_denominator / origin_scale)
    return {
        "gram_factor": sp.factor(
            ((frame + T * variation).T * (frame + T * variation)).det()
        ),
        "block_sizes": [len(block) for block in blocks],
        "block_indices": blocks,
        "block_factors": [
            sp.factor(value, extension=SQRT2) for value in block_determinants
        ],
        "numerator": numerator,
        "denominator": determinant_denominator,
        "numerator_degree": int(sp.Poly(numerator, T, extension=SQRT2).degree()),
        "denominator_degree": int(sp.Poly(determinant_denominator, T).degree()),
    }


def _quadratic_field_sign_witness(
    value: sp.Expr,
    radical: int,
) -> dict[str, object]:
    """Return an exact sign witness for ``a+b*sqrt(radical)``."""

    root = sp.sqrt(radical)
    generator = sp.Dummy("quadratic_generator")
    symbolic_value = sp.cancel(value.xreplace({root: generator}))
    numerator, denominator_expression = sp.fraction(symbolic_value)
    minimal_polynomial = sp.Poly(
        generator**2 - radical,
        generator,
        domain=sp.QQ,
    )
    numerator_polynomial = sp.Poly(
        numerator,
        generator,
        domain=sp.QQ,
    ).rem(minimal_polynomial)
    denominator_polynomial = sp.Poly(
        denominator_expression,
        generator,
        domain=sp.QQ,
    ).rem(minimal_polynomial)
    inverse_denominator = sp.invert(
        denominator_polynomial,
        minimal_polynomial,
    )
    reduced = (numerator_polynomial * inverse_denominator).rem(minimal_polynomial)
    rational_part = sp.Rational(reduced.nth(0))
    radical_part = sp.Rational(reduced.nth(1))

    denominator = sp.ilcm(
        int(sp.denom(rational_part)),
        int(sp.denom(radical_part)),
    )
    rational = sp.Integer(rational_part * denominator)
    radical_coefficient = sp.Integer(radical_part * denominator)
    squared_margin = sp.expand(radical * radical_coefficient**2 - rational**2)
    if radical_coefficient == 0:
        positive = bool(rational > 0)
    elif radical_coefficient > 0:
        positive = bool(rational >= 0 or squared_margin > 0)
    else:
        positive = bool(rational > 0 and squared_margin < 0)
    return {
        "rational_numerator": str(rational),
        "radical_numerator": str(radical_coefficient),
        "denominator": str(denominator),
        "squared_margin": str(squared_margin),
        "positive": positive,
    }


def diagnostics() -> dict[str, object]:
    """Replay both ray theorems and their exact comparison to the candidate."""

    frame = _cayley_null_frame()
    normal, metric, actions, casimir = _normal_slice_data()
    v5_basis, zero_coordinates, axis_coordinates = _v5_coordinates()
    zero_variation = sp.Matrix(16, 3, list(normal * zero_coordinates))
    axis_variation = sp.Matrix(16, 3, list(normal * axis_coordinates))

    v5_metric = sp.simplify(v5_basis.T * metric * v5_basis)
    left_inverse = sp.simplify(v5_metric.inv() * v5_basis.T * metric)
    restricted_actions = [
        sp.simplify(left_inverse * action * v5_basis) for action in actions
    ]
    x = sp.Matrix(sp.symbols("x0:5"))
    cubic = _v5_cubic(x)
    cubic_invariant = all(
        sp.expand(
            sum(sp.diff(cubic, x[index]) * (action * x)[index] for index in range(5))
        )
        == 0
        for action in restricted_actions
    )
    zero_v5 = sp.Matrix([1, 0, 0, 0, 0])
    axis_v5 = sp.Matrix([0, 0, SQRT2, 0, 1])

    zero_ray = _ray_determinant(frame, zero_variation)
    zero_gap = sp.Poly(sp.expand(zero_ray["denominator"] - zero_ray["numerator"]), T)
    zero_quotient, zero_remainder = sp.div(zero_gap, sp.Poly(T**2, T))
    zero_quotient = sp.Poly(zero_quotient, T, domain=sp.QQ)
    zero_real_roots = int(zero_quotient.count_roots(-sp.oo, sp.oo))

    axis_ray = _ray_determinant(frame, axis_variation)
    axis_gap = sp.expand(101 * axis_ray["denominator"] - 100 * axis_ray["numerator"])
    axis_conjugate = sp.expand(axis_gap.xreplace({SQRT2: -SQRT2}))
    axis_rational = sp.expand((axis_gap + axis_conjugate) / 2)
    axis_radical = sp.expand((axis_gap - axis_conjugate) / (2 * SQRT2))
    axis_norm = sp.Poly(
        sp.expand(axis_rational**2 - 2 * axis_radical**2),
        T,
        domain=sp.QQ,
    )
    axis_norm_real_roots = int(axis_norm.count_roots(-sp.oo, sp.oo))

    challenger_t = sp.Integer(-50)
    challenger_gap = sp.simplify(
        axis_ray["numerator"].subs(T, challenger_t)
        - axis_ray["denominator"].subs(T, challenger_t)
    )
    challenger_sign = _quadratic_field_sign_witness(challenger_gap, 2)

    q = sp.sqrt(241)
    c_star = (q - 17) / 24
    candidate_ratio = sp.simplify(
        (1 - c_star) ** 10 * (c_star + 2) ** 5 * (2 * c_star + 1) ** 3 / 32
    )
    candidate_gap = sp.simplify(candidate_ratio - sp.Rational(101, 100))
    candidate_sign = _quadratic_field_sign_witness(candidate_gap, 241)

    zero_numerator_rational = (
        sp.expand(
            zero_ray["numerator"].xreplace({SQRT2: -SQRT2}) - zero_ray["numerator"]
        )
        == 0
    )
    axis_gap_reconstruction = (
        sp.expand(axis_rational + SQRT2 * axis_radical - axis_gap) == 0
    )
    slice_checks = {
        "frame_orthonormal": frame.T * frame == sp.eye(3),
        "v5_basis_casimir_six": casimir * v5_basis == 6 * v5_basis,
        "zero_ray_casimir_six": casimir * zero_coordinates == 6 * zero_coordinates,
        "axis_ray_casimir_six": casimir * axis_coordinates == 6 * axis_coordinates,
        "zero_ray_horizontal": frame.T * zero_variation == sp.zeros(3),
        "axis_ray_horizontal": frame.T * axis_variation == sp.zeros(3),
        "axis_ray_so2_fixed": actions[0] * axis_coordinates == sp.zeros(6, 1),
        "cubic_infinitesimally_invariant": cubic_invariant,
        "zero_ray_cubic": sp.sstr(_v5_cubic(zero_v5)),
        "axis_ray_cubic": sp.sstr(_v5_cubic(axis_v5)),
        "zero_ray_norm_squared": sp.sstr(
            (zero_coordinates.T * metric * zero_coordinates)[0]
        ),
        "axis_ray_norm_squared": sp.sstr(
            (axis_coordinates.T * metric * axis_coordinates)[0]
        ),
    }

    report: dict[str, object] = {
        "schema_version": 1,
        "claim_scope": "two complete V5 graph rays at the Cayley-null Spin(9) plane",
        "slice_checks": slice_checks,
        "zero_cubic_ray": {
            "gram_factor": sp.sstr(zero_ray["gram_factor"]),
            "information_block_sizes": zero_ray["block_sizes"],
            "information_blocks": zero_ray["block_indices"],
            "block_determinant_factors": [
                sp.sstr(value) for value in zero_ray["block_factors"]
            ],
            "determinant_ratio_numerator_factor": sp.sstr(
                sp.factor(zero_ray["numerator"])
            ),
            "determinant_ratio_denominator_factor": sp.sstr(
                sp.factor(zero_ray["denominator"])
            ),
            "numerator_degree": zero_ray["numerator_degree"],
            "denominator_degree": zero_ray["denominator_degree"],
            "numerator_is_rational": zero_numerator_rational,
            "gap_has_t_squared_factor": zero_remainder.is_zero,
            "gap_quotient_degree": int(zero_quotient.degree()),
            "gap_quotient_real_root_count": zero_real_roots,
            "gap_quotient_at_zero": str(zero_quotient.eval(0)),
            "global_ray_bound": "det I(P_zero(t)) <= det I(P_0)",
            "finite_equality_parameter": "t = 0",
        },
        "axisymmetric_ray": {
            "gram_factor": sp.sstr(axis_ray["gram_factor"]),
            "information_block_sizes": axis_ray["block_sizes"],
            "information_blocks": axis_ray["block_indices"],
            "block_determinant_factors": [
                sp.sstr(value) for value in axis_ray["block_factors"]
            ],
            "determinant_ratio_numerator_factor": sp.sstr(
                sp.factor(axis_ray["numerator"], extension=SQRT2)
            ),
            "determinant_ratio_denominator_factor": sp.sstr(
                sp.factor(axis_ray["denominator"])
            ),
            "numerator_degree": axis_ray["numerator_degree"],
            "denominator_degree": axis_ray["denominator_degree"],
            "rational_upper_bound": "101/100",
            "upper_gap_norm_degree": int(axis_norm.degree()),
            "upper_gap_norm_real_root_count": axis_norm_real_roots,
            "upper_gap_at_zero": sp.sstr(axis_gap.subs(T, 0)),
            "global_ray_bound": "det I(P_axis(t)) < (101/100) det I(P_0)",
            "cayley_null_challenger_parameter": str(challenger_t),
            "cayley_null_challenger_sign_witness": challenger_sign,
            "cayley_null_global_ray_maximum": False,
        },
        "symmetric_candidate_comparison": {
            "c_star": sp.sstr(c_star),
            "determinant_ratio_to_cayley_null": sp.sstr(candidate_ratio),
            "ratio_exceeds_101_over_100": candidate_sign,
            "candidate_beats_both_complete_rays": True,
        },
        "all_v5_shapes_certified": False,
        "global_grassmann_quotient_certified": False,
        "global_rank_three_optimum_certified": False,
    }
    report["passed"] = bool(
        all(
            value
            for key, value in slice_checks.items()
            if key
            not in {
                "zero_ray_cubic",
                "axis_ray_cubic",
                "zero_ray_norm_squared",
                "axis_ray_norm_squared",
            }
        )
        and slice_checks["zero_ray_cubic"] == "0"
        and slice_checks["axis_ray_cubic"] == "-2"
        and slice_checks["zero_ray_norm_squared"] == "4"
        and slice_checks["axis_ray_norm_squared"] == "24"
        and zero_numerator_rational
        and zero_remainder.is_zero
        and zero_real_roots == 0
        and zero_quotient.eval(0) > 0
        and axis_gap_reconstruction
        and axis_norm_real_roots == 0
        and axis_gap.subs(T, 0) > 0
        and challenger_sign["positive"]
        and candidate_sign["positive"]
        and not report["all_v5_shapes_certified"]
        and not report["global_grassmann_quotient_certified"]
        and not report["global_rank_three_optimum_certified"]
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
