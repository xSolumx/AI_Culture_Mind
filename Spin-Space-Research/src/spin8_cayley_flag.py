"""Exact isotropy audit supporting the balanced Cayley flag quotient.

The classical Spin(7) action on oriented four-planes in R^8 has
cohomogeneity one.  The publication argument also needs a finer fact: the
principal four-plane stabilizer acts on that four-plane as the full SO(4), so
it removes the internal 2+2 splitting.  This module verifies that load-bearing
claim exactly and repeats the rank calculation at several rational points to
falsify accidental one-point behavior.  It does not reprove the global
cohomogeneity-one orbit classification; that remains an explicitly cited
classical input.
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
from spin8_triality import SPIN8_PAIRS


def _cayley_value_exact(frame: sp.Matrix) -> sp.Expr:
    """Evaluate the maintained Cayley four-form on a 4-by-8 row frame."""

    return sp.factor(
        sum(
            coefficient * frame[:, list(indices)].det()
            for indices, coefficient in CAYLEY_TERMS.items()
        )
    )


def _isotropy_row(
    spin7_generators: list[sp.Matrix], cayley: sp.Rational, sine: sp.Rational
) -> dict[str, object]:
    """Return exact plane- and split-isotropy data for one normal-form flag."""

    frame = sp.zeros(8, 4)
    frame[0, 0] = 1
    frame[1, 1] = 1
    frame[2, 2] = 1
    frame[3, 3] = cayley
    frame[4, 3] = sine
    projector = frame * frame.T

    preservation_constraints = sp.Matrix.hstack(
        *[
            ((sp.eye(8) - projector) * generator * frame).reshape(32, 1)
            for generator in spin7_generators
        ]
    )
    stabilizer_basis = preservation_constraints.nullspace()
    coefficient_basis = sp.Matrix.hstack(*stabilizer_basis)

    restricted_generators: list[sp.Matrix] = []
    for basis_index in range(coefficient_basis.cols):
        ambient = sum(
            (
                coefficient_basis[generator_index, basis_index]
                * spin7_generators[generator_index]
                for generator_index in range(len(spin7_generators))
            ),
            sp.zeros(8),
        )
        restricted_generators.append(sp.simplify(frame.T * ambient * frame))

    upper_coordinates = sp.Matrix.hstack(
        *[
            sp.Matrix(
                [
                    restricted[row, column]
                    for row in range(4)
                    for column in range(row + 1, 4)
                ]
            )
            for restricted in restricted_generators
        ]
    )
    split_constraints = sp.Matrix(
        [
            [restricted[row, column] for restricted in restricted_generators]
            for row in (2, 3)
            for column in (0, 1)
        ]
    )
    plane_stabilizer_dimension = len(stabilizer_basis)
    split_stabilizer_dimension = plane_stabilizer_dimension - split_constraints.rank()
    return {
        "c": str(cayley),
        "s": str(sine),
        "cayley_four_form_value": str(_cayley_value_exact(frame.T)),
        "plane_preservation_constraint_rank": preservation_constraints.rank(),
        "plane_stabilizer_dimension": plane_stabilizer_dimension,
        "restricted_generators_are_skew": all(
            restricted + restricted.T == sp.zeros(4)
            for restricted in restricted_generators
        ),
        "effective_restriction_rank_in_so4": upper_coordinates.rank(),
        "split_preservation_constraint_rank": split_constraints.rank(),
        "flag_stabilizer_dimension": split_stabilizer_dimension,
    }


def exact_principal_flag_certificate() -> dict[str, object]:
    """Certify the effective isotropy and local flag dimensions exactly."""

    # Fixing the vector probe leaves the 21 bivector generators whose vector
    # planes avoid coordinate zero.  Their positive-chiral matrices realize
    # the common eight-dimensional Spin(7) module used after triality
    # identification.
    generators = symbolic_triality_generators()
    stabilizer_indices = [
        index for index, pair in enumerate(SPIN8_PAIRS) if 0 not in pair
    ]
    spin7_generators = [sp.Matrix(generators[1][index]) for index in stabilizer_indices]

    principal_pairs = [
        (sp.Rational(3, 5), sp.Rational(4, 5)),
        (sp.Rational(0), sp.Rational(1)),
        (sp.Rational(-3, 5), sp.Rational(4, 5)),
        (sp.Rational(5, 13), sp.Rational(12, 13)),
        (sp.Rational(12, 13), sp.Rational(5, 13)),
    ]
    principal_cross_checks = [
        _isotropy_row(spin7_generators, cayley, sine)
        for cayley, sine in principal_pairs
    ]
    principal = principal_cross_checks[0]
    endpoints = [
        _isotropy_row(spin7_generators, sp.Rational(sign), sp.Rational(0))
        for sign in (1, -1)
    ]

    # A direct cross-split regression guards the bridge used by the
    # Dirac--Gram manuscripts.  QR/Cholesky and symmetric polar whitening can
    # differ by an SO(4) row action that mixes the two positive and two
    # negative rows.  The global proof uses the full isotropy image below; this
    # explicit rational example makes accidental block-only implementations
    # fail immediately.
    basis = [[sp.Integer(row == column) for column in range(8)] for row in range(8)]
    cayley, sine = sp.Rational(3, 5), sp.Rational(4, 5)
    normal_frame = sp.Matrix(
        [
            basis[0],
            basis[1],
            basis[2],
            [cayley * basis[3][k] + sine * basis[4][k] for k in range(8)],
        ]
    )
    first_rotation = sp.eye(4)
    first_rotation[0, 0] = sp.Rational(3, 5)
    first_rotation[0, 2] = sp.Rational(4, 5)
    first_rotation[2, 0] = -sp.Rational(4, 5)
    first_rotation[2, 2] = sp.Rational(3, 5)
    second_rotation = sp.eye(4)
    second_rotation[1, 1] = sp.Rational(5, 13)
    second_rotation[1, 3] = sp.Rational(12, 13)
    second_rotation[3, 1] = -sp.Rational(12, 13)
    second_rotation[3, 3] = sp.Rational(5, 13)
    cross_split_rotation = second_rotation * first_rotation
    mixed_frame = cross_split_rotation * normal_frame
    fixed_vector = symbolic_query_projector(0, basis[0], generators)

    def balanced_information(frame: sp.Matrix) -> sp.Matrix:
        information = fixed_vector
        for row in range(4):
            view = 1 if row < 2 else 2
            information += symbolic_query_projector(
                view, list(frame.row(row)), generators
            )
        return information

    normal_information = balanced_information(normal_frame)
    mixed_information = balanced_information(mixed_frame)
    cross_split_characteristic_identity = (
        sp.expand(
            normal_information.charpoly().as_expr()
            - mixed_information.charpoly().as_expr()
        )
        == 0
    )
    oriented_grassmannian_dimension = 4 * (8 - 4)
    splitting_fiber_dimension = 2 * (4 - 2)
    flag_space_dimension = oriented_grassmannian_dimension + splitting_fiber_dimension
    principal_flag_orbit_dimension = 21 - int(principal["flag_stabilizer_dimension"])
    quotient_dimension = flag_space_dimension - principal_flag_orbit_dimension
    principal_rows_pass = all(
        row["cayley_four_form_value"] == row["c"]
        and row["plane_preservation_constraint_rank"] == 15
        and row["plane_stabilizer_dimension"] == 6
        and row["restricted_generators_are_skew"]
        and row["effective_restriction_rank_in_so4"] == 6
        and row["split_preservation_constraint_rank"] == 4
        and row["flag_stabilizer_dimension"] == 2
        for row in principal_cross_checks
    )
    passed = (
        len(stabilizer_indices) == 21
        and principal_rows_pass
        and all(
            row["cayley_four_form_value"] == row["c"]
            and row["plane_stabilizer_dimension"] == 9
            and row["restricted_generators_are_skew"]
            and row["effective_restriction_rank_in_so4"] == 6
            and row["flag_stabilizer_dimension"] == 5
            for row in endpoints
        )
        and flag_space_dimension == 20
        and principal_flag_orbit_dimension == 19
        and quotient_dimension == 1
        and cross_split_rotation.det() == 1
        and cross_split_rotation * cross_split_rotation.T == sp.eye(4)
        and cross_split_characteristic_identity
    )
    return {
        "theorem_role": "exact isotropy audit for the balanced-flag quotient",
        "spin7_generator_count": len(stabilizer_indices),
        "principal_representative": principal,
        "principal_rational_cross_checks": principal_cross_checks,
        "principal_cross_checks_passed": principal_rows_pass,
        "singular_endpoint_representatives": endpoints,
        "principal_plane_stabilizer_dimension": principal["plane_stabilizer_dimension"],
        "restricted_generators_are_skew": principal["restricted_generators_are_skew"],
        "effective_restriction_rank_in_so4": principal[
            "effective_restriction_rank_in_so4"
        ],
        "so4_dimension": 6,
        "principal_flag_stabilizer_dimension": principal["flag_stabilizer_dimension"],
        "oriented_grassmannian_dimension": oriented_grassmannian_dimension,
        "splitting_fiber_dimension": splitting_fiber_dimension,
        "flag_space_dimension": flag_space_dimension,
        "principal_flag_orbit_dimension": principal_flag_orbit_dimension,
        "local_flag_quotient_dimension": quotient_dimension,
        "cross_split_regression": {
            "row_rotation": [
                [str(value) for value in cross_split_rotation.row(row)]
                for row in range(4)
            ],
            "rotation_is_so4": bool(
                cross_split_rotation.det() == 1
                and cross_split_rotation * cross_split_rotation.T == sp.eye(4)
            ),
            "mixes_positive_and_negative_pairs": True,
            "characteristic_polynomials_match_exactly": bool(
                cross_split_characteristic_identity
            ),
            "determinant": str(sp.factor(normal_information.det())),
        },
        # Retained for compatibility with the original public artifact.  The
        # accompanying proof-layer fields now prevent this local dimension
        # count from being mistaken for a self-contained global classification.
        "flag_quotient_dimension": quotient_dimension,
        "oriented_stiefel_frame_dimension": 22,
        "same_view_basis_gauge_dimension": 2,
        "oriented_split_flag_dimension": 20,
        "endpoint_flag_orbit_dimension": 21
        - int(endpoints[0]["flag_stabilizer_dimension"]),
        "proof_layers": {
            "exact_algebra": (
                "The maintained Spin(7) generators give plane stabilizer "
                "dimensions 6 on five rational non-endpoint representatives "
                "and 9 at c=+/-1; their restrictions span so(4), and the "
                "split stabilizers have dimensions 2 and 5 respectively."
            ),
            "classical_global_input": (
                "The Spin(7) action on the oriented four-plane Grassmannian "
                "has cohomogeneity one, Cayley value separates its oriented "
                "orbits, and the two Cayley orientations are the singular "
                "endpoints."
            ),
            "deduction": (
                "Full SO(4) isotropy removes the internal oriented 2+2 split; "
                "allowing O(2) basis changes in either observed pair identifies "
                "c with -c, so the information-family coordinate is z=c^2."
            ),
            "not_recomputed_here": (
                "The global separation of all four-plane orbits by c and the "
                "principal-isotropy conjugacy theorem are not derived from "
                "the finite rank calculations."
            ),
        },
        "scope": (
            "Exact isotropy and dimension calculations, with five rational "
            "principal-stratum falsifiers and both endpoints. Global orbit "
            "classification remains a separate classical theorem input."
        ),
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = exact_principal_flag_certificate()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("principal balanced-flag certificate failed")


if __name__ == "__main__":
    main()
