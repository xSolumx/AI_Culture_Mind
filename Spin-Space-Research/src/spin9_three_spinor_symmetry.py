"""Exact stabilizer and branching certificate for the symmetric Spin(9) curve.

The complete characteristic polynomial of the information operator factors
with powers 7, 5, and 3.  This module determines what those powers mean at the
exact Cayley-null point c = 0.  The stabilizer of the corresponding spinor
three-plane is a three-dimensional copy of so(3), and its adjoint action on
spin(9) decomposes as two 7-dimensional, two 5-dimensional, and four
3-dimensional irreducibles.

The certificate is deliberately pointwise.  It does not infer a common fixed
stabilizer subgroup for every point of the one-parameter curve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp

from spin9_dirac_clifford import build_spin9_clifford_system
from spin9_three_spinor_conditioning import _symbolic_template

CURVE_RANK_WITNESS_ROWS = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    41,
    43,
    44,
    47,
)
CURVE_RANK_WITNESS_COLUMNS = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    40,
    41,
    44,
)


def _canonical_integer_vector(vector: sp.Matrix) -> sp.Matrix:
    """Return the primitive integer representative of a rational vector."""

    denominator = sp.ilcm(*[sp.denom(value) for value in vector])
    integer = sp.Matrix([sp.expand(denominator * value) for value in vector])
    nonzero = [abs(int(value)) for value in integer if value]
    divisor = sp.igcd(*nonzero) if nonzero else 1
    integer /= divisor
    first = next((value for value in integer if value), 0)
    return -integer if first > 0 else integer


def _adjoint_matrix(
    element: sp.Matrix, generators: list[sp.Matrix]
) -> sp.Matrix:
    """Matrix of ``[element, -]`` in the orthogonal generator basis."""

    # Conventional generators have Frobenius norm squared four.
    return sp.Matrix.hstack(
        *[
            sp.Matrix(
                [
                    sp.trace(basis.T * (element * generator - generator * element))
                    / 4
                    for basis in generators
                ]
            )
            for generator in generators
        ]
    )


def _curve_stabilizer_certificate(
    generators: list[sp.Matrix],
) -> dict[str, object]:
    """Certify an so(3) plane stabilizer over the complete open curve."""

    u = sp.symbols("u", positive=True)
    radical = sp.sqrt((3 - u**2) * (1 + u**2))
    spinor_frame = sp.zeros(16, 3)
    spinor_frame[0, 0] = 1
    spinor_frame[1, 1] = 1
    spinor_frame[8, 1] = u
    # Scaling a spanning vector does not change the plane.  This doubled third
    # vector avoids fractions in the symbolic stabilizer equations.
    spinor_frame[2, 2] = -2
    spinor_frame[11, 2] = u * (1 - u**2)
    spinor_frame[12, 2] = u * radical

    generator_columns = [
        (generator * spinor_frame).reshape(48, 1) for generator in generators
    ]
    induced_columns = []
    for row in range(3):
        for column in range(3):
            elementary = sp.zeros(3)
            elementary[row, column] = 1
            induced_columns.append((-spinor_frame * elementary).reshape(48, 1))
    stabilizer_system = sp.Matrix.hstack(*(generator_columns + induced_columns))
    nullspace = stabilizer_system.nullspace(simplify=True)
    induced_actions = [sp.Matrix(3, 3, list(vector[36:])) for vector in nullspace]

    plane_metric = sp.simplify(spinor_frame.T * spinor_frame)
    expected_plane_metric = sp.diag(1, 1 + u**2, 4 * (1 + u**2))
    metric_skew_identities = [
        sp.simplify(action.T * plane_metric + plane_metric * action) == sp.zeros(3)
        for action in induced_actions
    ]
    induced_span_rank = sp.Matrix.hstack(
        *[action.reshape(9, 1) for action in induced_actions]
    ).rank()

    witness_minor = stabilizer_system.extract(
        CURVE_RANK_WITNESS_ROWS, CURVE_RANK_WITNESS_COLUMNS
    )
    witness_determinant = sp.factor(witness_minor.det(method="domain-ge"))
    expected_witness_determinant = (
        u**10
        * (u**2 - 3) ** 4
        * (u**2 + 1) ** sp.Rational(13, 2)
        / (256 * sp.sqrt(3 - u**2))
    )

    return {
        "parameterization": "u^2=(1-c)/(1+c), 0<u<sqrt(3)",
        "stabilizer_system_shape": list(stabilizer_system.shape),
        "symbolic_nullity": len(nullspace),
        "plane_metric": sp.sstr(plane_metric),
        "plane_metric_identity": plane_metric == expected_plane_metric,
        "induced_metric_skew_identities": metric_skew_identities,
        "induced_action_span_rank": int(induced_span_rank),
        "rank_witness_rows": list(CURVE_RANK_WITNESS_ROWS),
        "rank_witness_columns": list(CURVE_RANK_WITNESS_COLUMNS),
        "rank_witness_determinant": sp.sstr(witness_determinant),
        "rank_witness_identity": sp.simplify(
            witness_determinant - expected_witness_determinant
        )
        == 0,
        "rank_witness_nonzero_on_open_interval": True,
        "curve_plane_stabilizer_dimension": 3,
        "curve_plane_stabilizer_is_full_so3": bool(
            len(nullspace) == 3
            and plane_metric == expected_plane_metric
            and all(metric_skew_identities)
            and induced_span_rank == 3
        ),
    }


def diagnostics() -> dict[str, object]:
    """Reconstruct the exact c=0 plane stabilizer and its branching law."""

    c, spinors, information, substitution = _symbolic_template()
    spinor_frame = sp.Matrix.hstack(*spinors).subs(substitution).subs(c, 0)
    projector = sp.simplify(spinor_frame * spinor_frame.T)

    system = build_spin9_clifford_system()
    generators = [
        sp.Matrix(matrix) / 2 for matrix in system.doubled_spin_generators
    ]

    plane_action = sp.Matrix.hstack(
        *[
            sp.simplify((sp.eye(16) - projector) * generator * spinor_frame).reshape(
                48, 1
            )
            for generator in generators
        ]
    )
    pointwise_action = sp.Matrix.hstack(
        *[(generator * spinor_frame).reshape(48, 1) for generator in generators]
    )
    stabilizer_coordinates = [
        _canonical_integer_vector(vector) for vector in plane_action.nullspace()
    ]
    stabilizer = [
        sum(
            (coordinates[index] * generators[index] for index in range(36)),
            sp.zeros(16),
        )
        for coordinates in stabilizer_coordinates
    ]

    expected_brackets = {
        "[H0,H1]": stabilizer[2],
        "[H0,H2]": -2 * stabilizer[1],
        "[H1,H2]": stabilizer[0],
    }
    observed_brackets = {
        "[H0,H1]": stabilizer[0] * stabilizer[1]
        - stabilizer[1] * stabilizer[0],
        "[H0,H2]": stabilizer[0] * stabilizer[2]
        - stabilizer[2] * stabilizer[0],
        "[H1,H2]": stabilizer[1] * stabilizer[2]
        - stabilizer[2] * stabilizer[1],
    }
    bracket_identities = {
        key: observed_brackets[key] == expected
        for key, expected in expected_brackets.items()
    }

    adjoint = [_adjoint_matrix(element, generators) for element in stabilizer]
    # E0=H0, E1=sqrt(2)H1, E2=H2 have equal so(3) structure constants.
    # Dividing the negative sum of squares by two gives eigenvalue l(l+1)
    # on the real irreducible of dimension 2l+1.
    casimir = sp.simplify(
        -(adjoint[0] ** 2 + 2 * adjoint[1] ** 2 + adjoint[2] ** 2) / 2
    )
    casimir_eigenvalues = {
        int(eigenvalue): int(multiplicity)
        for eigenvalue, multiplicity in casimir.eigenvals().items()
    }
    expected_casimir_eigenvalues = {12: 14, 6: 10, 2: 12}

    information_zero = sp.simplify(information.subs(substitution).subs(c, 0))
    commutator_identities = [
        information_zero * action - action * information_zero == sp.zeros(36)
        for action in adjoint
    ]

    spectral_variable = sp.symbols("x")
    restricted_characteristic_polynomials: dict[str, str] = {}
    expected_restricted = {
        12: (8 * spectral_variable**2 - 8 * spectral_variable + 1) ** 7
        / sp.Integer(2) ** 21,
        6: (4 * spectral_variable**2 - 7 * spectral_variable + 2) ** 5
        / sp.Integer(2) ** 10,
        2: (
            16 * spectral_variable**4
            - 60 * spectral_variable**3
            + 64 * spectral_variable**2
            - 16 * spectral_variable
            + 1
        )
        ** 3
        / sp.Integer(2) ** 12,
    }
    restricted_identities: dict[str, bool] = {}
    for eigenvalue in (12, 6, 2):
        basis = sp.Matrix.hstack(
            *(casimir - eigenvalue * sp.eye(36)).nullspace()
        )
        restriction = sp.simplify(
            (basis.T * basis).inv() * basis.T * information_zero * basis
        )
        characteristic = sp.factor(
            restriction.charpoly(spectral_variable).as_expr()
        )
        restricted_characteristic_polynomials[str(eigenvalue)] = sp.sstr(
            characteristic
        )
        restricted_identities[str(eigenvalue)] = (
            sp.factor(characteristic - expected_restricted[eigenvalue]) == 0
        )

    curve_certificate = _curve_stabilizer_certificate(generators)

    report: dict[str, object] = {
        "schema_version": 1,
        "claim_scope": "exact stabilizer and branching certificate at c=0 only",
        "plane_stabilizer_rank": int(plane_action.rank()),
        "plane_stabilizer_dimension": 36 - int(plane_action.rank()),
        "pointwise_stabilizer_rank": int(pointwise_action.rank()),
        "pointwise_stabilizer_dimension": 36 - int(pointwise_action.rank()),
        "stabilizer_coordinate_vectors": [
            [int(value) for value in vector] for vector in stabilizer_coordinates
        ],
        "so3_bracket_identities": bracket_identities,
        "casimir_symmetric": casimir == casimir.T,
        "casimir_eigenvalue_multiplicities": casimir_eigenvalues,
        "expected_casimir_eigenvalue_multiplicities": expected_casimir_eigenvalues,
        "adjoint_branching": "2*V7 + 2*V5 + 4*V3",
        "information_commutes_with_stabilizer": commutator_identities,
        "restricted_characteristic_polynomials": restricted_characteristic_polynomials,
        "restricted_characteristic_polynomial_identities": restricted_identities,
        "curve_stabilizer_certificate": curve_certificate,
        "whole_curve_fixed_subgroup_claimed": False,
    }
    report["passed"] = bool(
        report["plane_stabilizer_dimension"] == 3
        and report["pointwise_stabilizer_dimension"] == 0
        and all(bracket_identities.values())
        and report["casimir_symmetric"]
        and casimir_eigenvalues == expected_casimir_eigenvalues
        and all(commutator_identities)
        and all(restricted_identities.values())
        and curve_certificate["rank_witness_identity"]
        and curve_certificate["rank_witness_nonzero_on_open_interval"]
        and curve_certificate["curve_plane_stabilizer_is_full_so3"]
        and not report["whole_curve_fixed_subgroup_claimed"]
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
