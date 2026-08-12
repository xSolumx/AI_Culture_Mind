"""Exact approximate-design audit for the balanced Spin(8) sensor.

The existing five-query problem assigns one unit of cost to each of five pure
queries.  Classical approximate design theory instead allows arbitrary
nonnegative observation weights.  These are different optimization domains.

This module applies the Kiefer--Wolfowitz sensitivity criterion exactly.  It
also gives an exact one-parameter reweighting witness showing that the equal
five-query design is not optimal in the broader weighted domain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp

from spin8_cayley_spectrum import symbolic_query_projector, symbolic_triality_generators


def _basis() -> list[list[sp.Integer]]:
    return [[sp.Integer(row == column) for column in range(8)] for row in range(8)]


def _sensitivity_matrix(
    inverse_information: sp.Matrix,
    generators: list[list[list[list[sp.Rational]]]],
    view: int,
) -> sp.Matrix:
    result = sp.zeros(8)
    for left in range(28):
        left_generator = sp.Matrix(generators[view][left])
        for right in range(28):
            coefficient = inverse_information[left, right]
            if coefficient:
                result += (
                    coefficient * left_generator.T * sp.Matrix(generators[view][right])
                )
    return sp.simplify((result + result.T) / 2)


def run() -> dict[str, object]:
    generators = symbolic_triality_generators()
    basis = _basis()
    query_rows = (
        (0, basis[0]),
        (1, basis[0]),
        (1, basis[1]),
        (2, basis[2]),
        (2, basis[4]),
    )
    projectors = [
        symbolic_query_projector(view, state, generators) for view, state in query_rows
    ]
    information = sum(projectors, sp.zeros(28))
    inverse = information.inv()

    sensitivity = [_sensitivity_matrix(inverse, generators, view) for view in range(3)]
    sensitivity_spectra = []
    for view, matrix in enumerate(sensitivity):
        eigenvalues = matrix.eigenvals()
        sensitivity_spectra.append(
            {
                "view": view,
                "diagonal_in_maintained_basis": all(
                    matrix[row, column] == 0
                    for row in range(8)
                    for column in range(8)
                    if row != column
                ),
                "eigenvalue_multiplicities": {
                    str(value): int(multiplicity)
                    for value, multiplicity in sorted(
                        eigenvalues.items(), key=lambda item: item[0]
                    )
                },
                "maximum": str(max(eigenvalues)),
            }
        )

    support_sensitivities = [
        sp.factor((sp.Matrix(state).T * sensitivity[view] * sp.Matrix(state))[0])
        for view, state in query_rows
    ]
    maximum_unnormalized = max(
        value for matrix in sensitivity for value in matrix.eigenvals()
    )
    # The approximate design information matrix is information/5, so its
    # sensitivity is five times the unnormalized value.  The parameter count
    # is 28.
    approximate_maximum = 5 * maximum_unnormalized

    alpha = sp.symbols("alpha")
    beta = (5 - alpha) / 4
    weighted_information = alpha * projectors[0] + beta * sum(
        projectors[1:], sp.zeros(28)
    )
    weighted_determinant = sp.factor(weighted_information.det())
    expected_determinant = sp.factor(
        -(alpha**3) * (alpha - 5) ** 21 * (7 * alpha + 5) ** 4 / 2**60
    )
    derivative = sp.factor(sp.diff(weighted_determinant, alpha))
    expected_derivative = sp.factor(
        -(alpha**2)
        * (alpha - 5) ** 20
        * (7 * alpha + 5) ** 3
        * (196 * alpha**2 - 125 * alpha - 75)
        / 2**60
    )
    optimum = sp.Rational(125, 392) + 5 * sp.sqrt(2977) / 392
    optimum_beta = sp.factor((5 - optimum) / 4)
    rational_alpha = sp.Rational(101, 100)
    rational_beta = sp.factor((5 - rational_alpha) / 4)
    balanced_determinant = sp.Rational(81, 1024)
    rational_determinant = sp.factor(weighted_determinant.subs(alpha, rational_alpha))
    rational_gain = sp.factor(rational_determinant - balanced_determinant)
    alpha_zero_information = weighted_information.subs(alpha, 0)
    alpha_five_information = weighted_information.subs(alpha, 5)

    # Eight orthogonal coordinate probes in any one view form an exact tight
    # design: their normalized information is I_28/4.  Its sensitivity is
    # identically 28, so the equivalence criterion is saturated globally.
    per_view_sums = []
    trace_quadratic_identities = []
    for view in range(3):
        coordinate_projectors = [
            symbolic_query_projector(view, basis[coordinate], generators)
            for coordinate in range(8)
        ]
        total = sum(
            coordinate_projectors,
            sp.zeros(28),
        )
        per_view_sums.append(total)
        diagonal = all(sp.trace(projector) == 7 for projector in coordinate_projectors)
        cross = all(
            sp.trace(
                symbolic_query_projector(
                    view,
                    [basis[left][index] + basis[right][index] for index in range(8)],
                    generators,
                )
                - coordinate_projectors[left]
                - coordinate_projectors[right]
            )
            == 0
            for left in range(8)
            for right in range(left + 1, 8)
        )
        trace_quadratic_identities.append(bool(diagonal and cross))
    uniform_design_exact = all(total == 2 * sp.eye(28) for total in per_view_sums)
    unit_probe_trace_exact = all(trace_quadratic_identities)

    passed = bool(
        information.det() == balanced_determinant
        and sum(support_sensitivities) == 28
        and support_sensitivities == [sp.Rational(17, 3)] + [sp.Rational(67, 12)] * 4
        and maximum_unnormalized == 15
        and approximate_maximum == 75
        and weighted_determinant == expected_determinant
        and derivative == expected_derivative
        and sp.factor(196 * optimum**2 - 125 * optimum - 75) == 0
        and 0 < optimum < 5
        and rational_gain > 0
        and alpha_zero_information.rank() == 25
        and alpha_five_information.rank() == 7
        and uniform_design_exact
        and unit_probe_trace_exact
    )
    return {
        "experiment": "exact Kiefer-Wolfowitz audit of Spin8 triality sensing",
        "domains": {
            "existing_problem": "exactly five unit-cost pure queries",
            "audited_broader_problem": (
                "an approximate design measure with arbitrary nonnegative weights"
            ),
        },
        "balanced_information": {
            "determinant": str(information.det()),
            "rank": int(information.rank()),
            "support_sensitivities_for_I_sum": [
                str(value) for value in support_sensitivities
            ],
            "support_sensitivities_sum": str(sum(support_sensitivities)),
            "view_sensitivity_spectra_for_I_sum": sensitivity_spectra,
            "maximum_sensitivity_for_I_sum": str(maximum_unnormalized),
            "maximum_sensitivity_for_normalized_I_over_5": str(approximate_maximum),
            "kiefer_wolfowitz_threshold": "28",
            "passes_approximate_D_optimality_criterion": bool(
                approximate_maximum <= 28
            ),
        },
        "exact_reweighting_counterexample": {
            "weight_convention": "alpha on V query; beta=(5-alpha)/4 on each other query",
            "determinant_polynomial": str(weighted_determinant),
            "derivative": str(derivative),
            "unique_interior_maximizer_alpha": str(optimum),
            "unique_interior_maximizer_alpha_decimal": float(optimum),
            "corresponding_beta": str(optimum_beta),
            "corresponding_beta_decimal": float(optimum_beta),
            "rational_witness_alpha": str(rational_alpha),
            "rational_witness_beta": str(rational_beta),
            "rational_witness_determinant": str(rational_determinant),
            "exact_gain_over_81_over_1024": str(rational_gain),
            "relative_gain_decimal": float(
                rational_determinant / balanced_determinant - 1
            ),
            "weight_simplex_boundaries": {
                "alpha_zero_rank": int(alpha_zero_information.rank()),
                "alpha_zero_determinant_vanishing_order": 3,
                "alpha_five_rank": int(alpha_five_information.rank()),
                "alpha_five_determinant_vanishing_order": 21,
                "interpretation": (
                    "Both ends of this fixed-support weight segment are singular; "
                    "the exact optimum lies in its interior."
                ),
            },
        },
        "global_approximate_design": {
            "construction": "uniform mass on one orthonormal basis in any one view",
            "exact_information": "M=I_28/4",
            "global_sensitivity": "28 at every unit probe in every view",
            "unit_probe_trace_seven_exact_by_polarization": unit_probe_trace_exact,
            "kiefer_wolfowitz_saturated": uniform_design_exact,
            "interpretation": (
                "This is globally D-optimal in the approximate-design domain. "
                "It uses eight support points and does not solve the exact "
                "five-query problem."
            ),
        },
        "proof_layers": {
            "exact_repository_premises": (
                "The eight-point information matrix, constant unit-probe "
                "sensitivity, trace-seven identity, and five-query reweighting "
                "counterexample are computed exactly."
            ),
            "classical_global_input": (
                "The Kiefer--Wolfowitz general equivalence theorem identifies "
                "a design with sensitivity bounded by the parameter dimension "
                "as globally D-optimal in the approximate-design domain."
            ),
            "deduction": (
                "The exact sensitivity is identically 28, so the eight-support "
                "isotropic design saturates the equivalence criterion globally."
            ),
        },
        "correction": (
            "The exact five-query and approximate weighted-design problems must "
            "not be called the same global D-optimality problem. Equal five-query "
            "optimality remains open; approximate-design optimality is solved by "
            "the isotropic eight-point construction, and the equal balanced sensor "
            "is not optimal in that broader domain."
        ),
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run()
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("approximate-design audit failed")


if __name__ == "__main__":
    main()
