"""Exact local and one-probe finite geometry of the balanced five-query sensor.

The global equal-cost five-query problem is still open.  This module closes a
strictly smaller question without sampling: is the balanced coordinate sensor
at least a genuine local maximum when probes may move continuously on their
seven-spheres?

The answer is yes.  The exact Riemannian Hessian of ``log det I`` on the
35-dimensional product of spheres has a 28-dimensional kernel, exactly equal
to the tangent space of the shared Spin(8) orbit, and is negative definite on
the seven-dimensional quotient.  We also enumerate all 35 coordinate
great-circle replacements and reduce their determinants exactly modulo
``c^2+s^2=1``.  This is a local theorem plus a finite curve atlas, not a global
five-query proof.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import sympy as sp

from spin8_cayley_spectrum import (
    symbolic_query_projector,
    symbolic_triality_generators,
)

DIMENSION = 8
ACTION_DIMENSION = 28
QUERY_ROWS = ((0, 0), (1, 0), (1, 1), (2, 2), (2, 4))
BALANCED_DETERMINANT = sp.Rational(81, 1024)


def _basis() -> list[sp.Matrix]:
    return [sp.eye(DIMENSION)[:, index] for index in range(DIMENSION)]


def _polarized_projector(
    view: int,
    left: sp.Matrix,
    right: sp.Matrix,
    generators: list[list[list[list[sp.Rational]]]],
) -> sp.Matrix:
    """Return the first derivative of ``P(left+t right)`` at zero."""

    return (
        symbolic_query_projector(view, list(left + right), generators)
        - symbolic_query_projector(view, list(left - right), generators)
    ) / 2


def _hessian_certificate(
    generators: list[list[list[list[sp.Rational]]]],
) -> dict[str, object]:
    basis = _basis()
    queries = [(view, basis[coordinate]) for view, coordinate in QUERY_ROWS]
    projectors = [
        symbolic_query_projector(view, list(state), generators)
        for view, state in queries
    ]
    information = sum(projectors, sp.zeros(ACTION_DIMENSION))
    inverse = information.inv()

    tangent_coordinates = []
    for query_index, ((view, radial), (_, state)) in enumerate(
        zip(QUERY_ROWS, queries, strict=True)
    ):
        for coordinate in range(DIMENSION):
            if coordinate == radial:
                continue
            direction = basis[coordinate]
            first = _polarized_projector(view, state, direction, generators)
            tangent_coordinates.append(
                (query_index, view, radial, coordinate, direction, first)
            )

    first_derivative = [sp.trace(inverse * row[5]) for row in tangent_coordinates]
    hessian = sp.zeros(len(tangent_coordinates))
    for left_index, left in enumerate(tangent_coordinates):
        left_query, left_view, _, left_coordinate, left_direction, left_first = left
        for right_index, right in enumerate(tangent_coordinates):
            (
                right_query,
                _,
                _,
                right_coordinate,
                right_direction,
                right_first,
            ) = right
            second = sp.zeros(ACTION_DIMENSION)
            if left_query == right_query:
                if left_coordinate == right_coordinate:
                    second = 2 * (
                        symbolic_query_projector(
                            left_view, list(left_direction), generators
                        )
                        - projectors[left_query]
                    )
                else:
                    second = _polarized_projector(
                        left_view,
                        left_direction,
                        right_direction,
                        generators,
                    )
            hessian[left_index, right_index] = sp.factor(
                sp.trace(inverse * second)
                - sp.trace(inverse * left_first * inverse * right_first)
            )

    # A shared infinitesimal Spin(8) action moves every query at once.  Express
    # those 28 orbit directions in the same 35 tangent coordinates.
    orbit_tangent = sp.Matrix(
        len(tangent_coordinates),
        ACTION_DIMENSION,
        lambda row, plane: generators[tangent_coordinates[row][1]][plane][
            tangent_coordinates[row][3]
        ][tangent_coordinates[row][2]],
    )
    eigenvalues = hessian.eigenvals()
    expected = {
        sp.Integer(0): 28,
        sp.Integer(-22): 4,
        -sp.Rational(158, 9): 2,
        -sp.Rational(232, 9): 1,
    }
    orbit_rank = int(orbit_tangent.rank())
    kernel_dimension = len(tangent_coordinates) - int(hessian.rank())
    orbit_is_kernel = hessian * orbit_tangent == sp.zeros(
        len(tangent_coordinates), ACTION_DIMENSION
    )
    passed = bool(
        information.det() == BALANCED_DETERMINANT
        and not any(first_derivative)
        and eigenvalues == expected
        and orbit_rank == ACTION_DIMENSION
        and kernel_dimension == ACTION_DIMENSION
        and orbit_is_kernel
    )
    return {
        "tangent_dimension": len(tangent_coordinates),
        "first_derivative_nonzero_count": sum(value != 0 for value in first_derivative),
        "hessian_rank": int(hessian.rank()),
        "hessian_nullity": kernel_dimension,
        "hessian_eigenvalue_multiplicities": {
            str(value): int(multiplicity)
            for value, multiplicity in sorted(
                eigenvalues.items(), key=lambda item: item[0]
            )
        },
        "shared_spin8_orbit_tangent_rank": orbit_rank,
        "hessian_annihilates_shared_spin8_orbit_exactly": orbit_is_kernel,
        "kernel_equals_shared_spin8_orbit_by_rank": (
            orbit_is_kernel and orbit_rank == kernel_dimension
        ),
        "quotient_dimension": len(tangent_coordinates) - orbit_rank,
        "quotient_hessian_is_negative_definite": all(
            value < 0 for value in eigenvalues if value != 0
        ),
        "interpretation": (
            "The balanced sensor is a strict local maximum of log det I after "
            "quotienting by the shared Spin(8) symmetry. This is not a global "
            "five-query optimality proof."
        ),
        "passed": passed,
    }


def _great_circle_certificate(
    generators: list[list[list[list[sp.Rational]]]],
) -> dict[str, object]:
    basis = _basis()
    queries = [(view, basis[coordinate]) for view, coordinate in QUERY_ROWS]
    projectors = [
        symbolic_query_projector(view, list(state), generators)
        for view, state in queries
    ]
    information = sum(projectors, sp.zeros(ACTION_DIMENSION))
    cosine, sine = sp.symbols("c s", real=True)
    squared_cosine = sp.symbols("z", real=True)
    classes: dict[str, dict[str, object]] = {}
    grouped: defaultdict[str, list[list[int]]] = defaultdict(list)

    for query_index, ((view, radial), (_, state)) in enumerate(
        zip(QUERY_ROWS, queries, strict=True)
    ):
        for target in range(DIMENSION):
            if target == radial:
                continue
            moved = cosine * state + sine * basis[target]
            determinant = (
                information
                - projectors[query_index]
                + symbolic_query_projector(view, list(moved), generators)
            ).det(method="domain-ge")
            reduced = sp.factor(
                sp.rem(
                    sp.Poly(determinant, sine),
                    sp.Poly(sine**2 - (1 - cosine**2), sine),
                ).as_expr()
            )
            key = str(reduced)
            grouped[key].append([query_index, radial, target])

    expected = {
        str(BALANCED_DETERMINANT): {
            "count": 15,
            "boundary_rank": 28,
            "strictly_lower_for_abs_c_less_than_one": False,
            "difference_positive_factor": None,
        },
        str(3 * cosine**6 * (cosine**2 + 2) * (cosine**2 + 5) ** 2 / 4096): {
            "count": 12,
            "boundary_rank": 25,
            "strictly_lower_for_abs_c_less_than_one": True,
            "difference_positive_factor": (
                3
                * (squared_cosine + 3)
                * (
                    squared_cosine**4
                    + 10 * squared_cosine**3
                    + 28 * squared_cosine**2
                    + 24 * squared_cosine
                    + 36
                )
                / 4096
            ),
        },
        str(cosine**6 * (cosine**2 + 1) * (4 * cosine**2 + 5) ** 2 / 2048): {
            "count": 4,
            "boundary_rank": 25,
            "strictly_lower_for_abs_c_less_than_one": True,
            "difference_positive_factor": (
                16 * squared_cosine**5
                + 72 * squared_cosine**4
                + 137 * squared_cosine**3
                + 162 * squared_cosine**2
                + 162 * squared_cosine
                + 162
            )
            / 2048,
        },
        str(cosine**6 * (cosine**2 + 8) ** 2 / 1024): {
            "count": 4,
            "boundary_rank": 25,
            "strictly_lower_for_abs_c_less_than_one": True,
            "difference_positive_factor": (
                squared_cosine**4
                + 17 * squared_cosine**3
                + 81 * squared_cosine**2
                + 81 * squared_cosine
                + 81
            )
            / 1024,
        },
    }

    for formula, rows in grouped.items():
        representative_query, _, representative_target = rows[0]
        view, _ = QUERY_ROWS[representative_query]
        boundary_information = (
            information
            - projectors[representative_query]
            + symbolic_query_projector(
                view, list(basis[representative_target]), generators
            )
        )
        classes[formula] = {
            "count": len(rows),
            "directions_query_radial_target": rows,
            "rank_at_c_zero": int(boundary_information.rank()),
            "strictly_lower_for_abs_c_less_than_one": expected.get(formula, {}).get(
                "strictly_lower_for_abs_c_less_than_one"
            ),
            "difference_positive_factor_in_z": (
                None
                if expected.get(formula, {}).get("difference_positive_factor") is None
                else str(sp.factor(expected[formula]["difference_positive_factor"]))
            ),
        }

    sign_certificates = []
    for formula, metadata in expected.items():
        expression = sp.sympify(formula, locals={"c": cosine})
        factor = metadata["difference_positive_factor"]
        if factor is None:
            identity = expression == BALANCED_DETERMINANT
            positive_coefficients = True
        else:
            identity = (
                sp.factor(
                    expression
                    - BALANCED_DETERMINANT
                    - (cosine**2 - 1) * factor.subs(squared_cosine, cosine**2)
                )
                == 0
            )
            positive_coefficients = all(
                coefficient > 0
                for coefficient in sp.Poly(factor, squared_cosine).all_coeffs()
            )
        sign_certificates.append(
            {
                "determinant": formula,
                "difference_factor_identity": identity,
                "positive_coefficients_on_z_nonnegative": positive_coefficients,
            }
        )

    matches = bool(
        set(grouped) == set(expected)
        and all(len(grouped[key]) == value["count"] for key, value in expected.items())
        and all(
            classes[key]["rank_at_c_zero"] == value["boundary_rank"]
            for key, value in expected.items()
        )
        and all(
            row["difference_factor_identity"]
            and row["positive_coefficients_on_z_nonnegative"]
            for row in sign_certificates
        )
    )
    return {
        "curve_count": sum(len(rows) for rows in grouped.values()),
        "determinant_class_count": len(grouped),
        "circle_relation": "c^2+s^2=1",
        "determinant_classes": classes,
        "exact_sign_certificates": sign_certificates,
        "flat_orbit_curve_count": len(grouped[str(BALANCED_DETERMINANT)]),
        "strictly_decreasing_nonorbit_curve_count": sum(
            len(grouped[key])
            for key, value in expected.items()
            if value["strictly_lower_for_abs_c_less_than_one"]
        ),
        "nonorbit_boundary_rank": 25,
        "nonorbit_boundary_determinant_vanishing_order_in_c": 6,
        "interpretation": (
            "Every coordinate great circle is either exactly flat, with its tangent "
            "in the shared symmetry kernel, or has strictly smaller determinant "
            "for -1<c<1 and reaches a rank-25 boundary at c=0. General coupled "
            "finite deformations remain outside this certificate."
        ),
        "passed": matches,
    }


@lru_cache(maxsize=1)
def run() -> dict[str, object]:
    generators = symbolic_triality_generators()
    hessian = _hessian_certificate(generators)
    circles = _great_circle_certificate(generators)
    return {
        "experiment": "exact continuous local geometry of the balanced Spin8 sensor",
        "design": {
            "allocation": [1, 2, 2],
            "queries_view_coordinate": [list(row) for row in QUERY_ROWS],
            "determinant": str(BALANCED_DETERMINANT),
        },
        "riemannian_hessian": hessian,
        "coordinate_great_circle_atlas": circles,
        "claim_boundary": (
            "Exact strict local maximality modulo Spin(8), plus all 35 single-probe "
            "coordinate great circles. This neither excludes distant coupled "
            "non-vertex competitors nor proves global equal-five-query optimality."
        ),
        "passed": bool(hessian["passed"] and circles["passed"]),
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
        raise SystemExit("five-query local-geometry certificate failed")


if __name__ == "__main__":
    main()
