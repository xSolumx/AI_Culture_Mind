"""Exact star-family certificate for the Spin(8) Dirac--Gram conjecture."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import sympy as sp

from spin8_cayley_spectrum import (
    symbolic_query_projector,
    symbolic_triality_generators,
)

VARIABLES = sp.symbols("u v w z")
U, V, W, Z = VARIABLES

NODE_SETS = {
    "discovery": {
        "u": (
            0,
            sp.Rational(1, 8),
            sp.Rational(1, 4),
            sp.Rational(1, 2),
            sp.Rational(3, 4),
        ),
        "z": (
            0,
            sp.Rational(1, 10),
            sp.Rational(1, 6),
            sp.Rational(1, 4),
            sp.Rational(1, 3),
            sp.Rational(1, 2),
            sp.Rational(2, 3),
            sp.Rational(3, 4),
        ),
    },
    "confirmation": {
        "u": (
            0,
            sp.Rational(1, 10),
            sp.Rational(1, 5),
            sp.Rational(2, 5),
            sp.Rational(3, 5),
        ),
        "z": (
            0,
            sp.Rational(1, 12),
            sp.Rational(1, 7),
            sp.Rational(1, 5),
            sp.Rational(2, 7),
            sp.Rational(2, 5),
            sp.Rational(3, 5),
            sp.Rational(4, 5),
        ),
    },
}

HOLDOUT_PARAMETERS = (
    sp.Rational(1, 11),
    sp.Rational(3, 11),
)
HOLDOUT_CAYLEY_PARAMETERS = (
    sp.Rational(1, 13),
    sp.Rational(3, 13),
)


def rational_circle(parameter: sp.Expr | int) -> tuple[sp.Expr, sp.Expr]:
    parameter = sp.sympify(parameter)
    denominator = 1 + parameter**2
    return 2 * parameter / denominator, (1 - parameter**2) / denominator


def _vector(
    leading: sp.Expr,
    complement: sp.Expr,
    residual: list[sp.Expr],
    basis: list[list[sp.Integer]],
) -> list[sp.Expr]:
    return [
        leading * basis[0][index] + complement * residual[index] for index in range(8)
    ]


def _tensor_interpolate(
    values: dict[tuple[int, int, int, int], sp.Expr],
    spatial_nodes: list[sp.Expr],
    cayley_nodes: list[sp.Expr],
) -> sp.Poly:
    level_three = {
        (left, middle, right): sp.interpolate(
            [
                (cayley_nodes[index], values[left, middle, right, index])
                for index in range(len(cayley_nodes))
            ],
            Z,
        )
        for left in range(len(spatial_nodes))
        for middle in range(len(spatial_nodes))
        for right in range(len(spatial_nodes))
    }
    level_two = {
        (left, middle): sp.interpolate(
            [
                (spatial_nodes[index], level_three[left, middle, index])
                for index in range(len(spatial_nodes))
            ],
            W,
        )
        for left in range(len(spatial_nodes))
        for middle in range(len(spatial_nodes))
    }
    level_one = {
        left: sp.interpolate(
            [
                (spatial_nodes[index], level_two[left, index])
                for index in range(len(spatial_nodes))
            ],
            V,
        )
        for left in range(len(spatial_nodes))
    }
    return sp.Poly(
        sp.interpolate(
            [
                (spatial_nodes[index], level_one[index])
                for index in range(len(spatial_nodes))
            ],
            U,
        ),
        *VARIABLES,
    )


def polynomial_records(polynomial: sp.Poly) -> list[dict[str, object]]:
    return [
        {"powers": list(powers), "coefficient": str(coefficient)}
        for powers, coefficient in polynomial.terms()
    ]


def polynomial_from_records(records: list[dict[str, object]]) -> sp.Poly:
    """Reconstruct an exact polynomial from canonical artifact records."""

    expression = sp.Integer(0)
    for record in records:
        powers = tuple(int(value) for value in record["powers"])
        coefficient = sp.Rational(str(record["coefficient"]))
        monomial = sp.prod(
            variable**power for variable, power in zip(VARIABLES, powers, strict=True)
        )
        expression += coefficient * monomial
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


def reconstruct_star_polynomials(node_set: str) -> tuple[sp.Poly, sp.Poly]:
    nodes = NODE_SETS[node_set]
    spatial_pairs = [rational_circle(value) for value in nodes["u"]]
    spatial_squares = [sp.factor(left**2) for left, _ in spatial_pairs]
    cayley_pairs = [rational_circle(value) for value in nodes["z"]]
    cayley_squares = [sp.factor(left**2) for left, _ in cayley_pairs]

    generators = symbolic_triality_generators()
    basis = [[sp.Integer(row == column) for column in range(8)] for row in range(8)]
    fixed = symbolic_query_projector(
        0, basis[0], generators
    ) + symbolic_query_projector(1, basis[0], generators)
    even_values: dict[tuple[int, int, int, int], sp.Expr] = {}
    odd_values: dict[tuple[int, int, int, int], sp.Expr] = {}

    for left_index, (left, left_complement) in enumerate(spatial_pairs):
        positive = symbolic_query_projector(
            1,
            _vector(left, left_complement, basis[1], basis),
            generators,
        )
        for middle_index, (middle, middle_complement) in enumerate(spatial_pairs):
            negative = symbolic_query_projector(
                2,
                _vector(middle, middle_complement, basis[2], basis),
                generators,
            )
            for right_index, (right, right_complement) in enumerate(spatial_pairs):
                delta = (
                    (1 - spatial_squares[left_index])
                    * (1 - spatial_squares[middle_index])
                    * (1 - spatial_squares[right_index])
                )
                for cayley_index, (cayley, sine) in enumerate(cayley_pairs):
                    determinants = []
                    signs = (1,) if cayley == 0 else (1, -1)
                    for sign in signs:
                        final_basis = [
                            sign * cayley * basis[3][index] + sine * basis[4][index]
                            for index in range(8)
                        ]
                        information = (
                            fixed
                            + positive
                            + negative
                            + symbolic_query_projector(
                                2,
                                _vector(right, right_complement, final_basis, basis),
                                generators,
                            )
                        )
                        determinants.append(
                            sp.factor(
                                1024 * information.det(method="domain-ge") / delta**3
                            )
                        )
                    even_values[left_index, middle_index, right_index, cayley_index] = (
                        determinants[0]
                        if cayley == 0
                        else sp.factor((determinants[0] + determinants[1]) / 2)
                    )
                    if left_index and middle_index and right_index and cayley_index:
                        odd_values[
                            left_index - 1,
                            middle_index - 1,
                            right_index - 1,
                            cayley_index - 1,
                        ] = sp.factor(
                            (determinants[0] - determinants[1])
                            / (
                                2
                                * left
                                * middle
                                * right
                                * left_complement
                                * middle_complement
                                * right_complement
                                * cayley
                            )
                        )

    even = _tensor_interpolate(even_values, spatial_squares, cayley_squares)
    odd = _tensor_interpolate(odd_values, spatial_squares[1:], cayley_squares[1:])
    return even, odd


def exact_holdout_certificate(
    even: sp.Poly,
    odd: sp.Poly,
) -> dict[str, object]:
    """Compare the reconstructed formula with exact off-grid determinants."""

    generators = symbolic_triality_generators()
    basis = [[sp.Integer(row == column) for column in range(8)] for row in range(8)]
    fixed = symbolic_query_projector(
        0, basis[0], generators
    ) + symbolic_query_projector(1, basis[0], generators)
    maximum_error = sp.Integer(0)
    case_count = 0

    for (
        left_parameter,
        middle_parameter,
        right_parameter,
        cayley_parameter,
    ) in itertools.product(
        HOLDOUT_PARAMETERS,
        HOLDOUT_PARAMETERS,
        HOLDOUT_PARAMETERS,
        HOLDOUT_CAYLEY_PARAMETERS,
    ):
        left, left_complement = rational_circle(left_parameter)
        middle, middle_complement = rational_circle(middle_parameter)
        right, right_complement = rational_circle(right_parameter)
        cayley, sine = rational_circle(cayley_parameter)
        delta = left_complement**2 * middle_complement**2 * right_complement**2
        substitutions = {
            U: left**2,
            V: middle**2,
            W: right**2,
            Z: cayley**2,
        }
        positive = symbolic_query_projector(
            1,
            _vector(left, left_complement, basis[1], basis),
            generators,
        )
        negative = symbolic_query_projector(
            2,
            _vector(middle, middle_complement, basis[2], basis),
            generators,
        )
        for sign in (1, -1):
            final_basis = [
                sign * cayley * basis[3][index] + sine * basis[4][index]
                for index in range(8)
            ]
            information = (
                fixed
                + positive
                + negative
                + symbolic_query_projector(
                    2,
                    _vector(right, right_complement, final_basis, basis),
                    generators,
                )
            )
            observed = sp.factor(1024 * information.det(method="domain-ge") / delta**3)
            orientation = (
                sign
                * left
                * middle
                * right
                * left_complement
                * middle_complement
                * right_complement
                * cayley
            )
            predicted = sp.factor(
                even.as_expr().subs(substitutions)
                + orientation * odd.as_expr().subs(substitutions)
            )
            error = sp.factor(observed - predicted)
            if error != 0:
                maximum_error = max(maximum_error, abs(error))
            case_count += 1

    return {
        "rational_frame_count": 16,
        "orientation_count_per_frame": 2,
        "exact_determinant_comparisons": case_count,
        "maximum_exact_error": str(maximum_error),
        "passed": maximum_error == 0 and case_count == 32,
    }


def certificate_from_polynomials(even: sp.Poly, odd: sp.Poly) -> dict[str, object]:
    target = (1 - Z) ** 3 * (9 - Z) ** 2
    margin = sp.Poly(sp.expand(target - even.as_expr()), *VARIABLES)
    orientation_discriminant = sp.Poly(
        sp.expand(
            margin.as_expr() ** 2
            - U * V * W * (1 - U) * (1 - V) * (1 - W) * Z * odd.as_expr() ** 2
        ),
        *VARIABLES,
    )
    margin_degrees, margin_bernstein = bernstein_records(margin)
    discriminant_degrees, discriminant_bernstein = bernstein_records(
        orientation_discriminant
    )
    even_records = polynomial_records(even)
    odd_records = polynomial_records(odd)
    return {
        "even_degrees": [int(even.degree(variable)) for variable in VARIABLES],
        "odd_degrees": [int(odd.degree(variable)) for variable in VARIABLES],
        "even_term_count": len(even.terms()),
        "odd_term_count": len(odd.terms()),
        "even_coefficients_sha256": records_hash(even_records),
        "odd_coefficients_sha256": records_hash(odd_records),
        "even_coefficients": even_records,
        "odd_coefficients": odd_records,
        "margin_bernstein_degrees": list(margin_degrees),
        "margin_bernstein_negative_count": sum(
            1 for value in margin_bernstein if sp.Rational(value) < 0
        ),
        "margin_bernstein_zero_count": margin_bernstein.count("0"),
        "margin_bernstein_sha256": records_hash(
            [{"coefficient": value} for value in margin_bernstein]
        ),
        "margin_bernstein_coefficients": margin_bernstein,
        "orientation_discriminant_bernstein_degrees": list(discriminant_degrees),
        "orientation_discriminant_negative_count": sum(
            1 for value in discriminant_bernstein if sp.Rational(value) < 0
        ),
        "orientation_discriminant_zero_count": discriminant_bernstein.count("0"),
        "orientation_discriminant_bernstein_sha256": records_hash(
            [{"coefficient": value} for value in discriminant_bernstein]
        ),
        "orientation_discriminant_bernstein_coefficients": discriminant_bernstein,
    }


def verify_report(report: dict[str, object]) -> bool:
    """Replay stored maps, Bernstein signs, and exact off-grid determinants.

    This verifier does not rerun the two interpolation grids.  A full replay is
    obtained by calling :func:`run` and comparing the resulting artifact.
    """

    discovery = report["discovery_node_set"]
    confirmation = report["confirmation_node_set"]
    reconstructed: list[tuple[sp.Poly, sp.Poly, dict[str, object]]] = []
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
        for prefix in ("margin", "orientation_discriminant"):
            coefficients = certificate[f"{prefix}_bernstein_coefficients"]
            coefficient_records = [{"coefficient": value} for value in coefficients]
            if (
                records_hash(coefficient_records)
                != certificate[f"{prefix}_bernstein_sha256"]
            ):
                return False
            if sum(1 for value in coefficients if sp.Rational(value) < 0) != 0:
                return False
            zero_count_key = (
                "margin_bernstein_zero_count"
                if prefix == "margin"
                else "orientation_discriminant_zero_count"
            )
            if coefficients.count("0") != certificate[zero_count_key]:
                return False
        even = polynomial_from_records(certificate["even_coefficients"])
        odd = polynomial_from_records(certificate["odd_coefficients"])
        fresh = certificate_from_polynomials(even, odd)
        replay_keys = (
            "even_degrees",
            "odd_degrees",
            "even_term_count",
            "odd_term_count",
            "even_coefficients_sha256",
            "odd_coefficients_sha256",
            "margin_bernstein_degrees",
            "margin_bernstein_negative_count",
            "margin_bernstein_zero_count",
            "margin_bernstein_sha256",
            "orientation_discriminant_bernstein_degrees",
            "orientation_discriminant_negative_count",
            "orientation_discriminant_zero_count",
            "orientation_discriminant_bernstein_sha256",
        )
        if any(fresh[key] != certificate[key] for key in replay_keys):
            return False
        reconstructed.append((even, odd, fresh))

    discovery_even, discovery_odd, _ = reconstructed[0]
    confirmation_even, confirmation_odd, confirmation_fresh = reconstructed[1]
    maps_match = (
        discovery_even == confirmation_even and discovery_odd == confirmation_odd
    )
    fresh_holdouts = exact_holdout_certificate(confirmation_even, confirmation_odd)
    return bool(
        maps_match
        and confirmation_fresh["even_degrees"] == [3, 3, 3, 5]
        and confirmation_fresh["odd_degrees"] == [2, 2, 2, 4]
        and confirmation_fresh["margin_bernstein_negative_count"] == 0
        and confirmation_fresh["orientation_discriminant_negative_count"] == 0
        and fresh_holdouts == report["off_grid_exact_holdouts"]
        and fresh_holdouts["passed"]
    )


def verify_artifact(path: Path) -> bool:
    return verify_report(json.loads(path.read_text(encoding="utf-8")))


def run() -> dict[str, object]:
    discovery_even, discovery_odd = reconstruct_star_polynomials("discovery")
    confirmation_even, confirmation_odd = reconstruct_star_polynomials("confirmation")
    discovery = certificate_from_polynomials(discovery_even, discovery_odd)
    confirmation = certificate_from_polynomials(confirmation_even, confirmation_odd)
    holdouts = exact_holdout_certificate(confirmation_even, confirmation_odd)
    coefficient_match = (
        discovery["even_coefficients_sha256"]
        == confirmation["even_coefficients_sha256"]
        and discovery["odd_coefficients_sha256"]
        == confirmation["odd_coefficients_sha256"]
    )
    passed = (
        coefficient_match
        and confirmation["even_degrees"] == [3, 3, 3, 5]
        and confirmation["odd_degrees"] == [2, 2, 2, 4]
        and confirmation["even_term_count"] == 360
        and confirmation["odd_term_count"] == 86
        and confirmation["margin_bernstein_negative_count"] == 0
        and confirmation["margin_bernstein_zero_count"] == 195
        and confirmation["orientation_discriminant_negative_count"] == 0
        and confirmation["orientation_discriminant_zero_count"] == 2078
        and holdouts["passed"]
    )
    return {
        "experiment": "Spin8 exact signed star-family Dirac--Gram theorem",
        "discovery_node_set": discovery,
        "confirmation_node_set": confirmation,
        "coefficient_maps_match": coefficient_match,
        "off_grid_exact_holdouts": holdouts,
        "global_dirac_gram_theorem_proved": False,
        "star_family_theorem_proved": passed,
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
                key: value
                for key, value in report.items()
                if not key.endswith("node_set")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
