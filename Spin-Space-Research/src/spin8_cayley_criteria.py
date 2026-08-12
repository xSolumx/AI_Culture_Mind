"""Exact design-criterion consequences of the balanced Cayley spectrum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp

from spin8_cayley_blocks import expected_block_characteristics


def _positive_bernstein_coefficients_on_unit_interval(
    polynomial: sp.Poly,
) -> list[sp.Rational]:
    """Return exact univariate Bernstein coefficients at the native degree."""

    variable = polynomial.gens[0]
    degree = int(polynomial.degree())
    power_coefficients = [polynomial.nth(index) for index in range(degree + 1)]
    return [
        sp.factor(
            sum(
                sp.Rational(sp.binomial(row, column), sp.binomial(degree, column))
                * power_coefficients[column]
                for column in range(row + 1)
            )
        )
        for row in range(degree + 1)
    ]


def exact_cayley_criteria_certificate() -> dict[str, object]:
    """Derive D-, A-, and inverse-Frobenius laws from the exact spectrum."""

    cayley, eigenvalue, square = sp.symbols("c lambda z", real=True)
    block_characteristics = expected_block_characteristics(cayley, eigenvalue)
    characteristic = sp.factor(sp.prod(block_characteristics))
    polynomial = sp.Poly(sp.expand(characteristic), eigenvalue)
    coefficients = polynomial.all_coeffs()

    trace = sp.factor(-coefficients[1])
    second_elementary = coefficients[2]
    trace_square = sp.factor(trace**2 - 2 * second_elementary)
    participation_ratio = sp.factor(trace**2 / trace_square)
    determinant = sp.factor(characteristic.subs(eigenvalue, 0))
    trace_inverse = sp.factor(
        -sp.diff(characteristic, eigenvalue).subs(eigenvalue, 0) / determinant
    )
    trace_inverse_square = sp.factor(
        -(
            sp.diff(characteristic, eigenvalue, 2).subs(eigenvalue, 0) * determinant
            - sp.diff(characteristic, eigenvalue).subs(eigenvalue, 0) ** 2
        )
        / determinant**2
    )

    determinant_in_square = (1 - square) ** 3 * (9 - square) ** 2 / sp.Integer(1024)
    trace_inverse_in_square = (11 * square**2 - 206 * square + 387) / (
        (1 - square) * (9 - square)
    )
    trace_inverse_square_in_square = (
        19 * square**4 - 76 * square**3 + 786 * square**2 + 2676 * square + 8883
    ) / ((1 - square) ** 2 * (9 - square) ** 2)

    determinant_derivative = sp.factor(sp.diff(determinant_in_square, square))
    trace_inverse_derivative = sp.factor(sp.diff(trace_inverse_in_square, square))
    trace_inverse_square_derivative = sp.factor(
        sp.diff(trace_inverse_square_in_square, square)
    )

    inverse_square_sign_polynomial = sp.Poly(
        -(19 * square**4 + 8 * square**3 + 630 * square**2 - 336 * square - 12609),
        square,
    )
    inverse_square_sign_bernstein = _positive_bernstein_coefficients_on_unit_interval(
        inverse_square_sign_polynomial
    )

    endpoint_blocks = [
        sp.factor(block.subs(cayley, 1)) for block in block_characteristics
    ]
    endpoint_small_eigenvalue_slopes = [
        sp.factor(
            sp.diff(block, cayley).subs({cayley: 1, eigenvalue: 0})
            / (2 * sp.diff(block, eigenvalue).subs({cayley: 1, eigenvalue: 0}))
        )
        for block in block_characteristics[:3]
    ]
    endpoint_maximum_eigenvalue = 2 + sp.sqrt(2)
    condition_number_leading_coefficient = sp.factor(8 * endpoint_maximum_eigenvalue)

    identities = {
        "determinant": sp.cancel(
            determinant - determinant_in_square.subs(square, cayley**2)
        )
        == 0,
        "trace_inverse": sp.cancel(
            trace_inverse - trace_inverse_in_square.subs(square, cayley**2)
        )
        == 0,
        "trace_inverse_square": sp.cancel(
            trace_inverse_square
            - trace_inverse_square_in_square.subs(square, cayley**2)
        )
        == 0,
    }
    passed = (
        all(identities.values())
        and trace == 35
        and trace_square == 67
        and participation_ratio == sp.Rational(1225, 67)
        and sp.factor(trace_inverse_in_square.subs(square, 0)) == 43
        and sp.factor(trace_inverse_square_in_square.subs(square, 0))
        == sp.Rational(329, 3)
        and sp.cancel(
            determinant_derivative
            + (square - 9) * (square - 1) ** 2 * (5 * square - 29) / 1024
        )
        == 0
        and sp.cancel(
            trace_inverse_derivative
            - 96
            * (square**2 - 6 * square + 21)
            / ((square - 9) ** 2 * (square - 1) ** 2)
        )
        == 0
        and all(bool(value > 0) for value in inverse_square_sign_bernstein)
        and endpoint_small_eigenvalue_slopes == [sp.Rational(1, 8)] * 3
        and endpoint_blocks[0]
        == eigenvalue
        * (eigenvalue - 1) ** 3
        * (eigenvalue**2 - 4 * eigenvalue + 2)
        * (eigenvalue**2 - 3 * eigenvalue + 1)
        and endpoint_blocks[1]
        == endpoint_blocks[2]
        == eigenvalue
        * (eigenvalue - 2) ** 2
        * (eigenvalue - 1) ** 3
        * (eigenvalue**2 - 3 * eigenvalue + 1)
    )
    return {
        "theorem": "simultaneous exact Cayley design criteria",
        "domain": "z = c^2 in [0, 1)",
        "trace": str(trace),
        "trace_square": str(trace_square),
        "second_moment_participation_ratio": str(participation_ratio),
        "determinant": str(sp.factor(determinant_in_square)),
        "trace_inverse": str(sp.factor(trace_inverse_in_square)),
        "trace_inverse_square": str(sp.factor(trace_inverse_square_in_square)),
        "determinant_derivative": str(determinant_derivative),
        "trace_inverse_derivative": str(trace_inverse_derivative),
        "trace_inverse_square_derivative": str(trace_inverse_square_derivative),
        "inverse_square_sign_bernstein_coefficients": [
            str(value) for value in inverse_square_sign_bernstein
        ],
        "balanced_trace_inverse": "43",
        "balanced_trace_inverse_square": "329/3",
        "endpoint_block_characteristics": [str(block) for block in endpoint_blocks],
        "endpoint_small_eigenvalue_slopes_in_1_minus_z": [
            str(value) for value in endpoint_small_eigenvalue_slopes
        ],
        "endpoint_maximum_eigenvalue": str(endpoint_maximum_eigenvalue),
        "condition_number_leading_coefficient": str(
            condition_number_leading_coefficient
        ),
        "endpoint_asymptotics": {
            "three_small_eigenvalues": "lambda_j = (1-z)/8 + O((1-z)^2)",
            "trace_inverse": "24/(1-z) + O(1)",
            "trace_inverse_square": "192/(1-z)^2 + O(1/(1-z))",
            "condition_number": "8*(2 + sqrt(2))/(1-z) + O(1)",
        },
        "identities": identities,
        "conclusion": (
            "The Cayley-null orbit uniquely maximizes determinant and uniquely "
            "minimizes trace(I^-1) and trace(I^-2) on the unoriented interior."
        ),
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = exact_cayley_criteria_certificate()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("exact Cayley design-criterion certificate failed")


if __name__ == "__main__":
    main()
