#!/usr/bin/env python3
"""Independent exact verifier for the balanced Cayley-spectrum theorem.

This file intentionally imports no project module and no computer-algebra
package.  It reconstructs the four displayed block characteristic polynomials
with ``fractions.Fraction`` arithmetic, multiplies them, and derives the
determinant, spectral moments, monotonicity numerators, and endpoint slopes.

It verifies the algebraic theorem *after* the four block laws are supplied.
It does not reconstruct those blocks from the Spin(8) generator matrices and
does not prove the global orbit-normal-form proposition.  See TRUST_BOUNDARY.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from math import comb
from pathlib import Path

Q = Fraction


def _fraction(value: int | Fraction) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def _qtext(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


@dataclass(frozen=True)
class Poly1:
    """Sparse exact polynomial in one variable, with ascending degrees."""

    terms: tuple[tuple[int, Fraction], ...]

    @staticmethod
    def from_dict(terms: dict[int, int | Fraction]) -> Poly1:
        cleaned = tuple(
            sorted(
                (degree, _fraction(value)) for degree, value in terms.items() if value
            )
        )
        return Poly1(cleaned)

    @staticmethod
    def constant(value: int | Fraction) -> Poly1:
        return Poly1.from_dict({0: value})

    @staticmethod
    def variable() -> Poly1:
        return Poly1.from_dict({1: 1})

    def as_dict(self) -> dict[int, Fraction]:
        return dict(self.terms)

    def __add__(self, other: int | Fraction | Poly1) -> Poly1:
        right = other if isinstance(other, Poly1) else Poly1.constant(other)
        result = self.as_dict()
        for degree, value in right.terms:
            result[degree] = result.get(degree, Q(0)) + value
        return Poly1.from_dict(result)

    __radd__ = __add__

    def __neg__(self) -> Poly1:
        return Poly1.from_dict({degree: -value for degree, value in self.terms})

    def __sub__(self, other: int | Fraction | Poly1) -> Poly1:
        right = other if isinstance(other, Poly1) else Poly1.constant(other)
        return self + (-right)

    def __rsub__(self, other: int | Fraction | Poly1) -> Poly1:
        return Poly1.constant(other) - self

    def __mul__(self, other: int | Fraction | Poly1) -> Poly1:
        right = other if isinstance(other, Poly1) else Poly1.constant(other)
        result: dict[int, Fraction] = {}
        for left_degree, left_value in self.terms:
            for right_degree, right_value in right.terms:
                degree = left_degree + right_degree
                result[degree] = result.get(degree, Q(0)) + left_value * right_value
        return Poly1.from_dict(result)

    __rmul__ = __mul__

    def __truediv__(self, scalar: int | Fraction) -> Poly1:
        divisor = _fraction(scalar)
        if divisor == 0:
            raise ZeroDivisionError
        return Poly1.from_dict(
            {degree: value / divisor for degree, value in self.terms}
        )

    def __pow__(self, exponent: int) -> Poly1:
        if exponent < 0:
            raise ValueError("polynomial exponent must be nonnegative")
        result = Poly1.constant(1)
        base = self
        power = exponent
        while power:
            if power & 1:
                result = result * base
            base = base * base
            power >>= 1
        return result

    def derivative(self) -> Poly1:
        return Poly1.from_dict(
            {degree - 1: degree * value for degree, value in self.terms if degree}
        )

    def evaluate(self, value: int | Fraction) -> Fraction:
        point = _fraction(value)
        return sum(coefficient * point**degree for degree, coefficient in self.terms)

    def coefficient_list(self) -> list[str]:
        maximum = max((degree for degree, _ in self.terms), default=0)
        values = self.as_dict()
        return [_qtext(values.get(degree, Q(0))) for degree in range(maximum + 1)]


@dataclass(frozen=True)
class Poly2:
    """Sparse exact polynomial in ``(c, lambda)``."""

    terms: tuple[tuple[tuple[int, int], Fraction], ...]

    @staticmethod
    def from_dict(
        terms: dict[tuple[int, int], int | Fraction],
    ) -> Poly2:
        cleaned = tuple(
            sorted(
                (degree, _fraction(value)) for degree, value in terms.items() if value
            )
        )
        return Poly2(cleaned)

    @staticmethod
    def constant(value: int | Fraction) -> Poly2:
        return Poly2.from_dict({(0, 0): value})

    @staticmethod
    def c_variable() -> Poly2:
        return Poly2.from_dict({(1, 0): 1})

    @staticmethod
    def lambda_variable() -> Poly2:
        return Poly2.from_dict({(0, 1): 1})

    def as_dict(self) -> dict[tuple[int, int], Fraction]:
        return dict(self.terms)

    def __add__(self, other: int | Fraction | Poly2) -> Poly2:
        right = other if isinstance(other, Poly2) else Poly2.constant(other)
        result = self.as_dict()
        for degree, value in right.terms:
            result[degree] = result.get(degree, Q(0)) + value
        return Poly2.from_dict(result)

    __radd__ = __add__

    def __neg__(self) -> Poly2:
        return Poly2.from_dict({degree: -value for degree, value in self.terms})

    def __sub__(self, other: int | Fraction | Poly2) -> Poly2:
        right = other if isinstance(other, Poly2) else Poly2.constant(other)
        return self + (-right)

    def __rsub__(self, other: int | Fraction | Poly2) -> Poly2:
        return Poly2.constant(other) - self

    def __mul__(self, other: int | Fraction | Poly2) -> Poly2:
        right = other if isinstance(other, Poly2) else Poly2.constant(other)
        result: dict[tuple[int, int], Fraction] = {}
        for (c_left, l_left), left_value in self.terms:
            for (c_right, l_right), right_value in right.terms:
                degree = (c_left + c_right, l_left + l_right)
                result[degree] = result.get(degree, Q(0)) + left_value * right_value
        return Poly2.from_dict(result)

    __rmul__ = __mul__

    def __truediv__(self, scalar: int | Fraction) -> Poly2:
        divisor = _fraction(scalar)
        if divisor == 0:
            raise ZeroDivisionError
        return Poly2.from_dict(
            {degree: value / divisor for degree, value in self.terms}
        )

    def __pow__(self, exponent: int) -> Poly2:
        if exponent < 0:
            raise ValueError("polynomial exponent must be nonnegative")
        result = Poly2.constant(1)
        base = self
        power = exponent
        while power:
            if power & 1:
                result = result * base
            base = base * base
            power >>= 1
        return result

    def derivative(self, axis: int) -> Poly2:
        if axis not in (0, 1):
            raise ValueError("axis must be 0 for c or 1 for lambda")
        result: dict[tuple[int, int], Fraction] = {}
        for degrees, value in self.terms:
            power = degrees[axis]
            if not power:
                continue
            reduced = list(degrees)
            reduced[axis] -= 1
            result[tuple(reduced)] = power * value
        return Poly2.from_dict(result)

    def evaluate(self, cayley: int | Fraction, eigenvalue: int | Fraction) -> Fraction:
        c_value = _fraction(cayley)
        l_value = _fraction(eigenvalue)
        return sum(
            coefficient * c_value**c_degree * l_value**l_degree
            for (c_degree, l_degree), coefficient in self.terms
        )

    def lambda_coefficient_as_z(self, degree: int) -> Poly1:
        c_terms: dict[int, Fraction] = {}
        for (c_degree, l_degree), value in self.terms:
            if l_degree != degree:
                continue
            if c_degree % 2:
                raise AssertionError(
                    f"lambda coefficient {degree} contains odd c degree {c_degree}"
                )
            z_degree = c_degree // 2
            c_terms[z_degree] = c_terms.get(z_degree, Q(0)) + value
        return Poly1.from_dict(c_terms)

    def at_c(self, cayley: int | Fraction) -> Poly1:
        c_value = _fraction(cayley)
        terms: dict[int, Fraction] = {}
        for (c_degree, l_degree), value in self.terms:
            terms[l_degree] = terms.get(l_degree, Q(0)) + value * c_value**c_degree
        return Poly1.from_dict(terms)


def _ratio_equal(
    left_numerator: Poly1,
    left_denominator: Poly1,
    right_numerator: Poly1,
    right_denominator: Poly1,
) -> bool:
    return left_numerator * right_denominator == right_numerator * left_denominator


def _ratio_derivative(numerator: Poly1, denominator: Poly1) -> tuple[Poly1, Poly1]:
    return (
        numerator.derivative() * denominator - numerator * denominator.derivative(),
        denominator**2,
    )


def _bernstein_coefficients(polynomial: Poly1) -> list[Fraction]:
    coefficients = polynomial.as_dict()
    degree = max(coefficients, default=0)
    return [
        sum(
            Q(comb(row, column), comb(degree, column)) * coefficients.get(column, Q(0))
            for column in range(row + 1)
        )
        for row in range(degree + 1)
    ]


def _block_polynomials() -> tuple[Poly2, Poly2, Poly2, Poly2]:
    c = Poly2.c_variable()
    l = Poly2.lambda_variable()

    chi0 = (
        -Q(1, 4)
        * (l - 1) ** 2
        * (2 * c * l - c - 2 * l**3 + 8 * l**2 - 6 * l + 1)
        * (2 * c * l - c + 2 * l**3 - 8 * l**2 + 6 * l - 1)
    )
    chi1 = (
        Q(1, 16)
        * (c - 2 * l**2 + 4 * l - 1)
        * (c - 2 * l**2 + 6 * l - 3)
        * (c + 2 * l**2 - 6 * l + 3)
        * (c + 2 * l**2 - 4 * l + 1)
    )
    chi3 = (l - 1) ** 2 * (l**2 - 3 * l + 1)
    return chi0, chi1, chi1, chi3


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def build_certificate() -> dict[str, object]:
    blocks = _block_polynomials()
    characteristic = Poly2.constant(1)
    for block in blocks:
        characteristic *= block

    coefficients = {
        degree: characteristic.lambda_coefficient_as_z(degree) for degree in range(29)
    }
    if coefficients[28] != Poly1.constant(1):
        raise AssertionError("the full characteristic polynomial is not monic")

    z = Poly1.variable()
    trace = -coefficients[27]
    trace_square = trace**2 - 2 * coefficients[26]
    determinant = coefficients[0]
    expected_determinant = (1 - z) ** 3 * (9 - z) ** 2 / 1024
    if determinant != expected_determinant:
        raise AssertionError("determinant factorization failed")
    if trace != Poly1.constant(35) or trace_square != Poly1.constant(67):
        raise AssertionError("direct spectral moments failed")

    p0, p1, p2 = coefficients[0], coefficients[1], coefficients[2]
    inverse_numerator, inverse_denominator = -p1, p0
    inverse_expected_numerator = 11 * z**2 - 206 * z + 387
    inverse_expected_denominator = (1 - z) * (9 - z)
    if not _ratio_equal(
        inverse_numerator,
        inverse_denominator,
        inverse_expected_numerator,
        inverse_expected_denominator,
    ):
        raise AssertionError("trace(I^-1) identity failed")

    inverse_square_numerator = p1**2 - 2 * p2 * p0
    inverse_square_denominator = p0**2
    inverse_square_expected_numerator = (
        19 * z**4 - 76 * z**3 + 786 * z**2 + 2676 * z + 8883
    )
    inverse_square_expected_denominator = (1 - z) ** 2 * (9 - z) ** 2
    if not _ratio_equal(
        inverse_square_numerator,
        inverse_square_denominator,
        inverse_square_expected_numerator,
        inverse_square_expected_denominator,
    ):
        raise AssertionError("trace(I^-2) identity failed")

    determinant_derivative = determinant.derivative()
    determinant_expected_derivative = -((1 - z) ** 2) * (9 - z) * (29 - 5 * z) / 1024
    if determinant_derivative != determinant_expected_derivative:
        raise AssertionError("determinant derivative factorization failed")

    inverse_derivative = _ratio_derivative(
        inverse_expected_numerator, inverse_expected_denominator
    )
    inverse_derivative_expected = (
        96 * (z**2 - 6 * z + 21),
        (1 - z) ** 2 * (9 - z) ** 2,
    )
    if not _ratio_equal(*inverse_derivative, *inverse_derivative_expected):
        raise AssertionError("trace(I^-1) derivative failed")

    inverse_square_derivative = _ratio_derivative(
        inverse_square_expected_numerator, inverse_square_expected_denominator
    )
    inverse_square_sign = 12609 + 336 * z - 630 * z**2 - 8 * z**3 - 19 * z**4
    inverse_square_derivative_expected = (
        16 * inverse_square_sign,
        (1 - z) ** 3 * (9 - z) ** 3,
    )
    if not _ratio_equal(
        *inverse_square_derivative, *inverse_square_derivative_expected
    ):
        raise AssertionError("trace(I^-2) derivative failed")
    bernstein = _bernstein_coefficients(inverse_square_sign)
    if bernstein != [Q(12609), Q(12693), Q(12672), Q(12544), Q(12288)]:
        raise AssertionError("inverse-square Bernstein certificate failed")
    if not all(value > 0 for value in bernstein):
        raise AssertionError("inverse-square derivative sign is not certified")

    l = Poly1.variable()
    expected_endpoint_blocks = [
        l * (l - 1) ** 3 * (l**2 - 4 * l + 2) * (l**2 - 3 * l + 1),
        l * (l - 2) ** 2 * (l - 1) ** 3 * (l**2 - 3 * l + 1),
        l * (l - 2) ** 2 * (l - 1) ** 3 * (l**2 - 3 * l + 1),
        (l - 1) ** 2 * (l**2 - 3 * l + 1),
    ]
    endpoint_blocks = [block.at_c(1) for block in blocks]
    if endpoint_blocks != expected_endpoint_blocks:
        raise AssertionError("endpoint block factorization failed")

    slopes: list[Fraction] = []
    for block in blocks[:3]:
        partial_c = block.derivative(0).evaluate(1, 0)
        partial_l = block.derivative(1).evaluate(1, 0)
        if partial_l == 0:
            raise AssertionError("endpoint zero root is not simple")
        slopes.append(partial_c / (2 * partial_l))
    if slopes != [Q(1, 8)] * 3:
        raise AssertionError("endpoint slope identity failed")

    coefficient_map = [coefficients[degree].coefficient_list() for degree in range(29)]
    coefficient_hash = hashlib.sha256(
        _canonical_json(coefficient_map).encode("ascii")
    ).hexdigest()

    report: dict[str, object] = {
        "schema": "spin8-cayley-referee-certificate-v1",
        "arithmetic": "Python standard library fractions.Fraction",
        "supplied_input": "four displayed block characteristic polynomials",
        "full_characteristic_coefficients": coefficient_map,
        "full_characteristic_coefficients_sha256": coefficient_hash,
        "block_determinants_in_z": [
            block.lambda_coefficient_as_z(0).coefficient_list() for block in blocks
        ],
        "derived_claims": {
            "determinant_coefficients": determinant.coefficient_list(),
            "trace": "35",
            "trace_square": "67",
            "trace_inverse_at_z0": _qtext(
                inverse_expected_numerator.evaluate(0)
                / inverse_expected_denominator.evaluate(0)
            ),
            "trace_inverse_square_at_z0": _qtext(
                inverse_square_expected_numerator.evaluate(0)
                / inverse_square_expected_denominator.evaluate(0)
            ),
            "inverse_square_sign_bernstein_coefficients": [
                _qtext(value) for value in bernstein
            ],
            "endpoint_small_eigenvalue_slopes_in_1_minus_z": [
                _qtext(value) for value in slopes
            ],
            "endpoint_rank": 25,
            "endpoint_maximum_eigenvalue": "2 + sqrt(2)",
        },
        "trust_boundary": {
            "recomputed": [
                "full characteristic polynomial from the four block laws",
                "determinant and direct/inverse spectral moments",
                "monotonicity derivative identities and exact Bernstein sign",
                "endpoint block factorizations and three first-order slopes",
            ],
            "not_recomputed": [
                "Spin(8) generator construction",
                "28x28 information matrix to four-block reduction",
                "global balanced-orbit normal-form proposition",
            ],
        },
        "passed": True,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit",
        action="store_true",
        help="print the freshly reconstructed exact certificate as canonical JSON",
    )
    parser.add_argument(
        "--certificate",
        type=Path,
        default=Path(__file__).with_name("artifacts") / "certificate.json",
        help="stored certificate to compare against",
    )
    args = parser.parse_args()

    report = build_certificate()
    if args.emit:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    stored = json.loads(args.certificate.read_text(encoding="utf-8"))
    if stored != report:
        raise SystemExit("FAIL: stored certificate differs from exact reconstruction")
    print("PASS: independent exact Cayley-spectrum certificate reproduced")
    print(
        f"coefficient map SHA-256: {report['full_characteristic_coefficients_sha256']}"
    )


if __name__ == "__main__":
    main()
