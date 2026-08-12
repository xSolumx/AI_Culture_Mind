"""Exact audit of Gaussian phase divisors behind a Machin identity.

The algebraic phase map used here is rho(z) = z / conjugate(z).  Unlike
z / abs(z), rho is defined over Q and lands in the rational norm-one torus.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from fractions import Fraction
from math import gcd, isqrt
from pathlib import Path

from sympy import factorint

GaussianInteger = tuple[int, int]
GaussianRational = tuple[Fraction, Fraction]


def gaussian_mul(left: GaussianInteger, right: GaussianInteger) -> GaussianInteger:
    a, b = left
    c, d = right
    return a * c - b * d, a * d + b * c


def gaussian_conjugate(value: GaussianInteger) -> GaussianInteger:
    return value[0], -value[1]


def gaussian_norm(value: GaussianInteger) -> int:
    return value[0] * value[0] + value[1] * value[1]


def gaussian_pow(value: GaussianInteger, exponent: int) -> GaussianInteger:
    if exponent < 0:
        raise ValueError("Gaussian-integer powers require a nonnegative exponent")
    result = (1, 0)
    base = value
    power = exponent
    while power:
        if power & 1:
            result = gaussian_mul(result, base)
        base = gaussian_mul(base, base)
        power //= 2
    return result


def rational_gaussian_mul(
    left: GaussianRational, right: GaussianRational
) -> GaussianRational:
    a, b = left
    c, d = right
    return a * c - b * d, a * d + b * c


def rational_gaussian_inverse(value: GaussianRational) -> GaussianRational:
    a, b = value
    norm = a * a + b * b
    if norm == 0:
        raise ZeroDivisionError("zero has no Gaussian-rational inverse")
    return a / norm, -b / norm


def rational_gaussian_pow(value: GaussianRational, exponent: int) -> GaussianRational:
    if exponent < 0:
        return rational_gaussian_pow(rational_gaussian_inverse(value), -exponent)
    result = (Fraction(1), Fraction(0))
    base = value
    power = exponent
    while power:
        if power & 1:
            result = rational_gaussian_mul(result, base)
        base = rational_gaussian_mul(base, base)
        power //= 2
    return result


def algebraic_phase(value: GaussianRational) -> GaussianRational:
    """Return rho(z)=z/conjugate(z), an exact rational point of the torus."""

    a, b = value
    return rational_gaussian_mul(value, rational_gaussian_inverse((a, -b)))


def rational_circle_point(n: int) -> tuple[Fraction, Fraction]:
    """Return rho(n+i), the rational stereographic point on x^2+y^2=1."""

    denominator = n * n + 1
    return Fraction(n * n - 1, denominator), Fraction(2 * n, denominator)


def circle_mul(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    return rational_gaussian_mul(left, right)


def machin_product(terms: Iterable[tuple[int, int]]) -> GaussianRational:
    result = (Fraction(1), Fraction(0))
    for denominator, coefficient in terms:
        factor = (Fraction(denominator), Fraction(1))
        result = rational_gaussian_mul(
            result, rational_gaussian_pow(factor, coefficient)
        )
    return result


def _divide_gaussian_exact(
    numerator: GaussianInteger, denominator: GaussianInteger
) -> GaussianInteger | None:
    norm = gaussian_norm(denominator)
    product = gaussian_mul(numerator, gaussian_conjugate(denominator))
    if product[0] % norm or product[1] % norm:
        return None
    return product[0] // norm, product[1] // norm


def _split_prime_generator(prime: int) -> GaussianInteger:
    for imaginary in range(1, isqrt(prime) + 1):
        real_square = prime - imaginary * imaginary
        real = isqrt(real_square)
        if real > 0 and real * real == real_square:
            return max(real, imaginary), min(real, imaginary)
    raise ValueError(f"failed to split prime {prime} as a sum of two squares")


def factor_gaussian_integer(value: GaussianInteger) -> dict[str, object]:
    """Factor a nonzero Gaussian integer into deterministic prime associates."""

    if value == (0, 0):
        raise ValueError("zero has no Gaussian prime factorization")

    remainder = value
    factors: dict[str, int] = {}
    for prime in sorted(factorint(gaussian_norm(value))):
        candidates: list[tuple[str, GaussianInteger]]
        if prime == 2:
            candidates = [("2:ramified", (1, 1))]
        elif prime % 4 == 1:
            generator = _split_prime_generator(prime)
            candidates = [
                (f"{prime}:+", generator),
                (f"{prime}:-", gaussian_conjugate(generator)),
            ]
        else:
            candidates = [(f"{prime}:inert", (prime, 0))]

        for label, candidate in candidates:
            valuation = 0
            while True:
                quotient = _divide_gaussian_exact(remainder, candidate)
                if quotient is None:
                    break
                remainder = quotient
                valuation += 1
            if valuation:
                factors[label] = valuation

    unit_powers = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
    if remainder not in unit_powers:
        raise AssertionError(f"Gaussian factorization left nonunit {remainder}")
    return {"unit_i_power_mod_4": unit_powers[remainder], "factors": factors}


def free_phase_divisor(terms: Iterable[tuple[int, int]]) -> dict[str, int]:
    """Return split-prime valuation differences for a formal Machin product."""

    divisor: dict[str, int] = {}
    for denominator, coefficient in terms:
        factorization = factor_gaussian_integer((denominator, 1))["factors"]
        assert isinstance(factorization, dict)
        split_primes = {
            label.split(":", 1)[0]
            for label in factorization
            if label.endswith((":+", ":-"))
        }
        for prime in split_primes:
            difference = factorization.get(f"{prime}:+", 0) - factorization.get(
                f"{prime}:-", 0
            )
            divisor[prime] = divisor.get(prime, 0) + coefficient * difference
    return {prime: value for prime, value in sorted(divisor.items()) if value}


def alferov_tangent_step(
    numerator: int, denominator: int, q: int | None = None
) -> dict[str, object]:
    """Perform the exact tangent subtraction used by the proposed reduction."""

    if numerator <= 0 or denominator <= 0:
        raise ValueError("this audited nearest-cotangent control requires a,b > 0")
    if q is None:
        q = (2 * denominator + numerator) // (2 * numerator)
    if q <= 0:
        raise ValueError("the arctangent denominator q must be positive")
    raw_numerator = numerator * q - denominator
    raw_denominator = denominator * q + numerator
    common = gcd(abs(raw_numerator), raw_denominator)
    reduced = Fraction(raw_numerator, raw_denominator)
    expected = (Fraction(numerator, denominator) - Fraction(1, q)) / (
        1 + Fraction(numerator, denominator * q)
    )
    return {
        "input_tangent": str(Fraction(numerator, denominator)),
        "nearest_cotangent_integer": q,
        "raw_remainder": [raw_numerator, raw_denominator],
        "reduced_remainder": str(reduced),
        "reduction_gcd": common,
        "tangent_subtraction_identity": reduced == expected,
        "nearest_integer_numerator_bound": 2 * abs(raw_numerator) <= numerator,
    }


def _complex_strings(value: GaussianRational | GaussianInteger) -> list[str]:
    return [str(value[0]), str(value[1])]


def run_diagnostics() -> dict[str, object]:
    terms = [(239, 1), (7, 2), (4, 2), (268, 2)]
    integer_product = (1, 0)
    for denominator, coefficient in terms:
        integer_product = gaussian_mul(
            integer_product, gaussian_pow((denominator, 1), coefficient)
        )

    target = gaussian_pow((1, 1), 1)
    target_rational = (Fraction(target[0]), Fraction(target[1]))
    product_rational = (
        Fraction(integer_product[0]),
        Fraction(integer_product[1]),
    )
    quotient = rational_gaussian_mul(
        product_rational, rational_gaussian_inverse(target_rational)
    )
    phase_product = algebraic_phase(product_rational)
    phase_target = algebraic_phase(target_rational)
    divisor = free_phase_divisor(terms)

    circle_controls = {}
    for denominator in (4, 7, 239, 268):
        point = rational_circle_point(denominator)
        direct = algebraic_phase((Fraction(denominator), Fraction(1)))
        circle_controls[str(denominator)] = {
            "point": _complex_strings(point),
            "equals_z_over_conjugate_z": point == direct,
            "norm_is_one": point[0] * point[0] + point[1] * point[1] == 1,
            "double_angle_tangent": str(point[1] / point[0]),
            "double_angle_formula": str(
                Fraction(2 * denominator, denominator * denominator - 1)
            ),
            "double_angle_identity": point[1] / point[0]
            == Fraction(2 * denominator, denominator * denominator - 1),
        }

    factorizations = {
        str(denominator): factor_gaussian_integer((denominator, 1))
        for denominator, _ in terms
    }
    exact_gates = {
        "four_term_integer_identity": integer_product
        == (10_317_661_250, 10_317_661_250),
        "quotient_by_one_plus_i_is_positive_rational": quotient[1] == 0
        and quotient[0] == 10_317_661_250,
        "algebraic_phase_matches_target": phase_product == phase_target,
        "all_free_split_prime_phase_valuations_cancel": not divisor,
        "rational_circle_controls_pass": all(
            control["equals_z_over_conjugate_z"]
            and control["norm_is_one"]
            and control["double_angle_identity"]
            for control in circle_controls.values()
        ),
    }

    alferov_control = alferov_tangent_step(5, 12)
    exact_gates["alferov_tangent_identity"] = bool(
        alferov_control["tangent_subtraction_identity"]
    )
    exact_gates["nearest_integer_numerator_bound"] = bool(
        alferov_control["nearest_integer_numerator_bound"]
    )

    return {
        "schema_version": 1,
        "claim_scope": (
            "exact Gaussian norm-one-torus and phase-divisor reformulation of "
            "one four-term Machin identity; no global Alferov termination, "
            "Lehmer-height, optimality, or novelty theorem"
        ),
        "phase_map": "rho(z)=z/conjugate(z)",
        "machin_terms": [
            {"denominator": denominator, "coefficient": coefficient}
            for denominator, coefficient in terms
        ],
        "integer_product": _complex_strings(integer_product),
        "target": _complex_strings(target),
        "positive_rational_quotient": _complex_strings(quotient),
        "free_phase_divisor": divisor,
        "input_factorizations": factorizations,
        "circle_controls": circle_controls,
        "alferov_control": alferov_control,
        "exact_gates": exact_gates,
        "corrections_to_supplied_reformulation": [
            "z/abs(z) is generally not a Q-rational point; z/conjugate(z) is",
            "(n+i)^2 is not itself a multiplicative Galois minus eigenvector",
            "the normalized spinor may require sqrt(n^2+1), while its square is rational",
            "nearest-cotangent subtraction gives an exact Euclidean numerator bound, not a proved Lehmer-height descent",
        ],
        "passed": all(exact_gates.values()),
    }


def write_report(path: Path) -> dict[str, object]:
    report = run_diagnostics()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_diagnostics()
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
