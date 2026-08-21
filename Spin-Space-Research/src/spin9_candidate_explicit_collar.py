"""Explicit exact mixed collars around the four Spin(9) candidate preimages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import sympy as sp
from spin9_candidate_normal_form import ISOLATING_CELLS
from spin9_candidate_normal_form import certificate as normal_form
from spin9_v1_candidate_line import _quadratic_field_parts
from spin9_v1_v5_gap import (
    DEFAULT_COEFFICIENT_ARTIFACT,
    _multiply_sparse,
    _shape_power_coefficients,
)
from spin9_v1_v5_reconstruction import COUPLED_MONOMIALS, ROOT, load_coefficients

DEFAULT_OUTPUT = ROOT / "artifacts" / "spin9_candidate_explicit_collar_20260821.json"
RADIUS_EXPONENTS = (47, 52, 34, 6)
SQRT_BOUND_DIGITS = 180


def _sha256_fraction(value: Fraction) -> str:
    digest = hashlib.sha256()
    for integer in (value.numerator, value.denominator):
        magnitude = abs(integer)
        encoded = magnitude.to_bytes(max(1, (magnitude.bit_length() + 7) // 8), "big")
        digest.update(b"-" if integer < 0 else b"+")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _candidate_parts() -> tuple[int, int, int]:
    q = sp.sqrt(241)
    ratio = sp.expand(
        (-41 + q) ** 10
        * (-5 + q) ** 3
        * (q + 31) ** 5
        / sp.Integer(27_916_588_318_337_525_511_880_704)
    )
    rational, radical = _quadratic_field_parts(ratio)
    denominator = math.lcm(rational.denominator, radical.denominator)
    return (
        rational.numerator * (denominator // rational.denominator),
        radical.numerator * (denominator // radical.denominator),
        denominator,
    )


def _gap_pairs() -> dict[tuple[int, int, int], tuple[int, int]]:
    rational, radical, denominator = _candidate_parts()
    delta = {
        (0, 0, 0): 1,
        (0, 1, 0): 8,
        (0, 2, 0): 20,
        (0, 3, 0): 16,
        (0, 0, 2): -16,
        (2, 0, 0): 6,
        (2, 1, 0): 24,
        (2, 2, 0): 8,
        (4, 0, 0): 12,
        (4, 1, 0): 16,
        (6, 0, 0): 8,
        (1, 0, 1): 24,
        (1, 1, 1): 80,
        (3, 0, 1): 80,
    }
    delta_power = {(0, 0, 0): 1}
    for _ in range(14):
        delta_power = _multiply_sparse(delta_power, delta)
    result = {
        monomial: (rational * value, radical * value)
        for monomial, value in delta_power.items()
    }
    coefficients = load_coefficients(DEFAULT_COEFFICIENT_ARTIFACT)
    for monomial, coefficient in zip(
        COUPLED_MONOMIALS, coefficients, strict=True
    ):
        left, right = result.get(monomial, (0, 0))
        result[monomial] = (
            left - denominator * coefficient.numerator,
            right,
        )
    return {key: value for key, value in result.items() if value != (0, 0)}


def _sqrt_bounds() -> tuple[Fraction, Fraction]:
    scale = 10**SQRT_BOUND_DIGITS
    lower_numerator = math.isqrt(241 * scale * scale)
    return Fraction(lower_numerator, scale), Fraction(lower_numerator + 1, scale)


def _lower_pair(
    rational: Fraction,
    radical: Fraction,
    sqrt_lower: Fraction,
    sqrt_upper: Fraction,
) -> Fraction:
    return rational + radical * (sqrt_lower if radical >= 0 else sqrt_upper)


def _radial_bernstein_lower(
    gap: dict[tuple[int, int, int], tuple[int, int]],
    lower: Fraction,
    upper: Fraction,
    sqrt_lower: Fraction,
    sqrt_upper: Fraction,
) -> Fraction:
    degree = max(x_power for x_power, p_power, y_power in gap if p_power == 1 and y_power == 0)
    power_x = [gap.get((index, 1, 0), (0, 0)) for index in range(degree + 1)]
    width = upper - lower
    power_u: list[tuple[Fraction, Fraction]] = []
    for index in range(degree + 1):
        rational = sum(
            Fraction(power_x[source][0])
            * math.comb(source, index)
            * lower ** (source - index)
            * width**index
            for source in range(index, degree + 1)
        )
        radical = sum(
            Fraction(power_x[source][1])
            * math.comb(source, index)
            * lower ** (source - index)
            * width**index
            for source in range(index, degree + 1)
        )
        power_u.append((rational, radical))
    controls: list[Fraction] = []
    for index in range(degree + 1):
        rational = sum(
            power_u[source][0]
            * Fraction(math.comb(index, source), math.comb(degree, source))
            for source in range(index + 1)
        )
        radical = sum(
            power_u[source][1]
            * Fraction(math.comb(index, source), math.comb(degree, source))
            for source in range(index + 1)
        )
        controls.append(
            _lower_pair(rational, radical, sqrt_lower, sqrt_upper)
        )
    return min(controls)


def _remainder_upper(
    gap: dict[tuple[int, int, int], tuple[int, int]],
    maximum_abs_x: Fraction,
) -> Fraction:
    bound = Fraction(0)
    for (x_power, p_power, y_power), (rational, radical) in gap.items():
        radial_degree = 2 * p_power + 3 * y_power
        if radial_degree < 3:
            continue
        shape_l1 = sum(
            abs(value)
            for value in _shape_power_coefficients(p_power, y_power).values()
        )
        coefficient_bound = abs(rational) + 16 * abs(radical)
        bound += (
            coefficient_bound
            * maximum_abs_x**x_power
            * Fraction(shape_l1, 2 ** (p_power + y_power))
        )
    return bound


def certificate() -> dict[str, object]:
    """Certify four explicit semialgebraic candidate collars."""

    gap = _gap_pairs()
    sqrt_lower, sqrt_upper = _sqrt_bounds()
    rows = []
    for (numerator, decimal_power), radius_exponent in zip(
        ISOLATING_CELLS, RADIUS_EXPONENTS, strict=True
    ):
        lower = Fraction(numerator, 10**decimal_power)
        upper = lower + Fraction(1, 10**decimal_power)
        radial_lower = _radial_bernstein_lower(
            gap, lower, upper, sqrt_lower, sqrt_upper
        )
        remainder_upper = _remainder_upper(gap, max(abs(lower), abs(upper)))
        radius = Fraction(1, 10**radius_exponent)
        margin = Fraction(3, 4) * radial_lower - radius * remainder_upper
        rows.append(
            {
                "x_lower": str(lower),
                "x_upper": str(upper),
                "r_upper": str(radius),
                "z_interval": ["0", "1"],
                "radial_bernstein_lower_sha256": _sha256_fraction(radial_lower),
                "remainder_l1_upper_sha256": _sha256_fraction(remainder_upper),
                "radial_lower_decimal": str(float(radial_lower)),
                "remainder_upper_decimal": str(float(remainder_upper)),
                "certified_margin_decimal": str(float(margin)),
                "domination_identity": "K >= (3/2)A_min-r*M >= (3/4)A_min > 0",
                "passed": bool(radial_lower > 0 and margin > 0 and radius <= 1),
            }
        )
    normal = normal_form()
    passed = bool(
        normal["passed"]
        and all(row["passed"] for row in rows)
        and sqrt_lower * sqrt_lower < 241 < sqrt_upper * sqrt_upper
    )
    return {
        "schema_version": 1,
        "claim_scope": (
            "four explicit exact finite-radius collars for the reconstructed "
            "Spin(9) candidate gap on the first V1+V5 graph slice"
        ),
        "coordinate_substitution": {
            "p": "(3+9*z^2)*r^2/2",
            "y": "(9*z^2-1)*r^3/2",
            "domain": "r>=0 and 0<=z<=1",
        },
        "normal_form_chained": normal["passed"],
        "sqrt_241_rational_bound_digits": SQRT_BOUND_DIGITS,
        "sqrt_241_bounds_verified": sqrt_lower * sqrt_lower < 241 < sqrt_upper * sqrt_upper,
        "collars": rows,
        "all_four_explicit_collars_certified": all(row["passed"] for row in rows),
        "candidate_maximality_on_collar_union_certified": passed,
        "compact_complement_certified_at_candidate_ratio": False,
        "second_v5_copy_certified": False,
        "unrestricted_quotient_certified": False,
        "passed": passed,
    }


def verify_report(report: dict[str, object]) -> bool:
    rows = report.get("collars", [])
    return bool(
        report.get("passed") is True
        and report.get("candidate_maximality_on_collar_union_certified") is True
        and report.get("compact_complement_certified_at_candidate_ratio") is False
        and report.get("second_v5_copy_certified") is False
        and report.get("unrestricted_quotient_certified") is False
        and len(rows) == 4
        and all(row.get("passed") is True for row in rows)
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    report = certificate()
    encoded = json.dumps(report, indent=2, sort_keys=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
