"""Four exact shape-uniform cusp charts around the Spin(9) equality fiber."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import sympy as sp

from spin9_candidate_explicit_collar import (
    _gap_pairs,
    _radial_bernstein_lower,
    _remainder_upper,
    _sqrt_bounds,
)
from spin9_candidate_normal_form import ISOLATING_CELLS, X, _candidate_fiber
from spin9_v1_candidate_line import _quadratic_field_parts
from spin9_v1_v5_reconstruction import ROOT

SQRT_241 = sp.sqrt(241)
DEFAULT_OUTPUT = ROOT / "artifacts" / "spin9_candidate_cusp_charts_20260821.json"
MACRO_INTERVALS = (
    (Fraction(-1, 2), Fraction(-2, 5)),
    (Fraction(-1, 50), Fraction(-1, 100)),
    (Fraction(1), Fraction(11, 10)),
    (Fraction(31), Fraction(33)),
)


def _sha256_fraction(value: Fraction) -> str:
    digest = hashlib.sha256()
    for integer in (value.numerator, value.denominator):
        magnitude = abs(integer)
        encoded = magnitude.to_bytes(max(1, (magnitude.bit_length() + 7) // 8), "big")
        digest.update(b"-" if integer < 0 else b"+")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _pair_coefficients(poly: sp.Poly) -> list[tuple[Fraction, Fraction]]:
    degree = poly.degree()
    rows = []
    for power in range(degree + 1):
        rational, radical = _quadratic_field_parts(poly.nth(power))
        rows.append((rational, radical))
    return rows


def _bernstein_lower(
    coefficients: list[tuple[Fraction, Fraction]],
    lower: Fraction,
    upper: Fraction,
) -> Fraction:
    degree = len(coefficients) - 1
    width = upper - lower
    sqrt_lower, sqrt_upper = _sqrt_bounds()
    power_u = []
    for index in range(degree + 1):
        power_u.append(
            tuple(
                sum(
                    coefficients[source][part]
                    * math.comb(source, index)
                    * lower ** (source - index)
                    * width**index
                    for source in range(index, degree + 1)
                )
                for part in range(2)
            )
        )
    controls = []
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
            rational + radical * (sqrt_lower if radical >= 0 else sqrt_upper)
        )
    return min(controls)


def _absolute_derivative_bound(
    coefficients: list[tuple[Fraction, Fraction]], maximum_abs_x: Fraction
) -> Fraction:
    return sum(
        power
        * (abs(rational) + 16 * abs(radical))
        * maximum_abs_x ** (power - 1)
        for power, (rational, radical) in enumerate(coefficients)
        if power
    )


def _normal_form_polynomials(
    gap: dict[tuple[int, int, int], tuple[int, int]],
) -> tuple[sp.Poly, sp.Poly, sp.Poly]:
    pure = sp.Poly(
        sum((a + b * SQRT_241) * X**i for (i, p, y), (a, b) in gap.items() if p == y == 0),
        X,
        extension=SQRT_241,
    )
    radial = sp.Poly(
        sum((a + b * SQRT_241) * X**i for (i, p, y), (a, b) in gap.items() if p == 1 and y == 0),
        X,
        extension=SQRT_241,
    )
    fiber = sp.Poly(_candidate_fiber(), X, extension=SQRT_241)
    quotient, remainder = sp.div(pure, fiber**2, domain=sp.QQ.algebraic_field(SQRT_241))
    if not remainder.is_zero:
        raise AssertionError("candidate square factorization failed")
    return fiber, quotient, radial


def certificate() -> dict[str, object]:
    """Certify four macroscopic-x, finite-r cusp neighborhoods."""

    gap = _gap_pairs()
    fiber, quotient, radial = _normal_form_polynomials(gap)
    fiber_derivative = fiber.diff()
    quotient_coefficients = _pair_coefficients(quotient)
    derivative_coefficients = _pair_coefficients(fiber_derivative)
    radial_coefficients = _pair_coefficients(radial)
    rows = []
    for root_index, ((numerator, decimal_power), macro) in enumerate(
        zip(ISOLATING_CELLS, MACRO_INTERVALS, strict=True)
    ):
        cell_lower = Fraction(numerator, 10**decimal_power)
        cell_upper = cell_lower + Fraction(1, 10**decimal_power)
        lower, upper = macro
        if not (lower < cell_lower < cell_upper < upper):
            raise AssertionError("macro interval does not contain its root cell")

        h_lower = _bernstein_lower(quotient_coefficients, lower, upper)
        midpoint = (cell_lower + cell_upper) / 2
        derivative_mid = fiber_derivative.eval(sp.Rational(midpoint.numerator, midpoint.denominator))
        derivative_sign = 1 if float(sp.N(derivative_mid, 30)) > 0 else -1
        signed_derivative = [
            (derivative_sign * a, derivative_sign * b)
            for a, b in derivative_coefficients
        ]
        q_derivative_lower = _bernstein_lower(signed_derivative, lower, upper)
        pure_distance_lower = h_lower * q_derivative_lower**2

        radial_cell_lower = _radial_bernstein_lower(
            gap, cell_lower, cell_upper, *_sqrt_bounds()
        )
        radial_lipschitz = _absolute_derivative_bound(
            radial_coefficients, max(abs(lower), abs(upper))
        )
        cross = 6 * radial_lipschitz
        young_quartic = cross**2 / (2 * pure_distance_lower)
        remainder = _remainder_upper(gap, max(abs(lower), abs(upper)))

        linear_radius = Fraction(3, 8) * radial_cell_lower / remainder
        quadratic_radius = sp.sqrt(
            sp.Rational(
                (Fraction(3, 8) * radial_cell_lower / young_quartic).numerator,
                (Fraction(3, 8) * radial_cell_lower / young_quartic).denominator,
            )
        )
        # A rational dyadic-free radius below both exact thresholds.
        exponent = max(
            1,
            math.ceil(-math.log10(float(linear_radius))),
            math.ceil(-math.log10(float(sp.N(quadratic_radius, 30)))),
        )
        radius = Fraction(1, 10**exponent)
        bracket = (
            Fraction(3, 2) * radial_cell_lower
            - remainder * radius
            - young_quartic * radius**2
        )
        passed = bool(
            h_lower > 0
            and q_derivative_lower > 0
            and radial_cell_lower > 0
            and bracket > 0
        )
        rows.append(
            {
                "root_index": root_index,
                "x_interval": [str(lower), str(upper)],
                "r_upper": str(radius),
                "r_radius_exponent": exponent,
                "z_interval": ["0", "1"],
                "pure_distance_lower_decimal": str(float(pure_distance_lower)),
                "radial_cell_lower_decimal": str(float(radial_cell_lower)),
                "radial_lipschitz_decimal": str(float(radial_lipschitz)),
                "remainder_upper_decimal": str(float(remainder)),
                "final_bracket_decimal": str(float(bracket)),
                "pure_distance_lower_sha256": _sha256_fraction(pure_distance_lower),
                "final_bracket_sha256": _sha256_fraction(bracket),
                "inequality": (
                    "G >= c*d^2+(3/2)*a*r^2-6*L*d*r^2-M*r^3; "
                    "Young gives G >= (c/2)*d^2+r^2*((3/2)*a-M*r-18*L^2*r^2/c)"
                ),
                "passed": passed,
            }
        )

    passed = all(row["passed"] for row in rows)
    return {
        "schema_version": 1,
        "claim_scope": (
            "four exact shape-uniform cusp charts around every algebraic "
            "candidate preimage on the first Spin(9) V1+V5 graph"
        ),
        "method": "Q(x)^2*H(x), exact Bernstein bounds, and Young absorption",
        "charts": rows,
        "all_four_cusp_charts_certified": passed,
        "compact_complement_certified_at_candidate_ratio": False,
        "second_v5_copy_certified": False,
        "unrestricted_quotient_certified": False,
        "passed": passed,
    }


def verify_report(report: dict[str, object]) -> bool:
    rows = report.get("charts", [])
    return bool(
        report.get("passed") is True
        and report.get("all_four_cusp_charts_certified") is True
        and report.get("compact_complement_certified_at_candidate_ratio") is False
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
    arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
