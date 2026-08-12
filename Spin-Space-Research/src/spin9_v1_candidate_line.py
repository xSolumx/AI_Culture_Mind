"""Exact global candidate theorem on the pure Spin(9) ``V1`` graph line.

The reconstructed coupled normal-slice ratio restricts at ``p=y=0`` to a
rational function of the graph coordinate ``x``.  This module proves that the
line determinant is a fourfold rational pullback of the already certified
symmetric equiangular determinant curve.  Consequently the algebraic candidate is the
global maximum on the complete real graph line, with exactly four finite graph
coordinates mapping to the same maximizing symmetric-curve parameter.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import sympy as sp

from spin9_v1_v5_reconstruction import ROOT

DEFAULT_OUTPUT = ROOT / "artifacts" / "spin9_v1_candidate_line_20260812.json"


def _quadratic_field_parts(expression: sp.Expr) -> tuple[Fraction, Fraction]:
    """Return rational ``a,b`` for ``expression=a+b*sqrt(241)``."""

    q = sp.sqrt(241)
    expanded = sp.expand(expression)
    b = sp.simplify(expanded.coeff(q))
    a = sp.simplify(expanded - b * q)
    if not (a.is_Rational and b.is_Rational):
        raise AssertionError("expression is not in Q(sqrt(241))")
    return Fraction(int(sp.numer(a)), int(sp.denom(a))), Fraction(
        int(sp.numer(b)), int(sp.denom(b))
    )


def _quadratic_field_sign(expression: sp.Expr) -> int:
    """Return the exact sign in the positive embedding of ``sqrt(241)``."""

    a, b = _quadratic_field_parts(expression)
    if not b:
        return (a > 0) - (a < 0)
    if not a:
        return (b > 0) - (b < 0)
    if a > 0 and b > 0:
        return 1
    if a < 0 and b < 0:
        return -1
    comparison = a * a - 241 * b * b
    if not comparison:
        raise AssertionError("unexpected rational square root of 241")
    if a > 0:  # b < 0
        return 1 if comparison > 0 else -1
    return 1 if comparison < 0 else -1  # a < 0 < b


def certificate() -> dict[str, object]:
    """Build the exact pure-``V1`` candidate certificate."""

    x, c = sp.symbols("x c", real=True)
    q = sp.sqrt(241)
    denominator = 1 + 2 * x**2
    first = 4 * x**4 + 8 * x**3 - 4 * x + 1
    second = 4 * x**4 - 16 * x**3 + 12 * x**2 + 8 * x + 1
    third = 4 * x**4 - 4 * x**3 + 6 * x**2 + 2 * x + 1
    line_ratio = sp.factor(first**10 * second**3 * third**5 / denominator**36)

    curve_coordinate = sp.factor(-4 * x * (x - 1) * (2 * x + 1) / denominator**2)
    curve_ratio = sp.factor((1 - c) ** 10 * (c + 2) ** 5 * (2 * c + 1) ** 3 / 32)
    coordinate_identities = {
        "one_minus_c": sp.factor(1 - curve_coordinate - first / denominator**2)
        == 0,
        "two_c_plus_one": sp.factor(
            2 * curve_coordinate + 1 - second / denominator**2
        )
        == 0,
        "c_plus_two": sp.factor(
            curve_coordinate + 2 - 2 * third / denominator**2
        )
        == 0,
        "ratio_composition": sp.factor(
            line_ratio - curve_ratio.subs(c, curve_coordinate)
        )
        == 0,
    }
    lower_square = sp.factor(curve_coordinate + sp.Rational(1, 2))
    upper_square = sp.factor(1 - curve_coordinate)
    expected_lower = (2 * x**2 - 4 * x - 1) ** 2 / (2 * denominator**2)
    expected_upper = (2 * x**2 + 2 * x - 1) ** 2 / denominator**2
    interval_certificate = {
        "c_plus_one_half_identity": sp.factor(lower_square - expected_lower) == 0,
        "one_minus_c_identity": sp.factor(upper_square - expected_upper) == 0,
        "image_contained_in_closed_interval": "-1/2 <= c(x) <= 1",
    }

    c_star = (-17 + q) / 24
    other_root = (-17 - q) / 24
    derivative = sp.factor(sp.diff(curve_ratio, c))
    expected_derivative = sp.factor(
        3
        * (c - 1) ** 9
        * (c + 2) ** 4
        * (2 * c + 1) ** 2
        * (12 * c**2 + 17 * c + 1)
        / 32
    )
    candidate_ratio = sp.factor(curve_ratio.subs(c, c_star))
    stationary_quadratic = 12 * c**2 + 17 * c + 1
    stationary_root_identity = sp.factor(
        stationary_quadratic - 12 * (c - c_star) * (c - other_root)
    ) == 0
    derivative_sign_certificate = {
        "stationary_quadratic_root_identity": stationary_root_identity,
        "other_root_below_minus_one_half": _quadratic_field_sign(
            other_root + sp.Rational(1, 2)
        )
        < 0,
        "candidate_inside_open_interval": bool(
            _quadratic_field_sign(c_star + sp.Rational(1, 2)) > 0
            and _quadratic_field_sign(1 - c_star) > 0
        ),
        "sign_logic": (
            "On -1/2<c<1 the nonquadratic derivative factor is negative; "
            "the stationary quadratic is negative before c_star and positive "
            "after c_star. Hence the ratio increases then decreases."
        ),
        "boundary_values_zero": bool(
            curve_ratio.subs(c, sp.Rational(-1, 2)) == 0
            and curve_ratio.subs(c, 1) == 0
        ),
    }
    derivative_sign_certificate["unique_global_maximum_on_closed_interval"] = all(
        value
        for key, value in derivative_sign_certificate.items()
        if key != "sign_logic"
    )

    plus_quartic = sp.expand(
        x**4
        - (17 + q) * x**3
        + (19 + q) * x**2 / 2
        + (17 + q) * x / 2
        + sp.Rational(1, 4)
    )
    minus_quartic = plus_quartic.xreplace({q: -q})
    stationary_octic = (
        16 * x**8
        - 544 * x**7
        + 1072 * x**6
        - 1040 * x**5
        - 280 * x**4
        + 520 * x**3
        + 268 * x**2
        + 68 * x
        + 1
    )
    fiber_numerator = sp.fraction(sp.together(curve_coordinate - c_star))[0]
    expected_fiber_proportionality = 68 - 4 * q
    fiber_identity = sp.factor(
        fiber_numerator - expected_fiber_proportionality * plus_quartic,
        extension=q,
    ) == 0

    isolating_intervals = (
        (sp.Rational(-1, 2), sp.Rational(-2, 5)),
        (sp.Rational(-1, 50), sp.Rational(-1, 100)),
        (sp.Integer(1), sp.Rational(11, 10)),
        (sp.Integer(31), sp.Integer(32)),
    )
    interval_rows = []
    for left, right in isolating_intervals:
        left_value = sp.simplify(plus_quartic.subs(x, left))
        right_value = sp.simplify(plus_quartic.subs(x, right))
        left_sign = _quadratic_field_sign(left_value)
        right_sign = _quadratic_field_sign(right_value)
        interval_rows.append(
            {
                "left": str(left),
                "right": str(right),
                "left_value": str(left_value),
                "right_value": str(right_value),
                "left_sign": left_sign,
                "right_sign": right_sign,
                "strict_sign_change": left_sign * right_sign == -1,
            }
        )

    octic_real_root_count = int(sp.Poly(stationary_octic, x).count_roots(-sp.oo, sp.oo))
    four_isolated_roots = bool(
        len(interval_rows) == 4
        and all(row["strict_sign_change"] for row in interval_rows)
    )
    all_real_octic_roots_are_candidate_fiber = bool(
        four_isolated_roots and octic_real_root_count == 4
    )
    maximizer_passed = bool(
        sp.factor(derivative - expected_derivative) == 0
        and derivative_sign_certificate["unique_global_maximum_on_closed_interval"]
    )
    passed = bool(
        all(coordinate_identities.values())
        and interval_certificate["c_plus_one_half_identity"]
        and interval_certificate["one_minus_c_identity"]
        and maximizer_passed
        and sp.factor(stationary_octic - 16 * plus_quartic * minus_quartic) == 0
        and fiber_identity
        and all_real_octic_roots_are_candidate_fiber
    )
    return {
        "schema_version": 1,
        "claim_scope": "complete real pure-V1 graph line in the coupled Spin(9) normal slice",
        "curve_coordinate": str(curve_coordinate),
        "coordinate_identities": coordinate_identities,
        "interval_certificate": interval_certificate,
        "symmetric_curve_ratio": str(curve_ratio),
        "pure_v1_ratio_is_curve_composition": coordinate_identities["ratio_composition"],
        "candidate": {
            "c_star": str(c_star),
            "other_stationary_root": str(other_root),
            "other_root_below_feasible_interval": True,
            "derivative_factorization": str(derivative),
            "derivative_identity_verified_exactly": derivative == expected_derivative,
            "derivative_sign_certificate": derivative_sign_certificate,
            "unique_curve_maximizer": maximizer_passed,
            "determinant_ratio": str(candidate_ratio),
        },
        "graph_fiber": {
            "stationary_octic": str(stationary_octic),
            "quartic_over_q_sqrt_241": str(plus_quartic),
            "conjugate_quartic": str(minus_quartic),
            "octic_factorization_verified_exactly": sp.factor(
                stationary_octic - 16 * plus_quartic * minus_quartic
            )
            == 0,
            "candidate_fiber_proportionality": str(expected_fiber_proportionality),
            "candidate_fiber_identity_verified_exactly": fiber_identity,
            "isolating_intervals": interval_rows,
            "octic_real_root_count_by_exact_sturm": octic_real_root_count,
            "all_real_octic_roots_are_the_four_candidate_preimages": (
                all_real_octic_roots_are_candidate_fiber
            ),
        },
        "pure_v1_global_candidate_optimality_certified": passed,
        "equality_classification": (
            "Equality holds at exactly four finite graph coordinates, one in "
            "each stored rational interval; all four map to c=c_star."
        ),
        "coupled_p_positive_candidate_optimality_certified": False,
        "global_grassmann_quotient_certified": False,
        "passed": passed,
    }


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
