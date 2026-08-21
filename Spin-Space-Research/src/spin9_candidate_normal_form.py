"""Exact invariant normal form at the four Spin(9) candidate preimages.

Let ``G = R_* delta**14 - N`` be the reconstructed candidate determinant gap
on the coupled ``V1 + V5`` graph slice.  The pure-``V1`` restriction factors
through the square of the quartic candidate fiber ``Q(x)``.  This module also
proves that the coefficient of the radial ``V5`` invariant ``p`` is positive
at all four real roots of ``Q``.

The first three transverse-coefficient roots are extraordinarily close to the
corresponding candidate roots.  Fixed rational intervals of width ``1e-140``
separate them.  Every root count and endpoint sign is replayed exactly over
``Q(sqrt(241))``; decimal numerics are not part of the certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import sympy as sp
from spin9_v1_candidate_line import _quadratic_field_sign
from spin9_v1_v5_gap import DEFAULT_COEFFICIENT_ARTIFACT
from spin9_v1_v5_reconstruction import COUPLED_MONOMIALS, ROOT, load_coefficients

DEFAULT_OUTPUT = ROOT / "artifacts" / "spin9_candidate_normal_form_20260821.json"
X = sp.symbols("x", real=True)
SQRT_241 = sp.sqrt(241)

# These cells were proposed numerically once, then frozen.  Certificate replay
# uses only exact Sturm variations and exact quadratic-field endpoint signs.
ISOLATING_CELLS = (
    (
        -47690053118170423976506191976351203824510092156471482459985563640850815222950325697430749977707137503777762590116301962540906591292071333952,
        140,
    ),
    (
        -1564050410342347850864384697273002911123110041593738354472152830138579085302509158637491245413959254404633911054174249698347636079345263006,
        140,
    ),
    (
        104843665986502039306264824228575628861211682068599272484006133440905232176032340136651731158263371574958169346025893418905843740524560017620,
        140,
    ),
    (3196827907168013105407, 20),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_ratio() -> sp.Expr:
    q = SQRT_241
    return sp.factor(
        (-41 + q) ** 10
        * (-5 + q) ** 3
        * (q + 31) ** 5
        / sp.Integer(27_916_588_318_337_525_511_880_704)
    )


def _candidate_fiber() -> sp.Expr:
    q = SQRT_241
    return (
        X**4
        - (17 + q) * X**3
        + (19 + q) * X**2 / 2
        + (17 + q) * X / 2
        + sp.Rational(1, 4)
    )


def _coefficient(
    coefficients: tuple,
    p_power: int,
    y_power: int,
) -> sp.Expr:
    return sum(
        sp.Rational(value.numerator, value.denominator) * X**x_power
        for (x_power, p_index, y_index), value in zip(
            COUPLED_MONOMIALS, coefficients, strict=True
        )
        if p_index == p_power and y_index == y_power
    )


def _variations(polynomial: sp.Poly, point: sp.Rational) -> int:
    signs = [
        _quadratic_field_sign(sp.expand(member.eval(point)))
        for member in polynomial.sturm()
    ]
    nonzero = [sign for sign in signs if sign]
    return sum(left != right for left, right in pairwise(nonzero))


def _root_count(
    polynomial: sp.Poly,
    lower: sp.Rational,
    upper: sp.Rational,
) -> int:
    return _variations(polynomial, lower) - _variations(polynomial, upper)


def _interval_row(
    fiber: sp.Poly,
    radial: sp.Poly,
    pure_quotient: sp.Poly,
    numerator: int,
    decimal_power: int,
) -> dict[str, object]:
    denominator = 10**decimal_power
    lower = sp.Rational(numerator, denominator)
    upper = lower + sp.Rational(1, denominator)
    radial_signs = [
        _quadratic_field_sign(radial.eval(endpoint))
        for endpoint in (lower, upper)
    ]
    quotient_signs = [
        _quadratic_field_sign(pure_quotient.eval(endpoint))
        for endpoint in (lower, upper)
    ]
    row = {
        "lower": str(lower),
        "upper": str(upper),
        "decimal_width_exponent": -decimal_power,
        "candidate_fiber_root_count": _root_count(fiber, lower, upper),
        "radial_coefficient_root_count": _root_count(radial, lower, upper),
        "radial_coefficient_endpoint_signs": radial_signs,
        "pure_quotient_root_count": _root_count(pure_quotient, lower, upper),
        "pure_quotient_endpoint_signs": quotient_signs,
    }
    row["candidate_fiber_isolated"] = row["candidate_fiber_root_count"] == 1
    row["radial_coefficient_positive_on_cell"] = bool(
        row["radial_coefficient_root_count"] == 0 and radial_signs == [1, 1]
    )
    row["pure_quotient_positive_on_cell"] = bool(
        row["pure_quotient_root_count"] == 0 and quotient_signs == [1, 1]
    )
    row["passed"] = bool(
        row["candidate_fiber_isolated"]
        and row["radial_coefficient_positive_on_cell"]
        and row["pure_quotient_positive_on_cell"]
    )
    return row


def certificate() -> dict[str, object]:
    """Build the exact candidate-fiber normal-form certificate."""

    coefficients = load_coefficients(DEFAULT_COEFFICIENT_ARTIFACT)
    candidate_ratio = _candidate_ratio()
    delta_zero = 1 + 6 * X**2 + 12 * X**4 + 8 * X**6
    delta_p = 8 + 24 * X**2 + 16 * X**4
    numerator_zero = _coefficient(coefficients, 0, 0)
    numerator_p = _coefficient(coefficients, 1, 0)
    fiber = sp.Poly(_candidate_fiber(), X, extension=SQRT_241)
    pure_gap = sp.Poly(
        sp.expand(candidate_ratio * delta_zero**14 - numerator_zero),
        X,
        extension=SQRT_241,
    )
    pure_quotient, pure_remainder = pure_gap.div(fiber**2)
    pure_quotient_on_fiber = pure_quotient.rem(fiber)
    radial_coefficient = sp.Poly(
        sp.expand(
            14 * candidate_ratio * delta_zero**13 * delta_p - numerator_p
        ),
        X,
        extension=SQRT_241,
    )
    radial_on_fiber = radial_coefficient.rem(fiber)
    intervals = [
        _interval_row(
            fiber,
            radial_on_fiber,
            pure_quotient_on_fiber,
            numerator,
            decimal_power,
        )
        for numerator, decimal_power in ISOLATING_CELLS
    ]
    fiber_simple = sp.polys.polytools.gcd(fiber, fiber.diff()).degree() == 0
    passed = bool(
        pure_remainder.is_zero
        and pure_quotient.degree() == 76
        and fiber_simple
        and sum(row["candidate_fiber_root_count"] for row in intervals) == 4
        and all(row["passed"] for row in intervals)
    )
    return {
        "schema_version": 1,
        "claim_scope": (
            "exact invariant leading normal form at all four pure-V1 graph "
            "preimages of the Spin(9) algebraic candidate"
        ),
        "coefficient_artifact": DEFAULT_COEFFICIENT_ARTIFACT.name,
        "coefficient_artifact_sha256": _sha256(DEFAULT_COEFFICIENT_ARTIFACT),
        "coefficient_field": "Q(sqrt(241)), positive real embedding",
        "candidate_ratio": str(candidate_ratio),
        "candidate_fiber_quartic": str(sp.factor(fiber.as_expr(), extension=SQRT_241)),
        "candidate_fiber_simple": fiber_simple,
        "pure_gap_degree": pure_gap.degree(),
        "pure_gap_divisible_by_candidate_fiber_squared": pure_remainder.is_zero,
        "pure_gap_quotient_degree": pure_quotient.degree(),
        "radial_coefficient_degree_before_reduction": radial_coefficient.degree(),
        "radial_coefficient_degree_mod_candidate_fiber": radial_on_fiber.degree(),
        "isolating_cells": intervals,
        "all_four_candidate_preimages_isolated": bool(
            sum(row["candidate_fiber_root_count"] for row in intervals) == 4
        ),
        "pure_quadratic_coefficient_positive_at_all_preimages": all(
            row["pure_quotient_positive_on_cell"] for row in intervals
        ),
        "mixed_radial_coefficient_positive_at_all_preimages": all(
            row["radial_coefficient_positive_on_cell"] for row in intervals
        ),
        "orbit_shape_constraint": "p>=0 and 27*y^2<=2*p^3",
        "local_consequence": (
            "G=Q(x)^2*H(x)+p*A(x)+y*B(x)+O(p^2,p*y,y^2), with "
            "H and A positive at every candidate preimage; since y=O(p^(3/2)), "
            "the reconstructed candidate gap is locally strict off the fiber"
        ),
        "explicit_finite_radius_certified": False,
        "second_v5_copy_certified": False,
        "unrestricted_quotient_certified": False,
        "passed": passed,
    }


def verify_report(report: dict[str, object]) -> bool:
    """Check source integrity and the stored exact claim topology."""

    rows = report.get("isolating_cells", [])
    return bool(
        report.get("passed") is True
        and report.get("coefficient_artifact_sha256")
        == _sha256(DEFAULT_COEFFICIENT_ARTIFACT)
        and report.get("pure_gap_divisible_by_candidate_fiber_squared") is True
        and report.get("all_four_candidate_preimages_isolated") is True
        and report.get("pure_quadratic_coefficient_positive_at_all_preimages")
        is True
        and report.get("mixed_radial_coefficient_positive_at_all_preimages")
        is True
        and report.get("explicit_finite_radius_certified") is False
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
