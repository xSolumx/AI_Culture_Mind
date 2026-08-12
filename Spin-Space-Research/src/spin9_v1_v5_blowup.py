"""Exact blow-up formulas and Bernstein bounds at coupled V1+V5 infinity.

The joint projective compactification of the coupled normal slice has two
rank-loss base points.  Boundary-adapted coordinates retain the finite third
spinor and reduce the reconstructed determinant ratio to two explicit
degree-28 rational functions.  This module derives those functions from the
maintained 18,600-coefficient numerator and certifies that both are below
26/25 on their complete real planes.

An independent raw characteristic-zero boundary artifact verifies both factor
formulas directly from the limiting information matrices.  The chained result
is therefore an exact boundary theorem, but it does not certify the full
finite-radius coupled slice or the global rank-three determinant problem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp

from spin9_v1_v5_reconstruction import (
    COUPLED_MONOMIALS,
    ROOT,
    load_coefficients,
)
from spin9_v5_cartan_certificate import _split_bernstein_half, _tensor_digest
from spin9_v5_ray_certificate import _quadratic_field_sign_witness

DEFAULT_COEFFICIENT_ARTIFACT = (
    ROOT / "artifacts" / "spin9_v1_v5_reconstruction_20260811.json"
)
DEFAULT_RAW_BOUNDARY_ARTIFACT = (
    ROOT / "artifacts" / "spin9_v1_v5_boundary_char0_20260811.json"
)
COMPACT_DEGREE = 28
UPPER_NUMERATOR = 26
UPPER_DENOMINATOR = 25


def _multiply_polynomials(
    left: list[Fraction],
    right: list[Fraction],
) -> list[Fraction]:
    result = [Fraction(0)] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            result[left_index + right_index] += left_value * right_value
    return result


def _polynomial_powers(
    polynomial: list[Fraction],
    maximum_power: int,
) -> tuple[list[Fraction], ...]:
    result = [[Fraction(1)]]
    for _ in range(maximum_power):
        result.append(_multiply_polynomials(result[-1], polynomial))
    return tuple(result)


def boundary_numerator_coefficients(
    coefficients: tuple[Fraction, ...],
    family: str,
) -> dict[tuple[int, int], Fraction]:
    """Extract the rho^56 coefficient at one projective rank-loss point."""

    if family == "A":
        scalar_leading = 1
        p_direction = [Fraction(3, 2), Fraction(0), Fraction(1, 2)]
        y_direction = [Fraction(-1, 2), Fraction(0), Fraction(1, 2)]
    elif family == "B":
        scalar_leading = -2
        p_direction = [Fraction(6), Fraction(-9), Fraction(9, 2)]
        y_direction = [Fraction(4), Fraction(-9), Fraction(9, 2)]
    else:
        raise ValueError("family must be 'A' or 'B'")

    p_powers = _polynomial_powers(p_direction, 42)
    y_powers = _polynomial_powers(y_direction, 28)
    direction_cache: dict[tuple[int, int], list[Fraction]] = {}
    result: dict[tuple[int, int], Fraction] = {}
    for (x_power, p_power, y_power), coefficient in zip(
        COUPLED_MONOMIALS,
        coefficients,
        strict=True,
    ):
        weight = x_power + 2 * p_power + 3 * y_power
        if weight < 56:
            continue
        direction_key = (p_power, y_power)
        if direction_key not in direction_cache:
            direction_cache[direction_key] = _multiply_polynomials(
                p_powers[p_power],
                y_powers[y_power],
            )
        direction = direction_cache[direction_key]
        for finite_scalar_power in range(x_power + 1):
            direction_power = weight - 56 - finite_scalar_power
            if 0 <= direction_power < len(direction):
                value = direction[direction_power]
                if value:
                    monomial = (finite_scalar_power, direction_power)
                    result[monomial] = result.get(monomial, Fraction(0)) + (
                        coefficient
                        * math.comb(x_power, finite_scalar_power)
                        * scalar_leading ** (x_power - finite_scalar_power)
                        * value
                    )
    return {monomial: value for monomial, value in result.items() if value}


def _polynomial_expression(
    coefficients: dict[tuple[int, int], Fraction],
) -> sp.Expr:
    d, w = sp.symbols("d w", real=True)
    return sp.expand(
        sum(
            sp.Rational(value.numerator, value.denominator) * d**d_power * w**w_power
            for (d_power, w_power), value in coefficients.items()
        )
    )


def expected_factorization(family: str) -> tuple[sp.Expr, sp.Expr, int]:
    """Return the normalized numerator product, norm, and Gram scale."""

    d, w = sp.symbols("d w", real=True)
    if family == "A":
        norm = 1 + 2 * d**2 + 2 * w**2
        factors = (
            (2 * d**2 + 2 * w**2 - 1) ** 6,
            4 * d**4 + 8 * d**2 * w**2 + 2 * d**2 + 4 * w**4 + 2 * w**2 + 1,
            4 * d**4 + 8 * d**2 * w**2 + 2 * d**2 + 4 * w**4 + 4 * w**2 + 1,
            8 * d**4
            + 4 * d**3
            + 16 * d**2 * w**2
            - 4 * d**2 * w
            + 7 * d**2
            + 4 * d * w**2
            - 2 * d * w
            + 2 * d
            + 8 * w**4
            - 4 * w**3
            + 7 * w**2
            - 2 * w
            + 2,
            8 * d**4
            + 4 * d**3
            + 16 * d**2 * w**2
            + 4 * d**2 * w
            + 7 * d**2
            + 4 * d * w**2
            + 2 * d * w
            + 2 * d
            + 8 * w**4
            + 4 * w**3
            + 7 * w**2
            + 2 * w
            + 2,
        )
        return sp.prod(factors) / 4, norm, 81
    if family == "B":
        norm = 1 + 2 * d**2 - 6 * d * w + 9 * w**2
        factors = (
            (2 * d**2 - 6 * d * w + 9 * w**2 - 1) ** 6,
            4 * d**4
            - 24 * d**3 * w
            + 72 * d**2 * w**2
            + 2 * d**2
            - 108 * d * w**3
            - 6 * d * w
            + 81 * w**4
            + 9 * w**2
            + 1,
            8 * d**4
            - 48 * d**3 * w
            + 144 * d**2 * w**2
            + 4 * d**2
            - 216 * d * w**3
            - 12 * d * w
            + 162 * w**4
            + 27 * w**2
            + 2,
            8 * d**4
            - 48 * d**3 * w
            + 4 * d**3
            + 144 * d**2 * w**2
            - 12 * d**2 * w
            + 7 * d**2
            - 216 * d * w**3
            + 18 * d * w**2
            - 18 * d * w
            + 2 * d
            + 162 * w**4
            + 27 * w**2
            + 2,
            8 * d**4
            - 48 * d**3 * w
            + 4 * d**3
            + 144 * d**2 * w**2
            - 24 * d**2 * w
            + 7 * d**2
            - 216 * d * w**3
            + 54 * d * w**2
            - 24 * d * w
            + 2 * d
            + 162 * w**4
            - 54 * w**3
            + 36 * w**2
            - 6 * w
            + 2,
        )
        return sp.prod(factors) / 8, norm, 1296
    raise ValueError("family must be 'A' or 'B'")


def compact_bernstein_controls(
    gap: sp.Expr,
    d_sign: int,
    w_sign: int,
) -> tuple[np.ndarray, int]:
    """Return exact controls for d=sign*a*rho, w=sign*(1-a)*rho."""

    if d_sign not in (-1, 1) or w_sign not in (-1, 1):
        raise ValueError("chart signs must be +/-1")
    d, w = sp.symbols("d w", real=True)
    polynomial = sp.Poly(sp.expand(gap), d, w, domain=sp.QQ)
    if polynomial.total_degree() > COMPACT_DEGREE:
        raise AssertionError("blow-up gap exceeded degree 28")
    binomials = [math.comb(COMPACT_DEGREE, index) for index in range(29)]
    axis_scale = math.lcm(*binomials)
    controls = np.empty((29, 29), dtype=object)
    controls.fill(0)
    for (d_power, w_power), coefficient in polynomial.terms():
        total_degree = d_power + w_power
        common = (
            int(coefficient)
            * d_sign**d_power
            * w_sign**w_power
            * (axis_scale // binomials[total_degree])
        )
        for chamber_index in range(
            d_power,
            d_power + COMPACT_DEGREE - total_degree + 1,
        ):
            controls[total_degree, chamber_index] += (
                common
                * math.comb(
                    COMPACT_DEGREE - total_degree,
                    chamber_index - d_power,
                )
                * (axis_scale // binomials[chamber_index])
            )
    return controls, axis_scale**2


def _dyadic_box(path: str) -> dict[str, str]:
    bounds = [[Fraction(0), Fraction(1)] for _ in range(2)]
    for step in filter(None, path.split(",")):
        axis = 0 if step[0] == "t" else 1
        midpoint = (bounds[axis][0] + bounds[axis][1]) / 2
        if step[1] == "0":
            bounds[axis][1] = midpoint
        else:
            bounds[axis][0] = midpoint
    return {
        "t_lower": str(bounds[0][0]),
        "t_upper": str(bounds[0][1]),
        "a_lower": str(bounds[1][0]),
        "a_upper": str(bounds[1][1]),
    }


def bernstein_atlas(
    controls: np.ndarray,
    maximum_depth: int = 20,
) -> dict[str, object]:
    native_negative = sum(value < 0 for value in controls.flat)
    native_zero = sum(value == 0 for value in controls.flat)
    stack = [(controls, 0, "")]
    leaves: list[dict[str, object]] = []
    split_count = 0
    while stack:
        cell, depth, path = stack.pop()
        minimum = min(cell.flat)
        if minimum > 0:
            leaves.append(
                {
                    "path": path.rstrip(",") or "root",
                    "depth": depth,
                    "box": _dyadic_box(path),
                    "minimum_scaled_coefficient": str(minimum),
                    "controls_sha256": _tensor_digest(cell),
                }
            )
            continue
        if depth >= maximum_depth:
            raise AssertionError(f"blow-up atlas failed at {path}")
        axis = depth % 2
        lower, upper = _split_bernstein_half(cell, axis)
        symbol = "t" if axis == 0 else "a"
        stack.append((upper, depth + 1, f"{path}{symbol}1,"))
        stack.append((lower, depth + 1, f"{path}{symbol}0,"))
        split_count += 1
    return {
        "native_negative_coefficient_count": native_negative,
        "native_zero_coefficient_count": native_zero,
        "native_controls_sha256": _tensor_digest(controls),
        "split_count": split_count,
        "leaf_count": len(leaves),
        "leaf_depth_histogram": dict(
            sorted(Counter(str(leaf["depth"]) for leaf in leaves).items())
        ),
        "all_leaf_controls_strictly_positive": True,
        "leaves": leaves,
    }


def certificate(
    coefficient_artifact: Path = DEFAULT_COEFFICIENT_ARTIFACT,
    raw_boundary_artifact: Path = DEFAULT_RAW_BOUNDARY_ARTIFACT,
) -> dict[str, object]:
    coefficients = load_coefficients(coefficient_artifact)
    raw_boundary_report = json.loads(raw_boundary_artifact.read_text(encoding="utf-8"))
    raw_boundary_passed = bool(
        raw_boundary_report["passed"]
        and not raw_boundary_report["modular_reconstruction_used"]
        and not raw_boundary_report["finite_radius_coupled_identity_certified"]
        and all(
            row["all_newton_grid_nodes_match"]
            and row["newton_grid_node_count"] == 2701
            for row in raw_boundary_report["families"]
        )
    )
    family_rows = []
    for family in ("A", "B"):
        extracted = boundary_numerator_coefficients(coefficients, family)
        observed = _polynomial_expression(extracted)
        normalized_numerator, norm, gram_scale = expected_factorization(family)
        expected = sp.expand(gram_scale**14 * normalized_numerator)
        factorization_valid = sp.expand(observed - expected) == 0
        gap = sp.expand(
            UPPER_NUMERATOR * sp.denom(normalized_numerator) * norm**14
            - UPPER_DENOMINATOR * sp.numer(normalized_numerator)
        )
        charts = []
        common_scale: int | None = None
        for d_sign in (-1, 1):
            for w_sign in (-1, 1):
                controls, scale = compact_bernstein_controls(gap, d_sign, w_sign)
                common_scale = scale if common_scale is None else common_scale
                if scale != common_scale:
                    raise AssertionError("quadrant control scales disagree")
                charts.append(
                    {
                        "d_sign": d_sign,
                        "w_sign": w_sign,
                        "atlas": bernstein_atlas(controls),
                    }
                )
        family_rows.append(
            {
                "family": family,
                "finite_norm": sp.sstr(norm),
                "gram_leading_scale": gram_scale,
                "boundary_numerator_term_count": len(extracted),
                "boundary_numerator_maximum_total_degree": max(
                    sum(monomial) for monomial in extracted
                ),
                "factorization": sp.sstr(sp.factor(normalized_numerator)),
                "extracted_factorization_identity_passed": bool(factorization_valid),
                "ratio": sp.sstr(sp.factor(normalized_numerator / norm**14)),
                "gap": (f"{UPPER_NUMERATOR}*denominator-{UPPER_DENOMINATOR}*numerator"),
                "bernstein_common_scale": str(common_scale),
                "quadrant_charts": charts,
                "all_quadrants_strictly_positive": all(
                    chart["atlas"]["all_leaf_controls_strictly_positive"]
                    for chart in charts
                ),
            }
        )

    q = sp.sqrt(241)
    c_star = (q - 17) / 24
    candidate_ratio = sp.factor(
        (1 - c_star) ** 10 * (c_star + 2) ** 5 * (2 * c_star + 1) ** 3 / 32
    )
    candidate_gap = sp.factor(
        candidate_ratio - sp.Rational(UPPER_NUMERATOR, UPPER_DENOMINATOR)
    )
    candidate_sign = _quadratic_field_sign_witness(candidate_gap, 241)
    passed = bool(
        all(
            row["extracted_factorization_identity_passed"]
            and row["all_quadrants_strictly_positive"]
            for row in family_rows
        )
        and candidate_sign["positive"]
        and raw_boundary_passed
    )
    canonical = json.dumps(family_rows, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "claim_scope": "raw characteristic-zero bound on both coupled boundary blow-up planes",
        "coefficient_artifact": coefficient_artifact.name,
        "raw_boundary_artifact": raw_boundary_artifact.name,
        "bound": "R_A(d,w), R_B(d,w) < 26/25 on R^2",
        "families": family_rows,
        "symmetric_candidate_comparison": {
            "candidate_ratio": sp.sstr(candidate_ratio),
            "candidate_ratio_approx": float(sp.N(candidate_ratio, 17)),
            "candidate_minus_26_over_25": sp.sstr(candidate_gap),
            "candidate_exceeds_26_over_25": candidate_sign,
        },
        "family_rows_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "raw_characteristic_zero_boundary_identities_certified": raw_boundary_passed,
        "raw_characteristic_zero_coupled_identity_certified": False,
        "global_coupled_determinant_theorem_claimed": False,
        "passed": passed,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coefficient-artifact",
        type=Path,
        default=DEFAULT_COEFFICIENT_ARTIFACT,
    )
    parser.add_argument(
        "--raw-boundary-artifact",
        type=Path,
        default=DEFAULT_RAW_BOUNDARY_ARTIFACT,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = certificate(args.coefficient_artifact, args.raw_boundary_artifact)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
