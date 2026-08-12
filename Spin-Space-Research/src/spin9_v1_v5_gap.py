"""Exact Bernstein bounds for the reconstructed coupled Spin(9) V1+V5 ratio.

This module studies the maintained rational function

    R(x,p,y) = N(x,p,y) / delta(x,p,y)**14

on the complete normal-slice orbit domain.  It uses the joint compactification

    x = sign*a*rho,  r = (1-a)*rho,  rho = t/(1-t)

and the Cartan chamber parameter ``0 <= z <= 1``.  The generated Bernstein
certificate is exact for the reconstructed polynomial.  Until the separate
raw characteristic-zero determinant identity is promoted, it is deliberately
reported as a bound on the reconstructed rational function rather than as a
determinant theorem.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

from spin9_v1_v5_reconstruction import (
    COUPLED_MONOMIALS,
    ROOT,
    load_coefficients,
)
from spin9_v5_cartan_certificate import _split_bernstein_half, _tensor_digest

DEFAULT_COEFFICIENT_ARTIFACT = (
    ROOT / "artifacts" / "spin9_v1_v5_reconstruction_20260811.json"
)
COMPACT_DEGREE = 84
DEFAULT_UPPER_NUMERATOR = 21
DEFAULT_UPPER_DENOMINATOR = 20


def _multiply_sparse(
    left: dict[tuple[int, int, int], int],
    right: dict[tuple[int, int, int], int],
) -> dict[tuple[int, int, int], int]:
    result: dict[tuple[int, int, int], int] = {}
    for left_powers, left_value in left.items():
        for right_powers, right_value in right.items():
            powers = tuple(
                left_power + right_power
                for left_power, right_power in zip(
                    left_powers,
                    right_powers,
                    strict=True,
                )
            )
            result[powers] = result.get(powers, 0) + left_value * right_value
    return {powers: value for powers, value in result.items() if value}


def invariant_gap_coefficients(
    coefficients: tuple[Fraction, ...],
    *,
    upper_numerator: int = DEFAULT_UPPER_NUMERATOR,
    upper_denominator: int = DEFAULT_UPPER_DENOMINATOR,
) -> dict[tuple[int, int, int], int]:
    """Return ``upper_numerator*delta^14-upper_denominator*N``."""

    if upper_numerator <= upper_denominator or upper_denominator <= 0:
        raise ValueError("the upper bound must be a positive ratio greater than one")
    if any(value.denominator != 1 for value in coefficients):
        raise ValueError("the maintained coupled numerator must be integral")
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
        monomial: upper_numerator * value for monomial, value in delta_power.items()
    }
    for monomial, coefficient in zip(
        COUPLED_MONOMIALS,
        coefficients,
        strict=True,
    ):
        result[monomial] = (
            result.get(monomial, 0) - upper_denominator * coefficient.numerator
        )
    return {monomial: value for monomial, value in result.items() if value}


def _shape_power_coefficients(p_power: int, y_power: int) -> dict[int, int]:
    p_direction = {
        index: math.comb(p_power, index) * 3 ** (p_power - index) * 9**index
        for index in range(p_power + 1)
    }
    y_direction = {
        index: math.comb(y_power, index) * (-1) ** (y_power - index) * 9**index
        for index in range(y_power + 1)
    }
    result: dict[int, int] = {}
    for left_index, left_value in p_direction.items():
        for right_index, right_value in y_direction.items():
            index = left_index + right_index
            result[index] = result.get(index, 0) + left_value * right_value
    return result


def compact_bernstein_controls(
    coefficients: tuple[Fraction, ...],
    *,
    scalar_sign: int,
    upper_numerator: int = DEFAULT_UPPER_NUMERATOR,
    upper_denominator: int = DEFAULT_UPPER_DENOMINATOR,
) -> tuple[np.ndarray, int, int]:
    """Return integer-scaled Bernstein controls on one compact unit cube."""

    if scalar_sign not in (-1, 1):
        raise ValueError("scalar_sign must be -1 or 1")
    gap = invariant_gap_coefficients(
        coefficients,
        upper_numerator=upper_numerator,
        upper_denominator=upper_denominator,
    )
    binomials = [math.comb(COMPACT_DEGREE, index) for index in range(85)]
    axis_scale = math.lcm(*binomials)
    inverse_binomial_scales = [axis_scale // value for value in binomials]
    denominator_scale = 2 ** (COMPACT_DEGREE // 2)

    shape_cache: dict[tuple[int, int], np.ndarray] = {}
    controls = np.empty((85, 85, 85), dtype=object)
    controls.fill(0)
    for (x_power, p_power, y_power), coefficient in gap.items():
        radial_degree = 2 * p_power + 3 * y_power
        weight = x_power + radial_degree
        shape_key = (p_power, y_power)
        if shape_key not in shape_cache:
            power = _shape_power_coefficients(p_power, y_power)
            shape_controls = np.zeros(85, dtype=object)
            for bernstein_index in range(85):
                shape_controls[bernstein_index] = sum(
                    value
                    * math.comb(bernstein_index, 2 * power_index)
                    * inverse_binomial_scales[2 * power_index]
                    for power_index, value in power.items()
                    if 2 * power_index <= bernstein_index
                )
            shape_cache[shape_key] = shape_controls
        shape_controls = shape_cache[shape_key]

        common = (
            coefficient
            * scalar_sign**x_power
            * inverse_binomial_scales[weight]
            * 2 ** (COMPACT_DEGREE // 2 - p_power - y_power)
        )
        for chamber_index in range(x_power, x_power + COMPACT_DEGREE - weight + 1):
            chamber_scale = (
                math.comb(COMPACT_DEGREE - weight, chamber_index - x_power)
                * inverse_binomial_scales[chamber_index]
            )
            controls[weight, chamber_index] += common * chamber_scale * shape_controls
    total_scale = axis_scale**3 * denominator_scale
    return controls, total_scale, len(gap)


def _dyadic_box(path: str) -> dict[str, str]:
    bounds = [[Fraction(0), Fraction(1)] for _ in range(3)]
    symbols = ("t", "a", "z")
    for step in filter(None, path.split(",")):
        axis = symbols.index(step[0])
        midpoint = (bounds[axis][0] + bounds[axis][1]) / 2
        if step[1] == "0":
            bounds[axis][1] = midpoint
        else:
            bounds[axis][0] = midpoint
    return {
        f"{symbol}_{side}": str(bounds[axis][side_index])
        for axis, symbol in enumerate(symbols)
        for side_index, side in enumerate(("lower", "upper"))
    }


def bernstein_atlas_partial(
    controls: np.ndarray,
    *,
    maximum_depth: int = 18,
) -> dict[str, object]:
    """Build a dyadic atlas, retaining every unresolved depth-limit cell.

    A retained cell is not a positivity certificate.  It is an exact handoff
    box for a separately certified local chart.  Keeping these boxes instead
    of raising immediately is what permits an auditable projective atlas near
    rank-loss directions.
    """

    native_negative = sum(value < 0 for value in controls.flat)
    native_zero = sum(value == 0 for value in controls.flat)
    stack = [(controls, 0, "")]
    leaves: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []
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
            unresolved.append(
                {
                    "path": path.rstrip(",") or "root",
                    "depth": depth,
                    "box": _dyadic_box(path),
                    "minimum_scaled_coefficient": str(minimum),
                    "controls_sha256": _tensor_digest(cell),
                }
            )
            continue
        axis = depth % 3
        lower, upper = _split_bernstein_half(cell, axis)
        symbol = ("t", "a", "z")[axis]
        stack.append((upper, depth + 1, f"{path}{symbol}1,"))
        stack.append((lower, depth + 1, f"{path}{symbol}0,"))
        split_count += 1
    return {
        "native_negative_coefficient_count": native_negative,
        "native_zero_coefficient_count": native_zero,
        "native_controls_sha256": _tensor_digest(controls),
        "split_rule": "cycle t, a, z midpoint splits",
        "split_count": split_count,
        "leaf_count": len(leaves),
        "leaf_depth_histogram": dict(
            sorted(Counter(str(leaf["depth"]) for leaf in leaves).items())
        ),
        "unresolved_count": len(unresolved),
        "unresolved": unresolved,
        "all_leaf_controls_strictly_positive": not unresolved,
        "leaves": leaves,
    }


def bernstein_atlas(
    controls: np.ndarray,
    *,
    maximum_depth: int = 18,
) -> dict[str, object]:
    """Build an exact strict dyadic atlas for one sign chart."""

    report = bernstein_atlas_partial(controls, maximum_depth=maximum_depth)
    if report["unresolved"]:
        first = report["unresolved"][0]
        raise AssertionError(
            "strict Bernstein atlas failed at depth "
            f"{first['depth']} on {first['path']},"
        )
    return report


def certificate(
    coefficient_artifact: Path = DEFAULT_COEFFICIENT_ARTIFACT,
    *,
    upper_numerator: int = DEFAULT_UPPER_NUMERATOR,
    upper_denominator: int = DEFAULT_UPPER_DENOMINATOR,
    maximum_depth: int = 18,
) -> dict[str, object]:
    coefficients = load_coefficients(coefficient_artifact)
    charts: dict[str, object] = {}
    common_scale: int | None = None
    gap_term_count: int | None = None
    for sign in (1, -1):
        controls, scale, terms = compact_bernstein_controls(
            coefficients,
            scalar_sign=sign,
            upper_numerator=upper_numerator,
            upper_denominator=upper_denominator,
        )
        common_scale = scale if common_scale is None else common_scale
        gap_term_count = terms if gap_term_count is None else gap_term_count
        if scale != common_scale or terms != gap_term_count:
            raise AssertionError("sign charts disagree on common certificate data")
        charts["nonnegative_x" if sign == 1 else "nonpositive_x"] = bernstein_atlas(
            controls,
            maximum_depth=maximum_depth,
        )
    passed = all(
        chart["all_leaf_controls_strictly_positive"] for chart in charts.values()
    )
    return {
        "schema_version": 1,
        "claim_scope": "exact bound on the reconstructed coupled V1+V5 rational function",
        "coefficient_artifact": coefficient_artifact.name,
        "upper_bound": f"{upper_numerator}/{upper_denominator}",
        "orbit_domain": "x real, p>=0, 27*y^2<=2*p^3",
        "joint_compactification": {
            "x": "+/- a*rho",
            "v5_radial": "r=(1-a)*rho",
            "rho": "t/(1-t)",
            "p": "(3+9*z^2)*r^2/2",
            "y": "(9*z^2-1)*r^3/2",
            "cube": "0<=t,a,z<=1",
        },
        "compact_polynomial": (
            f"(1-t)^84 * ({upper_numerator}*delta^14-{upper_denominator}*N)"
        ),
        "compact_multidegree": [84, 84, 84],
        "gap_invariant_term_count": gap_term_count,
        "bernstein_common_scale": str(common_scale),
        "charts": charts,
        "raw_characteristic_zero_determinant_identity_certified": False,
        "global_determinant_theorem_claimed": False,
        "passed": passed,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coefficient-artifact", type=Path, default=DEFAULT_COEFFICIENT_ARTIFACT
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--maximum-depth", type=int, default=18)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = certificate(
        args.coefficient_artifact,
        maximum_depth=args.maximum_depth,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
