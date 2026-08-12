"""Exact toric Bernstein charts for the coupled Spin(9) ``V1 + V5`` gap.

The reconstructed invariant ratio is ``R=N/delta**14``.  This module uses the
two boundary-adapted affine charts

``A: h=1/r, x=r+d,   w=3*z*r``
``B: h=1/r, x=-2*r+d, k=(1-z)*r``

and constructs the integer polynomial ``2**42 * (21*delta_chart**14-20*N_chart)``.
Both chart polynomials have multidegree ``(56, 84, 84)`` and satisfy the exact
Newton support inequality ``radial_degree-h_degree <= 28``.  Four affine unit
cubes cover each chart: a compact core, an ``h``-dominant end, and the two
``R``-dominant toric ends distinguished by ``q=h*R <= 1`` or ``q >= 1``.

This file certifies the reconstructed rational function.  Promotion to a raw
determinant theorem additionally requires the separate characteristic-zero
identity bridge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from spin9_v1_v5_gap import (
    bernstein_atlas,
    bernstein_atlas_partial,
    compact_bernstein_controls,
)
from spin9_v1_v5_reconstruction import (
    COUPLED_MONOMIALS,
    ROOT,
    load_coefficients,
)
from spin9_v5_cartan_certificate import _integer_bernstein_tensor

DEFAULT_COEFFICIENT_ARTIFACT = (
    ROOT / "artifacts" / "spin9_v1_v5_reconstruction_20260811.json"
)
CHARTS = (
    "core",
    "h_infinity",
    "r_infinity_q_low",
    "r_infinity_q_high",
    "r_infinity_q_high_physical",
)
LOCAL_HANDOFF_CHARTS = ("core", "r_infinity_q_low")
GAP_SCALE = 2**42
H_DEGREE = 56
RADIAL_DEGREE = 84
NEWTON_EXCESS = 28

DELTA = {
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

Sparse = dict[tuple[int, int, int], int]


def _multiply_sparse(left: Sparse, right: Sparse) -> Sparse:
    result: defaultdict[tuple[int, int, int], int] = defaultdict(int)
    for left_powers, left_value in left.items():
        for right_powers, right_value in right.items():
            powers = tuple(
                left_powers[index] + right_powers[index] for index in range(3)
            )
            result[powers] += left_value * right_value
    return {powers: value for powers, value in result.items() if value}


def _power_sparse(polynomial: Sparse, exponent: int) -> Sparse:
    result: Sparse = {(0, 0, 0): 1}
    factor = polynomial
    while exponent:
        if exponent & 1:
            result = _multiply_sparse(result, factor)
        exponent >>= 1
        if exponent:
            factor = _multiply_sparse(factor, factor)
    return result


def _convolve(left: list[int], right: list[int]) -> list[int]:
    result = [0] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            result[left_index + right_index] += left_value * right_value
    return result


def _power_univariate(polynomial: list[int], exponent: int) -> list[int]:
    result = [1]
    factor = polynomial
    while exponent:
        if exponent & 1:
            result = _convolve(result, factor)
        exponent >>= 1
        if exponent:
            factor = _convolve(factor, factor)
    return result


def _a_scaled_polynomial(rows: tuple[tuple[tuple[int, int, int], int], ...], *, total_weight: int, scale_power: int) -> Sparse:
    """Substitute ``x=(1+d*h)/h`` and the family-A invariants."""

    result: defaultdict[tuple[int, int, int], int] = defaultdict(int)
    for (x_power, p_power, y_power), coefficient in rows:
        common = coefficient * 2 ** (scale_power - p_power - y_power)
        weight = x_power + 2 * p_power + 3 * y_power
        for x_index in range(x_power + 1):
            x_value = math.comb(x_power, x_index)
            for p_index in range(p_power + 1):
                p_value = (
                    math.comb(p_power, p_index)
                    * 3 ** (p_power - p_index)
                )
                for y_index in range(y_power + 1):
                    powers = (
                        total_weight
                        - weight
                        + x_index
                        + 2 * p_index
                        + 2 * y_index,
                        x_index,
                        2 * (p_index + y_index),
                    )
                    result[powers] += (
                        common
                        * x_value
                        * p_value
                        * math.comb(y_power, y_index)
                        * (-1) ** (y_power - y_index)
                    )
    return {powers: value for powers, value in result.items() if value}


def _b_scaled_polynomial(rows: tuple[tuple[tuple[int, int, int], int], ...], *, total_weight: int, scale_power: int) -> Sparse:
    """Substitute ``x=(-2+d*h)/h`` and the family-B invariants."""

    result: defaultdict[tuple[int, int, int], int] = defaultdict(int)
    shape_cache: dict[tuple[int, int], list[int]] = {}
    x_cache = tuple(
        tuple(math.comb(power, index) * (-2) ** (power - index) for index in range(power + 1))
        for power in range(85)
    )
    for (x_power, p_power, y_power), coefficient in rows:
        key = (p_power, y_power)
        if key not in shape_cache:
            shape_cache[key] = _convolve(
                _power_univariate([12, -18, 9], p_power),
                _power_univariate([8, -18, 9], y_power),
            )
        common = coefficient * 2 ** (scale_power - p_power - y_power)
        weight = x_power + 2 * p_power + 3 * y_power
        for x_index, x_value in enumerate(x_cache[x_power]):
            for shape_index, shape_value in enumerate(shape_cache[key]):
                powers = (
                    total_weight - weight + x_index + shape_index,
                    x_index,
                    shape_index,
                )
                result[powers] += common * x_value * shape_value
    return {powers: value for powers, value in result.items() if value}


def _coefficient_rows(path: Path) -> tuple[tuple[tuple[int, int, int], int], ...]:
    coefficients = load_coefficients(path)
    if any(value.denominator != 1 for value in coefficients):
        raise ValueError("the maintained coupled numerator must be integral")
    return tuple(
        (monomial, coefficient.numerator)
        for monomial, coefficient in zip(
            COUPLED_MONOMIALS,
            coefficients,
            strict=True,
        )
    )


def chart_gap_coefficients(
    family: str,
    coefficient_artifact: Path = DEFAULT_COEFFICIENT_ARTIFACT,
) -> tuple[Sparse, Sparse, Sparse]:
    """Return the scaled numerator, chart Gram determinant, and ``21/20`` gap."""

    if family not in {"A", "B"}:
        raise ValueError("family must be 'A' or 'B'")
    builder = _a_scaled_polynomial if family == "A" else _b_scaled_polynomial
    numerator_weight_84 = builder(
        _coefficient_rows(coefficient_artifact),
        total_weight=84,
        scale_power=42,
    )
    if min(powers[0] for powers in numerator_weight_84) < 28:
        raise AssertionError("the expected h^28 numerator cancellation failed")
    numerator = {
        (h_power - 28, d_power, w_power): value
        for (h_power, d_power, w_power), value in numerator_weight_84.items()
    }

    delta_weight_6 = builder(
        tuple(DELTA.items()),
        total_weight=6,
        scale_power=3,
    )
    if min(powers[0] for powers in delta_weight_6) < 2:
        raise AssertionError("the expected h^2 Gram cancellation failed")
    if any(value % 8 for value in delta_weight_6.values()):
        raise AssertionError("the chart Gram polynomial lost integrality")
    chart_delta = {
        (h_power - 2, d_power, w_power): value // 8
        for (h_power, d_power, w_power), value in delta_weight_6.items()
    }
    delta_power = _power_sparse(chart_delta, 14)
    gap = {powers: 21 * GAP_SCALE * value for powers, value in delta_power.items()}
    for powers, value in numerator.items():
        gap[powers] = gap.get(powers, 0) - 20 * value
    gap = {powers: value for powers, value in gap.items() if value}
    if max(d_power + w_power - h_power for h_power, d_power, w_power in gap) != NEWTON_EXCESS:
        raise AssertionError("the toric Newton support bound changed")
    return numerator, chart_delta, gap


def _sparse_digest(polynomial: Sparse) -> str:
    digest = hashlib.sha256()
    for powers, value in sorted(polynomial.items()):
        digest.update(f"{powers}:{value}\n".encode())
    return digest.hexdigest()


def chart_power_tensor(gap: Sparse, *, chart: str, d_sign: int) -> np.ndarray:
    """Pull one toric chart back to an affine unit cube in power form."""

    if chart not in CHARTS:
        raise ValueError(f"unknown chart {chart!r}")
    if d_sign not in (-1, 1):
        raise ValueError("d_sign must be -1 or 1")
    if chart == "r_infinity_q_high_physical":
        # Q=1/q, U=1/R and s=h*k=1-z.  On the Cartan chamber all three
        # variables lie in [0,1], while
        #
        #   h=U/Q, |d|=(1-s*Q)/U, k=s*Q/U.
        #
        # Multiplication by Q^56 U^28 makes every exponent nonnegative.
        power = np.zeros(
            (RADIAL_DEGREE + 1,) * 3,
            dtype=object,
        )
        for (h_power, d_power, k_power), coefficient in gap.items():
            u_power = NEWTON_EXCESS + h_power - d_power - k_power
            if u_power < 0:
                raise AssertionError("negative physical toric exponent")
            common = coefficient * d_sign**d_power
            for increment in range(d_power + 1):
                q_power = H_DEGREE - h_power + k_power + increment
                s_power = k_power + increment
                power[q_power, u_power, s_power] += (
                    common
                    * math.comb(d_power, increment)
                    * (-1) ** increment
                )
        return power

    power = np.zeros(
        (H_DEGREE + 1, RADIAL_DEGREE + 1, RADIAL_DEGREE + 1),
        dtype=object,
    )
    for (h_power, d_power, w_power), coefficient in gap.items():
        radial_power = d_power + w_power
        toric_power = NEWTON_EXCESS + h_power - radial_power
        if toric_power < 0:
            raise AssertionError("negative toric exponent")
        if chart == "core":
            first, second = h_power, radial_power
        elif chart == "h_infinity":
            first, second = H_DEGREE - h_power, radial_power
        elif chart == "r_infinity_q_low":
            first, second = h_power, toric_power
        else:
            first, second = H_DEGREE - h_power, toric_power
        common = coefficient * d_sign**d_power
        for increment in range(w_power + 1):
            shape_power = d_power + increment
            power[first, second, shape_power] += (
                common
                * math.comb(w_power, increment)
                * (-1) ** increment
            )
    return power


def chart_certificate(
    family: str,
    chart: str,
    d_sign: int,
    *,
    coefficient_artifact: Path = DEFAULT_COEFFICIENT_ARTIFACT,
    maximum_depth: int = 18,
) -> dict[str, object]:
    numerator, chart_delta, gap = chart_gap_coefficients(
        family,
        coefficient_artifact,
    )
    power = chart_power_tensor(gap, chart=chart, d_sign=d_sign)
    controls, bernstein_scale = _integer_bernstein_tensor(power)
    native_negative = sum(value < 0 for value in controls.flat)
    native_zero = sum(value == 0 for value in controls.flat)
    failure: str | None = None
    atlas: dict[str, object] | None = None
    try:
        atlas = bernstein_atlas(controls, maximum_depth=maximum_depth)
    except AssertionError as error:
        failure = str(error)
    passed = atlas is not None
    return {
        "schema_version": 1,
        "claim_scope": "exact 21/20 gap for the reconstructed coupled V1+V5 rational function",
        "family": family,
        "chart": chart,
        "d_sign": d_sign,
        "coefficient_artifact": coefficient_artifact.name,
        "numerator_term_count": len(numerator),
        "chart_delta_term_count": len(chart_delta),
        "gap_term_count": len(gap),
        "gap_multidegree": [
            max(powers[axis] for powers in gap) for axis in range(3)
        ],
        "newton_support_bound": "d_degree+w_degree-h_degree<=28",
        "gap_rows_sha256": _sparse_digest(gap),
        "power_tensor_shape": list(power.shape),
        "bernstein_scale": str(bernstein_scale),
        "native_negative_coefficient_count": native_negative,
        "native_zero_coefficient_count": native_zero,
        "atlas": atlas,
        "failure": failure,
        "reconstructed_rational_function_bound_certified": passed,
        "raw_characteristic_zero_determinant_identity_certified": False,
        "global_determinant_theorem_claimed": False,
        "passed": passed,
    }


def compact_chart_certificate(
    scalar_sign: int,
    *,
    coefficient_artifact: Path = DEFAULT_COEFFICIENT_ARTIFACT,
    maximum_depth: int = 18,
) -> dict[str, object]:
    """Return the exact compact atlas and its local-chart handoff boxes."""

    coefficients = load_coefficients(coefficient_artifact)
    controls, bernstein_scale, gap_term_count = compact_bernstein_controls(
        coefficients,
        scalar_sign=scalar_sign,
    )
    atlas = bernstein_atlas_partial(controls, maximum_depth=maximum_depth)
    handoffs = [
        _compact_handoff(scalar_sign, unresolved["box"])
        for unresolved in atlas["unresolved"]
    ]
    return {
        "scalar_sign": scalar_sign,
        "coefficient_artifact": coefficient_artifact.name,
        "gap_term_count": gap_term_count,
        "bernstein_scale": str(bernstein_scale),
        "atlas": atlas,
        "local_handoffs": handoffs,
        "passed_or_handed_off": all(handoff["passed"] for handoff in handoffs),
    }


def _as_fraction(value: object) -> Fraction:
    return Fraction(str(value))


def _compact_handoff(
    scalar_sign: int,
    box: dict[str, object],
) -> dict[str, object]:
    """Prove a compact failure box lies in a pair of local toric charts."""

    t_lower = _as_fraction(box["t_lower"])
    a_lower = _as_fraction(box["a_lower"])
    a_upper = _as_fraction(box["a_upper"])
    z_lower = _as_fraction(box["z_lower"])
    z_upper = _as_fraction(box["z_upper"])
    if t_lower <= 0 or a_upper >= 1:
        raise AssertionError("handoff box meets a forbidden coordinate pole")
    h_upper = (1 - t_lower) / ((1 - a_upper) * t_lower)

    if scalar_sign == 1:
        family = "A"

        def direction(a_value: Fraction) -> Fraction:
            return abs(2 * a_value - 1) / (1 - a_value)

        q_upper = max(direction(a_lower), direction(a_upper)) + 3 * z_upper
        direction_lower = 2 * a_lower - 1
        direction_upper = 2 * a_upper - 1
    elif scalar_sign == -1:
        family = "B"

        def direction(a_value: Fraction) -> Fraction:
            return abs(2 - 3 * a_value) / (1 - a_value)

        q_upper = max(direction(a_lower), direction(a_upper)) + 1 - z_lower
        direction_lower = 2 - 3 * a_upper
        direction_upper = 2 - 3 * a_lower
    else:
        raise ValueError("scalar_sign must be -1 or 1")

    if direction_upper <= 0:
        d_signs = [-1]
    elif direction_lower >= 0:
        d_signs = [1]
    else:
        d_signs = [-1, 1]
    passed = h_upper <= 1 and q_upper <= 1
    return {
        "family": family,
        "compact_box": box,
        "exact_h_upper": str(h_upper),
        "exact_q_upper": str(q_upper),
        "d_signs": d_signs,
        "cover": ["core (R<=1)", "r_infinity_q_low (R>=1)"],
        "cover_logic": "h<=1 and q=h*R<=1; split exhaustively at R=1",
        "passed": passed,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_atlas_partition(
    atlas: dict[str, object],
    *,
    allow_unresolved: bool,
) -> None:
    """Validate positivity, prefix-freeness, and exhaustive binary coverage."""

    leaves = list(atlas["leaves"])
    unresolved = list(atlas.get("unresolved", []))
    if unresolved and not allow_unresolved:
        raise AssertionError("a local positivity atlas retained unresolved cells")
    if len(leaves) != atlas["leaf_count"]:
        raise AssertionError("leaf count does not match the stored atlas")
    if len(unresolved) != atlas.get("unresolved_count", 0):
        raise AssertionError("unresolved count does not match the stored atlas")
    if len(leaves) + len(unresolved) != atlas["split_count"] + 1:
        raise AssertionError("atlas terminals do not form a full binary tree")
    if any(int(leaf["minimum_scaled_coefficient"]) <= 0 for leaf in leaves):
        raise AssertionError("a claimed positive atlas leaf is not positive")
    if any(
        int(cell["minimum_scaled_coefficient"]) > 0 for cell in unresolved
    ):
        raise AssertionError("a positive cell was incorrectly handed off")

    terminals: set[tuple[str, ...]] = set()
    for terminal in [*leaves, *unresolved]:
        path = terminal["path"]
        steps = () if path == "root" else tuple(path.split(","))
        if len(steps) != terminal["depth"]:
            raise AssertionError("atlas path and depth disagree")
        for depth, step in enumerate(steps):
            if step not in {f"{('t', 'a', 'z')[depth % 3]}0", f"{('t', 'a', 'z')[depth % 3]}1"}:
                raise AssertionError("atlas path violates the cyclic split rule")
        if steps in terminals:
            raise AssertionError("duplicate terminal path")
        terminals.add(steps)
    for terminal in terminals:
        if any(terminal[:depth] in terminals for depth in range(len(terminal))):
            raise AssertionError("atlas terminal paths are not prefix-free")

    def covers(prefix: tuple[str, ...]) -> None:
        if prefix in terminals:
            return
        symbol = ("t", "a", "z")[len(prefix) % 3]
        for side in (0, 1):
            child = (*prefix, f"{symbol}{side}")
            if not any(
                terminal[: len(child)] == child for terminal in terminals
            ):
                raise AssertionError("atlas binary cover has a missing child")
            covers(child)

    covers(())


def assemble_global_certificate(
    report_directory: Path,
    *,
    coefficient_artifact: Path = DEFAULT_COEFFICIENT_ARTIFACT,
) -> dict[str, object]:
    """Assemble the compact and local reports into one global exact cover."""

    compact_reports: dict[str, object] = {}
    required_local: set[tuple[str, str, int]] = set()
    source_hashes: dict[str, str] = {}
    for scalar_sign in (1, -1):
        path = report_directory / f"compact_{scalar_sign}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["scalar_sign"] != scalar_sign:
            raise AssertionError("compact report has the wrong scalar sign")
        _validate_atlas_partition(report["atlas"], allow_unresolved=True)
        handoffs = [
            _compact_handoff(scalar_sign, cell["box"])
            for cell in report["atlas"]["unresolved"]
        ]
        if not all(handoff["passed"] for handoff in handoffs):
            raise AssertionError("a compact failure box escaped the local cover")
        for handoff in handoffs:
            for d_sign in handoff["d_signs"]:
                for chart in LOCAL_HANDOFF_CHARTS:
                    required_local.add((handoff["family"], chart, d_sign))
        report["local_handoffs"] = handoffs
        report["passed_or_handed_off"] = True
        compact_reports[str(scalar_sign)] = report
        source_hashes[path.name] = _file_sha256(path)

    local_reports: dict[str, object] = {}
    family_digests: dict[str, str] = {}
    for family, chart, d_sign in sorted(required_local):
        path = report_directory / f"{family}_{chart}_{d_sign}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if (report["family"], report["chart"], report["d_sign"]) != (
            family,
            chart,
            d_sign,
        ):
            raise AssertionError("local report metadata does not match its role")
        if not report["passed"]:
            raise AssertionError("a required local chart did not pass")
        _validate_atlas_partition(report["atlas"], allow_unresolved=False)
        old_digest = family_digests.setdefault(family, report["gap_rows_sha256"])
        if report["gap_rows_sha256"] != old_digest:
            raise AssertionError("local reports disagree on the family gap")
        key = f"{family}:{chart}:{d_sign}"
        local_reports[key] = report
        source_hashes[path.name] = _file_sha256(path)

    for family, expected_digest in family_digests.items():
        _, _, gap = chart_gap_coefficients(family, coefficient_artifact)
        if _sparse_digest(gap) != expected_digest:
            raise AssertionError("local report digest does not replay from coefficients")

    passed = (
        len(compact_reports) == 2
        and len(local_reports) == len(required_local)
        and all(
            report["passed_or_handed_off"]
            for report in compact_reports.values()
        )
        and all(report["passed"] for report in local_reports.values())
    )
    return {
        "schema_version": 1,
        "claim_scope": (
            "exact global 21/20 bound for the reconstructed coupled V1+V5 "
            "invariant rational function"
        ),
        "theorem": "N(x,p,y)/delta(x,p,y)^14 <= 21/20",
        "orbit_domain": "x real, p>=0, 27*y^2<=2*p^3",
        "coefficient_artifact": coefficient_artifact.name,
        "coefficient_artifact_sha256": _file_sha256(coefficient_artifact),
        "cover_argument": {
            "compact_chambers": "x>=0 and x<=0",
            "ordinary_cells": "strict exact Bernstein positivity",
            "rank_loss_handoff": "exact h<=1 and q=h*R<=1 bounds",
            "local_partition": "R<=1 core or R>=1 low-q toric chart",
        },
        "source_report_sha256": dict(sorted(source_hashes.items())),
        "compact_reports": compact_reports,
        "required_local_charts": sorted(
            f"{family}:{chart}:{d_sign}"
            for family, chart, d_sign in required_local
        ),
        "local_reports": local_reports,
        "reconstructed_rational_function_bound_certified": passed,
        "raw_characteristic_zero_determinant_identity_certified": False,
        "global_determinant_theorem_claimed": False,
        "passed": passed,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("A", "B"))
    parser.add_argument("--chart", choices=CHARTS)
    parser.add_argument("--d-sign", type=int, choices=(-1, 1))
    parser.add_argument("--compact-sign", type=int, choices=(-1, 1))
    parser.add_argument("--assemble-report-dir", type=Path)
    parser.add_argument(
        "--coefficient-artifact",
        type=Path,
        default=DEFAULT_COEFFICIENT_ARTIFACT,
    )
    parser.add_argument("--maximum-depth", type=int, default=18)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    if arguments.assemble_report_dir is not None:
        if any(
            value is not None
            for value in (
                arguments.family,
                arguments.chart,
                arguments.d_sign,
                arguments.compact_sign,
            )
        ):
            raise SystemExit(
                "--assemble-report-dir cannot be combined with chart arguments"
            )
        report = assemble_global_certificate(
            arguments.assemble_report_dir,
            coefficient_artifact=arguments.coefficient_artifact,
        )
        passed = bool(report["passed"])
    elif arguments.compact_sign is not None:
        if any(
            value is not None
            for value in (arguments.family, arguments.chart, arguments.d_sign)
        ):
            raise SystemExit("--compact-sign cannot be combined with local-chart arguments")
        report = compact_chart_certificate(
            arguments.compact_sign,
            coefficient_artifact=arguments.coefficient_artifact,
            maximum_depth=arguments.maximum_depth,
        )
        passed = bool(report["passed_or_handed_off"])
    else:
        if any(
            value is None
            for value in (arguments.family, arguments.chart, arguments.d_sign)
        ):
            raise SystemExit("--family, --chart, and --d-sign are required together")
        report = chart_certificate(
            arguments.family,
            arguments.chart,
            arguments.d_sign,
            coefficient_artifact=arguments.coefficient_artifact,
            maximum_depth=arguments.maximum_depth,
        )
        passed = bool(report["passed"])
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
