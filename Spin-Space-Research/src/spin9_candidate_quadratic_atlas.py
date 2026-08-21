"""Exact quadratic-field compact-atlas diagnostic at the Spin(9) candidate.

The candidate determinant ratio lies in ``Q(sqrt(241))``.  This module lifts
the existing rational compact Bernstein tensor to that ordered quadratic
field without floating-point signs.  A 180-digit rational enclosure of the
positive square root is applied coefficientwise, producing a rational
polynomial below the exact candidate gap.  Strict Bernstein leaves therefore
certify the exact algebraic target; unresolved leaves remain handoffs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from spin9_candidate_explicit_collar import (
    SQRT_BOUND_DIGITS,
    _candidate_parts,
    _sqrt_bounds,
)
from spin9_v1_v5_gap import bernstein_atlas_partial, compact_bernstein_controls
from spin9_v1_v5_reconstruction import ROOT, load_coefficients

DEFAULT_OUTPUT = ROOT / "artifacts" / "spin9_candidate_quadratic_atlas_20260821.json"
DEFAULT_MAXIMUM_DEPTH = 6


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_lower_controls(
    coefficients: tuple,
    *,
    scalar_sign: int,
) -> tuple[np.ndarray, int, dict[str, object]]:
    """Return exact rational controls lying below the candidate gap.

    If ``C1`` and ``C2`` are the common-scaled controls for
    ``21*Delta-20*N`` and ``2*Delta-N``, then

    ``Delta = (20*C2-C1)/19`` and ``N=2*Delta-C2``.

    This reconstructs both coefficients of
    ``(A+B*sqrt(241))*Delta-D*N`` without rebuilding the compactification.
    """

    c_21_20, scale_21_20, terms_21_20 = compact_bernstein_controls(
        coefficients,
        scalar_sign=scalar_sign,
        upper_numerator=21,
        upper_denominator=20,
    )
    c_2_1, scale_2_1, terms_2_1 = compact_bernstein_controls(
        coefficients,
        scalar_sign=scalar_sign,
        upper_numerator=2,
        upper_denominator=1,
    )
    if scale_21_20 != scale_2_1:
        raise AssertionError("rational basis tensors have different scales")

    rational_part, radical_part, denominator = _candidate_parts()
    sqrt_lower, sqrt_upper = _sqrt_bounds()
    root_scale = math.lcm(sqrt_lower.denominator, sqrt_upper.denominator)
    root_lower = sqrt_lower.numerator * (root_scale // sqrt_lower.denominator)
    root_upper = sqrt_upper.numerator * (root_scale // sqrt_upper.denominator)

    result = np.empty(c_21_20.shape, dtype=object)
    exact_divisions = True
    radical_negative_count = 0
    for index, (first, second) in enumerate(
        zip(c_21_20.flat, c_2_1.flat, strict=True)
    ):
        delta_numerator = 20 * second - first
        if delta_numerator % 19:
            exact_divisions = False
            raise AssertionError("compact Delta reconstruction lost integrality")
        delta_control = delta_numerator // 19
        rational_control = (
            (rational_part - 2 * denominator) * delta_control
            + denominator * second
        )
        radical_control = radical_part * delta_control
        if radical_control >= 0:
            root_numerator = root_lower
        else:
            root_numerator = root_upper
            radical_negative_count += 1
        result.flat[index] = (
            root_scale * rational_control + root_numerator * radical_control
        )

    del c_21_20, c_2_1
    gc.collect()
    return result, scale_21_20 * denominator * root_scale, {
        "rational_basis_term_counts": [terms_21_20, terms_2_1],
        "delta_reconstruction_exactly_divisible_by_19": exact_divisions,
        "radical_negative_native_control_count": radical_negative_count,
        "sqrt_241_bound_digits": SQRT_BOUND_DIGITS,
        "sqrt_241_bounds_verified": bool(
            sqrt_lower * sqrt_lower < 241 < sqrt_upper * sqrt_upper
        ),
        "coefficientwise_lowering": (
            "a+b*sqrt(241) is lowered with the rational lower root bound "
            "for b>=0 and the upper root bound for b<0"
        ),
    }


def certificate(maximum_depth: int = DEFAULT_MAXIMUM_DEPTH) -> dict[str, object]:
    """Build the exact ordered-quadratic compact localization report."""

    if maximum_depth < 0:
        raise ValueError("maximum_depth must be nonnegative")
    from spin9_v1_v5_gap import DEFAULT_COEFFICIENT_ARTIFACT

    coefficients = load_coefficients(DEFAULT_COEFFICIENT_ARTIFACT)
    charts = []
    common_scale = None
    for scalar_sign in (1, -1):
        controls, scale, construction = _candidate_lower_controls(
            coefficients,
            scalar_sign=scalar_sign,
        )
        common_scale = scale if common_scale is None else common_scale
        if scale != common_scale:
            raise AssertionError("sign charts have different exact scales")
        atlas = bernstein_atlas_partial(controls, maximum_depth=maximum_depth)
        charts.append(
            {
                "scalar_sign": scalar_sign,
                "construction": construction,
                "atlas": atlas,
                "strict_region_certified_at_exact_candidate": bool(
                    atlas["leaf_count"] > 0
                ),
            }
        )
        del controls
        gc.collect()

    unresolved_count = sum(
        int(chart["atlas"]["unresolved_count"]) for chart in charts
    )
    strict_leaf_count = sum(int(chart["atlas"]["leaf_count"]) for chart in charts)
    construction_passed = bool(
        unresolved_count > 0
        and all(
            chart["construction"]["sqrt_241_bounds_verified"] for chart in charts
        )
    )
    return {
        "schema_version": 1,
        "claim_scope": (
            "exact Q(sqrt(241)) compact localization at the Spin(9) candidate; "
            "strict leaves are proved and unresolved cells are local handoffs"
        ),
        "coefficient_artifact": DEFAULT_COEFFICIENT_ARTIFACT.name,
        "coefficient_artifact_sha256": _sha256(DEFAULT_COEFFICIENT_ARTIFACT),
        "maximum_depth": maximum_depth,
        "exact_lower_polynomial_common_scale": str(common_scale),
        "charts": charts,
        "strict_leaf_count": strict_leaf_count,
        "unresolved_count": unresolved_count,
        "quadratic_field_compact_atlas_built": construction_passed,
        "strict_region_localized": strict_leaf_count > 0,
        "compact_complement_certified_at_candidate_ratio": False,
        "second_v5_copy_certified": False,
        "unrestricted_quotient_certified": False,
        "passed": construction_passed,
    }


def verify_report(report: dict[str, object]) -> bool:
    charts = report.get("charts", [])
    return bool(
        report.get("passed") is True
        and report.get("quadratic_field_compact_atlas_built") is True
        and report.get("compact_complement_certified_at_candidate_ratio") is False
        and report.get("second_v5_copy_certified") is False
        and report.get("unrestricted_quotient_certified") is False
        and len(charts) == 2
        and {chart.get("scalar_sign") for chart in charts} == {-1, 1}
        and all(
            chart["construction"]["sqrt_241_bounds_verified"] for chart in charts
        )
        and int(report.get("unresolved_count", 0)) > 0
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maximum-depth", type=int, default=DEFAULT_MAXIMUM_DEPTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    report = certificate(arguments.maximum_depth)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
