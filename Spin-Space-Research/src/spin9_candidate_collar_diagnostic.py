"""Exact partial-atlas diagnostic just above the Spin(9) candidate ratio.

This module deliberately does not claim the candidate theorem.  It replaces
the coarse ``21/20`` target by ``26201/25000``, a rational number only about
``1.08e-7`` above the algebraic symmetric-candidate ratio, and records every
Bernstein cell that remains unresolved at a shallow deterministic depth.
Those boxes are inputs to local algebraic charts, not counterexamples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import sympy as sp
from spin9_v1_candidate_line import _quadratic_field_sign
from spin9_v1_v5_gap import (
    DEFAULT_COEFFICIENT_ARTIFACT,
    bernstein_atlas_partial,
    compact_bernstein_controls,
)
from spin9_v1_v5_reconstruction import ROOT, load_coefficients

DEFAULT_OUTPUT = ROOT / "artifacts" / "spin9_candidate_collar_20260821.json"
DEFAULT_MAXIMUM_DEPTH = 6
UPPER_NUMERATOR = 26_201
UPPER_DENOMINATOR = 25_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_ratio() -> sp.Expr:
    q = sp.sqrt(241)
    return sp.factor(
        (-41 + q) ** 10
        * (-5 + q) ** 3
        * (q + 31) ** 5
        / sp.Integer(27_916_588_318_337_525_511_880_704)
    )


def _chart_summary(
    coefficients: tuple,
    scalar_sign: int,
    maximum_depth: int,
) -> dict[str, object]:
    controls, scale, term_count = compact_bernstein_controls(
        coefficients,
        scalar_sign=scalar_sign,
        upper_numerator=UPPER_NUMERATOR,
        upper_denominator=UPPER_DENOMINATOR,
    )
    atlas = bernstein_atlas_partial(controls, maximum_depth=maximum_depth)
    return {
        "scalar_sign": scalar_sign,
        "bernstein_scale": str(scale),
        "gap_invariant_term_count": term_count,
        "atlas": atlas,
        "strict_global_bound_certified": not atlas["unresolved"],
    }


def certificate(maximum_depth: int = DEFAULT_MAXIMUM_DEPTH) -> dict[str, object]:
    """Build the exact rational-collar localization report."""

    if maximum_depth < 0:
        raise ValueError("maximum_depth must be nonnegative")
    coefficients = load_coefficients(DEFAULT_COEFFICIENT_ARTIFACT)
    candidate = _candidate_ratio()
    collar = sp.Rational(UPPER_NUMERATOR, UPPER_DENOMINATOR)
    margin = sp.collect(sp.expand(collar - candidate), sp.sqrt(241))
    charts = [
        _chart_summary(coefficients, scalar_sign, maximum_depth)
        for scalar_sign in (1, -1)
    ]
    unresolved_count = sum(
        int(chart["atlas"]["unresolved_count"]) for chart in charts
    )
    return {
        "schema_version": 1,
        "claim_scope": (
            "exact shallow localization for a rational collar above the "
            "Spin(9) algebraic candidate; no candidate bound is claimed"
        ),
        "coefficient_artifact": DEFAULT_COEFFICIENT_ARTIFACT.name,
        "coefficient_artifact_sha256": _sha256(DEFAULT_COEFFICIENT_ARTIFACT),
        "upper_bound": f"{UPPER_NUMERATOR}/{UPPER_DENOMINATOR}",
        "candidate_ratio": str(candidate),
        "candidate_ratio_decimal": str(sp.N(candidate, 30)),
        "exact_positive_collar_margin": str(margin),
        "collar_strictly_above_candidate": _quadratic_field_sign(margin) > 0,
        "maximum_depth": maximum_depth,
        "charts": charts,
        "unresolved_count": unresolved_count,
        "interpretation": (
            "unresolved cells are local-chart handoffs, not counterexamples"
        ),
        "mixed_candidate_optimality_certified": False,
        "second_v5_copy_certified": False,
        "unrestricted_quotient_certified": False,
        "diagnostic_completed": bool(unresolved_count > 0),
        "passed": bool(unresolved_count > 0 and _quadratic_field_sign(margin) > 0),
    }


def verify_report(report: dict[str, object]) -> bool:
    """Verify the stored report's claim boundary and source integrity."""

    charts = report.get("charts", [])
    return bool(
        report.get("passed") is True
        and report.get("collar_strictly_above_candidate") is True
        and report.get("mixed_candidate_optimality_certified") is False
        and report.get("second_v5_copy_certified") is False
        and report.get("unrestricted_quotient_certified") is False
        and report.get("coefficient_artifact_sha256")
        == _sha256(DEFAULT_COEFFICIENT_ARTIFACT)
        and len(charts) == 2
        and {chart.get("scalar_sign") for chart in charts} == {-1, 1}
        and all(int(chart["atlas"]["unresolved_count"]) > 0 for chart in charts)
        and int(report.get("unresolved_count", 0))
        == sum(int(chart["atlas"]["unresolved_count"]) for chart in charts)
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
