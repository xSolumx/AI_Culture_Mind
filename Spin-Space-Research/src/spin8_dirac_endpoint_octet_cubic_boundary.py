"""Exact selected-face audit for unresolved endpoint-octet cubic blow-ups."""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import reduce
from operator import mul
from pathlib import Path

from flint import ctx

from spin8_dirac_endpoint_octet_cubic import DEFAULT_COEFFICIENT_DIR, _build_cubic
from spin8_dirac_endpoint_octet_cubic_blowup import (
    _batched_bernstein_audit,
    _blowup_chart,
    _tangent_face_in_chart,
)
from spin8_dirac_endpoint_octet_cubic_tangent import _homogeneous_taylor
from spin8_dirac_endpoint_octet_quadratic import _atomic_json, _sha256
from spin8_resource_limits import constrain_current_process


def _coefficient_sha256(polynomial) -> str:
    digest = hashlib.sha256()
    for powers, coefficient in sorted(polynomial.to_dict().items()):
        digest.update(",".join(map(str, powers)).encode("ascii"))
        digest.update(b":")
        digest.update(str(coefficient).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _compact_factor_report(polynomial) -> tuple[dict[str, object], list[tuple]]:
    content, factors = polynomial.factor()
    rebuilt = reduce(
        mul,
        (factor ** int(exponent) for factor, exponent in factors),
        polynomial.context().constant(int(content)),
    )
    return (
        {
            "content": str(content),
            "factor_count": len(factors),
            "factors": [
                {
                    "exponent": int(exponent),
                    "power_term_count": len(factor.to_dict()),
                    "multidegree": list(map(int, factor.degrees())),
                    "coefficient_sha256": _coefficient_sha256(factor),
                }
                for factor, exponent in factors
            ],
            "identity_verified_exactly": rebuilt == polynomial,
        },
        factors,
    )


def _face_report(polynomial) -> dict[str, object]:
    factor_report, factors = _compact_factor_report(polynomial)
    native = _batched_bernstein_audit(polynomial, sample_limit=128)
    factor_audits = []
    certified_product_sign = 1 if int(factor_report["content"]) > 0 else -1
    every_odd_factor_signed = True
    for factor, exponent in factors:
        audit = None
        certified_sign = None
        if int(exponent) % 2:
            audit = _batched_bernstein_audit(factor, sample_limit=64)
            positive_count = (
                audit["coefficient_count"]
                - audit["negative_scaled_coefficient_count"]
                - audit["zero_scaled_coefficient_count"]
            )
            audit["positive_scaled_coefficient_count"] = positive_count
            if audit["negative_scaled_coefficient_count"] == 0:
                certified_sign = "nonnegative"
                certified_product_sign *= 1
            elif positive_count == 0:
                certified_sign = "nonpositive"
                certified_product_sign *= -1
            else:
                every_odd_factor_signed = False
        factor_audits.append(
            {
                "exponent": int(exponent),
                "odd_exponent_requires_sign": bool(int(exponent) % 2),
                "certified_sign": certified_sign,
                "native_bernstein": audit,
            }
        )
    factor_route_passed = bool(
        factor_report["identity_verified_exactly"]
        and every_odd_factor_signed
        and certified_product_sign > 0
    )
    return {
        "power_term_count": len(polynomial.to_dict()),
        "multidegree": list(map(int, polynomial.degrees())),
        "coefficient_sha256": _coefficient_sha256(polynomial),
        "native_bernstein": native,
        "factorization": factor_report,
        "factor_native_audits": factor_audits,
        "certified_factor_product_sign": certified_product_sign,
        "factor_route_passed": factor_route_passed,
        "passed": bool(
            native["negative_scaled_coefficient_count"] == 0 or factor_route_passed
        ),
    }


def run(
    coefficient_dir: Path,
    *,
    pivot: int,
    output: Path,
    flint_threads: int = 6,
) -> dict[str, object]:
    if pivot not in (0, 1, 2):
        raise ValueError("selected-face audit applies only to pivots 0, 1, and 2")
    resource = constrain_current_process(workers=flint_threads)
    ctx.threads = flint_threads
    cubic, _tau, _forced_product, _variables = _build_cubic(coefficient_dir)
    tangent, tangent_order = _homogeneous_taylor(cubic, max_order=12)
    quotient, order, nonpivots, _shifted_term_count = _blowup_chart(
        cubic, pivot=pivot, expected_order=tangent_order
    )
    exceptional = quotient.subs({"radius": 0})
    expected = _tangent_face_in_chart(
        tangent, nonpivots=nonpivots, context=quotient.context()
    )
    if exceptional // int(exceptional.content()) != expected // int(expected.content()):
        raise AssertionError("exceptional tangent face mismatch")
    radius = quotient.context().gens()[0]
    radial_degree = int(quotient.degrees()[0])
    first_remainder = quotient - exceptional * (1 - radius) ** radial_degree

    ui_axis = 1 + nonpivots.index(3)
    ui = quotient.context().gens()[ui_axis]
    ui_degree = int(quotient.degrees()[ui_axis])
    ui_face = first_remainder.subs({str(ui): 0})
    ui_complement = first_remainder - ui_face * (1 - ui) ** ui_degree
    ui_identity = first_remainder == ui_face * (1 - ui) ** ui_degree + ui_complement
    ui_face_report = _face_report(ui_face)

    double_face_report = None
    if pivot == 0:
        ue_axis = 1 + nonpivots.index(1)
        ug_axis = 1 + nonpivots.index(2)
        ue = quotient.context().gens()[ue_axis]
        ug = quotient.context().gens()[ug_axis]
        ue_degree = int(quotient.degrees()[ue_axis])
        ug_degree = int(quotient.degrees()[ug_axis])
        double_face = ui_complement.subs({str(ue): 0, str(ug): 0})
        final_complement = ui_complement - (
            double_face * (1 - ue) ** ue_degree * (1 - ug) ** ug_degree
        )
        double_identity = ui_complement == (
            double_face * (1 - ue) ** ue_degree * (1 - ug) ** ug_degree
            + final_complement
        )
        double_face_report = {
            "identity_verified_exactly": double_identity,
            "face": _face_report(double_face),
        }

    report = {
        "experiment": "adjacent endpoint octet cubic boundary-support selectors",
        "pivot_index": pivot,
        "pivot_deviation": ("ud", "ue", "ug")[pivot],
        "exact_radius_order": order,
        "ui_zero_selector": {
            "selector": f"(1-ui)^{ui_degree}",
            "identity_verified_exactly": ui_identity,
            "face": ui_face_report,
        },
        "ue_ug_zero_selector": double_face_report,
        "passed": bool(
            ui_identity
            and ui_face_report["passed"]
            and (
                double_face_report is None
                or (
                    double_face_report["identity_verified_exactly"]
                    and double_face_report["face"]["passed"]
                )
            )
        ),
        "scope_boundary": (
            "Selected faces only. Complementary Bernstein layers are certified "
            "by the complete support audit in the corresponding blow-up artifact."
        ),
        "resource_contract": resource,
    }
    _atomic_json(output, report)
    report["artifact_sha256"] = _sha256(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coefficient-dir", type=Path, default=DEFAULT_COEFFICIENT_DIR)
    parser.add_argument("--pivot", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flint-threads", type=int, default=6)
    arguments = parser.parse_args()
    report = run(
        arguments.coefficient_dir,
        pivot=arguments.pivot,
        output=arguments.output,
        flint_threads=arguments.flint_threads,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
