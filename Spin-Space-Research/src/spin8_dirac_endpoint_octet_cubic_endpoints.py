"""Exact two-endpoint selector audit for the adjacent-octet Schur cubic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flint import ctx

from spin8_dirac_endpoint_octet_cubic import (
    DEFAULT_COEFFICIENT_DIR,
    _build_cubic,
)
from spin8_dirac_endpoint_octet_quadratic import (
    _atomic_json,
    _native_bernstein_audit,
    _sha256,
)
from spin8_resource_limits import constrain_current_process

DEFAULT_Y_ZERO_CERTIFICATE = Path(
    "artifacts/spin8_dirac_endpoint_octet_cubic_yzero_20260808.json"
)


def run(
    coefficient_dir: Path,
    *,
    output: Path,
    y_zero_certificate: Path = DEFAULT_Y_ZERO_CERTIFICATE,
    flint_threads: int = 6,
) -> dict[str, object]:
    if not 1 <= flint_threads <= 6:
        raise ValueError("FLINT thread count must be between one and six")
    resource = constrain_current_process(workers=flint_threads)
    ctx.threads = flint_threads
    cubic, _tau, _forced_product, _variables = _build_cubic(coefficient_dir)
    y = cubic.context().gens()[4]
    y_degree = int(cubic.degrees()[4])
    y_zero_face = cubic.subs({"y": 0})
    y_one_face = cubic.subs({"y": 1})
    y_zero_audit = _native_bernstein_audit(y_zero_face, sample_limit=128)
    y_zero_native_passed = y_zero_audit["negative_scaled_coefficient_count"] == 0
    delegated_report = None
    delegated_passed = False
    delegated_sha256 = None
    if y_zero_certificate.exists():
        delegated_report = json.loads(y_zero_certificate.read_text(encoding="utf-8"))
        delegated_sha256 = _sha256(y_zero_certificate)
        delegated_passed = bool(delegated_report.get("passed"))
    y_zero_passed = bool(y_zero_native_passed or delegated_passed)

    dual_remainder = None
    dual_audit = None
    dual_identity = None
    endpoint_factor_certificate = None
    if y_zero_passed:
        dual_remainder = (
            cubic
            - y_zero_face * (1 - y) ** y_degree
            - y_one_face * y**y_degree
        )
        dual_identity = cubic == (
            y_zero_face * (1 - y) ** y_degree
            + y_one_face * y**y_degree
            + dual_remainder
        )
        dual_audit = _native_bernstein_audit(dual_remainder, sample_limit=256)
        if dual_audit["negative_scaled_coefficient_count"]:
            endpoint_factor = y * (1 - y)
            quotient, division_remainder = divmod(dual_remainder, endpoint_factor)
            factor_identity = bool(
                division_remainder == 0
                and dual_remainder == endpoint_factor * quotient
            )
            quotient_audit = (
                _native_bernstein_audit(quotient, sample_limit=1024)
                if factor_identity
                else None
            )
            endpoint_factor_certificate = {
                "factor": "y*(1-y)",
                "zero_division_remainder": division_remainder == 0,
                "identity_verified_exactly": factor_identity,
                "quotient_power_term_count": (
                    len(quotient.to_dict()) if factor_identity else None
                ),
                "quotient_native_bernstein": quotient_audit,
                "passed": bool(
                    factor_identity
                    and quotient_audit is not None
                    and quotient_audit["negative_scaled_coefficient_count"] == 0
                ),
            }

    dual_passed = bool(
        y_zero_passed
        and dual_identity
        and dual_audit is not None
        and (
            dual_audit["negative_scaled_coefficient_count"] == 0
            or (
                endpoint_factor_certificate is not None
                and endpoint_factor_certificate["passed"]
            )
        )
    )
    report = {
        "experiment": "adjacent endpoint octet cubic two-endpoint selector",
        "domain": "(ud,ue,ug,ui,y) in [0,1]^5",
        "y_degree": y_degree,
        "y_zero_face": {
            "power_term_count": len(y_zero_face.to_dict()),
            "native_bernstein": y_zero_audit,
            "native_passed": y_zero_native_passed,
            "delegated_certificate": {
                "path": y_zero_certificate.as_posix(),
                "sha256": delegated_sha256,
                "reported_passed": delegated_passed,
                "acceptance_note": (
                    "The final theorem verifier must replay this source; this "
                    "stage records its exact provenance."
                ),
            },
            "passed": y_zero_passed,
        },
        "dual_selector": {
            "identity": (
                "cubic=cubic|y=0*(1-y)^36+cubic|y=1*y^36+remainder"
            ),
            "identity_verified_exactly": dual_identity,
            "remainder_power_term_count": (
                len(dual_remainder.to_dict()) if dual_remainder is not None else None
            ),
            "remainder_native_bernstein": dual_audit,
            "endpoint_factor_certificate": endpoint_factor_certificate,
            "passed": dual_passed,
        },
        "passed": dual_passed,
        "scope_boundary": (
            "A pass proves the cubic principal-minor family only. A failed "
            "endpoint face or remainder rejects this selector, not the cubic."
        ),
        "resource_contract": resource,
    }
    _atomic_json(output, report)
    report["artifact_sha256"] = _sha256(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coefficient-dir", type=Path, default=DEFAULT_COEFFICIENT_DIR
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--y-zero-certificate", type=Path, default=DEFAULT_Y_ZERO_CERTIFICATE
    )
    parser.add_argument("--flint-threads", type=int, default=6)
    arguments = parser.parse_args()
    report = run(
        arguments.coefficient_dir,
        output=arguments.output,
        y_zero_certificate=arguments.y_zero_certificate,
        flint_threads=arguments.flint_threads,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=int))


if __name__ == "__main__":
    main()
