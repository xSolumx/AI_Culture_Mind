"""Exact native-Bernstein audit for the adjacent-endpoint Schur cubic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flint import ctx

from spin8_dirac_endpoint_octet import H0, TRIVIAL
from spin8_dirac_endpoint_octet_quadratic import (
    _atomic_json,
    _build_z_coefficients,
    _native_bernstein_audit,
    _sha256,
)
from spin8_resource_limits import constrain_current_process

DEFAULT_COEFFICIENT_DIR = Path(
    "artifacts/spin8_dirac_unrestricted_coefficients_20260807"
)
DEFAULT_ENDPOINT_ARTIFACT = Path("artifacts/spin8_dirac_endpoint_octet_20260807.json")


def _build_cubic(coefficient_dir: Path):
    z, forced_squares, variables = _build_z_coefficients(coefficient_dir)
    ud, ue, ug, ui, y = variables
    nontrivial = H0[1:]
    tau = (
        ud
        * ue
        * ug
        * ui
        * y**2
        * (1 - ud)
        * (1 - ue)
        * (1 - ug)
        * (1 - ui)
    )
    forced_product = forced_squares[nontrivial[0]]
    for mask in nontrivial[1:]:
        forced_product *= forced_squares[mask]
    if forced_product != tau**2:
        raise AssertionError("the three forced radicals do not cancel to tau")

    cubic = z[TRIVIAL] ** 3
    cubic -= z[TRIVIAL] * sum(
        (forced_squares[mask] * z[mask] ** 2 for mask in nontrivial),
        start=z[TRIVIAL].context().constant(0),
    )
    cubic += 2 * tau * z[nontrivial[0]] * z[nontrivial[1]] * z[nontrivial[2]]
    return cubic, tau, forced_product, variables


def run(
    coefficient_dir: Path,
    *,
    output: Path,
    flint_threads: int = 6,
    endpoint_artifact: Path = DEFAULT_ENDPOINT_ARTIFACT,
) -> dict[str, object]:
    if not 1 <= flint_threads <= 6:
        raise ValueError("FLINT thread count must be between one and six")
    resource = constrain_current_process(workers=flint_threads)
    ctx.threads = flint_threads
    cubic, tau, forced_product, _variables = _build_cubic(coefficient_dir)
    audit = _native_bernstein_audit(cubic, sample_limit=256)
    with endpoint_artifact.open("r", encoding="utf-8") as handle:
        endpoint_report = json.load(handle)
    x_face_passed = bool(
        endpoint_report["passed"]
        and endpoint_report["x_block"]["passed"]
        and endpoint_report["x_block"]["y_one_klein_certificate_passed"]
    )
    if not x_face_passed:
        raise AssertionError("the delegated y=1 X-block theorem did not pass")
    y = cubic.context().gens()[4]
    y_degree = int(cubic.degrees()[4])
    y_one_face = cubic.subs({"y": 1})
    selector = y**y_degree
    remainder = cubic - y_one_face * selector
    selector_identity = cubic == y_one_face * selector + remainder
    remainder_audit = _native_bernstein_audit(remainder, sample_limit=256)
    selector_passed = bool(
        selector_identity
        and x_face_passed
        and remainder_audit["negative_scaled_coefficient_count"] == 0
    )
    report = {
        "experiment": "adjacent endpoint octet cubic native-Bernstein audit",
        "domain": "(ud,ue,ug,ui,y) in [0,1]^5",
        "cubic_formula": (
            "Z0^3-Z0*(s1*Z1^2+s2*Z2^2+s3*Z3^2)+2*tau*Z1*Z2*Z3"
        ),
        "tau": (
            "ud*ue*ug*ui*y^2*(1-ud)*(1-ue)*(1-ug)*(1-ui)"
        ),
        "forced_radical_identity": {
            "identity": "s1*s2*s3=tau^2",
            "verified_exactly": forced_product == tau**2,
        },
        "power_term_count": len(cubic.to_dict()),
        "multidegree": list(map(int, cubic.degrees())),
        "native_bernstein": audit,
        "native_certificate_passed": (
            audit["negative_scaled_coefficient_count"] == 0
        ),
        "y_one_selector_certificate": {
            "identity": "cubic=cubic|y=1*y^36+remainder",
            "identity_verified_exactly": selector_identity,
            "y_degree": y_degree,
            "face_implication": "y=1 implies Z=X^2; X positive semidefinite",
            "delegated_endpoint_artifact": f"artifacts/{endpoint_artifact.name}",
            "delegated_endpoint_sha256": _sha256(endpoint_artifact),
            "delegated_x_face_passed": x_face_passed,
            "face_power_term_count": len(y_one_face.to_dict()),
            "remainder_power_term_count": len(remainder.to_dict()),
            "remainder_native_bernstein": remainder_audit,
            "passed": selector_passed,
        },
        "passed": (
            audit["negative_scaled_coefficient_count"] == 0 or selector_passed
        ),
        "scope_boundary": (
            "A native pass proves only the cubic principal-minor family on the "
            "adjacent endpoint octet. A native failure is inconclusive. The "
            "determinant and unrestricted Dirac--Gram theorem remain open."
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
    parser.add_argument("--flint-threads", type=int, default=6)
    parser.add_argument(
        "--endpoint-artifact", type=Path, default=DEFAULT_ENDPOINT_ARTIFACT
    )
    arguments = parser.parse_args()
    report = run(
        arguments.coefficient_dir,
        output=arguments.output,
        flint_threads=arguments.flint_threads,
        endpoint_artifact=arguments.endpoint_artifact,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=int))


if __name__ == "__main__":
    main()
