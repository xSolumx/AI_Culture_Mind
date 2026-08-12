"""Exact reconstruction and endpoint audit for the octet Schur determinant.

This is the fourth and final principal-minor family of the Klein-four
circulant ``Z`` in the adjacent Spin(8) endpoint octet.  The module deliberately
separates exact polynomial reconstruction and endpoint certificates from the
still-independent interior positivity problem.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import sympy as sp
from flint import ctx

from spin8_dirac_endpoint_octet import H0, H1, TRIVIAL
from spin8_dirac_endpoint_octet_quadratic import (
    _atomic_json,
    _build_z_coefficients,
    _native_bernstein_audit,
    _restrict_half_box,
    _sha256,
)
from spin8_resource_limits import constrain_current_process

DEFAULT_COEFFICIENT_DIR = Path(
    "artifacts/spin8_dirac_unrestricted_coefficients_20260807"
)
DEFAULT_ENDPOINT_ARTIFACT = Path("artifacts/spin8_dirac_endpoint_octet_20260807.json")


def _generic_determinant_identity() -> dict[str, bool]:
    """Check the radical-free formula against two independent expressions."""

    z0, a, b, c = sp.symbols("z0 a b c")
    matrix = sp.Matrix(
        (
            (z0, a, b, c),
            (a, z0, c, b),
            (b, c, z0, a),
            (c, b, a, z0),
        )
    )
    formula = (
        z0**4
        - 2 * z0**2 * (a**2 + b**2 + c**2)
        + 8 * z0 * a * b * c
        + a**4
        + b**4
        + c**4
        - 2 * (a**2 * b**2 + a**2 * c**2 + b**2 * c**2)
    )
    walsh_product = sp.prod(
        (
            z0 + a + b + c,
            z0 + a - b - c,
            z0 - a + b - c,
            z0 - a - b + c,
        )
    )
    return {
        "four_by_four_determinant": sp.expand(matrix.det() - formula) == 0,
        "four_walsh_eigenvalues": sp.expand(walsh_product - formula) == 0,
    }


def _build_determinant(coefficient_dir: Path):
    """Reconstruct the radical-free determinant in ``Z`` coefficient form."""

    z, forced_squares, variables = _build_z_coefficients(coefficient_dir)
    ud, ue, ug, ui, y = variables
    nontrivial = H0[1:]
    tau = ud * ue * ug * ui * y**2 * (1 - ud) * (1 - ue) * (1 - ug) * (1 - ui)
    forced_product = forced_squares[nontrivial[0]]
    for mask in nontrivial[1:]:
        forced_product *= forced_squares[mask]
    if forced_product != tau**2:
        raise AssertionError("the three determinant radicals do not cancel")

    squared_modes = [forced_squares[mask] * z[mask] ** 2 for mask in nontrivial]
    # Pairing the four Walsh eigenvalues around the first nontrivial character
    # avoids the much denser generic determinant expansion.
    paired_center = (
        z[TRIVIAL] ** 2 + squared_modes[0] - squared_modes[1] - squared_modes[2]
    )
    determinant = paired_center**2
    determinant -= 4 * z[TRIVIAL] ** 2 * squared_modes[0]
    determinant -= 4 * squared_modes[1] * squared_modes[2]
    determinant += (
        8 * tau * z[TRIVIAL] * z[nontrivial[0]] * z[nontrivial[1]] * z[nontrivial[2]]
    )
    return determinant, tau, forced_product, variables


def _compact_audit(audit: dict[str, object]) -> dict[str, object]:
    """Retain every sign count while bounding artifact sample volume."""

    return {
        key: value
        for key, value in audit.items()
        if key not in {"negative_rows_sample", "negative_rows_sample_limit"}
    }


def _y_zero_atlas(y_zero) -> dict[str, object]:
    """Certify the four-variable low face by a complete dyadic cover."""

    coarse_rows = []
    failed = []
    for bit_tuple in itertools.product("01", repeat=4):
        bits = "".join(bit_tuple)
        restricted = _restrict_half_box(y_zero, bits + "0")
        audit = _native_bernstein_audit(restricted, sample_limit=8)
        passed = audit["negative_scaled_coefficient_count"] == 0
        coarse_rows.append(
            {"bits": bits, "audit": _compact_audit(audit), "passed": passed}
        )
        if not passed:
            failed.append(bits)

    # The unique coarse failure is boundary-supported.  Its complete 16-child
    # subdivision certifies the polynomial itself without interpreting the
    # parent's negative controls as values.
    if failed != ["0001"]:
        raise AssertionError(f"unexpected y=0 coarse failures: {failed}")
    parent = _restrict_half_box(y_zero, "00010")
    child_rows = []
    for bit_tuple in itertools.product("01", repeat=4):
        bits = "".join(bit_tuple)
        restricted = _restrict_half_box(parent, bits + "0")
        audit = _native_bernstein_audit(restricted, sample_limit=8)
        passed = audit["negative_scaled_coefficient_count"] == 0
        child_rows.append(
            {
                "bits": f"0001/{bits}",
                "audit": _compact_audit(audit),
                "passed": passed,
            }
        )

    passed = bool(
        len(coarse_rows) == 16
        and len(child_rows) == 16
        and all(row["passed"] for row in coarse_rows if row["bits"] != "0001")
        and all(row["passed"] for row in child_rows)
    )
    return {
        "interval_convention": "0=[0,1/2], 1=[1/2,1]",
        "active_axes": ["ud", "ue", "ug", "ui"],
        "coarse_box_count": len(coarse_rows),
        "coarse_rows": coarse_rows,
        "delegated_coarse_box": "0001",
        "delegated_child_box_count": len(child_rows),
        "delegated_child_rows": child_rows,
        "certifying_leaf_count": 15 + len(child_rows),
        "complete_binary_cover": passed,
        "passed": passed,
    }


def _coefficient_hashes(coefficient_dir: Path) -> dict[str, str]:
    return {
        path.name: _sha256(path)
        for mask in H0 + H1
        for path in (
            coefficient_dir / f"alpha_sector_{''.join(map(str, mask))}.json.gz",
        )
    }


def verify_report(
    report_path: Path,
    *,
    coefficient_dir: Path = DEFAULT_COEFFICIENT_DIR,
    endpoint_artifact: Path = DEFAULT_ENDPOINT_ARTIFACT,
) -> dict[str, object]:
    """Replay the compact algebra, dependency hashes, and atlas cover.

    The 6,082,148-term reconstruction and Bernstein transforms remain the full
    source-harness tier.  This compact verifier checks that the stored exact
    controls form the declared complete cover and that no negative leaf was
    promoted.
    """

    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures = []
    if report.get("generic_identity") != _generic_determinant_identity():
        failures.append("generic determinant identity disagrees")
    if report.get("multidegree") != [24, 24, 24, 24, 48]:
        failures.append("determinant multidegree disagrees")
    if report.get("coefficient_source_sha256") != _coefficient_hashes(coefficient_dir):
        failures.append("coefficient-source hashes disagree")
    y_one = report.get("y_one_face", {})
    if y_one.get("delegated_endpoint_sha256") != _sha256(endpoint_artifact):
        failures.append("y=1 dependency hash disagrees")
    endpoint = json.loads(endpoint_artifact.read_text(encoding="utf-8"))
    if not endpoint.get("x_block", {}).get("passed"):
        failures.append("delegated X-block theorem is not passed")

    atlas = report.get("y_zero_face", {}).get("dyadic_atlas", {})
    expected_bits = {"".join(bits) for bits in itertools.product("01", repeat=4)}
    coarse = atlas.get("coarse_rows", [])
    children = atlas.get("delegated_child_rows", [])
    if {row.get("bits") for row in coarse} != expected_bits:
        failures.append("coarse y=0 atlas is not the complete 16-box cover")
    expected_children = {f"0001/{bits}" for bits in expected_bits}
    if {row.get("bits") for row in children} != expected_children:
        failures.append("delegated y=0 atlas is not the complete child cover")
    for row in coarse:
        negative = row.get("audit", {}).get("negative_scaled_coefficient_count")
        if row.get("bits") == "0001":
            if not isinstance(negative, int) or negative <= 0 or row.get("passed"):
                failures.append("delegated coarse box does not record its rejection")
        elif negative != 0 or not row.get("passed"):
            failures.append(f"coarse leaf {row.get('bits')} is not certified")
    for row in children:
        if row.get("audit", {}).get(
            "negative_scaled_coefficient_count"
        ) != 0 or not row.get("passed"):
            failures.append(f"child leaf {row.get('bits')} is not certified")
    if atlas.get("certifying_leaf_count") != 31:
        failures.append("y=0 certifying leaf count disagrees")
    if not report.get("endpoint_faces_proved") or not report.get("passed"):
        failures.append("report does not promote the endpoint theorem")
    return {
        "verified": not failures,
        "failures": failures,
        "compact_trust_boundary": (
            "Stored exact Bernstein summaries are checked for complete cover "
            "and sign consistency; rerun the source harness to recompute them."
        ),
    }


def run(
    coefficient_dir: Path,
    *,
    output: Path,
    endpoint_artifact: Path = DEFAULT_ENDPOINT_ARTIFACT,
    flint_threads: int = 6,
) -> dict[str, object]:
    """Reconstruct ``D`` and prove its low endpoint exactly."""

    if not 1 <= flint_threads <= 6:
        raise ValueError("FLINT thread count must be between one and six")
    resource = constrain_current_process(workers=flint_threads)
    ctx.threads = flint_threads

    identity = _generic_determinant_identity()
    determinant, tau, forced_product, _variables = _build_determinant(coefficient_dir)
    degrees = tuple(map(int, determinant.degrees()))
    if degrees != (24, 24, 24, 24, 48):
        raise AssertionError(f"unexpected determinant multidegree {degrees}")

    y_zero = determinant.subs({"y": 0})
    y_zero_audit = _native_bernstein_audit(y_zero, sample_limit=32)
    y_zero_atlas = _y_zero_atlas(y_zero)
    y_zero_passed = bool(y_zero_atlas["passed"])

    endpoint_report = json.loads(endpoint_artifact.read_text(encoding="utf-8"))
    y_one_dependency_passed = bool(
        endpoint_report.get("passed")
        and endpoint_report.get("x_block", {}).get("passed")
        and endpoint_report.get("exact_schur_reduction")
        == "K8>=0 iff X>=0 and Z=X^2-(1-y^2)R^2>=0"
    )
    y_one = determinant.subs({"y": 1})

    endpoint_passed = bool(
        all(identity.values())
        and forced_product == tau**2
        and y_zero_passed
        and y_one_dependency_passed
    )
    report = {
        "experiment": "adjacent endpoint-octet Schur determinant reconstruction",
        "domain": "(ud,ue,ug,ui,y) in [0,1]^5",
        "formula": (
            "D=Z0^4-2*Z0^2*sum(sj*Zj^2)+8*tau*Z0*Z1*Z2*Z3+"
            "sum((sj*Zj^2)^2)-2*sum_{j<k}(sj*Zj^2)*(sk*Zk^2)"
        ),
        "generic_identity": identity,
        "forced_radical_identity": {
            "identity": "s1*s2*s3=tau^2",
            "verified_exactly": forced_product == tau**2,
        },
        "power_term_count": len(determinant.to_dict()),
        "multidegree": list(degrees),
        "coefficient_source_sha256": _coefficient_hashes(coefficient_dir),
        "y_zero_face": {
            "power_term_count": len(y_zero.to_dict()),
            "multidegree": list(map(int, y_zero.degrees())),
            "native_bernstein": y_zero_audit,
            "dyadic_atlas": y_zero_atlas,
            "proved_nonnegative": y_zero_passed,
        },
        "y_one_face": {
            "power_term_count": len(y_one.to_dict()),
            "multidegree": list(map(int, y_one.degrees())),
            "exact_implication": "y=1 implies Z=X^2 and det(Z)=det(X)^2>=0",
            "delegated_endpoint_artifact": f"artifacts/{endpoint_artifact.name}",
            "delegated_endpoint_sha256": _sha256(endpoint_artifact),
            "proved_nonnegative": y_one_dependency_passed,
        },
        "endpoint_faces_proved": endpoint_passed,
        "passed": endpoint_passed,
        "resource_contract": resource,
        "scope_boundary": (
            "This proves exact determinant reconstruction and both y endpoint "
            "faces. Positivity for 0<y<1, the complete endpoint octet, and the "
            "unrestricted seven-variable Dirac--Gram theorem remain open."
        ),
    }
    _atomic_json(output, report)
    report["artifact_sha256"] = _sha256(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coefficient-dir", type=Path, default=DEFAULT_COEFFICIENT_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--endpoint-artifact", type=Path, default=DEFAULT_ENDPOINT_ARTIFACT
    )
    parser.add_argument("--flint-threads", type=int, default=6)
    arguments = parser.parse_args()
    report = run(
        arguments.coefficient_dir,
        output=arguments.output,
        endpoint_artifact=arguments.endpoint_artifact,
        flint_threads=arguments.flint_threads,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=int))


if __name__ == "__main__":
    main()
