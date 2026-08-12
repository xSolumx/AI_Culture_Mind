"""Assemble the exact finite-radius corner certificate for the cubic gate.

The expensive source harnesses reconstruct and transform the underlying
polynomials.  This compact verifier hash-binds those artifacts and checks that
their exact selectors, boundary-support partitions, and dyadic atlases form a
complete cover of the five max-coordinate blow-up charts.

It intentionally proves only the equality-corner neighbourhood.  The separate
first-level quotient box ``00010`` is not covered here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from spin8_dirac_endpoint_octet_cubic_tangent import verify_report as verify_tangent

DEFAULT_ARTIFACT_DIR = Path("artifacts")

SOURCE_FILES = {
    "tangent": "spin8_dirac_endpoint_octet_cubic_tangent_20260810.json",
    **{
        f"blowup_p{pivot}": (
            f"spin8_dirac_endpoint_octet_cubic_blowup_p{pivot}_20260810.json"
        )
        for pivot in range(5)
    },
    **{
        f"boundary_p{pivot}": (
            f"spin8_dirac_endpoint_octet_cubic_boundary_p{pivot}_20260810.json"
        )
        for pivot in range(3)
    },
    "atlas_p1": "spin8_dirac_endpoint_octet_cubic_boundary_atlas_p1_20260810.json",
    "atlas_p1_0010": (
        "spin8_dirac_endpoint_octet_cubic_boundary_atlas_p1_0010_20260810.json"
    ),
    "atlas_p2": "spin8_dirac_endpoint_octet_cubic_boundary_atlas_p2_20260810.json",
    "atlas_p2_0010": (
        "spin8_dirac_endpoint_octet_cubic_boundary_atlas_p2_0010_20260810.json"
    ),
    "atlas_p0_double": (
        "spin8_dirac_endpoint_octet_cubic_boundary_atlas_p0_double_20260810.json"
    ),
    "atlas_p0_ui": (
        "spin8_dirac_endpoint_octet_cubic_boundary_atlas_p0_ui_20260810.json"
    ),
    "atlas_p0_ui_0000_r0": (
        "spin8_dirac_endpoint_octet_cubic_boundary_atlas_p0_ui_0000_r0_"
        "20260810.json"
    ),
    "atlas_p0_ui_0000_r0_001": (
        "spin8_dirac_endpoint_octet_cubic_boundary_atlas_p0_ui_0000_r0_001_"
        "20260810.json"
    ),
    "atlas_p0_ui_0001_r0": (
        "spin8_dirac_endpoint_octet_cubic_boundary_atlas_p0_ui_0001_r0_"
        "20260810.json"
    ),
    "atlas_p0_ui_0001_r0_000": (
        "spin8_dirac_endpoint_octet_cubic_boundary_atlas_p0_ui_0001_r0_000_"
        "20260810.json"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _audit_passed(audit: dict[str, object]) -> bool:
    return int(audit["negative_scaled_coefficient_count"]) == 0


def _support_is_covered(
    audit: dict[str, object], predicate
) -> bool:
    """Check that the histogram accounts for every negative control."""

    histogram = audit["negative_boundary_histogram"]
    if not isinstance(histogram, dict):
        return False
    if sum(int(count) for count in histogram.values()) != int(
        audit["negative_scaled_coefficient_count"]
    ):
        return False
    return all(predicate(set(label.split(","))) for label in histogram)


def _check_blowup(report: dict[str, object], pivot: int) -> dict[str, object]:
    if int(report["pivot_index"]) != pivot:
        raise AssertionError(f"blow-up p{pivot} has the wrong pivot index")
    if not report["construction_passed"]:
        raise AssertionError(f"blow-up p{pivot} construction failed")
    if int(report["exact_radius_divisibility_order"]) != 6:
        raise AssertionError(f"blow-up p{pivot} does not have exact radius order six")
    if not report["exceptional_tangent_proportionality"]["passed"]:
        raise AssertionError(f"blow-up p{pivot} tangent face mismatch")

    native = report["quotient_native_bernstein"]
    selector = report["radial_selector_certificate"]
    if pivot in (3, 4):
        if not report["passed"] or not _audit_passed(native):
            raise AssertionError(f"native blow-up p{pivot} did not certify")
        mechanism = "native tensor-product Bernstein positivity"
    else:
        if selector is None or not selector["identity_verified_exactly"]:
            raise AssertionError(f"blow-up p{pivot} radial selector failed")
        if not selector["tangent_radical_factor_certificate"]["passed"]:
            raise AssertionError(f"blow-up p{pivot} tangent factor proof failed")
        remainder = selector["remainder_native_bernstein"]
        if pivot in (1, 2):
            covered = _support_is_covered(
                remainder, lambda support: "3:0" in support
            )
            mechanism = "radial selector plus ui=0 face atlas"
        else:
            covered = _support_is_covered(
                remainder,
                lambda support: (
                    "3:0" in support or {"1:0", "2:0"} <= support
                ),
            )
            mechanism = "radial selector plus ui=0 and ue=ug=0 face atlases"
        if not covered:
            raise AssertionError(f"blow-up p{pivot} has an uncovered negative control")
    return {"pivot": pivot, "mechanism": mechanism, "verified": True}


def _check_boundary(
    report: dict[str, object], pivot: int, *, require_double: bool
) -> dict[str, object]:
    if int(report["pivot_index"]) != pivot or int(report["exact_radius_order"]) != 6:
        raise AssertionError(f"boundary p{pivot} identity mismatch")
    ui = report["ui_zero_selector"]
    if not ui["identity_verified_exactly"]:
        raise AssertionError(f"boundary p{pivot} ui selector identity failed")
    if not ui["face"]["factorization"]["identity_verified_exactly"]:
        raise AssertionError(f"boundary p{pivot} ui factorization failed")
    double = report["ue_ug_zero_selector"]
    if require_double:
        if double is None or not double["identity_verified_exactly"]:
            raise AssertionError("boundary p0 double selector identity failed")
        if not double["face"]["factorization"]["identity_verified_exactly"]:
            raise AssertionError("boundary p0 double-face factorization failed")
    elif double is not None:
        raise AssertionError(f"boundary p{pivot} unexpectedly contains a double face")
    return {
        "pivot": pivot,
        "ui_selector": True,
        "double_selector": require_double,
        "verified": True,
    }


def _check_atlas(
    report: dict[str, object],
    *,
    pivot: int,
    active_dimension: int,
    expected_failures: set[str],
    expected_parent: list[str] | None = None,
    expected_post_zero: list[str] | None = None,
    require_zero_selector: bool = False,
) -> dict[str, object]:
    if int(report["pivot_index"]) != pivot:
        raise AssertionError("atlas pivot mismatch")
    rows = report["boxes"]
    expected_count = 2**active_dimension
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise AssertionError("atlas does not contain its complete binary partition")
    labels = {str(row["active_bits"]) for row in rows}
    expected_labels = {format(index, f"0{active_dimension}b") for index in range(expected_count)}
    if labels != expected_labels:
        raise AssertionError("atlas active labels are not the complete binary partition")
    failures = {str(row["active_bits"]) for row in rows if not row["passed"]}
    if failures != expected_failures:
        raise AssertionError(f"unexpected atlas failures: {sorted(failures)}")
    for row in rows:
        if bool(row["passed"]) != _audit_passed(row["audit"]):
            raise AssertionError("atlas pass flag disagrees with its exact sign count")
    parent = [str(row["active_bits"]) for row in report.get("parent_path", [])]
    if parent != (expected_parent or []):
        raise AssertionError("atlas parent path mismatch")
    post_zero = [str(row["active_bits"]) for row in report.get("post_zero_path", [])]
    if post_zero != (expected_post_zero or []):
        raise AssertionError("atlas post-zero path mismatch")
    selected_zero = report.get("selected_zero_face")
    if require_zero_selector:
        if selected_zero is None or not selected_zero["identity_verified_exactly"]:
            raise AssertionError("atlas zero-face selector identity failed")
    elif selected_zero is not None:
        raise AssertionError("unexpected zero-face selector")
    if not _audit_passed(report["monomial_native_bernstein"]):
        raise AssertionError("atlas monomial prefactor is not nonnegative")
    return {
        "pivot": pivot,
        "box_count": expected_count,
        "expected_failures": sorted(expected_failures),
        "verified": True,
    }


def _check_failed_box_support(
    report: dict[str, object], active_bits: str, *, required_axis: str
) -> None:
    row = next(
        (row for row in report["boxes"] if row["active_bits"] == active_bits), None
    )
    if row is None or row["passed"]:
        raise AssertionError("expected failed parent atlas box is absent")
    if not _support_is_covered(
        row["audit"], lambda support: required_axis in support
    ):
        raise AssertionError("parent atlas has negatives away from selected zero face")


def _bind_atlas_to_boundary_face(
    atlas: dict[str, object], face: dict[str, object]
) -> None:
    """Cross-check that an atlas has the stored selected-face fingerprint."""

    factorization = face["factorization"]
    if abs(int(factorization["content"])) != int(atlas["positive_content"]):
        raise AssertionError("atlas content does not match its boundary face")
    if int(face["power_term_count"]) != int(atlas["oriented_core_power_term_count"]):
        raise AssertionError("atlas term count does not match its boundary core")
    core_factors = [
        row for row in factorization["factors"] if int(row["power_term_count"]) != 1
    ]
    if len(core_factors) != 1:
        raise AssertionError("boundary factorization does not isolate one atlas core")
    if list(core_factors[0]["multidegree"]) != list(atlas["oriented_core_multidegree"]):
        raise AssertionError("atlas multidegree does not match its boundary core")


def assemble(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR, *, output: Path | None = None
) -> dict[str, object]:
    paths = {name: artifact_dir / filename for name, filename in SOURCE_FILES.items()}
    reports = {name: _read(path) for name, path in paths.items()}

    tangent = verify_tangent(reports["tangent"])
    if not tangent["verified"]:
        raise AssertionError(f"tangent replay failed: {tangent['failures']}")
    blowups = [
        _check_blowup(reports[f"blowup_p{pivot}"], pivot) for pivot in range(5)
    ]
    boundaries = [
        _check_boundary(
            reports[f"boundary_p{pivot}"], pivot, require_double=pivot == 0
        )
        for pivot in range(3)
    ]

    atlas_checks = {
        "p1": _check_atlas(
            reports["atlas_p1"],
            pivot=1,
            active_dimension=4,
            expected_failures={"0010"},
        ),
        "p1_0010": _check_atlas(
            reports["atlas_p1_0010"],
            pivot=1,
            active_dimension=4,
            expected_failures=set(),
            expected_parent=["0010"],
        ),
        "p2": _check_atlas(
            reports["atlas_p2"],
            pivot=2,
            active_dimension=4,
            expected_failures={"0010"},
        ),
        "p2_0010": _check_atlas(
            reports["atlas_p2_0010"],
            pivot=2,
            active_dimension=4,
            expected_failures=set(),
            expected_parent=["0010"],
        ),
        "p0_double": _check_atlas(
            reports["atlas_p0_double"],
            pivot=0,
            active_dimension=3,
            expected_failures=set(),
        ),
        "p0_ui": _check_atlas(
            reports["atlas_p0_ui"],
            pivot=0,
            active_dimension=4,
            expected_failures={"0000", "0001"},
        ),
        "p0_ui_0000_r0": _check_atlas(
            reports["atlas_p0_ui_0000_r0"],
            pivot=0,
            active_dimension=3,
            expected_failures={"001"},
            expected_parent=["0000"],
            require_zero_selector=True,
        ),
        "p0_ui_0000_r0_001": _check_atlas(
            reports["atlas_p0_ui_0000_r0_001"],
            pivot=0,
            active_dimension=3,
            expected_failures=set(),
            expected_parent=["0000"],
            expected_post_zero=["001"],
            require_zero_selector=True,
        ),
        "p0_ui_0001_r0": _check_atlas(
            reports["atlas_p0_ui_0001_r0"],
            pivot=0,
            active_dimension=3,
            expected_failures={"000"},
            expected_parent=["0001"],
            require_zero_selector=True,
        ),
        "p0_ui_0001_r0_000": _check_atlas(
            reports["atlas_p0_ui_0001_r0_000"],
            pivot=0,
            active_dimension=3,
            expected_failures=set(),
            expected_parent=["0001"],
            expected_post_zero=["000"],
            require_zero_selector=True,
        ),
    }
    _bind_atlas_to_boundary_face(
        reports["atlas_p1"], reports["boundary_p1"]["ui_zero_selector"]["face"]
    )
    _bind_atlas_to_boundary_face(
        reports["atlas_p2"], reports["boundary_p2"]["ui_zero_selector"]["face"]
    )
    _bind_atlas_to_boundary_face(
        reports["atlas_p0_ui"], reports["boundary_p0"]["ui_zero_selector"]["face"]
    )
    _bind_atlas_to_boundary_face(
        reports["atlas_p0_double"],
        reports["boundary_p0"]["ue_ug_zero_selector"]["face"],
    )
    # The only two failures of the p0 ui-face atlas occur on its radius-zero
    # Bernstein layer.  Selecting that layer is therefore exhaustive; the
    # two three-dimensional atlases and their single refined children cover
    # the selected faces themselves.
    _check_failed_box_support(
        reports["atlas_p0_ui"], "0000", required_axis="0:0"
    )
    _check_failed_box_support(
        reports["atlas_p0_ui"], "0001", required_axis="0:0"
    )

    report = {
        "experiment": "finite-radius equality-corner certificate for the cubic gate",
        "theorem": (
            "The adjacent endpoint-octet cubic is nonnegative whenever "
            "ud,ue,ug,ui,1-y all lie in [0,1/4]."
        ),
        "domain": {
            "deviations": ["ud", "ue", "ug", "ui", "1-y"],
            "interval": "[0,1/4]^5",
            "max_coordinate_cover": (
                "choose a largest deviation m; set radius=4m and every "
                "nonpivot ratio to deviation/m"
            ),
            "pivot_count": 5,
            "zero_deviation_value": 0,
        },
        "tangent_certificate": {
            "exact_order": 6,
            "factorization": "H6=2^36*F2*F4",
            "radical_factor_route_verified": True,
        },
        "blowup_checks": blowups,
        "boundary_checks": boundaries,
        "atlas_checks": atlas_checks,
        "source_artifacts": {
            name: {"path": f"artifacts/{path.name}", "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "verifier_contract": {
            "recomputes": [
                "all source SHA-256 hashes",
                "the stored tangent factor proof from exact factor coefficients",
                "all five blow-up construction, radius-order, and tangent-face predicates",
                "complete support-histogram coverage for every selected boundary",
                "all selector and factorization identities recorded by the source artifacts",
                "every binary atlas partition, parent path, failed cell, and exact sign count",
            ],
            "trusts": [
                "the source harnesses' stored exact polynomial identities",
                "the source harnesses' FLINT power-to-Bernstein transforms",
                "the source artifacts' complete negative-support histograms",
            ],
            "full_replay": [
                "spin8_dirac_endpoint_octet_cubic_tangent.py",
                "spin8_dirac_endpoint_octet_cubic_blowup.py",
                "spin8_dirac_endpoint_octet_cubic_boundary.py",
                "spin8_dirac_endpoint_octet_cubic_boundary_atlas.py",
            ],
        },
        "passed": True,
        "scope_boundary": (
            "This is an exact finite-radius theorem only for the equality "
            "corner [0,1/4]^5 in deviation coordinates. The separate quotient "
            "box 00010, the full cubic, determinant, endpoint octet, and "
            "unrestricted Dirac--Gram inequality remain open."
        ),
    }
    if output is not None:
        _atomic_json(output, report)
        report["artifact_sha256"] = _sha256(output)
    return report


def verify(report_path: Path) -> dict[str, object]:
    stored = _read(report_path)
    rebuilt = assemble(report_path.parent)
    for key in (
        "theorem",
        "domain",
        "tangent_certificate",
        "blowup_checks",
        "boundary_checks",
        "atlas_checks",
        "source_artifacts",
        "verifier_contract",
        "passed",
        "scope_boundary",
    ):
        if stored[key] != rebuilt[key]:
            raise AssertionError(f"corner certificate mismatch at {key}")
    return {
        "verified": True,
        "source_artifact_count": len(rebuilt["source_artifacts"]),
        "pivot_count": len(rebuilt["blowup_checks"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", type=Path)
    arguments = parser.parse_args()
    if arguments.verify is not None:
        report = verify(arguments.verify)
    else:
        report = assemble(arguments.artifact_dir, output=arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
