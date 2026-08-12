"""Assemble and verify the exact global certificate for octet quadratic 0.

This lightweight verifier checks the logical cover and the load-bearing fields
of the expensive exact artifacts.  It does not reconstruct their Bernstein
arrays; the source harnesses remain the full replay path for those arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

DEFAULT_ARTIFACT_DIR = Path("artifacts")
MODE_MASKS = {0: "0011001", 1: "0101010", 2: "0110011"}


def _atlas_files(minor_index: int) -> dict[str, str]:
    stem = f"spin8_dirac_endpoint_octet_quadratic_{minor_index}_half_atlas"
    return {
        "coarse": f"{stem}_20260808.json",
        "child_00001": f"{stem}_00001_20260808.json",
        "child_00010": f"{stem}_00010_20260808.json",
    }


def _blowup_files(minor_index: int) -> dict[int, str]:
    return {
        pivot: (
            f"spin8_dirac_endpoint_octet_blowup_q{minor_index}_p{pivot}_"
            "20260808.json"
        )
        for pivot in range(5)
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


def _atlas_rows(report: dict[str, object]) -> list[dict[str, object]]:
    rows = report["half_box_atlas"]
    if not isinstance(rows, list) or len(rows) != 32:
        raise AssertionError("each binary atlas must contain all 32 boxes")
    return rows


def _check_atlas(
    report: dict[str, object], *, expected_failures: set[str]
) -> dict[str, object]:
    rows = _atlas_rows(report)
    labels = {str(row["bits"]) for row in rows}
    expected_labels = {format(index, "05b") for index in range(32)}
    parent = report.get("half_box_atlas_parent")
    if parent:
        expected_labels = {f"{parent}/{label}" for label in expected_labels}
    if labels != expected_labels:
        raise AssertionError("atlas paths do not form the frozen binary partition")
    actual_failures = {str(row["bits"]) for row in rows if not row["passed"]}
    if actual_failures != expected_failures:
        raise AssertionError(
            f"unexpected atlas failures: {sorted(actual_failures)}"
        )
    for row in rows:
        audit_passed = _audit_passed(row["native_bernstein"])
        if bool(row["passed"]) != audit_passed:
            raise AssertionError("atlas pass flag disagrees with its coefficient array")
    return {
        "box_count": 32,
        "expected_failures": sorted(expected_failures),
        "verified": True,
    }


def _check_blowup(
    report: dict[str, object], pivot: int, *, minor_index: int
) -> dict[str, object]:
    if (
        int(report["minor_index"]) != minor_index
        or int(report["pivot_index"]) != pivot
    ):
        raise AssertionError("blow-up artifact identity mismatch")
    if int(report["exact_radius_divisibility_order"]) != 4:
        raise AssertionError("the equality germ must have exact radius order four")
    if not report["passed"]:
        raise AssertionError(f"pivot {pivot} did not pass")

    if minor_index in (1, 2) and pivot == 0:
        exceptional = report["exceptional_divisor"][
            "nested_ui_zero_certificate"
        ]["complete_exceptional_square"]
        radial = report["exceptional_boundary_selector"][
            "nested_ui_zero_certificate"
        ]["radial_exact_certificate"]
        if not exceptional["identity_verified"] or not exceptional["passed"]:
            raise AssertionError("shared exceptional square failed")
        boxes = radial["corner_four_box_atlas"]
        if len(boxes) != 4 or not all(
            row["passed"] and _audit_passed(row["native_bernstein"])
            for row in boxes
        ):
            raise AssertionError("shared radial corner atlas failed")
        if not _audit_passed(radial["second_remainder_native_bernstein"]):
            raise AssertionError("shared radial complement failed")
        mechanism = "complete tangent square plus four-box radial corner atlas"
    elif minor_index in (1, 2) or pivot in (3, 4):
        if not _audit_passed(report["quotient_native_bernstein"]):
            raise AssertionError("native-positive pivot contains a negative control")
        mechanism = "native tensor-product Bernstein positivity"
    elif pivot == 0:
        exceptional = report["exceptional_divisor"][
            "nested_ui_zero_certificate"
        ]
        radial = report["exceptional_boundary_selector"][
            "nested_ui_zero_certificate"
        ]["radial_ui_zero_exact_certificate"]
        if not exceptional["ui_zero_tangent_product"]["passed"]:
            raise AssertionError("pivot-0 signed tangent product failed")
        corner = exceptional["nested_ue_ug_zero_corner"]
        if not corner["corner_sign_factorization"]["passed"]:
            raise AssertionError("pivot-0 exceptional corner factorization failed")
        if not _audit_passed(corner["second_remainder_native_bernstein"]):
            raise AssertionError("pivot-0 exceptional remainder is not certified")
        if not radial["comparison_identity_verified"]:
            raise AssertionError("pivot-0 selector comparison identity failed")
        if not radial["linear_axis"]["corner_identity_verified"]:
            raise AssertionError("pivot-0 radial-axis identity failed")
        if not radial["linear_axis"]["cubic_strictly_positive"]:
            raise AssertionError("pivot-0 radial cubic is not positive")
        if not _audit_passed(
            radial["linear_axis"]["remainder_native_bernstein"]
        ):
            raise AssertionError("pivot-0 radial-axis remainder failed")
        boxes = radial["linear_radial_remainder"][
            "ue_ug_zero_corner_four_box_atlas"
        ]
        if len(boxes) != 4 or not all(
            row["passed"] and _audit_passed(row["native_bernstein"])
            for row in boxes
        ):
            raise AssertionError("pivot-0 four-box corner atlas failed")
        if not _audit_passed(
            radial["linear_radial_remainder"][
                "second_remainder_native_bernstein"
            ]
        ):
            raise AssertionError("pivot-0 radial remainder failed")
        mechanism = "exact signed factors plus nested Bernstein selectors"
    else:
        exceptional = report["exceptional_divisor"][
            "nested_ui_zero_certificate"
        ]
        boundary = report["exceptional_boundary_selector"][
            "nested_ui_zero_certificate"
        ]
        radial = boundary["radial_ui_zero_exact_certificate"]
        if not exceptional["ui_zero_tangent_product"]["passed"]:
            raise AssertionError("middle-pivot signed tangent product failed")
        if not _audit_passed(exceptional["remainder_native_bernstein"]):
            raise AssertionError("middle-pivot exceptional remainder failed")
        if not radial["comparison_identity_verified"]:
            raise AssertionError("middle-pivot selector comparison failed")
        if not _audit_passed(radial["alternate_quotient_native_bernstein"]):
            raise AssertionError("middle-pivot alternate quotient failed")
        if not _audit_passed(boundary["second_remainder_native_bernstein"]):
            raise AssertionError("middle-pivot ui remainder failed")
        mechanism = (
            "signed tangent product plus alternate-selector comparison"
        )
    return {"pivot": pivot, "mechanism": mechanism, "verified": True}


def assemble(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    *,
    minor_index: int = 0,
    output: Path | None = None,
) -> dict[str, object]:
    if minor_index not in (0, 1, 2):
        raise ValueError("minor index must be 0, 1, or 2")
    atlas_files = _atlas_files(minor_index)
    blowup_files = _blowup_files(minor_index)
    paths = {
        name: artifact_dir / filename for name, filename in atlas_files.items()
    }
    paths.update(
        {
            f"blowup_p{pivot}": artifact_dir / filename
            for pivot, filename in blowup_files.items()
        }
    )
    reports = {name: _read(path) for name, path in paths.items()}
    atlas_checks = {
        "coarse": _check_atlas(
            reports["coarse"], expected_failures={"00001", "00010"}
        ),
        "child_00010": _check_atlas(
            reports["child_00010"], expected_failures=set()
        ),
        "child_00001": _check_atlas(
            reports["child_00001"],
            expected_failures={"00001/00001"},
        ),
    }
    blowup_checks = [
        _check_blowup(
            reports[f"blowup_p{pivot}"], pivot, minor_index=minor_index
        )
        for pivot in range(5)
    ]
    mode_mask = MODE_MASKS[minor_index]
    theorem = (
        "Z_trivial^2 - s_mu*Z_mu^2 is nonnegative on "
        "[0,1]^4 x [-1,1] for the first nontrivial H0 mode"
        if minor_index == 0
        else (
            "Z_trivial^2 - s_mu*Z_mu^2 is nonnegative on "
            f"[0,1]^4 x [-1,1] for H0 mode {mode_mask}"
        )
    )
    report = {
        "experiment": (
            f"global certificate for endpoint-octet quadratic minor {minor_index}"
        ),
        "theorem": theorem,
        "domain_reduction": {
            "even_cayley_coordinate": "y=abs(c) in [0,1]",
            "coarse_binary_partition_box_count": 32,
            "refined_00010_box_count": 32,
            "refined_00001_certified_box_count": 31,
            "sole_residual_path": "00001/00001",
            "sole_residual_region": (
                "ud,ue,ug,ui in [0,1/4] and y in [3/4,1]"
            ),
            "deviation_form": (
                "ud,ue,ug,ui,1-y all lie in [0,1/4]"
            ),
        },
        "max_coordinate_cover": {
            "argument": (
                "choose a largest nonzero deviation m; set radius=4m and "
                "the other four coordinates to deviation/m"
            ),
            "radius_interval": "[0,1]",
            "ratio_intervals": "[0,1]^4",
            "pivot_count": 5,
            "zero_deviation_point": "the quartic germ vanishes exactly",
            "verified_pivots": blowup_checks,
        },
        "atlas_checks": atlas_checks,
        "source_artifacts": {
            name: {
                "path": f"artifacts/{path.name}",
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "verifier_contract": {
            "recomputes": [
                "all source SHA-256 hashes",
                "binary-atlas completeness and expected unresolved paths",
                "stored Bernstein sign counts for every accepted atlas cell",
                "all five radius-order and pivot acceptance predicates",
                "load-bearing stored factor, selector, and comparison identities",
            ],
            "trusts": [
                "the source harnesses' stored exact coefficient arrays",
                "the source harnesses' FLINT power-to-Bernstein transforms",
            ],
            "full_replay": [
                "spin8_dirac_endpoint_octet_quadratic.py",
                "spin8_dirac_endpoint_octet_blowup.py",
            ],
        },
        "passed": True,
        "scope_boundary": (
            "This proves one of three quadratic principal-minor inequalities "
            "on the adjacent endpoint octet. The other two quadratics, the "
            "cubic, determinant, full endpoint octet, and unrestricted "
            "Dirac--Gram theorem remain open."
            if minor_index == 0
            else (
                (
                    "This proves the second of three quadratic principal-minor "
                    "inequalities on the adjacent endpoint octet. The final "
                    "quadratic, cubic, determinant, full endpoint octet, and "
                    "unrestricted Dirac--Gram theorem remain open."
                )
                if minor_index == 1
                else (
                    "This proves the final quadratic principal-minor inequality "
                    "on the adjacent endpoint octet. Together the three global "
                    "quadratic certificates close the complete quadratic gate. "
                    "The cubic, determinant, full endpoint octet, and unrestricted "
                    "Dirac--Gram theorem remain open."
                )
            )
        ),
    }
    if output is not None:
        _atomic_json(output, report)
        report["artifact_sha256"] = _sha256(output)
    return report


def verify(report_path: Path) -> dict[str, object]:
    stored = _read(report_path)
    artifact_dir = report_path.parent
    theorem = str(stored["theorem"])
    matches = [index for index, mask in MODE_MASKS.items() if mask in theorem]
    if "first nontrivial H0 mode" in theorem:
        matches = [0]
    if len(matches) != 1 or matches[0] not in (0, 1, 2):
        raise AssertionError("could not infer the certified quadratic mode")
    rebuilt = assemble(artifact_dir, minor_index=matches[0])
    for key in (
        "theorem",
        "domain_reduction",
        "max_coordinate_cover",
        "atlas_checks",
        "source_artifacts",
        "verifier_contract",
        "passed",
        "scope_boundary",
    ):
        if stored[key] != rebuilt[key]:
            raise AssertionError(f"assembled certificate mismatch at {key}")
    return {
        "verified": True,
        "artifact_sha256": _sha256(report_path),
        "source_artifact_count": len(rebuilt["source_artifacts"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    assemble_parser = subparsers.add_parser("assemble")
    assemble_parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    assemble_parser.add_argument(
        "--minor-index", type=int, choices=(0, 1, 2), default=0
    )
    assemble_parser.add_argument("--output", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("report", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "assemble":
        report = assemble(
            arguments.artifact_dir,
            minor_index=arguments.minor_index,
            output=arguments.output,
        )
    else:
        report = verify(arguments.report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
