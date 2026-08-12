"""Exact dyadic atlas for the unresolved cubic ``00010`` selected face."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from flint import ctx

from spin8_dirac_endpoint_octet_cubic import DEFAULT_COEFFICIENT_DIR
from spin8_dirac_endpoint_octet_cubic_atlas import _build_quotient
from spin8_dirac_endpoint_octet_cubic_blowup import _batched_bernstein_audit
from spin8_dirac_endpoint_octet_quadratic import (
    _atomic_json,
    _restrict_half_box,
    _sha256,
)
from spin8_resource_limits import constrain_current_process

DEFAULT_BOUNDARY_REPORT = Path(
    "artifacts/spin8_dirac_endpoint_octet_cubic_boundary_00010_20260809.json"
)


def _coefficient_sha256(polynomial) -> str:
    digest = hashlib.sha256()
    for powers, coefficient in sorted(polynomial.to_dict().items()):
        digest.update(",".join(map(str, powers)).encode("ascii"))
        digest.update(b":")
        digest.update(str(coefficient).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _selected_face(coefficient_dir: Path):
    quotient = _build_quotient(coefficient_dir)
    restricted = _restrict_half_box(quotient, "00010")
    return restricted.subs({"ud": 0})


def _full_bits(active_bits: tuple[str, ...] | str) -> str:
    row = ["0"] * 5
    for axis, bit in zip((1, 2, 3, 4), active_bits, strict=True):
        row[axis] = bit
    return "".join(row)


def run(
    coefficient_dir: Path,
    *,
    boundary_report: Path,
    output: Path,
    batch_entry_limit: int = 500_000,
    flint_threads: int = 6,
) -> dict[str, object]:
    if not 1 <= flint_threads <= 6:
        raise ValueError("FLINT thread count must be between one and six")
    resource = constrain_current_process(workers=flint_threads)
    ctx.threads = flint_threads
    stored_boundary = json.loads(boundary_report.read_text(encoding="utf-8"))
    if stored_boundary.get("box") != "00010":
        raise AssertionError("the selector source is not box 00010")
    if not stored_boundary.get("identity_verified_exactly"):
        raise AssertionError("the box-00010 selector identity did not pass")
    if int(
        stored_boundary["remainder_native_bernstein"][
            "negative_scaled_coefficient_count"
        ]
    ):
        raise AssertionError("the box-00010 selector complement is not certified")

    face = _selected_face(coefficient_dir)
    if len(face.to_dict()) != int(stored_boundary["face_power_term_count"]):
        raise AssertionError("selected-face term count disagrees with its source")
    if list(map(int, face.degrees())) != list(
        stored_boundary["face_native_bernstein"]["multidegree"]
    ):
        raise AssertionError("selected-face multidegree disagrees with its source")

    source_sha = _sha256(boundary_report)
    previous_rows: dict[str, dict[str, object]] = {}
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("source_boundary_artifact_sha256") == source_sha:
            previous_rows = {row["active_bits"]: row for row in previous["boxes"]}

    labels = ["".join(bits) for bits in itertools.product("01", repeat=4)]
    for active_bits in labels:
        if active_bits in previous_rows:
            continue
        bits = _full_bits(active_bits)
        child = _restrict_half_box(face, bits)
        audit = _batched_bernstein_audit(
            child,
            sample_limit=64,
            batch_entry_limit=batch_entry_limit,
        )
        previous_rows[active_bits] = {
            "bits": bits,
            "active_bits": active_bits,
            "power_term_count": len(child.to_dict()),
            "native_bernstein": audit,
            "passed": audit["negative_scaled_coefficient_count"] == 0,
        }
        rows = [previous_rows[label] for label in labels if label in previous_rows]
        report = {
            "experiment": "cubic box 00010 selected-face dyadic atlas",
            "source_boundary_artifact": boundary_report.as_posix(),
            "source_boundary_artifact_sha256": source_sha,
            "source_face_coefficient_sha256": _coefficient_sha256(face),
            "source_face_power_term_count": len(face.to_dict()),
            "source_face_multidegree": list(map(int, face.degrees())),
            "selected_face": "first-level quotient box 00010 with ud=0",
            "active_axes": [1, 2, 3, 4],
            "boxes": rows,
            "completed_box_count": len(rows),
            "complete": len(rows) == 16,
            "passed": len(rows) == 16 and all(row["passed"] for row in rows),
            "resource_contract": resource,
            "scope_boundary": (
                "A pass certifies only the selected face of box 00010. The "
                "selector complement is delegated to the hash-bound source "
                "artifact; the rest of the cubic remains separate."
            ),
        }
        _atomic_json(output, report)

    result = json.loads(output.read_text(encoding="utf-8"))
    result["artifact_sha256"] = _sha256(output)
    return result


def verify_report(report_path: Path) -> dict[str, object]:
    """Replay the compact acceptance contract for a completed face atlas."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    source_path = Path(str(report["source_boundary_artifact"]))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if _sha256(source_path) != report["source_boundary_artifact_sha256"]:
        failures.append("source boundary SHA-256 mismatch")
    if source.get("box") != "00010" or not source.get("identity_verified_exactly"):
        failures.append("source selector identity did not pass")
    if int(
        source["remainder_native_bernstein"]["negative_scaled_coefficient_count"]
    ):
        failures.append("source selector complement is not certified")
    if int(report["source_face_power_term_count"]) != int(
        source["face_power_term_count"]
    ):
        failures.append("source face term count mismatch")
    if list(report["source_face_multidegree"]) != list(
        source["face_native_bernstein"]["multidegree"]
    ):
        failures.append("source face multidegree mismatch")
    rows = report.get("boxes", [])
    labels = {str(row["active_bits"]) for row in rows}
    if labels != {format(index, "04b") for index in range(16)}:
        failures.append("atlas is not the complete four-cube partition")
    for row in rows:
        passed = int(
            row["native_bernstein"]["negative_scaled_coefficient_count"]
        ) == 0
        if bool(row["passed"]) != passed or not passed:
            failures.append(f"atlas child {row['active_bits']} did not certify")
    if not report.get("complete") or not report.get("passed"):
        failures.append("atlas top-level acceptance flags are false")
    return {
        "verified": not failures,
        "failures": failures,
        "box_count": len(rows),
        "source_boundary_sha256": _sha256(source_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coefficient-dir", type=Path, default=DEFAULT_COEFFICIENT_DIR
    )
    parser.add_argument(
        "--boundary-report", type=Path, default=DEFAULT_BOUNDARY_REPORT
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-entry-limit", type=int, default=500_000)
    parser.add_argument("--flint-threads", type=int, default=6)
    parser.add_argument("--verify", type=Path)
    arguments = parser.parse_args()
    if arguments.verify is not None:
        report = verify_report(arguments.verify)
    else:
        report = run(
            arguments.coefficient_dir,
            boundary_report=arguments.boundary_report,
            output=arguments.output,
            batch_entry_limit=arguments.batch_entry_limit,
            flint_threads=arguments.flint_threads,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
