"""Complete first-level exact atlas for the endpoint-octet cubic quotient."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from flint import ctx

from spin8_dirac_endpoint_octet_cubic import DEFAULT_COEFFICIENT_DIR
from spin8_dirac_endpoint_octet_cubic_00010_atlas import (
    verify_report as verify_00010_report,
)
from spin8_dirac_endpoint_octet_cubic_atlas import (
    DEFAULT_ENDPOINT_REPORT,
    _build_quotient,
    _check_stored_bernstein_audit,
    _source_sha,
    verify_nested_00001_report,
)
from spin8_dirac_endpoint_octet_cubic_blowup import _batched_bernstein_audit
from spin8_dirac_endpoint_octet_quadratic import (
    _atomic_json,
    _restrict_half_box,
    _sha256,
)
from spin8_resource_limits import constrain_current_process

DEFAULT_00010_REPORT = Path(
    "artifacts/spin8_dirac_endpoint_octet_cubic_boundary_00010_atlas_20260810.json"
)
DEFAULT_00001_REPORT = Path(
    "artifacts/spin8_dirac_endpoint_octet_cubic_atlas_nested_00001_complete_"
    "20260810.json"
)


def _delegated_row(bits: str, report: Path, mechanism: str) -> dict[str, object]:
    return {
        "bits": bits,
        "interval_convention": "0=[0,1/2], 1=[1/2,1]",
        "certificate_kind": mechanism,
        "source_certificate": report.as_posix(),
        "source_certificate_sha256": _sha256(report),
        "passed": True,
    }


def run(
    coefficient_dir: Path,
    *,
    endpoint_report: Path,
    report_00010: Path,
    report_00001: Path,
    output: Path,
    batch_entry_limit: int = 500_000,
    flint_threads: int = 6,
) -> dict[str, object]:
    if not 1 <= flint_threads <= 6:
        raise ValueError("FLINT thread count must be between one and six")
    if not verify_00010_report(report_00010)["verified"]:
        raise AssertionError("box-00010 delegated certificate did not replay")
    if not verify_nested_00001_report(report_00001)["verified"]:
        raise AssertionError("box-00001 delegated certificate did not replay")
    resource = constrain_current_process(workers=flint_threads)
    ctx.threads = flint_threads
    endpoint_sha = _source_sha(endpoint_report)
    quotient = _build_quotient(coefficient_dir)
    labels = ["".join(bits) for bits in itertools.product("01", repeat=5)]

    stored: dict[str, dict[str, object]] = {}
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if (
            previous.get("source_endpoint_artifact_sha256") == endpoint_sha
            and previous.get("source_00010_artifact_sha256") == _sha256(report_00010)
            and previous.get("source_00001_artifact_sha256") == _sha256(report_00001)
        ):
            stored = {row["bits"]: row for row in previous.get("boxes", [])}
    stored["00010"] = _delegated_row(
        "00010", report_00010, "delegated_exact_face_plus_complement"
    )
    stored["00001"] = _delegated_row(
        "00001", report_00001, "delegated_exact_complete_child_atlas"
    )

    for bits in labels:
        if bits in stored:
            continue
        child = _restrict_half_box(quotient, bits)
        audit = _batched_bernstein_audit(
            child,
            sample_limit=64,
            batch_entry_limit=batch_entry_limit,
        )
        stored[bits] = {
            "bits": bits,
            "interval_convention": "0=[0,1/2], 1=[1/2,1]",
            "power_term_count": len(child.to_dict()),
            "native_bernstein": audit,
            "passed": audit["negative_scaled_coefficient_count"] == 0,
        }
        rows = [stored[label] for label in labels if label in stored]
        report = {
            "experiment": "complete coarse atlas for the endpoint-octet cubic quotient",
            "source_endpoint_artifact": endpoint_report.as_posix(),
            "source_endpoint_artifact_sha256": endpoint_sha,
            "source_00010_artifact": report_00010.as_posix(),
            "source_00010_artifact_sha256": _sha256(report_00010),
            "source_00001_artifact": report_00001.as_posix(),
            "source_00001_artifact_sha256": _sha256(report_00001),
            "requested_boxes": labels,
            "boxes": rows,
            "completed_box_count": len(rows),
            "complete": len(rows) == 32,
            "passed": len(rows) == 32 and all(row["passed"] for row in rows),
            "resource_contract": resource,
            "scope_boundary": (
                "A pass certifies the endpoint-factor quotient on [0,1]^5. "
                "The endpoint faces and the final cubic assembly remain "
                "separate obligations."
            ),
        }
        _atomic_json(output, report)

    result = json.loads(output.read_text(encoding="utf-8"))
    result["artifact_sha256"] = _sha256(output)
    return result


def verify_report(report_path: Path) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    labels = [f"{index:05b}" for index in range(32)]
    if report.get("requested_boxes") != labels:
        failures.append("requested paths are not the complete coarse atlas")
    rows = report.get("boxes", [])
    if [row.get("bits") for row in rows] != labels:
        failures.append("stored rows are not the ordered coarse atlas")
    endpoint = Path(str(report["source_endpoint_artifact"]))
    try:
        endpoint_sha = _source_sha(endpoint)
    except (FileNotFoundError, AssertionError, KeyError, TypeError) as error:
        failures.append(f"endpoint source did not replay its identity: {error}")
    else:
        if endpoint_sha != report["source_endpoint_artifact_sha256"]:
            failures.append("endpoint source SHA-256 mismatch")
    delegated = {
        "00010": (
            Path(str(report["source_00010_artifact"])),
            report["source_00010_artifact_sha256"],
            verify_00010_report,
            "delegated_exact_face_plus_complement",
        ),
        "00001": (
            Path(str(report["source_00001_artifact"])),
            report["source_00001_artifact_sha256"],
            verify_nested_00001_report,
            "delegated_exact_complete_child_atlas",
        ),
    }
    rows_by_bits = {str(row.get("bits")): row for row in rows}
    for bits, (path, expected_sha, verifier, mechanism) in delegated.items():
        if _sha256(path) != expected_sha or not verifier(path)["verified"]:
            failures.append(f"delegated coarse box {bits} did not replay")
        row = rows_by_bits.get(bits, {})
        if (
            row.get("certificate_kind") != mechanism
            or row.get("source_certificate") != path.as_posix()
            or row.get("source_certificate_sha256") != expected_sha
            or not row.get("passed")
        ):
            failures.append(f"delegated coarse box {bits} metadata is malformed")
    for row in rows:
        bits = str(row.get("bits"))
        if bits in delegated:
            continue
        label = f"coarse box {bits}"
        before = len(failures)
        _check_stored_bernstein_audit(
            row.get("native_bernstein"), label=label, failures=failures
        )
        if len(failures) == before and not row.get("passed"):
            failures.append(f"coarse box {bits} has a false pass flag")
    if not report.get("complete") or not report.get("passed"):
        failures.append("coarse atlas top-level acceptance flags are false")
    return {
        "verified": not failures,
        "failures": failures,
        "box_count": len(rows),
        "verification_scope": (
            "hash-bound delegated replay and stored-summary integrity; regenerate "
            "the atlas to replay every Bernstein transform"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coefficient-dir", type=Path, default=DEFAULT_COEFFICIENT_DIR)
    parser.add_argument("--endpoint-report", type=Path, default=DEFAULT_ENDPOINT_REPORT)
    parser.add_argument("--report-00010", type=Path, default=DEFAULT_00010_REPORT)
    parser.add_argument("--report-00001", type=Path, default=DEFAULT_00001_REPORT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-entry-limit", type=int, default=500_000)
    parser.add_argument("--flint-threads", type=int, default=6)
    parser.add_argument("--verify", type=Path)
    arguments = parser.parse_args()
    if arguments.verify is not None:
        report = verify_report(arguments.verify)
    else:
        if arguments.output is None:
            parser.error("--output is required unless --verify is used")
        report = run(
            arguments.coefficient_dir,
            endpoint_report=arguments.endpoint_report,
            report_00010=arguments.report_00010,
            report_00001=arguments.report_00001,
            output=arguments.output,
            batch_entry_limit=arguments.batch_entry_limit,
            flint_threads=arguments.flint_threads,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
