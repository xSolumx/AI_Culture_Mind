"""Exact dyadic atlas for the cubic two-endpoint quotient."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

from flint import ctx

from spin8_dirac_endpoint_octet_cubic import DEFAULT_COEFFICIENT_DIR, _build_cubic
from spin8_dirac_endpoint_octet_cubic_blowup import _batched_bernstein_audit
from spin8_dirac_endpoint_octet_cubic_corner_certificate import (
    verify as verify_corner_certificate,
)
from spin8_dirac_endpoint_octet_quadratic import (
    _atomic_json,
    _restrict_half_box,
    _sha256,
)
from spin8_resource_limits import constrain_current_process

DEFAULT_ENDPOINT_REPORT = Path(
    "artifacts/spin8_dirac_endpoint_octet_cubic_endpoints_20260808.json"
)
DEFAULT_CORNER_REPORT = Path(
    "artifacts/spin8_dirac_endpoint_octet_cubic_corner_20260810.json"
)


def _build_quotient(coefficient_dir: Path):
    cubic, _tau, _forced_product, _variables = _build_cubic(coefficient_dir)
    y = cubic.context().gens()[4]
    degree = int(cubic.degrees()[4])
    face_zero = cubic.subs({"y": 0})
    face_one = cubic.subs({"y": 1})
    remainder = cubic - face_zero * (1 - y) ** degree - face_one * y**degree
    quotient, division_remainder = divmod(remainder, y * (1 - y))
    if division_remainder != 0 or remainder != y * (1 - y) * quotient:
        raise AssertionError("the exact endpoint-factor identity failed")
    return quotient


def _source_sha(endpoint_report: Path) -> str:
    if not endpoint_report.exists():
        raise FileNotFoundError(endpoint_report)
    report = json.loads(endpoint_report.read_text(encoding="utf-8"))
    certificate = report.get("dual_selector", {}).get(
        "endpoint_factor_certificate"
    )
    if not certificate or not certificate.get("identity_verified_exactly"):
        raise AssertionError("endpoint-factor source did not verify its identity")
    return _sha256(endpoint_report)


def _restrict_path(polynomial, path: str):
    result = polynomial
    for bits in path.split("/"):
        if len(bits) != 5 or set(bits) - {"0", "1"}:
            raise ValueError(f"invalid dyadic box segment {bits!r}")
        result = _restrict_half_box(result, bits)
    return result


def _check_stored_bernstein_audit(
    audit: object, *, label: str, failures: list[str]
) -> None:
    """Check the internal arithmetic contract of a stored Bernstein summary.

    This is deliberately an integrity check, not a reconstruction of the
    underlying transform.  The expensive replay remains the atlas generator
    itself; this helper makes every quantity trusted by the compact verifier
    explicit and mutually consistent.
    """

    if not isinstance(audit, dict):
        failures.append(f"{label} has no Bernstein audit")
        return
    try:
        multidegree = [int(value) for value in audit["multidegree"]]
        tensor_shape = [int(value) for value in audit["tensor_shape"]]
        coefficient_count = int(audit["coefficient_count"])
        negative_count = int(audit["negative_scaled_coefficient_count"])
        zero_count = int(audit["zero_scaled_coefficient_count"])
        minimum = int(audit["minimum_scaled_coefficient"])
        scales = [int(value) for value in audit["axis_positive_scales"]]
    except (KeyError, TypeError, ValueError) as error:
        failures.append(f"{label} has a malformed Bernstein audit: {error}")
        return
    if tensor_shape != [degree + 1 for degree in multidegree]:
        failures.append(f"{label} tensor shape disagrees with its multidegree")
    if coefficient_count != math.prod(tensor_shape):
        failures.append(f"{label} coefficient count disagrees with its tensor shape")
    if len(scales) != len(multidegree) or any(scale <= 0 for scale in scales):
        failures.append(f"{label} has invalid Bernstein scaling factors")
    if negative_count != 0 or minimum < 0:
        failures.append(f"{label} contains a negative Bernstein coefficient")
    if zero_count < 0 or zero_count > coefficient_count:
        failures.append(f"{label} has an invalid zero-coefficient count")
    if (minimum == 0) != (zero_count > 0):
        failures.append(f"{label} minimum and zero-coefficient count disagree")
    if audit.get("negative_rows_sample"):
        failures.append(f"{label} stores negative rows despite a nonnegative count")


def run(
    coefficient_dir: Path,
    *,
    output: Path,
    endpoint_report: Path = DEFAULT_ENDPOINT_REPORT,
    corner_report: Path | None = None,
    delegate_corner_path: str | None = None,
    boxes: list[str],
    batch_entry_limit: int = 500_000,
    flint_threads: int = 6,
) -> dict[str, object]:
    if not 1 <= flint_threads <= 6:
        raise ValueError("FLINT thread count must be between one and six")
    for path in boxes:
        for bits in path.split("/"):
            if len(bits) != 5 or set(bits) - {"0", "1"}:
                raise ValueError(f"invalid dyadic box segment {bits!r}")
    resource = constrain_current_process(workers=flint_threads)
    ctx.threads = flint_threads
    source_sha = _source_sha(endpoint_report)
    corner_sha = None
    if delegate_corner_path is not None:
        if corner_report is None:
            raise ValueError("corner delegation requires a corner report")
        if delegate_corner_path != "00001/00001":
            raise ValueError("the finite corner delegates only path 00001/00001")
        corner_verification = verify_corner_certificate(corner_report)
        if not corner_verification["verified"]:
            raise AssertionError("finite-radius corner certificate did not replay")
        corner_sha = _sha256(corner_report)
    quotient = _build_quotient(coefficient_dir)

    stored: dict[str, dict[str, object]] = {}
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("source_endpoint_artifact_sha256") == source_sha:
            stored = {row["bits"]: row for row in previous.get("boxes", [])}

    if delegate_corner_path is not None:
        stored[delegate_corner_path] = {
            "bits": delegate_corner_path,
            "interval_convention": "0=[0,1/2], 1=[1/2,1]",
            "certificate_kind": "delegated_exact_finite_radius_corner",
            "source_corner_artifact": corner_report.as_posix(),
            "source_corner_artifact_sha256": corner_sha,
            "domain_containment": (
                "ud,ue,ug,ui in [0,1/4] and y in [3/4,1], hence 1-y in [0,1/4]"
            ),
            "passed": True,
        }

    for path in boxes:
        if path in stored:
            continue
        restricted = _restrict_path(quotient, path)
        audit = _batched_bernstein_audit(
            restricted,
            sample_limit=64,
            batch_entry_limit=batch_entry_limit,
        )
        stored[path] = {
            "bits": path,
            "interval_convention": "0=[0,1/2], 1=[1/2,1]",
            "power_term_count": len(restricted.to_dict()),
            "native_bernstein": audit,
            "passed": audit["negative_scaled_coefficient_count"] == 0,
        }
        ordered = [stored[key] for key in boxes if key in stored]
        checkpoint = {
            "experiment": "adjacent endpoint octet cubic quotient dyadic atlas",
            "source_endpoint_artifact": endpoint_report.as_posix(),
            "source_endpoint_artifact_sha256": source_sha,
            "source_corner_artifact": (
                corner_report.as_posix() if corner_report is not None else None
            ),
            "source_corner_artifact_sha256": corner_sha,
            "requested_boxes": boxes,
            "completed_box_count": len(ordered),
            "boxes": ordered,
            "complete": len(ordered) == len(boxes),
            "passed": len(ordered) == len(boxes)
            and all(row["passed"] for row in ordered),
            "resource_contract": resource,
        }
        _atomic_json(output, checkpoint)

    report = json.loads(output.read_text(encoding="utf-8"))
    report["artifact_sha256"] = _sha256(output)
    return report


def verify_nested_00001_report(report_path: Path) -> dict[str, object]:
    """Verify the exhaustive child cover of first-level box ``00001``."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    expected = [f"00001/{index:05b}" for index in range(32)]
    if list(report.get("requested_boxes", [])) != expected:
        failures.append("requested paths are not the frozen 32-child cover")
    rows = report.get("boxes", [])
    if [row.get("bits") for row in rows] != expected:
        failures.append("stored rows are not the ordered 32-child cover")
    endpoint_path = Path(str(report["source_endpoint_artifact"]))
    try:
        endpoint_sha = _source_sha(endpoint_path)
    except (FileNotFoundError, AssertionError, KeyError, TypeError) as error:
        failures.append(f"endpoint source did not replay its identity: {error}")
    else:
        if endpoint_sha != report["source_endpoint_artifact_sha256"]:
            failures.append("endpoint source SHA-256 mismatch")
    delegated_rows = [
        row
        for row in rows
        if row.get("certificate_kind") == "delegated_exact_finite_radius_corner"
    ]
    if len(delegated_rows) != 1 or delegated_rows[0].get("bits") != "00001/00001":
        failures.append("the unique corner delegation is missing or misplaced")
    else:
        delegated = delegated_rows[0]
        corner_path = Path(str(delegated["source_corner_artifact"]))
        if (
            report.get("source_corner_artifact") != corner_path.as_posix()
            or report.get("source_corner_artifact_sha256")
            != delegated["source_corner_artifact_sha256"]
        ):
            failures.append("corner delegation disagrees with top-level metadata")
        elif _sha256(corner_path) != delegated["source_corner_artifact_sha256"]:
            failures.append("corner source SHA-256 mismatch")
        elif not verify_corner_certificate(corner_path)["verified"]:
            failures.append("corner source did not replay")
    for row in rows:
        if row.get("certificate_kind") == "delegated_exact_finite_radius_corner":
            continue
        label = f"nested child {row.get('bits')}"
        before = len(failures)
        _check_stored_bernstein_audit(
            row.get("native_bernstein"), label=label, failures=failures
        )
        if len(failures) == before and not row.get("passed"):
            failures.append(f"nested child {row.get('bits')} has a false pass flag")
    if not report.get("complete") or not report.get("passed"):
        failures.append("nested atlas top-level acceptance flags are false")
    return {
        "verified": not failures,
        "failures": failures,
        "box_count": len(rows),
        "delegated_box_count": len(delegated_rows),
        "verification_scope": (
            "hash-bound source replay and stored-summary integrity; regenerate "
            "the atlas to replay every Bernstein transform"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coefficient-dir", type=Path, default=DEFAULT_COEFFICIENT_DIR
    )
    parser.add_argument("--endpoint-report", type=Path, default=DEFAULT_ENDPOINT_REPORT)
    parser.add_argument("--corner-report", type=Path, default=DEFAULT_CORNER_REPORT)
    parser.add_argument("--delegate-corner-path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flint-threads", type=int, default=6)
    parser.add_argument("--batch-entry-limit", type=int, default=500_000)
    parser.add_argument(
        "--box",
        action="append",
        dest="boxes",
        help="five-bit dyadic box; repeat to evaluate several boxes",
    )
    parser.add_argument("--all-boxes", action="store_true")
    arguments = parser.parse_args()
    if arguments.all_boxes:
        boxes = ["".join(bits) for bits in itertools.product("01", repeat=5)]
    else:
        boxes = arguments.boxes or ["00010", "00001"]
    report = run(
        arguments.coefficient_dir,
        output=arguments.output,
        endpoint_report=arguments.endpoint_report,
        corner_report=arguments.corner_report,
        delegate_corner_path=arguments.delegate_corner_path,
        boxes=boxes,
        batch_entry_limit=arguments.batch_entry_limit,
        flint_threads=arguments.flint_threads,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
