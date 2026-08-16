"""Exact complete coordinate-boundary theorem for the octet determinant.

The adjacent endpoint-octet Schur determinant is a five-variable polynomial
in ``(ud, ue, ug, ui, y)``.  Its two ``y`` endpoint faces were certified
previously.  This module exposes a stronger structural fact: on every
coordinate face except ``y=1``, at most one nontrivial Klein-four radical
mode survives and the mixed radical product vanishes.  The four-mode
determinant therefore collapses to a perfect square.  The remaining ``y=1``
face is the already-certified identity ``Z=X^2``.

The result proves the complete coordinate boundary of the five-cube.  It does
not prove positivity at any point with all five coordinates strictly between
zero and one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp
from flint import ctx

from spin8_dirac_endpoint_octet import H0
from spin8_dirac_endpoint_octet_determinant import (
    DEFAULT_COEFFICIENT_DIR,
    _coefficient_hashes,
)
from spin8_dirac_endpoint_octet_quadratic import (
    _atomic_json,
    _build_z_coefficients,
    _sha256,
)
from spin8_resource_limits import constrain_current_process

DEFAULT_DETERMINANT_ARTIFACT = Path(
    "artifacts/spin8_dirac_endpoint_octet_determinant_20260812.json"
)
DEFAULT_ENDPOINT_ARTIFACT = Path("artifacts/spin8_dirac_endpoint_octet_20260807.json")
VARIABLE_NAMES = ("ud", "ue", "ug", "ui", "y")
SQUARE_FACES = tuple(
    (name, endpoint)
    for name in VARIABLE_NAMES
    for endpoint in (0, 1)
    if (name, endpoint) != ("y", 1)
)


def _mask_label(mask: tuple[int, ...]) -> str:
    return "".join(map(str, mask))


def _generic_square_identities() -> dict[str, bool]:
    """Verify the determinant collapse with zero or one active mode."""

    z0, q1, q2, q3, mixed = sp.symbols("z0 q1 q2 q3 mixed")
    modes = (q1, q2, q3)
    determinant = (
        z0**4
        - 2 * z0**2 * sum(modes)
        + 8 * mixed
        + sum(mode**2 for mode in modes)
        - 2 * (q1 * q2 + q1 * q3 + q2 * q3)
    )
    identities = {
        "zero_active_modes": sp.expand(
            determinant.subs({q1: 0, q2: 0, q3: 0, mixed: 0}) - z0**4
        )
        == 0
    }
    for index, mode in enumerate(modes):
        substitutions = {other: 0 for other in modes if other != mode}
        substitutions[mixed] = 0
        identities[f"active_mode_{index}"] = (
            sp.expand(determinant.subs(substitutions) - (z0**2 - mode) ** 2) == 0
        )
    return identities


def _tau(variables):
    ud, ue, ug, ui, y = variables
    return ud * ue * ug * ui * y**2 * (1 - ud) * (1 - ue) * (1 - ug) * (1 - ui)


def _square_face_rows(coefficient_dir: Path) -> list[dict[str, object]]:
    """Reconstruct the nine perfect-square boundary reductions exactly."""

    _z, forced_squares, variables = _build_z_coefficients(coefficient_dir)
    nontrivial = H0[1:]
    generic = _generic_square_identities()
    rows: list[dict[str, object]] = []
    for variable, endpoint in SQUARE_FACES:
        restricted = {
            mask: forced_squares[mask].subs({variable: endpoint}) for mask in nontrivial
        }
        active = [mask for mask in nontrivial if restricted[mask] != 0]
        zero = [mask for mask in nontrivial if restricted[mask] == 0]
        tau_zero = _tau(variables).subs({variable: endpoint}) == 0
        if len(active) > 1:
            raise AssertionError(
                f"face {variable}={endpoint} retained {len(active)} modes"
            )
        if not tau_zero:
            raise AssertionError(f"face {variable}={endpoint} retained tau")
        active_index = nontrivial.index(active[0]) if active else None
        generic_key = (
            "zero_active_modes"
            if active_index is None
            else f"active_mode_{active_index}"
        )
        identity = generic[generic_key]
        reduction = (
            "D=Z0^4"
            if active_index is None
            else f"D=(Z0^2-s_{_mask_label(active[0])}*Z_{_mask_label(active[0])}^2)^2"
        )
        rows.append(
            {
                "face": f"{variable}={endpoint}",
                "variable": variable,
                "endpoint": endpoint,
                "tau_restricts_to_zero": tau_zero,
                "active_nontrivial_modes": [_mask_label(mask) for mask in active],
                "zero_nontrivial_modes": [_mask_label(mask) for mask in zero],
                "active_mode_count": len(active),
                "restricted_active_forced_square_term_count": (
                    len(restricted[active[0]].to_dict()) if active else 0
                ),
                "generic_identity_key": generic_key,
                "determinant_reduction": reduction,
                "perfect_square_identity_verified_exactly": identity,
                "proved_nonnegative": bool(tau_zero and len(active) <= 1 and identity),
            }
        )
    return rows


def _delegated_y_one_certificate(
    determinant_artifact: Path, endpoint_artifact: Path
) -> dict[str, object]:
    """Verify the existing ``y=1`` ``Z=X^2`` dependency and its hashes."""

    determinant = json.loads(determinant_artifact.read_text(encoding="utf-8"))
    endpoint = json.loads(endpoint_artifact.read_text(encoding="utf-8"))
    row = determinant.get("y_one_face", {})
    determinant_passed = bool(
        determinant.get("passed")
        and determinant.get("endpoint_faces_proved")
        and row.get("proved_nonnegative")
        and row.get("exact_implication") == "y=1 implies Z=X^2 and det(Z)=det(X)^2>=0"
    )
    endpoint_hash = _sha256(endpoint_artifact)
    endpoint_passed = bool(
        endpoint.get("passed") and row.get("delegated_endpoint_sha256") == endpoint_hash
    )
    return {
        "face": "y=1",
        "identity": "Z=X^2 and det(Z)=det(X)^2",
        "determinant_artifact": determinant_artifact.name,
        "determinant_artifact_sha256": _sha256(determinant_artifact),
        "endpoint_artifact": endpoint_artifact.name,
        "endpoint_artifact_sha256": endpoint_hash,
        "determinant_endpoint_claim_replayed": determinant_passed,
        "endpoint_dependency_hash_replayed": endpoint_passed,
        "proved_nonnegative": bool(determinant_passed and endpoint_passed),
    }


def build_certificate(
    coefficient_dir: Path = DEFAULT_COEFFICIENT_DIR,
    *,
    determinant_artifact: Path = DEFAULT_DETERMINANT_ARTIFACT,
    endpoint_artifact: Path = DEFAULT_ENDPOINT_ARTIFACT,
) -> dict[str, object]:
    """Build the exact boundary certificate without expanding the full determinant."""

    identities = _generic_square_identities()
    square_faces = _square_face_rows(coefficient_dir)
    y_one = _delegated_y_one_certificate(determinant_artifact, endpoint_artifact)
    square_passed = bool(
        len(square_faces) == 9
        and all(row["proved_nonnegative"] for row in square_faces)
    )
    passed = bool(
        all(identities.values()) and square_passed and y_one["proved_nonnegative"]
    )
    return {
        "experiment": "adjacent endpoint-octet determinant coordinate boundary",
        "evidence_class": "exact algebraic reduction plus hash-bound exact dependency",
        "domain": "(ud,ue,ug,ui,y) in [0,1]^5",
        "coefficient_source_sha256": _coefficient_hashes(coefficient_dir),
        "generic_perfect_square_identities": identities,
        "perfect_square_faces": square_faces,
        "perfect_square_face_count": len(square_faces),
        "delegated_y_one_face": y_one,
        "coordinate_face_count": len(square_faces) + 1,
        "complete_coordinate_boundary_proved": passed,
        "global_determinant_interior_proved": False,
        "passed": passed,
        "scope_boundary": (
            "The certificate proves D>=0 whenever at least one of ud, ue, ug, "
            "ui, or y is a coordinate endpoint. It does not prove D>=0 when "
            "all five coordinates are strictly interior."
        ),
        "next_exact_gate": (
            "Certify the compact strict interior after removing all ten exact "
            "coordinate faces, with a nested blow-up at the y=1 equality corner."
        ),
    }


def verify_report(
    report_or_path,
    *,
    coefficient_dir: Path = DEFAULT_COEFFICIENT_DIR,
    determinant_artifact: Path = DEFAULT_DETERMINANT_ARTIFACT,
    endpoint_artifact: Path = DEFAULT_ENDPOINT_ARTIFACT,
) -> dict[str, object]:
    """Reconstruct every face reduction and compare it with a stored report."""

    if isinstance(report_or_path, (str, Path)):
        report = json.loads(Path(report_or_path).read_text(encoding="utf-8"))
    else:
        report = report_or_path
    replay = build_certificate(
        coefficient_dir,
        determinant_artifact=determinant_artifact,
        endpoint_artifact=endpoint_artifact,
    )
    failures: list[str] = []
    for key in (
        "coefficient_source_sha256",
        "generic_perfect_square_identities",
        "perfect_square_faces",
        "perfect_square_face_count",
        "delegated_y_one_face",
        "coordinate_face_count",
        "complete_coordinate_boundary_proved",
        "global_determinant_interior_proved",
        "passed",
    ):
        if report.get(key) != replay.get(key):
            failures.append(f"stored {key} disagrees with exact replay")
    if report.get("complete_coordinate_boundary_proved") is not True:
        failures.append("stored report does not prove the complete coordinate boundary")
    if report.get("global_determinant_interior_proved") is not False:
        failures.append("stored report overclaims determinant interior positivity")
    return {
        "verified": not failures,
        "failures": failures,
        "replayed_square_face_count": replay["perfect_square_face_count"],
        "replayed_coordinate_face_count": replay["coordinate_face_count"],
        "complete_coordinate_boundary_proved": replay[
            "complete_coordinate_boundary_proved"
        ],
        "global_determinant_interior_proved": False,
    }


def run(
    coefficient_dir: Path,
    *,
    output: Path,
    determinant_artifact: Path = DEFAULT_DETERMINANT_ARTIFACT,
    endpoint_artifact: Path = DEFAULT_ENDPOINT_ARTIFACT,
    flint_threads: int = 6,
) -> dict[str, object]:
    """Build and write the exact complete coordinate-boundary artifact."""

    if not 1 <= flint_threads <= 6:
        raise ValueError("FLINT thread count must be between one and six")
    resource = constrain_current_process(workers=flint_threads)
    ctx.threads = flint_threads
    report = build_certificate(
        coefficient_dir,
        determinant_artifact=determinant_artifact,
        endpoint_artifact=endpoint_artifact,
    )
    report["resource_contract"] = resource
    _atomic_json(output, report)
    report["artifact_sha256"] = _sha256(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coefficient-dir", type=Path, default=DEFAULT_COEFFICIENT_DIR)
    parser.add_argument(
        "--determinant-artifact", type=Path, default=DEFAULT_DETERMINANT_ARTIFACT
    )
    parser.add_argument(
        "--endpoint-artifact", type=Path, default=DEFAULT_ENDPOINT_ARTIFACT
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flint-threads", type=int, default=6)
    arguments = parser.parse_args()
    report = run(
        arguments.coefficient_dir,
        output=arguments.output,
        determinant_artifact=arguments.determinant_artifact,
        endpoint_artifact=arguments.endpoint_artifact,
        flint_threads=arguments.flint_threads,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
