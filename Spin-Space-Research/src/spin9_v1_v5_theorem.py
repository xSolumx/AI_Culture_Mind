"""Assemble the exact coupled Spin(9) finite-radius theorem.

This lightweight assembler joins two independently replayable proof objects:
the characteristic-zero raw determinant identity and the global projective
Bernstein cover.  It deliberately distinguishes a theorem on the complete
``V1 + V5`` Grassmann normal slice from the still-open unrestricted quotient
and from exact optimality of the algebraic symmetric candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from spin9_v1_v5_reconstruction import ROOT

DEFAULT_GLOBAL_ARTIFACT = ROOT / "artifacts" / "spin9_v1_v5_global_20260812.json"
DEFAULT_CHAR0_ARTIFACT = ROOT / "artifacts" / "spin9_v1_v5_char0_20260812.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "spin9_v1_v5_theorem_20260812.json"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def certificate(
    global_artifact: Path = DEFAULT_GLOBAL_ARTIFACT,
    char0_artifact: Path = DEFAULT_CHAR0_ARTIFACT,
) -> dict[str, object]:
    """Validate both proof layers and return the promoted theorem receipt."""

    global_report = json.loads(global_artifact.read_text(encoding="utf-8"))
    char0_report = json.loads(char0_artifact.read_text(encoding="utf-8"))
    same_coefficients = (
        global_report["coefficient_artifact"]
        == char0_report["coefficient_artifact"]
    )
    positivity_passed = bool(
        global_report["passed"]
        and global_report["reconstructed_rational_function_bound_certified"]
    )
    identity_passed = bool(
        char0_report["passed"]
        and char0_report["characteristic_zero_identity_certified"]
        and char0_report["full_required_prime_set_checked"]
    )
    passed = same_coefficients and positivity_passed and identity_passed
    return {
        "schema_version": 1,
        "claim_scope": (
            "complete finite-radius coupled V1+V5 Grassmann normal slice at "
            "the Cayley-null Spin(9) plane"
        ),
        "theorem": (
            "det I(P)/det I(P_cayley-null) <= 21/20 for every graph in the "
            "coupled V1+V5 normal slice"
        ),
        "orbit_domain": "x real, p>=0, 27*y^2<=2*p^3",
        "proof_layers": {
            "raw_characteristic_zero_identity": {
                "artifact": char0_artifact.name,
                "sha256": _file_sha256(char0_artifact),
                "passed": identity_passed,
            },
            "global_projective_positivity": {
                "artifact": global_artifact.name,
                "sha256": _file_sha256(global_artifact),
                "passed": positivity_passed,
            },
        },
        "same_coefficient_artifact_used_by_both_layers": same_coefficients,
        "finite_radius_coupled_slice_bound_certified": passed,
        "global_on_stated_slice": passed,
        "algebraic_symmetric_candidate_optimality_certified": False,
        "second_v5_copy_certified": False,
        "global_grassmann_quotient_certified": False,
        "global_rank_three_optimum_certified": False,
        "passed": passed,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-artifact", type=Path, default=DEFAULT_GLOBAL_ARTIFACT)
    parser.add_argument("--char0-artifact", type=Path, default=DEFAULT_CHAR0_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    report = certificate(arguments.global_artifact, arguments.char0_artifact)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
