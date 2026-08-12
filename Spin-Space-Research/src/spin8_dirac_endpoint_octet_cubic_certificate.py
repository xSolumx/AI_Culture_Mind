"""Final exact assembly for the endpoint-octet cubic principal minor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flint import ctx

from spin8_dirac_endpoint_octet_cubic import (
    DEFAULT_COEFFICIENT_DIR,
    DEFAULT_ENDPOINT_ARTIFACT,
    _build_cubic,
)
from spin8_dirac_endpoint_octet_cubic_atlas import (
    DEFAULT_ENDPOINT_REPORT,
    _source_sha,
)
from spin8_dirac_endpoint_octet_cubic_coarse_atlas import (
    verify_report as verify_coarse_atlas,
)
from spin8_dirac_endpoint_octet_cubic_yzero import (
    verify_report as verify_yzero_report,
)
from spin8_dirac_endpoint_octet_quadratic import _atomic_json, _sha256
from spin8_resource_limits import constrain_current_process

DEFAULT_YZERO_REPORT = Path(
    "artifacts/spin8_dirac_endpoint_octet_cubic_yzero_20260808.json"
)
DEFAULT_COARSE_REPORT = Path(
    "artifacts/spin8_dirac_endpoint_octet_cubic_coarse_atlas_20260811.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/spin8_dirac_endpoint_octet_cubic_certificate_20260811.json"
)


def _endpoint_gate(path: Path) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    x_block = report.get("x_block", {})
    passed = bool(
        report.get("passed")
        and x_block.get("passed")
        and x_block.get("y_one_klein_certificate_passed")
        and report.get("exact_schur_reduction")
        == "K8>=0 iff X>=0 and Z=X^2-(1-y^2)R^2>=0"
    )
    return {
        "passed": passed,
        "endpoint_artifact_passed": bool(report.get("passed")),
        "x_block_passed": bool(x_block.get("passed")),
        "y_one_klein_certificate_passed": bool(
            x_block.get("y_one_klein_certificate_passed")
        ),
        "exact_implication": "y=1 implies Z=X^2; X positive semidefinite",
    }


def _assembly_identity(coefficient_dir: Path) -> dict[str, object]:
    cubic, tau, forced_product, variables = _build_cubic(coefficient_dir)
    y = variables[4]
    degree = int(cubic.degrees()[4])
    face_zero = cubic.subs({"y": 0})
    face_one = cubic.subs({"y": 1})
    remainder = cubic - face_zero * (1 - y) ** degree - face_one * y**degree
    quotient, division_remainder = divmod(remainder, y * (1 - y))
    identity = bool(
        division_remainder == 0
        and cubic
        == face_zero * (1 - y) ** degree + face_one * y**degree + y * (1 - y) * quotient
    )
    return {
        "identity": "C=C|y=0*(1-y)^36+C|y=1*y^36+y*(1-y)*Q",
        "identity_verified_exactly": identity,
        "zero_division_remainder": division_remainder == 0,
        "forced_radical_identity_verified_exactly": forced_product == tau**2,
        "y_degree": degree,
        "cubic_power_term_count": len(cubic.to_dict()),
        "cubic_multidegree": list(map(int, cubic.degrees())),
        "quotient_power_term_count": len(quotient.to_dict()),
        "quotient_multidegree": list(map(int, quotient.degrees())),
        "passed": bool(identity and forced_product == tau**2 and degree == 36),
    }


def run(
    coefficient_dir: Path,
    *,
    output: Path,
    yzero_report: Path = DEFAULT_YZERO_REPORT,
    endpoint_report: Path = DEFAULT_ENDPOINT_REPORT,
    endpoint_artifact: Path = DEFAULT_ENDPOINT_ARTIFACT,
    coarse_report: Path = DEFAULT_COARSE_REPORT,
    flint_threads: int = 6,
) -> dict[str, object]:
    if not 1 <= flint_threads <= 6:
        raise ValueError("FLINT thread count must be between one and six")
    resource = constrain_current_process(workers=flint_threads)
    ctx.threads = flint_threads

    yzero_payload = json.loads(yzero_report.read_text(encoding="utf-8"))
    yzero_verification = verify_yzero_report(yzero_payload)
    coarse_verification = verify_coarse_atlas(coarse_report)
    endpoint_gate = _endpoint_gate(endpoint_artifact)
    endpoint_factor_sha = _source_sha(endpoint_report)
    assembly = _assembly_identity(coefficient_dir)

    gates = {
        "y_zero_face": {
            "source": yzero_report.as_posix(),
            "source_sha256": _sha256(yzero_report),
            **yzero_verification,
            "passed": bool(yzero_verification["verified"]),
        },
        "y_one_face": {
            "source": endpoint_artifact.as_posix(),
            "source_sha256": _sha256(endpoint_artifact),
            **endpoint_gate,
        },
        "endpoint_factor_identity_source": {
            "source": endpoint_report.as_posix(),
            "source_sha256": endpoint_factor_sha,
            "passed": True,
        },
        "complete_quotient_atlas": {
            "source": coarse_report.as_posix(),
            "source_sha256": _sha256(coarse_report),
            **coarse_verification,
            "passed": bool(coarse_verification["verified"]),
        },
        "final_assembly": assembly,
    }
    passed = all(bool(gate["passed"]) for gate in gates.values())
    report = {
        "experiment": "complete adjacent endpoint-octet cubic certificate",
        "domain": "(ud,ue,ug,ui,y) in [0,1]^5",
        "theorem": "C(ud,ue,ug,ui,y) >= 0 on the complete five-cube",
        "proof_decomposition": (
            "The endpoint selectors, y*(1-y), and the complete quotient atlas "
            "are nonnegative on the domain."
        ),
        "exact_gates": gates,
        "passed": passed,
        "trust_boundary": (
            "The final identity and forced-radical cancellation are recomputed "
            "in characteristic zero. Source certificate verifiers replay their "
            "load-bearing algebra; full Bernstein-transform replay uses the "
            "documented expensive source harnesses."
        ),
        "scope_boundary": (
            "This proves the cubic principal-minor family on the adjacent "
            "endpoint octet. The fourth-order determinant of the second Schur "
            "block, the complete endpoint octet, and unrestricted Dirac--Gram "
            "positivity remain separate obligations."
        ),
        "resource_contract": resource,
    }
    _atomic_json(output, report)
    report["artifact_sha256"] = _sha256(output)
    return report


def verify_report(
    report_path: Path, *, coefficient_dir: Path | None = None
) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    gates = report.get("exact_gates", {})
    required = {
        "y_zero_face",
        "y_one_face",
        "endpoint_factor_identity_source",
        "complete_quotient_atlas",
        "final_assembly",
    }
    if set(gates) != required:
        failures.append("final certificate does not contain the five frozen gates")
        return {"verified": False, "failures": failures}

    yzero = gates["y_zero_face"]
    yzero_path = Path(str(yzero["source"]))
    if _sha256(yzero_path) != yzero["source_sha256"]:
        failures.append("y-zero source SHA-256 mismatch")
    else:
        payload = json.loads(yzero_path.read_text(encoding="utf-8"))
        if not verify_yzero_report(payload)["verified"]:
            failures.append("y-zero source did not replay")

    endpoint = gates["y_one_face"]
    endpoint_path = Path(str(endpoint["source"]))
    if _sha256(endpoint_path) != endpoint["source_sha256"]:
        failures.append("y-one endpoint source SHA-256 mismatch")
    elif not _endpoint_gate(endpoint_path)["passed"]:
        failures.append("y-one endpoint theorem did not replay")

    factor = gates["endpoint_factor_identity_source"]
    factor_path = Path(str(factor["source"]))
    try:
        factor_sha = _source_sha(factor_path)
    except (FileNotFoundError, AssertionError, KeyError, TypeError) as error:
        failures.append(f"endpoint-factor source failed: {error}")
    else:
        if factor_sha != factor["source_sha256"]:
            failures.append("endpoint-factor source SHA-256 mismatch")

    coarse = gates["complete_quotient_atlas"]
    coarse_path = Path(str(coarse["source"]))
    if _sha256(coarse_path) != coarse["source_sha256"]:
        failures.append("coarse-atlas source SHA-256 mismatch")
    elif not verify_coarse_atlas(coarse_path)["verified"]:
        failures.append("coarse-atlas source did not replay")

    assembly = gates["final_assembly"]
    if coefficient_dir is not None:
        recomputed = _assembly_identity(coefficient_dir)
        for key in (
            "identity_verified_exactly",
            "zero_division_remainder",
            "forced_radical_identity_verified_exactly",
            "y_degree",
            "cubic_power_term_count",
            "cubic_multidegree",
            "quotient_power_term_count",
            "quotient_multidegree",
            "passed",
        ):
            if recomputed[key] != assembly.get(key):
                failures.append(f"final assembly mismatch in {key}")
    elif not assembly.get("passed"):
        failures.append("stored final assembly gate is false")

    if not report.get("passed") or any(
        not gate.get("passed") for gate in gates.values()
    ):
        failures.append("final or component pass flag is false")
    return {
        "verified": not failures,
        "failures": failures,
        "recomputed_characteristic_zero_identity": coefficient_dir is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coefficient-dir", type=Path, default=DEFAULT_COEFFICIENT_DIR)
    parser.add_argument("--yzero-report", type=Path, default=DEFAULT_YZERO_REPORT)
    parser.add_argument("--endpoint-report", type=Path, default=DEFAULT_ENDPOINT_REPORT)
    parser.add_argument(
        "--endpoint-artifact", type=Path, default=DEFAULT_ENDPOINT_ARTIFACT
    )
    parser.add_argument("--coarse-report", type=Path, default=DEFAULT_COARSE_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--flint-threads", type=int, default=6)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--recompute-identity", action="store_true")
    arguments = parser.parse_args()
    if arguments.verify is not None:
        report = verify_report(
            arguments.verify,
            coefficient_dir=(
                arguments.coefficient_dir if arguments.recompute_identity else None
            ),
        )
    else:
        report = run(
            arguments.coefficient_dir,
            output=arguments.output,
            yzero_report=arguments.yzero_report,
            endpoint_report=arguments.endpoint_report,
            endpoint_artifact=arguments.endpoint_artifact,
            coarse_report=arguments.coarse_report,
            flint_threads=arguments.flint_threads,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
