"""Exact permutation audit between the two middle cubic blow-up cores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flint import ctx

from spin8_dirac_endpoint_octet_cubic import DEFAULT_COEFFICIENT_DIR, _build_cubic
from spin8_dirac_endpoint_octet_cubic_boundary_atlas import _selected_face_core
from spin8_dirac_endpoint_octet_quadratic import _atomic_json, _sha256
from spin8_resource_limits import constrain_current_process


def run(
    coefficient_dir: Path, *, output: Path, flint_threads: int = 6
) -> dict[str, object]:
    resource = constrain_current_process(workers=flint_threads)
    ctx.threads = flint_threads
    cubic, _tau, _forced_product, _variables = _build_cubic(coefficient_dir)
    ue_core, _ue_content, _ue_monomial, _ue_audit = _selected_face_core(cubic, pivot=1)
    ug_core, _ug_content, _ug_monomial, _ug_audit = _selected_face_core(cubic, pivot=2)
    radius, x0, x1, x2, x3 = ue_core.context().gens()
    identity = ue_core == ug_core
    swapped = ue_core == ug_core.compose(radius, x1, x0, x2, x3)
    report = {
        "experiment": "adjacent endpoint octet cubic middle-chart symmetry audit",
        "ue_core_power_term_count": len(ue_core.to_dict()),
        "ug_core_power_term_count": len(ug_core.to_dict()),
        "multidegrees_equal": ue_core.degrees() == ug_core.degrees(),
        "identity_coordinates_equal": identity,
        "degree18_ratio_swap_equal": swapped,
        "passed": bool(identity or swapped),
        "accepted_map": ("identity" if identity else "swap_x0_x1" if swapped else None),
        "scope_boundary": (
            "A failed audit preserves independent chart obligations; it is not "
            "evidence against positivity of either core."
        ),
        "resource_contract": resource,
    }
    _atomic_json(output, report)
    report["artifact_sha256"] = _sha256(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coefficient-dir", type=Path, default=DEFAULT_COEFFICIENT_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flint-threads", type=int, default=6)
    arguments = parser.parse_args()
    report = run(
        arguments.coefficient_dir,
        output=arguments.output,
        flint_threads=arguments.flint_threads,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
