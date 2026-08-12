"""Exact dyadic atlas for selected endpoint-octet cubic boundary cores."""

from __future__ import annotations

import argparse
import itertools
import json
from functools import reduce
from operator import mul
from pathlib import Path

from flint import ctx

from spin8_dirac_endpoint_octet_cubic import DEFAULT_COEFFICIENT_DIR, _build_cubic
from spin8_dirac_endpoint_octet_cubic_blowup import (
    _batched_bernstein_audit,
    _blowup_chart,
)
from spin8_dirac_endpoint_octet_cubic_tangent import _homogeneous_taylor
from spin8_dirac_endpoint_octet_quadratic import (
    _atomic_json,
    _restrict_half_box,
    _sha256,
)
from spin8_resource_limits import constrain_current_process


def _orient_factored_face(face):
    content, factors = face.factor()
    if any(int(exponent) != 1 for _factor, exponent in factors):
        raise AssertionError("expected square-free selected-face factors")
    monomials = [factor for factor, _exponent in factors if len(factor.to_dict()) == 1]
    cores = [factor for factor, _exponent in factors if len(factor.to_dict()) != 1]
    if not monomials or len(cores) != 1:
        raise AssertionError("expected monomial prefactors and one selected core")
    monomial = reduce(mul, monomials, face.context().constant(1))
    core = cores[0]
    oriented_core = core if int(content) > 0 else -core
    positive_content = abs(int(content))
    if face != positive_content * monomial * oriented_core:
        raise AssertionError("oriented selected-face factorization mismatch")
    monomial_audit = _batched_bernstein_audit(monomial, sample_limit=8)
    if monomial_audit["negative_scaled_coefficient_count"]:
        raise AssertionError("selected-face monomial is not Bernstein-nonnegative")
    return oriented_core, positive_content, monomial, monomial_audit


def _selected_face_core(cubic, *, pivot: int, face_kind: str = "ui"):
    _tangent, tangent_order = _homogeneous_taylor(cubic, max_order=12)
    quotient, _order, nonpivots, _shifted_term_count = _blowup_chart(
        cubic, pivot=pivot, expected_order=tangent_order
    )
    radius = quotient.context().gens()[0]
    exceptional = quotient.subs({"radius": 0})
    first_remainder = quotient - exceptional * (1 - radius) ** int(
        quotient.degrees()[0]
    )
    ui_axis = 1 + nonpivots.index(3)
    ui = quotient.context().gens()[ui_axis]
    ui_face = first_remainder.subs({str(ui): 0})
    if face_kind == "ui":
        return _orient_factored_face(ui_face)
    if face_kind != "double" or pivot != 0:
        raise ValueError("double face is defined only for pivot zero")
    ui_complement = first_remainder - ui_face * (1 - ui) ** int(
        quotient.degrees()[ui_axis]
    )
    ue_axis = 1 + nonpivots.index(1)
    ug_axis = 1 + nonpivots.index(2)
    ue = quotient.context().gens()[ue_axis]
    ug = quotient.context().gens()[ug_axis]
    double_face = ui_complement.subs({str(ue): 0, str(ug): 0})
    return _orient_factored_face(double_face)


def _full_bits(active_axes: list[int], active_bits: tuple[str, ...] | str) -> str:
    row = ["0"] * 5
    for axis, bit in zip(active_axes, active_bits, strict=True):
        row[axis] = bit
    return "".join(row)


def run(
    coefficient_dir: Path,
    *,
    pivot: int,
    output: Path,
    face_kind: str = "ui",
    parent_active_path: str | None = None,
    select_zero_axis: int | None = None,
    post_zero_active_path: str | None = None,
    flint_threads: int = 6,
) -> dict[str, object]:
    if pivot not in (0, 1, 2):
        raise ValueError("selected-face atlas applies only to pivots 0, 1, and 2")
    resource = constrain_current_process(workers=flint_threads)
    ctx.threads = flint_threads
    cubic, _tau, _forced_product, _variables = _build_cubic(coefficient_dir)
    core, content, monomial, monomial_audit = _selected_face_core(
        cubic, pivot=pivot, face_kind=face_kind
    )
    active_axes = [axis for axis, degree in enumerate(core.degrees()) if int(degree)]
    parent_rows: list[dict[str, str]] = []
    if parent_active_path:
        for active_bits in parent_active_path.split("/"):
            if len(active_bits) != len(active_axes) or set(active_bits) - {"0", "1"}:
                raise ValueError("parent component does not match active-axis count")
            bits = _full_bits(active_axes, active_bits)
            core = _restrict_half_box(core, bits)
            parent_rows.append({"active_bits": active_bits, "bits": bits})
    selected_zero_face = None
    if select_zero_axis is not None:
        if select_zero_axis not in active_axes:
            raise ValueError("selected zero axis must be active after factorization")
        variable = core.context().gens()[select_zero_axis]
        degree = int(core.degrees()[select_zero_axis])
        face = core.subs({str(variable): 0})
        complement = core - face * (1 - variable) ** degree
        if core != face * (1 - variable) ** degree + complement:
            raise AssertionError("secondary zero-face selector identity failed")
        core = face
        selected_zero_face = {
            "axis": select_zero_axis,
            "degree": degree,
            "selector": f"(1-axis{select_zero_axis})^{degree}",
            "identity_verified_exactly": True,
        }
        active_axes = [
            axis for axis, axis_degree in enumerate(core.degrees()) if int(axis_degree)
        ]
    post_zero_rows: list[dict[str, str]] = []
    if post_zero_active_path:
        if selected_zero_face is None:
            raise ValueError("post-zero path requires a selected zero face")
        for active_bits in post_zero_active_path.split("/"):
            if len(active_bits) != len(active_axes) or set(active_bits) - {"0", "1"}:
                raise ValueError("post-zero component does not match active axes")
            bits = _full_bits(active_axes, active_bits)
            core = _restrict_half_box(core, bits)
            post_zero_rows.append({"active_bits": active_bits, "bits": bits})
    report: dict[str, object] = {
        "experiment": "adjacent endpoint octet cubic selected-face dyadic atlas",
        "pivot_index": pivot,
        "pivot_deviation": ("ud", "ue", "ug")[pivot],
        "selected_face": face_kind,
        "positive_content": str(content),
        "monomial_power_term_count": len(monomial.to_dict()),
        "monomial_native_bernstein": monomial_audit,
        "oriented_core_power_term_count": len(core.to_dict()),
        "oriented_core_multidegree": list(map(int, core.degrees())),
        "active_axes": active_axes,
        "parent_path": parent_rows,
        "selected_zero_face": selected_zero_face,
        "post_zero_path": post_zero_rows,
        "boxes": [],
        "resource_contract": resource,
        "passed": False,
        "scope_boundary": (
            "This atlas certifies only one selected face core. The full chart "
            "also requires the exact selector identities and nonnegative "
            "complements recorded by the parent artifacts."
        ),
    }
    _atomic_json(output, report)
    boxes: list[dict[str, object]] = []
    for active_bits in itertools.product("01", repeat=len(active_axes)):
        bits = _full_bits(active_axes, active_bits)
        restricted = _restrict_half_box(core, bits)
        audit = _batched_bernstein_audit(restricted, sample_limit=64)
        boxes.append(
            {
                "bits": bits,
                "active_bits": "".join(active_bits),
                "power_term_count": len(restricted.to_dict()),
                "audit": audit,
                "passed": audit["negative_scaled_coefficient_count"] == 0,
            }
        )
        report["boxes"] = boxes
        report["passed"] = bool(
            len(boxes) == 2 ** len(active_axes) and all(row["passed"] for row in boxes)
        )
        _atomic_json(output, report)
    report["artifact_sha256"] = _sha256(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coefficient-dir", type=Path, default=DEFAULT_COEFFICIENT_DIR)
    parser.add_argument("--pivot", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--face-kind", choices=("ui", "double"), default="ui")
    parser.add_argument("--parent-active-path")
    parser.add_argument("--select-zero-axis", type=int, choices=range(5))
    parser.add_argument("--post-zero-active-path")
    parser.add_argument("--flint-threads", type=int, default=6)
    arguments = parser.parse_args()
    report = run(
        arguments.coefficient_dir,
        pivot=arguments.pivot,
        output=arguments.output,
        face_kind=arguments.face_kind,
        parent_active_path=arguments.parent_active_path,
        select_zero_axis=arguments.select_zero_axis,
        post_zero_active_path=arguments.post_zero_active_path,
        flint_threads=arguments.flint_threads,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
