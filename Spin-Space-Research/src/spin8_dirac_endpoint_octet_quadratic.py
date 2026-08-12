"""Exact native-Bernstein audit for one adjacent-endpoint Schur quadratic.

The three invocations are intentionally independent.  This keeps exact
working sets bounded and makes a partial campaign crash-resilient.  Negative
Bernstein controls are localized but are never interpreted as negative
polynomial values.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import tempfile
from collections import Counter
from math import comb
from pathlib import Path

from flint import ctx, fmpz, fmpz_mpoly_ctx

from spin8_dirac_endpoint_octet import (
    H0,
    H1,
    TRIVIAL,
    _convolution,
    _forced_square,
    _to_endpoint_chart,
    _z_quadratic_corner_certificate,
    _zero,
)
from spin8_dirac_final_residual import exact_full_chart_sign_certificate
from spin8_dirac_unrestricted_core import (
    _bernstein_matrix,
    _read_integer_polynomial,
    _transform_axis,
)
from spin8_resource_limits import constrain_current_process


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _build_z_coefficients(coefficient_dir: Path):
    """Reconstruct the four radical-free Klein-four coefficients of ``Z``."""

    chart = exact_full_chart_sign_certificate()
    if not chart["passed"]:
        raise AssertionError("full-chart sign certificate failed")
    rows = chart["chart_characters"]
    masks = tuple(tuple(row["lower_mask"]) for row in rows)
    complements = {
        tuple(row["lower_mask"]): tuple(row["complement_mask"]) for row in rows
    }

    context7 = fmpz_mpoly_ctx.get(["ua", "ud", "ue", "ug", "uh", "ui", "z"])
    context5 = fmpz_mpoly_ctx.get(["ud", "ue", "ug", "ui", "y"])
    variables = context5.gens()
    zero = _zero(context5)
    residuals = {
        mask: _to_endpoint_chart(
            _read_integer_polynomial(context7, coefficient_dir, mask), context5
        )
        for mask in H0 + H1
    }
    forced_squares = {
        mask: _forced_square(context5, variables, mask, complements[mask])
        for mask in masks
    }
    convolution_x = {
        target: _convolution(
            H0,
            target,
            residuals,
            complements,
            variables,
            zero,
            h_zero_subgroup=True,
        )
        for target in H0
    }
    convolution_r = {
        target: _convolution(
            H1,
            target,
            residuals,
            complements,
            variables,
            zero,
            h_zero_subgroup=False,
        )
        for target in H0
    }
    z = {
        target: convolution_x[target] - (1 - variables[4] ** 2) * convolution_r[target]
        for target in H0
    }
    return z, forced_squares, variables


def _build_quadratic(coefficient_dir: Path, minor_index: int):
    z, forced_squares, variables = _build_z_coefficients(coefficient_dir)
    mask = H0[1:][minor_index]
    polynomial = z[TRIVIAL] ** 2 - forced_squares[mask] * z[mask] ** 2
    return polynomial, mask, variables


def _native_bernstein_audit(polynomial, *, sample_limit: int = 256):
    degrees = tuple(int(value) for value in polynomial.degrees())
    shape = tuple(degree + 1 for degree in degrees)
    strides = tuple(math.prod(shape[axis + 1 :]) for axis in range(len(shape)))
    values = [fmpz(0)] * math.prod(shape)
    for powers, coefficient in polynomial.to_dict().items():
        flat = sum(
            power * stride for power, stride in zip(powers, strides, strict=True)
        )
        values[flat] = fmpz(coefficient)

    scales = []
    for axis, degree in enumerate(degrees):
        matrix, scale = _bernstein_matrix(degree)
        values = _transform_axis(values, axis=axis, shape=shape, matrix=matrix)
        scales.append(int(scale))

    negative_count = 0
    zero_count = 0
    negative_samples = []
    boundary_histogram: Counter[str] = Counter()
    minimum = None
    minimum_index = None
    for flat, value in enumerate(values):
        if minimum is None or value < minimum:
            minimum = value
            minimum_index = flat
        if value == 0:
            zero_count += 1
            continue
        if value > 0:
            continue
        negative_count += 1
        remainder = flat
        index = []
        boundary = []
        for axis, stride in enumerate(strides):
            coordinate, remainder = divmod(remainder, stride)
            index.append(coordinate)
            # A zero-degree axis is not a geometric boundary: every control
            # has coordinate zero there.  Recording it as a face obscures the
            # true support of a failed Bernstein certificate.
            if degrees[axis] == 0:
                continue
            if coordinate == 0:
                boundary.append(f"{axis}:0")
            elif coordinate == degrees[axis]:
                boundary.append(f"{axis}:1")
        boundary_histogram[",".join(boundary) or "interior-control"] += 1
        if len(negative_samples) < sample_limit:
            negative_samples.append(
                {"bernstein_index": index, "scaled_coefficient": str(value)}
            )

    def unravel(flat: int | None) -> list[int] | None:
        if flat is None:
            return None
        result = []
        remainder = flat
        for stride in strides:
            coordinate, remainder = divmod(remainder, stride)
            result.append(coordinate)
        return result

    return {
        "multidegree": list(degrees),
        "tensor_shape": list(shape),
        "coefficient_count": len(values),
        "axis_positive_scales": scales,
        "minimum_scaled_coefficient": str(minimum),
        "minimum_bernstein_index": unravel(minimum_index),
        "negative_scaled_coefficient_count": negative_count,
        "zero_scaled_coefficient_count": zero_count,
        "negative_boundary_histogram": dict(sorted(boundary_histogram.items())),
        "negative_rows_sample": negative_samples,
        "negative_rows_sample_limit": sample_limit,
    }


def _restrict_half_box(polynomial, bits: str):
    """Map one dyadic half-box back to the unit cube with integer scaling."""

    if len(bits) != 5 or set(bits) - {"0", "1"}:
        raise ValueError("half-box bits must contain five binary digits")
    result = polynomial
    degrees = tuple(int(value) for value in polynomial.degrees())
    for axis, (degree, upper) in enumerate(zip(degrees, bits, strict=True)):
        transformed: dict[tuple[int, ...], int] = {}
        for powers, coefficient in result.to_dict().items():
            power = powers[axis]
            base = int(coefficient) * 2 ** (degree - power)
            targets = range(power + 1) if upper == "1" else (power,)
            for target in targets:
                new_powers = list(powers)
                new_powers[axis] = target
                key = tuple(new_powers)
                value = base * (comb(power, target) if upper == "1" else 1)
                transformed[key] = transformed.get(key, 0) + value
        result = result.context().from_dict(
            {powers: coefficient for powers, coefficient in transformed.items() if coefficient}
        )
    return result


def _restrict_box_path(polynomial, path: str):
    result = polynomial
    for bits in path.split("/"):
        result = _restrict_half_box(result, bits)
    return result


def run(
    coefficient_dir: Path,
    *,
    minor_index: int,
    output: Path,
    flint_threads: int = 6,
    boundary_selector: bool = False,
    face_audit: bool = False,
    half_box_bits: str | None = None,
    all_half_boxes: bool = False,
    skip_native: bool = False,
    half_box_face_audit: bool = False,
    parent_box_path: str | None = None,
) -> dict[str, object]:
    if minor_index not in range(3):
        raise ValueError("minor-index must be 0, 1, or 2")
    if not 1 <= flint_threads <= 6:
        raise ValueError("FLINT thread count must be between one and six")
    resource = constrain_current_process(workers=flint_threads)
    ctx.threads = flint_threads
    polynomial, mask, variables = _build_quadratic(coefficient_dir, minor_index)
    audit = None if skip_native else _native_bernstein_audit(polynomial)
    face_rows = None
    if face_audit:
        names = ("ud", "ue", "ug", "ui", "y")
        face_rows = []
        for axis, name in enumerate(names):
            for value in (0, 1):
                face = polynomial.subs({name: value})
                face_audit_row = _native_bernstein_audit(face)
                face_rows.append(
                    {
                        "face": f"{name}={value}",
                        "power_term_count": len(face.to_dict()),
                        "native_bernstein": face_audit_row,
                        "native_certificate_passed": (
                            face_audit_row["negative_scaled_coefficient_count"] == 0
                        ),
                    }
                )
    selector_report = None
    if boundary_selector:
        ud, _ue, ug, _ui, _y = variables
        face = polynomial.subs({"ud": 0, "ug": 0})
        face_certificate = _z_quadratic_corner_certificate((face, face, face))
        ud_degree, _ue_degree, ug_degree, _ui_degree, _y_degree = map(
            int, polynomial.degrees()
        )
        selector = (1 - ud) ** ud_degree * (1 - ug) ** ug_degree
        remainder = polynomial - face * selector
        remainder_audit = _native_bernstein_audit(remainder)
        selector_report = {
            "face": "ud=ug=0",
            "selector": f"(1-ud)^{ud_degree}*(1-ug)^{ug_degree}",
            "face_square_certificate": face_certificate,
            "remainder_power_term_count": len(remainder.to_dict()),
            "remainder_native_bernstein": remainder_audit,
            "passed": bool(
                face_certificate["passed"]
                and remainder_audit["negative_scaled_coefficient_count"] == 0
            ),
        }
    half_box_report = None
    half_box_face_rows = None
    if half_box_bits is not None:
        restricted = _restrict_half_box(polynomial, half_box_bits)
        restricted_audit = _native_bernstein_audit(restricted)
        half_box_report = {
            "bits": half_box_bits,
            "interval_convention": "0=[0,1/2], 1=[1/2,1]",
            "positive_integer_rescaling_only": True,
            "power_term_count": len(restricted.to_dict()),
            "native_bernstein": restricted_audit,
            "passed": restricted_audit["negative_scaled_coefficient_count"] == 0,
        }
        if half_box_face_audit:
            names = ("ud", "ue", "ug", "ui", "y")
            half_box_face_rows = []
            for name in names:
                for value in (0, 1):
                    face = restricted.subs({name: value})
                    row_audit = _native_bernstein_audit(face, sample_limit=32)
                    half_box_face_rows.append(
                        {
                            "face_in_rescaled_box": f"{name}={value}",
                            "native_bernstein": row_audit,
                            "passed": row_audit["negative_scaled_coefficient_count"] == 0,
                        }
                    )
    half_box_atlas = None
    if all_half_boxes:
        atlas_base = (
            polynomial
            if parent_box_path is None
            else _restrict_box_path(polynomial, parent_box_path)
        )
        half_box_atlas = []
        for bit_tuple in itertools.product("01", repeat=5):
            bits = "".join(bit_tuple)
            restricted = _restrict_half_box(atlas_base, bits)
            restricted_audit = _native_bernstein_audit(restricted, sample_limit=32)
            half_box_atlas.append(
                {
                    "bits": bits if parent_box_path is None else f"{parent_box_path}/{bits}",
                    "power_term_count": len(restricted.to_dict()),
                    "native_bernstein": restricted_audit,
                    "passed": (
                        restricted_audit["negative_scaled_coefficient_count"] == 0
                    ),
                }
            )
            progress = {
                "experiment": "adjacent endpoint octet quadratic half-box atlas",
                "minor_index": minor_index,
                "parent_box_path": parent_box_path,
                "completed_box_count": len(half_box_atlas),
                "total_box_count": 32,
                "boxes": half_box_atlas,
                "complete": len(half_box_atlas) == 32,
            }
            _atomic_json(output, progress)
    report = {
        "experiment": "adjacent endpoint octet quadratic native audit",
        "minor_index": minor_index,
        "mode_mask": list(mask),
        "domain": "(ud,ue,ug,ui,y) in [0,1]^5",
        "power_term_count": len(polynomial.to_dict()),
        "native_bernstein": audit,
        "native_certificate_passed": (
            None if audit is None else audit["negative_scaled_coefficient_count"] == 0
        ),
        "common_square_boundary_selector": selector_report,
        "coordinate_face_audit": face_rows,
        "half_box_audit": half_box_report,
        "half_box_coordinate_face_audit": half_box_face_rows,
        "half_box_atlas": half_box_atlas,
        "half_box_atlas_parent": parent_box_path,
        "interpretation": (
            "A negative native coefficient rejects only the native Bernstein basis; "
            "it is not a counterexample to polynomial nonnegativity."
        ),
        "resource_contract": resource,
    }
    _atomic_json(output, report)
    report["artifact_sha256"] = _sha256(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minor-index", type=int, required=True, choices=range(3))
    parser.add_argument(
        "--coefficient-dir",
        type=Path,
        default=Path("artifacts/spin8_dirac_unrestricted_coefficients_20260807"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flint-threads", type=int, default=6)
    parser.add_argument(
        "--boundary-selector",
        action="store_true",
        help="subtract the exact ud=ug=0 square with its full-degree selector",
    )
    parser.add_argument(
        "--face-audit",
        action="store_true",
        help="audit all ten coordinate faces in exact native Bernstein form",
    )
    parser.add_argument(
        "--half-box-bits",
        type=str,
        help="audit one dyadic half-box, e.g. 00010 for L,L,L,U,L",
    )
    parser.add_argument(
        "--all-half-boxes",
        action="store_true",
        help="audit all 32 dyadic half-boxes and checkpoint after every box",
    )
    parser.add_argument(
        "--parent-box-path",
        type=str,
        help="with --all-half-boxes, first descend through slash-separated boxes",
    )
    parser.add_argument(
        "--half-box-face-audit",
        action="store_true",
        help="with --half-box-bits, audit all ten faces of that rescaled box",
    )
    parser.add_argument(
        "--skip-native",
        action="store_true",
        help="skip the already-recorded full-cube native audit",
    )
    arguments = parser.parse_args()
    report = run(
        arguments.coefficient_dir,
        minor_index=arguments.minor_index,
        output=arguments.output,
        flint_threads=arguments.flint_threads,
        boundary_selector=arguments.boundary_selector,
        face_audit=arguments.face_audit,
        half_box_bits=arguments.half_box_bits,
        all_half_boxes=arguments.all_half_boxes,
        skip_native=arguments.skip_native,
        half_box_face_audit=arguments.half_box_face_audit,
        parent_box_path=arguments.parent_box_path,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
