"""Exact finite-radius blow-up charts for the adjacent-octet Schur cubic."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from math import gcd
from pathlib import Path

from flint import ctx, fmpz, fmpz_mat, fmpz_mpoly_ctx

from spin8_dirac_endpoint_octet_cubic import DEFAULT_COEFFICIENT_DIR, _build_cubic
from spin8_dirac_endpoint_octet_cubic_tangent import (
    DEVIATIONS,
    _factor_report,
    _homogeneous_taylor,
    _radical_factor_certificate,
)
from spin8_dirac_endpoint_octet_quadratic import (
    _atomic_json,
    _native_bernstein_audit,
    _sha256,
)
from spin8_dirac_unrestricted_core import _bernstein_matrix
from spin8_resource_limits import constrain_current_process


def _content(polynomial) -> int:
    result = 0
    for coefficient in polynomial.to_dict().values():
        result = gcd(result, abs(int(coefficient)))
    return result


def _blowup_chart(polynomial, *, pivot: int, expected_order: int = 6):
    if pivot not in range(5):
        raise ValueError("pivot must be between zero and four")
    generators = polynomial.context().gens()
    y = generators[4]
    shifted = polynomial.compose(*generators[:4], 1 - y)
    nonpivots = tuple(axis for axis in range(5) if axis != pivot)
    ratio_index = {axis: index + 1 for index, axis in enumerate(nonpivots)}
    chart_context = fmpz_mpoly_ctx.get(["radius", "x0", "x1", "x2", "x3"])
    coefficients: dict[tuple[int, ...], int] = {}
    for powers, coefficient in shifted.to_dict().items():
        radial_power = sum(powers)
        target = [radial_power, 0, 0, 0, 0]
        for axis in nonpivots:
            target[ratio_index[axis]] = powers[axis]
        key = tuple(target)
        coefficients[key] = coefficients.get(key, 0) + int(coefficient)
    coefficients = {powers: value for powers, value in coefficients.items() if value}
    order = int(min(powers[0] for powers in coefficients))
    if order != expected_order:
        raise AssertionError(f"expected radius order {expected_order}, got {order}")
    divided = {
        (powers[0] - order,) + powers[1:]: coefficient
        for powers, coefficient in coefficients.items()
    }
    radial_degree = int(max(powers[0] for powers in divided))
    scaled = {
        powers: coefficient * 4 ** (radial_degree - powers[0])
        for powers, coefficient in divided.items()
    }
    return chart_context.from_dict(scaled), order, nonpivots, len(shifted.to_dict())


def _tangent_face_in_chart(tangent, *, nonpivots: tuple[int, ...], context):
    coefficients: dict[tuple[int, ...], int] = {}
    for powers, coefficient in tangent.to_dict().items():
        target = [0, 0, 0, 0, 0]
        for ratio_axis, source_axis in enumerate(nonpivots, start=1):
            target[ratio_axis] = powers[source_axis]
        key = tuple(target)
        coefficients[key] = coefficients.get(key, 0) + int(coefficient)
    return context.from_dict(
        {powers: value for powers, value in coefficients.items() if value}
    )


def _positive_proportionality(actual, expected) -> dict[str, object]:
    actual_content = _content(actual)
    expected_content = _content(expected)
    if not actual_content or not expected_content:
        return {"passed": False, "failure": "zero polynomial in proportionality"}
    actual_primitive = actual // actual_content
    expected_primitive = expected // expected_content
    same = actual_primitive == expected_primitive
    return {
        "actual_content": str(actual_content),
        "expected_content": str(expected_content),
        "primitive_polynomials_equal": same,
        "positive_integer_scalings": actual_content > 0 and expected_content > 0,
        "passed": bool(same and actual_content > 0 and expected_content > 0),
    }


def _transform_axis_batched_in_place(
    data: list[fmpz],
    *,
    axis: int,
    shape: tuple[int, ...],
    matrix: fmpz_mat,
    batch_entry_limit: int = 500_000,
) -> None:
    """Apply one tensor axis transform without allocating another full tensor.

    Each tensor line parallel to ``axis`` is independent.  Lines are gathered
    into modest FLINT matrices, transformed exactly, and written back only
    after their batch multiplication has finished.  Thus no later batch can
    observe partially transformed input, while peak memory is bounded by one
    dense tensor plus two small work matrices.
    """

    size = math.prod(shape)
    axis_size = shape[axis]
    stride = math.prod(shape[axis + 1 :])
    block = axis_size * stride
    column_count = size // axis_size
    batch_columns = max(1, batch_entry_limit // axis_size)
    for start in range(0, column_count, batch_columns):
        stop = min(start + batch_columns, column_count)
        columns = stop - start
        entries: list[fmpz] = []
        for coordinate in range(axis_size):
            for line in range(start, stop):
                outer, offset = divmod(line, stride)
                entries.append(data[outer * block + coordinate * stride + offset])
        transformed = (matrix * fmpz_mat(axis_size, columns, entries)).entries()
        for coordinate in range(axis_size):
            source = coordinate * columns
            for local, line in enumerate(range(start, stop)):
                outer, offset = divmod(line, stride)
                data[outer * block + coordinate * stride + offset] = transformed[
                    source + local
                ]


def _batched_bernstein_audit(
    polynomial, *, sample_limit: int = 256, batch_entry_limit: int = 500_000
) -> dict[str, object]:
    """Audit an exact Bernstein tensor with bounded transform memory."""

    degrees = tuple(int(value) for value in polynomial.degrees())
    shape = tuple(degree + 1 for degree in degrees)
    strides = tuple(math.prod(shape[axis + 1 :]) for axis in range(len(shape)))
    values = [fmpz(0)] * math.prod(shape)
    terms = polynomial.to_dict()
    for powers, coefficient in terms.items():
        flat = sum(
            power * stride for power, stride in zip(powers, strides, strict=True)
        )
        values[flat] = fmpz(coefficient)
    del terms

    scales: list[int] = []
    for axis, degree in enumerate(degrees):
        matrix, scale = _bernstein_matrix(degree)
        _transform_axis_batched_in_place(
            values,
            axis=axis,
            shape=shape,
            matrix=matrix,
            batch_entry_limit=batch_entry_limit,
        )
        scales.append(int(scale))

    negative_count = 0
    zero_count = 0
    negative_samples: list[dict[str, object]] = []
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
        index: list[int] = []
        boundary: list[str] = []
        for axis, stride in enumerate(strides):
            coordinate, remainder = divmod(remainder, stride)
            index.append(coordinate)
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
        result: list[int] = []
        remainder = flat
        for stride in strides:
            coordinate, remainder = divmod(remainder, stride)
            result.append(coordinate)
        return result

    return {
        "audit_engine": "exact_batched_in_place",
        "batch_entry_limit": batch_entry_limit,
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


def run(
    coefficient_dir: Path,
    *,
    output: Path,
    pivot: int = 4,
    audit: bool = False,
    audit_mode: str = "batched",
    flint_threads: int = 6,
) -> dict[str, object]:
    if not 1 <= flint_threads <= 6:
        raise ValueError("FLINT thread count must be between one and six")
    resource = constrain_current_process(workers=flint_threads)
    ctx.threads = flint_threads
    cubic, _tau, _forced_product, _variables = _build_cubic(coefficient_dir)
    tangent, tangent_order = _homogeneous_taylor(cubic, max_order=12)
    quotient, order, nonpivots, shifted_term_count = _blowup_chart(
        cubic, pivot=pivot, expected_order=tangent_order
    )
    exceptional = quotient.subs({"radius": 0})
    expected = _tangent_face_in_chart(
        tangent,
        nonpivots=nonpivots,
        context=quotient.context(),
    )
    proportionality = _positive_proportionality(exceptional, expected)
    construction_passed = bool(order == 6 and proportionality["passed"])
    native_audit = None
    selector_certificate = None
    chart_passed = False
    if audit:
        if audit_mode == "native":
            native_audit = _native_bernstein_audit(quotient, sample_limit=256)
            native_audit["audit_engine"] = "exact_full_tensor"
        elif audit_mode == "batched":
            native_audit = _batched_bernstein_audit(quotient, sample_limit=256)
        else:
            raise ValueError("audit_mode must be 'native' or 'batched'")
        if native_audit["negative_scaled_coefficient_count"] == 0:
            chart_passed = construction_passed
        else:
            factor_report, exact_factors = _factor_report(tangent)
            tangent_sign = _radical_factor_certificate(exact_factors)
            radius = quotient.context().gens()[0]
            radial_degree = int(quotient.degrees()[0])
            selector = (1 - radius) ** radial_degree
            remainder = quotient - exceptional * selector
            identity = quotient == exceptional * selector + remainder
            if audit_mode == "native":
                remainder_audit = _native_bernstein_audit(remainder, sample_limit=256)
                remainder_audit["audit_engine"] = "exact_full_tensor"
            else:
                remainder_audit = _batched_bernstein_audit(remainder, sample_limit=256)
            chart_passed = bool(
                construction_passed
                and factor_report["identity_verified_exactly"]
                and tangent_sign["passed"]
                and identity
                and remainder_audit["negative_scaled_coefficient_count"] == 0
            )
            selector_certificate = {
                "selector": f"(1-radius)^{radial_degree}",
                "identity_verified_exactly": identity,
                "tangent_radical_factor_certificate": tangent_sign,
                "remainder_power_term_count": len(remainder.to_dict()),
                "remainder_native_bernstein": remainder_audit,
                "passed": chart_passed,
            }
    report = {
        "experiment": "adjacent endpoint octet cubic finite-radius blow-up",
        "pivot_index": pivot,
        "pivot_deviation": DEVIATIONS[pivot],
        "ratio_deviations": [DEVIATIONS[index] for index in nonpivots],
        "chart": "pivot=radius; nonpivots=radius*x_j; radius=R/4",
        "shifted_power_term_count": shifted_term_count,
        "exact_radius_divisibility_order": order,
        "quotient_power_term_count": len(quotient.to_dict()),
        "quotient_multidegree": list(map(int, quotient.degrees())),
        "exceptional_power_term_count": len(exceptional.to_dict()),
        "exceptional_tangent_proportionality": proportionality,
        "construction_passed": construction_passed,
        "quotient_native_bernstein": native_audit,
        "radial_selector_certificate": selector_certificate,
        "passed": chart_passed,
        "scope_boundary": (
            "One of five finite-radius charts. All five must pass before the "
            "equality corner is certified."
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
    parser.add_argument("--pivot", type=int, default=4, choices=range(5))
    parser.add_argument("--audit", action="store_true")
    parser.add_argument(
        "--audit-mode", choices=("native", "batched"), default="batched"
    )
    parser.add_argument("--flint-threads", type=int, default=6)
    arguments = parser.parse_args()
    report = run(
        arguments.coefficient_dir,
        output=arguments.output,
        pivot=arguments.pivot,
        audit=arguments.audit,
        audit_mode=arguments.audit_mode,
        flint_threads=arguments.flint_threads,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
