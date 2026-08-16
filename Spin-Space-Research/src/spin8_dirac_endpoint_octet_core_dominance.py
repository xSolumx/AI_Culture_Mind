"""Exact diagonal-dominance theorem on the central endpoint-octet core.

On ``[1/4, 3/4]^5`` the trivial Walsh amplitude strictly dominates the sum
of the absolute values of all seven nontrivial amplitudes.  The proof uses a
32-box dyadic partition, exact Bernstein lower/upper bounds for the integer
residual polynomials, and outward rational square-root bounds for the forced
radical squares.  Consequently every physical orientation margin is strictly
positive on the complete central core.

This is an exact strict-interior theorem on a compact subcube, not a proof on
the collars between that core and the coordinate boundary.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

from flint import ctx, fmpz, fmpz_mpoly_ctx

from spin8_dirac_endpoint_octet import (
    H0,
    H1,
    _forced_square,
    _to_endpoint_chart,
)
from spin8_dirac_endpoint_octet_determinant import (
    DEFAULT_COEFFICIENT_DIR,
    _coefficient_hashes,
)
from spin8_dirac_endpoint_octet_quadratic import (
    _atomic_json,
    _restrict_half_box,
    _sha256,
)
from spin8_dirac_final_residual import exact_full_chart_sign_certificate
from spin8_dirac_unrestricted_core import (
    _bernstein_matrix,
    _read_integer_polynomial,
    _transform_axis,
)
from spin8_resource_limits import constrain_current_process

CORE_LOW = Fraction(1, 4)
CORE_HIGH = Fraction(3, 4)
SQRT_BITS = 80
VARIABLE_NAMES = ("ud", "ue", "ug", "ui", "y")


def _fraction_text(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _ceil_sqrt_fraction(value: Fraction, *, bits: int = SQRT_BITS) -> Fraction:
    """Return an exact dyadic upper bound for a nonnegative square root."""

    if value < 0 or bits < 1:
        raise ValueError("square-root input and precision must be nonnegative")
    if value == 0:
        return Fraction(0)
    scale = 1 << bits
    target = value.numerator * scale * scale
    numerator = math.isqrt(target // value.denominator)
    if numerator * numerator * value.denominator < target:
        numerator += 1
    result = Fraction(numerator, scale)
    if result * result < value:
        raise AssertionError("outward square-root bound rounded inward")
    if numerator > 0 and Fraction(numerator - 1, scale) ** 2 >= value:
        raise AssertionError("outward square-root bound is not the dyadic ceiling")
    return result


def _restrict_dyadic_path(polynomial, path: str):
    """Map a dyadic box to the unit cube and return its positive scale."""

    result = polynomial
    positive_scale = 1
    for bits in path.split("/"):
        if len(bits) != 5 or set(bits) - {"0", "1"}:
            raise ValueError("each dyadic path component must contain five bits")
        positive_scale *= 2 ** sum(map(int, result.degrees()))
        result = _restrict_half_box(result, bits)
    return result, positive_scale


def _bernstein_bounds(polynomial, path: str) -> tuple[Fraction, Fraction]:
    """Return exact Bernstein convex-hull bounds on one dyadic box."""

    restricted, denominator = _restrict_dyadic_path(polynomial, path)
    degrees = tuple(map(int, restricted.degrees()))
    shape = tuple(degree + 1 for degree in degrees)
    strides = tuple(math.prod(shape[axis + 1 :]) for axis in range(5))
    values = [fmpz(0)] * math.prod(shape)
    for powers, coefficient in restricted.to_dict().items():
        flat = sum(
            power * stride for power, stride in zip(powers, strides, strict=True)
        )
        values[flat] = fmpz(coefficient)
    for axis, degree in enumerate(degrees):
        matrix, scale = _bernstein_matrix(degree)
        values = _transform_axis(values, axis=axis, shape=shape, matrix=matrix)
        denominator *= int(scale)
    return (
        Fraction(int(min(values)), denominator),
        Fraction(int(max(values)), denominator),
    )


def _endpoint_amplitudes(coefficient_dir: Path):
    """Reconstruct the eight integer-scaled endpoint amplitude ingredients."""

    chart = exact_full_chart_sign_certificate()
    if not chart["passed"]:
        raise AssertionError("full-chart sign certificate failed")
    rows = chart["chart_characters"]
    masks = tuple(tuple(row["lower_mask"]) for row in rows)
    complements = {
        tuple(row["lower_mask"]): tuple(row["complement_mask"]) for row in rows
    }
    context7 = fmpz_mpoly_ctx.get(("ua", "ud", "ue", "ug", "uh", "ui", "z"))
    context5 = fmpz_mpoly_ctx.get(VARIABLE_NAMES)
    variables = context5.gens()
    surviving = H0 + H1
    residuals = {
        mask: _to_endpoint_chart(
            _read_integer_polynomial(context7, coefficient_dir, mask), context5
        )
        for mask in surviving
    }
    forced_squares = {
        mask: _forced_square(context5, variables, mask, complements[mask])
        for mask in masks
    }
    if forced_squares[H0[0]] != 1:
        raise AssertionError("trivial amplitude has a nontrivial forced square")
    return surviving, residuals, forced_squares


def _core_box_path(code: tuple[int, ...]) -> tuple[str, list[list[str]]]:
    """Return the dyadic path and exact intervals for one central-core box."""

    if len(code) != 5 or set(code) - {0, 1}:
        raise ValueError("central-core code must contain five binary entries")
    first = "".join(map(str, code))
    second = "".join(str(1 - bit) for bit in code)
    intervals = [["1/4", "1/2"] if bit == 0 else ["1/2", "3/4"] for bit in code]
    return f"{first}/{second}", intervals


def _core_rows(coefficient_dir: Path) -> list[dict[str, object]]:
    surviving, residuals, forced_squares = _endpoint_amplitudes(coefficient_dir)
    rows: list[dict[str, object]] = []
    for code in itertools.product((0, 1), repeat=5):
        path, intervals = _core_box_path(code)
        trivial_low, trivial_high = _bernstein_bounds(residuals[H0[0]], path)
        if trivial_low <= 0:
            raise AssertionError(f"trivial amplitude is not positive on {path}")
        component_rows = []
        nontrivial_sum = Fraction(0)
        for mask in surviving[1:]:
            residual_low, residual_high = _bernstein_bounds(residuals[mask], path)
            square_low, square_high = _bernstein_bounds(forced_squares[mask], path)
            if square_high < 0:
                raise AssertionError(
                    f"forced square has a negative upper bound on {path}"
                )
            residual_absolute = max(abs(residual_low), abs(residual_high))
            radical_upper = _ceil_sqrt_fraction(square_high)
            amplitude_upper = residual_absolute * radical_upper
            nontrivial_sum += amplitude_upper
            component_rows.append(
                {
                    "mask": "".join(map(str, mask)),
                    "residual_bernstein_lower": _fraction_text(residual_low),
                    "residual_bernstein_upper": _fraction_text(residual_high),
                    "forced_square_bernstein_lower": _fraction_text(square_low),
                    "forced_square_bernstein_upper": _fraction_text(square_high),
                    "outward_sqrt_upper": _fraction_text(radical_upper),
                    "absolute_amplitude_upper": _fraction_text(amplitude_upper),
                    "sqrt_upper_verified_exactly": radical_upper**2 >= square_high,
                }
            )
        scaled_gap = trivial_low - nontrivial_sum
        rows.append(
            {
                "box_code": "".join(map(str, code)),
                "dyadic_path": path,
                "variable_order": list(VARIABLE_NAMES),
                "intervals": intervals,
                "trivial_amplitude_bernstein_lower": _fraction_text(trivial_low),
                "trivial_amplitude_bernstein_upper": _fraction_text(trivial_high),
                "nontrivial_absolute_upper_bounds": component_rows,
                "nontrivial_absolute_sum_upper": _fraction_text(nontrivial_sum),
                "integer_scaled_dominance_gap_lower": _fraction_text(scaled_gap),
                "physical_dominance_gap_lower": _fraction_text(scaled_gap / 4),
                "strictly_positive": scaled_gap > 0,
            }
        )
    return rows


def build_certificate(coefficient_dir: Path = DEFAULT_COEFFICIENT_DIR):
    """Build the exact 32-box diagonal-dominance certificate."""

    rows = _core_rows(coefficient_dir)
    minimum = min(
        rows,
        key=lambda row: Fraction(row["integer_scaled_dominance_gap_lower"]),
    )
    passed = bool(len(rows) == 32 and all(row["strictly_positive"] for row in rows))
    return {
        "experiment": "adjacent endpoint-octet central-core diagonal dominance",
        "evidence_class": "exact rational Bernstein and outward-radical certificate",
        "domain": "(ud,ue,ug,ui,y) in [1/4,3/4]^5",
        "partition": {
            "box_count": len(rows),
            "per_axis_intervals": [["1/4", "1/2"], ["1/2", "3/4"]],
            "complete_cartesian_cover": len(rows) == 2**5,
        },
        "coefficient_source_sha256": _coefficient_hashes(coefficient_dir),
        "residual_common_integer_scale": 4,
        "outward_square_root_precision_bits": SQRT_BITS,
        "box_certificates": rows,
        "minimum_gap_box_code": minimum["box_code"],
        "minimum_integer_scaled_gap_lower": minimum[
            "integer_scaled_dominance_gap_lower"
        ],
        "minimum_physical_gap_lower": minimum["physical_dominance_gap_lower"],
        "strict_diagonal_dominance_proved": passed,
        "all_eight_physical_margins_strictly_positive": passed,
        "schur_matrix_positive_definite_on_core": passed,
        "determinant_strictly_positive_on_core": passed,
        "complete_adjacent_octet_proved": False,
        "global_dirac_gram_theorem_proved": False,
        "passed": passed,
        "scope_boundary": (
            "The theorem covers exactly [1/4,3/4]^5. It does not cover the "
            "boundary collars, the complete adjacent endpoint octet, or the "
            "unrestricted seven-variable Dirac--Gram domain."
        ),
        "next_exact_gate": (
            "Extend the dyadic dominance atlas into the boundary collars and "
            "delegate irreducible equality-corner boxes to nested blow-ups."
        ),
    }


def verify_report(
    report_or_path, *, coefficient_dir: Path = DEFAULT_COEFFICIENT_DIR
) -> dict[str, object]:
    """Rebuild all exact bounds and compare them with the stored certificate."""

    if isinstance(report_or_path, (str, Path)):
        report = json.loads(Path(report_or_path).read_text(encoding="utf-8"))
    else:
        report = report_or_path
    replay = build_certificate(coefficient_dir)
    failures: list[str] = []
    for key in (
        "partition",
        "coefficient_source_sha256",
        "residual_common_integer_scale",
        "outward_square_root_precision_bits",
        "box_certificates",
        "minimum_gap_box_code",
        "minimum_integer_scaled_gap_lower",
        "minimum_physical_gap_lower",
        "strict_diagonal_dominance_proved",
        "all_eight_physical_margins_strictly_positive",
        "schur_matrix_positive_definite_on_core",
        "determinant_strictly_positive_on_core",
        "complete_adjacent_octet_proved",
        "global_dirac_gram_theorem_proved",
        "passed",
    ):
        if report.get(key) != replay.get(key):
            failures.append(f"stored {key} disagrees with exact replay")
    for claim in (
        "strict_diagonal_dominance_proved",
        "all_eight_physical_margins_strictly_positive",
        "schur_matrix_positive_definite_on_core",
        "determinant_strictly_positive_on_core",
    ):
        if report.get(claim) is not True:
            failures.append(f"stored report does not establish {claim}")
    for nonclaim in (
        "complete_adjacent_octet_proved",
        "global_dirac_gram_theorem_proved",
    ):
        if report.get(nonclaim) is not False:
            failures.append(f"stored report overclaims {nonclaim}")
    return {
        "verified": not failures,
        "failures": failures,
        "replayed_box_count": replay["partition"]["box_count"],
        "minimum_gap_box_code": replay["minimum_gap_box_code"],
        "minimum_physical_gap_lower": replay["minimum_physical_gap_lower"],
        "strict_diagonal_dominance_proved": replay["strict_diagonal_dominance_proved"],
    }


def run(
    coefficient_dir: Path,
    *,
    output: Path,
    flint_threads: int = 6,
) -> dict[str, object]:
    if not 1 <= flint_threads <= 6:
        raise ValueError("FLINT thread count must be between one and six")
    resource = constrain_current_process(workers=flint_threads)
    ctx.threads = flint_threads
    report = build_certificate(coefficient_dir)
    report["resource_contract"] = resource
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
