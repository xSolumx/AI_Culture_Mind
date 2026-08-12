"""Rational-circle Bernstein atlas for the finite second Dirac edge.

This module changes every nonnegative circle pair to the half-angle chart

    q(t) = 2 t / (1 + t**2),
    Q(t) = (1 - t**2) / (1 + t**2),       0 <= t <= 1.

For each of the eight orientation margins, multiplication by the common
strictly-positive denominator produces an ordinary polynomial of multidegree
``(12, 12, 12, 12, 12, 8)``.  The conversion is performed as separable tensor
algebra from the maintained exact Walsh-sector coefficient map; it does not
expand a six-variable SymPy expression.

The floating-point Bernstein and dyadic-shell routines are discovery tools.
They deliberately do not promote a theorem: a publication certificate must
replay the same transformations with exact or outward-rounded arithmetic.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import sympy as sp

from spin8_dirac_two_edge_kernel import EVEN_MASKS, HADAMARD, ODD_MASKS, load_sectors
from spin8_resource_limits import constrain_current_process

MASK_ORDER = EVEN_MASKS + ODD_MASKS
TARGET_COEFFICIENTS = (81, -18, 1)


def common_denominator_exponents(sectors) -> tuple[int, ...]:
    """Return the least coordinatewise half-angle denominator exponents."""

    exponents = []
    for axis in range(6):
        target_degree = 4 if axis == 5 else 0
        exponents.append(
            max(
                [target_degree]
                + [
                    sector.mask[axis]
                    + sector.complement[axis]
                    + 2 * powers[axis]
                    for sector in sectors.values()
                    for powers, _coefficient in sector.terms
                ]
            )
        )
    return tuple(exponents)


def _polynomial_power(coefficients: np.ndarray, exponent: int) -> np.ndarray:
    result = np.array([1], dtype=np.int64)
    for _ in range(exponent):
        result = np.convolve(result, coefficients)
    return result


def _axis_rationalization_matrix(
    denominator_exponent: int,
    lower_bit: int,
    complement_bit: int,
    source_degree: int,
) -> np.ndarray:
    """Map powers of ``q(t)^2`` to one common polynomial numerator."""

    matrix = np.zeros(
        (2 * denominator_exponent + 1, source_degree + 1), dtype=np.int64
    )
    plus = np.array([1, 0, 1], dtype=np.int64)
    minus = np.array([1, 0, -1], dtype=np.int64)
    for power in range(source_degree + 1):
        residual = denominator_exponent - lower_bit - complement_bit - 2 * power
        if residual < 0:
            continue
        polynomial = np.convolve(
            _polynomial_power(plus, residual),
            _polynomial_power(minus, complement_bit),
        )
        polynomial = np.pad(polynomial, (lower_bit + 2 * power, 0))
        polynomial *= (2**lower_bit) * (4**power)
        matrix[: polynomial.size, power] = polynomial
    return matrix


def _mode_product(matrix: np.ndarray, tensor: np.ndarray, axis: int) -> np.ndarray:
    result = np.tensordot(matrix, tensor, axes=(1, axis))
    return np.moveaxis(result, 0, axis)


def _checked_integer_mode_product(
    matrix: np.ndarray, tensor: np.ndarray, axis: int
) -> np.ndarray:
    maximum = int(np.abs(tensor).max(initial=0))
    row_bound = int(np.abs(matrix).sum(axis=1).max(initial=0))
    if maximum * row_bound > np.iinfo(np.int64).max:
        raise OverflowError("int64 rationalization bound exceeded")
    return _mode_product(matrix, tensor, axis)


def rationalized_power_tensors(
    coefficient_path: Path,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], dict[str, object]]:
    """Return the scaled target and eight exact integer power tensors.

    All source coefficients are multiplied by their common denominator four.
    The returned tensors therefore encode four times the polynomial numerators;
    this positive scaling has no effect on signs.
    """

    sectors = load_sectors(coefficient_path)
    denominator_exponents = common_denominator_exponents(sectors)
    output_shape = tuple(2 * exponent + 1 for exponent in denominator_exponents)
    coefficient_scale = math.lcm(
        *(
            int(sp.denom(coefficient))
            for sector in sectors.values()
            for _powers, coefficient in sector.terms
        )
    )

    target = np.zeros((1, 1, 1, 1, 1, 3), dtype=np.int64)
    target[0, 0, 0, 0, 0, :] = np.asarray(TARGET_COEFFICIENTS) * coefficient_scale
    for axis, exponent in enumerate(denominator_exponents):
        source_degree = 2 if axis == 5 else 0
        matrix = _axis_rationalization_matrix(exponent, 0, 0, source_degree)
        target = _checked_integer_mode_product(matrix, target, axis)

    amplitudes = []
    rows = []
    for mask in MASK_ORDER:
        sector = sectors[mask]
        degrees = tuple(max(powers[axis] for powers, _ in sector.terms) for axis in range(6))
        tensor = np.zeros(tuple(degree + 1 for degree in degrees), dtype=np.int64)
        for powers, coefficient in sector.terms:
            scaled = coefficient * coefficient_scale
            if scaled.q != 1:
                raise AssertionError("common coefficient scale is incomplete")
            tensor[powers] = int(scaled)
        for axis, exponent in enumerate(denominator_exponents):
            matrix = _axis_rationalization_matrix(
                exponent,
                sector.mask[axis],
                sector.complement[axis],
                degrees[axis],
            )
            tensor = _checked_integer_mode_product(matrix, tensor, axis)
        if tensor.shape != output_shape:
            raise AssertionError("rationalized sector tensor has the wrong shape")
        amplitudes.append(tensor)
        rows.append(
            {
                "mask": list(mask),
                "source_multidegree": list(degrees),
                "power_minimum": int(tensor.min()),
                "power_maximum": int(tensor.max()),
            }
        )

    metadata = {
        "chart": "q=2t/(1+t^2), Q=(1-t^2)/(1+t^2), 0<=t<=1",
        "coefficient_scale": coefficient_scale,
        "common_denominator_exponents": list(denominator_exponents),
        "positive_common_denominator": (
            "product_j (1+t_j^2)^(common_denominator_exponents[j])"
        ),
        "numerator_multidegree": [size - 1 for size in output_shape],
        "numerator_coefficient_count": math.prod(output_shape),
        "sector_rows": rows,
    }
    return target, tuple(amplitudes), metadata


def orientation_power_tensor(
    target: np.ndarray,
    amplitudes: tuple[np.ndarray, ...],
    channel: int,
    odd_sign: int,
) -> np.ndarray:
    """Assemble one of the eight physical orientation-margin numerators."""

    if channel not in range(4) or odd_sign not in (-1, 1):
        raise ValueError("expected channel in range(4) and odd_sign in {-1,1}")
    result = target.copy()
    row = HADAMARD[channel]
    for index in range(4):
        result -= row[index] * amplitudes[index]
        result -= odd_sign * row[index] * amplitudes[4 + index]
    return result


def _power_to_bernstein_matrix(degree: int) -> np.ndarray:
    matrix = np.zeros((degree + 1, degree + 1), dtype=np.float64)
    for target in range(degree + 1):
        for source in range(target + 1):
            matrix[target, source] = math.comb(target, source) / math.comb(
                degree, source
            )
    return matrix


def float_bernstein_controls(power_tensor: np.ndarray) -> np.ndarray:
    """Convert an exact power tensor to float64 Bernstein controls.

    This function is a fast screen only.  Its output is not an exact sign
    certificate.
    """

    controls = power_tensor.astype(np.float64)
    for axis, size in enumerate(controls.shape):
        controls = _mode_product(_power_to_bernstein_matrix(size - 1), controls, axis)
    return controls


def _upward_nonnegative_mode_product(
    matrix: np.ndarray, tensor: np.ndarray, axis: int
) -> np.ndarray:
    """Upper-bound a nonnegative matrix--tensor product in binary64.

    If a dot product has ``n`` terms, the standard floating-point model bounds
    its relative summation error by ``gamma_n = n*u/(1-n*u)``.  Dividing the
    computed nonnegative result by ``1-gamma_n`` and rounding once toward
    positive infinity therefore encloses the exact product.  The routine is
    used only with finite, nonnegative binary64 inputs.
    """

    if np.any(matrix < 0) or np.any(tensor < 0):
        raise ValueError("upward product requires nonnegative inputs")
    result = _mode_product(matrix, tensor, axis)
    unit_roundoff = np.finfo(np.float64).eps / 2
    gamma = matrix.shape[1] * unit_roundoff
    gamma /= 1 - gamma
    return np.nextafter(result / (1 - gamma), np.inf)


def float_bernstein_enclosure(
    power_tensor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enclose every Bernstein control and track structural zeros.

    The integer power coefficients in the maintained rational chart are below
    ``2**53`` and therefore enter binary64 exactly.  At each separable
    Bernstein transform this routine propagates three error sources:

    * the enclosure already attached to the input controls;
    * rounding of each exact nonnegative rational transform weight;
    * the dot-product rounding error bounded by ``gamma_n``.

    The returned arrays are ``(centre, radius, possible_nonzero)``.  A control
    with ``possible_nonzero == False`` is exactly zero because no nonzero power
    coefficient can reach it through the lower-triangular positive Bernstein
    transform.  A control with ``centre - radius > 0`` is rigorously positive
    under the IEEE-754 round-to-nearest model used by NumPy.  Remaining entries
    require an exact integer or rational replay; this function never guesses
    their sign.
    """

    maximum = int(np.abs(power_tensor).max(initial=0))
    if maximum > 2**53:
        raise OverflowError("power coefficients are not exactly representable")

    centre = power_tensor.astype(np.float64)
    radius = np.zeros_like(centre)
    possible_nonzero = power_tensor != 0
    unit_roundoff = np.finfo(np.float64).eps / 2
    weight_relative_error = unit_roundoff / (1 - unit_roundoff)

    for axis, size in enumerate(centre.shape):
        matrix = _power_to_bernstein_matrix(size - 1)
        absolute_matrix = np.abs(matrix)
        absolute_centre = _upward_nonnegative_mode_product(
            absolute_matrix, np.abs(centre), axis
        )
        propagated_radius = _upward_nonnegative_mode_product(
            absolute_matrix, radius, axis
        )
        gamma = size * unit_roundoff
        gamma /= 1 - gamma

        next_centre = _mode_product(matrix, centre, axis)
        propagation_factor = np.nextafter(
            np.float64(1 + weight_relative_error), np.inf
        )
        rounding_factor = np.nextafter(
            np.float64(weight_relative_error + gamma), np.inf
        )
        propagation_term = np.nextafter(
            propagation_factor * propagated_radius, np.inf
        )
        rounding_term = np.nextafter(rounding_factor * absolute_centre, np.inf)
        next_radius = np.nextafter(propagation_term + rounding_term, np.inf)

        centre = next_centre
        radius = next_radius
        possible_nonzero = np.logical_or.accumulate(possible_nonzero, axis=axis)

    return centre, radius, possible_nonzero


def _scaled_integer_bernstein_rows(
    degree: int, selected_rows: tuple[int, ...]
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return positively scaled exact Bernstein rows as Python integers.

    For a degree-``n`` power polynomial, Bernstein control ``k`` is

    ``sum_{i<=k} a_i * C(k,i) / C(n,i)``.

    Each requested row is multiplied by the least common multiple of its
    rational denominators.  Row-dependent positive scaling preserves both
    sign and exact zero, and later transforms on other axes never mix these
    already-selected row indices.
    """

    if any(row < 0 or row > degree for row in selected_rows):
        raise ValueError("selected Bernstein row is outside the degree")
    rows = []
    scales = []
    for target in selected_rows:
        rationals = [
            sp.Rational(math.comb(target, source), math.comb(degree, source))
            if source <= target
            else sp.Integer(0)
            for source in range(degree + 1)
        ]
        scale = math.lcm(*(int(sp.denom(value)) for value in rationals))
        rows.append([int(value * scale) for value in rationals])
        scales.append(scale)
    return np.asarray(rows, dtype=object), tuple(scales)


def selected_scaled_integer_bernstein(
    power_tensor: np.ndarray,
    selected_rows: tuple[tuple[int, ...], ...],
) -> tuple[np.ndarray, dict[str, object]]:
    """Replay a Cartesian subset of Bernstein controls with exact integers.

    The output is scaled independently by a positive integer along each
    selected row of each axis.  Hence an output entry is positive, zero, or
    negative exactly when the corresponding rational Bernstein control is.
    Axes are processed in order of greatest dimensional reduction, keeping the
    object-integer working set small.
    """

    if len(selected_rows) != power_tensor.ndim:
        raise ValueError("one selected-row tuple is required per tensor axis")
    if power_tensor.dtype.kind not in "iu":
        raise TypeError("exact selected replay requires an integer power tensor")

    output = power_tensor
    rows = []
    order = sorted(
        range(power_tensor.ndim),
        key=lambda axis: len(selected_rows[axis]) / power_tensor.shape[axis],
    )
    for axis in order:
        degree = output.shape[axis] - 1
        selected = tuple(sorted(set(selected_rows[axis])))
        matrix, scales = _scaled_integer_bernstein_rows(degree, selected)

        if output.dtype == object:
            maximum = max(abs(int(value)) for value in output.reshape(-1))
        else:
            maximum = int(np.abs(output).max(initial=0))
        row_bound = max(sum(abs(int(value)) for value in row) for row in matrix)
        fits_int64 = maximum * row_bound <= np.iinfo(np.int64).max
        if fits_int64 and output.dtype != object:
            transformed = _mode_product(matrix.astype(np.int64), output, axis)
        else:
            if output.dtype != object:
                output = output.astype(object)
            transformed = _mode_product(matrix, output, axis)
        output = transformed
        rows.append(
            {
                "axis": axis,
                "source_degree": degree,
                "selected_rows": list(selected),
                "positive_row_scales": list(scales),
                "used_int64": fits_int64,
            }
        )

    return output, {"axis_order": order, "rows": rows}


def triangle_blowup_power_tensor(
    power_tensor: np.ndarray,
    first_axis: int,
    second_axis: int,
    *,
    upper: bool = False,
) -> np.ndarray:
    """Pull a polynomial back to one triangle of a coordinate square.

    The lower chart, in the two output coordinates ``(u,v)``, is

    ``x_first = u`` and ``x_second = u * v``.

    The upper chart is

    ``x_first = u * v`` and ``x_second = v``.

    In either case ``(u,v)`` ranges over the unit square.  The two
    charts cover the original coordinate square and meet on its diagonal.
    Coefficients are only re-indexed and added, so an integer input produces
    an exact integer output without symbolic expansion.
    """

    if first_axis == second_axis:
        raise ValueError("triangle axes must be distinct")
    if first_axis not in range(power_tensor.ndim) or second_axis not in range(
        power_tensor.ndim
    ):
        raise ValueError("triangle axis is outside the tensor rank")

    first_degree = power_tensor.shape[first_axis] - 1
    second_degree = power_tensor.shape[second_axis] - 1
    output_shape = list(power_tensor.shape)
    if upper:
        output_shape[first_axis] = first_degree + 1
        output_shape[second_axis] = first_degree + second_degree + 1
    else:
        output_shape[first_axis] = first_degree + second_degree + 1
        output_shape[second_axis] = second_degree + 1
    output = np.zeros(output_shape, dtype=power_tensor.dtype)

    for first_power in range(first_degree + 1):
        for second_power in range(second_degree + 1):
            source = [slice(None)] * power_tensor.ndim
            source[first_axis] = first_power
            source[second_axis] = second_power
            target = [slice(None)] * power_tensor.ndim
            if upper:
                target[first_axis] = first_power
                target[second_axis] = first_power + second_power
            else:
                target[first_axis] = first_power + second_power
                target[second_axis] = second_power
            output[tuple(target)] += power_tensor[tuple(source)]
    return output


def split_bernstein_half(
    controls: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split one Bernstein tensor at the dyadic midpoint by de Casteljau."""

    work = np.moveaxis(controls, axis, 0)
    degree = work.shape[0] - 1
    left = np.empty_like(work)
    right = np.empty_like(work)
    left[0] = work[0]
    right[degree] = work[degree]
    current = work
    for level in range(1, degree + 1):
        current = (current[:-1] + current[1:]) / 2
        left[level] = current[0]
        right[degree - level] = current[-1]
    return np.moveaxis(left, 0, axis), np.moveaxis(right, 0, axis)


def lexicographic_shell_screen(
    controls: np.ndarray,
    *,
    levels: int,
    shell_axes: tuple[int, ...] = (0, 1, 2, 3, 4),
    tolerance: float = 1e-9,
) -> dict[str, object]:
    """Screen dyadic shells around the common equality corner.

    At each level, the current all-lower box is split successively along the
    five non-Cayley axes.  The five high slabs and the next all-lower box form a
    disjoint cover of the current box.
    """

    current = controls
    rows = []
    for level in range(1, levels + 1):
        shell_rows = []
        for axis in shell_axes:
            lower, upper = split_bernstein_half(current, axis)
            shell_rows.append(
                {
                    "axis": axis,
                    "minimum": float(upper.min()),
                    "negative_count": int(np.count_nonzero(upper < -tolerance)),
                }
            )
            current = lower
        rows.append(
            {
                "level": level,
                "outer_radius": 2 ** (1 - level),
                "inner_radius": 2**-level,
                "shell_slabs": shell_rows,
                "all_lower_minimum": float(current.min()),
                "all_lower_negative_count": int(
                    np.count_nonzero(current < -tolerance)
                ),
            }
        )
    return {
        "levels": levels,
        "shell_axes": list(shell_axes),
        "rows": rows,
        "all_shell_slabs_screen_nonnegative": all(
            slab["negative_count"] == 0
            for row in rows
            for slab in row["shell_slabs"]
        ),
        "status": "float64 discovery screen only; not a positivity certificate",
    }


def run(
    coefficients: Path,
    output: Path,
    *,
    shell_levels: int,
    workers: int,
) -> dict[str, object]:
    resource = constrain_current_process(workers=workers)
    target, amplitudes, metadata = rationalized_power_tensors(coefficients)
    margins = []
    for channel in range(4):
        for odd_sign in (1, -1):
            power = orientation_power_tensor(target, amplitudes, channel, odd_sign)
            controls = float_bernstein_controls(power)
            margins.append(
                {
                    "channel": channel,
                    "odd_sign": odd_sign,
                    "native_negative_count": int(np.count_nonzero(controls < -1e-9)),
                    "native_minimum": float(controls.min()),
                    "native_zero_count_at_1e_minus_10": int(
                        np.count_nonzero(np.abs(controls) <= 1e-10)
                    ),
                    "shell_screen": lexicographic_shell_screen(
                        controls, levels=shell_levels
                    ),
                }
            )
    report = {
        "experiment": "finite two-edge rational-circle Bernstein atlas",
        "coefficient_artifact": str(coefficients),
        "rationalization": metadata,
        "orientation_margins": margins,
        "resource_contract": resource,
        "theorem_promoted": False,
        "scope_boundary": (
            "The rationalization is exact. Bernstein signs and dyadic shells in "
            "this artifact are float64 reconnaissance until replayed with exact "
            "or outward-rounded arithmetic and joined to a local equality proof."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coefficients",
        type=Path,
        default=Path(
            "artifacts/spin8_dirac_two_edge_all_sectors_coefficients_20260806.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/spin8_dirac_two_edge_rational_screen_20260807.json"),
    )
    parser.add_argument("--shell-levels", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    report = run(
        arguments.coefficients,
        arguments.output,
        shell_levels=arguments.shell_levels,
        workers=arguments.workers,
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "multidegree": report["rationalization"]["numerator_multidegree"],
                "native_negative_counts": [
                    row["native_negative_count"]
                    for row in report["orientation_margins"]
                ],
                "all_shell_screens_pass": all(
                    row["shell_screen"]["all_shell_slabs_screen_nonnegative"]
                    for row in report["orientation_margins"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
