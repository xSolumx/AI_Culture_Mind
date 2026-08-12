"""Exact coupled-core certificate for the unrestricted Dirac--Gram margin.

The unrestricted Fourier reduction has a trivial amplitude ``A0`` and two
amplitudes, ``Aeg`` and ``Adh``, at first radial order.  This module proves

    A0**2 - Aeg**2 - Adh**2 >= 0

on the complete seven-cube with ``0 <= c**2 <= 2/3``.  The square removes all
circle radicals.  Native FLINT multiplication and an exact tensor-product
Bernstein transform provide the sign certificate.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from fractions import Fraction
from pathlib import Path

import sympy as sp
from flint import ctx, fmpz, fmpz_mat, fmpz_mpoly_ctx

from spin8_resource_limits import constrain_current_process

TRIVIAL = (0, 0, 0, 0, 0, 0, 0)
EG = (0, 0, 1, 1, 0, 0, 1)
DH = (0, 1, 0, 0, 1, 0, 1)
CAYLEY_CUTOFF = Fraction(2, 3)


def _read_integer_polynomial(
    context: fmpz_mpoly_ctx, coefficient_dir: Path, mask: tuple[int, ...]
):
    path = coefficient_dir / f"alpha_sector_{''.join(map(str, mask))}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = json.load(handle)["coefficient_rows"]
    coefficients = {}
    for row in rows:
        value = Fraction(row["coefficient"])
        if 4 % value.denominator:
            raise AssertionError("unexpected unrestricted coefficient denominator")
        coefficients[tuple(row["powers"])] = int(4 * value)
    return context.from_dict(coefficients)


def _bernstein_matrix(degree: int) -> tuple[fmpz_mat, int]:
    scale = math.lcm(*(math.comb(degree, index) for index in range(degree + 1)))
    matrix = fmpz_mat(
        degree + 1,
        degree + 1,
        [
            fmpz(
                math.comb(target, source) * scale // math.comb(degree, source)
                if source <= target
                else 0
            )
            for target in range(degree + 1)
            for source in range(degree + 1)
        ],
    )
    return matrix, scale


def _transform_axis(
    data: list[fmpz],
    *,
    axis: int,
    shape: tuple[int, ...],
    matrix: fmpz_mat,
) -> list[fmpz]:
    size = math.prod(shape)
    axis_size = shape[axis]
    stride = math.prod(shape[axis + 1 :])
    block = axis_size * stride
    columns = size // axis_size
    entries = []
    for coordinate in range(axis_size):
        for outer in range(size // block):
            start = outer * block + coordinate * stride
            entries.extend(data[start : start + stride])
    transformed = (matrix * fmpz_mat(axis_size, columns, entries)).entries()
    result = [fmpz(0)] * size
    for coordinate in range(axis_size):
        offset = coordinate * columns
        for outer in range(size // block):
            source = offset + outer * stride
            target = outer * block + coordinate * stride
            result[target : target + stride] = transformed[source : source + stride]
    return result


def certificate(
    coefficient_dir: Path,
    *,
    flint_threads: int = 6,
) -> dict[str, object]:
    if not 1 <= flint_threads <= 7:
        raise ValueError("FLINT threads must leave at least one logical core free")
    resource = constrain_current_process(workers=flint_threads)
    ctx.threads = flint_threads
    polynomial_context = fmpz_mpoly_ctx.get(["ua", "ud", "ue", "ug", "uh", "ui", "z"])
    _ua, ud, ue, ug, uh, ui, z = polynomial_context.gens()
    trivial, eg, dh = (
        _read_integer_polynomial(polynomial_context, coefficient_dir, mask)
        for mask in (TRIVIAL, EG, DH)
    )
    eg_forced_square = ue * (1 - ue) * ug * (1 - ug) * (1 - uh) * (1 - ui) * z
    dh_forced_square = ud * (1 - ud) * (1 - ue) * uh * (1 - uh) * (1 - ui) * z
    # Every loaded polynomial is four times its exact amplitude polynomial, so
    # ``core`` is sixteen times the desired discriminant.
    core = trivial**2 - eg_forced_square * eg**2 - dh_forced_square * dh**2
    degrees = tuple(int(value) for value in core.degrees())
    expected_degrees = (6, 6, 6, 6, 6, 6, 4)
    if degrees != expected_degrees:
        raise AssertionError(f"unexpected core multidegree {degrees}")

    numerator = CAYLEY_CUTOFF.numerator
    denominator = CAYLEY_CUTOFF.denominator
    z_degree = degrees[-1]
    restricted = {
        powers: int(coefficient)
        * numerator ** powers[-1]
        * denominator ** (z_degree - powers[-1])
        for powers, coefficient in core.to_dict().items()
    }
    shape = tuple(degree + 1 for degree in degrees)
    size = math.prod(shape)
    values = [fmpz(0)] * size
    strides = tuple(math.prod(shape[axis + 1 :]) for axis in range(7))
    for powers, coefficient in restricted.items():
        flat = sum(
            power * stride for power, stride in zip(powers, strides, strict=True)
        )
        values[flat] = fmpz(coefficient)

    axis_scales = []
    for axis, degree in enumerate(degrees):
        matrix, scale = _bernstein_matrix(degree)
        values = _transform_axis(
            values,
            axis=axis,
            shape=shape,
            matrix=matrix,
        )
        axis_scales.append(scale)

    minimum = min(values)
    negative_count = sum(value < 0 for value in values)
    zero_count = sum(value == 0 for value in values)

    tau = sp.symbols("tau", real=True)
    obstruction = sp.factor(
        sp.Rational(32, 9) * (tau - 9) ** 2 * (tau**2 - 14 * tau + 9)
    )
    native_basis_cutoff = 7 - 2 * sp.sqrt(10)
    passed = bool(
        minimum == 0
        and negative_count == 0
        and len(values) == 588245
        and sp.factor(obstruction.subs(tau, native_basis_cutoff)) == 0
    )
    return {
        "experiment": "unrestricted coupled-core exact Bernstein certificate",
        "inequality": "A0^2 - Aeg^2 - Adh^2 >= 0",
        "domain": "(ua,ud,ue,ug,uh,ui) in [0,1]^6; z=c^2 in [0,2/3]",
        "integer_scaling": (
            "The stored sector polynomials are multiplied by 4; the certified "
            "integer polynomial is therefore 16 times the core discriminant."
        ),
        "core_term_count": len(core.to_dict()),
        "core_multidegree": list(degrees),
        "bernstein_tensor_shape": list(shape),
        "bernstein_coefficient_count": len(values),
        "axis_positive_scales": axis_scales,
        "minimum_scaled_bernstein_coefficient": str(minimum),
        "negative_scaled_bernstein_coefficient_count": negative_count,
        "zero_scaled_bernstein_coefficient_count": zero_count,
        "native_basis_first_obstruction_coefficient": str(obstruction),
        "native_basis_algebraic_cutoff": str(native_basis_cutoff),
        "certificate_basis_boundary": (
            "The obstruction beyond 7-2*sqrt(10) is a negative coefficient in "
            "this native Bernstein basis. It is not a counterexample to the "
            "core inequality."
        ),
        "resource_contract": resource,
        "scope_boundary": (
            "This controls the two first-order Fourier amplitudes. It does not "
            "bound the remaining thirteen amplitudes and does not prove the "
            "complete unrestricted physical margins."
        ),
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coefficient-dir",
        type=Path,
        default=Path("artifacts/spin8_dirac_unrestricted_coefficients_20260807"),
    )
    parser.add_argument("--flint-threads", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/spin8_dirac_unrestricted_core_20260807.json"),
    )
    arguments = parser.parse_args()
    report = certificate(
        arguments.coefficient_dir,
        flint_threads=arguments.flint_threads,
    )
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("unrestricted coupled-core certificate failed")


if __name__ == "__main__":
    main()
