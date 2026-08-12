"""Exact full-sector Fourier-energy certificate for the Dirac--Gram margin.

Let ``A0`` denote the trivial Walsh amplitude of the sixteen physical
orientation margins and let ``A_mu`` denote the other fifteen amplitudes.
This module proves

    A0**2 - sum(A_mu**2 for mu != 0) >= 0

on the complete seven-cube.  All circle radicals disappear after squaring.
The proof uses native FLINT polynomial arithmetic, two symmetric triangular
face charts, and exact tensor-product Bernstein transforms.

This is an aggregate Fourier-energy theorem.  It is not, by itself, a proof
that every physical orientation margin is nonnegative.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

import sympy as sp
from flint import ctx, fmpz, fmpz_mpoly_ctx

from spin8_dirac_final_residual import exact_full_chart_sign_certificate
from spin8_dirac_unrestricted_core import (
    _bernstein_matrix,
    _read_integer_polynomial,
    _transform_axis,
)
from spin8_resource_limits import constrain_current_process

TRIVIAL = (0, 0, 0, 0, 0, 0, 0)
CAYLEY_CUTOFF = Fraction(2, 3)


def _flint_bernstein_stats(
    polynomial,
    degrees: tuple[int, ...],
    *,
    z_cutoff: Fraction | None = None,
) -> dict[str, object]:
    coefficients = polynomial.to_dict()
    if z_cutoff is not None:
        z_degree = degrees[-1]
        coefficients = {
            powers: int(coefficient)
            * z_cutoff.numerator ** powers[-1]
            * z_cutoff.denominator ** (z_degree - powers[-1])
            for powers, coefficient in coefficients.items()
        }
    shape = tuple(degree + 1 for degree in degrees)
    size = math.prod(shape)
    values = [fmpz(0)] * size
    strides = tuple(math.prod(shape[axis + 1 :]) for axis in range(len(shape)))
    for powers, coefficient in coefficients.items():
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

    negative_rows = []
    for flat, value in enumerate(values):
        if value >= 0:
            continue
        remainder = flat
        index = []
        for stride in strides:
            coordinate, remainder = divmod(remainder, stride)
            index.append(coordinate)
        negative_rows.append(
            {"bernstein_index": index, "scaled_coefficient": str(value)}
        )
    return {
        "tensor_shape": list(shape),
        "coefficient_count": len(values),
        "axis_positive_scales": axis_scales,
        "minimum_scaled_coefficient": str(min(values)),
        "negative_scaled_coefficient_count": len(negative_rows),
        "zero_scaled_coefficient_count": sum(value == 0 for value in values),
        "negative_rows": negative_rows,
    }


def _sympy_bernstein_stats(
    expression: sp.Expr, variables: tuple[sp.Symbol, ...]
) -> dict[str, object]:
    polynomial = sp.Poly(sp.expand(expression), *variables)
    degrees = tuple(polynomial.degree(variable) for variable in variables)
    coefficients = {
        powers: Fraction(coefficient) for powers, coefficient in polynomial.terms()
    }
    for axis, degree in enumerate(degrees):
        transformed = {}
        other_indices = {powers[:axis] + powers[axis + 1 :] for powers in coefficients}
        for other in other_indices:
            power_line = [
                coefficients.get(other[:axis] + (source,) + other[axis:], Fraction(0))
                for source in range(degree + 1)
            ]
            for target in range(degree + 1):
                transformed[other[:axis] + (target,) + other[axis:]] = sum(
                    power_line[source]
                    * Fraction(math.comb(target, source), math.comb(degree, source))
                    for source in range(target + 1)
                )
        coefficients = transformed
    values = list(coefficients.values())
    return {
        "multidegree": list(degrees),
        "power_term_count": len(polynomial.terms()),
        "bernstein_coefficient_count": len(values),
        "minimum_bernstein_coefficient": str(min(values)),
        "negative_bernstein_coefficient_count": sum(value < 0 for value in values),
        "zero_bernstein_coefficient_count": sum(value == 0 for value in values),
    }


def _high_cayley_face_charts(face) -> dict[str, object]:
    x, y, z = sp.symbols("x y z", real=True)
    expression = sum(
        int(coefficient) * x ** powers[2] * y ** powers[3] * z ** powers[6]
        for powers, coefficient in face.to_dict().items()
    )
    symmetric = sp.expand(expression - expression.xreplace({x: y, y: x})) == 0
    symmetric_form, remainder, mapping = sp.symmetrize(expression, [x, y], formal=True)
    if remainder != 0:
        raise AssertionError("endpoint face is not symmetric")
    sigma = next(left for left, right in mapping if sp.expand(right - x - y) == 0)
    product = next(left for left, right in mapping if sp.expand(right - x * y) == 0)
    radius, balance, cayley = sp.symbols("r q t", real=True)
    z_high = sp.Rational(2, 3) + cayley / 3

    lower_numerator, lower_denominator = sp.cancel(
        symmetric_form.subs(
            {
                sigma: radius,
                product: radius**2 * balance / 4,
                z: z_high,
            }
        )
        / radius**2
    ).as_numer_denom()
    if lower_denominator.free_symbols or lower_denominator <= 0:
        raise AssertionError("lower triangular chart has a nonconstant denominator")
    lower = _sympy_bernstein_stats(
        lower_numerator / lower_denominator, (radius, balance, cayley)
    )

    complement_x, complement_y = sp.symbols("X Y", real=True)
    complementary_expression = sp.expand(
        expression.subs({x: 1 - complement_x, y: 1 - complement_y})
    )
    upper_form, upper_remainder, upper_mapping = sp.symmetrize(
        complementary_expression, [complement_x, complement_y], formal=True
    )
    if upper_remainder != 0:
        raise AssertionError("complementary endpoint face is not symmetric")
    upper_sigma = next(
        left
        for left, right in upper_mapping
        if sp.expand(right - complement_x - complement_y) == 0
    )
    upper_product = next(
        left
        for left, right in upper_mapping
        if sp.expand(right - complement_x * complement_y) == 0
    )
    upper_expression = upper_form.subs(
        {
            upper_sigma: radius,
            upper_product: radius**2 * balance / 4,
            z: z_high,
        }
    )
    upper = _sympy_bernstein_stats(upper_expression, (radius, balance, cayley))
    return {
        "face_symmetric_in_coupled_pair": symmetric,
        "high_cayley_parameter": "z=2/3+t/3, 0<=t<=1",
        "lower_triangle": {
            "domain": "x+y<=1; r=x+y; q=4xy/r^2",
            "removed_boundary_factor": "r^2",
            **lower,
        },
        "upper_triangle": {
            "domain": "x+y>=1; r=(1-x)+(1-y); q=4(1-x)(1-y)/r^2",
            **upper,
        },
        "passed": bool(
            symmetric
            and lower["negative_bernstein_coefficient_count"] == 0
            and upper["negative_bernstein_coefficient_count"] == 0
        ),
    }


def _third_symmetric_certificate(
    masks: list[tuple[int, ...]],
) -> dict[str, object]:
    a0, sector_energy, triple_sum = sp.symbols("A0 S T", real=True)
    power_one = 16 * a0
    power_two = 16 * (a0**2 + sector_energy)
    power_three = 16 * (a0**3 + 3 * a0 * sector_energy + 6 * triple_sum)
    elementary_one = power_one
    elementary_two = sp.expand((elementary_one * power_one - power_two) / 2)
    elementary_three = sp.factor(
        (elementary_two * power_one - elementary_one * power_two + power_three) / 3
    )
    energy = a0**2 - sector_energy
    expected = 112 * a0 * (4 * a0**2 + energy) + 32 * triple_sum
    zero = (0,) * 7
    triple_count = sum(
        tuple(left ^ middle ^ right for left, middle, right in zip(*triple)) == zero
        for triple in itertools.combinations(sorted(masks)[1:], 3)
    )
    identity = sp.expand(elementary_three - expected) == 0
    rational_constant_check = 3 * 448 - 64 == 1280
    return {
        "newton_identity": str(elementary_three),
        "identity": "e3 = 112*A0*(4*A0^2 + E) + 32*T",
        "identity_verified": identity,
        "nontrivial_zero_sum_triple_count": triple_count,
        "triple_convolution_identity": "6*T = <a*a,a>",
        "young_cauchy_bound": (
            "|T| <= sqrt(15)/6*(sum_{mu!=0} A_mu^2)^(3/2) " "<= 2/3*A0^3"
        ),
        "global_lower_bound": "e3 >= 1280/3*A0^3 >= 0",
        "rational_constant_check": rational_constant_check,
        "passed": bool(identity and triple_count == 35 and rational_constant_check),
    }


def _fourth_symmetric_certificate() -> dict[str, object]:
    radius = sp.symbols("r", nonnegative=True)
    lower_envelope = (
        sp.Integer(1820)
        - sp.Rational(91, 2) * radius
        - sp.Rational(13, 3) * radius ** sp.Rational(3, 2)
        - sp.Rational(1, 8) * radius**2
    )
    endpoint = sp.simplify(lower_envelope.subs(radius, 16))
    derivative = sp.diff(lower_envelope, radius)
    expected_derivative = (
        -sp.Rational(91, 2) - sp.Rational(13, 2) * sp.sqrt(radius) - radius / 4
    )
    passed = bool(
        endpoint == sp.Rational(2348, 3)
        and sp.simplify(derivative - expected_derivative) == 0
    )
    return {
        "normalization": "M_i=A0*(1+y_i); sum_i y_i=0; r=sum_i y_i^2<=16",
        "exact_expansion": (
            "e4/A0^4 = 1820 - 91*r/2 + 13*sum(y_i^3)/3 " "+ r^2/8 - sum(y_i^4)/4"
        ),
        "moment_bounds": ("sum(y_i^3)>=-r^(3/2) and sum(y_i^4)<=r^2"),
        "lower_envelope": str(lower_envelope),
        "lower_envelope_derivative": str(derivative),
        "endpoint_value_at_r_16": str(endpoint),
        "global_lower_bound": "e4 >= 2348/3*A0^4 >= 0",
        "passed": passed,
    }


def certificate(
    coefficient_dir: Path,
    *,
    flint_threads: int = 6,
) -> dict[str, object]:
    if not 1 <= flint_threads <= 7:
        raise ValueError("FLINT threads must leave at least one logical core free")
    resource = constrain_current_process(workers=flint_threads)
    ctx.threads = flint_threads

    chart = exact_full_chart_sign_certificate()
    if not chart["passed"]:
        raise AssertionError("full-chart sign certificate failed")
    character_rows = chart["chart_characters"]
    masks = [tuple(row["lower_mask"]) for row in character_rows]
    complements = {
        tuple(row["lower_mask"]): tuple(row["complement_mask"])
        for row in character_rows
    }

    polynomial_context = fmpz_mpoly_ctx.get(["ua", "ud", "ue", "ug", "uh", "ui", "z"])
    variables = polynomial_context.gens()
    trivial = _read_integer_polynomial(polynomial_context, coefficient_dir, TRIVIAL)
    energy = trivial**2
    for mask in masks:
        if mask == TRIVIAL:
            continue
        amplitude = _read_integer_polynomial(polynomial_context, coefficient_dir, mask)
        forced_square = 1
        for variable, lower_bit, complement_bit in zip(
            variables, mask, complements[mask], strict=True
        ):
            if lower_bit:
                forced_square *= variable
            if complement_bit:
                forced_square *= 1 - variable
        energy -= forced_square * amplitude**2

    degrees = tuple(int(value) for value in energy.degrees())
    expected_degrees = (6, 6, 6, 6, 6, 6, 4)
    if degrees != expected_degrees:
        raise AssertionError(f"unexpected Fourier-energy multidegree {degrees}")

    trivial_degrees = tuple(int(value) for value in trivial.degrees())
    trivial_sign = _flint_bernstein_stats(trivial, trivial_degrees)

    low_cayley = _flint_bernstein_stats(energy, degrees, z_cutoff=CAYLEY_CUTOFF)
    native_full = _flint_bernstein_stats(energy, degrees)

    ua, ud, ue, ug, uh, ui, _z = variables
    eg_face = energy.subs({"ua": 0, "ud": 0, "uh": 0, "ui": 0})
    dh_face = energy.subs({"ua": 0, "ue": 0, "ug": 0, "ui": 0})
    face_pair_identity = bool(eg_face.compose(ua, ud, ud, uh, uh, ui, _z) == dh_face)
    high_cayley_faces = _high_cayley_face_charts(eg_face)
    third_symmetric = _third_symmetric_certificate(masks)
    fourth_symmetric = _fourth_symmetric_certificate()

    eg_extension = (
        eg_face * (1 - ua) ** 6 * (1 - ud) ** 6 * (1 - uh) ** 6 * (1 - ui) ** 6
    )
    dh_extension = (
        dh_face * (1 - ua) ** 6 * (1 - ue) ** 6 * (1 - ug) ** 6 * (1 - ui) ** 6
    )
    remainder = energy - eg_extension - dh_extension
    remainder_stats = _flint_bernstein_stats(remainder, degrees)

    expected_obstruction_indices = {
        (0, 0, 1, 1, 0, 0, 3),
        (0, 0, 1, 1, 0, 0, 4),
        (0, 1, 0, 0, 1, 0, 3),
        (0, 1, 0, 0, 1, 0, 4),
    }
    observed_obstruction_indices = {
        tuple(row["bernstein_index"]) for row in native_full["negative_rows"]
    }
    passed = bool(
        low_cayley["negative_scaled_coefficient_count"] == 0
        and low_cayley["zero_scaled_coefficient_count"] == 35
        and trivial_sign["negative_scaled_coefficient_count"] == 0
        and trivial_sign["zero_scaled_coefficient_count"] == 3
        and observed_obstruction_indices == expected_obstruction_indices
        and face_pair_identity
        and high_cayley_faces["passed"]
        and remainder_stats["negative_scaled_coefficient_count"] == 0
        and remainder_stats["zero_scaled_coefficient_count"] == 495
        and third_symmetric["passed"]
        and fourth_symmetric["passed"]
    )
    return {
        "experiment": "unrestricted full-sector Fourier-energy certificate",
        "inequality": "A0^2 - sum_{mu != 0} A_mu^2 >= 0",
        "domain": "(ua,ud,ue,ug,uh,ui,z=c^2) in [0,1]^7",
        "integer_scaling": (
            "Each stored sector polynomial is multiplied by 4; the certified "
            "integer polynomial is therefore 16 times the exact energy margin."
        ),
        "sector_count": len(masks),
        "nontrivial_sector_count": len(masks) - 1,
        "energy_term_count": len(energy.to_dict()),
        "energy_multidegree": list(degrees),
        "nonnegative_mean_certificate": {
            "amplitude": "A0",
            "multidegree": list(trivial_degrees),
            **trivial_sign,
        },
        "low_cayley_native_certificate": {
            "domain": "z in [0,2/3]",
            **low_cayley,
        },
        "full_cube_native_basis_audit": native_full,
        "native_obstructions_are_exactly_two_coupled_faces": (
            observed_obstruction_indices == expected_obstruction_indices
        ),
        "coupled_face_pair_identity": face_pair_identity,
        "high_cayley_coupled_face_certificate": high_cayley_faces,
        "boundary_supported_decomposition": (
            "E = Eeg*(1-ua)^6*(1-ud)^6*(1-uh)^6*(1-ui)^6 + "
            "Edh*(1-ua)^6*(1-ue)^6*(1-ug)^6*(1-ui)^6 + R"
        ),
        "global_remainder_certificate": remainder_stats,
        "parseval_consequence": (
            "The root-mean-square deviation of the sixteen physical margins "
            "from their common nonnegative mean A0 is at most A0."
        ),
        "third_elementary_symmetric_certificate": third_symmetric,
        "fourth_elementary_symmetric_certificate": fourth_symmetric,
        "resource_contract": resource,
        "scope_boundary": (
            "This globally bounds the aggregate energy of all fifteen "
            "nontrivial Walsh amplitudes. It does not imply that every "
            "physical orientation margin is nonnegative and does not prove "
            "unrestricted Dirac--Gram."
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
        default=Path("artifacts/spin8_dirac_unrestricted_energy_20260807.json"),
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
        raise SystemExit("unrestricted Fourier-energy certificate failed")


if __name__ == "__main__":
    main()
