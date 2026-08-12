"""Exact PSD certificate on a four-variable Cayley-endpoint face.

On ``ua=uh=0`` and ``z=c**2=1``, exactly three nontrivial Walsh
amplitudes survive.  Their masks form the nonzero elements of a Klein-four
subgroup.  The sixteen physical orientation margins therefore reduce to the
four eigenvalues of a symmetric 4-by-4 group-circulant matrix, each repeated
four times.

This module proves that matrix positive semidefinite on the complete
``(ud,ue,ug,ui)`` cube.  It is an exact theorem on this endpoint face, not a
proof of the unrestricted seven-variable Dirac--Gram inequality.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import sympy as sp
from flint import ctx, fmpz_mpoly_ctx

from spin8_dirac_final_residual import exact_full_chart_sign_certificate
from spin8_dirac_unrestricted_core import _read_integer_polynomial
from spin8_dirac_unrestricted_energy import (
    _flint_bernstein_stats,
    _sympy_bernstein_stats,
)
from spin8_dirac_unrestricted_grid import _sector_metadata
from spin8_resource_limits import constrain_current_process

TRIVIAL = (0, 0, 0, 0, 0, 0, 0)
FIRST = (0, 0, 1, 1, 0, 0, 1)
SECOND = (0, 1, 0, 1, 0, 1, 0)
THIRD = (0, 1, 1, 0, 0, 1, 1)
SURVIVING = (TRIVIAL, FIRST, SECOND, THIRD)
FACE_SUBSTITUTIONS = {"ua": 0, "uh": 0, "z": 1}


def _zero(context: fmpz_mpoly_ctx):
    return context.from_dict({})


def _one(context: fmpz_mpoly_ctx):
    return context.from_dict({(0,) * 7: 1})


def _forced_square(context, variables, mask, complement):
    result = _one(context)
    for axis, variable in enumerate(variables):
        result *= variable ** mask[axis]
        result *= (1 - variable) ** complement[axis]
    return result.subs(FACE_SUBSTITUTIONS)


def _triple_radical_product(context, variables, masks, complements):
    """Return the polynomial product of the three forced square roots."""

    result = _one(context)
    for axis, variable in enumerate(variables):
        lower_power = sum(mask[axis] for mask in masks)
        complement_power = sum(complements[mask][axis] for mask in masks)
        if lower_power % 2 or complement_power % 2:
            raise AssertionError("Klein-face triple product retained a radical")
        result *= variable ** (lower_power // 2)
        result *= (1 - variable) ** (complement_power // 2)
    return result.subs(FACE_SUBSTITUTIONS)


def _two_variable_expression(polynomial) -> tuple[sp.Expr, sp.Symbol, sp.Symbol]:
    left, right = sp.symbols("U G", real=True)
    expression = sp.expand(
        sum(
            int(coefficient) * left ** powers[2] * right ** powers[3]
            for powers, coefficient in polynomial.to_dict().items()
        )
    )
    return expression, left, right


def _triangular_face_certificate(polynomial) -> dict[str, object]:
    expression, left, right = _two_variable_expression(polynomial)
    symmetric = (
        sp.expand(expression - expression.xreplace({left: right, right: left})) == 0
    )
    symmetric_form, remainder, mapping = sp.symmetrize(
        expression, [left, right], formal=True
    )
    if remainder != 0 or not symmetric:
        raise AssertionError("one-mode endpoint face is not symmetric")
    total = next(
        symbol for symbol, value in mapping if sp.expand(value - left - right) == 0
    )
    product = next(
        symbol for symbol, value in mapping if sp.expand(value - left * right) == 0
    )
    radius, balance = sp.symbols("r q", real=True)
    lower = sp.cancel(
        symmetric_form.subs({total: radius, product: radius**2 * balance / 4})
        / radius**2
    )
    lower_stats = _sympy_bernstein_stats(lower, (radius, balance))

    complement_left, complement_right = sp.symbols("X Y", real=True)
    upper_expression = sp.expand(
        expression.subs({left: 1 - complement_left, right: 1 - complement_right})
    )
    upper_form, upper_remainder, upper_mapping = sp.symmetrize(
        upper_expression, [complement_left, complement_right], formal=True
    )
    if upper_remainder != 0:
        raise AssertionError("complementary one-mode face is not symmetric")
    upper_total = next(
        symbol
        for symbol, value in upper_mapping
        if sp.expand(value - complement_left - complement_right) == 0
    )
    upper_product = next(
        symbol
        for symbol, value in upper_mapping
        if sp.expand(value - complement_left * complement_right) == 0
    )
    upper = upper_form.subs(
        {upper_total: radius, upper_product: radius**2 * balance / 4}
    )
    upper_stats = _sympy_bernstein_stats(upper, (radius, balance))
    passed = bool(
        lower_stats["negative_bernstein_coefficient_count"] == 0
        and upper_stats["negative_bernstein_coefficient_count"] == 0
    )
    return {
        "lower_triangle": {
            "chart": "r=U+G; q=4*U*G/r^2; exact r^2 factor removed",
            **lower_stats,
        },
        "upper_triangle": {
            "chart": "r=(1-U)+(1-G); q=4*(1-U)*(1-G)/r^2",
            **upper_stats,
        },
        "passed": passed,
    }


def _stats(polynomial) -> dict[str, object]:
    degrees = tuple(int(value) for value in polynomial.degrees())
    return {
        "power_term_count": len(polynomial.to_dict()),
        "multidegree": list(degrees),
        **_flint_bernstein_stats(polynomial, degrees),
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
    rows = chart["chart_characters"]
    masks = [tuple(row["lower_mask"]) for row in rows]
    complements = {
        tuple(row["lower_mask"]): tuple(row["complement_mask"]) for row in rows
    }
    context = fmpz_mpoly_ctx.get(["ua", "ud", "ue", "ug", "uh", "ui", "z"])
    variables = context.gens()
    zero = _zero(context)
    residuals = {
        mask: _read_integer_polynomial(context, coefficient_dir, mask).subs(
            FACE_SUBSTITUTIONS
        )
        for mask in SURVIVING
    }
    forced_squares = {
        mask: _forced_square(context, variables, mask, complements[mask])
        for mask in masks
    }
    actual_survivors = tuple(
        mask for mask in masks if mask == TRIVIAL or forced_squares[mask] != 0
    )
    subgroup_closed = bool(
        {
            tuple(left ^ right for left, right in zip(a, b, strict=True))
            for a in SURVIVING
            for b in SURVIVING
        }
        == set(SURVIVING)
    )
    if actual_survivors != SURVIVING or not subgroup_closed:
        raise AssertionError("endpoint face did not reduce to the expected Klein four")

    metadata_masks, _metadata_complements, _representatives, hadamard = (
        _sector_metadata()
    )
    metadata_index = {mask: index for index, mask in enumerate(metadata_masks)}
    sign_patterns = Counter(
        tuple(hadamard[metadata_index[mask]][column] for mask in SURVIVING[1:])
        for column in range(16)
    )
    expected_patterns = {
        (1, 1, 1): 4,
        (1, -1, -1): 4,
        (-1, 1, -1): 4,
        (-1, -1, 1): 4,
    }
    eigenvalue_multiplicities_verified = sign_patterns == expected_patterns

    center = residuals[TRIVIAL]
    modes = tuple(residuals[mask] for mask in SURVIVING[1:])
    mode_squares = tuple(
        forced_squares[mask] * mode**2
        for mask, mode in zip(SURVIVING[1:], modes, strict=True)
    )
    triple = (
        _triple_radical_product(context, variables, SURVIVING[1:], complements)
        * modes[0]
        * modes[1]
        * modes[2]
    )

    one_by_one = center
    two_by_two = tuple(center**2 - square for square in mode_squares)
    three_by_three = center**3 - center * sum(mode_squares, zero) + 2 * triple
    determinant = (
        center**4
        - 2 * center**2 * sum(mode_squares, zero)
        + 8 * center * triple
        + sum((square**2 for square in mode_squares), zero)
        - 2
        * (
            mode_squares[0] * mode_squares[1]
            + mode_squares[0] * mode_squares[2]
            + mode_squares[1] * mode_squares[2]
        )
    )

    corner_substitutions = {"ud": 0, "ui": 0}
    one_mode_face = two_by_two[0].subs(corner_substitutions)
    one_mode_certificate = _triangular_face_certificate(one_mode_face)

    pair_remainder = (
        two_by_two[0]
        - one_mode_face * (1 - variables[1]) ** 6 * (1 - variables[5]) ** 6
    )
    pair_remainder_stats = _stats(pair_remainder)
    two_stats = [_stats(value) for value in two_by_two]

    cubic_face = three_by_three.subs(corner_substitutions)
    cubic_face_identity = (
        cubic_face == center.subs(corner_substitutions) * one_mode_face
    )
    cubic_remainder = (
        three_by_three - cubic_face * (1 - variables[1]) ** 9 * (1 - variables[5]) ** 9
    )
    cubic_remainder_stats = _stats(cubic_remainder)

    ud_face = determinant.subs({"ud": 0})
    ud_face_square_identity = ud_face == two_by_two[0].subs({"ud": 0}) ** 2
    first_remainder = determinant - ud_face * (1 - variables[1]) ** 12
    ui_boundary = first_remainder.subs({"ui": 0})
    quotient, quotient_remainder = divmod(ui_boundary, variables[1])
    if quotient_remainder:
        raise AssertionError("nested determinant boundary is not divisible by ud")
    quotient_corner = quotient.subs({"ud": 0})
    quotient_interior = quotient - quotient_corner * (1 - variables[1]) ** 11
    quotient_interior_stats = _stats(quotient_interior)

    quotient_corner_expression, left, right = _two_variable_expression(quotient_corner)
    one_mode_expression, _, _ = _two_variable_expression(one_mode_face)
    positive_factor, division_remainder = sp.div(
        quotient_corner_expression, one_mode_expression, left, right
    )
    corner_factor_identity = sp.expand(division_remainder) == 0
    positive_factor_stats = _sympy_bernstein_stats(positive_factor, (left, right))

    second_remainder = first_remainder - ui_boundary * (1 - variables[5]) ** 11
    second_remainder_stats = _stats(second_remainder)

    first_stats = _stats(one_by_one)
    passed = bool(
        actual_survivors == SURVIVING
        and subgroup_closed
        and eigenvalue_multiplicities_verified
        and first_stats["negative_scaled_coefficient_count"] == 0
        and one_mode_certificate["passed"]
        and pair_remainder_stats["negative_scaled_coefficient_count"] == 0
        and all(row["negative_scaled_coefficient_count"] == 0 for row in two_stats[1:])
        and cubic_face_identity
        and cubic_remainder_stats["negative_scaled_coefficient_count"] == 0
        and ud_face_square_identity
        and corner_factor_identity
        and positive_factor_stats["negative_bernstein_coefficient_count"] == 0
        and quotient_interior_stats["negative_scaled_coefficient_count"] == 0
        and second_remainder_stats["negative_scaled_coefficient_count"] == 0
    )
    return {
        "experiment": "Cayley-endpoint Klein-four face PSD certificate",
        "domain": "ua=uh=0; z=1; (ud,ue,ug,ui) in [0,1]^4",
        "surviving_masks": [list(mask) for mask in actual_survivors],
        "surviving_masks_form_klein_four": subgroup_closed,
        "physical_sign_patterns": [
            {"signs": list(pattern), "multiplicity": multiplicity}
            for pattern, multiplicity in sorted(sign_patterns.items())
        ],
        "group_circulant_eigenvalue_multiplicities_verified": (
            eigenvalue_multiplicities_verified
        ),
        "orientation_multiplicity": 4,
        "group_circulant": (
            "[[x,a,b,c],[a,x,c,b],[b,c,x,a],[c,b,a,x]]; "
            "its four eigenvalues are the physical margins on this face"
        ),
        "one_by_one_minor": first_stats,
        "two_by_two_minors": two_stats,
        "one_mode_boundary_certificate": one_mode_certificate,
        "first_pair_boundary_remainder": pair_remainder_stats,
        "three_by_three_minor": {
            "formula": "x^3-x*(a^2+b^2+c^2)+2*a*b*c",
            "corner_identity": "x*(x^2-a^2)",
            "corner_identity_verified": cubic_face_identity,
            "remainder": cubic_remainder_stats,
        },
        "four_by_four_determinant": {
            "formula": (
                "x^4-2*x^2*(a^2+b^2+c^2)+8*x*a*b*c+"
                "a^4+b^4+c^4-2*(a^2*b^2+a^2*c^2+b^2*c^2)"
            ),
            "ud_zero_face_is_square": ud_face_square_identity,
            "nested_ui_boundary_divisible_by_ud": True,
            "nested_corner_factorization_verified": corner_factor_identity,
            "nested_corner_positive_factor": positive_factor_stats,
            "nested_quotient_interior_remainder": quotient_interior_stats,
            "final_interior_remainder": second_remainder_stats,
        },
        "resource_contract": resource,
        "scope_boundary": (
            "This proves all physical margins on one complete four-variable "
            "Cayley-endpoint face. It does not prove the other endpoint faces "
            "or the unrestricted seven-variable inequality."
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
        default=Path("artifacts/spin8_dirac_endpoint_klein_face_20260807.json"),
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
        raise SystemExit("Cayley-endpoint Klein-four face certificate failed")


if __name__ == "__main__":
    main()
