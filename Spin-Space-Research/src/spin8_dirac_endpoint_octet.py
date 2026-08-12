"""Exact octet reduction and first Schur block on the adjacent endpoint face.

On ``ua=0`` and ``z=c**2=1``, exactly eight Walsh amplitudes survive.  They
form a three-dimensional binary subgroup.  Splitting that subgroup by the
``uh`` bit gives a four-element subgroup and one coset, hence the exact block
form

    K8 = [[X, sqrt(1-y**2) R], [sqrt(1-y**2) R, X]],

where ``uh=1-y**2`` and ``X,R`` are commuting Klein-four circulants.  This
module proves ``X >= 0`` on the complete five-cube and verifies the exact
radical-free second Schur block

    Z = X**2 - (1-y**2) R**2.

It also proves the scalar (one-by-one) principal minor of ``Z`` from the
already-certified global Fourier-energy inequality.  Separate modules now
prove the quadratic and cubic principal-minor families.  The fourth-order
determinant remains open, so the complete adjacent face is not yet a theorem.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from math import comb
from pathlib import Path

import sympy as sp
from flint import ctx, fmpz_mpoly_ctx

from spin8_dirac_endpoint_klein_face import certificate as klein_certificate
from spin8_dirac_final_residual import exact_full_chart_sign_certificate
from spin8_dirac_unrestricted_core import _read_integer_polynomial
from spin8_dirac_unrestricted_energy import (
    _flint_bernstein_stats,
    _sympy_bernstein_stats,
)
from spin8_dirac_unrestricted_grid import _sector_metadata
from spin8_resource_limits import constrain_current_process

TRIVIAL = (0, 0, 0, 0, 0, 0, 0)
H0 = (
    TRIVIAL,
    (0, 0, 1, 1, 0, 0, 1),
    (0, 1, 0, 1, 0, 1, 0),
    (0, 1, 1, 0, 0, 1, 1),
)
H1 = (
    (0, 0, 0, 1, 1, 1, 1),
    (0, 0, 1, 0, 1, 1, 0),
    (0, 1, 0, 0, 1, 0, 1),
    (0, 1, 1, 1, 1, 0, 0),
)
SURVIVING = H0 + H1


def _xor(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(a ^ b for a, b in zip(left, right, strict=True))


def _zero(context: fmpz_mpoly_ctx):
    return context.from_dict({})


def _one(context: fmpz_mpoly_ctx):
    return context.from_dict({(0,) * 5: 1})


def _to_endpoint_chart(polynomial, context: fmpz_mpoly_ctx):
    """Apply ``ua=0,z=1,uh=1-y**2`` using exact integer arithmetic."""

    coefficients: dict[tuple[int, ...], int] = {}
    for powers, coefficient in polynomial.to_dict().items():
        if powers[0]:
            continue
        base = (powers[1], powers[2], powers[3], powers[5])
        for index in range(powers[4] + 1):
            target = base + (2 * index,)
            coefficients[target] = coefficients.get(target, 0) + (
                int(coefficient) * comb(powers[4], index) * (-1) ** index
            )
    return context.from_dict(
        {
            powers: coefficient
            for powers, coefficient in coefficients.items()
            if coefficient
        }
    )


def _forced_square(context, variables, mask, complement, *, strip_h: bool = False):
    result = _one(context)
    for variable, axis in zip(variables[:4], (1, 2, 3, 5), strict=True):
        result *= variable ** mask[axis]
        result *= (1 - variable) ** complement[axis]
    if not strip_h:
        if mask[4]:
            result *= (1 - variables[4] ** 2) ** mask[4]
        result *= variables[4] ** (2 * complement[4])
    return result


def _triple_factor(context, variables, masks, complements):
    result = _one(context)
    for variable, axis in zip(variables[:4], (1, 2, 3, 5), strict=True):
        lower = sum(mask[axis] for mask in masks)
        upper = sum(complements[mask][axis] for mask in masks)
        if lower % 2 or upper % 2:
            raise AssertionError("triple product retained a spatial radical")
        result *= variable ** (lower // 2)
        result *= (1 - variable) ** (upper // 2)
    y_power = sum(complements[mask][4] for mask in masks)
    if y_power % 2:
        raise AssertionError("triple product retained the y radical")
    result *= variables[4] ** y_power
    return result


def _stats(polynomial) -> dict[str, object]:
    degrees = tuple(int(value) for value in polynomial.degrees())
    return {
        "power_term_count": len(polynomial.to_dict()),
        "multidegree": list(degrees),
        **_flint_bernstein_stats(polynomial, degrees),
    }


def _three_variable_expression(polynomial):
    ue, ug, y = sp.symbols("ue ug y", real=True)
    expression = sp.expand(
        sum(
            int(coefficient) * ue ** powers[1] * ug ** powers[2] * y ** powers[4]
            for powers, coefficient in polynomial.to_dict().items()
        )
    )
    return expression, ue, ug, y


def _z_quadratic_corner_certificate(faces) -> dict[str, object]:
    """Prove the three ``ud=ug=0`` quadratic faces are one exact square."""

    common = faces[0]
    faces_identical = all(face == common for face in faces[1:])
    ue, ui, y = sp.symbols("ue ui y", real=True)
    expression = sp.expand(
        sum(
            int(coefficient) * ue ** powers[1] * ui ** powers[3] * y ** powers[4]
            for powers, coefficient in common.to_dict().items()
        )
    )
    constant, factors = sp.factor_list(expression)
    square_verified = bool(
        constant == 1
        and len(factors) == 1
        and factors[0][1] == 2
        and tuple(sp.Poly(factors[0][0], ue, ui, y).degree_list()) == (6, 6, 12)
    )
    root = factors[0][0] if square_verified else sp.Integer(0)
    return {
        "face": "ud=ug=0",
        "three_quadratic_faces_are_identical": faces_identical,
        "common_face_multidegree": list(sp.Poly(expression, ue, ui, y).degree_list()),
        "common_face_power_term_count": len(sp.Poly(expression, ue, ui, y).terms()),
        "perfect_square": "P_6,6,12(ue,ui,y)^2",
        "square_root_multidegree": (
            list(sp.Poly(root, ue, ui, y).degree_list()) if square_verified else None
        ),
        "square_root_power_term_count": (
            len(sp.Poly(root, ue, ui, y).terms()) if square_verified else None
        ),
        "exact_factorization_verified": square_verified,
        "passed": bool(faces_identical and square_verified),
    }


def _determinant_corner_certificate(corner, *, ud_degree: int, ui_degree: int):
    """Certify the only non-native-positive remainder of ``det(X)``."""

    _ud, _ue, _ug, _ui, y_variable = corner.context().gens()
    quotient, remainder = divmod(corner, 1 - y_variable**2)
    if remainder:
        raise AssertionError("determinant corner lacks the expected 1-y**2 factor")

    expression, ue, ug, y = _three_variable_expression(quotient)
    radius, balance = sp.symbols("r s", real=True)
    lower = sp.expand(
        expression.subs({ue: radius * balance, ug: radius * (1 - balance)})
    )
    upper = sp.expand(
        expression.subs({ue: 1 - radius * balance, ug: 1 - radius * (1 - balance)})
    )
    upper_stats = _sympy_bernstein_stats(upper, (radius, balance, y))

    lower_face = sp.expand(lower.subs(y, 1))
    lower_interior = sp.expand(lower - lower_face * y ** sp.degree(lower, y))
    lower_interior_stats = _sympy_bernstein_stats(lower_interior, (radius, balance, y))

    constant, factors = sp.factor_list(lower_face)
    factor_by_degree = {
        tuple(sp.Poly(factor, radius, balance).degree_list()): (factor, multiplicity)
        for factor, multiplicity in factors
    }
    radius_factor, radius_multiplicity = factor_by_degree[(1, 0)]
    low_factor, low_multiplicity = factor_by_degree[(5, 6)]
    high_factor, high_multiplicity = factor_by_degree[(8, 8)]
    factorization_verified = bool(
        constant == -64
        and radius_factor == radius
        and radius_multiplicity == 3
        and low_multiplicity == high_multiplicity == 1
    )

    positive_low = -low_factor
    low_boundary = sp.expand(positive_low.subs(radius, 0))
    low_boundary_identity = sp.expand(low_boundary - 4096 * (2 * balance - 1) ** 2) == 0
    low_remainder = sp.expand(positive_low - low_boundary * (1 - radius) ** 5)
    low_remainder_stats = _sympy_bernstein_stats(low_remainder, (radius, balance))
    high_factor_stats = _sympy_bernstein_stats(high_factor, (radius, balance))

    passed = bool(
        upper_stats["negative_bernstein_coefficient_count"] == 0
        and lower_interior_stats["negative_bernstein_coefficient_count"] == 0
        and factorization_verified
        and low_boundary_identity
        and low_remainder_stats["negative_bernstein_coefficient_count"] == 0
        and high_factor_stats["negative_bernstein_coefficient_count"] == 0
    )
    return {
        "corner_selector_degrees": {"ud": ud_degree, "ui": ui_degree},
        "exact_factor": "1-y^2",
        "upper_triangle": {
            "chart": "ue=1-r*s; ug=1-r*(1-s)",
            **upper_stats,
        },
        "lower_triangle": {
            "chart": "ue=r*s; ug=r*(1-s)",
            "y_one_factorization": "-64*r^3*F_5,6*F_8,8",
            "factorization_verified": factorization_verified,
            "negative_low_factor_boundary_identity": ("(-F_5,6)|r=0 = 4096*(2*s-1)^2"),
            "negative_low_factor_boundary_identity_verified": low_boundary_identity,
            "negative_low_factor_remainder": low_remainder_stats,
            "positive_high_factor": high_factor_stats,
            "y_interior_remainder": lower_interior_stats,
        },
        "passed": passed,
    }


def _product_residual(
    residuals,
    complements,
    variables,
    left,
    right,
    target,
    *,
    h_zero_subgroup: bool,
):
    result = residuals[left] * residuals[right]
    for variable, axis in zip(variables[:4], (1, 2, 3, 5), strict=True):
        lower = left[axis] + right[axis] - target[axis]
        upper = (
            complements[left][axis]
            + complements[right][axis]
            - complements[target][axis]
        )
        if lower < 0 or upper < 0 or lower % 2 or upper % 2:
            raise AssertionError("coset convolution retained a spatial radical")
        result *= variable ** (lower // 2)
        result *= (1 - variable) ** (upper // 2)
    if h_zero_subgroup:
        y_power = complements[left][4] + complements[right][4] - complements[target][4]
        if y_power < 0 or y_power % 2:
            raise AssertionError("subgroup convolution retained the y radical")
        result *= variables[4] ** y_power
    return result


def _convolution(
    family,
    target,
    residuals,
    complements,
    variables,
    zero,
    *,
    h_zero_subgroup: bool,
):
    return sum(
        (
            _product_residual(
                residuals,
                complements,
                variables,
                left,
                _xor(left, target),
                target,
                h_zero_subgroup=h_zero_subgroup,
            )
            for left in family
        ),
        zero,
    )


def certificate(
    coefficient_dir: Path,
    *,
    flint_threads: int = 6,
    energy_artifact: Path = Path(
        "artifacts/spin8_dirac_unrestricted_energy_20260807.json"
    ),
) -> dict[str, object]:
    if not 1 <= flint_threads <= 7:
        raise ValueError("FLINT threads must leave at least one logical core free")
    resource = constrain_current_process(workers=flint_threads)
    ctx.threads = flint_threads

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
        for mask in SURVIVING
    }
    forced_squares = {
        mask: _forced_square(context5, variables, mask, complements[mask])
        for mask in masks
    }
    # ``_forced_square`` lives in the reduced five-variable chart, so the two
    # discarded face factors must be tested here rather than inferred from that
    # reduced polynomial.  At ua=0 a lower ``ua`` factor kills the sector; at
    # z=1 an upper ``1-z`` factor kills it.
    actual_survivors = tuple(
        mask
        for mask in masks
        if mask[0] == 0
        and complements[mask][6] == 0
        and (mask == TRIVIAL or forced_squares[mask] != 0)
    )
    subgroup_closed = {
        _xor(left, right) for left in SURVIVING for right in SURVIVING
    } == set(SURVIVING)
    h0_closed = {_xor(left, right) for left in H0 for right in H0} == set(H0)
    h1_is_coset = all(_xor(H1[0], mask) in H0 for mask in H1)

    metadata_masks, _metadata_complements, _representatives, hadamard = (
        _sector_metadata()
    )
    metadata_index = {mask: index for index, mask in enumerate(metadata_masks)}
    sign_patterns = Counter(
        tuple(hadamard[metadata_index[mask]][column] for mask in SURVIVING[1:])
        for column in range(16)
    )
    multiplicities_verified = len(sign_patterns) == 8 and set(
        sign_patterns.values()
    ) == {2}

    center = residuals[TRIVIAL]
    x_modes = H0[1:]
    x_squares = tuple(forced_squares[mask] * residuals[mask] ** 2 for mask in x_modes)
    x_triple = (
        _triple_factor(context5, variables, x_modes, complements)
        * residuals[x_modes[0]]
        * residuals[x_modes[1]]
        * residuals[x_modes[2]]
    )
    x_minors = (
        center,
        *(center**2 - square for square in x_squares),
        center**3 - center * sum(x_squares, zero) + 2 * x_triple,
        (
            center**4
            - 2 * center**2 * sum(x_squares, zero)
            + 8 * center * x_triple
            + sum((square**2 for square in x_squares), zero)
            - 2
            * (
                x_squares[0] * x_squares[1]
                + x_squares[0] * x_squares[2]
                + x_squares[1] * x_squares[2]
            )
        ),
    )

    klein = klein_certificate(coefficient_dir, flint_threads=flint_threads)
    x_rows = []
    x_native_audit = []
    for index, polynomial in enumerate(x_minors):
        degree_y = int(polynomial.degrees()[4])
        face = polynomial.subs({"y": 1})
        remainder = polynomial - face * variables[4] ** degree_y
        x_native_audit.append(_stats(polynomial))
        if index < len(x_minors) - 1:
            remainder_stats = _stats(remainder)
            x_rows.append(
                {
                    "minor_index": index,
                    "y_degree": degree_y,
                    "y_one_face_delegated_to_klein_certificate": True,
                    "interior_remainder": remainder_stats,
                    "passed": remainder_stats["negative_scaled_coefficient_count"] == 0,
                }
            )
            continue

        ud_degree, _ue_degree, _ug_degree, ui_degree, _y_degree = map(
            int, remainder.degrees()
        )
        corner = remainder.subs({"ud": 0, "ui": 0})
        interior = (
            remainder
            - corner * (1 - variables[0]) ** ud_degree * (1 - variables[3]) ** ui_degree
        )
        interior_stats = _stats(interior)
        corner_report = _determinant_corner_certificate(
            corner, ud_degree=ud_degree, ui_degree=ui_degree
        )
        x_rows.append(
            {
                "minor_index": index,
                "y_degree": degree_y,
                "y_one_face_delegated_to_klein_certificate": True,
                "corner_selector": (f"(1-ud)^{ud_degree}*(1-ui)^{ui_degree}"),
                "corner": corner_report,
                "five_variable_interior_remainder": interior_stats,
                "passed": bool(
                    corner_report["passed"]
                    and interior_stats["negative_scaled_coefficient_count"] == 0
                ),
            }
        )

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
    z_residuals = {
        target: convolution_x[target] - (1 - variables[4] ** 2) * convolution_r[target]
        for target in H0
    }

    energy = center**2 - sum(x_squares, zero)
    for mask in H1:
        stripped_square = _forced_square(
            context5,
            variables,
            mask,
            complements[mask],
            strip_h=True,
        )
        energy -= (1 - variables[4] ** 2) * stripped_square * residuals[mask] ** 2
    z_center_identity = z_residuals[TRIVIAL] == energy + 2 * sum(x_squares, zero)
    energy_report = json.loads(energy_artifact.read_text(encoding="utf-8"))
    energy_dependency_passed = bool(energy_report.get("passed"))

    z_quadratic_corner_faces = []
    for mask in H0[1:]:
        z_square = forced_squares[mask] * z_residuals[mask] ** 2
        z_quadratic_corner_faces.append(
            (z_residuals[TRIVIAL] ** 2 - z_square).subs({"ud": 0, "ug": 0})
        )
    z_quadratic_corner = _z_quadratic_corner_certificate(
        tuple(z_quadratic_corner_faces)
    )

    x_passed = bool(klein["passed"] and all(row["passed"] for row in x_rows))
    reduction_passed = bool(
        set(actual_survivors) == set(SURVIVING)
        and subgroup_closed
        and h0_closed
        and h1_is_coset
        and multiplicities_verified
        and z_center_identity
        and energy_dependency_passed
        and z_quadratic_corner["passed"]
    )
    passed = bool(x_passed and reduction_passed)
    return {
        "experiment": "adjacent Cayley-endpoint octet reduction",
        "domain": "ua=0; z=1; uh=1-y^2; (ud,ue,ug,ui,y) in [0,1]^5",
        "surviving_masks": [list(mask) for mask in actual_survivors],
        "survivors_form_z2_cubed": subgroup_closed,
        "h_zero_subgroup_is_klein_four": h0_closed,
        "h_one_family_is_its_coset": h1_is_coset,
        "physical_sign_patterns": [
            {"signs": list(pattern), "multiplicity": multiplicity}
            for pattern, multiplicity in sorted(sign_patterns.items())
        ],
        "eight_patterns_each_have_multiplicity_two": multiplicities_verified,
        "exact_block_reduction": (
            "K8=[[X,sqrt(1-y^2)R],[sqrt(1-y^2)R,X]], "
            "with commuting Klein-four circulants X and R"
        ),
        "exact_schur_reduction": ("K8>=0 iff X>=0 and Z=X^2-(1-y^2)R^2>=0"),
        "x_block": {
            "native_basis_audit": x_native_audit,
            "y_one_klein_certificate_passed": bool(klein["passed"]),
            "principal_minor_certificates": x_rows,
            "passed": x_passed,
        },
        "z_block": {
            "coefficient_term_counts": {
                "center": len(z_residuals[TRIVIAL].to_dict()),
                "nontrivial": [len(z_residuals[mask].to_dict()) for mask in H0[1:]],
            },
            "center_identity": (
                "Z_0 = FourierEnergy + 2*sum(nontrivial X amplitudes squared)"
            ),
            "center_identity_verified": z_center_identity,
            "global_fourier_energy_dependency_passed": energy_dependency_passed,
            "center_nonnegative": bool(z_center_identity and energy_dependency_passed),
            "common_quadratic_corner": z_quadratic_corner,
            "higher_principal_minors": "open",
        },
        "resource_contract": resource,
        "scope_boundary": (
            "This proves the exact octet/Schur reduction, the complete X-block, "
            "and the scalar Z minor. The three quadratic/cubic/determinant Z "
            "families remain open; therefore this is not a proof of the complete "
            "adjacent endpoint face or the unrestricted seven-variable theorem."
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
        "--energy-artifact",
        type=Path,
        default=Path("artifacts/spin8_dirac_unrestricted_energy_20260807.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/spin8_dirac_endpoint_octet_20260807.json"),
    )
    arguments = parser.parse_args()
    report = certificate(
        arguments.coefficient_dir,
        flint_threads=arguments.flint_threads,
        energy_artifact=arguments.energy_artifact,
    )
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("adjacent endpoint octet reduction failed")


if __name__ == "__main__":
    main()
