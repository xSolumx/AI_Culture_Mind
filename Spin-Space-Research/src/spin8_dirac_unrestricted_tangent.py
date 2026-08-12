"""Exact tangent cone of the unrestricted Dirac--Gram margin.

The orthonormal equality line is ``a=d=e=g=h=i=0`` with arbitrary Cayley
coordinate ``z=c^2``.  This certificate extracts the first nonzero radial
terms from the independently reconstructed sector polynomials and proves that
the resulting quadratic form is positive semidefinite for ``0 <= z <= 1``.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import sympy as sp

from spin8_dirac_final_residual import exact_full_chart_sign_certificate

Z = sp.symbols("z")
EPSILON, P, Q, R, T = sp.symbols("epsilon p q r t", real=True)
ALPHA, IOTA, X, Y, W = sp.symbols("alpha iota x y w", real=True)


def _read_rows(root: Path, mask: tuple[int, ...]) -> list[dict[str, object]]:
    path = root / f"alpha_sector_{''.join(map(str, mask))}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)["coefficient_rows"]


def _univariate_slice(
    rows: list[dict[str, object]], *, radial_degree: int, axis: int | None = None
) -> sp.Expr:
    result = sp.Integer(0)
    for row in rows:
        powers = row["powers"]
        if sum(powers[:6]) != radial_degree:
            continue
        if axis is not None and (
            powers[axis] != radial_degree
            or any(powers[index] for index in range(6) if index != axis)
        ):
            continue
        result += sp.Rational(row["coefficient"]) * Z ** powers[6]
    return sp.factor(result)


def _endpoint_null_cone_certificate(
    rows: dict[tuple[int, ...], list[dict[str, object]]],
    masks: list[tuple[int, ...]],
    characters: list[dict[str, object]],
) -> dict[str, object]:
    """Resolve the quartic lift of every tangent-null endpoint direction.

    At ``z=1`` the two nontrivial tangent blocks each have a one-dimensional
    kernel.  The signs of those kernels depend on the physical orientation.
    This calculation substitutes the appropriate signed null line into every
    one of the sixteen exact orientation margins and extracts its first
    nonzero term.
    """

    from spin8_dirac_unrestricted_grid import _sector_metadata

    complements = {
        tuple(row["lower_mask"]): tuple(row["complement_mask"]) for row in characters
    }
    epsilon = EPSILON
    # Lower coordinates are ordered (a,d,e,g,h,i,c).  The four live endpoint
    # directions are d=q, e=p, g=r, h=t; a=i=0 and c=1.
    lower = (0, epsilon * Q, epsilon * P, epsilon * R, epsilon * T, 0, 1)
    squared = tuple(value**2 for value in lower)
    complement = (
        1,
        sp.sqrt(1 - squared[1]),
        sp.sqrt(1 - squared[2]),
        sp.sqrt(1 - squared[3]),
        sp.sqrt(1 - squared[4]),
        1,
        0,
    )
    amplitudes = []
    for mask in masks:
        polynomial = sp.Integer(0)
        for row in rows[mask]:
            term = sp.Rational(row["coefficient"])
            for coordinate, power in zip(squared, row["powers"], strict=True):
                if power:
                    term *= coordinate**power
            polynomial += term
        forced = sp.prod(
            value**lower_bit * other**complement_bit
            for value, other, lower_bit, complement_bit in zip(
                lower, complement, mask, complements[mask], strict=True
            )
        )
        amplitudes.append(
            sp.series(forced * polynomial, epsilon, 0, 6).removeO().expand()
        )

    metadata_masks, _, representatives, hadamard = _sector_metadata()
    if tuple(masks) != metadata_masks:
        raise AssertionError("sector metadata order changed")
    expected_quartic = 128 * (P**2 + Q**2) ** 2
    orientation_rows = []
    for orientation, representative in enumerate(representatives):
        margin = sp.expand(
            sum(
                hadamard[sector][orientation] * amplitudes[sector]
                for sector in range(16)
            )
        )
        quadratic = sp.factor(margin.coeff(epsilon, 2))
        quartic = margin.coeff(epsilon, 4)
        dh_cross = sp.expand(quadratic).coeff(Q * T)
        eg_cross = sp.expand(quadratic).coeff(P * R)
        if abs(dh_cross) != 64 or abs(eg_cross) != 64:
            raise AssertionError("unexpected endpoint tangent block")
        h_over_d = -sp.sign(dh_cross)
        g_over_e = -sp.sign(eg_cross)
        null_quartic = sp.factor(quartic.subs({T: h_over_d * Q, R: g_over_e * P}))
        orientation_rows.append(
            {
                "orientation_index": orientation,
                "orientation_representative": list(representative),
                "quadratic_form": str(quadratic),
                "null_relation_h_over_d": int(h_over_d),
                "null_relation_g_over_e": int(g_over_e),
                "quartic_on_tangent_null_cone": str(null_quartic),
                "quartic_identity_verified": sp.expand(null_quartic - expected_quartic)
                == 0,
            }
        )
    passed = all(row["quartic_identity_verified"] for row in orientation_rows)
    return {
        "endpoint": "z=c^2=1",
        "orientation_rows": orientation_rows,
        "common_quartic_on_all_sixteen_tangent_null_cones": str(expected_quartic),
        "strictness_statement": (
            "The quartic is strictly positive for every nonzero endpoint "
            "tangent-null direction and vanishes only at p=q=0."
        ),
        "passed": passed,
    }


def _weighted_endpoint_blowup_certificate(
    rows: dict[tuple[int, ...], list[dict[str, object]]],
    masks: list[tuple[int, ...]],
    characters: list[dict[str, object]],
    endpoint: dict[str, object],
) -> dict[str, object]:
    """Extract the complete weighted leading form at the singular endpoint.

    The tangent-null residuals have weight one.  Distance from the calibrated
    endpoint also has weight one through ``s=epsilon*w``.  The two transverse
    residuals and deviations from each null relation have weight two.  Terms
    above weighted degree four are discarded exactly during multiplication.
    """

    from spin8_dirac_unrestricted_grid import _sector_metadata

    order = 5

    def add(left: list[sp.Expr], right: list[sp.Expr]) -> list[sp.Expr]:
        return [left[index] + right[index] for index in range(order)]

    def multiply(left: list[sp.Expr], right: list[sp.Expr]) -> list[sp.Expr]:
        result = [sp.Integer(0)] * order
        for left_degree, left_value in enumerate(left):
            if left_value == 0:
                continue
            for right_degree, right_value in enumerate(right[: order - left_degree]):
                if right_value != 0:
                    result[left_degree + right_degree] += left_value * right_value
        return result

    def power(value: list[sp.Expr], exponent: int) -> list[sp.Expr]:
        result = [sp.Integer(1)] + [sp.Integer(0)] * (order - 1)
        for _ in range(exponent):
            result = multiply(result, value)
        return result

    complements = {
        tuple(row["lower_mask"]): tuple(row["complement_mask"]) for row in characters
    }
    metadata_masks, _, representatives, hadamard = _sector_metadata()
    if tuple(masks) != metadata_masks:
        raise AssertionError("sector metadata order changed")

    sign_pairs = sorted(
        {
            (
                int(row["null_relation_h_over_d"]),
                int(row["null_relation_g_over_e"]),
            )
            for row in endpoint["orientation_rows"]
        }
    )
    expected = 16 * (
        5 * ALPHA**2
        + 5 * IOTA**2
        + 8 * P**4
        + 16 * P**2 * Q**2
        + 4 * P**2 * W**2
        + 8 * Q**4
        + 4 * Q**2 * W**2
        + 2 * X**2
        + 2 * Y**2
    )
    sign_rows = []
    for h_over_d, g_over_e in sign_pairs:
        # Coefficient lists are indexed by powers of epsilon.
        lower = (
            [0, 0, ALPHA, 0, 0],
            [0, Q, 0, 0, 0],
            [0, P, 0, 0, 0],
            [0, g_over_e * P, X, 0, 0],
            [0, h_over_d * Q, Y, 0, 0],
            [0, 0, IOTA, 0, 0],
            [1, 0, -(W**2) / 2, 0, -(W**4) / 8],
        )
        squared = tuple(multiply(value, value) for value in lower)
        complement = []
        for value in squared[:6]:
            complement.append(
                [
                    1,
                    0,
                    -value[2] / 2,
                    -value[3] / 2,
                    -value[4] / 2 - value[2] ** 2 / 8,
                ]
            )
        complement.append([0, W, 0, 0, 0])

        amplitudes = []
        powers = [
            [power(coordinate, exponent) for exponent in range(4)]
            for coordinate in squared
        ]
        one = [sp.Integer(1)] + [sp.Integer(0)] * (order - 1)
        for mask in masks:
            polynomial = [sp.Integer(0)] * order
            for row in rows[mask]:
                term = one
                for axis, exponent in enumerate(row["powers"]):
                    term = multiply(term, powers[axis][exponent])
                polynomial = add(
                    polynomial,
                    [sp.Rational(row["coefficient"]) * value for value in term],
                )
            forced = one
            for axis, (lower_bit, complement_bit) in enumerate(
                zip(mask, complements[mask], strict=True)
            ):
                if lower_bit:
                    forced = multiply(forced, lower[axis])
                if complement_bit:
                    forced = multiply(forced, complement[axis])
            amplitudes.append(multiply(forced, polynomial))

        orientation = next(
            row["orientation_index"]
            for row in endpoint["orientation_rows"]
            if row["null_relation_h_over_d"] == h_over_d
            and row["null_relation_g_over_e"] == g_over_e
        )
        leading = sp.factor(
            sum(
                hadamard[sector][orientation] * amplitudes[sector][4]
                for sector in range(16)
            )
        )
        sign_rows.append(
            {
                "null_relation_h_over_d": h_over_d,
                "null_relation_g_over_e": g_over_e,
                "representative_orientation_index": orientation,
                "orientation_representative": list(representatives[orientation]),
                "weighted_degree_four_form": str(leading),
                "common_form_identity_verified": sp.expand(leading - expected) == 0,
            }
        )

    passed = len(sign_rows) == 4 and all(
        row["common_form_identity_verified"] for row in sign_rows
    )
    return {
        "weighted_substitution": (
            "a=epsilon^2 alpha, i=epsilon^2 iota, d=epsilon q, "
            "e=epsilon p, h=sigma_d epsilon q+epsilon^2 y, "
            "g=sigma_e epsilon p+epsilon^2 x, s=epsilon w, "
            "c=sqrt(1-epsilon^2 w^2)"
        ),
        "sign_rows": sign_rows,
        "common_weighted_degree_four_form": str(sp.factor(expected)),
        "manifestly_nonnegative_decomposition": (
            "16[5 alpha^2 + 5 iota^2 + 8(p^2+q^2)^2 + "
            "4 w^2(p^2+q^2) + 2 x^2 + 2 y^2]"
        ),
        "strictness_statement": (
            "The weighted leading form is positive away from the blow-up origin."
        ),
        "passed": passed,
    }


def certificate(coefficient_dir: Path) -> dict[str, object]:
    chart = exact_full_chart_sign_certificate()
    characters = chart["chart_characters"]
    masks = [tuple(row["lower_mask"]) for row in characters]
    rows = {mask: _read_rows(coefficient_dir, mask) for mask in masks}
    trivial = (0, 0, 0, 0, 0, 0, 0)
    linear = [
        _univariate_slice(rows[trivial], radial_degree=1, axis=axis)
        for axis in range(6)
    ]
    z = Z
    alpha = sp.Rational(5, 2) * (9 - z) * (5 - z)
    beta = 2 * (9 - z) * (3 - z)
    expected_linear = [alpha, beta, beta, beta, beta, alpha]
    linear_identity = all(
        sp.expand(observed - expected) == 0
        for observed, expected in zip(linear, expected_linear, strict=True)
    )

    eg_mask = (0, 0, 1, 1, 0, 0, 1)
    dh_mask = (0, 1, 0, 0, 1, 0, 1)
    eg_constant = _univariate_slice(rows[eg_mask], radial_degree=0)
    dh_constant = _univariate_slice(rows[dh_mask], radial_degree=0)
    expected_eg = 8 * (9 - z)
    expected_dh = -8 * (9 - z)
    coupling_identity = bool(
        sp.expand(eg_constant - expected_eg) == 0
        and sp.expand(dh_constant - expected_dh) == 0
    )

    radial_rows = []
    for mask in masks:
        minimum_polynomial_degree = min(sum(row["powers"][:6]) for row in rows[mask])
        radial_order = sp.Rational(2 * minimum_polynomial_degree + sum(mask[:6]), 2)
        radial_rows.append(
            {
                "mask": list(mask),
                "minimum_sector_polynomial_degree": minimum_polynomial_degree,
                "physical_radial_order": str(radial_order),
            }
        )
    order_one_masks = {
        tuple(row["mask"])
        for row in radial_rows
        if sp.Rational(row["physical_radial_order"]) == 1
    }
    expected_order_one = {trivial, eg_mask, dh_mask}

    discriminant = sp.factor(4 * beta**2 - z * expected_eg**2)
    expected_discriminant = 16 * (9 - z) ** 3 * (1 - z)
    discriminant_identity = sp.expand(discriminant - expected_discriminant) == 0
    smaller_block_eigenvalue = sp.factor(2 * (9 - z) * (3 - z - 2 * sp.sqrt(z)))
    endpoint_factorization = sp.factor(3 - z - 2 * sp.sqrt(z))
    endpoint_null_cone = _endpoint_null_cone_certificate(rows, masks, characters)
    weighted_endpoint = _weighted_endpoint_blowup_certificate(
        rows, masks, characters, endpoint_null_cone
    )
    passed = bool(
        chart["passed"]
        and linear_identity
        and coupling_identity
        and order_one_masks == expected_order_one
        and discriminant_identity
        and endpoint_null_cone["passed"]
        and weighted_endpoint["passed"]
    )
    return {
        "experiment": "unrestricted orthonormal-line tangent certificate",
        "equality_line": "a=d=e=g=h=i=0; 0<=z=c^2<=1",
        "trivial_sector_linear_coefficients": [str(value) for value in linear],
        "expected_diagonal_coefficients": [str(value) for value in expected_linear],
        "linear_identity_verified": linear_identity,
        "order_one_sector_rows": [
            row for row in radial_rows if tuple(row["mask"]) in expected_order_one
        ],
        "all_sector_radial_orders": radial_rows,
        "exact_order_one_masks": [list(mask) for mask in sorted(order_one_masks)],
        "eg_coupling_polynomial": str(eg_constant),
        "dh_coupling_polynomial": str(dh_constant),
        "coupling_identity_verified": coupling_identity,
        "two_by_two_block_discriminant": str(discriminant),
        "expected_discriminant": str(expected_discriminant),
        "discriminant_identity_verified": discriminant_identity,
        "smaller_block_eigenvalue": str(smaller_block_eigenvalue),
        "endpoint_factorization": str(endpoint_factorization),
        "endpoint_null_cone_quartic_certificate": endpoint_null_cone,
        "weighted_endpoint_blowup_certificate": weighted_endpoint,
        "positivity_argument": (
            "For 0<=z<=1, alpha and beta are positive. Each coupled block has "
            "equal diagonal beta and off-diagonal magnitude 4(9-z)sqrt(z), "
            "so its smaller eigenvalue is 2(9-z)(3-z-2sqrt(z)). The final "
            "factor equals 2(9-z)(1-sqrt(z))(3+sqrt(z)) and is nonnegative, "
            "vanishing only at z=1. All other Fourier amplitudes are higher "
            "than first radial order."
        ),
        "scope_boundary": (
            "This proves the complete tangent cone along the orthonormal equality "
            "line. It is not a global positivity certificate away from that line."
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/spin8_dirac_unrestricted_tangent_20260807.json"),
    )
    arguments = parser.parse_args()
    report = certificate(arguments.coefficient_dir)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("unrestricted tangent certificate failed")


if __name__ == "__main__":
    main()
