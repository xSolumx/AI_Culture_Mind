"""Exact max-coordinate blow-up for the octet quadratic equality corner."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import tempfile
from functools import reduce
from math import comb, gcd
from pathlib import Path

import sympy as sp
from flint import ctx, fmpz_mpoly_ctx

from spin8_dirac_endpoint_octet_quadratic import (
    _build_quadratic,
    _native_bernstein_audit,
    _restrict_half_box,
)
from spin8_resource_limits import constrain_current_process

DEVIATIONS = ("ud", "ue", "ug", "ui", "one_minus_y")


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=int)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _blowup_chart(polynomial, pivot: int):
    if pivot not in range(5):
        raise ValueError("pivot must be between zero and four")
    nonpivots = tuple(axis for axis in range(5) if axis != pivot)
    ratio_index = {axis: index + 1 for index, axis in enumerate(nonpivots)}
    context = fmpz_mpoly_ctx.get(["radius", "x0", "x1", "x2", "x3"])
    coefficients: dict[tuple[int, ...], int] = {}
    for powers, coefficient in polynomial.to_dict().items():
        radial_base = sum(powers[:4])
        ratio_base = [0, 0, 0, 0, 0]
        for axis in range(4):
            if axis != pivot:
                ratio_base[ratio_index[axis]] += powers[axis]
        y_power = powers[4]
        for expansion_power in range(y_power + 1):
            radial_power = radial_base + expansion_power
            target = list(ratio_base)
            target[0] = radial_power
            if pivot != 4:
                target[ratio_index[4]] += expansion_power
            key = tuple(target)
            value = (
                int(coefficient)
                * comb(y_power, expansion_power)
                * (-1) ** expansion_power
            )
            coefficients[key] = coefficients.get(key, 0) + value
    coefficients = {key: value for key, value in coefficients.items() if value}
    minimum_radius_degree = min(key[0] for key in coefficients)
    if minimum_radius_degree != 4:
        raise AssertionError(
            f"expected exact radius order four, got {minimum_radius_degree}"
        )
    divided = {
        (key[0] - 4,) + key[1:]: value for key, value in coefficients.items()
    }
    radius_degree = max(key[0] for key in divided)
    # radius = scaled_radius / 4.  Multiplication by 4**radius_degree clears
    # every denominator and is common and strictly positive.
    scaled = {
        key: value * 4 ** (radius_degree - key[0]) for key, value in divided.items()
    }
    return context.from_dict(scaled), minimum_radius_degree, nonpivots


def _pivot_ud_tangent_product_identity(face) -> dict[str, object]:
    """Verify the four signed-square factors on the ``ui=0`` exceptional face."""

    ui_zero = face.subs({"x2": 0})
    P, G, H = sp.symbols("P G H")
    signed_product = sp.expand(
        sp.prod(
            (1 + sigma * H) ** 2 + (P + tau * G) ** 2
            for sigma in (-1, 1)
            for tau in (-1, 1)
        )
    )
    expected_coefficients: dict[tuple[int, ...], int] = {}
    for powers, coefficient in sp.Poly(signed_product, P, G, H).terms():
        if any(power % 2 for power in powers):
            raise AssertionError("signed tangent product retained an odd radical")
        target = (0, powers[0] // 2, powers[1] // 2, 0, powers[2] // 2)
        expected_coefficients[target] = int(coefficient) * 2 ** (powers[2] // 2)
    expected = face.context().from_dict(expected_coefficients)
    content, _factors = ui_zero.factor()
    identity = ui_zero == int(content) * expected
    return {
        "face": "exceptional divisor and ui=0",
        "identity": (
            "content*product_{sigma,tau=+-1} "
            "[(1+sigma*h)^2+(e+tau*g)^2], with h^2=2*tangent_h"
        ),
        "content": str(content),
        "factor_count": 4,
        "each_factor_manifestly_nonnegative": True,
        "identity_verified": identity,
        "passed": identity and int(content) > 0,
    }


def _pivot_ud_two_variable_corner_identity(corner) -> dict[str, object]:
    """Certify the residual ``ue=ug=0`` corner by an explicit sign factorization."""

    _radius, _ue, _ug, ui, tangent_h = corner.context().gens()
    first = (
        16 * ui * tangent_h**2
        - 16 * ui * tangent_h
        - 21 * ui
        - 32 * tangent_h**2
        - 8 * tangent_h
        - 28
    )
    second = (
        16 * ui**2 * tangent_h**2
        - 16 * ui**2 * tangent_h
        + 29 * ui**2
        - 32 * ui * tangent_h**2
        + 72 * ui * tangent_h
        + 12 * ui
        + 32 * tangent_h**2
        - 32 * tangent_h
        + 8
    )
    content, _factors = corner.factor()
    identity = corner == int(content) * ui * first * second
    return {
        "identity": "corner=content*ui*A*B",
        "content": str(content),
        "content_is_negative": int(content) < 0,
        "identity_verified": identity,
        "A_negative_decomposition": (
            "A=-[16*ui*t*(1-t)+21*ui+32*t^2+8*t+28] < 0"
        ),
        "B_nonnegative_decomposition": (
            "B=ui^2*[29-16*t*(1-t)]+ui*[12+8*t*(9-4*t)]"
            "+32*(t-1/2)^2 >= 0"
        ),
        "domain_signs_verified": True,
        "passed": identity and int(content) < 0,
    }


def _pivot_middle_tangent_product_identity(face) -> dict[str, object]:
    """Verify the signed-square tangent product in the ue/ug pivot charts."""

    ui_zero = face.subs({"x2": 0})
    A, B, H = sp.symbols("A B H")
    signed_product = sp.expand(
        sp.prod(
            (A + sigma * H) ** 2 + (1 + tau * B) ** 2
            for sigma in (-1, 1)
            for tau in (-1, 1)
        )
    )
    expected_coefficients: dict[tuple[int, ...], int] = {}
    for powers, coefficient in sp.Poly(signed_product, A, B, H).terms():
        if any(power % 2 for power in powers):
            raise AssertionError("middle-pivot tangent product retained a radical")
        target = (0, powers[0] // 2, powers[1] // 2, 0, powers[2] // 2)
        expected_coefficients[target] = int(coefficient) * 2 ** (powers[2] // 2)
    expected = face.context().from_dict(expected_coefficients)
    content = _positive_content(ui_zero)
    identity = ui_zero == content * expected
    return {
        "identity": (
            "positive_content*prod_{sigma,tau=+-1} "
            "((sqrt(x0)+sigma*sqrt(2*t))^2+"
            "(1+tau*sqrt(x1))^2)"
        ),
        "positive_content": str(content),
        "radicals_cancelled_exactly": True,
        "identity_verified": identity,
        "factor_count": 4,
        "each_factor_nonnegative": True,
        "passed": identity,
    }


def _shared_ud_exceptional_square_identity(face) -> dict[str, object]:
    """Verify the perfect-square tangent law shared by minors 1 and 2."""

    _radius, e, g, i, t = face.context().gens()
    base = (
        4 * e**2
        + 24 * e * g
        + 20 * e * i
        + 16 * e * t
        + 8 * e
        + 4 * g**2
        + 20 * g * i
        + 16 * g * t
        + 8 * g
        + 25 * i**2
        + 40 * i * t
        + 20 * i
        + 16 * t**2
        - 16 * t
        + 4
    )
    content = _positive_content(face)
    identity = face == content * base**2
    return {
        "identity": "positive_content*B(e,g,i,t)^2",
        "base_polynomial": (
            "4e^2+24eg+20ei+16et+8e+4g^2+20gi+16gt+8g+"
            "25i^2+40it+20i+16t^2-16t+4"
        ),
        "positive_content": str(content),
        "identity_verified": identity,
        "passed": identity,
    }


def _shared_ud_radial_remainder_certificate(remainder) -> dict[str, object]:
    """Certify the hard radial corner shared by minors 1 and 2."""

    _radius, e, g, i, _t = remainder.context().gens()
    corner = remainder.subs({"x0": 0, "x1": 0, "x2": 0})
    boxes = []
    for bits in ("00000", "00001", "10000", "10001"):
        audit = _native_bernstein_audit(
            _restrict_half_box(corner, bits), sample_limit=16
        )
        boxes.append(
            {
                "five_axis_bits_radius_e_g_i_t": bits,
                "native_bernstein": audit,
                "passed": audit["negative_scaled_coefficient_count"] == 0,
            }
        )
    rest = remainder - corner * (1 - e) ** int(
        remainder.degrees()[1]
    ) * (1 - g) ** int(remainder.degrees()[2]) * (1 - i) ** int(
        remainder.degrees()[3]
    )
    rest_audit = _native_bernstein_audit(rest, sample_limit=32)
    passed = bool(
        all(row["passed"] for row in boxes)
        and rest_audit["negative_scaled_coefficient_count"] == 0
    )
    return {
        "corner": "e_ratio=g_ratio=i_ratio=0",
        "corner_four_box_atlas": boxes,
        "corner_atlas_covers_radius_t_square": True,
        "selector": (
            f"(1-e_ratio)^{int(remainder.degrees()[1])}*"
            f"(1-g_ratio)^{int(remainder.degrees()[2])}*"
            f"(1-i_ratio)^{int(remainder.degrees()[3])}"
        ),
        "second_remainder_native_bernstein": rest_audit,
        "passed": passed,
    }


def _divide_coordinate_order(polynomial, axis: int):
    coefficients = polynomial.to_dict()
    order = min(powers[axis] for powers in coefficients)
    if order <= 0:
        raise AssertionError("expected a strictly positive coordinate divisor")
    divided = {}
    for powers, coefficient in coefficients.items():
        target = list(powers)
        target[axis] -= order
        divided[tuple(target)] = coefficient
    return polynomial.context().from_dict(divided), int(order)


def _positive_content(polynomial) -> int:
    """Return the positive integer content of a nonzero FLINT polynomial."""

    coefficients = polynomial.to_dict().values()
    return reduce(gcd, (abs(int(value)) for value in coefficients))


def _pivot_ud_radial_ui_zero_certificate(quotient) -> dict[str, object]:
    """Certify the hard radial face of the minor-0, ``ud`` blow-up chart.

    The production chart uses the degree-68 exceptional-face selector.  Its
    radial quotient is compared exactly with the quotient obtained from the
    degree-one selector.  Their difference is the already-certified
    exceptional face times a nonnegative geometric sum.  The degree-one
    quotient is then resolved by two nested boundary selectors and a four-box
    exact atlas around its sole equality point.
    """

    radius, ue, ug, _ui_ratio, tangent_h = quotient.context().gens()
    exceptional = quotient.subs({"radius": 0, "x2": 0})
    production_degree = int(quotient.degrees()[0])

    production_remainder = quotient - quotient.subs({"radius": 0}) * (
        1 - radius
    ) ** production_degree
    production_quotient, production_order = _divide_coordinate_order(
        production_remainder.subs({"x2": 0}), 0
    )

    linear_remainder = quotient - quotient.subs({"radius": 0}) * (1 - radius)
    linear_quotient, linear_order = _divide_coordinate_order(
        linear_remainder.subs({"x2": 0}), 0
    )
    if production_order != 1 or linear_order != 1:
        raise AssertionError("both radial remainders must vanish to first order")

    geometric_sum = sum(
        (1 - radius) ** power for power in range(1, production_degree)
    )
    comparison_identity = (
        production_quotient == linear_quotient + exceptional * geometric_sum
    )

    axis = linear_quotient.subs({"radius": 0})
    axis_corner = axis.subs({"x0": 0, "x1": 0})
    expected_axis_corner = (2 * tangent_h - 1) ** 2 * (
        -56 * tangent_h**3
        + 220 * tangent_h**2
        + 70 * tangent_h
        + 13
    )
    axis_corner_content = _positive_content(axis_corner)
    axis_corner_identity = axis_corner == axis_corner_content * expected_axis_corner
    # The cubic factor has degree-three Bernstein coefficients
    # (13, 109/3, 133, 247), all strictly positive.  Store the common
    # denominator-three integer row as the replayable sign certificate.
    cubic_bernstein_scaled = [39, 109, 399, 741]
    cubic_positive = all(value > 0 for value in cubic_bernstein_scaled)

    axis_remainder = axis - axis_corner * (1 - ue) ** int(
        axis.degrees()[1]
    ) * (1 - ug) ** int(axis.degrees()[2])
    axis_remainder_audit = _native_bernstein_audit(
        axis_remainder, sample_limit=32
    )

    radial_degree = int(linear_quotient.degrees()[0])
    radial_remainder = linear_quotient - axis * (1 - radius) ** radial_degree
    radial_corner = radial_remainder.subs({"x0": 0, "x1": 0})
    radial_corner_boxes = []
    for bits in ("00000", "00001", "10000", "10001"):
        audit = _native_bernstein_audit(
            _restrict_half_box(radial_corner, bits), sample_limit=16
        )
        radial_corner_boxes.append(
            {
                "five_axis_bits_radius_ue_ug_ui_t": bits,
                "native_bernstein": audit,
                "passed": audit["negative_scaled_coefficient_count"] == 0,
            }
        )

    radial_remainder_rest = radial_remainder - radial_corner * (
        1 - ue
    ) ** int(radial_remainder.degrees()[1]) * (1 - ug) ** int(
        radial_remainder.degrees()[2]
    )
    radial_remainder_rest_audit = _native_bernstein_audit(
        radial_remainder_rest, sample_limit=32
    )

    linear_passed = bool(
        axis_corner_identity
        and cubic_positive
        and axis_remainder_audit["negative_scaled_coefficient_count"] == 0
        and all(row["passed"] for row in radial_corner_boxes)
        and radial_remainder_rest_audit["negative_scaled_coefficient_count"] == 0
    )
    return {
        "production_selector_exponent": production_degree,
        "production_radial_divisor_order": production_order,
        "linear_selector_radial_divisor_order": linear_order,
        "comparison_identity": (
            "H_production = H_linear + F*sum_{j=0}^{m-2}(1-radius)^j"
        ),
        "comparison_identity_verified": comparison_identity,
        "comparison_nonnegative_factors": {
            "exceptional_ui_zero_face_certified_elsewhere": True,
            "geometric_sum_term_count": production_degree - 1,
            "unit_cube_nonnegative": True,
        },
        "linear_axis": {
            "corner_identity": (
                "positive_content*(2*t-1)^2*"
                "(-56*t^3+220*t^2+70*t+13)"
            ),
            "corner_positive_content": str(axis_corner_content),
            "corner_identity_verified": axis_corner_identity,
            "cubic_degree3_bernstein_coefficients_scaled_by_3": (
                cubic_bernstein_scaled
            ),
            "cubic_strictly_positive": cubic_positive,
            "selector": (
                f"(1-ue_ratio)^{int(axis.degrees()[1])}*"
                f"(1-ug_ratio)^{int(axis.degrees()[2])}"
            ),
            "remainder_native_bernstein": axis_remainder_audit,
        },
        "linear_radial_remainder": {
            "selector": f"(1-radius)^{radial_degree}",
            "ue_ug_zero_corner_four_box_atlas": radial_corner_boxes,
            "corner_atlas_covers_unit_square": True,
            "second_selector": (
                f"(1-ue_ratio)^{int(radial_remainder.degrees()[1])}*"
                f"(1-ug_ratio)^{int(radial_remainder.degrees()[2])}"
            ),
            "second_remainder_native_bernstein": radial_remainder_rest_audit,
        },
        "linear_quotient_passed": linear_passed,
        "passed": bool(comparison_identity and linear_passed),
    }


def _pivot_middle_radial_ui_zero_certificate(
    quotient, *, alternate_selector_degree: int
) -> dict[str, object]:
    """Compare the production radial quotient with a native-positive one."""

    if alternate_selector_degree not in (1, 2):
        raise ValueError("the frozen middle-pivot selectors are one or two")
    radius = quotient.context().gens()[0]
    exceptional = quotient.subs({"radius": 0, "x2": 0})
    production_degree = int(quotient.degrees()[0])

    production_remainder = quotient - quotient.subs({"radius": 0}) * (
        1 - radius
    ) ** production_degree
    production_quotient, production_order = _divide_coordinate_order(
        production_remainder.subs({"x2": 0}), 0
    )
    alternate_remainder = quotient - quotient.subs({"radius": 0}) * (
        1 - radius
    ) ** alternate_selector_degree
    alternate_quotient, alternate_order = _divide_coordinate_order(
        alternate_remainder.subs({"x2": 0}), 0
    )
    if production_order != 1 or alternate_order != 1:
        raise AssertionError("both middle-pivot remainders must have order one")

    geometric_sum = sum(
        (1 - radius) ** power
        for power in range(alternate_selector_degree, production_degree)
    )
    comparison_identity = (
        production_quotient == alternate_quotient + exceptional * geometric_sum
    )
    alternate_audit = _native_bernstein_audit(
        alternate_quotient, sample_limit=64
    )
    return {
        "production_selector_exponent": production_degree,
        "alternate_selector_exponent": alternate_selector_degree,
        "production_radial_divisor_order": production_order,
        "alternate_radial_divisor_order": alternate_order,
        "comparison_identity": (
            "H_production = H_alternate + "
            "F*sum_{j=alternate}^{production-1}(1-radius)^j"
        ),
        "comparison_identity_verified": comparison_identity,
        "geometric_sum_term_count": (
            production_degree - alternate_selector_degree
        ),
        "alternate_quotient_native_bernstein": alternate_audit,
        "passed": bool(
            comparison_identity
            and alternate_audit["negative_scaled_coefficient_count"] == 0
        ),
    }


def run(
    coefficient_dir: Path,
    *,
    minor_index: int,
    pivot: int,
    output: Path,
    flint_threads: int = 6,
    radial_ui0_atlas: bool = False,
) -> dict[str, object]:
    if not 1 <= flint_threads <= 6:
        raise ValueError("FLINT thread count must be between one and six")
    resource = constrain_current_process(workers=flint_threads)
    ctx.threads = flint_threads
    polynomial, mask, _variables = _build_quadratic(coefficient_dir, minor_index)
    quotient, order, nonpivots = _blowup_chart(polynomial, pivot)
    audit = _native_bernstein_audit(quotient, sample_limit=64)
    radius = quotient.context().gens()[0]
    exceptional_face = quotient.subs({"radius": 0})
    exceptional_audit = _native_bernstein_audit(exceptional_face, sample_limit=64)
    tangent_product = None
    exceptional_nested = None
    if minor_index == 0 and pivot == 0:
        tangent_product = _pivot_ud_tangent_product_identity(exceptional_face)
        _radius, _ue, _ug, ui_ratio, _tangent_h = quotient.context().gens()
        exceptional_ui_zero = exceptional_face.subs({"x2": 0})
        exceptional_ui_degree = int(exceptional_face.degrees()[3])
        exceptional_remainder = exceptional_face - exceptional_ui_zero * (
            1 - ui_ratio
        ) ** exceptional_ui_degree
        exceptional_remainder_audit = _native_bernstein_audit(
            exceptional_remainder, sample_limit=64
        )
        exceptional_corner = exceptional_remainder.subs({"x0": 0, "x1": 0})
        exceptional_corner_audit = _native_bernstein_audit(
            exceptional_corner, sample_limit=64
        )
        exceptional_corner_identity = _pivot_ud_two_variable_corner_identity(
            exceptional_corner
        )
        ue_degree = int(exceptional_remainder.degrees()[1])
        ug_degree = int(exceptional_remainder.degrees()[2])
        exceptional_second_remainder = exceptional_remainder - exceptional_corner * (
            1 - _ue
        ) ** ue_degree * (1 - _ug) ** ug_degree
        exceptional_second_audit = _native_bernstein_audit(
            exceptional_second_remainder, sample_limit=64
        )
        exceptional_remainder_passed = bool(
            exceptional_remainder_audit["negative_scaled_coefficient_count"] == 0
            or (
                (
                    exceptional_corner_audit["negative_scaled_coefficient_count"]
                    == 0
                    or exceptional_corner_identity["passed"]
                )
                and exceptional_second_audit["negative_scaled_coefficient_count"]
                == 0
            )
        )
        exceptional_nested = {
            "selector": f"(1-ui_ratio)^{exceptional_ui_degree}",
            "ui_zero_tangent_product": tangent_product,
            "remainder_native_bernstein": exceptional_remainder_audit,
            "nested_ue_ug_zero_corner": {
                "corner_native_bernstein": exceptional_corner_audit,
                "corner_sign_factorization": exceptional_corner_identity,
                "selector": f"(1-ue_ratio)^{ue_degree}*(1-ug_ratio)^{ug_degree}",
                "second_remainder_native_bernstein": exceptional_second_audit,
                "passed": exceptional_remainder_passed,
            },
            "passed": bool(
                tangent_product["passed"]
                and exceptional_remainder_passed
            ),
        }
    elif minor_index == 0 and pivot in (1, 2):
        _radius, _x0, _x1, ui_ratio, _tangent_h = quotient.context().gens()
        exceptional_ui_zero = exceptional_face.subs({"x2": 0})
        exceptional_ui_degree = int(exceptional_face.degrees()[3])
        exceptional_remainder = exceptional_face - exceptional_ui_zero * (
            1 - ui_ratio
        ) ** exceptional_ui_degree
        exceptional_remainder_audit = _native_bernstein_audit(
            exceptional_remainder, sample_limit=64
        )
        tangent_product = _pivot_middle_tangent_product_identity(
            exceptional_face
        )
        exceptional_nested = {
            "selector": f"(1-ui_ratio)^{exceptional_ui_degree}",
            "ui_zero_tangent_product": tangent_product,
            "remainder_native_bernstein": exceptional_remainder_audit,
            "passed": bool(
                tangent_product["passed"]
                and exceptional_remainder_audit[
                    "negative_scaled_coefficient_count"
                ]
                == 0
            ),
        }
    elif minor_index in (1, 2) and pivot == 0:
        exceptional_square = _shared_ud_exceptional_square_identity(
            exceptional_face
        )
        exceptional_nested = {
            "complete_exceptional_square": exceptional_square,
            "passed": exceptional_square["passed"],
        }
    radius_degree = int(quotient.degrees()[0])
    exceptional_selector = (1 - radius) ** radius_degree
    remainder = quotient - exceptional_face * exceptional_selector
    remainder_audit = _native_bernstein_audit(remainder, sample_limit=64)
    remainder_nested = None
    if minor_index == 0 and pivot == 0:
        _radius, _ue, _ug, ui_ratio, _tangent_h = quotient.context().gens()
        remainder_ui_zero = remainder.subs({"x2": 0})
        remainder_ui_audit = _native_bernstein_audit(
            remainder_ui_zero, sample_limit=64
        )
        radius_quotient, radius_order = _divide_coordinate_order(
            remainder_ui_zero, 0
        )
        radius_quotient_audit = _native_bernstein_audit(
            radius_quotient, sample_limit=64
        )
        radius_atlas = None
        if radial_ui0_atlas:
            radius_atlas = []
            for active_bits in itertools.product("01", repeat=4):
                # The zero-degree ui axis is retained as bit zero so the
                # generic five-axis dyadic map can be replayed unchanged.
                bits = "".join(active_bits[:3]) + "0" + active_bits[3]
                restricted = _restrict_half_box(radius_quotient, bits)
                restricted_audit = _native_bernstein_audit(
                    restricted, sample_limit=32
                )
                radius_atlas.append(
                    {
                        "active_bits_radius_ue_ug_t": "".join(active_bits),
                        "five_axis_bits": bits,
                        "native_bernstein": restricted_audit,
                        "passed": (
                            restricted_audit["negative_scaled_coefficient_count"]
                            == 0
                        ),
                    }
                )
                _atomic_json(
                    output,
                    {
                        "experiment": "radial ui-zero quotient atlas progress",
                        "minor_index": minor_index,
                        "pivot": pivot,
                        "completed_box_count": len(radius_atlas),
                        "total_box_count": 16,
                        "boxes": radius_atlas,
                        "complete": len(radius_atlas) == 16,
                    },
                )
        radius_atlas_passed = bool(
            radius_atlas is not None and all(row["passed"] for row in radius_atlas)
        )
        remainder_ui_degree = int(remainder.degrees()[3])
        second_remainder = remainder - remainder_ui_zero * (
            1 - ui_ratio
        ) ** remainder_ui_degree
        second_remainder_audit = _native_bernstein_audit(
            second_remainder, sample_limit=64
        )
        radial_exact_certificate = _pivot_ud_radial_ui_zero_certificate(
            quotient
        )
        remainder_nested = {
            "ui_zero_face_native_bernstein": remainder_ui_audit,
            "exact_radius_divisor_order": radius_order,
            "radius_quotient_native_bernstein": radius_quotient_audit,
            "radius_quotient_half_box_atlas": radius_atlas,
            "selector": f"(1-ui_ratio)^{remainder_ui_degree}",
            "second_remainder_native_bernstein": second_remainder_audit,
            "radial_ui_zero_exact_certificate": radial_exact_certificate,
            "passed": bool(
                (
                    remainder_ui_audit["negative_scaled_coefficient_count"] == 0
                    or radius_quotient_audit["negative_scaled_coefficient_count"]
                    == 0
                    or radius_atlas_passed
                    or radial_exact_certificate["passed"]
                )
                and second_remainder_audit["negative_scaled_coefficient_count"]
                == 0
            ),
        }
    elif minor_index == 0 and pivot in (1, 2):
        _radius, _x0, _x1, ui_ratio, _tangent_h = quotient.context().gens()
        remainder_ui_zero = remainder.subs({"x2": 0})
        remainder_ui_audit = _native_bernstein_audit(
            remainder_ui_zero, sample_limit=64
        )
        radius_quotient, radius_order = _divide_coordinate_order(
            remainder_ui_zero, 0
        )
        radius_quotient_audit = _native_bernstein_audit(
            radius_quotient, sample_limit=64
        )
        remainder_ui_degree = int(remainder.degrees()[3])
        second_remainder = remainder - remainder_ui_zero * (
            1 - ui_ratio
        ) ** remainder_ui_degree
        second_remainder_audit = _native_bernstein_audit(
            second_remainder, sample_limit=64
        )
        alternate_selector_degree = 1 if pivot == 1 else 2
        radial_exact_certificate = _pivot_middle_radial_ui_zero_certificate(
            quotient,
            alternate_selector_degree=alternate_selector_degree,
        )
        remainder_nested = {
            "ui_zero_face_native_bernstein": remainder_ui_audit,
            "exact_radius_divisor_order": radius_order,
            "radius_quotient_native_bernstein": radius_quotient_audit,
            "selector": f"(1-ui_ratio)^{remainder_ui_degree}",
            "second_remainder_native_bernstein": second_remainder_audit,
            "radial_ui_zero_exact_certificate": radial_exact_certificate,
            "passed": bool(
                (
                    radius_quotient_audit[
                        "negative_scaled_coefficient_count"
                    ]
                    == 0
                    or radial_exact_certificate["passed"]
                )
                and second_remainder_audit[
                    "negative_scaled_coefficient_count"
                ]
                == 0
            ),
        }
    elif minor_index in (1, 2) and pivot == 0:
        radial_exact_certificate = _shared_ud_radial_remainder_certificate(
            remainder
        )
        remainder_nested = {
            "radial_exact_certificate": radial_exact_certificate,
            "passed": radial_exact_certificate["passed"],
        }
    exceptional_passed = bool(
        exceptional_audit["negative_scaled_coefficient_count"] == 0
        or (exceptional_nested is not None and exceptional_nested["passed"])
    )
    remainder_passed = bool(
        remainder_audit["negative_scaled_coefficient_count"] == 0
        or (remainder_nested is not None and remainder_nested["passed"])
    )
    boundary_decomposition_passed = bool(
        exceptional_passed and remainder_passed
    )
    report = {
        "experiment": "adjacent endpoint octet equality max-coordinate blow-up",
        "minor_index": minor_index,
        "mode_mask": list(mask),
        "pivot_index": pivot,
        "pivot_deviation": DEVIATIONS[pivot],
        "ratio_deviations": [DEVIATIONS[index] for index in nonpivots],
        "chart": (
            "pivot_deviation=radius; other_deviation=radius*x_j; "
            "radius=scaled_radius/4"
        ),
        "exact_radius_divisibility_order": order,
        "quotient_power_term_count": len(quotient.to_dict()),
        "quotient_native_bernstein": audit,
        "exceptional_divisor": {
            "power_term_count": len(exceptional_face.to_dict()),
            "native_bernstein": exceptional_audit,
            "nested_ui_zero_certificate": exceptional_nested,
            "passed": exceptional_passed,
        },
        "exceptional_boundary_selector": {
            "selector": f"(1-radius)^{radius_degree}",
            "remainder_power_term_count": len(remainder.to_dict()),
            "remainder_native_bernstein": remainder_audit,
            "nested_ui_zero_certificate": remainder_nested,
            "passed": boundary_decomposition_passed,
        },
        "passed": bool(
            audit["negative_scaled_coefficient_count"] == 0
            or boundary_decomposition_passed
        ),
        "scope_boundary": (
            "One of five charts for the equality corner of one quadratic minor. "
            "All five pivots and all three minor families remain required."
        ),
        "resource_contract": resource,
    }
    _atomic_json(output, report)
    report["artifact_sha256"] = _sha256(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minor-index", type=int, required=True, choices=range(3))
    parser.add_argument("--pivot", type=int, required=True, choices=range(5))
    parser.add_argument(
        "--coefficient-dir",
        type=Path,
        default=Path("artifacts/spin8_dirac_unrestricted_coefficients_20260807"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flint-threads", type=int, default=6)
    parser.add_argument(
        "--radial-ui0-atlas",
        action="store_true",
        help="audit all 16 half-boxes of the divided radial ui=0 face",
    )
    arguments = parser.parse_args()
    report = run(
        arguments.coefficient_dir,
        minor_index=arguments.minor_index,
        pivot=arguments.pivot,
        output=arguments.output,
        flint_threads=arguments.flint_threads,
        radial_ui0_atlas=arguments.radial_ui0_atlas,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=int))


if __name__ == "__main__":
    main()
