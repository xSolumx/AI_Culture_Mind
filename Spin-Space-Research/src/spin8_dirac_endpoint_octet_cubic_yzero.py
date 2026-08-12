"""Nested exact boundary-selector audit for the cubic ``y=0`` face."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import sympy as sp
from flint import ctx

from spin8_dirac_endpoint_octet_cubic import DEFAULT_COEFFICIENT_DIR, _build_cubic
from spin8_dirac_endpoint_octet_quadratic import (
    _atomic_json,
    _native_bernstein_audit,
    _restrict_half_box,
    _sha256,
)
from spin8_resource_limits import constrain_current_process


def _all_negative_rows_on_axis(audit: dict[str, object], axis: int, value: int) -> bool:
    rows = audit["negative_rows_sample"]
    return bool(
        len(rows) == audit["negative_scaled_coefficient_count"]
        and all(row["bernstein_index"][axis] == value for row in rows)
    )


def _exact_sturm_certificate(polynomial, *, axis: int) -> dict[str, object]:
    """Prove strict positivity of a univariate FLINT polynomial on [0, 1]."""

    active = [index for index, degree in enumerate(polynomial.degrees()) if degree]
    if active != [axis]:
        raise AssertionError("Sturm certificate requires exactly one active axis")
    u = sp.Symbol("u")
    expression = sum(
        sp.Integer(int(coefficient)) * u ** int(powers[axis])
        for powers, coefficient in polynomial.to_dict().items()
    )
    return _sympy_sturm_certificate(sp.Poly(expression, u, domain=sp.ZZ))


def _sympy_sturm_certificate(exact: sp.Poly) -> dict[str, object]:
    """Return an exact, replayable strict-positivity certificate on [0, 1]."""

    if len(exact.gens) != 1:
        raise AssertionError("Sturm certificate requires a univariate polynomial")
    u = exact.gens[0]
    chain = [sp.Poly(row, u, domain=sp.QQ) for row in sp.sturm(exact.as_expr(), u)]

    def variation_at(value: int) -> tuple[int, list[int]]:
        signs = []
        for row in chain:
            evaluated = sp.sign(row.eval(value))
            if evaluated:
                signs.append(int(evaluated))
        changes = sum(left != right for left, right in itertools.pairwise(signs))
        return changes, signs

    variations_zero, signs_zero = variation_at(0)
    variations_one, signs_one = variation_at(1)
    sturm_roots = variations_zero - variations_one
    independent_roots = int(exact.count_roots(0, 1))
    value_zero = int(exact.eval(0))
    value_one = int(exact.eval(1))
    passed = bool(
        sturm_roots == 0
        and independent_roots == 0
        and value_zero > 0
        and value_one > 0
    )
    return {
        "polynomial_coefficients_descending": [
            str(value) for value in exact.all_coeffs()
        ],
        "sturm_chain_coefficients_descending": [
            [str(value) for value in row.all_coeffs()] for row in chain
        ],
        "signs_at_zero": signs_zero,
        "signs_at_one": signs_one,
        "variations_at_zero": variations_zero,
        "variations_at_one": variations_one,
        "sturm_root_count_open_interval": sturm_roots,
        "independent_exact_root_count": independent_roots,
        "value_at_zero": str(value_zero),
        "value_at_one": str(value_one),
        "continuity_and_no_roots_imply_strict_positivity": True,
        "passed": passed,
    }


def _quadratic_discriminant_certificate(q_factor) -> dict[str, object]:
    """Certify the remaining bivariate factor as a positive quadratic."""

    u, v = sp.symbols("u v")
    expression = sum(
        sp.Integer(int(coefficient))
        * u ** int(powers[1])
        * v ** int(powers[3])
        for powers, coefficient in q_factor.to_dict().items()
    )
    quadratic = sp.Poly(expression, v)
    if quadratic.degree() != 2:
        raise AssertionError("the final factor is not quadratic in ui")
    leading, linear, constant = quadratic.all_coeffs()
    discriminant = sp.expand(linear**2 - 4 * leading * constant)
    sextic = (
        17800 * u**6
        - 123800 * u**5
        + 337225 * u**4
        - 444820 * u**3
        + 270590 * u**2
        - 43420 * u
        + 2809
    )
    cubic = 30 * u**3 - 135 * u**2 + 180 * u + 53
    discriminant_identity = sp.expand(
        discriminant + 3 * cubic**4 * sextic**2
    ) == 0
    leading_certificate = _sympy_sturm_certificate(
        sp.Poly(leading, u, domain=sp.ZZ)
    )
    sextic_certificate = _sympy_sturm_certificate(
        sp.Poly(sextic, u, domain=sp.ZZ)
    )
    cubic_certificate = _sympy_sturm_certificate(
        sp.Poly(cubic, u, domain=sp.ZZ)
    )
    passed = bool(
        discriminant_identity
        and leading_certificate["passed"]
        and sextic_certificate["passed"]
        and cubic_certificate["passed"]
    )
    return {
        "quadratic_variable": "ui",
        "parameter_interval": "ue in [0,1]",
        "leading_coefficients_descending": [
            str(value) for value in sp.Poly(leading, u).all_coeffs()
        ],
        "linear_coefficients_descending": [
            str(value) for value in sp.Poly(linear, u).all_coeffs()
        ],
        "constant_coefficients_descending": [
            str(value) for value in sp.Poly(constant, u).all_coeffs()
        ],
        "discriminant_identity": "disc_ui(Q)=-3*S(ue)^4*P(ue)^2",
        "discriminant_identity_verified_exactly": discriminant_identity,
        "leading_coefficient_sturm": leading_certificate,
        "sextic_factor_sturm": sextic_certificate,
        "cubic_factor_sturm": cubic_certificate,
        "leading_positive_and_discriminant_strictly_negative": passed,
        "conclusion": "Q(ue,ui)>0 for ue in [0,1] and every real ui",
        "passed": passed,
    }


def _poly_from_descending(values: list[str], variable: sp.Symbol) -> sp.Poly:
    degree = len(values) - 1
    expression = sum(
        sp.Integer(value) * variable ** (degree - index)
        for index, value in enumerate(values)
    )
    return sp.Poly(expression, variable, domain=sp.ZZ)


def verify_report(report: dict[str, object]) -> dict[str, object]:
    """Replay the load-bearing algebra stored in a y-zero report."""

    failures: list[str] = []
    if not report.get("passed"):
        failures.append("report does not record a pass")
    stages = report.get("stages", [])
    if len(stages) != 2:
        failures.append("expected exactly two completed selector stages")
        return {"verified": False, "failures": failures}
    for index, stage in enumerate(stages):
        if not stage.get("identity_verified_exactly"):
            failures.append(f"stage {index} selector identity failed")
        if not stage.get("face_passed"):
            failures.append(f"stage {index} face did not pass")
    final_audit = stages[-1]["remainder_native_bernstein"]
    if final_audit["negative_scaled_coefficient_count"] != 0:
        failures.append("final selector remainder has negative controls")

    nested = stages[1]["nested_face_certificate"]
    factor = nested["corner_factor_certificate"]
    certificate = factor["q_quadratic_discriminant_certificate"]
    u, v = sp.symbols("u v")
    leading = _poly_from_descending(
        certificate["leading_coefficients_descending"], u
    )
    linear = _poly_from_descending(
        certificate["linear_coefficients_descending"], u
    )
    constant = _poly_from_descending(
        certificate["constant_coefficients_descending"], u
    )
    sextic = sp.Poly(
        17800 * u**6
        - 123800 * u**5
        + 337225 * u**4
        - 444820 * u**3
        + 270590 * u**2
        - 43420 * u
        + 2809,
        u,
        domain=sp.ZZ,
    )
    cubic = sp.Poly(30 * u**3 - 135 * u**2 + 180 * u + 53, u, domain=sp.ZZ)
    q = leading.as_expr() * v**2 + linear.as_expr() * v + constant.as_expr()
    discriminant = sp.discriminant(q, v)
    if sp.expand(discriminant + 3 * cubic.as_expr() ** 4 * sextic.as_expr() ** 2):
        failures.append("stored quadratic coefficients fail the discriminant identity")
    for name, polynomial in (
        ("leading", leading),
        ("sextic", sextic),
        ("cubic", cubic),
    ):
        if not _sympy_sturm_certificate(polynomial)["passed"]:
            failures.append(f"{name} strict-positivity replay failed")
    return {
        "verified": not failures,
        "failures": failures,
        "recomputed_discriminant_identity": not failures
        or "stored quadratic coefficients fail the discriminant identity" not in failures,
    }


def _selector_stage(
    polynomial,
    *,
    axis: int,
    endpoint: int,
    cube_factor_allowed: bool = False,
    nested_face_axis: int | None = None,
) -> dict[str, object]:
    variable = polynomial.context().gens()[axis]
    degree = int(polynomial.degrees()[axis])
    face = polynomial.subs({str(variable): endpoint})
    selector = variable**degree if endpoint == 1 else (1 - variable) ** degree
    remainder = polynomial - face * selector
    identity = polynomial == face * selector + remainder
    face_audit = _native_bernstein_audit(face, sample_limit=1024)
    nested_face_certificate = None
    nested_face_passed = False
    if (
        nested_face_axis is not None
        and face_audit["negative_scaled_coefficient_count"]
        and _all_negative_rows_on_axis(face_audit, nested_face_axis, 0)
    ):
        nested_variable = face.context().gens()[nested_face_axis]
        nested_face = face.subs({str(nested_variable): 0})
        nested_selector = (1 - nested_variable) ** int(
            face.degrees()[nested_face_axis]
        )
        nested_remainder = face - nested_face * nested_selector
        nested_identity = face == nested_face * nested_selector + nested_remainder
        nested_face_audit = _native_bernstein_audit(
            nested_face, sample_limit=256
        )
        nested_factor_certificate = None
        nested_factor_passed = False
        if nested_face_audit["negative_scaled_coefficient_count"]:
            content, factors = nested_face.factor()
            ui = nested_face.context().gens()[3]
            linear_rows = [
                factor
                for factor, exponent in factors
                if int(exponent) == 1 and factor == ui - 1
            ]
            square_rows = [
                factor for factor, exponent in factors if int(exponent) == 2
            ]
            q_rows = [
                factor
                for factor, exponent in factors
                if int(exponent) == 1 and factor != ui - 1
            ]
            shape_passed = bool(
                int(content) == -64
                and len(factors) == 3
                and len(linear_rows) == 1
                and len(square_rows) == 1
                and len(q_rows) == 1
            )
            q_factor = q_rows[0] if shape_passed else None
            factor_identity = bool(
                shape_passed
                and nested_face
                == int(content)
                * linear_rows[0]
                * q_factor
                * square_rows[0] ** 2
            )
            q_audit = (
                _native_bernstein_audit(q_factor, sample_limit=256)
                if factor_identity
                else None
            )
            q_discriminant = (
                _quadratic_discriminant_certificate(q_factor)
                if factor_identity
                and q_audit is not None
                and q_audit["negative_scaled_coefficient_count"]
                else None
            )
            nested_factor_passed = bool(
                factor_identity
                and q_audit is not None
                and (
                    q_audit["negative_scaled_coefficient_count"] == 0
                    or (q_discriminant is not None and q_discriminant["passed"])
                )
            )
            nested_factor_certificate = {
                "identity": "corner=64*(1-ui)*Q*P^2",
                "content": int(content),
                "factor_exponents": [int(row[1]) for row in factors],
                "shape_verified": shape_passed,
                "identity_verified_exactly": factor_identity,
                "q_power_term_count": (
                    len(q_factor.to_dict()) if q_factor is not None else None
                ),
                "q_multidegree": (
                    list(map(int, q_factor.degrees()))
                    if q_factor is not None
                    else None
                ),
                "q_native_bernstein": q_audit,
                "q_quadratic_discriminant_certificate": q_discriminant,
                "passed": nested_factor_passed,
            }
        nested_remainder_audit = _native_bernstein_audit(
            nested_remainder, sample_limit=256
        )
        nested_face_passed = bool(
            nested_identity
            and (
                nested_face_audit["negative_scaled_coefficient_count"] == 0
                or nested_factor_passed
            )
            and nested_remainder_audit["negative_scaled_coefficient_count"] == 0
        )
        nested_face_certificate = {
            "axis": nested_face_axis,
            "endpoint": 0,
            "identity_verified_exactly": nested_identity,
            "corner_power_term_count": len(nested_face.to_dict()),
            "corner_native_bernstein": nested_face_audit,
            "corner_factor_certificate": nested_factor_certificate,
            "remainder_power_term_count": len(nested_remainder.to_dict()),
            "remainder_native_bernstein": nested_remainder_audit,
            "passed": nested_face_passed,
        }
    factor_certificate = None
    factor_passed = False
    if cube_factor_allowed and face_audit["negative_scaled_coefficient_count"]:
        content, factors = face.factor()
        factor_shape_passed = bool(
            int(content) == 64 and len(factors) == 1 and int(factors[0][1]) == 3
        )
        base = factors[0][0] if factor_shape_passed else None
        identity = bool(
            factor_shape_passed and face == int(content) * base ** int(factors[0][1])
        )
        base_audit = (
            _native_bernstein_audit(base, sample_limit=256)
            if identity
            else None
        )
        base_corner_certificate = None
        base_corner_passed = False
        if (
            identity
            and base_audit is not None
            and base_audit["negative_scaled_coefficient_count"] == 1
            and _all_negative_rows_on_axis(base_audit, 0, 0)
            and _all_negative_rows_on_axis(base_audit, 2, 0)
        ):
            ud, _ue, ug, _ui, _y = base.context().gens()
            corner = base.subs({"ud": 0, "ug": 0})
            selector = (1 - ud) ** int(base.degrees()[0]) * (
                1 - ug
            ) ** int(base.degrees()[2])
            corner_remainder = base - corner * selector
            corner_identity = base == corner * selector + corner_remainder
            corner_audit = _native_bernstein_audit(corner, sample_limit=64)
            corner_half_atlas = []
            if corner_audit["negative_scaled_coefficient_count"]:
                for bits in ("00000", "01000"):
                    half_audit = _native_bernstein_audit(
                        _restrict_half_box(corner, bits), sample_limit=32
                    )
                    corner_half_atlas.append(
                        {
                            "five_axis_bits": bits,
                            "ue_interval": (
                                "[0,1/2]" if bits == "00000" else "[1/2,1]"
                            ),
                            "native_bernstein": half_audit,
                            "passed": (
                                half_audit["negative_scaled_coefficient_count"] == 0
                            ),
                        }
                    )
            corner_passed = bool(
                corner_audit["negative_scaled_coefficient_count"] == 0
                or (
                    len(corner_half_atlas) == 2
                    and all(row["passed"] for row in corner_half_atlas)
                )
            )
            corner_sturm = None
            if not corner_passed:
                corner_sturm = _exact_sturm_certificate(corner, axis=1)
                corner_passed = bool(corner_sturm["passed"])
            corner_remainder_audit = _native_bernstein_audit(
                corner_remainder, sample_limit=256
            )
            base_corner_passed = bool(
                corner_identity
                and corner_passed
                and corner_remainder_audit["negative_scaled_coefficient_count"]
                == 0
            )
            base_corner_certificate = {
                "identity": (
                    "P=P|ud=ug=0*(1-ud)^6*(1-ug)^4+remainder"
                ),
                "identity_verified_exactly": corner_identity,
                "corner_power_term_count": len(corner.to_dict()),
                "corner_native_bernstein": corner_audit,
                "corner_half_interval_atlas": corner_half_atlas,
                "corner_exact_sturm_certificate": corner_sturm,
                "remainder_power_term_count": len(corner_remainder.to_dict()),
                "remainder_native_bernstein": corner_remainder_audit,
                "passed": base_corner_passed,
            }
        factor_passed = bool(
            identity
            and base_audit is not None
            and (
                base_audit["negative_scaled_coefficient_count"] == 0
                or base_corner_passed
            )
        )
        factor_certificate = {
            "identity": "face=64*P^3",
            "positive_content": int(content),
            "factor_count": len(factors),
            "factor_exponents": [int(row[1]) for row in factors],
            "identity_verified_exactly": identity,
            "base_power_term_count": len(base.to_dict()) if base is not None else None,
            "base_multidegree": (
                list(map(int, base.degrees())) if base is not None else None
            ),
            "base_native_bernstein": base_audit,
            "base_corner_certificate": base_corner_certificate,
            "passed": factor_passed,
        }
    remainder_audit = _native_bernstein_audit(remainder, sample_limit=1024)
    return {
        "axis": axis,
        "endpoint": endpoint,
        "degree": degree,
        "identity_verified_exactly": identity,
        "face_power_term_count": len(face.to_dict()),
        "face_native_bernstein": face_audit,
        "nested_face_certificate": nested_face_certificate,
        "face_factor_certificate": factor_certificate,
        "face_passed": bool(
            face_audit["negative_scaled_coefficient_count"] == 0
            or factor_passed
            or nested_face_passed
        ),
        "remainder_power_term_count": len(remainder.to_dict()),
        "remainder_native_bernstein": remainder_audit,
        "remainder": remainder,
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
    cubic, _tau, _forced_product, _variables = _build_cubic(coefficient_dir)
    y_zero = cubic.subs({"y": 0})

    stages = []
    current = y_zero
    escaped_boundary = None
    for stage_index, (axis, endpoint) in enumerate(((3, 1), (0, 0), (2, 0))):
        stage = _selector_stage(
            current,
            axis=axis,
            endpoint=endpoint,
            cube_factor_allowed=stage_index == 0,
            nested_face_axis=2 if stage_index == 1 else None,
        )
        remainder = stage.pop("remainder")
        stages.append(stage)
        audit = stage["remainder_native_bernstein"]
        if not stage["identity_verified_exactly"] or not stage["face_passed"]:
            break
        if audit["negative_scaled_coefficient_count"] == 0:
            current = remainder
            break
        next_index = len(stages)
        if next_index >= 3:
            current = remainder
            break
        next_axis, next_endpoint = ((0, 0), (2, 0))[next_index - 1]
        next_value = 0 if next_endpoint == 0 else int(remainder.degrees()[next_axis])
        if not _all_negative_rows_on_axis(audit, next_axis, next_value):
            escaped_boundary = {
                "after_stage": next_index,
                "expected_axis": next_axis,
                "expected_endpoint": next_endpoint,
            }
            current = remainder
            break
        current = remainder

    final_audit = stages[-1]["remainder_native_bernstein"]
    passed = bool(
        escaped_boundary is None
        and all(
            stage["identity_verified_exactly"] and stage["face_passed"]
            for stage in stages
        )
        and final_audit["negative_scaled_coefficient_count"] == 0
    )
    report = {
        "experiment": "adjacent endpoint octet cubic y-zero selector cascade",
        "domain": "(ud,ue,ug,ui) in [0,1]^4",
        "coordinate_order": ["ud", "ue", "ug", "ui", "y"],
        "frozen_stage_order": ["ui=1", "ud=0", "ug=0"],
        "completed_stage_count": len(stages),
        "stages": stages,
        "escaped_preregistered_boundary": escaped_boundary,
        "passed": passed,
        "scope_boundary": (
            "A pass proves only the y=0 face of the cubic. The complete "
            "five-cube still requires a nonnegative two-endpoint remainder."
        ),
        "resource_contract": resource,
    }
    _atomic_json(output, report)
    report["artifact_sha256"] = _sha256(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coefficient-dir", type=Path, default=DEFAULT_COEFFICIENT_DIR
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flint-threads", type=int, default=6)
    arguments = parser.parse_args()
    report = run(
        arguments.coefficient_dir,
        output=arguments.output,
        flint_threads=arguments.flint_threads,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=int))


if __name__ == "__main__":
    main()
