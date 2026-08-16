"""Exact tangent theorem and selector rejection for the octet determinant.

The determinant is even in ``y``, so this module first descends exactly to
``t=y^2``.  It then records two logically separate results:

* the degree-matched two-endpoint selector quotient is *not* nonnegative;
  an exact rational witness rejects that proposed proof route;
* the determinant itself has order-eight leading form at the remaining
  equality corner, and that form is a positive constant times the square of
  an explicitly nonnegative quartic.

Neither statement is promoted as finite-radius positivity of the determinant.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

from flint import ctx, fmpz_mpoly_ctx

from spin8_dirac_endpoint_octet_cubic_tangent import (
    _factor_report,
)
from spin8_dirac_endpoint_octet_determinant import (
    DEFAULT_COEFFICIENT_DIR,
    _build_determinant,
    _coefficient_hashes,
)
from spin8_dirac_endpoint_octet_quadratic import _atomic_json, _sha256
from spin8_resource_limits import constrain_current_process

T_VARIABLES = ("ud", "ue", "ug", "ui", "t")
T_DEVIATIONS = ("ud", "ue", "ug", "ui", "one_minus_t")
SELECTOR_DEGREE = 24
WITNESS_DENOMINATOR = 100


def _descend_to_t(determinant):
    """Replace every exact even power ``y^(2j)`` by ``t^j``."""

    context = fmpz_mpoly_ctx.get(T_VARIABLES)
    coefficients = {}
    for powers, coefficient in determinant.to_dict().items():
        if powers[4] % 2:
            raise AssertionError(f"determinant contains odd y power {powers[4]}")
        target = tuple(powers[:4]) + (powers[4] // 2,)
        if target in coefficients:
            raise AssertionError("even-power descent was not injective")
        coefficients[target] = int(coefficient)
    return context.from_dict(coefficients)


def _selector_quotient(determinant_t, *, degree: int = SELECTOR_DEGREE):
    """Return the exact two-endpoint quotient and its endpoint faces."""

    t = determinant_t.context().gens()[4]
    low = determinant_t.subs({"t": 0})
    high = determinant_t.subs({"t": 1})
    remainder = determinant_t - low * (1 - t) ** degree - high * t**degree
    quotient, division_remainder = divmod(remainder, t * (1 - t))
    if division_remainder:
        raise AssertionError("endpoint selector remainder is not divisible by t(1-t)")
    identity = determinant_t == (
        low * (1 - t) ** degree + high * t**degree + t * (1 - t) * quotient
    )
    return quotient, low, high, identity


def _manifest_quartic():
    """Return the nonnegative quartic governing the determinant tangent."""

    context = fmpz_mpoly_ctx.get(T_DEVIATIONS)
    d, e, g, i, r = context.gens()
    linear = 2 * d + 2 * e + 2 * g + 5 * i + 2 * r
    quadratic = (
        4 * (d - r) ** 2
        + 4 * e**2
        + 4 * g**2
        + 25 * i**2
        + 8 * d * e
        + 8 * d * g
        + 20 * d * i
        + 24 * e * g
        + 20 * e * i
        + 8 * e * r
        + 20 * g * i
        + 8 * g * r
        + 20 * i * r
    )
    quartic = quadratic**2 - 64 * e * g * linear**2
    return quartic, quadratic, linear


def _homogeneous_taylor_t(polynomial, *, max_order: int):
    """Return the first homogeneous form at ``(0,0,0,0,t=1)``."""

    context = fmpz_mpoly_ctx.get(T_DEVIATIONS)
    by_order: dict[int, dict[tuple[int, ...], int]] = {
        order: {} for order in range(max_order + 1)
    }
    for powers, coefficient in polynomial.to_dict().items():
        base_order = sum(powers[:4])
        if base_order > max_order:
            continue
        for t_power in range(min(powers[4], max_order - base_order) + 1):
            order = base_order + t_power
            target = tuple(powers[:4]) + (t_power,)
            value = int(coefficient) * comb(powers[4], t_power) * (-1) ** t_power
            row = by_order[order]
            row[target] = row.get(target, 0) + value
    for order in range(max_order + 1):
        cleaned = {powers: value for powers, value in by_order[order].items() if value}
        if cleaned:
            return context.from_dict(cleaned), order
    raise AssertionError(f"no nonzero Taylor component through order {max_order}")


def _quartic_sign_certificate(quartic) -> dict[str, object]:
    """Replay the radical factor argument on the nonnegative cone."""

    manifest, quadratic, linear = _manifest_quartic()
    d, e, g, _i, r = manifest.context().gens()
    quadratic_identity = quadratic == linear**2 + 16 * e * g - 16 * d * r
    difference_identity = quartic == quadratic**2 - 64 * e * g * linear**2
    manifest_identity = quartic == manifest
    passed = bool(manifest_identity and quadratic_identity and difference_identity)
    return {
        "linear_form": "L=2d+2e+2g+5i+2r",
        "quadratic_identity": "F2=L^2+16*e*g-16*d*r",
        "quartic_identity": "F4=F2^2-64*e*g*L^2",
        "manifest_quartic_identity_verified_exactly": manifest_identity,
        "quadratic_linear_identity_verified_exactly": quadratic_identity,
        "difference_of_squares_identity_verified_exactly": difference_identity,
        "radical_factorization": (
            "F4=(L-a-b)(L-a+b)(L+a-b)(L+a+b), "
            "a=4*sqrt(e*g), b=4*sqrt(d*r)"
        ),
        "am_gm_bounds": [
            "2e+2g>=4*sqrt(e*g)=a",
            "2d+2r>=4*sqrt(d*r)=b",
            "therefore L>=a+b on the nonnegative cone",
        ],
        "nonnegative_on_the_deviation_cone": passed,
        "passed": passed,
    }


def _restrict_ud_t(polynomial):
    """Restrict a five-variable polynomial to ``ue=ug=ui=0``."""

    context = fmpz_mpoly_ctx.get(("ud", "t"))
    coefficients = {
        (powers[0], powers[4]): int(coefficient)
        for powers, coefficient in polynomial.to_dict().items()
        if powers[1] == powers[2] == powers[3] == 0
    }
    return context.from_dict(coefficients)


def _scaled_rational_numerator(
    polynomial, *, numerators: tuple[int, ...], denominator: int
) -> int:
    """Evaluate exactly, clearing one positive common denominator."""

    if denominator <= 0 or len(numerators) != polynomial.context().nvars():
        raise ValueError("invalid common-denominator rational point")
    degrees = tuple(map(int, polynomial.degrees()))
    total_degree = sum(degrees)
    result = 0
    for powers, coefficient in polynomial.to_dict().items():
        term = int(coefficient)
        for numerator, power in zip(numerators, powers, strict=True):
            term *= numerator**power
        term *= denominator ** (total_degree - sum(powers))
        result += term
    return result


def _coefficient_rows(polynomial) -> list[dict[str, object]]:
    return [
        {"powers": list(map(int, powers)), "coefficient": str(coefficient)}
        for powers, coefficient in sorted(polynomial.to_dict().items())
    ]


def verify_report(
    report_or_path, *, coefficient_dir: Path = DEFAULT_COEFFICIENT_DIR
) -> dict[str, object]:
    """Compactly replay the stored quartic and rational route rejection."""

    if isinstance(report_or_path, (str, Path)):
        report = json.loads(Path(report_or_path).read_text(encoding="utf-8"))
    else:
        report = report_or_path
    failures: list[str] = []
    if report.get("coefficient_source_sha256") != _coefficient_hashes(coefficient_dir):
        failures.append("coefficient-source hashes disagree")

    tangent = report.get("determinant_tangent", {})
    factorization = tangent.get("factorization", {})
    rows = factorization.get("factors", [])
    factor = None
    exponent = None
    if len(rows) == 1:
        row = rows[0]
        context = fmpz_mpoly_ctx.get(T_DEVIATIONS)
        factor = context.from_dict(
            {
                tuple(item["powers"]): int(item["coefficient"])
                for item in row.get("coefficient_rows", [])
            }
        )
        exponent = row.get("exponent")
    if factor is None or exponent != 2:
        failures.append("tangent does not contain one squared quartic factor")
        sign = None
    else:
        sign = _quartic_sign_certificate(factor)
        if not sign["passed"]:
            failures.append("stored tangent quartic fails its sign certificate")
    if tangent.get("exact_vanishing_order") != 8:
        failures.append("determinant tangent order is not eight")
    if factorization.get("content") != str(2**48):
        failures.append("determinant tangent content is not 2^48")
    if not tangent.get("leading_form_is_content_times_quartic_squared"):
        failures.append("source harness did not verify the squared leading form")

    rejection = report.get("selector_route_rejection", {})
    if rejection.get("identity_verified_exactly") is not True:
        failures.append("selector identity was not verified exactly")
    face = rejection.get("restricted_ue_ug_ui_zero", {})
    face_context = fmpz_mpoly_ctx.get(("ud", "t"))
    face_polynomial = face_context.from_dict(
        {
            tuple(item["powers"]): int(item["coefficient"])
            for item in face.get("coefficient_rows", [])
        }
    )
    witness = rejection.get("rational_witness", {})
    numerator = _scaled_rational_numerator(
        face_polynomial,
        numerators=(1, 99),
        denominator=WITNESS_DENOMINATOR,
    )
    if str(numerator) != witness.get("scaled_exact_numerator") or numerator >= 0:
        failures.append("selector-quotient rational rejection did not replay")
    if rejection.get("selector_degree") != SELECTOR_DEGREE:
        failures.append("selector degree disagrees")
    if rejection.get("selector_quotient_nonnegative") is not False:
        failures.append("rejected selector route was not marked false")

    if report.get("global_determinant_interior_proved") is not False:
        failures.append("local artifact overclaims global determinant positivity")
    if not report.get("passed"):
        failures.append("local tangent/rejection artifact is not marked passed")
    return {
        "verified": not failures,
        "failures": failures,
        "replayed_quartic_sign_certificate": sign,
        "replayed_selector_witness_numerator": str(numerator),
        "compact_trust_boundary": (
            "The compact verifier reconstructs the stored quartic and restricted "
            "selector face. Full determinant reconstruction and Taylor extraction "
            "remain the source-harness tier."
        ),
    }


def run(
    coefficient_dir: Path,
    *,
    output: Path,
    flint_threads: int = 6,
) -> dict[str, object]:
    """Reconstruct and certify the exact local determinant geometry."""

    if not 1 <= flint_threads <= 6:
        raise ValueError("FLINT thread count must be between one and six")
    resource = constrain_current_process(workers=flint_threads)
    ctx.threads = flint_threads

    determinant, _tau, _forced_product, _variables = _build_determinant(
        coefficient_dir
    )
    determinant_t = _descend_to_t(determinant)
    if tuple(map(int, determinant_t.degrees())) != (24, 24, 24, 24, 24):
        raise AssertionError("unexpected t-descended determinant multidegree")
    quotient, low, high, selector_identity = _selector_quotient(determinant_t)

    quotient_face = _restrict_ud_t(quotient)
    witness_numerator = _scaled_rational_numerator(
        quotient_face, numerators=(1, 99), denominator=WITNESS_DENOMINATOR
    )
    if witness_numerator >= 0:
        raise AssertionError("frozen selector rejection witness is not negative")

    tangent, order = _homogeneous_taylor_t(determinant_t, max_order=16)
    factorization, factors = _factor_report(tangent)
    if order != 8 or len(factors) != 1 or int(factors[0][1]) != 2:
        raise AssertionError("unexpected determinant tangent factorization")
    quartic = factors[0][0]
    sign = _quartic_sign_certificate(quartic)
    squared_identity = tangent == (2**48) * quartic**2
    tangent_passed = bool(
        factorization["identity_verified_exactly"]
        and squared_identity
        and sign["passed"]
    )
    passed = bool(selector_identity and witness_numerator < 0 and tangent_passed)

    report = {
        "experiment": "adjacent endpoint-octet determinant tangent and route audit",
        "domain": "(ud,ue,ug,ui,t=y^2) in [0,1]^5",
        "even_power_descent": {
            "identity": "t=y^2",
            "all_y_powers_even": True,
            "power_term_count": len(determinant_t.to_dict()),
            "multidegree": list(map(int, determinant_t.degrees())),
        },
        "coefficient_source_sha256": _coefficient_hashes(coefficient_dir),
        "selector_route_rejection": {
            "identity": (
                "D=D0*(1-t)^24+D1*t^24+t*(1-t)*Q24"
            ),
            "identity_verified_exactly": selector_identity,
            "selector_degree": SELECTOR_DEGREE,
            "quotient_power_term_count": len(quotient.to_dict()),
            "quotient_multidegree": list(map(int, quotient.degrees())),
            "endpoint_power_term_counts": {
                "t_zero": len(low.to_dict()),
                "t_one": len(high.to_dict()),
            },
            "restricted_ue_ug_ui_zero": {
                "power_term_count": len(quotient_face.to_dict()),
                "multidegree": list(map(int, quotient_face.degrees())),
                "coefficient_rows": _coefficient_rows(quotient_face),
            },
            "rational_witness": {
                "point": {
                    "ud": "1/100",
                    "ue": "0",
                    "ug": "0",
                    "ui": "0",
                    "t": "99/100",
                },
                "common_denominator": WITNESS_DENOMINATOR,
                "scaled_exact_numerator": str(witness_numerator),
                "strictly_negative": witness_numerator < 0,
            },
            "selector_quotient_nonnegative": False,
            "interpretation": (
                "This exactly rejects the degree-matched endpoint-selector proof "
                "route. It is not a negative value of D."
            ),
        },
        "determinant_tangent": {
            "equality_point": "ud=ue=ug=ui=0, t=1",
            "deviation_order": list(T_DEVIATIONS),
            "exact_vanishing_order": order,
            "tangent_power_term_count": len(tangent.to_dict()),
            "tangent_multidegree": list(map(int, tangent.degrees())),
            "homogeneity_verified_exactly": all(
                sum(powers) == order for powers in tangent.to_dict()
            ),
            "factorization": factorization,
            "leading_form_identity": "T8=2^48*F4^2",
            "leading_form_is_content_times_quartic_squared": squared_identity,
            "quartic_sign_certificate": sign,
            "proved_nonnegative_on_deviation_cone": tangent_passed,
        },
        "passed": passed,
        "global_determinant_interior_proved": False,
        "next_exact_gate": (
            "A finite-radius max-coordinate blow-up for D itself, retaining the "
            "nonnegative order-eight exceptional divisor."
        ),
        "scope_boundary": (
            "This proves the equality-corner tangent theorem and exactly rejects "
            "one over-strong selector decomposition. It does not prove D>=0 on "
            "the punctured interior or the complete adjacent endpoint octet."
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
