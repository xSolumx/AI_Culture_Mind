"""Exact homogeneous tangent cone of the adjacent-octet Schur cubic."""

from __future__ import annotations

import argparse
import json
from functools import reduce
from math import comb
from operator import mul
from pathlib import Path

import sympy as sp
from flint import ctx, fmpz_mpoly_ctx

from spin8_dirac_endpoint_octet_cubic import DEFAULT_COEFFICIENT_DIR, _build_cubic
from spin8_dirac_endpoint_octet_quadratic import (
    _atomic_json,
    _native_bernstein_audit,
    _sha256,
)
from spin8_resource_limits import constrain_current_process

DEVIATIONS = ("ud", "ue", "ug", "ui", "one_minus_y")


def _homogeneous_taylor(polynomial, *, max_order: int):
    context = fmpz_mpoly_ctx.get(DEVIATIONS)
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


def _factor_report(polynomial):
    content, factors = polynomial.factor()
    rebuilt = reduce(
        mul,
        (factor ** int(exponent) for factor, exponent in factors),
        polynomial.context().constant(int(content)),
    )
    report = {
        "content": str(content),
        "factor_count": len(factors),
        "factors": [
            {
                "exponent": int(exponent),
                "power_term_count": len(factor.to_dict()),
                "multidegree": list(map(int, factor.degrees())),
                "coefficient_rows": [
                    {
                        "powers": list(map(int, powers)),
                        "coefficient": str(coefficient),
                    }
                    for powers, coefficient in sorted(factor.to_dict().items())
                ],
            }
            for factor, exponent in factors
        ],
        "identity_verified_exactly": rebuilt == polynomial,
    }
    return report, factors


def _to_sympy(polynomial, variables: tuple[sp.Symbol, ...]) -> sp.Expr:
    return sp.expand(
        sum(
            sp.Integer(int(coefficient))
            * sp.prod(variable**power for variable, power in zip(variables, powers))
            for powers, coefficient in polynomial.to_dict().items()
        )
    )


def _radical_factor_certificate(exact_factors) -> dict[str, object]:
    d, e, g, i, t = sp.symbols("d e g i t", nonnegative=True)
    variables = (d, e, g, i, t)
    ordered = sorted(
        (factor for factor, exponent in exact_factors if int(exponent) == 1),
        key=lambda factor: max(map(int, factor.degrees())),
    )
    if len(ordered) != 2:
        return {"passed": False, "failure": "expected two exponent-one factors"}
    f2 = _to_sympy(ordered[0], variables)
    f4 = _to_sympy(ordered[1], variables)
    linear = 2 * d + 2 * e + 2 * g + 5 * i + 4 * t
    manifest_f2 = (
        4 * (d - 2 * t) ** 2
        + 4 * e**2
        + 4 * g**2
        + 25 * i**2
        + 8 * d * e
        + 8 * d * g
        + 20 * d * i
        + 24 * e * g
        + 20 * e * i
        + 16 * e * t
        + 20 * g * i
        + 16 * g * t
        + 40 * i * t
    )
    f2_manifest_identity = sp.expand(f2 - manifest_f2) == 0
    f2_linear_identity = sp.expand(f2 - (linear**2 + 16 * e * g - 32 * d * t)) == 0
    f4_difference_identity = sp.expand(f4 - (f2**2 - 64 * e * g * linear**2)) == 0
    passed = bool(
        f2_manifest_identity and f2_linear_identity and f4_difference_identity
    )
    return {
        "linear_form": "L=2d+2e+2g+5i+4t",
        "f2_manifest_copositive_identity_verified": f2_manifest_identity,
        "f2_linear_identity_verified": f2_linear_identity,
        "f4_difference_of_squares_identity_verified": f4_difference_identity,
        "radical_factorization": (
            "F4=(L-a-b)(L-a+b)(L+a-b)(L+a+b), "
            "a=4*sqrt(e*g), b=4*sqrt(2*d*t)"
        ),
        "am_gm_bounds": [
            "2e+2g>=4*sqrt(e*g)=a",
            "2d+4t>=4*sqrt(2*d*t)=b",
            "therefore L>=a+b",
        ],
        "every_radical_factor_nonnegative_on_the_nonnegative_cone": passed,
        "passed": passed,
    }


def verify_report(report: dict[str, object]) -> dict[str, object]:
    """Replay the compact order-six factor certificate from stored rows."""

    failures: list[str] = []
    if report.get("exact_vanishing_order") != 6:
        failures.append("the stored vanishing order is not six")
    if report.get("tangent_power_term_count") != 210:
        failures.append("the stored tangent term count is not 210")
    factorization = report.get("factorization", {})
    if factorization.get("content") != str(2**36):
        failures.append("the tangent content is not 2^36")
    rows = factorization.get("factors", [])
    context = fmpz_mpoly_ctx.get(DEVIATIONS)
    factors = []
    for row in rows:
        coefficients = {
            tuple(item["powers"]): int(item["coefficient"])
            for item in row.get("coefficient_rows", [])
        }
        factors.append((context.from_dict(coefficients), row.get("exponent")))
    replay = _radical_factor_certificate(factors) if len(factors) == 2 else None
    if replay is None or not replay["passed"]:
        failures.append("the radical factor certificate did not replay")
    return {"verified": not failures, "failures": failures, "replay": replay}


def run(
    coefficient_dir: Path,
    *,
    output: Path,
    max_order: int = 12,
    flint_threads: int = 6,
) -> dict[str, object]:
    if not 1 <= flint_threads <= 6:
        raise ValueError("FLINT thread count must be between one and six")
    if max_order < 1:
        raise ValueError("max-order must be positive")
    resource = constrain_current_process(workers=flint_threads)
    ctx.threads = flint_threads
    cubic, _tau, _forced_product, _variables = _build_cubic(coefficient_dir)
    tangent, order = _homogeneous_taylor(cubic, max_order=max_order)
    terms = tangent.to_dict()
    homogeneous = all(sum(powers) == order for powers in terms)
    factor, exact_factors = _factor_report(tangent)
    chart_rows = []
    for axis, name in enumerate(DEVIATIONS):
        face = tangent.subs({name: 1})
        audit = _native_bernstein_audit(face, sample_limit=128)
        chart_rows.append(
            {
                "pivot_axis": axis,
                "pivot_deviation": name,
                "power_term_count": len(face.to_dict()),
                "native_bernstein": audit,
                "passed": audit["negative_scaled_coefficient_count"] == 0,
            }
        )
    factor_chart_rows = []
    for factor_index, (exact_factor, exponent) in enumerate(exact_factors):
        faces = []
        for axis, name in enumerate(DEVIATIONS):
            face = exact_factor.subs({name: 1})
            audit = _native_bernstein_audit(face, sample_limit=64)
            faces.append(
                {
                    "pivot_axis": axis,
                    "pivot_deviation": name,
                    "native_bernstein": audit,
                    "passed": audit["negative_scaled_coefficient_count"] == 0,
                }
            )
        factor_chart_rows.append(
            {
                "factor_index": factor_index,
                "exponent": int(exponent),
                "faces": faces,
                "passed": all(row["passed"] for row in faces),
            }
        )
    factors_passed = bool(
        int(factor["content"]) > 0
        and all(row["passed"] for row in factor_chart_rows)
    )
    radical_factor_certificate = _radical_factor_certificate(exact_factors)
    passed = bool(
        homogeneous
        and factor["identity_verified_exactly"]
        and (
            all(row["passed"] for row in chart_rows)
            or factors_passed
            or radical_factor_certificate["passed"]
        )
    )
    report = {
        "experiment": "adjacent endpoint octet cubic equality tangent cone",
        "equality_point": "ud=ue=ug=ui=0, y=1",
        "deviation_order": list(DEVIATIONS),
        "maximum_reconstructed_order": max_order,
        "exact_vanishing_order": order,
        "tangent_power_term_count": len(terms),
        "tangent_multidegree": list(map(int, tangent.degrees())),
        "homogeneity_verified_exactly": homogeneous,
        "factorization": factor,
        "factor_chart_certificates": factor_chart_rows,
        "positive_factor_route_passed": factors_passed,
        "radical_factor_certificate": radical_factor_certificate,
        "max_coordinate_chart_faces": chart_rows,
        "passed": passed,
        "scope_boundary": (
            "A pass proves only nonnegativity of the exceptional divisor; "
            "every full radial chart still needs a remainder certificate."
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
    parser.add_argument("--max-order", type=int, default=12)
    parser.add_argument("--flint-threads", type=int, default=6)
    arguments = parser.parse_args()
    report = run(
        arguments.coefficient_dir,
        output=arguments.output,
        max_order=arguments.max_order,
        flint_threads=arguments.flint_threads,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
