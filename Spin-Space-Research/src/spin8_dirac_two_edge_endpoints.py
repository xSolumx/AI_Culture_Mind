"""Exact endpoint-jet atlas for the finite second Dirac edge.

The all-sector reconstruction is polynomial in the squared edge coordinate
``i2``.  This module evaluates every sector at ``i2=0`` and ``i2=1`` and
certifies the common flag ideal in the first derivative at the fully active
endpoint.  It is a structural reduction, not a positivity certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import sympy as sp

from spin8_dirac_two_edge_kernel import load_sectors
from spin8_resource_limits import constrain_current_process

VARIABLES = sp.symbols("a2 d2 e2 g2 i2 c2")
BASE_VARIABLES = VARIABLES[:4] + VARIABLES[5:]
A2, D2, E2, G2, I2, C2 = VARIABLES


EXPECTED_ENDPOINT_MULTIPLICITIES = {
    (0, 0, 0, 0, 0, 0): (0, 0, 0, 0),
    (0, 0, 1, 1, 0, 1): (1, 1, 0, 0),
    (0, 1, 0, 1, 1, 0): (0, 0, 0, 0),
    (0, 1, 1, 0, 1, 1): (1, 1, 1, 1),
    (1, 0, 0, 0, 1, 1): (1, 1, 1, 1),
    (1, 0, 1, 1, 1, 0): (0, 1, 0, 0),
    (1, 1, 0, 1, 0, 1): (1, 0, 0, 0),
    (1, 1, 1, 0, 0, 0): (0, 0, 0, 0),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sector_poly(sector) -> sp.Poly:
    return sp.Poly.from_dict(dict(sector.terms), VARIABLES)


def _factor_multiplicity(polynomial: sp.Poly, variable: sp.Symbol) -> int:
    factor = sp.Poly(1 - variable, *BASE_VARIABLES)
    multiplicity = 0
    current = polynomial
    while current.rem(factor).is_zero:
        current = current.exquo(factor)
        multiplicity += 1
    return multiplicity


def _divide_coordinate_factors(
    polynomial: sp.Poly, multiplicities: tuple[int, ...]
) -> sp.Poly:
    current = polynomial
    for variable, multiplicity in zip(BASE_VARIABLES[:4], multiplicities, strict=True):
        if multiplicity:
            current = current.exquo(
                sp.Poly((1 - variable) ** multiplicity, *BASE_VARIABLES)
            )
    return current


def exact_endpoint_jet_certificate(
    coefficient_artifact: Path,
    *,
    workers: int = 6,
) -> dict[str, object]:
    resource = constrain_current_process(workers=workers)
    sectors = load_sectors(coefficient_artifact)
    rows = []
    passed = True
    for mask, sector in sorted(sectors.items()):
        polynomial = _sector_poly(sector)
        endpoint = sp.Poly(polynomial.as_expr().subs(I2, 1), *BASE_VARIABLES)
        derivative = sp.Poly(
            sp.diff(polynomial.as_expr(), I2).subs(I2, 1), *BASE_VARIABLES
        )
        one_edge = sp.Poly(polynomial.as_expr().subs(I2, 0), *BASE_VARIABLES)
        endpoint_multiplicities = tuple(
            _factor_multiplicity(endpoint, variable) for variable in BASE_VARIABLES[:4]
        )
        derivative_multiplicities = tuple(
            _factor_multiplicity(derivative, variable)
            for variable in BASE_VARIABLES[:4]
        )
        expected_endpoint = EXPECTED_ENDPOINT_MULTIPLICITIES[mask]
        expected_derivative = tuple(
            value + increment
            for value, increment in zip(expected_endpoint, (0, 1, 1, 1), strict=True)
        )
        endpoint_quotient = _divide_coordinate_factors(
            endpoint, endpoint_multiplicities
        )
        derivative_quotient = _divide_coordinate_factors(
            derivative, derivative_multiplicities
        )
        row_passed = bool(
            endpoint.degree(C2) == 0
            and derivative.degree(C2) <= 1
            and endpoint_multiplicities == expected_endpoint
            and derivative_multiplicities == expected_derivative
        )
        passed &= row_passed
        rows.append(
            {
                "mask": list(mask),
                "one_edge_term_count": len(one_edge.terms()),
                "fully_active_endpoint_term_count": len(endpoint.terms()),
                "fully_active_endpoint_c2_degree": int(endpoint.degree(C2)),
                "endpoint_factor_multiplicities_a2_d2_e2_g2": list(
                    endpoint_multiplicities
                ),
                "endpoint_quotient_term_count": len(endpoint_quotient.terms()),
                "i2_derivative_term_count": len(derivative.terms()),
                "i2_derivative_c2_degree": int(derivative.degree(C2)),
                "derivative_factor_multiplicities_a2_d2_e2_g2": list(
                    derivative_multiplicities
                ),
                "derivative_quotient_term_count": len(derivative_quotient.terms()),
                "passed": row_passed,
            }
        )
    return {
        "theorem": "finite two-edge exact endpoint-jet flag law",
        "coefficient_artifact": coefficient_artifact.name,
        "coefficient_artifact_sha256": _sha256(coefficient_artifact),
        "sector_count": len(rows),
        "fully_active_endpoint_core_is_cayley_independent_in_every_sector": all(
            row["fully_active_endpoint_c2_degree"] == 0 for row in rows
        ),
        "first_i2_derivative_has_universal_extra_flag_factor": ("(1-d2)(1-e2)(1-g2)"),
        "endpoint_rows": rows,
        "finite_y_consequence": (
            "With i2=1-y^2, core variation begins at order y^2. The order-y "
            "endpoint layer comes only from sectors carrying the forced "
            "complement coordinate I=y; their endpoint cores are c2-independent."
        ),
        "scope_boundary": (
            "The assembled channel amplitudes retain Cayley dependence through "
            "their forced character monomials. This certificate does not prove "
            "the y=0 center or separation polynomials nonnegative."
        ),
        "resource_contract": resource,
        "passed": bool(passed and len(rows) == 8),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coefficients",
        type=Path,
        default=Path(
            "artifacts/spin8_dirac_two_edge_all_sectors_coefficients_20260806.json"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    report = exact_endpoint_jet_certificate(
        arguments.coefficients,
        workers=arguments.workers,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("two-edge endpoint-jet certificate failed")


if __name__ == "__main__":
    main()
