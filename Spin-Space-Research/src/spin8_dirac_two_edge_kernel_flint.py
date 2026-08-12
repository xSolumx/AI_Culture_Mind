"""Independent FLINT replay of the exact two-edge local kernel jet.

The maintained coefficient artifact is parsed once, then the few univariate
coefficient slices controlling the quadratic jet are assembled directly as
FLINT rational polynomials.  This independently checks the SymPy extraction
and the factorization of the only potentially singular ``(e,g)`` block.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp

try:
    from flint import ctx, fmpq, fmpq_poly
except ImportError:  # pragma: no cover - minimal install only
    ctx = fmpq = fmpq_poly = None

from spin8_dirac_two_edge_kernel import (
    EVEN_MASKS,
    ODD_MASKS,
    SectorPolynomial,
    exact_local_kernel_certificate,
    load_sectors,
)

FLINT_AVAILABLE = fmpq_poly is not None


def _flint_rational(value: sp.Expr) -> fmpq:
    rational = sp.Rational(value)
    return fmpq(int(rational.p), int(rational.q))


def _slice_polynomial(
    sector: SectorPolynomial,
    *,
    selected_axis: int | None = None,
) -> fmpq_poly:
    """Extract a polynomial in c2 at the origin of the other coordinates."""

    coefficients: dict[int, sp.Rational] = {}
    for powers, coefficient in sector.terms:
        if powers[4] != 0:
            continue
        desired = [0, 0, 0, 0]
        if selected_axis is not None:
            desired[selected_axis] = 1
        if list(powers[:4]) != desired:
            continue
        coefficients[powers[5]] = coefficients.get(powers[5], sp.Rational(0)) + coefficient
    degree = max(coefficients, default=0)
    return fmpq_poly(
        [_flint_rational(coefficients.get(index, 0)) for index in range(degree + 1)]
    )


def _poly_coefficients(polynomial: fmpq_poly) -> list[str]:
    return [str(value) for value in polynomial]


def run(coefficients: Path, *, flint_threads: int = 6) -> dict[str, object]:
    if not FLINT_AVAILABLE:
        raise RuntimeError("python-flint is required; install the 'exact' extra")
    if not 1 <= flint_threads <= 7:
        raise ValueError("flint_threads must leave at least one CPU core free")
    ctx.threads = flint_threads
    sectors = load_sectors(coefficients)
    sympy_certificate = exact_local_kernel_certificate(sectors)

    trivial = sectors[EVEN_MASKS[0]]
    diagonal = [-_slice_polynomial(trivial, selected_axis=axis) for axis in range(4)]
    even_eg = _slice_polynomial(sectors[EVEN_MASKS[1]])
    odd_dg = _slice_polynomial(sectors[ODD_MASKS[0]])
    odd_de = _slice_polynomial(sectors[ODD_MASKS[1]])
    odd_a = _slice_polynomial(sectors[ODD_MASKS[2]])
    z = fmpq_poly([0, 1])
    expected_diagonal = [
        fmpq(5, 2) * (z - 9) * (z - 5),
        2 * (z - 9) * (z - 3),
        2 * (z - 9) * (z - 3),
        2 * (z - 9) * (z - 3),
    ]
    expected_even_eg = 8 * (z - 9)
    expected_odd_dg = -(z - 23) * (z - 9)
    block_determinant = diagonal[2] * diagonal[3] - z * (even_eg / 2) ** 2
    expected_block_determinant = 4 * (z - 9) ** 3 * (z - 1)

    checks = {
        "diagonal_coefficients_match": diagonal == expected_diagonal,
        "even_eg_core_matches": even_eg == expected_even_eg,
        "odd_dg_core_matches": odd_dg == expected_odd_dg,
        "odd_de_core_is_zero": odd_de == 0,
        "odd_a_core_is_zero": odd_a == 0,
        "eg_block_factorization_matches": (
            block_determinant == expected_block_determinant
        ),
        "sympy_certificate_passed": sympy_certificate["passed"],
    }
    return {
        "experiment": "FLINT replay of two-edge exact local kernel jet",
        "source_coefficient_artifact": str(coefficients),
        "flint_threads": flint_threads,
        "direct_flint_slices": {
            "lambda_diagonal": [_poly_coefficients(row) for row in diagonal],
            "even_eg_core": _poly_coefficients(even_eg),
            "odd_dg_core": _poly_coefficients(odd_dg),
            "odd_de_core": _poly_coefficients(odd_de),
            "odd_a_core": _poly_coefficients(odd_a),
            "eg_block_determinant": _poly_coefficients(block_determinant),
        },
        "checks": checks,
        "scope_boundary": (
            "FLINT independently checks exact rational coefficient arithmetic "
            "after loading the maintained reconstruction artifact; it does not "
            "independently reconstruct the 28 by 28 determinant data."
        ),
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coefficients", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run(arguments.coefficients, flint_threads=arguments.threads)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(arguments.output), "passed": report["passed"]}))
    if not report["passed"]:
        raise SystemExit("two-edge FLINT local-kernel replay failed")


if __name__ == "__main__":
    main()
