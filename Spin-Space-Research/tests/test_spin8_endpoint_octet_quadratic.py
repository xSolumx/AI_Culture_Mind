"""Focused regression tests for the endpoint-octet quadratic certificate."""

from __future__ import annotations

import unittest
from pathlib import Path

import sympy as sp
from flint import fmpz_mpoly_ctx

from spin8_dirac_endpoint_octet_quadratic import (
    _native_bernstein_audit,
    _restrict_half_box,
)
from spin8_dirac_endpoint_octet_quadratic_certificate import verify

ROOT = Path(__file__).resolve().parents[1]


def _to_sympy(polynomial, symbols: tuple[sp.Symbol, ...]) -> sp.Expr:
    return sp.expand(
        sum(
            sp.Integer(int(coefficient))
            * sp.prod(symbol**power for symbol, power in zip(symbols, powers))
            for powers, coefficient in polynomial.to_dict().items()
        )
    )


class EndpointOctetQuadraticTests(unittest.TestCase):
    def test_dyadic_restriction_is_exact_integer_pullback(self) -> None:
        context = fmpz_mpoly_ctx.get(["x0", "x1", "x2", "x3", "x4"])
        x0, x1, x2, x3, x4 = context.gens()
        polynomial = 7 + 3 * x0**2 * x4 + 5 * x1 * x3**3 - 2 * x2 * x4**2
        bits = "10101"
        transformed = _restrict_half_box(polynomial, bits)

        symbols = sp.symbols("x0:5")
        original = _to_sympy(polynomial, symbols)
        degrees = tuple(int(value) for value in polynomial.degrees())
        scale = 2 ** sum(degrees)
        substitutions = {
            symbol: (symbol + int(bit)) / 2
            for symbol, bit in zip(symbols, bits, strict=True)
        }
        expected = sp.expand(scale * original.subs(substitutions, simultaneous=True))
        self.assertEqual(_to_sympy(transformed, symbols), expected)

    def test_zero_degree_axes_are_not_reported_as_boundaries(self) -> None:
        context = fmpz_mpoly_ctx.get(["x0", "x1", "x2", "x3", "x4"])
        audit = _native_bernstein_audit(context.from_dict({(0, 0, 0, 0, 0): -1}))
        self.assertEqual(audit["negative_boundary_histogram"], {"interior-control": 1})

    def test_assembled_global_quadratic_certificates_replay(self) -> None:
        for minor_index in (0, 1, 2):
            report = (
                ROOT
                / "artifacts"
                / (
                    f"spin8_dirac_endpoint_octet_quadratic_{minor_index}_global_"
                    "20260808.json"
                )
            )
            verification = verify(report)
            self.assertIs(verification["verified"], True)
            self.assertEqual(verification["source_artifact_count"], 8)


if __name__ == "__main__":
    unittest.main()
