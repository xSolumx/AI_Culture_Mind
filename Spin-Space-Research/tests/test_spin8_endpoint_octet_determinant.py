"""Regression tests for the endpoint-octet determinant endpoint theorem."""

from __future__ import annotations

import unittest
from pathlib import Path

from spin8_dirac_endpoint_octet_determinant import (
    _generic_determinant_identity,
    verify_report,
)

ROOT = Path(__file__).resolve().parents[1]


class EndpointOctetDeterminantTests(unittest.TestCase):
    def test_generic_formula_is_the_klein_determinant(self) -> None:
        self.assertEqual(
            _generic_determinant_identity(),
            {
                "four_by_four_determinant": True,
                "four_walsh_eigenvalues": True,
            },
        )

    def test_endpoint_certificate_has_a_complete_compact_replay(self) -> None:
        verification = verify_report(
            ROOT / "artifacts" / "spin8_dirac_endpoint_octet_determinant_20260812.json",
            coefficient_dir=(
                ROOT / "artifacts" / "spin8_dirac_unrestricted_coefficients_20260807"
            ),
            endpoint_artifact=(
                ROOT / "artifacts" / "spin8_dirac_endpoint_octet_20260807.json"
            ),
        )
        self.assertIs(verification["verified"], True)
        self.assertEqual(verification["failures"], [])


if __name__ == "__main__":
    unittest.main()
