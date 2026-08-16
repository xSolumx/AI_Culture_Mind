"""Regression tests for the endpoint-octet determinant endpoint theorem."""

from __future__ import annotations

import unittest
from pathlib import Path

from spin8_dirac_endpoint_octet_determinant import (
    _generic_determinant_identity,
    verify_report,
)
from spin8_dirac_endpoint_octet_determinant_tangent import (
    _manifest_quartic,
    _quartic_sign_certificate,
)
from spin8_dirac_endpoint_octet_determinant_tangent import (
    verify_report as verify_tangent_report,
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

    def test_determinant_tangent_quartic_is_manifestly_nonnegative(self) -> None:
        quartic, _quadratic, _linear = _manifest_quartic()
        certificate = _quartic_sign_certificate(quartic)
        self.assertIs(certificate["passed"], True)
        self.assertIs(certificate["nonnegative_on_the_deviation_cone"], True)

    def test_tangent_and_selector_rejection_artifact_replays(self) -> None:
        verification = verify_tangent_report(
            ROOT
            / "artifacts"
            / "spin8_dirac_endpoint_octet_determinant_tangent_20260812.json",
            coefficient_dir=(
                ROOT / "artifacts" / "spin8_dirac_unrestricted_coefficients_20260807"
            ),
        )
        self.assertIs(verification["verified"], True)
        self.assertEqual(verification["failures"], [])
        self.assertLess(int(verification["replayed_selector_witness_numerator"]), 0)


if __name__ == "__main__":
    unittest.main()
