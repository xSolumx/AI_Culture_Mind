"""Tests for the complete coordinate-boundary determinant theorem."""

from __future__ import annotations

import unittest
from pathlib import Path

from spin8_dirac_endpoint_octet_determinant_boundary import (
    _generic_square_identities,
    _square_face_rows,
    verify_report,
)

ROOT = Path(__file__).resolve().parents[1]
COEFFICIENT_DIR = ROOT / "artifacts" / "spin8_dirac_unrestricted_coefficients_20260807"
DETERMINANT_ARTIFACT = (
    ROOT / "artifacts" / "spin8_dirac_endpoint_octet_determinant_20260812.json"
)
ENDPOINT_ARTIFACT = ROOT / "artifacts" / "spin8_dirac_endpoint_octet_20260807.json"
BOUNDARY_ARTIFACT = (
    ROOT / "artifacts" / "spin8_dirac_endpoint_octet_determinant_boundary_20260816.json"
)


class EndpointOctetDeterminantBoundaryTests(unittest.TestCase):
    def test_generic_one_mode_determinants_are_perfect_squares(self) -> None:
        self.assertEqual(
            _generic_square_identities(),
            {
                "zero_active_modes": True,
                "active_mode_0": True,
                "active_mode_1": True,
                "active_mode_2": True,
            },
        )

    def test_nine_coordinate_faces_have_at_most_one_active_mode(self) -> None:
        rows = _square_face_rows(COEFFICIENT_DIR)
        self.assertEqual(len(rows), 9)
        expected_active = {
            "ud=0": ["0011001"],
            "ud=1": ["0011001"],
            "ue=0": ["0101010"],
            "ue=1": ["0110011"],
            "ug=0": ["0110011"],
            "ug=1": ["0110011"],
            "ui=0": ["0011001"],
            "ui=1": ["0101010"],
            "y=0": ["0110011"],
        }
        self.assertEqual(
            {row["face"]: row["active_nontrivial_modes"] for row in rows},
            expected_active,
        )
        self.assertTrue(all(row["tau_restricts_to_zero"] for row in rows))
        self.assertTrue(all(row["proved_nonnegative"] for row in rows))

    def test_stored_complete_boundary_artifact_replays_exactly(self) -> None:
        verification = verify_report(
            BOUNDARY_ARTIFACT,
            coefficient_dir=COEFFICIENT_DIR,
            determinant_artifact=DETERMINANT_ARTIFACT,
            endpoint_artifact=ENDPOINT_ARTIFACT,
        )
        self.assertIs(verification["verified"], True)
        self.assertEqual(verification["failures"], [])
        self.assertEqual(verification["replayed_square_face_count"], 9)
        self.assertEqual(verification["replayed_coordinate_face_count"], 10)
        self.assertIs(verification["complete_coordinate_boundary_proved"], True)
        self.assertIs(verification["global_determinant_interior_proved"], False)


if __name__ == "__main__":
    unittest.main()
