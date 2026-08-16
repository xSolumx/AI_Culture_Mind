"""Integrity checks for the frozen octonion operator audit artifact."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


class OctonionOperatorAuditTests(unittest.TestCase):
    ARTIFACT = (
        Path(__file__).resolve().parent
        / "experiments"
        / "artifacts"
        / "octonion_operator_scan_rtx2070s_20260816.json"
    )
    SHA256 = "992ca268f50e366b1a024c9e7ac63814f981312219097c178039b31d2e7f1830"

    def test_frozen_artifact_hash_and_required_gates(self) -> None:
        payload = self.ARTIFACT.read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), self.SHA256)
        report = json.loads(payload)
        self.assertTrue(report["all_required_checks_passed"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["streaming_cache_scalars_per_lane"], 8)
        self.assertEqual(report["parallel_operator_matrix_scalars_per_lane"], 64)
        self.assertEqual(
            report["parallel_homogeneous_affine_matrix_scalars_per_lane"], 81
        )

    def test_exact_lie_and_associator_certificates_are_present(self) -> None:
        report = json.loads(self.ARTIFACT.read_text())
        algebra = report["algebra"]
        self.assertEqual(algebra["left_lie_coordinate_rank"], 28)
        self.assertEqual(algebra["left_lie_coordinate_determinant"], str(-(2**49)))
        self.assertEqual(algebra["e1_e2_e4_associator_norm"], 2.0)
        self.assertEqual(algebra["operator_collapse_frobenius_discrepancy"], 4.0)

    def test_systems_numbers_match_frozen_protocol(self) -> None:
        report = json.loads(self.ARTIFACT.read_text())
        systems = report["systems"]
        self.assertEqual(systems["length"], 4096)
        self.assertEqual(systems["batch"], 8)
        self.assertEqual(systems["lanes"], 4)
        self.assertEqual(systems["composition_counts"]["work_efficient"], 12286)
        self.assertEqual(systems["composition_counts"]["hillis_steele"], 45057)
        self.assertGreater(
            systems["hillis_over_work_efficient_prebuilt_median_ratio"], 3.0
        )


if __name__ == "__main__":
    unittest.main()
