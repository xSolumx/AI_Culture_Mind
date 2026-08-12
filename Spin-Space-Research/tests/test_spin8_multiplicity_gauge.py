"""Contracts for the repeated-view multiplicity gauge theorem."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from spin8_multiplicity_gauge import exact_multiplicity_gauge_certificate


class Spin8MultiplicityGaugeTests(unittest.TestCase):
    def test_exact_certificate(self) -> None:
        report = exact_multiplicity_gauge_certificate()
        self.assertTrue(report["passed"])
        self.assertEqual(report["example_pair_correlation"], "3/13")
        self.assertEqual(report["orthogonalized_inner_product"], "0")
        self.assertEqual(report["orthogonalized_norm_squares"], ["16/13", "10/13"])
        self.assertEqual(
            report["same_view_projector_identity_verified"], [True, True, True]
        )

    def test_published_artifact_matches_fresh_certificate(self) -> None:
        artifact = (
            Path(__file__).parents[1]
            / "artifacts"
            / "spin8_multiplicity_gauge_20260806.json"
        )
        stored = json.loads(artifact.read_text(encoding="utf-8"))
        self.assertEqual(stored, exact_multiplicity_gauge_certificate())


if __name__ == "__main__":
    unittest.main()
