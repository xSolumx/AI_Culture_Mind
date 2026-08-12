"""Tests for the exact D4/24-cell and triality-projector bridge."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from spin8_d4_24cell_bridge import run


class Spin8D4BridgeTests(unittest.TestCase):
    def test_exact_bridge_and_non_equivalence_certificate(self) -> None:
        report = run()
        self.assertTrue(report["passed"])

        weights = report["minuscule_weight_24cell"]
        self.assertEqual(weights["orbit_sizes"], {"8v": 8, "8s_plus": 8, "8s_minus": 8})
        self.assertEqual(weights["union_size"], 24)
        self.assertTrue(weights["triality_cycle_exact"])
        self.assertEqual(
            weights["spherical_monomials_checked_through_degree_five"], 126
        )
        self.assertEqual(weights["spherical_five_design_failures"], [])

        projectors = report["coordinate_sensor_projectors"]
        self.assertTrue(projectors["per_view_tight_fusion_frame_exact"])
        self.assertTrue(projectors["all_views_tight_fusion_frame_exact"])
        self.assertTrue(
            projectors["same_view_distinct_subspaces_intersect_in_one_line"]
        )
        self.assertTrue(projectors["cross_view_isoclinic_exact"])
        self.assertEqual(projectors["cross_view_squared_cosine"], "1/4")
        self.assertTrue(
            projectors["continuous_nonvertex_deformation"]["deformed_view_sum_exact"]
        )
        packing = projectors["standard_grassmannian_packing_audit"]
        self.assertFalse(packing["spectrally_optimal"])
        self.assertFalse(packing["meets_chordal_simplex_bound"])
        self.assertEqual(packing["chordal_optimality_status"], "open from this audit")

    def test_stored_artifact_replays_exactly(self) -> None:
        artifact = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "spin8_d4_24cell_bridge_20260806.json"
        )
        self.assertEqual(json.loads(artifact.read_text(encoding="utf-8")), run())


if __name__ == "__main__":
    unittest.main()
