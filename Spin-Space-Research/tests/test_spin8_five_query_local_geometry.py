"""Tests for the exact five-query local-geometry certificate."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from spin8_five_query_local_geometry import run


class Spin8FiveQueryLocalGeometryTests(unittest.TestCase):
    def test_exact_hessian_and_finite_circle_atlas(self) -> None:
        report = run()
        self.assertTrue(report["passed"])

        hessian = report["riemannian_hessian"]
        self.assertEqual(hessian["hessian_nullity"], 28)
        self.assertEqual(hessian["shared_spin8_orbit_tangent_rank"], 28)
        self.assertTrue(hessian["kernel_equals_shared_spin8_orbit_by_rank"])
        self.assertEqual(hessian["quotient_dimension"], 7)
        self.assertTrue(hessian["quotient_hessian_is_negative_definite"])

        circles = report["coordinate_great_circle_atlas"]
        self.assertEqual(circles["curve_count"], 35)
        self.assertEqual(circles["determinant_class_count"], 4)
        self.assertEqual(circles["flat_orbit_curve_count"], 15)
        self.assertEqual(circles["strictly_decreasing_nonorbit_curve_count"], 20)
        self.assertEqual(circles["nonorbit_boundary_rank"], 25)

    def test_stored_artifact_replays_exactly(self) -> None:
        artifact = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "spin8_five_query_local_geometry_20260806.json"
        )
        self.assertEqual(json.loads(artifact.read_text(encoding="utf-8")), run())


if __name__ == "__main__":
    unittest.main()
