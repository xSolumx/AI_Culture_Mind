"""Replay contracts for the first reconstructed two-edge Walsh sector."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import sympy as sp

from spin8_dirac_star import rational_circle
from spin8_dirac_two_edge_reconstruct import (
    VARIABLE_ORDER,
    _point_worker,
    _target_setup,
    verify_coefficient_report,
    verify_comparison_report,
    verify_face_report,
    verify_factor_report,
    verify_holdout_report,
)


class Spin8DiracTwoEdgeReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).parents[1]
        cls.coefficients = json.loads(
            (
                root
                / "artifacts/spin8_dirac_two_edge_sector_110101_coefficients_20260806.json"
            ).read_text(encoding="utf-8")
        )
        cls.comparison = json.loads(
            (
                root / "artifacts/spin8_dirac_two_edge_sector_110101_20260806.json"
            ).read_text(encoding="utf-8")
        )
        cls.holdouts = json.loads(
            (
                root
                / "artifacts/spin8_dirac_two_edge_sector_110101_holdouts_20260806.json"
            ).read_text(encoding="utf-8")
        )
        cls.factor = json.loads(
            (
                root
                / "artifacts/spin8_dirac_two_edge_sector_110101_factor_20260806.json"
            ).read_text(encoding="utf-8")
        )
        cls.faces = json.loads(
            (
                root
                / "artifacts/spin8_dirac_two_edge_sector_110101_faces_20260806.json"
            ).read_text(encoding="utf-8")
        )

    def test_published_reports_replay_without_trusting_pass_flags(self) -> None:
        self.assertTrue(verify_coefficient_report(self.coefficients))
        self.assertTrue(verify_comparison_report(self.comparison, self.coefficients))
        self.assertTrue(verify_holdout_report(self.holdouts, self.coefficients))
        self.assertTrue(verify_factor_report(self.factor, self.coefficients))
        self.assertTrue(verify_face_report(self.faces, self.coefficients))

    def test_exact_factor_has_the_published_shape(self) -> None:
        self.assertEqual(self.factor["exact_factor"], "1-a2")
        self.assertEqual(self.factor["source_nonzero_coefficient_count"], 243)
        self.assertEqual(self.factor["source_multidegree"], [2, 2, 2, 2, 1, 1])
        self.assertEqual(self.factor["quotient_nonzero_coefficient_count"], 162)
        self.assertEqual(self.factor["quotient_multidegree"], [1, 2, 2, 2, 1, 1])
        nested = self.factor["nested_boundary_decomposition"]
        self.assertEqual(nested["geometric_late_boundary_factor"], "D^2 E^2 G^2")
        self.assertEqual(nested["base_nonzero_coefficient_count"], 42)
        self.assertEqual(nested["correction_nonzero_coefficient_count"], 28)
        self.assertEqual(nested["correction_multidegree"], [1, 1, 1, 1, 1, 1])
        self.assertTrue(nested["compact_formula_matches"])

    def test_opposite_signed_face_identities_are_explicit(self) -> None:
        self.assertEqual(self.faces["d2_equals_one_identity"], "Q=3(a2-1)(g2-1)^2")
        self.assertEqual(
            self.faces["g2_equals_one_identity"],
            "Q=3(a2-1)(d2-1)^2(e2-1)(3e2+1)",
        )
        self.assertEqual(
            self.faces["unit_cube_signs"],
            {"d2_equals_one": "nonpositive", "g2_equals_one": "nonnegative"},
        )

    def test_one_fresh_exact_point_matches_direct_determinants(self) -> None:
        parameters = tuple(sp.Rational(2 + axis, 41 + 2 * axis) for axis in range(6))
        signs, weights, complement_mask, _degree = _target_setup()
        _, observed_text = _point_worker(
            ((0,) * 6, parameters, signs, weights, complement_mask)
        )
        squared = tuple(rational_circle(value)[0] ** 2 for value in parameters)
        predicted = sum(
            sp.Rational(row["coefficient"])
            * sp.prod(squared[axis] ** int(row["powers"][axis]) for axis in range(6))
            for row in self.coefficients["coefficient_rows"]
        )
        self.assertEqual(sp.Rational(observed_text), sp.factor(predicted))
        self.assertEqual(tuple(self.coefficients["variable_order"]), VARIABLE_ORDER)


if __name__ == "__main__":
    unittest.main()
