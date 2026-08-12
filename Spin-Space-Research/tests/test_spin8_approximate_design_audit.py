"""Tests for the exact weighted experimental-design correction."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from spin8_approximate_design_audit import run


class Spin8ApproximateDesignAuditTests(unittest.TestCase):
    def test_exact_domain_separation_and_reweighting_witness(self) -> None:
        report = run()
        self.assertTrue(report["passed"])
        balanced = report["balanced_information"]
        self.assertFalse(balanced["passes_approximate_D_optimality_criterion"])
        self.assertEqual(balanced["maximum_sensitivity_for_normalized_I_over_5"], "75")
        self.assertGreater(
            report["exact_reweighting_counterexample"]["relative_gain_decimal"], 0
        )
        boundaries = report["exact_reweighting_counterexample"][
            "weight_simplex_boundaries"
        ]
        self.assertEqual(boundaries["alpha_zero_rank"], 25)
        self.assertEqual(boundaries["alpha_five_rank"], 7)
        self.assertTrue(
            report["global_approximate_design"]["kiefer_wolfowitz_saturated"]
        )
        self.assertTrue(
            report["global_approximate_design"][
                "unit_probe_trace_seven_exact_by_polarization"
            ]
        )
        self.assertIn("classical_global_input", report["proof_layers"])
        self.assertIn(
            "Kiefer--Wolfowitz",
            report["proof_layers"]["classical_global_input"],
        )

    def test_stored_artifact_replays_exactly(self) -> None:
        artifact = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "spin8_approximate_design_audit_20260806.json"
        )
        self.assertEqual(json.loads(artifact.read_text(encoding="utf-8")), run())


if __name__ == "__main__":
    unittest.main()
