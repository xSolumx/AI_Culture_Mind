from __future__ import annotations

import json
import unittest
from pathlib import Path

from spin_dirac_a5_cohomology import diagnostics

ARTIFACTS = Path(__file__).parent / "experiments" / "artifacts"
COHOMOLOGY_ARTIFACT = ARTIFACTS / "spin_dirac_a5_cohomology_20260816.json"
RIGIDITY_ARTIFACT = ARTIFACTS / "spin_dirac_a5_rigidity_20260816.json"


class SpinDiracA5CohomologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.live = diagnostics()

    def test_exact_group_and_low_degree_contraction(self) -> None:
        report = self.live
        self.assertTrue(report["passed"])
        self.assertTrue(report["group"]["passed"])
        self.assertEqual(report["group"]["order"], 120)
        self.assertTrue(all(report["group"]["checks"].values()))
        for name, outputs in (("degree_one", 120), ("degree_two", 14_400)):
            degree = report["contraction"][name]
            self.assertTrue(degree["passed"])
            self.assertEqual(degree["outputs_checked"], outputs)
            self.assertEqual(degree["maximum_unexpected_terms"], 0)
            self.assertEqual(degree["observed_identity_coefficient_range"], [120, 120])
        self.assertEqual(
            report["consequences"]["h1_dimension_for_every_linear_module"], 0
        )
        self.assertEqual(
            report["consequences"]["h2_dimension_for_every_linear_module"], 0
        )

    def test_artifact_matches_live_group_and_contraction(self) -> None:
        artifact = json.loads(COHOMOLOGY_ARTIFACT.read_text(encoding="utf-8"))
        self.assertTrue(artifact["passed"])
        self.assertEqual(
            artifact["group"]["multiplication_table_sha256"],
            self.live["group"]["multiplication_table_sha256"],
        )
        self.assertEqual(artifact["contraction"], self.live["contraction"])

    def test_raw_presentation_cokernel_is_not_mislabeled_h2(self) -> None:
        rigidity = json.loads(RIGIDITY_ARTIFACT.read_text(encoding="utf-8"))
        self.assertTrue(rigidity["passed"])
        for stage in rigidity["stages"].values():
            raw_relation_cokernel = (
                3 * stage["lie_dimension"] - stage["certified_relation_rank"]
            )
            self.assertGreater(raw_relation_cokernel, 0)
        self.assertEqual(
            self.live["consequences"]["h2_dimension_for_every_linear_module"], 0
        )


if __name__ == "__main__":
    unittest.main()
