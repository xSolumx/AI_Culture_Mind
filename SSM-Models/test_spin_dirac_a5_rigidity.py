from __future__ import annotations

import json
import unittest
from pathlib import Path

from spin_dirac_a5_ladder import LADDER_DIMENSIONS
from spin_dirac_a5_rigidity import (
    CERTIFICATE_PRIMES,
    exact_binary_icosahedral_certificate,
    stage_diagnostics,
)

ARTIFACT = (
    Path(__file__).parent
    / "experiments"
    / "artifacts"
    / "spin_dirac_a5_rigidity_20260816.json"
)


class SpinDiracA5RigidityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binary_group = exact_binary_icosahedral_certificate()
        cls.live_stages = {
            dimension: stage_diagnostics(dimension) for dimension in (3, 8)
        }

    def test_binary_icosahedral_group_is_exact_over_q_sqrt5(self) -> None:
        self.assertTrue(self.binary_group["passed"])
        self.assertEqual(self.binary_group["binary_group_order"], 120)
        self.assertEqual(self.binary_group["projective_group_order"], 60)
        self.assertTrue(all(self.binary_group["checks"].values()))

    def test_live_exact_relation_and_conjugacy_certificates(self) -> None:
        expected = {
            3: {"centralizer": 0, "orbit": 3, "relation_rank": 3},
            8: {"centralizer": 10, "orbit": 18, "relation_rank": 38},
        }
        for dimension, stage in self.live_stages.items():
            self.assertTrue(stage["passed"])
            self.assertTrue(all(stage["exact_checks"].values()))
            self.assertEqual(stage["h1_dimension"], 0)
            self.assertEqual(
                stage["expected_centralizer_dimension"],
                expected[dimension]["centralizer"],
            )
            self.assertEqual(
                stage["certified_conjugacy_rank"], expected[dimension]["orbit"]
            )
            self.assertEqual(
                stage["certified_relation_rank"],
                expected[dimension]["relation_rank"],
            )
            for family in (
                stage["relation_rank_certificates"],
                stage["conjugacy_rank_certificates"],
            ):
                self.assertEqual(
                    [certificate["prime"] for certificate in family],
                    list(CERTIFICATE_PRIMES),
                )
                for certificate in family:
                    self.assertNotEqual(
                        certificate["pivot_minor_determinant_mod_prime"], 0
                    )
                    self.assertEqual(
                        len(certificate["pivot_rows"]), certificate["rank"]
                    )
                    self.assertEqual(
                        len(certificate["pivot_columns"]), certificate["rank"]
                    )

    def test_full_ladder_artifact_records_zero_h1(self) -> None:
        report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])
        self.assertEqual(tuple(report["ladder"]), LADDER_DIMENSIONS)
        self.assertTrue(report["exact_binary_icosahedral_group"]["passed"])
        expected_relation_ranks = {3: 3, 8: 38, 9: 51, 10: 66, 11: 83, 12: 102}
        for dimension, relation_rank in expected_relation_ranks.items():
            stage = report["stages"][str(dimension)]
            self.assertTrue(stage["passed"])
            self.assertEqual(stage["h1_dimension"], 0)
            self.assertEqual(stage["certified_relation_rank"], relation_rank)
            self.assertEqual(
                stage["certified_relation_kernel_dimension"],
                stage["certified_conjugacy_rank"],
            )


if __name__ == "__main__":
    unittest.main()
