from __future__ import annotations

import json
import unittest

from spin9_v1_v5_boundary_char0 import DEFAULT_OUTPUT, certificate


class Spin9V1V5BoundaryCharacteristicZeroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = certificate(workers=4)
        cls.artifact = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))

    def test_raw_boundary_identities_replay(self) -> None:
        self.assertTrue(self.report["passed"], self.report)
        self.assertEqual(
            self.report["base_information_block_determinants"],
            [65_536, 262_144],
        )
        families = {row["family"]: row for row in self.report["families"]}
        self.assertEqual(set(families), {"A", "B"})
        self.assertEqual(
            families["A"]["raw_block_determinant_rows_sha256"],
            "f761c1c838eaf7095b1df235208523b88bf3180b53d546af3007b4d507b91703",
        )
        self.assertEqual(
            families["B"]["raw_block_determinant_rows_sha256"],
            "739a26fe38a6aa61dea42172948d76ac01169ea9ff59a3078c7362aabc675c1b",
        )
        for row in families.values():
            self.assertEqual(row["newton_grid_node_count"], 2701)
            self.assertEqual(row["determinant_identity_total_degree_bound"], 72)
            self.assertTrue(all(row["structural_checks"].values()))
            self.assertTrue(row["all_newton_grid_nodes_match"])

    def test_artifact_and_claim_boundary(self) -> None:
        self.assertEqual(self.artifact, self.report)
        self.assertFalse(self.report["modular_reconstruction_used"])
        self.assertFalse(self.report["finite_radius_coupled_identity_certified"])
        self.assertFalse(self.report["global_coupled_determinant_theorem_claimed"])


if __name__ == "__main__":
    unittest.main()
