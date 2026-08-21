import hashlib
import json
import unittest
from pathlib import Path

from spin8_mixed_closure_so8_theorem import certificate

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "spin8_mixed_closure_so8_theorem_20260821.json"
EXPECTED_ARTIFACT_SHA256 = "5962ba103bbea7115b85095da30ef64375c5af8a86f22d3e3699df4aaeb9e6fb"


class Spin8MixedClosureSO8TheoremTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = certificate()
        cls.stored = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_exact_replay_matches_artifact(self) -> None:
        self.assertEqual(self.report, self.stored)
        self.assertTrue(self.report["passed"])

    def test_exact_topological_closure_conclusion(self) -> None:
        conclusion = self.report["proved_conclusion"]
        self.assertEqual(
            conclusion["statement"],
            "the topological closure of the stated mixed monomial--golden subgroup is SO(8)",
        )
        self.assertEqual(
            conclusion["equivalent_statement"],
            "the stated mixed subgroup is dense in SO(8)",
        )

    def test_irreducible_grade_decomposition_forces_full_lie_algebra(self) -> None:
        decomposition = self.report["clifford_grade_decomposition"]
        self.assertEqual(decomposition["grade_one_dimension"], 7)
        self.assertEqual(decomposition["grade_two_dimension"], 21)
        self.assertEqual(decomposition["combined_dimension"], 28)
        character = decomposition["character_orthogonality"]
        self.assertEqual(character["grade_one_character_inner_product"], 1)
        self.assertEqual(character["grade_two_character_inner_product"], 1)
        self.assertEqual(character["cross_character_inner_product"], 0)
        self.assertEqual(
            self.report["grade_two_non_normalization_witness"]
            ["grade_one_frobenius_coefficients"][0],
            "-1/4",
        )
        self.assertTrue(
            self.report["checks"]["grade_decomposition_is_frobenius_orthogonal"]
        )

    def test_artifact_hash(self) -> None:
        self.assertEqual(
            hashlib.sha256(ARTIFACT.read_bytes()).hexdigest(),
            EXPECTED_ARTIFACT_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
