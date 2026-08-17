import hashlib
import json
import unittest
from pathlib import Path

from mixed_monomial_golden_closure import certificate

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "mixed_monomial_golden_closure_20260817.json"
EXPECTED_ARTIFACT_SHA256 = (
    "4d6840bd89a0c58ac086c308e88f823928cc4d3206450f24bcc035f499d6d8c7"
)


class MixedMonomialGoldenClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = certificate()
        cls.stored = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_exact_replay_matches_artifact(self) -> None:
        self.assertEqual(self.report, self.stored)
        self.assertTrue(self.report["passed"])

    def test_source_groups_are_reconstructed_in_the_same_basis(self) -> None:
        self.assertEqual(self.report["monomial_group"]["order"], 21_504)
        self.assertEqual(
            self.report["monomial_group"]["structure"],
            "2_+^(1+6):PSL(2,7)",
        )
        self.assertEqual(self.report["views"]["vector"]["source_subgroup_order"], 60)
        self.assertEqual(
            self.report["views"]["positive_half_spin"]["source_subgroup_order"],
            120,
        )
        self.assertEqual(
            self.report["views"]["negative_half_spin"]["source_subgroup_order"],
            120,
        )

    def test_symmetric_length_two_words_are_exhausted_exactly(self) -> None:
        expected_counts = {
            "vector": 51,
            "positive_half_spin": 68,
            "negative_half_spin": 68,
        }
        for view, expected in expected_counts.items():
            with self.subTest(view=view):
                screen = self.report["views"][view]["length_two_screen"]
                self.assertEqual(screen["mixed_products_checked"], expected)
                self.assertEqual(screen["products_without_order_at_most_120"], 0)
                self.assertTrue(
                    screen["checks"][
                        "every_symmetric_length_two_mixed_word_has_exact_finite_order"
                    ]
                )

    def test_same_length_three_word_proves_all_three_closures_infinite(self) -> None:
        for view, report in self.report["views"].items():
            with self.subTest(view=view):
                witness = report["infinite_order_witness"]
                characteristic = witness["characteristic"]
                self.assertEqual(witness["word"], ["FanoA", "FanoB", "b"])
                self.assertEqual(witness["word_length"], 3)
                self.assertEqual(
                    characteristic["first_nonintegral_coefficient"],
                    "1/4 - sqrt(5)/4",
                )
                self.assertEqual(
                    characteristic[
                        "characteristic_norm_coefficient_denominators"
                    ],
                    [1, 2, 4],
                )
                self.assertTrue(report["mixed_closure_is_infinite"])
                self.assertTrue(all(witness["checks"].values()))

    def test_clifford_adjoint_gate_proves_so8_density(self) -> None:
        adjoint = self.report["clifford_adjoint_density"]
        self.assertEqual(adjoint["grade_one_dimension"], 7)
        self.assertEqual(adjoint["grade_two_dimension"], 21)
        self.assertEqual(adjoint["union_dimension"], 28)
        self.assertEqual(
            adjoint["grade_one_centralizer"]["centralizer_dimension"], 1
        )
        self.assertEqual(
            adjoint["grade_two_centralizer"]["centralizer_dimension"], 1
        )
        self.assertEqual(adjoint["full_centralizer"]["centralizer_dimension"], 2)
        self.assertEqual(adjoint["grade_one_bracket_span_dimension"], 21)
        self.assertTrue(adjoint["passed"])
        for view, report in self.report["views"].items():
            with self.subTest(view=view):
                self.assertEqual(report["topological_closure"], "SO(8)")
                self.assertTrue(report["mixed_group_is_topologically_dense_in_SO8"])
                self.assertTrue(
                    adjoint["view_normalizer_failures"][view][
                        "golden_b_does_not_normalize_grade_two"
                    ]
                )

    def test_density_does_not_promote_rate_or_model_claims(self) -> None:
        nonclaims = self.report["claim_scope"]["not_claimed"]
        self.assertIn("a quantitative spectral gap or equidistribution rate", nonclaims)
        self.assertIn("an ML-quality or kernel-speed advantage", nonclaims)

    def test_artifact_hash(self) -> None:
        digest = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_ARTIFACT_SHA256)


if __name__ == "__main__":
    unittest.main()
