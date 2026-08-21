import hashlib
import json
import unittest
from pathlib import Path

from mixed_monomial_golden_mixing import certificate

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "mixed_monomial_golden_mixing_20260817.json"
EXPECTED_ARTIFACT_SHA256 = (
    "0082b9df621dfe4b14c41a26cc914f124cebde37abb38974ed79c19411f7eac9"
)


class MixedMonomialGoldenMixingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = certificate()
        cls.stored = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_exact_replay_matches_artifact(self) -> None:
        self.assertEqual(self.report, self.stored)
        self.assertTrue(self.report["passed"])

    def test_symmetric_alphabets_preserve_label_multiplicity(self) -> None:
        vector = self.report["views"]["vector"]
        self.assertEqual(vector["symmetric_alphabet_size"], 20)
        self.assertEqual(vector["distinct_matrix_count"], 20)
        for view in ("positive_half_spin", "negative_half_spin"):
            with self.subTest(view=view):
                report = self.report["views"][view]
                self.assertEqual(report["symmetric_alphabet_size"], 21)
                self.assertEqual(report["distinct_matrix_count"], 19)
                self.assertEqual(report["uniform_label_weight"], "1/21")

    def test_defining_representation_root_certificates(self) -> None:
        vector = self.report["views"]["vector"]["defining_8"]
        self.assertEqual(vector["rational_norm_degree"], 16)
        self.assertEqual(
            vector["rational_norm_roots_strictly_inside_interval"], 16
        )
        self.assertEqual(vector["radius_upper_bound"], "7/25")
        self.assertEqual(vector["gap_lower_bound"], "18/25")
        self.assertTrue(vector["inequalities_are_strict"])

        for view in ("positive_half_spin", "negative_half_spin"):
            with self.subTest(view=view):
                defining = self.report["views"][view]["defining_8"]
                self.assertTrue(defining["characteristic_factorization_matches"])
                self.assertEqual(
                    defining["quintic_roots_strictly_inside_interval"], 5
                )
                self.assertEqual(defining["radius"], "4/21")
                self.assertEqual(defining["gap"], "17/21")
                self.assertTrue(defining["radius_is_attained_by_linear_factor"])

    def test_exact_ldlt_bounds_in_dimensions_28_and_35(self) -> None:
        expected = {
            "vector": {"adjoint_28": ("5/8", "3/8"), "traceless_symmetric_35": ("1/3", "2/3")},
            "positive_half_spin": {"adjoint_28": ("3/4", "1/4"), "traceless_symmetric_35": ("3/8", "5/8")},
            "negative_half_spin": {"adjoint_28": ("3/4", "1/4"), "traceless_symmetric_35": ("3/8", "5/8")},
        }
        for view, representations in expected.items():
            for representation, (radius, gap) in representations.items():
                with self.subTest(view=view, representation=representation):
                    report = self.report["views"][view][representation]
                    self.assertEqual(report["radius_upper_bound"], radius)
                    self.assertEqual(report["gap_lower_bound"], gap)
                    self.assertTrue(report["mean_is_self_adjoint_in_metric"])
                    self.assertTrue(
                        report["positive_definite_cG_plus_GM"]["passed"]
                    )
                    self.assertTrue(
                        report["positive_definite_cG_minus_GM"]["passed"]
                    )

    def test_claim_scope_does_not_promote_low_degree_bounds(self) -> None:
        nonclaims = self.report["claim_scope"]["not_claimed"]
        self.assertIn(
            "a spectral gap on the full mean-zero L2(SO(8))", nonclaims
        )
        self.assertIn(
            "an SSM accuracy, efficiency, or hardware-speed advantage",
            nonclaims,
        )

    def test_artifact_hash(self) -> None:
        digest = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_ARTIFACT_SHA256)


if __name__ == "__main__":
    unittest.main()
