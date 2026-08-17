import hashlib
import json
import unittest
from pathlib import Path

from mixed_monomial_golden_higher_weight import certificate

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT / "artifacts" / "mixed_monomial_golden_higher_weight_20260817.json"
)
EXPECTED_ARTIFACT_SHA256 = (
    "d5b3fb35092d2fae603c546530502a95e04eb31ac1566035ef866fc7e033b1d6"
)


class MixedMonomialGoldenHigherWeightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = certificate()
        cls.stored = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_exact_replay_matches_artifact(self) -> None:
        self.assertEqual(self.report, self.stored)
        self.assertTrue(self.report["passed"])

    def test_monomial_group_has_one_fixed_cayley_line(self) -> None:
        cayley = self.report["monomial_cayley_fixed_line"]
        self.assertEqual(cayley["hodge_sector"], "minus")
        self.assertEqual(cayley["fixed_space_equation_rank"], 34)
        self.assertEqual(cayley["fixed_space_dimension"], 1)
        self.assertEqual(cayley["reduced_basis_norm_squared"], 7)
        self.assertTrue(cayley["fixed_by_all_seventeen_monomial_steps"])
        self.assertTrue(cayley["passed"])

    def test_original_walk_extends_to_the_56_and_both_hodge_35s(self) -> None:
        expected = {
            "vector": {
                "exterior_3_56": "7/25",
                "hodge_plus_35": "1/3",
                "hodge_minus_35": "24/25",
            },
            "positive_half_spin": {
                "exterior_3_56": "1/5",
                "hodge_plus_35": "3/8",
                "hodge_minus_35": "99/100",
            },
            "negative_half_spin": {
                "exterior_3_56": "1/5",
                "hodge_plus_35": "3/8",
                "hodge_minus_35": "99/100",
            },
        }
        for view, representations in expected.items():
            original = self.report["views"][view][
                "original_uniform_label_measure"
            ]
            self.assertTrue(original["all_six_representation_bounds_passed"])
            for name, radius in representations.items():
                with self.subTest(view=view, representation=name):
                    report = original["representations"][name]
                    self.assertEqual(report["radius_upper_bound"], radius)
                    self.assertTrue(report["passed"])

    def test_sparse_rayleigh_witnesses_certify_the_bottleneck(self) -> None:
        vector = self.report["views"]["vector"][
            "bottleneck_rayleigh_witness"
        ]
        self.assertEqual(
            vector["exact_rayleigh_quotient"], "(sqrt(5) + 98)/105"
        )
        self.assertEqual(vector["comparison_threshold"], "19/20")
        self.assertTrue(vector["rayleigh_quotient_exceeds_threshold"])

        for view in ("positive_half_spin", "negative_half_spin"):
            with self.subTest(view=view):
                witness = self.report["views"][view][
                    "bottleneck_rayleigh_witness"
                ]
                self.assertEqual(witness["exact_rayleigh_quotient"], "221/224")
                self.assertEqual(
                    witness[
                        "exact_original_band_gap_upper_bound_from_witness"
                    ],
                    "3/224",
                )
                self.assertTrue(witness["passed"])

    def test_compiled_sandwich_improves_the_certified_macro_step_band(self) -> None:
        expected = {
            "vector": ("17/20", "3/20", "> 3", 867),
            "positive_half_spin": ("97/100", "3/100", "> 56/25", 1156),
            "negative_half_spin": ("97/100", "3/100", "> 56/25", 1156),
        }
        for view, (radius, gap, factor, support) in expected.items():
            with self.subTest(view=view):
                report = self.report["views"][view]
                sandwich = report[
                    "monomial_golden_monomial_sandwich_measure"
                ]
                self.assertEqual(sandwich["operator_formula"], "M_N M_H M_N")
                self.assertEqual(sandwich["primitive_word_length"], 3)
                self.assertTrue(sandwich["distribution_is_symmetric"])
                self.assertEqual(sandwich["band_radius_upper_bound"], radius)
                self.assertEqual(sandwich["band_gap_lower_bound"], gap)
                self.assertEqual(
                    sandwich["precompiled_macro_support_upper_bound"], support
                )
                self.assertEqual(
                    report["certified_macro_step_band_gap_improvement_factor"],
                    factor,
                )
                self.assertTrue(
                    sandwich["all_six_representation_bounds_passed"]
                )

    def test_macro_claim_does_not_hide_primitive_cost(self) -> None:
        nonclaims = self.report["claim_scope"]["not_claimed"]
        self.assertIn(
            "an improvement per primitive matrix multiplication; one sandwich macro consumes three primitive letters unless precompiled",
            nonclaims,
        )
        self.assertIn(
            "a spectral gap on the full mean-zero L2(SO(8))", nonclaims
        )

    def test_artifact_hash(self) -> None:
        digest = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_ARTIFACT_SHA256)


if __name__ == "__main__":
    unittest.main()
