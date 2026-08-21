import hashlib
import json
import unittest
from pathlib import Path

from mixed_monomial_golden_macro_compiler import certificate

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT / "artifacts" / "mixed_monomial_golden_macro_compiler_20260817.json"
)
EXPECTED_ARTIFACT_SHA256 = (
    "9595578918bd46387ce5773607d1ded3d6117b11cb2b2353b586a1c6fc0cd438"
)


class MixedMonomialGoldenMacroCompilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = certificate()
        cls.stored = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_exact_replay_matches_artifact(self) -> None:
        self.assertEqual(self.report, self.stored)
        self.assertTrue(self.report["passed"])

    def test_exact_dictionary_sizes_and_multiplicities(self) -> None:
        vector = self.report["views"]["vector"]
        self.assertEqual(vector["labelled_triple_count"], 867)
        self.assertEqual(vector["distinct_exact_matrix_count"], 530)
        self.assertEqual(
            vector["multiplicity_histogram"],
            {"1": 274, "2": 223, "3": 1, "4": 30, "12": 2},
        )

        expected_half_histogram = {
            "1": 148,
            "2": 116,
            "4": 82,
            "8": 38,
            "9": 4,
            "12": 2,
            "13": 2,
            "29": 2,
        }
        for view in ("positive_half_spin", "negative_half_spin"):
            with self.subTest(view=view):
                report = self.report["views"][view]
                self.assertEqual(report["labelled_triple_count"], 1156)
                self.assertEqual(report["distinct_exact_matrix_count"], 394)
                self.assertEqual(
                    report["multiplicity_histogram"], expected_half_histogram
                )

    def test_compiled_measure_preserves_the_exact_sandwich_operator(self) -> None:
        for view, report in self.report["views"].items():
            with self.subTest(view=view):
                checks = report["checks"]
                self.assertTrue(
                    checks["weighted_dictionary_is_inverse_symmetric"]
                )
                self.assertTrue(
                    checks["weighted_dictionary_mean_equals_MN_MH_MN"]
                )
                self.assertTrue(
                    checks["multiplicities_sum_to_all_labelled_triples"]
                )

    def test_compiled_tables_fit_the_declared_small_memory_envelope(self) -> None:
        vector = self.report["views"]["vector"]["runtime_storage_bytes"]
        self.assertEqual(vector["float32_table_plus_uint16_lookup"], 137414)
        for view in ("positive_half_spin", "negative_half_spin"):
            storage = self.report["views"][view]["runtime_storage_bytes"]
            self.assertEqual(storage["float32_table_plus_uint16_lookup"], 103176)

    def test_uniform_distinct_sampling_is_explicitly_excluded(self) -> None:
        nonclaims = self.report["claim_scope"]["not_claimed"]
        self.assertIn(
            "uniform sampling over distinct matrices has the certified sandwich spectrum",
            nonclaims,
        )

    def test_artifact_hash(self) -> None:
        digest = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_ARTIFACT_SHA256)


if __name__ == "__main__":
    unittest.main()
