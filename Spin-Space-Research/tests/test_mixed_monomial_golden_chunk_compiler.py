import hashlib
import json
import unittest
from pathlib import Path

from mixed_monomial_golden_chunk_compiler import certificate

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT / "artifacts" / "mixed_monomial_golden_chunk_compiler_20260817.json"
)
EXPECTED_ARTIFACT_SHA256 = (
    "ed1ae7e8ac98c5e037be4e45d10f22ec3236e7d6f8337fbc2b9f9a499e13e5de"
)


class MixedMonomialGoldenChunkCompilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = certificate()
        cls.stored = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_exact_replay_matches_artifact(self) -> None:
        self.assertEqual(self.report, self.stored)
        self.assertTrue(self.report["passed"])

    def test_all_three_causal_blocks_are_exact(self) -> None:
        for view, report in self.report["views"].items():
            with self.subTest(view=view):
                self.assertEqual(report["prefix_operator_shape"], [24, 8])
                self.assertTrue(
                    report["causal_contract"]["emits_every_primitive_prefix"]
                )
                self.assertEqual(
                    report["causal_contract"]["primitive_application_order"],
                    ["right", "middle", "left"],
                )
                self.assertTrue(report["checks"]["first_block_is_right_action"])
                self.assertTrue(
                    report["checks"]["second_block_is_middle_times_right"]
                )
                self.assertTrue(
                    report["checks"][
                        "third_block_matches_exact_macro_endpoint"
                    ]
                )

    def test_prefix_tables_are_unique_and_below_one_megabyte(self) -> None:
        expected = {
            "vector": (867, 665856),
            "positive_half_spin": (1156, 887808),
            "negative_half_spin": (1156, 887808),
        }
        for view, (count, byte_count) in expected.items():
            with self.subTest(view=view):
                report = self.report["views"][view]
                self.assertEqual(report["labelled_triple_count"], count)
                self.assertEqual(
                    report["distinct_exact_prefix_operator_count"], count
                )
                self.assertEqual(
                    report["runtime_storage_bytes"][
                        "float32_full_labelled_prefix_table"
                    ],
                    byte_count,
                )
                self.assertLess(byte_count, 1_000_000)

    def test_continuous_transition_generalization_remains_open(self) -> None:
        self.assertIn(
            "the table applies to arbitrary learned continuous transitions",
            self.report["claim_scope"]["not_claimed"],
        )

    def test_artifact_hash(self) -> None:
        digest = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_ARTIFACT_SHA256)


if __name__ == "__main__":
    unittest.main()
