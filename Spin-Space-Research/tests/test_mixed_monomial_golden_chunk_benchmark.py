import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT / "artifacts" / "mixed_monomial_golden_chunk_benchmark_20260817.json"
)
EXPECTED_ARTIFACT_SHA256 = (
    "35aef11f6e2577ac5848c800d5afb1dcbd38def2db3dcb5f3deb4dc820793f74"
)
EXPECTED_COMPILER_SHA256 = (
    "ed1ae7e8ac98c5e037be4e45d10f22ec3236e7d6f8337fbc2b9f9a499e13e5de"
)


class MixedMonomialGoldenChunkBenchmarkArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_artifact_is_linked_to_the_exact_prefix_compiler(self) -> None:
        self.assertEqual(
            self.report["exact_chunk_compiler_artifact_sha256"],
            EXPECTED_COMPILER_SHA256,
        )
        self.assertTrue(self.report["passed"])

    def test_complete_every_prefix_grid_is_present(self) -> None:
        self.assertEqual(len(self.report["results"]), 6)
        for report in self.report["results"]:
            self.assertEqual(len(report["results"]), 12)
            self.assertEqual(report["prefix_table_shape"][-2:], [24, 8])
        self.assertTrue(self.report["checks"]["all_result_grids_are_complete"])

    def test_compiled_path_won_every_recorded_cell(self) -> None:
        for report in self.report["results"]:
            for row in report["results"]:
                with self.subTest(
                    view=report["view"],
                    device=report["device"],
                    batch=row["batch_size"],
                    chunks=row["chunk_count"],
                ):
                    self.assertGreater(
                        row["endpoint_only"]["compiled_speedup_vs_online"], 1
                    )
                    self.assertGreater(
                        row["every_prefix"]["compiled_speedup_vs_online"], 1
                    )

    def test_recurrent_float32_parity_passed_through_length_192(self) -> None:
        self.assertEqual(self.report["settings"]["primitive_lengths"][-1], 192)
        self.assertTrue(
            self.report["checks"]["all_float32_parity_checks_passed"]
        )
        maximum_error = max(
            row[section]["max_abs_error"]
            for report in self.report["results"]
            for row in report["results"]
            for section in ("endpoint_only", "every_prefix")
        )
        self.assertLessEqual(
            maximum_error,
            self.report["settings"]["parity_tolerance_max_abs"],
        )
        self.assertEqual(self.report["settings"]["torch_cpu_threads"], 1)
        self.assertTrue(
            self.report["settings"]["cuda_synchronized_each_complete_call"]
        )

    def test_parallel_scan_and_training_are_not_promoted(self) -> None:
        nonclaims = self.report["claim_scope"]["not_claimed"]
        self.assertIn(
            "a parallel prefix-scan kernel has been implemented or benchmarked",
            nonclaims,
        )
        self.assertIn("backward or training throughput improves", nonclaims)

    def test_artifact_hash(self) -> None:
        digest = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_ARTIFACT_SHA256)


if __name__ == "__main__":
    unittest.main()
