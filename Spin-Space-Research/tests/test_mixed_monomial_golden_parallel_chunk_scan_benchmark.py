import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "artifacts"
    / "mixed_monomial_golden_parallel_chunk_scan_benchmark_20260817.json"
)
EXPECTED_ARTIFACT_SHA256 = (
    "73816cbcf8733ad9e2a4be8a87376ebfa96e369be64b3aa4093b3f0cafa93900"
)
EXPECTED_COMPILER_SHA256 = (
    "ed1ae7e8ac98c5e037be4e45d10f22ec3236e7d6f8337fbc2b9f9a499e13e5de"
)


class MixedMonomialGoldenParallelChunkScanBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_artifact_is_linked_to_the_exact_chunk_compiler(self) -> None:
        self.assertEqual(
            self.report["exact_chunk_compiler_artifact_sha256"],
            EXPECTED_COMPILER_SHA256,
        )
        self.assertTrue(self.report["passed"])

    def test_complete_parallel_grid_and_composition_reduction(self) -> None:
        self.assertEqual(len(self.report["results"]), 6)
        for report in self.report["results"]:
            self.assertEqual(len(report["results"]), 9)
            for row in report["results"]:
                counts = row["composition_counts"]
                self.assertLess(
                    counts["compiled_work_efficient_products"],
                    counts["primitive_work_efficient_products"],
                )
        self.assertTrue(self.report["checks"]["all_result_grids_are_complete"])

    def test_compiled_forward_and_initial_state_backward_won_every_cell(self) -> None:
        for report in self.report["results"]:
            for row in report["results"]:
                with self.subTest(
                    view=report["view"],
                    device=report["device"],
                    batch=row["batch_size"],
                    chunks=row["chunk_count"],
                ):
                    self.assertGreater(
                        row["forward"]["compiled_speedup_vs_primitive"], 1
                    )
                    self.assertGreater(
                        row["forward_plus_initial_state_backward"][
                            "compiled_speedup_vs_primitive"
                        ],
                        1,
                    )
                    self.assertGreaterEqual(
                        row["forward_plus_initial_state_backward"][
                            "compiled_two_stage_scan"
                        ]["repeats"],
                        10,
                    )

    def test_forward_and_initial_state_gradient_parity_passed(self) -> None:
        self.assertTrue(
            self.report["checks"][
                "all_float32_forward_backward_parity_checks_passed"
            ]
        )
        maximum_forward_error = max(
            error
            for report in self.report["results"]
            for row in report["results"]
            for error in (
                row["forward"]["primitive_vs_recurrent_max_abs_error"],
                row["forward"]["compiled_vs_recurrent_max_abs_error"],
                row["forward_plus_initial_state_backward"][
                    "forward_max_abs_error"
                ],
            )
        )
        maximum_gradient_error = max(
            row["forward_plus_initial_state_backward"][
                "initial_state_gradient_max_abs_error"
            ]
            for report in self.report["results"]
            for row in report["results"]
        )
        self.assertLessEqual(
            maximum_forward_error,
            self.report["settings"]["forward_parity_tolerance_max_abs"],
        )
        self.assertLessEqual(
            maximum_gradient_error,
            self.report["settings"][
                "initial_gradient_parity_tolerance_max_abs"
            ],
        )

    def test_artifact_records_its_then_open_fused_kernel_boundary(self) -> None:
        self.assertEqual(
            self.report["algorithm"]["implementation"],
            "eager PyTorch composition; no fused custom kernel",
        )
        nonclaims = self.report["claim_scope"]["not_claimed"]
        self.assertIn(
            "a fused Triton or custom CUDA scan kernel has been implemented",
            nonclaims,
        )
        self.assertIn(
            "full model backward, optimizer-step, or end-to-end SSM throughput",
            nonclaims,
        )

    def test_artifact_hash(self) -> None:
        digest = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_ARTIFACT_SHA256)


if __name__ == "__main__":
    unittest.main()
