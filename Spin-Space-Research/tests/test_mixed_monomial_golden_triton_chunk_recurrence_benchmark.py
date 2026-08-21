import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "artifacts"
    / "mixed_monomial_golden_triton_chunk_recurrence_benchmark_20260817.json"
)
EXPECTED_ARTIFACT_SHA256 = (
    "b6897f60a011f2dcb0788fccc32195277a54c600f87199d8d13f75b043f37a7d"
)
EXPECTED_COMPILER_SHA256 = (
    "ed1ae7e8ac98c5e037be4e45d10f22ec3236e7d6f8337fbc2b9f9a499e13e5de"
)


class MixedMonomialGoldenTritonChunkRecurrenceBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_artifact_is_complete_and_linked_to_exact_compiler(self) -> None:
        self.assertEqual(
            self.report["exact_chunk_compiler_artifact_sha256"],
            EXPECTED_COMPILER_SHA256,
        )
        self.assertTrue(self.report["passed"])
        self.assertTrue(
            self.report["checks"][
                "triton_forward_and_backward_kernels_compiled_and_executed"
            ]
        )
        self.assertEqual(len(self.report["results"]), 3)
        self.assertTrue(self.report["checks"]["all_result_grids_are_complete"])
        for view in self.report["results"]:
            self.assertEqual(len(view["results"]), 9)

    def test_fused_recurrence_won_every_recorded_parallel_control_cell(self) -> None:
        for view in self.report["results"]:
            for row in view["results"]:
                for phase in (
                    "forward",
                    "forward_plus_initial_state_backward",
                ):
                    with self.subTest(
                        view=view["view"],
                        batch=row["batch_size"],
                        chunks=row["chunk_count"],
                        phase=phase,
                    ):
                        self.assertGreater(
                            row[phase][
                                "fused_speedup_vs_parallel_preselected_eager"
                            ],
                            1,
                        )
                        self.assertGreater(
                            row[phase][
                                "fused_speedup_vs_parallel_indexed_triton_local"
                            ],
                            1,
                        )
                        self.assertGreaterEqual(
                            row[phase]["timings"]["fused_triton_recurrence"][
                                "repeats"
                            ],
                            10,
                        )

    def test_forward_and_initial_gradient_parity_passed(self) -> None:
        self.assertTrue(
            self.report["checks"][
                "all_float32_forward_backward_parity_checks_passed"
            ]
        )
        maximum_forward_error = max(
            error
            for view in self.report["results"]
            for row in view["results"]
            for error in row["forward"][
                "max_abs_error_vs_parallel_preselected_eager"
            ].values()
        )
        maximum_gradient_error = max(
            error
            for view in self.report["results"]
            for row in view["results"]
            for error in row["forward_plus_initial_state_backward"][
                "max_abs_error_vs_parallel_preselected_eager"
            ].values()
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

    def test_serial_depth_and_training_boundaries_are_explicit(self) -> None:
        nonclaims = self.report["claim_scope"]["not_claimed"]
        self.assertIn("the candidate is a parallel prefix scan", nonclaims)
        self.assertIn("prefix-table gradients in the Triton path", nonclaims)
        self.assertIn(
            "full model backward, optimizer-step, or end-to-end SSM throughput",
            nonclaims,
        )

    def test_artifact_hash(self) -> None:
        self.assertEqual(
            hashlib.sha256(ARTIFACT.read_bytes()).hexdigest(),
            EXPECTED_ARTIFACT_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
