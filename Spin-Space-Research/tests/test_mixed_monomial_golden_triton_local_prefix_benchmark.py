import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "artifacts"
    / "mixed_monomial_golden_triton_local_prefix_benchmark_20260817.json"
)
EXPECTED_ARTIFACT_SHA256 = (
    "b124288e9e68d0513e66f39e5fd20c5c831b49d824be28d73ae76305040c6f17"
)
EXPECTED_COMPILER_SHA256 = (
    "ed1ae7e8ac98c5e037be4e45d10f22ec3236e7d6f8337fbc2b9f9a499e13e5de"
)


class MixedMonomialGoldenTritonLocalPrefixBenchmarkTests(unittest.TestCase):
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
            self.report["checks"]["triton_kernel_compiled_and_executed"]
        )
        self.assertEqual(len(self.report["results"]), 3)
        self.assertTrue(self.report["checks"]["all_result_grids_are_complete"])
        for view in self.report["results"]:
            self.assertEqual(len(view["results"]), 9)

    def test_parity_and_empirical_win_counts_are_preserved(self) -> None:
        self.assertTrue(
            self.report["checks"][
                "all_float32_forward_backward_parity_checks_passed"
            ]
        )
        rows = [
            row
            for view in self.report["results"]
            for row in view["results"]
        ]
        local_forward_wins = sum(
            row["local_expansion_only"]["forward"][
                "triton_speedup_vs_indexed_eager"
            ]
            > 1
            for row in rows
        )
        local_backward_wins = sum(
            row["local_expansion_only"]["forward_plus_state_backward"][
                "triton_speedup_vs_indexed_eager"
            ]
            > 1
            for row in rows
        )
        full_forward_wins = sum(
            row["full_two_stage_scan"]["forward"][
                "triton_speedup_vs_indexed_eager"
            ]
            > 1
            for row in rows
        )
        full_backward_wins = sum(
            row["full_two_stage_scan"]["forward_plus_state_backward"][
                "triton_speedup_vs_indexed_eager"
            ]
            > 1
            for row in rows
        )
        self.assertEqual(
            (
                local_forward_wins,
                local_backward_wins,
                full_forward_wins,
                full_backward_wins,
            ),
            (26, 25, 16, 14),
        )

    def test_gradient_scope_and_unfused_endpoint_boundary_are_explicit(self) -> None:
        nonclaims = self.report["claim_scope"]["not_claimed"]
        self.assertIn("prefix-table gradients in the Triton path", nonclaims)
        self.assertIn("a fused endpoint matrix scan", nonclaims)
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
