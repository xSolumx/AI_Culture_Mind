import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT / "artifacts" / "mixed_monomial_golden_macro_benchmark_20260817.json"
)
EXPECTED_ARTIFACT_SHA256 = (
    "93103d6fa5b4c36a43c89d8cac012d22deaa41a020c516d4c2b9ad9fc2bd8add"
)
EXPECTED_COMPILER_SHA256 = (
    "9595578918bd46387ce5773607d1ded3d6117b11cb2b2353b586a1c6fc0cd438"
)


class MixedMonomialGoldenMacroBenchmarkArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_benchmark_is_linked_to_the_exact_compiler(self) -> None:
        self.assertEqual(
            self.report["exact_compiler_artifact_sha256"],
            EXPECTED_COMPILER_SHA256,
        )
        self.assertTrue(self.report["passed"])

    def test_benchmark_covers_every_view_device_and_batch(self) -> None:
        keys = {
            (report["view"], report["device"])
            for report in self.report["results"]
        }
        self.assertEqual(
            keys,
            {
                (view, device)
                for view in (
                    "vector",
                    "positive_half_spin",
                    "negative_half_spin",
                )
                for device in ("cpu", "cuda")
            },
        )
        for report in self.report["results"]:
            self.assertEqual(
                [row["batch_size"] for row in report["batch_results"]],
                [1, 64, 1024, 16384],
            )

    def test_labelled_compilation_won_every_recorded_comparison(self) -> None:
        for report in self.report["results"]:
            for batch in report["batch_results"]:
                with self.subTest(
                    view=report["view"],
                    device=report["device"],
                    batch=batch["batch_size"],
                ):
                    self.assertGreater(
                        batch["transition_materialization"][
                            "labelled_speedup_vs_online"
                        ],
                        1,
                    )
                    self.assertGreater(
                        batch["state_application"][
                            "labelled_speedup_vs_online"
                        ],
                        1,
                    )

    def test_float32_parity_and_timing_controls_passed(self) -> None:
        self.assertTrue(
            self.report["checks"]["all_float32_parity_checks_passed"]
        )
        self.assertTrue(self.report["checks"]["every_timing_is_positive"])
        self.assertEqual(self.report["settings"]["torch_cpu_threads"], 1)
        self.assertEqual(self.report["settings"]["torch_interop_threads"], 1)
        self.assertTrue(
            self.report["settings"]["cuda_synchronized_each_timing"]
        )
        maximum_error = max(
            error
            for report in self.report["results"]
            for batch in report["batch_results"]
            for section in (
                batch["transition_materialization"],
                batch["state_application"],
            )
            for error in (
                section["deduplicated_max_abs_error"],
                section["labelled_max_abs_error"],
            )
        )
        self.assertLessEqual(maximum_error, 2e-6)

    def test_artifact_keeps_end_to_end_claim_open(self) -> None:
        nonclaims = self.report["claim_scope"]["not_claimed"]
        self.assertIn(
            "an end-to-end SSM accuracy or throughput advantage", nonclaims
        )

    def test_artifact_hash(self) -> None:
        digest = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_ARTIFACT_SHA256)


if __name__ == "__main__":
    unittest.main()
