"""Integrity checks for the frozen WSL/Triton octonion benchmark."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


class OctonionTritonBenchmarkTests(unittest.TestCase):
    ARTIFACT = (
        Path(__file__).resolve().parent
        / "experiments"
        / "artifacts"
        / "octonion_triton_scan_wsl_rtx2070s_20260816.json"
    )
    SHA256 = "3af7bb5d0e8711d96c9ef2e0bca60eef98f1958e1bcb5c78570b8bd4c78a1d2c"

    def test_artifact_hash_environment_and_gates(self) -> None:
        payload = self.ARTIFACT.read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), self.SHA256)
        report = json.loads(payload)
        self.assertTrue(report["all_required_checks_passed"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["environment"]["triton"], "3.0.0")
        self.assertEqual(report["environment"]["compute_capability"], [7, 5])

    def test_frozen_speedups_and_backward_are_present(self) -> None:
        report = json.loads(self.ARTIFACT.read_text())
        timings = report["timings"]
        self.assertGreater(
            timings["speedup_ratios"]["4096"]["work_efficient_over_triton_forward"],
            4.5,
        )
        self.assertGreater(
            timings["speedup_ratios"]["512"]["raw_recurrent_over_triton_forward"],
            150,
        )
        backward = timings["forward_backward"]["1024"]
        fused = backward["triton_fused_recurrent"]["median_ms"]
        work_efficient = backward["pytorch_work_efficient_operator"]["median_ms"]
        self.assertGreater(work_efficient / fused, 6.0)


if __name__ == "__main__":
    unittest.main()
