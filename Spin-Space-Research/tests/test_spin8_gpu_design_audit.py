"""Acceptance tests for the stored CUDA continuous-design cohort."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


class Spin8GPUDesignAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.report = json.loads(
            (root / "artifacts" / "spin8_gpu_design_cohort_20260806.json").read_text(
                encoding="utf-8"
            )
        )
        cls.resource = json.loads(
            (
                root / "artifacts" / "spin8_gpu_design_cohort_resource_20260806.json"
            ).read_text(encoding="utf-8")
        )

    def test_every_seed_replays_frozen_falsification_gates(self) -> None:
        self.assertEqual(self.report["seed_count"], 10)
        self.assertEqual(self.report["pass_count"], 10)
        self.assertEqual(self.report["total_dense_interior_samples"], 860_160)
        self.assertEqual(self.report["total_gradient_restarts"], 1_680)
        for row in self.report["rows"]:
            self.assertTrue(row["passed"])
            self.assertEqual(row["dense_candidate_count"], 0)
            self.assertEqual(row["gradient_candidate_count"], 0)
            self.assertEqual(row["noise_uphill_count"], 0)
            self.assertEqual(row["sensitivity_maximum"], 75.0)
            self.assertLess(row["reweighting_maximum_weight_error"], 1e-12)

    def test_resource_contract_was_measured(self) -> None:
        self.assertTrue(self.resource["passed"])
        self.assertEqual(self.resource["workers"], 6)
        self.assertEqual(len(self.resource["cpu_affinity"]), 6)
        self.assertEqual(self.resource["memory_limit_gib"], 15.0)
        self.assertFalse(self.resource["memory_limit_exceeded"])
        self.assertLess(self.resource["peak_process_tree_rss_gib"], 15.0)
        self.assertLess(
            max(row["peak_cuda_memory_mib"] for row in self.report["rows"]),
            8192,
        )


if __name__ == "__main__":
    unittest.main()
