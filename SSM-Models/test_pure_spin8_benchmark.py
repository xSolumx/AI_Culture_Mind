"""Regression contracts for the frozen Pure Spin(8)/Mamba-2 cohort."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import torch
from benchmark_pure_spin8_vs_mamba2 import MaintainedPureSpin8Tracker

ROOT = Path(__file__).resolve().parent
ARTIFACT = (
    ROOT
    / "experiments"
    / "artifacts"
    / "pure_spin8_vs_mamba2_triality_transport1000.json"
)
ARTIFACT_SHA256 = "d265e28a132c28261ae317958adfa34619c5dd0c58a0859b26e7afd653ad9876"


class FrozenPureSpin8BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = ARTIFACT.read_bytes()
        cls.report = json.loads(cls.payload)

    def test_artifact_hash_version_and_all_gates(self) -> None:
        self.assertEqual(hashlib.sha256(self.payload).hexdigest(), ARTIFACT_SHA256)
        self.assertTrue(self.report["all_required_checks_passed"])
        self.assertEqual(self.report["pure_spin8_version"], "1.0.0")
        self.assertEqual(len(self.report["seed_reports"]), 3)
        for seed_report in self.report["seed_reports"]:
            self.assertTrue(seed_report["all_required_checks_passed"])
            self.assertTrue(all(seed_report["checks"].values()))

    def test_pure_spin8_beats_references_and_retains_center(self) -> None:
        for seed_report in self.report["seed_reports"]:
            results = seed_report["results"]
            pure = results["maintained_pure_spin8"]
            mamba = results["transformers_mamba2"]
            delta = results["delta_product_reference"]
            pure_mse = pure["evaluation"]["128"]["mse"]
            self.assertEqual(pure["parameters"], 836)
            self.assertEqual(pure["recurrent_state_scalars"], 24)
            self.assertLess(pure_mse, 1e-4)
            self.assertLess(pure_mse, mamba["evaluation"]["128"]["mse"])
            self.assertLess(pure_mse, delta["evaluation"]["128"]["mse"])
            self.assertEqual(
                pure["center_pair_evaluation"]["center_classification_accuracy"],
                1.0,
            )
            self.assertLessEqual(
                mamba["center_pair_evaluation"]["center_classification_accuracy"],
                0.5,
            )

    def test_all_nine_checkpoints_rehash_and_reload(self) -> None:
        count = 0
        for seed_report in self.report["seed_reports"]:
            seed = seed_report["seed"]
            for name, record in seed_report["results"].items():
                count += 1
                checkpoint = ROOT / Path(record["checkpoint"])
                self.assertEqual(
                    hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    record["checkpoint_sha256"],
                )
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                self.assertEqual(payload["candidate"], name)
                self.assertTrue(payload["state_dict"])
                if name == "maintained_pure_spin8":
                    self.assertEqual(payload["model_version"], "1.0.0")
                    model = MaintainedPureSpin8Tracker(seed)
                    model.load_state_dict(payload["state_dict"])
        self.assertEqual(count, 9)


if __name__ == "__main__":
    unittest.main()
