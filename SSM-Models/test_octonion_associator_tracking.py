"""Regression contracts for the frozen continuous associator benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
ARTIFACT = (
    ROOT / "experiments" / "artifacts" / "octonion_associator_tracking_pilot300.json"
)
ARTIFACT_SHA256 = "c282b21a2050006f69a1c31c42dd28d2fcd9311e7ba52ef310c3c5cde49d802e"
TRAINING_SCHEDULE_SHA256 = (
    "92dcd37d893e73a4656074b30b332ac44370642073823cdff3533687bc82a1c7"
)
EVALUATION_SCHEDULE_SHA256 = (
    "2c26c8bae29e5c18a0aa0881efafa1c9e4377c46b5ccccafab8e16d2fb9ab8fb"
)


class FrozenAssociatorTrackingArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = ARTIFACT.read_bytes()
        cls.report = json.loads(cls.payload)

    def test_artifact_hash_and_frozen_schedules(self) -> None:
        self.assertEqual(hashlib.sha256(self.payload).hexdigest(), ARTIFACT_SHA256)
        self.assertEqual(
            self.report["schedules"]["training_schedule_sha256"],
            TRAINING_SCHEDULE_SHA256,
        )
        self.assertEqual(
            self.report["schedules"]["evaluation_schedule_sha256"],
            EVALUATION_SCHEDULE_SHA256,
        )

    def test_required_gates_and_claim_boundary(self) -> None:
        self.assertTrue(self.report["all_required_checks_passed"])
        self.assertTrue(all(self.report["checks"].values()))
        self.assertIn("One-seed", self.report["claim_boundary"])
        self.assertIn("not natural-task", self.report["claim_boundary"])

        results = self.report["results"]
        oracle = results["exact_operator_oracle"]["evaluation"]["128"]["mse"]
        learned = results["learned_octonion_operator"]["evaluation"]["128"]["mse"]
        collapsed = results["collapsed_octonion_ablation"]["evaluation"]["128"]["mse"]
        mamba = results["transformers_mamba2"]["evaluation"]["128"]["mse"]
        delta = results["delta_product_reference"]["evaluation"]["128"]["mse"]

        self.assertLess(oracle, 1e-12)
        self.assertLess(learned, 1e-10)
        self.assertGreater(collapsed, 0.2)
        self.assertLess(learned, min(collapsed, mamba, delta))
        self.assertTrue(
            all(
                math.isfinite(row["mse"])
                for result in results.values()
                for row in result["evaluation"].values()
            )
        )

    def test_learned_checkpoints_rehash_and_reload(self) -> None:
        for name in (
            "learned_octonion_operator",
            "transformers_mamba2",
            "delta_product_reference",
        ):
            record = self.report["results"][name]
            checkpoint = ROOT / Path(record["checkpoint"])
            self.assertTrue(checkpoint.is_file(), checkpoint)
            self.assertEqual(
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                record["checkpoint_sha256"],
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(payload["candidate"], name)
            self.assertEqual(payload["format_version"], 1)
            self.assertTrue(payload["state_dict"])


if __name__ == "__main__":
    unittest.main()
