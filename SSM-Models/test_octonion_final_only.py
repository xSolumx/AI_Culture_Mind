"""Regression contracts for the frozen final-only Haar-basis cohort."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
ARTIFACT = (
    ROOT / "experiments" / "artifacts" / "octonion_final_only_replication1000.json"
)
ARTIFACT_SHA256 = "7def2ca25ce7b04f11f282c06c16dea51ccf86d2a18eae1a4db679fa4d9e8f4a"
PARITY_ARTIFACT = (
    ROOT / "experiments" / "artifacts" / "octonion_final_only_parity_gauge_audit.json"
)
PARITY_ARTIFACT_SHA256 = (
    "32545cb8174bfd84cf0cb2d1a84202570af34dce20ff4f0e48dd9573118c8d30"
)


class FrozenFinalOnlyArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = ARTIFACT.read_bytes()
        cls.report = json.loads(cls.payload)

    def test_hash_and_top_level_failure_are_locked(self) -> None:
        self.assertEqual(hashlib.sha256(self.payload).hexdigest(), ARTIFACT_SHA256)
        self.assertFalse(self.report["all_required_checks_passed"])
        self.assertIn("Final-only", self.report["claim_boundary"])

    def test_basis_specific_success_and_failures(self) -> None:
        reports = {row["basis_seed"]: row for row in self.report["basis_reports"]}
        self.assertEqual(set(reports), {0, 1, 2})
        self.assertFalse(reports[0]["all_required_checks_passed"])
        self.assertTrue(reports[1]["all_required_checks_passed"])
        self.assertFalse(reports[2]["all_required_checks_passed"])

        for seed in (0, 2):
            self.assertFalse(reports[seed]["checks"]["all_structured_l128_below_1e_3"])
            for record in reports[seed]["learned_basis_operator"].values():
                self.assertGreater(record["evaluation"]["128"]["mse"], 0.24)

        self.assertTrue(reports[1]["checks"]["all_structured_l128_below_1e_3"])
        for record in reports[1]["learned_basis_operator"].values():
            self.assertLess(record["evaluation"]["128"]["mse"], 4e-11)

        for report in reports.values():
            for record in report["dense_linear_operator"].values():
                self.assertGreater(record["evaluation"]["128"]["mse"], 0.124)
                self.assertLess(record["evaluation"]["128"]["mse"], 0.126)

    def test_all_twenty_four_checkpoints_rehash_and_reload(self) -> None:
        count = 0
        for report in self.report["basis_reports"]:
            families = (
                report["learned_basis_operator"],
                report["dense_linear_operator"],
                report["reference_results"],
            )
            for family in families:
                for name, record in family.items():
                    count += 1
                    checkpoint = ROOT / Path(record["checkpoint"])
                    self.assertEqual(
                        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        record["checkpoint_sha256"],
                    )
                    payload = torch.load(
                        checkpoint, map_location="cpu", weights_only=False
                    )
                    if name in {"0", "1", "2"}:
                        self.assertEqual(
                            payload["candidate"],
                            (
                                "learned_basis_operator"
                                if family is report["learned_basis_operator"]
                                else "dense_linear_operator"
                            ),
                        )
                    else:
                        self.assertEqual(payload["candidate"], name)
                    self.assertTrue(payload["state_dict"])
        self.assertEqual(count, 24)


class FinalOnlyParityAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = PARITY_ARTIFACT.read_bytes()
        cls.report = json.loads(cls.payload)

    def test_hash_and_signed_g2_gate_are_locked(self) -> None:
        self.assertEqual(
            hashlib.sha256(self.payload).hexdigest(), PARITY_ARTIFACT_SHA256
        )
        self.assertTrue(self.report["all_required_checks_passed"])
        self.assertEqual(self.report["positive_g2_coset_models"], 4)
        self.assertEqual(self.report["negative_g2_coset_models"], 5)

    def test_odd_length_exposes_the_negative_coset(self) -> None:
        models = [
            model for basis in self.report["basis_audits"] for model in basis["models"]
        ]
        self.assertEqual(len(models), 9)
        for model in models:
            self.assertTrue(model["all_required_checks_passed"])
            sign = model["gauge"]["parity_sign"]
            ordinary = model["odd_length_evaluation"]["ordinary_mse"]
            corrected = model["odd_length_evaluation"]["parity_corrected_mse"]
            self.assertLess(corrected, 1e-8)
            if sign == 1:
                self.assertLess(ordinary, 1e-8)
            else:
                self.assertGreater(ordinary, 0.49)


if __name__ == "__main__":
    unittest.main()
