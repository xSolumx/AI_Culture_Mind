"""Regression contracts for the frozen Haar-basis replication and audit."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
COHORT = (
    ROOT / "experiments" / "artifacts" / "octonion_basis_transport_replication300.json"
)
COHORT_SHA256 = "b96bed5d0e4c33e229816f6ce2db24d2c42c5b33ae1b978950a2a0bd9960daf5"
AUDIT = ROOT / "experiments" / "artifacts" / "octonion_basis_identification_audit.json"
AUDIT_SHA256 = "bb149467ebc4cfe32b8ded8311d6a0b5ec47a7b606d13973d371ee7ae955d5c6"


class FrozenHaarBasisArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cohort_bytes = COHORT.read_bytes()
        cls.cohort = json.loads(cls.cohort_bytes)
        cls.audit_bytes = AUDIT.read_bytes()
        cls.audit = json.loads(cls.audit_bytes)

    def test_authoritative_hashes_and_explicit_frozen_failure(self) -> None:
        self.assertEqual(hashlib.sha256(self.cohort_bytes).hexdigest(), COHORT_SHA256)
        self.assertEqual(hashlib.sha256(self.audit_bytes).hexdigest(), AUDIT_SHA256)
        self.assertFalse(self.cohort["all_required_checks_passed"])
        self.assertTrue(self.audit["all_required_checks_passed"])
        self.assertTrue(self.audit["source_artifact_hash_matches_frozen"])

    def test_three_basis_result_and_failed_gates_are_locked(self) -> None:
        self.assertEqual(
            [report["basis_seed"] for report in self.cohort["basis_reports"]],
            [0, 1, 2],
        )
        for report in self.cohort["basis_reports"]:
            checks = report["checks"]
            self.assertTrue(checks["learned_basis_l128_below_1e_3"])
            self.assertTrue(checks["learned_basis_beats_fixed_canonical"])
            self.assertFalse(checks["dense_operator_l128_below_1e_3"])
            self.assertTrue(checks["dense_operator_beats_fixed_canonical"])
            results = report["results"]
            self.assertLess(
                results["learned_basis_operator"]["evaluation"]["128"]["mse"],
                1e-6,
            )
            self.assertGreater(
                results["dense_linear_operator"]["evaluation"]["128"]["mse"],
                0.12,
            )
        self.assertFalse(self.cohort["basis_reports"][0]["checks"]["oracle_is_exact"])
        self.assertTrue(self.cohort["basis_reports"][1]["checks"]["oracle_is_exact"])
        self.assertTrue(self.cohort["basis_reports"][2]["checks"]["oracle_is_exact"])

    def test_post_protocol_dense_and_g2_diagnostics(self) -> None:
        self.assertEqual(len(self.audit["basis_audits"]), 3)
        for report in self.audit["basis_audits"]:
            self.assertTrue(report["all_required_checks_passed"])
            self.assertTrue(all(report["schedule_replay"].values()))
            self.assertTrue(all(report["checkpoint_rehash"].values()))
            identified = report["identified_dense_operator"]
            self.assertEqual(identified["design_rank"], 8)
            self.assertLess(identified["evaluation"]["128"]["mse"], 1e-10)
            gauge = report["learned_gauge"]
            self.assertLess(gauge["gauge_identity_residual"], 1e-3)
            self.assertLess(gauge["g2_left_action_intertwiner_residual"], 1e-3)

    def test_all_twelve_learned_checkpoints_rehash_and_reload(self) -> None:
        count = 0
        for report in self.cohort["basis_reports"]:
            for name, record in report["results"].items():
                if "checkpoint" not in record:
                    continue
                count += 1
                checkpoint = ROOT / Path(record["checkpoint"])
                self.assertEqual(
                    hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    record["checkpoint_sha256"],
                )
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                self.assertEqual(payload["candidate"], name)
                self.assertTrue(payload["state_dict"])
        self.assertEqual(count, 12)


if __name__ == "__main__":
    unittest.main()
