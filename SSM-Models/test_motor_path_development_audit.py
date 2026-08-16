"""Structural tests for the rigid-motor path-development audit."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import torch
from audit_motor_path_development import (
    MotorAuditConfig,
    audit_length,
    deterministic_motors,
    matrix_prefix_scan,
)
from pure_rotor_ssm.motor_scan import motor_to_matrix


class MotorPathDevelopmentAuditTests(unittest.TestCase):
    def test_frozen_artifact_contract(self) -> None:
        artifact_path = (
            Path(__file__).parent
            / "experiments"
            / "artifacts"
            / "motor_path_development_20260816.json"
        )
        artifact_bytes = artifact_path.read_bytes()
        self.assertEqual(
            hashlib.sha256(artifact_bytes).hexdigest(),
            "3496e374ddb68d48f3105b41ac39c23a40be3edaa872f76dcf6ce9e06ea8f95a",
        )
        report = json.loads(artifact_bytes)
        self.assertEqual(report["status"], "completed numerical gate")
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(
            [row["length"] for row in report["length_results"]],
            [16, 128, 1024, 4096],
        )

    def test_matrix_parallel_recurrent_and_cache_contract(self) -> None:
        motors = deterministic_motors(
            batch_size=2,
            length=31,
            lanes=3,
            dtype=torch.float64,
            device=torch.device("cpu"),
            seed=147,
            angle_scale=0.3,
            translation_scale=0.05,
        )
        matrices = motor_to_matrix(motors)
        parallel, parallel_final = matrix_prefix_scan(matrices, mode="parallel")
        recurrent, recurrent_final = matrix_prefix_scan(matrices, mode="recurrent")
        torch.testing.assert_close(parallel, recurrent, rtol=2e-12, atol=2e-12)
        torch.testing.assert_close(
            parallel_final, recurrent_final, rtol=2e-12, atol=2e-12
        )
        first, cache = matrix_prefix_scan(matrices[:, :11], mode="parallel")
        second, cache = matrix_prefix_scan(matrices[:, 11:], cache, mode="parallel")
        torch.testing.assert_close(
            parallel, torch.cat((first, second), dim=1), rtol=2e-12, atol=2e-12
        )
        torch.testing.assert_close(parallel_final, cache, rtol=2e-12, atol=2e-12)

    def test_small_exact_gate_passes(self) -> None:
        config = MotorAuditConfig(
            lengths=(16,),
            batch_size=2,
            lanes=2,
            timing_warmups=1,
            timing_repeats=1,
        )
        row = audit_length(config, 16)
        self.assertTrue(row["all_outputs_finite"])
        self.assertLess(row["motor_parallel_recurrent_max_abs_error"], 2e-12)
        self.assertLess(row["motor_chunked_full_max_abs_error"], 2e-12)
        self.assertLess(row["motor_matrix_prefix_max_abs_error"], 2e-12)
        self.assertLess(row["maximum_study_condition_error"], 2e-12)
        self.assertLess(row["central_negation_state_antipode_max_abs_error"], 2e-12)
        self.assertLess(row["central_negation_physical_matrix_max_abs_error"], 2e-12)
        self.assertGreater(row["maximum_absolute_translation"], 0.05)


if __name__ == "__main__":
    unittest.main()
