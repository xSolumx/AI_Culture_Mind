"""Contracts for signed-pose noise in local motor identification."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import torch
from audit_noisy_motor_identification import noisy_training_batches
from benchmark_spin_motor_rigid_2a5 import (
    RigidSpinConfig,
    identify_direct_motor_from_prefixes,
    make_rigid_spin_task,
    make_training_batches,
)


class NoisyMotorIdentificationTests(unittest.TestCase):
    ARTIFACT_SHA256 = "2adc41d821e110e8b8e05624a0587c2aaa04492c2c47f8eabd7e35251020a6d5"

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = RigidSpinConfig(steps=8, batch_size=16)
        cls.task = make_rigid_spin_task("e", cls.config.translation_step)
        cls.training = make_training_batches(cls.task, cls.config)

    def test_noise_is_deterministic_signed_and_seed_sensitive(self) -> None:
        first, first_audit = noisy_training_batches(
            self.training,
            rotation_std_degrees=5.0,
            translation_std=0.05,
            noise_seed=3,
        )
        repeated, repeated_audit = noisy_training_batches(
            self.training,
            rotation_std_degrees=5.0,
            translation_std=0.05,
            noise_seed=3,
        )
        different, different_audit = noisy_training_batches(
            self.training,
            rotation_std_degrees=5.0,
            translation_std=0.05,
            noise_seed=4,
        )
        self.assertEqual(
            first_audit["noisy_pose_schedule_sha256"],
            repeated_audit["noisy_pose_schedule_sha256"],
        )
        self.assertNotEqual(
            first_audit["noisy_pose_schedule_sha256"],
            different_audit["noisy_pose_schedule_sha256"],
        )
        self.assertTrue(torch.equal(first[0].pose_targets, repeated[0].pose_targets))
        self.assertFalse(torch.equal(first[0].pose_targets, different[0].pose_targets))
        signed_dot = (
            first[0].pose_targets[..., :4] * self.training[0].pose_targets[..., :4]
        ).sum(dim=-1)
        self.assertTrue(bool((signed_dot > 0).all()))

    def test_medium_noise_still_identifies_finite_token_motors(self) -> None:
        noisy, _ = noisy_training_batches(
            self.training,
            rotation_std_degrees=5.0,
            translation_std=0.05,
            noise_seed=0,
        )
        model, audit = identify_direct_motor_from_prefixes(self.task, noisy)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 49)
        self.assertTrue(
            torch.isfinite(model.composition.normalized_token_motors()).all()
        )
        # This deliberately tiny fixture has far fewer prefix observations than
        # the frozen 76,800-observation audit, so it checks a broad finite-
        # sample bound rather than importing the artifact's much tighter error.
        self.assertLess(audit["maximum_exact_token_quaternion_error_degrees"], 3.0)
        self.assertLess(audit["maximum_exact_token_translation_error"], 0.05)

    def test_frozen_noise_artifact_and_checkpoint_hashes(self) -> None:
        path = (
            Path(__file__).parent
            / "experiments"
            / "artifacts"
            / "spin_motor_noisy_identification_4tiers_5seeds.json"
        )
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(), self.ARTIFACT_SHA256
        )
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["run_count"], 20)
        self.assertTrue(report["required_clean_low_medium_tiers_passed"])
        for tier in ("clean", "low", "medium"):
            summary = report["tier_summaries"][tier]
            self.assertEqual(summary["minimum_joint_signed_pose_accuracy"], 1.0)
            self.assertEqual(summary["minimum_paired_double_cover_pose_accuracy"], 1.0)
        high = report["tier_summaries"]["high"]
        self.assertTrue(high["all_center_gates_passed"])
        self.assertFalse(high["all_joint_pose_gates_passed"])
        self.assertLess(high["minimum_joint_signed_pose_accuracy"], 0.5)
        for run in report["runs"]:
            checkpoint = Path(__file__).parent / run["checkpoint"]
            self.assertEqual(
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                run["checkpoint_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
