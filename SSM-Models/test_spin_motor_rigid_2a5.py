"""Contract tests for the rigid ``2.A5`` Spin/motor benchmark."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from benchmark_pure_rotor_2a5 import parameter_count
from benchmark_spin_motor_rigid_2a5 import (
    ORACLE_CANDIDATES,
    RigidSpinConfig,
    build_models,
    evaluate_relation_pairs,
    identify_direct_motor_from_prefixes,
    make_relation_pair_batches,
    make_rigid_spin_task,
    make_training_batches,
    recurrent_state_scalars,
    relation_pair_audit,
    rigid_prefix_targets,
    training_split_audit,
)
from pure_rotor_ssm.motor_scan import DirectMotorPoseTracker, DirectProductPoseTracker


class RigidSpinBenchmarkTests(unittest.TestCase):
    ARTIFACT_HASHES: ClassVar[dict[str, str]] = {
        "spin_motor_rigid_2a5_pilot300.json": "49d5d031a6e496e36e8404b666d07d49ff4af6469aa7b4a2af6f3f5c01e2d3e2",
        "spin_motor_direct_readout_pilot300.json": "7a364b61ba51666db65f0ced909fc78d81855582fd14e9dd5e598d2d4d3ab1f2",
        "spin_motor_identified_e_seed0.json": "df132600d8be86505a4e5156b161a7e2ee33ce84dc4dba1dae26d3207653c62b",
        "spin_motor_identification_replication_3x3.json": "97ffc994889278b21da7482ff49a597d9799f4ac38472e43b459465468a00aa5",
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = RigidSpinConfig(
            steps=8,
            batch_size=16,
            validation_batches=1,
            validation_pairs_per_batch=2,
            evaluation_lengths=(16,),
        )
        cls.task = make_rigid_spin_task("e", cls.config.translation_step)

    def test_body_frame_translation_is_noncommutative_with_rotation(self) -> None:
        # ``tx,a`` translates before the rotation; ``a,tx`` translates in the
        # rotated body frame.  Their endpoints must differ.
        inputs = np.asarray([[4, 0], [0, 4]], dtype=np.int64)
        groups, poses = rigid_prefix_targets(self.task, inputs)
        self.assertEqual(int(groups[0, -1]), int(groups[1, -1]))
        self.assertGreater(
            float(np.linalg.norm(poses[0, -1, 4:] - poses[1, -1, 4:])), 0.1
        )

    def test_training_omits_relations_but_covers_all_rotation_states(self) -> None:
        first = make_training_batches(self.task, self.config)
        second = make_training_batches(self.task, self.config)
        audit = training_split_audit(self.task, first)
        repeated = training_split_audit(self.task, second)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["exact_legal_rotation_state_coverage"], 120)
        self.assertEqual(audit["observed_binary_rotation_states"], 120)
        self.assertEqual(audit["observed_projective_rotation_states"], 60)
        self.assertEqual(audit["observed_center_bits"], [0, 1])
        self.assertEqual(
            audit["input_group_pose_schedule_sha256"],
            repeated["input_group_pose_schedule_sha256"],
        )
        self.assertTrue(
            all(
                value == 0 for value in audit["forbidden_relation_occurrences"].values()
            )
        )
        for symbol in ("tx", "ty", "tz"):
            self.assertGreater(audit["token_counts"][symbol], 0)

    def test_all_relation_pairs_are_antipodal_but_pose_equal(self) -> None:
        for relation in self.task.rotation.relations:
            for position in ("early", "late"):
                batches = make_relation_pair_batches(
                    self.task, relation, self.config, 16, position
                )
                audit = relation_pair_audit(self.task, relation, batches)
                self.assertTrue(audit["passed"], (relation.key, position, audit))
                self.assertEqual(
                    audit["antipodal_quaternion_checks"],
                    audit["scored_pair_positions"],
                )
                self.assertEqual(
                    audit["translation_equality_checks"],
                    audit["scored_pair_positions"],
                )
                self.assertGreater(audit["nonzero_translation_scored_positions"], 0)

    def test_quotient_oracle_loses_exactly_the_pair_sign_bit(self) -> None:
        relation = self.task.rotation.relations[0]
        batches = make_relation_pair_batches(
            self.task, relation, self.config, 16, "early"
        )
        metrics = {
            name: evaluate_relation_pairs(
                name,
                None,
                batches,
                torch.device("cpu"),
                self.config,
                spin_scan_mode="parallel",
                motor_scan_mode="parallel",
                delta_scan_mode="parallel",
            )
            for name in ORACLE_CANDIDATES
        }
        exact = metrics["exact_spin_motor_oracle"]
        quotient = metrics["se3_quotient_oracle"]
        self.assertEqual(exact["joint_signed_pose_accuracy"], 1.0)
        self.assertEqual(exact["paired_double_cover_pose_accuracy"], 1.0)
        self.assertEqual(quotient["signed_rotation_threshold_accuracy"], 0.5)
        self.assertEqual(quotient["physical_rotation_threshold_accuracy"], 1.0)
        self.assertEqual(quotient["translation_threshold_accuracy"], 1.0)
        self.assertEqual(quotient["paired_antipodal_threshold_accuracy"], 0.0)
        self.assertEqual(quotient["paired_physical_pose_agreement_accuracy"], 1.0)

    def test_direct_motor_realizes_exact_task_but_direct_product_does_not(self) -> None:
        motor = DirectMotorPoseTracker(len(self.task.input_symbols)).double()
        direct_product = DirectProductPoseTracker(len(self.task.input_symbols)).double()
        quaternions = torch.tensor(
            [
                self.task.rotation.binary.quaternions[element]
                for element in self.task.input_elements
            ],
            dtype=torch.double,
        )
        translations = torch.tensor(self.task.token_translations, dtype=torch.double)
        with torch.no_grad():
            motor.composition.token_rotations.copy_(quaternions[:, None])
            motor.composition.token_translations.copy_(translations[:, None])
            direct_product.rotation.token_rotors.copy_(quaternions[:, None])
            direct_product.token_translations.copy_(translations)
        inputs = torch.tensor([[0, 4, 1, 5, 2, 6, 0, 0]], dtype=torch.long)
        _, targets = rigid_prefix_targets(self.task, inputs.numpy())
        expected = torch.from_numpy(targets)
        parallel = motor(inputs, scan_mode="parallel")
        recurrent = motor(inputs, scan_mode="recurrent")
        self.assertTrue(torch.allclose(parallel, expected, atol=2e-12, rtol=2e-12))
        self.assertTrue(torch.allclose(recurrent, expected, atol=2e-12, rtol=2e-12))
        self.assertTrue(torch.allclose(parallel, recurrent, atol=2e-12, rtol=2e-12))
        ablation = direct_product(inputs, scan_mode="parallel")
        self.assertTrue(
            torch.allclose(ablation[..., :4], expected[..., :4], atol=2e-12)
        )
        self.assertGreater(
            float(
                torch.linalg.vector_norm(ablation[..., 4:] - expected[..., 4:]).detach()
            ),
            0.1,
        )

    def test_legal_prefix_identification_recovers_exact_token_motors(self) -> None:
        training = make_training_batches(self.task, self.config)
        model, audit = identify_direct_motor_from_prefixes(self.task, training)
        self.assertFalse(audit["uses_evaluation_data"])
        self.assertFalse(audit["uses_forbidden_relation_occurrences"])
        self.assertEqual(
            audit["total_prefix_observations"],
            self.config.steps * self.config.batch_size * self.config.training_length,
        )
        self.assertLess(audit["maximum_exact_token_quaternion_error_degrees"], 0.1)
        self.assertLess(audit["maximum_exact_token_translation_error"], 1e-6)
        relation = self.task.rotation.relations[-1]
        batches = make_relation_pair_batches(
            self.task, relation, self.config, 16, "early"
        )
        metrics = evaluate_relation_pairs(
            "direct_motor_pose_scan",
            model,
            batches,
            torch.device("cpu"),
            self.config,
            spin_scan_mode="parallel",
            motor_scan_mode="parallel",
            delta_scan_mode="parallel",
        )
        self.assertEqual(metrics["signed_rotation_threshold_accuracy"], 1.0)
        self.assertEqual(metrics["translation_threshold_accuracy"], 1.0)
        self.assertEqual(metrics["paired_double_cover_pose_accuracy"], 1.0)

    def test_parameter_and_state_contracts(self) -> None:
        models = build_models(self.config)
        counts = {name: parameter_count(model) for name, model in models.items()}
        for group in (
            ("direct_product_pose_scan", "direct_motor_pose_scan"),
            (
                "spin_quaternion_scan",
                "spin_motor_scan",
                "mamba2_transformers",
                "delta_product_reference",
            ),
        ):
            values = [counts[name] for name in group]
            self.assertLessEqual(
                (max(values) - min(values)) / max(values),
                0.02,
            )
        self.assertEqual(counts["direct_product_pose_scan"], 49)
        self.assertEqual(counts["direct_motor_pose_scan"], 49)
        self.assertEqual(recurrent_state_scalars(self.config)["spin_motor_scan"], 32)
        self.assertEqual(
            recurrent_state_scalars(self.config)["spin_quaternion_scan"], 32
        )
        inputs = torch.tensor([[0, 1, 4, 2, 5, 6]], dtype=torch.long)
        for name in (
            "direct_product_pose_scan",
            "direct_motor_pose_scan",
            "spin_quaternion_scan",
            "spin_motor_scan",
        ):
            output = models[name](inputs, scan_mode="parallel")
            self.assertEqual(tuple(output.shape), (1, 6, 7))
            loss = output.square().mean()
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in models[name].parameters()
                if parameter.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(
                all(torch.isfinite(gradient).all() for gradient in gradients)
            )

    def test_frozen_artifacts_preserve_negative_and_positive_results(self) -> None:
        artifact_directory = Path(__file__).parent / "experiments" / "artifacts"
        reports = {}
        for filename, expected_hash in self.ARTIFACT_HASHES.items():
            path = artifact_directory / filename
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), expected_hash
            )
            reports[filename] = json.loads(path.read_text(encoding="utf-8"))

        learned = reports["spin_motor_rigid_2a5_pilot300.json"]
        self.assertTrue(
            all(
                not value["all_long_splits_joint_pose_gate_80pct"]
                for value in learned["gate_summary"].values()
            )
        )
        direct = reports["spin_motor_direct_readout_pilot300.json"]["gate_summary"]
        self.assertTrue(
            direct["direct_product_pose_scan"]["all_long_splits_center_gate_90pct"]
        )
        self.assertFalse(
            direct["direct_product_pose_scan"]["all_long_splits_joint_pose_gate_80pct"]
        )
        self.assertFalse(
            direct["direct_motor_pose_scan"]["all_long_splits_center_gate_90pct"]
        )
        identified = reports["spin_motor_identified_e_seed0.json"]
        self.assertTrue(identified["identification_gates"]["all_passed"])
        replicated = reports["spin_motor_identification_replication_3x3.json"]
        self.assertEqual(replicated["run_count"], 9)
        self.assertEqual(replicated["unique_training_schedule_hashes"], 9)
        self.assertTrue(replicated["all_runs_passed"])
        self.assertEqual(
            replicated["aggregate"]["minimum_joint_signed_pose_accuracy"], 1.0
        )
        self.assertEqual(
            replicated["aggregate"]["minimum_paired_double_cover_pose_accuracy"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
