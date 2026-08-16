"""Structural and numerical tests for the center-sensitive ``2.A5`` benchmark."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import torch
from benchmark_pure_rotor_2a5 import (
    ORACLE_CANDIDATES,
    BinaryA5BenchmarkConfig,
    binary_icosahedral_task,
    build_models,
    central_pair_evaluation_audit,
    evaluate_central_pairs,
    logits_for,
    make_central_pair_evaluation_batches,
    make_training_batches,
    mamba2_model,
    parameter_count,
    pure_rotor_model,
    quaternion_prefix_scan,
    training_split_audit,
)


class PureRotorBinaryA5BenchmarkTests(unittest.TestCase):
    def test_frozen_pilot300_artifact_contract_and_registered_gate(self) -> None:
        artifact_path = (
            Path(__file__).parent
            / "experiments"
            / "artifacts"
            / "pure_rotor_2a5_center_pilot300.json"
        )
        artifact_bytes = artifact_path.read_bytes()
        self.assertEqual(
            hashlib.sha256(artifact_bytes).hexdigest(),
            "911815d9e104fa08e632161f97f41a966991a9102c70ca65e52a5f07d28d4476",
        )
        report = json.loads(artifact_bytes)
        self.assertEqual(report["seeds"], [0, 1, 2])
        self.assertEqual(len(report["results"]), 12)
        expected_pairs = {
            (seed, candidate)
            for seed in range(3)
            for candidate in (
                "pure_rotor",
                "identity_rotation_ablation",
                "spin_quaternion_scan",
                "mamba2_transformers",
            )
        }
        self.assertEqual(
            {(row["seed"], row["name"]) for row in report["results"]},
            expected_pairs,
        )
        for seed in range(3):
            split = report["task"]["split"][str(seed)]
            training = split["training"]
            self.assertEqual(training["training_pair_audit"]["pair_occurrences"], 0)
            self.assertEqual(training["observed_binary_target_states"], 120)
            self.assertEqual(training["observed_projective_target_states"], 60)
            self.assertEqual(training["observed_center_bits"], [0, 1])
            for audit in split["evaluations"].values():
                self.assertTrue(audit["passed"])
                self.assertEqual(len(audit["evaluation_schedule_sha256"]), 64)

        spin_rows = [
            row for row in report["results"] if row["name"] == "spin_quaternion_scan"
        ]
        self.assertEqual(len(spin_rows), 3)
        for row in spin_rows:
            for length in (16, 64, 128):
                metrics = row["final_center_pair_metrics"][f"early_L{length}"]
                self.assertGreater(
                    metrics["post_relation_central_margin_accuracy"], 0.75
                )

    def test_binary_presentation_training_language_and_pair_split(self) -> None:
        task = binary_icosahedral_task()
        presentation = task.presentation
        self.assertEqual(task.group.order, 120)
        self.assertEqual(presentation["projective_group_order"], 60)
        self.assertTrue(presentation["a_squared_is_center"])
        self.assertTrue(presentation["b_cubed_is_center"])
        self.assertTrue(presentation["ab_to_fifth_is_center"])
        self.assertTrue(presentation["center_squared_is_identity"])
        self.assertTrue(presentation["b_times_b_inverse_is_identity"])
        self.assertEqual(presentation["generated_subgroup_order"], 120)
        self.assertEqual(
            task.group_table_sha256,
            "ee26aff5719e54cf28eccc7a0259c79eeb27c7e06bb18134bf7ca910773f58c9",
        )
        self.assertTrue(
            all(
                task.central_partner[task.central_partner[state]] == state
                for state in range(120)
            )
        )

        config = BinaryA5BenchmarkConfig(
            steps=4,
            batch_size=32,
            training_length=16,
            validation_batches=1,
            validation_pairs_per_batch=4,
            evaluation_lengths=(2, 16),
        )
        training = make_training_batches(task, config)
        audit = training_split_audit(task, training)
        self.assertEqual(audit["training_pair_audit"]["pair_occurrences"], 0)
        self.assertEqual(audit["observed_binary_target_states"], 120)
        self.assertEqual(audit["observed_projective_target_states"], 60)
        self.assertEqual(audit["observed_center_bits"], [0, 1])
        self.assertTrue(
            audit["exact_training_language_coverage"]["all_binary_states_reachable"]
        )
        self.assertLessEqual(
            audit["exact_training_language_coverage"][
                "maximum_shortest_witness_length"
            ],
            16,
        )

        for length in config.evaluation_lengths:
            positions = ("early",) if length == 2 else ("early", "late")
            for position in positions:
                batches = make_central_pair_evaluation_batches(
                    task, config, length, position
                )
                pair_audit = central_pair_evaluation_audit(task, batches)
                self.assertTrue(pair_audit["passed"])
                self.assertEqual(
                    pair_audit["exact_central_partner_checks"],
                    pair_audit["post_relation_pair_positions"],
                )
                self.assertEqual(
                    pair_audit["forced_center_relation_checks"],
                    pair_audit["paired_sequences"],
                )
                self.assertEqual(
                    pair_audit["forced_identity_relation_checks"],
                    pair_audit["paired_sequences"],
                )
                self.assertEqual(len(pair_audit["evaluation_schedule_sha256"]), 64)

    def test_parameter_near_models_emit_120_state_logits(self) -> None:
        task = binary_icosahedral_task()
        config = BinaryA5BenchmarkConfig(
            steps=1,
            batch_size=2,
            training_length=5,
            validation_batches=1,
            validation_pairs_per_batch=2,
            evaluation_lengths=(2,),
        )
        torch.manual_seed(2718)
        models = build_models(task, config)
        counts = {name: parameter_count(model) for name, model in models.items()}
        self.assertEqual(
            counts,
            {
                "pure_rotor": 29370,
                "identity_rotation_ablation": 29370,
                "spin_quaternion_scan": 29592,
                "mamba2_transformers": 29300,
            },
        )
        self.assertLess(
            (max(counts.values()) - min(counts.values())) / max(counts.values()), 0.02
        )
        tokens = torch.randint(0, 3, (2, 5))
        for name, model in models.items():
            logits = logits_for(name, model.eval(), tokens, "parallel", "parallel")
            self.assertEqual(tuple(logits.shape), (2, 5, 120))
            self.assertTrue(bool(torch.isfinite(logits).all()))

        torch.manual_seed(17)
        pure = pure_rotor_model(task, config, max_rotor_angle=torch.pi)
        torch.manual_seed(17)
        identity = pure_rotor_model(task, config, max_rotor_angle=0.0)
        self.assertTrue(
            torch.equal(
                pure(tokens, scan_mode="parallel"),
                identity(tokens, scan_mode="parallel"),
            )
        )

    def test_mamba_shape_constraint_remains_explicit(self) -> None:
        task = binary_icosahedral_task()
        with self.assertRaisesRegex(ValueError, "2\\*hidden_size"):
            mamba2_model(task, BinaryA5BenchmarkConfig(mamba_hidden_size=31))

    def test_quaternion_parallel_scan_matches_recurrence_and_gradients(self) -> None:
        torch.manual_seed(31415)
        recurrent_input = torch.randn(2, 17, 3, 4, requires_grad=True)
        parallel_input = recurrent_input.detach().clone().requires_grad_(True)
        recurrent = quaternion_prefix_scan(recurrent_input, mode="recurrent")
        parallel = quaternion_prefix_scan(parallel_input, mode="parallel")
        self.assertTrue(torch.allclose(recurrent, parallel, atol=2e-6, rtol=2e-6))
        weights = torch.randn_like(recurrent)
        (recurrent * weights).sum().backward()
        (parallel * weights).sum().backward()
        self.assertIsNotNone(recurrent_input.grad)
        self.assertIsNotNone(parallel_input.grad)
        self.assertTrue(
            torch.allclose(
                recurrent_input.grad,
                parallel_input.grad,
                atol=2e-5,
                rtol=2e-5,
            )
        )

    def test_oracles_separate_binary_accuracy_from_projective_accuracy(self) -> None:
        task = binary_icosahedral_task()
        config = BinaryA5BenchmarkConfig(
            steps=1,
            batch_size=2,
            training_length=4,
            validation_batches=1,
            validation_pairs_per_batch=8,
            evaluation_lengths=(16,),
        )
        batches = make_central_pair_evaluation_batches(task, config, 16, "early")
        metrics = {
            name: evaluate_central_pairs(
                name,
                None,
                batches,
                task,
                torch.device("cpu"),
                "parallel",
                "parallel",
                3,
            )
            for name in ORACLE_CANDIDATES
        }
        for name in ("exact_table_oracle", "float64_quaternion_oracle"):
            self.assertEqual(metrics[name]["post_relation_exact_accuracy"], 1.0)
            self.assertEqual(metrics[name]["post_relation_center_bit_accuracy"], 1.0)
            self.assertEqual(metrics[name]["paired_final_exact_accuracy"], 1.0)
        projective = metrics["projective_a5_oracle"]
        self.assertEqual(projective["post_relation_projective_accuracy"], 1.0)
        self.assertEqual(projective["post_relation_exact_accuracy"], 0.5)
        self.assertEqual(projective["post_relation_center_bit_accuracy"], 0.5)
        self.assertEqual(projective["post_relation_central_margin_accuracy"], 0.5)
        self.assertEqual(projective["paired_final_exact_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
