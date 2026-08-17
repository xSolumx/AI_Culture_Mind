"""Contracts for the latent-increment Pure Spin(8) development harness."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import torch

from benchmark_pure_spin8_latent_increment import (
    CANDIDATES,
    LatentIncrementConfig,
    LatentPureSpin8Tracker,
    build_models,
    make_relation_batches,
    make_training_schedule,
    parameter_count,
    relation_batch_audit,
    teacher_contract,
    training_split_audit,
)
from validate_pure_spin8_latent_increment import seed_gate_checks

ROOT = Path(__file__).resolve().parent
VALIDATION_ARTIFACT = (
    ROOT
    / "experiments"
    / "artifacts"
    / "pure_spin8_latent_increment_validation_seeds1_3.json"
)
VALIDATION_ARTIFACT_SHA256 = (
    "c3d49145fb710c43aa087262212e4005f887995ffb67f399a49afb57e8ae51a2"
)


class PureSpin8LatentIncrementTests(unittest.TestCase):
    def test_teacher_center_and_identity_relations(self) -> None:
        contract = teacher_contract(torch.device("cpu"))
        self.assertTrue(contract["passed"])
        for key, value in contract.items():
            if key != "passed":
                self.assertLess(value, 1e-6)

    def test_training_schedule_excludes_only_held_pair_and_covers_rest(self) -> None:
        config = LatentIncrementConfig(
            steps=16,
            batch_size=16,
            training_length=16,
            evaluation_pairs=4,
            evaluation_lengths=(16,),
        )
        schedule = make_training_schedule(config, torch.device("cpu"))
        audit = training_split_audit(schedule)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["held_out_pair_count"], 0)
        self.assertGreater(audit["minimum_allowed_pair_count"], 0)

    def test_early_and_late_relation_batches_are_paired_and_center_sensitive(self) -> None:
        config = LatentIncrementConfig(
            steps=1,
            batch_size=2,
            training_length=4,
            evaluation_pairs=4,
            evaluation_lengths=(16,),
        )
        for position in ("early", "late"):
            with self.subTest(position=position):
                batch = make_relation_batches(
                    config, 16, position, torch.device("cpu")
                )[0]
                self.assertTrue(relation_batch_audit(batch)["passed"])
                center = batch.targets[0::2, -1]
                identity = batch.targets[1::2, -1]
                self.assertLess(
                    float((center[:, 0] - identity[:, 0]).abs().max()), 2e-6
                )
                self.assertLess(
                    float((center[:, 1:] + identity[:, 1:]).abs().max()),
                    2e-6,
                )

    def test_candidate_parameter_counts_are_near_and_shapes_match(self) -> None:
        models = build_models()
        self.assertEqual(tuple(models), CANDIDATES)
        counts = {name: parameter_count(model) for name, model in models.items()}
        self.assertEqual(
            counts,
            {
                "latent_pure_spin8": 892,
                "mamba2_transformers": 891,
                "gru_reference": 887,
                "token_only_ablation": 874,
            },
        )
        tokens = torch.randint(8, (2, 5))
        for name, model in models.items():
            with self.subTest(name=name):
                self.assertEqual(tuple(model(tokens).shape), (2, 5, 3, 8))

    def test_latent_spin8_router_and_scan_receive_finite_gradients(self) -> None:
        model = LatentPureSpin8Tracker()
        tokens = torch.tensor([[0, 1, 3, 0, 2], [4, 5, 6, 7, 1]])
        outputs = model(tokens)
        loss = outputs.square().mean()
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in gradients)
        )

    def test_frozen_gates_accept_the_excluded_development_seed(self) -> None:
        artifact = (
            ROOT
            / "experiments"
            / "artifacts"
            / "pure_spin8_latent_increment_development_seed0.json"
        )
        report = json.loads(artifact.read_text(encoding="utf-8"))
        checks = seed_gate_checks(report)
        self.assertTrue(checks)
        self.assertTrue(all(checks.values()))

    def test_fresh_validation_artifact_and_all_twelve_checkpoints(self) -> None:
        payload = VALIDATION_ARTIFACT.read_bytes()
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(), VALIDATION_ARTIFACT_SHA256
        )
        report = json.loads(payload)
        self.assertTrue(report["passed"])
        self.assertEqual(report["fresh_seeds"], [1, 2, 3])
        self.assertTrue(all(report["cohort_checks"].values()))
        checkpoint_count = 0
        for seed_report in report["seed_reports"]:
            self.assertTrue(seed_report["passed"])
            self.assertTrue(all(seed_report["checks"].values()))
            pure = seed_report["results"]["latent_pure_spin8"]
            self.assertLessEqual(pure["action_identification"]["action_rmse"], 0.02)
            for key in ("early_L128", "late_L128"):
                metrics = pure["evaluation"][key]
                self.assertEqual(metrics["center_classification_accuracy"], 1.0)
                self.assertEqual(metrics["center_rows_correct"], 1.0)
                self.assertEqual(metrics["identity_rows_correct"], 1.0)
                self.assertLessEqual(metrics["post_relation_mse"], 0.002)
            for record in seed_report["results"].values():
                checkpoint_count += 1
                checkpoint = ROOT / Path(record["checkpoint"])
                self.assertEqual(
                    hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    record["checkpoint_sha256"],
                )
                checkpoint_payload = torch.load(
                    checkpoint, map_location="cpu", weights_only=False
                )
                self.assertTrue(checkpoint_payload["state_dict"])
        self.assertEqual(checkpoint_count, 12)


if __name__ == "__main__":
    unittest.main()
