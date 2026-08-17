"""Contracts for noisy continuous-observation Pure Spin(8) development."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import torch

from benchmark_pure_spin8_continuous_observation import (
    CANDIDATES,
    OBSERVATION_DIMENSION,
    ContinuousObservationConfig,
    IndependentSO8TripletTracker,
    SharedPureSpin8Tracker,
    build_models,
    make_observation_system,
    make_relation_batch,
    make_training_schedule,
    observe_coordinates,
    parameter_count,
    relation_batch_audit,
    teacher_contract,
    training_split_audit,
)
from validate_pure_spin8_continuous_observation import seed_gate_checks

ROOT = Path(__file__).resolve().parent
DEVELOPMENT_ARTIFACT = (
    ROOT
    / "experiments"
    / "artifacts"
    / "pure_spin8_continuous_observation_development_seed0.json"
)
DEVELOPMENT_ARTIFACT_SHA256 = (
    "64aed46a68bda6523690e14673000f1d916df51d62ccf99718526cf9afe67094"
)
VALIDATION_AGGREGATE = (
    ROOT
    / "experiments"
    / "artifacts"
    / "pure_spin8_continuous_observation_validation_seeds1_3.json"
)
VALIDATION_AGGREGATE_SHA256 = (
    "34238a1d98fa467e8f8f38b1f90d1a24bc2495cc510934b4327cba81e09ebc6c"
)


class PureSpin8ContinuousObservationTests(unittest.TestCase):
    def test_teacher_center_and_identity_contract(self) -> None:
        contract = teacher_contract(torch.device("cpu"))
        self.assertTrue(contract["passed"])
        for key, value in contract.items():
            if key != "passed":
                self.assertLess(value, 1e-6)

    def test_observation_chart_is_seeded_noisy_and_not_coordinate_exposure(
        self,
    ) -> None:
        system = make_observation_system(0)
        self.assertEqual(tuple(system.projection.shape), (7, OBSERVATION_DIMENSION))
        coordinates = torch.zeros(4, 28)
        coordinates[:, 0] = torch.tensor([-0.7, 0.2, 1.0, 3.1])
        first = observe_coordinates(
            coordinates,
            system,
            noise_std=0.01,
            generator=torch.Generator().manual_seed(1),
        )
        replay = observe_coordinates(
            coordinates,
            system,
            noise_std=0.01,
            generator=torch.Generator().manual_seed(1),
        )
        clean = observe_coordinates(
            coordinates,
            system,
            noise_std=0.0,
            generator=torch.Generator().manual_seed(1),
        )
        self.assertTrue(torch.equal(first, replay))
        self.assertFalse(torch.equal(first, clean))
        self.assertEqual(tuple(first.shape), (4, OBSERVATION_DIMENSION))
        self.assertEqual(torch.unique(first, dim=0).shape[0], 4)

    def test_training_has_unique_observations_and_excludes_half_center_pair(
        self,
    ) -> None:
        config = ContinuousObservationConfig(
            steps=16,
            batch_size=16,
            training_length=16,
            evaluation_pairs=4,
            evaluation_lengths=(16,),
        )
        schedule = make_training_schedule(
            config, make_observation_system(0), torch.device("cpu")
        )
        audit = training_split_audit(schedule)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["held_out_adjacent_half_center_count"], 0)
        self.assertEqual(
            audit["unique_observation_count"], audit["observation_count"]
        )

    def test_early_and_late_relation_batches_have_exact_target_signature(
        self,
    ) -> None:
        config = ContinuousObservationConfig(
            steps=1,
            batch_size=2,
            training_length=4,
            evaluation_pairs=4,
            evaluation_lengths=(16,),
        )
        system = make_observation_system(0)
        for position in ("early", "late"):
            with self.subTest(position=position):
                batch = make_relation_batch(
                    config, system, 16, position, torch.device("cpu")
                )
                self.assertTrue(relation_batch_audit(batch)["passed"])
                center = batch.targets[0::2, -1]
                identity = batch.targets[1::2, -1]
                self.assertLess(
                    float((center[:, 0] - identity[:, 0]).abs().max()), 2e-6
                )
                self.assertLess(
                    float((center[:, 1:] + identity[:, 1:]).abs().max()), 2e-6
                )

    def test_candidate_counts_state_axes_and_shapes(self) -> None:
        models = build_models()
        self.assertEqual(tuple(models), CANDIDATES)
        counts = {name: parameter_count(model) for name, model in models.items()}
        self.assertEqual(
            counts,
            {
                "shared_pure_spin8": 930,
                "independent_so8_triplet": 957,
                "mamba2_parameter_near": 931,
                "gru_parameter_near": 960,
                "observation_only_ablation": 949,
                "gru_state_matched": 3312,
            },
        )
        self.assertEqual(models["shared_pure_spin8"].recurrent_state_scalars, 24)
        self.assertEqual(
            models["independent_so8_triplet"].recurrent_state_scalars, 24
        )
        self.assertEqual(models["gru_state_matched"].recurrent_state_scalars, 24)
        observations = torch.randn(2, 5, OBSERVATION_DIMENSION)
        for name, model in models.items():
            with self.subTest(name=name):
                self.assertEqual(tuple(model(observations).shape), (2, 5, 3, 8))

    def test_structured_routers_receive_finite_gradients(self) -> None:
        observations = torch.randn(2, 5, OBSERVATION_DIMENSION)
        for model in (SharedPureSpin8Tracker(), IndependentSO8TripletTracker()):
            with self.subTest(model=type(model).__name__):
                loss = model(observations).square().mean()
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
        payload = DEVELOPMENT_ARTIFACT.read_bytes()
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(), DEVELOPMENT_ARTIFACT_SHA256
        )
        report = json.loads(payload)
        checks = seed_gate_checks(report)
        self.assertTrue(checks)
        self.assertTrue(all(checks.values()))

    def test_fresh_validation_aggregate_is_content_locked_and_passed(self) -> None:
        payload = VALIDATION_AGGREGATE.read_bytes()
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(), VALIDATION_AGGREGATE_SHA256
        )
        report = json.loads(payload)
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["cohort_checks"].values()))
        self.assertEqual(len(report["seed_reports"]), 3)
        self.assertTrue(all(seed["passed"] for seed in report["seed_reports"]))


if __name__ == "__main__":
    unittest.main()
