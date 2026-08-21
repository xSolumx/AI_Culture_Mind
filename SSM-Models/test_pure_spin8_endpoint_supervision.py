"""Contracts for endpoint-only continuous Spin(8) identification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

import benchmark_pure_spin8_continuous_observation as continuous
from benchmark_pure_spin8_endpoint_supervision import (
    EndpointTrainingBatch,
    endpoint_loss,
    endpoint_training_split_audit,
    make_endpoint_training_schedule,
    parse_candidates,
)
from validate_pure_spin8_endpoint_supervision import seed_gate_checks

ROOT = Path(__file__).resolve().parent
DEVELOPMENT_ARTIFACT = (
    ROOT
    / "experiments"
    / "artifacts"
    / "pure_spin8_endpoint_supervision_development_seed0.json"
)
DEVELOPMENT_ARTIFACT_SHA256 = (
    "87626dd9ac5a4f81695999a8832cb6eb3fb58ef312e86edfe642d1b54163a1d2"
)
VALIDATION_AGGREGATE = (
    ROOT
    / "experiments"
    / "artifacts"
    / "pure_spin8_endpoint_supervision_validation_seeds1_3.json"
)
VALIDATION_AGGREGATE_SHA256 = (
    "1cf51a4af05303bc3ca9e781478e2352e8dbb077d1c9b367f46af2f384653880"
)


def test_endpoint_loss_has_no_intermediate_gradient_support() -> None:
    predictions = torch.randn(2, 5, 3, 8, requires_grad=True)
    targets = torch.randn(2, 3, 8)
    endpoint_loss(predictions, targets).backward()
    assert predictions.grad is not None
    assert torch.count_nonzero(predictions.grad[:, :-1]) == 0
    assert torch.count_nonzero(predictions.grad[:, -1]) > 0


def test_training_batch_schema_cannot_hold_prefix_targets() -> None:
    assert "targets" not in EndpointTrainingBatch.__dataclass_fields__
    assert tuple(EndpointTrainingBatch.__dataclass_fields__) == (
        "observations",
        "endpoint_targets",
        "coordinates",
        "events",
    )


def test_endpoint_schedule_is_unique_finite_and_excludes_relation() -> None:
    config = continuous.ContinuousObservationConfig(
        steps=8,
        batch_size=8,
        training_length=8,
        evaluation_pairs=2,
        evaluation_lengths=(8,),
    )
    schedule = make_endpoint_training_schedule(
        config,
        continuous.make_observation_system(0),
        torch.device("cpu"),
    )
    audit = endpoint_training_split_audit(schedule)
    assert audit["passed"]
    assert audit["retained_intermediate_target_count"] == 0
    assert audit["supervised_scalars_per_sequence"] == 24
    assert audit["unique_observation_count"] == audit["observation_count"]
    assert audit["held_out_adjacent_half_center_count"] == 0


def test_candidate_parser_preserves_declared_order() -> None:
    assert parse_candidates("shared_pure_spin8, independent_so8_triplet") == (
        "shared_pure_spin8",
        "independent_so8_triplet",
    )


def test_frozen_gates_accept_content_locked_development_seed() -> None:
    payload = DEVELOPMENT_ARTIFACT.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == DEVELOPMENT_ARTIFACT_SHA256
    checks = seed_gate_checks(json.loads(payload))
    assert checks
    assert all(checks.values())


def test_fresh_validation_aggregate_is_content_locked_and_passed() -> None:
    payload = VALIDATION_AGGREGATE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == VALIDATION_AGGREGATE_SHA256
    report = json.loads(payload)
    assert report["passed"]
    assert all(report["cohort_checks"].values())
    assert all(seed["passed"] for seed in report["seed_reports"])
    assert len(report["seed_reports"]) == 3
