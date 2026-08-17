from __future__ import annotations

import hashlib
import json
from pathlib import Path

import benchmark_pure_spin8_lift_bit_calibration as calibration
import benchmark_pure_spin8_scrambled_alignment as benchmark
import torch

ROOT = Path(__file__).resolve().parent


def test_zero_alignment_exactly_recovers_shared_tracker() -> None:
    torch.manual_seed(123)
    model = benchmark.ScrambledSharedLatentSO8TripletTracker(alignment_std=0.0)
    observations = torch.randn(3, 5, 12)
    expected = model.router(observations)
    actual = model(observations)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_scrambled_control_has_matched_state_and_only_56_extra_parameters() -> None:
    audit = benchmark.matched_initialization_audit(seed=0)
    assert audit["passed"]
    assert audit["parameter_counts"][benchmark.SCRAMBLED] == 986
    assert audit["parameter_counts"][benchmark.SHARED] == 930


def test_alignment_actions_are_orthogonal() -> None:
    torch.manual_seed(456)
    model = benchmark.ScrambledSharedLatentSO8TripletTracker()
    actions = model.alignment_actions().double()
    identity = torch.eye(8, dtype=torch.float64).expand_as(actions)
    assert torch.allclose(
        actions.transpose(-1, -2) @ actions,
        identity,
        atol=2e-6,
        rtol=2e-6,
    )


def test_adaptive_loss_updates_shared_head_but_not_negative_alignment() -> None:
    torch.manual_seed(789)
    model = benchmark.ScrambledSharedLatentSO8TripletTracker()
    observations = torch.randn(4, 6, 12)
    targets = torch.randn(4, 3, 8)
    predictions = model(observations)
    loss, _ = calibration.mode_loss(
        "vector_plus_adaptive_lift_bit", predictions, targets
    )
    loss.backward()
    assert model.spinor_alignment_coordinates.grad is not None
    assert torch.count_nonzero(model.spinor_alignment_coordinates.grad[0]) > 0
    assert torch.count_nonzero(model.spinor_alignment_coordinates.grad[1]) == 0
    assert model.router.coordinate_head.weight.grad is not None
    row_norms = torch.linalg.vector_norm(
        model.router.coordinate_head.weight.grad, dim=-1
    )
    assert torch.count_nonzero(row_norms) == 28


def test_full_triality_loss_updates_both_alignments() -> None:
    torch.manual_seed(987)
    model = benchmark.ScrambledSharedLatentSO8TripletTracker()
    observations = torch.randn(4, 6, 12)
    targets = torch.randn(4, 3, 8)
    predictions = model(observations)
    loss, _ = calibration.mode_loss("full_triality", predictions, targets)
    loss.backward()
    assert model.spinor_alignment_coordinates.grad is not None
    assert torch.count_nonzero(model.spinor_alignment_coordinates.grad[0]) > 0
    assert torch.count_nonzero(model.spinor_alignment_coordinates.grad[1]) > 0


def test_protocol_and_development_artifacts_are_content_locked() -> None:
    assert benchmark.PROTOCOL_FROZEN_AT == "2026-08-17T08:40:40.5270068+02:00"
    artifacts = ROOT / "experiments" / "artifacts"
    expected = {
        "pure_spin8_scrambled_alignment_development_seed0.json": (
            "327e862e883a9add78331542155ae327962f46fb665e31226ad1793ff9bdc0e8"
        ),
        "pure_spin8_scrambled_alignment_development_seed0_validated.json": (
            "7b19b034873fcccffa3cf25e0d7a013aa9eb44cb024d22b0bb60661c1d22bff4"
        ),
    }
    for name, digest in expected.items():
        path = artifacts / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        assert json.loads(path.read_text(encoding="utf-8"))["passed"]


def test_failed_fresh_cohort_is_content_locked_without_rescue() -> None:
    artifacts = ROOT / "experiments" / "artifacts"
    expected = {
        "pure_spin8_scrambled_alignment_validation_seed7.json": (
            "3bc885cbaa5264b3b8eaabe816d71ae36447a73a78a599eb8c08186640fd2992"
        ),
        "pure_spin8_scrambled_alignment_validation_seed8.json": (
            "fd7a12f19c24da821dfe5d41e74c7cb87425829ddbf794d4c7a46029dec6c5e0"
        ),
        "pure_spin8_scrambled_alignment_validation_seed9.json": (
            "b56e99b19f980bcff6f810066f14cad099a17cac9283a004c6580bf1253fe03b"
        ),
        "pure_spin8_scrambled_alignment_validation_seeds7_9.json": (
            "ec6802d9c55f318aa85aaacb9ce4030df697f716158a3bdc5752432394f044a7"
        ),
    }
    for name, digest in expected.items():
        path = artifacts / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest

    aggregate = json.loads(
        (artifacts / "pure_spin8_scrambled_alignment_validation_seeds7_9.json")
        .read_text(encoding="utf-8")
    )
    assert not aggregate["passed"]
    assert aggregate["seeds"] == [7, 8, 9]
    assert not aggregate["global_checks"][
        "every_frozen_gate_passes_without_median_rescue"
    ]
    assert not aggregate["frozen_seed_gates"]["7"][
        "shared_beats_scrambled_adaptive_every_l128_view"
    ]
    assert all(aggregate["frozen_seed_gates"][str(seed)]["shared_beats_scrambled_adaptive_every_l128_view"] for seed in (8, 9))
