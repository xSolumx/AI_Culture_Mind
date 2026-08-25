"""Contracts for the prospectively frozen G15A-F full-frame cohort."""

from __future__ import annotations

import torch

from hybrid_memory_v1_4.g15a_spin_dirac_cohort import _oracle_memory
from hybrid_memory_v1_4.g15af_full_frame_cohort import (
    ARM_NAMES,
    PROBE_COUNT,
    QUALITY_SEEDS,
    RETENTION,
    _adjudicate,
    _broken_lie_bracket_certificate,
    _frame_prediction,
    _observability_certificate,
    _probe_bank,
    generate_frame_batch,
    generate_singleton_inverse_batch,
    quality_config,
)


def test_quality_protocol_and_arm_set_are_frozen() -> None:
    config = quality_config()
    assert config.seeds == QUALITY_SEEDS == (2203, 2207, 2213)
    assert config.training_updates == 300
    assert config.training_length == 16
    assert config.evaluation_specs == ((64, 8), (256, 12), (1024, 16))
    assert PROBE_COUNT == 4
    assert ARM_NAMES == ("I", "C", "S", "S-broken")


def test_probe_bank_is_deterministic_orthogonal_and_seed_specific() -> None:
    first = _probe_bank(2203)
    second = _probe_bank(2203)
    other = _probe_bank(2207)
    eye = torch.eye(8, dtype=torch.float64).expand(PROBE_COUNT, 8, 8)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert torch.allclose(first.transpose(-1, -2) @ first, eye, atol=1e-12)


def test_batches_repeat_the_same_probe_bank_for_each_composition() -> None:
    batch = generate_frame_batch(
        3,
        16,
        seed=11,
        model_seed=2203,
        minimum_actions=2,
        maximum_actions=6,
    )
    expected = _probe_bank(2203).to(torch.float32)
    assert batch.initial_frames.shape == (3, PROBE_COUNT, 8, 8)
    assert torch.equal(batch.initial_frames[0], expected)
    assert torch.equal(batch.initial_frames[1], expected)
    assert (
        batch.fingerprint()
        == generate_frame_batch(
            3,
            16,
            seed=11,
            model_seed=2203,
            minimum_actions=2,
            maximum_actions=6,
        ).fingerprint()
    )


def test_frozen_probe_banks_and_broken_control_pass_pretraining_screens() -> None:
    for seed in QUALITY_SEEDS:
        certificate = _observability_certificate(seed)
        assert certificate["independent_carrier_jacobian_shape"] == [256, 56]
        assert certificate["passed"] is True
    bracket = _broken_lie_bracket_certificate()
    assert bracket["generator_pairs_checked"] == 28**2
    assert bracket["mismatch_count"] > 0
    assert bracket["witness"]["integer_residual_max_abs"] > 0
    assert bracket["passed"] is True


def test_event_sparse_full_frame_transport_matches_dense_recurrence_float64() -> None:
    device = torch.device("cpu")
    batch = generate_frame_batch(
        2,
        12,
        seed=23,
        model_seed=2203,
        minimum_actions=2,
        maximum_actions=5,
    ).to(device, torch.float64)
    memory = _oracle_memory("S", dtype=torch.float64, device=device)
    sparse = _frame_prediction(memory, batch, batch.exact_coordinates, device=device)

    batch_size, length = batch.token_ids.shape
    repeated = batch_size * PROBE_COUNT
    carrier = torch.zeros(repeated, length, 1, 8, dtype=torch.float64)
    gate = torch.zeros(repeated, length, 1, 1, dtype=torch.float64)
    retention = torch.full_like(gate, RETENTION)
    coordinates = (
        batch.exact_coordinates[:, None]
        .expand(-1, PROBE_COUNT, -1, -1, -1)
        .reshape(repeated, length, 1, 28)
    )
    initial_state = (batch.initial_frames / RETENTION).reshape(repeated, 1, 8, 8)
    _, dense_final = memory.forward_controls(
        carrier,
        carrier,
        carrier,
        gate,
        gate,
        retention,
        coordinates,
        initial_state,
        scan_mode="recurrent",
    )
    dense = dense_final[:, 0].reshape(batch_size, PROBE_COUNT, 8, 8)
    assert torch.allclose(sparse, dense, atol=1e-10, rtol=1e-10)


def test_singleton_inverse_diagnostic_covers_all_signed_primitives() -> None:
    batch = generate_singleton_inverse_batch(2203)
    assert batch.token_ids.shape == (24, 4)
    assert torch.all((batch.token_ids[:16] != 0).sum(dim=1) == 1)
    assert torch.all((batch.token_ids[16:] != 0).sum(dim=1) == 2)
    assert torch.allclose(
        batch.exact_coordinates[16:].sum(dim=1),
        torch.zeros_like(batch.exact_coordinates[16:, 0]),
    )


def _fake_arm(error: float) -> dict[str, object]:
    return {
        "trainable_parameters": 476,
        "initial_state_sha256": "same",
        "probe_bank_sha256": "same",
        "training_schedule_sha256": "same",
        "evaluation": {
            str(length): {
                "mean_relative_frobenius_error": error,
                "p95_relative_frobenius_error": error,
                "maximum_relative_frobenius_error": error,
            }
            for length in (64, 256, 1024)
        },
    }


def test_adjudicator_requires_absolute_accuracy_and_broken_separation() -> None:
    arms = {
        "I": _fake_arm(0.4),
        "C": _fake_arm(0.3),
        "S": _fake_arm(0.04),
        "S-broken": _fake_arm(0.12),
    }
    assert _adjudicate([{"seed": 1, "arms": arms}])["passed"] is True
    arms["S-broken"] = _fake_arm(0.04)
    assert _adjudicate([{"seed": 1, "arms": arms}])["passed"] is False
