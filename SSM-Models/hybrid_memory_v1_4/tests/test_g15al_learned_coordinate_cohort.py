"""Contracts for the prospectively frozen G15A-L coordinate cohort."""

from __future__ import annotations

import torch

from hybrid_memory_v1_4.g15al_learned_coordinate_cohort import (
    ACTION_ANGLE,
    ARM_NAMES,
    QUALITY_SEEDS,
    TokenCoordinateController,
    _adjudicate,
    generate_batch,
    quality_config,
)


def test_quality_protocol_and_arm_set_are_frozen() -> None:
    config = quality_config()
    assert config.seeds == QUALITY_SEEDS == (2153, 2161, 2179)
    assert config.training_updates == 300
    assert config.training_length == 16
    assert config.evaluation_specs == ((64, 8), (256, 12), (1024, 16))
    assert ARM_NAMES == ("I", "C", "S", "S+identity-read", "S-broken")


def test_batches_are_deterministic_sparse_and_hide_the_token_map() -> None:
    first = generate_batch(
        4,
        16,
        seed=11,
        model_seed=2153,
        minimum_actions=2,
        maximum_actions=6,
    )
    second = generate_batch(
        4,
        16,
        seed=11,
        model_seed=2153,
        minimum_actions=2,
        maximum_actions=6,
    )
    assert first.fingerprint() == second.fingerprint()
    assert torch.equal(first.token_ids, second.token_ids)
    active = first.token_ids != 0
    assert torch.all(first.exact_coordinates[~active].eq(0))
    magnitudes = first.exact_coordinates[active].abs().sum(dim=(-2, -1))
    assert torch.allclose(magnitudes, torch.full_like(magnitudes, ACTION_ANGLE))


def test_controller_hard_masks_filler_and_has_exact_parameter_budget() -> None:
    controller = TokenCoordinateController()
    tokens = torch.tensor([[0, 1, 16]])
    coordinates = controller(tokens)
    assert sum(parameter.numel() for parameter in controller.parameters()) == 476
    assert torch.equal(coordinates[:, 0], torch.zeros_like(coordinates[:, 0]))


def _fake_arm(mean: float, minimum: float, raw: float = 0.0) -> dict[str, object]:
    return {
        "trainable_parameters": 476,
        "initial_state_sha256": "same",
        "training_schedule_sha256": "same",
        "learned_raw_coordinates": [[raw] * 28 for _ in range(17)],
        "evaluation": {
            str(length): {"mean_cosine": mean, "minimum_cosine": minimum}
            for length in (64, 256, 1024)
        },
    }


def test_adjudicator_rejects_a_broken_arm_that_absorbs_the_chart() -> None:
    arms = {
        "I": _fake_arm(0.5, 0.4),
        "C": _fake_arm(0.6, 0.5),
        "S": _fake_arm(0.999, 0.99),
        "S+identity-read": _fake_arm(0.999, 0.99),
        "S-broken": _fake_arm(0.7, 0.6),
    }
    passing = _adjudicate([{"seed": 1, "arms": arms}])
    assert passing["passed"] is True
    arms["S-broken"] = _fake_arm(0.999, 0.99)
    absorbed = _adjudicate([{"seed": 1, "arms": arms}])
    assert absorbed["passed"] is False
