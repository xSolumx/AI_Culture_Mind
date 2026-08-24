"""Acceptance gates for the isolated experimental structured tier."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selected_block import LowRankLinear
from structured_tier import (
    StructuredSpin8Tier,
    StructuredTierConfig,
    cumulative_generator_gates,
    generator_mask_for_rung,
    rung_complexity_regularizer,
    rung_entropy_regularizer,
    rung_generator_masks,
    subgroup_generator_indices,
)

DTYPE = torch.float64


def test_exact_global_generator_index_sets() -> None:
    assert subgroup_generator_indices(3) == (0, 1, 7)
    assert subgroup_generator_indices(4) == (0, 1, 2, 7, 8, 13)
    assert subgroup_generator_indices(6) == (
        0,
        1,
        2,
        3,
        4,
        7,
        8,
        9,
        10,
        13,
        14,
        15,
        18,
        19,
        22,
    )
    assert subgroup_generator_indices(8) == tuple(range(28))
    assert [len(subgroup_generator_indices(rung)) for rung in (3, 4, 6, 8)] == [
        3,
        6,
        15,
        28,
    ]


def test_hard_rung_masks_are_exact_and_nested() -> None:
    masks = rung_generator_masks()
    assert masks.dtype == torch.bool
    assert masks.shape == (4, 28)
    assert masks.sum(dim=-1).tolist() == [3, 6, 15, 28]
    for index, rung in enumerate((3, 4, 6, 8)):
        expected = torch.zeros(28, dtype=torch.bool)
        expected[list(subgroup_generator_indices(rung))] = True
        assert torch.equal(masks[index], expected)
        assert torch.equal(generator_mask_for_rung(rung), expected)
    assert torch.all(masks[:-1] <= masks[1:])


def test_soft_masks_are_cumulative_probability_mass() -> None:
    probabilities = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=DTYPE)
    gates = cumulative_generator_gates(probabilities)
    masks = rung_generator_masks()
    assert torch.allclose(gates[:, masks[0]], torch.ones(1, 3, dtype=DTYPE))
    assert torch.allclose(
        gates[:, masks[1] & ~masks[0]], torch.full((1, 3), 0.9, dtype=DTYPE)
    )
    assert torch.allclose(
        gates[:, masks[2] & ~masks[1]], torch.full((1, 9), 0.7, dtype=DTYPE)
    )
    assert torch.allclose(
        gates[:, masks[3] & ~masks[2]], torch.full((1, 13), 0.4, dtype=DTYPE)
    )


def test_training_probabilities_are_normalized_and_labeled_continuous() -> None:
    torch.manual_seed(1)
    tier = StructuredSpin8Tier(
        StructuredTierConfig(model_dim=5, channels=2, temperature=0.7)
    ).double()
    tier.train()
    output = tier(torch.randn(3, 4, 5, dtype=DTYPE))
    probabilities = output.diagnostics["rung_probabilities"]
    assert isinstance(probabilities, torch.Tensor)
    assert probabilities.shape == (3, 4, 2, 4)
    assert torch.all(probabilities > 0)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(3, 4, 2, dtype=DTYPE))
    assert output.coordinates.shape == (3, 4, 2, 28)
    assert output.actions.shape == (3, 4, 2, 3, 8, 8)
    assert output.diagnostics["rung_regime"] == "continuous_regularization"
    assert output.diagnostics["soft_subgroup_claim"] == (
        "none_soft_mixtures_are_not_discrete_subgroups"
    )


def test_hard_eval_uses_one_exact_rung_per_token_and_channel() -> None:
    tier = StructuredSpin8Tier(
        StructuredTierConfig(model_dim=3, channels=1, hard_eval=True)
    ).double()
    with torch.no_grad():
        tier.rung_controller.weight.zero_()
        tier.rung_controller.bias.copy_(torch.tensor([0.0, 1.0, 5.0, 2.0]))
        assert isinstance(tier.coefficient_controller, torch.nn.Linear)
        tier.coefficient_controller.weight.zero_()
        tier.coefficient_controller.bias.fill_(1.0)
    tier.eval()
    output = tier(torch.randn(2, 5, 3, dtype=DTYPE))
    probabilities = output.diagnostics["rung_probabilities"]
    gates = output.diagnostics["generator_gates"]
    selected = output.diagnostics["selected_rung"]
    expected_mask = generator_mask_for_rung(6).to(dtype=DTYPE)
    assert isinstance(probabilities, torch.Tensor)
    assert isinstance(gates, torch.Tensor)
    assert isinstance(selected, torch.Tensor)
    assert torch.equal(
        probabilities, torch.tensor([0.0, 0.0, 1.0, 0.0]).expand_as(probabilities)
    )
    assert torch.equal(gates, expected_mask.expand_as(gates))
    assert torch.equal(output.coordinates, expected_mask.expand_as(output.coordinates))
    assert torch.equal(selected, torch.full_like(selected, 6))
    assert torch.equal(
        output.diagnostics["expected_factor_count"],
        torch.full((2, 5, 1), 15.0, dtype=DTYPE),
    )
    assert output.diagnostics["hard_selection"] is True
    assert output.diagnostics["rung_regime"] == "exact_rung"


def test_gradients_reach_rung_and_coordinate_controllers() -> None:
    torch.manual_seed(2)
    tier = StructuredSpin8Tier(
        StructuredTierConfig(model_dim=6, channels=2, hard_eval=True)
    ).double()
    tier.train()
    inputs = torch.randn(2, 3, 6, dtype=DTYPE, requires_grad=True)
    output = tier(inputs)
    expected = output.diagnostics["expected_factor_count"]
    assert isinstance(expected, torch.Tensor)
    probe = torch.randn_like(output.actions)
    loss = (
        (output.actions * probe).sum()
        + output.coordinates.square().sum()
        + expected.sum()
    )
    loss.backward()
    assert inputs.grad is not None and torch.count_nonzero(inputs.grad) > 0
    assert tier.rung_controller.weight.grad is not None
    assert torch.count_nonzero(tier.rung_controller.weight.grad) > 0
    for parameter in tier.coefficient_controller.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) > 0


def test_actions_preserve_orthogonality_and_two_pi_center() -> None:
    torch.manual_seed(3)
    tier = StructuredSpin8Tier(
        StructuredTierConfig(model_dim=4, channels=1, hard_eval=True)
    ).double()
    tier.train()
    random_output = tier(torch.randn(2, 3, 4, dtype=DTYPE))
    identity = torch.eye(8, dtype=DTYPE)
    orthogonality = random_output.actions.transpose(-1, -2) @ random_output.actions
    assert torch.allclose(orthogonality, identity, atol=2e-14, rtol=0.0)

    with torch.no_grad():
        tier.rung_controller.weight.zero_()
        tier.rung_controller.bias.copy_(torch.tensor([5.0, 0.0, 0.0, 0.0]))
        assert isinstance(tier.coefficient_controller, torch.nn.Linear)
        tier.coefficient_controller.weight.zero_()
        tier.coefficient_controller.bias.zero_()
        tier.coefficient_controller.bias[0] = 2.0 * math.pi
    tier.eval()
    center = tier(torch.zeros(1, 1, 4, dtype=DTYPE)).actions[0, 0, 0]
    assert torch.allclose(center[0], identity, atol=2e-14, rtol=0.0)
    assert torch.allclose(center[1], -identity, atol=2e-14, rtol=0.0)
    assert torch.allclose(center[2], -identity, atol=2e-14, rtol=0.0)


def test_regularizers_are_pure_tensor_functions() -> None:
    probabilities = torch.tensor(
        [[0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0]],
        dtype=DTYPE,
        requires_grad=True,
    )
    entropy = rung_entropy_regularizer(probabilities, reduction="none")
    complexity = rung_complexity_regularizer(probabilities, reduction="none")
    assert torch.allclose(entropy, torch.tensor([math.log(4.0), 0.0], dtype=DTYPE))
    assert torch.allclose(complexity, torch.tensor([13.0, 3.0], dtype=DTYPE))
    (entropy.mean() + complexity.mean()).backward()
    assert probabilities.grad is not None


def test_low_rank_coefficient_controller_stays_rank_constrained() -> None:
    torch.manual_seed(4)
    config = StructuredTierConfig(model_dim=7, channels=2, controller_rank=3)
    tier = StructuredSpin8Tier(config).double()
    controller = tier.coefficient_controller
    assert isinstance(controller, LowRankLinear)
    assert "weight" not in dict(controller.named_parameters())
    expected_parameters = 3 * (7 + 56) + 56
    assert (
        sum(parameter.numel() for parameter in controller.parameters())
        == expected_parameters
    )
    optimizer = torch.optim.SGD(tier.parameters(), lr=0.01)
    output = tier(torch.randn(2, 3, 7, dtype=DTYPE))
    output.coordinates.square().mean().backward()
    optimizer.step()
    assert torch.linalg.matrix_rank(controller.effective_weight()).item() <= 3


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_dim": 0},
        {"channels": 0},
        {"rungs": (3, 3, 8)},
        {"rungs": (3, 9)},
        {"temperature": 0.0},
        {"hard_eval": 1},
        {"controller_rank": 0},
        {"controller_rank": 5, "model_dim": 4},
    ],
)
def test_config_guards(overrides: dict[str, object]) -> None:
    values: dict[str, object] = {"model_dim": 4}
    values.update(overrides)
    with pytest.raises((TypeError, ValueError)):
        StructuredTierConfig(**values)  # type: ignore[arg-type]
