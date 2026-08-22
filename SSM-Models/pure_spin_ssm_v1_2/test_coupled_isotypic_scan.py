from __future__ import annotations

import copy

import torch

from pure_spin_ssm_v1_2.coupled_isotypic_scan import (
    CoupledIsotypicTransition,
    apply_coupled_transition,
    compose_coupled_transition,
    contractive_givens_left,
    parallel_coupled_scan,
    recurrent_coupled_scan,
)


def _random_transition(*shape: int, dtype: torch.dtype) -> CoupledIsotypicTransition:
    channels, representations, dimension = 2, 3, 8
    scale = 0.7 + 0.2 * torch.rand(*shape, channels, dtype=dtype)
    angles = 0.4 * torch.randn(*shape, 1, dtype=dtype)
    left = contractive_givens_left(scale, angles, ((0, 1),))
    skew = torch.randn(*shape, representations, dimension, dimension, dtype=dtype)
    right = torch.matrix_exp(0.1 * (skew - skew.transpose(-1, -2)))
    drive = 0.05 * torch.randn(
        *shape, channels, representations, dimension, dtype=dtype
    )
    return CoupledIsotypicTransition(left, right, drive)


def _maximum_difference(
    left: CoupledIsotypicTransition, right: CoupledIsotypicTransition
) -> float:
    return max(
        float((left_value - right_value).abs().max())
        for left_value, right_value in zip(
            (left.left, left.right, left.drive),
            (right.left, right.right, right.drive),
        )
    )


def test_coupled_composition_is_associative() -> None:
    torch.manual_seed(202_608_26)
    first, second, third = (_random_transition(dtype=torch.float64) for _ in range(3))
    expected = compose_coupled_transition(
        third, compose_coupled_transition(second, first)
    )
    actual = compose_coupled_transition(
        compose_coupled_transition(third, second), first
    )
    assert _maximum_difference(actual, expected) < 2e-15


def test_parallel_recurrent_output_and_gradient_parity() -> None:
    torch.manual_seed(202_608_27)
    recurrent_transition = _random_transition(2, 7, dtype=torch.float64)
    parallel_transition = copy.deepcopy(recurrent_transition)
    initial_recurrent = torch.randn(2, 2, 3, 8, dtype=torch.float64)
    initial_recurrent.requires_grad_()
    initial_parallel = initial_recurrent.detach().clone().requires_grad_()
    recurrent_values = tuple(
        value.detach().clone().requires_grad_()
        for value in (
            recurrent_transition.left,
            recurrent_transition.right,
            recurrent_transition.drive,
        )
    )
    parallel_values = tuple(
        value.detach().clone().requires_grad_()
        for value in (
            parallel_transition.left,
            parallel_transition.right,
            parallel_transition.drive,
        )
    )
    recurrent_states, _ = recurrent_coupled_scan(
        CoupledIsotypicTransition(*recurrent_values), initial_recurrent
    )
    parallel_states, _ = parallel_coupled_scan(
        CoupledIsotypicTransition(*parallel_values), initial_parallel
    )
    output_gradient = torch.randn_like(recurrent_states)
    recurrent_gradients = torch.autograd.grad(
        recurrent_states,
        (*recurrent_values, initial_recurrent),
        output_gradient,
    )
    parallel_gradients = torch.autograd.grad(
        parallel_states,
        (*parallel_values, initial_parallel),
        output_gradient,
    )
    torch.testing.assert_close(
        parallel_states, recurrent_states, rtol=2e-12, atol=2e-12
    )
    for actual, expected in zip(parallel_gradients, recurrent_gradients):
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)


def test_zero_angle_reduces_to_independent_shared_action_and_is_contractive() -> None:
    torch.manual_seed(202_608_28)
    scale = torch.tensor([[[0.7, 0.9]]], dtype=torch.float64)
    left = contractive_givens_left(
        scale, torch.zeros(1, 1, 1, dtype=torch.float64), ((0, 1),)
    )
    torch.testing.assert_close(left, torch.diag_embed(scale), rtol=0.0, atol=0.0)
    singular_values = torch.linalg.svdvals(left)
    torch.testing.assert_close(singular_values[..., 0], scale.max(dim=-1).values)

    state = torch.randn(1, 2, 3, 8, dtype=torch.float64)
    right = torch.eye(8, dtype=torch.float64).expand(1, 3, 8, 8)
    drive = torch.randn_like(state)
    actual = apply_coupled_transition(
        CoupledIsotypicTransition(left[:, 0], right, drive), state
    )
    expected = scale[:, 0, :, None, None] * state + drive
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
