"""Algebraic and differential contracts for independent-action block scans."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from coupled_isotypic_scan import contractive_givens_left
from independent_block_scan import (
    IndependentBlockTransition,
    apply_independent_block_transition,
    compose_independent_block_transition,
    materialize_independent_block_operator,
    parallel_independent_block_scan,
    recurrent_independent_block_scan,
)


def random_orthogonal(shape: tuple[int, ...], dimension: int) -> torch.Tensor:
    raw = 0.15 * torch.randn(*shape, dimension, dimension, dtype=torch.float64)
    return torch.matrix_exp(raw - raw.transpose(-1, -2))


def test_independent_block_affine_composition_is_associative() -> None:
    torch.manual_seed(202_608_32)
    batch, channels, representations, dimension = 2, 2, 3, 4
    transitions = []
    for _ in range(3):
        left = random_orthogonal((batch,), channels)
        actions = random_orthogonal((batch, channels, representations), dimension)
        transitions.append(
            IndependentBlockTransition(
                materialize_independent_block_operator(left, actions),
                torch.randn(
                    batch,
                    channels,
                    representations,
                    dimension,
                    dtype=torch.float64,
                ),
            )
        )
    first = compose_independent_block_transition(
        transitions[2],
        compose_independent_block_transition(transitions[1], transitions[0]),
    )
    second = compose_independent_block_transition(
        compose_independent_block_transition(transitions[2], transitions[1]),
        transitions[0],
    )
    state = torch.randn(
        batch, channels, representations, dimension, dtype=torch.float64
    )
    torch.testing.assert_close(first.operator, second.operator, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(first.drive, second.drive, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        apply_independent_block_transition(first, state),
        apply_independent_block_transition(second, state),
        rtol=1e-12,
        atol=1e-12,
    )


def test_parallel_and_recurrent_independent_block_outputs_and_gradients_match() -> None:
    torch.manual_seed(202_608_33)
    batch, length, channels, representations, dimension = 2, 7, 2, 3, 4
    scale = 0.7 + 0.2 * torch.rand(batch, length, channels, dtype=torch.float64)
    angles = 0.2 * torch.randn(batch, length, 1, dtype=torch.float64)
    left = contractive_givens_left(scale, angles, ((0, 1),))
    actions = random_orthogonal(
        (batch, length, channels, representations), dimension
    )
    drive = torch.randn(
        batch, length, channels, representations, dimension, dtype=torch.float64
    )
    initial = torch.randn(
        batch, channels, representations, dimension, dtype=torch.float64
    )
    base = (left, actions, drive, initial)
    recurrent_inputs = tuple(value.detach().clone().requires_grad_() for value in base)
    parallel_inputs = tuple(value.detach().clone().requires_grad_() for value in base)
    expected, expected_final = recurrent_independent_block_scan(*recurrent_inputs)
    actual, actual_final = parallel_independent_block_scan(*parallel_inputs)
    weights = torch.randn_like(expected)
    expected_gradients = torch.autograd.grad((expected * weights).sum(), recurrent_inputs)
    actual_gradients = torch.autograd.grad((actual * weights).sum(), parallel_inputs)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(actual_final, expected_final, rtol=2e-11, atol=2e-11)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=3e-10, atol=3e-10
        )


def test_independent_block_is_contractive_and_identity_reduces_to_v12() -> None:
    torch.manual_seed(202_608_34)
    batch, length, channels, representations, dimension = 2, 5, 2, 3, 4
    scale = 0.7 + 0.2 * torch.rand(batch, length, channels, dtype=torch.float64)
    angles = torch.zeros(batch, length, 1, dtype=torch.float64)
    left = contractive_givens_left(scale, angles, ((0, 1),))
    actions = random_orthogonal(
        (batch, length, channels, representations), dimension
    )
    operator = materialize_independent_block_operator(left, actions)
    spectral_norm = torch.linalg.matrix_norm(operator, ord=2, dim=(-2, -1))
    expected_bound = scale.amax(dim=-1, keepdim=True).expand_as(spectral_norm)
    torch.testing.assert_close(spectral_norm, expected_bound, rtol=2e-12, atol=2e-12)

    drive = torch.randn(
        batch, length, channels, representations, dimension, dtype=torch.float64
    )
    initial = torch.randn(
        batch, channels, representations, dimension, dtype=torch.float64
    )
    actual, _ = recurrent_independent_block_scan(left, actions, drive, initial)
    state = initial
    expected = []
    for position in range(length):
        state = (
            scale[:, position, :, None, None]
            * torch.einsum(
                "...crij,...crj->...cri", actions[:, position], state
            )
            + drive[:, position]
        )
        expected.append(state)
    torch.testing.assert_close(
        actual, torch.stack(expected, dim=1), rtol=2e-12, atol=2e-12
    )


def test_independent_block_is_equivariant_under_common_frame_changes() -> None:
    torch.manual_seed(202_608_35)
    batch, length, channels, representations, dimension = 1, 5, 2, 3, 4
    scale = 0.7 + 0.2 * torch.rand(batch, length, channels, dtype=torch.float64)
    angles = 0.2 * torch.randn(batch, length, 1, dtype=torch.float64)
    left = contractive_givens_left(scale, angles, ((0, 1),))
    actions = random_orthogonal(
        (batch, length, channels, representations), dimension
    )
    frame = random_orthogonal((representations,), dimension)
    drive = torch.randn(
        batch, length, channels, representations, dimension, dtype=torch.float64
    )
    initial = torch.randn(
        batch, channels, representations, dimension, dtype=torch.float64
    )
    expected, _ = recurrent_independent_block_scan(left, actions, drive, initial)
    transformed_actions = torch.einsum(
        "rij,...crjk,rlk->...cril", frame, actions, frame
    )
    transformed_drive = torch.einsum("rij,...crj->...cri", frame, drive)
    transformed_initial = torch.einsum("rij,...crj->...cri", frame, initial)
    actual, _ = recurrent_independent_block_scan(
        left, transformed_actions, transformed_drive, transformed_initial
    )
    transformed_expected = torch.einsum("rij,...crj->...cri", frame, expected)
    torch.testing.assert_close(actual, transformed_expected, rtol=2e-11, atol=2e-11)
