from __future__ import annotations

import torch

from spin_delta_scan import (
    SpinDeltaTransition,
    apply_spin_delta,
    compose_spin_delta,
    contractive_delta_left,
    parallel_spin_delta_scan,
    read_delta_state,
    recurrent_spin_delta_scan,
    route_delta_drive,
)


def _random_transition(
    *, batch: int = 2, length: int = 7, heads: int = 2, slots: int = 2
) -> SpinDeltaTransition:
    generator = torch.Generator().manual_seed(317)
    dtype = torch.float64
    left = 0.2 * torch.randn(
        batch, length, heads, slots, slots, generator=generator, dtype=dtype
    )
    action = 0.2 * torch.randn(
        batch, length, heads, 3, 8, 8, generator=generator, dtype=dtype
    )
    drive = 0.2 * torch.randn(
        batch, length, heads, slots, 3, 8, generator=generator, dtype=dtype
    )
    return SpinDeltaTransition(left, action, drive)


def test_composition_matches_direct_application_and_is_associative() -> None:
    transition = _random_transition(length=3)
    state = torch.randn(2, 2, 2, 3, 8, dtype=torch.float64)
    tokens = [
        SpinDeltaTransition(
            transition.left[:, index],
            transition.action[:, index],
            transition.drive[:, index],
        )
        for index in range(3)
    ]
    direct = apply_spin_delta(
        tokens[2], apply_spin_delta(tokens[1], apply_spin_delta(tokens[0], state))
    )
    left_grouped = compose_spin_delta(tokens[2], compose_spin_delta(tokens[1], tokens[0]))
    right_grouped = compose_spin_delta(compose_spin_delta(tokens[2], tokens[1]), tokens[0])
    torch.testing.assert_close(apply_spin_delta(left_grouped, state), direct)
    torch.testing.assert_close(apply_spin_delta(right_grouped, state), direct)


def test_parallel_scan_matches_recurrence_and_gradients() -> None:
    transition = _random_transition()
    transition = SpinDeltaTransition(
        transition.left.requires_grad_(),
        transition.action.requires_grad_(),
        transition.drive.requires_grad_(),
    )
    initial = torch.randn(2, 2, 2, 3, 8, dtype=torch.float64, requires_grad=True)
    recurrent, recurrent_final = recurrent_spin_delta_scan(transition, initial)
    parallel, parallel_final = parallel_spin_delta_scan(transition, initial)
    torch.testing.assert_close(parallel, recurrent, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(parallel_final, recurrent_final, rtol=2e-12, atol=2e-12)

    inputs = (*transition.__dict__.values(), initial)
    recurrent_gradients = torch.autograd.grad(
        recurrent.square().sum(), inputs, retain_graph=True
    )
    parallel_gradients = torch.autograd.grad(parallel.square().sum(), inputs)
    for actual, expected in zip(parallel_gradients, recurrent_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)


def test_contractive_left_and_routing_contracts() -> None:
    scale = torch.tensor([[[0.9, 0.7]]], dtype=torch.float64)
    erase_key = torch.tensor([[[[3.0, 4.0], [1.0, -2.0]]]], dtype=torch.float64)
    erase_strength = torch.tensor([[[0.25, 1.0]]], dtype=torch.float64)
    left = contractive_delta_left(scale, erase_key, erase_strength)
    singular_values = torch.linalg.svdvals(left)
    assert torch.all(singular_values <= scale[..., None] + 1e-12)

    drive = torch.randn(1, 1, 2, 3, 8, dtype=torch.float64)
    write_key = torch.tensor([[[[1.0, 0.0], [0.6, 0.8]]]], dtype=torch.float64)
    routed = route_delta_drive(write_key, drive)
    assert routed.shape == (1, 1, 2, 2, 3, 8)
    query = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]], dtype=torch.float64)
    torch.testing.assert_close(
        read_delta_state(routed, query),
        drive * write_key.sum(dim=-1)[..., None, None],
    )


def test_baseline_embedding_and_controller_directions_are_live() -> None:
    dtype = torch.float64
    state0 = torch.randn(1, 2, 3, 8, dtype=dtype)
    state = torch.stack((state0, torch.zeros_like(state0)), dim=2)
    action = torch.eye(8, dtype=dtype).expand(1, 2, 3, 8, 8).clone()
    scale = torch.tensor([[0.83, 0.91]], dtype=dtype)
    base_drive = torch.randn(1, 2, 3, 8, dtype=dtype)

    write_angle = torch.zeros(1, 2, dtype=dtype, requires_grad=True)
    erase_angle = torch.zeros(1, 2, dtype=dtype, requires_grad=True)
    erase_logit = torch.zeros(1, 2, dtype=dtype, requires_grad=True)
    query_delta = torch.zeros(1, 2, dtype=dtype, requires_grad=True)
    write_key = torch.stack((torch.cos(write_angle), torch.sin(write_angle)), dim=-1)
    erase_key = torch.stack((torch.sin(erase_angle), torch.cos(erase_angle)), dim=-1)
    query = torch.stack((1.0 + query_delta, 1.0 - query_delta), dim=-1)
    transition = SpinDeltaTransition(
        contractive_delta_left(scale, erase_key, torch.sigmoid(erase_logit)),
        action,
        route_delta_drive(write_key, base_drive),
    )
    result = apply_spin_delta(transition, state)
    read = read_delta_state(result, query)
    expected = scale[..., None, None] * torch.einsum(
        "...grij,...grj->...gri", action, state0
    ) + base_drive
    torch.testing.assert_close(read, expected, rtol=0.0, atol=0.0)

    read.square().sum().backward()
    assert write_angle.grad is not None and torch.count_nonzero(write_angle.grad)
    assert erase_angle.grad is not None and torch.count_nonzero(erase_angle.grad)
    assert query_delta.grad is not None and torch.count_nonzero(query_delta.grad)
    # At the exact embedding, strength acts only on an empty slot. Its zero
    # first derivative is structural; the learned erase direction activates it.
    assert erase_logit.grad is not None
    torch.testing.assert_close(erase_logit.grad, torch.zeros_like(erase_logit.grad))
