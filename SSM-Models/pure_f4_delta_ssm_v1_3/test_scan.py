from __future__ import annotations

import torch

from .scan import (
    compile_delta_transition,
    compile_one_sided_delta_transition,
    compose_transition,
    parallel_delta_scan,
    parallel_one_sided_delta_scan,
    recurrent_delta_scan,
    recurrent_one_sided_delta_scan,
)


def _fixture() -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    torch.manual_seed(20260821)
    dtype = torch.float64
    batch, length, rank, key_dim, value_dim = 2, 7, 3, 4, 5
    retention = (0.8 + 0.1 * torch.rand(batch, length, key_dim, dtype=dtype)).requires_grad_()
    write_key = torch.randn(batch, length, rank, key_dim, dtype=dtype).requires_grad_()
    erase_key = (0.03 * torch.randn(batch, length, rank, key_dim, dtype=dtype)).requires_grad_()
    write_value = torch.randn(batch, length, rank, value_dim, dtype=dtype).requires_grad_()
    tangent = (0.03 * torch.randn(batch, length, value_dim, value_dim, dtype=dtype)).requires_grad_()
    action = torch.matrix_exp(tangent)
    initial = torch.randn(batch, key_dim, value_dim, dtype=dtype).requires_grad_()
    query = torch.randn(batch, length, key_dim, dtype=dtype).requires_grad_()
    return [retention, write_key, erase_key, write_value, tangent, initial, query], action, query


def _run(inputs: list[torch.Tensor], scanner):
    retention, write_key, erase_key, write_value, tangent, initial, query = inputs
    transition = compile_delta_transition(
        retention, write_key, erase_key, write_value, torch.matrix_exp(tangent)
    )
    return scanner(transition, initial, query)


def test_two_sided_composition_matches_direct_application() -> None:
    inputs, action, _ = _fixture()
    retention, write_key, erase_key, write_value = inputs[:4]
    transition = compile_delta_transition(
        retention, write_key, erase_key, write_value, action
    )
    first = type(transition)(
        transition.left[:, 0], transition.right[:, 0], transition.bias[:, 0]
    )
    second = type(transition)(
        transition.left[:, 1], transition.right[:, 1], transition.bias[:, 1]
    )
    state = inputs[-2]
    direct = second.left @ (first.left @ state @ first.right.transpose(-1, -2) + first.bias) @ second.right.transpose(-1, -2) + second.bias
    composed = compose_transition(second, first)
    actual = composed.left @ state @ composed.right.transpose(-1, -2) + composed.bias
    torch.testing.assert_close(actual, direct, rtol=2e-13, atol=2e-13)


def test_parallel_and_recurrent_scans_match_outputs_states_and_gradients() -> None:
    recurrent_inputs, _, _ = _fixture()
    parallel_inputs = [value.detach().clone().requires_grad_(True) for value in recurrent_inputs]
    recurrent_reads, recurrent_states, recurrent_final = _run(
        recurrent_inputs, recurrent_delta_scan
    )
    parallel_reads, parallel_states, parallel_final = _run(
        parallel_inputs, parallel_delta_scan
    )
    torch.testing.assert_close(parallel_reads, recurrent_reads, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(parallel_states, recurrent_states, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(parallel_final, recurrent_final, rtol=2e-11, atol=2e-11)
    output_gradient = torch.randn_like(recurrent_reads)
    recurrent_gradients = torch.autograd.grad(
        recurrent_reads, recurrent_inputs, output_gradient
    )
    parallel_gradients = torch.autograd.grad(parallel_reads, parallel_inputs, output_gradient)
    for actual, expected in zip(parallel_gradients, recurrent_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=3e-10, atol=3e-10)


def test_rank_one_tied_keys_recover_delta_prediction_error_update() -> None:
    dtype = torch.float64
    state = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=dtype)
    key = torch.tensor([[[[1.0, 0.0]]]], dtype=dtype)
    value = torch.tensor([[[[7.0, 11.0]]]], dtype=dtype)
    retention = torch.ones(1, 1, 2, dtype=dtype)
    action = torch.eye(2, dtype=dtype).reshape(1, 1, 2, 2)
    transition = compile_delta_transition(retention, key, key, value, action)
    _, states, _ = recurrent_delta_scan(transition, state)
    torch.testing.assert_close(states[0, 0, 0], value[0, 0, 0])
    torch.testing.assert_close(states[0, 0, 1], state[0, 1])


def test_one_sided_identity_path_matches_generic_outputs_and_gradients() -> None:
    torch.manual_seed(47)
    dtype = torch.float64
    batch, length, rank, key_dim, value_dim = 2, 9, 2, 3, 5
    original = [
        (0.8 + 0.1 * torch.rand(batch, length, key_dim, dtype=dtype)).requires_grad_(),
        torch.randn(batch, length, rank, key_dim, dtype=dtype).requires_grad_(),
        (0.03 * torch.randn(batch, length, rank, key_dim, dtype=dtype)).requires_grad_(),
        torch.randn(batch, length, rank, value_dim, dtype=dtype).requires_grad_(),
        torch.randn(batch, key_dim, value_dim, dtype=dtype).requires_grad_(),
        torch.randn(batch, length, key_dim, dtype=dtype).requires_grad_(),
    ]
    fast_inputs = [value.detach().clone().requires_grad_(True) for value in original]

    retention, write_key, erase_key, write_value, initial, query = original
    identity = torch.eye(value_dim, dtype=dtype).reshape(1, 1, value_dim, value_dim)
    generic_transition = compile_delta_transition(
        retention,
        write_key,
        erase_key,
        write_value,
        identity.expand(batch, length, -1, -1),
    )
    expected, expected_states, expected_final = parallel_delta_scan(
        generic_transition, initial, query
    )

    retention, write_key, erase_key, write_value, initial, query = fast_inputs
    fast_transition = compile_one_sided_delta_transition(
        retention, write_key, erase_key, write_value
    )
    actual, actual_states, actual_final = parallel_one_sided_delta_scan(
        fast_transition, initial, query
    )
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-12)
    torch.testing.assert_close(actual_states, expected_states, rtol=3e-12, atol=3e-12)
    torch.testing.assert_close(actual_final, expected_final, rtol=3e-12, atol=3e-12)

    output_gradient = torch.randn_like(expected)
    expected_gradients = torch.autograd.grad(expected, original, output_gradient)
    actual_gradients = torch.autograd.grad(actual, fast_inputs, output_gradient)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=2e-11, atol=2e-11
        )


def test_one_sided_parallel_and_recurrent_scans_match() -> None:
    torch.manual_seed(53)
    transition = compile_one_sided_delta_transition(
        0.9 + 0.05 * torch.rand(1, 11, 3, dtype=torch.float64),
        torch.randn(1, 11, 2, 3, dtype=torch.float64),
        0.02 * torch.randn(1, 11, 2, 3, dtype=torch.float64),
        torch.randn(1, 11, 2, 7, dtype=torch.float64),
    )
    initial = torch.randn(1, 3, 7, dtype=torch.float64)
    query = torch.randn(1, 11, 3, dtype=torch.float64)
    expected = recurrent_one_sided_delta_scan(transition, initial, query)
    actual = parallel_one_sided_delta_scan(transition, initial, query)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_tensor, expected_tensor, rtol=2e-11, atol=2e-11
        )
