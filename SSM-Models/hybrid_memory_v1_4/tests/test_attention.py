"""Correctness gates for bounded streaming causal self-attention."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attention import AttentionConfig, AttentionState, CausalSelfAttention


def _attention(*, window_size: int = 7) -> CausalSelfAttention:
    torch.manual_seed(7)
    return (
        CausalSelfAttention(
            AttentionConfig(model_dim=24, heads=3, window_size=window_size)
        )
        .double()
        .eval()
    )


def _stream(
    module: CausalSelfAttention,
    inputs: torch.Tensor,
    boundaries: list[int],
) -> tuple[torch.Tensor, AttentionState]:
    outputs = []
    state = None
    start = 0
    for stop in boundaries:
        output, state = module(inputs[:, start:stop], state, use_cache=True)
        outputs.append(output)
        start = stop
    assert state is not None
    return torch.cat(outputs, dim=1), state


def test_no_future_leakage_and_window_is_local() -> None:
    module = _attention(window_size=4)
    inputs = torch.randn(2, 11, 24, dtype=torch.float64)

    changed_future = inputs.clone()
    changed_future[:, 6:] = torch.randn_like(changed_future[:, 6:])
    assert torch.allclose(
        module(inputs)[:, :6], module(changed_future)[:, :6], atol=1e-12, rtol=1e-12
    )

    changed_old_history = inputs.clone()
    changed_old_history[:, :7] = torch.randn_like(changed_old_history[:, :7])
    assert torch.allclose(
        module(inputs)[:, 10],
        module(changed_old_history)[:, 10],
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "boundaries",
    [
        [19],
        [1, 4, 5, 11, 19],
        list(range(1, 20)),
        [2, 10, 12, 19],
    ],
)
def test_full_chunk_and_token_step_parity(boundaries: list[int]) -> None:
    module = _attention(window_size=6)
    inputs = torch.randn(2, 19, 24, dtype=torch.float64)
    full = module(inputs)
    streamed, state = _stream(module, inputs, boundaries)

    assert torch.allclose(full, streamed, atol=1e-11, rtol=1e-11)
    assert state.position == inputs.shape[1]


def test_cache_bound_after_long_stream() -> None:
    module = _attention(window_size=5)
    state = None
    for index in range(137):
        chunk_length = index % 9 + 1
        inputs = torch.randn(3, chunk_length, 24, dtype=torch.float64)
        _, state = module(inputs, state, use_cache=True)
        assert state.cache_length <= module.max_cache_length
        assert state.key_cache.shape == state.value_cache.shape
        assert state.key_cache.storage_offset() == 0
        assert state.value_cache.storage_offset() == 0
        assert state.key_cache.untyped_storage().nbytes() == (
            state.key_cache.numel() * state.key_cache.element_size()
        )
        assert state.value_cache.untyped_storage().nbytes() == (
            state.value_cache.numel() * state.value_cache.element_size()
        )
    assert state is not None
    assert state.cache_length == 4
    assert state.position == sum(index % 9 + 1 for index in range(137))


def test_rope_frequency_and_absolute_position_continuation() -> None:
    module = _attention(window_size=8)
    dimensions = torch.arange(0, module.head_dim, 2, dtype=torch.float64)
    expected = 1.0 / (module.config.rope_base ** (dimensions / module.head_dim))
    assert torch.equal(module.inv_freq, expected)

    inputs = torch.randn(1, 17, 24, dtype=torch.float64)
    _, full_state = module(inputs, use_cache=True)
    _, chunk_state = _stream(module, inputs, [3, 9, 10, 17])
    assert full_state.position == chunk_state.position == 17
    assert torch.allclose(
        full_state.key_cache, chunk_state.key_cache, atol=1e-14, rtol=1e-14
    )
    assert torch.allclose(
        full_state.value_cache, chunk_state.value_cache, atol=1e-14, rtol=1e-14
    )

    raw_key = module._split_heads(
        module.qkv_projection(inputs[:, 9:10]).chunk(3, dim=-1)[1]
    )
    rotated_at_nine = module._apply_rope(raw_key, 9)
    rotated_at_zero = module._apply_rope(raw_key, 0)
    assert not torch.allclose(rotated_at_nine, rotated_at_zero)


@pytest.mark.parametrize(
    "config",
    [
        AttentionConfig(24, 3),
    ],
)
def test_valid_configuration(config: AttentionConfig) -> None:
    assert CausalSelfAttention(config).head_dim == 8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_dim": 0, "heads": 1},
        {"model_dim": 24, "heads": 0},
        {"model_dim": 25, "heads": 3},
        {"model_dim": 21, "heads": 3},
        {"model_dim": 24, "heads": 3, "window_size": 0},
        {"model_dim": 24, "heads": 3, "rope_base": 0.0},
        {"model_dim": 24, "heads": 3, "dropout": 1.0},
    ],
)
def test_invalid_configuration_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        AttentionConfig(**kwargs)  # type: ignore[arg-type]


def test_invalid_input_and_masks_rejected() -> None:
    module = _attention()
    with pytest.raises(ValueError, match="shape"):
        module(torch.randn(2, 24, dtype=torch.float64))
    with pytest.raises(ValueError, match="shape"):
        module(torch.randn(2, 3, 23, dtype=torch.float64))
    with pytest.raises(ValueError, match="nonempty"):
        module(torch.randn(2, 0, 24, dtype=torch.float64))
    with pytest.raises(NotImplementedError, match="valid_mask"):
        module(
            torch.randn(2, 3, 24, dtype=torch.float64),
            valid_mask=torch.ones(2, 3, dtype=torch.bool),
        )


def test_invalid_states_rejected() -> None:
    module = _attention(window_size=5)
    inputs = torch.randn(2, 3, 24, dtype=torch.float64)
    _, state = module(inputs, use_cache=True)

    bad_shape = AttentionState(state.key_cache[:, :2], state.value_cache[:, :2], 3)
    with pytest.raises(ValueError, match="shape"):
        module(inputs, bad_shape)

    bad_length = replace(state, position=4)
    with pytest.raises(ValueError, match="length"):
        module(inputs, bad_length)

    bad_dtype = AttentionState(state.key_cache.float(), state.value_cache.float(), 3)
    with pytest.raises(ValueError, match="dtype"):
        module(inputs, bad_dtype)

    bad_position = replace(state, position=-1)
    with pytest.raises(ValueError, match="position"):
        module(inputs, bad_position)

    with pytest.raises(TypeError, match="AttentionState"):
        module(inputs, state=torch.zeros(1))  # type: ignore[arg-type]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_actual_and_capacity_byte_accounting(dtype: torch.dtype) -> None:
    module = (
        CausalSelfAttention(AttentionConfig(model_dim=24, heads=3, window_size=5))
        .to(dtype=dtype)
        .eval()
    )
    inputs = torch.randn(2, 3, 24, dtype=dtype)
    _, state = module(inputs, use_cache=True)
    element_size = inputs.element_size()
    expected_actual = 2 * 2 * 3 * 3 * 8 * element_size + 8
    expected_capacity = 2 * 2 * 3 * 4 * 8 * element_size + 8

    assert state.actual_bytes == expected_actual
    assert state.nbytes == expected_actual
    assert module.state_actual_bytes(state) == expected_actual
    assert module.state_capacity_bytes(2, dtype) == expected_capacity

    _, full_state = module(torch.randn(2, 8, 24, dtype=dtype), use_cache=True)
    assert full_state.actual_bytes == expected_capacity


def test_gradients_are_finite() -> None:
    torch.manual_seed(11)
    module = (
        CausalSelfAttention(
            AttentionConfig(model_dim=24, heads=3, window_size=6, dropout=0.0)
        )
        .double()
        .train()
    )
    inputs = torch.randn(2, 13, 24, dtype=torch.float64, requires_grad=True)
    loss = module(inputs).square().mean()
    loss.backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
