"""Bounded sliding-window causal self-attention with a streaming KV cache."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

_POSITION_BYTES = 8
_MAX_POSITION = torch.iinfo(torch.int64).max


@dataclass(frozen=True)
class AttentionConfig:
    """Configuration for :class:`CausalSelfAttention`."""

    model_dim: int
    heads: int
    window_size: int = 2048
    rope_base: float = 10_000.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("model_dim", self.model_dim),
            ("heads", self.heads),
            ("window_size", self.window_size),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.window_size > _MAX_POSITION:
            raise ValueError("window_size exceeds the signed 64-bit position range")
        if self.model_dim % self.heads != 0:
            raise ValueError("model_dim must be divisible by heads")
        if (self.model_dim // self.heads) % 2 != 0:
            raise ValueError("attention head_dim must be even for RoPE")
        if not math.isfinite(self.rope_base) or self.rope_base <= 0:
            raise ValueError("rope_base must be finite and positive")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")


@dataclass(frozen=True)
class AttentionState:
    """Rotated keys, values, and the next absolute sequence position.

    ``actual_bytes`` is the exact serialized state payload: logical tensor
    bytes plus one signed 64-bit position. Python and allocator overhead are
    intentionally not part of the model-state accounting contract.
    """

    key_cache: torch.Tensor
    value_cache: torch.Tensor
    position: int

    @property
    def cache_length(self) -> int:
        return self.key_cache.shape[-2]

    @property
    def actual_bytes(self) -> int:
        return (
            self.key_cache.numel() * self.key_cache.element_size()
            + self.value_cache.numel() * self.value_cache.element_size()
            + _POSITION_BYTES
        )

    @property
    def nbytes(self) -> int:
        """Alias for ``actual_bytes``."""

        return self.actual_bytes


class CausalSelfAttention(nn.Module):
    """Multi-head RoPE attention with a bounded sliding-window cache.

    A normal ``module(inputs)`` call returns only the output tensor. Pass
    ``use_cache=True`` for the first streaming chunk; subsequent calls may
    pass the returned state and automatically return another ``(output,
    state)`` pair.
    """

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        if not isinstance(config, AttentionConfig):
            raise TypeError("config must be an AttentionConfig")
        self.config = config
        self.head_dim = config.model_dim // config.heads
        self.qkv_projection = nn.Linear(
            config.model_dim, 3 * config.model_dim, bias=False
        )
        self.output_projection = nn.Linear(
            config.model_dim, config.model_dim, bias=False
        )
        self.register_buffer(
            "_rope_dimensions",
            torch.arange(0, self.head_dim, 2, dtype=torch.int64),
            persistent=False,
        )

    @property
    def max_cache_length(self) -> int:
        return self.config.window_size - 1

    @property
    def inv_freq(self) -> torch.Tensor:
        """Conventional RoPE frequencies ``base**(-2i / head_dim)``."""

        calculation_dtype = (
            torch.float64
            if self.qkv_projection.weight.dtype == torch.float64
            else torch.float32
        )
        dimensions = self._rope_dimensions.to(dtype=calculation_dtype)
        return self.config.rope_base ** (-dimensions / self.head_dim)

    def state_capacity_bytes(
        self, batch_size: int, dtype: torch.dtype | None = None
    ) -> int:
        """Return maximum cache payload bytes for a batch and dtype."""

        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if dtype is None:
            dtype = self.qkv_projection.weight.dtype
        try:
            element_size = torch.empty((), dtype=dtype).element_size()
        except (TypeError, RuntimeError) as error:
            raise TypeError("dtype must be a torch.dtype") from error
        cache_elements = (
            2 * batch_size * self.config.heads * self.max_cache_length * self.head_dim
        )
        return cache_elements * element_size + _POSITION_BYTES

    def state_actual_bytes(self, state: AttentionState) -> int:
        """Validate ``state`` and return its exact current payload bytes."""

        if not isinstance(state, AttentionState):
            raise TypeError("state must be an AttentionState")
        if not isinstance(state.key_cache, torch.Tensor):
            raise TypeError("state key_cache must be a tensor")
        if state.key_cache.ndim != 4:
            raise ValueError(
                "state caches must have shape (batch, heads, length, head_dim)"
            )
        self._validate_state(
            state,
            batch_size=state.key_cache.shape[0],
            dtype=state.key_cache.dtype,
            device=state.key_cache.device,
        )
        return state.actual_bytes

    def forward(
        self,
        inputs: torch.Tensor,
        state: AttentionState | None = None,
        *,
        use_cache: bool = False,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionState]:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a tensor")
        if inputs.ndim != 3 or inputs.shape[-1] != self.config.model_dim:
            raise ValueError("inputs must have shape (batch, length, model_dim)")
        if inputs.shape[0] == 0 or inputs.shape[1] == 0:
            raise ValueError("inputs must have nonempty batch and sequence dimensions")
        if type(use_cache) is not bool:
            raise TypeError("use_cache must be a bool")
        if valid_mask is not None:
            raise NotImplementedError(
                "valid_mask is not supported: padded batches require per-example cache lengths"
            )
        if state is not None and not isinstance(state, AttentionState):
            raise TypeError("state must be an AttentionState or None")

        batch_size, sequence_length, _ = inputs.shape
        qkv = self.qkv_projection(inputs)
        query, key, value = qkv.chunk(3, dim=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        if state is None:
            position = 0
            past_key = key[:, :, :0, :]
            past_value = value[:, :, :0, :]
        else:
            self._validate_state(
                state,
                batch_size=batch_size,
                dtype=key.dtype,
                device=key.device,
            )
            position = state.position
            past_key = state.key_cache
            past_value = state.value_cache

        if sequence_length > _MAX_POSITION - position:
            raise ValueError("sequence would overflow the signed 64-bit position range")
        query = self._apply_rope(query, position)
        key = self._apply_rope(key, position)

        combined_key = torch.cat((past_key, key), dim=-2)
        combined_value = torch.cat((past_value, value), dim=-2)
        attention_mask = self._local_causal_mask(
            query_length=sequence_length,
            past_length=past_key.shape[-2],
            position=position,
            device=query.device,
            dtype=query.dtype,
        )
        attended = F.scaled_dot_product_attention(
            query,
            combined_key,
            combined_value,
            attn_mask=attention_mask,
            dropout_p=self.config.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.config.model_dim)
        )
        output = self.output_projection(attended)

        if not use_cache and state is None:
            return output

        cache_length = min(self.max_cache_length, combined_key.shape[-2])
        cache_start = combined_key.shape[-2] - cache_length
        # Clone so a small cache cannot retain a larger chunk's backing storage.
        next_key = combined_key[:, :, cache_start:, :].clone()
        next_value = combined_value[:, :, cache_start:, :].clone()
        next_state = AttentionState(
            key_cache=next_key,
            value_cache=next_value,
            position=position + sequence_length,
        )
        return output, next_state

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(
            batch_size, sequence_length, self.config.heads, self.head_dim
        ).transpose(1, 2)

    def _apply_rope(self, tensor: torch.Tensor, position: int) -> torch.Tensor:
        calculation_dtype = (
            torch.float64 if tensor.dtype == torch.float64 else torch.float32
        )
        positions = torch.arange(
            position,
            position + tensor.shape[-2],
            device=tensor.device,
            dtype=calculation_dtype,
        )
        inverse_frequencies = self.inv_freq.to(dtype=calculation_dtype)
        angles = torch.outer(positions, inverse_frequencies)
        cosine = angles.cos().to(dtype=tensor.dtype)[None, None, :, :]
        sine = angles.sin().to(dtype=tensor.dtype)[None, None, :, :]
        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]
        rotated = torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
        )
        return rotated.flatten(-2)

    def _local_causal_mask(
        self,
        *,
        query_length: int,
        past_length: int,
        position: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        query_positions = torch.arange(
            position, position + query_length, device=device, dtype=torch.int64
        )
        past_start = position - past_length
        key_positions = torch.arange(
            past_start,
            position + query_length,
            device=device,
            dtype=torch.int64,
        )
        allowed = (key_positions[None, :] <= query_positions[:, None]) & (
            key_positions[None, :]
            >= query_positions[:, None] - (self.config.window_size - 1)
        )
        mask = torch.full(allowed.shape, float("-inf"), device=device, dtype=dtype)
        return mask.masked_fill(allowed, 0.0)

    def _validate_state(
        self,
        state: AttentionState,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if not isinstance(state.key_cache, torch.Tensor) or not isinstance(
            state.value_cache, torch.Tensor
        ):
            raise TypeError("state key_cache and value_cache must be tensors")
        if type(state.position) is not int:
            raise TypeError("state position must be an integer")
        if not 0 <= state.position <= _MAX_POSITION:
            raise ValueError("state position is outside the signed 64-bit range")
        expected_prefix = (batch_size, self.config.heads)
        if state.key_cache.ndim != 4 or state.value_cache.ndim != 4:
            raise ValueError(
                "state caches must have shape (batch, heads, length, head_dim)"
            )
        if state.key_cache.shape != state.value_cache.shape:
            raise ValueError("state key and value caches must have identical shapes")
        if (
            state.key_cache.shape[:2] != expected_prefix
            or state.key_cache.shape[-1] != self.head_dim
        ):
            raise ValueError(
                "state cache shape is incompatible with the input and configuration"
            )
        expected_length = min(state.position, self.max_cache_length)
        if state.key_cache.shape[-2] != expected_length:
            raise ValueError(
                "state cache length is inconsistent with its absolute position"
            )
        if state.key_cache.dtype != dtype or state.value_cache.dtype != dtype:
            raise ValueError("state cache dtype must match projected keys and values")
        if state.key_cache.device != device or state.value_cache.device != device:
            raise ValueError("state cache device must match the input device")


__all__ = ["AttentionConfig", "AttentionState", "CausalSelfAttention"]
