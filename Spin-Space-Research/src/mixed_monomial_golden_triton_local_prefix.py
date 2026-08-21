"""Fused indexed local-prefix expansion for compiled N-H-N chunks.

The exact compiler stores a labelled table with shape ``(N,H,N,24,8)``.
Given one label triple and one incoming eight-vector per chunk, this module
fuses the three-dimensional table lookup with the 24-by-8 matrix-vector
product.  The output rows are the three chronological causal states
``[R x; H R x; L H R x]``.

The Triton path intentionally has a narrow differentiation contract: it
supports gradients with respect to incoming states, which is sufficient for
the maintained initial-state backward benchmark.  A trainable prefix table
uses the eager path, preserving full PyTorch autograd instead of silently
discarding table gradients.
"""

from __future__ import annotations

from typing import Literal

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional environment boundary
    triton = None
    tl = None

LocalPrefixBackend = Literal["auto", "eager", "triton"]
STATE_DIMENSION = 8
PREFIX_ROWS = 24


if triton is not None:

    @triton.jit
    def _indexed_local_prefix_forward_kernel(
        table_pointer,
        left_pointer,
        middle_pointer,
        right_pointer,
        incoming_pointer,
        output_pointer,
        middle_count: tl.constexpr,
        right_count: tl.constexpr,
        row_block: tl.constexpr,
    ):
        token = tl.program_id(0)
        rows = tl.arange(0, row_block)
        columns = tl.arange(0, 8)
        row_mask = rows < 24

        left = tl.load(left_pointer + token)
        middle = tl.load(middle_pointer + token)
        right = tl.load(right_pointer + token)
        table_index = (left * middle_count + middle) * right_count + right
        table_offsets = (
            table_index * 24 * 8
            + rows[:, None] * 8
            + columns[None, :]
        )
        operator = tl.load(
            table_pointer + table_offsets,
            mask=row_mask[:, None],
            other=0.0,
        )
        incoming = tl.load(
            incoming_pointer + token * 8 + columns
        )
        packed = tl.sum(operator * incoming[None, :], axis=1)
        tl.store(
            output_pointer + token * 24 + rows,
            packed,
            mask=row_mask,
        )

    @triton.jit
    def _indexed_local_prefix_backward_kernel(
        table_pointer,
        left_pointer,
        middle_pointer,
        right_pointer,
        output_gradient_pointer,
        incoming_gradient_pointer,
        middle_count: tl.constexpr,
        right_count: tl.constexpr,
        row_block: tl.constexpr,
    ):
        token = tl.program_id(0)
        rows = tl.arange(0, row_block)
        columns = tl.arange(0, 8)
        row_mask = rows < 24

        left = tl.load(left_pointer + token)
        middle = tl.load(middle_pointer + token)
        right = tl.load(right_pointer + token)
        table_index = (left * middle_count + middle) * right_count + right
        table_offsets = (
            table_index * 24 * 8
            + rows[:, None] * 8
            + columns[None, :]
        )
        operator = tl.load(
            table_pointer + table_offsets,
            mask=row_mask[:, None],
            other=0.0,
        )
        output_gradient = tl.load(
            output_gradient_pointer + token * 24 + rows,
            mask=row_mask,
            other=0.0,
        )
        incoming_gradient = tl.sum(
            operator * output_gradient[:, None], axis=0
        )
        tl.store(
            incoming_gradient_pointer + token * 8 + columns,
            incoming_gradient,
        )

    class _IndexedLocalPrefix(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: torch.autograd.function.FunctionCtx,
            prefix_table: torch.Tensor,
            left_index: torch.Tensor,
            middle_index: torch.Tensor,
            right_index: torch.Tensor,
            incoming: torch.Tensor,
        ) -> torch.Tensor:
            token_count = incoming.numel() // STATE_DIMENSION
            output = torch.empty(
                *incoming.shape[:-1],
                PREFIX_ROWS,
                dtype=incoming.dtype,
                device=incoming.device,
            )
            _indexed_local_prefix_forward_kernel[(token_count,)](
                prefix_table,
                left_index,
                middle_index,
                right_index,
                incoming,
                output,
                prefix_table.shape[1],
                prefix_table.shape[2],
                row_block=32,
                num_warps=1,
            )
            ctx.save_for_backward(
                prefix_table, left_index, middle_index, right_index
            )
            return output

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx,
            output_gradient: torch.Tensor,
        ) -> tuple[None, None, None, None, torch.Tensor]:
            prefix_table, left_index, middle_index, right_index = (
                ctx.saved_tensors
            )
            output_gradient = output_gradient.contiguous()
            incoming_gradient = torch.empty(
                *output_gradient.shape[:-1],
                STATE_DIMENSION,
                dtype=output_gradient.dtype,
                device=output_gradient.device,
            )
            token_count = incoming_gradient.numel() // STATE_DIMENSION
            _indexed_local_prefix_backward_kernel[(token_count,)](
                prefix_table,
                left_index,
                middle_index,
                right_index,
                output_gradient,
                incoming_gradient,
                prefix_table.shape[1],
                prefix_table.shape[2],
                row_block=32,
                num_warps=1,
            )
            return None, None, None, None, incoming_gradient


def triton_is_available() -> bool:
    """Return whether the optional Triton CUDA path can be attempted."""

    return triton is not None and torch.cuda.is_available()


def _validate_inputs(
    prefix_table: torch.Tensor,
    left_index: torch.Tensor,
    middle_index: torch.Tensor,
    right_index: torch.Tensor,
    incoming: torch.Tensor,
) -> tuple[int, int]:
    if prefix_table.ndim != 5 or prefix_table.shape[-2:] != (
        PREFIX_ROWS,
        STATE_DIMENSION,
    ):
        raise ValueError("prefix_table must have shape (N,H,N,24,8)")
    if prefix_table.shape[0] != prefix_table.shape[2]:
        raise ValueError("the left and right table dimensions must agree")
    if (
        left_index.shape != middle_index.shape
        or left_index.shape != right_index.shape
        or left_index.ndim != 2
    ):
        raise ValueError("label indices must share shape (batch,chunks)")
    batch, chunks = left_index.shape
    if incoming.shape != (batch, chunks, STATE_DIMENSION):
        raise ValueError("incoming must have shape (batch,chunks,8)")
    tensors = (left_index, middle_index, right_index, incoming)
    if any(value.device != prefix_table.device for value in tensors):
        raise ValueError("the table, indices, and incoming states must share a device")
    if any(value.dtype not in (torch.int32, torch.int64) for value in tensors[:3]):
        raise ValueError("label indices must use int32 or int64")
    if prefix_table.dtype != incoming.dtype:
        raise ValueError("the table and incoming states must share a dtype")
    if not prefix_table.is_floating_point():
        raise ValueError("the table and incoming states must be floating point")
    return batch, chunks


def eager_indexed_local_prefix_states(
    prefix_table: torch.Tensor,
    left_index: torch.Tensor,
    middle_index: torch.Tensor,
    right_index: torch.Tensor,
    incoming: torch.Tensor,
) -> torch.Tensor:
    """Reference indexed lookup and local-prefix application in PyTorch."""

    batch, chunks = _validate_inputs(
        prefix_table,
        left_index,
        middle_index,
        right_index,
        incoming,
    )
    operators = prefix_table[left_index, middle_index, right_index]
    packed = (operators @ incoming[..., None]).squeeze(-1)
    return packed.reshape(batch, chunks * 3, STATE_DIMENSION)


def triton_indexed_local_prefix_states(
    prefix_table: torch.Tensor,
    left_index: torch.Tensor,
    middle_index: torch.Tensor,
    right_index: torch.Tensor,
    incoming: torch.Tensor,
) -> torch.Tensor:
    """Run the fused lookup/matvec with incoming-state backward support."""

    batch, chunks = _validate_inputs(
        prefix_table,
        left_index,
        middle_index,
        right_index,
        incoming,
    )
    if not triton_is_available():
        raise RuntimeError("the fused local-prefix path requires Triton and CUDA")
    if prefix_table.device.type != "cuda" or prefix_table.dtype != torch.float32:
        raise ValueError("the fused local-prefix path requires CUDA float32 inputs")
    if prefix_table.requires_grad:
        raise ValueError(
            "the fused path does not implement prefix-table gradients; use eager"
        )
    packed = _IndexedLocalPrefix.apply(
        prefix_table.contiguous(),
        left_index.contiguous(),
        middle_index.contiguous(),
        right_index.contiguous(),
        incoming.contiguous(),
    )
    return packed.reshape(batch, chunks * 3, STATE_DIMENSION)


def indexed_local_prefix_states(
    prefix_table: torch.Tensor,
    left_index: torch.Tensor,
    middle_index: torch.Tensor,
    right_index: torch.Tensor,
    incoming: torch.Tensor,
    *,
    backend: LocalPrefixBackend = "auto",
) -> torch.Tensor:
    """Dispatch to Triton when safe, otherwise preserve eager semantics."""

    if backend not in ("auto", "eager", "triton"):
        raise ValueError("backend must be 'auto', 'eager', or 'triton'")
    use_triton = (
        backend == "triton"
        or (
            backend == "auto"
            and triton_is_available()
            and prefix_table.device.type == "cuda"
            and prefix_table.dtype == torch.float32
            and not prefix_table.requires_grad
        )
    )
    if use_triton:
        return triton_indexed_local_prefix_states(
            prefix_table,
            left_index,
            middle_index,
            right_index,
            incoming,
        )
    return eager_indexed_local_prefix_states(
        prefix_table,
        left_index,
        middle_index,
        right_index,
        incoming,
    )


__all__ = [
    "LocalPrefixBackend",
    "eager_indexed_local_prefix_states",
    "indexed_local_prefix_states",
    "triton_indexed_local_prefix_states",
    "triton_is_available",
]
