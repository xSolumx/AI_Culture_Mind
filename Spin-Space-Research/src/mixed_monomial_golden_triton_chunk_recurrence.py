"""Register-resident Triton recurrence over exact compiled N-H-N chunks.

One CUDA program owns one batch element, keeps its eight-scalar state in
registers, and walks a sequence of exact dictionary labels.  At each chunk it
loads the selected 24-by-8 operator, emits all three causal prefix states, and
uses the final eight-row block as the next recurrent state.

This is a streaming/chunk complement to the logarithmic-depth endpoint scan,
not a parallel prefix algorithm.  Its reverse kernel propagates gradients to
the initial state.  Trainable table gradients remain on the eager reference
path.
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

ChunkRecurrenceBackend = Literal["auto", "eager", "triton"]
STATE_DIMENSION = 8
PREFIX_ROWS = 24


if triton is not None:

    @triton.jit
    def _compiled_chunk_recurrence_forward_kernel(
        table_pointer,
        left_pointer,
        middle_pointer,
        right_pointer,
        initial_pointer,
        output_pointer,
        chunks,
        middle_count: tl.constexpr,
        right_count: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        rows = tl.arange(0, 8)
        columns = tl.arange(0, 8)
        state = tl.load(initial_pointer + batch_index * 8 + columns)

        for position in tl.range(0, chunks):
            token = batch_index * chunks + position
            left = tl.load(left_pointer + token)
            middle = tl.load(middle_pointer + token)
            right = tl.load(right_pointer + token)
            table_index = (left * middle_count + middle) * right_count + right
            table_base = table_index * 24 * 8
            output_base = token * 24
            for block in tl.static_range(0, 3):
                operator_offsets = (
                    table_base
                    + (block * 8 + rows[:, None]) * 8
                    + columns[None, :]
                )
                operator = tl.load(table_pointer + operator_offsets)
                value = tl.sum(operator * state[None, :], axis=1)
                tl.store(output_pointer + output_base + block * 8 + rows, value)
                if block == 2:
                    state = value

    @triton.jit
    def _compiled_chunk_recurrence_backward_kernel(
        table_pointer,
        left_pointer,
        middle_pointer,
        right_pointer,
        output_gradient_pointer,
        initial_gradient_pointer,
        chunks,
        middle_count: tl.constexpr,
        right_count: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        rows = tl.arange(0, 8)
        columns = tl.arange(0, 8)
        carry = tl.zeros([8], dtype=tl.float32)

        for reverse_position in tl.range(0, chunks):
            position = chunks - 1 - reverse_position
            token = batch_index * chunks + position
            left = tl.load(left_pointer + token)
            middle = tl.load(middle_pointer + token)
            right = tl.load(right_pointer + token)
            table_index = (left * middle_count + middle) * right_count + right
            table_base = table_index * 24 * 8
            output_base = token * 24
            state_gradient = tl.zeros([8], dtype=tl.float32)
            for block in tl.static_range(0, 3):
                operator_offsets = (
                    table_base
                    + (block * 8 + rows[:, None]) * 8
                    + columns[None, :]
                )
                operator = tl.load(table_pointer + operator_offsets)
                direct = tl.load(
                    output_gradient_pointer + output_base + block * 8 + rows
                )
                if block == 2:
                    direct += carry
                state_gradient += tl.sum(operator * direct[:, None], axis=0)
            carry = state_gradient
        tl.store(initial_gradient_pointer + batch_index * 8 + columns, carry)

    class _CompiledChunkRecurrence(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: torch.autograd.function.FunctionCtx,
            prefix_table: torch.Tensor,
            left_index: torch.Tensor,
            middle_index: torch.Tensor,
            right_index: torch.Tensor,
            initial_state: torch.Tensor,
        ) -> torch.Tensor:
            batch, chunks = left_index.shape
            output = torch.empty(
                batch,
                chunks,
                PREFIX_ROWS,
                dtype=initial_state.dtype,
                device=initial_state.device,
            )
            _compiled_chunk_recurrence_forward_kernel[(batch,)](
                prefix_table,
                left_index,
                middle_index,
                right_index,
                initial_state,
                output,
                chunks,
                prefix_table.shape[1],
                prefix_table.shape[2],
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
            batch, chunks = left_index.shape
            initial_gradient = torch.empty(
                batch,
                STATE_DIMENSION,
                dtype=output_gradient.dtype,
                device=output_gradient.device,
            )
            _compiled_chunk_recurrence_backward_kernel[(batch,)](
                prefix_table,
                left_index,
                middle_index,
                right_index,
                output_gradient,
                initial_gradient,
                chunks,
                prefix_table.shape[1],
                prefix_table.shape[2],
                num_warps=1,
            )
            return None, None, None, None, initial_gradient


def triton_is_available() -> bool:
    """Return whether the optional Triton CUDA path can be attempted."""

    return triton is not None and torch.cuda.is_available()


def _validate_inputs(
    prefix_table: torch.Tensor,
    left_index: torch.Tensor,
    middle_index: torch.Tensor,
    right_index: torch.Tensor,
    initial_state: torch.Tensor,
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
    if chunks < 1:
        raise ValueError("the chunk sequence must be nonempty")
    if initial_state.shape != (batch, STATE_DIMENSION):
        raise ValueError("initial_state must have shape (batch,8)")
    tensors = (left_index, middle_index, right_index, initial_state)
    if any(value.device != prefix_table.device for value in tensors):
        raise ValueError("the table, indices, and initial state must share a device")
    if any(value.dtype not in (torch.int32, torch.int64) for value in tensors[:3]):
        raise ValueError("label indices must use int32 or int64")
    if prefix_table.dtype != initial_state.dtype:
        raise ValueError("the table and initial state must share a dtype")
    if not prefix_table.is_floating_point():
        raise ValueError("the table and initial state must be floating point")
    return batch, chunks


def eager_indexed_chunk_recurrence(
    prefix_table: torch.Tensor,
    left_index: torch.Tensor,
    middle_index: torch.Tensor,
    right_index: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """Sequential PyTorch oracle for the exact compiled recurrence."""

    batch, chunks = _validate_inputs(
        prefix_table,
        left_index,
        middle_index,
        right_index,
        initial_state,
    )
    state = initial_state
    outputs = []
    for position in range(chunks):
        operator = prefix_table[
            left_index[:, position],
            middle_index[:, position],
            right_index[:, position],
        ]
        packed = (operator @ state[..., None]).squeeze(-1)
        prefixes = packed.reshape(batch, 3, STATE_DIMENSION)
        outputs.append(prefixes)
        state = prefixes[:, 2]
    return torch.stack(outputs, dim=1).reshape(
        batch, chunks * 3, STATE_DIMENSION
    )


def triton_indexed_chunk_recurrence(
    prefix_table: torch.Tensor,
    left_index: torch.Tensor,
    middle_index: torch.Tensor,
    right_index: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """Run the one-program-per-sequence recurrence and initial-state backward."""

    batch, chunks = _validate_inputs(
        prefix_table,
        left_index,
        middle_index,
        right_index,
        initial_state,
    )
    if not triton_is_available():
        raise RuntimeError("the fused chunk recurrence requires Triton and CUDA")
    if prefix_table.device.type != "cuda" or prefix_table.dtype != torch.float32:
        raise ValueError("the fused chunk recurrence requires CUDA float32 inputs")
    if prefix_table.requires_grad:
        raise ValueError(
            "the fused path does not implement prefix-table gradients; use eager"
        )
    packed = _CompiledChunkRecurrence.apply(
        prefix_table.contiguous(),
        left_index.contiguous(),
        middle_index.contiguous(),
        right_index.contiguous(),
        initial_state.contiguous(),
    )
    return packed.reshape(batch, chunks * 3, STATE_DIMENSION)


def indexed_chunk_recurrence(
    prefix_table: torch.Tensor,
    left_index: torch.Tensor,
    middle_index: torch.Tensor,
    right_index: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    backend: ChunkRecurrenceBackend = "auto",
) -> torch.Tensor:
    """Dispatch to fused CUDA when safe, retaining full eager autograd otherwise."""

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
        return triton_indexed_chunk_recurrence(
            prefix_table,
            left_index,
            middle_index,
            right_index,
            initial_state,
        )
    return eager_indexed_chunk_recurrence(
        prefix_table,
        left_index,
        middle_index,
        right_index,
        initial_state,
    )


__all__ = [
    "ChunkRecurrenceBackend",
    "eager_indexed_chunk_recurrence",
    "indexed_chunk_recurrence",
    "triton_indexed_chunk_recurrence",
    "triton_is_available",
]
