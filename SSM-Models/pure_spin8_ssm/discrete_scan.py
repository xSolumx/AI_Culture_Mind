"""Compiled discrete-token inference for frozen Pure Spin(8) actions.

Training may identify a finite token dictionary in continuous Lie coordinates.
After identification, those coordinates can be compiled once into a table of
faithful triality actions.  This module scans that frozen table directly and
therefore removes the per-token matrix exponential/factorization and the eager
prefix tree from inference.

The Triton backend is a streaming recurrence: one program owns one
``(batch, representation)`` state, keeps its eight scalars in registers, and
emits every causal prefix.  It differentiates the initial state only.  A
trainable action table automatically uses the eager reference path.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from pure_spin8_ssm import SPIN8_DIM, __version__

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional environment boundary
    triton = None
    tl = None

DiscreteScanBackend = Literal["auto", "eager", "triton"]


if triton is not None:

    @triton.jit
    def _discrete_spin8_forward_kernel(
        action_table_pointer,
        token_pointer,
        initial_pointer,
        output_pointer,
        length,
        representations: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        representation_index = tl.program_id(1)
        rows = tl.arange(0, 8)
        columns = tl.arange(0, 8)
        state_base = (
            batch_index * representations + representation_index
        ) * 8
        state = tl.load(initial_pointer + state_base + columns)

        for position in tl.range(0, length):
            token_offset = batch_index * length + position
            token = tl.load(token_pointer + token_offset)
            table_base = (
                token * representations + representation_index
            ) * 8 * 8
            offsets = (
                table_base
                + rows[:, None] * 8
                + columns[None, :]
            )
            action = tl.load(action_table_pointer + offsets)
            state = tl.sum(action * state[None, :], axis=1)
            output_base = (
                (token_offset * representations + representation_index)
                * 8
            )
            tl.store(output_pointer + output_base + rows, state)

    @triton.jit
    def _discrete_spin8_backward_kernel(
        action_table_pointer,
        token_pointer,
        output_gradient_pointer,
        initial_gradient_pointer,
        length,
        representations: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        representation_index = tl.program_id(1)
        rows = tl.arange(0, 8)
        columns = tl.arange(0, 8)
        carry = tl.zeros([8], dtype=tl.float32)

        for reverse_position in tl.range(0, length):
            position = length - 1 - reverse_position
            token_offset = batch_index * length + position
            token = tl.load(token_pointer + token_offset)
            table_base = (
                token * representations + representation_index
            ) * 8 * 8
            offsets = (
                table_base
                + rows[:, None] * 8
                + columns[None, :]
            )
            action = tl.load(action_table_pointer + offsets)
            output_base = (
                (token_offset * representations + representation_index)
                * 8
            )
            direct = tl.load(output_gradient_pointer + output_base + rows)
            direct += carry
            carry = tl.sum(action * direct[:, None], axis=0)

        state_base = (
            batch_index * representations + representation_index
        ) * 8
        tl.store(initial_gradient_pointer + state_base + columns, carry)

    class _DiscreteSpin8Recurrence(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: torch.autograd.function.FunctionCtx,
            action_table: torch.Tensor,
            tokens: torch.Tensor,
            initial_state: torch.Tensor,
        ) -> torch.Tensor:
            batch, length = tokens.shape
            representations = action_table.shape[1]
            output = torch.empty(
                batch,
                length,
                representations,
                SPIN8_DIM,
                dtype=initial_state.dtype,
                device=initial_state.device,
            )
            _discrete_spin8_forward_kernel[(batch, representations)](
                action_table,
                tokens,
                initial_state,
                output,
                length,
                representations,
                num_warps=1,
            )
            ctx.save_for_backward(action_table, tokens)
            return output

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx,
            output_gradient: torch.Tensor,
        ) -> tuple[None, None, torch.Tensor]:
            action_table, tokens = ctx.saved_tensors
            output_gradient = output_gradient.contiguous()
            batch, length = tokens.shape
            representations = action_table.shape[1]
            initial_gradient = torch.empty(
                batch,
                representations,
                SPIN8_DIM,
                dtype=output_gradient.dtype,
                device=output_gradient.device,
            )
            _discrete_spin8_backward_kernel[(batch, representations)](
                action_table,
                tokens,
                output_gradient,
                initial_gradient,
                length,
                representations,
                num_warps=1,
            )
            return None, None, initial_gradient


def triton_is_available() -> bool:
    """Return whether the optional Triton CUDA path can be attempted."""

    return triton is not None and torch.cuda.is_available()


def _validate_inputs(
    action_table: torch.Tensor,
    tokens: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    validate_token_range: bool,
) -> tuple[int, int, int]:
    if action_table.ndim != 4 or action_table.shape[-2:] != (
        SPIN8_DIM,
        SPIN8_DIM,
    ):
        raise ValueError("action_table must have shape (vocabulary,reps,8,8)")
    if action_table.shape[0] < 1 or action_table.shape[1] < 1:
        raise ValueError("action_table vocabulary and representations must be nonempty")
    if tokens.ndim != 2 or tokens.shape[1] < 1:
        raise ValueError("tokens must have nonempty shape (batch,length)")
    if tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("tokens must use int32 or int64")
    batch, length = tokens.shape
    representations = action_table.shape[1]
    if initial_state.shape != (batch, representations, SPIN8_DIM):
        raise ValueError(
            "initial_state must have shape (batch,representations,8)"
        )
    if action_table.device != tokens.device or tokens.device != initial_state.device:
        raise ValueError("action_table, tokens, and initial_state must share a device")
    if action_table.dtype != initial_state.dtype or not action_table.is_floating_point():
        raise ValueError("action_table and initial_state must share a floating dtype")
    if validate_token_range:
        minimum, maximum = torch.aminmax(tokens)
        if int(minimum) < 0 or int(maximum) >= action_table.shape[0]:
            raise ValueError("tokens contain an index outside the action table")
    return batch, length, representations


def eager_discrete_spin8_scan(
    action_table: torch.Tensor,
    tokens: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    validate_token_range: bool = True,
) -> torch.Tensor:
    """Differentiable sequential PyTorch oracle over a token-action table."""

    _, length, _ = _validate_inputs(
        action_table,
        tokens,
        initial_state,
        validate_token_range=validate_token_range,
    )
    state = initial_state
    outputs = []
    for position in range(length):
        action = action_table[tokens[:, position]]
        state = torch.einsum("brij,brj->bri", action, state)
        outputs.append(state)
    return torch.stack(outputs, dim=1)


def triton_discrete_spin8_scan(
    action_table: torch.Tensor,
    tokens: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    validate_token_range: bool = True,
) -> torch.Tensor:
    """Run fused frozen-table recurrence with initial-state differentiation."""

    batch, _, representations = _validate_inputs(
        action_table,
        tokens,
        initial_state,
        validate_token_range=validate_token_range,
    )
    if not triton_is_available():
        raise RuntimeError("the fused discrete Spin(8) scan requires Triton and CUDA")
    if action_table.device.type != "cuda" or action_table.dtype != torch.float32:
        raise ValueError("the fused discrete Spin(8) scan requires CUDA float32")
    if action_table.requires_grad:
        raise ValueError(
            "the fused path does not implement action-table gradients; use eager"
        )
    if initial_state.shape != (batch, representations, SPIN8_DIM):
        raise AssertionError("validated state shape changed unexpectedly")
    return _DiscreteSpin8Recurrence.apply(
        action_table.contiguous(), tokens.contiguous(), initial_state.contiguous()
    )


def discrete_spin8_scan(
    action_table: torch.Tensor,
    tokens: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    backend: DiscreteScanBackend = "auto",
    validate_token_range: bool = True,
) -> torch.Tensor:
    """Dispatch to fused inference when safe and eager autograd otherwise."""

    if backend not in ("auto", "eager", "triton"):
        raise ValueError("backend must be 'auto', 'eager', or 'triton'")
    use_triton = backend == "triton" or (
        backend == "auto"
        and triton_is_available()
        and action_table.device.type == "cuda"
        and action_table.dtype == torch.float32
        and not action_table.requires_grad
    )
    if use_triton:
        return triton_discrete_spin8_scan(
            action_table,
            tokens,
            initial_state,
            validate_token_range=validate_token_range,
        )
    return eager_discrete_spin8_scan(
        action_table,
        tokens,
        initial_state,
        validate_token_range=validate_token_range,
    )


class CompiledSpin8TokenTracker(nn.Module):
    """Inference module containing a frozen token-action dictionary."""

    def __init__(
        self,
        action_table: torch.Tensor,
        initial_state: torch.Tensor,
        *,
        representations: Sequence[str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        representations = tuple(representations)
        if action_table.shape != (
            action_table.shape[0],
            len(representations),
            SPIN8_DIM,
            SPIN8_DIM,
        ):
            raise ValueError("action_table shape does not match representations")
        if initial_state.shape != (len(representations), SPIN8_DIM):
            raise ValueError("initial_state must have shape (representations,8)")
        self.representations = representations
        self.metadata = dict(metadata or {})
        self.register_buffer("action_table", action_table.detach().clone())
        self.register_buffer("initial_state", initial_state.detach().clone())

    @property
    def recurrent_state_scalars(self) -> int:
        return len(self.representations) * SPIN8_DIM

    def forward(
        self,
        tokens: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        backend: DiscreteScanBackend = "auto",
        validate_token_range: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            state = self.initial_state.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        states = discrete_spin8_scan(
            self.action_table,
            tokens,
            state,
            backend=backend,
            validate_token_range=validate_token_range,
        )
        return states, states[:, -1]

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save(
            {
                "format_version": 1,
                "model_type": "compiled_spin8_token_tracker",
                "model_version": __version__,
                "representations": self.representations,
                "action_table": self.action_table.detach().cpu(),
                "initial_state": self.initial_state.detach().cpu(),
                "metadata": self.metadata,
            },
            Path(path),
        )

    @classmethod
    def load_checkpoint(
        cls, path: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> CompiledSpin8TokenTracker:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if payload.get("model_type") != "compiled_spin8_token_tracker":
            raise ValueError("not a compiled Spin(8) token checkpoint")
        if payload.get("format_version") != 1:
            raise ValueError("unsupported compiled Spin(8) checkpoint format")
        return cls(
            payload["action_table"],
            payload["initial_state"],
            representations=payload["representations"],
            metadata=payload.get("metadata", {}),
        )


__all__ = [
    "CompiledSpin8TokenTracker",
    "DiscreteScanBackend",
    "discrete_spin8_scan",
    "eager_discrete_spin8_scan",
    "triton_discrete_spin8_scan",
    "triton_is_available",
]
