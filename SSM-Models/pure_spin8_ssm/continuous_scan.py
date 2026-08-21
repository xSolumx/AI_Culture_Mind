"""Compiled continuous-action Spin(8) recurrences.

The scalar Triton path is a full-training primitive: it differentiates actions,
retentions, drives, and initial states.  One program owns one
``(batch,channel,representation)`` eight-vector and walks time in registers.

The Tensor-Core path is inference-only and requires a genuinely shared action
across isotypic channels.  It packs sixteen copies as columns of one FP16
matrix product.  It is never selected merely because FP16 is available.
"""

from __future__ import annotations

from typing import Literal

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional WSL/CUDA boundary
    triton = None
    tl = None


ContinuousBackend = Literal[
    "auto", "eager", "triton_scalar", "triton_tensor_core"
]


if triton is not None:

    @triton.jit
    def _continuous_forward_kernel(
        action_pointer,
        scale_pointer,
        drive_pointer,
        initial_pointer,
        output_pointer,
        length,
        channels: tl.constexpr,
        representations: tl.constexpr,
        shared_action: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        channel_index = tl.program_id(1)
        representation_index = tl.program_id(2)
        rows = tl.arange(0, 8)
        columns = tl.arange(0, 8)
        state_base = (
            (batch_index * channels + channel_index) * representations
            + representation_index
        ) * 8
        state = tl.load(initial_pointer + state_base + columns)

        for position in tl.range(0, length):
            step = batch_index * length + position
            if shared_action:
                action_base = (step * representations + representation_index) * 64
            else:
                action_base = (
                    (step * channels + channel_index) * representations
                    + representation_index
                ) * 64
            action = tl.load(
                action_pointer
                + action_base
                + rows[:, None] * 8
                + columns[None, :]
            )
            scale = tl.load(scale_pointer + step * channels + channel_index)
            drive_base = (
                (step * channels + channel_index) * representations
                + representation_index
            ) * 8
            drive = tl.load(drive_pointer + drive_base + rows)
            state = scale * tl.sum(action * state[None, :], axis=1) + drive
            tl.store(output_pointer + drive_base + rows, state)

    @triton.jit
    def _continuous_backward_kernel(
        action_pointer,
        scale_pointer,
        initial_pointer,
        output_pointer,
        output_gradient_pointer,
        action_gradient_pointer,
        scale_gradient_pointer,
        drive_gradient_pointer,
        initial_gradient_pointer,
        length,
        channels: tl.constexpr,
        representations: tl.constexpr,
        shared_action: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        channel_index = tl.program_id(1)
        representation_index = tl.program_id(2)
        rows = tl.arange(0, 8)
        columns = tl.arange(0, 8)
        carry = tl.zeros([8], dtype=tl.float32)
        initial_base = (
            (batch_index * channels + channel_index) * representations
            + representation_index
        ) * 8

        for reverse_position in tl.range(0, length):
            position = length - 1 - reverse_position
            step = batch_index * length + position
            state_base = (
                (step * channels + channel_index) * representations
                + representation_index
            ) * 8
            direct = tl.load(output_gradient_pointer + state_base + rows) + carry
            if position == 0:
                previous = tl.load(initial_pointer + initial_base + columns)
            else:
                previous_step = batch_index * length + position - 1
                previous_base = (
                    (previous_step * channels + channel_index) * representations
                    + representation_index
                ) * 8
                previous = tl.load(output_pointer + previous_base + columns)

            if shared_action:
                action_base = (step * representations + representation_index) * 64
            else:
                action_base = (
                    (step * channels + channel_index) * representations
                    + representation_index
                ) * 64
            action_offsets = (
                action_base + rows[:, None] * 8 + columns[None, :]
            )
            action = tl.load(action_pointer + action_offsets)
            scale_offset = step * channels + channel_index
            scale = tl.load(scale_pointer + scale_offset)
            rotated = tl.sum(action * previous[None, :], axis=1)

            tl.store(drive_gradient_pointer + state_base + rows, direct)
            tl.atomic_add(
                scale_gradient_pointer + scale_offset,
                tl.sum(direct * rotated, axis=0),
            )
            action_gradient = scale * direct[:, None] * previous[None, :]
            if shared_action:
                tl.atomic_add(
                    action_gradient_pointer + action_offsets,
                    action_gradient,
                )
            else:
                tl.store(
                    action_gradient_pointer + action_offsets,
                    action_gradient,
                )
            carry = scale * tl.sum(action * direct[:, None], axis=0)

        tl.store(initial_gradient_pointer + initial_base + columns, carry)

    @triton.jit
    def _continuous_tensor_core_kernel(
        action_pointer,
        scale_pointer,
        drive_pointer,
        initial_pointer,
        output_pointer,
        length,
        channels,
        representations: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        representation_index = tl.program_id(1)
        channel_block = tl.program_id(2)
        channel_offsets = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
        padded_rows = tl.arange(0, 16)
        padded_columns = tl.arange(0, 16)
        state = tl.load(
            initial_pointer
            + (
                (batch_index * channels + channel_offsets[None, :])
                * representations
                + representation_index
            )
            * 8
            + padded_columns[:, None],
            mask=(padded_columns[:, None] < 8)
            & (channel_offsets[None, :] < channels),
            other=0.0,
        ).to(tl.float16)

        for position in tl.range(0, length):
            step = batch_index * length + position
            action_base = (step * representations + representation_index) * 64
            action = tl.load(
                action_pointer
                + action_base
                + padded_rows[:, None] * 8
                + padded_columns[None, :],
                mask=(padded_rows[:, None] < 8)
                & (padded_columns[None, :] < 8),
                other=0.0,
            ).to(tl.float16)
            rotated = tl.dot(action, state)
            scale = tl.load(
                scale_pointer + step * channels + channel_offsets,
                mask=channel_offsets < channels,
                other=0.0,
            )
            drive_base = (
                (step * channels + channel_offsets[None, :]) * representations
                + representation_index
            ) * 8
            drive = tl.load(
                drive_pointer + drive_base + padded_rows[:, None],
                mask=(padded_rows[:, None] < 8)
                & (channel_offsets[None, :] < channels),
                other=0.0,
            )
            state = (rotated * scale[None, :] + drive).to(tl.float16)
            tl.store(
                output_pointer + drive_base + padded_rows[:, None],
                state,
                mask=(padded_rows[:, None] < 8)
                & (channel_offsets[None, :] < channels),
            )


def triton_continuous_is_available() -> bool:
    return triton is not None and torch.cuda.is_available()


def _validate(
    action: torch.Tensor,
    scale: torch.Tensor,
    drive: torch.Tensor,
    initial: torch.Tensor,
) -> tuple[int, int, int, int, bool]:
    if scale.ndim != 3:
        raise ValueError("scale must have shape (batch,length,channels)")
    batch, length, channels = scale.shape
    if drive.ndim != 5 or drive.shape[:3] != scale.shape or drive.shape[-1] != 8:
        raise ValueError("drive must have shape (batch,length,channels,reps,8)")
    representations = drive.shape[-2]
    shared = action.ndim == 5
    expected_action = (
        (batch, length, representations, 8, 8)
        if shared
        else (batch, length, channels, representations, 8, 8)
    )
    if action.shape != expected_action:
        raise ValueError(f"action must have shape {expected_action}")
    if initial.shape != (batch, channels, representations, 8):
        raise ValueError(
            "initial must have shape (batch,channels,representations,8)"
        )
    if len({action.device, scale.device, drive.device, initial.device}) != 1:
        raise ValueError("all recurrence tensors must share a device")
    if len({action.dtype, scale.dtype, drive.dtype, initial.dtype}) != 1:
        raise ValueError("all recurrence tensors must share a dtype")
    return batch, length, channels, representations, shared


def eager_continuous_spin8_scan(
    action: torch.Tensor,
    scale: torch.Tensor,
    drive: torch.Tensor,
    initial: torch.Tensor,
) -> torch.Tensor:
    """Differentiable sequential oracle for shared or channel-wise actions."""

    _, length, _, _, shared = _validate(action, scale, drive, initial)
    state = initial
    outputs = []
    for position in range(length):
        selected = action[:, position]
        if shared:
            rotated = torch.einsum("brij,bcrj->bcri", selected, state)
        else:
            rotated = torch.einsum("bcrij,bcrj->bcri", selected, state)
        state = scale[:, position, :, None, None] * rotated + drive[:, position]
        outputs.append(state)
    return torch.stack(outputs, dim=1)


if triton is not None:

    class _ContinuousSpin8Recurrence(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: torch.autograd.function.FunctionCtx,
            action: torch.Tensor,
            scale: torch.Tensor,
            drive: torch.Tensor,
            initial: torch.Tensor,
        ) -> torch.Tensor:
            batch, length, channels, representations, shared = _validate(
                action, scale, drive, initial
            )
            output = torch.empty_like(drive)
            _continuous_forward_kernel[(batch, channels, representations)](
                action,
                scale,
                drive,
                initial,
                output,
                length,
                channels,
                representations,
                shared,
                num_warps=1,
            )
            ctx.shared = shared
            ctx.save_for_backward(action, scale, initial, output)
            return output

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx,
            output_gradient: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            action, scale, initial, output = ctx.saved_tensors
            output_gradient = output_gradient.contiguous()
            batch, length, channels, representations, _ = _validate(
                action, scale, output, initial
            )
            action_gradient = torch.zeros_like(action)
            scale_gradient = torch.zeros_like(scale)
            drive_gradient = torch.empty_like(output)
            initial_gradient = torch.empty_like(initial)
            _continuous_backward_kernel[(batch, channels, representations)](
                action,
                scale,
                initial,
                output,
                output_gradient,
                action_gradient,
                scale_gradient,
                drive_gradient,
                initial_gradient,
                length,
                channels,
                representations,
                ctx.shared,
                num_warps=1,
            )
            return (
                action_gradient,
                scale_gradient,
                drive_gradient,
                initial_gradient,
            )


def triton_scalar_continuous_spin8_scan(
    action: torch.Tensor,
    scale: torch.Tensor,
    drive: torch.Tensor,
    initial: torch.Tensor,
) -> torch.Tensor:
    """Run the full-gradient float32 register-resident recurrence."""

    _validate(action, scale, drive, initial)
    if not triton_continuous_is_available():
        raise RuntimeError("the continuous Triton scan requires CUDA and Triton")
    if action.device.type != "cuda" or action.dtype not in (
        torch.float16,
        torch.float32,
    ):
        raise ValueError("the scalar Triton scan requires CUDA float16/float32 tensors")
    if action.dtype == torch.float16 and any(
        tensor.requires_grad for tensor in (action, scale, drive, initial)
    ):
        raise ValueError("float16 scalar recurrence is inference-only in v2.1.1")
    return _ContinuousSpin8Recurrence.apply(
        action.contiguous(),
        scale.contiguous(),
        drive.contiguous(),
        initial.contiguous(),
    )


@torch.inference_mode()
def triton_tensor_core_continuous_spin8_scan(
    action: torch.Tensor,
    scale: torch.Tensor,
    drive: torch.Tensor,
    initial: torch.Tensor,
) -> torch.Tensor:
    """Run an FP16 shared-action isotypic recurrence using ``mma.sync``."""

    batch, length, channels, representations, shared = _validate(
        action, scale, drive, initial
    )
    if not triton_continuous_is_available():
        raise RuntimeError("the Tensor-Core scan requires CUDA and Triton")
    if action.device.type != "cuda" or action.dtype != torch.float16:
        raise ValueError("the Tensor-Core scan requires CUDA float16 tensors")
    if not shared:
        raise ValueError("Tensor-Core packing requires one shared isotypic action")
    if any(tensor.requires_grad for tensor in (action, scale, drive, initial)):
        raise ValueError("the Tensor-Core path is inference-only")
    output = torch.empty_like(drive)
    _continuous_tensor_core_kernel[
        (batch, representations, triton.cdiv(channels, 16))
    ](
        action.contiguous(),
        scale.contiguous(),
        drive.contiguous(),
        initial.contiguous(),
        output,
        length,
        channels,
        representations,
        BLOCK_C=16,
        num_warps=4,
    )
    return output


def continuous_spin8_scan(
    action: torch.Tensor,
    scale: torch.Tensor,
    drive: torch.Tensor,
    initial: torch.Tensor,
    *,
    backend: ContinuousBackend = "auto",
) -> torch.Tensor:
    """Dispatch a continuous recurrence without weakening precision contracts."""

    if backend not in ("auto", "eager", "triton_scalar", "triton_tensor_core"):
        raise ValueError("unknown continuous Spin(8) backend")
    if backend == "eager":
        return eager_continuous_spin8_scan(action, scale, drive, initial)
    if backend == "triton_scalar":
        return triton_scalar_continuous_spin8_scan(action, scale, drive, initial)
    if backend == "triton_tensor_core":
        return triton_tensor_core_continuous_spin8_scan(
            action, scale, drive, initial
        )
    if (
        triton_continuous_is_available()
        and action.device.type == "cuda"
        and action.dtype in (torch.float16, torch.float32)
    ):
        return triton_scalar_continuous_spin8_scan(
            action, scale, drive, initial
        )
    return eager_continuous_spin8_scan(action, scale, drive, initial)


@torch.inference_mode()
def continuous_ptx_evidence(
    action: torch.Tensor,
    scale: torch.Tensor,
    drive: torch.Tensor,
    initial: torch.Tensor,
    *,
    backend: Literal["triton_scalar", "triton_tensor_core"],
) -> dict[str, int]:
    """Compile one kernel and count instruction evidence in generated PTX."""

    batch, length, channels, representations, shared = _validate(
        action, scale, drive, initial
    )
    output = torch.empty_like(drive)
    if backend == "triton_scalar":
        compiled = _continuous_forward_kernel[(batch, channels, representations)](
            action.contiguous(),
            scale.contiguous(),
            drive.contiguous(),
            initial.contiguous(),
            output,
            length,
            channels,
            representations,
            shared,
            num_warps=1,
        )
    elif backend == "triton_tensor_core":
        if not shared:
            raise ValueError("Tensor-Core evidence requires shared actions")
        compiled = _continuous_tensor_core_kernel[
            (batch, representations, triton.cdiv(channels, 16))
        ](
            action.contiguous(),
            scale.contiguous(),
            drive.contiguous(),
            initial.contiguous(),
            output,
            length,
            channels,
            representations,
            BLOCK_C=16,
            num_warps=4,
        )
    else:
        raise ValueError("unknown PTX evidence backend")
    ptx = compiled.asm.get("ptx", "")
    return {
        "ptx_characters": len(ptx),
        "mma_sync_occurrences": ptx.count("mma.sync"),
        "fma_occurrences": ptx.count("fma."),
    }


__all__ = [
    "ContinuousBackend",
    "continuous_ptx_evidence",
    "continuous_spin8_scan",
    "eager_continuous_spin8_scan",
    "triton_continuous_is_available",
    "triton_scalar_continuous_spin8_scan",
    "triton_tensor_core_continuous_spin8_scan",
]
