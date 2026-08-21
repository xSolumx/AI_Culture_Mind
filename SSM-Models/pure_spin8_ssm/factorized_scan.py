"""Trainable coordinate-to-recurrence Spin(8) scan.

The maintained factorized chart represents one Spin(8) element as 28 ordered
plane factors.  Materializing their three ``8 x 8`` products is unnecessary
when the only consumer is a recurrent state: apply each factor directly to the
eight-vector in chronological order.

The Triton reverse pass uses orthogonality.  Starting from the rotated state,
each pre-factor state is recovered by the inverse factor while the adjoint is
transported by the same inverse.  This gives exact coordinate gradients in
``O(28)`` factor applications per token rather than storing 28 intermediate
states or recomputing an ``O(28^2)`` tape.  Gradients through a learned linear
controller follow through the returned coordinate gradient.

Version 2.1.2 deliberately supports the canonical full triality order
``(8v,8s+,8s-)`` in FP32.  Other representation selections and lower-precision
training remain explicit future compiler targets.
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


FactorizedBackend = Literal["auto", "eager", "triton"]


def _validate(
    coordinates: torch.Tensor,
    generators: torch.Tensor,
    scale: torch.Tensor,
    drive: torch.Tensor,
    initial: torch.Tensor,
) -> tuple[int, int, int]:
    if coordinates.ndim != 4 or coordinates.shape[-1] != 28:
        raise ValueError("coordinates must have shape (batch,length,channels,28)")
    batch, length, channels, _ = coordinates.shape
    if generators.shape != (3, 28, 8, 8):
        raise ValueError("generators must have canonical triality shape (3,28,8,8)")
    if scale.shape != (batch, length, channels):
        raise ValueError("scale must have shape (batch,length,channels)")
    if drive.shape != (batch, length, channels, 3, 8):
        raise ValueError("drive must have shape (batch,length,channels,3,8)")
    if initial.shape != (batch, channels, 3, 8):
        raise ValueError("initial must have shape (batch,channels,3,8)")
    tensors = (coordinates, generators, scale, drive, initial)
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all factorized-scan tensors must share a device")
    if len({tensor.dtype for tensor in tensors}) != 1:
        raise ValueError("all factorized-scan tensors must share a dtype")
    return batch, length, channels


def _validate_controller_scan(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    generators: torch.Tensor,
    scale: torch.Tensor,
    drive: torch.Tensor,
    initial: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[int, int, int, int]:
    if features.ndim != 3:
        raise ValueError("features must have shape (batch,length,input_size)")
    batch, length, input_size = features.shape
    if scale.ndim != 3 or scale.shape[:2] != (batch, length):
        raise ValueError("scale must have shape (batch,length,channels)")
    channels = scale.shape[-1]
    if weight.shape != (channels * 28, input_size):
        raise ValueError("controller weight has incompatible shape")
    if bias.shape != (channels * 28,):
        raise ValueError("controller bias has incompatible shape")
    if generators.shape != (3, 28, 8, 8):
        raise ValueError("generators must have canonical triality shape (3,28,8,8)")
    if drive.shape != (batch, length, channels, 3, 8):
        raise ValueError("drive must have shape (batch,length,channels,3,8)")
    if initial.shape != (batch, channels, 3, 8):
        raise ValueError("initial must have shape (batch,channels,3,8)")
    if gate.shape != (batch, length):
        raise ValueError("coordinate gate must have shape (batch,length)")
    tensors = (features, weight, bias, generators, scale, drive, initial, gate)
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all controller-scan tensors must share a device")
    if len({tensor.dtype for tensor in tensors}) != 1:
        raise ValueError("all controller-scan tensors must share a dtype")
    return batch, length, channels, input_size


def eager_factorized_coordinate_scan(
    coordinates: torch.Tensor,
    generators: torch.Tensor,
    scale: torch.Tensor,
    drive: torch.Tensor,
    initial: torch.Tensor,
) -> torch.Tensor:
    """Sequential differentiable oracle without materialized group actions."""

    _, length, _ = _validate(coordinates, generators, scale, drive, initial)
    state = initial
    outputs = []
    identity = torch.eye(8, dtype=state.dtype, device=state.device)
    generator_square = generators @ generators
    for position in range(length):
        angles = coordinates[:, position]
        for coordinate_index in range(28):
            angle = angles[..., coordinate_index]
            factors = []
            vector_generator = generators[0, coordinate_index]
            vector_factor = (
                identity
                + torch.sin(angle)[..., None, None] * vector_generator
                + (1.0 - torch.cos(angle))[..., None, None]
                * generator_square[0, coordinate_index]
            )
            factors.append(vector_factor)
            for representation_index in (1, 2):
                spinor_generator = generators[
                    representation_index, coordinate_index
                ]
                factors.append(
                    torch.cos(0.5 * angle)[..., None, None] * identity
                    + 2.0
                    * torch.sin(0.5 * angle)[..., None, None]
                    * spinor_generator
                )
            factor = torch.stack(factors, dim=-3)
            state = torch.einsum("bcrij,bcrj->bcri", factor, state)
        state = scale[:, position, :, None, None] * state + drive[:, position]
        outputs.append(state)
    return torch.stack(outputs, dim=1)


if triton is not None:

    @triton.jit
    def _apply_factor(
        state,
        generator_pointer,
        representation_index,
        coordinate_index,
        angle,
    ):
        rows = tl.arange(0, 8)
        columns = tl.arange(0, 8)
        generator_base = (
            representation_index * 28 + coordinate_index
        ) * 64
        generator = tl.load(
            generator_pointer
            + generator_base
            + rows[:, None] * 8
            + columns[None, :]
        )
        first = tl.sum(generator * state[None, :], axis=1)
        second = tl.sum(generator * first[None, :], axis=1)
        vector = representation_index == 0
        identity_coefficient = tl.where(vector, 1.0, tl.cos(0.5 * angle))
        first_coefficient = tl.where(
            vector, tl.sin(angle), 2.0 * tl.sin(0.5 * angle)
        )
        second_coefficient = tl.where(vector, 1.0 - tl.cos(angle), 0.0)
        return (
            identity_coefficient * state
            + first_coefficient * first
            + second_coefficient * second
        )

    @triton.jit
    def _factor_derivative(
        state,
        generator_pointer,
        representation_index,
        coordinate_index,
        angle,
    ):
        rows = tl.arange(0, 8)
        columns = tl.arange(0, 8)
        generator_base = (
            representation_index * 28 + coordinate_index
        ) * 64
        generator = tl.load(
            generator_pointer
            + generator_base
            + rows[:, None] * 8
            + columns[None, :]
        )
        first = tl.sum(generator * state[None, :], axis=1)
        second = tl.sum(generator * first[None, :], axis=1)
        vector = representation_index == 0
        vector_derivative = tl.cos(angle) * first + tl.sin(angle) * second
        spinor_derivative = (
            -0.5 * tl.sin(0.5 * angle) * state
            + tl.cos(0.5 * angle) * first
        )
        return tl.where(vector, vector_derivative, spinor_derivative)

    @triton.jit
    def _controller_angle(
        feature,
        weight_pointer,
        bias_pointer,
        channel_index,
        coordinate_index,
        input_size,
        gate,
        BLOCK_F: tl.constexpr,
    ):
        feature_offsets = tl.arange(0, BLOCK_F)
        row = channel_index * 28 + coordinate_index
        weight = tl.load(
            weight_pointer + row * input_size + feature_offsets,
            mask=feature_offsets < input_size,
            other=0.0,
        )
        bias = tl.load(bias_pointer + row)
        return gate * (tl.sum(weight * feature, axis=0) + bias)

    @triton.jit
    def _factorized_forward_kernel(
        coordinate_pointer,
        generator_pointer,
        scale_pointer,
        drive_pointer,
        initial_pointer,
        output_pointer,
        length,
        channels: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        channel_index = tl.program_id(1)
        representation_index = tl.program_id(2)
        rows = tl.arange(0, 8)
        state_base = (
            (batch_index * channels + channel_index) * 3
            + representation_index
        ) * 8
        state = tl.load(initial_pointer + state_base + rows).to(tl.float32)

        for position in tl.range(0, length):
            step = batch_index * length + position
            coordinate_base = (step * channels + channel_index) * 28
            for coordinate_index in tl.range(0, 28):
                angle = tl.load(
                    coordinate_pointer + coordinate_base + coordinate_index
                )
                state = _apply_factor(
                    state,
                    generator_pointer,
                    representation_index,
                    coordinate_index,
                    angle,
                )
            scale = tl.load(scale_pointer + step * channels + channel_index)
            output_base = (
                (step * channels + channel_index) * 3
                + representation_index
            ) * 8
            drive = tl.load(drive_pointer + output_base + rows)
            state = scale * state + drive
            tl.store(output_pointer + output_base + rows, state)

    @triton.jit
    def _factorized_backward_kernel(
        coordinate_pointer,
        generator_pointer,
        scale_pointer,
        initial_pointer,
        output_pointer,
        output_gradient_pointer,
        coordinate_gradient_pointer,
        scale_gradient_pointer,
        drive_gradient_pointer,
        initial_gradient_pointer,
        length,
        channels: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        channel_index = tl.program_id(1)
        representation_index = tl.program_id(2)
        rows = tl.arange(0, 8)
        carry = tl.zeros([8], dtype=tl.float32)
        initial_base = (
            (batch_index * channels + channel_index) * 3
            + representation_index
        ) * 8

        for reverse_position in tl.range(0, length):
            position = length - 1 - reverse_position
            step = batch_index * length + position
            output_base = (
                (step * channels + channel_index) * 3
                + representation_index
            ) * 8
            direct = tl.load(output_gradient_pointer + output_base + rows) + carry
            if position == 0:
                previous = tl.load(initial_pointer + initial_base + rows)
            else:
                previous_step = batch_index * length + position - 1
                previous_base = (
                    (previous_step * channels + channel_index) * 3
                    + representation_index
                ) * 8
                previous = tl.load(output_pointer + previous_base + rows)

            coordinate_base = (step * channels + channel_index) * 28
            rotated = previous.to(tl.float32)
            for coordinate_index in tl.range(0, 28):
                angle = tl.load(
                    coordinate_pointer + coordinate_base + coordinate_index
                )
                rotated = _apply_factor(
                    rotated,
                    generator_pointer,
                    representation_index,
                    coordinate_index,
                    angle,
                )

            scale_offset = step * channels + channel_index
            scale = tl.load(scale_pointer + scale_offset)
            tl.atomic_add(
                scale_gradient_pointer + scale_offset,
                tl.sum(direct * rotated, axis=0),
            )
            tl.store(drive_gradient_pointer + output_base + rows, direct)

            adjoint = scale * direct
            state_after = rotated
            for reverse_coordinate in tl.range(0, 28):
                coordinate_index = 27 - reverse_coordinate
                angle = tl.load(
                    coordinate_pointer + coordinate_base + coordinate_index
                )
                state_before = _apply_factor(
                    state_after,
                    generator_pointer,
                    representation_index,
                    coordinate_index,
                    -angle,
                )
                derivative = _factor_derivative(
                    state_before,
                    generator_pointer,
                    representation_index,
                    coordinate_index,
                    angle,
                )
                tl.atomic_add(
                    coordinate_gradient_pointer
                    + coordinate_base
                    + coordinate_index,
                    tl.sum(adjoint * derivative, axis=0),
                )
                adjoint = _apply_factor(
                    adjoint,
                    generator_pointer,
                    representation_index,
                    coordinate_index,
                    -angle,
                )
                state_after = state_before
            carry = adjoint

        tl.store(initial_gradient_pointer + initial_base + rows, carry)

    @triton.jit
    def _controller_factorized_forward_kernel(
        feature_pointer,
        weight_pointer,
        bias_pointer,
        generator_pointer,
        scale_pointer,
        drive_pointer,
        initial_pointer,
        gate_pointer,
        output_pointer,
        length,
        input_size,
        channels: tl.constexpr,
        BLOCK_F: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        channel_index = tl.program_id(1)
        representation_index = tl.program_id(2)
        rows = tl.arange(0, 8)
        feature_offsets = tl.arange(0, BLOCK_F)
        state_base = (
            (batch_index * channels + channel_index) * 3
            + representation_index
        ) * 8
        state = tl.load(initial_pointer + state_base + rows).to(tl.float32)

        for position in tl.range(0, length):
            step = batch_index * length + position
            feature = tl.load(
                feature_pointer + step * input_size + feature_offsets,
                mask=feature_offsets < input_size,
                other=0.0,
            )
            gate = tl.load(gate_pointer + step)
            for coordinate_index in tl.range(0, 28):
                angle = _controller_angle(
                    feature,
                    weight_pointer,
                    bias_pointer,
                    channel_index,
                    coordinate_index,
                    input_size,
                    gate,
                    BLOCK_F,
                )
                state = _apply_factor(
                    state,
                    generator_pointer,
                    representation_index,
                    coordinate_index,
                    angle,
                )
            scale = tl.load(scale_pointer + step * channels + channel_index)
            output_base = (
                (step * channels + channel_index) * 3
                + representation_index
            ) * 8
            drive = tl.load(drive_pointer + output_base + rows)
            state = scale * state + drive
            tl.store(output_pointer + output_base + rows, state)

    @triton.jit
    def _controller_factorized_backward_kernel(
        feature_pointer,
        weight_pointer,
        bias_pointer,
        generator_pointer,
        scale_pointer,
        initial_pointer,
        gate_pointer,
        output_pointer,
        output_gradient_pointer,
        feature_gradient_pointer,
        weight_gradient_pointer,
        bias_gradient_pointer,
        scale_gradient_pointer,
        drive_gradient_pointer,
        initial_gradient_pointer,
        length,
        input_size,
        channels: tl.constexpr,
        BLOCK_F: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        channel_index = tl.program_id(1)
        representation_index = tl.program_id(2)
        rows = tl.arange(0, 8)
        feature_offsets = tl.arange(0, BLOCK_F)
        carry = tl.zeros([8], dtype=tl.float32)
        initial_base = (
            (batch_index * channels + channel_index) * 3
            + representation_index
        ) * 8

        for reverse_position in tl.range(0, length):
            position = length - 1 - reverse_position
            step = batch_index * length + position
            output_base = (
                (step * channels + channel_index) * 3
                + representation_index
            ) * 8
            direct = tl.load(output_gradient_pointer + output_base + rows) + carry
            if position == 0:
                previous = tl.load(initial_pointer + initial_base + rows)
            else:
                previous_step = batch_index * length + position - 1
                previous_base = (
                    (previous_step * channels + channel_index) * 3
                    + representation_index
                ) * 8
                previous = tl.load(output_pointer + previous_base + rows)
            feature_base = step * input_size
            feature = tl.load(
                feature_pointer + feature_base + feature_offsets,
                mask=feature_offsets < input_size,
                other=0.0,
            )
            gate = tl.load(gate_pointer + step)

            rotated = previous.to(tl.float32)
            for coordinate_index in tl.range(0, 28):
                angle = _controller_angle(
                    feature,
                    weight_pointer,
                    bias_pointer,
                    channel_index,
                    coordinate_index,
                    input_size,
                    gate,
                    BLOCK_F,
                )
                rotated = _apply_factor(
                    rotated,
                    generator_pointer,
                    representation_index,
                    coordinate_index,
                    angle,
                )

            scale_offset = step * channels + channel_index
            scale = tl.load(scale_pointer + scale_offset)
            tl.atomic_add(
                scale_gradient_pointer + scale_offset,
                tl.sum(direct * rotated, axis=0),
            )
            tl.store(drive_gradient_pointer + output_base + rows, direct)

            adjoint = scale * direct
            state_after = rotated
            for reverse_coordinate in tl.range(0, 28):
                coordinate_index = 27 - reverse_coordinate
                angle = _controller_angle(
                    feature,
                    weight_pointer,
                    bias_pointer,
                    channel_index,
                    coordinate_index,
                    input_size,
                    gate,
                    BLOCK_F,
                )
                state_before = _apply_factor(
                    state_after,
                    generator_pointer,
                    representation_index,
                    coordinate_index,
                    -angle,
                )
                derivative = _factor_derivative(
                    state_before,
                    generator_pointer,
                    representation_index,
                    coordinate_index,
                    angle,
                )
                angle_gradient = gate * tl.sum(adjoint * derivative, axis=0)
                controller_row = channel_index * 28 + coordinate_index
                weight_offsets = (
                    controller_row * input_size + feature_offsets
                )
                weight = tl.load(
                    weight_pointer + weight_offsets,
                    mask=feature_offsets < input_size,
                    other=0.0,
                )
                tl.atomic_add(
                    weight_gradient_pointer + weight_offsets,
                    angle_gradient * feature,
                    mask=feature_offsets < input_size,
                )
                tl.atomic_add(
                    bias_gradient_pointer + controller_row,
                    angle_gradient,
                )
                tl.atomic_add(
                    feature_gradient_pointer + feature_base + feature_offsets,
                    angle_gradient * weight,
                    mask=feature_offsets < input_size,
                )
                adjoint = _apply_factor(
                    adjoint,
                    generator_pointer,
                    representation_index,
                    coordinate_index,
                    -angle,
                )
                state_after = state_before
            carry = adjoint

        tl.store(initial_gradient_pointer + initial_base + rows, carry)

    class _FactorizedCoordinateRecurrence(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: torch.autograd.function.FunctionCtx,
            coordinates: torch.Tensor,
            generators: torch.Tensor,
            scale: torch.Tensor,
            drive: torch.Tensor,
            initial: torch.Tensor,
        ) -> torch.Tensor:
            batch, length, channels = _validate(
                coordinates, generators, scale, drive, initial
            )
            output = torch.empty_like(drive)
            _factorized_forward_kernel[(batch, channels, 3)](
                coordinates,
                generators,
                scale,
                drive,
                initial,
                output,
                length,
                channels,
                num_warps=1,
            )
            ctx.save_for_backward(coordinates, generators, scale, initial, output)
            return output

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx,
            output_gradient: torch.Tensor,
        ) -> tuple[
            torch.Tensor,
            None,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            coordinates, generators, scale, initial, output = ctx.saved_tensors
            batch, length, channels = coordinates.shape[:3]
            coordinate_gradient = torch.zeros_like(coordinates)
            scale_gradient = torch.zeros_like(scale)
            drive_gradient = torch.empty_like(output)
            initial_gradient = torch.empty_like(initial)
            _factorized_backward_kernel[(batch, channels, 3)](
                coordinates,
                generators,
                scale,
                initial,
                output,
                output_gradient.contiguous(),
                coordinate_gradient,
                scale_gradient,
                drive_gradient,
                initial_gradient,
                length,
                channels,
                num_warps=1,
            )
            return (
                coordinate_gradient,
                None,
                scale_gradient,
                drive_gradient,
                initial_gradient,
            )

    class _ControllerFactorizedRecurrence(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: torch.autograd.function.FunctionCtx,
            features: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            generators: torch.Tensor,
            scale: torch.Tensor,
            drive: torch.Tensor,
            initial: torch.Tensor,
            gate: torch.Tensor,
        ) -> torch.Tensor:
            batch, length, channels, input_size = _validate_controller_scan(
                features,
                weight,
                bias,
                generators,
                scale,
                drive,
                initial,
                gate,
            )
            block_f = triton.next_power_of_2(input_size)
            output = torch.empty_like(drive)
            _controller_factorized_forward_kernel[(batch, channels, 3)](
                features,
                weight,
                bias,
                generators,
                scale,
                drive,
                initial,
                gate,
                output,
                length,
                input_size,
                channels,
                BLOCK_F=block_f,
                num_warps=1,
            )
            ctx.channels = channels
            ctx.input_size = input_size
            ctx.block_f = block_f
            ctx.save_for_backward(
                features, weight, bias, generators, scale, initial, gate, output
            )
            return output

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx,
            output_gradient: torch.Tensor,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            None,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            None,
        ]:
            (
                features,
                weight,
                bias,
                generators,
                scale,
                initial,
                gate,
                output,
            ) = ctx.saved_tensors
            batch, length = features.shape[:2]
            feature_gradient = torch.zeros_like(features)
            weight_gradient = torch.zeros_like(weight)
            bias_gradient = torch.zeros_like(bias)
            scale_gradient = torch.zeros_like(scale)
            drive_gradient = torch.empty_like(output)
            initial_gradient = torch.empty_like(initial)
            _controller_factorized_backward_kernel[
                (batch, ctx.channels, 3)
            ](
                features,
                weight,
                bias,
                generators,
                scale,
                initial,
                gate,
                output,
                output_gradient.contiguous(),
                feature_gradient,
                weight_gradient,
                bias_gradient,
                scale_gradient,
                drive_gradient,
                initial_gradient,
                length,
                ctx.input_size,
                ctx.channels,
                BLOCK_F=ctx.block_f,
                num_warps=1,
            )
            return (
                feature_gradient,
                weight_gradient,
                bias_gradient,
                None,
                scale_gradient,
                drive_gradient,
                initial_gradient,
                None,
            )


def triton_factorized_coordinate_scan(
    coordinates: torch.Tensor,
    generators: torch.Tensor,
    scale: torch.Tensor,
    drive: torch.Tensor,
    initial: torch.Tensor,
) -> torch.Tensor:
    """Run the FP32 full-gradient coordinate-to-recurrence kernel."""

    _validate(coordinates, generators, scale, drive, initial)
    if triton is None or not torch.cuda.is_available():
        raise RuntimeError("the factorized Triton scan requires CUDA and Triton")
    if coordinates.device.type != "cuda" or coordinates.dtype != torch.float32:
        raise ValueError("the factorized Triton scan requires CUDA float32 tensors")
    return _FactorizedCoordinateRecurrence.apply(
        coordinates.contiguous(),
        generators.contiguous(),
        scale.contiguous(),
        drive.contiguous(),
        initial.contiguous(),
    )


def triton_controller_factorized_scan(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    generators: torch.Tensor,
    scale: torch.Tensor,
    drive: torch.Tensor,
    initial: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Fuse the learned coordinate controller, 28 factors, and recurrence."""

    tensors = (features, weight, bias, generators, scale, drive, initial, gate)
    if triton is None or not torch.cuda.is_available():
        raise RuntimeError("the fused controller scan requires CUDA and Triton")
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("the fused controller scan requires CUDA tensors")
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise ValueError("the fused controller scan requires float32 tensors")
    return _ControllerFactorizedRecurrence.apply(
        *(tensor.contiguous() for tensor in tensors)
    )


def factorized_coordinate_spin8_scan(
    coordinates: torch.Tensor,
    generators: torch.Tensor,
    scale: torch.Tensor,
    drive: torch.Tensor,
    initial: torch.Tensor,
    *,
    backend: FactorizedBackend = "auto",
) -> torch.Tensor:
    """Dispatch the maintained canonical factorized recurrence."""

    if backend not in ("auto", "eager", "triton"):
        raise ValueError("unknown factorized-coordinate backend")
    if backend == "eager":
        return eager_factorized_coordinate_scan(
            coordinates, generators, scale, drive, initial
        )
    if backend == "triton":
        return triton_factorized_coordinate_scan(
            coordinates, generators, scale, drive, initial
        )
    if (
        triton is not None
        and torch.cuda.is_available()
        and coordinates.device.type == "cuda"
        and coordinates.dtype == torch.float32
    ):
        return triton_factorized_coordinate_scan(
            coordinates, generators, scale, drive, initial
        )
    return eager_factorized_coordinate_scan(
        coordinates, generators, scale, drive, initial
    )


__all__ = [
    "FactorizedBackend",
    "eager_factorized_coordinate_scan",
    "factorized_coordinate_spin8_scan",
    "triton_controller_factorized_scan",
    "triton_factorized_coordinate_scan",
]
