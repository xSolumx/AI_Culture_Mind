"""Optional fused Triton recurrence for the octonion operator experiment.

The work-efficient matrix tree in :mod:`octonion_operator_scan` is the parallel
training path.  This module targets the complementary Linux/CUDA streaming and
chunk-inference path: one Triton program keeps each lane's eight-scalar state in
registers while iterating over a chunk.  A reverse kernel implements the exact
vector-Jacobian recurrence, so this path remains differentiable.

Triton is optional and is not supported by the repository's native Windows
Python environment.  The module imports safely there and reports the missing
capability instead of changing the canonical backend.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .octonion_operator_scan import OCTONION_DIM, unit_octonion

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by the native Windows suite
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _quaternion_product(
        a0,
        a1,
        a2,
        a3,
        b0,
        b1,
        b2,
        b3,
    ):
        return (
            a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
        )

    @triton.jit
    def _octonion_product(
        a0,
        a1,
        a2,
        a3,
        a4,
        a5,
        a6,
        a7,
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
    ):
        # Cayley--Dickson coordinates (a,b)(c,d) =
        # (a c - conjugate(d) b, d a + b conjugate(c)).  This convention is
        # exactly the repository's (123),(145),(176),(246),(257),(347),(365)
        # Fano orientation.
        ac0, ac1, ac2, ac3 = _quaternion_product(a0, a1, a2, a3, c0, c1, c2, c3)
        db0, db1, db2, db3 = _quaternion_product(c4, -c5, -c6, -c7, a4, a5, a6, a7)
        da0, da1, da2, da3 = _quaternion_product(c4, c5, c6, c7, a0, a1, a2, a3)
        bc0, bc1, bc2, bc3 = _quaternion_product(a4, a5, a6, a7, c0, -c1, -c2, -c3)
        return (
            ac0 - db0,
            ac1 - db1,
            ac2 - db2,
            ac3 - db3,
            da0 + bc0,
            da1 + bc1,
            da2 + bc2,
            da3 + bc3,
        )

    @triton.jit
    def _load_octonion(pointer, base, stride):
        return (
            tl.load(pointer + base + 0 * stride),
            tl.load(pointer + base + 1 * stride),
            tl.load(pointer + base + 2 * stride),
            tl.load(pointer + base + 3 * stride),
            tl.load(pointer + base + 4 * stride),
            tl.load(pointer + base + 5 * stride),
            tl.load(pointer + base + 6 * stride),
            tl.load(pointer + base + 7 * stride),
        )

    @triton.jit
    def _store_octonion(pointer, base, stride, v0, v1, v2, v3, v4, v5, v6, v7):
        tl.store(pointer + base + 0 * stride, v0)
        tl.store(pointer + base + 1 * stride, v1)
        tl.store(pointer + base + 2 * stride, v2)
        tl.store(pointer + base + 3 * stride, v3)
        tl.store(pointer + base + 4 * stride, v4)
        tl.store(pointer + base + 5 * stride, v5)
        tl.store(pointer + base + 6 * stride, v6)
        tl.store(pointer + base + 7 * stride, v7)

    @triton.jit
    def _octonion_recurrence_forward_kernel(
        tokens,
        initial,
        outputs,
        length,
        lanes: tl.constexpr,
        token_batch_stride: tl.constexpr,
        token_length_stride: tl.constexpr,
        token_lane_stride: tl.constexpr,
        token_component_stride: tl.constexpr,
        initial_batch_stride: tl.constexpr,
        initial_lane_stride: tl.constexpr,
        initial_component_stride: tl.constexpr,
        output_batch_stride: tl.constexpr,
        output_length_stride: tl.constexpr,
        output_lane_stride: tl.constexpr,
        output_component_stride: tl.constexpr,
    ):
        program = tl.program_id(0)
        batch_index = program // lanes
        lane_index = program - batch_index * lanes
        initial_base = (
            batch_index * initial_batch_stride + lane_index * initial_lane_stride
        )
        h0, h1, h2, h3, h4, h5, h6, h7 = _load_octonion(
            initial, initial_base, initial_component_stride
        )
        for position in tl.range(0, length):
            token_base = (
                batch_index * token_batch_stride
                + position * token_length_stride
                + lane_index * token_lane_stride
            )
            u0, u1, u2, u3, u4, u5, u6, u7 = _load_octonion(
                tokens, token_base, token_component_stride
            )
            h0, h1, h2, h3, h4, h5, h6, h7 = _octonion_product(
                u0,
                u1,
                u2,
                u3,
                u4,
                u5,
                u6,
                u7,
                h0,
                h1,
                h2,
                h3,
                h4,
                h5,
                h6,
                h7,
            )
            output_base = (
                batch_index * output_batch_stride
                + position * output_length_stride
                + lane_index * output_lane_stride
            )
            _store_octonion(
                outputs,
                output_base,
                output_component_stride,
                h0,
                h1,
                h2,
                h3,
                h4,
                h5,
                h6,
                h7,
            )

    @triton.jit
    def _octonion_recurrence_backward_kernel(
        tokens,
        initial,
        outputs,
        output_gradients,
        token_gradients,
        initial_gradients,
        length,
        lanes: tl.constexpr,
        token_batch_stride: tl.constexpr,
        token_length_stride: tl.constexpr,
        token_lane_stride: tl.constexpr,
        token_component_stride: tl.constexpr,
        initial_batch_stride: tl.constexpr,
        initial_lane_stride: tl.constexpr,
        initial_component_stride: tl.constexpr,
        output_batch_stride: tl.constexpr,
        output_length_stride: tl.constexpr,
        output_lane_stride: tl.constexpr,
        output_component_stride: tl.constexpr,
    ):
        program = tl.program_id(0)
        batch_index = program // lanes
        lane_index = program - batch_index * lanes
        carry0 = 0.0
        carry1 = 0.0
        carry2 = 0.0
        carry3 = 0.0
        carry4 = 0.0
        carry5 = 0.0
        carry6 = 0.0
        carry7 = 0.0
        for reverse_position in tl.range(0, length):
            position = length - 1 - reverse_position
            output_base = (
                batch_index * output_batch_stride
                + position * output_length_stride
                + lane_index * output_lane_stride
            )
            direct0, direct1, direct2, direct3, direct4, direct5, direct6, direct7 = (
                _load_octonion(output_gradients, output_base, output_component_stride)
            )
            total0 = direct0 + carry0
            total1 = direct1 + carry1
            total2 = direct2 + carry2
            total3 = direct3 + carry3
            total4 = direct4 + carry4
            total5 = direct5 + carry5
            total6 = direct6 + carry6
            total7 = direct7 + carry7
            if position == 0:
                previous_base = (
                    batch_index * initial_batch_stride
                    + lane_index * initial_lane_stride
                )
                p0, p1, p2, p3, p4, p5, p6, p7 = _load_octonion(
                    initial, previous_base, initial_component_stride
                )
            else:
                previous_base = (
                    batch_index * output_batch_stride
                    + (position - 1) * output_length_stride
                    + lane_index * output_lane_stride
                )
                p0, p1, p2, p3, p4, p5, p6, p7 = _load_octonion(
                    outputs, previous_base, output_component_stride
                )

            # R_previous^T total = total * conjugate(previous).
            g0, g1, g2, g3, g4, g5, g6, g7 = _octonion_product(
                total0,
                total1,
                total2,
                total3,
                total4,
                total5,
                total6,
                total7,
                p0,
                -p1,
                -p2,
                -p3,
                -p4,
                -p5,
                -p6,
                -p7,
            )
            token_base = (
                batch_index * token_batch_stride
                + position * token_length_stride
                + lane_index * token_lane_stride
            )
            _store_octonion(
                token_gradients,
                token_base,
                token_component_stride,
                g0,
                g1,
                g2,
                g3,
                g4,
                g5,
                g6,
                g7,
            )

            u0, u1, u2, u3, u4, u5, u6, u7 = _load_octonion(
                tokens, token_base, token_component_stride
            )
            # L_u^T total = conjugate(u) * total for unit u.
            carry0, carry1, carry2, carry3, carry4, carry5, carry6, carry7 = (
                _octonion_product(
                    u0,
                    -u1,
                    -u2,
                    -u3,
                    -u4,
                    -u5,
                    -u6,
                    -u7,
                    total0,
                    total1,
                    total2,
                    total3,
                    total4,
                    total5,
                    total6,
                    total7,
                )
            )
        initial_base = (
            batch_index * initial_batch_stride + lane_index * initial_lane_stride
        )
        _store_octonion(
            initial_gradients,
            initial_base,
            initial_component_stride,
            carry0,
            carry1,
            carry2,
            carry3,
            carry4,
            carry5,
            carry6,
            carry7,
        )

    class _FusedOctonionRecurrence(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: torch.autograd.function.FunctionCtx,
            normalized_tokens: torch.Tensor,
            initial_state: torch.Tensor,
        ) -> torch.Tensor:
            batch, length, lanes, _ = normalized_tokens.shape
            outputs = torch.empty_like(normalized_tokens)
            _octonion_recurrence_forward_kernel[(batch * lanes,)](
                normalized_tokens,
                initial_state,
                outputs,
                length,
                lanes,
                *normalized_tokens.stride(),
                *initial_state.stride(),
                *outputs.stride(),
                num_warps=1,
            )
            ctx.save_for_backward(normalized_tokens, initial_state, outputs)
            return outputs

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx,
            output_gradients: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            normalized_tokens, initial_state, outputs = ctx.saved_tensors
            batch, length, lanes, _ = normalized_tokens.shape
            output_gradients = output_gradients.contiguous()
            token_gradients = torch.empty_like(normalized_tokens)
            initial_gradients = torch.empty_like(initial_state)
            _octonion_recurrence_backward_kernel[(batch * lanes,)](
                normalized_tokens,
                initial_state,
                outputs,
                output_gradients,
                token_gradients,
                initial_gradients,
                length,
                lanes,
                *normalized_tokens.stride(),
                *initial_state.stride(),
                *outputs.stride(),
                num_warps=1,
            )
            return token_gradients, initial_gradients


def triton_is_available() -> bool:
    return triton is not None and torch.cuda.is_available()


def fused_octonion_state_scan(
    token_octonions: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a fused left-octonion recurrence on one CUDA sequence chunk.

    Inputs have shape ``(batch,length,lanes,8)``.  Tokens are normalized by the
    ordinary PyTorch primitive before entering the custom autograd function, so
    its backward composes with the radial-normalization gradient.
    """

    if not triton_is_available():
        raise RuntimeError("the fused octonion scan requires Triton and CUDA")
    if (
        token_octonions.ndim != 4
        or token_octonions.shape[1] < 1
        or token_octonions.shape[-1] != OCTONION_DIM
        or token_octonions.device.type != "cuda"
        or token_octonions.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError(
            "token_octonions must be CUDA float32/float64 with nonempty shape "
            "(batch,length,lanes,8)"
        )
    batch, _, lanes, _ = token_octonions.shape
    if initial_state is None:
        initial_state = torch.zeros(
            batch,
            lanes,
            OCTONION_DIM,
            dtype=token_octonions.dtype,
            device=token_octonions.device,
        )
        initial_state[..., 0] = 1
    elif initial_state.shape != (batch, lanes, OCTONION_DIM):
        raise ValueError("initial_state must have shape (batch,lanes,8)")
    if initial_state.device != token_octonions.device:
        raise ValueError("tokens and initial state must share a CUDA device")
    normalized = unit_octonion(token_octonions).contiguous()
    initial_state = initial_state.contiguous()
    sequence = _FusedOctonionRecurrence.apply(normalized, initial_state)
    return sequence, sequence[:, -1]


__all__: Sequence[str] = (
    "fused_octonion_state_scan",
    "triton_is_available",
)
