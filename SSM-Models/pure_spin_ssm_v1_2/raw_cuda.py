"""JIT loader for the raw CUDA materialized-action recurrence."""

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def extension():
    if not torch.cuda.is_available():
        raise RuntimeError("raw CUDA Spin scan requires CUDA")
    cuda_flags = ["-O3", "--use_fast_math", "-lineinfo"]
    cxx_flags = ["-O3"]
    if os.name == "nt":
        cuda_flags.append("-Xcompiler=/Zc:preprocessor")
        cxx_flags = ["/O2", "/Zc:preprocessor"]
    return load(
        name="pure_spin_v12_raw_cuda",
        sources=[
            str(ROOT / "csrc" / "spin_scan.cpp"),
            str(ROOT / "csrc" / "spin_scan_cuda.cu"),
        ],
        extra_cuda_cflags=cuda_flags,
        extra_cflags=cxx_flags,
        verbose=False,
    )


def raw_cuda_spin8_scan(action, scale, drive, initial):
    """Forward-only shared-action scan; gradients are intentionally refused."""
    if torch.is_grad_enabled() and any(t.requires_grad for t in (action, scale, drive, initial)):
        raise RuntimeError("raw CUDA v1.2 comparison kernel is inference-only")
    return extension().forward(
        action.contiguous(),
        scale.contiguous(),
        drive.contiguous(),
        initial.contiguous(),
    )


class _RawCudaControllerFactorized(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        features,
        weight,
        bias,
        generators,
        scale,
        drive,
        initial,
        gate,
    ):
        tensors = (features, weight, bias, generators, scale, drive, initial, gate)
        if any(tensor.device.type != "cuda" for tensor in tensors):
            raise ValueError("raw controller backend requires CUDA tensors")
        if any(tensor.dtype != torch.float32 for tensor in tensors):
            raise ValueError("raw controller backend requires float32 tensors")
        contiguous = tuple(tensor.contiguous() for tensor in tensors)
        output = extension().controller_forward(*contiguous)
        ctx.save_for_backward(
            contiguous[0],
            contiguous[1],
            contiguous[2],
            contiguous[3],
            contiguous[4],
            contiguous[5],
            contiguous[6],
            contiguous[7],
            output,
        )
        return output

    @staticmethod
    def backward(ctx, output_gradient):
        features, weight, bias, generators, scale, drive, initial, gate, output = (
            ctx.saved_tensors
        )
        gradients = extension().controller_backward(
            features,
            weight,
            bias,
            generators,
            scale,
            drive,
            initial,
            gate,
            output,
            output_gradient.contiguous(),
        )
        return (*gradients[:3], None, *gradients[3:], None)


class _RawCudaCoordinateFactorized(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coordinates, generators, scale, drive, initial):
        tensors = (coordinates, generators, scale, drive, initial)
        if any(tensor.device.type != "cuda" for tensor in tensors):
            raise ValueError("raw coordinate backend requires CUDA tensors")
        if any(tensor.dtype != torch.float32 for tensor in tensors):
            raise ValueError("raw coordinate backend requires float32 tensors")
        contiguous = tuple(tensor.contiguous() for tensor in tensors)
        output = extension().coordinate_forward(*contiguous)
        ctx.save_for_backward(
            contiguous[0],
            contiguous[1],
            contiguous[2],
            contiguous[3],
            contiguous[4],
            output,
        )
        return output

    @staticmethod
    def backward(ctx, output_gradient):
        coordinates, generators, scale, drive, initial, output = ctx.saved_tensors
        gradients = extension().coordinate_backward(
            coordinates,
            generators,
            scale,
            drive,
            initial,
            output,
            output_gradient.contiguous(),
        )
        return gradients[0], None, *gradients[1:]


def raw_cuda_controller_factorized_scan(
    features,
    weight,
    bias,
    generators,
    scale,
    drive,
    initial,
    gate,
):
    """Raw-CUDA fused controller, 28 factors, and full backward.

    Backward normally reconstructs each pre-affine rotated state as
    ``(output-drive) / scale``. Zero or tiny scales take an exact factor-replay
    fallback so the public backend remains well-defined for arbitrary scales.
    """
    return _RawCudaControllerFactorized.apply(
        features,
        weight,
        bias,
        generators,
        scale,
        drive,
        initial,
        gate,
    )


def raw_cuda_coordinate_factorized_scan(
    coordinates, generators, scale, drive, initial
):
    """Raw-CUDA factor recurrence with coordinate-level backward.

    Uses the same guarded reconstruction as the controller backend.
    """
    return _RawCudaCoordinateFactorized.apply(
        coordinates, generators, scale, drive, initial
    )
