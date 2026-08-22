"""JIT loader for the raw CUDA materialized-action recurrence."""

import hashlib
import os
import platform
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent


def _cuda_dependency_include_paths() -> list[str]:
    """Expose CUDA math headers bundled beside the active PyTorch wheel.

    CUDA 12's minimal nvcc package contains the compiler and runtime headers,
    while PyTorch owns the matching cuBLAS/cuSPARSE/cuSOLVER/cuDNN packages.
    Reusing those headers avoids a second, nearly gigabyte-sized system SDK.
    """

    nvidia_root = Path(torch.__file__).resolve().parent.parent / "nvidia"
    candidates = (
        nvidia_root / "cublas" / "include",
        nvidia_root / "cusparse" / "include",
        nvidia_root / "cusolver" / "include",
        nvidia_root / "cudnn" / "include",
        nvidia_root / "cuda_runtime" / "include",
    )
    return [str(path) for path in candidates if path.is_dir()]


def _extension_name() -> str:
    """Use a source/ABI-qualified name so stale binary modules cannot survive."""

    sources = (ROOT / "csrc" / "spin_scan.cpp", ROOT / "csrc" / "spin_scan_cuda.cu")
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.read_bytes())
    digest.update(torch.__version__.encode())
    digest.update(str(torch.version.cuda).encode())
    digest.update(platform.python_version().encode())
    digest.update(os.name.encode())
    return f"pure_spin_v12_raw_cuda_{digest.hexdigest()[:16]}"


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
        name=_extension_name(),
        sources=[
            str(ROOT / "csrc" / "spin_scan.cpp"),
            str(ROOT / "csrc" / "spin_scan_cuda.cu"),
        ],
        extra_include_paths=_cuda_dependency_include_paths(),
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


class _RawCudaIsotypicCoordinateFactorized(torch.autograd.Function):
    """Schedule each inequivalent triality representation as its own warp."""

    @staticmethod
    def forward(ctx, coordinates, generators, scale, drive, initial):
        tensors = (coordinates, generators, scale, drive, initial)
        if any(tensor.device.type != "cuda" for tensor in tensors):
            raise ValueError("raw isotypic backend requires CUDA tensors")
        if any(tensor.dtype != torch.float32 for tensor in tensors):
            raise ValueError("raw isotypic backend requires float32 tensors")
        contiguous = tuple(tensor.contiguous() for tensor in tensors)
        output = extension().isotypic_coordinate_forward(*contiguous)
        ctx.save_for_backward(*contiguous, output)
        return output

    @staticmethod
    def backward(ctx, output_gradient):
        coordinates, generators, scale, drive, initial, output = ctx.saved_tensors
        gradients = extension().isotypic_coordinate_backward(
            coordinates,
            generators,
            scale,
            drive,
            initial,
            output,
            output_gradient.contiguous(),
        )
        return gradients[0], None, *gradients[1:]


class _RawCudaHybridCoordinateFactorized(torch.autograd.Function):
    """Use isotypic forward occupancy and packed shared-gradient backward."""

    @staticmethod
    def forward(ctx, coordinates, generators, scale, drive, initial):
        tensors = (coordinates, generators, scale, drive, initial)
        if any(tensor.device.type != "cuda" for tensor in tensors):
            raise ValueError("raw hybrid backend requires CUDA tensors")
        if any(tensor.dtype != torch.float32 for tensor in tensors):
            raise ValueError("raw hybrid backend requires float32 tensors")
        contiguous = tuple(tensor.contiguous() for tensor in tensors)
        output = extension().isotypic_coordinate_forward(*contiguous)
        ctx.save_for_backward(*contiguous, output)
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


class _RawCudaCoupledCoordinateFactorized(torch.autograd.Function):
    """Shared Spin action plus a learned 2x2 multiplicity transition."""

    @staticmethod
    def forward(ctx, coordinates, generators, left, drive, initial):
        tensors = (coordinates, generators, left, drive, initial)
        if any(tensor.device.type != "cuda" for tensor in tensors):
            raise ValueError("raw coupled backend requires CUDA tensors")
        if any(tensor.dtype != torch.float32 for tensor in tensors):
            raise ValueError("raw coupled backend requires float32 tensors")
        contiguous = tuple(tensor.contiguous() for tensor in tensors)
        output = extension().coupled_coordinate_forward(*contiguous)
        ctx.save_for_backward(*contiguous, output)
        return output

    @staticmethod
    def backward(ctx, output_gradient):
        coordinates, generators, left, drive, initial, output = ctx.saved_tensors
        gradients = extension().coupled_coordinate_backward(
            coordinates,
            generators,
            left,
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


def raw_cuda_isotypic_coordinate_scan(
    coordinates, generators, scale, drive, initial
):
    """Coordinate recurrence with one CUDA warp per real-Schur triality block."""
    return _RawCudaIsotypicCoordinateFactorized.apply(
        coordinates, generators, scale, drive, initial
    )


def raw_cuda_hybrid_coordinate_scan(
    coordinates, generators, scale, drive, initial
):
    """Isotypic-split forward followed by packed-warp reverse mode."""
    return _RawCudaHybridCoordinateFactorized.apply(
        coordinates, generators, scale, drive, initial
    )


def raw_cuda_coupled_coordinate_scan(
    coordinates, generators, left, drive, initial
):
    """Full-training CUDA lowering of ``H <- L H R^T + D``.

    The current kernel deliberately fixes multiplicity to two copies, matching
    the v1.2 architecture, while retaining all three inequivalent triality
    representations and one shared ordered Spin factorization.
    """
    return _RawCudaCoupledCoordinateFactorized.apply(
        coordinates, generators, left, drive, initial
    )
