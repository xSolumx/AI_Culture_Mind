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
