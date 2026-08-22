"""Fail-closed audit of the pinned Pure Spin v1.2 WSL CUDA environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path

import torch
from data import tiny_shakespeare_bytes
from mamba2_baseline import fused_mamba2_available
from raw_cuda import _extension_name, extension
from torch.utils.cpp_extension import is_ninja_available

EXPECTED_DISTRIBUTIONS = {
    "causal-conv1d": "1.7.0",
    "datasets": "5.0.1",
    "einops": "0.8.2",
    "huggingface-hub": "1.28.0",
    "mamba-ssm": "2.3.2.post1",
    "ninja": "1.13.0",
    "numpy": "2.2.6",
    "packaging": "26.3",
    "pytest": "9.1.1",
    "ruff": "0.16.1",
    "setuptools": "84.0.0",
    "torch": "2.10.0",
    "transformers": "5.15.1",
    "triton": "3.6.0",
}
EXPECTED_EXTENSION_SYMBOLS = {
    "forward",
    "controller_forward",
    "controller_backward",
    "coordinate_forward",
    "coordinate_backward",
    "isotypic_coordinate_forward",
    "isotypic_coordinate_backward",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(*, prepare_data: bool) -> dict[str, object]:
    if platform.system() != "Linux" or "microsoft" not in platform.release().lower():
        raise RuntimeError("Pure Spin v1.2 CUDA validation must run inside WSL2")
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(f"expected Python 3.10, got {platform.python_version()}")

    versions = {
        distribution: importlib.metadata.version(distribution)
        for distribution in EXPECTED_DISTRIBUTIONS
    }
    mismatches = {
        distribution: {"expected": expected, "actual": versions[distribution]}
        for distribution, expected in EXPECTED_DISTRIBUTIONS.items()
        if not versions[distribution].startswith(expected)
    }
    if mismatches:
        raise RuntimeError(f"distribution version mismatch: {mismatches}")
    if torch.__version__ != "2.10.0+cu126" or torch.version.cuda != "12.6":
        raise RuntimeError(
            f"expected Torch 2.10.0+cu126/CUDA 12.6, got "
            f"{torch.__version__}/{torch.version.cuda}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside WSL")
    if not is_ninja_available() or shutil.which("ninja") is None:
        raise RuntimeError("Ninja is not executable inside the active WSL environment")

    cuda_home = Path(os.environ.get("CUDA_HOME", ""))
    if not cuda_home.is_dir() or not (cuda_home / "bin" / "nvcc").is_file():
        raise RuntimeError(f"invalid CUDA_HOME: {cuda_home}")
    required_cuda_files = (
        cuda_home / "include" / "cuda_runtime.h",
        cuda_home / "lib64" / "libcudart.so",
        cuda_home / "lib64" / "libcudart_static.a",
        cuda_home / "lib64" / "libcudadevrt.a",
    )
    missing_cuda_files = [
        str(path) for path in required_cuda_files if not path.exists()
    ]
    if missing_cuda_files:
        raise RuntimeError(f"incomplete CUDA toolkit layout: {missing_cuda_files}")
    extensions_dir = Path(os.environ.get("TORCH_EXTENSIONS_DIR", ""))
    if not extensions_dir.is_dir():
        raise RuntimeError(f"invalid TORCH_EXTENSIONS_DIR: {extensions_dir}")
    large_root = Path(os.environ.get("PURE_SPIN_V12_LARGE_ROOT", ""))
    if not large_root.is_dir() or not str(large_root).startswith("/mnt/e/"):
        raise RuntimeError(f"large-artifact root must live on E:; got {large_root}")

    e_backed_paths = {
        name: Path(os.environ.get(name, ""))
        for name in (
            "HF_DATASETS_CACHE",
            "HF_HOME",
            "HF_HUB_CACHE",
            "PIP_CACHE_DIR",
            "PURE_SPIN_V12_DATA_CACHE",
        )
    }
    misplaced = {
        name: str(path)
        for name, path in e_backed_paths.items()
        if not path.is_dir() or not str(path).startswith(f"{large_root}/")
    }
    if misplaced:
        raise RuntimeError(f"download caches are not rooted on E:: {misplaced}")

    local_cache_root = Path(os.environ.get("PURE_SPIN_V12_LOCAL_CACHE_ROOT", ""))
    local_cache_paths = {
        name: Path(os.environ.get(name, ""))
        for name in (
            "CUDA_CACHE_PATH",
            "TORCH_EXTENSIONS_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
        )
    }
    misplaced_local = {
        name: str(path)
        for name, path in local_cache_paths.items()
        if not path.is_dir()
        or not str(path).startswith(f"{local_cache_root}/")
        or str(path).startswith("/mnt/")
    }
    if misplaced_local:
        raise RuntimeError(f"compiler caches are not on WSL ext4: {misplaced_local}")

    fused_available, fused_detail = fused_mamba2_available()
    if not fused_available:
        raise RuntimeError(f"official fused Mamba-2 unavailable: {fused_detail}")

    module = extension()
    exported = {name for name in EXPECTED_EXTENSION_SYMBOLS if hasattr(module, name)}
    if exported != EXPECTED_EXTENSION_SYMBOLS:
        missing = sorted(EXPECTED_EXTENSION_SYMBOLS - exported)
        raise RuntimeError(f"raw CUDA extension is missing symbols: {missing}")

    dataset = None
    if prepare_data:
        train, train_hash = tiny_shakespeare_bytes("train", offline=False)
        validation, validation_hash = tiny_shakespeare_bytes(
            "validation", offline=True
        )
        dataset = {
            "train_bytes": train.numel(),
            "validation_bytes": validation.numel(),
            "train_sha256": train_hash,
            "validation_sha256": validation_hash,
        }

    root = Path(__file__).resolve().parent
    return {
        "schema_version": 1,
        "status": "passed",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "distributions": versions,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_home": str(cuda_home),
        "ninja": shutil.which("ninja"),
        "torch_extensions_dir": str(extensions_dir),
        "large_artifact_root": str(large_root),
        "e_backed_paths": {name: str(path) for name, path in e_backed_paths.items()},
        "local_cache_root": str(local_cache_root),
        "local_cache_paths": {
            name: str(path) for name, path in local_cache_paths.items()
        },
        "raw_extension_name": _extension_name(),
        "raw_extension_symbols": sorted(exported),
        "mamba2_fused": {"available": fused_available, "version": fused_detail},
        "dataset": dataset,
        "source_sha256": {
            name: _sha256(root / name)
            for name in (
                "data.py",
                "raw_cuda.py",
                "verify_wsl_environment.py",
                "wsl_env.sh",
            )
        },
        "dependency_boundary": (
            "The pinned no-deps Mamba-2 wheel intentionally omits TileLang, "
            "Quack, and TVM-FFI packages required by current combined-package "
            "metadata but unused by this frozen Mamba-2 route; the exercised "
            "fused SSD symbol is imported and tested directly."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(prepare_data=args.prepare_data)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(payload, end="")


if __name__ == "__main__":
    main()
