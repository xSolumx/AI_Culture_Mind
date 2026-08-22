"""Paired Tiny Shakespeare gate for direction and triality-invariant readouts."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, replace
from pathlib import Path

import torch
from benchmark import (
    BenchmarkConfig,
    build_model,
    file_sha256,
    package_version,
    parameter_count,
    run_one,
)
from data import (
    TINY_SHAKESPEARE_REVISION,
    TINY_SHAKESPEARE_SHA256,
    TINY_SHAKESPEARE_URL,
    random_batch,
    tiny_shakespeare_bytes,
)
from mamba2_baseline import fused_mamba2_available

ROOT = Path(__file__).resolve().parent
READOUTS = ("direction", "triality_invariants")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    base_config = BenchmarkConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        validation_batches=args.validation_batches,
        spin_backend="raw_cuda_hybrid",
        spin_group_schedule=(3, 4, 6, 8),
        seed=args.seed,
    )
    train, train_sha = tiny_shakespeare_bytes("train", offline=args.offline)
    valid, valid_sha = tiny_shakespeare_bytes("validation", offline=args.offline)
    train_stream = torch.as_tensor(train, dtype=torch.long)
    valid_stream = torch.as_tensor(valid, dtype=torch.long)
    validation_generator = torch.Generator().manual_seed(args.seed + 1)
    validation = [
        random_batch(
            valid_stream,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            generator=validation_generator,
        )
        for _ in range(args.validation_batches)
    ]
    device = torch.device("cuda")
    results = []
    counts = {}
    for readout in READOUTS:
        config = replace(base_config, spin_readout=readout)
        counts[readout] = parameter_count(build_model("pure_spin_v1_2", config))
        result = run_one("pure_spin_v1_2", config, train_stream, validation, device)
        result["readout"] = readout
        results.append(result)
    mamba_count = parameter_count(build_model("mamba2_fused", base_config))

    implementation_paths = (
        Path(__file__),
        ROOT / "benchmark.py",
        ROOT / "model.py",
        ROOT / "data.py",
        ROOT / "raw_cuda.py",
        ROOT / "csrc" / "spin_scan.cpp",
        ROOT / "csrc" / "spin_scan_cuda.cu",
    )
    report = {
        "schema_version": 1,
        "claim_scope": (
            "paired internal v1.2 readout-quality gate; sequential timers are "
            "not an order-balanced speed claim"
        ),
        "config": asdict(base_config),
        "readouts": list(READOUTS),
        "parameter_counts": {
            **counts,
            "mamba2_fused_reference": mamba_count,
        },
        "dataset": {
            "name": "tiny_shakespeare",
            "encoding": "raw UTF-8 bytes",
            "split": "90/5/5 contiguous bytes",
            "source": {
                "url": TINY_SHAKESPEARE_URL,
                "revision": TINY_SHAKESPEARE_REVISION,
                "full_sha256": TINY_SHAKESPEARE_SHA256,
            },
            "train_sha256": train_sha,
            "validation_sha256": valid_sha,
        },
        "environment": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "triton": package_version("triton", "triton-windows"),
            "mamba2_fused": list(fused_mamba2_available()),
        },
        "results": results,
        "implementation_sha256": {
            path.relative_to(ROOT).as_posix(): file_sha256(path)
            for path in implementation_paths
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
