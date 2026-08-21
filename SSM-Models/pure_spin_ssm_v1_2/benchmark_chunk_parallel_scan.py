"""Benchmark the continuous affine chunk compiler against full-tree and raw scans."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from chunk_parallel_scan import (
    chunk_parallel_spin8_scan,
    factorized_triality_actions,
)
from pure_spin8_ssm.torch_backend import (
    Spin8AffineTransition,
    apply_spin8_affine,
    work_efficient_spin8_scan,
)
from raw_cuda import raw_cuda_coordinate_factorized_scan
from spin8_triality import torch_triality_generators


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timed(function, *, warmups: int, repetitions: int) -> tuple[object, float]:
    for _ in range(warmups):
        result = function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)
    return result, statistics.median(samples)


def composition_count(length: int, chunk_size: int) -> dict[str, int]:
    chunks = (length + chunk_size - 1) // chunk_size
    padded_length = chunks * chunk_size
    padded_chunks = 1 << (chunks - 1).bit_length()
    return {
        "chunks": chunks,
        "padded_length": padded_length,
        "full_tree_matrix_compositions": 0 if length == 1 else 3 * (1 << (length - 1).bit_length()) - 2,
        "chunk_local_matrix_compositions": chunks * (chunk_size - 1),
        "chunk_endpoint_tree_matrix_compositions": 0 if chunks == 1 else 3 * padded_chunks - 2,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--factors", type=int, default=28)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/chunk_parallel_scan_rtx2070s.json"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20_260_821)
    coordinates = (
        0.02
        * torch.randn(
            args.batch,
            args.length,
            args.channels,
            args.factors,
            device="cuda",
        )
    ).requires_grad_()
    generators = torch_triality_generators(device="cuda")[:, : args.factors].contiguous()
    scale = (
        0.8
        + 0.1
        * torch.rand(args.batch, args.length, args.channels, device="cuda")
    ).requires_grad_()
    drive = (
        0.01
        * torch.randn(
            args.batch, args.length, args.channels, 3, 8, device="cuda"
        )
    ).requires_grad_()
    initial = torch.randn(
        args.batch, args.channels, 3, 8, device="cuda", requires_grad=True
    )
    differentiable = (coordinates, scale, drive, initial)
    output_gradient = torch.randn_like(drive)

    def transition() -> Spin8AffineTransition:
        return Spin8AffineTransition(
            scale=scale,
            action=factorized_triality_actions(coordinates, generators),
            drive=drive,
        )

    def full_tree() -> torch.Tensor:
        prefixes = work_efficient_spin8_scan(transition())
        return apply_spin8_affine(prefixes, initial[:, None])

    def chunked(chunk_size: int) -> torch.Tensor:
        return chunk_parallel_spin8_scan(
            transition(), initial, chunk_size=chunk_size
        )[0]

    def raw() -> torch.Tensor:
        return raw_cuda_coordinate_factorized_scan(
            coordinates, generators, scale, drive, initial
        )

    def training(function):
        output = function()
        gradients = torch.autograd.grad(
            output, differentiable, output_gradient, create_graph=False
        )
        return output, gradients

    with torch.no_grad():
        raw_forward, raw_forward_us = timed(
            raw, warmups=args.warmups, repetitions=args.repetitions
        )
        full_forward, full_forward_us = timed(
            full_tree, warmups=args.warmups, repetitions=args.repetitions
        )
    raw_training, raw_training_us = timed(
        lambda: training(raw),
        warmups=args.warmups,
        repetitions=args.repetitions,
    )
    full_training, full_training_us = timed(
        lambda: training(full_tree),
        warmups=args.warmups,
        repetitions=args.repetitions,
    )
    torch.testing.assert_close(full_forward, raw_forward, rtol=2e-4, atol=2e-4)
    rows = []
    for chunk_size in args.chunk_sizes:
        with torch.no_grad():
            chunk_forward, chunk_forward_us = timed(
                lambda chunk_size=chunk_size: chunked(chunk_size),
                warmups=args.warmups,
                repetitions=args.repetitions,
            )
        chunk_training, chunk_training_us = timed(
            lambda chunk_size=chunk_size: training(
                lambda: chunked(chunk_size)
            ),
            warmups=args.warmups,
            repetitions=args.repetitions,
        )
        torch.testing.assert_close(
            chunk_forward, raw_forward, rtol=2e-4, atol=2e-4
        )
        torch.testing.assert_close(
            chunk_training[0], raw_training[0], rtol=2e-4, atol=2e-4
        )
        for actual, expected in zip(
            chunk_training[1], full_training[1], strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=2e-3, atol=3e-3)
        counts = composition_count(args.length, chunk_size)
        chunk_compositions = (
            counts["chunk_local_matrix_compositions"]
            + counts["chunk_endpoint_tree_matrix_compositions"]
        )
        rows.append(
            {
                "chunk_size": chunk_size,
                "composition_count": counts,
                "chunk_over_full_composition_work": (
                    chunk_compositions / counts["full_tree_matrix_compositions"]
                ),
                "median_forward_microseconds": chunk_forward_us,
                "median_forward_backward_microseconds": chunk_training_us,
                "speedup_over_full_tree": {
                    "forward": full_forward_us / chunk_forward_us,
                    "forward_backward": full_training_us / chunk_training_us,
                },
                "slowdown_over_raw": {
                    "forward": chunk_forward_us / raw_forward_us,
                    "forward_backward": chunk_training_us / raw_training_us,
                },
            }
        )
    implementation_paths = (
        Path(__file__),
        ROOT / "chunk_parallel_scan.py",
        ROOT / "raw_cuda.py",
        ROOT / "csrc" / "spin_scan.cpp",
        ROOT / "csrc" / "spin_scan_cuda.cu",
    )
    report = {
        "schema_version": 1,
        "claim_scope": (
            "continuous learned Spin affine chunk compiler semantic and local "
            "CUDA performance gate; not an end-to-end model claim"
        ),
        "shape": {
            "batch": args.batch,
            "length": args.length,
            "channels": args.channels,
            "factors": args.factors,
            "representations": 3,
            "state_dimension": 8,
        },
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "timing": {
            "method": "CUDA events on current stream",
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "baselines": {
                "raw_serial": {
                    "forward_microseconds": raw_forward_us,
                    "forward_backward_microseconds": raw_training_us,
                },
                "materialized_full_tree": {
                    "forward_microseconds": full_forward_us,
                    "forward_backward_microseconds": full_training_us,
                },
            },
            "chunk_rows": rows,
        },
        "implementation_sha256": {
            path.relative_to(ROOT).as_posix(): file_sha256(path)
            for path in implementation_paths
        },
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
