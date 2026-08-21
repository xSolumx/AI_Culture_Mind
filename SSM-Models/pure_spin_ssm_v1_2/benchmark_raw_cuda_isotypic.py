"""Benchmark packed versus real-isotypic raw CUDA Spin recurrence schedules."""

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

from raw_cuda import (
    raw_cuda_coordinate_factorized_scan,
    raw_cuda_hybrid_coordinate_scan,
    raw_cuda_isotypic_coordinate_scan,
)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--factor-counts", type=int, nargs="+", default=[3, 6, 15, 28])
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/raw_cuda_isotypic_rtx2070s.json"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20_260_821)
    generators = torch_triality_generators(device="cuda")
    rows = []
    for factors in args.factor_counts:
        coordinates = (
            0.02
            * torch.randn(
                args.batch,
                args.length,
                args.channels,
                factors,
                device="cuda",
            )
        ).requires_grad_()
        selected_generators = generators[:, :factors].contiguous()
        scale = (
            0.8
            + 0.1
            * torch.rand(
                args.batch, args.length, args.channels, device="cuda"
            )
        ).requires_grad_()
        drive = (
            0.01
            * torch.randn(
                args.batch,
                args.length,
                args.channels,
                3,
                8,
                device="cuda",
            )
        ).requires_grad_()
        initial = torch.randn(
            args.batch,
            args.channels,
            3,
            8,
            device="cuda",
            requires_grad=True,
        )
        inputs = (coordinates, selected_generators, scale, drive, initial)
        differentiable = (coordinates, scale, drive, initial)
        output_gradient = torch.randn_like(drive)

        def forward(backend, scan_inputs=inputs):
            return backend(*scan_inputs)

        def training(
            backend,
            scan_inputs=inputs,
            differentiable_inputs=differentiable,
            gradient=output_gradient,
        ):
            output = backend(*scan_inputs)
            gradients = torch.autograd.grad(
                output, differentiable_inputs, gradient, create_graph=False
            )
            return output, gradients

        packed_forward, packed_forward_us = timed(
            lambda: forward(raw_cuda_coordinate_factorized_scan),
            warmups=10,
            repetitions=args.repetitions,
        )
        split_forward, split_forward_us = timed(
            lambda: forward(raw_cuda_isotypic_coordinate_scan),
            warmups=10,
            repetitions=args.repetitions,
        )
        packed_training, packed_training_us = timed(
            lambda: training(raw_cuda_coordinate_factorized_scan),
            warmups=10,
            repetitions=args.repetitions,
        )
        split_training, split_training_us = timed(
            lambda: training(raw_cuda_isotypic_coordinate_scan),
            warmups=10,
            repetitions=args.repetitions,
        )
        hybrid_forward, hybrid_forward_us = timed(
            lambda: forward(raw_cuda_hybrid_coordinate_scan),
            warmups=10,
            repetitions=args.repetitions,
        )
        hybrid_training, hybrid_training_us = timed(
            lambda: training(raw_cuda_hybrid_coordinate_scan),
            warmups=10,
            repetitions=args.repetitions,
        )
        torch.testing.assert_close(
            split_forward, packed_forward, rtol=4e-5, atol=4e-5
        )
        torch.testing.assert_close(
            split_training[0], packed_training[0], rtol=4e-5, atol=4e-5
        )
        torch.testing.assert_close(
            hybrid_forward, packed_forward, rtol=4e-5, atol=4e-5
        )
        torch.testing.assert_close(
            hybrid_training[0], packed_training[0], rtol=4e-5, atol=4e-5
        )
        for actual, expected in zip(
            split_training[1], packed_training[1], strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=9e-4, atol=2e-3)
        for actual, expected in zip(
            hybrid_training[1], packed_training[1], strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=9e-4, atol=2e-3)
        rows.append(
            {
                "factors": factors,
                "median_forward_microseconds": {
                    "packed": packed_forward_us,
                    "isotypic": split_forward_us,
                    "hybrid": hybrid_forward_us,
                },
                "median_forward_backward_microseconds": {
                    "packed": packed_training_us,
                    "isotypic": split_training_us,
                    "hybrid": hybrid_training_us,
                },
                "isotypic_over_packed": {
                    "forward": split_forward_us / packed_forward_us,
                    "forward_backward": split_training_us / packed_training_us,
                },
                "hybrid_over_packed": {
                    "forward": hybrid_forward_us / packed_forward_us,
                    "forward_backward": hybrid_training_us / packed_training_us,
                },
            }
        )
    implementation_paths = (
        Path(__file__),
        ROOT / "raw_cuda.py",
        ROOT / "csrc" / "spin_scan.cpp",
        ROOT / "csrc" / "spin_scan_cuda.cu",
    )
    report = {
        "schema_version": 1,
        "claim_scope": (
            "packed versus isotypic-split coordinate recurrence at the exact "
            "four-block Spin ladder shapes"
        ),
        "shape": {
            "batch": args.batch,
            "length": args.length,
            "channels": args.channels,
            "representations": 3,
            "state_dimension": 8,
        },
        "factor_counts": args.factor_counts,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "timing": {
            "method": "CUDA events on current stream",
            "warmups": 10,
            "repetitions": args.repetitions,
            "rows": rows,
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
