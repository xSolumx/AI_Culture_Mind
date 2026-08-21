"""Benchmark raw CUDA versus Triton fused Spin controller training."""

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

from pure_spin8_ssm.factorized_scan import triton_controller_factorized_scan
from raw_cuda import raw_cuda_controller_factorized_scan
from spin8_triality import torch_triality_generators


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timed(function, *, warmups: int, repetitions: int):
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
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/raw_cuda_training_rtx2070s.json"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20_260_821)
    shape = (args.batch, args.length, args.channels, args.input_size)
    features = torch.randn(
        args.batch,
        args.length,
        args.input_size,
        device="cuda",
        requires_grad=True,
    )
    weight = (
        0.02
        * torch.randn(args.channels * 28, args.input_size, device="cuda")
    ).requires_grad_()
    bias = torch.zeros(args.channels * 28, device="cuda", requires_grad=True)
    generators = torch_triality_generators(device="cuda")
    scale = (
        0.8
        + 0.1 * torch.rand(args.batch, args.length, args.channels, device="cuda")
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
    gate = torch.ones(args.batch, args.length, device="cuda")
    output_gradient = torch.randn_like(drive)
    differentiable = (features, weight, bias, scale, drive, initial)

    def run(backend):
        output = backend(
            features, weight, bias, generators, scale, drive, initial, gate
        )
        gradients = torch.autograd.grad(
            output, differentiable, output_gradient, create_graph=False
        )
        return output, gradients

    triton_result, triton_us = timed(
        lambda: run(triton_controller_factorized_scan),
        warmups=10,
        repetitions=args.repetitions,
    )
    raw_result, raw_us = timed(
        lambda: run(raw_cuda_controller_factorized_scan),
        warmups=10,
        repetitions=args.repetitions,
    )
    torch.testing.assert_close(raw_result[0], triton_result[0], rtol=4e-5, atol=4e-5)
    for actual, expected in zip(raw_result[1], triton_result[1], strict=True):
        torch.testing.assert_close(actual, expected, rtol=9e-4, atol=2e-3)
    report = {
        "schema_version": 1,
        "claim_scope": "fused coefficient-controller, 28-factor, recurrence forward+backward",
        "shape": {
            "batch": shape[0],
            "length": shape[1],
            "channels": shape[2],
            "input_size": shape[3],
            "representations": 3,
            "state_dimension": 8,
        },
        "dtype": "float32",
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "timing": {
            "method": "CUDA events on current stream",
            "warmups": 10,
            "repetitions": args.repetitions,
            "median_forward_backward_microseconds": {
                "triton": triton_us,
                "raw_cuda": raw_us,
            },
            "raw_over_triton": raw_us / triton_us,
        },
        "maximum_absolute_error": {
            "output": float(
                (raw_result[0] - triton_result[0]).abs().max().detach()
            ),
            "gradients": [
                float((actual - expected).abs().max().detach())
                for actual, expected in zip(
                    raw_result[1], triton_result[1], strict=True
                )
            ],
        },
        "implementation_sha256": {
            path.relative_to(ROOT).as_posix(): file_sha256(path)
            for path in (
                Path(__file__),
                ROOT / "raw_cuda.py",
                ROOT / "csrc" / "spin_scan.cpp",
                ROOT / "csrc" / "spin_scan_cuda.cu",
            )
        },
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
