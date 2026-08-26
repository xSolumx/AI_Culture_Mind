"""Focused SM75 benchmark for exact canonical primitive exceptional transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch

from .primitive_action import (
    PrimitiveExceptionalAction,
    _extension_name,
    dense_primitive_product_oracle,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _measure(operation, warmups: int, repeats: int) -> list[float]:
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def _summary(samples: list[float]) -> dict[str, object]:
    return {
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "samples_ms": samples,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError("this benchmark requires the exact local SM75 GPU")
    torch.manual_seed(args.seed)
    rows = {}
    for algebra in ("f4", "e6"):
        action = PrimitiveExceptionalAction(algebra, backend="cuda").cuda().float()
        values = torch.randn(
            args.batch,
            args.length,
            args.copies,
            27,
            device="cuda",
            requires_grad=True,
        )
        coordinates = (
            args.coordinate_scale
            * torch.randn(
                args.batch,
                args.length,
                action.coordinate_dim,
                device="cuda",
            )
        ).requires_grad_()
        cotangent = torch.randn_like(values)

        def native_forward() -> None:
            action(values, coordinates)

        def native_forward_backward() -> None:
            values.grad = None
            coordinates.grad = None
            action(values, coordinates).backward(cotangent)

        def dense_forward() -> None:
            dense_primitive_product_oracle(values, coordinates, algebra)

        def dense_forward_backward() -> None:
            values.grad = None
            coordinates.grad = None
            dense_primitive_product_oracle(values, coordinates, algebra).backward(
                cotangent
            )

        actual = action(values, coordinates)
        expected = dense_primitive_product_oracle(values, coordinates, algebra)
        actual_gradients = torch.autograd.grad(
            actual, (values, coordinates), cotangent, retain_graph=True
        )
        expected_gradients = torch.autograd.grad(
            expected, (values, coordinates), cotangent
        )
        native_forward_samples = _measure(
            native_forward, args.warmups, args.repeats
        )
        dense_forward_samples = _measure(
            dense_forward, max(1, args.warmups // 2), args.repeats
        )
        native_backward_samples = _measure(
            native_forward_backward, args.warmups, args.repeats
        )
        dense_backward_samples = _measure(
            dense_forward_backward,
            max(1, args.warmups // 2),
            max(2, args.repeats // 2),
        )
        rows[algebra] = {
            "factor_count": action.coordinate_dim,
            "compact_factor_count": int((~action.content_mask).sum()),
            "content_factor_count": int(action.content_mask.sum()),
            "maximum_operator_norm": float(action.operator_norms.max()),
            "forward_max_abs_error": float(
                (actual - expected).abs().max().detach()
            ),
            "value_gradient_max_abs_error": float(
                (actual_gradients[0] - expected_gradients[0]).abs().max().detach()
            ),
            "coordinate_gradient_max_abs_error": float(
                (actual_gradients[1] - expected_gradients[1]).abs().max().detach()
            ),
            "native_forward": _summary(native_forward_samples),
            "dense_forward": _summary(dense_forward_samples),
            "native_forward_backward": _summary(native_backward_samples),
            "dense_forward_backward": _summary(dense_backward_samples),
            "forward_speedup": statistics.median(dense_forward_samples)
            / statistics.median(native_forward_samples),
            "forward_backward_speedup": statistics.median(dense_backward_samples)
            / statistics.median(native_backward_samples),
        }
    root = Path(__file__).resolve().parent
    repository = root.parents[1]
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "schema_version": 1,
        "experiment": "exact canonical primitive exceptional transport SM75",
        "status": "focused kernel qualification; not model-quality evidence",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
        "shape": {
            "batch": args.batch,
            "length": args.length,
            "copies": args.copies,
            "coordinate_scale": args.coordinate_scale,
        },
        "seed": args.seed,
        "native_extension_name": _extension_name(),
        "git": {
            "revision": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "dirty": bool(status.strip()),
            "working_patch_sha256": hashlib.sha256(diff).hexdigest(),
        },
        "source_sha256": {
            name: _sha256(root / name)
            for name in (
                "albert.py",
                "benchmark_primitive_action.py",
                "primitive_action.py",
                "primitive_action_bindings.cpp",
                "primitive_action_cuda.cu",
            )
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--copies", type=int, default=4)
    parser.add_argument("--coordinate-scale", type=float, default=0.02)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.time()
    report = run(args)
    report["wall_seconds"] = time.time() - started
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(payload, end="")


if __name__ == "__main__":
    main()
