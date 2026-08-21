"""Development profiler for exceptional action construction and model steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch

from .action import build_exceptional_action
from .model import ExceptionalDeltaConfig, ExceptionalDeltaLM, parameter_count


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _measure_cuda(
    operation: Callable[[], None], *, warmups: int, repetitions: int
) -> list[float]:
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def _measure_cpu(
    operation: Callable[[], None], *, warmups: int, repetitions: int
) -> list[float]:
    for _ in range(warmups):
        operation()
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        operation()
        samples.append(1e3 * (time.perf_counter() - start))
    return samples


def _summary(samples: list[float]) -> dict[str, object]:
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
    }


def profile(args: argparse.Namespace) -> dict[str, object]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    measure = _measure_cuda if device.type == "cuda" else _measure_cpu
    torch.manual_seed(args.seed)

    rows: dict[str, object] = {}
    action_specs = (
        ("identity", "identity", "direct"),
        ("spin8", "spin8", "direct"),
        ("spin9", "spin9", "direct"),
        ("f4", "f4", "direct"),
        ("e6_direct", "e6", "direct"),
        ("e6_polar", "e6", "polar"),
        ("e6_cartan", "e6", "cartan"),
    )
    for row_name, algebra, geometry in action_specs:
        action = build_exceptional_action(algebra, geometry=geometry).to(
            device=device, dtype=dtype
        )
        coordinates = (
            args.coordinate_scale
            * torch.randn(
                args.batch,
                args.length,
                args.action_factors,
                action.coordinate_dim,
                device=device,
                dtype=dtype,
            )
        ).requires_grad_()

        def action_training_forward(
            _action: torch.nn.Module = action,
            _coordinates: torch.Tensor = coordinates,
        ) -> None:
            _action.ordered(_coordinates)

        def action_inference(
            _action: torch.nn.Module = action,
            _coordinates: torch.Tensor = coordinates,
        ) -> None:
            with torch.inference_mode():
                _action.ordered(_coordinates)

        def action_forward_backward(
            _action: torch.nn.Module = action,
            _coordinates: torch.Tensor = coordinates,
        ) -> None:
            _coordinates.grad = None
            matrix = _action.ordered(_coordinates)
            matrix.square().mean().backward()

        action_training_samples = measure(
            action_training_forward,
            warmups=args.warmups,
            repetitions=args.repetitions,
        )
        action_inference_samples = measure(
            action_inference,
            warmups=args.warmups,
            repetitions=args.repetitions,
        )
        action_backward_summary = None
        if action.coordinate_dim:
            action_backward_samples = measure(
                action_forward_backward,
                warmups=max(1, args.warmups // 2),
                repetitions=max(1, args.repetitions // 2),
            )
            action_backward_summary = _summary(action_backward_samples)

        config = ExceptionalDeltaConfig(
            d_model=args.d_model,
            num_layers=1,
            memory_width=args.memory_width,
            update_rank=args.update_rank,
            action_algebra=algebra,
            action_geometry=geometry,
            action_factors=args.action_factors,
            d_conv=args.d_conv,
            channel_mixer=args.channel_mixer,
        )
        model = ExceptionalDeltaLM(config).to(device=device, dtype=dtype)
        tokens = torch.randint(
            0, config.vocab_size, (args.batch, args.length), device=device
        )

        def model_forward(
            _model: ExceptionalDeltaLM = model,
            _tokens: torch.Tensor = tokens,
        ) -> None:
            _model(_tokens, scan_mode=args.scan_mode)["logits"]

        def model_inference(
            _model: ExceptionalDeltaLM = model,
            _tokens: torch.Tensor = tokens,
        ) -> None:
            with torch.inference_mode():
                _model(_tokens, scan_mode=args.scan_mode)["logits"]

        def model_forward_backward(
            _model: ExceptionalDeltaLM = model,
            _tokens: torch.Tensor = tokens,
        ) -> None:
            _model.zero_grad(set_to_none=True)
            logits = _model(_tokens, scan_mode=args.scan_mode)["logits"]
            logits.square().mean().backward()

        forward_samples = measure(
            model_forward, warmups=args.warmups, repetitions=args.repetitions
        )
        inference_samples = measure(
            model_inference, warmups=args.warmups, repetitions=args.repetitions
        )
        backward_samples = measure(
            model_forward_backward,
            warmups=max(1, args.warmups // 2),
            repetitions=max(1, args.repetitions // 2),
        )
        rows[row_name] = {
            "algebra": algebra,
            "geometry": geometry,
            "coordinate_dim": action.coordinate_dim,
            "parameter_count": parameter_count(model),
            "cache_scalars": model.cache_scalars,
            "action_training_forward": _summary(action_training_samples),
            "action_inference": _summary(action_inference_samples),
            "action_forward_backward": action_backward_summary,
            "model_training_forward": _summary(forward_samples),
            "model_inference": _summary(inference_samples),
            "model_forward_backward": _summary(backward_samples),
        }

    root = Path(__file__).resolve().parent
    return {
        "schema_version": 1,
        "experiment": "Pure Exceptional Delta SSM v1.3 development profile",
        "status": "development bottleneck measurement; not a promoted benchmark",
        "seed": args.seed,
        "shape": {
            "batch": args.batch,
            "length": args.length,
            "d_model": args.d_model,
            "memory_width": args.memory_width,
            "update_rank": args.update_rank,
            "action_factors": args.action_factors,
            "d_conv": args.d_conv,
            "channel_mixer": args.channel_mixer,
            "scan_mode": args.scan_mode,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "cuda_runtime": torch.version.cuda,
            "dtype": args.dtype,
        },
        "source_sha256": {
            name: _file_sha256(root / name)
            for name in (
                "action.py",
                "albert.py",
                "benchmark_profile.py",
                "model.py",
                "scan.py",
            )
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--memory-width", type=int, default=4)
    parser.add_argument("--update-rank", type=int, default=2)
    parser.add_argument("--action-factors", type=int, default=1)
    parser.add_argument("--coordinate-scale", type=float, default=0.02)
    parser.add_argument("--d-conv", type=int, default=3)
    parser.add_argument("--channel-mixer", choices=("swiglu", "jordan", "none"), default="jordan")
    parser.add_argument(
        "--scan-mode", choices=("auto", "recurrent", "parallel"), default="auto"
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = profile(args)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
