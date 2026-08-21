"""Order-balanced steady GPU training-step benchmark for Spin and Mamba-2."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
from benchmark import (
    BenchmarkConfig,
    build_model,
    package_version,
    parameter_match,
    seed_all,
)
from mamba2_baseline import fused_mamba2_available
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    name: str,
    spin_backend: str,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    kwargs = {"scan_mode": spin_backend} if name == "pure_spin_v1_2" else {}
    logits = model(inputs, **kwargs)["logits"]
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def measure(
    name: str,
    config: BenchmarkConfig,
    *,
    warmup_steps: int,
    windows: int,
    steps_per_window: int,
) -> dict[str, object]:
    seed_all(config.seed)
    model = build_model(name, config).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    generator = torch.Generator(device="cuda").manual_seed(config.seed + 91)
    inputs = torch.randint(
        0,
        256,
        (config.batch_size, config.sequence_length),
        device="cuda",
        generator=generator,
    )
    targets = torch.randint(
        0,
        256,
        (config.batch_size, config.sequence_length),
        device="cuda",
        generator=generator,
    )
    loss = 0.0
    for _ in range(warmup_steps):
        loss = train_step(model, optimizer, inputs, targets, name, config.spin_backend)
    torch.cuda.synchronize()
    window_ms: list[float] = []
    for _ in range(windows):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(steps_per_window):
            loss = train_step(
                model, optimizer, inputs, targets, name, config.spin_backend
            )
        end.record()
        end.synchronize()
        window_ms.append(start.elapsed_time(end))
    tokens = config.batch_size * config.sequence_length * steps_per_window
    throughput = [tokens / (milliseconds * 1e-3) for milliseconds in window_ms]
    del optimizer, model, inputs, targets
    torch.cuda.empty_cache()
    return {
        "name": name,
        "final_training_loss": loss,
        "window_milliseconds": window_ms,
        "tokens_per_second": {
            "median": statistics.median(throughput),
            "mean": statistics.fmean(throughput),
            "minimum": min(throughput),
            "maximum": max(throughput),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--steps-per-window", type=int, default=10)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/steady_step_spin_ladder_vs_mamba2.json"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    available, detail = fused_mamba2_available()
    if not available:
        raise RuntimeError(f"official fused Mamba-2 unavailable: {detail}")
    config = BenchmarkConfig(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        spin_backend="raw_cuda_factorized",
        spin_group_schedule=(3, 4, 6, 8),
        seed=args.seed,
    )
    cycle_results = []
    names = ("pure_spin_v1_2", "mamba2_fused")
    for cycle in range(args.cycles):
        order = names if cycle % 2 == 0 else tuple(reversed(names))
        cycle_results.append(
            {
                "cycle": cycle + 1,
                "order": list(order),
                "results": [
                    measure(
                        name,
                        config,
                        warmup_steps=args.warmup_steps,
                        windows=args.windows,
                        steps_per_window=args.steps_per_window,
                    )
                    for name in order
                ],
            }
        )
    medians: dict[str, list[float]] = {name: [] for name in names}
    for cycle in cycle_results:
        for result in cycle["results"]:
            medians[result["name"]].append(result["tokens_per_second"]["median"])
    aggregate = {
        name: {
            "median_of_cycle_medians_tokens_per_second": statistics.median(values),
            "cycle_medians_tokens_per_second": values,
        }
        for name, values in medians.items()
    }
    aggregate["mamba2_throughput_over_spin"] = (
        aggregate["mamba2_fused"]["median_of_cycle_medians_tokens_per_second"]
        / aggregate["pure_spin_v1_2"]["median_of_cycle_medians_tokens_per_second"]
    )
    aggregate["spin_throughput_over_mamba2"] = (
        aggregate["pure_spin_v1_2"]["median_of_cycle_medians_tokens_per_second"]
        / aggregate["mamba2_fused"]["median_of_cycle_medians_tokens_per_second"]
    )
    implementation_paths = (
        Path(__file__),
        ROOT / "benchmark.py",
        ROOT / "model.py",
        ROOT / "mamba2_baseline.py",
        ROOT / "raw_cuda.py",
        ROOT / "csrc" / "spin_scan.cpp",
        ROOT / "csrc" / "spin_scan_cuda.cu",
    )
    report = {
        "schema_version": 1,
        "claim_scope": (
            "order-balanced fixed-batch GPU training-step throughput; "
            "not a convergence or general superiority claim"
        ),
        "config": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "warmup_steps": args.warmup_steps,
            "windows": args.windows,
            "steps_per_window": args.steps_per_window,
            "cycles": args.cycles,
            "spin_backend": config.spin_backend,
            "spin_group_schedule": list(config.spin_group_schedule or ()),
        },
        "parameter_match": parameter_match(config),
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": package_version("triton", "triton-windows"),
            "mamba2_fused": list(fused_mamba2_available()),
        },
        "timing": {
            "method": "CUDA events around windows of complete training steps",
            "includes": "forward, backward, gradient clipping, and AdamW update",
            "excludes": "data loading, host-to-device copies, and validation",
        },
        "cycles": cycle_results,
        "aggregate": aggregate,
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
