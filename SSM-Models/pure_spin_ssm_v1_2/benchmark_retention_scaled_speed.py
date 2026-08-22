"""Order-balanced complete-step timing for retention-scaled block coupling."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import asdict, replace
from pathlib import Path

import torch
from benchmark import BenchmarkConfig, build_model, parameter_count, seed_all
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
VARIANTS = ("maintained", "retention_scaled")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variant_config(label: str, base: BenchmarkConfig) -> BenchmarkConfig:
    if label == "maintained":
        return replace(
            base,
            spin_backend="raw_cuda_hybrid",
            spin_recurrence="independent",
            spin_recurrent_multiplicity="identity",
            spin_recurrent_coupling_scale="unit",
        )
    return replace(
        base,
        spin_backend="raw_cuda_block",
        spin_recurrence="independent_block",
        spin_recurrent_multiplicity="orthogonal",
        spin_recurrent_coupling_scale="retention_step",
    )


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    backend: str,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs, scan_mode=backend)["logits"]
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def measure(
    label: str,
    config: BenchmarkConfig,
    *,
    warmup_steps: int,
    windows: int,
    steps_per_window: int,
) -> dict[str, object]:
    seed_all(config.seed)
    model = build_model("pure_spin_v1_2", config).cuda().train()
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
        loss = train_step(model, optimizer, inputs, targets, config.spin_backend)
    torch.cuda.synchronize()
    window_ms = []
    for _ in range(windows):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(steps_per_window):
            loss = train_step(model, optimizer, inputs, targets, config.spin_backend)
        end.record()
        end.synchronize()
        window_ms.append(start.elapsed_time(end))
    tokens = config.batch_size * config.sequence_length * steps_per_window
    throughput = [tokens / (milliseconds * 1.0e-3) for milliseconds in window_ms]
    result = {
        "variant": label,
        "backend": config.spin_backend,
        "parameters": parameter_count(model),
        "final_training_loss": loss,
        "window_milliseconds": window_ms,
        "tokens_per_second": {
            "median": statistics.median(throughput),
            "mean": statistics.fmean(throughput),
            "minimum": min(throughput),
            "maximum": max(throughput),
        },
    }
    del optimizer, model, inputs, targets
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--steps-per-window", type=int, default=10)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    base = BenchmarkConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        spin_group_schedule=(3, 4, 6, 8),
    )
    configs = {label: variant_config(label, base) for label in VARIANTS}
    cycles = []
    for cycle in range(args.cycles):
        order = VARIANTS if cycle % 2 == 0 else tuple(reversed(VARIANTS))
        cycles.append(
            {
                "cycle": cycle + 1,
                "order": list(order),
                "results": [
                    measure(
                        label,
                        configs[label],
                        warmup_steps=args.warmup_steps,
                        windows=args.windows,
                        steps_per_window=args.steps_per_window,
                    )
                    for label in order
                ],
            }
        )

    medians = {label: [] for label in VARIANTS}
    for cycle in cycles:
        for result in cycle["results"]:
            medians[result["variant"]].append(
                result["tokens_per_second"]["median"]
            )
    aggregate = {
        label: {
            "cycle_medians_tokens_per_second": values,
            "median_of_cycle_medians_tokens_per_second": statistics.median(values),
        }
        for label, values in medians.items()
    }
    ratio = (
        aggregate["retention_scaled"]["median_of_cycle_medians_tokens_per_second"]
        / aggregate["maintained"]["median_of_cycle_medians_tokens_per_second"]
    )
    aggregate["retention_scaled_over_maintained"] = ratio
    aggregate["maximum_allowed_regression"] = 0.10
    aggregate["passes_no_more_than_ten_percent_regression"] = ratio >= 0.90

    implementation_paths = (
        Path(__file__),
        ROOT / "benchmark.py",
        ROOT / "model.py",
        ROOT / "coupled_isotypic_scan.py",
        ROOT / "raw_cuda.py",
        ROOT / "csrc" / "spin_scan.cpp",
        ROOT / "csrc" / "spin_scan_cuda.cu",
    )
    report = {
        "schema_version": 1,
        "claim_scope": (
            "order-balanced fixed-batch complete GPU training steps; not a "
            "convergence, end-to-end, or cross-device speed claim"
        ),
        "config": {
            **asdict(base),
            "warmup_steps": args.warmup_steps,
            "windows": args.windows,
            "steps_per_window": args.steps_per_window,
            "cycles": args.cycles,
        },
        "variant_configs": {
            label: {
                "backend": config.spin_backend,
                "recurrence": config.spin_recurrence,
                "recurrent_multiplicity": config.spin_recurrent_multiplicity,
                "recurrent_coupling_scale": config.spin_recurrent_coupling_scale,
            }
            for label, config in configs.items()
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "timing": {
            "method": "CUDA events around windows of complete training steps",
            "includes": "forward, backward, gradient clipping, and AdamW update",
            "excludes": "data loading, host-to-device copies, and validation",
        },
        "cycles": cycles,
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
