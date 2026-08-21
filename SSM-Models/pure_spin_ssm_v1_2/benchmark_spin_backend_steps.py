"""Order-balanced complete-step comparison of packed and hybrid Spin backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import replace
from pathlib import Path

import torch
from benchmark import BenchmarkConfig, parameter_match
from benchmark_steady_step import measure

ROOT = Path(__file__).resolve().parent


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--steps-per-window", type=int, default=10)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/spin_backend_complete_steps.json"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    base = BenchmarkConfig(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        spin_group_schedule=(3, 4, 6, 8),
        seed=args.seed,
    )
    backends = ("raw_cuda_factorized", "raw_cuda_hybrid")
    cycles = []
    medians: dict[str, list[float]] = {backend: [] for backend in backends}
    for cycle in range(args.cycles):
        order = backends if cycle % 2 == 0 else tuple(reversed(backends))
        results = []
        for backend in order:
            result = measure(
                "pure_spin_v1_2",
                replace(base, spin_backend=backend),
                warmup_steps=args.warmup_steps,
                windows=args.windows,
                steps_per_window=args.steps_per_window,
            )
            result["backend"] = backend
            medians[backend].append(result["tokens_per_second"]["median"])
            results.append(result)
        cycles.append({"cycle": cycle + 1, "order": list(order), "results": results})
    aggregate = {
        backend: {
            "cycle_medians_tokens_per_second": values,
            "median_of_cycle_medians_tokens_per_second": statistics.median(values),
        }
        for backend, values in medians.items()
    }
    aggregate["hybrid_throughput_over_packed"] = (
        aggregate["raw_cuda_hybrid"]["median_of_cycle_medians_tokens_per_second"]
        / aggregate["raw_cuda_factorized"][
            "median_of_cycle_medians_tokens_per_second"
        ]
    )
    implementation_paths = (
        Path(__file__),
        ROOT / "benchmark.py",
        ROOT / "benchmark_steady_step.py",
        ROOT / "model.py",
        ROOT / "raw_cuda.py",
        ROOT / "csrc" / "spin_scan.cpp",
        ROOT / "csrc" / "spin_scan_cuda.cu",
    )
    report = {
        "schema_version": 1,
        "claim_scope": (
            "order-balanced complete Spin training steps on one fixed batch; "
            "not a convergence or cross-model claim"
        ),
        "config": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "warmup_steps": args.warmup_steps,
            "windows": args.windows,
            "steps_per_window": args.steps_per_window,
            "cycles": args.cycles,
            "spin_group_schedule": list(base.spin_group_schedule or ()),
        },
        "parameter_match_reference": parameter_match(base),
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
