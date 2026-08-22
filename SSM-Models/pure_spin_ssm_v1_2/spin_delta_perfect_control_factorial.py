"""One cell of the frozen perfect-control Spin-Delta factorial."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from benchmark import file_sha256, package_version, parameter_count, seed_all
from spin_delta_capability_gate import overwrite_retrieval_batch
from spin_delta_oracle_gate import build

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FactorialConfig:
    init_seed: int
    data_seed: int
    steps: int = 800
    batch_size: int = 128
    training_writes: int = 8
    evaluation_writes: tuple[int, ...] = (8, 16, 32)
    evaluation_batches: int = 16
    learning_rate: float = 3.0e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    d_model: int = 64
    layers: int = 2


def evaluation_rows(config: FactorialConfig, device: torch.device):
    rows = {}
    for writes in config.evaluation_writes:
        generator = torch.Generator().manual_seed(990_001 + 43 * writes)
        rows[writes] = [
            overwrite_retrieval_batch(
                config.batch_size,
                writes,
                generator=generator,
                device=device,
                return_oracle=True,
            )
            for _ in range(config.evaluation_batches)
        ]
    return rows


@torch.no_grad()
def evaluate(model, rows) -> dict[str, float | int]:
    model.eval()
    correct = examples = 0
    loss_sum = 0.0
    for inputs, target, oracle in rows:
        logits = model(
            inputs,
            scan_mode="raw_cuda_delta",
            delta_oracle_slots=oracle,
        )["logits"][:, -1]
        loss_sum += float(F.cross_entropy(logits, target, reduction="sum"))
        correct += int((logits.argmax(dim=-1) == target).sum())
        examples += target.numel()
    return {
        "accuracy": correct / examples,
        "nats_per_query": loss_sum / examples,
        "bits_per_query": loss_sum / examples / math.log(2.0),
        "queries": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = FactorialConfig(args.init_seed, args.data_seed)
    device = torch.device("cuda")
    seed_all(config.init_seed)
    model = build(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.data_seed)
    rows = evaluation_rows(config, device)
    initial = {str(w): evaluate(model, value) for w, value in rows.items()}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model.train()
    for step in range(1, config.steps + 1):
        inputs, target, oracle = overwrite_retrieval_batch(
            config.batch_size,
            config.training_writes,
            generator=generator,
            device=device,
            return_oracle=True,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            inputs,
            scan_mode="raw_cuda_delta",
            delta_oracle_slots=oracle,
        )["logits"][:, -1]
        loss = F.cross_entropy(logits, target)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"nonfinite gradient at step {step}")
        optimizer.step()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    final = {str(w): evaluate(model, value) for w, value in rows.items()}
    paths = (
        Path(__file__),
        ROOT / "spin_delta_oracle_gate.py",
        ROOT / "model.py",
        ROOT / "spin_delta_scan.py",
        ROOT / "raw_cuda.py",
        ROOT / "csrc" / "spin_scan.cpp",
        ROOT / "csrc" / "spin_scan_cuda.cu",
    )
    report = {
        "schema_version": 1,
        "stage": "spin_delta_perfect_control_factorial",
        "protocol": "SPIN_DELTA_PERFECT_CONTROL_FACTORIAL_PREREGISTRATION.md",
        "config": asdict(config),
        "intervention": "exact causal controls from optimizer step one",
        "parameters": parameter_count(model),
        "initial": initial,
        "final": final,
        "training_seconds": seconds,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "environment": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "triton": package_version("triton", "triton-windows"),
        },
        "implementation_sha256": {
            path.relative_to(ROOT).as_posix(): file_sha256(path) for path in paths
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
