"""Matched two-key overwrite/retrieval capability gate for Spin-Delta."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch.nn import functional as F

from benchmark import (
    BenchmarkConfig,
    build_model,
    file_sha256,
    package_version,
    parameter_count,
    seed_all,
)

ROOT = Path(__file__).resolve().parent
WRITE_TOKEN = 2
QUERY_TOKEN = 3
VALUE_OFFSET = 16
VALUE_COUNT = 32
VARIANTS = ("independent_v1_2", "spin_delta")


@dataclass(frozen=True)
class CapabilityConfig:
    steps: int = 800
    batch_size: int = 128
    training_writes: int = 8
    evaluation_writes: tuple[int, ...] = (8, 16, 32)
    evaluation_batches: int = 32
    learning_rate: float = 3.0e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    d_model: int = 64
    layers: int = 2
    seed: int = 401


def overwrite_retrieval_batch(
    batch_size: int,
    writes: int,
    *,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate balanced two-key streams and the latest queried value.

    Each episode is ``[WRITE,key,value] * writes + [QUERY,key]``.  The first
    two writes cover both keys in random order; subsequent writes are random.
    The target is the most recently written value for the final query key.
    """

    if batch_size < 1 or writes < 2:
        raise ValueError("batch_size must be positive and writes must be at least two")
    keys = torch.randint(2, (batch_size, writes), generator=generator)
    first_key = torch.randint(2, (batch_size,), generator=generator)
    keys[:, 0] = first_key
    keys[:, 1] = 1 - first_key
    values = torch.randint(
        VALUE_COUNT, (batch_size, writes), generator=generator
    ) + VALUE_OFFSET
    query = torch.randint(2, (batch_size,), generator=generator)
    positions = torch.arange(writes).expand(batch_size, -1)
    latest = torch.where(keys == query[:, None], positions, -1).max(dim=1).values
    target = values.gather(1, latest[:, None]).squeeze(1)

    sequence = torch.empty(batch_size, 3 * writes + 2, dtype=torch.long)
    sequence[:, 0 : 3 * writes : 3] = WRITE_TOKEN
    sequence[:, 1 : 3 * writes : 3] = keys
    sequence[:, 2 : 3 * writes : 3] = values
    sequence[:, -2] = QUERY_TOKEN
    sequence[:, -1] = query
    return sequence.to(device), target.to(device)


def benchmark_config(config: CapabilityConfig, variant: str) -> BenchmarkConfig:
    base = BenchmarkConfig(
        d_model=config.d_model,
        layers=config.layers,
        spin_channels=2,
        spin_group_schedule=tuple(8 for _ in range(config.layers)),
        seed=config.seed,
    )
    if variant == "independent_v1_2":
        return replace(
            base,
            spin_backend="raw_cuda_hybrid",
            spin_recurrence="independent",
        )
    if variant == "spin_delta":
        return replace(
            base,
            spin_backend="raw_cuda_delta",
            spin_recurrence="spin_delta",
        )
    raise ValueError(f"unknown variant {variant!r}")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    backend: str,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    model.eval()
    correct = 0
    examples = 0
    loss_sum = 0.0
    for inputs, target in batches:
        logits = model(inputs, scan_mode=backend)["logits"][:, -1]
        loss_sum += float(F.cross_entropy(logits, target, reduction="sum"))
        correct += int((logits.argmax(dim=-1) == target).sum())
        examples += target.numel()
    return {
        "accuracy": correct / examples,
        "nats_per_query": loss_sum / examples,
        "bits_per_query": loss_sum / examples / math.log(2.0),
        "queries": examples,
    }


def evaluation_batches(
    config: CapabilityConfig, device: torch.device
) -> dict[int, list[tuple[torch.Tensor, torch.Tensor]]]:
    rows = {}
    for writes in config.evaluation_writes:
        generator = torch.Generator().manual_seed(
            910_000 + 1009 * config.seed + 31 * writes
        )
        rows[writes] = [
            overwrite_retrieval_batch(
                config.batch_size,
                writes,
                generator=generator,
                device=device,
            )
            for _ in range(config.evaluation_batches)
        ]
    return rows


def run_variant(
    variant: str,
    config: CapabilityConfig,
    batches: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
    device: torch.device,
) -> dict[str, object]:
    model_config = benchmark_config(config, variant)
    seed_all(config.seed)
    model = build_model("pure_spin_v1_2", model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed)
    initial = {
        str(writes): evaluate(model, model_config.spin_backend, rows)
        for writes, rows in batches.items()
    }
    sampled_losses = {}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model.train()
    for step in range(1, config.steps + 1):
        inputs, target = overwrite_retrieval_batch(
            config.batch_size,
            config.training_writes,
            generator=generator,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs, scan_mode=model_config.spin_backend)["logits"][:, -1]
        loss = F.cross_entropy(logits, target)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"nonfinite gradient at step {step}")
        optimizer.step()
        if step in {1, config.steps // 2, config.steps}:
            sampled_losses[str(step)] = float(loss.detach())
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    final = {
        str(writes): evaluate(model, model_config.spin_backend, rows)
        for writes, rows in batches.items()
    }
    return {
        "variant": variant,
        "parameters": parameter_count(model),
        "backend": model_config.spin_backend,
        "initial": initial,
        "final": final,
        "sampled_training_losses": sampled_losses,
        "training_seconds": seconds,
        "training_examples_per_second": (
            config.steps * config.batch_size / seconds
        ),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }


def pairing_audit(
    config: CapabilityConfig, device: torch.device
) -> dict[str, object]:
    models = []
    model_configs = []
    for variant in VARIANTS:
        model_config = benchmark_config(config, variant)
        seed_all(config.seed)
        models.append(build_model("pure_spin_v1_2", model_config).to(device).eval())
        model_configs.append(model_config)
    baseline_state = models[0].state_dict()
    candidate_state = models[1].state_dict()
    common_equal = all(
        torch.equal(candidate_state[name], value)
        for name, value in baseline_state.items()
    )
    generator = torch.Generator().manual_seed(920_000 + config.seed)
    inputs, _ = overwrite_retrieval_batch(
        config.batch_size,
        config.training_writes,
        generator=generator,
        device=device,
    )
    with torch.no_grad():
        logits = [
            model(inputs, scan_mode=model_config.spin_backend)["logits"][:, -1]
            for model, model_config in zip(models, model_configs, strict=True)
        ]
    maximum = float((logits[0] - logits[1]).abs().max())
    if not common_equal or maximum > 2.0e-6:
        raise RuntimeError("capability variants violate the frozen pairing bound")
    return {
        "common_parameters_bitwise_equal": common_equal,
        "maximum_absolute_logit_difference": maximum,
        "maximum_allowed_logit_difference": 2.0e-6,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--training-writes", type=int, default=8)
    parser.add_argument("--evaluation-batches", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = CapabilityConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        training_writes=args.training_writes,
        evaluation_batches=args.evaluation_batches,
        seed=args.seed,
    )
    device = torch.device("cuda")
    pairing = pairing_audit(config, device)
    batches = evaluation_batches(config, device)
    results = [run_variant(row, config, batches, device) for row in VARIANTS]
    implementation_paths = (
        Path(__file__),
        ROOT / "model.py",
        ROOT / "spin_delta_scan.py",
        ROOT / "raw_cuda.py",
        ROOT / "csrc" / "spin_scan.cpp",
        ROOT / "csrc" / "spin_scan_cuda.cu",
    )
    report = {
        "schema_version": 1,
        "stage": "spin_delta_overwrite_capability",
        "claim_scope": (
            "matched synthetic two-key overwrite/retrieval capability; not a "
            "language-quality or general memory claim"
        ),
        "protocol": "SPIN_DELTA_CAPABILITY_PREREGISTRATION.md",
        "config": asdict(config),
        "variant_order": list(VARIANTS),
        "pairing": pairing,
        "task": {
            "keys": 2,
            "values": VALUE_COUNT,
            "sequence": "[WRITE,key,value] * writes + [QUERY,key]",
            "target": "latest value written to queried key",
            "chance_accuracy": 1.0 / VALUE_COUNT,
        },
        "environment": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "triton": package_version("triton", "triton-windows"),
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
