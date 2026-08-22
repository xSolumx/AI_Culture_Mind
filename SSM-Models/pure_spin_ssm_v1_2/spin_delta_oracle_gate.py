"""Oracle-address intervention for the Spin-Delta overwrite task."""

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
from model import PureSpinSSMV12, PureSpinV12Config
from spin_delta_capability_gate import overwrite_retrieval_batch

ROOT = Path(__file__).resolve().parent
VARIANTS = ("learned_addresses", "oracle_addresses")


@dataclass(frozen=True)
class OracleConfig:
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
    seed: int = 431


def build(config: OracleConfig) -> PureSpinSSMV12:
    return PureSpinSSMV12(
        PureSpinV12Config(
            d_model=config.d_model,
            num_layers=config.layers,
            spin_channels=2,
            group_schedule=tuple(8 for _ in range(config.layers)),
            recurrence="spin_delta",
        )
    )


def frozen_batches(
    config: OracleConfig, device: torch.device
) -> dict[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    rows = {}
    for writes in config.evaluation_writes:
        generator = torch.Generator().manual_seed(
            930_000 + 1013 * config.seed + 37 * writes
        )
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
def evaluate(
    model: PureSpinSSMV12,
    rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    use_oracle: bool,
) -> dict[str, float | int]:
    model.eval()
    correct = 0
    examples = 0
    loss_sum = 0.0
    for inputs, target, oracle in rows:
        logits = model(
            inputs,
            scan_mode="raw_cuda_delta",
            delta_oracle_slots=oracle if use_oracle else None,
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


def run_variant(
    variant: str,
    config: OracleConfig,
    batches: dict[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    device: torch.device,
) -> dict[str, object]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    use_oracle = variant == "oracle_addresses"
    seed_all(config.seed)
    model = build(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed)
    initial = {
        str(writes): evaluate(model, rows, use_oracle=use_oracle)
        for writes, rows in batches.items()
    }
    sampled_losses = {}
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
            delta_oracle_slots=oracle if use_oracle else None,
        )["logits"][:, -1]
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
        str(writes): evaluate(model, rows, use_oracle=use_oracle)
        for writes, rows in batches.items()
    }
    return {
        "variant": variant,
        "parameters": parameter_count(model),
        "oracle_intervention": use_oracle,
        "initial": initial,
        "final": final,
        "sampled_training_losses": sampled_losses,
        "training_seconds": seconds,
        "training_examples_per_second": config.steps * config.batch_size / seconds,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }


def pairing_audit(config: OracleConfig) -> dict[str, object]:
    seed_all(config.seed)
    learned = build(config)
    seed_all(config.seed)
    oracle = build(config)
    exact = all(
        torch.equal(oracle.state_dict()[name], value)
        for name, value in learned.state_dict().items()
    )
    if not exact:
        raise RuntimeError("oracle intervention models must start bitwise equal")
    return {
        "all_parameters_bitwise_equal": exact,
        "initial_logit_equality_required": False,
        "reason": "the supplied causal address tensor is the intervention",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batches", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = OracleConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        evaluation_batches=args.evaluation_batches,
        seed=args.seed,
    )
    device = torch.device("cuda")
    pairing = pairing_audit(config)
    batches = frozen_batches(config, device)
    results = [run_variant(row, config, batches, device) for row in VARIANTS]
    implementation_paths = (
        Path(__file__),
        ROOT / "spin_delta_capability_gate.py",
        ROOT / "model.py",
        ROOT / "spin_delta_scan.py",
        ROOT / "raw_cuda.py",
        ROOT / "csrc" / "spin_scan.cpp",
        ROOT / "csrc" / "spin_scan_cuda.cu",
    )
    report = {
        "schema_version": 1,
        "stage": "spin_delta_oracle_address_intervention",
        "claim_scope": (
            "causal oracle-address intervention on the frozen two-key overwrite "
            "task; not a language-quality or autonomous-addressing claim"
        ),
        "protocol": "SPIN_DELTA_ORACLE_ADDRESS_PREREGISTRATION.md",
        "config": asdict(config),
        "variant_order": list(VARIANTS),
        "pairing": pairing,
        "intervention": {
            "write_timing": "value-token positions only",
            "write_and_erase_slot": "supplied binary semantic key",
            "erase_strength": 1.0,
            "query_slot": "supplied only at final query key",
            "unchanged": [
                "model parameters",
                "Spin transport",
                "two-slot state",
                "raw CUDA recurrence",
                "optimizer",
                "training and evaluation batches",
            ],
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
