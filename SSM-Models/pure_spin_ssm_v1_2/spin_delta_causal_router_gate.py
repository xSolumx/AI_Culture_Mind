"""Frozen learned causal-router gate for the Spin-Delta overwrite task."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from benchmark import file_sha256, package_version, parameter_count, seed_all
from model import PureSpinSSMV12, PureSpinV12Config
from spin_delta_capability_gate import overwrite_retrieval_batch
from spin_delta_router import (
    CausalLowEntropyRouter,
    RoutedSpinDelta,
    RouterOutput,
    router_supervision_loss,
)

ROOT = Path(__file__).resolve().parent
VARIANTS = ("learned_continuous", "causal_discrete_aux")


@dataclass(frozen=True)
class RouterGateConfig:
    steps: int = 800
    batch_size: int = 128
    training_writes: int = 8
    evaluation_writes: tuple[int, ...] = (8, 16, 32)
    evaluation_batches: int = 32
    learning_rate: float = 3.0e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    auxiliary_weight: float = 1.0
    d_model: int = 64
    layers: int = 2
    router_width: int = 32
    router_kernel_size: int = 3
    router_temperature: float = 1.0
    seed: int = 449


def build_core(config: RouterGateConfig) -> PureSpinSSMV12:
    return PureSpinSSMV12(
        PureSpinV12Config(
            d_model=config.d_model,
            num_layers=config.layers,
            spin_channels=2,
            group_schedule=tuple(8 for _ in range(config.layers)),
            recurrence="spin_delta",
        )
    )


def build_candidate(config: RouterGateConfig) -> RoutedSpinDelta:
    return RoutedSpinDelta(
        build_core(config),
        CausalLowEntropyRouter(
            width=config.router_width,
            kernel_size=config.router_kernel_size,
            temperature=config.router_temperature,
        ),
    )


def frozen_batches(
    config: RouterGateConfig, device: torch.device
) -> dict[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    rows = {}
    for writes in config.evaluation_writes:
        generator = torch.Generator().manual_seed(
            950_000 + 1019 * config.seed + 41 * writes
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


def _router_counts(routing: RouterOutput, oracle: torch.Tensor) -> dict[str, int]:
    controls = routing.controls
    write_prediction = controls[..., 0] >= 0.5
    query_prediction = controls[..., 3] >= 0.5
    write_target = oracle[..., 0] >= 0
    query_target = oracle[..., 1] >= 0
    write_slot_prediction = controls[..., 1:3].argmax(dim=-1)
    query_slot_prediction = controls[..., 4:6].argmax(dim=-1)
    return {
        "write_true_positive": int((write_prediction & write_target).sum()),
        "write_false_positive": int((write_prediction & ~write_target).sum()),
        "write_false_negative": int((~write_prediction & write_target).sum()),
        "query_true_positive": int((query_prediction & query_target).sum()),
        "query_false_positive": int((query_prediction & ~query_target).sum()),
        "query_false_negative": int((~query_prediction & query_target).sum()),
        "write_slot_correct": int(
            (write_slot_prediction[write_target] == oracle[..., 0][write_target]).sum()
        ),
        "write_slot_total": int(write_target.sum()),
        "query_slot_correct": int(
            (query_slot_prediction[query_target] == oracle[..., 1][query_target]).sum()
        ),
        "query_slot_total": int(query_target.sum()),
    }


def _merge_counts(total: dict[str, int], row: dict[str, int]) -> None:
    for name, value in row.items():
        total[name] = total.get(name, 0) + value


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def _router_metrics(counts: dict[str, int]) -> dict[str, float | int]:
    return {
        "write_event_f1": _f1(
            counts["write_true_positive"],
            counts["write_false_positive"],
            counts["write_false_negative"],
        ),
        "query_event_f1": _f1(
            counts["query_true_positive"],
            counts["query_false_positive"],
            counts["query_false_negative"],
        ),
        "write_slot_accuracy": (
            counts["write_slot_correct"] / counts["write_slot_total"]
        ),
        "query_slot_accuracy": (
            counts["query_slot_correct"] / counts["query_slot_total"]
        ),
        **counts,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    routed: bool,
) -> dict[str, object]:
    model.eval()
    correct = 0
    examples = 0
    loss_sum = 0.0
    counts: dict[str, int] = {}
    for inputs, target, oracle in rows:
        result = model(inputs, scan_mode="raw_cuda_delta")
        logits = result["logits"][:, -1]
        loss_sum += float(F.cross_entropy(logits, target, reduction="sum"))
        correct += int((logits.argmax(dim=-1) == target).sum())
        examples += target.numel()
        if routed:
            _merge_counts(counts, _router_counts(result["router"], oracle))
    report: dict[str, object] = {
        "accuracy": correct / examples,
        "nats_per_query": loss_sum / examples,
        "bits_per_query": loss_sum / examples / math.log(2.0),
        "queries": examples,
    }
    if routed:
        report["router"] = _router_metrics(counts)
    return report


def run_variant(
    variant: str,
    config: RouterGateConfig,
    batches: dict[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    device: torch.device,
) -> dict[str, object]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    routed = variant == "causal_discrete_aux"
    seed_all(config.seed)
    model: nn.Module = build_candidate(config) if routed else build_core(config)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed)
    initial = {
        str(writes): evaluate(model, rows, routed=routed)
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
        result = model(inputs, scan_mode="raw_cuda_delta")
        retrieval_loss = F.cross_entropy(result["logits"][:, -1], target)
        if routed:
            auxiliary = router_supervision_loss(result["router"], oracle)
            loss = retrieval_loss + config.auxiliary_weight * auxiliary["total"]
        else:
            auxiliary = None
            loss = retrieval_loss
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"nonfinite gradient at step {step}")
        optimizer.step()
        if step in {1, config.steps // 2, config.steps}:
            sampled_losses[str(step)] = {
                "total": float(loss.detach()),
                "retrieval": float(retrieval_loss.detach()),
                "auxiliary": (
                    float(auxiliary["total"].detach()) if auxiliary else None
                ),
            }
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    final = {
        str(writes): evaluate(model, rows, routed=routed)
        for writes, rows in batches.items()
    }
    return {
        "variant": variant,
        "parameters": parameter_count(model),
        "autonomous_evaluation": True,
        "training_uses_router_labels": routed,
        "initial": initial,
        "final": final,
        "sampled_training_losses": sampled_losses,
        "training_seconds": seconds,
        "training_examples_per_second": config.steps * config.batch_size / seconds,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }


def pairing_audit(config: RouterGateConfig) -> dict[str, object]:
    seed_all(config.seed)
    baseline = build_core(config)
    seed_all(config.seed)
    candidate = build_candidate(config)
    exact = all(
        torch.equal(candidate.core.state_dict()[name], value)
        for name, value in baseline.state_dict().items()
    )
    if not exact:
        raise RuntimeError("candidate and baseline Spin-Delta cores must match")
    return {
        "all_core_tensors_bitwise_equal": exact,
        "candidate_extra_parameters": parameter_count(candidate.router),
        "initial_logit_equality_required": False,
        "reason": "the learned causal router is the architectural intervention",
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
    config = RouterGateConfig(
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
        ROOT / "spin_delta_router.py",
        ROOT / "spin_delta_capability_gate.py",
        ROOT / "model.py",
        ROOT / "spin_delta_scan.py",
        ROOT / "raw_cuda.py",
        ROOT / "csrc" / "spin_scan.cpp",
        ROOT / "csrc" / "spin_scan_cuda.cu",
    )
    report = {
        "schema_version": 1,
        "stage": "spin_delta_causal_router_gate",
        "claim_scope": (
            "autonomous evaluation with train-time synthetic grammar labels; "
            "not a natural-language or parameter-matched superiority claim"
        ),
        "protocol": "SPIN_DELTA_CAUSAL_ROUTER_PREREGISTRATION.md",
        "config": asdict(config),
        "variant_order": list(VARIANTS),
        "pairing": pairing,
        "intervention": {
            "router": "token embedding, causal width-three convolution, hard-ST heads",
            "training_labels": "causal synthetic event and slot labels only",
            "evaluation_inputs": "token IDs only",
            "unchanged": [
                "Spin-Delta two-slot state",
                "independent Spin transports",
                "value drive",
                "raw CUDA recurrence and backward",
                "readout and language head",
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
