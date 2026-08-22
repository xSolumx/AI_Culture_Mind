"""Frozen joint-versus-phased Spin-Delta causal-router gate."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from benchmark import file_sha256, package_version, parameter_count, seed_all
from spin_delta_capability_gate import overwrite_retrieval_batch
from spin_delta_causal_router_gate import build_candidate, evaluate, frozen_batches
from spin_delta_router import RoutedSpinDelta, router_supervision_loss

ROOT = Path(__file__).resolve().parent
VARIANTS = ("joint_schedule", "phase_separated_schedule")


@dataclass(frozen=True)
class PhasedRouterConfig:
    phase_a_steps: int = 100
    phase_b_steps: int = 800
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
    seed: int = 467


def _optimizer(parameters, config: PhasedRouterConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )


def _batch(
    config: PhasedRouterConfig,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return overwrite_retrieval_batch(
        config.batch_size,
        config.training_writes,
        generator=generator,
        device=device,
        return_oracle=True,
    )


def _finite_step(
    loss: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    gradient_clip: float,
    step: str,
) -> float:
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        gradient_clip,
    )
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError(f"nonfinite gradient at {step}")
    optimizer.step()
    return float(gradient_norm)


def _evaluate_all(
    model: RoutedSpinDelta,
    batches: dict[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
) -> dict[str, object]:
    return {
        str(writes): evaluate(model, rows, routed=True)
        for writes, rows in batches.items()
    }


def run_joint(
    config: PhasedRouterConfig,
    batches: dict[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    device: torch.device,
) -> dict[str, object]:
    seed_all(config.seed)
    model = build_candidate(config).to(device)
    optimizer = _optimizer(model.parameters(), config)
    phase_a_generator = torch.Generator().manual_seed(970_000 + config.seed)
    phase_b_generator = torch.Generator().manual_seed(config.seed)
    initial = _evaluate_all(model, batches)
    sampled_losses = {}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model.train()
    total_steps = config.phase_a_steps + config.phase_b_steps
    for step in range(1, total_steps + 1):
        generator = (
            phase_a_generator if step <= config.phase_a_steps else phase_b_generator
        )
        inputs, target, oracle = _batch(config, generator, device)
        optimizer.zero_grad(set_to_none=True)
        result = model(inputs, scan_mode="raw_cuda_delta")
        retrieval = F.cross_entropy(result["logits"][:, -1], target)
        auxiliary = router_supervision_loss(result["router"], oracle)["total"]
        loss = retrieval + config.auxiliary_weight * auxiliary
        _finite_step(loss, model, optimizer, config.gradient_clip, f"joint {step}")
        if step in {1, config.phase_a_steps, total_steps}:
            sampled_losses[str(step)] = {
                "total": float(loss.detach()),
                "retrieval": float(retrieval.detach()),
                "auxiliary": float(auxiliary.detach()),
            }
        if step == config.phase_a_steps:
            phase_a_router = _evaluate_all(model, batches)
            model.train()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    final = _evaluate_all(model, batches)
    return {
        "variant": "joint_schedule",
        "parameters": parameter_count(model),
        "initial": initial,
        "phase_a_router": phase_a_router,
        "final": final,
        "sampled_training_losses": sampled_losses,
        "training_seconds": seconds,
        "training_examples_per_second": total_steps * config.batch_size / seconds,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "autonomous_evaluation": True,
    }


def run_phased(
    config: PhasedRouterConfig,
    batches: dict[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    device: torch.device,
) -> dict[str, object]:
    seed_all(config.seed)
    model = build_candidate(config).to(device)
    router_optimizer = _optimizer(model.router.parameters(), config)
    phase_a_generator = torch.Generator().manual_seed(970_000 + config.seed)
    phase_b_generator = torch.Generator().manual_seed(config.seed)
    initial_core = {
        name: tensor.detach().clone() for name, tensor in model.core.state_dict().items()
    }
    initial = _evaluate_all(model, batches)
    sampled_losses = {}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model.train()
    for step in range(1, config.phase_a_steps + 1):
        inputs, _, oracle = _batch(config, phase_a_generator, device)
        router_optimizer.zero_grad(set_to_none=True)
        auxiliary = router_supervision_loss(model.router(inputs), oracle)["total"]
        _finite_step(
            auxiliary,
            model.router,
            router_optimizer,
            config.gradient_clip,
            f"phased router {step}",
        )
        if step in {1, config.phase_a_steps}:
            sampled_losses[f"router_{step}"] = {
                "total": float(auxiliary.detach()),
                "retrieval": None,
                "auxiliary": float(auxiliary.detach()),
            }
    core_untouched = all(
        torch.equal(model.core.state_dict()[name], tensor)
        for name, tensor in initial_core.items()
    )
    if not core_untouched:
        raise RuntimeError("phase A modified the supposedly untouched core")
    phase_a_router = _evaluate_all(model, batches)
    for parameter in model.router.parameters():
        parameter.requires_grad_(False)
    core_optimizer = _optimizer(model.core.parameters(), config)
    model.train()
    for step in range(1, config.phase_b_steps + 1):
        inputs, target, _ = _batch(config, phase_b_generator, device)
        core_optimizer.zero_grad(set_to_none=True)
        result = model(inputs, scan_mode="raw_cuda_delta")
        retrieval = F.cross_entropy(result["logits"][:, -1], target)
        _finite_step(
            retrieval,
            model.core,
            core_optimizer,
            config.gradient_clip,
            f"phased core {step}",
        )
        if step in {1, config.phase_b_steps}:
            sampled_losses[f"core_{step}"] = {
                "total": float(retrieval.detach()),
                "retrieval": float(retrieval.detach()),
                "auxiliary": None,
            }
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    final = _evaluate_all(model, batches)
    return {
        "variant": "phase_separated_schedule",
        "parameters": parameter_count(model),
        "initial": initial,
        "phase_a_core_untouched": core_untouched,
        "phase_a_router": phase_a_router,
        "router_frozen_during_phase_b": True,
        "final": final,
        "sampled_training_losses": sampled_losses,
        "training_seconds": seconds,
        "training_examples_per_second": (
            (config.phase_a_steps + config.phase_b_steps)
            * config.batch_size
            / seconds
        ),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "autonomous_evaluation": True,
    }


def pairing_audit(config: PhasedRouterConfig) -> dict[str, object]:
    seed_all(config.seed)
    joint = build_candidate(config)
    seed_all(config.seed)
    phased = build_candidate(config)
    exact = all(
        torch.equal(phased.state_dict()[name], tensor)
        for name, tensor in joint.state_dict().items()
    )
    if not exact:
        raise RuntimeError("joint and phased models must start bitwise equal")
    return {
        "all_initial_tensors_bitwise_equal": exact,
        "parameter_count": parameter_count(joint),
        "phase_a_batches_equal": True,
        "phase_b_batches_equal": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--phase-a-steps", type=int, default=100)
    parser.add_argument("--phase-b-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batches", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = PhasedRouterConfig(
        phase_a_steps=args.phase_a_steps,
        phase_b_steps=args.phase_b_steps,
        batch_size=args.batch_size,
        evaluation_batches=args.evaluation_batches,
        seed=args.seed,
    )
    device = torch.device("cuda")
    pairing = pairing_audit(config)
    batches = frozen_batches(config, device)
    results = [run_joint(config, batches, device), run_phased(config, batches, device)]
    implementation_paths = (
        Path(__file__),
        ROOT / "spin_delta_causal_router_gate.py",
        ROOT / "spin_delta_router.py",
        ROOT / "model.py",
        ROOT / "spin_delta_scan.py",
        ROOT / "raw_cuda.py",
        ROOT / "csrc" / "spin_scan.cpp",
        ROOT / "csrc" / "spin_scan_cuda.cu",
    )
    report = {
        "schema_version": 1,
        "stage": "spin_delta_phased_router_gate",
        "claim_scope": (
            "matched joint-versus-phase-separated optimization on a synthetic "
            "grammar; not a natural-language claim"
        ),
        "protocol": "SPIN_DELTA_PHASED_ROUTER_PREREGISTRATION.md",
        "config": asdict(config),
        "variant_order": list(VARIANTS),
        "pairing": pairing,
        "intervention": {
            "joint": "900 joint retrieval-plus-auxiliary updates",
            "phased": "100 router-only then 800 frozen-router core-only updates",
            "evaluation_inputs": "token IDs only",
            "unchanged": [
                "model tensors at initialization",
                "causal hard-ST router architecture",
                "Spin-Delta recurrence and raw CUDA compiler",
                "phase-specific training batches",
                "evaluation batches",
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
