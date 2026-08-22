"""One paired cell of the frozen learned-router curriculum transfer gate."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from benchmark import file_sha256, package_version, parameter_count, seed_all
from spin_delta_capability_gate import overwrite_retrieval_batch
from spin_delta_causal_router_gate import (
    _merge_counts,
    _router_counts,
    _router_metrics,
    build_candidate,
    evaluate,
)
from spin_delta_router import RoutedSpinDelta, router_supervision_loss
from spin_delta_write_curriculum_gate import state_digest

ROOT = Path(__file__).resolve().parent
ARMS = ("fixed", "curriculum")
SCORED_WRITES = (8, 16, 32)
SCHEDULES = {
    "fixed": ((8, 800),),
    "curriculum": ((2, 100), (3, 100), (5, 200), (8, 400)),
}


@dataclass(frozen=True)
class RouterTransferConfig:
    init_seed: int
    data_seed: int
    router_steps: int = 100
    core_steps: int = 800
    batch_size: int = 128
    router_training_writes: int = 8
    evaluation_writes: tuple[int, ...] = (2, 3, 5, 8, 16, 32)
    evaluation_batches: int = 16
    learning_rate: float = 3.0e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    d_model: int = 64
    layers: int = 2
    router_width: int = 32
    router_kernel_size: int = 3
    router_temperature: float = 1.0


def _optimizer(parameters, config: RouterTransferConfig):
    return torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )


def evaluation_rows(config: RouterTransferConfig, device: torch.device):
    rows = {}
    for writes in config.evaluation_writes:
        generator = torch.Generator().manual_seed(1_150_001 + 47 * writes)
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


def _evaluate_all(model: RoutedSpinDelta, rows) -> dict[str, object]:
    return {
        str(writes): evaluate(model, batches, routed=True)
        for writes, batches in rows.items()
    }


def _finite_step(loss, module, optimizer, config, label: str) -> float:
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        (parameter for parameter in module.parameters() if parameter.requires_grad),
        config.gradient_clip,
    )
    if not torch.isfinite(norm):
        raise FloatingPointError(f"nonfinite gradient at {label}")
    optimizer.step()
    return float(norm)


def train_router(
    config: RouterTransferConfig,
    rows,
    device: torch.device,
) -> tuple[RoutedSpinDelta, dict[str, object]]:
    seed_all(config.init_seed)
    model = build_candidate(config).to(device)
    initial_core = {
        name: tensor.detach().clone()
        for name, tensor in model.core.state_dict().items()
    }
    optimizer = _optimizer(model.router.parameters(), config)
    generator = torch.Generator().manual_seed(1_170_000 + config.init_seed)
    sampled_losses = {}
    torch.cuda.synchronize()
    start = time.perf_counter()
    model.train()
    for step in range(1, config.router_steps + 1):
        inputs, _, oracle = overwrite_retrieval_batch(
            config.batch_size,
            config.router_training_writes,
            generator=generator,
            device=device,
            return_oracle=True,
        )
        optimizer.zero_grad(set_to_none=True)
        auxiliary = router_supervision_loss(model.router(inputs), oracle)["total"]
        _finite_step(auxiliary, model.router, optimizer, config, f"router {step}")
        if step in {1, config.router_steps}:
            sampled_losses[str(step)] = float(auxiliary.detach())
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    core_untouched = all(
        torch.equal(model.core.state_dict()[name], tensor)
        for name, tensor in initial_core.items()
    )
    if not core_untouched:
        raise RuntimeError("router pretraining modified the core")
    readiness = _evaluate_all(model, rows)
    return model, {
        "core_untouched": core_untouched,
        "sampled_losses": sampled_losses,
        "training_seconds": seconds,
        "readiness": readiness,
        "post_router_state_sha256": state_digest(model),
        "post_router_core_sha256": state_digest(model.core),
        "post_router_router_sha256": state_digest(model.router),
    }


def clone_frozen_router(
    source: RoutedSpinDelta,
    config: RouterTransferConfig,
    device: torch.device,
) -> RoutedSpinDelta:
    seed_all(config.init_seed)
    model = build_candidate(config).to(device)
    model.load_state_dict(source.state_dict())
    for parameter in model.router.parameters():
        parameter.requires_grad_(False)
    model.router.eval()
    return model


def train_core_arm(
    arm: str,
    source: RoutedSpinDelta,
    config: RouterTransferConfig,
    rows,
    device: torch.device,
) -> dict[str, object]:
    model = clone_frozen_router(source, config, device)
    initial_state_sha256 = state_digest(model)
    optimizer = _optimizer(model.core.parameters(), config)
    generator = torch.Generator().manual_seed(config.data_seed)
    sampled_losses = {}
    router_counts: dict[str, int] = {}
    global_step = 0
    schedule = SCHEDULES[arm]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model.train()
    model.router.eval()
    for writes, stage_steps in schedule:
        for local_step in range(1, stage_steps + 1):
            global_step += 1
            inputs, target, oracle = overwrite_retrieval_batch(
                config.batch_size,
                writes,
                generator=generator,
                device=device,
                return_oracle=True,
            )
            optimizer.zero_grad(set_to_none=True)
            result = model(inputs, scan_mode="raw_cuda_delta")
            _merge_counts(router_counts, _router_counts(result["router"], oracle))
            retrieval = F.cross_entropy(result["logits"][:, -1], target)
            _finite_step(
                retrieval,
                model.core,
                optimizer,
                config,
                f"{arm} core {global_step}",
            )
            if local_step == stage_steps:
                sampled_losses[str(global_step)] = {
                    "writes": writes,
                    "loss": float(retrieval.detach()),
                }
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    final = {
        str(writes): evaluate(model, rows[writes], routed=True)
        for writes in SCORED_WRITES
    }
    examples = config.core_steps * config.batch_size
    tokens = sum(
        stage_steps * config.batch_size * (3 * writes + 2)
        for writes, stage_steps in schedule
    )
    return {
        "arm": arm,
        "training_schedule": [
            {"writes": writes, "steps": stage_steps} for writes, stage_steps in schedule
        ],
        "initial_state_sha256": initial_state_sha256,
        "router_frozen": not any(
            parameter.requires_grad for parameter in model.router.parameters()
        ),
        "training_router_metrics": _router_metrics(router_counts),
        "final": final,
        "sampled_training_losses": sampled_losses,
        "training_examples": examples,
        "training_tokens": tokens,
        "training_seconds": seconds,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = RouterTransferConfig(args.init_seed, args.data_seed)
    if any(
        sum(steps for _, steps in schedule) != config.core_steps
        for schedule in SCHEDULES.values()
    ):
        raise RuntimeError("core schedule violates the frozen step budget")
    device = torch.device("cuda")
    rows = evaluation_rows(config, device)
    shared, router_phase = train_router(config, rows, device)
    arms = [train_core_arm(arm, shared, config, rows, device) for arm in ARMS]
    if arms[0]["initial_state_sha256"] != arms[1]["initial_state_sha256"]:
        raise RuntimeError("core arms did not begin from the same cloned state")
    paths = (
        Path(__file__),
        ROOT / "spin_delta_causal_router_gate.py",
        ROOT / "spin_delta_router.py",
        ROOT / "spin_delta_write_curriculum_gate.py",
        ROOT / "model.py",
        ROOT / "spin_delta_scan.py",
        ROOT / "raw_cuda.py",
        ROOT / "csrc" / "spin_scan.cpp",
        ROOT / "csrc" / "spin_scan_cuda.cu",
    )
    report = {
        "schema_version": 1,
        "stage": "spin_delta_router_curriculum_transfer",
        "protocol": "SPIN_DELTA_ROUTER_CURRICULUM_TRANSFER_PREREGISTRATION.md",
        "config": asdict(config),
        "claim_scope": "autonomous learned-router synthetic transfer",
        "parameters": parameter_count(shared),
        "router_phase": router_phase,
        "arms": arms,
        "autonomous_evaluation": True,
        "oracle_controls_supplied_to_model": False,
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
