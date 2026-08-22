"""Run one initialization cohort of the label-free Spin-Delta curriculum."""

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
from spin_delta_write_curriculum_gate import state_digest

ROOT = Path(__file__).resolve().parent
INIT_SEEDS = (653, 659, 661)
DATA_SEEDS = (673, 677, 683)
ARMS = ("fixed", "curriculum")
READINESS_WRITES = (2, 3, 5, 8, 16, 32)
SCORED_WRITES = (8, 16, 32)
SCHEDULES = {
    "fixed": ((8, 800),),
    "curriculum": ((2, 100), (3, 100), (5, 200), (8, 400)),
}
PROTOCOL = "SPIN_DELTA_LABEL_FREE_CURRICULUM_PREREGISTRATION.md"


@dataclass(frozen=True)
class LabelFreeConfig:
    init_seed: int
    data_seed: int
    steps: int = 800
    batch_size: int = 128
    evaluation_writes: tuple[int, ...] = READINESS_WRITES
    evaluation_batches: int = 16
    learning_rate: float = 3.0e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    d_model: int = 64
    layers: int = 2
    router_width: int = 32
    router_kernel_size: int = 3
    router_temperature: float = 1.0


def evaluation_rows(config: LabelFreeConfig, device: torch.device):
    rows = {}
    for writes in config.evaluation_writes:
        generator = torch.Generator().manual_seed(1_210_001 + 53 * writes)
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


def clone_model(source, config: LabelFreeConfig, device: torch.device):
    seed_all(config.init_seed)
    model = build_candidate(config).to(device)
    model.load_state_dict(source.state_dict())
    return model


def train_arm(arm, source, config: LabelFreeConfig, rows, device: torch.device):
    model = clone_model(source, config, device)
    initial_digest = state_digest(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.data_seed)
    schedule = SCHEDULES[arm]
    sampled_losses = {}
    stage_router_metrics = {}
    global_step = 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model.train()
    for writes, stage_steps in schedule:
        counts: dict[str, int] = {}
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
            _merge_counts(counts, _router_counts(result["router"], oracle))
            retrieval = F.cross_entropy(result["logits"][:, -1], target)
            retrieval.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            if not torch.isfinite(norm):
                raise FloatingPointError(f"nonfinite gradient at {arm} {global_step}")
            optimizer.step()
            if local_step == stage_steps:
                sampled_losses[str(global_step)] = {
                    "writes": writes,
                    "retrieval": float(retrieval.detach()),
                }
        stage_router_metrics[str(global_step)] = {
            "writes": writes,
            "metrics": _router_metrics(counts),
        }
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    readiness = {
        str(writes): evaluate(model, batches, routed=True)
        for writes, batches in rows.items()
    }
    return {
        "arm": arm,
        "training_schedule": [
            {"writes": writes, "steps": stage_steps} for writes, stage_steps in schedule
        ],
        "initial_state_sha256": initial_digest,
        "training_uses_router_labels": False,
        "router_auxiliary_loss_weight": 0.0,
        "oracle_controls_supplied_to_model": False,
        "audit_labels_detached_from_loss": True,
        "final": {str(writes): readiness[str(writes)] for writes in SCORED_WRITES},
        "router_readiness": readiness,
        "stage_training_router_metrics": stage_router_metrics,
        "sampled_training_losses": sampled_losses,
        "training_examples": config.steps * config.batch_size,
        "training_tokens": sum(
            stage_steps * config.batch_size * (3 * writes + 2)
            for writes, stage_steps in schedule
        ),
        "training_seconds": seconds,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-seed", type=int, choices=INIT_SEEDS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    base_config = LabelFreeConfig(args.init_seed, DATA_SEEDS[0])
    seed_all(args.init_seed)
    source = build_candidate(base_config).to(device)
    source_digest = state_digest(source)
    execution_id = f"init-{args.init_seed}-state-{source_digest}"
    rows = evaluation_rows(base_config, device)
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
    implementation = {
        path.relative_to(ROOT).as_posix(): file_sha256(path) for path in paths
    }
    environment = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "triton": package_version("triton", "triton-windows"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for data_seed in DATA_SEEDS:
        config = LabelFreeConfig(args.init_seed, data_seed)
        arms = [train_arm(arm, source, config, rows, device) for arm in ARMS]
        if any(arm["initial_state_sha256"] != source_digest for arm in arms):
            raise RuntimeError("joint arms did not begin from the shared state")
        report = {
            "schema_version": 1,
            "stage": "spin_delta_label_free_curriculum",
            "protocol": PROTOCOL,
            "config": asdict(config),
            "claim_scope": "retrieval-only joint router/core synthetic learning",
            "parameters": parameter_count(source),
            "source_state_sha256": source_digest,
            "arms": arms,
            "contract": {
                "training_uses_router_labels": False,
                "router_auxiliary_loss_weight": 0.0,
                "oracle_controls_supplied_to_model": False,
                "audit_labels_detached_from_loss": True,
                "router_and_core_jointly_trainable": True,
            },
            "cohort_execution": {
                "shared_initial_single_execution": True,
                "execution_id": execution_id,
                "init_seed": args.init_seed,
                "data_seeds": list(DATA_SEEDS),
            },
            "environment": environment,
            "implementation_sha256": implementation,
        }
        output = args.output_dir / f"i{args.init_seed}_d{data_seed}.json"
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
