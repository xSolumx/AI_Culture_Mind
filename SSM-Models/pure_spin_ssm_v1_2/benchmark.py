"""Matched Shakespeare byte-LM training for Pure Spin v1.2 and fused Mamba-2."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data import (
    TINY_SHAKESPEARE_REVISION,
    TINY_SHAKESPEARE_SHA256,
    TINY_SHAKESPEARE_URL,
    random_batch,
    tiny_shakespeare_bytes,
    wikitext_bytes,
)
from mamba2_baseline import OfficialMamba2LM, fused_mamba2_available
from model import PureSpinSSMV12, PureSpinV12Config, parameter_count


@dataclass(frozen=True)
class BenchmarkConfig:
    steps: int = 300
    batch_size: int = 8
    sequence_length: int = 256
    validation_batches: int = 16
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    d_model: int = 128
    layers: int = 4
    spin_channels: int = 2
    spin_backend: str = "raw_cuda_hybrid"
    spin_mixer: str = "swiglu"
    spin_readout: str = "direction"
    spin_multiplicity_router: str = "none"
    spin_recurrence: str = "independent"
    spin_recurrent_multiplicity: str = "identity"
    spin_recurrent_coupling_scale: str = "unit"
    spin_retention_mode: str = "shared"
    spin_expansion: int = 2
    spin_group_schedule: tuple[int, ...] | None = None
    spin_chunk_size: int = 32
    mamba_d_model: int = 144
    mamba_d_state: int = 64
    maximum_parameter_gap: float = 0.05
    seed: int = 17


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_version(*distribution_names: str) -> str:
    for name in distribution_names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return "unavailable"


def build_model(name: str, config: BenchmarkConfig) -> torch.nn.Module:
    if name == "pure_spin_v1_2":
        return PureSpinSSMV12(
            PureSpinV12Config(
                d_model=config.d_model,
                num_layers=config.layers,
                spin_channels=config.spin_channels,
                mixer=config.spin_mixer,
                readout=config.spin_readout,
                multiplicity_router=config.spin_multiplicity_router,
                recurrence=config.spin_recurrence,
                recurrent_multiplicity=config.spin_recurrent_multiplicity,
                recurrent_coupling_scale=config.spin_recurrent_coupling_scale,
                retention_mode=config.spin_retention_mode,
                expansion=config.spin_expansion,
                group_schedule=config.spin_group_schedule,
                scan_chunk_size=config.spin_chunk_size,
            )
        )
    if name == "mamba2_fused":
        return OfficialMamba2LM(
            vocab_size=256,
            d_model=config.mamba_d_model,
            num_layers=config.layers,
            d_state=config.mamba_d_state,
            headdim=32,
        )
    raise ValueError(f"unknown model {name!r}")


def parameter_match(config: BenchmarkConfig) -> dict[str, object]:
    """Construct both candidates and fail unless trainable counts are matched."""
    counts = {
        name: parameter_count(build_model(name, config))
        for name in ("pure_spin_v1_2", "mamba2_fused")
    }
    gap = abs(counts["pure_spin_v1_2"] - counts["mamba2_fused"]) / max(counts.values())
    if gap > config.maximum_parameter_gap:
        raise RuntimeError(
            f"parameter gap {gap:.3%} exceeds {config.maximum_parameter_gap:.3%}: {counts}"
        )
    return {"counts": counts, "relative_gap": gap, "denominator": "larger model"}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    name: str,
    spin_backend: str,
) -> float:
    model.eval()
    loss_sum = 0.0
    token_count = 0
    for inputs, targets in batches:
        inputs, targets = inputs.to(device), targets.to(device)
        kwargs = {"scan_mode": spin_backend} if name == "pure_spin_v1_2" else {}
        logits = model(inputs, **kwargs)["logits"]
        loss_sum += float(
            F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="sum")
        )
        token_count += targets.numel()
    return loss_sum / token_count


def run_one(
    name: str,
    config: BenchmarkConfig,
    train_stream: torch.Tensor,
    validation: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, object]:
    seed_all(config.seed)
    model = build_model(name, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed)
    initial_loss = evaluate(model, validation, device, name, config.spin_backend)
    warmup_generator = torch.Generator().manual_seed(config.seed + 2)
    warmup_inputs, warmup_targets = random_batch(
        train_stream,
        batch_size=config.batch_size,
        sequence_length=config.sequence_length,
        generator=warmup_generator,
    )
    warmup_inputs, warmup_targets = warmup_inputs.to(device), warmup_targets.to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    warmup_kwargs = (
        {"scan_mode": config.spin_backend} if name == "pure_spin_v1_2" else {}
    )
    warmup_logits = model(warmup_inputs, **warmup_kwargs)["logits"]
    F.cross_entropy(warmup_logits.flatten(0, 1), warmup_targets.flatten()).backward()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    sampled_losses = {}
    model.train()
    for step in range(1, config.steps + 1):
        inputs, targets = random_batch(
            train_stream,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            generator=generator,
        )
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        kwargs = {"scan_mode": config.spin_backend} if name == "pure_spin_v1_2" else {}
        logits = model(inputs, **kwargs)["logits"]
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
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
    final_loss = evaluate(model, validation, device, name, config.spin_backend)
    tokens = config.steps * config.batch_size * config.sequence_length
    return {
        "name": name,
        "parameters": parameter_count(model),
        "initial_nats_per_byte": initial_loss,
        "final_nats_per_byte": final_loss,
        "final_bits_per_byte": final_loss / math.log(2),
        "sampled_training_losses": sampled_losses,
        "training_seconds": seconds,
        "training_tokens_per_second": tokens / seconds,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--spin-backend",
        choices=[
            "compiled_controller",
            "compiled_factorized",
            "raw_cuda_controller",
            "raw_cuda_factorized",
            "raw_cuda_isotypic",
            "raw_cuda_hybrid",
            "raw_cuda_coupled",
            "raw_cuda_block",
            "chunk_parallel",
            "delta_recurrent",
            "delta_parallel",
        ],
        default="raw_cuda_hybrid",
    )
    parser.add_argument(
        "--spin-mixer",
        choices=["swiglu", "sol_bounded_quadratic", "sol_self_gate"],
        default="swiglu",
    )
    parser.add_argument(
        "--spin-readout",
        choices=["direction", "triality_invariants"],
        default="direction",
    )
    parser.add_argument(
        "--spin-multiplicity-router",
        choices=["none", "orthogonal_query"],
        default="none",
    )
    parser.add_argument(
        "--spin-recurrence",
        choices=[
            "independent",
            "coupled_isotypic",
            "independent_block",
            "spin_delta",
        ],
        default="independent",
    )
    parser.add_argument(
        "--spin-recurrent-multiplicity",
        choices=["identity", "orthogonal"],
        default="identity",
    )
    parser.add_argument(
        "--spin-recurrent-coupling-scale",
        choices=["unit", "retention_step"],
        default="unit",
    )
    parser.add_argument(
        "--spin-retention-mode",
        choices=["shared", "isotypic", "isotypic_spectrum"],
        default="shared",
    )
    parser.add_argument("--spin-expansion", type=int, default=2)
    parser.add_argument("--spin-d-model", type=int, default=128)
    parser.add_argument("--spin-group-schedule", type=int, nargs="+")
    parser.add_argument("--spin-chunk-size", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument(
        "--models", nargs="+", default=["pure_spin_v1_2", "mamba2_fused"]
    )
    parser.add_argument(
        "--dataset",
        choices=["tiny_shakespeare", "wikitext2_legacy"],
        default="tiny_shakespeare",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/shakespeare_byte_300.json")
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the fused comparison requires CUDA")
    if "mamba2_fused" in args.models:
        available, detail = fused_mamba2_available()
        if not available:
            raise RuntimeError(f"official fused Mamba-2 unavailable: {detail}")
    config = BenchmarkConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        validation_batches=args.validation_batches,
        spin_backend=args.spin_backend,
        spin_mixer=args.spin_mixer,
        spin_readout=args.spin_readout,
        spin_multiplicity_router=args.spin_multiplicity_router,
        spin_recurrence=args.spin_recurrence,
        spin_recurrent_multiplicity=args.spin_recurrent_multiplicity,
        spin_recurrent_coupling_scale=args.spin_recurrent_coupling_scale,
        spin_retention_mode=args.spin_retention_mode,
        spin_expansion=args.spin_expansion,
        d_model=args.spin_d_model,
        spin_group_schedule=(
            tuple(args.spin_group_schedule)
            if args.spin_group_schedule is not None
            else None
        ),
        spin_chunk_size=args.spin_chunk_size,
        layers=args.layers,
        seed=args.seed,
    )
    if args.dataset == "tiny_shakespeare":
        train, train_sha = tiny_shakespeare_bytes("train", offline=args.offline)
        valid, valid_sha = tiny_shakespeare_bytes("validation", offline=args.offline)
        dataset = {
            "name": "tiny_shakespeare",
            "encoding": "raw UTF-8 bytes",
            "split": "90/5/5 contiguous bytes",
            "source": {
                "url": TINY_SHAKESPEARE_URL,
                "revision": TINY_SHAKESPEARE_REVISION,
                "full_sha256": TINY_SHAKESPEARE_SHA256,
            },
            "train_sha256": train_sha,
            "validation_sha256": valid_sha,
        }
    else:
        train, train_sha = wikitext_bytes("train", offline=args.offline)
        valid, valid_sha = wikitext_bytes("validation", offline=args.offline)
        dataset = {
            "name": "Salesforce/wikitext wikitext-2-raw-v1",
            "encoding": "raw UTF-8 bytes",
            "status": "legacy replay only",
            "train_sha256": train_sha,
            "validation_sha256": valid_sha,
        }
    train_stream = torch.as_tensor(train, dtype=torch.long)
    valid_stream = torch.as_tensor(valid, dtype=torch.long)
    generator = torch.Generator().manual_seed(config.seed + 1)
    validation = [
        random_batch(
            valid_stream,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            generator=generator,
        )
        for _ in range(config.validation_batches)
    ]
    device = torch.device("cuda")
    report = {
        "schema_version": 1,
        "claim_scope": (
            "natural-data training run; matched only when parameter_match is "
            "present; empirical, not a general superiority claim"
        ),
        "config": asdict(config),
        "dataset": dataset,
        "parameter_match": parameter_match(config)
        if set(args.models) == {"pure_spin_v1_2", "mamba2_fused"}
        else None,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "total_cuda_bytes": torch.cuda.get_device_properties(0).total_memory,
            "triton": package_version("triton", "triton-windows"),
            "causal_conv1d": package_version("causal-conv1d"),
            "mamba2_fused": fused_mamba2_available(),
        },
        "implementation_sha256": {
            path.relative_to(Path(__file__).parent).as_posix(): file_sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("model.py"),
                Path(__file__).with_name("mamba2_baseline.py"),
                Path(__file__).with_name("data.py"),
                Path(__file__).with_name("raw_cuda.py"),
                Path(__file__).with_name("chunk_parallel_scan.py"),
                Path(__file__).with_name("spin_delta_scan.py"),
                Path(__file__).with_name("csrc") / "spin_scan.cpp",
                Path(__file__).with_name("csrc") / "spin_scan_cuda.cu",
            )
        },
        "results": [
            run_one(name, config, train_stream, validation, device)
            for name in args.models
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
