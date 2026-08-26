"""Matched-data natural-text development screen for v1.3 architecture choices."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import (
    TINY_SHAKESPEARE_REVISION,
    TINY_SHAKESPEARE_SHA256,
    TINY_SHAKESPEARE_URL,
    random_batch,
    tiny_shakespeare_bytes,
)
from .model import ExceptionalDeltaConfig, ExceptionalDeltaLM, parameter_count


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 100
    batch_size: int = 4
    sequence_length: int = 64
    validation_batches: int = 8
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    optimizer: str = "adamw"
    d_model: int = 32
    layers: int = 2
    memory_width: int = 4
    update_rank: int = 2
    d_conv: int = 4
    seed: int = 17


VARIANTS = {
    "identity_legacy": {
        "action_algebra": "identity",
        "action_geometry": "direct",
        "identity_fast_path": False,
        "albert_determinant_backend": "jordan",
        "albert_product_backend": "dense",
    },
    "identity_generic": {
        "action_algebra": "identity",
        "action_geometry": "direct",
        "identity_fast_path": False,
    },
    "identity_delta": {"action_algebra": "identity", "action_geometry": "direct"},
    "identity_explicit": {
        "action_algebra": "identity",
        "action_geometry": "direct",
        "albert_determinant_backend": "explicit",
    },
    "identity_safe": {"action_algebra": "identity", "action_geometry": "direct"},
    "identity_matched": {
        "action_algebra": "identity",
        "action_geometry": "direct",
        "d_model_override": 36,
    },
    "g2_safe": {"action_algebra": "g2", "action_geometry": "direct"},
    "spin7_safe": {"action_algebra": "spin7", "action_geometry": "direct"},
    "spin8_safe": {"action_algebra": "spin8", "action_geometry": "direct"},
    "spin9_safe": {"action_algebra": "spin9", "action_geometry": "direct"},
    "f4_safe": {"action_algebra": "f4", "action_geometry": "direct"},
    "f4_matched": {
        "action_algebra": "f4",
        "action_geometry": "direct",
        "d_model_override": 33,
    },
    "e6_safe": {"action_algebra": "e6", "action_geometry": "direct"},
    "mamba2_official": {"baseline": "mamba2_official"},
    "f4_delta": {"action_algebra": "f4", "action_geometry": "direct"},
    "e6_direct_delta": {"action_algebra": "e6", "action_geometry": "direct"},
    "late_e6_delta": {
        "action_geometry": "direct",
        "schedule_mode": "late_e6",
    },
    "early_e6_delta": {
        "action_geometry": "direct",
        "schedule_mode": "early_e6",
    },
    "e6_polar_delta": {"action_algebra": "e6", "action_geometry": "polar"},
    "e6_cartan_delta": {"action_algebra": "e6", "action_geometry": "cartan"},
}


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_model(name: str, config: TrainingConfig) -> torch.nn.Module:
    if name not in VARIANTS:
        raise ValueError(f"unknown v1.3 variant {name!r}")
    variant = dict(VARIANTS[name])
    d_model = int(variant.pop("d_model_override", config.d_model))
    baseline = variant.pop("baseline", None)
    if baseline == "mamba2_official":
        from pure_spin_ssm_v1_2.mamba2_baseline import OfficialMamba2LM

        # At d_model=32 this is a 40,848-parameter official fused Mamba-2,
        # only ten parameters from the 40,858-parameter safe E6 arm.
        return OfficialMamba2LM(
            vocab_size=256,
            d_model=d_model,
            num_layers=config.layers,
            d_state=128,
            expand=2,
            headdim=8,
        )
    schedule_mode = variant.pop("schedule_mode", None)
    if schedule_mode == "late_e6":
        variant["action_schedule"] = ("identity",) * (config.layers - 1) + ("e6",)
    elif schedule_mode == "early_e6":
        variant["action_schedule"] = ("e6",) + ("identity",) * (config.layers - 1)
    return ExceptionalDeltaLM(
        ExceptionalDeltaConfig(
            d_model=d_model,
            num_layers=config.layers,
            memory_width=config.memory_width,
            update_rank=config.update_rank,
            d_conv=config.d_conv,
            channel_mixer="jordan",
            readout_mode="albert_invariants",
            **variant,
        )
    )


def _copy_shared_exceptional_initialization(
    target: ExceptionalDeltaLM, reference: ExceptionalDeltaLM
) -> None:
    """Copy every common tensor and every common controller field exactly."""

    reference_parameters = dict(reference.named_parameters())
    with torch.no_grad():
        for name, parameter in target.named_parameters():
            source = reference_parameters.get(name)
            if source is not None and source.shape == parameter.shape:
                parameter.copy_(source)
        for target_block, reference_block in zip(
            target.blocks, reference.blocks, strict=True
        ):
            target_offset = 0
            reference_offsets: dict[str, tuple[int, int]] = {}
            offset = 0
            for field, width in reference_block._segments.items():
                reference_offsets[field] = (offset, offset + width)
                offset += width
            for field, width in target_block._segments.items():
                if field in reference_offsets:
                    start, stop = reference_offsets[field]
                    if stop - start == width:
                        target_block.controller.weight[
                            target_offset : target_offset + width
                        ].copy_(reference_block.controller.weight[start:stop])
                        target_block.controller.bias[
                            target_offset : target_offset + width
                        ].copy_(reference_block.controller.bias[start:stop])
                target_offset += width


def _forward_logits(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    if isinstance(model, ExceptionalDeltaLM):
        return model(inputs, scan_mode="auto")["logits"]
    output = model(inputs)
    if not isinstance(output, dict) or "logits" not in output:
        raise TypeError("language-model baseline must return a logits dictionary")
    return output["logits"]


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    mark_compile_step: bool = False,
) -> float:
    model.eval()
    loss_sum = 0.0
    token_count = 0
    for inputs, targets in batches:
        if mark_compile_step:
            torch.compiler.cudagraph_mark_step_begin()
        inputs, targets = inputs.to(device), targets.to(device)
        logits = _forward_logits(model, inputs)
        loss_sum += float(
            F.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), reduction="sum"
            )
        )
        token_count += targets.numel()
    return loss_sum / token_count


def _run_one(
    name: str,
    config: TrainingConfig,
    train_stream: torch.Tensor,
    validation: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    execution: str,
    checkpoint_root: Path | None,
) -> dict[str, object]:
    _seed_all(config.seed)
    base_model = _build_model(name, config).to(device)
    if isinstance(base_model, ExceptionalDeltaLM) and name != "identity_safe":
        _seed_all(config.seed)
        reference = _build_model(
            "identity_safe", replace(config, d_model=base_model.config.d_model)
        )
        if not isinstance(reference, ExceptionalDeltaLM):
            raise AssertionError("identity reference must be ExceptionalDeltaLM")
        _copy_shared_exceptional_initialization(base_model, reference.to(device))
        del reference
    if execution == "reduce-overhead" and name != "identity_safe":
        raise RuntimeError(
            "reduce-overhead is qualified only for identity transport on SM75; "
            "exceptional matrix_exp cannot be captured and baselines have their own kernels"
        )
    if name == "mamba2_official" and execution != "eager":
        raise RuntimeError("official fused Mamba-2 must run through its native eager wrapper")
    model = (
        base_model
        if execution == "eager"
        else torch.compile(base_model, fullgraph=True, mode=execution)
    )
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            base_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        optimizer_report: dict[str, object] = {"name": "adamw"}
    elif config.optimizer == "harmonic_muon_adamw":
        from hybrid_memory_v1_4.optimizers import HarmonicMuonAdamW

        optimizer = HarmonicMuonAdamW(
            base_model,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        optimizer_report = {
            "name": "harmonic_muon_adamw",
            "partition": optimizer.partition_report(),
        }
    else:
        raise ValueError(f"unknown optimizer {config.optimizer!r}")
    preparation_start = time.perf_counter()
    mark_compile_step = execution == "reduce-overhead"
    initial_loss = _evaluate(
        model, validation, device, mark_compile_step=mark_compile_step
    )
    generator = torch.Generator().manual_seed(config.seed)

    warmup_generator = torch.Generator().manual_seed(config.seed + 1)
    warmup_inputs, warmup_targets = random_batch(
        train_stream,
        batch_size=config.batch_size,
        sequence_length=config.sequence_length,
        generator=warmup_generator,
    )
    warmup_inputs, warmup_targets = warmup_inputs.to(device), warmup_targets.to(device)
    model.train()
    if mark_compile_step:
        torch.compiler.cudagraph_mark_step_begin()
    warmup_logits = _forward_logits(model, warmup_inputs)
    F.cross_entropy(
        warmup_logits.flatten(0, 1), warmup_targets.flatten()
    ).backward()
    base_model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    preparation_seconds = time.perf_counter() - preparation_start

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    sampled_losses: dict[str, float] = {}
    maximum_gradient_norm = 0.0
    target_digest = hashlib.sha256()
    for step in range(1, config.steps + 1):
        if mark_compile_step:
            torch.compiler.cudagraph_mark_step_begin()
        inputs, targets = random_batch(
            train_stream,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            generator=generator,
        )
        inputs, targets = inputs.to(device), targets.to(device)
        target_digest.update(targets.detach().cpu().numpy().tobytes())
        optimizer.zero_grad(set_to_none=True)
        logits = _forward_logits(model, inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"nonfinite gradient in {name} at step {step}")
        maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm))
        optimizer.step()
        if step in {1, max(1, config.steps // 2), config.steps}:
            sampled_losses[str(step)] = float(loss.detach())
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    final_loss = _evaluate(
        model, validation, device, mark_compile_step=mark_compile_step
    )
    tokens = config.steps * config.batch_size * config.sequence_length
    checkpoint = None
    checkpoint_sha256 = None
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_root / f"{name}_seed{config.seed}.pt"
        torch.save(
            {
                "model": base_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": asdict(config),
                "variant": name,
                "training_target_sha256": target_digest.hexdigest(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                ),
            },
            checkpoint,
        )
        checkpoint_sha256 = _file_sha256(checkpoint)
    return {
        "name": name,
        "parameters": parameter_count(base_model),
        "cache_scalars": getattr(base_model, "cache_scalars", None),
        "execution": execution,
        "optimizer": optimizer_report,
        "preparation_seconds": preparation_seconds,
        "initial_nats_per_byte": initial_loss,
        "final_nats_per_byte": final_loss,
        "final_bits_per_byte": final_loss / math.log(2.0),
        "sampled_training_losses": sampled_losses,
        "maximum_preclip_gradient_norm": maximum_gradient_norm,
        "training_seconds": seconds,
        "training_tokens_per_second": tokens / seconds,
        "training_target_sha256": target_digest.hexdigest(),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "checkpoint_sha256": checkpoint_sha256,
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=["identity_safe", "f4_safe", "e6_safe", "mamba2_official"],
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--memory-width", type=int, default=4)
    parser.add_argument("--update-rank", type=int, default=2)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "harmonic_muon_adamw"),
        default="adamw",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--execution",
        choices=("eager", "default", "reduce-overhead"),
        default="eager",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--require-sm75", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.require_sm75 and (
        args.device != "cuda" or torch.cuda.get_device_capability() != (7, 5)
    ):
        raise RuntimeError("this cohort requires exact CUDA compute capability 7.5")
    config = TrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        validation_batches=args.validation_batches,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        d_model=args.d_model,
        layers=args.layers,
        memory_width=args.memory_width,
        update_rank=args.update_rank,
        d_conv=args.d_conv,
        seed=args.seed,
    )
    train_stream, train_sha = tiny_shakespeare_bytes(
        "train", offline=args.offline, cache_root=args.cache_root
    )
    valid_stream, valid_sha = tiny_shakespeare_bytes(
        "validation", offline=args.offline, cache_root=args.cache_root
    )
    validation_generator = torch.Generator().manual_seed(config.seed + 10_000)
    validation = [
        random_batch(
            valid_stream,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            generator=validation_generator,
        )
        for _ in range(config.validation_batches)
    ]
    device = torch.device(args.device)
    rows = [
        _run_one(
            name,
            config,
            train_stream,
            validation,
            device,
            args.execution,
            args.checkpoint_dir,
        )
        for name in args.variants
    ]
    root = Path(__file__).resolve().parent
    report = {
        "schema_version": 2,
        "experiment": "Pure Exceptional Delta SSM v1.3.1 natural-text development screen",
        "status": "development evidence; interpret through an explicit cohort contract",
        "config": asdict(config),
        "variants": args.variants,
        "execution": args.execution,
        "dataset": {
            "name": "tiny_shakespeare",
            "tokenization": "UTF-8 bytes",
            "train_sha256": train_sha,
            "validation_sha256": valid_sha,
            "source": {
                "url": TINY_SHAKESPEARE_URL,
                "revision": TINY_SHAKESPEARE_REVISION,
                "full_sha256": TINY_SHAKESPEARE_SHA256,
                "split": "90/5/5 contiguous bytes",
            },
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "compute_capability": (
                list(torch.cuda.get_device_capability(device))
                if device.type == "cuda"
                else None
            ),
        },
        "git": {
            "revision": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root.parents[1],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "dirty": bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=root.parents[1],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            ),
        },
        "source_sha256": {
            name: _file_sha256(root / name)
            for name in (
                "action.py",
                "albert.py",
                "benchmark_train.py",
                "data.py",
                "model.py",
                "scan.py",
            )
        },
        "rows": rows,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
