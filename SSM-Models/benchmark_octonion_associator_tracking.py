"""Frozen continuous associator-tracking benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from delta_product_reference import DeltaProductReferenceLayer
from pure_rotor_ssm.octonion_operator_scan import (
    OCTONION_DIM,
    octonion_left_multiplication_matrix,
    octonion_operator_prefix_scan,
    octonion_product,
    unit_octonion,
)
from torch import nn
from torch.nn import functional as F
from transformers import Mamba2Config, Mamba2ForCausalLM

PROTOCOL_FROZEN_AT = "2026-08-16T21:02:38+02:00"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "experiments"
    / "artifacts"
    / "octonion_associator_tracking_pilot300.json"
)
DEFAULT_CHECKPOINT_DIRECTORY = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "octonion_associator_tracking_pilot300"
)


@dataclass(frozen=True)
class AssociatorConfig:
    seed: int = 0
    steps: int = 300
    batch_size: int = 32
    train_length: int = 16
    evaluation_batch_size: int = 128
    evaluation_lengths: tuple[int, ...] = (16, 64, 128)
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    operator_noise_std: float = 0.05
    mamba_hidden_size: int = 32
    mamba_state_size: int = 16
    mamba_heads: int = 4
    mamba_head_dim: int = 16
    delta_hidden_size: int = 32
    delta_heads: int = 4
    delta_householder: int = 4


def now() -> str:
    return datetime.now().astimezone().isoformat()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_hash(tensors: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(contiguous.shape)).encode())
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def exact_targets(inputs: torch.Tensor) -> torch.Tensor:
    prefixes, _ = octonion_operator_prefix_scan(inputs, mode="work_efficient")
    return prefixes[:, :, 0].flatten(start_dim=-2)


def make_schedules(
    config: AssociatorConfig,
) -> tuple[
    list[tuple[torch.Tensor, torch.Tensor]],
    dict[int, tuple[torch.Tensor, torch.Tensor]],
    dict[str, str],
]:
    generator = torch.Generator().manual_seed(config.seed + 1000)
    training = []
    training_inputs = []
    for _ in range(config.steps):
        inputs = unit_octonion(
            torch.randn(
                config.batch_size,
                config.train_length,
                1,
                OCTONION_DIM,
                generator=generator,
            )
        )
        targets = exact_targets(inputs)
        training.append((inputs[:, :, 0], targets))
        training_inputs.append(inputs)
    evaluations = {}
    evaluation_inputs = []
    for length in config.evaluation_lengths:
        inputs = unit_octonion(
            torch.randn(
                config.evaluation_batch_size,
                length,
                1,
                OCTONION_DIM,
                generator=generator,
            )
        )
        evaluations[length] = (inputs[:, :, 0], exact_targets(inputs))
        evaluation_inputs.append(inputs)
    return (
        training,
        evaluations,
        {
            "training_schedule_sha256": tensor_hash(training_inputs),
            "evaluation_schedule_sha256": tensor_hash(evaluation_inputs),
        },
    )


class LearnedOctonionOperatorTracker(nn.Module):
    def __init__(self, noise_std: float) -> None:
        super().__init__()
        self.encoder = nn.Linear(OCTONION_DIM, OCTONION_DIM)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.eye(OCTONION_DIM))
            self.encoder.weight.add_(noise_std * torch.randn_like(self.encoder.weight))
            self.encoder.bias.zero_()

    @property
    def recurrent_state_scalars(self) -> int:
        return OCTONION_DIM**2

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(inputs)[:, :, None]
        prefixes, _ = octonion_operator_prefix_scan(encoded, mode="work_efficient")
        return prefixes[:, :, 0].flatten(start_dim=-2)


class ContinuousMamba2Tracker(nn.Module):
    def __init__(self, config: AssociatorConfig) -> None:
        super().__init__()
        self.input_projection = nn.Linear(OCTONION_DIM, config.mamba_hidden_size)
        self.backbone = Mamba2ForCausalLM(
            Mamba2Config(
                vocab_size=OCTONION_DIM**2,
                hidden_size=config.mamba_hidden_size,
                state_size=config.mamba_state_size,
                num_hidden_layers=1,
                num_heads=config.mamba_heads,
                head_dim=config.mamba_head_dim,
                expand=2,
                conv_kernel=4,
                n_groups=1,
                tie_word_embeddings=False,
                use_cache=False,
            )
        )
        self.recurrent_state_scalars = (
            config.mamba_heads * config.mamba_head_dim * config.mamba_state_size
            + (2 * config.mamba_hidden_size + 2 * config.mamba_state_size) * 4
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(inputs)
        return self.backbone(inputs_embeds=hidden, use_cache=False).logits


class ContinuousDeltaProductTracker(nn.Module):
    def __init__(self, config: AssociatorConfig) -> None:
        super().__init__()
        self.input_projection = nn.Linear(OCTONION_DIM, config.delta_hidden_size)
        self.norm = nn.RMSNorm(config.delta_hidden_size)
        self.delta = DeltaProductReferenceLayer(
            hidden_size=config.delta_hidden_size,
            num_heads=config.delta_heads,
            num_householder=config.delta_householder,
        )
        self.output = nn.Linear(config.delta_hidden_size, OCTONION_DIM**2)

    @property
    def recurrent_state_scalars(self) -> int:
        return self.delta.num_heads * self.delta.head_dim**2

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(inputs)
        update, _ = self.delta(self.norm(hidden), scan_mode="parallel")
        return self.output(hidden + update)


def collapsed_octonion_predictions(inputs: torch.Tensor) -> torch.Tensor:
    prefix = torch.zeros_like(inputs[:, 0])
    prefix[..., 0] = 1
    rows = []
    for position in range(inputs.shape[1]):
        prefix = octonion_product(inputs[:, position], prefix)
        rows.append(octonion_left_multiplication_matrix(prefix).flatten(start_dim=-2))
    return torch.stack(rows, dim=1)


def metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    error = (predictions - targets).reshape(*targets.shape[:2], 8, 8)
    target = targets.reshape(*targets.shape[:2], 8, 8)
    return {
        "mse": float(F.mse_loss(predictions, targets)),
        "mean_relative_frobenius_error": float(
            (
                torch.linalg.matrix_norm(error)
                / torch.linalg.matrix_norm(target).clamp_min(1e-12)
            ).mean()
        ),
        "maximum_absolute_error": float(error.abs().max()),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module | None,
    candidate: str,
    evaluations: dict[int, tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    if model is not None:
        model.eval()
    result = {}
    for length, (inputs, targets) in evaluations.items():
        inputs, targets = inputs.to(device), targets.to(device)
        if candidate == "exact_operator_oracle":
            predictions = exact_targets(inputs[:, :, None])
        elif candidate == "collapsed_octonion_ablation":
            predictions = collapsed_octonion_predictions(inputs)
        else:
            assert model is not None
            predictions = model(inputs)
        result[str(length)] = metrics(predictions, targets)
    return result


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_candidate(
    name: str,
    model: nn.Module,
    training: list[tuple[torch.Tensor, torch.Tensor]],
    evaluations: dict[int, tuple[torch.Tensor, torch.Tensor]],
    config: AssociatorConfig,
    device: torch.device,
    checkpoint_directory: Path,
) -> dict[str, Any]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    started = time.perf_counter()
    loss_samples = {}
    model.train()
    for step, (inputs, targets) in enumerate(training, start=1):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = F.mse_loss(predictions, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name} produced nonfinite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in (1, 50, 100, 200, config.steps):
            loss_samples[str(step)] = float(loss.detach())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    evaluation = evaluate(model, name, evaluations, device)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_directory / f"{name}_seed{config.seed}.pt"
    torch.save(
        {
            "format_version": 1,
            "candidate": name,
            "config": asdict(config),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "evaluation": evaluation,
        },
        checkpoint,
    )
    return {
        "parameters": parameter_count(model),
        "recurrent_state_scalars": int(model.recurrent_state_scalars),
        "loss_samples": loss_samples,
        "training_wall_seconds": elapsed,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "evaluation": evaluation,
        "checkpoint": str(
            checkpoint.resolve().relative_to(Path(__file__).resolve().parent)
        ),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = AssociatorConfig(seed=args.seed, steps=args.steps)
    seed_everything(config.seed)
    device = torch.device(args.device)
    training, evaluations, schedules = make_schedules(config)
    results = {
        "exact_operator_oracle": {
            "parameters": 0,
            "recurrent_state_scalars": 64,
            "evaluation": evaluate(None, "exact_operator_oracle", evaluations, device),
        },
        "collapsed_octonion_ablation": {
            "parameters": 0,
            "recurrent_state_scalars": 8,
            "evaluation": evaluate(
                None, "collapsed_octonion_ablation", evaluations, device
            ),
        },
    }
    factories = {
        "learned_octonion_operator": lambda: LearnedOctonionOperatorTracker(
            config.operator_noise_std
        ),
        "transformers_mamba2": lambda: ContinuousMamba2Tracker(config),
        "delta_product_reference": lambda: ContinuousDeltaProductTracker(config),
    }
    for offset, (name, factory) in enumerate(factories.items()):
        seed_everything(config.seed + offset)
        results[name] = train_candidate(
            name,
            factory(),
            training,
            evaluations,
            config,
            device,
            args.checkpoint_directory,
        )
    checks = {
        "oracle_is_exact": max(
            row["mse"]
            for row in results["exact_operator_oracle"]["evaluation"].values()
        )
        < 1e-12,
        "collapsed_ablation_fails_l128": results["collapsed_octonion_ablation"][
            "evaluation"
        ]["128"]["mse"]
        > 1e-2,
        "learned_operator_beats_collapsed_l128": results["learned_octonion_operator"][
            "evaluation"
        ]["128"]["mse"]
        < results["collapsed_octonion_ablation"]["evaluation"]["128"]["mse"],
        "all_learned_metrics_finite": all(
            np.isfinite(row["mse"])
            for name in factories
            for row in results[name]["evaluation"].values()
        ),
    }
    return {
        "schema_version": 1,
        "experiment": "continuous octonion associator tracking",
        "protocol_frozen_at": PROTOCOL_FROZEN_AT,
        "started_at": args.started_at,
        "finished_at": now(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor()
            ),
        },
        "config": asdict(config),
        "schedules": schedules,
        "results": results,
        "checks": checks,
        "all_required_checks_passed": all(checks.values()),
        "claim_boundary": (
            "One-seed algebra-matched synthetic realizability/length pilot; "
            "not natural-task, triality-specific, or fused-baseline superiority."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--checkpoint-directory", type=Path, default=DEFAULT_CHECKPOINT_DIRECTORY
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    args.started_at = now()
    return args


def main() -> None:
    args = parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_required_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
