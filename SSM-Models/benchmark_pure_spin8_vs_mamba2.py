"""Frozen center-sensitive Pure Spin(8) transport comparison."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
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
from pure_spin8_ssm import __version__ as PURE_SPIN8_VERSION
from pure_spin8_ssm.torch_backend import (
    PureSpin8SSMLayer,
    Spin8AffineTransition,
    apply_spin8_affine,
    spin8_group_actions,
    work_efficient_spin8_scan,
)
from spin8_triality import (
    SPIN8_BIVECTOR_DIM,
    SPIN8_DIM,
    TRIALITY_REPRESENTATIONS,
    torch_triality_generators,
)
from torch import nn
from torch.nn import functional as F
from transformers import Mamba2Config, Mamba2ForCausalLM

PROTOCOL_FROZEN_AT = "2026-08-16T21:54:00+02:00"
CURRICULUM = ((2, 250), (4, 250), (8, 250), (16, 250))
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "experiments"
    / "artifacts"
    / "pure_spin8_vs_mamba2_triality_transport1000.json"
)
DEFAULT_CHECKPOINT_DIRECTORY = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "pure_spin8_vs_mamba2_triality_transport1000"
)
CANDIDATES = (
    "maintained_pure_spin8",
    "transformers_mamba2",
    "delta_product_reference",
)


@dataclass(frozen=True)
class Spin8TransportConfig:
    steps: int = 1_000
    batch_size: int = 32
    evaluation_batch_size: int = 128
    evaluation_lengths: tuple[int, ...] = (16, 64, 128)
    coordinate_std: float = 0.15
    center_probability: float = 0.10
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    mamba_hidden_size: int = 32
    mamba_state_size: int = 16
    mamba_heads: int = 4
    mamba_head_dim: int = 16
    delta_hidden_size: int = 32
    delta_heads: int = 4
    delta_householder: int = 4
    seed: int = 0


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


def teacher_initial_state() -> torch.Tensor:
    generator = torch.Generator().manual_seed(202_608_16)
    initial = torch.randn(
        1, len(TRIALITY_REPRESENTATIONS), SPIN8_DIM, generator=generator
    )
    return F.normalize(initial, dim=-1)


def sample_coordinates(
    batch_size: int,
    length: int,
    config: Spin8TransportConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    coordinates = config.coordinate_std * torch.randn(
        batch_size, length, SPIN8_BIVECTOR_DIM, generator=generator
    )
    central = (
        torch.rand(batch_size, length, generator=generator) < config.center_probability
    )
    signs = torch.where(
        torch.rand(batch_size, length, generator=generator) < 0.5, -1.0, 1.0
    )
    coordinates[central] = 0.0
    coordinates[..., 0] = torch.where(
        central, signs * (2.0 * math.pi), coordinates[..., 0]
    )
    return coordinates


@torch.no_grad()
def teacher_outputs(
    coordinates: torch.Tensor,
    initial_state: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    coordinates_device = coordinates.to(device)
    generators = torch_triality_generators(
        dtype=coordinates_device.dtype, device=device
    )
    actions = spin8_group_actions(
        coordinates_device[:, :, None],
        generators,
        TRIALITY_REPRESENTATIONS,
        mode="exponential",
    )
    batch, length = coordinates.shape[:2]
    transition = Spin8AffineTransition(
        scale=torch.ones(batch, length, 1, device=device),
        action=actions,
        drive=torch.zeros(
            batch,
            length,
            1,
            len(TRIALITY_REPRESENTATIONS),
            SPIN8_DIM,
            device=device,
        ),
    )
    prefixes = work_efficient_spin8_scan(transition)
    initial = initial_state.to(device).expand(batch, -1, -1, -1)
    return apply_spin8_affine(prefixes, initial[:, None])[:, -1, 0].cpu()


def make_schedules(
    config: Spin8TransportConfig, device: torch.device
) -> tuple[
    list[tuple[torch.Tensor, torch.Tensor]],
    dict[int, tuple[torch.Tensor, torch.Tensor]],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    dict[str, str],
]:
    generator = torch.Generator().manual_seed(200_000 + config.seed)
    initial = teacher_initial_state()
    training = []
    training_hash_tensors = [initial]
    for length, updates in CURRICULUM:
        for _ in range(updates):
            inputs = sample_coordinates(config.batch_size, length, config, generator)
            targets = teacher_outputs(inputs, initial, device)
            training.append((inputs, targets))
            training_hash_tensors.extend((inputs, targets))

    evaluations = {}
    evaluation_hash_tensors = [initial]
    for length in config.evaluation_lengths:
        inputs = sample_coordinates(
            config.evaluation_batch_size, length, config, generator
        )
        targets = teacher_outputs(inputs, initial, device)
        evaluations[length] = (inputs, targets)
        evaluation_hash_tensors.extend((inputs, targets))

    tail = sample_coordinates(config.evaluation_batch_size, 127, config, generator)
    identity_inputs = torch.cat(
        (torch.zeros(config.evaluation_batch_size, 1, SPIN8_BIVECTOR_DIM), tail),
        dim=1,
    )
    center_token = torch.zeros(config.evaluation_batch_size, 1, SPIN8_BIVECTOR_DIM)
    center_token[..., 0] = 2.0 * math.pi
    center_inputs = torch.cat((center_token, tail), dim=1)
    identity_targets = teacher_outputs(identity_inputs, initial, device)
    center_targets = teacher_outputs(center_inputs, initial, device)
    center_pair = (
        identity_inputs,
        identity_targets,
        center_inputs,
        center_targets,
    )
    pair_hash = tensor_hash(list(center_pair))
    return (
        training,
        evaluations,
        center_pair,
        {
            "training_schedule_sha256": tensor_hash(training_hash_tensors),
            "evaluation_schedule_sha256": tensor_hash(evaluation_hash_tensors),
            "center_pair_schedule_sha256": pair_hash,
            "teacher_initial_state_sha256": tensor_hash([initial]),
        },
    )


class MaintainedPureSpin8Tracker(nn.Module):
    recurrent_state_scalars = 24

    def __init__(self, model_seed: int) -> None:
        super().__init__()
        self.layer = PureSpin8SSMLayer(
            SPIN8_BIVECTOR_DIM,
            channels=1,
            representations=TRIALITY_REPRESENTATIONS,
            action_mode="exponential",
            triality_coupling=False,
            transport_only=True,
            normalize_inputs=False,
        )
        generator = torch.Generator().manual_seed(300_000 + model_seed)
        with torch.no_grad():
            self.layer.coefficient_controller.weight.copy_(
                torch.eye(SPIN8_BIVECTOR_DIM)
            )
            self.layer.coefficient_controller.weight.add_(
                0.05
                * torch.randn(
                    self.layer.coefficient_controller.weight.shape,
                    generator=generator,
                )
            )
            self.layer.coefficient_controller.bias.zero_()
            initial = teacher_initial_state()
            initial = initial + 0.05 * torch.randn(initial.shape, generator=generator)
            self.layer.initial_state.copy_(F.normalize(initial, dim=-1))
            self.layer.coupling_logits.zero_()
        self.layer.coupling_logits.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        states, _ = self.layer(
            inputs, scan_mode="work_efficient", return_raw_states=True
        )
        return states[:, -1, 0]


class ContinuousMamba2TrialityTracker(nn.Module):
    def __init__(self, config: Spin8TransportConfig) -> None:
        super().__init__()
        self.input_projection = nn.Linear(SPIN8_BIVECTOR_DIM, config.mamba_hidden_size)
        self.backbone = Mamba2ForCausalLM(
            Mamba2Config(
                vocab_size=24,
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
        return (
            self.backbone(inputs_embeds=hidden, use_cache=False)
            .logits[:, -1]
            .reshape(inputs.shape[0], len(TRIALITY_REPRESENTATIONS), SPIN8_DIM)
        )


class ContinuousDeltaTrialityTracker(nn.Module):
    def __init__(self, config: Spin8TransportConfig) -> None:
        super().__init__()
        self.input_projection = nn.Linear(SPIN8_BIVECTOR_DIM, config.delta_hidden_size)
        self.norm = nn.RMSNorm(config.delta_hidden_size)
        self.delta = DeltaProductReferenceLayer(
            hidden_size=config.delta_hidden_size,
            num_heads=config.delta_heads,
            num_householder=config.delta_householder,
        )
        self.output = nn.Linear(config.delta_hidden_size, 24)

    @property
    def recurrent_state_scalars(self) -> int:
        return self.delta.num_heads * self.delta.head_dim**2

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(inputs)
        update, _ = self.delta(self.norm(hidden), scan_mode="parallel")
        return self.output(hidden[:, -1] + update[:, -1]).reshape(
            inputs.shape[0], len(TRIALITY_REPRESENTATIONS), SPIN8_DIM
        )


def parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    error = predictions - targets
    return {
        "mse": float(F.mse_loss(predictions, targets)),
        "mean_relative_error": float(
            (
                torch.linalg.vector_norm(error.flatten(start_dim=1), dim=-1)
                / torch.linalg.vector_norm(
                    targets.flatten(start_dim=1), dim=-1
                ).clamp_min(1e-12)
            ).mean()
        ),
        "maximum_absolute_error": float(error.abs().max()),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    evaluations: dict[int, tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    model.eval()
    result = {}
    for length, (inputs, targets) in evaluations.items():
        predictions = model(inputs.to(device))
        result[str(length)] = metrics(predictions, targets.to(device))
    return result


@torch.no_grad()
def evaluate_center_pair(
    model: nn.Module,
    center_pair: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    identity_inputs, identity_targets, center_inputs, center_targets = center_pair
    identity_targets = identity_targets.to(device)
    center_targets = center_targets.to(device)
    identity_predictions = model(identity_inputs.to(device))
    center_predictions = model(center_inputs.to(device))

    identity_correct = (identity_predictions - identity_targets).square().flatten(
        start_dim=1
    ).sum(-1) < (identity_predictions - center_targets).square().flatten(
        start_dim=1
    ).sum(-1)
    center_correct = (center_predictions - center_targets).square().flatten(
        start_dim=1
    ).sum(-1) < (center_predictions - identity_targets).square().flatten(
        start_dim=1
    ).sum(-1)
    spinor_negation = torch.cat(
        (
            center_predictions[:, 1] + identity_predictions[:, 1],
            center_predictions[:, 2] + identity_predictions[:, 2],
        ),
        dim=-1,
    )
    return {
        "paired_mse": float(
            0.5
            * (
                F.mse_loss(identity_predictions, identity_targets)
                + F.mse_loss(center_predictions, center_targets)
            )
        ),
        "center_classification_accuracy": float(
            torch.cat((identity_correct, center_correct)).float().mean()
        ),
        "predicted_spinor_negation_rmse": float(spinor_negation.square().mean().sqrt()),
        "teacher_vector_identity_residual": float(
            (identity_targets[:, 0] - center_targets[:, 0]).abs().max()
        ),
        "teacher_spinor_negation_residual": float(
            torch.stack(
                (
                    identity_targets[:, 1] + center_targets[:, 1],
                    identity_targets[:, 2] + center_targets[:, 2],
                )
            )
            .abs()
            .max()
        ),
    }


def train_candidate(
    name: str,
    model: nn.Module,
    training: list[tuple[torch.Tensor, torch.Tensor]],
    evaluations: dict[int, tuple[torch.Tensor, torch.Tensor]],
    center_pair: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    config: Spin8TransportConfig,
    device: torch.device,
    checkpoint_directory: Path,
) -> dict[str, Any]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    losses = {}
    model.train()
    for step, (inputs, targets) in enumerate(training, start=1):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = F.mse_loss(predictions, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name} produced nonfinite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        if step in (1, 100, 250, 500, 750, config.steps):
            losses[str(step)] = float(loss.detach())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    evaluation = evaluate(model, evaluations, device)
    pair_evaluation = evaluate_center_pair(model, center_pair, device)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_directory / f"{name}_seed{config.seed}.pt"
    torch.save(
        {
            "format_version": 1,
            "candidate": name,
            "model_version": PURE_SPIN8_VERSION
            if name == "maintained_pure_spin8"
            else None,
            "config": asdict(config),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "evaluation": evaluation,
            "center_pair_evaluation": pair_evaluation,
        },
        checkpoint,
    )
    result = {
        "parameters": parameter_count(model),
        "recurrent_state_scalars": int(model.recurrent_state_scalars),
        "loss_samples": losses,
        "training_wall_seconds": elapsed,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "evaluation": evaluation,
        "center_pair_evaluation": pair_evaluation,
        "checkpoint": str(
            checkpoint.resolve().relative_to(Path(__file__).resolve().parent)
        ),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    seed_reports = []
    for seed in args.seeds:
        config = Spin8TransportConfig(seed=seed, steps=args.steps)
        training, evaluations, center_pair, schedules = make_schedules(config, device)
        factories = {
            "maintained_pure_spin8": lambda seed=seed: MaintainedPureSpin8Tracker(seed),
            "transformers_mamba2": lambda config=config: (
                ContinuousMamba2TrialityTracker(config)
            ),
            "delta_product_reference": lambda config=config: (
                ContinuousDeltaTrialityTracker(config)
            ),
        }
        results = {}
        for offset, (name, factory) in enumerate(factories.items()):
            seed_everything(400_000 + 1_000 * seed + offset)
            results[name] = train_candidate(
                name,
                factory(),
                training,
                evaluations,
                center_pair,
                config,
                device,
                args.checkpoint_directory,
            )
        teacher_self_replay = {
            str(length): metrics(targets, targets)
            for length, (_, targets) in evaluations.items()
        }
        pure_l128 = results["maintained_pure_spin8"]["evaluation"]["128"]["mse"]
        checks = {
            "teacher_self_replay_below_1e_12": max(
                row["mse"] for row in teacher_self_replay.values()
            )
            < 1e-12,
            "teacher_center_signature_is_exact": max(
                results[name]["center_pair_evaluation"][
                    "teacher_vector_identity_residual"
                ]
                for name in results
            )
            < 1e-5
            and max(
                results[name]["center_pair_evaluation"][
                    "teacher_spinor_negation_residual"
                ]
                for name in results
            )
            < 1e-5,
            "pure_spin8_l128_below_1e_4": pure_l128 < 1e-4,
            "pure_spin8_beats_mamba2_l128": pure_l128
            < results["transformers_mamba2"]["evaluation"]["128"]["mse"],
            "pure_spin8_beats_delta_l128": pure_l128
            < results["delta_product_reference"]["evaluation"]["128"]["mse"],
            "pure_spin8_center_accuracy_above_0_99": results["maintained_pure_spin8"][
                "center_pair_evaluation"
            ]["center_classification_accuracy"]
            > 0.99,
            "all_metrics_finite": all(
                math.isfinite(row["mse"])
                for result in results.values()
                for row in result["evaluation"].values()
            ),
        }
        seed_reports.append(
            {
                "seed": seed,
                "config": asdict(config),
                "schedules": schedules,
                "teacher_self_replay": teacher_self_replay,
                "results": results,
                "checks": checks,
                "all_required_checks_passed": all(checks.values()),
            }
        )
    return {
        "schema_version": 1,
        "experiment": "Pure Spin8 center-sensitive triality transport versus Mamba2",
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
        "pure_spin8_version": PURE_SPIN8_VERSION,
        "seed_reports": seed_reports,
        "all_required_checks_passed": all(
            report["all_required_checks_passed"] for report in seed_reports
        ),
        "claim_boundary": (
            "Algebra-matched center-sensitive synthetic transport result; not "
            "generic language-model or fused-kernel superiority."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--checkpoint-directory", type=Path, default=DEFAULT_CHECKPOINT_DIRECTORY
    )
    args = parser.parse_args()
    if args.steps != sum(updates for _, updates in CURRICULUM):
        parser.error("steps must equal the frozen curriculum total")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be nonempty and unique")
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
