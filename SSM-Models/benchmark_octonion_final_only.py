"""Frozen final-only Haar-basis octonion composition benchmark."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from benchmark_octonion_associator_tracking import (
    AssociatorConfig,
    ContinuousDeltaProductTracker,
    ContinuousMamba2Tracker,
    collapsed_octonion_predictions,
    exact_targets,
    parameter_count,
    seed_everything,
    tensor_hash,
)
from benchmark_octonion_basis_transport import (
    BASIS_SEEDS,
    DenseLinearOperatorTracker,
    LearnedBasisOperatorTracker,
    haar_special_orthogonal,
    transport_operators,
    transport_tokens,
)
from pure_rotor_ssm.octonion_operator_scan import OCTONION_DIM, unit_octonion
from torch import nn
from torch.nn import functional as F

PROTOCOL_FROZEN_AT = "2026-08-16T21:23:00+02:00"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "experiments"
    / "artifacts"
    / "octonion_final_only_replication1000.json"
)
DEFAULT_CHECKPOINT_DIRECTORY = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "octonion_final_only_replication1000"
)
MODEL_INITIALIZATIONS = (0, 1, 2)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def terminal_metrics(
    predictions: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    error = (predictions - targets).reshape(-1, OCTONION_DIM, OCTONION_DIM)
    target = targets.reshape(-1, OCTONION_DIM, OCTONION_DIM)
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


def canonical_terminal_targets(tokens: torch.Tensor) -> torch.Tensor:
    return exact_targets(tokens[:, :, None])[:, -1]


def make_final_only_schedules(
    config: AssociatorConfig, basis: torch.Tensor
) -> tuple[
    list[tuple[torch.Tensor, torch.Tensor]],
    dict[int, tuple[torch.Tensor, torch.Tensor]],
    dict[str, str],
]:
    generator = torch.Generator().manual_seed(50_000 + config.seed)
    training: list[tuple[torch.Tensor, torch.Tensor]] = []
    training_hash_tensors: list[torch.Tensor] = [basis]
    for _ in range(config.steps):
        canonical = unit_octonion(
            torch.randn(
                config.batch_size,
                config.train_length,
                OCTONION_DIM,
                generator=generator,
            )
        )
        observed = transport_tokens(canonical, basis)
        terminal = canonical_terminal_targets(canonical).reshape(
            config.batch_size, OCTONION_DIM, OCTONION_DIM
        )
        targets = transport_operators(terminal[:, None], basis)[:, 0].flatten(
            start_dim=-2
        )
        training.append((observed, targets))
        training_hash_tensors.extend((observed, targets))

    evaluations: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    evaluation_hash_tensors: list[torch.Tensor] = [basis]
    for length in config.evaluation_lengths:
        canonical = unit_octonion(
            torch.randn(
                config.evaluation_batch_size,
                length,
                OCTONION_DIM,
                generator=generator,
            )
        )
        observed = transport_tokens(canonical, basis)
        terminal = canonical_terminal_targets(canonical).reshape(
            config.evaluation_batch_size, OCTONION_DIM, OCTONION_DIM
        )
        targets = transport_operators(terminal[:, None], basis)[:, 0].flatten(
            start_dim=-2
        )
        evaluations[length] = (observed, targets)
        evaluation_hash_tensors.extend((observed, targets))

    return (
        training,
        evaluations,
        {
            "basis_sha256": tensor_hash([basis]),
            "training_schedule_sha256": tensor_hash(training_hash_tensors),
            "evaluation_schedule_sha256": tensor_hash(evaluation_hash_tensors),
        },
    )


def initialize_structured(model_seed: int) -> LearnedBasisOperatorTracker:
    model = LearnedBasisOperatorTracker()
    generator = torch.Generator().manual_seed(60_000 + model_seed)
    with torch.no_grad():
        model.coordinates.copy_(
            0.05 * torch.randn(model.coordinates.shape, generator=generator)
        )
    return model


def initialize_dense(model_seed: int) -> DenseLinearOperatorTracker:
    model = DenseLinearOperatorTracker()
    generator = torch.Generator().manual_seed(70_000 + model_seed)
    with torch.no_grad():
        model.encoder.weight.add_(
            0.01 * torch.randn(model.encoder.weight.shape, generator=generator)
        )
    return model


@torch.no_grad()
def evaluate_terminal(
    candidate: str,
    model: nn.Module | None,
    evaluations: dict[int, tuple[torch.Tensor, torch.Tensor]],
    basis: torch.Tensor,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    if model is not None:
        model.eval()
    basis_device = basis.to(device)
    result = {}
    for length, (observed, targets) in evaluations.items():
        observed, targets = observed.to(device), targets.to(device)
        if candidate == "exact_transported_oracle":
            canonical = observed @ basis_device
            terminal = canonical_terminal_targets(canonical).reshape(
                observed.shape[0], OCTONION_DIM, OCTONION_DIM
            )
            predictions = transport_operators(terminal[:, None], basis_device)[
                :, 0
            ].flatten(start_dim=-2)
        elif candidate == "fixed_canonical_operator":
            predictions = canonical_terminal_targets(observed)
        elif candidate == "transported_collapsed_octonion":
            canonical = observed @ basis_device
            terminal = collapsed_octonion_predictions(canonical)[:, -1].reshape(
                observed.shape[0], OCTONION_DIM, OCTONION_DIM
            )
            predictions = transport_operators(terminal[:, None], basis_device)[
                :, 0
            ].flatten(start_dim=-2)
        else:
            assert model is not None
            predictions = model(observed)[:, -1]
        result[str(length)] = terminal_metrics(predictions, targets)
    return result


def serialize_checkpoint(checkpoint: Path) -> str:
    resolved = checkpoint.resolve()
    try:
        return str(resolved.relative_to(Path(__file__).resolve().parent))
    except ValueError:
        return str(resolved)


def train_terminal_candidate(
    name: str,
    model: nn.Module,
    model_seed: int,
    training: list[tuple[torch.Tensor, torch.Tensor]],
    evaluations: dict[int, tuple[torch.Tensor, torch.Tensor]],
    basis: torch.Tensor,
    config: AssociatorConfig,
    device: torch.device,
    checkpoint_directory: Path,
) -> dict[str, Any]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    loss_samples: dict[str, float] = {}
    model.train()
    for step, (inputs, targets) in enumerate(training, start=1):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)[:, -1]
        loss = F.mse_loss(predictions, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name} produced nonfinite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in (1, 100, 300, 500, 750, config.steps):
            loss_samples[str(step)] = float(loss.detach())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    evaluation = evaluate_terminal(name, model, evaluations, basis, device)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_directory / (
        f"{name}_basis{config.seed}_init{model_seed}.pt"
    )
    torch.save(
        {
            "format_version": 1,
            "candidate": name,
            "basis_seed": config.seed,
            "model_seed": model_seed,
            "config": asdict(config),
            "basis_sha256": tensor_hash([basis]),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "evaluation": evaluation,
        },
        checkpoint,
    )
    result = {
        "parameters": parameter_count(model),
        "recurrent_state_scalars": int(model.recurrent_state_scalars),
        "loss_samples": loss_samples,
        "training_wall_seconds": elapsed,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "evaluation": evaluation,
        "checkpoint": serialize_checkpoint(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    if name == "learned_basis_operator":
        learned_basis = model.basis().detach().cpu()
        result["learned_basis_orthogonality_residual"] = float(
            (learned_basis.T @ learned_basis - torch.eye(OCTONION_DIM)).abs().max()
        )
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_basis(
    basis_seed: int, args: argparse.Namespace, device: torch.device
) -> dict[str, Any]:
    config = AssociatorConfig(seed=basis_seed, steps=args.steps)
    basis = haar_special_orthogonal(basis_seed)
    training, evaluations, schedules = make_final_only_schedules(config, basis)
    fixed = {}
    for name, state in (
        ("exact_transported_oracle", 64),
        ("fixed_canonical_operator", 64),
        ("transported_collapsed_octonion", 8),
    ):
        fixed[name] = {
            "parameters": 0,
            "recurrent_state_scalars": state,
            "evaluation": evaluate_terminal(name, None, evaluations, basis, device),
        }

    structured = {}
    dense = {}
    for model_seed in args.model_seeds:
        seed_everything(80_000 + 1_000 * basis_seed + model_seed)
        structured[str(model_seed)] = train_terminal_candidate(
            "learned_basis_operator",
            initialize_structured(1_000 * basis_seed + model_seed),
            model_seed,
            training,
            evaluations,
            basis,
            config,
            device,
            args.checkpoint_directory,
        )
        seed_everything(90_000 + 1_000 * basis_seed + model_seed)
        dense[str(model_seed)] = train_terminal_candidate(
            "dense_linear_operator",
            initialize_dense(1_000 * basis_seed + model_seed),
            model_seed,
            training,
            evaluations,
            basis,
            config,
            device,
            args.checkpoint_directory,
        )

    references = {}
    for offset, (name, factory) in enumerate(
        (
            ("transformers_mamba2", lambda: ContinuousMamba2Tracker(config)),
            ("delta_product_reference", lambda: ContinuousDeltaProductTracker(config)),
        )
    ):
        seed_everything(100_000 + 1_000 * basis_seed + offset)
        references[name] = train_terminal_candidate(
            name,
            factory(),
            0,
            training,
            evaluations,
            basis,
            config,
            device,
            args.checkpoint_directory,
        )

    checks = {
        "oracle_is_within_float32_gate": max(
            row["mse"]
            for row in fixed["exact_transported_oracle"]["evaluation"].values()
        )
        < 2e-12,
        "collapsed_ablation_fails_l128": fixed["transported_collapsed_octonion"][
            "evaluation"
        ]["128"]["mse"]
        > 1e-2,
        "all_structured_l128_below_1e_3": all(
            record["evaluation"]["128"]["mse"] < 1e-3 for record in structured.values()
        ),
        "all_structured_beat_matched_dense_l128": all(
            structured[seed]["evaluation"]["128"]["mse"]
            < dense[seed]["evaluation"]["128"]["mse"]
            for seed in structured
        ),
        "all_learned_metrics_finite": all(
            math.isfinite(row["mse"])
            for family in (structured, dense, references)
            for record in family.values()
            for row in record["evaluation"].values()
        ),
    }
    return {
        "basis_seed": basis_seed,
        "config": asdict(config),
        "schedules": schedules,
        "fixed_results": fixed,
        "learned_basis_operator": structured,
        "dense_linear_operator": dense,
        "reference_results": references,
        "checks": checks,
        "all_required_checks_passed": all(checks.values()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    basis_reports = [run_basis(seed, args, device) for seed in args.basis_seeds]
    return {
        "schema_version": 1,
        "experiment": "Haar-basis final-only octonion operator recovery",
        "protocol_frozen_at": PROTOCOL_FROZEN_AT,
        "started_at": args.started_at,
        "finished_at": now(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "numpy": np.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor()
            ),
        },
        "basis_reports": basis_reports,
        "all_required_checks_passed": all(
            report["all_required_checks_passed"] for report in basis_reports
        ),
        "claim_boundary": (
            "Final-only synthetic multi-basis/multi-initialization recovery; "
            "not natural-task, triality-specific, or fused-baseline superiority."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis-seeds", type=int, nargs="+", default=BASIS_SEEDS)
    parser.add_argument(
        "--model-seeds", type=int, nargs="+", default=MODEL_INITIALIZATIONS
    )
    parser.add_argument("--steps", type=int, default=1_000)
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
    if len(set(args.basis_seeds)) != len(args.basis_seeds):
        parser.error("basis seeds must be unique")
    if len(set(args.model_seeds)) != len(args.model_seeds):
        parser.error("model seeds must be unique")
    if not args.basis_seeds or not args.model_seeds:
        parser.error("basis and model seeds must be nonempty")
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
