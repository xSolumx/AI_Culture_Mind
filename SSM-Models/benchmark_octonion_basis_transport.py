"""Frozen Haar-basis replication of continuous associator tracking."""

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
    metrics,
    parameter_count,
    seed_everything,
    tensor_hash,
)
from pure_rotor_ssm.octonion_operator_scan import (
    OCTONION_DIM,
    associative_matrix_prefix_scan,
    octonion_left_multiplication_matrix,
    unit_octonion,
)
from torch import nn
from torch.nn import functional as F

PROTOCOL_FROZEN_AT = "2026-08-16T21:11:30+02:00"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "experiments"
    / "artifacts"
    / "octonion_basis_transport_replication300.json"
)
DEFAULT_CHECKPOINT_DIRECTORY = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "octonion_basis_transport_replication300"
)
BASIS_SEEDS = (0, 1, 2)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def haar_special_orthogonal(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20_000 + seed)
    raw = torch.randn(OCTONION_DIM, OCTONION_DIM, generator=generator)
    basis, triangular = torch.linalg.qr(raw)
    signs = torch.sign(torch.diagonal(triangular))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    basis = basis * signs
    if torch.linalg.det(basis) < 0:
        basis[:, 0] = -basis[:, 0]
    return basis


def transport_tokens(tokens: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,btj->bti", basis, tokens)


def transport_operators(operators: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,btjk,lk->btil", basis, operators, basis)


def make_transported_schedules(
    config: AssociatorConfig, basis: torch.Tensor
) -> tuple[
    list[tuple[torch.Tensor, torch.Tensor]],
    dict[int, tuple[torch.Tensor, torch.Tensor]],
    dict[str, str],
]:
    generator = torch.Generator().manual_seed(30_000 + config.seed)
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
        canonical_targets = exact_targets(canonical[:, :, None]).reshape(
            config.batch_size, config.train_length, OCTONION_DIM, OCTONION_DIM
        )
        targets = transport_operators(canonical_targets, basis).flatten(start_dim=-2)
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
        canonical_targets = exact_targets(canonical[:, :, None]).reshape(
            config.evaluation_batch_size,
            length,
            OCTONION_DIM,
            OCTONION_DIM,
        )
        targets = transport_operators(canonical_targets, basis).flatten(start_dim=-2)
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


def skew_basis() -> torch.Tensor:
    result = torch.zeros(28, OCTONION_DIM, OCTONION_DIM)
    row = 0
    for left in range(OCTONION_DIM):
        for right in range(left + 1, OCTONION_DIM):
            result[row, left, right] = 1
            result[row, right, left] = -1
            row += 1
    return result


class LearnedBasisOperatorTracker(nn.Module):
    recurrent_state_scalars = OCTONION_DIM**2

    def __init__(self) -> None:
        super().__init__()
        self.coordinates = nn.Parameter(torch.zeros(28))
        self.register_buffer("skew_basis", skew_basis(), persistent=False)

    def basis(self) -> torch.Tensor:
        skew = torch.einsum("k,kij->ij", self.coordinates, self.skew_basis)
        return torch.matrix_exp(skew)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        basis = self.basis()
        canonical = inputs @ basis
        prefixes = exact_targets(canonical[:, :, None]).reshape(
            inputs.shape[0], inputs.shape[1], OCTONION_DIM, OCTONION_DIM
        )
        return transport_operators(prefixes, basis).flatten(start_dim=-2)


class DenseLinearOperatorTracker(nn.Module):
    recurrent_state_scalars = OCTONION_DIM**2

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(OCTONION_DIM, OCTONION_DIM**2, bias=False)
        basis_tokens = torch.eye(OCTONION_DIM)
        canonical_actions = octonion_left_multiplication_matrix(basis_tokens)
        with torch.no_grad():
            self.encoder.weight.copy_(canonical_actions.flatten(start_dim=-2).T)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        leaves = self.encoder(inputs).reshape(
            inputs.shape[0], inputs.shape[1], OCTONION_DIM, OCTONION_DIM
        )
        prefixes = associative_matrix_prefix_scan(leaves, backend="work_efficient")
        return prefixes.flatten(start_dim=-2)


@torch.no_grad()
def evaluate_predictions(
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
        observed = observed.to(device)
        targets = targets.to(device)
        if candidate == "exact_transported_oracle":
            canonical = observed @ basis_device
            canonical_predictions = exact_targets(canonical[:, :, None]).reshape(
                observed.shape[0], length, OCTONION_DIM, OCTONION_DIM
            )
            predictions = transport_operators(
                canonical_predictions, basis_device
            ).flatten(start_dim=-2)
        elif candidate == "fixed_canonical_operator":
            predictions = exact_targets(observed[:, :, None])
        elif candidate == "transported_collapsed_octonion":
            canonical = observed @ basis_device
            collapsed = collapsed_octonion_predictions(canonical).reshape(
                observed.shape[0], length, OCTONION_DIM, OCTONION_DIM
            )
            predictions = transport_operators(collapsed, basis_device).flatten(
                start_dim=-2
            )
        else:
            assert model is not None
            predictions = model(observed)
        result[str(length)] = metrics(predictions, targets)
    return result


def train_candidate(
    name: str,
    model: nn.Module,
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
    evaluation = evaluate_predictions(name, model, evaluations, basis, device)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_directory / f"{name}_basis{config.seed}.pt"
    torch.save(
        {
            "format_version": 1,
            "candidate": name,
            "config": asdict(config),
            "basis_sha256": tensor_hash([basis]),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "evaluation": evaluation,
        },
        checkpoint,
    )
    resolved_checkpoint = checkpoint.resolve()
    try:
        serialized_checkpoint = str(
            resolved_checkpoint.relative_to(Path(__file__).resolve().parent)
        )
    except ValueError:
        serialized_checkpoint = str(resolved_checkpoint)
    result = {
        "parameters": parameter_count(model),
        "recurrent_state_scalars": int(model.recurrent_state_scalars),
        "loss_samples": loss_samples,
        "training_wall_seconds": elapsed,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "evaluation": evaluation,
        "checkpoint": serialized_checkpoint,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    if name == "learned_basis_operator":
        learned_basis = model.basis().detach().cpu()
        gauge_residual = learned_basis.T @ basis
        result["learned_basis_orthogonality_residual"] = float(
            (learned_basis.T @ learned_basis - torch.eye(OCTONION_DIM)).abs().max()
        )
        result["learned_to_true_gauge_trace"] = float(torch.trace(gauge_residual))
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_basis(
    basis_seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    config = AssociatorConfig(seed=basis_seed, steps=args.steps)
    basis = haar_special_orthogonal(basis_seed)
    training, evaluations, schedules = make_transported_schedules(config, basis)
    results: dict[str, Any] = {}
    for name, state in (
        ("exact_transported_oracle", 64),
        ("fixed_canonical_operator", 64),
        ("transported_collapsed_octonion", 8),
    ):
        results[name] = {
            "parameters": 0,
            "recurrent_state_scalars": state,
            "evaluation": evaluate_predictions(name, None, evaluations, basis, device),
        }

    factories: dict[str, Any] = {
        "learned_basis_operator": LearnedBasisOperatorTracker,
        "dense_linear_operator": DenseLinearOperatorTracker,
        "transformers_mamba2": lambda: ContinuousMamba2Tracker(config),
        "delta_product_reference": lambda: ContinuousDeltaProductTracker(config),
    }
    for offset, (name, factory) in enumerate(factories.items()):
        seed_everything(40_000 + 100 * basis_seed + offset)
        results[name] = train_candidate(
            name,
            factory(),
            training,
            evaluations,
            basis,
            config,
            device,
            args.checkpoint_directory,
        )

    checks = {
        "oracle_is_exact": max(
            row["mse"]
            for row in results["exact_transported_oracle"]["evaluation"].values()
        )
        < 1e-12,
        "collapsed_ablation_fails_l128": results["transported_collapsed_octonion"][
            "evaluation"
        ]["128"]["mse"]
        > 1e-2,
        "learned_basis_l128_below_1e_3": results["learned_basis_operator"][
            "evaluation"
        ]["128"]["mse"]
        < 1e-3,
        "learned_basis_beats_fixed_canonical": results["learned_basis_operator"][
            "evaluation"
        ]["128"]["mse"]
        < results["fixed_canonical_operator"]["evaluation"]["128"]["mse"],
        "dense_operator_l128_below_1e_3": results["dense_linear_operator"][
            "evaluation"
        ]["128"]["mse"]
        < 1e-3,
        "dense_operator_beats_fixed_canonical": results["dense_linear_operator"][
            "evaluation"
        ]["128"]["mse"]
        < results["fixed_canonical_operator"]["evaluation"]["128"]["mse"],
        "all_learned_metrics_finite": all(
            math.isfinite(row["mse"])
            for name in factories
            for row in results[name]["evaluation"].values()
        ),
    }
    return {
        "basis_seed": basis_seed,
        "basis_determinant": float(torch.linalg.det(basis)),
        "basis_orthogonality_residual": float(
            (basis.T @ basis - torch.eye(OCTONION_DIM)).abs().max()
        ),
        "config": asdict(config),
        "schedules": schedules,
        "results": results,
        "checks": checks,
        "all_required_checks_passed": all(checks.values()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    basis_reports = [run_basis(seed, args, device) for seed in args.basis_seeds]
    return {
        "schema_version": 1,
        "experiment": "Haar-basis continuous octonion operator replication",
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
            "Three-basis synthetic operator-identification replication; not a "
            "natural-task, triality-specific, or fused-baseline superiority claim."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis-seeds", type=int, nargs="+", default=BASIS_SEEDS)
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
    if not args.basis_seeds:
        parser.error("at least one basis seed is required")
    if len(set(args.basis_seeds)) != len(args.basis_seeds):
        parser.error("basis seeds must be unique")
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
