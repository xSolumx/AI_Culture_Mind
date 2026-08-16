"""Frozen terminal-only composition-depth curriculum benchmark."""

from __future__ import annotations

import argparse
import json
import math
import platform
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
    seed_everything,
    tensor_hash,
)
from benchmark_octonion_basis_transport import (
    BASIS_SEEDS,
    haar_special_orthogonal,
    transport_operators,
    transport_tokens,
)
from benchmark_octonion_final_only import (
    MODEL_INITIALIZATIONS,
    canonical_terminal_targets,
    evaluate_terminal,
    initialize_dense,
    initialize_structured,
    train_terminal_candidate,
)
from pure_rotor_ssm.octonion_operator_scan import OCTONION_DIM, unit_octonion

PROTOCOL_FROZEN_AT = "2026-08-16T21:33:30+02:00"
CURRICULUM = ((2, 250), (4, 250), (8, 250), (16, 250))
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "experiments"
    / "artifacts"
    / "octonion_final_only_curriculum1000.json"
)
DEFAULT_CHECKPOINT_DIRECTORY = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "octonion_final_only_curriculum1000"
)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def terminal_batch(
    batch_size: int,
    length: int,
    basis: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    canonical = unit_octonion(
        torch.randn(batch_size, length, OCTONION_DIM, generator=generator)
    )
    observed = transport_tokens(canonical, basis)
    terminal = canonical_terminal_targets(canonical).reshape(
        batch_size, OCTONION_DIM, OCTONION_DIM
    )
    targets = transport_operators(terminal[:, None], basis)[:, 0].flatten(start_dim=-2)
    return observed, targets


def make_curriculum_schedules(
    config: AssociatorConfig, basis: torch.Tensor
) -> tuple[
    list[tuple[torch.Tensor, torch.Tensor]],
    dict[int, tuple[torch.Tensor, torch.Tensor]],
    dict[str, str],
]:
    if config.steps != sum(updates for _, updates in CURRICULUM):
        raise ValueError("steps must equal the frozen curriculum total")
    generator = torch.Generator().manual_seed(110_000 + config.seed)
    training: list[tuple[torch.Tensor, torch.Tensor]] = []
    training_hash_tensors: list[torch.Tensor] = [basis]
    for length, updates in CURRICULUM:
        for _ in range(updates):
            batch = terminal_batch(config.batch_size, length, basis, generator)
            training.append(batch)
            training_hash_tensors.extend(batch)

    evaluations: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    evaluation_hash_tensors: list[torch.Tensor] = [basis]
    for length in config.evaluation_lengths:
        batch = terminal_batch(config.evaluation_batch_size, length, basis, generator)
        evaluations[length] = batch
        evaluation_hash_tensors.extend(batch)
    return (
        training,
        evaluations,
        {
            "basis_sha256": tensor_hash([basis]),
            "training_schedule_sha256": tensor_hash(training_hash_tensors),
            "evaluation_schedule_sha256": tensor_hash(evaluation_hash_tensors),
        },
    )


def run_basis(
    basis_seed: int, args: argparse.Namespace, device: torch.device
) -> dict[str, Any]:
    config = AssociatorConfig(seed=basis_seed, steps=args.steps)
    basis = haar_special_orthogonal(basis_seed)
    training, evaluations, schedules = make_curriculum_schedules(config, basis)
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
        initialization_seed = 10_000 * basis_seed + model_seed
        seed_everything(120_000 + initialization_seed)
        structured[str(model_seed)] = train_terminal_candidate(
            "learned_basis_operator",
            initialize_structured(initialization_seed),
            model_seed,
            training,
            evaluations,
            basis,
            config,
            device,
            args.checkpoint_directory,
        )
        seed_everything(130_000 + initialization_seed)
        dense[str(model_seed)] = train_terminal_candidate(
            "dense_linear_operator",
            initialize_dense(initialization_seed),
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
        seed_everything(140_000 + 1_000 * basis_seed + offset)
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
        "curriculum": [
            {"length": length, "updates": updates} for length, updates in CURRICULUM
        ],
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
    reports = [run_basis(seed, args, device) for seed in args.basis_seeds]
    return {
        "schema_version": 1,
        "experiment": "terminal-only octonion composition-depth homotopy",
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
        "basis_reports": reports,
        "all_required_checks_passed": all(
            report["all_required_checks_passed"] for report in reports
        ),
        "claim_boundary": (
            "Terminal-only synthetic composition-depth curriculum; not a global "
            "optimization theorem, natural-task result, or fused SSM comparison."
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
    if args.steps != sum(updates for _, updates in CURRICULUM):
        parser.error("steps must equal the frozen curriculum total")
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
