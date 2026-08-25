"""Prospectively frozen G15A-R first-order learning-repair cohort."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from spin8_triality import SPIN8_PAIRS
from torch.nn import functional as F

if __package__:
    from .g15a_spin_dirac_cohort import (
        OFF_TORUS_PAIRS,
        _atomic_json,
        _git_state,
        _now,
        _oracle_memory,
        _sha256,
        _stable_seed,
    )
    from .g15af_full_frame_cohort import (
        ARM_NAMES,
        FullFrameBatch,
        _adjudicate,
        _frame_prediction,
        _metrics,
        _probe_bank,
        _semantic_coordinate,
        _teacher_target,
        generate_frame_batch,
    )
    from .g15al_learned_coordinate_cohort import (
        ACTION_VOCABULARY,
        TokenCoordinateController,
        _token_map,
    )
    from .optimizers import (
        BlockScalarSecondMomentAdamW,
        ScalarSecondMomentAdamW,
    )
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4.g15a_spin_dirac_cohort import (  # type: ignore[no-redef]
        OFF_TORUS_PAIRS,
        _atomic_json,
        _git_state,
        _now,
        _oracle_memory,
        _sha256,
        _stable_seed,
    )
    from hybrid_memory_v1_4.g15af_full_frame_cohort import (  # type: ignore[no-redef]
        ARM_NAMES,
        FullFrameBatch,
        _adjudicate,
        _frame_prediction,
        _metrics,
        _probe_bank,
        _semantic_coordinate,
        _teacher_target,
        generate_frame_batch,
    )
    from hybrid_memory_v1_4.g15al_learned_coordinate_cohort import (  # type: ignore[no-redef]
        ACTION_VOCABULARY,
        TokenCoordinateController,
        _token_map,
    )
    from hybrid_memory_v1_4.optimizers import (  # type: ignore[no-redef]
        BlockScalarSecondMomentAdamW,
        ScalarSecondMomentAdamW,
    )


PROTOCOL = Path(__file__).with_name("G15AR_FIRST_ORDER_PROTOCOL_2026-08-25.md")
G15AF_ARTIFACT_SHA256 = (
    "cdfdcb1785e2bf2a85ea592e2100a61596d1a06ea219a9d75c058f1d11e74296"
)
DIAGNOSTIC_ARTIFACT_SHA256 = (
    "6b36752060da872b14047da38d592a5a51d32e4e215fe00d577e92186877cb67"
)
DEVELOPMENT_SEEDS = (2237, 2239, 2243)
CONFIRMATION_SEEDS = (2251, 2267, 2273)
DEVELOPMENT_RECIPES = (
    "G-fixed/random",
    "G-decay/random",
    "B-decay/random",
    "G-decay/curriculum",
    "B-decay/curriculum",
)
SELECTION_ORDER = DEVELOPMENT_RECIPES[1:]
VALIDATION_MILESTONES = (1, 25, 50, 100, 150, 200, 300, 450, 600)
EVALUATION_SPECS = ((64, 8), (256, 12), (1024, 16))


@dataclass(frozen=True)
class RepairConfig:
    mode: str
    development_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    updates: int
    batch_size: int
    training_length: int
    validation_examples: int
    final_examples: int
    validation_milestones: tuple[int, ...]
    evaluation_specs: tuple[tuple[int, int], ...]
    dtype: str = "float32"


def quality_config() -> RepairConfig:
    return RepairConfig(
        mode="quality",
        development_seeds=DEVELOPMENT_SEEDS,
        confirmation_seeds=CONFIRMATION_SEEDS,
        updates=600,
        batch_size=16,
        training_length=16,
        validation_examples=32,
        final_examples=80,
        validation_milestones=VALIDATION_MILESTONES,
        evaluation_specs=EVALUATION_SPECS,
    )


def smoke_config() -> RepairConfig:
    return RepairConfig(
        mode="smoke",
        development_seeds=(31,),
        confirmation_seeds=(),
        updates=6,
        batch_size=16,
        training_length=16,
        validation_examples=8,
        final_examples=8,
        validation_milestones=(1, 3, 6),
        evaluation_specs=((16, 3), (32, 4)),
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _singleton_batch(model_seed: int) -> FullFrameBatch:
    batch_size = ACTION_VOCABULARY
    length = 16
    tokens = torch.zeros(batch_size, length, dtype=torch.long)
    coordinates = torch.zeros(batch_size, length, 1, len(SPIN8_PAIRS))
    positions = torch.full((batch_size, 2), -1, dtype=torch.long)
    token_map = _token_map(model_seed)
    for semantic in range(ACTION_VOCABULARY):
        tokens[semantic, 1] = token_map[semantic]
        coordinates[semantic, 1] = _semantic_coordinate(semantic)
        positions[semantic, 0] = 1
    frames = (
        _probe_bank(model_seed)
        .to(torch.float32)
        .unsqueeze(0)
        .expand(batch_size, -1, -1, -1)
        .clone()
    )
    return FullFrameBatch(tokens, coordinates, frames, positions)


def _ordered_inverse_batch(model_seed: int) -> FullFrameBatch:
    inverse_count = len(OFF_TORUS_PAIRS)
    batch_size = 2 * inverse_count
    length = 16
    tokens = torch.zeros(batch_size, length, dtype=torch.long)
    coordinates = torch.zeros(batch_size, length, 1, len(SPIN8_PAIRS))
    positions = torch.full((batch_size, 2), -1, dtype=torch.long)
    token_map = _token_map(model_seed)
    for pair_index in range(inverse_count):
        negative = pair_index + inverse_count
        for order_index, semantics in enumerate(
            ((pair_index, negative), (negative, pair_index))
        ):
            row = 2 * pair_index + order_index
            for position, semantic in enumerate(semantics, start=1):
                tokens[row, position] = token_map[semantic]
                coordinates[row, position] = _semantic_coordinate(semantic)
            positions[row] = torch.tensor((1, 2))
    frames = (
        _probe_bank(model_seed)
        .to(torch.float32)
        .unsqueeze(0)
        .expand(batch_size, -1, -1, -1)
        .clone()
    )
    return FullFrameBatch(tokens, coordinates, frames, positions)


def _uses_curriculum(recipe: str) -> bool:
    return recipe.endswith("/curriculum")


def _uses_block_moment(recipe: str) -> bool:
    return recipe.startswith("B-")


def _learning_rate(recipe: str, update: int) -> float:
    if recipe == "G-fixed/random":
        return 0.05
    if update <= 100:
        return 0.05
    if update <= 300:
        return 0.01
    return 0.002


def _training_batch(
    recipe: str,
    *,
    seed: int,
    update: int,
    stage: str,
    config: RepairConfig,
) -> FullFrameBatch:
    if _uses_curriculum(recipe) and update <= 100:
        return _singleton_batch(seed)
    if _uses_curriculum(recipe) and update <= 200:
        return _ordered_inverse_batch(seed)
    return generate_frame_batch(
        config.batch_size,
        config.training_length,
        seed=_stable_seed("g15ar-train", stage, seed, update),
        model_seed=seed,
        minimum_actions=2,
        maximum_actions=6,
    )


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    relative = torch.tensor(
        [value for row in rows for value in row["relative_frobenius_errors"]],
        dtype=torch.float64,
    )
    return {
        "mean_relative_frobenius_error": float(relative.mean()),
        "p95_relative_frobenius_error": float(torch.quantile(relative, 0.95)),
        "maximum_relative_frobenius_error": float(relative.max()),
        "raw_elementwise_mse": sum(row["raw_elementwise_mse"] for row in rows)
        / len(rows),
        "mean_matrix_cosine": sum(row["mean_matrix_cosine"] for row in rows)
        / len(rows),
        "minimum_matrix_cosine": min(row["minimum_matrix_cosine"] for row in rows),
    }


@torch.no_grad()
def _evaluate(
    controller: TokenCoordinateController,
    memory: torch.nn.Module,
    teacher: torch.nn.Module,
    *,
    seed: int,
    examples: int,
    specs: tuple[tuple[int, int], ...],
    prefix: str,
    device: torch.device,
) -> dict[str, dict[str, float | int]]:
    evaluations = {}
    for length, actions in specs:
        rows = []
        for offset in range(0, examples, 8):
            size = min(8, examples - offset)
            batch = generate_frame_batch(
                size,
                length,
                seed=_stable_seed(prefix, seed, length, offset),
                model_seed=seed,
                minimum_actions=actions,
                maximum_actions=actions,
            ).to(device)
            target = _teacher_target(teacher, batch, device=device)
            prediction = _frame_prediction(
                memory, batch, controller(batch.token_ids), device=device
            )
            rows.append(_metrics(prediction, target))
        evaluations[str(length)] = {
            **_aggregate(rows),
            "actions_per_episode": actions,
            "examples": examples,
        }
    return evaluations


def _build_optimizer(
    recipe: str, controller: TokenCoordinateController
) -> ScalarSecondMomentAdamW:
    optimizer_class = (
        BlockScalarSecondMomentAdamW
        if _uses_block_moment(recipe)
        else ScalarSecondMomentAdamW
    )
    return optimizer_class(controller.parameters(), lr=0.05, weight_decay=0.0)


def _train(
    arm: str,
    recipe: str,
    config: RepairConfig,
    *,
    seed: int,
    stage: str,
    device: torch.device,
    checkpoint_directory: Path,
    collect_validation_trace: bool,
) -> dict[str, Any]:
    _seed_everything(_stable_seed("g15ar-controller", stage, seed))
    controller = TokenCoordinateController().to(device)
    memory = _oracle_memory(arm, dtype=torch.float32, device=device)
    teacher = _oracle_memory("S", dtype=torch.float32, device=device)
    optimizer = _build_optimizer(recipe, controller)
    initial_hash = hashlib.sha256(
        controller.raw_coordinates.detach().cpu().numpy().tobytes()
    ).hexdigest()
    probe_hash = hashlib.sha256(
        _probe_bank(seed).detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    schedule = hashlib.sha256()
    loss_samples: dict[str, dict[str, float]] = {}
    validation_trace: dict[str, Any] = {}
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for update in range(1, config.updates + 1):
        learning_rate = _learning_rate(recipe, update)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        batch = _training_batch(
            recipe,
            seed=seed,
            update=update,
            stage=stage,
            config=config,
        )
        schedule.update(batch.fingerprint().encode())
        batch = batch.to(device)
        target = _teacher_target(teacher, batch, device=device)
        optimizer.zero_grad(set_to_none=True)
        prediction = _frame_prediction(
            memory, batch, controller(batch.token_ids), device=device
        )
        loss = F.mse_loss(prediction, target)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"non-finite G15A-R loss for {arm}/{recipe} seed {seed}"
            )
        if loss.requires_grad:
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        else:
            gradient_norm = loss.new_zeros(())
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("non-finite G15A-R gradient norm")
        optimizer.step()
        if update in {1, 100, 200, 300, 450, config.updates}:
            loss_samples[str(update)] = {
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm.detach()),
                "learning_rate": learning_rate,
                "coordinate_max_abs": float(
                    controller(batch.token_ids).detach().abs().max()
                ),
            }
        if collect_validation_trace and update in config.validation_milestones:
            validation_trace[str(update)] = _evaluate(
                controller,
                memory,
                teacher,
                seed=seed,
                examples=config.validation_examples,
                specs=config.evaluation_specs,
                prefix="g15ar-fixed-validation",
                device=device,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    evaluations = _evaluate(
        controller,
        memory,
        teacher,
        seed=seed,
        examples=config.final_examples,
        specs=config.evaluation_specs,
        prefix=f"g15ar-{stage}-final",
        device=device,
    )
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    safe_recipe = recipe.replace("/", "-")
    checkpoint = checkpoint_directory / f"g15ar_{stage}_{arm}_{safe_recipe}_{seed}.pt"
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(
        {
            "schema_version": 1,
            "arm": arm,
            "recipe": recipe,
            "seed": seed,
            "raw_coordinates": controller.raw_coordinates.detach().cpu(),
            "optimizer_state_dict": optimizer.state_dict(),
            "evaluation": evaluations,
            "training_schedule_sha256": schedule.hexdigest(),
        },
        temporary,
    )
    os.replace(temporary, checkpoint)
    result = {
        "recipe": recipe,
        "trainable_parameters": sum(
            parameter.numel() for parameter in controller.parameters()
        ),
        "structurally_unused_parameters": len(SPIN8_PAIRS),
        "initial_state_sha256": initial_hash,
        "probe_bank_sha256": probe_hash,
        "training_schedule_sha256": schedule.hexdigest(),
        "loss_samples": loss_samples,
        "validation_trace": validation_trace,
        "training_wall_seconds": training_seconds,
        "mean_synchronized_step_seconds": training_seconds / config.updates,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "evaluation": evaluations,
        "learned_raw_coordinates": controller.raw_coordinates.detach().cpu().tolist(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "optimizer": {
            "name": type(optimizer).__name__,
            "second_moment": (
                "one scalar per 28-coordinate token row"
                if _uses_block_moment(recipe)
                else "one scalar for the 17x28 coordinate tensor"
            ),
            "learning_rate_schedule": (
                "constant_0.05"
                if recipe == "G-fixed/random"
                else "0.05_updates_1_100__0.01_101_300__0.002_301_600"
            ),
            "weight_decay": 0.0,
        },
    }
    del optimizer, controller, memory, teacher
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _absolute_checks(evaluation: dict[str, Any]) -> dict[str, bool]:
    return {
        "mean_relative_error_at_most_0_05": evaluation["mean_relative_frobenius_error"]
        <= 0.05 + 1e-12,
        "p95_relative_error_at_most_0_10": evaluation["p95_relative_frobenius_error"]
        <= 0.10 + 1e-12,
        "maximum_relative_error_at_most_0_20": evaluation[
            "maximum_relative_frobenius_error"
        ]
        <= 0.20 + 1e-12,
    }


def _select_recipe(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = {}
    for recipe in DEVELOPMENT_RECIPES:
        per_seed = []
        for report in seed_reports:
            per_length = {}
            for length, evaluation in report["recipes"][recipe]["evaluation"].items():
                checks = _absolute_checks(evaluation)
                per_length[length] = {
                    "evaluation": evaluation,
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            per_seed.append(
                {
                    "seed": report["seed"],
                    "per_length": per_length,
                    "passed": all(row["passed"] for row in per_length.values()),
                }
            )
        candidates[recipe] = {
            "per_seed": per_seed,
            "qualified": all(row["passed"] for row in per_seed),
            "selectable": recipe in SELECTION_ORDER,
        }
    selected = next(
        (recipe for recipe in SELECTION_ORDER if candidates[recipe]["qualified"]),
        None,
    )
    return {
        "candidates": candidates,
        "selection_order": list(SELECTION_ORDER),
        "selected_recipe": selected,
        "passed": selected is not None,
    }


def run(
    config: RepairConfig,
    *,
    device: torch.device,
    checkpoint_directory: Path,
    commit: str,
    status_at_start: list[str],
) -> dict[str, Any]:
    started_at = _now()
    started = time.perf_counter()
    development = []
    for seed in config.development_seeds:
        recipes = {
            recipe: _train(
                "S",
                recipe,
                config,
                seed=seed,
                stage="development",
                device=device,
                checkpoint_directory=checkpoint_directory,
                collect_validation_trace=True,
            )
            for recipe in DEVELOPMENT_RECIPES
        }
        development.append({"seed": seed, "recipes": recipes})
    selection = _select_recipe(development)

    confirmation = None
    selected = selection["selected_recipe"]
    if config.mode == "quality" and selected is not None:
        confirmation_reports = []
        for seed in config.confirmation_seeds:
            arms = {
                arm: _train(
                    arm,
                    selected,
                    config,
                    seed=seed,
                    stage="confirmation",
                    device=device,
                    checkpoint_directory=checkpoint_directory,
                    collect_validation_trace=False,
                )
                for arm in ARM_NAMES
            }
            confirmation_reports.append({"seed": seed, "arms": arms})
        confirmation = {
            "seed_reports": confirmation_reports,
            "adjudication": _adjudicate(confirmation_reports),
        }
    passed = bool(
        selection["passed"]
        and confirmation is not None
        and confirmation["adjudication"]["passed"]
    )
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("g15af_full_frame_cohort.py"),
        Path(__file__).with_name("g15al_learned_coordinate_cohort.py"),
        Path(__file__).with_name("optimizers.py"),
        Path(__file__).with_name("spin_dirac_memory.py"),
        PROTOCOL,
    )
    return {
        "schema_version": 1,
        "experiment": "G15A-R first-order chart-learning repair",
        "claim_status": (
            "paired development ablation plus fresh four-transport confirmation"
        ),
        "mode": config.mode,
        "evidentiary": config.mode == "quality" and not status_at_start,
        "passed": passed,
        "started_at": started_at,
        "finished_at": _now(),
        "elapsed_wall_seconds": time.perf_counter() - started,
        "git_commit_at_start": commit,
        "git_status_at_start": status_at_start,
        "g15af_artifact_sha256": G15AF_ARTIFACT_SHA256,
        "diagnostic_artifact_sha256": DIAGNOSTIC_ARTIFACT_SHA256,
        "protocol": asdict(config),
        "protocol_file_sha256": _sha256(PROTOCOL),
        "source_files": {
            str(path.relative_to(Path(__file__).parent)): _sha256(path)
            for path in source_paths
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "dtype": "float32",
        },
        "development_seed_reports": development,
        "selection": selection,
        "confirmation": confirmation,
        "decision": (
            "G15A-R passes development selection and fresh confirmation"
            if passed
            else "G15A-R fails its frozen development or confirmation gate"
        ),
        "explicit_nonclaims": [
            "the four-probe initial frames and edit timing remain oracle supplied",
            "only the token-to-coordinate lookup is learned",
            "a curriculum winner would weaken the claim to curriculum-assisted identification",
            "no generic association, natural-text, scaling, or fused claim follows",
        ],
    }


def _load_bound(path: Path, sha256: str, experiment: str) -> dict[str, Any]:
    if _sha256(path) != sha256:
        raise RuntimeError(f"{experiment} artifact does not match the bound hash")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "quality"), required=True)
    parser.add_argument("--g15af-artifact", type=Path, required=True)
    parser.add_argument("--diagnostic-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    g15af = _load_bound(args.g15af_artifact, G15AF_ARTIFACT_SHA256, "G15A-F")
    diagnostic = _load_bound(
        args.diagnostic_artifact, DIAGNOSTIC_ARTIFACT_SHA256, "G15A-F diagnostic"
    )
    if g15af.get("adjudication", {}).get("passed") is not False:
        raise RuntimeError("bound G15A-F artifact is not the frozen failure")
    if (
        diagnostic.get("claim_status")
        != "descriptive posthoc diagnostic; no promotion gate"
    ):
        raise RuntimeError("bound diagnostic has the wrong claim status")
    config = quality_config() if args.mode == "quality" else smoke_config()
    commit, status_at_start = _git_state()
    if args.mode == "quality" and status_at_start:
        raise RuntimeError("evidentiary G15A-R requires a clean committed worktree")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the local G15A-R cohort requires CUDA")
    if torch.cuda.get_device_capability(device) != (7, 5):
        raise RuntimeError("the local G15A-R cohort requires exact SM75")
    report = run(
        config,
        device=device,
        checkpoint_directory=args.checkpoint_directory,
        commit=commit,
        status_at_start=status_at_start,
    )
    _atomic_json(args.output, report)
    print(args.output)
    print(
        json.dumps(
            {"selection": report["selection"], "passed": report["passed"]}, indent=2
        )
    )
    if args.mode == "quality" and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
