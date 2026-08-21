"""Partial-readout observability benchmark for endpoint-only Pure Spin(8).

Training retains the established noisy continuous input task and final-only
loss, but exposes only selected triality representations at the endpoint.  The
benchmark asks whether a shared Spin(8) action learned through one view transfers
to the hidden views, and contrasts that empirical question with the separate
quotient-collision impossibility certificate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

import analyze_spin8_endpoint_observability as observability
import benchmark_pure_spin8_continuous_observation as continuous
import benchmark_pure_spin8_endpoint_supervision as endpoint

ROOT = Path(__file__).resolve().parent
PROTOCOL_FROZEN_AT = "2026-08-17T06:14:38.9940486+02:00"
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "artifacts"
    / "pure_spin8_endpoint_observability_development_seed0.json"
)
DEFAULT_CHECKPOINT_DIRECTORY = (
    ROOT / "checkpoints" / "pure_spin8_endpoint_observability_development"
)
READOUTS: dict[str, tuple[int, ...]] = observability.READOUTS
CANDIDATES = ("shared_pure_spin8", "independent_so8_triplet")
CENTER_VISIBILITY_RMSE_THRESHOLD = 1e-5
FACTORIES: dict[str, Callable[[], nn.Module]] = {
    "shared_pure_spin8": continuous.SharedPureSpin8Tracker,
    "independent_so8_triplet": continuous.IndependentSO8TripletTracker,
}


def partial_endpoint_loss(
    predictions: torch.Tensor,
    selected_endpoint_targets: torch.Tensor,
    representation_indices: tuple[int, ...],
) -> torch.Tensor:
    """Final-only MSE over exactly the selected representation blocks."""

    if not representation_indices or len(set(representation_indices)) != len(
        representation_indices
    ):
        raise ValueError("representation indices must be nonempty and unique")
    if any(index < 0 or index >= len(continuous.TRIALITY_REPRESENTATIONS) for index in representation_indices):
        raise ValueError("representation index is out of range")
    expected_shape = (
        predictions.shape[0],
        len(representation_indices),
        predictions.shape[-1],
    )
    if selected_endpoint_targets.shape != expected_shape:
        raise ValueError(
            f"selected endpoint targets must have shape {expected_shape}"
        )
    return F.mse_loss(
        predictions[:, -1, representation_indices],
        selected_endpoint_targets,
    )


@torch.no_grad()
def evaluate_by_representation(
    model: nn.Module,
    batches: Sequence[continuous.ContinuousRelationBatch],
    device: torch.device,
    microbatch_size: int,
) -> dict[str, Any]:
    accumulators = {
        representation: {
            "all_squared": 0.0,
            "all_count": 0,
            "post_squared": 0.0,
            "post_count": 0,
            "final_squared": 0.0,
            "final_count": 0,
            "center_correct": 0,
            "identity_correct": 0,
            "pair_count": 0,
            "target_pair_squared": 0.0,
            "predicted_pair_squared": 0.0,
            "pair_scalar_count": 0,
        }
        for representation in continuous.TRIALITY_REPRESENTATIONS
    }
    model.eval()
    for batch in batches:
        for start in range(0, batch.observations.shape[0], microbatch_size):
            stop = start + microbatch_size
            observations = batch.observations[start:stop].to(device)
            targets = batch.targets[start:stop].to(device)
            mask = batch.post_relation_mask[start:stop].to(device)
            predictions = model(observations)
            for index, representation in enumerate(
                continuous.TRIALITY_REPRESENTATIONS
            ):
                item = accumulators[representation]
                errors = predictions[..., index, :] - targets[..., index, :]
                item["all_squared"] += float(errors.square().sum())
                item["all_count"] += errors.numel()
                selected = errors[mask]
                item["post_squared"] += float(selected.square().sum())
                item["post_count"] += selected.numel()
                final = errors[:, -1]
                item["final_squared"] += float(final.square().sum())
                item["final_count"] += final.numel()

                center_prediction = predictions[0::2, -1, index]
                identity_prediction = predictions[1::2, -1, index]
                center_target = targets[0::2, -1, index]
                identity_target = targets[1::2, -1, index]
                target_difference = center_target - identity_target
                item["target_pair_squared"] += float(target_difference.square().sum())
                item["predicted_pair_squared"] += float(
                    (center_prediction - identity_prediction).square().sum()
                )
                item["pair_scalar_count"] += target_difference.numel()
                target_pair_rmse = float(target_difference.square().mean().sqrt())
                if target_pair_rmse > CENTER_VISIBILITY_RMSE_THRESHOLD:
                    item["center_correct"] += int(
                        (
                            (center_prediction - center_target)
                            .square()
                            .sum(dim=-1)
                            < (center_prediction - identity_target)
                            .square()
                            .sum(dim=-1)
                        ).sum()
                    )
                    item["identity_correct"] += int(
                        (
                            (identity_prediction - identity_target)
                            .square()
                            .sum(dim=-1)
                            < (identity_prediction - center_target)
                            .square()
                            .sum(dim=-1)
                        ).sum()
                    )
                    item["pair_count"] += center_prediction.shape[0]
    result = {}
    for representation, item in accumulators.items():
        pair_count = item["pair_count"]
        result[representation] = {
            "all_prefix_mse": item["all_squared"] / item["all_count"],
            "post_relation_mse": item["post_squared"] / item["post_count"],
            "final_mse": item["final_squared"] / item["final_count"],
            "target_center_identity_rmse": math.sqrt(
                item["target_pair_squared"] / item["pair_scalar_count"]
            ),
            "predicted_center_identity_rmse": math.sqrt(
                item["predicted_pair_squared"] / item["pair_scalar_count"]
            ),
            "center_visible_in_target": pair_count > 0,
            "center_classification_accuracy": (
                (item["center_correct"] + item["identity_correct"])
                / (2 * pair_count)
                if pair_count
                else None
            ),
            "center_rows_correct": (
                item["center_correct"] / pair_count if pair_count else None
            ),
            "identity_rows_correct": (
                item["identity_correct"] / pair_count if pair_count else None
            ),
        }
    return result


@torch.no_grad()
def action_rmse_by_representation(
    model: nn.Module,
    evaluations: dict[str, list[continuous.ContinuousRelationBatch]],
    device: torch.device,
) -> dict[str, float]:
    squared = torch.zeros(len(continuous.TRIALITY_REPRESENTATIONS), dtype=torch.float64)
    counts = torch.zeros_like(squared)
    model.eval()
    for batches in evaluations.values():
        batch = batches[0]
        observations = batch.observations.to(device)
        predicted = model.observation_actions(observations)
        target = continuous.action_table_from_coordinates(
            batch.coordinates, device=device
        )
        errors = (predicted - target).double()
        squared += errors.square().sum(dim=(0, 1, 3, 4)).cpu()
        counts += torch.tensor(
            [errors[..., index, :, :].numel() for index in range(3)],
            dtype=torch.float64,
        )
    return {
        representation: math.sqrt(float(squared[index] / counts[index]))
        for index, representation in enumerate(continuous.TRIALITY_REPRESENTATIONS)
    }


def train_candidate(
    name: str,
    readout_name: str,
    representation_indices: tuple[int, ...],
    schedule: Sequence[endpoint.EndpointTrainingBatch],
    evaluations: dict[str, list[continuous.ContinuousRelationBatch]],
    config: continuous.ContinuousObservationConfig,
    device: torch.device,
    checkpoint_directory: Path | None,
) -> dict[str, Any]:
    model = FACTORIES[name]().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    samples = {}
    gradient_norm = torch.tensor(0.0)
    started = time.perf_counter()
    model.train()
    for step, batch in enumerate(schedule, start=1):
        observations = batch.observations.to(device)
        # Slice on CPU before transfer: hidden endpoint blocks never enter the
        # candidate's accelerator-side training step or loss function.
        targets = batch.endpoint_targets[:, representation_indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(observations)
        loss = partial_endpoint_loss(predictions, targets, representation_indices)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"{name}/{readout_name} produced nonfinite loss at step {step}"
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == config.steps:
            samples[str(step)] = float(loss.detach())
            print(
                f"{name} readout={readout_name} seed={config.seed} "
                f"step={step}/{config.steps} loss={samples[str(step)]:.8f}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    evaluation = {
        key: evaluate_by_representation(
            model, batches, device, config.evaluation_microbatch_size
        )
        for key, batches in evaluations.items()
    }
    result: dict[str, Any] = {
        "parameters": continuous.parameter_count(model),
        "recurrent_state_scalars": int(model.recurrent_state_scalars),
        "supervised_representations": [
            continuous.TRIALITY_REPRESENTATIONS[index]
            for index in representation_indices
        ],
        "supervised_scalars_per_sequence": 8 * len(representation_indices),
        "loss_samples": samples,
        "final_training_loss": samples[str(config.steps)],
        "last_preclip_gradient_norm": float(gradient_norm),
        "training_wall_seconds": elapsed,
        "training_tokens_per_second": (
            config.steps * config.batch_size * config.training_length / elapsed
        ),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "evaluation": evaluation,
        "action_rmse_by_representation": action_rmse_by_representation(
            model, evaluations, device
        ),
    }
    if checkpoint_directory is not None:
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_directory / (
            f"{name}_{readout_name}_endpoint_seed{config.seed}_step{config.steps}.pt"
        )
        torch.save(
            {
                "format_version": 1,
                "candidate": name,
                "readout": readout_name,
                "supervised_representation_indices": representation_indices,
                "training_supervision": "partial_endpoint_only_signed_state",
                "pure_spin8_version": (
                    continuous.PURE_SPIN8_VERSION
                    if name == "shared_pure_spin8"
                    else None
                ),
                "config": asdict(config),
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "result": result,
            },
            checkpoint,
        )
        result["checkpoint"] = str(checkpoint)
        result["checkpoint_sha256"] = hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest()
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _all_numeric_values_finite(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_numeric_values_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numeric_values_finite(item) for item in value)
    return True


def run_benchmark(
    config: continuous.ContinuousObservationConfig,
    *,
    device: torch.device,
    readouts: tuple[str, ...],
    candidates: tuple[str, ...],
    checkpoint_directory: Path | None,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if not readouts or len(set(readouts)) != len(readouts):
        raise ValueError("readouts must be nonempty and unique")
    unknown_readouts = set(readouts) - set(READOUTS)
    if unknown_readouts:
        raise ValueError(f"unknown readouts: {sorted(unknown_readouts)}")
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("candidates must be nonempty and unique")
    unknown_candidates = set(candidates) - set(CANDIDATES)
    if unknown_candidates:
        raise ValueError(f"unknown candidates: {sorted(unknown_candidates)}")

    teacher_contract = continuous.teacher_contract(device)
    if not teacher_contract["passed"]:
        raise RuntimeError("teacher relation contract failed")
    exact_certificate = observability.build_certificate()
    if not exact_certificate["passed"]:
        raise RuntimeError("endpoint observability certificate failed")
    system = continuous.make_observation_system(config.seed)
    schedule = endpoint.make_endpoint_training_schedule(config, system, device)
    split = endpoint.endpoint_training_split_audit(schedule)
    if not split["passed"]:
        raise RuntimeError("partial-readout training split audit failed")

    evaluations = {}
    evaluation_audits = {}
    evaluation_hashes = {}
    for length in config.evaluation_lengths:
        for position in ("early", "late"):
            key = f"{position}_L{length}"
            batch = continuous.make_relation_batch(
                config, system, length, position, device
            )
            evaluations[key] = [batch]
            evaluation_audits[key] = continuous.relation_batch_audit(batch)
            evaluation_hashes[key] = continuous.tensor_hash(
                (
                    batch.observations,
                    batch.targets,
                    batch.coordinates,
                    batch.post_relation_mask,
                )
            )
    if not all(audit["passed"] for audit in evaluation_audits.values()):
        raise RuntimeError("partial-readout evaluation split audit failed")

    shapes = continuous.build_models()
    counts = {name: continuous.parameter_count(shapes[name]) for name in candidates}
    states = {name: int(shapes[name].recurrent_state_scalars) for name in candidates}
    del shapes

    results = {}
    for readout_name in readouts:
        results[readout_name] = {}
        for name in candidates:
            candidate_offset = CANDIDATES.index(name)
            # Identical candidate initialization across readouts isolates the
            # supervision mask; candidates retain their established seed offsets.
            continuous.seed_everything(
                1_060_000 + 1_000 * config.seed + candidate_offset
            )
            results[readout_name][name] = train_candidate(
                name,
                readout_name,
                READOUTS[readout_name],
                schedule,
                evaluations,
                config,
                device,
                checkpoint_directory,
            )
    all_finite = _all_numeric_values_finite(results)
    return {
        "schema_version": 1,
        "experiment": "Pure Spin8 endpoint partial-readout observability",
        "status": "development" if config.seed == 0 else "unadjudicated",
        "protocol_frozen_at": PROTOCOL_FROZEN_AT,
        "recorded_at": continuous.now(),
        "pure_spin8_version": continuous.PURE_SPIN8_VERSION,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor()
            ),
            "torch_cpu_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "config": asdict(config),
        "readouts": {
            name: {
                "indices": READOUTS[name],
                "representations": [
                    continuous.TRIALITY_REPRESENTATIONS[index]
                    for index in READOUTS[name]
                ],
                "supervised_scalars_per_sequence": 8 * len(READOUTS[name]),
            }
            for name in readouts
        },
        "candidates": candidates,
        "task": {
            "training_supervision": "selected final representation blocks only",
            "intermediate_prefix_targets_retained_by_training_schedule": False,
            "observation_system_sha256": continuous.tensor_hash(
                (system.projection, system.bias)
            ),
            "teacher_contract": teacher_contract,
            "training_split": split,
            "evaluation_audits": evaluation_audits,
            "evaluation_schedule_sha256": evaluation_hashes,
        },
        "observability_certificate": exact_certificate,
        "integrity": {
            "same_precomputed_schedule_for_every_readout_and_candidate": True,
            "candidate_initialization_identical_across_readouts": True,
            "coordinates_and_events_never_passed_to_models": True,
            "hidden_endpoint_blocks_sliced_before_device_transfer": True,
            "parameter_counts": counts,
            "recurrent_state_scalars": states,
            "all_metrics_finite": all_finite,
        },
        "results": results,
        "claim_scope": {
            "empirical": [
                "partial signed endpoint readout transfer on one synthetic teacher family",
                "same candidate initialization and schedule across readout masks",
            ],
            "exact_or_information_theoretic": [
                "single-view seven-probe local Lie rank",
                "tested-center visibility by representation",
                "balanced quotient-input hidden-lift Bayes lower bound",
            ],
            "not_claimed": [
                "global identifiability from one representation",
                "unsigned hidden-lift recovery from quotient observations",
                "natural-data or language-model superiority",
            ],
        },
        "passed": bool(
            teacher_contract["passed"]
            and split["passed"]
            and exact_certificate["passed"]
            and all_finite
        ),
    }


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-length", type=int, default=16)
    parser.add_argument("--evaluation-pairs", type=int, default=64)
    parser.add_argument("--evaluation-lengths", default="16,64,128")
    parser.add_argument("--evaluation-microbatch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--observation-noise-std", type=float, default=0.01)
    parser.add_argument("--half-center-probability", type=float, default=0.12)
    parser.add_argument("--regular-coordinate-std", type=float, default=0.32)
    parser.add_argument("--half-center-delta", type=float, default=0.25)
    parser.add_argument("--readouts", default=",".join(READOUTS))
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--checkpoint-directory", type=Path, default=DEFAULT_CHECKPOINT_DIRECTORY
    )
    parser.add_argument("--no-checkpoints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = continuous.ContinuousObservationConfig(
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        training_length=args.training_length,
        evaluation_pairs=args.evaluation_pairs,
        evaluation_lengths=continuous.parse_lengths(args.evaluation_lengths),
        evaluation_microbatch_size=args.evaluation_microbatch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        observation_noise_std=args.observation_noise_std,
        half_center_probability=args.half_center_probability,
        regular_coordinate_std=args.regular_coordinate_std,
        half_center_delta=args.half_center_delta,
    )
    result = run_benchmark(
        config,
        device=torch.device(args.device),
        readouts=parse_csv(args.readouts),
        candidates=parse_csv(args.candidates),
        checkpoint_directory=(
            None if args.no_checkpoints else args.checkpoint_directory
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
