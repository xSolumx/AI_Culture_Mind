"""Endpoint-only noisy continuous-observation Spin(8) identification.

This runner changes one axis of the validated continuous-observation task:
training batches contain only the final signed triality target.  No intermediate
prefix target is retained by the training schedule or read by the loss.  The
observation chart, held-out adjacent center relation, candidate architectures,
and long-context evaluation remain the same.

The purpose is to test credit assignment and local-action identifiability, not
to claim natural-data or language-model performance.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import benchmark_pure_spin8_continuous_observation as continuous

DEVELOPMENT_STARTED_AT = "2026-08-17T04:54:00+02:00"
DEVELOPMENT_TIMESTAMP_CORRECTED_AT = "2026-08-17T05:01:31+02:00"
PROTOCOL_FROZEN_AT = "2026-08-17T05:02:15+02:00"
ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "artifacts"
    / "pure_spin8_endpoint_supervision_development_seed0.json"
)
DEFAULT_CHECKPOINT_DIRECTORY = (
    ROOT / "checkpoints" / "pure_spin8_endpoint_supervision_development"
)

FACTORIES: dict[str, Callable[[], nn.Module]] = {
    "shared_pure_spin8": continuous.SharedPureSpin8Tracker,
    "independent_so8_triplet": continuous.IndependentSO8TripletTracker,
    "mamba2_parameter_near": continuous.ContinuousMamba2Tracker,
    "gru_parameter_near": continuous.ParameterNearGRUTracker,
    "observation_only_ablation": continuous.ObservationOnlyAblation,
    "gru_state_matched": continuous.StateMatchedGRUTracker,
}


@dataclass(frozen=True)
class EndpointTrainingBatch:
    observations: torch.Tensor
    endpoint_targets: torch.Tensor
    coordinates: torch.Tensor
    events: torch.Tensor


def endpoint_loss(
    predictions: torch.Tensor, endpoint_targets: torch.Tensor
) -> torch.Tensor:
    """MSE whose gradient support is restricted to the final model output."""

    return F.mse_loss(predictions[:, -1], endpoint_targets)


def make_endpoint_training_schedule(
    config: continuous.ContinuousObservationConfig,
    system: continuous.ObservationSystem,
    device: torch.device,
) -> list[EndpointTrainingBatch]:
    coordinate_generator = np.random.default_rng(820_000 + config.seed)
    noise_generator = torch.Generator().manual_seed(830_000 + config.seed)
    schedule = []
    for _ in range(config.steps):
        coordinates, events = continuous._sample_coordinate_sequence(
            coordinate_generator,
            config.batch_size,
            config.training_length,
            config,
        )
        observations = continuous.observe_coordinates(
            coordinates,
            system,
            noise_std=config.observation_noise_std,
            generator=noise_generator,
        )
        endpoint_targets = continuous.teacher_outputs(coordinates, device)[:, -1]
        schedule.append(
            EndpointTrainingBatch(
                observations=observations,
                endpoint_targets=endpoint_targets,
                coordinates=coordinates,
                events=events,
            )
        )
    return schedule


def endpoint_training_split_audit(
    schedule: Sequence[EndpointTrainingBatch],
) -> dict[str, Any]:
    events = torch.cat([batch.events for batch in schedule])
    observations = torch.cat([batch.observations for batch in schedule])
    coordinates = torch.cat([batch.coordinates for batch in schedule])
    endpoints = torch.cat([batch.endpoint_targets for batch in schedule])
    adjacent_half = int(
        (
            (events[:, :-1] == continuous.HALF_CENTER_EVENT)
            & (events[:, 1:] == continuous.HALF_CENTER_EVENT)
        )
        .sum()
        .item()
    )
    flattened = observations.reshape(-1, continuous.OBSERVATION_DIMENSION).numpy()
    row_bytes = np.ascontiguousarray(flattened).view(
        np.dtype((np.void, flattened.dtype.itemsize * flattened.shape[1]))
    )
    unique_observations = int(np.unique(row_bytes).size)
    half_values = coordinates[..., 0][events == continuous.HALF_CENTER_EVENT]
    batch = schedule[0]
    checks = {
        "held_out_adjacent_half_center_count_zero": adjacent_half == 0,
        "half_center_events_present": int((events == continuous.HALF_CENTER_EVENT).sum())
        > 0,
        "regular_events_present": int((events == continuous.REGULAR_EVENT).sum()) > 0,
        "half_center_delta_spans_both_signs": bool(
            (half_values < math.pi).any() and (half_values > math.pi).any()
        ),
        "every_observation_is_unique": unique_observations == flattened.shape[0],
        "observations_are_finite": bool(torch.isfinite(observations).all()),
        "endpoint_targets_are_finite": bool(torch.isfinite(endpoints).all()),
        "endpoint_target_shape_exact": batch.endpoint_targets.shape[1:] == (3, 8),
        "no_intermediate_targets_in_training_batch": not hasattr(batch, "targets"),
    }
    return {
        "schedule_sha256": continuous.tensor_hash(
            [
                value
                for item in schedule
                for value in (
                    item.observations,
                    item.endpoint_targets,
                    item.coordinates,
                    item.events,
                )
            ]
        ),
        "observation_count": int(flattened.shape[0]),
        "unique_observation_count": unique_observations,
        "supervised_endpoint_count": int(endpoints.shape[0]),
        "supervised_scalars_per_sequence": int(endpoints[0].numel()),
        "retained_intermediate_target_count": 0,
        "half_center_event_count": int((events == continuous.HALF_CENTER_EVENT).sum()),
        "regular_event_count": int((events == continuous.REGULAR_EVENT).sum()),
        "held_out_adjacent_half_center_count": adjacent_half,
        "half_center_coordinate_range": [
            float(half_values.min()),
            float(half_values.max()),
        ],
        "checks": checks,
        "passed": all(checks.values()),
    }


def train_endpoint_candidate(
    name: str,
    factory: Callable[[], nn.Module],
    schedule: Sequence[EndpointTrainingBatch],
    evaluations: dict[str, list[continuous.ContinuousRelationBatch]],
    config: continuous.ContinuousObservationConfig,
    device: torch.device,
    checkpoint_directory: Path | None,
) -> dict[str, Any]:
    model = factory().to(device)
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
        targets = batch.endpoint_targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(observations)
        loss = endpoint_loss(predictions, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name} produced nonfinite loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == config.steps:
            samples[str(step)] = float(loss.detach())
            print(
                f"{name} endpoint seed={config.seed} step={step}/{config.steps} "
                f"loss={samples[str(step)]:.8f}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result: dict[str, Any] = {
        "parameters": continuous.parameter_count(model),
        "recurrent_state_scalars": int(model.recurrent_state_scalars),
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
        "evaluation": {
            key: continuous.evaluate_relation_batches(
                model, batches, device, config.evaluation_microbatch_size
            )
            for key, batches in evaluations.items()
        },
    }
    if isinstance(
        model,
        (
            continuous.SharedPureSpin8Tracker,
            continuous.IndependentSO8TripletTracker,
        ),
    ):
        result["action_identification"] = continuous.action_identification_diagnostics(
            model, evaluations, device
        )
    if checkpoint_directory is not None:
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        checkpoint = (
            checkpoint_directory
            / f"{name}_endpoint_seed{config.seed}_step{config.steps}.pt"
        )
        torch.save(
            {
                "format_version": 1,
                "candidate": name,
                "training_supervision": "endpoint_only_signed_triality_state",
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


def run_benchmark(
    config: continuous.ContinuousObservationConfig,
    *,
    device: torch.device,
    candidates: tuple[str, ...],
    checkpoint_directory: Path | None,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("candidate subset must be nonempty and unique")
    unknown = set(candidates) - set(continuous.CANDIDATES)
    if unknown:
        raise ValueError(f"unknown candidates: {sorted(unknown)}")

    contract = continuous.teacher_contract(device)
    if not contract["passed"]:
        raise RuntimeError("teacher relation contract failed")
    system = continuous.make_observation_system(config.seed)
    schedule = make_endpoint_training_schedule(config, system, device)
    split = endpoint_training_split_audit(schedule)
    if not split["passed"]:
        raise RuntimeError("endpoint-only training split audit failed")

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
        raise RuntimeError("endpoint-only evaluation split audit failed")

    shapes = continuous.build_models()
    counts = {name: continuous.parameter_count(model) for name, model in shapes.items()}
    states = {name: int(model.recurrent_state_scalars) for name, model in shapes.items()}
    del shapes

    results = {}
    for name in candidates:
        offset = continuous.CANDIDATES.index(name)
        continuous.seed_everything(960_000 + 1_000 * config.seed + offset)
        results[name] = train_endpoint_candidate(
            name,
            FACTORIES[name],
            schedule,
            evaluations,
            config,
            device,
            checkpoint_directory,
        )
    all_finite = all(
        math.isfinite(value)
        for result in results.values()
        for metrics in result["evaluation"].values()
        for value in metrics.values()
    )
    return {
        "schema_version": 1,
        "experiment": "endpoint-only noisy continuous-observation Pure Spin8 identification",
        "status": "development" if config.seed == 0 else "unadjudicated",
        "development_started_at": DEVELOPMENT_STARTED_AT,
        "development_timestamp_corrected_at": DEVELOPMENT_TIMESTAMP_CORRECTED_AT,
        "protocol_frozen_at": PROTOCOL_FROZEN_AT,
        "recorded_at": continuous.now(),
        "pure_spin8_version": continuous.PURE_SPIN8_VERSION,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor()
            ),
            "mamba2_backend": "huggingface_transformers_naive_fallback",
            "torch_cpu_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "config": asdict(config),
        "candidates": candidates,
        "task": {
            "training_supervision": "final signed 24-real triality state only",
            "intermediate_prefix_targets_retained_by_training_schedule": False,
            "observation_system_sha256": continuous.tensor_hash(
                (system.projection, system.bias)
            ),
            "teacher_contract": contract,
            "training_split": split,
            "evaluation_audits": evaluation_audits,
            "evaluation_schedule_sha256": evaluation_hashes,
            "held_out_relation": (
                "two adjacent fresh observations with plane-0 coordinates "
                "pi+delta and pi-delta"
            ),
        },
        "integrity": {
            "same_precomputed_schedule_for_every_candidate": True,
            "candidate_initialized_after_seed": True,
            "parameter_counts": counts,
            "recurrent_state_scalars": states,
            "all_metrics_finite": all_finite,
        },
        "results": results,
        "claim_scope": {
            "empirical": [
                "endpoint-only synthetic group-action credit assignment",
                "no intermediate prefix targets retained in training batches",
            ],
            "not_claimed": [
                "a frozen or replicated result before an explicit protocol",
                "natural-data, unsigned-state, or sparse input supervision",
                "all-28-coordinate Spin8 action identification",
                "fused-Mamba or language-model superiority",
            ],
        },
        "passed": bool(contract["passed"] and split["passed"] and all_finite),
    }


def parse_candidates(value: str) -> tuple[str, ...]:
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
    parser.add_argument("--observation-noise-std", type=float, default=0.01)
    parser.add_argument("--half-center-probability", type=float, default=0.12)
    parser.add_argument("--regular-coordinate-std", type=float, default=0.40)
    parser.add_argument("--half-center-delta", type=float, default=0.25)
    parser.add_argument(
        "--candidates",
        default="shared_pure_spin8,independent_so8_triplet",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIRECTORY,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = continuous.ContinuousObservationConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        training_length=args.training_length,
        evaluation_pairs=args.evaluation_pairs,
        evaluation_lengths=continuous.parse_lengths(args.evaluation_lengths),
        evaluation_microbatch_size=args.evaluation_microbatch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        observation_noise_std=args.observation_noise_std,
        half_center_probability=args.half_center_probability,
        regular_coordinate_std=args.regular_coordinate_std,
        half_center_delta=args.half_center_delta,
        seed=args.seed,
    )
    report = run_benchmark(
        config,
        device=torch.device(args.device),
        candidates=parse_candidates(args.candidates),
        checkpoint_directory=args.checkpoint_directory,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
