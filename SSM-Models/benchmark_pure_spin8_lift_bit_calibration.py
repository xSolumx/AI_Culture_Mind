"""Minimal one-bit calibration for vector-only Pure Spin(8) endpoint learning.

The vector endpoint is invariant under the tested central element.  This runner
adds one binary measurement of the positive-spinor endpoint,
``sign(<e_0, y_+>)``, and tests whether that information-theoretically minimal
fiber label closes the exact hidden-lift failures of vector-only supervision.
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

import benchmark_pure_spin8_continuous_observation as continuous
import benchmark_pure_spin8_endpoint_observability as observability_benchmark
import benchmark_pure_spin8_endpoint_supervision as endpoint
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
PROTOCOL_FROZEN_AT = "2026-08-17T07:05:22.8794763+02:00"
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "artifacts"
    / "pure_spin8_lift_bit_calibration_development_seed0.json"
)
DEFAULT_CHECKPOINT_DIRECTORY = (
    ROOT / "checkpoints" / "pure_spin8_lift_bit_calibration_development"
)
CANDIDATES = observability_benchmark.CANDIDATES
FACTORIES: dict[str, Callable[[], nn.Module]] = observability_benchmark.FACTORIES
MODES = (
    "vector_only",
    "vector_plus_positive_bit",
    "vector_plus_adaptive_lift_bit",
    "positive_only",
    "full_triality",
)
POSITIVE_PROBE_INDEX = 0
BIT_LOGIT_SCALE = 8.0
BIT_LOSS_WEIGHT = 0.10


def lift_bit_from_positive_endpoint(endpoint: torch.Tensor) -> torch.Tensor:
    """Return the one-bit hemisphere label without moving hidden blocks."""

    if endpoint.shape[-2:] != (3, 8):
        raise ValueError("endpoint must have final shape (3,8)")
    return endpoint[..., 1, POSITIVE_PROBE_INDEX] >= 0


def adaptive_lift_address_and_bit(
    endpoint: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a lift-invariant chart address and its one lift-selecting bit."""

    if endpoint.shape[-2:] != (3, 8):
        raise ValueError("endpoint must have final shape (3,8)")
    positive = endpoint[..., 1, :]
    address = positive.abs().argmax(dim=-1)
    selected = positive.gather(-1, address.unsqueeze(-1)).squeeze(-1)
    return address, selected >= 0


def vector_plus_bit_loss(
    predictions: torch.Tensor,
    vector_targets: torch.Tensor,
    bit_targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Vector MSE plus a single binary half-spin calibration measurement."""

    expected_vector_shape = (predictions.shape[0], predictions.shape[-1])
    if vector_targets.shape != expected_vector_shape:
        raise ValueError(f"vector targets must have shape {expected_vector_shape}")
    if bit_targets.shape != (predictions.shape[0],):
        raise ValueError("bit targets must have shape (batch,)")
    vector_loss = F.mse_loss(predictions[:, -1, 0], vector_targets)
    bit_logits = (
        BIT_LOGIT_SCALE * predictions[:, -1, 1, POSITIVE_PROBE_INDEX]
    )
    bit_loss = F.binary_cross_entropy_with_logits(
        bit_logits, bit_targets.to(dtype=bit_logits.dtype)
    )
    total = vector_loss + BIT_LOSS_WEIGHT * bit_loss
    return total, {
        "vector_mse": vector_loss,
        "bit_bce": bit_loss,
        "bit_accuracy": (
            (bit_logits >= 0) == bit_targets
        ).to(dtype=torch.float32).mean(),
    }


def vector_plus_adaptive_bit_loss(
    predictions: torch.Tensor,
    vector_targets: torch.Tensor,
    addresses: torch.Tensor,
    bit_targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Vector MSE plus one sign bit in a robust adaptive spinor chart."""

    expected_vector_shape = (predictions.shape[0], predictions.shape[-1])
    if vector_targets.shape != expected_vector_shape:
        raise ValueError(f"vector targets must have shape {expected_vector_shape}")
    if addresses.shape != (predictions.shape[0],) or addresses.dtype != torch.long:
        raise ValueError("addresses must be int64 with shape (batch,)")
    if bit_targets.shape != (predictions.shape[0],):
        raise ValueError("bit targets must have shape (batch,)")
    vector_loss = F.mse_loss(predictions[:, -1, 0], vector_targets)
    positive_prediction = predictions[:, -1, 1]
    selected_prediction = positive_prediction.gather(
        -1, addresses.unsqueeze(-1)
    ).squeeze(-1)
    bit_logits = BIT_LOGIT_SCALE * selected_prediction
    bit_loss = F.binary_cross_entropy_with_logits(
        bit_logits, bit_targets.to(dtype=bit_logits.dtype)
    )
    total = vector_loss + BIT_LOSS_WEIGHT * bit_loss
    return total, {
        "vector_mse": vector_loss,
        "adaptive_bit_bce": bit_loss,
        "adaptive_bit_accuracy": (
            (bit_logits >= 0) == bit_targets
        ).to(dtype=torch.float32).mean(),
    }


def mode_loss(
    mode: str,
    predictions: torch.Tensor,
    endpoint_targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Slice every target on CPU before the selected tensors reach the device."""

    if mode == "vector_plus_positive_bit":
        vector_targets = endpoint_targets[:, 0].to(predictions.device)
        bit_targets = lift_bit_from_positive_endpoint(endpoint_targets).to(
            predictions.device
        )
        return vector_plus_bit_loss(predictions, vector_targets, bit_targets)
    if mode == "vector_plus_adaptive_lift_bit":
        vector_targets = endpoint_targets[:, 0].to(predictions.device)
        addresses, bit_targets = adaptive_lift_address_and_bit(endpoint_targets)
        return vector_plus_adaptive_bit_loss(
            predictions,
            vector_targets,
            addresses.to(predictions.device),
            bit_targets.to(predictions.device),
        )
    if mode == "vector_only":
        indices = (0,)
    elif mode == "positive_only":
        indices = (1,)
    elif mode == "full_triality":
        indices = (0, 1, 2)
    else:
        raise ValueError(f"unknown calibration mode: {mode}")
    selected = endpoint_targets[:, indices].to(predictions.device)
    loss = observability_benchmark.partial_endpoint_loss(
        predictions, selected, indices
    )
    return loss, {"selected_endpoint_mse": loss}


def exact_one_bit_certificate() -> dict[str, Any]:
    """Return the exact fiber argument and a deterministic numerical witness."""

    y = torch.tensor(
        [1, -2, 3, -4, 5, -6, 7, -8], dtype=torch.int64
    )
    positive_bit = bool(y[POSITIVE_PROBE_INDEX] >= 0)
    negative_bit = bool((-y)[POSITIVE_PROBE_INDEX] >= 0)
    checks = {
        "double_cover_fiber_has_two_elements": True,
        "binary_measurement_has_two_values": True,
        "witness_bits_are_opposite": positive_bit != negative_bit,
        "probe_is_nonzero_on_witness": int(y[POSITIVE_PROBE_INDEX]) != 0,
        "one_bit_meets_log2_fiber_lower_bound": math.ceil(math.log2(2)) == 1,
        "adaptive_address_is_antipodally_invariant": int(y.abs().argmax())
        == int((-y).abs().argmax()),
        "adaptive_sign_bit_flips": bool(
            y[int(y.abs().argmax())] >= 0
        )
        != bool((-y)[int(y.abs().argmax())] >= 0),
        "unit_spinor_adaptive_margin_lower_bound": math.isclose(
            1.0 / math.sqrt(8.0), math.sqrt(1.0 / 8.0)
        ),
    }
    return {
        "fiber": "{y,-y}",
        "measurement": "sign(<e_0,y_+>)",
        "exceptional_set": "<e_0,y_+>=0, a codimension-one hyperplane",
        "minimum_binary_measurements": 1,
        "robust_adaptive_chart": {
            "address": "argmax_j |y_j|, invariant under y -> -y",
            "address_bits": 3,
            "external_lift_bits_given_address": 1,
            "unit_state_margin_lower_bound": 1.0 / math.sqrt(8.0),
        },
        "proof": (
            "For nonzero l(y), l(-y)=-l(y), so the sign bit differs on the two "
            "lifts. A two-element fiber needs at least ceil(log2(2))=1 bit."
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }


def lift_bit_training_audit(
    schedule: Sequence[endpoint.EndpointTrainingBatch],
) -> dict[str, Any]:
    endpoints = torch.cat([batch.endpoint_targets for batch in schedule])
    scalars = endpoints[:, 1, POSITIVE_PROBE_INDEX]
    bits = scalars >= 0
    absolute = scalars.abs()
    quantiles = torch.quantile(
        absolute.float(), torch.tensor([0.0, 0.001, 0.01, 0.05, 0.5])
    )
    positive_fraction = float(bits.float().mean())
    checks = {
        "both_bit_values_present": bool(bits.any() and (~bits).any()),
        "bit_balance_between_45_and_55_percent": 0.45 <= positive_fraction <= 0.55,
        "no_exact_probe_zero": int((scalars == 0).sum()) == 0,
        "all_probe_scalars_finite": bool(torch.isfinite(scalars).all()),
    }
    return {
        "probe_representation": "positive",
        "probe_index": POSITIVE_PROBE_INDEX,
        "endpoint_count": int(endpoints.shape[0]),
        "positive_bit_fraction": positive_fraction,
        "negative_bit_fraction": 1.0 - positive_fraction,
        "exact_zero_count": int((scalars == 0).sum()),
        "absolute_probe_quantiles": {
            name: float(value)
            for name, value in zip(
                ("min", "q0.001", "q0.01", "q0.05", "median"),
                quantiles,
                strict=True,
            )
        },
        "bit_sha256": continuous.tensor_hash((bits,)),
        "checks": checks,
        "passed": all(checks.values()),
    }


def adaptive_lift_training_audit(
    schedule: Sequence[endpoint.EndpointTrainingBatch],
) -> dict[str, Any]:
    endpoints = torch.cat([batch.endpoint_targets for batch in schedule])
    addresses, bits = adaptive_lift_address_and_bit(endpoints)
    positive = endpoints[:, 1]
    selected = positive.gather(-1, addresses.unsqueeze(-1)).squeeze(-1)
    minimum_margin = float(selected.abs().min())
    theoretical_margin = 1.0 / math.sqrt(8.0)
    counts = torch.bincount(addresses, minlength=8)
    positive_fraction = float(bits.float().mean())
    checks = {
        "all_eight_chart_addresses_present": bool((counts > 0).all()),
        "both_bit_values_present": bool(bits.any() and (~bits).any()),
        "bit_balance_between_45_and_55_percent": 0.45 <= positive_fraction <= 0.55,
        "observed_margin_respects_unit_sphere_bound": minimum_margin
        >= theoretical_margin - 2e-5,
        "no_chart_tie_at_recorded_precision": bool(
            (
                positive.abs().topk(2, dim=-1).values[:, 0]
                > positive.abs().topk(2, dim=-1).values[:, 1]
            ).all()
        ),
    }
    return {
        "address_rule": "argmax absolute positive-spinor endpoint coordinate",
        "address_counts": counts.tolist(),
        "positive_bit_fraction": positive_fraction,
        "minimum_selected_absolute_probe": minimum_margin,
        "theoretical_unit_state_margin": theoretical_margin,
        "address_sha256": continuous.tensor_hash((addresses,)),
        "bit_sha256": continuous.tensor_hash((bits,)),
        "checks": checks,
        "passed": all(checks.values()),
    }


@torch.no_grad()
def evaluate_lift_bit(
    model: nn.Module,
    batches: Sequence[continuous.ContinuousRelationBatch],
    device: torch.device,
    microbatch_size: int,
) -> dict[str, Any]:
    correct = 0
    count = 0
    pair_opposite = 0
    pair_count = 0
    minimum_absolute_target_probe = math.inf
    model.eval()
    for batch in batches:
        for start in range(0, batch.observations.shape[0], microbatch_size):
            stop = start + microbatch_size
            observations = batch.observations[start:stop].to(device)
            target_endpoint = batch.targets[start:stop, -1]
            predictions = model(observations)
            predicted_bits = (
                predictions[:, -1, 1, POSITIVE_PROBE_INDEX] >= 0
            ).cpu()
            target_bits = lift_bit_from_positive_endpoint(target_endpoint)
            correct += int((predicted_bits == target_bits).sum())
            count += target_bits.numel()
            pair_opposite += int((target_bits[0::2] != target_bits[1::2]).sum())
            pair_count += target_bits[0::2].numel()
            minimum_absolute_target_probe = min(
                minimum_absolute_target_probe,
                float(target_endpoint[:, 1, POSITIVE_PROBE_INDEX].abs().min()),
            )
    return {
        "positive_lift_bit_accuracy": correct / count,
        "target_center_identity_bits_opposite_fraction": pair_opposite / pair_count,
        "minimum_absolute_target_probe": minimum_absolute_target_probe,
    }


@torch.no_grad()
def evaluate_adaptive_lift_bit(
    model: nn.Module,
    batches: Sequence[continuous.ContinuousRelationBatch],
    device: torch.device,
    microbatch_size: int,
) -> dict[str, Any]:
    correct = 0
    count = 0
    pair_same_address = 0
    pair_opposite_bit = 0
    pair_count = 0
    minimum_margin = math.inf
    model.eval()
    for batch in batches:
        for start in range(0, batch.observations.shape[0], microbatch_size):
            stop = start + microbatch_size
            observations = batch.observations[start:stop].to(device)
            target_endpoint = batch.targets[start:stop, -1]
            addresses, target_bits = adaptive_lift_address_and_bit(target_endpoint)
            predictions = model(observations).cpu()
            selected_prediction = predictions[:, -1, 1].gather(
                -1, addresses.unsqueeze(-1)
            ).squeeze(-1)
            predicted_bits = selected_prediction >= 0
            correct += int((predicted_bits == target_bits).sum())
            count += target_bits.numel()
            pair_same_address += int(
                (addresses[0::2] == addresses[1::2]).sum()
            )
            pair_opposite_bit += int(
                (target_bits[0::2] != target_bits[1::2]).sum()
            )
            pair_count += target_bits[0::2].numel()
            selected_target = target_endpoint[:, 1].gather(
                -1, addresses.unsqueeze(-1)
            ).squeeze(-1)
            minimum_margin = min(minimum_margin, float(selected_target.abs().min()))
    return {
        "adaptive_lift_bit_accuracy": correct / count,
        "target_center_identity_same_address_fraction": pair_same_address / pair_count,
        "target_center_identity_opposite_bit_fraction": pair_opposite_bit / pair_count,
        "minimum_selected_target_margin": minimum_margin,
    }


def train_candidate(
    candidate: str,
    mode: str,
    schedule: Sequence[endpoint.EndpointTrainingBatch],
    evaluations: dict[str, list[continuous.ContinuousRelationBatch]],
    config: continuous.ContinuousObservationConfig,
    device: torch.device,
    checkpoint_directory: Path | None,
    factory: Callable[[], nn.Module] | None = None,
) -> dict[str, Any]:
    model = (FACTORIES[candidate] if factory is None else factory)().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    samples = {}
    component_samples = {}
    gradient_norm = torch.tensor(0.0)
    started = time.perf_counter()
    model.train()
    for step, batch in enumerate(schedule, start=1):
        observations = batch.observations.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(observations)
        loss, components = mode_loss(mode, predictions, batch.endpoint_targets)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"{candidate}/{mode} produced nonfinite loss at step {step}"
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == config.steps:
            samples[str(step)] = float(loss.detach())
            component_samples[str(step)] = {
                key: float(value.detach()) for key, value in components.items()
            }
            print(
                f"{candidate} mode={mode} seed={config.seed} "
                f"step={step}/{config.steps} loss={samples[str(step)]:.8f} "
                f"components={component_samples[str(step)]}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result: dict[str, Any] = {
        "parameters": continuous.parameter_count(model),
        "recurrent_state_scalars": int(model.recurrent_state_scalars),
        "final_trainable_parameter_sha256": continuous.tensor_hash(
            tuple(
                parameter.detach()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
        "loss_samples": samples,
        "component_samples": component_samples,
        "final_training_loss": samples[str(config.steps)],
        "last_preclip_gradient_norm": float(gradient_norm),
        "training_wall_seconds": elapsed,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "evaluation": {
            key: observability_benchmark.evaluate_by_representation(
                model, batches, device, config.evaluation_microbatch_size
            )
            for key, batches in evaluations.items()
        },
        "lift_bit_evaluation": {
            key: evaluate_lift_bit(
                model, batches, device, config.evaluation_microbatch_size
            )
            for key, batches in evaluations.items()
        },
        "adaptive_lift_evaluation": {
            key: evaluate_adaptive_lift_bit(
                model, batches, device, config.evaluation_microbatch_size
            )
            for key, batches in evaluations.items()
        },
        "action_rmse_by_representation": (
            observability_benchmark.action_rmse_by_representation(
                model, evaluations, device
            )
        ),
    }
    if checkpoint_directory is not None:
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_directory / (
            f"{candidate}_{mode}_seed{config.seed}_step{config.steps}.pt"
        )
        torch.save(
            {
                "format_version": 1,
                "candidate": candidate,
                "mode": mode,
                "training_supervision": "endpoint_vector_plus_optional_one_lift_bit",
                "positive_probe_index": POSITIVE_PROBE_INDEX,
                "bit_logit_scale": BIT_LOGIT_SCALE,
                "bit_loss_weight": BIT_LOSS_WEIGHT,
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
    modes: tuple[str, ...],
    candidates: tuple[str, ...],
    checkpoint_directory: Path | None,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if not modes or len(set(modes)) != len(modes) or set(modes) - set(MODES):
        raise ValueError("modes must be a nonempty unique subset of the known modes")
    if (
        not candidates
        or len(set(candidates)) != len(candidates)
        or set(candidates) - set(CANDIDATES)
    ):
        raise ValueError("candidates must be a nonempty unique known subset")
    certificate = exact_one_bit_certificate()
    if not certificate["passed"]:
        raise RuntimeError("one-bit lift certificate failed")
    teacher_contract = continuous.teacher_contract(device)
    system = continuous.make_observation_system(config.seed)
    schedule = endpoint.make_endpoint_training_schedule(config, system, device)
    split = endpoint.endpoint_training_split_audit(schedule)
    bit_audit = lift_bit_training_audit(schedule)
    adaptive_bit_audit = adaptive_lift_training_audit(schedule)
    if (
        not teacher_contract["passed"]
        or not split["passed"]
        or not bit_audit["passed"]
        or not adaptive_bit_audit["passed"]
    ):
        raise RuntimeError("one-bit task audit failed")

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
        raise RuntimeError("one-bit evaluation audit failed")

    results = {}
    for mode in modes:
        results[mode] = {}
        for candidate in candidates:
            candidate_offset = CANDIDATES.index(candidate)
            continuous.seed_everything(
                1_160_000 + 1_000 * config.seed + candidate_offset
            )
            results[mode][candidate] = train_candidate(
                candidate,
                mode,
                schedule,
                evaluations,
                config,
                device,
                checkpoint_directory,
            )
    all_finite = observability_benchmark._all_numeric_values_finite(results)
    return {
        "schema_version": 1,
        "experiment": "Pure Spin8 minimal one-bit lift calibration",
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
        },
        "config": asdict(config),
        "modes": modes,
        "candidates": candidates,
        "one_bit_certificate": certificate,
        "task": {
            "training_split": split,
            "lift_bit_audit": bit_audit,
            "adaptive_lift_bit_audit": adaptive_bit_audit,
            "observation_system_sha256": continuous.tensor_hash(
                (system.projection, system.bias)
            ),
            "evaluation_audits": evaluation_audits,
            "evaluation_schedule_sha256": evaluation_hashes,
            "positive_probe_index": POSITIVE_PROBE_INDEX,
            "bit_logit_scale": BIT_LOGIT_SCALE,
            "bit_loss_weight": BIT_LOSS_WEIGHT,
            "hidden_full_spinor_target_transferred_for_bit_mode": False,
            "intermediate_targets_retained": False,
        },
        "integrity": {
            "same_schedule_for_every_mode_and_candidate": True,
            "candidate_initialization_identical_across_modes": True,
            "bit_computed_on_cpu_before_device_transfer": True,
            "coordinates_and_events_never_passed_to_models": True,
            "all_metrics_finite": all_finite,
        },
        "results": results,
        "claim_scope": {
            "empirical": [
                "one binary center-visible calibration on an injective synthetic endpoint task",
                "comparison against vector-only, one full spinor, and full triality supervision",
            ],
            "exact": [
                "one bit is necessary and sufficient to distinguish a nonexceptional two-element lift fiber",
            ],
            "not_claimed": [
                "a globally continuous section of the Spin(8) double cover",
                "recovery on the probe-zero hyperplane",
                "physical unsigned-observation or natural-task performance",
            ],
        },
        "passed": bool(
            certificate["passed"]
            and teacher_contract["passed"]
            and split["passed"]
            and bit_audit["passed"]
            and adaptive_bit_audit["passed"]
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
    parser.add_argument("--modes", default=",".join(MODES))
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
        modes=parse_csv(args.modes),
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
