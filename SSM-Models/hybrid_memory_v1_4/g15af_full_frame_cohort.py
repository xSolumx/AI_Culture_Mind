"""Prospectively frozen G15A-F full-frame observability cohort."""

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
from spin8_triality import (
    SPIN8_PAIRS,
    TRIALITY_REPRESENTATIONS,
    torch_triality_generators,
)
from torch import nn
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
    from .g15al_learned_coordinate_cohort import (
        ACTION_ANGLE,
        ACTION_VOCABULARY,
        TokenCoordinateController,
        _carrier_totals,
        _token_map,
        generate_batch,
    )
    from .optimizers import ScalarSecondMomentAdamW
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
    from hybrid_memory_v1_4.g15al_learned_coordinate_cohort import (  # type: ignore[no-redef]
        ACTION_ANGLE,
        ACTION_VOCABULARY,
        TokenCoordinateController,
        _carrier_totals,
        _token_map,
        generate_batch,
    )
    from hybrid_memory_v1_4.optimizers import (  # type: ignore[no-redef]
        ScalarSecondMomentAdamW,
    )


PROTOCOL = Path(__file__).with_name("G15AF_FULL_FRAME_PROTOCOL_2026-08-25.md")
G15AL_ARTIFACT_SHA256 = (
    "7716b75e43964d479bd5fef0cfbd06d0328a315c00a4a0fa107b26f785af1108"
)
OBSERVABILITY_ARTIFACT_SHA256 = (
    "8c53afd308763c6f25042f702d6fcbd667315c35384483a956119d252e3c8d21"
)
QUALITY_SEEDS = (2203, 2207, 2213)
ARM_NAMES = ("I", "C", "S", "S-broken")
RETENTION = 0.999999
PROBE_COUNT = 4
VECTOR_INDEX = TRIALITY_REPRESENTATIONS.index("vector")
POSITIVE_INDEX = TRIALITY_REPRESENTATIONS.index("positive")


@dataclass(frozen=True)
class FullFrameConfig:
    mode: str
    seeds: tuple[int, ...]
    training_updates: int
    training_batch_size: int
    training_length: int
    minimum_training_actions: int
    maximum_training_actions: int
    evaluation_examples: int
    evaluation_microbatch: int
    evaluation_specs: tuple[tuple[int, int], ...]
    learning_rate: float
    dtype: str = "float32"


def quality_config() -> FullFrameConfig:
    return FullFrameConfig(
        mode="quality",
        seeds=QUALITY_SEEDS,
        training_updates=300,
        training_batch_size=16,
        training_length=16,
        minimum_training_actions=2,
        maximum_training_actions=6,
        evaluation_examples=80,
        evaluation_microbatch=8,
        evaluation_specs=((64, 8), (256, 12), (1024, 16)),
        learning_rate=0.05,
    )


def smoke_config() -> FullFrameConfig:
    return FullFrameConfig(
        mode="smoke",
        seeds=(19,),
        training_updates=8,
        training_batch_size=4,
        training_length=12,
        minimum_training_actions=1,
        maximum_training_actions=3,
        evaluation_examples=8,
        evaluation_microbatch=4,
        evaluation_specs=((16, 3), (32, 4)),
        learning_rate=0.05,
    )


@dataclass
class FullFrameBatch:
    token_ids: torch.Tensor
    exact_coordinates: torch.Tensor
    initial_frames: torch.Tensor
    action_positions: torch.Tensor

    def to(
        self, device: torch.device, dtype: torch.dtype | None = None
    ) -> FullFrameBatch:
        dtype = dtype or self.exact_coordinates.dtype
        return FullFrameBatch(
            token_ids=self.token_ids.to(device),
            exact_coordinates=self.exact_coordinates.to(device=device, dtype=dtype),
            initial_frames=self.initial_frames.to(device=device, dtype=dtype),
            action_positions=self.action_positions.to(device),
        )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for tensor in (
            self.token_ids,
            self.exact_coordinates,
            self.initial_frames,
            self.action_positions,
        ):
            value = tensor.detach().cpu().contiguous()
            digest.update(str(value.dtype).encode())
            digest.update(json.dumps(tuple(value.shape)).encode())
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _probe_bank(model_seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(
        _stable_seed("g15af-probe-bank", model_seed)
    )
    samples = torch.randn(
        PROBE_COUNT, 8, 8, generator=generator, dtype=torch.float64
    )
    q, r = torch.linalg.qr(samples)
    signs = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs[:, None, :]


def generate_frame_batch(
    batch_size: int,
    length: int,
    *,
    seed: int,
    model_seed: int,
    minimum_actions: int,
    maximum_actions: int,
) -> FullFrameBatch:
    base = generate_batch(
        batch_size,
        length,
        seed=seed,
        model_seed=model_seed,
        minimum_actions=minimum_actions,
        maximum_actions=maximum_actions,
    )
    frames = (
        _probe_bank(model_seed)
        .to(torch.float32)
        .unsqueeze(0)
        .expand(batch_size, -1, -1, -1)
        .clone()
    )
    return FullFrameBatch(
        token_ids=base.token_ids,
        exact_coordinates=base.exact_coordinates,
        initial_frames=frames,
        action_positions=base.action_positions,
    )


def _semantic_coordinate(semantic: int) -> torch.Tensor:
    coordinate = torch.zeros(1, len(SPIN8_PAIRS), dtype=torch.float32)
    pair_index = semantic % len(OFF_TORUS_PAIRS)
    sign = 1.0 if semantic < len(OFF_TORUS_PAIRS) else -1.0
    coordinate[0, SPIN8_PAIRS.index(OFF_TORUS_PAIRS[pair_index])] = (
        sign * ACTION_ANGLE
    )
    return coordinate


def generate_singleton_inverse_batch(model_seed: int) -> FullFrameBatch:
    primitive_count = ACTION_VOCABULARY
    inverse_count = len(OFF_TORUS_PAIRS)
    batch_size = primitive_count + inverse_count
    length = 4
    tokens = torch.zeros(batch_size, length, dtype=torch.long)
    coordinates = torch.zeros(batch_size, length, 1, len(SPIN8_PAIRS))
    positions = torch.full((batch_size, 2), -1, dtype=torch.long)
    token_map = _token_map(model_seed)
    for semantic in range(primitive_count):
        tokens[semantic, 1] = token_map[semantic]
        coordinates[semantic, 1] = _semantic_coordinate(semantic)
        positions[semantic, 0] = 1
    for pair_index in range(inverse_count):
        row = primitive_count + pair_index
        inverse_semantic = pair_index + inverse_count
        tokens[row, 1] = token_map[pair_index]
        tokens[row, 2] = token_map[inverse_semantic]
        coordinates[row, 1] = _semantic_coordinate(pair_index)
        coordinates[row, 2] = _semantic_coordinate(inverse_semantic)
        positions[row] = torch.tensor((1, 2))
    frames = (
        _probe_bank(model_seed)
        .to(torch.float32)
        .unsqueeze(0)
        .expand(batch_size, -1, -1, -1)
        .clone()
    )
    return FullFrameBatch(tokens, coordinates, frames, positions)


def _broken_positive_generators() -> torch.Tensor:
    generators = torch_triality_generators(dtype=torch.float64)
    positive = generators[POSITIVE_INDEX]
    memory = _oracle_memory("S-broken", dtype=torch.float64, device=torch.device("cpu"))
    permutation = memory.broken_coordinate_permutation.cpu()
    signs = memory.broken_coordinate_signs.to(dtype=torch.float64, device="cpu")
    inverse = torch.argsort(permutation)
    return signs[inverse, None, None] * positive[inverse]


def _broken_lie_bracket_certificate() -> dict[str, Any]:
    generators = torch_triality_generators(dtype=torch.float64)
    vector = torch.round(generators[VECTOR_INDEX]).to(torch.int64)
    broken_positive = torch.round(2.0 * _broken_positive_generators()).to(torch.int64)
    mismatch_count = 0
    maximum_residual = 0
    witness: dict[str, Any] | None = None
    for first in range(len(SPIN8_PAIRS)):
        for second in range(len(SPIN8_PAIRS)):
            vector_commutator = (
                vector[first] @ vector[second] - vector[second] @ vector[first]
            )
            coefficients = torch.stack(
                [
                    (vector_commutator * basis).sum() // 2
                    for basis in vector
                ]
            )
            expected = 2 * torch.einsum(
                "a,aij->ij", coefficients, broken_positive
            )
            observed = (
                broken_positive[first] @ broken_positive[second]
                - broken_positive[second] @ broken_positive[first]
            )
            residual = observed - expected
            residual_max = int(residual.abs().max())
            maximum_residual = max(maximum_residual, residual_max)
            if residual_max:
                mismatch_count += 1
                if witness is None:
                    witness = {
                        "first_index": first,
                        "first_pair": list(SPIN8_PAIRS[first]),
                        "second_index": second,
                        "second_pair": list(SPIN8_PAIRS[second]),
                        "integer_residual_max_abs": residual_max,
                        "integer_residual_matrix": residual.tolist(),
                    }
    return {
        "arithmetic": (
            "exact int64 after multiplying positive-spin generators by two"
        ),
        "generator_pairs_checked": len(SPIN8_PAIRS) ** 2,
        "mismatch_count": mismatch_count,
        "maximum_integer_residual_abs": maximum_residual,
        "witness": witness,
        "passed": mismatch_count > 0 and witness is not None,
    }


def _observability_certificate(model_seed: int) -> dict[str, Any]:
    probes = _probe_bank(model_seed)
    generators = torch_triality_generators(dtype=torch.float64)
    vector = generators[VECTOR_INDEX]
    positive = generators[POSITIVE_INDEX]
    broken_positive = _broken_positive_generators()

    vector_columns = [
        torch.stack([vector[index] @ probe for probe in probes]).reshape(-1)
        for index in range(len(SPIN8_PAIRS))
    ]
    positive_columns = [
        torch.stack([-probe @ positive[index] for probe in probes]).reshape(-1)
        for index in range(len(SPIN8_PAIRS))
    ]
    independent_jacobian = torch.stack(vector_columns + positive_columns, dim=1)
    singular_values = torch.linalg.svdvals(independent_jacobian)
    tolerance = 1e-10 * float(singular_values.max())
    rank = int((singular_values > tolerance).sum())
    condition_ratio = float(singular_values.min() / singular_values.max())

    target_tied = torch.stack(
        [
            torch.stack(
                [vector[index] @ probe - probe @ positive[index] for probe in probes]
            ).reshape(-1)
            for index in range(len(SPIN8_PAIRS))
        ],
        dim=1,
    )
    broken_tied = torch.stack(
        [
            torch.stack(
                [
                    vector[index] @ probe - probe @ broken_positive[index]
                    for probe in probes
                ]
            ).reshape(-1)
            for index in range(len(SPIN8_PAIRS))
        ],
        dim=1,
    )
    broken_basis = torch.linalg.qr(broken_tied, mode="reduced").Q
    primitive_residuals = {}
    for pair in OFF_TORUS_PAIRS:
        index = SPIN8_PAIRS.index(pair)
        target = target_tied[:, index]
        projected = broken_basis @ (broken_basis.T @ target)
        primitive_residuals[str(pair)] = float(
            (target - projected).norm() / target.norm().clamp_min(1e-15)
        )
    probe_hash = hashlib.sha256(
        probes.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    checks = {
        "independent_carrier_jacobian_rank_is_56": rank == 56,
        "condition_ratio_at_least_1e_3": condition_ratio + 1e-15 >= 1e-3,
        "every_primitive_outside_broken_tangent_by_0_05": all(
            residual + 1e-15 >= 0.05 for residual in primitive_residuals.values()
        ),
    }
    return {
        "model_seed": model_seed,
        "probe_count": PROBE_COUNT,
        "probe_bank_sha256": probe_hash,
        "independent_carrier_jacobian_shape": list(independent_jacobian.shape),
        "rank_tolerance": tolerance,
        "numerical_rank": rank,
        "singular_value_maximum": float(singular_values.max()),
        "singular_value_minimum": float(singular_values.min()),
        "condition_ratio": condition_ratio,
        "primitive_relative_projection_residuals": primitive_residuals,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _frame_prediction(
    memory: nn.Module,
    batch: FullFrameBatch,
    coordinates: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    vector, positive = _carrier_totals(memory, batch, coordinates, device=device)
    transported = (
        vector[:, None]
        @ batch.initial_frames
        @ positive.transpose(-1, -2)[:, None]
    )
    return (RETENTION ** (batch.token_ids.shape[1] - 1)) * transported


@torch.no_grad()
def _teacher_target(
    teacher: nn.Module,
    batch: FullFrameBatch,
    *,
    device: torch.device,
) -> torch.Tensor:
    return _frame_prediction(
        teacher, batch, batch.exact_coordinates, device=device
    )


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    difference = (prediction - target).flatten(1)
    target_flat = target.flatten(1)
    relative = difference.norm(dim=-1) / target_flat.norm(dim=-1).clamp_min(1e-12)
    cosine = F.cosine_similarity(prediction.flatten(1), target_flat, dim=-1)
    return {
        "mean_relative_frobenius_error": float(relative.mean()),
        "p95_relative_frobenius_error": float(torch.quantile(relative, 0.95)),
        "maximum_relative_frobenius_error": float(relative.max()),
        "raw_elementwise_mse": float(F.mse_loss(prediction, target)),
        "mean_matrix_cosine": float(cosine.mean()),
        "minimum_matrix_cosine": float(cosine.min()),
        "relative_frobenius_errors": relative.detach().cpu().tolist(),
    }


def _train_arm(
    arm: str,
    config: FullFrameConfig,
    *,
    seed: int,
    device: torch.device,
    checkpoint_directory: Path,
) -> dict[str, Any]:
    _seed_everything(_stable_seed("g15af-controller", seed))
    controller = TokenCoordinateController().to(device)
    memory = _oracle_memory(arm, dtype=torch.float32, device=device)
    teacher = _oracle_memory("S", dtype=torch.float32, device=device)
    optimizer = ScalarSecondMomentAdamW(
        controller.parameters(), lr=config.learning_rate, weight_decay=0.0
    )
    initial_hash = hashlib.sha256(
        controller.raw_coordinates.detach().cpu().numpy().tobytes()
    ).hexdigest()
    probe_bank_hash = hashlib.sha256(
        _probe_bank(seed).detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    schedule = hashlib.sha256()
    loss_samples: dict[str, dict[str, float]] = {}
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(config.training_updates):
        batch = generate_frame_batch(
            config.training_batch_size,
            config.training_length,
            seed=_stable_seed("g15af-train", seed, step),
            model_seed=seed,
            minimum_actions=config.minimum_training_actions,
            maximum_actions=config.maximum_training_actions,
        )
        schedule.update(batch.fingerprint().encode())
        batch = batch.to(device)
        target = _teacher_target(teacher, batch, device=device)
        optimizer.zero_grad(set_to_none=True)
        prediction = _frame_prediction(
            memory,
            batch,
            controller(batch.token_ids),
            device=device,
        )
        loss = F.mse_loss(prediction, target)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite G15A-F loss for {arm} seed {seed}")
        if loss.requires_grad:
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        else:
            gradient_norm = loss.new_zeros(())
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("non-finite G15A-F gradient norm")
        optimizer.step()
        if step in {0, config.training_updates // 2, config.training_updates - 1}:
            loss_samples[str(step + 1)] = {
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm.detach()),
                "coordinate_max_abs": float(
                    controller(batch.token_ids).detach().abs().max()
                ),
            }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    evaluations: dict[str, dict[str, float | int]] = {}
    controller.eval()
    for length, actions_per_episode in config.evaluation_specs:
        metric_rows = []
        for offset in range(
            0, config.evaluation_examples, config.evaluation_microbatch
        ):
            size = min(
                config.evaluation_microbatch, config.evaluation_examples - offset
            )
            batch = generate_frame_batch(
                size,
                length,
                seed=_stable_seed("g15af-eval", seed, length, offset),
                model_seed=seed,
                minimum_actions=actions_per_episode,
                maximum_actions=actions_per_episode,
            ).to(device)
            target = _teacher_target(teacher, batch, device=device)
            with torch.no_grad():
                prediction = _frame_prediction(
                    memory,
                    batch,
                    controller(batch.token_ids),
                    device=device,
                )
            metric_rows.append(_metrics(prediction, target))
        relative_errors = torch.tensor(
            [
                value
                for row in metric_rows
                for value in row["relative_frobenius_errors"]
            ],
            dtype=torch.float64,
        )
        evaluations[str(length)] = {
            "mean_relative_frobenius_error": sum(
                row["mean_relative_frobenius_error"] for row in metric_rows
            )
            / len(metric_rows),
            "p95_relative_frobenius_error": float(
                torch.quantile(relative_errors, 0.95)
            ),
            "maximum_relative_frobenius_error": max(
                row["maximum_relative_frobenius_error"] for row in metric_rows
            ),
            "raw_elementwise_mse": sum(
                row["raw_elementwise_mse"] for row in metric_rows
            )
            / len(metric_rows),
            "mean_matrix_cosine": sum(
                row["mean_matrix_cosine"] for row in metric_rows
            )
            / len(metric_rows),
            "minimum_matrix_cosine": min(
                row["minimum_matrix_cosine"] for row in metric_rows
            ),
            "actions_per_episode": actions_per_episode,
            "examples": config.evaluation_examples,
        }
    diagnostic_batch = generate_singleton_inverse_batch(seed).to(device)
    diagnostic_target = _teacher_target(teacher, diagnostic_batch, device=device)
    with torch.no_grad():
        diagnostic_prediction = _frame_prediction(
            memory,
            diagnostic_batch,
            controller(diagnostic_batch.token_ids),
            device=device,
        )
    primitive_count = ACTION_VOCABULARY
    primitive_diagnostic = {
        "singleton": _metrics(
            diagnostic_prediction[:primitive_count],
            diagnostic_target[:primitive_count],
        ),
        "inverse_pair": _metrics(
            diagnostic_prediction[primitive_count:],
            diagnostic_target[primitive_count:],
        ),
        "gated": False,
    }
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_directory / f"g15af_{arm}_seed{seed}.pt"
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(
        {
            "schema_version": 1,
            "arm": arm,
            "seed": seed,
            "protocol": asdict(config),
            "raw_coordinates": controller.raw_coordinates.detach().cpu(),
            "optimizer_state_dict": optimizer.state_dict(),
            "evaluation": evaluations,
            "training_schedule_sha256": schedule.hexdigest(),
        },
        temporary,
    )
    os.replace(temporary, checkpoint)
    result = {
        "trainable_parameters": sum(
            parameter.numel() for parameter in controller.parameters()
        ),
        "structurally_unused_parameters": len(SPIN8_PAIRS),
        "initial_state_sha256": initial_hash,
        "probe_bank_sha256": probe_bank_hash,
        "training_schedule_sha256": schedule.hexdigest(),
        "loss_samples": loss_samples,
        "training_wall_seconds": training_seconds,
        "mean_synchronized_step_seconds": training_seconds / config.training_updates,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "evaluation": evaluations,
        "primitive_diagnostic": primitive_diagnostic,
        "learned_raw_coordinates": controller.raw_coordinates.detach().cpu().tolist(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "optimizer": {
            "name": "ScalarSecondMomentAdamW",
            "learning_rate": config.learning_rate,
            "weight_decay": 0.0,
            "second_moment": "one scalar for the 17x28 coordinate tensor",
        },
    }
    del optimizer, controller, memory, teacher
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _adjudicate(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    per_seed = []
    for report in seed_reports:
        arms = report["arms"]
        per_length = {}
        for length in arms["S"]["evaluation"]:
            spin = arms["S"]["evaluation"][length]
            comparator_margins = {
                arm: arms[arm]["evaluation"][length][
                    "mean_relative_frobenius_error"
                ]
                - spin["mean_relative_frobenius_error"]
                for arm in ("I", "C", "S-broken")
            }
            checks = {
                "spin_mean_relative_error_at_most_0_05": spin[
                    "mean_relative_frobenius_error"
                ]
                <= 0.05 + 1e-12,
                "spin_p95_relative_error_at_most_0_10": spin[
                    "p95_relative_frobenius_error"
                ]
                <= 0.10 + 1e-12,
                "spin_maximum_relative_error_at_most_0_20": spin[
                    "maximum_relative_frobenius_error"
                ]
                <= 0.20 + 1e-12,
                "each_comparator_error_margin_at_least_0_05": all(
                    margin + 1e-12 >= 0.05 for margin in comparator_margins.values()
                ),
                "broken_error_at_least_twice_spin": arms["S-broken"][
                    "evaluation"
                ][length]["mean_relative_frobenius_error"]
                + 1e-12
                >= 2.0 * spin["mean_relative_frobenius_error"],
            }
            per_length[length] = {
                "spin": spin,
                "comparator_mean_relative_error_margins": comparator_margins,
                "checks": checks,
                "passed": all(checks.values()),
            }
        exact_pairing = (
            len({arms[arm]["trainable_parameters"] for arm in ARM_NAMES}) == 1
            and len({arms[arm]["initial_state_sha256"] for arm in ARM_NAMES}) == 1
            and len({arms[arm]["probe_bank_sha256"] for arm in ARM_NAMES}) == 1
            and len({arms[arm]["training_schedule_sha256"] for arm in ARM_NAMES}) == 1
        )
        checks = {
            "all_lengths_pass": all(row["passed"] for row in per_length.values()),
            "exact_parameter_initialization_schedule_pairing": exact_pairing,
        }
        per_seed.append(
            {
                "seed": report["seed"],
                "per_length": per_length,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    passed = all(row["passed"] for row in per_seed)
    return {
        "passed": passed,
        "per_seed": per_seed,
        "decision": (
            "G15A-F passes: full-frame loss identifies the shared Spin chart"
            if passed
            else "G15A-F fails: full-frame learned shared-chart attribution is not established"
        ),
    }


def _load_bound_artifact(path: Path, expected_sha256: str) -> dict[str, Any]:
    if _sha256(path) != expected_sha256:
        raise RuntimeError(f"bound artifact does not match frozen hash: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run(
    config: FullFrameConfig,
    *,
    device: torch.device,
    checkpoint_directory: Path,
    commit: str,
    status_at_start: list[str],
) -> dict[str, Any]:
    started_at = _now()
    started = time.perf_counter()
    observability_certificates = [
        _observability_certificate(seed) for seed in config.seeds
    ]
    bracket_certificate = _broken_lie_bracket_certificate()
    if not all(row["passed"] for row in observability_certificates):
        raise RuntimeError("G15A-F probe-bank observability screen failed")
    if not bracket_certificate["passed"]:
        raise RuntimeError("G15A-F broken-arm Lie-bracket certificate failed")
    seed_reports = []
    for seed in config.seeds:
        arms = {
            arm: _train_arm(
                arm,
                config,
                seed=seed,
                device=device,
                checkpoint_directory=checkpoint_directory,
            )
            for arm in ARM_NAMES
        }
        seed_reports.append({"seed": seed, "arms": arms})
    adjudication = _adjudicate(seed_reports)
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("g15al_learned_coordinate_cohort.py"),
        Path(__file__).with_name("g15a_spin_dirac_cohort.py"),
        Path(__file__).with_name("spin_dirac_memory.py"),
        Path(__file__).with_name("optimizers.py"),
        PROTOCOL,
    )
    return {
        "schema_version": 1,
        "experiment": "G15A-F full-frame learned-coordinate cohort",
        "claim_status": (
            "learned shared vector/positive Spin chart under full-frame observation "
            "and oracle edit timing"
        ),
        "mode": config.mode,
        "evidentiary": config.mode == "quality" and not status_at_start,
        "started_at": started_at,
        "finished_at": _now(),
        "elapsed_wall_seconds": time.perf_counter() - started,
        "git_commit_at_start": commit,
        "git_status_at_start": status_at_start,
        "g15al_artifact_sha256": G15AL_ARTIFACT_SHA256,
        "observability_artifact_sha256": OBSERVABILITY_ARTIFACT_SHA256,
        "protocol": asdict(config),
        "arm_names": list(ARM_NAMES),
        "protocol_file_sha256": _sha256(PROTOCOL),
        "execution_path": "exact_event_sparse_two_sided_fast_weight_transport",
        "observability_certificates": observability_certificates,
        "broken_lie_bracket_certificate": bracket_certificate,
        "source_files": {
            str(path.relative_to(Path(__file__).parent)): _sha256(path)
            for path in source_paths
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "compute_capability": (
                list(torch.cuda.get_device_capability(device))
                if device.type == "cuda"
                else None
            ),
            "dtype": "float32",
        },
        "seed_reports": seed_reports,
        "adjudication": adjudication,
        "explicit_nonclaims": [
            "edit timing and the initial full-rank association frame are oracle supplied",
            "only the token-to-coordinate lookup is learned",
            "the four-probe observation remains insensitive to a common discrete center",
            "the negative spin carrier and Clifford read are not used",
            "no generic association, natural-text, scaling, or fused claim follows",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "quality"), required=True)
    parser.add_argument("--g15al-artifact", type=Path, required=True)
    parser.add_argument("--observability-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    g15al_artifact = _load_bound_artifact(
        args.g15al_artifact, G15AL_ARTIFACT_SHA256
    )
    if g15al_artifact.get("adjudication", {}).get("passed") is not False:
        raise RuntimeError("bound G15A-L artifact is not the frozen failure")
    _load_bound_artifact(
        args.observability_artifact, OBSERVABILITY_ARTIFACT_SHA256
    )
    config = quality_config() if args.mode == "quality" else smoke_config()
    commit, status_at_start = _git_state()
    if args.mode == "quality" and status_at_start:
        raise RuntimeError("evidentiary G15A-F requires a clean committed worktree")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if torch.cuda.get_device_capability(device) != (7, 5):
            raise RuntimeError("the frozen local G15A-F cohort requires exact SM75")
    report = run(
        config,
        device=device,
        checkpoint_directory=args.checkpoint_directory,
        commit=commit,
        status_at_start=status_at_start,
    )
    _atomic_json(args.output, report)
    print(args.output)
    print(json.dumps(report["adjudication"], indent=2, sort_keys=True))
    if args.mode == "quality" and not report["adjudication"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
