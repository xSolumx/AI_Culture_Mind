"""Run the prospectively frozen G15A-S spanning/center transfer cohort."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
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
    algebra_diagnostics,
    torch_triality_generators,
)
from torch import nn
from torch.nn import functional as F

if __package__:
    from .g15a_spin_dirac_cohort import (
        _atomic_json,
        _git_state,
        _now,
        _oracle_memory,
        _sha256,
        _stable_seed,
    )
    from .g15af_full_frame_cohort import _broken_positive_generators
    from .optimizers import ScalarSecondMomentAdamW
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4.g15a_spin_dirac_cohort import (  # type: ignore[no-redef]
        _atomic_json,
        _git_state,
        _now,
        _oracle_memory,
        _sha256,
        _stable_seed,
    )
    from hybrid_memory_v1_4.g15af_full_frame_cohort import (  # type: ignore[no-redef]
        _broken_positive_generators,
    )
    from hybrid_memory_v1_4.optimizers import (  # type: ignore[no-redef]
        ScalarSecondMomentAdamW,
    )


PROTOCOL = Path(__file__).with_name("G15AS_SPANNING_CENTER_PROTOCOL_2026-08-25.md")
G15AR_ARTIFACT_SHA256 = (
    "be004dea821e9bf38e140f627fe387829d67949767be045310cf2080ac7d6fe8"
)
QUALITY_SEEDS = (2281, 2287, 2293)
ARM_NAMES = ("I", "C", "S", "S-broken")
PAIR_COUNT = len(SPIN8_PAIRS)
ACTION_VOCABULARY = 2 * PAIR_COUNT
VOCABULARY_SIZE = ACTION_VOCABULARY + 1
STEP_ANGLE = math.pi / 16.0
MAXIMUM_COORDINATE = 0.25
RETENTION = 0.999999
PROBE_COUNT = 4
PROBE_POOL_SIZE = 64
VECTOR_INDEX = TRIALITY_REPRESENTATIONS.index("vector")
POSITIVE_INDEX = TRIALITY_REPRESENTATIONS.index("positive")
OFF_TORUS_CONTINUATIONS = (
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 4),
    (2, 5),
    (3, 6),
    (3, 7),
)


@dataclass(frozen=True)
class SpanningConfig:
    mode: str
    seeds: tuple[int, ...]
    updates: int
    batch_size: int
    training_length: int
    minimum_actions: int
    maximum_actions: int
    evaluation_examples: int
    evaluation_microbatch: int
    evaluation_specs: tuple[tuple[int, int], ...]
    probe_pool_size: int
    dtype: str = "float32"


def quality_config() -> SpanningConfig:
    return SpanningConfig(
        mode="quality",
        seeds=QUALITY_SEEDS,
        updates=600,
        batch_size=32,
        training_length=16,
        minimum_actions=2,
        maximum_actions=6,
        evaluation_examples=80,
        evaluation_microbatch=8,
        evaluation_specs=((64, 8), (256, 12), (1024, 16)),
        probe_pool_size=PROBE_POOL_SIZE,
    )


def smoke_config() -> SpanningConfig:
    return SpanningConfig(
        mode="smoke",
        seeds=(37,),
        updates=8,
        batch_size=8,
        training_length=12,
        minimum_actions=1,
        maximum_actions=3,
        evaluation_examples=8,
        evaluation_microbatch=4,
        evaluation_specs=((16, 3), (32, 4)),
        probe_pool_size=4,
    )


@dataclass
class SpanningBatch:
    token_ids: torch.Tensor
    exact_coordinates: torch.Tensor
    initial_frames: torch.Tensor
    action_positions: torch.Tensor

    def to(
        self, device: torch.device, dtype: torch.dtype | None = None
    ) -> SpanningBatch:
        target_dtype = dtype or self.exact_coordinates.dtype
        return SpanningBatch(
            self.token_ids.to(device),
            self.exact_coordinates.to(device=device, dtype=target_dtype),
            self.initial_frames.to(device=device, dtype=target_dtype),
            self.action_positions.to(device),
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


class SpanningCoordinateController(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.raw_coordinates = nn.Parameter(torch.zeros(VOCABULARY_SIZE, PAIR_COUNT))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        coordinates = MAXIMUM_COORDINATE * torch.tanh(self.raw_coordinates[token_ids])
        coordinates = torch.where(
            token_ids[..., None] == 0, torch.zeros_like(coordinates), coordinates
        )
        return coordinates.unsqueeze(2)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _token_map(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(_stable_seed("g15as-token-map", seed))
    return torch.randperm(ACTION_VOCABULARY, generator=generator) + 1


def _semantic_coordinate(semantic: int, *, dtype: torch.dtype) -> torch.Tensor:
    if not 0 <= semantic < ACTION_VOCABULARY:
        raise ValueError("semantic action is out of range")
    coordinate = torch.zeros(1, PAIR_COUNT, dtype=dtype)
    pair_index = semantic % PAIR_COUNT
    sign = 1.0 if semantic < PAIR_COUNT else -1.0
    coordinate[0, pair_index] = sign * STEP_ANGLE
    return coordinate


def _orthogonal_bank(seed: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    samples = torch.randn(PROBE_COUNT, 8, 8, generator=generator, dtype=dtype)
    q, r = torch.linalg.qr(samples)
    signs = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs[:, None, :]


def _probe_pool(model_seed: int, split: str, count: int) -> torch.Tensor:
    if split not in ("training", "evaluation"):
        raise ValueError("unknown probe split")
    return torch.stack(
        [
            _orthogonal_bank(_stable_seed("g15as-probe", split, model_seed, index))
            for index in range(count)
        ]
    )


def _pool_hash(pool: torch.Tensor) -> str:
    return hashlib.sha256(pool.cpu().contiguous().numpy().tobytes()).hexdigest()


def generate_batch(
    batch_size: int,
    length: int,
    *,
    seed: int,
    model_seed: int,
    minimum_actions: int,
    maximum_actions: int,
    probe_pool: torch.Tensor,
) -> SpanningBatch:
    if length < maximum_actions + 2:
        raise ValueError("length must leave room for actions")
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.zeros(batch_size, length, dtype=torch.long)
    coordinates = torch.zeros(batch_size, length, 1, PAIR_COUNT)
    positions = torch.full((batch_size, maximum_actions), -1, dtype=torch.long)
    token_map = _token_map(model_seed)
    for row in range(batch_size):
        count = int(
            torch.randint(
                minimum_actions,
                maximum_actions + 1,
                (),
                generator=generator,
            )
        )
        selected_positions = (
            (torch.randperm(length - 2, generator=generator)[:count] + 1).sort().values
        )
        semantics = torch.randint(ACTION_VOCABULARY, (count,), generator=generator)
        positions[row, :count] = selected_positions
        for position, semantic in zip(
            selected_positions.tolist(), semantics.tolist(), strict=True
        ):
            tokens[row, position] = token_map[semantic]
            coordinates[row, position] = _semantic_coordinate(
                semantic, dtype=coordinates.dtype
            )
    pool_indices = torch.randint(
        probe_pool.shape[0], (batch_size,), generator=generator
    )
    frames = probe_pool[pool_indices].to(torch.float32)
    return SpanningBatch(tokens, coordinates, frames, positions)


def _carrier_totals(
    memory: nn.Module,
    batch: SpanningBatch,
    coordinates: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, maximum_actions = batch.action_positions.shape
    rows = torch.arange(batch_size, device=device)[:, None]
    positions = batch.action_positions.clamp_min(0)
    event_coordinates = coordinates[rows, positions]
    valid = batch.action_positions >= 0
    event_coordinates = torch.where(
        valid[..., None, None], event_coordinates, torch.zeros_like(event_coordinates)
    )
    event_coordinates = memory._apply_transport_mode(event_coordinates)
    carrier = torch.zeros(
        batch_size,
        maximum_actions,
        1,
        8,
        dtype=coordinates.dtype,
        device=device,
    )
    gate = torch.zeros(
        batch_size,
        maximum_actions,
        1,
        1,
        dtype=coordinates.dtype,
        device=device,
    )
    actions = memory._transitions(
        carrier,
        carrier,
        gate,
        gate,
        torch.ones_like(gate),
        event_coordinates,
        None,
    )[3]
    eye = torch.eye(8, dtype=coordinates.dtype, device=device).expand(batch_size, 8, 8)
    vector_total = eye.clone()
    positive_total = eye.clone()
    for column in range(maximum_actions):
        active = valid[:, column, None, None]
        vector = torch.where(active, actions[:, column, 0, VECTOR_INDEX], eye)
        positive = torch.where(active, actions[:, column, 0, POSITIVE_INDEX], eye)
        vector_total = vector @ vector_total
        positive_total = positive @ positive_total
    return vector_total, positive_total


def _frame_prediction(
    memory: nn.Module,
    batch: SpanningBatch,
    coordinates: torch.Tensor,
    *,
    device: torch.device,
    apply_retention: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vector, positive = _carrier_totals(memory, batch, coordinates, device=device)
    frames = (
        vector[:, None] @ batch.initial_frames @ positive.transpose(-1, -2)[:, None]
    )
    if apply_retention:
        frames = (RETENTION ** (batch.token_ids.shape[1] - 1)) * frames
    return frames, vector, positive


@torch.no_grad()
def _teacher_target(
    teacher: nn.Module,
    batch: SpanningBatch,
    *,
    device: torch.device,
    apply_retention: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _frame_prediction(
        teacher,
        batch,
        batch.exact_coordinates,
        device=device,
        apply_retention=apply_retention,
    )


def _frame_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    difference = (prediction - target).flatten(1)
    target_flat = target.flatten(1)
    relative = difference.norm(dim=-1) / target_flat.norm(dim=-1).clamp_min(1e-12)
    return {
        "mean_relative_frobenius_error": float(relative.mean()),
        "p95_relative_frobenius_error": float(torch.quantile(relative, 0.95)),
        "maximum_relative_frobenius_error": float(relative.max()),
        "raw_elementwise_mse": float(F.mse_loss(prediction, target)),
        "relative_frobenius_errors": relative.detach().cpu().tolist(),
    }


def _bank_certificate(bank: torch.Tensor) -> dict[str, Any]:
    generators = torch_triality_generators(dtype=torch.float64)
    vector = generators[VECTOR_INDEX]
    positive = generators[POSITIVE_INDEX]
    broken_positive = _broken_positive_generators()
    vector_columns = [
        torch.stack([vector[index] @ probe for probe in bank]).reshape(-1)
        for index in range(PAIR_COUNT)
    ]
    positive_columns = [
        torch.stack([-probe @ positive[index] for probe in bank]).reshape(-1)
        for index in range(PAIR_COUNT)
    ]
    independent = torch.stack(vector_columns + positive_columns, dim=1)
    singular = torch.linalg.svdvals(independent)
    tolerance = 1e-10 * float(singular.max())
    rank = int((singular > tolerance).sum())
    condition = float(singular.min() / singular.max())
    target = torch.stack(
        [
            torch.stack(
                [vector[index] @ probe - probe @ positive[index] for probe in bank]
            ).reshape(-1)
            for index in range(PAIR_COUNT)
        ],
        dim=1,
    )
    broken = torch.stack(
        [
            torch.stack(
                [
                    vector[index] @ probe - probe @ broken_positive[index]
                    for probe in bank
                ]
            ).reshape(-1)
            for index in range(PAIR_COUNT)
        ],
        dim=1,
    )
    basis = torch.linalg.qr(broken, mode="reduced").Q
    residuals = []
    for index in range(PAIR_COUNT):
        column = target[:, index]
        projected = basis @ (basis.T @ column)
        residuals.append(float((column - projected).norm() / column.norm()))
    passed = rank == 56 and condition >= 0.10 and min(residuals) >= 0.05
    return {
        "rank": rank,
        "condition_ratio": condition,
        "minimum_broken_projection_residual": min(residuals),
        "maximum_broken_projection_residual": max(residuals),
        "passed": passed,
    }


def _pool_certificate(pool: torch.Tensor) -> dict[str, Any]:
    certificates = [_bank_certificate(bank) for bank in pool]
    return {
        "bank_count": len(certificates),
        "minimum_rank": min(row["rank"] for row in certificates),
        "minimum_condition_ratio": min(row["condition_ratio"] for row in certificates),
        "minimum_broken_projection_residual": min(
            row["minimum_broken_projection_residual"] for row in certificates
        ),
        "all_passed": all(row["passed"] for row in certificates),
    }


def _structured_words() -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(SPIN8_PAIRS):
        for sign_name, semantic in (
            ("positive", pair_index),
            ("negative", pair_index + PAIR_COUNT),
        ):
            words.append(
                {
                    "name": f"two_pi_{sign_name}_{pair_index}",
                    "kind": "center",
                    "center": "minus_one",
                    "pair": list(pair),
                    "semantics": [semantic] * 32,
                }
            )
            words.append(
                {
                    "name": f"four_pi_{sign_name}_{pair_index}",
                    "kind": "center",
                    "center": "identity",
                    "pair": list(pair),
                    "semantics": [semantic] * 64,
                }
            )
    torus_indices = [
        SPIN8_PAIRS.index(pair) for pair in ((0, 1), (2, 3), (4, 5), (6, 7))
    ]
    for sign_name, offset in (("positive", 0), ("negative", PAIR_COUNT)):
        volume = [index + offset for index in torus_indices for _ in range(16)]
        words.append(
            {
                "name": f"volume_{sign_name}",
                "kind": "center",
                "center": "omega",
                "semantics": volume,
            }
        )
        loop_index = SPIN8_PAIRS.index((0, 2)) + offset
        words.append(
            {
                "name": f"minus_volume_{sign_name}",
                "kind": "center",
                "center": "minus_omega",
                "semantics": [loop_index] * 32 + volume,
            }
        )
    loop_index = SPIN8_PAIRS.index((0, 2))
    for pair in OFF_TORUS_CONTINUATIONS:
        pair_index = SPIN8_PAIRS.index(pair)
        for sign_name, semantic in (
            ("positive", pair_index),
            ("negative", pair_index + PAIR_COUNT),
        ):
            words.append(
                {
                    "name": f"loop_then_{sign_name}_{pair_index}",
                    "kind": "continuation",
                    "center": None,
                    "pair": list(pair),
                    "semantics": [loop_index] * 32 + [semantic],
                }
            )
    return words


def _structured_batch(
    model_seed: int,
    probe_pool: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> tuple[SpanningBatch, list[dict[str, Any]]]:
    words = _structured_words()
    maximum_actions = max(len(row["semantics"]) for row in words)
    length = maximum_actions + 2
    tokens = torch.zeros(len(words), length, dtype=torch.long)
    coordinates = torch.zeros(len(words), length, 1, PAIR_COUNT, dtype=dtype)
    positions = torch.full((len(words), maximum_actions), -1, dtype=torch.long)
    token_map = _token_map(model_seed)
    for row_index, word in enumerate(words):
        semantics = word["semantics"]
        for column, semantic in enumerate(semantics, start=1):
            tokens[row_index, column] = token_map[semantic]
            coordinates[row_index, column] = _semantic_coordinate(semantic, dtype=dtype)
            positions[row_index, column - 1] = column
    pool_indices = torch.arange(len(words)).remainder(probe_pool.shape[0])
    frames = probe_pool[pool_indices].to(dtype)
    return SpanningBatch(tokens, coordinates, frames, positions), words


def _center_signs(name: str) -> tuple[float, float]:
    return {
        "identity": (1.0, 1.0),
        "minus_one": (1.0, -1.0),
        "omega": (-1.0, 1.0),
        "minus_omega": (-1.0, -1.0),
    }[name]


def _oracle_certificate(model_seed: int, pool: torch.Tensor) -> dict[str, Any]:
    batch, words = _structured_batch(model_seed, pool, dtype=torch.float64)
    teacher = _oracle_memory("S", dtype=torch.float64, device=torch.device("cpu"))
    frames, vector, positive = _teacher_target(
        teacher,
        batch,
        device=torch.device("cpu"),
        apply_retention=False,
    )
    eye = torch.eye(8, dtype=torch.float64)
    maximum_sign_residual = 0.0
    projective_witnesses = []
    for index, word in enumerate(words):
        center = word["center"]
        if center is None:
            continue
        vector_sign, positive_sign = _center_signs(center)
        maximum_sign_residual = max(
            maximum_sign_residual,
            float((vector[index] - vector_sign * eye).abs().max()),
            float((positive[index] - positive_sign * eye).abs().max()),
        )
        frame_identity_residual = float(
            (frames[index] - batch.initial_frames[index]).abs().max()
        )
        carrier_nonidentity = max(
            float((vector[index] - eye).abs().max()),
            float((positive[index] - eye).abs().max()),
        )
        if frame_identity_residual <= 1e-10 and carrier_nonidentity >= 1.0:
            projective_witnesses.append(word["name"])
    algebra = algebra_diagnostics(seed=model_seed)
    passed = (
        maximum_sign_residual <= 1e-10
        and bool(projective_witnesses)
        and bool(algebra["checks"]["full_center_signatures"])
    )
    return {
        "structured_word_count": len(words),
        "maximum_analytic_center_sign_residual": maximum_sign_residual,
        "projective_center_witnesses": projective_witnesses,
        "algebra_center_signature_max_abs": algebra["center_signature_max_abs"],
        "passed": passed,
    }


@torch.no_grad()
def _evaluate_random(
    controller: SpanningCoordinateController,
    memory: nn.Module,
    teacher: nn.Module,
    *,
    seed: int,
    examples: int,
    microbatch: int,
    specs: tuple[tuple[int, int], ...],
    probe_pool: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    results = {}
    for length, actions in specs:
        errors = []
        mse = 0.0
        batches = 0
        for offset in range(0, examples, microbatch):
            size = min(microbatch, examples - offset)
            batch = generate_batch(
                size,
                length,
                seed=_stable_seed("g15as-eval", seed, length, offset),
                model_seed=seed,
                minimum_actions=actions,
                maximum_actions=actions,
                probe_pool=probe_pool,
            ).to(device)
            target = _teacher_target(teacher, batch, device=device)[0]
            prediction = _frame_prediction(
                memory, batch, controller(batch.token_ids), device=device
            )[0]
            metrics = _frame_metrics(prediction, target)
            errors.extend(metrics["relative_frobenius_errors"])
            mse += metrics["raw_elementwise_mse"]
            batches += 1
        values = torch.tensor(errors, dtype=torch.float64)
        results[str(length)] = {
            "actions_per_episode": actions,
            "examples": examples,
            "mean_relative_frobenius_error": float(values.mean()),
            "p95_relative_frobenius_error": float(torch.quantile(values, 0.95)),
            "maximum_relative_frobenius_error": float(values.max()),
            "raw_elementwise_mse": mse / batches,
        }
    return results


@torch.no_grad()
def _coordinate_metrics(
    controller: SpanningCoordinateController, seed: int
) -> dict[str, float]:
    token_map = _token_map(seed).to(controller.raw_coordinates.device)
    token_ids = token_map
    learned = controller(token_ids)[:, 0]
    target = torch.stack(
        [
            _semantic_coordinate(index, dtype=learned.dtype)[0]
            for index in range(ACTION_VOCABULARY)
        ]
    ).to(learned.device)
    active_indices = torch.arange(ACTION_VOCABULARY, device=learned.device) % PAIR_COUNT
    rows = torch.arange(ACTION_VOCABULARY, device=learned.device)
    active_error = (learned[rows, active_indices] - target[rows, active_indices]).abs()
    inactive_mask = torch.ones_like(learned, dtype=torch.bool)
    inactive_mask[rows, active_indices] = False
    inactive = learned[inactive_mask]
    return {
        "maximum_active_coordinate_abs_error": float(active_error.max()),
        "mean_active_coordinate_abs_error": float(active_error.mean()),
        "inactive_coordinate_rms": float(inactive.square().mean().sqrt()),
        "inactive_coordinate_max_abs": float(inactive.abs().max()),
    }


@torch.no_grad()
def _evaluate_structured(
    controller: SpanningCoordinateController,
    memory: nn.Module,
    teacher: nn.Module,
    *,
    seed: int,
    probe_pool: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    batch, words = _structured_batch(seed, probe_pool, dtype=torch.float32)
    rows = []
    for start in range(0, len(words), 8):
        stop = min(start + 8, len(words))
        subset = SpanningBatch(
            batch.token_ids[start:stop],
            batch.exact_coordinates[start:stop],
            batch.initial_frames[start:stop],
            batch.action_positions[start:stop],
        ).to(device)
        target_frame, target_vector, target_positive = _teacher_target(
            teacher, subset, device=device, apply_retention=False
        )
        prediction_frame, prediction_vector, prediction_positive = _frame_prediction(
            memory,
            subset,
            controller(subset.token_ids),
            device=device,
            apply_retention=False,
        )
        for local, word in enumerate(words[start:stop]):

            def relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
                return float(
                    (actual - expected).norm() / expected.norm().clamp_min(1e-12)
                )

            rows.append(
                {
                    "name": word["name"],
                    "kind": word["kind"],
                    "center": word["center"],
                    "actions": len(word["semantics"]),
                    "frame_relative_error": relative(
                        prediction_frame[local], target_frame[local]
                    ),
                    "vector_relative_error": relative(
                        prediction_vector[local], target_vector[local]
                    ),
                    "positive_relative_error": relative(
                        prediction_positive[local], target_positive[local]
                    ),
                }
            )
    return {
        "word_count": len(rows),
        "maximum_frame_relative_error": max(
            row["frame_relative_error"] for row in rows
        ),
        "maximum_vector_relative_error": max(
            row["vector_relative_error"] for row in rows
        ),
        "maximum_positive_relative_error": max(
            row["positive_relative_error"] for row in rows
        ),
        "rows": rows,
    }


def _learning_rate(update: int) -> float:
    if update <= 100:
        return 0.05
    if update <= 300:
        return 0.01
    return 0.002


def _train_arm(
    arm: str,
    config: SpanningConfig,
    *,
    seed: int,
    training_pool: torch.Tensor,
    evaluation_pool: torch.Tensor,
    device: torch.device,
    checkpoint_directory: Path,
) -> dict[str, Any]:
    _seed_everything(_stable_seed("g15as-controller", seed))
    controller = SpanningCoordinateController().to(device)
    memory = _oracle_memory(arm, dtype=torch.float32, device=device)
    teacher = _oracle_memory("S", dtype=torch.float32, device=device)
    optimizer = ScalarSecondMomentAdamW(
        controller.parameters(), lr=0.05, weight_decay=0.0
    )
    initialization_hash = hashlib.sha256(
        controller.raw_coordinates.detach().cpu().numpy().tobytes()
    ).hexdigest()
    schedule_hash = hashlib.sha256()
    loss_samples = {}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for update in range(1, config.updates + 1):
        learning_rate = _learning_rate(update)
        optimizer.param_groups[0]["lr"] = learning_rate
        batch = generate_batch(
            config.batch_size,
            config.training_length,
            seed=_stable_seed("g15as-training", seed, update),
            model_seed=seed,
            minimum_actions=config.minimum_actions,
            maximum_actions=config.maximum_actions,
            probe_pool=training_pool,
        )
        schedule_hash.update(batch.fingerprint().encode())
        batch = batch.to(device)
        target = _teacher_target(teacher, batch, device=device)[0]
        optimizer.zero_grad(set_to_none=True)
        prediction = _frame_prediction(
            memory, batch, controller(batch.token_ids), device=device
        )[0]
        loss = F.mse_loss(prediction, target)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite G15A-S loss for {arm}/{seed}")
        if loss.requires_grad:
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        else:
            gradient_norm = loss.new_zeros(())
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("non-finite G15A-S gradient")
        optimizer.step()
        if update in (1, 100, 200, 300, 450, config.updates):
            loss_samples[str(update)] = {
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm.detach()),
                "learning_rate": learning_rate,
            }
    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    random_evaluation = _evaluate_random(
        controller,
        memory,
        teacher,
        seed=seed,
        examples=config.evaluation_examples,
        microbatch=config.evaluation_microbatch,
        specs=config.evaluation_specs,
        probe_pool=evaluation_pool,
        device=device,
    )
    coordinate_evaluation = _coordinate_metrics(controller, seed)
    structured_evaluation = _evaluate_structured(
        controller,
        memory,
        teacher,
        seed=seed,
        probe_pool=evaluation_pool,
        device=device,
    )
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_directory / f"g15as_{arm}_{seed}.pt"
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(
        {
            "schema_version": 1,
            "arm": arm,
            "seed": seed,
            "raw_coordinates": controller.raw_coordinates.detach().cpu(),
            "optimizer_state_dict": optimizer.state_dict(),
            "schedule_sha256": schedule_hash.hexdigest(),
            "training_probe_pool_sha256": _pool_hash(training_pool),
            "evaluation_probe_pool_sha256": _pool_hash(evaluation_pool),
        },
        temporary,
    )
    os.replace(temporary, checkpoint)
    return {
        "arm": arm,
        "seed": seed,
        "trainable_parameters": sum(p.numel() for p in controller.parameters()),
        "initialization_sha256": initialization_hash,
        "schedule_sha256": schedule_hash.hexdigest(),
        "training_probe_pool_sha256": _pool_hash(training_pool),
        "evaluation_probe_pool_sha256": _pool_hash(evaluation_pool),
        "loss_samples": loss_samples,
        "random_evaluation": random_evaluation,
        "coordinate_evaluation": coordinate_evaluation,
        "structured_evaluation": structured_evaluation,
        "learned_raw_coordinates": controller.raw_coordinates.detach().cpu().tolist(),
        "training_wall_seconds": training_seconds,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
    }


def _adjudicate(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    per_seed = []
    for report in seed_reports:
        arms = report["arms"]
        spin = arms["S"]
        random_checks = {}
        for length, spin_row in spin["random_evaluation"].items():
            margins = {
                arm: arms[arm]["random_evaluation"][length][
                    "mean_relative_frobenius_error"
                ]
                - spin_row["mean_relative_frobenius_error"]
                for arm in ("I", "C", "S-broken")
            }
            checks = {
                "mean_at_most_0_01": spin_row["mean_relative_frobenius_error"] <= 0.01,
                "p95_at_most_0_02": spin_row["p95_relative_frobenius_error"] <= 0.02,
                "maximum_at_most_0_05": spin_row["maximum_relative_frobenius_error"]
                <= 0.05,
                "all_comparator_margins_at_least_0_05": all(
                    margin >= 0.05 for margin in margins.values()
                ),
                "broken_at_least_twice_spin": arms["S-broken"]["random_evaluation"][
                    length
                ]["mean_relative_frobenius_error"]
                >= 2.0 * spin_row["mean_relative_frobenius_error"],
            }
            random_checks[length] = {
                "checks": checks,
                "margins": margins,
                "passed": all(checks.values()),
            }
        coordinate = spin["coordinate_evaluation"]
        coordinate_checks = {
            "active_max_at_most_5e_4": coordinate["maximum_active_coordinate_abs_error"]
            <= 5e-4,
            "inactive_rms_at_most_1e_4": coordinate["inactive_coordinate_rms"] <= 1e-4,
        }
        structured = spin["structured_evaluation"]
        structured_checks = {
            "vector_max_at_most_0_01": structured["maximum_vector_relative_error"]
            <= 0.01,
            "positive_max_at_most_0_01": structured["maximum_positive_relative_error"]
            <= 0.01,
        }
        pairing_fields = (
            "trainable_parameters",
            "initialization_sha256",
            "schedule_sha256",
            "training_probe_pool_sha256",
            "evaluation_probe_pool_sha256",
        )
        pairing = all(
            len({json.dumps(arms[arm][field], sort_keys=True) for arm in ARM_NAMES})
            == 1
            for field in pairing_fields
        )
        passed = (
            all(row["passed"] for row in random_checks.values())
            and all(coordinate_checks.values())
            and all(structured_checks.values())
            and pairing
        )
        per_seed.append(
            {
                "seed": report["seed"],
                "random": random_checks,
                "coordinate_checks": coordinate_checks,
                "structured_checks": structured_checks,
                "exact_pairing": pairing,
                "passed": passed,
            }
        )
    passed = all(row["passed"] for row in per_seed)
    return {
        "passed": passed,
        "decision": (
            "G15A-S passes spanning-chart and center-sensitive transfer"
            if passed
            else "G15A-S fails spanning-chart and center-sensitive transfer"
        ),
        "per_seed": per_seed,
    }


def _validate_predecessor(path: Path) -> dict[str, Any]:
    actual = _sha256(path)
    if actual != G15AR_ARTIFACT_SHA256:
        raise RuntimeError("G15A-R predecessor artifact hash mismatch")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not bool(report["confirmation"]["adjudication"]["passed"]):
        raise RuntimeError("G15A-R predecessor did not pass")
    return {"path": str(path), "sha256": actual}


def run(
    config: SpanningConfig,
    *,
    output: Path,
    predecessor: Path,
    checkpoint_directory: Path,
    device: torch.device,
) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("G15A-S requires CUDA")
    if torch.cuda.get_device_capability(device) != (7, 5):
        raise RuntimeError("G15A-S quality requires exact SM75")
    predecessor_record = _validate_predecessor(predecessor)
    commit, status = _git_state()
    if config.mode == "quality" and status:
        raise RuntimeError("G15A-S quality must start from a clean committed worktree")
    protocol_hash = _sha256(PROTOCOL)
    started = time.perf_counter()
    seed_reports = []
    structural = []
    for seed in config.seeds:
        training_pool = _probe_pool(seed, "training", config.probe_pool_size)
        evaluation_pool = _probe_pool(seed, "evaluation", config.probe_pool_size)
        if _pool_hash(training_pool) == _pool_hash(evaluation_pool):
            raise RuntimeError("training and evaluation probe pools collide")
        training_certificate = _pool_certificate(training_pool)
        evaluation_certificate = _pool_certificate(evaluation_pool)
        oracle_certificate = _oracle_certificate(seed, evaluation_pool)
        if not (
            training_certificate["all_passed"]
            and evaluation_certificate["all_passed"]
            and oracle_certificate["passed"]
        ):
            raise RuntimeError(f"G15A-S structural certificate failed for seed {seed}")
        structural.append(
            {
                "seed": seed,
                "training_pool_sha256": _pool_hash(training_pool),
                "evaluation_pool_sha256": _pool_hash(evaluation_pool),
                "training_pool": training_certificate,
                "evaluation_pool": evaluation_certificate,
                "oracle": oracle_certificate,
            }
        )
        arms = {}
        for arm in ARM_NAMES:
            arms[arm] = _train_arm(
                arm,
                config,
                seed=seed,
                training_pool=training_pool,
                evaluation_pool=evaluation_pool,
                device=device,
                checkpoint_directory=checkpoint_directory,
            )
            gc.collect()
            torch.cuda.empty_cache()
        seed_reports.append({"seed": seed, "arms": arms})
    adjudication = _adjudicate(seed_reports)
    report = {
        "schema_version": 1,
        "stage": "G15A-S",
        "created_at": _now(),
        "mode": config.mode,
        "config": asdict(config),
        "step_angle": STEP_ANGLE,
        "action_vocabulary": ACTION_VOCABULARY,
        "controller_parameters": VOCABULARY_SIZE * PAIR_COUNT,
        "arms": list(ARM_NAMES),
        "predecessor": predecessor_record,
        "protocol": str(PROTOCOL),
        "protocol_sha256": protocol_hash,
        "git_commit_at_start": commit,
        "git_status_at_start": status,
        "structural_certificates": structural,
        "seed_reports": seed_reports,
        "adjudication": adjudication,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        },
        "elapsed_wall_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "composition-only 28-generator signed dictionary transfer to unseen "
            "frame banks and global center words under oracle edit timing; not "
            "learned topology, association, language, or full triality utility"
        ),
    }
    _atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "quality"), default="smoke")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = quality_config() if args.mode == "quality" else smoke_config()
    report = run(
        config,
        output=args.output,
        predecessor=args.predecessor,
        checkpoint_directory=args.checkpoint_dir,
        device=torch.device(args.device),
    )
    print(args.output)
    print(json.dumps(report["adjudication"], indent=2, sort_keys=True))
    if args.mode == "quality" and not report["adjudication"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
