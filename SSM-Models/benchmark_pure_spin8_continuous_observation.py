"""Noisy continuous-observation Pure Spin(8) identification benchmark.

Every input is a fresh real observation of a hidden seven-coordinate Spin(8)
increment.  A seed-specific injective nonlinear chart and additive noise hide
the Lie coordinates.  Training supervises every triality prefix but excludes
adjacent half-center events.  Evaluation forces complementary continuous
half-center observations whose actions compose to the nontrivial center and
pairs them with continuous inverse observations composing to identity.

The decisive structural control is an independent three-SO(8) tracker with
the same 24 recurrent scalars and a parameter-near observation router.  It can
represent each triality stream but does not require one shared Spin(8) group
element.  Mamba-2, parameter-near GRU, state-matched GRU, and observation-only
controls separate other capacity axes.  This is a synthetic online action-
identification test, not a natural-data or language-model benchmark.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import Mamba2Config, Mamba2Model

from benchmark_pure_spin8_latent_increment import teacher_initial_state
from pure_spin8_ssm import __version__ as PURE_SPIN8_VERSION
from pure_spin8_ssm.torch_backend import (
    PureSpin8SSMLayer,
    Spin8AffineTransition,
    apply_spin8_affine,
    spin8_factorized_actions,
    spin8_group_actions,
    work_efficient_spin8_scan,
)
from spin8_triality import (
    SPIN8_BIVECTOR_DIM,
    SPIN8_DIM,
    TRIALITY_REPRESENTATIONS,
    torch_triality_generators,
)

PROTOCOL_DEVELOPMENT_STARTED_AT = "2026-08-17T04:02:10+02:00"
PROTOCOL_SPLIT_CORRECTION_AT = "2026-08-17T04:15:00+02:00"
OBSERVATION_DIMENSION = 12
ACTIVE_COORDINATES = 7
HALF_CENTER_EVENT = 1
REGULAR_EVENT = 0
PARAMETER_NEAR_CANDIDATES = (
    "shared_pure_spin8",
    "independent_so8_triplet",
    "mamba2_parameter_near",
    "gru_parameter_near",
    "observation_only_ablation",
)
CANDIDATES = (*PARAMETER_NEAR_CANDIDATES, "gru_state_matched")
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "experiments"
    / "artifacts"
    / "pure_spin8_continuous_observation_development_seed0.json"
)
DEFAULT_CHECKPOINT_DIRECTORY = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "pure_spin8_continuous_observation_development"
)


@dataclass(frozen=True)
class ContinuousObservationConfig:
    steps: int = 800
    batch_size: int = 32
    training_length: int = 16
    evaluation_pairs: int = 64
    evaluation_lengths: tuple[int, ...] = (16, 64, 128)
    evaluation_microbatch_size: int = 32
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    observation_noise_std: float = 0.01
    half_center_probability: float = 0.12
    regular_coordinate_std: float = 0.40
    half_center_delta: float = 0.25
    seed: int = 0


@dataclass(frozen=True)
class ObservationSystem:
    projection: torch.Tensor
    bias: torch.Tensor


@dataclass(frozen=True)
class TrainingBatch:
    observations: torch.Tensor
    targets: torch.Tensor
    coordinates: torch.Tensor
    events: torch.Tensor


@dataclass(frozen=True)
class ContinuousRelationBatch:
    observations: torch.Tensor
    targets: torch.Tensor
    coordinates: torch.Tensor
    post_relation_mask: torch.Tensor
    relation_position: str


def now() -> str:
    return datetime.now().astimezone().isoformat()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def tensor_hash(tensors: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def make_observation_system(seed: int) -> ObservationSystem:
    generator = torch.Generator().manual_seed(810_000 + seed)
    raw = torch.randn(OBSERVATION_DIMENSION, ACTIVE_COORDINATES, generator=generator)
    projection = torch.linalg.qr(raw, mode="reduced").Q.T * 1.6
    bias = 0.10 * torch.randn(OBSERVATION_DIMENSION, generator=generator)
    return ObservationSystem(projection=projection, bias=bias)


def observe_coordinates(
    coordinates: torch.Tensor,
    system: ObservationSystem,
    *,
    noise_std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    active = coordinates[..., :ACTIVE_COORDINATES] / math.pi
    warped = active + 0.15 * active.pow(3)
    clean = torch.tanh(warped @ system.projection + system.bias)
    noise = noise_std * torch.randn(clean.shape, generator=generator)
    return clean + noise


def action_table_from_coordinates(
    coordinates: torch.Tensor, *, device: torch.device
) -> torch.Tensor:
    coordinates = coordinates.to(device)
    generators = torch_triality_generators(
        TRIALITY_REPRESENTATIONS,
        dtype=coordinates.dtype,
        device=device,
    )
    return spin8_group_actions(
        coordinates[..., None, :],
        generators,
        TRIALITY_REPRESENTATIONS,
        mode="factorized",
    )[..., 0, :, :, :]


@torch.no_grad()
def teacher_outputs(
    coordinates: torch.Tensor, device: torch.device
) -> torch.Tensor:
    actions = action_table_from_coordinates(coordinates, device=device)
    batch, length = coordinates.shape[:2]
    transition = Spin8AffineTransition(
        scale=torch.ones(batch, length, 1, device=device),
        action=actions[:, :, None],
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
    initial = teacher_initial_state().to(device).expand(batch, -1, -1)
    return apply_spin8_affine(prefixes, initial[:, None, None])[:, :, 0].cpu()


def teacher_contract(device: torch.device) -> dict[str, float | bool]:
    coordinates = torch.zeros(4, SPIN8_BIVECTOR_DIM)
    coordinates[0, 0] = math.pi + 0.17
    coordinates[1, 0] = math.pi - 0.17
    coordinates[2, 0] = 0.83
    coordinates[3, 0] = -0.83
    actions = action_table_from_coordinates(coordinates, device=device).double()
    center = actions[1] @ actions[0]
    identity_control = actions[3] @ actions[2]
    identity = torch.eye(SPIN8_DIM, dtype=torch.float64, device=device)
    result = {
        "center_vector_identity_max_abs": float((center[0] - identity).abs().max()),
        "center_spinor_minus_identity_max_abs": float(
            (center[1:] + identity).abs().max()
        ),
        "inverse_pair_identity_max_abs": float(
            (identity_control - identity).abs().max()
        ),
    }
    result["passed"] = all(value < 1e-6 for value in result.values())
    return result


def _sample_coordinate_sequence(
    generator: np.random.Generator,
    rows: int,
    length: int,
    config: ContinuousObservationConfig,
    *,
    forbid_first_half: bool = False,
    forbid_last_half: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = np.zeros((rows, length, SPIN8_BIVECTOR_DIM), dtype=np.float32)
    events = np.zeros((rows, length), dtype=np.int64)
    for row in range(rows):
        previous_half = False
        for position in range(length):
            half = bool(generator.random() < config.half_center_probability)
            if previous_half:
                half = False
            if position == 0 and forbid_first_half:
                half = False
            if position == length - 1 and forbid_last_half:
                half = False
            if half:
                delta = generator.uniform(
                    -config.half_center_delta, config.half_center_delta
                )
                coordinates[row, position, 0] = math.pi + delta
                events[row, position] = HALF_CENTER_EVENT
            else:
                values = generator.normal(
                    0.0, config.regular_coordinate_std, ACTIVE_COORDINATES
                )
                coordinates[row, position, :ACTIVE_COORDINATES] = np.clip(
                    values, -0.9, 0.9
                )
            previous_half = half
    return torch.from_numpy(coordinates), torch.from_numpy(events)


def make_training_schedule(
    config: ContinuousObservationConfig,
    system: ObservationSystem,
    device: torch.device,
) -> list[TrainingBatch]:
    coordinate_generator = np.random.default_rng(820_000 + config.seed)
    noise_generator = torch.Generator().manual_seed(830_000 + config.seed)
    schedule = []
    for step in range(config.steps):
        coordinates, events = _sample_coordinate_sequence(
            coordinate_generator,
            config.batch_size,
            config.training_length,
            config,
        )
        observations = observe_coordinates(
            coordinates,
            system,
            noise_std=config.observation_noise_std,
            generator=noise_generator,
        )
        schedule.append(
            TrainingBatch(
                observations=observations,
                targets=teacher_outputs(coordinates, device),
                coordinates=coordinates,
                events=events,
            )
        )
    return schedule


def training_split_audit(schedule: Sequence[TrainingBatch]) -> dict[str, Any]:
    events = torch.cat([batch.events for batch in schedule])
    observations = torch.cat([batch.observations for batch in schedule])
    coordinates = torch.cat([batch.coordinates for batch in schedule])
    targets = torch.cat([batch.targets for batch in schedule])
    adjacent_half = int(
        ((events[:, :-1] == HALF_CENTER_EVENT) & (events[:, 1:] == HALF_CENTER_EVENT))
        .sum()
        .item()
    )
    flattened = observations.reshape(-1, OBSERVATION_DIMENSION).numpy()
    row_bytes = np.ascontiguousarray(flattened).view(
        np.dtype((np.void, flattened.dtype.itemsize * flattened.shape[1]))
    )
    unique_observations = int(np.unique(row_bytes).size)
    half_values = coordinates[..., 0][events == HALF_CENTER_EVENT]
    checks = {
        "held_out_adjacent_half_center_count_zero": adjacent_half == 0,
        "half_center_events_present": int((events == HALF_CENTER_EVENT).sum()) > 0,
        "regular_events_present": int((events == REGULAR_EVENT).sum()) > 0,
        "half_center_delta_spans_both_signs": bool(
            (half_values < math.pi).any() and (half_values > math.pi).any()
        ),
        "every_observation_is_unique": unique_observations == flattened.shape[0],
        "observations_are_finite": bool(torch.isfinite(observations).all()),
        "targets_are_finite": bool(torch.isfinite(targets).all()),
    }
    return {
        "schedule_sha256": tensor_hash(
            [
                value
                for batch in schedule
                for value in (
                    batch.observations,
                    batch.targets,
                    batch.coordinates,
                    batch.events,
                )
            ]
        ),
        "observation_count": int(flattened.shape[0]),
        "unique_observation_count": unique_observations,
        "half_center_event_count": int((events == HALF_CENTER_EVENT).sum()),
        "regular_event_count": int((events == REGULAR_EVENT).sum()),
        "held_out_adjacent_half_center_count": adjacent_half,
        "half_center_coordinate_range": [
            float(half_values.min()),
            float(half_values.max()),
        ],
        "checks": checks,
        "passed": all(checks.values()),
    }


def make_relation_batch(
    config: ContinuousObservationConfig,
    system: ObservationSystem,
    length: int,
    relation_position: str,
    device: torch.device,
) -> ContinuousRelationBatch:
    if relation_position not in ("early", "late"):
        raise ValueError("relation_position must be early or late")
    if length < 2:
        raise ValueError("relation length must be at least two")
    coordinate_generator = np.random.default_rng(
        840_000
        + 10_000 * config.seed
        + 10 * length
        + int(relation_position == "late")
    )
    context_coordinates, _ = _sample_coordinate_sequence(
        coordinate_generator,
        config.evaluation_pairs,
        length - 2,
        config,
        forbid_first_half=relation_position == "early",
        forbid_last_half=relation_position == "late",
    )
    center = torch.zeros(config.evaluation_pairs, 2, SPIN8_BIVECTOR_DIM)
    identity = torch.zeros_like(center)
    delta = torch.from_numpy(
        coordinate_generator.uniform(
            -config.half_center_delta,
            config.half_center_delta,
            config.evaluation_pairs,
        ).astype(np.float32)
    )
    beta = torch.from_numpy(
        coordinate_generator.uniform(0.45, 1.15, config.evaluation_pairs).astype(
            np.float32
        )
    )
    center[:, 0, 0] = math.pi + delta
    center[:, 1, 0] = math.pi - delta
    identity[:, 0, 0] = beta
    identity[:, 1, 0] = -beta
    if relation_position == "early":
        center_coordinates = torch.cat((center, context_coordinates), dim=1)
        identity_coordinates = torch.cat((identity, context_coordinates), dim=1)
        relation_start = 0
    else:
        center_coordinates = torch.cat((context_coordinates, center), dim=1)
        identity_coordinates = torch.cat((context_coordinates, identity), dim=1)
        relation_start = length - 2

    noise_generator = torch.Generator().manual_seed(
        850_000
        + 10_000 * config.seed
        + 10 * length
        + int(relation_position == "late")
    )
    context_observations = observe_coordinates(
        context_coordinates,
        system,
        noise_std=config.observation_noise_std,
        generator=noise_generator,
    )
    center_relation_observations = observe_coordinates(
        center,
        system,
        noise_std=config.observation_noise_std,
        generator=noise_generator,
    )
    identity_relation_observations = observe_coordinates(
        identity,
        system,
        noise_std=config.observation_noise_std,
        generator=noise_generator,
    )
    if relation_position == "early":
        center_observations = torch.cat(
            (center_relation_observations, context_observations), dim=1
        )
        identity_observations = torch.cat(
            (identity_relation_observations, context_observations), dim=1
        )
    else:
        center_observations = torch.cat(
            (context_observations, center_relation_observations), dim=1
        )
        identity_observations = torch.cat(
            (context_observations, identity_relation_observations), dim=1
        )
    observations = torch.empty(
        2 * config.evaluation_pairs, length, OBSERVATION_DIMENSION
    )
    coordinates = torch.empty(
        2 * config.evaluation_pairs, length, SPIN8_BIVECTOR_DIM
    )
    observations[0::2] = center_observations
    observations[1::2] = identity_observations
    coordinates[0::2] = center_coordinates
    coordinates[1::2] = identity_coordinates
    mask = torch.zeros(2 * config.evaluation_pairs, length, dtype=torch.bool)
    mask[:, relation_start + 1 :] = True
    return ContinuousRelationBatch(
        observations=observations,
        targets=teacher_outputs(coordinates, device),
        coordinates=coordinates,
        post_relation_mask=mask,
        relation_position=relation_position,
    )


def relation_batch_audit(batch: ContinuousRelationBatch) -> dict[str, Any]:
    length = batch.observations.shape[1]
    relation_start = 0 if batch.relation_position == "early" else length - 2
    center_coordinates = batch.coordinates[0::2]
    identity_coordinates = batch.coordinates[1::2]
    if batch.relation_position == "early":
        paired_context = torch.equal(
            batch.observations[0::2, 2:], batch.observations[1::2, 2:]
        )
    else:
        paired_context = torch.equal(
            batch.observations[0::2, :-2], batch.observations[1::2, :-2]
        )
    center_sum = center_coordinates[
        :, relation_start : relation_start + 2, 0
    ].sum(dim=1)
    identity_sum = identity_coordinates[
        :, relation_start : relation_start + 2, 0
    ].sum(dim=1)
    center_targets = batch.targets[0::2, -1]
    identity_targets = batch.targets[1::2, -1]
    checks = {
        "center_coordinates_sum_to_two_pi": bool(
            torch.allclose(
                center_sum,
                torch.full_like(center_sum, 2.0 * math.pi),
                atol=5e-7,
                rtol=0.0,
            )
        ),
        "identity_coordinates_sum_to_zero": bool(
            torch.allclose(identity_sum, torch.zeros_like(identity_sum), atol=1e-7)
        ),
        "paired_context_observations_identical": paired_context,
        "relation_observations_are_distinct": not torch.equal(
            batch.observations[0::2, relation_start : relation_start + 2],
            batch.observations[1::2, relation_start : relation_start + 2],
        ),
        "post_mask_starts_after_second_relation_observation": int(
            batch.post_relation_mask[0].nonzero()[0]
        )
        == relation_start + 1,
        "teacher_vector_targets_agree": float(
            (center_targets[:, 0] - identity_targets[:, 0]).abs().max()
        )
        < 2e-6,
        "teacher_spinor_targets_negate": float(
            (center_targets[:, 1:] + identity_targets[:, 1:]).abs().max()
        )
        < 2e-6,
        "all_values_finite": bool(
            torch.isfinite(batch.observations).all()
            and torch.isfinite(batch.coordinates).all()
            and torch.isfinite(batch.targets).all()
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


class SharedPureSpin8Tracker(nn.Module):
    recurrent_state_scalars = 24

    def __init__(self) -> None:
        super().__init__()
        self.observation_hidden = nn.Linear(OBSERVATION_DIMENSION, 22)
        self.coordinate_head = nn.Linear(22, SPIN8_BIVECTOR_DIM)
        self.layer = PureSpin8SSMLayer(
            SPIN8_BIVECTOR_DIM,
            channels=1,
            representations=TRIALITY_REPRESENTATIONS,
            action_mode="factorized",
            triality_coupling=False,
            transport_only=True,
            normalize_inputs=False,
        )
        with torch.no_grad():
            self.layer.coefficient_controller.weight.copy_(
                torch.eye(SPIN8_BIVECTOR_DIM)
            )
            self.layer.coefficient_controller.bias.zero_()
            self.layer.initial_state.copy_(teacher_initial_state())
            self.layer.coupling_logits.zero_()
        for parameter in self.layer.parameters():
            parameter.requires_grad_(False)
        nn.init.normal_(self.observation_hidden.weight, std=0.12)
        nn.init.zeros_(self.observation_hidden.bias)
        nn.init.normal_(self.coordinate_head.weight, std=0.05)
        nn.init.zeros_(self.coordinate_head.bias)

    def observation_coordinates(self, observations: torch.Tensor) -> torch.Tensor:
        return self.coordinate_head(F.silu(self.observation_hidden(observations)))

    def observation_actions(self, observations: torch.Tensor) -> torch.Tensor:
        coordinates = self.observation_coordinates(observations)
        return spin8_group_actions(
            coordinates[..., None, :],
            self.layer.generators.to(observations),
            TRIALITY_REPRESENTATIONS,
            mode="factorized",
        )[..., 0, :, :, :]

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        coordinates = self.observation_coordinates(observations)
        states, _ = self.layer(
            coordinates,
            scan_mode="work_efficient",
            return_raw_states=True,
        )
        return states[:, :, 0]


class IndependentSO8TripletTracker(nn.Module):
    recurrent_state_scalars = 24

    def __init__(self) -> None:
        super().__init__()
        self.observation_hidden = nn.Linear(OBSERVATION_DIMENSION, 9)
        self.coordinate_head = nn.Linear(
            9, len(TRIALITY_REPRESENTATIONS) * SPIN8_BIVECTOR_DIM
        )
        self.register_buffer(
            "generators",
            torch_triality_generators(TRIALITY_REPRESENTATIONS),
            persistent=True,
        )
        self.register_buffer(
            "initial_state", teacher_initial_state()[0], persistent=True
        )
        nn.init.normal_(self.observation_hidden.weight, std=0.12)
        nn.init.zeros_(self.observation_hidden.bias)
        nn.init.normal_(self.coordinate_head.weight, std=0.04)
        nn.init.zeros_(self.coordinate_head.bias)

    def observation_coordinates(self, observations: torch.Tensor) -> torch.Tensor:
        return self.coordinate_head(
            F.silu(self.observation_hidden(observations))
        ).reshape(
            *observations.shape[:-1],
            len(TRIALITY_REPRESENTATIONS),
            SPIN8_BIVECTOR_DIM,
        )

    def observation_actions(self, observations: torch.Tensor) -> torch.Tensor:
        coordinates = self.observation_coordinates(observations)
        actions = []
        for index, representation in enumerate(TRIALITY_REPRESENTATIONS):
            actions.append(
                spin8_factorized_actions(
                    coordinates[..., index, :],
                    self.generators[index : index + 1].to(observations),
                    (representation,),
                )[..., 0, :, :]
            )
        return torch.stack(actions, dim=-3)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        actions = self.observation_actions(observations)
        batch, length = observations.shape[:2]
        transition = Spin8AffineTransition(
            scale=torch.ones(batch, length, 1, device=observations.device),
            action=actions[:, :, None],
            drive=torch.zeros(
                batch,
                length,
                1,
                len(TRIALITY_REPRESENTATIONS),
                SPIN8_DIM,
                device=observations.device,
            ),
        )
        prefixes = work_efficient_spin8_scan(transition)
        initial = self.initial_state.to(observations).expand(batch, -1, -1)
        return apply_spin8_affine(prefixes, initial[:, None, None])[:, :, 0]


class ContinuousMamba2Tracker(nn.Module):
    recurrent_state_scalars = 160

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(OBSERVATION_DIMENSION, 8)
        self.backbone = Mamba2Model(
            Mamba2Config(
                vocab_size=8,
                hidden_size=8,
                state_size=4,
                num_hidden_layers=1,
                num_heads=1,
                head_dim=16,
                expand=2,
                conv_kernel=4,
                n_groups=1,
                use_cache=False,
            )
        )
        self.backbone.embeddings = nn.Identity()
        self.output = nn.Linear(8, 24)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(
            inputs_embeds=self.input_projection(observations), use_cache=False
        ).last_hidden_state
        return self.output(hidden).reshape(
            observations.shape[0],
            observations.shape[1],
            len(TRIALITY_REPRESENTATIONS),
            SPIN8_DIM,
        )


class ParameterNearGRUTracker(nn.Module):
    recurrent_state_scalars = 10

    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(OBSERVATION_DIMENSION, 10, batch_first=True)
        self.output = nn.Linear(10, 24, bias=False)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gru(observations)
        return self.output(hidden).reshape(
            observations.shape[0],
            observations.shape[1],
            len(TRIALITY_REPRESENTATIONS),
            SPIN8_DIM,
        )


class StateMatchedGRUTracker(nn.Module):
    recurrent_state_scalars = 24

    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(OBSERVATION_DIMENSION, 24, batch_first=True)
        self.output = nn.Linear(24, 24, bias=False)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gru(observations)
        return self.output(hidden).reshape(
            observations.shape[0],
            observations.shape[1],
            len(TRIALITY_REPRESENTATIONS),
            SPIN8_DIM,
        )


class ObservationOnlyAblation(nn.Module):
    recurrent_state_scalars = 0

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(OBSERVATION_DIMENSION, 25)
        self.output = nn.Linear(25, 24)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        output = self.output(F.silu(self.hidden(observations)))
        return output.reshape(
            observations.shape[0],
            observations.shape[1],
            len(TRIALITY_REPRESENTATIONS),
            SPIN8_DIM,
        )


def build_models() -> dict[str, nn.Module]:
    models: dict[str, nn.Module] = {
        "shared_pure_spin8": SharedPureSpin8Tracker(),
        "independent_so8_triplet": IndependentSO8TripletTracker(),
        "mamba2_parameter_near": ContinuousMamba2Tracker(),
        "gru_parameter_near": ParameterNearGRUTracker(),
        "observation_only_ablation": ObservationOnlyAblation(),
        "gru_state_matched": StateMatchedGRUTracker(),
    }
    counts = {name: parameter_count(model) for name, model in models.items()}
    near = [counts[name] for name in PARAMETER_NEAR_CANDIDATES]
    if (max(near) - min(near)) / max(near) > 0.05:
        raise RuntimeError(f"parameter-near candidates exceed 5% spread: {counts}")
    if models["shared_pure_spin8"].recurrent_state_scalars != models[
        "independent_so8_triplet"
    ].recurrent_state_scalars:
        raise RuntimeError("shared and independent orthogonal controls must match state")
    return models


@torch.no_grad()
def evaluate_relation_batches(
    model: nn.Module,
    batches: Sequence[ContinuousRelationBatch],
    device: torch.device,
    microbatch_size: int,
) -> dict[str, float]:
    model.eval()
    squared_error_sum = 0.0
    scalar_count = 0
    post_error_sum = 0.0
    post_scalar_count = 0
    final_error_sum = 0.0
    final_scalar_count = 0
    center_correct = 0
    identity_correct = 0
    pair_count = 0
    vector_pair_squared = 0.0
    spinor_pair_squared = 0.0
    pair_vector_scalars = 0
    pair_spinor_scalars = 0
    teacher_vector_residual = 0.0
    teacher_spinor_residual = 0.0
    for batch in batches:
        outputs = []
        for start in range(0, batch.observations.shape[0], microbatch_size):
            outputs.append(
                model(
                    batch.observations[start : start + microbatch_size].to(device)
                ).cpu()
            )
        predictions = torch.cat(outputs)
        targets = batch.targets
        errors = predictions - targets
        squared_error_sum += float(errors.square().sum())
        scalar_count += errors.numel()
        selected = errors[batch.post_relation_mask]
        post_error_sum += float(selected.square().sum())
        post_scalar_count += selected.numel()
        final = errors[:, -1]
        final_error_sum += float(final.square().sum())
        final_scalar_count += final.numel()
        center_prediction = predictions[0::2, -1]
        identity_prediction = predictions[1::2, -1]
        center_target = targets[0::2, -1]
        identity_target = targets[1::2, -1]
        center_correct += int(
            (
                (center_prediction - center_target).square().flatten(1).sum(1)
                < (center_prediction - identity_target).square().flatten(1).sum(1)
            ).sum()
        )
        identity_correct += int(
            (
                (identity_prediction - identity_target).square().flatten(1).sum(1)
                < (identity_prediction - center_target).square().flatten(1).sum(1)
            ).sum()
        )
        pair_count += center_prediction.shape[0]
        vector_pair_squared += float(
            (center_prediction[:, 0] - identity_prediction[:, 0]).square().sum()
        )
        spinor_pair_squared += float(
            torch.cat(
                (
                    center_prediction[:, 1] + identity_prediction[:, 1],
                    center_prediction[:, 2] + identity_prediction[:, 2],
                ),
                dim=-1,
            )
            .square()
            .sum()
        )
        pair_vector_scalars += center_prediction[:, 0].numel()
        pair_spinor_scalars += center_prediction[:, 1:].numel()
        teacher_vector_residual = max(
            teacher_vector_residual,
            float((center_target[:, 0] - identity_target[:, 0]).abs().max()),
        )
        teacher_spinor_residual = max(
            teacher_spinor_residual,
            float((center_target[:, 1:] + identity_target[:, 1:]).abs().max()),
        )
    return {
        "all_prefix_mse": squared_error_sum / scalar_count,
        "post_relation_mse": post_error_sum / post_scalar_count,
        "final_mse": final_error_sum / final_scalar_count,
        "center_classification_accuracy": (center_correct + identity_correct)
        / (2 * pair_count),
        "center_rows_correct": center_correct / pair_count,
        "identity_rows_correct": identity_correct / pair_count,
        "predicted_vector_pair_rmse": math.sqrt(
            vector_pair_squared / pair_vector_scalars
        ),
        "predicted_spinor_negation_rmse": math.sqrt(
            spinor_pair_squared / pair_spinor_scalars
        ),
        "teacher_vector_pair_max_abs": teacher_vector_residual,
        "teacher_spinor_negation_max_abs": teacher_spinor_residual,
    }


@torch.no_grad()
def action_identification_diagnostics(
    model: SharedPureSpin8Tracker | IndependentSO8TripletTracker,
    batches: dict[str, list[ContinuousRelationBatch]],
    device: torch.device,
) -> dict[str, Any]:
    squared_error = 0.0
    scalar_count = 0
    relation_residuals = {}
    for key, values in batches.items():
        batch = values[0]
        observations = batch.observations.to(device)
        predicted = model.observation_actions(observations)
        target = action_table_from_coordinates(batch.coordinates, device=device)
        squared_error += float((predicted - target).square().sum())
        scalar_count += predicted.numel()
        relation_start = 0 if batch.relation_position == "early" else observations.shape[1] - 2
        center = predicted[0::2, relation_start + 1] @ predicted[
            0::2, relation_start
        ]
        identity_control = predicted[1::2, relation_start + 1] @ predicted[
            1::2, relation_start
        ]
        identity = torch.eye(SPIN8_DIM, device=device)
        relation_residuals[key] = {
            "center_vector_identity_rmse": float(
                (center[:, 0] - identity).square().mean().sqrt()
            ),
            "center_spinor_minus_identity_rmse": float(
                (center[:, 1:] + identity).square().mean().sqrt()
            ),
            "identity_pair_rmse": float(
                (identity_control - identity).square().mean().sqrt()
            ),
        }
    return {
        "action_rmse": math.sqrt(squared_error / scalar_count),
        "relation_action_residuals": relation_residuals,
    }


def train_candidate(
    name: str,
    factory: Callable[[], nn.Module],
    schedule: Sequence[TrainingBatch],
    evaluations: dict[str, list[ContinuousRelationBatch]],
    config: ContinuousObservationConfig,
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
        targets = batch.targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(observations)
        loss = F.mse_loss(predictions, targets)
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
                f"{name} seed={config.seed} step={step}/{config.steps} "
                f"loss={samples[str(step)]:.8f}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result: dict[str, Any] = {
        "parameters": parameter_count(model),
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
            key: evaluate_relation_batches(
                model, batches, device, config.evaluation_microbatch_size
            )
            for key, batches in evaluations.items()
        },
    }
    if isinstance(model, (SharedPureSpin8Tracker, IndependentSO8TripletTracker)):
        result["action_identification"] = action_identification_diagnostics(
            model, evaluations, device
        )
    if checkpoint_directory is not None:
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_directory / f"{name}_seed{config.seed}_step{config.steps}.pt"
        torch.save(
            {
                "format_version": 1,
                "candidate": name,
                "pure_spin8_version": (
                    PURE_SPIN8_VERSION if name == "shared_pure_spin8" else None
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
    config: ContinuousObservationConfig,
    *,
    device: torch.device,
    checkpoint_directory: Path | None,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    contract = teacher_contract(device)
    if not contract["passed"]:
        raise RuntimeError("teacher relation contract failed")
    system = make_observation_system(config.seed)
    schedule = make_training_schedule(config, system, device)
    split = training_split_audit(schedule)
    if not split["passed"]:
        raise RuntimeError("training split audit failed")
    evaluations = {}
    evaluation_audits = {}
    evaluation_hashes = {}
    for length in config.evaluation_lengths:
        for position in ("early", "late"):
            key = f"{position}_L{length}"
            batch = make_relation_batch(config, system, length, position, device)
            evaluations[key] = [batch]
            evaluation_audits[key] = relation_batch_audit(batch)
            if not evaluation_audits[key]["passed"]:
                raise RuntimeError(f"evaluation split audit failed for {key}")
            evaluation_hashes[key] = tensor_hash(
                (
                    batch.observations,
                    batch.targets,
                    batch.coordinates,
                    batch.post_relation_mask,
                )
            )

    shapes = build_models()
    counts = {name: parameter_count(model) for name, model in shapes.items()}
    states = {
        name: int(model.recurrent_state_scalars) for name, model in shapes.items()
    }
    del shapes
    factories: dict[str, Callable[[], nn.Module]] = {
        "shared_pure_spin8": SharedPureSpin8Tracker,
        "independent_so8_triplet": IndependentSO8TripletTracker,
        "mamba2_parameter_near": ContinuousMamba2Tracker,
        "gru_parameter_near": ParameterNearGRUTracker,
        "observation_only_ablation": ObservationOnlyAblation,
        "gru_state_matched": StateMatchedGRUTracker,
    }
    results = {}
    for offset, name in enumerate(CANDIDATES):
        seed_everything(860_000 + 1_000 * config.seed + offset)
        results[name] = train_candidate(
            name,
            factories[name],
            schedule,
            evaluations,
            config,
            device,
            checkpoint_directory,
        )
    all_metrics_finite = all(
        math.isfinite(value)
        for result in results.values()
        for metrics in result["evaluation"].values()
        for value in metrics.values()
    )
    near_counts = [counts[name] for name in PARAMETER_NEAR_CANDIDATES]
    return {
        "schema_version": 1,
        "experiment": "noisy continuous-observation Pure Spin8 identification",
        "status": "development" if config.seed == 0 else "unadjudicated",
        "development_protocol_started_at": PROTOCOL_DEVELOPMENT_STARTED_AT,
        "protocol_split_correction_at": PROTOCOL_SPLIT_CORRECTION_AT,
        "recorded_at": now(),
        "pure_spin8_version": PURE_SPIN8_VERSION,
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
        "task": {
            "observation_dimension": OBSERVATION_DIMENSION,
            "active_teacher_coordinates": ACTIVE_COORDINATES,
            "observation_chart": "tanh(P @ (u + 0.15*u^3) + bias) + Gaussian noise",
            "observation_system_sha256": tensor_hash(
                (system.projection, system.bias)
            ),
            "teacher_initial_state_sha256": tensor_hash((teacher_initial_state(),)),
            "teacher_contract": contract,
            "target": "every prefix in 8v, 8+, and 8-",
            "held_out_relation": (
                "two adjacent fresh observations with plane-0 coordinates "
                "pi+delta and pi-delta"
            ),
            "identity_control_relation": (
                "two fresh observations with plane-0 coordinates beta and -beta"
            ),
            "training_split": split,
            "evaluation_audits": evaluation_audits,
            "evaluation_schedule_sha256": evaluation_hashes,
        },
        "integrity": {
            "same_precomputed_schedule_for_every_candidate": True,
            "candidate_initialized_after_seed": True,
            "parameter_counts": counts,
            "parameter_near_candidates": PARAMETER_NEAR_CANDIDATES,
            "maximum_relative_parameter_gap_near_cohort": (
                (max(near_counts) - min(near_counts)) / max(near_counts)
            ),
            "recurrent_state_scalars": states,
            "shared_vs_independent_so8_state_matched": (
                states["shared_pure_spin8"] == states["independent_so8_triplet"]
            ),
            "shared_vs_state_gru_state_matched": (
                states["shared_pure_spin8"] == states["gru_state_matched"]
            ),
            "all_metrics_finite": all_metrics_finite,
        },
        "results": results,
        "claim_scope": {
            "empirical": [
                "single-seed noisy continuous online action-identification development",
                "parameter-near comparison including an exact-state-matched independent orthogonal tracker",
            ],
            "not_claimed": [
                "a replicated or adjudicated cohort",
                "natural-data or language-model utility",
                "measured-compute matching before a separately frozen continuation",
                "a fused-Mamba or fused-training comparison",
                "a theorem about Spin8, Mamba-2, or global optimality",
            ],
        },
        "passed": bool(contract["passed"] and split["passed"] and all_metrics_finite),
    }


def parse_lengths(value: str) -> tuple[int, ...]:
    lengths = tuple(int(item) for item in value.split(",") if item.strip())
    if not lengths or any(length < 2 for length in lengths):
        raise ValueError("evaluation lengths must be distinct integers at least two")
    if len(set(lengths)) != len(lengths):
        raise ValueError("evaluation lengths must be distinct")
    return lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=800)
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
    config = ContinuousObservationConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        training_length=args.training_length,
        evaluation_pairs=args.evaluation_pairs,
        evaluation_lengths=parse_lengths(args.evaluation_lengths),
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
        checkpoint_directory=args.checkpoint_directory,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
