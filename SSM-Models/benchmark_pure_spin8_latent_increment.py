"""Latent-increment Spin(8) relation benchmark with parameter-near controls.

The teacher assigns eight symbolic tokens fixed but hidden Spin(8) Lie
increments.  Training supervises every prefix state in the vector and both
half-spin representations, but never contains the ordered pair ``a,a``.  At
evaluation that pair is forced and produces the nontrivial central action;
``b,b_inverse`` is the paired identity control.

The maintained Pure Spin(8) candidate must infer token-local 28-dimensional
increments before applying its exact triality action and associative scan.
Mamba-2, GRU, and token-only controls receive exactly the same symbolic inputs,
targets, schedules, update count, and optimizer settings.  This is a synthetic
representation-identification gate, not a language-model benchmark.
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

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import Mamba2Config, Mamba2Model

from pure_spin8_ssm import __version__ as PURE_SPIN8_VERSION
from pure_spin8_ssm.torch_backend import (
    PureSpin8SSMLayer,
    Spin8AffineTransition,
    apply_spin8_affine,
    spin8_group_actions,
    work_efficient_spin8_scan,
)
from spin8_triality import (
    SPIN8_BIVECTOR_DIM,
    SPIN8_DIM,
    TRIALITY_REPRESENTATIONS,
    torch_triality_generators,
)

PROTOCOL_DEVELOPMENT_STARTED_AT = "2026-08-17T03:24:00+02:00"
VOCABULARY = (
    "a_half_center",
    "b",
    "b_inverse",
    "c",
    "d",
    "e",
    "f",
    "g",
)
HELD_OUT_PAIR = (0, 0)
IDENTITY_CONTROL_PAIR = (1, 2)
CANDIDATES = (
    "latent_pure_spin8",
    "mamba2_transformers",
    "gru_reference",
    "token_only_ablation",
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "experiments"
    / "artifacts"
    / "pure_spin8_latent_increment_development_seed0.json"
)
DEFAULT_CHECKPOINT_DIRECTORY = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "pure_spin8_latent_increment_development"
)


@dataclass(frozen=True)
class LatentIncrementConfig:
    steps: int = 500
    batch_size: int = 32
    training_length: int = 16
    evaluation_pairs: int = 64
    evaluation_lengths: tuple[int, ...] = (16, 64, 128)
    evaluation_microbatch_size: int = 32
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    seed: int = 0


@dataclass(frozen=True)
class RelationBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
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


def teacher_coordinates() -> torch.Tensor:
    """Return the fixed hidden token increments used by the teacher."""

    coordinates = torch.zeros(len(VOCABULARY), SPIN8_BIVECTOR_DIM)
    coordinates[0, 0] = math.pi
    coordinates[1, 1] = 0.70
    coordinates[2, 1] = -0.70
    coordinates[3, 2] = 0.45
    coordinates[4, 3] = -0.55
    coordinates[5, 4] = 1.10
    coordinates[6, 5] = 0.35
    coordinates[7, 6] = -0.80
    return coordinates


def teacher_initial_state() -> torch.Tensor:
    generator = torch.Generator().manual_seed(202_608_17)
    initial = torch.randn(
        1,
        len(TRIALITY_REPRESENTATIONS),
        SPIN8_DIM,
        generator=generator,
    )
    return F.normalize(initial, dim=-1)


def token_action_table(
    coordinates: torch.Tensor, *, device: torch.device
) -> torch.Tensor:
    coordinates = coordinates.to(device)
    generators = torch_triality_generators(
        TRIALITY_REPRESENTATIONS,
        dtype=coordinates.dtype,
        device=device,
    )
    return spin8_group_actions(
        coordinates[:, None],
        generators,
        TRIALITY_REPRESENTATIONS,
        mode="factorized",
    )[:, 0]


def teacher_contract(device: torch.device) -> dict[str, float | bool]:
    actions = token_action_table(teacher_coordinates(), device=device).double()
    identity = torch.eye(SPIN8_DIM, dtype=torch.float64, device=device)
    center = actions[0] @ actions[0]
    identity_control = actions[2] @ actions[1]
    return {
        "a_square_vector_identity_max_abs": float(
            (center[0] - identity).abs().max()
        ),
        "a_square_positive_minus_identity_max_abs": float(
            (center[1] + identity).abs().max()
        ),
        "a_square_negative_minus_identity_max_abs": float(
            (center[2] + identity).abs().max()
        ),
        "b_inverse_b_identity_max_abs": float(
            (identity_control - identity).abs().max()
        ),
        "passed": bool(
            (center[0] - identity).abs().max() < 1e-6
            and (center[1:] + identity).abs().max() < 1e-6
            and (identity_control - identity).abs().max() < 1e-6
        ),
    }


@torch.no_grad()
def teacher_outputs(tokens: torch.Tensor, device: torch.device) -> torch.Tensor:
    actions = token_action_table(teacher_coordinates(), device=device)
    selected = actions[tokens.to(device)]
    batch, length = tokens.shape
    transition = Spin8AffineTransition(
        scale=torch.ones(batch, length, 1, device=device),
        action=selected[:, :, None],
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


def _sample_pair_free_tokens(
    generator: np.random.Generator,
    rows: int,
    length: int,
    *,
    forbid_first_a: bool = False,
    forbid_last_a: bool = False,
) -> np.ndarray:
    tokens = np.empty((rows, length), dtype=np.int64)
    for row in range(rows):
        previous: int | None = None
        for position in range(length):
            while True:
                token = int(generator.integers(0, len(VOCABULARY)))
                if previous == 0 and token == 0:
                    continue
                if position == 0 and forbid_first_a and token == 0:
                    continue
                if position == length - 1 and forbid_last_a and token == 0:
                    continue
                break
            tokens[row, position] = token
            previous = token
    return tokens


def make_training_schedule(
    config: LatentIncrementConfig, device: torch.device
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = np.random.default_rng(500_000 + config.seed)
    schedule = []
    for step in range(config.steps):
        tokens = _sample_pair_free_tokens(
            generator, config.batch_size, config.training_length
        )
        if step == 0:
            tokens[0, : len(VOCABULARY)] = np.arange(len(VOCABULARY))
        inputs = torch.from_numpy(tokens)
        schedule.append((inputs, teacher_outputs(inputs, device)))
    return schedule


def training_split_audit(
    schedule: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, object]:
    token_counts = np.zeros(len(VOCABULARY), dtype=np.int64)
    pair_counts = np.zeros(
        (len(VOCABULARY), len(VOCABULARY)), dtype=np.int64
    )
    tensors = []
    for inputs, targets in schedule:
        values = inputs.numpy()
        token_counts += np.bincount(values.ravel(), minlength=len(VOCABULARY))
        for left, right in zip(values[:, :-1].ravel(), values[:, 1:].ravel()):
            pair_counts[left, right] += 1
        tensors.extend((inputs, targets))
    allowed = np.ones_like(pair_counts, dtype=bool)
    allowed[HELD_OUT_PAIR] = False
    checks = {
        "held_out_a_a_count_is_zero": int(pair_counts[HELD_OUT_PAIR]) == 0,
        "all_tokens_occur": bool(np.all(token_counts > 0)),
        "all_allowed_pairs_occur": bool(np.all(pair_counts[allowed] > 0)),
        "targets_are_finite": all(torch.isfinite(targets).all() for _, targets in schedule),
    }
    return {
        "schedule_sha256": tensor_hash(tensors),
        "token_counts": token_counts.tolist(),
        "held_out_pair": list(HELD_OUT_PAIR),
        "held_out_pair_symbols": [VOCABULARY[index] for index in HELD_OUT_PAIR],
        "held_out_pair_count": int(pair_counts[HELD_OUT_PAIR]),
        "identity_control_pair": list(IDENTITY_CONTROL_PAIR),
        "minimum_allowed_pair_count": int(pair_counts[allowed].min()),
        "checks": checks,
        "passed": all(checks.values()),
    }


def make_relation_batches(
    config: LatentIncrementConfig,
    length: int,
    relation_position: str,
    device: torch.device,
) -> list[RelationBatch]:
    if length < 2:
        raise ValueError("relation evaluation requires length at least two")
    if relation_position not in ("early", "late"):
        raise ValueError("relation_position must be early or late")
    generator = np.random.default_rng(
        600_000
        + 10_000 * config.seed
        + 10 * length
        + int(relation_position == "late")
    )
    contexts = _sample_pair_free_tokens(
        generator,
        config.evaluation_pairs,
        length - 2,
        forbid_first_a=relation_position == "early",
        forbid_last_a=relation_position == "late",
    )
    center = np.tile(np.asarray(HELD_OUT_PAIR), (config.evaluation_pairs, 1))
    identity = np.tile(
        np.asarray(IDENTITY_CONTROL_PAIR), (config.evaluation_pairs, 1)
    )
    if relation_position == "early":
        center_words = np.concatenate((center, contexts), axis=1)
        identity_words = np.concatenate((identity, contexts), axis=1)
        post_start = 1
    else:
        center_words = np.concatenate((contexts, center), axis=1)
        identity_words = np.concatenate((contexts, identity), axis=1)
        post_start = length - 1
    interleaved = np.empty((2 * config.evaluation_pairs, length), dtype=np.int64)
    interleaved[0::2] = center_words
    interleaved[1::2] = identity_words
    inputs = torch.from_numpy(interleaved)
    mask = torch.zeros(inputs.shape, dtype=torch.bool)
    mask[:, post_start:] = True
    return [
        RelationBatch(
            inputs=inputs,
            targets=teacher_outputs(inputs, device),
            post_relation_mask=mask,
            relation_position=relation_position,
        )
    ]


def relation_batch_audit(batch: RelationBatch) -> dict[str, object]:
    inputs = batch.inputs.numpy()
    length = inputs.shape[1]
    relation_start = (
        0 if batch.relation_position == "early" else length - 2
    )
    center_rows = inputs[0::2]
    identity_rows = inputs[1::2]
    context_equal = (
        np.array_equal(center_rows[:, 2:], identity_rows[:, 2:])
        if batch.relation_position == "early"
        else np.array_equal(center_rows[:, :-2], identity_rows[:, :-2])
    )
    checks = {
        "center_rows_force_a_a": bool(
            np.all(
                center_rows[:, relation_start : relation_start + 2]
                == np.asarray(HELD_OUT_PAIR)
            )
        ),
        "identity_rows_force_b_b_inverse": bool(
            np.all(
                identity_rows[:, relation_start : relation_start + 2]
                == np.asarray(IDENTITY_CONTROL_PAIR)
            )
        ),
        "paired_contexts_are_identical": context_equal,
        "post_relation_mask_starts_after_second_relation_token": int(
            batch.post_relation_mask[0].nonzero()[0]
        )
        == relation_start + 1,
        "targets_are_finite": bool(torch.isfinite(batch.targets).all()),
    }
    return {"checks": checks, "passed": all(checks.values())}


class LatentPureSpin8Tracker(nn.Module):
    recurrent_state_scalars = 24

    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(len(VOCABULARY), 24)
        self.coordinate_head = nn.Linear(24, SPIN8_BIVECTOR_DIM)
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
        nn.init.normal_(self.token_embedding.weight, std=0.10)
        nn.init.normal_(self.coordinate_head.weight, std=0.05)
        nn.init.zeros_(self.coordinate_head.bias)

    def token_coordinates(self) -> torch.Tensor:
        tokens = torch.arange(
            len(VOCABULARY), device=self.token_embedding.weight.device
        )
        return self.coordinate_head(F.silu(self.token_embedding(tokens)))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        coordinates = self.coordinate_head(F.silu(self.token_embedding(tokens)))
        states, _ = self.layer(
            coordinates,
            scan_mode="work_efficient",
            return_raw_states=True,
        )
        return states[:, :, 0]


class Mamba2TrialityTracker(nn.Module):
    recurrent_state_scalars = 160

    def __init__(self) -> None:
        super().__init__()
        self.backbone = Mamba2Model(
            Mamba2Config(
                vocab_size=len(VOCABULARY),
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
        self.output = nn.Linear(8, 24)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids=tokens, use_cache=False).last_hidden_state
        return self.output(hidden).reshape(
            tokens.shape[0], tokens.shape[1], len(TRIALITY_REPRESENTATIONS), 8
        )


class GRUTrialityTracker(nn.Module):
    recurrent_state_scalars = 9

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(len(VOCABULARY), 10)
        self.gru = nn.GRU(10, 9, batch_first=True)
        self.output = nn.Linear(9, 24)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gru(self.embedding(tokens))
        return self.output(hidden).reshape(
            tokens.shape[0], tokens.shape[1], len(TRIALITY_REPRESENTATIONS), 8
        )


class TokenOnlyTrialityAblation(nn.Module):
    recurrent_state_scalars = 0

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(len(VOCABULARY), 17)
        self.hidden = nn.Linear(17, 17)
        self.output = nn.Linear(17, 24)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.hidden(self.embedding(tokens)))
        return self.output(hidden).reshape(
            tokens.shape[0], tokens.shape[1], len(TRIALITY_REPRESENTATIONS), 8
        )


def build_models() -> dict[str, nn.Module]:
    models: dict[str, nn.Module] = {
        "latent_pure_spin8": LatentPureSpin8Tracker(),
        "mamba2_transformers": Mamba2TrialityTracker(),
        "gru_reference": GRUTrialityTracker(),
        "token_only_ablation": TokenOnlyTrialityAblation(),
    }
    counts = {name: parameter_count(model) for name, model in models.items()}
    if (max(counts.values()) - min(counts.values())) / max(counts.values()) > 0.03:
        raise RuntimeError(f"candidate parameter counts are not near: {counts}")
    return models


@torch.no_grad()
def evaluate_relation_batches(
    model: nn.Module,
    batches: Sequence[RelationBatch],
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
        for start in range(0, batch.inputs.shape[0], microbatch_size):
            outputs.append(
                model(batch.inputs[start : start + microbatch_size].to(device)).cpu()
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
        "center_classification_accuracy": (
            center_correct + identity_correct
        )
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


def action_identification_diagnostics(
    model: LatentPureSpin8Tracker, device: torch.device
) -> dict[str, object]:
    with torch.no_grad():
        learned_coordinates = model.token_coordinates()
        learned_actions = token_action_table(learned_coordinates, device=device)
        target_actions = token_action_table(teacher_coordinates(), device=device)
        per_token = (
            (learned_actions - target_actions)
            .square()
            .flatten(start_dim=1)
            .mean(dim=1)
            .sqrt()
        )
        identity = torch.eye(SPIN8_DIM, device=device)
        center = learned_actions[0] @ learned_actions[0]
        identity_control = learned_actions[2] @ learned_actions[1]
        return {
            "action_rmse": float(
                (learned_actions - target_actions).square().mean().sqrt()
            ),
            "per_token_action_rmse": per_token.cpu().tolist(),
            "a_square_vector_identity_rmse": float(
                (center[0] - identity).square().mean().sqrt()
            ),
            "a_square_spinor_minus_identity_rmse": float(
                (center[1:] + identity).square().mean().sqrt()
            ),
            "b_inverse_b_identity_rmse": float(
                (identity_control - identity).square().mean().sqrt()
            ),
            "learned_coordinates": learned_coordinates.detach().cpu().tolist(),
        }


def train_candidate(
    name: str,
    factory: Callable[[], nn.Module],
    schedule: Sequence[tuple[torch.Tensor, torch.Tensor]],
    evaluations: dict[str, list[RelationBatch]],
    config: LatentIncrementConfig,
    device: torch.device,
    checkpoint_directory: Path | None,
) -> dict[str, object]:
    model = factory().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    losses = {}
    gradient_norm = torch.tensor(0.0)
    started = time.perf_counter()
    model.train()
    for step, (inputs, targets) in enumerate(schedule, start=1):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = F.mse_loss(predictions, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name} produced nonfinite loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == config.steps:
            losses[str(step)] = float(loss.detach())
            print(
                f"{name} seed={config.seed} step={step}/{config.steps} "
                f"loss={losses[str(step)]:.8f}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result: dict[str, object] = {
        "parameters": parameter_count(model),
        "recurrent_state_scalars": int(model.recurrent_state_scalars),
        "loss_samples": losses,
        "final_training_loss": losses[str(config.steps)],
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
                model,
                batches,
                device,
                config.evaluation_microbatch_size,
            )
            for key, batches in evaluations.items()
        },
    }
    if isinstance(model, LatentPureSpin8Tracker):
        result["action_identification"] = action_identification_diagnostics(
            model, device
        )
    if checkpoint_directory is not None:
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_directory / (
            f"{name}_seed{config.seed}_step{config.steps}.pt"
        )
        torch.save(
            {
                "format_version": 1,
                "candidate": name,
                "pure_spin8_version": (
                    PURE_SPIN8_VERSION if name == "latent_pure_spin8" else None
                ),
                "config": asdict(config),
                "vocabulary": VOCABULARY,
                "held_out_pair": HELD_OUT_PAIR,
                "identity_control_pair": IDENTITY_CONTROL_PAIR,
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
    config: LatentIncrementConfig,
    *,
    device: torch.device,
    checkpoint_directory: Path | None,
) -> dict[str, object]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    contract = teacher_contract(device)
    if not contract["passed"]:
        raise RuntimeError("teacher center/identity contract failed")
    schedule = make_training_schedule(config, device)
    split = training_split_audit(schedule)
    if not split["passed"]:
        raise RuntimeError("training split audit failed")
    evaluations = {}
    evaluation_audits = {}
    evaluation_hashes = {}
    for length in config.evaluation_lengths:
        for position in ("early", "late"):
            key = f"{position}_L{length}"
            evaluations[key] = make_relation_batches(
                config, length, position, device
            )
            evaluation_audits[key] = relation_batch_audit(evaluations[key][0])
            if not evaluation_audits[key]["passed"]:
                raise RuntimeError(f"evaluation split audit failed for {key}")
            batch = evaluations[key][0]
            evaluation_hashes[key] = tensor_hash(
                (batch.inputs, batch.targets, batch.post_relation_mask)
            )

    model_shapes = build_models()
    counts = {name: parameter_count(model) for name, model in model_shapes.items()}
    states = {
        name: int(model.recurrent_state_scalars)
        for name, model in model_shapes.items()
    }
    del model_shapes
    results = {}
    factories: dict[str, Callable[[], nn.Module]] = {
        "latent_pure_spin8": LatentPureSpin8Tracker,
        "mamba2_transformers": Mamba2TrialityTracker,
        "gru_reference": GRUTrialityTracker,
        "token_only_ablation": TokenOnlyTrialityAblation,
    }
    for offset, name in enumerate(CANDIDATES):
        seed_everything(700_000 + 1_000 * config.seed + offset)
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
    return {
        "schema_version": 1,
        "experiment": "latent-token Pure Spin8 center-relation identification",
        "status": "development" if config.seed == 0 else "unadjudicated",
        "development_protocol_started_at": PROTOCOL_DEVELOPMENT_STARTED_AT,
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
            "vocabulary": VOCABULARY,
            "teacher_coordinates_sha256": tensor_hash((teacher_coordinates(),)),
            "teacher_initial_state_sha256": tensor_hash((teacher_initial_state(),)),
            "teacher_contract": contract,
            "target": "every prefix in 8v, 8+, and 8-",
            "held_out_relation": "a_half_center * a_half_center = central z",
            "identity_control_relation": "b * b_inverse = identity",
            "training_split": split,
            "evaluation_audits": evaluation_audits,
            "evaluation_schedule_sha256": evaluation_hashes,
        },
        "integrity": {
            "same_precomputed_schedule_for_every_candidate": True,
            "candidate_initialized_after_seed": True,
            "parameter_counts": counts,
            "maximum_relative_parameter_gap": (
                (max(counts.values()) - min(counts.values())) / max(counts.values())
            ),
            "recurrent_state_scalars": states,
            "state_matched": False,
            "all_metrics_finite": all_metrics_finite,
        },
        "results": results,
        "claim_scope": {
            "empirical": [
                "single-seed optimization and forced-relation metrics on the recorded schedule",
                "parameter-near comparison with architecture-specific state sizes",
            ],
            "not_claimed": [
                "a replicated or adjudicated cohort without the separate frozen validator",
                "state- or compute-matched superiority",
                "a fused Mamba comparison",
                "generic language-model quality",
                "a theorem about Mamba-2 or diagonal SSMs",
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
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-length", type=int, default=16)
    parser.add_argument("--evaluation-pairs", type=int, default=64)
    parser.add_argument("--evaluation-lengths", default="16,64,128")
    parser.add_argument("--evaluation-microbatch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
    config = LatentIncrementConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        training_length=args.training_length,
        evaluation_pairs=args.evaluation_pairs,
        evaluation_lengths=parse_lengths(args.evaluation_lengths),
        evaluation_microbatch_size=args.evaluation_microbatch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
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
