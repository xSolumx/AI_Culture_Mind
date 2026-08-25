"""Prospectively frozen G15A-L learned-coordinate cohort."""

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
from spin8_triality import SPIN8_PAIRS, TRIALITY_REPRESENTATIONS
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
    from hybrid_memory_v1_4.optimizers import (  # type: ignore[no-redef]
        ScalarSecondMomentAdamW,
    )


PROTOCOL = Path(__file__).with_name("G15AL_LEARNED_COORDINATE_PROTOCOL_2026-08-25.md")
EXECUTION_AMENDMENT = Path(__file__).with_name(
    "G15AL_EXECUTION_AMENDMENT_2026-08-25.md"
)
CONDITIONAL_ARTIFACT_SHA256 = (
    "78cb06c0e7d088db74651fc93f3c40380f7d3f0f04d447bf54b75ff263c3ffe9"
)
QUALITY_SEEDS = (2153, 2161, 2179)
ARM_NAMES = ("I", "C", "S", "S+identity-read", "S-broken")
ACTION_ANGLE = 0.12
MAXIMUM_COORDINATE = 0.25
ACTION_VOCABULARY = 16
VOCABULARY_SIZE = 17
VECTOR_INDEX = TRIALITY_REPRESENTATIONS.index("vector")


@dataclass(frozen=True)
class LearnedCoordinateConfig:
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


def quality_config() -> LearnedCoordinateConfig:
    return LearnedCoordinateConfig(
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


def smoke_config() -> LearnedCoordinateConfig:
    return LearnedCoordinateConfig(
        mode="smoke",
        seeds=(17,),
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
class CoordinateBatch:
    token_ids: torch.Tensor
    exact_coordinates: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    action_positions: torch.Tensor

    def to(
        self, device: torch.device, dtype: torch.dtype | None = None
    ) -> CoordinateBatch:
        dtype = dtype or self.exact_coordinates.dtype
        return CoordinateBatch(
            token_ids=self.token_ids.to(device),
            exact_coordinates=self.exact_coordinates.to(device=device, dtype=dtype),
            keys=self.keys.to(device=device, dtype=dtype),
            values=self.values.to(device=device, dtype=dtype),
            action_positions=self.action_positions.to(device),
        )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for tensor in (
            self.token_ids,
            self.exact_coordinates,
            self.keys,
            self.values,
            self.action_positions,
        ):
            value = tensor.detach().cpu().contiguous()
            digest.update(str(value.dtype).encode())
            digest.update(json.dumps(tuple(value.shape)).encode())
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()


class TokenCoordinateController(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.raw_coordinates = nn.Parameter(
            torch.zeros(VOCABULARY_SIZE, len(SPIN8_PAIRS))
        )

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
    generator = torch.Generator().manual_seed(_stable_seed("g15al-token-map", seed))
    return torch.randperm(ACTION_VOCABULARY, generator=generator) + 1


def generate_batch(
    batch_size: int,
    length: int,
    *,
    seed: int,
    model_seed: int,
    minimum_actions: int,
    maximum_actions: int,
) -> CoordinateBatch:
    if length < maximum_actions + 2:
        raise ValueError("length must leave room for write, actions, and final query")
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.zeros(batch_size, length, dtype=torch.long)
    coordinates = torch.zeros(batch_size, length, 1, len(SPIN8_PAIRS))
    keys = F.normalize(torch.randn(batch_size, 8, generator=generator), dim=-1)
    values = F.normalize(torch.randn(batch_size, 8, generator=generator), dim=-1)
    positions = torch.full((batch_size, maximum_actions), -1, dtype=torch.long)
    token_map = _token_map(model_seed)
    for row in range(batch_size):
        count = int(
            torch.randint(
                minimum_actions,
                maximum_actions + 1,
                (1,),
                generator=generator,
            )
        )
        selected_positions = (
            (torch.randperm(length - 2, generator=generator)[:count] + 1).sort().values
        )
        semantic_actions = torch.randint(
            0, ACTION_VOCABULARY, (count,), generator=generator
        )
        positions[row, :count] = selected_positions
        for position, semantic in zip(
            selected_positions.tolist(), semantic_actions.tolist(), strict=True
        ):
            pair_index = semantic % len(OFF_TORUS_PAIRS)
            sign = 1.0 if semantic < len(OFF_TORUS_PAIRS) else -1.0
            tokens[row, position] = token_map[semantic]
            coordinate_index = SPIN8_PAIRS.index(OFF_TORUS_PAIRS[pair_index])
            coordinates[row, position, 0, coordinate_index] = sign * ACTION_ANGLE
    return CoordinateBatch(tokens, coordinates, keys, values, positions)


def _controls(
    batch: CoordinateBatch,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    batch_size, length = batch.token_ids.shape
    carrier = torch.zeros(
        batch_size, length, 1, 8, device=device, dtype=batch.keys.dtype
    )
    query = carrier.clone()
    key = carrier.clone()
    value = carrier.clone()
    key[:, 0, 0] = batch.keys
    value[:, 0, 0] = batch.values
    erase = torch.zeros(batch_size, length, 1, 1, device=device, dtype=batch.keys.dtype)
    write = torch.zeros_like(erase)
    write[:, 0] = 1.0
    retention = torch.full_like(erase, 0.999999)
    return query, key, value, erase, write, retention


def _carrier_totals(
    memory: nn.Module,
    batch: CoordinateBatch,
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
        valid[..., None, None],
        event_coordinates,
        torch.zeros_like(event_coordinates),
    )
    event_coordinates = memory._apply_transport_mode(event_coordinates)
    carrier = torch.zeros(
        batch_size,
        maximum_actions,
        1,
        8,
        device=device,
        dtype=coordinates.dtype,
    )
    gate = torch.zeros(
        batch_size,
        maximum_actions,
        1,
        1,
        device=device,
        dtype=coordinates.dtype,
    )
    retention = torch.ones_like(gate)
    actions = memory._transitions(
        carrier,
        carrier,
        gate,
        gate,
        retention,
        event_coordinates,
        None,
    )[3]
    eye = torch.eye(8, device=device, dtype=coordinates.dtype).expand(batch_size, 8, 8)
    total_vector = eye.clone()
    total_positive = eye.clone()
    positive_index = TRIALITY_REPRESENTATIONS.index("positive")
    for column in range(maximum_actions):
        vector = actions[:, column, 0, VECTOR_INDEX]
        positive = actions[:, column, 0, positive_index]
        column_valid = valid[:, column, None, None]
        vector = torch.where(column_valid, vector, eye)
        positive = torch.where(column_valid, positive, eye)
        total_vector = vector @ total_vector
        total_positive = positive @ total_positive
    return total_vector, total_positive


def _event_sparse_prediction(
    memory: nn.Module,
    batch: CoordinateBatch,
    coordinates: torch.Tensor,
    exact_query: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    vector, positive = _carrier_totals(memory, batch, coordinates, device=device)
    transported_key = torch.einsum("bij,bj->bi", vector, batch.keys)
    alignment = (exact_query * transported_key).sum(dim=-1, keepdim=True)
    transported_value = torch.einsum("bij,bj->bi", positive, batch.values)
    retention = 0.999999 ** (batch.token_ids.shape[1] - 1)
    return retention * alignment * transported_value


@torch.no_grad()
def _teacher_target(
    teacher: nn.Module,
    batch: CoordinateBatch,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    vector, positive = _carrier_totals(
        teacher, batch, batch.exact_coordinates, device=device
    )
    exact_query = torch.einsum("bij,bj->bi", vector, batch.keys)
    retention = 0.999999 ** (batch.token_ids.shape[1] - 1)
    alignment = (exact_query * exact_query).sum(dim=-1, keepdim=True)
    target = retention * alignment * torch.einsum("bij,bj->bi", positive, batch.values)
    return exact_query, target


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    normalized_prediction = F.normalize(prediction, dim=-1)
    normalized_target = F.normalize(target, dim=-1)
    cosine = (normalized_prediction * normalized_target).sum(dim=-1)
    return {
        "mean_cosine": float(cosine.mean()),
        "minimum_cosine": float(cosine.min()),
        "normalized_mse": float(F.mse_loss(normalized_prediction, normalized_target)),
    }


def _train_arm(
    arm: str,
    config: LearnedCoordinateConfig,
    *,
    seed: int,
    device: torch.device,
    checkpoint_directory: Path,
) -> dict[str, Any]:
    _seed_everything(_stable_seed("g15al-controller", seed))
    controller = TokenCoordinateController().to(device)
    memory = _oracle_memory(arm, dtype=torch.float32, device=device)
    teacher = _oracle_memory("S", dtype=torch.float32, device=device)
    optimizer = ScalarSecondMomentAdamW(
        controller.parameters(), lr=config.learning_rate, weight_decay=0.0
    )
    initial_hash = hashlib.sha256(
        controller.raw_coordinates.detach().cpu().numpy().tobytes()
    ).hexdigest()
    schedule = hashlib.sha256()
    loss_samples: dict[str, dict[str, float]] = {}
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(config.training_updates):
        batch = generate_batch(
            config.training_batch_size,
            config.training_length,
            seed=_stable_seed("g15al-train", seed, step),
            model_seed=seed,
            minimum_actions=config.minimum_training_actions,
            maximum_actions=config.maximum_training_actions,
        )
        schedule.update(batch.fingerprint().encode())
        batch = batch.to(device)
        exact_query, target = _teacher_target(teacher, batch, device=device)
        optimizer.zero_grad(set_to_none=True)
        learned_coordinates = controller(batch.token_ids)
        prediction = _event_sparse_prediction(
            memory,
            batch,
            learned_coordinates,
            exact_query,
            device=device,
        )
        cosine = (F.normalize(prediction, dim=-1) * F.normalize(target, dim=-1)).sum(
            dim=-1
        )
        loss = (1.0 - cosine).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite G15A-L loss for {arm} seed {seed}")
        if loss.requires_grad:
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        else:
            gradient_norm = loss.new_zeros(())
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("non-finite G15A-L gradient norm")
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
    evaluations = {}
    controller.eval()
    for length, actions_per_episode in config.evaluation_specs:
        metric_rows = []
        for offset in range(
            0, config.evaluation_examples, config.evaluation_microbatch
        ):
            size = min(
                config.evaluation_microbatch, config.evaluation_examples - offset
            )
            batch = generate_batch(
                size,
                length,
                seed=_stable_seed("g15al-eval", seed, length, offset),
                model_seed=seed,
                minimum_actions=actions_per_episode,
                maximum_actions=actions_per_episode,
            ).to(device)
            exact_query, target = _teacher_target(teacher, batch, device=device)
            with torch.no_grad():
                prediction = _event_sparse_prediction(
                    memory,
                    batch,
                    controller(batch.token_ids),
                    exact_query,
                    device=device,
                )
            metric_rows.append(_metrics(prediction, target))
        evaluations[str(length)] = {
            "mean_cosine": sum(row["mean_cosine"] for row in metric_rows)
            / len(metric_rows),
            "minimum_cosine": min(row["minimum_cosine"] for row in metric_rows),
            "normalized_mse": sum(row["normalized_mse"] for row in metric_rows)
            / len(metric_rows),
            "actions_per_episode": actions_per_episode,
            "examples": config.evaluation_examples,
        }
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        checkpoint_directory / f"g15al_{arm.replace('+', 'plus')}_seed{seed}.pt"
    )
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
        "training_schedule_sha256": schedule.hexdigest(),
        "loss_samples": loss_samples,
        "training_wall_seconds": training_seconds,
        "mean_synchronized_step_seconds": training_seconds / config.training_updates,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "evaluation": evaluations,
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
            identity_read = arms["S+identity-read"]["evaluation"][length]
            comparator_margins = {
                arm: spin["mean_cosine"]
                - arms[arm]["evaluation"][length]["mean_cosine"]
                for arm in ("I", "C", "S-broken")
            }
            checks = {
                "spin_mean_cosine_at_least_0_995": spin["mean_cosine"] + 1e-12 >= 0.995,
                "spin_minimum_cosine_at_least_0_98": spin["minimum_cosine"] + 1e-12
                >= 0.98,
                "identity_read_mean_parity_within_1e_5": abs(
                    spin["mean_cosine"] - identity_read["mean_cosine"]
                )
                <= 1e-5,
                "each_comparator_margin_at_least_0_05": all(
                    margin + 1e-12 >= 0.05 for margin in comparator_margins.values()
                ),
            }
            per_length[length] = {
                "spin": spin,
                "identity_read": identity_read,
                "comparator_mean_cosine_margins": comparator_margins,
                "checks": checks,
                "passed": all(checks.values()),
            }
        coordinate_parity = max(
            abs(left - right)
            for left_row, right_row in zip(
                arms["S"]["learned_raw_coordinates"],
                arms["S+identity-read"]["learned_raw_coordinates"],
                strict=True,
            )
            for left, right in zip(left_row, right_row, strict=True)
        )
        exact_pairing = (
            len({arms[arm]["trainable_parameters"] for arm in ARM_NAMES}) == 1
            and len({arms[arm]["initial_state_sha256"] for arm in ARM_NAMES}) == 1
            and len({arms[arm]["training_schedule_sha256"] for arm in ARM_NAMES}) == 1
        )
        checks = {
            "all_lengths_pass": all(row["passed"] for row in per_length.values()),
            "spin_identity_read_coordinate_parity_within_1e_5": coordinate_parity
            <= 1e-5,
            "exact_parameter_initialization_schedule_pairing": exact_pairing,
        }
        per_seed.append(
            {
                "seed": report["seed"],
                "per_length": per_length,
                "spin_identity_read_coordinate_max_abs_difference": coordinate_parity,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    passed = all(row["passed"] for row in per_seed)
    return {
        "passed": passed,
        "per_seed": per_seed,
        "decision": (
            "G15A-L passes: learned coordinates preserve the shared-lift separation"
            if passed
            else "G15A-L fails: learned-coordinate attribution is not established"
        ),
    }


def _load_conditional_artifact(path: Path) -> dict[str, Any]:
    if _sha256(path) != CONDITIONAL_ARTIFACT_SHA256:
        raise RuntimeError("conditional artifact does not match the frozen hash")
    report = json.loads(path.read_text(encoding="utf-8"))
    adjudication = report.get("adjudication", {})
    if (
        report.get("evidentiary") is not True
        or adjudication.get("integrity_passed") is not True
        or adjudication.get("shared_triality_coupling_contribution_supported")
        is not True
        or adjudication.get("clifford_read_contribution_supported") is not False
    ):
        raise RuntimeError("conditional artifact does not have the required outcome")
    return report


def run(
    config: LearnedCoordinateConfig,
    *,
    device: torch.device,
    checkpoint_directory: Path,
    commit: str,
    status_at_start: list[str],
    conditional_artifact: dict[str, Any],
) -> dict[str, Any]:
    started_at = _now()
    started = time.perf_counter()
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
        Path(__file__).with_name("g15a_spin_dirac_cohort.py"),
        Path(__file__).with_name("spin_dirac_memory.py"),
        Path(__file__).with_name("optimizers.py"),
        PROTOCOL,
    )
    return {
        "schema_version": 1,
        "experiment": "G15A-L learned token-to-Spin-coordinate cohort",
        "claim_status": (
            "learned coordinate mechanism under oracle edit controls and oracle "
            "transported final query"
        ),
        "mode": config.mode,
        "evidentiary": config.mode == "quality" and not status_at_start,
        "started_at": started_at,
        "finished_at": _now(),
        "elapsed_wall_seconds": time.perf_counter() - started,
        "git_commit_at_start": commit,
        "git_status_at_start": status_at_start,
        "conditional_artifact_sha256": CONDITIONAL_ARTIFACT_SHA256,
        "conditional_artifact_commit": conditional_artifact["git_commit_at_start"],
        "protocol": asdict(config),
        "arm_names": list(ARM_NAMES),
        "protocol_file_sha256": _sha256(PROTOCOL),
        "protocol_files": {
            PROTOCOL.name: _sha256(PROTOCOL),
            EXECUTION_AMENDMENT.name: _sha256(EXECUTION_AMENDMENT),
        },
        "execution_path": "exact_event_sparse_affine_recurrence",
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
            "the final transported query and memory edit controls are oracle supplied",
            "only the token-to-coordinate lookup is learned on the symmetry task",
            "no generic association, natural-text, scaling, or fused claim follows",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "quality"), required=True)
    parser.add_argument("--conditional-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    for path in (PROTOCOL, EXECUTION_AMENDMENT):
        if not path.is_file():
            raise FileNotFoundError(path)
    conditional_artifact = _load_conditional_artifact(args.conditional_artifact)
    config = quality_config() if args.mode == "quality" else smoke_config()
    commit, status_at_start = _git_state()
    if args.mode == "quality" and status_at_start:
        raise RuntimeError("evidentiary G15A-L requires a clean committed worktree")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if torch.cuda.get_device_capability(device) != (7, 5):
            raise RuntimeError("the frozen local G15A-L cohort requires exact SM75")
    report = run(
        config,
        device=device,
        checkpoint_directory=args.checkpoint_directory,
        commit=commit,
        status_at_start=status_at_start,
        conditional_artifact=conditional_artifact,
    )
    _atomic_json(args.output, report)
    print(args.output)
    print(json.dumps(report["adjudication"], indent=2, sort_keys=True))
    if args.mode == "quality" and not report["adjudication"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
