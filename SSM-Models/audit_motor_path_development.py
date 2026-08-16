"""Numerical and systems audit for dual-quaternion path development.

The audit compares the experimental eight-scalar motor scan with homogeneous
4 by 4 matrix prefix products on identical deterministic ``SE(3)`` increments.
It also inserts the nontrivial central sign in the double cover and verifies
that physical rigid motions remain unchanged while the recurrent motor state
does not.  Timings are local eager-PyTorch diagnostics, not fused-kernel claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from pure_rotor_ssm.motor_scan import (
    motor_from_rotation_translation,
    motor_prefix_scan,
    motor_to_matrix,
)


@dataclass(frozen=True)
class MotorAuditConfig:
    lengths: tuple[int, ...] = (16, 128, 1024, 4096)
    batch_size: int = 4
    lanes: int = 4
    rotation_angle_scale: float = 0.35
    translation_scale: float = 0.05
    seed: int = 20260816
    timing_length: int = 1024
    timing_batch_size: int = 8
    timing_warmups: int = 5
    timing_repeats: int = 20


def _axis_angle_quaternion(raw: torch.Tensor, angle_scale: float) -> torch.Tensor:
    axes = torch.nn.functional.normalize(raw[..., :3], dim=-1)
    angles = angle_scale * torch.tanh(raw[..., 3])
    half = 0.5 * angles
    return torch.cat(
        (torch.cos(half)[..., None], torch.sin(half)[..., None] * axes), -1
    )


def deterministic_motors(
    *,
    batch_size: int,
    length: int,
    lanes: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    angle_scale: float,
    translation_scale: float,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rotation_raw = torch.randn(
        batch_size, length, lanes, 4, generator=generator, dtype=dtype
    )
    translations = translation_scale * torch.randn(
        batch_size, length, lanes, 3, generator=generator, dtype=dtype
    )
    rotations = _axis_angle_quaternion(rotation_raw, angle_scale)
    return motor_from_rotation_translation(
        rotations.to(device), translations.to(device)
    )


def matrix_prefix_scan(
    token_matrices: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive homogeneous-matrix prefix products with matching orientation."""

    if (
        token_matrices.ndim != 5
        or token_matrices.shape[1] == 0
        or token_matrices.shape[-2:] != (4, 4)
    ):
        raise ValueError(
            "token_matrices must have nonempty shape (batch,length,lanes,4,4)"
        )
    batch, length, lanes, _, _ = token_matrices.shape
    if initial_state is None:
        initial_state = torch.eye(
            4, dtype=token_matrices.dtype, device=token_matrices.device
        ).expand(batch, lanes, 4, 4)
    elif initial_state.shape != (batch, lanes, 4, 4):
        raise ValueError("initial_state must have shape (batch,lanes,4,4)")
    if mode == "recurrent":
        state = initial_state
        states = []
        for position in range(length):
            state = state @ token_matrices[:, position]
            states.append(state)
        sequence = torch.stack(states, dim=1)
        return sequence, state
    if mode != "parallel":
        raise ValueError("mode must be parallel or recurrent")
    prefixes = token_matrices
    offset = 1
    while offset < length:
        products = prefixes[:, :-offset] @ prefixes[:, offset:]
        prefixes = torch.cat((prefixes[:, :offset], products), dim=1)
        offset *= 2
    sequence = initial_state[:, None] @ prefixes
    return sequence, sequence[:, -1]


def _tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def audit_length(config: MotorAuditConfig, length: int) -> dict[str, object]:
    motors = deterministic_motors(
        batch_size=config.batch_size,
        length=length,
        lanes=config.lanes,
        dtype=torch.float64,
        device=torch.device("cpu"),
        seed=config.seed + length,
        angle_scale=config.rotation_angle_scale,
        translation_scale=config.translation_scale,
    )
    motor_parallel, motor_final = motor_prefix_scan(motors, mode="parallel")
    motor_recurrent, recurrent_final = motor_prefix_scan(motors, mode="recurrent")
    matrices = motor_to_matrix(motors)
    matrix_parallel, matrix_final = matrix_prefix_scan(matrices, mode="parallel")
    matrix_recurrent, recurrent_matrix_final = matrix_prefix_scan(
        matrices, mode="recurrent"
    )
    motor_matrices = motor_to_matrix(motor_parallel)

    split = max(1, length // 3)
    first, cache = motor_prefix_scan(motors[:, :split], mode="parallel")
    second, cache = motor_prefix_scan(motors[:, split:], cache, mode="parallel")
    chunked = torch.cat((first, second), dim=1)

    negated = motors.clone()
    negated[:, 0] = -negated[:, 0]
    negative_states, _ = motor_prefix_scan(negated, mode="parallel")
    negative_matrices = motor_to_matrix(negative_states)

    real, dual = motor_parallel.split(4, dim=-1)
    rotation = motor_matrices[..., :3, :3]
    identity3 = torch.eye(3, dtype=torch.float64).expand_as(rotation)
    determinants = torch.linalg.det(rotation)
    translations = motor_matrices[..., :3, 3]
    return {
        "length": length,
        "input_sha256": _tensor_sha256(motors),
        "motor_parallel_recurrent_max_abs_error": float(
            (motor_parallel - motor_recurrent).abs().max()
        ),
        "motor_final_parallel_recurrent_max_abs_error": float(
            (motor_final - recurrent_final).abs().max()
        ),
        "motor_chunked_full_max_abs_error": float(
            (chunked - motor_parallel).abs().max()
        ),
        "motor_chunked_final_max_abs_error": float((cache - motor_final).abs().max()),
        "matrix_parallel_recurrent_max_abs_error": float(
            (matrix_parallel - matrix_recurrent).abs().max()
        ),
        "matrix_final_parallel_recurrent_max_abs_error": float(
            (matrix_final - recurrent_matrix_final).abs().max()
        ),
        "motor_matrix_prefix_max_abs_error": float(
            (motor_matrices - matrix_recurrent).abs().max()
        ),
        "motor_matrix_final_max_abs_error": float(
            (motor_to_matrix(motor_final) - recurrent_matrix_final).abs().max()
        ),
        "maximum_rotation_norm_error": float(
            (torch.linalg.vector_norm(real, dim=-1) - 1).abs().max()
        ),
        "maximum_study_condition_error": float((real * dual).sum(dim=-1).abs().max()),
        "maximum_rotation_orthogonality_error": float(
            (rotation.transpose(-1, -2) @ rotation - identity3).abs().max()
        ),
        "maximum_rotation_determinant_error": float((determinants - 1).abs().max()),
        "maximum_absolute_translation": float(translations.abs().max()),
        "central_negation_state_antipode_max_abs_error": float(
            (negative_states + motor_parallel).abs().max()
        ),
        "central_negation_physical_matrix_max_abs_error": float(
            (negative_matrices - motor_matrices).abs().max()
        ),
        "all_outputs_finite": bool(
            torch.isfinite(motor_parallel).all()
            and torch.isfinite(matrix_parallel).all()
        ),
    }


def _time_cuda(operation, warmups: int, repeats: int) -> float:
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        operation()
    torch.cuda.synchronize()
    return 1_000 * (time.perf_counter() - start) / repeats


@torch.no_grad()
def cuda_timing(config: MotorAuditConfig, device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        return {"available": False, "reason": "CUDA timing not requested"}
    motors = deterministic_motors(
        batch_size=config.timing_batch_size,
        length=config.timing_length,
        lanes=config.lanes,
        dtype=torch.float32,
        device=device,
        seed=config.seed + 900_000,
        angle_scale=config.rotation_angle_scale,
        translation_scale=config.translation_scale,
    )
    matrices = motor_to_matrix(motors)
    timings = {
        "motor_parallel_ms": _time_cuda(
            lambda: motor_prefix_scan(motors, mode="parallel"),
            config.timing_warmups,
            config.timing_repeats,
        ),
        "motor_recurrent_ms": _time_cuda(
            lambda: motor_prefix_scan(motors, mode="recurrent"),
            config.timing_warmups,
            config.timing_repeats,
        ),
        "matrix_parallel_ms": _time_cuda(
            lambda: matrix_prefix_scan(matrices, mode="parallel"),
            config.timing_warmups,
            config.timing_repeats,
        ),
        "matrix_recurrent_ms": _time_cuda(
            lambda: matrix_prefix_scan(matrices, mode="recurrent"),
            config.timing_warmups,
            config.timing_repeats,
        ),
    }
    token_count = config.timing_batch_size * config.timing_length * config.lanes
    return {
        "available": True,
        "device": torch.cuda.get_device_name(device),
        "length": config.timing_length,
        "batch_size": config.timing_batch_size,
        "lanes": config.lanes,
        "warmups": config.timing_warmups,
        "repeats": config.timing_repeats,
        **timings,
        "motor_parallel_million_lane_tokens_per_second": (
            token_count / timings["motor_parallel_ms"] / 1_000
        ),
        "matrix_parallel_million_lane_tokens_per_second": (
            token_count / timings["matrix_parallel_ms"] / 1_000
        ),
        "fused_kernel_claimed": False,
    }


def run_audit(config: MotorAuditConfig, device: torch.device) -> dict[str, object]:
    started = datetime.now(ZoneInfo("Africa/Johannesburg"))
    lengths = [audit_length(config, length) for length in config.lengths]
    tolerances = {
        "tree_recurrence": 2e-9,
        "chunk_cache": 2e-9,
        "motor_matrix": 2e-9,
        "unit_and_study": 2e-12,
        "rigid_matrix": 2e-10,
        "central_action": 2e-12,
    }
    checks = {
        "parallel_recurrent": all(
            row["motor_parallel_recurrent_max_abs_error"]
            < tolerances["tree_recurrence"]
            for row in lengths
        ),
        "chunk_cache": all(
            row["motor_chunked_full_max_abs_error"] < tolerances["chunk_cache"]
            and row["motor_chunked_final_max_abs_error"] < tolerances["chunk_cache"]
            for row in lengths
        ),
        "homogeneous_matrix_equivalence": all(
            row["motor_matrix_prefix_max_abs_error"] < tolerances["motor_matrix"]
            for row in lengths
        ),
        "unit_motor_constraints": all(
            row["maximum_rotation_norm_error"] < tolerances["unit_and_study"]
            and row["maximum_study_condition_error"] < tolerances["unit_and_study"]
            for row in lengths
        ),
        "rigid_matrix_constraints": all(
            row["maximum_rotation_orthogonality_error"] < tolerances["rigid_matrix"]
            and row["maximum_rotation_determinant_error"] < tolerances["rigid_matrix"]
            for row in lengths
        ),
        "central_sign_state_and_action": all(
            row["central_negation_state_antipode_max_abs_error"]
            < tolerances["central_action"]
            and row["central_negation_physical_matrix_max_abs_error"]
            < tolerances["central_action"]
            for row in lengths
        ),
        "translation_nontrivial": all(
            row["maximum_absolute_translation"] > config.translation_scale
            for row in lengths
        ),
        "finite": all(row["all_outputs_finite"] for row in lengths),
    }
    timing = cuda_timing(config, device)
    finished = datetime.now(ZoneInfo("Africa/Johannesburg"))
    return {
        "experiment": "dual-quaternion Spin(3) semidirect R3 path-development audit",
        "status": "completed numerical gate" if all(checks.values()) else "failed gate",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "torch_version": torch.__version__,
        "config": asdict(config),
        "conventions": {
            "motor": "q_real + epsilon q_dual, scalar-first quaternions",
            "rigid_action": "x -> R(q_real) x + translation",
            "composition": "left product matches left homogeneous-matrix product",
            "prefix_update": "state_next = state * token_motor",
            "unit_constraints": "norm(q_real)=1 and dot(q_real,q_dual)=0",
            "central_element": "simultaneous negation of real and dual parts",
        },
        "tolerances": tolerances,
        "checks": checks,
        "length_results": lengths,
        "cuda_timing": timing,
        "claim_scope": {
            "established": [
                "numerical equivalence to homogeneous SE(3) prefix composition",
                "center-sensitive state with center-blind physical action",
                "parallel/recurrent and chunk/cache parity within recorded tolerances",
            ],
            "not_established": [
                "learned rigid-motion superiority",
                "bounded translation state",
                "fused-kernel performance",
                "a general conformal or noncompact Lie-group scan theorem",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lengths", default="16,128,1024,4096")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--timing-length", type=int, default=1024)
    parser.add_argument("--timing-batch-size", type=int, default=8)
    parser.add_argument("--timing-warmups", type=int, default=5)
    parser.add_argument("--timing-repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quiet-report", action="store_true")
    args = parser.parse_args()
    lengths = tuple(int(value) for value in args.lengths.split(",") if value)
    positive = (
        *lengths,
        args.batch_size,
        args.lanes,
        args.timing_length,
        args.timing_batch_size,
        args.timing_warmups,
        args.timing_repeats,
    )
    if not lengths or min(positive) < 1:
        raise ValueError(
            "lengths, batches, lanes, warmups, and repeats must be positive"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config = MotorAuditConfig(
        lengths=lengths,
        batch_size=args.batch_size,
        lanes=args.lanes,
        timing_length=args.timing_length,
        timing_batch_size=args.timing_batch_size,
        timing_warmups=args.timing_warmups,
        timing_repeats=args.timing_repeats,
    )
    report = run_audit(config, torch.device(args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not args.quiet_report:
        print(json.dumps(report, indent=2))
    if report["status"] != "completed numerical gate":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
