"""Controlled learnability cohort for the exceptional Albert ladder.

The task hides signed primitive group actions behind discrete tokens.  A
candidate must learn token-to-Lie-coordinate assignments from transported
Albert probes and then compose them on unseen words and unseen probes.  This
tests representation/controller learnability under oracle event timing.  It is
not a language-model or autonomous-routing claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .action import build_exceptional_action
from .albert import ALBERT_DIM, albert_determinant, build_albert_algebra


LADDER = ("identity", "g2", "spin7", "spin8", "spin9", "f4", "e6")
SAME_RUNG_RELATIVE_ERROR_GATE = 5e-3


@dataclass(frozen=True)
class LearningConfig:
    target_algebra: str = "f4"
    candidate_algebra: str = "f4"
    primitive_count: int = 4
    primitive_scale: float = 0.08
    train_word_length: int = 4
    train_probes: int = 4
    batch_size: int = 16
    steps: int = 1000
    learning_rate: float = 0.01
    seed: int = 20260826


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _bank(algebra: str) -> np.ndarray:
    data = build_albert_algebra()
    if algebra == "identity":
        return np.zeros((0, ALBERT_DIM, ALBERT_DIM), dtype=np.float64)
    return np.asarray(getattr(data, algebra))


def _novel_generator_indices(
    target_algebra: str, primitive_count: int
) -> tuple[list[int], str]:
    """Choose target directions maximally outside the predecessor subalgebra."""

    target_index = LADDER.index(target_algebra)
    if target_index == 0:
        raise ValueError("identity has no learnable primitive generators")
    predecessor = LADDER[target_index - 1]
    target = _bank(target_algebra)
    previous = _bank(predecessor)
    if previous.shape[0] == 0:
        residuals = np.linalg.norm(target.reshape(target.shape[0], -1), axis=1)
    else:
        basis, _ = np.linalg.qr(previous.reshape(previous.shape[0], -1).T)
        flat = target.reshape(target.shape[0], -1)
        residuals = np.linalg.norm(flat - (flat @ basis) @ basis.T, axis=1)
    order = np.argsort(-residuals, kind="stable")
    count = min(primitive_count, len(order))
    if count < 1:
        raise ValueError("primitive_count must select at least one generator")
    return [int(index) for index in order[:count]], predecessor


def _signed_coordinates(
    token_ids: torch.Tensor,
    *,
    generator_count: int,
    generator_indices: list[int],
    scale: float,
) -> torch.Tensor:
    primitive_count = len(generator_indices)
    coordinates = torch.zeros(
        *token_ids.shape,
        generator_count,
        dtype=torch.float64,
        device=token_ids.device,
    )
    primitive = token_ids.remainder(primitive_count)
    signs = torch.where(token_ids < primitive_count, 1.0, -1.0).to(
        dtype=coordinates.dtype
    )
    selected = token_ids.new_tensor(generator_indices)[primitive]
    coordinates.scatter_(-1, selected[..., None], (scale * signs)[..., None])
    return coordinates


def _sample_batch(
    *,
    generator: torch.Generator,
    batch_size: int,
    word_length: int,
    probes: int,
    token_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(
        token_count,
        (batch_size, word_length),
        generator=generator,
        device="cpu",
    ).to(device)
    values = torch.randn(
        batch_size,
        probes,
        ALBERT_DIM,
        generator=generator,
        dtype=torch.float64,
        device="cpu",
    ).to(device)
    return tokens, values


def _transport(values: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    return values @ action.transpose(-1, -2)


class LearnedPrimitiveCoordinates(nn.Module):
    def __init__(self, token_count: int, coordinate_dim: int) -> None:
        super().__init__()
        self.coordinates = nn.Embedding(token_count, coordinate_dim, dtype=torch.float64)
        nn.init.zeros_(self.coordinates.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.coordinates(tokens)


@torch.no_grad()
def _evaluate(
    learner: LearnedPrimitiveCoordinates | None,
    candidate_action: nn.Module,
    target_action: nn.Module,
    *,
    generator_indices: list[int],
    config: LearningConfig,
    word_length: int,
    device: torch.device,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(
        config.seed + 100_000 + 97 * word_length
    )
    tokens, probes = _sample_batch(
        generator=generator,
        batch_size=64,
        word_length=word_length,
        probes=8,
        token_count=2 * len(generator_indices),
        device=device,
    )
    target_coordinates = _signed_coordinates(
        tokens,
        generator_count=target_action.coordinate_dim,
        generator_indices=generator_indices,
        scale=config.primitive_scale,
    )
    expected_action = target_action.ordered(target_coordinates)
    expected = _transport(probes, expected_action)
    if learner is None:
        actual_action = torch.eye(
            ALBERT_DIM, dtype=torch.float64, device=device
        ).expand(tokens.shape[0], ALBERT_DIM, ALBERT_DIM)
    else:
        predicted = learner(tokens)
        actual_action = candidate_action.ordered(predicted)
    actual = _transport(probes, actual_action)
    relative = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(
        expected
    ).clamp_min(1e-12)
    action_relative = torch.linalg.matrix_norm(
        actual_action - expected_action, ord="fro"
    ).mean() / torch.linalg.matrix_norm(expected_action, ord="fro").mean().clamp_min(
        1e-12
    )
    determinant_error = (
        albert_determinant(actual) - albert_determinant(probes)
    ).abs().max()
    return {
        "word_length": word_length,
        "relative_probe_error": float(relative),
        "relative_action_error": float(action_relative),
        "maximum_cubic_error": float(determinant_error),
    }


def run(config: LearningConfig, device: torch.device) -> dict[str, object]:
    if config.target_algebra not in LADDER[1:]:
        raise ValueError("target algebra must be a nonidentity ladder rung")
    if config.candidate_algebra not in LADDER:
        raise ValueError("candidate algebra is not a ladder rung")
    _seed_all(config.seed)
    generator_indices, predecessor = _novel_generator_indices(
        config.target_algebra, config.primitive_count
    )
    target_action = build_exceptional_action(config.target_algebra).to(
        device=device, dtype=torch.float64
    )
    candidate_action = build_exceptional_action(config.candidate_algebra).to(
        device=device, dtype=torch.float64
    )
    token_count = 2 * len(generator_indices)
    learner = None
    optimizer = None
    if candidate_action.coordinate_dim:
        learner = LearnedPrimitiveCoordinates(
            token_count, candidate_action.coordinate_dim
        ).to(device)
        optimizer = torch.optim.Adam(learner.parameters(), lr=config.learning_rate)
    train_generator = torch.Generator().manual_seed(config.seed + 1)
    sampled_losses: dict[str, float] = {}
    started = time.perf_counter()
    maximum_gradient_norm = 0.0
    for step in range(1, config.steps + 1):
        # First identify each signed primitive, then learn its noncommutative
        # placement inside longer products.  This separates coordinate
        # discovery from composition without revealing the coordinates.
        word_length = 1 if step <= config.steps // 2 else config.train_word_length
        tokens, probes = _sample_batch(
            generator=train_generator,
            batch_size=config.batch_size,
            word_length=word_length,
            probes=config.train_probes,
            token_count=token_count,
            device=device,
        )
        with torch.no_grad():
            target_coordinates = _signed_coordinates(
                tokens,
                generator_count=target_action.coordinate_dim,
                generator_indices=generator_indices,
                scale=config.primitive_scale,
            )
            expected_action = target_action.ordered(target_coordinates)
            expected = _transport(probes, expected_action)
        if learner is None or optimizer is None:
            actual = probes
            loss = (actual - expected).square().mean()
        else:
            optimizer.zero_grad(set_to_none=True)
            predicted = learner(tokens)
            actual_action = candidate_action.ordered(predicted)
            actual = _transport(probes, actual_action)
            loss = (actual - expected).square().mean()
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError(f"nonfinite gradient at step {step}")
            maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm))
            optimizer.step()
        if step in {1, max(1, config.steps // 2), config.steps}:
            sampled_losses[str(step)] = float(loss.detach())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    evaluations = [
        _evaluate(
            learner,
            candidate_action,
            target_action,
            generator_indices=generator_indices,
            config=config,
            word_length=length,
            device=device,
        )
        for length in (1, config.train_word_length, 2 * config.train_word_length, 16)
    ]
    longest = evaluations[-1]
    passed = bool(
        config.candidate_algebra == config.target_algebra
        and longest["relative_probe_error"] <= SAME_RUNG_RELATIVE_ERROR_GATE
        and longest["maximum_cubic_error"] <= 1e-8
    )
    return {
        "config": asdict(config),
        "predecessor_algebra": predecessor,
        "target_generator_indices": generator_indices,
        "learner_parameters": (
            sum(parameter.numel() for parameter in learner.parameters())
            if learner is not None
            else 0
        ),
        "sampled_training_losses": sampled_losses,
        "maximum_preclip_gradient_norm": maximum_gradient_norm,
        "training_seconds": elapsed,
        "evaluations": evaluations,
        "same_rung_gate_passed": passed,
        "same_rung_relative_error_gate": SAME_RUNG_RELATIVE_ERROR_GATE,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=LADDER[1:], default="f4")
    parser.add_argument("--candidates", nargs="+", choices=LADDER, default=["spin9", "f4"])
    parser.add_argument("--primitive-count", type=int, default=4)
    parser.add_argument("--primitive-scale", type=float, default=0.08)
    parser.add_argument("--train-word-length", type=int, default=4)
    parser.add_argument("--train-probes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--require-sm75", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    if args.require_sm75 and (
        device.type != "cuda" or torch.cuda.get_device_capability(device) != (7, 5)
    ):
        raise RuntimeError("this cohort requires exact CUDA compute capability 7.5")
    rows = [
        run(
            LearningConfig(
                target_algebra=args.target,
                candidate_algebra=candidate,
                primitive_count=args.primitive_count,
                primitive_scale=args.primitive_scale,
                train_word_length=args.train_word_length,
                train_probes=args.train_probes,
                batch_size=args.batch_size,
                steps=args.steps,
                learning_rate=args.learning_rate,
                seed=args.seed,
            ),
            device,
        )
        for candidate in args.candidates
    ]
    root = Path(__file__).resolve().parent
    report = {
        "schema_version": 1,
        "experiment": "exceptional ladder hidden-coordinate composition learning",
        "status": "oracle-event representation/controller evidence; not language quality",
        "target": args.target,
        "candidates": args.candidates,
        "rows": rows,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "compute_capability": (
                list(torch.cuda.get_device_capability(device))
                if device.type == "cuda"
                else None
            ),
        },
        "source_sha256": {
            name: _file_sha256(root / name)
            for name in (
                "action.py",
                "albert.py",
                "benchmark_exceptional_learning.py",
            )
        },
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
