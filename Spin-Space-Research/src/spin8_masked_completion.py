"""Masked Spin(8) cross-representation completion mechanism gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from spin8_triality import SPIN8_BIVECTOR_DIM, spin8_actions, torch_triality_generators
from spin8_triality_lift import triality_bind, triality_tensor, triality_unbind_negative


class LearnedBilinearCompletion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tensor = nn.Parameter(torch.empty(8, 8, 8))
        nn.init.normal_(self.tensor, std=0.05)

    def forward(self, positive: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...i,vji,...v->...j", positive, self.tensor, vector)


class MLPCompletion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(16, 24), nn.GELU(), nn.Linear(24, 8))

    def forward(self, positive: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((positive, vector), dim=-1))


class LinearCompletion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(16, 8)

    def forward(self, positive: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat((positive, vector), dim=-1))


def fixed_generator_actions(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(20260803)
    coefficients = 0.45 * torch.randn(
        3, SPIN8_BIVECTOR_DIM, generator=generator, device=device, dtype=dtype
    )
    return spin8_actions(
        coefficients, torch_triality_generators(dtype=dtype, device=device)
    )


def sample_transported_batch(
    actions: torch.Tensor,
    *,
    batch_size: int,
    min_length: int,
    max_length: int,
    allowed_generators: tuple[int, ...],
    generator: torch.Generator,
    rho: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device, dtype = actions.device, actions.dtype
    positive_center = torch.zeros(batch_size, 8, device=device, dtype=dtype)
    positive_center[:, 0] = 1.0
    positive = F.normalize(
        positive_center
        + 0.08
        * torch.randn(
            batch_size, 8, generator=generator, device=device, dtype=dtype
        ),
        dim=-1,
    )
    negative_indices = torch.randint(
        0, 4, (batch_size,), generator=generator, device=device
    )
    negative_center = F.one_hot(negative_indices, num_classes=8).to(dtype=dtype)
    negative = F.normalize(
        negative_center
        + 0.08
        * torch.randn(
            batch_size, 8, generator=generator, device=device, dtype=dtype
        ),
        dim=-1,
    )
    vector = triality_bind(positive, negative, rho)
    lengths = torch.randint(
        min_length,
        max_length + 1,
        (batch_size,),
        generator=generator,
        device=device,
    )
    choices = torch.as_tensor(allowed_generators, device=device)
    token_choices = torch.randint(
        0,
        len(allowed_generators),
        (batch_size, max_length),
        generator=generator,
        device=device,
    )
    tokens = choices[token_choices]
    for position in range(max_length):
        active = position < lengths
        if not bool(active.any()):
            break
        step = actions[tokens[:, position]]
        next_vector = torch.einsum("bij,bj->bi", step[:, 0], vector)
        next_positive = torch.einsum("bij,bj->bi", step[:, 1], positive)
        next_negative = torch.einsum("bij,bj->bi", step[:, 2], negative)
        active_column = active[:, None]
        vector = torch.where(active_column, next_vector, vector)
        positive = torch.where(active_column, next_positive, positive)
        negative = torch.where(active_column, next_negative, negative)
    return positive, vector, negative


@torch.no_grad()
def evaluate_family(
    model: nn.Module | None,
    actions: torch.Tensor,
    rho: torch.Tensor,
    *,
    length: int,
    allowed_generators: tuple[int, ...],
    seed: int,
    examples: int = 1024,
) -> dict[str, float]:
    generator = torch.Generator(device=actions.device).manual_seed(seed)
    positive, vector, target = sample_transported_batch(
        actions,
        batch_size=examples,
        min_length=length,
        max_length=length,
        allowed_generators=allowed_generators,
        generator=generator,
        rho=rho,
    )
    prediction = (
        triality_unbind_negative(positive, vector, rho)
        if model is None
        else model(positive, vector)
    )
    return {
        "mse": float(F.mse_loss(prediction, target)),
        "mean_cosine": float(F.cosine_similarity(prediction, target, dim=-1).mean()),
        "minimum_cosine": float(F.cosine_similarity(prediction, target, dim=-1).min()),
        "maximum_norm_error": float((prediction.norm(dim=-1) - 1.0).abs().max()),
    }


def train_family(
    family: str,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, object]]:
    torch.manual_seed(seed)
    actions = fixed_generator_actions(device, torch.float32)
    rho = triality_tensor(dtype=torch.float32, device=device)
    model: nn.Module
    if family == "bilinear":
        model = LearnedBilinearCompletion()
    elif family == "mlp":
        model = MLPCompletion()
    elif family == "linear":
        model = LinearCompletion()
    else:
        raise ValueError(f"unknown family {family!r}")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    data_generator = torch.Generator(device=device).manual_seed(10000 + seed)
    trajectory = []
    for step in range(steps):
        positive, vector, target = sample_transported_batch(
            actions,
            batch_size=batch_size,
            min_length=1,
            max_length=8,
            allowed_generators=(0, 1),
            generator=data_generator,
            rho=rho,
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = model(positive, vector)
        loss = F.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()
        if step in (0, 49, 199, 499, steps - 1):
            trajectory.append({"step": step + 1, "loss": float(loss.detach())})
    result: dict[str, object] = {
        "family": family,
        "seed": seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trajectory": trajectory,
    }
    if family == "bilinear":
        learned = F.normalize(model.tensor.detach().flatten(), dim=0)
        exact = F.normalize(rho.flatten(), dim=0)
        result["tensor_cosine_with_triality"] = float((learned * exact).sum())
        result["tensor_relative_error"] = float((model.tensor.detach() - rho).norm() / rho.norm())
    return model, result


def run_experiment(
    *,
    device: torch.device,
    steps: int,
    batch_size: int,
    seeds: tuple[int, ...],
    lengths: tuple[int, ...] = (8, 32, 128, 512),
    evaluation_examples: int = 1024,
) -> dict[str, object]:
    actions = fixed_generator_actions(device, torch.float32)
    rho = triality_tensor(dtype=torch.float32, device=device)
    splits = {"held_out_generator": (2,), "mixed_generators": (0, 1, 2)}
    oracle_actions = fixed_generator_actions(torch.device("cpu"), torch.float64)
    oracle_rho = triality_tensor(dtype=torch.float64, device="cpu")
    oracle = {
        split: {
            str(length): evaluate_family(
                None,
                oracle_actions,
                oracle_rho,
                length=length,
                allowed_generators=allowed,
                seed=9000 + length,
                examples=evaluation_examples,
            )
            for length in lengths
        }
        for split, allowed in splits.items()
    }
    learned_results = []
    for family in ("bilinear", "mlp", "linear"):
        for seed in seeds:
            model, result = train_family(
                family,
                seed=seed,
                steps=steps,
                batch_size=batch_size,
                device=device,
            )
            result["evaluation"] = {
                split: {
                    str(length): evaluate_family(
                        model,
                        actions,
                        rho,
                        length=length,
                        allowed_generators=allowed,
                        seed=20000 + 1000 * seed + length,
                        examples=evaluation_examples,
                    )
                    for length in lengths
                }
                for split, allowed in splits.items()
            }
            learned_results.append(result)
    return {
        "experiment": "masked Spin(8) cross-representation completion",
        "device": str(device),
        "steps": steps,
        "batch_size": batch_size,
        "evaluation_examples": evaluation_examples,
        "evaluation_lengths": list(lengths),
        "initial_distribution": (
            "continuous 0.08-radius caps around positive e0 and negative e0..e3"
        ),
        "seeds": list(seeds),
        "oracle": oracle,
        "learned": learned_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--lengths", default="8,32,128,512")
    parser.add_argument("--evaluation-examples", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    lengths = tuple(int(value) for value in args.lengths.split(",") if value)
    report = run_experiment(
        device=torch.device(args.device),
        steps=args.steps,
        batch_size=args.batch_size,
        seeds=seeds,
        lengths=lengths,
        evaluation_examples=args.evaluation_examples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
