"""Invariant-space and rank-deficient identifiability gate for Spin(8) triality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from spin8_masked_completion import (
    LearnedBilinearCompletion,
    MLPCompletion,
    evaluate_family,
    fixed_generator_actions,
)
from spin8_triality import build_spin8_triality_algebra
from spin8_triality_lift import triality_bind, triality_tensor, triality_unbind_negative


class InvariantTrialityCompletion(nn.Module):
    def __init__(self, rho: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("rho", rho.detach().clone())
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, positive: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        return self.scale * triality_unbind_negative(positive, vector, self.rho)


class ExactTrialityCompletion(nn.Module):
    def __init__(self, rho: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("rho", rho.detach().clone())

    def forward(self, positive: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        return triality_unbind_negative(positive, vector, self.rho)


def invariant_space_audit() -> dict[str, object]:
    """Compute the Lie-algebra intertwiner nullspace in all 512 coordinates."""

    algebra = build_spin8_triality_algebra()
    vector = torch.as_tensor(algebra.vector_generators, dtype=torch.float64)
    positive = torch.as_tensor(algebra.positive_generators, dtype=torch.float64)
    negative = torch.as_tensor(algebra.negative_generators, dtype=torch.float64)
    basis = torch.eye(512, dtype=torch.float64).reshape(512, 8, 8, 8)
    blocks = []
    for vector_generator, positive_generator, negative_generator in zip(
        vector, positive, negative
    ):
        # Residual indices are (output v, negative l, positive k).
        output_term = torch.einsum("vw,bwlk->bvlk", vector_generator, basis)
        negative_term = torch.einsum("jl,bvjk->bvlk", negative_generator, basis)
        positive_term = torch.einsum("bvli,ik->bvlk", basis, positive_generator)
        residual = output_term - negative_term - positive_term
        blocks.append(residual.reshape(512, 512).T)
    constraints = torch.cat(blocks, dim=0)
    # The matrix is tall (14336 x 512).  Its Gram matrix has the same
    # right-nullspace and avoids materialising the large left singular basis.
    gram = constraints.T @ constraints
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    singular_values = eigenvalues.clamp_min(0.0).sqrt()
    tolerance = 1e-10 * singular_values[-1]
    nullity = int((singular_values < tolerance).sum())
    candidate = eigenvectors[:, 0]
    exact = triality_tensor(dtype=torch.float64).flatten()
    cosine = float(
        (F.normalize(candidate, dim=0) * F.normalize(exact, dim=0)).sum().abs()
    )
    return {
        "constraint_shape": list(constraints.shape),
        "nullity": nullity,
        "largest_singular_value": float(singular_values[-1]),
        "second_smallest_singular_value": float(singular_values[1]),
        "smallest_singular_value": float(singular_values[0]),
        "null_vector_abs_cosine_with_triality": cosine,
    }


def rank_deficient_batch(
    actions: torch.Tensor,
    *,
    batch_size: int,
    generator: torch.Generator,
    rho: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device, dtype = actions.device, actions.dtype
    positive = torch.zeros(batch_size, 8, device=device, dtype=dtype)
    positive[:, 0] = 1.0
    negative_index = torch.randint(
        0, 4, (batch_size,), generator=generator, device=device
    )
    negative = F.one_hot(negative_index, num_classes=8).to(dtype=dtype)
    vector = triality_bind(positive, negative, rho)
    lengths = torch.randint(1, 5, (batch_size,), generator=generator, device=device)
    action = actions[0]
    for position in range(4):
        active = (position < lengths)[:, None]
        next_vector = torch.einsum("ij,bj->bi", action[0], vector)
        next_positive = torch.einsum("ij,bj->bi", action[1], positive)
        next_negative = torch.einsum("ij,bj->bi", action[2], negative)
        vector = torch.where(active, next_vector, vector)
        positive = torch.where(active, next_positive, positive)
        negative = torch.where(active, next_negative, negative)
    return positive, vector, negative


def design_rank(actions: torch.Tensor, rho: torch.Tensor) -> dict[str, float | int]:
    generator = torch.Generator(device=actions.device).manual_seed(20260804)
    positive, vector, _ = rank_deficient_batch(
        actions, batch_size=4096, generator=generator, rho=rho
    )
    features = torch.einsum("bi,bv->bvi", positive, vector).flatten(1).double()
    singular_values = torch.linalg.svdvals(features)
    tolerance = 1e-10 * singular_values[0]
    nonzero = singular_values[singular_values > tolerance]
    return {
        "rank": int(nonzero.numel()),
        "ambient_dimension": 64,
        "nonzero_condition_number": float(nonzero[0] / nonzero[-1]),
    }


def train_model(
    family: str,
    *,
    seed: int,
    actions: torch.Tensor,
    rho: torch.Tensor,
    steps: int,
    batch_size: int,
) -> tuple[nn.Module, dict[str, object]]:
    torch.manual_seed(seed)
    if family == "invariant":
        model: nn.Module = InvariantTrialityCompletion(rho)
    elif family == "bilinear":
        model = LearnedBilinearCompletion()
    elif family == "mlp":
        model = MLPCompletion()
    else:
        raise ValueError(f"unknown family {family!r}")
    model = model.to(actions.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    generator = torch.Generator(device=actions.device).manual_seed(30000 + seed)
    trajectory = []
    for step in range(steps):
        positive, vector, target = rank_deficient_batch(
            actions, batch_size=batch_size, generator=generator, rho=rho
        )
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(positive, vector), target)
        loss.backward()
        optimizer.step()
        if step in (0, 49, 199, 499, steps - 1):
            trajectory.append({"step": step + 1, "loss": float(loss.detach())})
    result: dict[str, object] = {
        "family": family,
        "seed": seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trajectory": trajectory,
        "final_training_mse": trajectory[-1]["loss"],
    }
    if family == "invariant":
        result["learned_scale"] = float(model.scale.detach())
    elif family == "bilinear":
        result["tensor_cosine_with_triality"] = float(
            (
                F.normalize(model.tensor.detach().flatten(), dim=0)
                * F.normalize(rho.flatten(), dim=0)
            ).sum()
        )
    return model, result


def run(
    *,
    device: torch.device,
    steps: int,
    batch_size: int,
    seeds: tuple[int, ...],
) -> dict[str, object]:
    actions = fixed_generator_actions(device, torch.float32)
    rho = triality_tensor(dtype=torch.float32, device=device)
    lengths = (8, 32, 128, 512)
    splits = {"held_out_generator": (2,), "mixed_generators": (0, 1, 2)}
    exact = ExactTrialityCompletion(rho)
    results = [
        {
            "family": "exact",
            "parameters": 0,
            "evaluation": {
                split: {
                    str(length): evaluate_family(
                        exact,
                        actions,
                        rho,
                        length=length,
                        allowed_generators=allowed,
                        seed=39000 + length,
                    )
                    for length in lengths
                }
                for split, allowed in splits.items()
            },
        }
    ]
    for family in ("invariant", "bilinear", "mlp"):
        for seed in seeds:
            model, result = train_model(
                family,
                seed=seed,
                actions=actions,
                rho=rho,
                steps=steps,
                batch_size=batch_size,
            )
            result["evaluation"] = {
                split: {
                    str(length): evaluate_family(
                        model,
                        actions,
                        rho,
                        length=length,
                        allowed_generators=allowed,
                        seed=40000 + 1000 * seed + length,
                    )
                    for length in lengths
                }
                for split, allowed in splits.items()
            }
            results.append(result)
    return {
        "experiment": "rank-deficient Spin(8) triality identifiability",
        "device": str(device),
        "steps": steps,
        "batch_size": batch_size,
        "seeds": list(seeds),
        "invariant_space": invariant_space_audit(),
        "training_design": design_rank(actions, rho),
        "learned": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        device=torch.device(args.device),
        steps=args.steps,
        batch_size=args.batch_size,
        seeds=tuple(int(value) for value in args.seeds.split(",") if value),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
