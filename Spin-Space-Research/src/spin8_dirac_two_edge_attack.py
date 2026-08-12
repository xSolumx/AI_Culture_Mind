"""CUDA falsifier for the preregistered variable-Cayley two-edge family."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from spin8_dirac_one_edge import _projector
from spin8_dirac_two_edge import exact_sign_symmetry_certificate
from spin8_triality import torch_triality_generators


def log_advantage(
    parameters: torch.Tensor, generators: torch.Tensor, fixed: torch.Tensor
) -> torch.Tensor:
    a, d, e, g, residual_i, cayley = parameters.unbind(dim=-1)
    A, D, E, G, I, sine = (
        torch.sqrt(torch.clamp(1 - value.square(), min=1e-30))
        for value in (a, d, e, g, residual_i, cayley)
    )
    zeros = torch.zeros_like(a)
    positive = torch.stack((a, A, zeros, zeros, zeros, zeros, zeros, zeros), -1)
    first_negative = torch.stack(
        (d, D * e, D * E, zeros, zeros, zeros, zeros, zeros), -1
    )
    second_negative = torch.stack(
        (
            g,
            zeros,
            G * residual_i,
            G * I * cayley,
            G * I * sine,
            zeros,
            zeros,
            zeros,
        ),
        -1,
    )
    information = (
        fixed
        + _projector(generators, 1, positive)
        + _projector(generators, 2, first_negative)
        + _projector(generators, 2, second_negative)
    )
    sign, logdet = torch.linalg.slogdet(information)
    delta = A.square() * D.square() * E.square() * G.square() * I.square()
    target = (1 - cayley.square()).pow(3) * (9 - cayley.square()).pow(2)
    result = math.log(1024.0) + logdet - 3 * torch.log(delta) - torch.log(target)
    return torch.where(sign > 0, result, torch.full_like(result, -torch.inf))


def _update_best(values, parameters, best_value, best_parameters):
    value, index = values.max(dim=0)
    if value > best_value:
        return value.detach(), parameters[index].detach().clone()
    return best_value, best_parameters


def numerical_attack(
    *,
    seed: int,
    sample_count: int,
    boundary_samples_per_face: int,
    restarts: int,
    steps: int,
    cap: float,
    device: torch.device,
) -> dict[str, object]:
    dtype = torch.float64
    generators = torch_triality_generators(dtype=dtype, device=device)
    basis_zero = torch.zeros(1, 8, dtype=dtype, device=device)
    basis_zero[0, 0] = 1
    fixed = _projector(generators, 0, basis_zero) + _projector(
        generators, 1, basis_zero
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_value = torch.tensor(-torch.inf, dtype=dtype, device=device)
    best_parameters = torch.zeros(6, dtype=dtype, device=device)
    chunk_size = 4096
    remaining = sample_count
    while remaining:
        count = min(chunk_size, remaining)
        parameters = (
            2 * torch.rand(count, 6, generator=generator, dtype=dtype) - 1
        ) * cap
        parameters = parameters.to(device)
        best_value, best_parameters = _update_best(
            log_advantage(parameters, generators, fixed),
            parameters,
            best_value,
            best_parameters,
        )
        remaining -= count

    boundary_best = torch.tensor(-torch.inf, dtype=dtype, device=device)
    boundary_parameters = best_parameters
    for axis in range(6):
        for sign in (-1, 1):
            parameters = (
                2
                * torch.rand(
                    boundary_samples_per_face,
                    6,
                    generator=generator,
                    dtype=dtype,
                )
                - 1
            ) * cap
            # Log-uniform distance from each true coordinate face.  Staying at
            # least 1e-4 away keeps float64 slogdet meaningful near rank loss.
            exponents = 3 * torch.rand(
                boundary_samples_per_face, generator=generator, dtype=dtype
            )
            parameters[:, axis] = sign * (1 - torch.pow(10.0, -exponents - 1))
            parameters = parameters.to(device)
            boundary_best, boundary_parameters = _update_best(
                log_advantage(parameters, generators, fixed),
                parameters,
                boundary_best,
                boundary_parameters,
            )

    initial = (2 * torch.rand(restarts, 6, generator=generator, dtype=dtype) - 1) * (
        0.8 * cap
    )
    initial[0] = best_parameters.cpu()
    raw = torch.atanh((initial / cap).clamp(-0.999999, 0.999999)).to(device)
    raw.requires_grad_(True)
    optimizer = torch.optim.Adam((raw,), lr=0.025)
    optimized_best = best_value
    optimized_parameters = best_parameters
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        parameters = cap * torch.tanh(raw)
        values = log_advantage(parameters, generators, fixed)
        (-values.mean()).backward()
        optimizer.step()
        optimized_best, optimized_parameters = _update_best(
            values.detach(), parameters.detach(), optimized_best, optimized_parameters
        )

    maximum = torch.maximum(optimized_best, boundary_best)
    return {
        "device": str(device),
        "seed": seed,
        "uniform_sample_count": sample_count,
        "boundary_samples_per_signed_face": boundary_samples_per_face,
        "boundary_sample_count": boundary_samples_per_face * 12,
        "restarts": restarts,
        "optimization_steps": steps,
        "optimization_parameter_cap": cap,
        "random_maximum_log_advantage": float(best_value.cpu()),
        "boundary_maximum_log_advantage": float(boundary_best.cpu()),
        "optimized_maximum_log_advantage": float(optimized_best.cpu()),
        "maximum_log_advantage": float(maximum.cpu()),
        "optimized_parameters": optimized_parameters.cpu().tolist(),
        "boundary_best_parameters": boundary_parameters.cpu().tolist(),
        "violation_above_1e_minus_8": bool(maximum > 1e-8),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--boundary-samples-per-face", type=int, default=4096)
    parser.add_argument("--restarts", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--cap", type=float, default=0.98)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    device = torch.device(
        arguments.device
        if arguments.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    report = {
        "experiment": "variable-Cayley two-edge Dirac--Gram falsifier",
        "exact_sign_symmetry": exact_sign_symmetry_certificate(),
        "numerical_attack": numerical_attack(
            seed=arguments.seed,
            sample_count=arguments.samples,
            boundary_samples_per_face=arguments.boundary_samples_per_face,
            restarts=arguments.restarts,
            steps=arguments.steps,
            cap=arguments.cap,
            device=device,
        ),
    }
    report["passed_falsifier_gate"] = bool(
        report["exact_sign_symmetry"]["passed"]
        and not report["numerical_attack"]["violation_above_1e_minus_8"]
    )
    report["exact_theorem_proved"] = False
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
