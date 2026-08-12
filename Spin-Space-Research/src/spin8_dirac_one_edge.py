"""Variable-Cayley one-edge Dirac--Gram falsifier and orientation audit."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import sympy as sp
import torch

from spin8_cayley_spectrum import (
    symbolic_query_projector,
    symbolic_triality_generators,
)
from spin8_dirac_edge import _character
from spin8_dirac_star import rational_circle
from spin8_triality import torch_triality_generators

SIGN_CHARACTERS = {
    "trivial": (0, 0, 0, 0, 0),
    "egc": (0, 0, 1, 1, 1),
    "adgc": (1, 1, 0, 1, 1),
    "ade": (1, 1, 1, 0, 0),
}
SIGNS = tuple(itertools.product((1, -1), repeat=5))


def _symbolic_vector(
    coefficients: tuple[sp.Expr, ...],
    indices: tuple[int, ...],
    basis: list[list[sp.Integer]],
) -> list[sp.Expr]:
    return [
        sum(
            coefficient * basis[index][column]
            for coefficient, index in zip(coefficients, indices, strict=True)
        )
        for column in range(8)
    ]


def exact_walsh_support_certificate() -> dict[str, object]:
    generators = symbolic_triality_generators()
    basis = [[sp.Integer(row == column) for column in range(8)] for row in range(8)]
    fixed = symbolic_query_projector(
        0, basis[0], generators
    ) + symbolic_query_projector(1, basis[0], generators)
    pairs = tuple(
        rational_circle(value)
        for value in (
            sp.Rational(1, 3),
            sp.Rational(2, 5),
            sp.Rational(1, 4),
            sp.Rational(3, 7),
            sp.Rational(2, 7),
        )
    )
    (
        (a, diagonal_a),
        (d, diagonal_d),
        (e, diagonal_e),
        (
            g,
            diagonal_g,
        ),
        (cayley, sine),
    ) = pairs
    delta = diagonal_a**2 * diagonal_d**2 * diagonal_e**2 * diagonal_g**2
    determinants = {}
    for signs in SIGNS:
        sign_a, sign_d, sign_e, sign_g, sign_c = signs
        positive = _symbolic_vector((sign_a * a, diagonal_a), (0, 1), basis)
        first_negative = _symbolic_vector(
            (sign_d * d, diagonal_d * sign_e * e, diagonal_d * diagonal_e),
            (0, 1, 2),
            basis,
        )
        final_basis = _symbolic_vector((sign_c * cayley, sine), (3, 4), basis)
        second_negative = [
            sign_g * g * basis[0][index] + diagonal_g * final_basis[index]
            for index in range(8)
        ]
        information = (
            fixed
            + symbolic_query_projector(1, positive, generators)
            + symbolic_query_projector(2, first_negative, generators)
            + symbolic_query_projector(2, second_negative, generators)
        )
        determinants[signs] = sp.factor(
            1024 * information.det(method="domain-ge") / delta**3
        )
    coefficients = {}
    for mask in itertools.product((0, 1), repeat=5):
        coefficient = sp.factor(
            sum(
                _character(signs, mask) * determinant
                for signs, determinant in determinants.items()
            )
            / 32
        )
        if coefficient != 0:
            coefficients[mask] = coefficient
    expected = set(SIGN_CHARACTERS.values())
    return {
        "rational_circle_parameters": ["1/3", "2/5", "1/4", "3/7", "2/7"],
        "nonzero_masks": [list(mask) for mask in sorted(coefficients)],
        "expected_masks": [list(mask) for mask in sorted(expected)],
        "coefficient_signs": {
            name: int(sp.sign(coefficients[mask]))
            for name, mask in SIGN_CHARACTERS.items()
        },
        "passed": set(coefficients) == expected,
    }


def _projector(
    generators: torch.Tensor, view: int, states: torch.Tensor
) -> torch.Tensor:
    jacobian = torch.einsum("pij,bj->bip", generators[view], states)
    return jacobian.transpose(-1, -2) @ jacobian


def log_advantage(
    parameters: torch.Tensor, generators: torch.Tensor, fixed: torch.Tensor
) -> torch.Tensor:
    a, d, e, g, cayley = parameters.unbind(dim=-1)
    diagonal_a = torch.sqrt(torch.clamp(1 - a.square(), min=1e-30))
    diagonal_d = torch.sqrt(torch.clamp(1 - d.square(), min=1e-30))
    diagonal_e = torch.sqrt(torch.clamp(1 - e.square(), min=1e-30))
    diagonal_g = torch.sqrt(torch.clamp(1 - g.square(), min=1e-30))
    sine = torch.sqrt(torch.clamp(1 - cayley.square(), min=1e-30))
    zeros = torch.zeros_like(a)

    positive = torch.stack(
        (a, diagonal_a, zeros, zeros, zeros, zeros, zeros, zeros), -1
    )
    first_negative = torch.stack(
        (
            d,
            diagonal_d * e,
            diagonal_d * diagonal_e,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
        ),
        -1,
    )
    second_negative = torch.stack(
        (
            g,
            zeros,
            zeros,
            diagonal_g * cayley,
            diagonal_g * sine,
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
    delta = (
        diagonal_a.square()
        * diagonal_d.square()
        * diagonal_e.square()
        * diagonal_g.square()
    )
    target = (1 - cayley.square()).pow(3) * (9 - cayley.square()).pow(2)
    normalized_logdet = math.log(1024.0) + logdet - 3 * torch.log(delta)
    result = normalized_logdet - torch.log(target)
    return torch.where(sign > 0, result, torch.full_like(result, -torch.inf))


def numerical_attack(
    *,
    seed: int,
    sample_count: int,
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
    best_value = -torch.inf
    best_parameters = None
    chunk_size = 4096
    remaining = sample_count
    while remaining:
        count = min(chunk_size, remaining)
        parameters = (
            2 * torch.rand(count, 5, generator=generator, dtype=dtype) - 1
        ) * cap
        parameters = parameters.to(device)
        values = log_advantage(parameters, generators, fixed)
        value, index = values.max(dim=0)
        if value > best_value:
            best_value = value.detach()
            best_parameters = parameters[index].detach().clone()
        remaining -= count

    initial = (2 * torch.rand(restarts, 5, generator=generator, dtype=dtype) - 1) * (
        0.8 * cap
    )
    if best_parameters is not None:
        initial[0] = best_parameters.cpu()
    raw = torch.atanh((initial / cap).clamp(-0.999999, 0.999999)).to(device)
    raw.requires_grad_(True)
    optimizer = torch.optim.Adam((raw,), lr=0.025)
    best_optimized_value = best_value
    best_optimized_parameters = best_parameters
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        parameters = cap * torch.tanh(raw)
        values = log_advantage(parameters, generators, fixed)
        loss = -values.mean()
        loss.backward()
        optimizer.step()
        value, index = values.detach().max(dim=0)
        if value > best_optimized_value:
            best_optimized_value = value
            best_optimized_parameters = parameters.detach()[index].clone()

    return {
        "device": str(device),
        "seed": seed,
        "sample_count": sample_count,
        "restarts": restarts,
        "optimization_steps": steps,
        "strict_parameter_cap": cap,
        "random_maximum_log_advantage": float(best_value.cpu()),
        "optimized_maximum_log_advantage": float(best_optimized_value.cpu()),
        "optimized_parameters": [
            float(value) for value in best_optimized_parameters.cpu().tolist()
        ],
        "violation_above_1e_minus_8": bool(best_optimized_value > 1e-8),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    walsh = exact_walsh_support_certificate()
    attack = numerical_attack(
        seed=args.seed,
        sample_count=args.samples,
        restarts=args.restarts,
        steps=args.steps,
        cap=args.cap,
        device=device,
    )
    return {
        "experiment": "variable-Cayley one-edge Dirac--Gram falsifier",
        "exact_walsh_support": walsh,
        "numerical_attack": attack,
        "exact_theorem_proved": False,
        "passed_falsifier_gate": bool(
            walsh["passed"] and not attack["violation_above_1e_minus_8"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--restarts", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--cap", type=float, default=0.98)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
