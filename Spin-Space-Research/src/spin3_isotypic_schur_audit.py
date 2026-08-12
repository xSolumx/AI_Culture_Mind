"""Execute the frozen Spin(3) isotypic and Schur-scan foundational audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rotor_ssm_torch import GA_DIM, GradeLinear, rotor_from_bivector, rotor_sandwich
from schur_scan import (
    SchurAffineTransition,
    Spin3IsotypicLinear,
    apply_schur_affine,
    associative_schur_scan,
    compose_schur_affine,
    pack_cl3_isotypic,
    unpack_cl3_isotypic,
)


def linear_matrix(function, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    basis = torch.eye(GA_DIM, dtype=dtype, device=device).unsqueeze(-2)
    return function(basis).squeeze(-2).T


def hodge_copy_swap(inputs: torch.Tensor) -> torch.Tensor:
    trivial, active = pack_cl3_isotypic(inputs)
    trivial = trivial.reshape(*trivial.shape[:-1], 1, 2).flip(-1)
    active = active.reshape(*active.shape[:-2], 1, 2, 3).flip(-2)
    return unpack_cl3_isotypic(trivial.flatten(-2), active.flatten(-3, -2))


def projection_residual(target: torch.Tensor, basis: list[torch.Tensor]) -> float:
    design = torch.stack([value.reshape(-1) for value in basis], dim=1)
    coefficients = torch.linalg.lstsq(design, target.reshape(-1, 1)).solution
    residual = target.reshape(-1, 1) - design @ coefficients
    return float(residual.norm() / target.norm())


def centralizer_audit(dtype: torch.dtype, device: torch.device) -> dict[str, object]:
    generator = torch.Generator(device=device).manual_seed(20260803)
    rotors = rotor_from_bivector(
        torch.randn(7, 3, generator=generator, dtype=dtype, device=device)
    )
    action_matrices = [
        linear_matrix(lambda x, rotor=rotor: rotor_sandwich(rotor, x), dtype=dtype, device=device)
        for rotor in rotors
    ]
    constraints = []
    elementary = []
    for row in range(GA_DIM):
        for column in range(GA_DIM):
            matrix = torch.zeros(GA_DIM, GA_DIM, dtype=dtype, device=device)
            matrix[row, column] = 1.0
            elementary.append(matrix)
    for action in action_matrices:
        constraints.append(
            torch.stack([(matrix @ action - action @ matrix).reshape(-1) for matrix in elementary], dim=1)
        )
    singular_values = torch.linalg.svdvals(torch.cat(constraints, dim=0))
    tolerance = 1e-9 * float(singular_values.max())
    centralizer_dimension = int((singular_values < tolerance).sum())

    grade_basis = []
    for start, stop in ((0, 1), (1, 4), (4, 7), (7, 8)):
        matrix = torch.zeros(GA_DIM, GA_DIM, dtype=dtype, device=device)
        matrix[start:stop, start:stop] = torch.eye(stop - start, dtype=dtype, device=device)
        grade_basis.append(matrix)

    isotypic_basis = []
    for sector in ("trivial", "active"):
        for output_copy in range(2):
            for input_copy in range(2):
                def apply(inputs, sector=sector, output_copy=output_copy, input_copy=input_copy):
                    trivial, active = pack_cl3_isotypic(inputs)
                    trivial = trivial.reshape(*trivial.shape[:-1], 1, 2)
                    active = active.reshape(*active.shape[:-2], 1, 2, 3)
                    out_trivial = torch.zeros_like(trivial)
                    out_active = torch.zeros_like(active)
                    if sector == "trivial":
                        out_trivial[..., 0, output_copy] = trivial[..., 0, input_copy]
                    else:
                        out_active[..., 0, output_copy, :] = active[..., 0, input_copy, :]
                    return unpack_cl3_isotypic(
                        out_trivial.flatten(-2), out_active.flatten(-3, -2)
                    )
                isotypic_basis.append(linear_matrix(apply, dtype=dtype, device=device))

    target = linear_matrix(hodge_copy_swap, dtype=dtype, device=device)
    max_isotypic_commutator = max(
        float((basis @ action - action @ basis).abs().max())
        for basis in isotypic_basis
        for action in action_matrices
    )
    return {
        "centralizer_dimension": centralizer_dimension,
        "smallest_singular_values": [float(value) for value in singular_values[-10:]],
        "grade_basis_rank": int(torch.linalg.matrix_rank(torch.stack([x.reshape(-1) for x in grade_basis], 1))),
        "isotypic_basis_rank": int(torch.linalg.matrix_rank(torch.stack([x.reshape(-1) for x in isotypic_basis], 1))),
        "isotypic_max_commutator": max_isotypic_commutator,
        "hodge_swap_grade_projection_relative_residual": projection_residual(target, grade_basis),
        "hodge_swap_isotypic_projection_relative_residual": projection_residual(target, isotypic_basis),
        "hodge_swap_equivariance_max_error": max(
            float((target @ action - action @ target).abs().max()) for action in action_matrices
        ),
    }


def fit_audit(device: torch.device) -> dict[str, float | str]:
    torch.manual_seed(20260804)
    inputs = torch.randn(4096, 1, GA_DIM, device=device)
    targets = hodge_copy_swap(inputs)
    models: dict[str, torch.nn.Module] = {
        "grade_linear": GradeLinear(1, 1, use_bias=False),
        "wide_two_layer_grade_linear": torch.nn.Sequential(
            GradeLinear(1, 2, use_bias=False),
            GradeLinear(2, 1, use_bias=False),
        ),
        "isotypic_linear": Spin3IsotypicLinear(1, 1, use_bias=False),
    }
    output: dict[str, float | str] = {"device": str(device)}
    for name, model in models.items():
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        for _ in range(300):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
        output[f"{name}_parameters"] = float(sum(p.numel() for p in model.parameters()))
        output[f"{name}_mse"] = float(
            torch.nn.functional.mse_loss(model(inputs), targets).detach()
        )
    return output


def random_transition(
    *shape: int, multiplicity: int, dtype: torch.dtype
) -> SchurAffineTransition:
    def near_identity() -> torch.Tensor:
        eye = torch.eye(multiplicity, dtype=dtype).expand(*shape, -1, -1)
        return 0.8 * eye + 0.05 * torch.randn(*shape, multiplicity, multiplicity, dtype=dtype)
    skew = torch.randn(*shape, 3, 3, dtype=dtype)
    skew = 0.15 * (skew - skew.transpose(-1, -2))
    return SchurAffineTransition(
        trivial_action=near_identity(),
        active_multiplicity=near_identity(),
        rotation=torch.matrix_exp(skew),
        trivial_drive=0.1 * torch.randn(*shape, multiplicity, dtype=dtype),
        active_drive=0.1 * torch.randn(*shape, multiplicity, 3, dtype=dtype),
    )


def transition_difference(left: SchurAffineTransition, right: SchurAffineTransition) -> float:
    return max(float((a - b).abs().max()) for a, b in zip(left.__dict__.values(), right.__dict__.values()))


def scan_audit() -> dict[str, float]:
    torch.manual_seed(20260805)
    dtype = torch.float64
    first, second, third = (random_transition(multiplicity=4, dtype=dtype) for _ in range(3))
    associativity = transition_difference(
        compose_schur_affine(third, compose_schur_affine(second, first)),
        compose_schur_affine(compose_schur_affine(third, second), first),
    )
    transition = random_transition(2, 17, multiplicity=4, dtype=dtype)
    initial = (torch.randn(2, 4, dtype=dtype), torch.randn(2, 4, 3, dtype=dtype))
    prefixes = associative_schur_scan(transition)
    parallel = apply_schur_affine(prefixes, (initial[0][:, None], initial[1][:, None]))
    state = initial
    recurrent_trivial, recurrent_active = [], []
    for position in range(17):
        step = SchurAffineTransition(
            *(value[:, position] for value in transition.__dict__.values())
        )
        state = apply_schur_affine(step, state)
        recurrent_trivial.append(state[0])
        recurrent_active.append(state[1])
    recurrent = (torch.stack(recurrent_trivial, 1), torch.stack(recurrent_active, 1))
    return {
        "composition_associativity_max_error": associativity,
        "parallel_recurrent_trivial_max_error": float((parallel[0] - recurrent[0]).abs().max()),
        "parallel_recurrent_active_max_error": float((parallel[1] - recurrent[1]).abs().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    report = {
        "experiment": "Spin(3) isotypic commutant and representation-factored Schur scan",
        "centralizer": centralizer_audit(torch.float64, torch.device("cpu")),
        "optimization": fit_audit(device),
        "scan": scan_audit(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
