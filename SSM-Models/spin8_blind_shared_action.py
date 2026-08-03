"""Blind completion of shared Spin(8) token actions from partial endpoints."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from spin8_triality import (
    AffineTransition,
    SPIN8_BIVECTOR_DIM,
    apply_affine,
    associative_prefix_scan,
    spin8_actions,
    torch_triality_generators,
)
from spin8_triality_lift import triality_bind, triality_tensor


REPRESENTATION_NAMES = ("vector", "positive", "negative")
TOKEN_COUNT = 4
OBSERVED_COLUMNS = 5


@dataclass(frozen=True)
class TeacherFamily:
    coefficients: torch.Tensor
    actions: torch.Tensor
    design: dict[str, object]
    resamples: int


def polar_so8(matrices: torch.Tensor) -> torch.Tensor:
    """Project each matrix independently to SO(8) in Frobenius norm."""

    left, _, right = torch.linalg.svd(matrices)
    determinant = torch.linalg.det(left @ right)
    correction = torch.ones(*matrices.shape[:-2], 8, device=matrices.device, dtype=matrices.dtype)
    correction[..., -1] = torch.where(
        determinant < 0,
        -torch.ones_like(determinant),
        torch.ones_like(determinant),
    )
    return (left * correction[..., None, :]) @ right


def observed_action(actions: torch.Tensor) -> torch.Tensor:
    return actions[:, :2, :, :OBSERVED_COLUMNS]


def action_design_audit(
    coefficients: torch.Tensor, generators: torch.Tensor
) -> dict[str, object]:
    reports = []
    for token in range(coefficients.shape[0]):
        point = coefficients[token].detach().clone().requires_grad_(True)

        def observation(coordinate: torch.Tensor) -> torch.Tensor:
            action = spin8_actions(coordinate, generators)
            return action[:2, :, :OBSERVED_COLUMNS].reshape(-1)

        jacobian = torch.autograd.functional.jacobian(
            observation, point, vectorize=True
        )
        singular_values = torch.linalg.svdvals(jacobian)
        tolerance = 1e-9 * singular_values[0]
        nonzero = singular_values[singular_values > tolerance]
        reports.append(
            {
                "token": token,
                "rank": int(nonzero.numel()),
                "smallest_singular_value": float(nonzero[-1]),
                "condition_number": float(nonzero[0] / nonzero[-1]),
            }
        )
    return {
        "tokens": reports,
        "minimum_rank": min(int(row["rank"]) for row in reports),
        "maximum_condition_number": max(
            float(row["condition_number"]) for row in reports
        ),
    }


def vector_commutator(actions: torch.Tensor) -> float:
    first, second = actions[0, 0], actions[1, 0]
    return float(torch.linalg.matrix_norm(first @ second - second @ first))


def sample_teacher(
    *,
    seed: int,
    generators: torch.Tensor,
    minimum_commutator: float = 0.35,
) -> TeacherFamily:
    generator = torch.Generator(device=generators.device).manual_seed(51000 + seed)
    for attempt in range(32):
        coefficients = 0.18 * torch.randn(
            TOKEN_COUNT,
            SPIN8_BIVECTOR_DIM,
            generator=generator,
            device=generators.device,
            dtype=generators.dtype,
        )
        actions = spin8_actions(coefficients, generators)
        if vector_commutator(actions) < minimum_commutator:
            continue
        design = action_design_audit(coefficients, generators)
        if int(design["minimum_rank"]) == SPIN8_BIVECTOR_DIM:
            return TeacherFamily(coefficients, actions, design, attempt)
    raise RuntimeError("failed to sample an identifiable noncommuting teacher")


def fit_unconstrained(
    target_observation: torch.Tensor,
    *,
    seed: int,
    steps: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    device, dtype = target_observation.device, target_observation.dtype
    generator = torch.Generator(device=device).manual_seed(52000 + seed)
    identity = torch.eye(8, device=device, dtype=dtype).expand(
        TOKEN_COUNT, 3, 8, 8
    )
    matrices = nn.Parameter(
        identity.clone()
        + 0.02
        * torch.randn(
            TOKEN_COUNT,
            3,
            8,
            8,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    )
    optimizer = torch.optim.Adam((matrices,), lr=3e-2)
    trajectory = []
    log_steps = {0, 49, 199, steps - 1}
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(observed_action(matrices), target_observation)
        loss.backward()
        optimizer.step()
        if step in log_steps:
            trajectory.append({"step": step + 1, "loss": float(loss.detach())})
    return matrices.detach(), {
        "steps": steps,
        "trajectory": trajectory,
        "final_observed_mse": float(
            F.mse_loss(observed_action(matrices), target_observation).detach()
        ),
    }


def joint_shared_retraction(
    fitted_observation: torch.Tensor,
    *,
    seed: int,
    generators: torch.Tensor,
    adam_steps: int,
    lbfgs_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    generator = torch.Generator(device=generators.device).manual_seed(53000 + seed)
    coefficients = nn.Parameter(
        0.01
        * torch.randn(
            TOKEN_COUNT,
            SPIN8_BIVECTOR_DIM,
            generator=generator,
            device=generators.device,
            dtype=generators.dtype,
        )
    )

    def objective() -> torch.Tensor:
        return F.mse_loss(
            observed_action(spin8_actions(coefficients, generators)),
            fitted_observation,
        )

    adam = torch.optim.Adam((coefficients,), lr=3e-2)
    trajectory = []
    log_steps = {0, 49, 199, 499, adam_steps - 1}
    for step in range(adam_steps):
        adam.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        adam.step()
        if step in log_steps:
            trajectory.append({"stage": "adam", "step": step + 1, "loss": float(loss.detach())})

    lbfgs = torch.optim.LBFGS(
        (coefficients,),
        lr=1.0,
        max_iter=lbfgs_steps,
        tolerance_grad=1e-14,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        lbfgs.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        return loss

    lbfgs.step(closure)
    actions = spin8_actions(coefficients, generators).detach()
    final_loss = float(objective().detach())
    trajectory.append(
        {"stage": "lbfgs", "step": lbfgs_steps, "loss": final_loss}
    )
    return actions, coefficients.detach(), {
        "adam_steps": adam_steps,
        "lbfgs_steps": lbfgs_steps,
        "trajectory": trajectory,
        "final_observed_mse": final_loss,
    }


def independent_lie_retraction(
    fitted_observation: torch.Tensor,
    *,
    generators: torch.Tensor,
    adam_steps: int,
    lbfgs_steps: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    coordinates = nn.Parameter(
        torch.zeros(
            TOKEN_COUNT,
            3,
            SPIN8_BIVECTOR_DIM,
            device=generators.device,
            dtype=generators.dtype,
        )
    )

    def actions() -> torch.Tensor:
        tangent = torch.einsum(
            "trk,rkij->trij", coordinates, generators
        ).contiguous()
        return torch.matrix_exp(tangent)

    def objective() -> torch.Tensor:
        return F.mse_loss(observed_action(actions()), fitted_observation)

    adam = torch.optim.Adam((coordinates,), lr=3e-2)
    trajectory = []
    for step in range(adam_steps):
        adam.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        adam.step()
        if step in {0, 49, 199, 499, adam_steps - 1}:
            trajectory.append(
                {"stage": "adam", "step": step + 1, "loss": float(loss.detach())}
            )
    lbfgs = torch.optim.LBFGS(
        (coordinates,),
        lr=1.0,
        max_iter=lbfgs_steps,
        tolerance_grad=1e-14,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        lbfgs.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        return loss

    lbfgs.step(closure)
    final_actions = actions().detach()
    final_loss = float(objective().detach())
    trajectory.append(
        {"stage": "lbfgs", "step": lbfgs_steps, "loss": final_loss}
    )
    return final_actions, {
        "adam_steps": adam_steps,
        "lbfgs_steps": lbfgs_steps,
        "trajectory": trajectory,
        "final_observed_mse": final_loss,
    }


def one_step_metrics(
    actions: torch.Tensor,
    oracle: torch.Tensor,
    *,
    seed: int,
    examples: int = 1024,
) -> dict[str, object]:
    generator = torch.Generator(device=actions.device).manual_seed(seed)
    tokens = torch.randint(
        0, TOKEN_COUNT, (examples,), generator=generator, device=actions.device
    )
    states = F.normalize(
        torch.randn(
            examples,
            3,
            8,
            generator=generator,
            device=actions.device,
            dtype=actions.dtype,
        ),
        dim=-1,
    )
    predicted = torch.einsum("erij,erj->eri", actions[tokens], states)
    expected = torch.einsum("erij,erj->eri", oracle[tokens], states)
    result = {}
    for index, name in enumerate(REPRESENTATION_NAMES):
        cosine = F.cosine_similarity(predicted[:, index], expected[:, index], dim=-1)
        result[name] = {
            "mean_cosine": float(cosine.mean()),
            "minimum_cosine": float(cosine.min()),
            "mse": float(F.mse_loss(predicted[:, index], expected[:, index])),
        }
    return result


def composition_metrics(
    families: dict[str, torch.Tensor],
    *,
    seed: int,
    lengths: tuple[int, ...] = (8, 32, 128, 512, 2048),
    examples: int = 128,
) -> dict[str, object]:
    names = tuple(families)
    stacked = torch.stack(tuple(families.values()))
    oracle_index = names.index("oracle")
    generator = torch.Generator(device=stacked.device).manual_seed(seed)
    report: dict[str, object] = {name: {} for name in names}
    for length in lengths:
        tokens = torch.randint(
            0,
            TOKEN_COUNT,
            (examples, length),
            generator=generator,
            device=stacked.device,
        )
        initial = F.normalize(
            torch.randn(
                examples,
                3,
                8,
                generator=generator,
                device=stacked.device,
                dtype=stacked.dtype,
            ),
            dim=-1,
        )
        state = initial[None].expand(len(names), -1, -1, -1).clone()
        log_norm = torch.zeros(
            len(names), examples, 3, device=stacked.device, dtype=stacked.dtype
        )
        for position in range(length):
            selected = stacked[:, tokens[:, position]]
            state = torch.einsum("ferij,ferj->feri", selected, state)
            norm = state.norm(dim=-1).clamp_min(torch.finfo(state.dtype).tiny)
            log_norm = log_norm + norm.log()
            state = state / norm[..., None]
        expected = state[oracle_index]
        for family_index, name in enumerate(names):
            family_row = {}
            for representation, representation_name in enumerate(
                REPRESENTATION_NAMES
            ):
                cosine = (
                    state[family_index, :, representation]
                    * expected[:, representation]
                ).sum(dim=-1)
                family_row[representation_name] = {
                    "mean_cosine": float(cosine.mean()),
                    "minimum_cosine": float(cosine.min()),
                    "maximum_absolute_log_norm": float(
                        log_norm[family_index, :, representation].abs().max()
                    ),
                }
            report[name][str(length)] = family_row
    return report


def triality_residual(
    actions: torch.Tensor,
    rho: torch.Tensor,
    *,
    seed: int,
    examples: int = 256,
) -> float:
    generator = torch.Generator(device=actions.device).manual_seed(seed)
    positive = F.normalize(
        torch.randn(
            examples,
            8,
            generator=generator,
            device=actions.device,
            dtype=actions.dtype,
        ),
        dim=-1,
    )
    negative = F.normalize(
        torch.randn(
            examples,
            8,
            generator=generator,
            device=actions.device,
            dtype=actions.dtype,
        ),
        dim=-1,
    )
    bound = triality_bind(positive, negative, rho)
    maximum = 0.0
    for token in range(TOKEN_COUNT):
        vector_action, positive_action, negative_action = actions[token]
        transformed_positive = torch.einsum(
            "ij,bj->bi", positive_action, positive
        )
        transformed_negative = torch.einsum(
            "ij,bj->bi", negative_action, negative
        )
        left = triality_bind(
            transformed_positive, transformed_negative, rho
        )
        right = torch.einsum("ij,bj->bi", vector_action, bound)
        maximum = max(maximum, float((left - right).abs().max()))
    return maximum


def scan_parity(
    actions: torch.Tensor,
    *,
    seed: int,
    length: int = 257,
) -> float:
    generator = torch.Generator(device=actions.device).manual_seed(seed)
    batch = 2
    tokens = torch.randint(
        0,
        TOKEN_COUNT,
        (batch, length),
        generator=generator,
        device=actions.device,
    )
    selected = actions[tokens]
    transition = AffineTransition(
        scale=torch.ones(batch, length, device=actions.device, dtype=actions.dtype),
        action=selected,
        drive=torch.zeros(batch, length, 3, 8, device=actions.device, dtype=actions.dtype),
    )
    initial = torch.randn(
        batch,
        3,
        8,
        generator=generator,
        device=actions.device,
        dtype=actions.dtype,
    )
    parallel = apply_affine(associative_prefix_scan(transition), initial[:, None])
    state = initial
    recurrent = []
    for position in range(length):
        state = torch.einsum("brij,brj->bri", selected[:, position], state)
        recurrent.append(state)
    return float((parallel - torch.stack(recurrent, dim=1)).abs().max())


def family_diagnostics(
    name: str,
    actions: torch.Tensor,
    oracle: torch.Tensor,
    rho: torch.Tensor,
    *,
    seed: int,
) -> dict[str, object]:
    observed_mse = float(
        F.mse_loss(observed_action(actions), observed_action(oracle))
    )
    commutator = vector_commutator(actions)
    oracle_commutator = vector_commutator(oracle)
    return {
        "family": name,
        "observed_column_mse": observed_mse,
        "one_step": one_step_metrics(actions, oracle, seed=61000 + seed),
        "triality_equivariance_max_error": triality_residual(
            actions, rho, seed=62000 + seed
        ),
        "vector_commutator": commutator,
        "commutator_ratio_to_oracle": commutator / oracle_commutator,
        "scan_parallel_recurrent_max_error": scan_parity(
            actions, seed=63000 + seed
        ),
    }


def joint_seed_pass(
    diagnostics: dict[str, object], composition: dict[str, object]
) -> bool:
    if float(diagnostics["observed_column_mse"]) >= 1e-8:
        return False
    if float(diagnostics["triality_equivariance_max_error"]) >= 1e-8:
        return False
    if float(diagnostics["scan_parallel_recurrent_max_error"]) >= 1e-10:
        return False
    if float(diagnostics["commutator_ratio_to_oracle"]) < 0.90:
        return False
    for row in diagnostics["one_step"].values():
        if float(row["mean_cosine"]) < 0.9999:
            return False
    for length in composition.values():
        for row in length.values():
            if float(row["mean_cosine"]) < 0.99:
                return False
            if float(row["maximum_absolute_log_norm"]) >= 1e-5:
                return False
    return True


def run_seed(
    seed: int,
    *,
    generators: torch.Tensor,
    rho: torch.Tensor,
    raw_steps: int,
    retraction_steps: int,
    lbfgs_steps: int,
) -> dict[str, object]:
    teacher = sample_teacher(seed=seed, generators=generators)
    target_observation = observed_action(teacher.actions).detach()
    raw, raw_training = fit_unconstrained(
        target_observation, seed=seed, steps=raw_steps
    )
    independent = polar_so8(raw)
    independent_lie, independent_lie_training = independent_lie_retraction(
        observed_action(raw),
        generators=generators,
        adam_steps=retraction_steps,
        lbfgs_steps=lbfgs_steps,
    )
    joint, coefficients, retraction = joint_shared_retraction(
        observed_action(raw),
        seed=seed,
        generators=generators,
        adam_steps=retraction_steps,
        lbfgs_steps=lbfgs_steps,
    )
    families = {
        "unconstrained": raw,
        "independent_polar": independent,
        "independent_lie": independent_lie,
        "joint_shared": joint,
        "oracle": teacher.actions,
    }
    diagnostics = {
        name: family_diagnostics(
            name, action, teacher.actions, rho, seed=seed
        )
        for name, action in families.items()
    }
    compositions = composition_metrics(families, seed=64000 + seed)
    joint_pass = joint_seed_pass(
        diagnostics["joint_shared"], compositions["joint_shared"]
    )
    return {
        "seed": seed,
        "teacher_resamples": teacher.resamples,
        "design": teacher.design,
        "oracle_vector_commutator": vector_commutator(teacher.actions),
        "raw_training": raw_training,
        "independent_lie_training": independent_lie_training,
        "joint_retraction": retraction,
        "joint_coefficient_cosine_with_hidden_teacher": float(
            F.cosine_similarity(
                coefficients.flatten(), teacher.coefficients.flatten(), dim=0
            )
        ),
        "diagnostics": diagnostics,
        "composition": compositions,
        "joint_pass": joint_pass,
    }


def summarize(seeds: list[dict[str, object]]) -> dict[str, object]:
    pass_count = sum(bool(row["joint_pass"]) for row in seeds)
    joint_beats_negative = 0
    joint_beats_long = 0
    raw_control_fits = 0
    for row in seeds:
        diagnostics = row["diagnostics"]
        composition = row["composition"]
        joint_negative = diagnostics["joint_shared"]["one_step"]["negative"][
            "mean_cosine"
        ]
        polar_negative = diagnostics["independent_lie"]["one_step"]["negative"][
            "mean_cosine"
        ]
        joint_beats_negative += float(joint_negative) > float(polar_negative)
        joint_long = min(
            float(value["mean_cosine"])
            for value in composition["joint_shared"]["2048"].values()
        )
        control_long = max(
            min(
                float(value["mean_cosine"])
                for value in composition[name]["2048"].values()
            )
            for name in ("unconstrained", "independent_polar", "independent_lie")
        )
        joint_beats_long += joint_long > control_long
        raw_control_fits += (
            float(diagnostics["unconstrained"]["observed_column_mse"]) < 1e-6
            and float(diagnostics["independent_lie"]["observed_column_mse"]) < 1e-6
        )
    return {
        "joint_pass_count": pass_count,
        "seed_count": len(seeds),
        "reliability_gate": pass_count >= 8,
        "joint_beats_independent_negative_count": joint_beats_negative,
        "joint_beats_controls_length2048_count": joint_beats_long,
        "unconstrained_fit_count": raw_control_fits,
        "strong_support": (
            pass_count >= 8
            and joint_beats_negative == len(seeds)
            and joint_beats_long >= 8
            and raw_control_fits == len(seeds)
        ),
    }


def run(
    *,
    device: torch.device,
    seeds: tuple[int, ...],
    raw_steps: int,
    retraction_steps: int,
    lbfgs_steps: int,
) -> dict[str, object]:
    dtype = torch.float64
    generators = torch_triality_generators(dtype=dtype, device=device)
    rho = triality_tensor(dtype=dtype, device=device)
    rows = [
        run_seed(
            seed,
            generators=generators,
            rho=rho,
            raw_steps=raw_steps,
            retraction_steps=retraction_steps,
            lbfgs_steps=lbfgs_steps,
        )
        for seed in seeds
    ]
    return {
        "experiment": "blind shared Spin(8) action completion",
        "device": str(device),
        "dtype": str(dtype),
        "seeds": list(seeds),
        "raw_steps": raw_steps,
        "retraction_steps": retraction_steps,
        "lbfgs_steps": lbfgs_steps,
        "results": rows,
        "summary": summarize(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--raw-steps", type=int, default=500)
    parser.add_argument("--retraction-steps", type=int, default=1500)
    parser.add_argument("--lbfgs-steps", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        device=torch.device(args.device),
        seeds=tuple(int(value) for value in args.seeds.split(",") if value),
        raw_steps=args.raw_steps,
        retraction_steps=args.retraction_steps,
        lbfgs_steps=args.lbfgs_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
