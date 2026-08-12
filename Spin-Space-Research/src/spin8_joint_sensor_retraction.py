"""Joint late retraction for five-query Spin(8) triality sensors."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from spin8_active_sensing import (
    DENSE_LENGTHS,
    NOISE_STD,
    QUERY_COUNT,
    REPRESENTATION_COUNT,
    SensorDesign,
    action_independence_audit,
    fit_noisy_actions,
    information_matrix,
    information_metrics,
    learn_hard_sensor,
    observe_actions,
    oracle_doptimal_sensor,
    random_mixed_sensor,
    recovery_report,
)
from spin8_blind_shared_action import composition_metrics, sample_teacher
from spin8_triality import SPIN8_BIVECTOR_DIM, torch_triality_generators
from spin8_triality_lift import triality_tensor


@dataclass(frozen=True)
class SoftSensorBank:
    logits: torch.Tensor  # (5, 3)
    vectors: torch.Tensor  # (5, 3, 8), normalized
    training: dict[str, object]


def soft_information(
    logits: torch.Tensor,
    vectors: torch.Tensor,
    generators: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    weights = F.softmax(logits / temperature, dim=-1)
    normalized = F.normalize(vectors, dim=-1)
    blocks = torch.einsum("rpij,qrj->qrip", generators, normalized)
    selected = torch.einsum("qr,qrip->qip", weights, blocks)
    jacobian = selected.reshape(QUERY_COUNT * 8, SPIN8_BIVECTOR_DIM)
    return jacobian.T @ jacobian


def learn_soft_sensor_bank(
    seed: int,
    generators: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
    final_temperature: float,
) -> SoftSensorBank:
    """Optimize the complete query family before any hard view selection."""

    cpu_generator = torch.Generator(device="cpu").manual_seed(93000 + seed)
    logits = nn.Parameter(
        0.02
        * torch.randn(
            QUERY_COUNT,
            REPRESENTATION_COUNT,
            generator=cpu_generator,
            dtype=torch.float64,
        ).to(generators.device)
    )
    vectors = nn.Parameter(
        torch.randn(
            QUERY_COUNT,
            REPRESENTATION_COUNT,
            8,
            generator=cpu_generator,
            dtype=torch.float64,
        ).to(generators.device)
    )
    identity = torch.eye(
        SPIN8_BIVECTOR_DIM, dtype=generators.dtype, device=generators.device
    )
    optimizer = torch.optim.Adam((logits, vectors), lr=learning_rate)
    trajectory = []
    log_steps = {0, 49, 199, 499, steps - 1}
    for step in range(steps):
        fraction = step / max(steps - 1, 1)
        temperature = 1.5 * (final_temperature / 1.5) ** fraction
        optimizer.zero_grad(set_to_none=True)
        information = soft_information(
            logits, vectors, generators, temperature=temperature
        )
        _, logdet = torch.linalg.slogdet(information + 1e-7 * identity)
        (-logdet).backward()
        optimizer.step()
        if step in log_steps:
            probabilities = F.softmax(logits.detach() / temperature, dim=-1)
            trajectory.append(
                {
                    "step": step + 1,
                    "temperature": temperature,
                    "regularized_soft_logdet": float(logdet.detach()),
                    "probabilities": probabilities.cpu().tolist(),
                    "independent_allocation": [
                        int(value)
                        for value in torch.bincount(
                            probabilities.argmax(dim=-1).cpu(), minlength=3
                        ).tolist()
                    ],
                }
            )
    return SoftSensorBank(
        logits.detach(),
        F.normalize(vectors.detach(), dim=-1),
        {
            "steps": steps,
            "learning_rate": learning_rate,
            "final_temperature": final_temperature,
            "trajectory": trajectory,
            "final_logits": logits.detach().cpu().tolist(),
        },
    )


def independent_argmax_design(bank: SoftSensorBank) -> SensorDesign:
    views = bank.logits.argmax(dim=-1)
    vectors = bank.vectors[torch.arange(QUERY_COUNT, device=views.device), views]
    return SensorDesign(views, vectors, "soft_independent_argmax")


def all_hard_assignments(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        list(itertools.product(range(REPRESENTATION_COUNT), repeat=QUERY_COUNT)),
        dtype=torch.long,
        device=device,
    )


def joint_retract_sensor(
    bank: SoftSensorBank, generators: torch.Tensor
) -> tuple[SensorDesign, dict[str, object]]:
    """Select the complete hard assignment by exact family log determinant."""

    assignments = all_hard_assignments(generators.device)
    query_indices = torch.arange(QUERY_COUNT, device=generators.device)
    vectors = bank.vectors[query_indices[None], assignments]
    selected_generators = generators[assignments]
    blocks = torch.einsum("aqpij,aqj->aqip", selected_generators, vectors)
    jacobian = blocks.reshape(-1, QUERY_COUNT * 8, SPIN8_BIVECTOR_DIM)
    information = jacobian.transpose(-1, -2) @ jacobian
    eigenvalues = torch.linalg.eigvalsh(information)
    tolerance = (
        SPIN8_BIVECTOR_DIM
        * torch.finfo(information.dtype).eps
        * eigenvalues[:, -1]
        * 32
    )
    ranks = (eigenvalues > tolerance[:, None]).sum(dim=-1)
    _, logdet = torch.linalg.slogdet(information)
    scores = torch.where(
        ranks == SPIN8_BIVECTOR_DIM,
        logdet,
        torch.full_like(logdet, -torch.inf),
    )
    best = int(scores.argmax())
    design = SensorDesign(assignments[best], vectors[best], "soft_joint_retracted")
    return design, {
        "assignment_count": int(assignments.shape[0]),
        "full_rank_assignment_count": int((ranks == SPIN8_BIVECTOR_DIM).sum()),
        "selected_index": best,
        "selected_log_determinant": float(scores[best]),
        "maximum_log_determinant": float(scores.max()),
        "selection_gap": float(scores.max() - scores[best]),
    }


def polish_sensor(
    design: SensorDesign,
    generators: torch.Tensor,
    *,
    steps: int,
    learning_rate: float = 4e-2,
) -> tuple[SensorDesign, dict[str, object]]:
    """Optimize physical vectors while keeping the retracted views frozen."""

    vectors = nn.Parameter(design.vectors.detach().clone())
    identity = torch.eye(28, dtype=generators.dtype, device=generators.device)
    optimizer = torch.optim.Adam((vectors,), lr=learning_rate)
    trajectory = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        candidate = SensorDesign(
            design.views, F.normalize(vectors, dim=-1), "candidate"
        )
        information = information_matrix(candidate, generators)
        _, logdet = torch.linalg.slogdet(information + 1e-7 * identity)
        (-logdet).backward()
        optimizer.step()
        if step in {0, 49, 199, steps - 1}:
            trajectory.append(
                {"step": step + 1, "regularized_logdet": float(logdet.detach())}
            )
    polished = SensorDesign(
        design.views,
        F.normalize(vectors.detach(), dim=-1),
        "soft_joint_polished",
    )
    return polished, {
        "steps": steps,
        "learning_rate": learning_rate,
        "trajectory": trajectory,
    }


def multiply_polynomials(left: list[Fraction], right: list[Fraction]) -> list[Fraction]:
    result = [Fraction(0) for _ in range(len(left) + len(right) - 1)]
    for first, left_value in enumerate(left):
        for second, right_value in enumerate(right):
            result[first + second] += left_value * right_value
    return result


def polynomial_power(base: list[Fraction], exponent: int) -> list[Fraction]:
    result = [Fraction(1)]
    for _ in range(exponent):
        result = multiply_polynomials(result, base)
    return result


def exact_characteristic_coefficients() -> list[Fraction]:
    """Return monic coefficients in descending order for the frozen factor law."""

    # Ascending coefficient order during exact multiplication.
    factors = [
        polynomial_power([Fraction(-1), Fraction(1)], 4),
        [Fraction(1), Fraction(-3), Fraction(1)],
        polynomial_power([Fraction(3), Fraction(-6), Fraction(2)], 4),
        polynomial_power([Fraction(1), Fraction(-4), Fraction(2)], 4),
        polynomial_power([Fraction(-1), Fraction(6), Fraction(-8), Fraction(2)], 2),
    ]
    result = [Fraction(1)]
    for factor in factors:
        result = multiply_polynomials(result, factor)
    result = [coefficient / 1024 for coefficient in result]
    return list(reversed(result))


def factor_polynomial_value(eigenvalues: torch.Tensor) -> torch.Tensor:
    value = (eigenvalues - 1) ** 4
    value = value * (eigenvalues**2 - 3 * eigenvalues + 1)
    value = value * (2 * eigenvalues**2 - 6 * eigenvalues + 3) ** 4
    value = value * (2 * eigenvalues**2 - 4 * eigenvalues + 1) ** 4
    value = value * (2 * eigenvalues**3 - 8 * eigenvalues**2 + 6 * eigenvalues - 1) ** 2
    return value / 1024


def single_query_projector_audit(
    generators: torch.Tensor, *, seed: int, probes_per_representation: int = 100
) -> dict[str, object]:
    cpu_generator = torch.Generator(device="cpu").manual_seed(94000 + seed)
    maximum_idempotence = 0.0
    maximum_trace_error = 0.0
    ranks = []
    for representation in range(REPRESENTATION_COUNT):
        probes = F.normalize(
            torch.randn(
                probes_per_representation,
                8,
                generator=cpu_generator,
                dtype=torch.float64,
            ).to(generators.device),
            dim=-1,
        )
        jacobian = torch.einsum("pij,bj->bip", generators[representation], probes)
        projector = jacobian.transpose(-1, -2) @ jacobian
        maximum_idempotence = max(
            maximum_idempotence,
            float((projector @ projector - projector).abs().max()),
        )
        maximum_trace_error = max(
            maximum_trace_error,
            float(
                (torch.diagonal(projector, dim1=-2, dim2=-1).sum(dim=-1) - 7)
                .abs()
                .max()
            ),
        )
        ranks.extend(
            int(value)
            for value in torch.linalg.matrix_rank(projector, atol=1e-10).tolist()
        )
    return {
        "probe_count": REPRESENTATION_COUNT * probes_per_representation,
        "maximum_idempotence_error": maximum_idempotence,
        "maximum_trace_error": maximum_trace_error,
        "minimum_rank": min(ranks),
        "maximum_rank": max(ranks),
        "passed": (
            maximum_idempotence < 1e-12
            and maximum_trace_error < 1e-12
            and min(ranks) == max(ranks) == 7
        ),
    }


def spectral_signature_audit(
    design: SensorDesign, generators: torch.Tensor
) -> dict[str, object]:
    information = information_matrix(design, generators)
    eigenvalues = torch.linalg.eigvalsh(information)
    observed_coefficients = np.poly(eigenvalues.detach().cpu().numpy())
    exact_coefficients = np.asarray(
        [float(value) for value in exact_characteristic_coefficients()]
    )
    coefficient_error = np.abs(observed_coefficients - exact_coefficients)
    relative_error = coefficient_error / np.maximum(1.0, np.abs(exact_coefficients))
    determinant = float(torch.linalg.det(information))
    trace = float(torch.trace(information))
    trace_inverse = float(torch.trace(torch.linalg.inv(information)))
    return {
        "maximum_factor_residual": float(
            factor_polynomial_value(eigenvalues).abs().max()
        ),
        "maximum_characteristic_coefficient_absolute_error": float(
            coefficient_error.max()
        ),
        "maximum_characteristic_coefficient_relative_error": float(
            relative_error.max()
        ),
        "determinant": determinant,
        "determinant_error_from_81_over_1024": abs(determinant - 81.0 / 1024.0),
        "trace": trace,
        "trace_error_from_35": abs(trace - 35.0),
        "trace_inverse": trace_inverse,
        "trace_inverse_error_from_43": abs(trace_inverse - 43.0),
        "passed": (
            float(factor_polynomial_value(eigenvalues).abs().max()) < 1e-10
            and float(relative_error.max()) < 1e-10
            and abs(determinant - 81.0 / 1024.0) < 1e-10
            and abs(trace - 35.0) < 1e-10
            and abs(trace_inverse - 43.0) < 1e-10
        ),
    }


def finite_metric(value: object, *, worst: float) -> float:
    return float(value) if value is not None else worst


def run_seed(
    seed: int,
    *,
    generators: torch.Tensor,
    rho: torch.Tensor,
    soft_steps: int,
    soft_learning_rate: float,
    final_temperature: float,
    hard_steps: int,
    polish_steps: int,
    oracle_steps: int,
    oracle_restarts: int,
    action_adam_steps: int,
    action_lbfgs_steps: int,
) -> dict[str, object]:
    bank = learn_soft_sensor_bank(
        seed,
        generators,
        steps=soft_steps,
        learning_rate=soft_learning_rate,
        final_temperature=final_temperature,
    )
    independent = independent_argmax_design(bank)
    joint, retraction = joint_retract_sensor(bank, generators)
    polished, polish = polish_sensor(joint, generators, steps=polish_steps)
    hard, hard_training = learn_hard_sensor(
        seed, generators, steps=hard_steps, learning_rate=2e-2
    )
    oracle, oracle_search, _ = oracle_doptimal_sensor(
        seed, generators, steps=oracle_steps, restarts=oracle_restarts
    )
    random = random_mixed_sensor(seed, generators.device)
    designs = {
        "hard_straight_through": SensorDesign(
            hard.views, hard.vectors, "hard_straight_through"
        ),
        "soft_independent_argmax": independent,
        "soft_joint_retracted": joint,
        "soft_joint_polished": polished,
        "oracle_doptimal": oracle,
        "random_mixed": random,
    }
    reports = {
        name: information_metrics(design, generators)
        for name, design in designs.items()
    }
    teacher = sample_teacher(seed=95000 + seed, generators=generators)
    independence = {
        name: action_independence_audit(design, generators, teacher.actions)
        for name, design in designs.items()
    }
    spectral = spectral_signature_audit(oracle, generators)

    noise_generator = torch.Generator(device="cpu").manual_seed(96000 + seed)
    matched_noise = (
        NOISE_STD
        * torch.randn(
            4,
            QUERY_COUNT,
            8,
            generator=noise_generator,
            dtype=torch.float64,
        )
    ).to(generators.device)
    recovery_names = (
        "hard_straight_through",
        "soft_joint_polished",
        "oracle_doptimal",
        "random_mixed",
    )
    learned_actions = {}
    recovery = {}
    for offset, name in enumerate(recovery_names):
        design = designs[name]
        target = observe_actions(teacher.actions, design) + matched_noise
        actions, fit = fit_noisy_actions(
            target,
            design,
            generators,
            seed=100 * seed + offset,
            adam_steps=action_adam_steps,
            lbfgs_steps=action_lbfgs_steps,
        )
        learned_actions[name] = actions
        recovery[name] = recovery_report(
            actions, teacher.actions, rho, design, seed=seed, fit=fit
        )
    composition = composition_metrics(
        {**learned_actions, "oracle": teacher.actions},
        seed=97000 + seed,
        lengths=DENSE_LENGTHS,
    )

    oracle_report = reports["oracle_doptimal"]
    hard_report = reports["hard_straight_through"]
    independent_report = reports["soft_independent_argmax"]
    joint_report = reports["soft_joint_retracted"]
    polished_report = reports["soft_joint_polished"]
    polished_gap = float(oracle_report["log_determinant"]) - finite_metric(
        polished_report["log_determinant"], worst=-math.inf
    )
    polished_trace_ratio = finite_metric(
        polished_report["trace_inverse"], worst=math.inf
    ) / float(oracle_report["trace_inverse"])
    hard_gap = float(oracle_report["log_determinant"]) - finite_metric(
        hard_report["log_determinant"], worst=-math.inf
    )
    hard_trace_ratio = finite_metric(
        hard_report["trace_inverse"], worst=math.inf
    ) / float(oracle_report["trace_inverse"])
    independent_logdet = finite_metric(
        independent_report["log_determinant"], worst=-math.inf
    )
    joint_logdet = finite_metric(joint_report["log_determinant"], worst=-math.inf)
    independent_trace = finite_metric(
        independent_report["trace_inverse"], worst=math.inf
    )
    joint_trace = finite_metric(joint_report["trace_inverse"], worst=math.inf)
    joint_improves = (
        joint_logdet > independent_logdet and joint_trace < independent_trace
    )

    polished_long = min(
        float(row["mean_cosine"])
        for row in composition["soft_joint_polished"]["2048"].values()
    )
    random_long = min(
        float(row["mean_cosine"])
        for row in composition["random_mixed"]["2048"].values()
    )
    oracle_long = min(
        float(row["mean_cosine"])
        for row in composition["oracle_doptimal"]["2048"].values()
    )
    polished_least = float(
        recovery["soft_joint_polished"]["least_observed_one_step_mean_cosine"]
    )
    random_least = float(
        recovery["random_mixed"]["least_observed_one_step_mean_cosine"]
    )
    polished_norm = max(
        float(row["maximum_absolute_log_norm"])
        for length in composition["soft_joint_polished"].values()
        for row in length.values()
    )
    polished_recovery = recovery["soft_joint_polished"]
    noisy_pass = (
        polished_least > random_least
        and polished_long > random_long
        and polished_long >= oracle_long - 0.02
        and float(polished_recovery["triality_equivariance_max_error"]) < 1e-8
        and float(polished_recovery["scan_parallel_recurrent_max_error"]) < 1e-9
        and polished_norm < 1e-5
    )
    return {
        "seed": seed,
        "soft_training": bank.training,
        "joint_retraction": retraction,
        "joint_polish": polish,
        "hard_training": hard_training,
        "oracle_search": oracle_search,
        "designs": reports,
        "action_independence": independence,
        "spectral_signature": spectral,
        "recovery": recovery,
        "composition": composition,
        "polished_logdet_gap_to_oracle": polished_gap,
        "polished_trace_inverse_ratio_to_oracle": polished_trace_ratio,
        "hard_logdet_gap_to_oracle": hard_gap,
        "hard_trace_inverse_ratio_to_oracle": hard_trace_ratio,
        "joint_improves_independent": joint_improves,
        "polished_conditioning_pass": (
            sorted(polished_report["allocation"]) == [1, 2, 2]
            and polished_gap <= 0.10
            and polished_trace_ratio <= 1.10
        ),
        "hard_conditioning_pass": (
            sorted(hard_report["allocation"]) == [1, 2, 2]
            and hard_gap <= 0.10
            and hard_trace_ratio <= 1.10
        ),
        "polished_least_observed_advantage": polished_least - random_least,
        "polished_length2048_advantage": polished_long - random_long,
        "polished_length2048_gap_to_oracle": oracle_long - polished_long,
        "noisy_recovery_pass": noisy_pass,
    }


def summarize(
    rows: list[dict[str, object]], projector_audit: dict[str, object]
) -> dict[str, object]:
    count = len(rows)
    retraction_valid = sum(
        row["joint_retraction"]["assignment_count"] == 243
        and row["joint_retraction"]["selection_gap"] == 0.0
        and row["designs"]["soft_joint_retracted"]["rank"] == 28
        and sum(
            value > 0 for value in row["designs"]["soft_joint_retracted"]["allocation"]
        )
        >= 2
        for row in rows
    )
    polished_passes = sum(bool(row["polished_conditioning_pass"]) for row in rows)
    hard_passes = sum(bool(row["hard_conditioning_pass"]) for row in rows)
    causal_passes = sum(bool(row["joint_improves_independent"]) for row in rows)
    noisy_passes = sum(bool(row["noisy_recovery_pass"]) for row in rows)
    spectral_passes = sum(bool(row["spectral_signature"]["passed"]) for row in rows)
    return {
        "seed_count": count,
        "retraction_valid_pass_count": retraction_valid,
        "polished_conditioning_pass_count": polished_passes,
        "hard_conditioning_pass_count": hard_passes,
        "joint_beats_independent_pass_count": causal_passes,
        "noisy_recovery_pass_count": noisy_passes,
        "spectral_signature_pass_count": spectral_passes,
        "single_query_projector_gate_passed": bool(projector_audit["passed"]),
        "retraction_validity_gate_passed": retraction_valid == count,
        "conditioning_reliability_gate_passed": (
            polished_passes >= math.ceil(0.8 * count)
            and polished_passes >= hard_passes + 2
        ),
        "causal_retraction_gate_passed": causal_passes >= math.ceil(0.8 * count),
        "noisy_recovery_gate_passed": noisy_passes >= math.ceil(0.8 * count),
        "spectral_replication_gate_passed": spectral_passes == count,
    }


def run(
    *,
    device: torch.device,
    seeds: tuple[int, ...],
    soft_steps: int,
    soft_learning_rate: float,
    final_temperature: float,
    hard_steps: int,
    polish_steps: int,
    oracle_steps: int,
    oracle_restarts: int,
    action_adam_steps: int,
    action_lbfgs_steps: int,
) -> dict[str, object]:
    generators = torch_triality_generators(dtype=torch.float64, device=device)
    rho = triality_tensor(dtype=torch.float64, device=device)
    projector_audit = single_query_projector_audit(generators, seed=20260803)
    rows = [
        run_seed(
            seed,
            generators=generators,
            rho=rho,
            soft_steps=soft_steps,
            soft_learning_rate=soft_learning_rate,
            final_temperature=final_temperature,
            hard_steps=hard_steps,
            polish_steps=polish_steps,
            oracle_steps=oracle_steps,
            oracle_restarts=oracle_restarts,
            action_adam_steps=action_adam_steps,
            action_lbfgs_steps=action_lbfgs_steps,
        )
        for seed in seeds
    ]
    return {
        "experiment": "Spin8 joint five-query sensor retraction",
        "device": str(device),
        "dtype": str(generators.dtype),
        "seeds": list(seeds),
        "noise_std": NOISE_STD,
        "soft_steps": soft_steps,
        "soft_learning_rate": soft_learning_rate,
        "final_temperature": final_temperature,
        "hard_steps": hard_steps,
        "polish_steps": polish_steps,
        "oracle_steps": oracle_steps,
        "oracle_restarts": oracle_restarts,
        "action_adam_steps": action_adam_steps,
        "action_lbfgs_steps": action_lbfgs_steps,
        "dense_lengths": list(DENSE_LENGTHS),
        "single_query_projector_audit": projector_audit,
        "exact_characteristic_coefficients": [
            str(value) for value in exact_characteristic_coefficients()
        ],
        "results": rows,
        "summary": summarize(rows, projector_audit),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seeds", default="20,21,22,23,24,25,26,27,28,29")
    parser.add_argument("--soft-steps", type=int, default=1500)
    parser.add_argument("--soft-lr", type=float, default=2e-2)
    parser.add_argument("--final-temperature", type=float, default=0.35)
    parser.add_argument("--hard-steps", type=int, default=1500)
    parser.add_argument("--polish-steps", type=int, default=600)
    parser.add_argument("--oracle-steps", type=int, default=600)
    parser.add_argument("--oracle-restarts", type=int, default=4)
    parser.add_argument("--action-adam-steps", type=int, default=1500)
    parser.add_argument("--action-lbfgs-steps", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        device=torch.device(args.device),
        seeds=tuple(int(value) for value in args.seeds.split(",") if value),
        soft_steps=args.soft_steps,
        soft_learning_rate=args.soft_lr,
        final_temperature=args.final_temperature,
        hard_steps=args.hard_steps,
        polish_steps=args.polish_steps,
        oracle_steps=args.oracle_steps,
        oracle_restarts=args.oracle_restarts,
        action_adam_steps=args.action_adam_steps,
        action_lbfgs_steps=args.action_lbfgs_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
