"""Active five-query sensing for shared Spin(8) triality actions."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from spin8_blind_shared_action import (
    REPRESENTATION_NAMES,
    TOKEN_COUNT,
    composition_metrics,
    one_step_metrics,
    sample_teacher,
    scan_parity,
    triality_residual,
    vector_commutator,
)
from spin8_five_probe_identifiability import DENSE_LENGTHS
from spin8_triality import SPIN8_BIVECTOR_DIM, spin8_actions, torch_triality_generators
from spin8_triality_lift import triality_tensor

QUERY_COUNT = 5
REPRESENTATION_COUNT = 3
NOISE_STD = 1e-3


@dataclass(frozen=True)
class SensorDesign:
    """Five hard representation/state queries."""

    views: torch.Tensor  # (5,), integer representation indices
    vectors: torch.Tensor  # (5, 8), unit probes
    name: str

    def to(self, device: torch.device, dtype: torch.dtype) -> "SensorDesign":
        return SensorDesign(
            self.views.to(device=device),
            self.vectors.to(device=device, dtype=dtype),
            self.name,
        )


def canonical_random_vectors(seed: int, count: int = QUERY_COUNT) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return F.normalize(
        torch.randn(count, 8, generator=generator, dtype=torch.float64), dim=-1
    )


def design_jacobian(design: SensorDesign, generators: torch.Tensor) -> torch.Tensor:
    """Left-invariant endpoint Jacobian with shape ``(5*8, 28)``."""

    selected = generators[design.views]
    blocks = torch.einsum("qpij,qj->qip", selected, design.vectors)
    return blocks.reshape(QUERY_COUNT * 8, SPIN8_BIVECTOR_DIM)


def information_matrix(design: SensorDesign, generators: torch.Tensor) -> torch.Tensor:
    jacobian = design_jacobian(design, generators)
    return jacobian.T @ jacobian


def information_metrics(
    design: SensorDesign, generators: torch.Tensor
) -> dict[str, object]:
    information = information_matrix(design, generators)
    eigenvalues = torch.linalg.eigvalsh(information)
    leading = float(eigenvalues[-1])
    tolerance = SPIN8_BIVECTOR_DIM * torch.finfo(information.dtype).eps * leading * 32
    positive = eigenvalues[eigenvalues > tolerance]
    rank = int(positive.numel())
    counts = torch.bincount(design.views, minlength=REPRESENTATION_COUNT)
    result: dict[str, object] = {
        "name": design.name,
        "views": [int(value) for value in design.views.tolist()],
        "view_names": [REPRESENTATION_NAMES[int(value)] for value in design.views],
        "vectors": design.vectors.detach().cpu().tolist(),
        "allocation": [int(value) for value in counts.tolist()],
        "rank": rank,
        "nullity": SPIN8_BIVECTOR_DIM - rank,
        "tolerance": tolerance,
        "minimum_positive_eigenvalue": float(positive[0]),
        "maximum_eigenvalue": leading,
        "condition_number_on_image": float(positive[-1] / positive[0]),
    }
    if rank == SPIN8_BIVECTOR_DIM:
        sign, logabsdet = torch.linalg.slogdet(information)
        result.update(
            {
                "log_determinant": float(logabsdet) if float(sign) > 0 else None,
                "trace_inverse": float(torch.trace(torch.linalg.inv(information))),
                "minimum_eigenvalue": float(eigenvalues[0]),
                "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
            }
        )
    else:
        result.update(
            {
                "log_determinant": None,
                "trace_inverse": None,
                "minimum_eigenvalue": float(eigenvalues[0]),
                "condition_number": None,
            }
        )
    return result


def action_information_matrix(
    design: SensorDesign, generators: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
    """Information after an unknown orthogonal action, one matrix per token."""

    base = design_jacobian(design, generators).reshape(QUERY_COUNT, 8, -1)
    selected = actions[:, design.views]
    transported = torch.einsum("tqij,qjp->tqip", selected, base)
    flat = transported.reshape(actions.shape[0], QUERY_COUNT * 8, -1)
    return flat.transpose(-1, -2) @ flat


def action_independence_audit(
    design: SensorDesign, generators: torch.Tensor, actions: torch.Tensor
) -> dict[str, float]:
    reference = information_matrix(design, generators)
    transported = action_information_matrix(design, generators, actions)
    reference_spectrum = torch.linalg.eigvalsh(reference)
    transported_spectrum = torch.linalg.eigvalsh(transported)
    return {
        "information_max_absolute_error": float(
            (transported - reference[None]).abs().max()
        ),
        "spectrum_max_absolute_error": float(
            (transported_spectrum - reference_spectrum[None]).abs().max()
        ),
    }


def soft_design_information(
    logits: torch.Tensor,
    vectors: torch.Tensor,
    generators: torch.Tensor,
    *,
    temperature: float,
    gumbel: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hard-forward, soft-backward information matrix for sensor learning."""

    normalized = F.normalize(vectors, dim=-1)
    scores = logits if gumbel is None else logits + gumbel
    soft = F.softmax(scores / temperature, dim=-1)
    indices = soft.argmax(dim=-1)
    hard = F.one_hot(indices, num_classes=REPRESENTATION_COUNT).to(soft.dtype)
    weights = hard + soft - soft.detach()
    blocks = torch.einsum("rpij,qrj->qrip", generators, normalized)
    selected = torch.einsum("qr,qrip->qip", weights, blocks)
    jacobian = selected.reshape(QUERY_COUNT * 8, SPIN8_BIVECTOR_DIM)
    return jacobian.T @ jacobian, indices


def sample_gumbel(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    uniform = torch.rand(shape, generator=generator, dtype=torch.float64)
    return -torch.log(-torch.log(uniform.clamp(1e-9, 1 - 1e-9)))


def learn_hard_sensor(
    seed: int,
    generators: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
) -> tuple[SensorDesign, dict[str, object]]:
    """Learn view choices and probe vectors with no diversity supervision."""

    cpu_generator = torch.Generator(device="cpu").manual_seed(82000 + seed)
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
    optimizer = torch.optim.Adam((logits, vectors), lr=learning_rate)
    trajectory = []
    log_steps = {0, 49, 199, 499, steps - 1}
    for step in range(steps):
        fraction = step / max(steps - 1, 1)
        temperature = 1.5 * (0.08 / 1.5) ** fraction
        gumbel = sample_gumbel(tuple(logits.shape), cpu_generator).to(generators.device)
        optimizer.zero_grad(set_to_none=True)
        information, indices = soft_design_information(
            logits,
            vectors,
            generators,
            temperature=temperature,
            gumbel=gumbel,
        )
        _, logdet = torch.linalg.slogdet(
            information
            + 1e-7 * torch.eye(28, dtype=information.dtype, device=information.device)
        )
        loss = -logdet
        loss.backward()
        optimizer.step()
        if step in log_steps:
            trajectory.append(
                {
                    "step": step + 1,
                    "temperature": temperature,
                    "regularized_logdet": float(logdet.detach()),
                    "sampled_allocation": [
                        int(value)
                        for value in torch.bincount(
                            indices.detach().cpu(), minlength=REPRESENTATION_COUNT
                        ).tolist()
                    ],
                }
            )

    hard_views = logits.detach().argmax(dim=-1)
    normalized = F.normalize(vectors.detach(), dim=-1)
    hard_vectors = normalized[
        torch.arange(QUERY_COUNT, device=generators.device), hard_views
    ]
    design = SensorDesign(hard_views, hard_vectors, "learned_hard_doptimal")
    return design, {
        "steps": steps,
        "learning_rate": learning_rate,
        "final_logits": logits.detach().cpu().tolist(),
        "trajectory": trajectory,
    }


def all_five_allocations() -> list[tuple[int, int, int]]:
    return [
        (first, second, QUERY_COUNT - first - second)
        for first in range(QUERY_COUNT + 1)
        for second in range(QUERY_COUNT - first + 1)
    ]


def views_from_allocation(allocation: tuple[int, int, int]) -> torch.Tensor:
    return torch.tensor(
        [
            representation
            for representation, count in enumerate(allocation)
            for _ in range(count)
        ],
        dtype=torch.long,
    )


def oracle_doptimal_sensor(
    seed: int,
    generators: torch.Tensor,
    *,
    steps: int,
    restarts: int,
) -> tuple[SensorDesign, dict[str, object], SensorDesign]:
    """Enumerate every allocation and optimize its continuous probe vectors."""

    allocations = all_five_allocations()
    candidate_views = []
    candidate_vectors = []
    for allocation_index, allocation in enumerate(allocations):
        views = views_from_allocation(allocation)
        for restart in range(restarts):
            generator = torch.Generator(device="cpu").manual_seed(
                83000 + 10000 * seed + 100 * allocation_index + restart
            )
            candidate_views.append(views)
            candidate_vectors.append(
                torch.randn(QUERY_COUNT, 8, generator=generator, dtype=torch.float64)
            )

    views_batch = torch.stack(candidate_views).to(generators.device)
    vectors = nn.Parameter(torch.stack(candidate_vectors).to(generators.device))
    selected_generators = generators[views_batch]
    identity = torch.eye(28, dtype=generators.dtype, device=generators.device)
    optimizer = torch.optim.Adam((vectors,), lr=4e-2)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        normalized = F.normalize(vectors, dim=-1)
        blocks = torch.einsum("cqpij,cqj->cqip", selected_generators, normalized)
        jacobian = blocks.reshape(-1, QUERY_COUNT * 8, SPIN8_BIVECTOR_DIM)
        information = jacobian.transpose(-1, -2) @ jacobian
        _, logdet = torch.linalg.slogdet(information + 1e-7 * identity)
        (-logdet.sum()).backward()
        optimizer.step()

    normalized = F.normalize(vectors.detach(), dim=-1)
    candidates = [
        SensorDesign(views_batch[index], normalized[index], "candidate")
        for index in range(views_batch.shape[0])
    ]
    rows = []
    best_design: SensorDesign | None = None
    best_logdet = -math.inf
    single_design: SensorDesign | None = None
    single_score = -math.inf
    for allocation_index, allocation in enumerate(allocations):
        allocation_best: SensorDesign | None = None
        allocation_score = -math.inf
        for restart in range(restarts):
            candidate = candidates[allocation_index * restarts + restart]
            metrics = information_metrics(candidate, generators)
            score = (
                float(metrics["log_determinant"])
                if metrics["log_determinant"] is not None
                else -math.inf
            )
            if allocation_best is None or score > allocation_score:
                allocation_score = score
                allocation_best = candidate
        assert allocation_best is not None
        metrics = information_metrics(allocation_best, generators)
        rows.append({"allocation": list(allocation), **metrics})
        if allocation_score > best_logdet:
            best_logdet = allocation_score
            best_design = allocation_best
        if allocation == (5, 0, 0):
            single_design = allocation_best
            single_score = allocation_score
    assert best_design is not None and single_design is not None
    return (
        SensorDesign(best_design.views, best_design.vectors, "oracle_doptimal"),
        {
            "steps_per_restart": steps,
            "restarts": restarts,
            "allocations": rows,
            "best_log_determinant": best_logdet,
            "single_view_log_determinant": (
                None if single_score == -math.inf else single_score
            ),
        },
        SensorDesign(
            single_design.views, single_design.vectors, "single_view_doptimal"
        ),
    )


def random_mixed_sensor(seed: int, device: torch.device) -> SensorDesign:
    generator = torch.Generator(device="cpu").manual_seed(84000 + seed)
    while True:
        views = torch.randint(0, 3, (QUERY_COUNT,), generator=generator)
        if torch.unique(views).numel() >= 2:
            break
    vectors = F.normalize(
        torch.randn(QUERY_COUNT, 8, generator=generator, dtype=torch.float64), dim=-1
    )
    return SensorDesign(views.to(device), vectors.to(device), "random_mixed")


def fixed_sensor(seed: int, device: torch.device) -> SensorDesign:
    views = torch.tensor([0, 1, 1, 1, 1], dtype=torch.long, device=device)
    vectors = canonical_random_vectors(85000 + seed).to(device)
    return SensorDesign(views, vectors, "fixed_1_4_0")


def observe_actions(actions: torch.Tensor, design: SensorDesign) -> torch.Tensor:
    selected = actions[:, design.views]
    return torch.einsum("tqij,qj->tqi", selected, design.vectors)


def fit_noisy_actions(
    target: torch.Tensor,
    design: SensorDesign,
    generators: torch.Tensor,
    *,
    seed: int,
    adam_steps: int,
    lbfgs_steps: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    cpu_generator = torch.Generator(device="cpu").manual_seed(86000 + seed)
    coordinates = nn.Parameter(
        (
            0.01
            * torch.randn(
                TOKEN_COUNT,
                SPIN8_BIVECTOR_DIM,
                generator=cpu_generator,
                dtype=torch.float64,
            )
        ).to(generators.device)
    )

    def actions() -> torch.Tensor:
        return spin8_actions(coordinates, generators)

    def objective() -> torch.Tensor:
        return F.mse_loss(observe_actions(actions(), design), target)

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
    trajectory.append({"stage": "lbfgs", "step": lbfgs_steps, "loss": final_loss})
    return final_actions, {
        "final_noisy_endpoint_mse": final_loss,
        "adam_steps": adam_steps,
        "lbfgs_steps": lbfgs_steps,
        "trajectory": trajectory,
    }


def recovery_report(
    actions: torch.Tensor,
    oracle: torch.Tensor,
    rho: torch.Tensor,
    design: SensorDesign,
    *,
    seed: int,
    fit: dict[str, object],
) -> dict[str, object]:
    metrics = one_step_metrics(actions, oracle, seed=87000 + seed)
    allocation = torch.bincount(design.views, minlength=3)
    least_observed = int(torch.argmin(allocation))
    commutator = vector_commutator(actions)
    oracle_commutator = vector_commutator(oracle)
    return {
        "fit": fit,
        "least_observed_representation": REPRESENTATION_NAMES[least_observed],
        "least_observed_one_step_mean_cosine": metrics[
            REPRESENTATION_NAMES[least_observed]
        ]["mean_cosine"],
        "one_step": metrics,
        "triality_equivariance_max_error": triality_residual(
            actions, rho, seed=88000 + seed
        ),
        "scan_parallel_recurrent_max_error": scan_parity(actions, seed=89000 + seed),
        "commutator_ratio_to_oracle": commutator / oracle_commutator,
    }


def run_seed(
    seed: int,
    *,
    generators: torch.Tensor,
    rho: torch.Tensor,
    policy_steps: int,
    policy_lr: float,
    oracle_steps: int,
    oracle_restarts: int,
    adam_steps: int,
    lbfgs_steps: int,
) -> dict[str, object]:
    learned, learned_training = learn_hard_sensor(
        seed, generators, steps=policy_steps, learning_rate=policy_lr
    )
    oracle, oracle_search, single = oracle_doptimal_sensor(
        seed,
        generators,
        steps=oracle_steps,
        restarts=oracle_restarts,
    )
    designs = {
        "learned_hard_doptimal": learned,
        "oracle_doptimal": oracle,
        "random_mixed": random_mixed_sensor(seed, generators.device),
        "fixed_1_4_0": fixed_sensor(seed, generators.device),
        "single_view_doptimal": single,
    }
    teacher = sample_teacher(seed=90000 + seed, generators=generators)
    noise_generator = torch.Generator(device="cpu").manual_seed(91000 + seed)
    matched_noise = (
        NOISE_STD
        * torch.randn(
            TOKEN_COUNT,
            QUERY_COUNT,
            8,
            generator=noise_generator,
            dtype=torch.float64,
        )
    ).to(generators.device)

    learned_actions: dict[str, torch.Tensor] = {}
    recovery = {}
    design_reports = {}
    independence = {}
    for offset, (name, design) in enumerate(designs.items()):
        design_reports[name] = information_metrics(design, generators)
        independence[name] = action_independence_audit(
            design, generators, teacher.actions
        )
        noisy_target = observe_actions(teacher.actions, design) + matched_noise
        actions, fit = fit_noisy_actions(
            noisy_target,
            design,
            generators,
            seed=100 * seed + offset,
            adam_steps=adam_steps,
            lbfgs_steps=lbfgs_steps,
        )
        learned_actions[name] = actions
        recovery[name] = recovery_report(
            actions, teacher.actions, rho, design, seed=seed, fit=fit
        )
    composition = composition_metrics(
        {**learned_actions, "oracle": teacher.actions},
        seed=92000 + seed,
        lengths=DENSE_LENGTHS,
    )

    learned_info = design_reports["learned_hard_doptimal"]
    oracle_info = design_reports["oracle_doptimal"]
    learned_logdet_gap = (
        float(oracle_info["log_determinant"]) - float(learned_info["log_determinant"])
        if learned_info["log_determinant"] is not None
        else math.inf
    )
    learned_trace_ratio = (
        float(learned_info["trace_inverse"]) / float(oracle_info["trace_inverse"])
        if learned_info["trace_inverse"] is not None
        else math.inf
    )
    learned_design_pass = (
        sum(value > 0 for value in learned_info["allocation"]) >= 2
        and int(learned_info["rank"]) == 28
        and learned_logdet_gap <= 0.10
        and learned_trace_ratio <= 1.10
    )
    learned_least = float(
        recovery["learned_hard_doptimal"]["least_observed_one_step_mean_cosine"]
    )
    random_least = float(
        recovery["random_mixed"]["least_observed_one_step_mean_cosine"]
    )
    learned_long = min(
        float(row["mean_cosine"])
        for row in composition["learned_hard_doptimal"]["2048"].values()
    )
    random_long = min(
        float(row["mean_cosine"])
        for row in composition["random_mixed"]["2048"].values()
    )
    oracle_long = min(
        float(row["mean_cosine"])
        for row in composition["oracle_doptimal"]["2048"].values()
    )
    learned_recovery = recovery["learned_hard_doptimal"]
    learned_norm_drift = max(
        float(row["maximum_absolute_log_norm"])
        for length in composition["learned_hard_doptimal"].values()
        for row in length.values()
    )
    noisy_recovery_pass = (
        learned_least > random_least
        and learned_long > random_long
        and learned_long >= oracle_long - 0.02
        and float(learned_recovery["triality_equivariance_max_error"]) < 1e-8
        and float(learned_recovery["scan_parallel_recurrent_max_error"]) < 1e-9
        and learned_norm_drift < 1e-5
    )
    return {
        "seed": seed,
        "noise_std": NOISE_STD,
        "learned_sensor_training": learned_training,
        "oracle_search": oracle_search,
        "designs": design_reports,
        "action_independence": independence,
        "recovery": recovery,
        "composition": composition,
        "learned_logdet_gap_to_oracle": learned_logdet_gap,
        "learned_trace_inverse_ratio_to_oracle": learned_trace_ratio,
        "learned_design_pass": learned_design_pass,
        "learned_least_observed_advantage": learned_least - random_least,
        "learned_length2048_advantage": learned_long - random_long,
        "learned_length2048_gap_to_oracle": oracle_long - learned_long,
        "noisy_recovery_pass": noisy_recovery_pass,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    independence_passes = sum(
        max(
            float(report["information_max_absolute_error"])
            for report in row["action_independence"].values()
        )
        < 1e-10
        and max(
            float(report["spectrum_max_absolute_error"])
            for report in row["action_independence"].values()
        )
        < 1e-10
        for row in rows
    )
    design_passes = sum(bool(row["learned_design_pass"]) for row in rows)
    noisy_passes = sum(bool(row["noisy_recovery_pass"]) for row in rows)
    structural_passes = sum(
        row["designs"]["single_view_doptimal"]["rank"] == 25
        and row["designs"]["single_view_doptimal"]["nullity"] == 3
        and row["designs"]["random_mixed"]["rank"] == 28
        and row["designs"]["fixed_1_4_0"]["rank"] == 28
        for row in rows
    )
    count = len(rows)
    return {
        "seed_count": count,
        "action_independence_pass_count": independence_passes,
        "learned_design_pass_count": design_passes,
        "noisy_recovery_pass_count": noisy_passes,
        "structural_control_pass_count": structural_passes,
        "action_independence_gate_passed": independence_passes == count,
        "learned_design_gate_passed": design_passes >= math.ceil(0.8 * count),
        "noisy_recovery_gate_passed": noisy_passes >= math.ceil(0.8 * count),
        "structural_control_gate_passed": structural_passes == count,
    }


def run(
    *,
    device: torch.device,
    seeds: tuple[int, ...],
    policy_steps: int,
    policy_lr: float,
    oracle_steps: int,
    oracle_restarts: int,
    adam_steps: int,
    lbfgs_steps: int,
) -> dict[str, object]:
    generators = torch_triality_generators(dtype=torch.float64, device=device)
    rho = triality_tensor(dtype=torch.float64, device=device)
    rows = [
        run_seed(
            seed,
            generators=generators,
            rho=rho,
            policy_steps=policy_steps,
            policy_lr=policy_lr,
            oracle_steps=oracle_steps,
            oracle_restarts=oracle_restarts,
            adam_steps=adam_steps,
            lbfgs_steps=lbfgs_steps,
        )
        for seed in seeds
    ]
    return {
        "experiment": "Spin8 active five-query triality sensing",
        "device": str(device),
        "dtype": str(generators.dtype),
        "seeds": list(seeds),
        "query_count": QUERY_COUNT,
        "noise_std": NOISE_STD,
        "policy_steps": policy_steps,
        "policy_learning_rate": policy_lr,
        "oracle_steps": oracle_steps,
        "oracle_restarts": oracle_restarts,
        "action_adam_steps": adam_steps,
        "action_lbfgs_steps": lbfgs_steps,
        "dense_lengths": list(DENSE_LENGTHS),
        "results": rows,
        "summary": summarize(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seeds", default="10,11,12,13,14,15,16,17,18,19")
    parser.add_argument("--policy-steps", type=int, default=1500)
    parser.add_argument("--policy-lr", type=float, default=2e-2)
    parser.add_argument("--oracle-steps", type=int, default=600)
    parser.add_argument("--oracle-restarts", type=int, default=4)
    parser.add_argument("--adam-steps", type=int, default=1500)
    parser.add_argument("--lbfgs-steps", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        device=torch.device(args.device),
        seeds=tuple(int(value) for value in args.seeds.split(",") if value),
        policy_steps=args.policy_steps,
        policy_lr=args.policy_lr,
        oracle_steps=args.oracle_steps,
        oracle_restarts=args.oracle_restarts,
        adam_steps=args.adam_steps,
        lbfgs_steps=args.lbfgs_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
