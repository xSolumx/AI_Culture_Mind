"""Sharp five-probe identifiability gate for shared Spin(8) triality actions.

This harness deliberately observes transformed generic states rather than
matrix columns.  It separates three questions:

* differential identifiability of one shared 28-dimensional action family;
* constructive non-identifiability when one probe is removed; and
* whether optimization recovers the hidden chiral action from the minimal
  identifiable evidence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from spin8_blind_shared_action import (
    TOKEN_COUNT,
    composition_metrics,
    one_step_metrics,
    sample_teacher,
    scan_parity,
    triality_residual,
    vector_commutator,
)
from spin8_triality import SPIN8_BIVECTOR_DIM, spin8_actions, torch_triality_generators
from spin8_triality_lift import triality_tensor

Allocation = tuple[int, int, int]
FIVE_MIXED: Allocation = (1, 4, 0)
FOUR_MIXED: Allocation = (1, 3, 0)
FIVE_SINGLE: Allocation = (5, 0, 0)
DENSE_LENGTHS = (16, 32, 64, 128, 256, 512, 1024, 2048)


@dataclass(frozen=True)
class ProbeFamily:
    """Canonical orthonormal probe bases, one complete basis per triality view."""

    bases: torch.Tensor  # (3, 8, 8), columns are orthonormal probes
    seed: int


def make_probe_family(
    seed: int, *, dtype: torch.dtype, device: torch.device
) -> ProbeFamily:
    """Generate probes on CPU so device choice cannot alter the evidence."""

    generator = torch.Generator(device="cpu").manual_seed(71000 + seed)
    matrices = torch.randn(3, 8, 8, generator=generator, dtype=torch.float64)
    bases = []
    for representation in range(3):
        q, r = torch.linalg.qr(matrices[representation])
        # Remove QR's sign gauge for reproducible columns.
        sign = torch.where(torch.diagonal(r) < 0, -1.0, 1.0)
        bases.append(q * sign[None, :])
    return ProbeFamily(torch.stack(bases).to(dtype=dtype, device=device), seed)


def apply_probes(
    actions: torch.Tensor, probes: ProbeFamily, allocation: Allocation
) -> torch.Tensor:
    """Flatten the supplied state/action pairs for all tokens."""

    pieces = []
    for representation, count in enumerate(allocation):
        if count:
            basis = probes.bases[representation, :, :count]
            transformed = torch.einsum("tij,jp->tip", actions[:, representation], basis)
            pieces.append(transformed.reshape(actions.shape[0], -1))
    if not pieces:
        return actions.new_zeros(actions.shape[0], 0)
    return torch.cat(pieces, dim=-1)


def shared_observation_jacobian(
    coefficients: torch.Tensor,
    generators: torch.Tensor,
    probes: ProbeFamily,
    allocation: Allocation,
) -> torch.Tensor:
    """Jacobian for one token's shared action observation."""

    point = coefficients.detach().clone().requires_grad_(True)

    def observation(coordinate: torch.Tensor) -> torch.Tensor:
        return apply_probes(
            spin8_actions(coordinate[None], generators), probes, allocation
        ).reshape(-1)

    return torch.autograd.functional.jacobian(observation, point, vectorize=True)


def independent_observation_jacobian(
    coordinates: torch.Tensor,
    generators: torch.Tensor,
    probes: ProbeFamily,
    allocation: Allocation,
) -> torch.Tensor:
    """Jacobian for three unrelated SO(8) charts, flattened to 84 parameters."""

    point = coordinates.detach().clone().requires_grad_(True)

    def observation(flat: torch.Tensor) -> torch.Tensor:
        coordinate = flat.reshape(3, SPIN8_BIVECTOR_DIM)
        tangent = torch.einsum("rp,rpij->rij", coordinate, generators).contiguous()
        actions = torch.matrix_exp(tangent)[None]
        return apply_probes(actions, probes, allocation).reshape(-1)

    return torch.autograd.functional.jacobian(
        observation, point.reshape(-1), vectorize=True
    )


def numerical_rank(jacobian: torch.Tensor) -> dict[str, object]:
    singular = torch.linalg.svdvals(jacobian)
    leading = float(singular[0]) if singular.numel() else 0.0
    tolerance = max(jacobian.shape) * torch.finfo(jacobian.dtype).eps * leading * 32
    nonzero = singular[singular > tolerance]
    return {
        "rank": int(nonzero.numel()),
        "nullity": int(jacobian.shape[1] - nonzero.numel()),
        "tolerance": tolerance,
        "smallest_nonzero_singular_value": (
            float(nonzero[-1]) if nonzero.numel() else 0.0
        ),
        "largest_zero_singular_value": (
            float(singular[nonzero.numel()])
            if nonzero.numel() < singular.numel()
            else 0.0
        ),
    }


def allocation_rank_audit(
    generators: torch.Tensor, probes: ProbeFamily
) -> dict[str, object]:
    """Audit every allocation of one through five probes at the identity."""

    zero = torch.zeros(
        SPIN8_BIVECTOR_DIM, dtype=generators.dtype, device=generators.device
    )
    rows = []
    for total in range(1, 6):
        for first in range(total + 1):
            for second in range(total - first + 1):
                allocation = (first, second, total - first - second)
                if max(allocation) > 8:
                    continue
                rank = numerical_rank(
                    shared_observation_jacobian(zero, generators, probes, allocation)
                )
                rows.append(
                    {
                        "allocation": list(allocation),
                        "total": total,
                        "view_count": sum(value > 0 for value in allocation),
                        **rank,
                    }
                )
    envelopes = []
    for total in range(1, 6):
        single = [
            row["rank"]
            for row in rows
            if row["total"] == total and row["view_count"] == 1
        ]
        mixed = [
            row["rank"]
            for row in rows
            if row["total"] == total and row["view_count"] >= 2
        ]
        envelopes.append(
            {
                "total": total,
                "single_view_ranks": sorted(set(single)),
                "mixed_view_ranks": sorted(set(mixed)),
            }
        )
    return {"allocations": rows, "envelopes": envelopes}


def exact_four_probe_witness(
    teacher_actions: torch.Tensor,
    generators: torch.Tensor,
    probes: ProbeFamily,
    rho: torch.Tensor,
    *,
    seed: int,
    tangent_size: float = 1.0,
) -> dict[str, object]:
    """Construct a shared action invisible to four probes but active on S-."""

    zero = torch.zeros(
        SPIN8_BIVECTOR_DIM, dtype=generators.dtype, device=generators.device
    )
    jacobian = shared_observation_jacobian(zero, generators, probes, FOUR_MIXED)
    _, _, vh = torch.linalg.svd(jacobian, full_matrices=True)
    rank = numerical_rank(jacobian)
    tangent = vh[-1]
    tangent = tangent_size * tangent / tangent.norm()
    stabilizer = spin8_actions(tangent, generators)
    alternative = torch.einsum("trij,rjk->trik", teacher_actions, stabilizer)
    visible_error = float(
        (
            apply_probes(alternative, probes, FOUR_MIXED)
            - apply_probes(teacher_actions, probes, FOUR_MIXED)
        )
        .abs()
        .max()
    )
    metrics = one_step_metrics(
        alternative, teacher_actions, seed=72000 + seed, examples=2048
    )
    return {
        "jacobian": rank,
        "tangent_norm": float(tangent.norm()),
        "infinitesimal_visible_max_error": float((jacobian @ tangent).abs().max()),
        "visible_endpoint_max_error": visible_error,
        "hidden_negative_mean_cosine": metrics["negative"]["mean_cosine"],
        "hidden_negative_minimum_cosine": metrics["negative"]["minimum_cosine"],
        "teacher_triality_max_error": triality_residual(
            teacher_actions, rho, seed=73000 + seed
        ),
        "alternative_triality_max_error": triality_residual(
            alternative, rho, seed=74000 + seed
        ),
    }


def independent_actions(
    coordinates: torch.Tensor, generators: torch.Tensor
) -> torch.Tensor:
    tangent = torch.einsum("trp,rpij->trij", coordinates, generators).contiguous()
    return torch.matrix_exp(tangent)


def fit_family(
    target: torch.Tensor,
    probes: ProbeFamily,
    allocation: Allocation,
    generators: torch.Tensor,
    *,
    seed: int,
    shared: bool,
    adam_steps: int,
    lbfgs_steps: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Fit either a shared or three-independent tangent family."""

    cpu_generator = torch.Generator(device="cpu").manual_seed(75000 + seed)
    shape = (
        (TOKEN_COUNT, SPIN8_BIVECTOR_DIM)
        if shared
        else (TOKEN_COUNT, 3, SPIN8_BIVECTOR_DIM)
    )
    initial = 0.01 * torch.randn(shape, generator=cpu_generator, dtype=torch.float64)
    coordinate = nn.Parameter(
        initial.to(device=generators.device, dtype=generators.dtype)
    )

    def actions() -> torch.Tensor:
        if shared:
            return spin8_actions(coordinate, generators)
        return independent_actions(coordinate, generators)

    def objective() -> torch.Tensor:
        return F.mse_loss(apply_probes(actions(), probes, allocation), target)

    trajectory = []
    adam = torch.optim.Adam((coordinate,), lr=3e-2)
    log_steps = {0, 49, 199, 499, adam_steps - 1}
    for step in range(adam_steps):
        adam.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        adam.step()
        if step in log_steps:
            trajectory.append(
                {"stage": "adam", "step": step + 1, "loss": float(loss.detach())}
            )

    lbfgs = torch.optim.LBFGS(
        (coordinate,),
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
    learned = actions().detach()
    final_loss = float(objective().detach())
    trajectory.append({"stage": "lbfgs", "step": lbfgs_steps, "loss": final_loss})
    return learned, {
        "shared": shared,
        "allocation": list(allocation),
        "parameter_count": coordinate.numel(),
        "adam_steps": adam_steps,
        "lbfgs_steps": lbfgs_steps,
        "trajectory": trajectory,
        "final_visible_mse": final_loss,
    }


def family_report(
    actions: torch.Tensor,
    oracle: torch.Tensor,
    rho: torch.Tensor,
    *,
    seed: int,
    visible_mse: float,
) -> dict[str, object]:
    oracle_commutator = vector_commutator(oracle)
    commutator = vector_commutator(actions)
    return {
        "visible_mse": visible_mse,
        "one_step": one_step_metrics(actions, oracle, seed=76000 + seed),
        "triality_equivariance_max_error": triality_residual(
            actions, rho, seed=77000 + seed
        ),
        "vector_commutator": commutator,
        "commutator_ratio_to_oracle": commutator / oracle_commutator,
        "scan_parallel_recurrent_max_error": scan_parity(actions, seed=78000 + seed),
    }


def minimum_composition_cosine(
    composition: dict[str, object], family: str, length: int
) -> float:
    return min(
        float(row["mean_cosine"]) for row in composition[family][str(length)].values()
    )


def shared_five_pass(report: dict[str, object], composition: dict[str, object]) -> bool:
    if float(report["visible_mse"]) >= 1e-8:
        return False
    if float(report["one_step"]["negative"]["mean_cosine"]) < 0.9999:
        return False
    if float(report["triality_equivariance_max_error"]) >= 1e-8:
        return False
    if float(report["commutator_ratio_to_oracle"]) < 0.90:
        return False
    if float(report["scan_parallel_recurrent_max_error"]) >= 1e-9:
        return False
    for length in DENSE_LENGTHS:
        for row in composition[str(length)].values():
            if float(row["mean_cosine"]) < 0.999:
                return False
            if float(row["maximum_absolute_log_norm"]) >= 1e-5:
                return False
    return True


def run_seed(
    seed: int,
    *,
    generators: torch.Tensor,
    rho: torch.Tensor,
    adam_steps: int,
    lbfgs_steps: int,
    include_phase_diagram: bool,
) -> dict[str, object]:
    probes = make_probe_family(seed, dtype=generators.dtype, device=generators.device)
    teacher = sample_teacher(seed=seed, generators=generators)
    targets = {
        "shared_five_mixed": apply_probes(teacher.actions, probes, FIVE_MIXED),
        "shared_four_mixed": apply_probes(teacher.actions, probes, FOUR_MIXED),
        "shared_five_single": apply_probes(teacher.actions, probes, FIVE_SINGLE),
        "independent_five_mixed": apply_probes(teacher.actions, probes, FIVE_MIXED),
    }
    specifications = {
        "shared_five_mixed": (FIVE_MIXED, True),
        "shared_four_mixed": (FOUR_MIXED, True),
        "shared_five_single": (FIVE_SINGLE, True),
        "independent_five_mixed": (FIVE_MIXED, False),
    }
    families: dict[str, torch.Tensor] = {}
    training: dict[str, object] = {}
    reports: dict[str, object] = {}
    for offset, (name, (allocation, shared)) in enumerate(specifications.items()):
        action, train = fit_family(
            targets[name],
            probes,
            allocation,
            generators,
            seed=100 * seed + offset,
            shared=shared,
            adam_steps=adam_steps,
            lbfgs_steps=lbfgs_steps,
        )
        families[name] = action
        training[name] = train
        reports[name] = family_report(
            action,
            teacher.actions,
            rho,
            seed=seed,
            visible_mse=float(train["final_visible_mse"]),
        )
    families["oracle"] = teacher.actions
    composition = composition_metrics(
        {**families, "oracle": teacher.actions},
        seed=79000 + seed,
        lengths=DENSE_LENGTHS,
    )
    phase = allocation_rank_audit(generators, probes) if include_phase_diagram else None

    zero = torch.zeros(
        SPIN8_BIVECTOR_DIM, dtype=generators.dtype, device=generators.device
    )
    independent_zero = torch.zeros(
        3, SPIN8_BIVECTOR_DIM, dtype=generators.dtype, device=generators.device
    )
    frozen_ranks = {
        "shared_four_mixed": numerical_rank(
            shared_observation_jacobian(zero, generators, probes, FOUR_MIXED)
        ),
        "shared_five_mixed": numerical_rank(
            shared_observation_jacobian(zero, generators, probes, FIVE_MIXED)
        ),
        "shared_five_single": numerical_rank(
            shared_observation_jacobian(zero, generators, probes, FIVE_SINGLE)
        ),
        "independent_five_mixed": numerical_rank(
            independent_observation_jacobian(
                independent_zero, generators, probes, FIVE_MIXED
            )
        ),
    }
    witness = exact_four_probe_witness(
        teacher.actions, generators, probes, rho, seed=seed
    )
    five_pass = shared_five_pass(
        reports["shared_five_mixed"], composition["shared_five_mixed"]
    )
    strongest_control_hidden = max(
        float(reports[name]["one_step"]["negative"]["mean_cosine"])
        for name in (
            "shared_four_mixed",
            "shared_five_single",
            "independent_five_mixed",
        )
    )
    strongest_control_long = max(
        minimum_composition_cosine(composition, name, 2048)
        for name in (
            "shared_four_mixed",
            "shared_five_single",
            "independent_five_mixed",
        )
    )
    hidden_margin = (
        float(reports["shared_five_mixed"]["one_step"]["negative"]["mean_cosine"])
        - strongest_control_hidden
    )
    long_margin = (
        minimum_composition_cosine(composition, "shared_five_mixed", 2048)
        - strongest_control_long
    )
    return {
        "seed": seed,
        "teacher_resamples": teacher.resamples,
        "probe_orthogonality_max_error": float(
            (
                probes.bases.transpose(-1, -2) @ probes.bases
                - torch.eye(8, dtype=generators.dtype, device=generators.device)
            )
            .abs()
            .max()
        ),
        "frozen_ranks": frozen_ranks,
        "phase_diagram": phase,
        "four_probe_witness": witness,
        "training": training,
        "reports": reports,
        "composition": composition,
        "shared_five_pass": five_pass,
        "hidden_negative_margin_over_best_control": hidden_margin,
        "length2048_margin_over_best_control": long_margin,
        "causal_separation_pass": hidden_margin >= 0.05 and long_margin >= 0.05,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    rank_passes = 0
    witness_passes = 0
    fit_passes = 0
    shared_passes = 0
    causal_passes = 0
    for row in rows:
        ranks = row["frozen_ranks"]
        rank_passes += (
            ranks["shared_four_mixed"]["rank"] == 25
            and ranks["shared_four_mixed"]["nullity"] == 3
            and ranks["shared_five_mixed"]["rank"] == 28
            and ranks["shared_five_mixed"]["nullity"] == 0
            and ranks["shared_five_single"]["rank"] == 25
            and ranks["shared_five_single"]["nullity"] == 3
            and ranks["independent_five_mixed"]["rank"] == 29
            and ranks["independent_five_mixed"]["nullity"] == 55
        )
        witness = row["four_probe_witness"]
        witness_passes += (
            float(witness["visible_endpoint_max_error"]) < 1e-9
            and float(witness["hidden_negative_mean_cosine"]) < 0.99
            and float(witness["teacher_triality_max_error"]) < 1e-9
            and float(witness["alternative_triality_max_error"]) < 1e-9
        )
        fit_passes += all(
            float(train["final_visible_mse"]) < 1e-8
            for train in row["training"].values()
        )
        shared_passes += bool(row["shared_five_pass"])
        causal_passes += bool(row["causal_separation_pass"])
    count = len(rows)
    return {
        "seed_count": count,
        "rank_gate_pass_count": rank_passes,
        "four_probe_witness_pass_count": witness_passes,
        "all_controls_fit_pass_count": fit_passes,
        "shared_five_completion_pass_count": shared_passes,
        "causal_separation_pass_count": causal_passes,
        "theorem_gate_passed": rank_passes == count and witness_passes == count,
        "optimization_gate_passed": shared_passes >= 8 and fit_passes == count,
        "causal_gate_passed": causal_passes >= 8,
    }


def run(
    *,
    device: torch.device,
    seeds: tuple[int, ...],
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
            adam_steps=adam_steps,
            lbfgs_steps=lbfgs_steps,
            include_phase_diagram=index == 0,
        )
        for index, seed in enumerate(seeds)
    ]
    return {
        "experiment": "Spin8 five-probe triality identifiability",
        "device": str(device),
        "dtype": str(generators.dtype),
        "seeds": list(seeds),
        "adam_steps": adam_steps,
        "lbfgs_steps": lbfgs_steps,
        "dense_lengths": list(DENSE_LENGTHS),
        "results": rows,
        "summary": summarize(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--adam-steps", type=int, default=1500)
    parser.add_argument("--lbfgs-steps", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        device=torch.device(args.device),
        seeds=tuple(int(value) for value in args.seeds.split(",") if value),
        adam_steps=args.adam_steps,
        lbfgs_steps=args.lbfgs_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
