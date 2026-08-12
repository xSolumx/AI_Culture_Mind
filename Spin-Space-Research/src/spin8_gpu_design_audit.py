"""CUDA falsifiers for continuous Spin(8) five-query sensor geometry.

This program deliberately cannot promote a theorem.  It uses the GPU for four
numerical jobs that are complementary to the exact CPU certificates:

* dense random interior search over every five-query view allocation;
* gradient refinement of possible determinant challengers;
* continuous Kiefer--Wolfowitz sensitivity mapping and fixed-support
  reweighting;
* Monte Carlo query noise and float32/float64 roundoff profiling.

All seeds, batch sizes, tolerances, hardware metadata, and peak CUDA memory are
stored in the output artifact.  Any apparent challenger must be rationalized
and checked by exact arithmetic before it changes theorem status.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import psutil
import torch
from torch import nn
from torch.nn import functional as F

from spin8_active_sensing import all_five_allocations, views_from_allocation
from spin8_resource_limits import constrain_current_process
from spin8_triality import torch_triality_generators

QUERY_COUNT = 5
STATE_DIMENSION = 8
ACTION_DIMENSION = 28
BALANCED_DETERMINANT = 81.0 / 1024.0
BALANCED_LOGDETERMINANT = math.log(BALANCED_DETERMINANT)


def _balanced_design(
    device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    basis = torch.eye(STATE_DIMENSION, dtype=dtype, device=device)
    views = torch.tensor([0, 1, 1, 2, 2], dtype=torch.long, device=device)
    vectors = torch.stack((basis[0], basis[0], basis[1], basis[2], basis[4]))
    return views, vectors


def _information_batch(
    generators: torch.Tensor, views: torch.Tensor, vectors: torch.Tensor
) -> torch.Tensor:
    """Return information matrices for ``vectors`` shaped ``(B,5,8)``."""

    if views.ndim == 1:
        selected = generators[views]
        blocks = torch.einsum("qpij,bqj->bqip", selected, vectors)
    else:
        selected = generators[views]
        blocks = torch.einsum("bqpij,bqj->bqip", selected, vectors)
    jacobian = blocks.reshape(vectors.shape[0], QUERY_COUNT * STATE_DIMENSION, -1)
    return jacobian.transpose(-1, -2) @ jacobian


def _summary(values: torch.Tensor) -> dict[str, float]:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {
            name: float("nan")
            for name in ("minimum", "q01", "median", "q99", "maximum")
        }
    quantiles = torch.quantile(
        finite,
        torch.tensor([0.01, 0.5, 0.99], dtype=finite.dtype, device=finite.device),
    )
    return {
        "minimum": float(finite.min()),
        "q01": float(quantiles[0]),
        "median": float(quantiles[1]),
        "q99": float(quantiles[2]),
        "maximum": float(finite.max()),
    }


def dense_interior_falsifier(
    generators: torch.Tensor,
    random: torch.Generator,
    *,
    samples_per_allocation: int,
    batch_size: int,
) -> dict[str, object]:
    rows = []
    best = -math.inf
    challenger_count = 0
    for allocation in all_five_allocations():
        views = views_from_allocation(allocation).to(generators.device)
        allocation_best = -math.inf
        full_rank = 0
        remaining = samples_per_allocation
        while remaining:
            count = min(batch_size, remaining)
            vectors = F.normalize(
                torch.randn(
                    count,
                    QUERY_COUNT,
                    STATE_DIMENSION,
                    generator=random,
                    dtype=generators.dtype,
                    device=generators.device,
                ),
                dim=-1,
            )
            information = _information_batch(generators, views, vectors)
            signs, logdet = torch.linalg.slogdet(information)
            valid = signs > 0
            full_rank += int(valid.sum())
            if valid.any():
                values = logdet[valid]
                allocation_best = max(allocation_best, float(values.max()))
                challenger_count += int((values > BALANCED_LOGDETERMINANT + 1e-9).sum())
            remaining -= count
        best = max(best, allocation_best)
        rows.append(
            {
                "allocation": list(allocation),
                "samples": samples_per_allocation,
                "positive_determinant_count": full_rank,
                "best_log_determinant": (
                    allocation_best if math.isfinite(allocation_best) else None
                ),
            }
        )
    return {
        "total_samples": samples_per_allocation * len(rows),
        "samples_per_allocation": samples_per_allocation,
        "batch_size": batch_size,
        "target_log_determinant": BALANCED_LOGDETERMINANT,
        "best_observed_log_determinant": best,
        "candidate_above_target_plus_1e_minus_9": challenger_count,
        "allocations": rows,
        "status": "falsification evidence only",
    }


def gradient_challenger_search(
    generators: torch.Tensor,
    random: torch.Generator,
    *,
    restarts_per_allocation: int,
    steps: int,
) -> dict[str, object]:
    allocations = all_five_allocations()
    view_rows = [views_from_allocation(row) for row in allocations]
    views = torch.cat(
        [row.repeat(restarts_per_allocation, 1) for row in view_rows], dim=0
    ).to(generators.device)
    vectors = nn.Parameter(
        torch.randn(
            views.shape[0],
            QUERY_COUNT,
            STATE_DIMENSION,
            generator=random,
            dtype=generators.dtype,
            device=generators.device,
        )
    )
    optimizer = torch.optim.Adam((vectors,), lr=4e-2)
    identity = torch.eye(
        ACTION_DIMENSION, dtype=generators.dtype, device=generators.device
    )
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        information = _information_batch(
            generators, views, F.normalize(vectors, dim=-1)
        )
        _, regularized = torch.linalg.slogdet(information + 1e-10 * identity)
        (-regularized.mean()).backward()
        optimizer.step()

    with torch.no_grad():
        information = _information_batch(
            generators, views, F.normalize(vectors, dim=-1)
        )
        eigenvalues = torch.linalg.eigvalsh(information)
        signs, logdet = torch.linalg.slogdet(information)
        valid = (signs > 0) & (eigenvalues[:, 0] > 1e-9)
        rows = []
        best = -math.inf
        challenger_count = 0
        for allocation_index, allocation in enumerate(allocations):
            start = allocation_index * restarts_per_allocation
            stop = start + restarts_per_allocation
            mask = valid[start:stop]
            values = logdet[start:stop][mask]
            allocation_best = float(values.max()) if values.numel() else None
            if allocation_best is not None:
                best = max(best, allocation_best)
                challenger_count += int((values > BALANCED_LOGDETERMINANT + 1e-8).sum())
            rows.append(
                {
                    "allocation": list(allocation),
                    "full_rank_restarts": int(mask.sum()),
                    "best_log_determinant": allocation_best,
                }
            )
    return {
        "steps": steps,
        "restarts_per_allocation": restarts_per_allocation,
        "target_log_determinant": BALANCED_LOGDETERMINANT,
        "best_refined_log_determinant": best,
        "candidate_above_target_plus_1e_minus_8": challenger_count,
        "allocations": rows,
        "status": "falsification evidence only",
    }


def sensitivity_and_reweighting(
    generators: torch.Tensor,
    random: torch.Generator,
    *,
    sensitivity_samples: int,
    reweight_steps: int,
) -> dict[str, object]:
    views, vectors = _balanced_design(generators.device, generators.dtype)
    information = _information_batch(generators, views, vectors[None])[0]
    inverse = torch.linalg.inv(information)
    sensitivity_matrices = torch.stack(
        [
            5
            * torch.einsum(
                "pq,pij,qik->jk", inverse, generators[view], generators[view]
            )
            for view in range(3)
        ]
    )
    rows = []
    for view in range(3):
        samples = F.normalize(
            torch.randn(
                sensitivity_samples,
                STATE_DIMENSION,
                generator=random,
                dtype=generators.dtype,
                device=generators.device,
            ),
            dim=-1,
        )
        values = torch.einsum(
            "bi,ij,bj->b", samples, sensitivity_matrices[view], samples
        )
        probes = nn.Parameter(
            torch.randn(
                64,
                STATE_DIMENSION,
                generator=random,
                dtype=generators.dtype,
                device=generators.device,
            )
        )
        optimizer = torch.optim.Adam((probes,), lr=8e-2)
        for _ in range(150):
            optimizer.zero_grad(set_to_none=True)
            normalized = F.normalize(probes, dim=-1)
            objective = torch.einsum(
                "bi,ij,bj->b",
                normalized,
                sensitivity_matrices[view],
                normalized,
            )
            (-objective.mean()).backward()
            optimizer.step()
        normalized = F.normalize(probes.detach(), dim=-1)
        optimized = torch.einsum(
            "bi,ij,bj->b",
            normalized,
            sensitivity_matrices[view],
            normalized,
        )
        spectrum = torch.linalg.eigvalsh(sensitivity_matrices[view])
        rows.append(
            {
                "view": view,
                "random_sample_count": sensitivity_samples,
                "random_sensitivity": _summary(values),
                "gradient_maximum": float(optimized.max()),
                "matrix_minimum_eigenvalue": float(spectrum[0]),
                "matrix_maximum_eigenvalue": float(spectrum[-1]),
            }
        )

    projectors = []
    for index in range(QUERY_COUNT):
        jacobian = torch.einsum("pij,j->ip", generators[views[index]], vectors[index])
        projectors.append(jacobian.T @ jacobian)
    projectors = torch.stack(projectors)
    logits = nn.Parameter(
        torch.zeros(QUERY_COUNT, dtype=generators.dtype, device=generators.device)
    )
    optimizer = torch.optim.Adam((logits,), lr=5e-2)
    for _ in range(reweight_steps):
        optimizer.zero_grad(set_to_none=True)
        weights = 5 * torch.softmax(logits, dim=0)
        weighted = torch.einsum("q,qij->ij", weights, projectors)
        _, logdet = torch.linalg.slogdet(weighted)
        (-logdet).backward()
        optimizer.step()
    weights = 5 * torch.softmax(logits.detach(), dim=0)
    weighted = torch.einsum("q,qij->ij", weights, projectors)
    _, reweighted_logdet = torch.linalg.slogdet(weighted)
    exact_alpha = (125 + 5 * math.sqrt(2977)) / 392
    exact_beta = (5 - exact_alpha) / 4
    return {
        "normalized_balanced_design": "M=I_sum/5",
        "kiefer_wolfowitz_parameter_dimension": ACTION_DIMENSION,
        "sensitivity_maps": rows,
        "global_sampled_or_optimized_maximum": max(
            max(
                row["random_sensitivity"]["maximum"],
                row["gradient_maximum"],
                row["matrix_maximum_eigenvalue"],
            )
            for row in rows
        ),
        "fixed_support_gradient_reweighting": {
            "steps": reweight_steps,
            "weights_sum": float(weights.sum()),
            "learned_weights": [float(value) for value in weights],
            "log_determinant": float(reweighted_logdet),
            "exact_optimum_alpha": exact_alpha,
            "exact_optimum_beta": exact_beta,
            "maximum_weight_error_from_exact_symmetric_solution": max(
                abs(float(weights[0]) - exact_alpha),
                *(abs(float(value) - exact_beta) for value in weights[1:]),
            ),
        },
        "status": "numerical map checked against the exact sensitivity theorem",
    }


def noise_profile(
    generators: torch.Tensor,
    random: torch.Generator,
    *,
    samples: int,
) -> dict[str, object]:
    views, balanced = _balanced_design(generators.device, generators.dtype)
    rows = []
    target = BALANCED_LOGDETERMINANT
    for sigma in (0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 5e-2):
        noise = torch.randn(
            samples,
            QUERY_COUNT,
            STATE_DIMENSION,
            generator=random,
            dtype=generators.dtype,
            device=generators.device,
        )
        perturbed = F.normalize(balanced[None] + sigma * noise, dim=-1)
        information64 = _information_batch(generators, views, perturbed)
        eigenvalues64 = torch.linalg.eigvalsh(information64)
        _, logdet64 = torch.linalg.slogdet(information64)

        information32 = _information_batch(generators.float(), views, perturbed.float())
        _, logdet32 = torch.linalg.slogdet(information32)
        condition64 = eigenvalues64[:, -1] / eigenvalues64[:, 0]
        rows.append(
            {
                "noise_standard_deviation": sigma,
                "samples": samples,
                "float64_log_determinant": _summary(logdet64),
                "float64_minimum_eigenvalue": _summary(eigenvalues64[:, 0]),
                "float64_condition_number": _summary(condition64),
                "samples_above_exact_local_optimum_plus_1e_minus_9": int(
                    (logdet64 > target + 1e-9).sum()
                ),
                "float32_minus_float64_logdet": _summary(logdet32.double() - logdet64),
                "maximum_absolute_float32_logdet_error": float(
                    (logdet32.double() - logdet64).abs().max()
                ),
            }
        )
    return {
        "interpretation": (
            "Gaussian query perturbations test conditioning and numerical "
            "roundoff near the exact local optimum. They do not model every "
            "physical hardware fault and cannot certify a theorem."
        ),
        "rows": rows,
    }


def run(
    *,
    seed: int,
    workers: int,
    samples_per_allocation: int,
    batch_size: int,
    restarts_per_allocation: int,
    optimization_steps: int,
    sensitivity_samples: int,
    reweight_steps: int,
    noise_samples: int,
) -> dict[str, object]:
    affinity = constrain_current_process(workers)
    torch.set_num_threads(workers)
    requested_interop = min(2, workers)
    if torch.get_num_interop_threads() != requested_interop:
        torch.set_num_interop_threads(requested_interop)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this numerical falsifier")
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(seed)
    random = torch.Generator(device=device).manual_seed(seed)
    torch.cuda.reset_peak_memory_stats(device)
    generators = torch_triality_generators(dtype=torch.float64, device=device)

    dense = dense_interior_falsifier(
        generators,
        random,
        samples_per_allocation=samples_per_allocation,
        batch_size=batch_size,
    )
    optimized = gradient_challenger_search(
        generators,
        random,
        restarts_per_allocation=restarts_per_allocation,
        steps=optimization_steps,
    )
    sensitivity = sensitivity_and_reweighting(
        generators,
        random,
        sensitivity_samples=sensitivity_samples,
        reweight_steps=reweight_steps,
    )
    noise = noise_profile(generators, random, samples=noise_samples)
    torch.cuda.synchronize(device)

    pass_conditions = {
        "dense_sweep_found_no_target_violation": dense[
            "candidate_above_target_plus_1e_minus_9"
        ]
        == 0,
        "gradient_search_found_no_target_violation": optimized[
            "candidate_above_target_plus_1e_minus_8"
        ]
        == 0,
        "sensitivity_recovers_exact_maximum_75": abs(
            sensitivity["global_sampled_or_optimized_maximum"] - 75.0
        )
        < 1e-8,
        "reweighting_recovers_exact_fixed_support_optimum": sensitivity[
            "fixed_support_gradient_reweighting"
        ]["maximum_weight_error_from_exact_symmetric_solution"]
        < 1e-5,
        "local_noise_found_no_uphill_sample": all(
            row["samples_above_exact_local_optimum_plus_1e_minus_9"] == 0
            for row in noise["rows"]
        ),
    }
    return {
        "experiment": "CUDA continuous Spin8 design falsification and sensitivity audit",
        "seed": seed,
        "hardware": {
            "platform": platform.platform(),
            "cpu_affinity": affinity,
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "system_ram_gib": psutil.virtual_memory().total / 2**30,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "gpu_total_memory_mib": torch.cuda.get_device_properties(
                device
            ).total_memory
            / 2**20,
            "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "cpu_worker_cap": workers,
        },
        "dense_interior_falsifier": dense,
        "gradient_challenger_search": optimized,
        "continuous_sensitivity_and_reweighting": sensitivity,
        "monte_carlo_noise_profile": noise,
        "pass_conditions": pass_conditions,
        "passed": all(pass_conditions.values()),
        "claim_boundary": (
            "A numerical pass is counterexample-search evidence only. A numerical "
            "failure is a candidate requiring exact replay; neither outcome can "
            "promote a global theorem without rational certification."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--samples-per-allocation", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--restarts-per-allocation", type=int, default=8)
    parser.add_argument("--optimization-steps", type=int, default=250)
    parser.add_argument("--sensitivity-samples", type=int, default=65536)
    parser.add_argument("--reweight-steps", type=int, default=1000)
    parser.add_argument("--noise-samples", type=int, default=4096)
    arguments = parser.parse_args()
    report = run(
        seed=arguments.seed,
        workers=arguments.workers,
        samples_per_allocation=arguments.samples_per_allocation,
        batch_size=arguments.batch_size,
        restarts_per_allocation=arguments.restarts_per_allocation,
        optimization_steps=arguments.optimization_steps,
        sensitivity_samples=arguments.sensitivity_samples,
        reweight_steps=arguments.reweight_steps,
        noise_samples=arguments.noise_samples,
    )
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("one or more numerical falsification gates failed")


if __name__ == "__main__":
    main()
