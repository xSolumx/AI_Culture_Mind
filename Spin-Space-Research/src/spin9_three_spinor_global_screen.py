"""Unrestricted numerical falsifier for the Spin(9) three-spinor candidate.

This is deliberately separate from the exact theorem certificates.  It uses
full-dimensional unit-spinor optimization to search for a point above the
algebraic one-parameter candidate and profiles a transverse rank-two boundary.
Passing this screen is evidence against a counterexample, not a global proof.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import platform
from pathlib import Path

import numpy as np
import torch

from spin9_dirac_clifford import build_spin9_clifford_system


def algebraic_candidate() -> tuple[float, np.ndarray]:
    """Return the exact-curve maximizer and its canonical spinor triple."""

    c = (math.sqrt(241.0) - 17.0) / 24.0
    d = math.sqrt((1.0 + c) / 2.0)
    b = math.sqrt((1.0 - c) / 2.0)
    y = c * (1.0 - c) / (2.0 * b * (1.0 + c))
    z = math.sqrt((1.0 - c) * (1.0 + 2.0 * c) / (2.0 * (1.0 + c) ** 2))
    spinors = np.zeros((3, 16), dtype=np.float64)
    spinors[0, 0] = 1.0
    spinors[1, 1], spinors[1, 8] = d, b
    spinors[2, 2], spinors[2, 11], spinors[2, 12] = -d, y, z
    return c, spinors


def candidate_log_determinant(c: float) -> float:
    return (
        10.0 * math.log(1.0 - c)
        + 5.0 * math.log(c + 2.0)
        + 3.0 * math.log(2.0 * c + 1.0)
        - 43.0 * math.log(2.0)
    )


def _information(generators: torch.Tensor, spinors: torch.Tensor) -> torch.Tensor:
    observations = torch.einsum("aij,brj->brai", generators, spinors)
    return torch.einsum("brai,brci->bac", observations, observations)


def _optimize_seed(
    seed: int,
    starts: int,
    steps: int,
    learning_rate: float,
    device: torch.device,
    generators: torch.Tensor,
    involutions: torch.Tensor,
    target_logdet: float,
    target_c: float,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    raw = torch.tensor(
        rng.standard_normal((starts, 3, 16)),
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([raw], lr=learning_rate)

    def metrics() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spinors = raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
        information = _information(generators, spinors)
        signs, logdets = torch.linalg.slogdet(information)
        return spinors, signs, logdets

    with torch.no_grad():
        _, initial_signs, initial_logdets = metrics()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _, signs, logdets = metrics()
        if not bool(torch.all(signs > 0)):
            raise RuntimeError("a nonsingular random triple produced nonpositive sign")
        (-logdets.mean()).backward()
        optimizer.step()

    with torch.no_grad():
        spinors, signs, logdets = metrics()
        grams = spinors @ spinors.transpose(-1, -2)
        hopf = torch.einsum("bri,kij,brj->brk", spinors, involutions, spinors)
        hopf_grams = hopf @ hopf.transpose(-1, -2)
        identity = torch.eye(3, dtype=torch.float64, device=device)
        target_hopf = identity + target_c * (torch.ones_like(identity) - identity)
        gram_error = (
            torch.max(torch.abs(grams - identity), dim=-1).values.max(dim=-1).values
        )
        hopf_error = (
            torch.max(torch.abs(hopf_grams - target_hopf), dim=-1)
            .values.max(dim=-1)
            .values
        )
        gaps = logdets - target_logdet
    return {
        "seed": seed,
        "starts": starts,
        "initial_positive_signs": int((initial_signs > 0).sum().item()),
        "initial_best_logdet": float(initial_logdets.max().item()),
        "initial_mean_logdet": float(initial_logdets.mean().item()),
        "final_positive_signs": int((signs > 0).sum().item()),
        "final_best_logdet": float(logdets.max().item()),
        "final_worst_logdet": float(logdets.min().item()),
        "final_mean_logdet": float(logdets.mean().item()),
        "max_absolute_candidate_gap": float(torch.max(torch.abs(gaps)).item()),
        "max_spinor_gram_error": float(gram_error.max().item()),
        "max_hopf_gram_error": float(hopf_error.max().item()),
    }


def _boundary_profile(
    candidate: np.ndarray, generators: torch.Tensor, device: torch.device
) -> dict[str, object]:
    epsilon_values = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    first = torch.tensor(candidate[0], dtype=torch.float64, device=device)
    second = torch.tensor(candidate[1], dtype=torch.float64, device=device)
    transverse = torch.tensor(candidate[2], dtype=torch.float64, device=device)
    logdets: list[float] = []
    minimum_eigenvalues: list[float] = []
    with torch.no_grad():
        for epsilon in epsilon_values:
            third = first + epsilon * transverse
            third = third / torch.linalg.vector_norm(third)
            spinors = torch.stack((first, second, third))[None]
            information = _information(generators, spinors)[0]
            eigenvalues = torch.linalg.eigvalsh(information)
            logdets.append(float(torch.log(eigenvalues).sum().item()))
            minimum_eigenvalues.append(float(eigenvalues[0].item()))
    slopes = [
        (right - left) / math.log(epsilon_values[index + 1] / epsilon_values[index])
        for index, (left, right) in enumerate(itertools.pairwise(logdets))
    ]
    return {
        "epsilon_values": epsilon_values,
        "log_determinants": logdets,
        "minimum_eigenvalues": minimum_eigenvalues,
        "successive_log_log_slopes": slopes,
        "expected_transverse_order": 16,
        "last_slope_error": abs(slopes[-1] - 16.0),
    }


def _local_hessian_profile(
    candidate: np.ndarray, generators: torch.Tensor, device: torch.device
) -> dict[str, object]:
    """Measure the constrained Hessian with the first spinor gauge-fixed."""

    first, second, third = [
        torch.tensor(row, dtype=torch.float64, device=device) for row in candidate
    ]
    tangent_bases = []
    for spinor in (second, third):
        _, _, right = torch.linalg.svd(spinor.reshape(1, -1), full_matrices=True)
        tangent_bases.append(right[1:].T.contiguous())
    second_basis, third_basis = tangent_bases

    def objective(coordinates: torch.Tensor) -> torch.Tensor:
        moved_second = second + second_basis @ coordinates[:15]
        moved_third = third + third_basis @ coordinates[15:]
        moved_second = moved_second / torch.linalg.vector_norm(moved_second)
        moved_third = moved_third / torch.linalg.vector_norm(moved_third)
        spinors = torch.stack((first, moved_second, moved_third))[None]
        _, logdet = torch.linalg.slogdet(_information(generators, spinors)[0])
        return -logdet

    zero = torch.zeros(30, dtype=torch.float64, device=device, requires_grad=True)
    gradient = torch.autograd.grad(objective(zero), zero)[0]
    hessian = torch.autograd.functional.hessian(objective, zero, vectorize=True)
    hessian = (hessian + hessian.T) / 2.0
    eigenvalues = torch.linalg.eigvalsh(hessian)

    # Rotating the second and third probes into one another preserves their
    # frame operator exactly.  This flat direction is contained in the
    # residual Spin(7) orbit rather than adding a 22nd independent null mode.
    slot_rotation = torch.cat((second_basis.T @ third, third_basis.T @ (-second)))
    slot_rotation = slot_rotation / torch.linalg.vector_norm(slot_rotation)
    slot_residual = torch.linalg.vector_norm(hessian @ slot_rotation)
    tolerance = 1e-8
    return {
        "coordinate_dimension_after_fixing_first_spinor": 30,
        "gradient_norm": float(torch.linalg.vector_norm(gradient).item()),
        "hessian_eigenvalues": [float(value) for value in eigenvalues.tolist()],
        "null_modes_at_1e-8": int((torch.abs(eigenvalues) < tolerance).sum().item()),
        "positive_modes_at_1e-8": int((eigenvalues > tolerance).sum().item()),
        "negative_modes_at_1e-8": int((eigenvalues < -tolerance).sum().item()),
        "smallest_positive_eigenvalue": float(
            eigenvalues[eigenvalues > tolerance].min().item()
        ),
        "slot_rotation_hessian_residual_norm": float(slot_residual.item()),
        "interpretation": "numerical strict local maximum modulo the residual group orbit",
    }


def run(
    seeds: list[int],
    starts: int,
    steps: int,
    learning_rate: float,
    device_name: str,
    cpu_threads: int,
) -> dict[str, object]:
    """Run the frozen numerical falsifier and return its complete report."""

    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)
    device = torch.device(device_name)
    system = build_spin9_clifford_system()
    generators = (
        torch.tensor(
            system.doubled_spin_generators,
            dtype=torch.float64,
            device=device,
        )
        / 2.0
    )
    involutions = torch.tensor(
        system.involutions,
        dtype=torch.float64,
        device=device,
    )
    target_c, candidate = algebraic_candidate()
    target_logdet = candidate_log_determinant(target_c)
    seed_reports = [
        _optimize_seed(
            seed,
            starts,
            steps,
            learning_rate,
            device,
            generators,
            involutions,
            target_logdet,
            target_c,
        )
        for seed in seeds
    ]
    boundary = _boundary_profile(candidate, generators, device)
    local_hessian = _local_hessian_profile(candidate, generators, device)
    worst_gap = max(report["max_absolute_candidate_gap"] for report in seed_reports)
    worst_gram = max(report["max_spinor_gram_error"] for report in seed_reports)
    worst_hopf = max(report["max_hopf_gram_error"] for report in seed_reports)
    report: dict[str, object] = {
        "schema_version": 1,
        "claim_scope": "unrestricted numerical falsification, not global proof",
        "seeds": seeds,
        "starts_per_seed": starts,
        "optimization_steps": steps,
        "learning_rate": learning_rate,
        "dtype": "float64",
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor()
        ),
        "torch_version": torch.__version__,
        "cpu_threads": cpu_threads,
        "candidate_c": target_c,
        "candidate_log_determinant": target_logdet,
        "seed_reports": seed_reports,
        "worst_absolute_candidate_gap": worst_gap,
        "worst_spinor_gram_error": worst_gram,
        "worst_hopf_gram_error": worst_hopf,
        "boundary_profile": boundary,
        "local_hessian_profile": local_hessian,
        "counterexample_found": any(
            seed_report["final_best_logdet"] > target_logdet + 1e-9
            for seed_report in seed_reports
        ),
        "global_optimality_claimed": False,
    }
    report["passed"] = bool(
        not report["counterexample_found"]
        and worst_gap < 1e-9
        and worst_gram < 1e-7
        and worst_hopf < 1e-7
        and boundary["last_slope_error"] < 1e-5
        and local_hessian["gradient_norm"] < 1e-10
        and local_hessian["null_modes_at_1e-8"] == 21
        and local_hessian["positive_modes_at_1e-8"] == 9
        and local_hessian["negative_modes_at_1e-8"] == 0
        and local_hessian["slot_rotation_hessian_residual_norm"] < 1e-10
        and not report["global_optimality_claimed"]
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--starts", type=int, default=32)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run(
        seeds=args.seeds,
        starts=args.starts,
        steps=args.steps,
        learning_rate=args.learning_rate,
        device_name=args.device,
        cpu_threads=args.cpu_threads,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    raise SystemExit(main())
