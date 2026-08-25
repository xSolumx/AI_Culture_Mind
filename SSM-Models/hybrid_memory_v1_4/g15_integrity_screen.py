"""Deterministic pre-training integrity artifact for G15 SpinDirac.

This screen executes the three prospective amendment obligations that are too
large or too diagnostic for ordinary unit tests: training-dtype state growth,
mapped optimizer covariance, and delayed control/query descent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .optimizers import ScalarSecondMomentAdamW
from .spin_dirac_memory import SpinDiracConfig, SpinDiracMemory

SEED = 20_260_825
COVARIANCE_SEEDS = (2039, 2053, 2063)
HORIZON = 4096
CONTROL_STEP = 1.0e-4
MINIMUM_READ_CHANGE = 1.0e-6
MINIMUM_LOSS_REDUCTION = 1.0e-8
MAXIMUM_COVARIANCE_RELATIVE_ERROR = 1.0e-10


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = max(float(torch.linalg.vector_norm(expected)), 1.0e-30)
    return float(torch.linalg.vector_norm(actual - expected)) / denominator


def _mapped_optimizer_covariance(seed: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    dimension = 11
    matrix, _ = torch.linalg.qr(
        torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    )
    initial = torch.randn(dimension, generator=generator, dtype=torch.float64)
    gradient_rows = [
        torch.randn(dimension, generator=generator, dtype=torch.float64)
        for _ in range(8)
    ]
    rows: dict[str, Any] = {}
    for name in ("scalar_second_moment", "sgd"):
        parameter = torch.nn.Parameter(initial.clone())
        mapped = torch.nn.Parameter(matrix @ initial)
        if name == "scalar_second_moment":
            optimizer = ScalarSecondMomentAdamW(
                [parameter], lr=2.0e-3, weight_decay=0.01
            )
            mapped_optimizer = ScalarSecondMomentAdamW(
                [mapped], lr=2.0e-3, weight_decay=0.01
            )
        else:
            optimizer = torch.optim.SGD([parameter], lr=2.0e-3, weight_decay=0.01)
            mapped_optimizer = torch.optim.SGD([mapped], lr=2.0e-3, weight_decay=0.01)
        parameter_errors = []
        update_errors = []
        for gradient in gradient_rows:
            before = parameter.detach().clone()
            mapped_before = mapped.detach().clone()
            parameter.grad = gradient.clone()
            mapped.grad = matrix @ gradient
            optimizer.step()
            mapped_optimizer.step()
            update = parameter.detach() - before
            mapped_update = mapped.detach() - mapped_before
            parameter_errors.append(
                _relative_error(mapped.detach(), matrix @ parameter.detach())
            )
            update_errors.append(_relative_error(mapped_update, matrix @ update))
            optimizer.zero_grad(set_to_none=True)
            mapped_optimizer.zero_grad(set_to_none=True)
        rows[name] = {
            "maximum_parameter_relative_error": max(parameter_errors),
            "maximum_update_relative_error": max(update_errors),
            "steps": 8,
        }
    passed = all(
        max(
            row["maximum_parameter_relative_error"],
            row["maximum_update_relative_error"],
        )
        <= MAXIMUM_COVARIANCE_RELATIVE_ERROR
        for row in rows.values()
    )
    return {"seed": seed, "optimizers": rows, "passed": passed}


def _reads_from_controls(
    memory: SpinDiracMemory,
    inputs: torch.Tensor,
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    erase: torch.Tensor,
    write: torch.Tensor,
    retention: torch.Tensor,
    coordinates: torch.Tensor,
) -> torch.Tensor:
    left, right, injection, _ = memory._transitions(
        key, value, erase, write, retention, coordinates, None
    )
    initial = inputs.new_zeros(inputs.shape[0], *memory.config.state_shape)
    states, _ = memory._recurrent_states(left, right, injection, initial)
    positive = torch.einsum("bthv,bhtvp->bthp", query, states)
    negative = torch.einsum(
        "...i,vji,...v->...j", positive, memory.rho.to(positive), query
    )
    return torch.cat((positive, negative), dim=-1)


def _delayed_observability(seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    memory = SpinDiracMemory(
        SpinDiracConfig(
            8,
            heads=1,
            tie_query_key=False,
            gate_mode="equivariant_scalar",
        )
    ).double()
    with torch.no_grad():
        memory.coordinate_projection.weight.normal_(std=0.08)
    inputs = torch.randn(1, 6, 8, dtype=torch.float64)
    controls = tuple(item.detach() for item in memory._controls(inputs))
    query, key, value, erase, write, retention, coordinates = controls
    target_direction = F.normalize(torch.randn(16, dtype=torch.float64), dim=0)

    def read(
        current_query: torch.Tensor, current_coordinates: torch.Tensor
    ) -> torch.Tensor:
        reads = _reads_from_controls(
            memory,
            inputs,
            query=current_query,
            key=key,
            value=value,
            erase=erase,
            write=write,
            retention=retention,
            coordinates=current_coordinates,
        )
        return reads[0, -1, 0]

    query_variable = query.clone().requires_grad_(True)
    coordinate_variable = coordinates.clone().requires_grad_(True)
    baseline_read = read(query_variable, coordinate_variable)
    target = baseline_read.detach() + 5.0 * target_direction
    baseline_loss = (baseline_read - target).square().sum()
    coordinate_gradient, query_gradient = torch.autograd.grad(
        baseline_loss, (coordinate_variable, query_variable)
    )

    perturbed_coordinates = coordinates.clone()
    delayed_position = 1
    first_gradient = coordinate_gradient[:, delayed_position]
    perturbed_coordinates[:, delayed_position] -= (
        CONTROL_STEP
        * first_gradient
        / torch.linalg.vector_norm(first_gradient).clamp_min(1.0e-30)
    )
    coordinate_read = read(query, perturbed_coordinates)
    coordinate_loss = (coordinate_read - target).square().sum()

    perturbed_query = query.clone()
    final_gradient = query_gradient[:, -1]
    perturbed_query[:, -1] -= (
        CONTROL_STEP
        * final_gradient
        / torch.linalg.vector_norm(final_gradient).clamp_min(1.0e-30)
    )
    perturbed_query[:, -1] = F.normalize(perturbed_query[:, -1], dim=-1)
    query_read = read(perturbed_query, coordinates)
    query_loss = (query_read - target).square().sum()

    baseline_norm = max(
        float(torch.linalg.vector_norm(baseline_read).detach()), 1.0e-30
    )
    coordinate_change = (
        float(torch.linalg.vector_norm(coordinate_read - baseline_read).detach())
        / baseline_norm
    )
    query_change = (
        float(torch.linalg.vector_norm(query_read - baseline_read).detach())
        / baseline_norm
    )
    coordinate_reduction = float((baseline_loss - coordinate_loss).detach())
    query_reduction = float((baseline_loss - query_loss).detach())
    passed = (
        coordinate_change >= MINIMUM_READ_CHANGE
        and query_change >= MINIMUM_READ_CHANGE
        and coordinate_reduction >= MINIMUM_LOSS_REDUCTION
        and query_reduction >= MINIMUM_LOSS_REDUCTION
    )
    return {
        "seed": seed,
        "coordinate_position": delayed_position,
        "scored_position": inputs.shape[1] - 1,
        "coordinate_relative_read_change": coordinate_change,
        "query_relative_read_change": query_change,
        "coordinate_loss_reduction": coordinate_reduction,
        "query_loss_reduction": query_reduction,
        "passed": passed,
    }


def _state_growth(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    memory = SpinDiracMemory(
        SpinDiracConfig(8, heads=2, gate_mode="equivariant_scalar")
    ).to(device=device, dtype=torch.float32)
    memory.eval()
    inputs = 2.0 * torch.rand(1, HORIZON, 8, device=device) - 1.0
    with torch.no_grad():
        output, final_state, diagnostics = memory(
            inputs, scan_mode="recurrent", return_diagnostics=True
        )
        read = torch.cat(
            (diagnostics["read_positive"], diagnostics["read_negative"]), dim=-1
        )
        normalized_read = memory.output_norm(read)
    state_norm = diagnostics["state_norm"]
    if not isinstance(state_norm, torch.Tensor):
        raise TypeError("state_norm diagnostic is missing")
    singular_values = torch.linalg.svdvals(final_state.float())
    final_frobenius = torch.linalg.matrix_norm(final_state.float(), ord="fro")
    ceiling = (1.0 - memory.config.maximum_retention**HORIZON) / (
        1.0 - memory.config.maximum_retention
    )
    finite = all(
        bool(torch.isfinite(tensor).all())
        for tensor in (output, final_state, read, normalized_read)
    )
    passed = finite and float(state_norm.max()) <= ceiling + 1.0e-4
    return {
        "device": str(device),
        "dtype": str(inputs.dtype),
        "training_dtype": "torch.float32",
        "horizon": HORIZON,
        "input_absolute_bound": 1.0,
        "analytic_frobenius_ceiling": ceiling,
        "maximum_state_frobenius_by_head": state_norm.amax(dim=(0, 2)).tolist(),
        "final_state_frobenius_by_head": final_frobenius[0].tolist(),
        "final_largest_singular_value_by_head": singular_values[0, :, 0].tolist(),
        "maximum_state_to_ceiling_ratio_by_head": (
            state_norm.amax(dim=(0, 2)) / ceiling
        ).tolist(),
        "read_rms": float(read.float().square().mean().sqrt()),
        "read_max_abs": float(read.abs().max()),
        "normalized_read_rms": float(normalized_read.float().square().mean().sqrt()),
        "normalized_read_max_abs": float(normalized_read.abs().max()),
        "mixer_output_rms": float(output.float().square().mean().sqrt()),
        "finite": finite,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    root = Path(__file__).resolve().parents[2]
    covariance = [_mapped_optimizer_covariance(seed) for seed in COVARIANCE_SEEDS]
    observability = [_delayed_observability(seed) for seed in COVARIANCE_SEEDS]
    growth = _state_growth(device)
    passed = (
        growth["passed"]
        and all(row["passed"] for row in covariance)
        and all(row["passed"] for row in observability)
    )
    result = {
        "schema_version": 1,
        "seed": SEED,
        "status": "pass" if passed else "fail",
        "claim_boundary": (
            "pre-training SpinDirac integrity only; no learned mechanism, "
            "language quality, long-range recall, or speed result"
        ),
        "thresholds": {
            "control_step": CONTROL_STEP,
            "minimum_relative_read_change": MINIMUM_READ_CHANGE,
            "minimum_loss_reduction": MINIMUM_LOSS_REDUCTION,
            "maximum_covariance_relative_error": MAXIMUM_COVARIANCE_RELATIVE_ERROR,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "compute_capability": (
                list(torch.cuda.get_device_capability(device))
                if device.type == "cuda"
                else None
            ),
        },
        "git_commit_before_worktree_changes": _git_commit(root),
        "source": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "state_growth": growth,
        "optimizer_covariance": covariance,
        "delayed_observability": observability,
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
