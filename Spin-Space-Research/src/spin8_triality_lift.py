"""Exact polynomial lift for triangular Spin(8) triality binding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spin8_triality import (
    SPIN8_BIVECTOR_DIM,
    SPIN8_DIM,
    build_spin8_triality_algebra,
    spin8_actions,
    torch_triality_generators,
)


LIFT_DIM = 1 + SPIN8_DIM + SPIN8_DIM + SPIN8_DIM * SPIN8_DIM
POSITIVE_SLICE = slice(1, 1 + SPIN8_DIM)
NEGATIVE_SLICE = slice(1 + SPIN8_DIM, 1 + 2 * SPIN8_DIM)
TENSOR_SLICE = slice(1 + 2 * SPIN8_DIM, LIFT_DIM)
HOMOGENEOUS_DIM = 1 + SPIN8_DIM


def triality_tensor(
    *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None
) -> torch.Tensor:
    """Return ``rho[v, negative, positive]``."""

    return torch.as_tensor(
        build_spin8_triality_algebra().rho, dtype=dtype, device=device
    )


def triality_bind(
    positive: torch.Tensor, negative: torch.Tensor, rho: torch.Tensor
) -> torch.Tensor:
    """Evaluate the invariant bilinear map ``S+ x S- -> V``."""

    return torch.einsum("...j,vji,...i->...v", negative, rho, positive)


def triality_unbind_negative(
    positive: torch.Tensor, vector: torch.Tensor, rho: torch.Tensor
) -> torch.Tensor:
    """Recover ``S-`` from a unit ``S+`` key and its bound vector."""

    return torch.einsum("...i,vji,...v->...j", positive, rho, vector)


def lift_spinor_pair(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """Construct homogeneous coordinates ``[1,s+,s-,s+ tensor s-]``."""

    if positive.shape != negative.shape or positive.shape[-1] != SPIN8_DIM:
        raise ValueError("spinor pairs must have equal shape (..., 8)")
    one = torch.ones(*positive.shape[:-1], 1, dtype=positive.dtype, device=positive.device)
    tensor = torch.einsum("...i,...j->...ij", positive, negative).flatten(-2)
    return torch.cat((one, positive, negative, tensor), dim=-1)


def lifted_spinor_affine_matrix(
    positive_action: torch.Tensor,
    positive_drive: torch.Tensor,
    negative_action: torch.Tensor,
    negative_drive: torch.Tensor,
) -> torch.Tensor:
    """Build the exact 81D homogeneous matrix for two independent affine streams."""

    leading = positive_action.shape[:-2]
    expected_action = (*leading, SPIN8_DIM, SPIN8_DIM)
    expected_drive = (*leading, SPIN8_DIM)
    if (
        positive_action.shape != expected_action
        or negative_action.shape != expected_action
        or positive_drive.shape != expected_drive
        or negative_drive.shape != expected_drive
    ):
        raise ValueError("incompatible Spin(8) affine shapes")
    matrix = positive_action.new_zeros(*leading, LIFT_DIM, LIFT_DIM)
    matrix[..., 0, 0] = 1.0
    matrix[..., POSITIVE_SLICE, 0] = positive_drive
    matrix[..., POSITIVE_SLICE, POSITIVE_SLICE] = positive_action
    matrix[..., NEGATIVE_SLICE, 0] = negative_drive
    matrix[..., NEGATIVE_SLICE, NEGATIVE_SLICE] = negative_action

    constant = torch.einsum(
        "...i,...j->...ij", positive_drive, negative_drive
    ).flatten(-2)
    from_positive = torch.einsum(
        "...ik,...j->...ijk", positive_action, negative_drive
    ).reshape(*leading, SPIN8_DIM * SPIN8_DIM, SPIN8_DIM)
    from_negative = torch.einsum(
        "...i,...jk->...ijk", positive_drive, negative_action
    ).reshape(*leading, SPIN8_DIM * SPIN8_DIM, SPIN8_DIM)
    tensor_action = torch.einsum(
        "...ik,...jl->...ijkl", positive_action, negative_action
    ).reshape(
        *leading,
        SPIN8_DIM * SPIN8_DIM,
        SPIN8_DIM * SPIN8_DIM,
    )
    matrix[..., TENSOR_SLICE, 0] = constant
    matrix[..., TENSOR_SLICE, POSITIVE_SLICE] = from_positive
    matrix[..., TENSOR_SLICE, NEGATIVE_SLICE] = from_negative
    matrix[..., TENSOR_SLICE, TENSOR_SLICE] = tensor_action
    return matrix


def triality_readout_from_lift(lifted: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    tensor = lifted[..., TENSOR_SLICE].reshape(
        *lifted.shape[:-1], SPIN8_DIM, SPIN8_DIM
    )
    return torch.einsum("vji,...ij->...v", rho, tensor)


def associative_matrix_scan(matrices: torch.Tensor) -> torch.Tensor:
    """Inclusive ordered matrix prefixes over sequence axis 1."""

    if matrices.ndim < 4 or matrices.shape[-2:] != (LIFT_DIM, LIFT_DIM):
        raise ValueError("matrices must have shape (batch, length, 81, 81)")
    prefixes = matrices
    offset = 1
    while offset < matrices.shape[1]:
        composed = prefixes[:, offset:] @ prefixes[:, :-offset]
        prefixes = torch.cat((prefixes[:, :offset], composed), dim=1)
        offset *= 2
    return prefixes


def homogeneous_affine_matrix(
    action: torch.Tensor, drive: torch.Tensor
) -> torch.Tensor:
    """Embed an 8D affine transition in one 9D homogeneous matrix."""

    if action.shape[:-2] != drive.shape[:-1] or action.shape[-2:] != (
        SPIN8_DIM,
        SPIN8_DIM,
    ) or drive.shape[-1] != SPIN8_DIM:
        raise ValueError("incompatible homogeneous affine shapes")
    matrix = action.new_zeros(*action.shape[:-2], HOMOGENEOUS_DIM, HOMOGENEOUS_DIM)
    matrix[..., 0, 0] = 1.0
    matrix[..., 1:, 0] = drive
    matrix[..., 1:, 1:] = action
    return matrix


def associative_homogeneous_scan(matrices: torch.Tensor) -> torch.Tensor:
    if matrices.ndim < 4 or matrices.shape[-2:] != (
        HOMOGENEOUS_DIM,
        HOMOGENEOUS_DIM,
    ):
        raise ValueError("matrices must have shape (batch, length, 9, 9)")
    prefixes = matrices
    offset = 1
    while offset < matrices.shape[1]:
        composed = prefixes[:, offset:] @ prefixes[:, :-offset]
        prefixes = torch.cat((prefixes[:, :offset], composed), dim=1)
        offset *= 2
    return prefixes


def staged_triality_scan(
    positive_action: torch.Tensor,
    positive_drive: torch.Tensor,
    negative_action: torch.Tensor,
    negative_drive: torch.Tensor,
    vector_action: torch.Tensor,
    vector_base_drive: torch.Tensor,
    initial_positive: torch.Tensor,
    initial_negative: torch.Tensor,
    initial_vector: torch.Tensor,
    rho: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-stage parallel scan for a triangular triality-coupled recurrence.

    The two spinor streams are scanned first. Their current-prefix triality
    binding is then an ordinary per-token vector drive for the second scan.
    Streaming retains only the three 8D states; the 64D polynomial lift is a
    single-scan proof device, not a required recurrent cache.
    """

    positive_prefix = associative_homogeneous_scan(
        homogeneous_affine_matrix(positive_action, positive_drive)
    )
    negative_prefix = associative_homogeneous_scan(
        homogeneous_affine_matrix(negative_action, negative_drive)
    )
    one = torch.ones(
        *initial_positive.shape[:-1], 1,
        dtype=initial_positive.dtype,
        device=initial_positive.device,
    )
    positive_initial_h = torch.cat((one, initial_positive), dim=-1)
    negative_initial_h = torch.cat((one, initial_negative), dim=-1)
    positive = torch.einsum(
        "blij,bj->bli", positive_prefix, positive_initial_h
    )[..., 1:]
    negative = torch.einsum(
        "blij,bj->bli", negative_prefix, negative_initial_h
    )[..., 1:]
    binding = triality_bind(positive, negative, rho)
    vector_prefix = associative_homogeneous_scan(
        homogeneous_affine_matrix(vector_action, vector_base_drive + binding)
    )
    vector_initial_h = torch.cat((one, initial_vector), dim=-1)
    vector = torch.einsum("blij,bj->bli", vector_prefix, vector_initial_h)[..., 1:]
    return positive, negative, vector


def degree_growth(steps: int) -> dict[str, list[int]]:
    triangular = []
    feedback = []
    positive_degree = negative_degree = 1
    for _ in range(steps):
        triangular.append(2)
        binding_degree = positive_degree + negative_degree
        positive_degree = max(positive_degree, binding_degree)
        negative_degree = max(negative_degree, binding_degree)
        feedback.append(binding_degree)
    return {"triangular_max_degree": triangular, "two_way_feedback_degree": feedback}


def diagnostics(seed: int = 20260803) -> dict[str, object]:
    torch.manual_seed(seed)
    dtype = torch.float64
    batch, length = 2, 17
    positive_skew = torch.randn(
        batch, length, SPIN8_DIM, SPIN8_DIM, dtype=dtype
    )
    positive_action = torch.matrix_exp(
        0.15 * (positive_skew - positive_skew.transpose(-1, -2))
    )
    # Rebuild negative actions from a separate skew draw; closure of the lift
    # needs only affine independence, not shared Spin(8) coefficients.
    negative_skew = torch.randn(batch, length, SPIN8_DIM, SPIN8_DIM, dtype=dtype)
    negative_action = torch.matrix_exp(0.15 * (negative_skew - negative_skew.transpose(-1, -2)))
    positive_drive = 0.1 * torch.randn(batch, length, SPIN8_DIM, dtype=dtype)
    negative_drive = 0.1 * torch.randn(batch, length, SPIN8_DIM, dtype=dtype)
    matrices = lifted_spinor_affine_matrix(
        positive_action, positive_drive, negative_action, negative_drive
    )
    positive = torch.randn(batch, SPIN8_DIM, dtype=dtype)
    negative = torch.randn(batch, SPIN8_DIM, dtype=dtype)
    initial = lift_spinor_pair(positive, negative)

    direct_positive = positive_action[:, 0] @ positive.unsqueeze(-1)
    direct_positive = direct_positive.squeeze(-1) + positive_drive[:, 0]
    direct_negative = negative_action[:, 0] @ negative.unsqueeze(-1)
    direct_negative = direct_negative.squeeze(-1) + negative_drive[:, 0]
    lifted_one = torch.einsum("bij,bj->bi", matrices[:, 0], initial)
    expected_one = lift_spinor_pair(direct_positive, direct_negative)

    prefixes = associative_matrix_scan(matrices)
    parallel = torch.einsum("blij,bj->bli", prefixes, initial)
    state = initial
    recurrent = []
    for position in range(length):
        state = torch.einsum("bij,bj->bi", matrices[:, position], state)
        recurrent.append(state)
    recurrent = torch.stack(recurrent, dim=1)

    rho = triality_tensor(dtype=dtype)
    direct_binding = triality_bind(positive, negative, rho)
    lifted_binding = triality_readout_from_lift(initial, rho)

    coefficients = 0.2 * torch.randn(SPIN8_BIVECTOR_DIM, dtype=dtype)
    vector_action, positive_group_action, negative_group_action = spin8_actions(
        coefficients, torch_triality_generators(dtype=dtype)
    )
    transformed_binding = triality_bind(
        torch.einsum("ij,bj->bi", positive_group_action, positive),
        torch.einsum("ij,bj->bi", negative_group_action, negative),
        rho,
    )
    expected_binding = torch.einsum(
        "ij,bj->bi", vector_action, direct_binding
    )

    vector_skew = torch.randn(batch, length, SPIN8_DIM, SPIN8_DIM, dtype=dtype)
    vector_actions = torch.matrix_exp(
        0.15 * (vector_skew - vector_skew.transpose(-1, -2))
    )
    vector_base_drive = 0.05 * torch.randn(
        batch, length, SPIN8_DIM, dtype=dtype
    )
    initial_vector = torch.randn(batch, SPIN8_DIM, dtype=dtype)
    staged = staged_triality_scan(
        positive_action,
        positive_drive,
        negative_action,
        negative_drive,
        vector_actions,
        vector_base_drive,
        positive,
        negative,
        initial_vector,
        rho,
    )
    sequential_positive, sequential_negative, sequential_vector = (
        positive,
        negative,
        initial_vector,
    )
    sequential = [[], [], []]
    for position in range(length):
        sequential_positive = torch.einsum(
            "bij,bj->bi", positive_action[:, position], sequential_positive
        ) + positive_drive[:, position]
        sequential_negative = torch.einsum(
            "bij,bj->bi", negative_action[:, position], sequential_negative
        ) + negative_drive[:, position]
        binding = triality_bind(sequential_positive, sequential_negative, rho)
        sequential_vector = torch.einsum(
            "bij,bj->bi", vector_actions[:, position], sequential_vector
        ) + vector_base_drive[:, position] + binding
        sequential[0].append(sequential_positive)
        sequential[1].append(sequential_negative)
        sequential[2].append(sequential_vector)
    sequential_stacked = tuple(torch.stack(values, dim=1) for values in sequential)
    staged_error = max(
        float((parallel_value - recurrent_value).abs().max())
        for parallel_value, recurrent_value in zip(staged, sequential_stacked)
    )
    checks = {
        "one_step": float((lifted_one - expected_one).abs().max()) <= 1e-12,
        "parallel_recurrent": float((parallel - recurrent).abs().max()) <= 1e-11,
        "lifted_readout": float((direct_binding - lifted_binding).abs().max()) <= 1e-12,
        "triality_equivariance": float((transformed_binding - expected_binding).abs().max()) <= 1e-11,
        "staged_parallel_recurrent": staged_error <= 1e-11,
    }
    return {
        "experiment": "exact triangular Spin(8) triality polynomial lift",
        "lift_dimension": LIFT_DIM,
        "one_step_max_abs_error": float((lifted_one - expected_one).abs().max()),
        "parallel_recurrent_max_abs_error": float((parallel - recurrent).abs().max()),
        "lifted_readout_max_abs_error": float((direct_binding - lifted_binding).abs().max()),
        "triality_equivariance_max_abs_error": float((transformed_binding - expected_binding).abs().max()),
        "staged_parallel_recurrent_max_abs_error": staged_error,
        "streaming_cache_scalars": 3 * SPIN8_DIM,
        "parallel_scan_stages": 2,
        "degree_growth": degree_growth(8),
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnostics()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
