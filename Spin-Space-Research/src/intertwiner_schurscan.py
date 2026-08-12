"""Generic exact SchurScan for triangular bilinear intertwiners.

For a bilinear map beta: U x V -> W, independently affine U and V streams
may drive a downstream affine W stream through beta(u_t, v_t).  The practical
algorithm is two staged associative scans and stores only U + V + W recurrent
coordinates.  A homogeneous state containing U tensor V gives an exact
single-scan linear lift and a mechanical proof of closure.

No group theory is required for scan compatibility.  Equivariance is an extra
property of beta.  Spin(8) triality is one exceptional instance; the SO(3)
cross product below is a nonexceptional control.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch


@dataclass(frozen=True)
class LiftLayout:
    u: slice
    v: slice
    tensor: slice
    w: slice
    dimension: int


def lift_layout(u_dim: int, v_dim: int, w_dim: int) -> LiftLayout:
    u = slice(1, 1 + u_dim)
    v = slice(u.stop, u.stop + v_dim)
    tensor = slice(v.stop, v.stop + u_dim * v_dim)
    w = slice(tensor.stop, tensor.stop + w_dim)
    return LiftLayout(u=u, v=v, tensor=tensor, w=w, dimension=w.stop)


def bilinear_contract(
    u: torch.Tensor, v: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """Return beta(u, v) for beta shaped ``(W, U, V)``."""

    if u.shape[:-1] != v.shape[:-1]:
        raise ValueError("u and v must have equal leading shapes")
    if beta.ndim != 3 or beta.shape[1:] != (u.shape[-1], v.shape[-1]):
        raise ValueError("beta must have shape (W, U, V)")
    return torch.einsum("...i,oij,...j->...o", u, beta, v)


def homogeneous_affine_matrix(
    action: torch.Tensor, drive: torch.Tensor
) -> torch.Tensor:
    if action.shape[:-2] != drive.shape[:-1] or action.shape[-1] != action.shape[-2]:
        raise ValueError("incompatible affine action and drive")
    dimension = action.shape[-1]
    matrix = action.new_zeros(*action.shape[:-2], dimension + 1, dimension + 1)
    matrix[..., 0, 0] = 1
    matrix[..., 1:, 0] = drive
    matrix[..., 1:, 1:] = action
    return matrix


ScanBackend = Literal[
    "work_efficient",
    "work_efficient_affine",
    "work_efficient_homogeneous",
    "hillis_steele",
]


def _validate_matrix_sequence(matrices: torch.Tensor) -> tuple[int, int, int]:
    if matrices.ndim != 4 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError("matrices must have shape (batch, length, D, D)")
    batch, length, dimension = matrices.shape[:3]
    if length < 1:
        raise ValueError("scan length must be positive")
    return batch, length, dimension


def associative_matrix_scan(matrices: torch.Tensor) -> torch.Tensor:
    """Inclusive Hillis--Steele prefix products over sequence axis one.

    This reference implementation has logarithmic dependency depth but
    ``O(N log N)`` matrix-multiply work.  Use
    :func:`work_efficient_associative_matrix_scan` when operation count matters.
    """

    _validate_matrix_sequence(matrices)
    prefixes = matrices
    offset = 1
    while offset < matrices.shape[1]:
        composed = prefixes[:, offset:] @ prefixes[:, :-offset]
        prefixes = torch.cat((prefixes[:, :offset], composed), dim=1)
        offset *= 2
    return prefixes


def work_efficient_associative_matrix_scan(matrices: torch.Tensor) -> torch.Tensor:
    """Inclusive ordered prefix products using a Blelloch-style tree.

    For the chronological transition convention used here, a right segment
    acts after a left segment, so their aggregate is ``right @ left``.  The
    upsweep, downsweep, and final inclusive conversion require fewer than
    ``3P`` matrix products for ``P`` the next power of two above the sequence
    length.  All updates are out of place, so PyTorch autograd can traverse the
    tree.
    """

    batch, length, dimension = _validate_matrix_sequence(matrices)
    if length == 1:
        return matrices
    padded_length = 1 << (length - 1).bit_length()
    if padded_length == length:
        leaves = matrices
    else:
        identity = torch.eye(
            dimension, dtype=matrices.dtype, device=matrices.device
        ).reshape(1, 1, dimension, dimension)
        padding = identity.expand(batch, padded_length - length, -1, -1)
        leaves = torch.cat((matrices, padding), dim=1)

    levels = [leaves]
    nodes = leaves
    while nodes.shape[1] > 1:
        nodes = nodes[:, 1::2] @ nodes[:, 0::2]
        levels.append(nodes)

    identity = torch.eye(
        dimension, dtype=matrices.dtype, device=matrices.device
    ).reshape(1, 1, dimension, dimension)
    exclusive = identity.expand(batch, 1, -1, -1)
    for children in reversed(levels[:-1]):
        left_totals = children[:, 0::2]
        right_exclusive = left_totals @ exclusive
        exclusive = torch.stack((exclusive, right_exclusive), dim=2).reshape(
            batch, -1, dimension, dimension
        )

    inclusive = leaves @ exclusive
    return inclusive[:, :length]


def matrix_scan(
    matrices: torch.Tensor, *, backend: ScanBackend = "work_efficient"
) -> torch.Tensor:
    if backend in (
        "work_efficient",
        "work_efficient_affine",
        "work_efficient_homogeneous",
    ):
        return work_efficient_associative_matrix_scan(matrices)
    if backend == "hillis_steele":
        return associative_matrix_scan(matrices)
    raise ValueError(f"unknown scan backend: {backend}")


def work_efficient_affine_prefixes(
    action: torch.Tensor, drive: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return inclusive affine prefixes without homogeneous augmentation."""

    if (
        action.ndim != 4
        or drive.ndim != 3
        or action.shape[:2] != drive.shape[:2]
        or action.shape[-1] != action.shape[-2]
        or action.shape[-1] != drive.shape[-1]
    ):
        raise ValueError(
            "action and drive must have shapes (batch, length, D, D) and "
            "(batch, length, D)"
        )
    batch, length, dimension = drive.shape
    if length < 1:
        raise ValueError("scan length must be positive")
    if length == 1:
        return action, drive

    padded_length = 1 << (length - 1).bit_length()
    if padded_length == length:
        leaf_action, leaf_drive = action, drive
    else:
        identity = torch.eye(
            dimension, dtype=action.dtype, device=action.device
        ).reshape(1, 1, dimension, dimension)
        zero = torch.zeros(1, 1, dimension, dtype=drive.dtype, device=drive.device)
        leaf_action = torch.cat(
            (
                action,
                identity.expand(batch, padded_length - length, -1, -1),
            ),
            dim=1,
        )
        leaf_drive = torch.cat(
            (drive, zero.expand(batch, padded_length - length, -1)), dim=1
        )

    levels: list[tuple[torch.Tensor, torch.Tensor]] = [(leaf_action, leaf_drive)]
    node_action, node_drive = leaf_action, leaf_drive
    while node_action.shape[1] > 1:
        left_action, right_action = node_action[:, 0::2], node_action[:, 1::2]
        left_drive, right_drive = node_drive[:, 0::2], node_drive[:, 1::2]
        node_action = right_action @ left_action
        node_drive = (
            torch.einsum("bnij,bnj->bni", right_action, left_drive) + right_drive
        )
        levels.append((node_action, node_drive))

    identity = torch.eye(dimension, dtype=action.dtype, device=action.device).reshape(
        1, 1, dimension, dimension
    )
    prefix_action = identity.expand(batch, 1, -1, -1)
    prefix_drive = torch.zeros(
        batch, 1, dimension, dtype=drive.dtype, device=drive.device
    )
    for child_action, child_drive in reversed(levels[:-1]):
        left_action = child_action[:, 0::2]
        left_drive = child_drive[:, 0::2]
        right_prefix_action = left_action @ prefix_action
        right_prefix_drive = (
            torch.einsum("bnij,bnj->bni", left_action, prefix_drive) + left_drive
        )
        prefix_action = torch.stack(
            (prefix_action, right_prefix_action), dim=2
        ).reshape(batch, -1, dimension, dimension)
        prefix_drive = torch.stack((prefix_drive, right_prefix_drive), dim=2).reshape(
            batch, -1, dimension
        )

    inclusive_action = leaf_action @ prefix_action
    inclusive_drive = (
        torch.einsum("bnij,bnj->bni", leaf_action, prefix_drive) + leaf_drive
    )
    return inclusive_action[:, :length], inclusive_drive[:, :length]


def scan_composition_counts(length: int) -> dict[str, int]:
    """Return exact composition counts for the two reference scan trees.

    Counts describe the Python tensor programs in this module, not GPU kernel
    launches or wall-clock cost.  One affine composition comprises one matrix
    product, one matrix-vector product, and one vector addition.
    """

    if length < 1:
        raise ValueError("scan length must be positive")
    hillis = 0
    offset = 1
    while offset < length:
        hillis += length - offset
        offset *= 2
    if length == 1:
        work_efficient = 0
    else:
        padded_length = 1 << (length - 1).bit_length()
        work_efficient = 3 * padded_length - 2
    return {"hillis_steele": hillis, "work_efficient": work_efficient}


def scan_dependency_depths(length: int) -> dict[str, int]:
    """Return matrix-composition dependency depth of each reference tree."""

    if length < 1:
        raise ValueError("scan length must be positive")
    if length == 1:
        return {"hillis_steele": 0, "work_efficient": 0}
    levels = (length - 1).bit_length()
    return {
        "hillis_steele": levels,
        "work_efficient": 2 * levels + 1,
    }


def _scan_affine(
    action: torch.Tensor,
    drive: torch.Tensor,
    initial: torch.Tensor,
    *,
    backend: ScanBackend = "work_efficient",
) -> torch.Tensor:
    if initial.shape != (action.shape[0], action.shape[-1]):
        raise ValueError("initial must have shape (batch, D)")
    if backend == "work_efficient_affine":
        prefix_action, prefix_drive = work_efficient_affine_prefixes(action, drive)
        return torch.einsum("blij,bj->bli", prefix_action, initial) + prefix_drive
    prefixes = matrix_scan(homogeneous_affine_matrix(action, drive), backend=backend)
    one = torch.ones(*initial.shape[:-1], 1, dtype=initial.dtype, device=initial.device)
    initial_h = torch.cat((one, initial), dim=-1)
    return torch.einsum("blij,bj->bli", prefixes, initial_h)[..., 1:]


def staged_intertwiner_scan(
    u_action: torch.Tensor,
    u_drive: torch.Tensor,
    v_action: torch.Tensor,
    v_drive: torch.Tensor,
    w_action: torch.Tensor,
    w_drive: torch.Tensor,
    initial_u: torch.Tensor,
    initial_v: torch.Tensor,
    initial_w: torch.Tensor,
    beta: torch.Tensor,
    *,
    scan_backend: ScanBackend = "work_efficient",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the exact compact two-stage triangular scan."""

    u = _scan_affine(u_action, u_drive, initial_u, backend=scan_backend)
    v = _scan_affine(v_action, v_drive, initial_v, backend=scan_backend)
    binding = bilinear_contract(u, v, beta)
    w = _scan_affine(w_action, w_drive + binding, initial_w, backend=scan_backend)
    return u, v, w


def recurrent_intertwiner_scan(
    u_action: torch.Tensor,
    u_drive: torch.Tensor,
    v_action: torch.Tensor,
    v_drive: torch.Tensor,
    w_action: torch.Tensor,
    w_drive: torch.Tensor,
    initial_u: torch.Tensor,
    initial_v: torch.Tensor,
    initial_w: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sequential ground truth for the triangular recurrence."""

    batch, length, u_dim = u_drive.shape
    v_dim = v_drive.shape[-1]
    w_dim = w_drive.shape[-1]
    if (
        u_action.shape != (batch, length, u_dim, u_dim)
        or v_action.shape != (batch, length, v_dim, v_dim)
        or w_action.shape != (batch, length, w_dim, w_dim)
        or v_drive.shape[:2] != (batch, length)
        or w_drive.shape[:2] != (batch, length)
        or initial_u.shape != (batch, u_dim)
        or initial_v.shape != (batch, v_dim)
        or initial_w.shape != (batch, w_dim)
        or beta.shape != (w_dim, u_dim, v_dim)
    ):
        raise ValueError("incompatible triangular recurrence shapes")
    if length < 1:
        raise ValueError("scan length must be positive")

    state_u, state_v, state_w = initial_u, initial_v, initial_w
    rows: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]] = (
        [],
        [],
        [],
    )
    for position in range(u_action.shape[1]):
        state_u = (
            torch.einsum("bij,bj->bi", u_action[:, position], state_u)
            + u_drive[:, position]
        )
        state_v = (
            torch.einsum("bij,bj->bi", v_action[:, position], state_v)
            + v_drive[:, position]
        )
        state_w = (
            torch.einsum("bij,bj->bi", w_action[:, position], state_w)
            + w_drive[:, position]
            + bilinear_contract(state_u, state_v, beta)
        )
        rows[0].append(state_u)
        rows[1].append(state_v)
        rows[2].append(state_w)
    return (
        torch.stack(rows[0], dim=1),
        torch.stack(rows[1], dim=1),
        torch.stack(rows[2], dim=1),
    )


def lift_state(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    if u.shape[:-1] != v.shape[:-1] or u.shape[:-1] != w.shape[:-1]:
        raise ValueError("lifted states must have equal leading shapes")
    one = torch.ones(*u.shape[:-1], 1, dtype=u.dtype, device=u.device)
    tensor = torch.einsum("...i,...j->...ij", u, v).flatten(-2)
    return torch.cat((one, u, v, tensor, w), dim=-1)


def lifted_triangular_matrix(
    u_action: torch.Tensor,
    u_drive: torch.Tensor,
    v_action: torch.Tensor,
    v_drive: torch.Tensor,
    w_action: torch.Tensor,
    w_drive: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Build the exact homogeneous lift of one triangular bilinear step.

    The returned matrix advances ``[1, u, v, u tensor v, w]`` and implements

        u' = A u + a
        v' = B v + b
        w' = C w + c + beta(u', v').
    """

    leading = u_action.shape[:-2]
    u_dim = u_action.shape[-1]
    v_dim = v_action.shape[-1]
    w_dim = w_action.shape[-1]
    if (
        u_action.shape[-2:] != (u_dim, u_dim)
        or v_action.shape[:-2] != leading
        or v_action.shape[-2:] != (v_dim, v_dim)
        or w_action.shape[:-2] != leading
        or w_action.shape[-2:] != (w_dim, w_dim)
        or u_drive.shape != (*leading, u_dim)
        or v_drive.shape != (*leading, v_dim)
        or w_drive.shape != (*leading, w_dim)
        or beta.shape != (w_dim, u_dim, v_dim)
    ):
        raise ValueError("incompatible triangular lift shapes")

    layout = lift_layout(u_dim, v_dim, w_dim)
    matrix = u_action.new_zeros(*leading, layout.dimension, layout.dimension)
    matrix[..., 0, 0] = 1
    matrix[..., layout.u, 0] = u_drive
    matrix[..., layout.u, layout.u] = u_action
    matrix[..., layout.v, 0] = v_drive
    matrix[..., layout.v, layout.v] = v_action

    tensor_constant = torch.einsum("...i,...j->...ij", u_drive, v_drive).flatten(-2)
    tensor_from_u = torch.einsum("...ik,...j->...ijk", u_action, v_drive).reshape(
        *leading, u_dim * v_dim, u_dim
    )
    tensor_from_v = torch.einsum("...i,...jl->...ijl", u_drive, v_action).reshape(
        *leading, u_dim * v_dim, v_dim
    )
    tensor_action = torch.einsum("...ik,...jl->...ijkl", u_action, v_action).reshape(
        *leading, u_dim * v_dim, u_dim * v_dim
    )
    matrix[..., layout.tensor, 0] = tensor_constant
    matrix[..., layout.tensor, layout.u] = tensor_from_u
    matrix[..., layout.tensor, layout.v] = tensor_from_v
    matrix[..., layout.tensor, layout.tensor] = tensor_action

    beta_flat = beta.reshape(w_dim, u_dim * v_dim)
    matrix[..., layout.w, 0] = w_drive + torch.einsum(
        "oi,...i->...o", beta_flat, tensor_constant
    )
    matrix[..., layout.w, layout.u] = torch.einsum(
        "oi,...ij->...oj", beta_flat, tensor_from_u
    )
    matrix[..., layout.w, layout.v] = torch.einsum(
        "oi,...ij->...oj", beta_flat, tensor_from_v
    )
    matrix[..., layout.w, layout.tensor] = torch.einsum(
        "oi,...ij->...oj", beta_flat, tensor_action
    )
    matrix[..., layout.w, layout.w] = w_action
    return matrix


def lifted_intertwiner_scan(
    u_action: torch.Tensor,
    u_drive: torch.Tensor,
    v_action: torch.Tensor,
    v_drive: torch.Tensor,
    w_action: torch.Tensor,
    w_drive: torch.Tensor,
    initial_u: torch.Tensor,
    initial_v: torch.Tensor,
    initial_w: torch.Tensor,
    beta: torch.Tensor,
    *,
    scan_backend: ScanBackend = "work_efficient",
) -> torch.Tensor:
    matrices = lifted_triangular_matrix(
        u_action,
        u_drive,
        v_action,
        v_drive,
        w_action,
        w_drive,
        beta,
    )
    prefixes = matrix_scan(matrices, backend=scan_backend)
    initial = lift_state(initial_u, initial_v, initial_w)
    return torch.einsum("blij,bj->bli", prefixes, initial)


def so3_cross_product_tensor(
    *, dtype: torch.dtype, device: torch.device | str | None = None
) -> torch.Tensor:
    beta = torch.zeros(3, 3, 3, dtype=dtype, device=device)
    beta[0, 1, 2], beta[1, 2, 0], beta[2, 0, 1] = 1, 1, 1
    beta[0, 2, 1], beta[1, 0, 2], beta[2, 1, 0] = -1, -1, -1
    return beta


def feedback_degree_growth(steps: int) -> dict[str, list[int]]:
    """Formal degree recurrences for two explicitly scoped feedback graphs.

    'one_source' means that 'w' enters one source linearly while the other
    source remains affine of degree one.  'two_source' means that 'w'
    enters both sources linearly.  The returned degrees assume that generic
    leading coefficients do not cancel.  They obstruct a fixed finite
    monomial linear lift; they are not a no-go theorem for arbitrary
    nonlinear coordinates or every possible associative algorithm.
    """

    triangular = []
    one_source_feedback = []
    two_source_feedback = []
    one_degree = 1
    two_degree = 1
    for _ in range(steps):
        triangular.append(2)
        one_degree += 1
        two_degree *= 2
        one_source_feedback.append(one_degree)
        two_source_feedback.append(two_degree)
    return {
        "triangular": triangular,
        "feedback_into_one_source": one_source_feedback,
        "feedback_into_both_sources": two_source_feedback,
    }


def diagnostics(seed: int = 20260806) -> dict[str, object]:
    torch.manual_seed(seed)
    dtype = torch.float64
    batch, length, dimension = 3, 31, 3
    beta = so3_cross_product_tensor(dtype=dtype)

    def random_actions() -> torch.Tensor:
        raw = torch.randn(batch, length, dimension, dimension, dtype=dtype)
        return torch.matrix_exp(0.08 * (raw - raw.transpose(-1, -2)))

    u_action, v_action, w_action = random_actions(), random_actions(), random_actions()
    u_drive = 0.07 * torch.randn(batch, length, dimension, dtype=dtype)
    v_drive = 0.07 * torch.randn(batch, length, dimension, dtype=dtype)
    w_drive = 0.04 * torch.randn(batch, length, dimension, dtype=dtype)
    initial_u = torch.randn(batch, dimension, dtype=dtype)
    initial_v = torch.randn(batch, dimension, dtype=dtype)
    initial_w = torch.randn(batch, dimension, dtype=dtype)

    staged = staged_intertwiner_scan(
        u_action,
        u_drive,
        v_action,
        v_drive,
        w_action,
        w_drive,
        initial_u,
        initial_v,
        initial_w,
        beta,
    )
    staged_hillis = staged_intertwiner_scan(
        u_action,
        u_drive,
        v_action,
        v_drive,
        w_action,
        w_drive,
        initial_u,
        initial_v,
        initial_w,
        beta,
        scan_backend="hillis_steele",
    )
    staged_affine_tree = staged_intertwiner_scan(
        u_action,
        u_drive,
        v_action,
        v_drive,
        w_action,
        w_drive,
        initial_u,
        initial_v,
        initial_w,
        beta,
        scan_backend="work_efficient_affine",
    )
    lifted = lifted_intertwiner_scan(
        u_action,
        u_drive,
        v_action,
        v_drive,
        w_action,
        w_drive,
        initial_u,
        initial_v,
        initial_w,
        beta,
    )
    layout = lift_layout(dimension, dimension, dimension)
    lifted_components = (
        lifted[..., layout.u],
        lifted[..., layout.v],
        lifted[..., layout.w],
    )

    sequential = recurrent_intertwiner_scan(
        u_action,
        u_drive,
        v_action,
        v_drive,
        w_action,
        w_drive,
        initial_u,
        initial_v,
        initial_w,
        beta,
    )

    staged_error = max(
        float((left - right).abs().max()) for left, right in zip(staged, sequential)
    )
    hillis_error = max(
        float((left - right).abs().max())
        for left, right in zip(staged_hillis, sequential)
    )
    backend_error = max(
        float((left - right).abs().max()) for left, right in zip(staged, staged_hillis)
    )
    affine_tree_error = max(
        float((left - right).abs().max())
        for left, right in zip(staged_affine_tree, sequential)
    )
    lift_error = max(
        float((left - right).abs().max())
        for left, right in zip(lifted_components, sequential)
    )

    skew = torch.randn(batch, dimension, dimension, dtype=dtype)
    rotation = torch.matrix_exp(0.4 * (skew - skew.transpose(-1, -2)))
    u = torch.randn(batch, dimension, dtype=dtype)
    v = torch.randn(batch, dimension, dtype=dtype)
    transformed = bilinear_contract(
        torch.einsum("bij,bj->bi", rotation, u),
        torch.einsum("bij,bj->bi", rotation, v),
        beta,
    )
    expected = torch.einsum("bij,bj->bi", rotation, bilinear_contract(u, v, beta))
    equivariance_error = float((transformed - expected).abs().max())

    checks = {
        "staged_scan_matches_recurrence": staged_error <= 1e-11,
        "hillis_steele_matches_recurrence": hillis_error <= 1e-11,
        "scan_backends_agree": backend_error <= 1e-11,
        "affine_tree_matches_recurrence": affine_tree_error <= 1e-11,
        "single_lift_matches_recurrence": lift_error <= 1e-11,
        "so3_cross_product_is_equivariant": equivariance_error <= 1e-11,
        "streaming_state_excludes_tensor_lift": 3 * dimension < layout.dimension,
        "feedback_degree_recurrence_matches_formula": feedback_degree_growth(8)[
            "feedback_into_both_sources"
        ][-1]
        == 256,
    }
    return {
        "experiment": "generic triangular Intertwiner SchurScan",
        "nonexceptional_instance": "SO(3) cross product",
        "dimensions": {"U": dimension, "V": dimension, "W": dimension},
        "streaming_cache_scalars": 3 * dimension,
        "homogeneous_proof_lift_scalars": layout.dimension,
        "parallel_scan_stages": 2,
        "default_scan_backend": "work_efficient",
        "staged_recurrent_max_abs_error": staged_error,
        "hillis_recurrent_max_abs_error": hillis_error,
        "backend_max_abs_difference": backend_error,
        "affine_tree_recurrent_max_abs_error": affine_tree_error,
        "lifted_recurrent_max_abs_error": lift_error,
        "equivariance_max_abs_error": equivariance_error,
        "degree_growth": feedback_degree_growth(8),
        "checks": checks,
        "claims": {
            "triangular_bilinear_drive_is_scan_compatible": checks[
                "staged_scan_matches_recurrence"
            ],
            "fixed_finite_homogeneous_lift_exists": checks[
                "single_lift_matches_recurrence"
            ],
            "construction_is_not_triality_specific": checks[
                "so3_cross_product_is_equivariant"
            ],
            "generic_cyclic_feedback_has_fixed_finite_monomial_linear_lift": False,
            "all_scan_algorithms_for_cyclic_feedback_ruled_out": False,
        },
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnostics()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
