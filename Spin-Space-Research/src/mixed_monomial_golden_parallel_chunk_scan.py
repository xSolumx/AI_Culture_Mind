"""Differentiable two-stage parallel scan for compiled N-H-N chunks.

The chronological primitive transitions of one labelled chunk are ``R, H, L``.
The chunk endpoint is ``L @ H @ R`` and the exact local-prefix operator is the
24-by-8 row stack ``[R; H@R; L@H@R]``.

The compiled parallel algorithm has two stages:

1. scan the ``C`` endpoint matrices to obtain every chunk's incoming state;
2. apply all ``C`` local-prefix operators in parallel and reshape to ``3C``
   causal states.

The primitive control scans all ``3C`` matrices.  Both Hillis--Steele and the
maintained ordered work-efficient Blelloch-style tree are provided.  Updates
are out of place, so PyTorch autograd can traverse every path.
"""

from __future__ import annotations

from typing import Literal

import torch

from mixed_monomial_golden_triton_local_prefix import (
    LocalPrefixBackend,
    indexed_local_prefix_states,
)

ScanBackend = Literal["work_efficient", "hillis_steele"]


def _validate_square_sequence(matrices: torch.Tensor) -> tuple[int, int, int]:
    if matrices.ndim != 4 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError("matrices must have shape (batch,length,D,D)")
    batch, length, dimension = matrices.shape[:3]
    if length < 1:
        raise ValueError("matrix sequence must be nonempty")
    return batch, length, dimension


def hillis_steele_matrix_scan(matrices: torch.Tensor) -> torch.Tensor:
    """Inclusive chronological prefix products with logarithmic depth."""

    _validate_square_sequence(matrices)
    prefixes = matrices
    offset = 1
    while offset < matrices.shape[1]:
        composed = prefixes[:, offset:] @ prefixes[:, :-offset]
        prefixes = torch.cat((prefixes[:, :offset], composed), dim=1)
        offset *= 2
    return prefixes


def work_efficient_matrix_scan(matrices: torch.Tensor) -> torch.Tensor:
    """Inclusive ordered Blelloch-style scan with fewer than ``3P`` products."""

    batch, length, dimension = _validate_square_sequence(matrices)
    if length == 1:
        return matrices
    padded_length = 1 << (length - 1).bit_length()
    identity = torch.eye(
        dimension,
        dtype=matrices.dtype,
        device=matrices.device,
    ).reshape(1, 1, dimension, dimension)
    if padded_length == length:
        leaves = matrices
    else:
        padding = identity.expand(batch, padded_length - length, -1, -1)
        leaves = torch.cat((matrices, padding), dim=1)

    levels = [leaves]
    nodes = leaves
    while nodes.shape[1] > 1:
        nodes = nodes[:, 1::2] @ nodes[:, 0::2]
        levels.append(nodes)

    exclusive = identity.expand(batch, 1, -1, -1)
    for children in reversed(levels[:-1]):
        left_totals = children[:, 0::2]
        right_exclusive = left_totals @ exclusive
        exclusive = torch.stack((exclusive, right_exclusive), dim=2).reshape(
            batch, -1, dimension, dimension
        )
    inclusive = leaves @ exclusive
    return inclusive[:, :length]


def matrix_prefix_scan(
    matrices: torch.Tensor,
    *,
    backend: ScanBackend = "work_efficient",
) -> torch.Tensor:
    if backend == "work_efficient":
        return work_efficient_matrix_scan(matrices)
    if backend == "hillis_steele":
        return hillis_steele_matrix_scan(matrices)
    raise ValueError("backend must be 'work_efficient' or 'hillis_steele'")


def compile_chunk_operators(
    left: torch.Tensor,
    middle: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiably construct endpoint and stacked local-prefix operators."""

    if (
        left.shape != middle.shape
        or left.shape != right.shape
        or left.ndim != 4
        or left.shape[-1] != left.shape[-2]
    ):
        raise ValueError("left, middle, right must share shape (B,C,D,D)")
    second = middle @ right
    endpoint = left @ second
    local_prefix = torch.cat((right, second, endpoint), dim=-2)
    return endpoint, local_prefix


def primitive_step_sequence(
    left: torch.Tensor,
    middle: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Return chronological primitive matrices ``R,H,L`` for every chunk."""

    if left.shape != middle.shape or left.shape != right.shape:
        raise ValueError("left, middle, right shapes must agree")
    batch, chunks, dimension, _ = left.shape
    return torch.stack((right, middle, left), dim=2).reshape(
        batch, 3 * chunks, dimension, dimension
    )


def apply_matrix_prefixes(
    prefixes: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    batch, _, dimension = prefixes.shape[:3]
    if initial_state.shape != (batch, dimension):
        raise ValueError("initial_state must have shape (batch,D)")
    return (prefixes @ initial_state[:, None, :, None]).squeeze(-1)


def primitive_parallel_states(
    left: torch.Tensor,
    middle: torch.Tensor,
    right: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    backend: ScanBackend = "work_efficient",
) -> torch.Tensor:
    """Scan all ``3C`` primitive transitions and emit every state."""

    steps = primitive_step_sequence(left, middle, right)
    return apply_matrix_prefixes(
        matrix_prefix_scan(steps, backend=backend), initial_state
    )


def primitive_recurrent_states(
    left: torch.Tensor,
    middle: torch.Tensor,
    right: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """Sequential semantic oracle for all ``3C`` causal states."""

    steps = primitive_step_sequence(left, middle, right)
    batch, _, dimension = steps.shape[:3]
    if initial_state.shape != (batch, dimension):
        raise ValueError("initial_state must have shape (batch,D)")
    value = initial_state
    outputs = []
    for position in range(steps.shape[1]):
        value = (steps[:, position] @ value[..., None]).squeeze(-1)
        outputs.append(value)
    return torch.stack(outputs, dim=1)


def compiled_parallel_states(
    endpoint: torch.Tensor,
    local_prefix: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    backend: ScanBackend = "work_efficient",
) -> torch.Tensor:
    """Two-stage scan of ``C`` endpoints plus parallel local-prefix expansion."""

    batch, chunks, dimension = _validate_square_sequence(endpoint)
    if local_prefix.shape != (batch, chunks, 3 * dimension, dimension):
        raise ValueError("local_prefix must have shape (B,C,3D,D)")
    if initial_state.shape != (batch, dimension):
        raise ValueError("initial_state must have shape (B,D)")

    endpoint_prefixes = matrix_prefix_scan(endpoint, backend=backend)
    if chunks == 1:
        incoming = initial_state[:, None]
    else:
        earlier_states = (
            endpoint_prefixes[:, :-1]
            @ initial_state[:, None, :, None]
        ).squeeze(-1)
        incoming = torch.cat((initial_state[:, None], earlier_states), dim=1)
    packed = (local_prefix @ incoming[..., None]).squeeze(-1)
    return packed.reshape(batch, chunks * 3, dimension)


def compiled_parallel_indexed_states(
    endpoint: torch.Tensor,
    prefix_table: torch.Tensor,
    left_index: torch.Tensor,
    middle_index: torch.Tensor,
    right_index: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    backend: ScanBackend = "work_efficient",
    local_backend: LocalPrefixBackend = "auto",
) -> torch.Tensor:
    """Scan endpoints, then fuse labelled prefix lookup and local expansion."""

    batch, chunks, dimension = _validate_square_sequence(endpoint)
    if dimension != 8:
        raise ValueError("the indexed exact prefix compiler has dimension 8")
    if initial_state.shape != (batch, dimension):
        raise ValueError("initial_state must have shape (B,8)")

    endpoint_prefixes = matrix_prefix_scan(endpoint, backend=backend)
    if chunks == 1:
        incoming = initial_state[:, None]
    else:
        earlier_states = (
            endpoint_prefixes[:, :-1]
            @ initial_state[:, None, :, None]
        ).squeeze(-1)
        incoming = torch.cat((initial_state[:, None], earlier_states), dim=1)
    return indexed_local_prefix_states(
        prefix_table,
        left_index,
        middle_index,
        right_index,
        incoming,
        backend=local_backend,
    )


def compiled_parallel_from_primitives(
    left: torch.Tensor,
    middle: torch.Tensor,
    right: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    backend: ScanBackend = "work_efficient",
) -> torch.Tensor:
    """Differentiable compilation followed by the two-stage parallel scan."""

    endpoint, local_prefix = compile_chunk_operators(left, middle, right)
    return compiled_parallel_states(
        endpoint,
        local_prefix,
        initial_state,
        backend=backend,
    )


def scan_composition_counts(chunk_count: int) -> dict[str, int]:
    """Return tree-product counts, excluding local-prefix application."""

    if chunk_count < 1:
        raise ValueError("chunk_count must be positive")

    def work_efficient(length: int) -> int:
        return 0 if length == 1 else 3 * (1 << (length - 1).bit_length()) - 2

    def hillis(length: int) -> int:
        total = 0
        offset = 1
        while offset < length:
            total += length - offset
            offset *= 2
        return total

    return {
        "primitive_length": 3 * chunk_count,
        "compiled_chunk_length": chunk_count,
        "primitive_work_efficient_products": work_efficient(3 * chunk_count),
        "compiled_work_efficient_products": work_efficient(chunk_count),
        "primitive_hillis_steele_products": hillis(3 * chunk_count),
        "compiled_hillis_steele_products": hillis(chunk_count),
    }


__all__ = [
    "ScanBackend",
    "apply_matrix_prefixes",
    "compile_chunk_operators",
    "compiled_parallel_from_primitives",
    "compiled_parallel_indexed_states",
    "compiled_parallel_states",
    "hillis_steele_matrix_scan",
    "matrix_prefix_scan",
    "primitive_parallel_states",
    "primitive_recurrent_states",
    "primitive_step_sequence",
    "scan_composition_counts",
    "work_efficient_matrix_scan",
]
