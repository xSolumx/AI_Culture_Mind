"""Exact co-moving-frame compilation of transported DeltaRule memory.

For row-oriented memory states, the transported recurrence is

    S_t = A_t S_{t-1} R_t^T + k_t v_t^T,
    A_t = I - beta_t k_t k_t^T.

For invertible actions let ``P_t = R_t ... R_1`` and define
``S_bar_t = S_t P_t^{-T}``.  Then

    S_bar_t = A_t S_bar_{t-1} + k_t (P_t^{-1} v_t)^T.

For orthogonal actions this reduces to the cheaper ``P_t^T v_t`` formula.
The transformed recurrence is an ordinary DeltaRule scan.  Physical-frame
reads are recovered as ``P_t (q_t^T S_bar_t)``.  The compilation is exact for
invertible actions, but a streaming implementation must retain the cumulative
action ``P_t`` in addition to the DeltaRule state; that extra state is part of
the contract.
"""

from __future__ import annotations

import torch

from intertwiner_schurscan import ScanBackend, matrix_scan
from schurscan_delta_memory import (
    delta_read,
    delta_write_transitions,
    scanned_delta_states,
)


def cumulative_actions(
    actions: torch.Tensor, *, backend: ScanBackend = "work_efficient"
) -> torch.Tensor:
    """Return inclusive chronological products ``R_t ... R_1``."""

    return matrix_scan(actions, backend=backend)


def values_to_comoving_frame(
    values: torch.Tensor,
    action_prefixes: torch.Tensor,
    *,
    assume_orthogonal: bool = False,
) -> torch.Tensor:
    """Return ``P_t^{-1} v_t``, or ``P_t^T v_t`` under an orthogonal promise."""

    if (
        values.ndim != 3
        or action_prefixes.ndim != 4
        or values.shape[:2] != action_prefixes.shape[:2]
        or action_prefixes.shape[-1] != action_prefixes.shape[-2]
        or values.shape[-1] != action_prefixes.shape[-1]
    ):
        raise ValueError(
            "values and action prefixes must have shapes (B,L,V) and (B,L,V,V)"
        )
    if assume_orthogonal:
        return torch.einsum(
            "blij,blj->bli", action_prefixes.transpose(-1, -2), values
        )
    return torch.linalg.solve(action_prefixes, values.unsqueeze(-1)).squeeze(-1)


def reads_to_physical_frame(
    reads: torch.Tensor, action_prefixes: torch.Tensor
) -> torch.Tensor:
    """Return ``P_t y_bar_t`` from co-moving-frame read vectors."""

    if (
        reads.ndim != 3
        or action_prefixes.ndim != 4
        or reads.shape[:2] != action_prefixes.shape[:2]
        or action_prefixes.shape[-1] != action_prefixes.shape[-2]
        or reads.shape[-1] != action_prefixes.shape[-1]
    ):
        raise ValueError(
            "reads and action prefixes must have shapes (B,L,V) and (B,L,V,V)"
        )
    return torch.einsum("blij,blj->bli", action_prefixes, reads)


def comoving_delta_read_sequence(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    actions: torch.Tensor,
    *,
    beta: torch.Tensor | None = None,
    action_backend: ScanBackend = "work_efficient",
    delta_backend: str = "chunkwise",
    delta_chunk_size: int = 64,
    assume_orthogonal: bool = False,
) -> torch.Tensor:
    """Compile transported memory into an ordinary DeltaRule scan and read it."""

    if keys.shape[:2] != values.shape[:2] or queries.shape[:2] != values.shape[:2]:
        raise ValueError("keys, values, and queries must share batch/length axes")
    if keys.shape != queries.shape:
        raise ValueError("keys and queries must have equal shapes")
    if beta is None:
        beta = torch.ones(values.shape[:2], dtype=values.dtype, device=values.device)
    if beta.shape != values.shape[:2]:
        raise ValueError("beta must have shape (batch, length)")

    prefixes = cumulative_actions(actions, backend=action_backend)
    transformed_values = values_to_comoving_frame(
        values, prefixes, assume_orthogonal=assume_orthogonal
    )
    transitions = delta_write_transitions(keys, transformed_values, beta)
    initial = torch.zeros(
        values.shape[0],
        keys.shape[-1],
        values.shape[-1],
        dtype=values.dtype,
        device=values.device,
    )
    states = scanned_delta_states(
        transitions,
        initial,
        backend=delta_backend,
        chunk_size=delta_chunk_size,
    )
    return reads_to_physical_frame(delta_read(states, queries), prefixes)


def recurrent_state_scalars(key_dimension: int, value_dimension: int) -> int:
    """Count DeltaRule state plus the cumulative value-space action."""

    if key_dimension < 1 or value_dimension < 1:
        raise ValueError("dimensions must be positive")
    return key_dimension * value_dimension + value_dimension * value_dimension
