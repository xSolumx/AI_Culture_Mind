"""Exact factored scans for delta-rule matrix memory.

The recurrent state is a key-by-value matrix. Each token applies

    S' = L S R^T + B.

Standard delta overwrite has L = I - beta k k^T, R = I, and
B = beta k v^T. An orthogonal value transport is represented by a
nontrivial R. This factorization stays closed under composition and
avoids materializing a dense operator on vec(S).

The two-level chunked implementation is an eager correctness reference. It
is not the compact-WY fused kernel used by production DeltaNet systems.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeltaTransition:
    key_action: torch.Tensor
    value_action: torch.Tensor
    drive: torch.Tensor


def _validate_transition(transition: DeltaTransition) -> tuple[int, int, int, int]:
    key_action, value_action, drive = (
        transition.key_action,
        transition.value_action,
        transition.drive,
    )
    if (
        key_action.ndim != 4
        or value_action.ndim != 4
        or drive.ndim != 4
        or key_action.shape[:2] != value_action.shape[:2]
        or key_action.shape[:2] != drive.shape[:2]
        or key_action.shape[-1] != key_action.shape[-2]
        or value_action.shape[-1] != value_action.shape[-2]
        or key_action.shape[-1] != drive.shape[-2]
        or value_action.shape[-1] != drive.shape[-1]
    ):
        raise ValueError(
            "transition must have key_action (B,L,K,K), "
            "value_action (B,L,V,V), and drive (B,L,K,V)"
        )
    batch, length = drive.shape[:2]
    if length < 1:
        raise ValueError("scan length must be positive")
    return batch, length, drive.shape[-2], drive.shape[-1]


def compose_delta(after: DeltaTransition, before: DeltaTransition) -> DeltaTransition:
    """Compose chronological factored affine transitions.

    after acts after before. The formula is valid with arbitrary
    broadcast-compatible leading dimensions.
    """

    # Express L B R^T as batched matrix products.  Besides making the
    # transpose in the affine action explicit, this avoids the appreciable
    # equation-planning/dispatch overhead of a three-operand einsum for the
    # small matrices used by the memory scanners.
    transported_drive = (
        after.key_action @ before.drive @ after.value_action.transpose(-1, -2)
    )
    return DeltaTransition(
        key_action=after.key_action @ before.key_action,
        value_action=after.value_action @ before.value_action,
        drive=transported_drive + after.drive,
    )


def apply_delta(transition: DeltaTransition, state: torch.Tensor) -> torch.Tensor:
    """Apply L S R^T + B to a key-by-value state."""

    return (
        transition.key_action @ state @ transition.value_action.transpose(-1, -2)
        + transition.drive
    )


def identity_delta(
    batch: int,
    length: int,
    key_dimension: int,
    value_dimension: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> DeltaTransition:
    key_identity = torch.eye(key_dimension, dtype=dtype, device=device).reshape(
        1, 1, key_dimension, key_dimension
    )
    value_identity = torch.eye(value_dimension, dtype=dtype, device=device).reshape(
        1, 1, value_dimension, value_dimension
    )
    return DeltaTransition(
        key_action=key_identity.expand(batch, length, -1, -1),
        value_action=value_identity.expand(batch, length, -1, -1),
        drive=torch.zeros(
            batch,
            length,
            key_dimension,
            value_dimension,
            dtype=dtype,
            device=device,
        ),
    )


def delta_write_transitions(
    keys: torch.Tensor,
    values: torch.Tensor,
    beta: torch.Tensor,
) -> DeltaTransition:
    """Build standard corrective delta-write transitions."""

    if (
        keys.ndim != 3
        or values.ndim != 3
        or beta.ndim != 2
        or keys.shape[:2] != values.shape[:2]
        or keys.shape[:2] != beta.shape
    ):
        raise ValueError("keys, values, beta must have shapes (B,L,K), (B,L,V), (B,L)")
    batch, length, key_dimension = keys.shape
    value_dimension = values.shape[-1]
    key_identity = torch.eye(
        key_dimension, dtype=keys.dtype, device=keys.device
    ).reshape(1, 1, key_dimension, key_dimension)
    value_identity = torch.eye(
        value_dimension, dtype=values.dtype, device=values.device
    ).reshape(1, 1, value_dimension, value_dimension)
    return DeltaTransition(
        key_action=key_identity
        - beta[..., None, None] * keys[..., :, None] * keys[..., None, :],
        value_action=value_identity.expand(batch, length, -1, -1),
        drive=beta[..., None, None] * keys[..., :, None] * values[..., None, :],
    )


def value_transport_transitions(
    actions: torch.Tensor, *, key_dimension: int | None = None
) -> DeltaTransition:
    """Build transitions that rotate only the value axis."""

    if (
        actions.ndim != 4
        or actions.shape[-1] != actions.shape[-2]
        or actions.shape[1] < 1
    ):
        raise ValueError("actions must have shape (B,L,V,V) with positive length")
    batch, length, value_dimension, _ = actions.shape
    key_dimension = value_dimension if key_dimension is None else key_dimension
    return DeltaTransition(
        key_action=torch.eye(key_dimension, dtype=actions.dtype, device=actions.device)
        .reshape(1, 1, key_dimension, key_dimension)
        .expand(batch, length, -1, -1),
        value_action=actions,
        drive=torch.zeros(
            batch,
            length,
            key_dimension,
            value_dimension,
            dtype=actions.dtype,
            device=actions.device,
        ),
    )


def work_efficient_delta_scan(transition: DeltaTransition) -> DeltaTransition:
    """Inclusive ordered Blelloch scan of factored delta transitions."""

    batch, length, key_dimension, value_dimension = _validate_transition(transition)
    if length == 1:
        return transition
    padded_length = 1 << (length - 1).bit_length()
    if padded_length == length:
        leaves = transition
    else:
        padding = identity_delta(
            batch,
            padded_length - length,
            key_dimension,
            value_dimension,
            dtype=transition.drive.dtype,
            device=transition.drive.device,
        )
        leaves = DeltaTransition(
            key_action=torch.cat((transition.key_action, padding.key_action), dim=1),
            value_action=torch.cat(
                (transition.value_action, padding.value_action), dim=1
            ),
            drive=torch.cat((transition.drive, padding.drive), dim=1),
        )

    levels = [leaves]
    nodes = leaves
    while nodes.drive.shape[1] > 1:
        nodes = compose_delta(
            DeltaTransition(
                nodes.key_action[:, 1::2],
                nodes.value_action[:, 1::2],
                nodes.drive[:, 1::2],
            ),
            DeltaTransition(
                nodes.key_action[:, 0::2],
                nodes.value_action[:, 0::2],
                nodes.drive[:, 0::2],
            ),
        )
        levels.append(nodes)

    prefix = identity_delta(
        batch,
        1,
        key_dimension,
        value_dimension,
        dtype=transition.drive.dtype,
        device=transition.drive.device,
    )
    for children in reversed(levels[:-1]):
        left = DeltaTransition(
            children.key_action[:, 0::2],
            children.value_action[:, 0::2],
            children.drive[:, 0::2],
        )
        right_prefix = compose_delta(left, prefix)
        prefix = DeltaTransition(
            key_action=torch.stack(
                (prefix.key_action, right_prefix.key_action), dim=2
            ).reshape(batch, -1, key_dimension, key_dimension),
            value_action=torch.stack(
                (prefix.value_action, right_prefix.value_action), dim=2
            ).reshape(batch, -1, value_dimension, value_dimension),
            drive=torch.stack((prefix.drive, right_prefix.drive), dim=2).reshape(
                batch, -1, key_dimension, value_dimension
            ),
        )

    inclusive = compose_delta(leaves, prefix)
    return DeltaTransition(
        inclusive.key_action[:, :length],
        inclusive.value_action[:, :length],
        inclusive.drive[:, :length],
    )


def chunkwise_delta_scan(
    transition: DeltaTransition,
    *,
    chunk_size: int = 64,
) -> DeltaTransition:
    """Inclusive two-level chunked scan of the exact delta recurrence."""

    batch, length, key_dimension, value_dimension = _validate_transition(transition)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    chunks = (length + chunk_size - 1) // chunk_size
    padded_length = chunks * chunk_size
    if padded_length == length:
        leaves = transition
    else:
        padding = identity_delta(
            batch,
            padded_length - length,
            key_dimension,
            value_dimension,
            dtype=transition.drive.dtype,
            device=transition.drive.device,
        )
        leaves = DeltaTransition(
            torch.cat((transition.key_action, padding.key_action), dim=1),
            torch.cat((transition.value_action, padding.value_action), dim=1),
            torch.cat((transition.drive, padding.drive), dim=1),
        )

    def chunk_batch(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(batch, chunks, chunk_size, *tensor.shape[2:]).reshape(
            batch * chunks, chunk_size, *tensor.shape[2:]
        )

    local = work_efficient_delta_scan(
        DeltaTransition(
            chunk_batch(leaves.key_action),
            chunk_batch(leaves.value_action),
            chunk_batch(leaves.drive),
        )
    )
    totals = DeltaTransition(
        local.key_action[:, -1].reshape(batch, chunks, key_dimension, key_dimension),
        local.value_action[:, -1].reshape(
            batch, chunks, value_dimension, value_dimension
        ),
        local.drive[:, -1].reshape(batch, chunks, key_dimension, value_dimension),
    )
    if chunks == 1:
        before_chunks = identity_delta(
            batch,
            1,
            key_dimension,
            value_dimension,
            dtype=transition.drive.dtype,
            device=transition.drive.device,
        )
    else:
        chunk_prefix = work_efficient_delta_scan(totals)
        identity = identity_delta(
            batch,
            1,
            key_dimension,
            value_dimension,
            dtype=transition.drive.dtype,
            device=transition.drive.device,
        )
        before_chunks = DeltaTransition(
            torch.cat((identity.key_action, chunk_prefix.key_action[:, :-1]), dim=1),
            torch.cat(
                (identity.value_action, chunk_prefix.value_action[:, :-1]), dim=1
            ),
            torch.cat((identity.drive, chunk_prefix.drive[:, :-1]), dim=1),
        )

    def expand_before(tensor: torch.Tensor) -> torch.Tensor:
        return (
            tensor[:, :, None]
            .expand(batch, chunks, chunk_size, *tensor.shape[2:])
            .reshape(batch * chunks, chunk_size, *tensor.shape[2:])
        )

    global_prefix = compose_delta(
        local,
        DeltaTransition(
            expand_before(before_chunks.key_action),
            expand_before(before_chunks.value_action),
            expand_before(before_chunks.drive),
        ),
    )

    def restore(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(batch, padded_length, *tensor.shape[2:])[:, :length]

    return DeltaTransition(
        restore(global_prefix.key_action),
        restore(global_prefix.value_action),
        restore(global_prefix.drive),
    )


def recurrent_delta_states(
    transition: DeltaTransition,
    initial: torch.Tensor,
) -> torch.Tensor:
    """Sequential reference states for every token."""

    batch, length, key_dimension, value_dimension = _validate_transition(transition)
    if initial.shape != (batch, key_dimension, value_dimension):
        raise ValueError("initial must have shape (B,K,V)")
    state = initial
    states = []
    for position in range(length):
        state = apply_delta(
            DeltaTransition(
                transition.key_action[:, position],
                transition.value_action[:, position],
                transition.drive[:, position],
            ),
            state,
        )
        states.append(state)
    return torch.stack(states, dim=1)


def scanned_delta_states(
    transition: DeltaTransition,
    initial: torch.Tensor,
    *,
    backend: str = "chunkwise",
    chunk_size: int = 64,
) -> torch.Tensor:
    """Apply all prefixes from the selected exact scan backend."""

    if backend == "chunkwise":
        prefix = chunkwise_delta_scan(transition, chunk_size=chunk_size)
    elif backend == "work_efficient":
        prefix = work_efficient_delta_scan(transition)
    else:
        raise ValueError(f"unknown delta backend: {backend}")
    return apply_delta(prefix, initial[:, None])


def delta_read(state: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Read a value with query^T state."""

    if state.shape[:-2] != query.shape[:-1] or state.shape[-2] != query.shape[-1]:
        raise ValueError("state and query shapes are incompatible")
    return torch.einsum("...k,...kv->...v", query, state)
