"""Associative two-sided affine scans with generalized delta updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class TwoSidedAffineTransition:
    """The map ``S -> left @ S @ right.T + bias``."""

    left: torch.Tensor
    right: torch.Tensor
    bias: torch.Tensor


@dataclass(frozen=True)
class OneSidedAffineTransition:
    """The transport-free map ``S -> left @ S + bias``."""

    left: torch.Tensor
    bias: torch.Tensor


def compose_one_sided_transition(
    later: OneSidedAffineTransition, earlier: OneSidedAffineTransition
) -> OneSidedAffineTransition:
    """Compose chronological transport-free transitions."""

    return OneSidedAffineTransition(
        left=later.left @ earlier.left,
        bias=later.bias + later.left @ earlier.bias,
    )


def compose_transition(
    later: TwoSidedAffineTransition, earlier: TwoSidedAffineTransition
) -> TwoSidedAffineTransition:
    """Compose chronological transitions: ``later(earlier(S))``."""

    return TwoSidedAffineTransition(
        left=later.left @ earlier.left,
        right=later.right @ earlier.right,
        bias=later.bias + later.left @ earlier.bias @ later.right.transpose(-1, -2),
    )


def compile_delta_transition(
    retention: torch.Tensor,
    write_key: torch.Tensor,
    erase_key: torch.Tensor,
    write_value: torch.Tensor,
    action: torch.Tensor,
) -> TwoSidedAffineTransition:
    """Compile a rank-r independent erase/write update into an affine map.

    Shapes are ``retention (...,H)``, keys ``(...,R,H)``, values
    ``(...,R,V)``, and action ``(...,V,V)``.  The update is

    ``S' = (I - sum_r k_r e_r^T) diag(retention) S action^T``
    ``     + sum_r k_r z_r^T``.

    Setting ``erase_key = beta * write_key`` recovers a block DeltaRule.
    Independent keys are supported because the local memory evidence did not
    justify tying erase and write as a universal architectural law.
    """

    if write_key.shape != erase_key.shape:
        raise ValueError("write and erase keys must have identical shapes")
    if write_key.ndim < 2 or write_key.shape[-1] != retention.shape[-1]:
        raise ValueError("keys must end in (rank,key_dimension)")
    if write_value.shape[:-1] != write_key.shape[:-1]:
        raise ValueError("write values must share key leading and rank axes")
    if action.shape[-1] != action.shape[-2] or action.shape[-1] != write_value.shape[-1]:
        raise ValueError("action and write-value dimensions must agree")
    key_dimension = retention.shape[-1]
    identity = torch.eye(key_dimension, dtype=retention.dtype, device=retention.device)
    erase = torch.einsum("...rh,...rk->...hk", write_key, erase_key)
    # Multiplication by diag(retention) scales the columns.
    left = (identity - erase) * retention.unsqueeze(-2)
    bias = torch.einsum("...rh,...rv->...hv", write_key, write_value)
    return TwoSidedAffineTransition(left=left, right=action, bias=bias)


def compile_one_sided_delta_transition(
    retention: torch.Tensor,
    write_key: torch.Tensor,
    erase_key: torch.Tensor,
    write_value: torch.Tensor,
) -> OneSidedAffineTransition:
    """Compile the same rank-r update without materializing identity transport."""

    if write_key.shape != erase_key.shape:
        raise ValueError("write and erase keys must have identical shapes")
    if write_key.ndim < 2 or write_key.shape[-1] != retention.shape[-1]:
        raise ValueError("keys must end in (rank,key_dimension)")
    if write_value.shape[:-1] != write_key.shape[:-1]:
        raise ValueError("write values must share key leading and rank axes")
    key_dimension = retention.shape[-1]
    identity = torch.eye(key_dimension, dtype=retention.dtype, device=retention.device)
    erase = torch.einsum("...rh,...rk->...hk", write_key, erase_key)
    left = (identity - erase) * retention.unsqueeze(-2)
    bias = torch.einsum("...rh,...rv->...hv", write_key, write_value)
    return OneSidedAffineTransition(left=left, bias=bias)


def transition_prefix_scan(
    transition: TwoSidedAffineTransition,
) -> TwoSidedAffineTransition:
    """Inclusive Hillis-Steele scan along axis 1."""

    if transition.left.ndim < 4:
        raise ValueError("transitions must include batch and sequence axes")
    length = transition.left.shape[1]
    current = transition
    offset = 1
    while offset < length:
        later = TwoSidedAffineTransition(
            current.left[:, offset:], current.right[:, offset:], current.bias[:, offset:]
        )
        earlier = TwoSidedAffineTransition(
            current.left[:, :-offset], current.right[:, :-offset], current.bias[:, :-offset]
        )
        composed = compose_transition(later, earlier)
        current = TwoSidedAffineTransition(
            left=torch.cat((current.left[:, :offset], composed.left), dim=1),
            right=torch.cat((current.right[:, :offset], composed.right), dim=1),
            bias=torch.cat((current.bias[:, :offset], composed.bias), dim=1),
        )
        offset *= 2
    return current


def one_sided_transition_prefix_scan(
    transition: OneSidedAffineTransition,
) -> OneSidedAffineTransition:
    """Inclusive Hillis-Steele scan without a value-space identity matrix."""

    if transition.left.ndim < 4:
        raise ValueError("transitions must include batch and sequence axes")
    length = transition.left.shape[1]
    current = transition
    offset = 1
    while offset < length:
        later = OneSidedAffineTransition(
            current.left[:, offset:], current.bias[:, offset:]
        )
        earlier = OneSidedAffineTransition(
            current.left[:, :-offset], current.bias[:, :-offset]
        )
        composed = compose_one_sided_transition(later, earlier)
        current = OneSidedAffineTransition(
            left=torch.cat((current.left[:, :offset], composed.left), dim=1),
            bias=torch.cat((current.bias[:, :offset], composed.bias), dim=1),
        )
        offset *= 2
    return current


def apply_transition(
    transition: TwoSidedAffineTransition, state: torch.Tensor
) -> torch.Tensor:
    return transition.left @ state @ transition.right.transpose(-1, -2) + transition.bias


def apply_one_sided_transition(
    transition: OneSidedAffineTransition, state: torch.Tensor
) -> torch.Tensor:
    return transition.left @ state + transition.bias


def _read(states: torch.Tensor, query: torch.Tensor | None) -> torch.Tensor | None:
    if query is None:
        return None
    if states.shape[:-2] != query.shape[:-1] or states.shape[-2] != query.shape[-1]:
        raise ValueError("query must match state batch/sequence and key axes")
    return torch.einsum("...hv,...h->...v", states, query)


def recurrent_delta_scan(
    transition: TwoSidedAffineTransition,
    initial_state: torch.Tensor,
    query: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Sequential semantic oracle returning reads, every state, and final state."""

    state = initial_state
    states = []
    for position in range(transition.left.shape[1]):
        local = TwoSidedAffineTransition(
            transition.left[:, position],
            transition.right[:, position],
            transition.bias[:, position],
        )
        state = apply_transition(local, state)
        states.append(state)
    stacked = torch.stack(states, dim=1)
    return _read(stacked, query), stacked, state


def recurrent_one_sided_delta_scan(
    transition: OneSidedAffineTransition,
    initial_state: torch.Tensor,
    query: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Sequential transport-free oracle."""

    state = initial_state
    states = []
    for position in range(transition.left.shape[1]):
        local = OneSidedAffineTransition(
            transition.left[:, position], transition.bias[:, position]
        )
        state = apply_one_sided_transition(local, state)
        states.append(state)
    stacked = torch.stack(states, dim=1)
    return _read(stacked, query), stacked, state


def direct_recurrent_delta_scan(
    retention: torch.Tensor,
    write_key: torch.Tensor,
    erase_key: torch.Tensor,
    write_value: torch.Tensor,
    initial_state: torch.Tensor,
    query: torch.Tensor | None = None,
    action: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Exact bounded-memory rank-r recurrence without dense transition prefixes.

    This is the semantic form targeted by the SM75 kernel.  It avoids
    materializing the ``H x H`` erase matrix and, unlike Hillis--Steele, never
    stores ``log2(length)`` copies of the full ``V x V`` right action.  The
    Python loop is a correctness/long-context backend, not a fused speed claim.
    """

    if write_key.shape != erase_key.shape:
        raise ValueError("write and erase keys must have identical shapes")
    if write_value.shape[:-1] != write_key.shape[:-1]:
        raise ValueError("write values must share key leading and rank axes")
    if retention.shape[:2] != write_key.shape[:2]:
        raise ValueError("retention and keys must share batch/sequence axes")
    if action is not None and (
        action.shape[:2] != retention.shape[:2]
        or action.shape[-2:] != (write_value.shape[-1], write_value.shape[-1])
    ):
        raise ValueError("action must match batch, sequence, and value dimensions")
    state = initial_state
    states = []
    for position in range(retention.shape[1]):
        state = retention[:, position, :, None] * state
        erased_values = torch.einsum(
            "brh,bhv->brv", erase_key[:, position], state
        )
        state = state - torch.einsum(
            "brh,brv->bhv", write_key[:, position], erased_values
        )
        if action is not None:
            state = state @ action[:, position].transpose(-1, -2)
        state = state + torch.einsum(
            "brh,brv->bhv", write_key[:, position], write_value[:, position]
        )
        states.append(state)
    stacked = torch.stack(states, dim=1)
    return _read(stacked, query), stacked, state


def segmented_primitive_delta_scan(
    retention: torch.Tensor,
    write_key: torch.Tensor,
    erase_key: torch.Tensor,
    write_value: torch.Tensor,
    initial_state: torch.Tensor,
    query: torch.Tensor | None,
    event_positions: tuple[int, ...],
    event_coordinates: torch.Tensor,
    action_apply: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Exact sparse-event scan without materializing dense value actions.

    Identity spans use the one-sided associative scan. At each declared event
    the Delta edit is evaluated directly and the primitive action is applied
    between erase and write, matching ``compile_delta_transition``.
    """

    length = retention.shape[1]
    if tuple(sorted(set(event_positions))) != event_positions:
        raise ValueError("event positions must be unique and sorted")
    if any(position < 0 or position >= length for position in event_positions):
        raise ValueError("event position lies outside the sequence")
    if event_coordinates.shape[:2] != (
        retention.shape[0],
        len(event_positions),
    ):
        raise ValueError("event coordinates must have shape (batch,events,factors)")

    transition = compile_one_sided_delta_transition(
        retention, write_key, erase_key, write_value
    )
    state = initial_state
    state_chunks: list[torch.Tensor] = []
    read_chunks: list[torch.Tensor] = []
    cursor = 0
    for event_index, position in enumerate(event_positions):
        if position > cursor:
            span = OneSidedAffineTransition(
                transition.left[:, cursor:position],
                transition.bias[:, cursor:position],
            )
            span_query = None if query is None else query[:, cursor:position]
            span_reads, span_states, state = parallel_one_sided_delta_scan(
                span, state, span_query
            )
            state_chunks.append(span_states)
            if span_reads is not None:
                read_chunks.append(span_reads)

        state = retention[:, position, :, None] * state
        erased_values = torch.einsum(
            "brh,bhv->brv", erase_key[:, position], state
        )
        state = state - torch.einsum(
            "brh,brv->bhv", write_key[:, position], erased_values
        )
        state = action_apply(state, event_coordinates[:, event_index])
        state = state + torch.einsum(
            "brh,brv->bhv", write_key[:, position], write_value[:, position]
        )
        event_state = state[:, None]
        state_chunks.append(event_state)
        if query is not None:
            event_read = _read(event_state, query[:, position : position + 1])
            if event_read is None:
                raise AssertionError("event query unexpectedly returned no read")
            read_chunks.append(event_read)
        cursor = position + 1

    if cursor < length:
        span = OneSidedAffineTransition(
            transition.left[:, cursor:], transition.bias[:, cursor:]
        )
        span_query = None if query is None else query[:, cursor:]
        span_reads, span_states, state = parallel_one_sided_delta_scan(
            span, state, span_query
        )
        state_chunks.append(span_states)
        if span_reads is not None:
            read_chunks.append(span_reads)

    states = torch.cat(state_chunks, dim=1)
    reads = torch.cat(read_chunks, dim=1) if query is not None else None
    return reads, states, state


def parallel_delta_scan(
    transition: TwoSidedAffineTransition,
    initial_state: torch.Tensor,
    query: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Log-depth semantic prefix implementation with complete autograd."""

    prefix = transition_prefix_scan(transition)
    initial = initial_state[:, None]
    states = apply_transition(prefix, initial)
    return _read(states, query), states, states[:, -1]


def parallel_one_sided_delta_scan(
    transition: OneSidedAffineTransition,
    initial_state: torch.Tensor,
    query: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Log-depth transport-free scan with complete autograd."""

    prefix = one_sided_transition_prefix_scan(transition)
    states = apply_one_sided_transition(prefix, initial_state[:, None])
    return _read(states, query), states, states[:, -1]


def parallel_chunked_primitive_delta_scan(
    retention: torch.Tensor,
    write_key: torch.Tensor,
    erase_key: torch.Tensor,
    write_value: torch.Tensor,
    initial_state: torch.Tensor,
    query: torch.Tensor | None,
    event_actions: torch.Tensor,
    *,
    event_stride: int,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Exact log-depth scan for end-of-chunk two-sided event actions.

    Delta edits act on the head axis while exceptional transport acts on the
    value axis.  A stride-sized block therefore compiles to the separable map
    ``S -> A @ S @ R.T + B``.  The block maps are scanned associatively, then
    the token states inside all blocks are reconstructed in parallel.
    """

    if retention.ndim != 3 or event_stride < 1:
        raise ValueError("retention must be (B,L,H) and stride must be positive")
    batch, length, heads = retention.shape
    if length % event_stride:
        raise ValueError("length must be divisible by the end-event stride")
    if write_key.shape != erase_key.shape or write_key.ndim != 4:
        raise ValueError("keys must share shape (B,L,R,H)")
    rank = write_key.shape[2]
    value_dim = write_value.shape[-1]
    if write_key.shape != (batch, length, rank, heads):
        raise ValueError("key shape mismatch")
    if write_value.shape != (batch, length, rank, value_dim):
        raise ValueError("write value shape mismatch")
    if initial_state.shape != (batch, heads, value_dim):
        raise ValueError("initial state shape mismatch")
    if query is not None and query.shape != retention.shape:
        raise ValueError("query must match retention")
    chunks = length // event_stride
    if event_actions.shape != (batch, chunks, value_dim, value_dim):
        raise ValueError("event actions must be (B,L/stride,V,V)")

    transition = compile_one_sided_delta_transition(
        retention, write_key, erase_key, write_value
    )
    left = transition.left.reshape(
        batch, chunks, event_stride, heads, heads
    )
    bias = transition.bias.reshape(
        batch, chunks, event_stride, heads, value_dim
    )
    event_left = left[:, :, -1]
    event_bias = bias[:, :, -1]

    if event_stride == 1:
        chunk_transition = TwoSidedAffineTransition(
            left=event_left,
            right=event_actions,
            bias=event_bias,
        )
        pre_prefix = None
    else:
        pre_length = event_stride - 1
        pre_transition = OneSidedAffineTransition(
            left=left[:, :, :pre_length].reshape(
                batch * chunks, pre_length, heads, heads
            ),
            bias=bias[:, :, :pre_length].reshape(
                batch * chunks, pre_length, heads, value_dim
            ),
        )
        pre_prefix = one_sided_transition_prefix_scan(pre_transition)
        pre_final_left = pre_prefix.left[:, -1].reshape(
            batch, chunks, heads, heads
        )
        pre_final_bias = pre_prefix.bias[:, -1].reshape(
            batch, chunks, heads, value_dim
        )
        chunk_transition = TwoSidedAffineTransition(
            left=event_left @ pre_final_left,
            right=event_actions,
            bias=(
                event_bias
                + event_left @ pre_final_bias @ event_actions.transpose(-1, -2)
            ),
        )

    chunk_prefix = transition_prefix_scan(chunk_transition)
    chunk_ends = apply_transition(chunk_prefix, initial_state[:, None])
    chunk_starts = torch.cat((initial_state[:, None], chunk_ends[:, :-1]), dim=1)

    if event_stride == 1:
        states = chunk_ends[:, :, None]
    else:
        if pre_prefix is None:
            raise AssertionError("pre-event prefix was not constructed")
        flat_starts = chunk_starts.reshape(batch * chunks, heads, value_dim)
        pre_states = apply_one_sided_transition(
            pre_prefix, flat_starts[:, None]
        ).reshape(batch, chunks, event_stride - 1, heads, value_dim)
        states = torch.cat((pre_states, chunk_ends[:, :, None]), dim=2)
    states = states.reshape(batch, length, heads, value_dim)
    return _read(states, query), states, chunk_ends[:, -1]


__all__ = [
    "OneSidedAffineTransition",
    "TwoSidedAffineTransition",
    "apply_one_sided_transition",
    "apply_transition",
    "compile_delta_transition",
    "compile_one_sided_delta_transition",
    "compose_one_sided_transition",
    "compose_transition",
    "direct_recurrent_delta_scan",
    "one_sided_transition_prefix_scan",
    "parallel_delta_scan",
    "parallel_chunked_primitive_delta_scan",
    "parallel_one_sided_delta_scan",
    "segmented_primitive_delta_scan",
    "recurrent_delta_scan",
    "recurrent_one_sided_delta_scan",
    "transition_prefix_scan",
]
