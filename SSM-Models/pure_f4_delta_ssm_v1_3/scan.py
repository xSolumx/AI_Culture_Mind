"""Associative two-sided affine scans with generalized delta updates."""

from __future__ import annotations

from dataclasses import dataclass

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
    "parallel_one_sided_delta_scan",
    "recurrent_delta_scan",
    "recurrent_one_sided_delta_scan",
    "transition_prefix_scan",
]
