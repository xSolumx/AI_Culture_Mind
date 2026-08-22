"""Block-affine scan for independently transported Spin multiplicity copies.

The token factorization

    H'_a = sum_c L[a,c] R_c H_c + D_a

is globally equivariant and contractive when ``R_c`` and the multiplicity
factor of ``L`` are orthogonal.  Unlike the shared-action special case, this
factorization is not closed under composition.  Its exact associative closure
is the ordinary affine monoid on each representation's flattened ``C x D``
state, which this module materializes as a transparent semantic oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class IndependentBlockTransition:
    """Affine operators on one flattened multiplicity block per representation."""

    operator: torch.Tensor
    drive: torch.Tensor


def materialize_independent_block_operator(
    left: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
    """Return blocks ``A_r[a,i,c,j] = left[a,c] actions[c,r,i,j]``.

    ``left`` ends in ``(C,C)`` and ``actions`` ends in ``(C,R,D,D)``.  The
    result ends in ``(R,C*D,C*D)``.
    """

    if left.ndim < 2 or left.shape[-1] != left.shape[-2]:
        raise ValueError("left must end in (C,C)")
    if actions.ndim != left.ndim + 2:
        raise ValueError("actions must end in (C,R,D,D)")
    channels = left.shape[-1]
    if actions.shape[-4] != channels or actions.shape[-1] != actions.shape[-2]:
        raise ValueError("left and action multiplicity/state dimensions must agree")
    if left.shape[:-2] != actions.shape[:-4]:
        raise ValueError("left and actions must share leading dimensions")
    dimension = actions.shape[-1]
    blocks = torch.einsum("...ac,...crij->...raicj", left, actions)
    return blocks.reshape(*left.shape[:-2], actions.shape[-3], channels * dimension,
        channels * dimension)


def _flatten_state(state: torch.Tensor) -> torch.Tensor:
    # (...,C,R,D) -> (...,R,C*D)
    return state.movedim(-3, -2).flatten(start_dim=-2)


def _unflatten_state(flattened: torch.Tensor, channels: int) -> torch.Tensor:
    dimension = flattened.shape[-1] // channels
    # (...,R,C*D) -> (...,C,R,D)
    return flattened.unflatten(-1, (channels, dimension)).movedim(-3, -2)


def apply_independent_block_transition(
    transition: IndependentBlockTransition, state: torch.Tensor
) -> torch.Tensor:
    channels = transition.drive.shape[-3]
    flattened = _flatten_state(state)
    next_flattened = torch.einsum(
        "...rij,...rj->...ri", transition.operator, flattened
    )
    return _unflatten_state(next_flattened, channels) + transition.drive


def compose_independent_block_transition(
    later: IndependentBlockTransition,
    earlier: IndependentBlockTransition,
) -> IndependentBlockTransition:
    """Return ``later(earlier(H))`` in chronological order."""

    channels = later.drive.shape[-3]
    driven = torch.einsum(
        "...rij,...rj->...ri", later.operator, _flatten_state(earlier.drive)
    )
    return IndependentBlockTransition(
        operator=later.operator @ earlier.operator,
        drive=later.drive + _unflatten_state(driven, channels),
    )


def independent_block_prefix_scan(
    transition: IndependentBlockTransition,
) -> IndependentBlockTransition:
    """Inclusive logarithmic-depth Hillis--Steele scan on sequence axis one."""

    if transition.operator.ndim != 5:
        raise ValueError("operator must have shape (B,L,R,CD,CD)")
    if transition.drive.ndim != 5:
        raise ValueError("drive must have shape (B,L,C,R,D)")
    length = transition.operator.shape[1]
    if length < 1:
        raise ValueError("sequence length must be positive")
    current = transition
    offset = 1
    while offset < length:
        composed = compose_independent_block_transition(
            IndependentBlockTransition(
                current.operator[:, offset:], current.drive[:, offset:]
            ),
            IndependentBlockTransition(
                current.operator[:, :-offset], current.drive[:, :-offset]
            ),
        )
        current = IndependentBlockTransition(
            operator=torch.cat(
                (current.operator[:, :offset], composed.operator), dim=1
            ),
            drive=torch.cat((current.drive[:, :offset], composed.drive), dim=1),
        )
        offset *= 2
    return current


def recurrent_independent_block_scan(
    left: torch.Tensor,
    actions: torch.Tensor,
    drive: torch.Tensor,
    initial: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the structured token recurrence without dense operators."""

    state = initial
    states = []
    for position in range(left.shape[1]):
        rotated = torch.einsum(
            "...crij,...crj->...cri", actions[:, position], state
        )
        state = (
            torch.einsum("...ac,...cri->...ari", left[:, position], rotated)
            + drive[:, position]
        )
        states.append(state)
    stacked = torch.stack(states, dim=1)
    return stacked, state


def parallel_independent_block_scan(
    left: torch.Tensor,
    actions: torch.Tensor,
    drive: torch.Tensor,
    initial: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    transition = IndependentBlockTransition(
        materialize_independent_block_operator(left, actions), drive
    )
    prefixes = independent_block_prefix_scan(transition)
    states = apply_independent_block_transition(prefixes, initial[:, None])
    return states, states[:, -1]


__all__ = [
    "IndependentBlockTransition",
    "apply_independent_block_transition",
    "compose_independent_block_transition",
    "independent_block_prefix_scan",
    "materialize_independent_block_operator",
    "parallel_independent_block_scan",
    "recurrent_independent_block_scan",
]
