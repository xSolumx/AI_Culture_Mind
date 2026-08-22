"""Associative affine scan for addressable Spin(8) delta memory.

The state has shape ``(G, M, R, 8)``: independent Spin transport heads,
addressable memory slots, and the three triality representations.  A token
acts from the right by a Spin(8) representation matrix and from the left by
an address-space contraction.  These actions commute because they act on
different tensor factors, so the resulting affine maps are closed under
composition and admit a parallel prefix scan.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpinDeltaTransition:
    """A map ``H -> left @ (action @ H) + drive``.

    ``left`` ends in ``(G,M,M)``, ``action`` in ``(G,R,D,D)``, and ``drive``
    in ``(G,M,R,D)``.  Any leading dimensions are batch or sequence axes.
    """

    left: torch.Tensor
    action: torch.Tensor
    drive: torch.Tensor


def apply_spin_delta(
    transition: SpinDeltaTransition, state: torch.Tensor
) -> torch.Tensor:
    """Apply one or a broadcast family of Spin-Delta affine maps."""

    rotated = torch.einsum(
        "...grij,...gmrj->...gmri", transition.action, state
    )
    mixed = torch.einsum("...gmn,...gnri->...gmri", transition.left, rotated)
    return mixed + transition.drive


def compose_spin_delta(
    later: SpinDeltaTransition, earlier: SpinDeltaTransition
) -> SpinDeltaTransition:
    """Compose chronological maps, returning ``later o earlier``."""

    rotated_drive = torch.einsum(
        "...grij,...gmrj->...gmri", later.action, earlier.drive
    )
    propagated_drive = torch.einsum(
        "...gmn,...gnri->...gmri", later.left, rotated_drive
    )
    return SpinDeltaTransition(
        left=later.left @ earlier.left,
        action=later.action @ earlier.action,
        drive=later.drive + propagated_drive,
    )


def recurrent_spin_delta_scan(
    transition: SpinDeltaTransition, initial_state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference sequential recurrence over sequence dimension one."""

    _validate_scan_inputs(transition, initial_state)
    state = initial_state
    states = []
    for position in range(transition.left.shape[1]):
        token = SpinDeltaTransition(
            transition.left[:, position],
            transition.action[:, position],
            transition.drive[:, position],
        )
        state = apply_spin_delta(token, state)
        states.append(state)
    sequence = torch.stack(states, dim=1)
    return sequence, state


def parallel_spin_delta_scan(
    transition: SpinDeltaTransition, initial_state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive Hillis-Steele prefix scan of the closed affine monoid."""

    _validate_scan_inputs(transition, initial_state)
    prefix = transition
    offset = 1
    length = transition.left.shape[1]
    while offset < length:
        later = SpinDeltaTransition(
            prefix.left[:, offset:],
            prefix.action[:, offset:],
            prefix.drive[:, offset:],
        )
        earlier = SpinDeltaTransition(
            prefix.left[:, :-offset],
            prefix.action[:, :-offset],
            prefix.drive[:, :-offset],
        )
        combined = compose_spin_delta(later, earlier)
        prefix = SpinDeltaTransition(
            torch.cat((prefix.left[:, :offset], combined.left), dim=1),
            torch.cat((prefix.action[:, :offset], combined.action), dim=1),
            torch.cat((prefix.drive[:, :offset], combined.drive), dim=1),
        )
        offset *= 2
    states = apply_spin_delta(prefix, initial_state[:, None])
    return states, states[:, -1]


def contractive_delta_left(
    scale: torch.Tensor, erase_key: torch.Tensor, erase_strength: torch.Tensor
) -> torch.Tensor:
    """Build ``scale * (I - beta ee^T)`` with ``||e||=1`` and ``beta in [0,1]``."""

    if scale.shape != erase_strength.shape:
        raise ValueError("scale and erase_strength must have the same shape")
    if erase_key.shape[:-1] != scale.shape:
        raise ValueError("erase_key must append one slot dimension")
    if torch.any((erase_strength < 0) | (erase_strength > 1)):
        raise ValueError("erase_strength must lie in [0,1]")
    unit_key = erase_key * torch.rsqrt(
        erase_key.square().sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(erase_key.dtype).tiny
        )
    )
    slots = erase_key.shape[-1]
    identity = torch.eye(
        slots, dtype=erase_key.dtype, device=erase_key.device
    )
    projector = unit_key.unsqueeze(-1) * unit_key.unsqueeze(-2)
    return scale[..., None, None] * (
        identity - erase_strength[..., None, None] * projector
    )


def route_delta_drive(write_key: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
    """Route a triality drive into the addressable slot axis."""

    if write_key.shape[:-1] != drive.shape[:-2]:
        raise ValueError("write_key and drive leading dimensions are incompatible")
    return write_key[..., :, None, None] * drive[..., None, :, :]


def read_delta_state(state: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Contract the slot axis, retaining head and triality axes."""

    if query.shape[:-1] != state.shape[:-3] or query.shape[-1] != state.shape[-3]:
        raise ValueError("query and state slot dimensions are incompatible")
    return torch.einsum("...gmri,...gm->...gri", state, query)


def _validate_scan_inputs(
    transition: SpinDeltaTransition, initial_state: torch.Tensor
) -> None:
    if transition.left.ndim != 5:
        raise ValueError("left must have shape (B,L,G,M,M)")
    batch, length, heads, slots, slots_again = transition.left.shape
    if length < 1 or slots != slots_again:
        raise ValueError("left must be a nonempty sequence of square slot maps")
    if transition.action.ndim != 6:
        raise ValueError("action must have shape (B,L,G,R,D,D)")
    representations = transition.action.shape[-3]
    dimension = transition.action.shape[-1]
    if transition.action.shape != (
        batch, length, heads, representations, dimension, dimension
    ):
        raise ValueError("action has incompatible shape")
    expected_drive = (batch, length, heads, slots, representations, dimension)
    if transition.drive.shape != expected_drive:
        raise ValueError("drive has incompatible shape")
    if initial_state.shape != (batch, heads, slots, representations, dimension):
        raise ValueError("initial_state has incompatible shape")


__all__ = [
    "SpinDeltaTransition",
    "apply_spin_delta",
    "compose_spin_delta",
    "contractive_delta_left",
    "parallel_spin_delta_scan",
    "read_delta_state",
    "recurrent_spin_delta_scan",
    "route_delta_drive",
]
