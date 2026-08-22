"""Associative two-sided affine scan for shared-action Spin isotypic copies."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CoupledIsotypicTransition:
    """The replicated map ``H_r -> left @ H_r @ right_r.T + drive_r``."""

    left: torch.Tensor
    right: torch.Tensor
    drive: torch.Tensor


def _linear_apply(
    left: torch.Tensor, right: torch.Tensor, state: torch.Tensor
) -> torch.Tensor:
    mixed = torch.einsum("...cd,...drj->...crj", left, state)
    return torch.einsum("...rij,...crj->...cri", right, mixed)


def apply_coupled_transition(
    transition: CoupledIsotypicTransition, state: torch.Tensor
) -> torch.Tensor:
    return _linear_apply(transition.left, transition.right, state) + transition.drive


def compose_coupled_transition(
    later: CoupledIsotypicTransition,
    earlier: CoupledIsotypicTransition,
) -> CoupledIsotypicTransition:
    """Return ``later(earlier(H))`` in chronological order."""

    return CoupledIsotypicTransition(
        left=later.left @ earlier.left,
        right=later.right @ earlier.right,
        drive=later.drive + _linear_apply(later.left, later.right, earlier.drive),
    )


def coupled_transition_prefix_scan(
    transition: CoupledIsotypicTransition,
) -> CoupledIsotypicTransition:
    """Inclusive logarithmic-depth Hillis-Steele scan on sequence axis one."""

    if transition.left.ndim != 4:
        raise ValueError("left transitions must have shape (B,L,C,C)")
    if transition.right.ndim != 5:
        raise ValueError("right transitions must have shape (B,L,R,D,D)")
    if transition.drive.ndim != 5:
        raise ValueError("drive must have shape (B,L,C,R,D)")
    length = transition.left.shape[1]
    if length < 1:
        raise ValueError("sequence length must be positive")
    current = transition
    offset = 1
    while offset < length:
        later = CoupledIsotypicTransition(
            current.left[:, offset:],
            current.right[:, offset:],
            current.drive[:, offset:],
        )
        earlier = CoupledIsotypicTransition(
            current.left[:, :-offset],
            current.right[:, :-offset],
            current.drive[:, :-offset],
        )
        composed = compose_coupled_transition(later, earlier)
        current = CoupledIsotypicTransition(
            left=torch.cat((current.left[:, :offset], composed.left), dim=1),
            right=torch.cat((current.right[:, :offset], composed.right), dim=1),
            drive=torch.cat((current.drive[:, :offset], composed.drive), dim=1),
        )
        offset *= 2
    return current


def recurrent_coupled_scan(
    transition: CoupledIsotypicTransition, initial: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial
    states = []
    for position in range(transition.left.shape[1]):
        state = apply_coupled_transition(
            CoupledIsotypicTransition(
                transition.left[:, position],
                transition.right[:, position],
                transition.drive[:, position],
            ),
            state,
        )
        states.append(state)
    stacked = torch.stack(states, dim=1)
    return stacked, state


def parallel_coupled_scan(
    transition: CoupledIsotypicTransition, initial: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    prefixes = coupled_transition_prefix_scan(transition)
    states = apply_coupled_transition(prefixes, initial[:, None])
    return states, states[:, -1]


def contractive_givens_left(
    scale: torch.Tensor,
    angles: torch.Tensor,
    pairs: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    """Return ``diag(scale) @ Q`` for an ordered orthogonal Givens product."""

    if scale.ndim < 1:
        raise ValueError("scale must have a multiplicity axis")
    channels = scale.shape[-1]
    if angles.shape != (*scale.shape[:-1], len(pairs)):
        raise ValueError("angles must have one coordinate per Givens pair")
    identity = torch.eye(channels, dtype=scale.dtype, device=scale.device)
    rotation = identity.expand(*scale.shape[:-1], channels, channels)
    for factor, (left, right) in enumerate(pairs):
        if not 0 <= left < right < channels:
            raise ValueError("Givens pairs must be ordered channel indices")
        cosine = torch.cos(angles[..., factor])
        sine = torch.sin(angles[..., factor])
        givens = identity.expand(*scale.shape[:-1], channels, channels).clone()
        givens[..., left, left] = cosine
        givens[..., left, right] = -sine
        givens[..., right, left] = sine
        givens[..., right, right] = cosine
        rotation = givens @ rotation
    return scale[..., :, None] * rotation


__all__ = [
    "CoupledIsotypicTransition",
    "apply_coupled_transition",
    "compose_coupled_transition",
    "contractive_givens_left",
    "coupled_transition_prefix_scan",
    "parallel_coupled_scan",
    "recurrent_coupled_scan",
]
