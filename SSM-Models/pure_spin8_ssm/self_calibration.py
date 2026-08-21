"""Seven-probe reconstruction and an exact local Spin(8) lift.

Seven ordered vector images determine an oriented ``SO(8)`` action: their
Hodge cofactor is the missing eighth image.  An ordered adjacent-Givens QR
factorization then lifts that vector action through the maintained Spin(8)
generators.  The remaining sign is exactly the kernel of ``Spin(8) -> SO(8)``
and must be supplied as one lift-odd bit.

The Givens chart is differentiable on its regular cells.  No globally
continuous canonicalization is claimed; zero pivots use an explicit identity
convention and chart-boundary diagnostics remain the caller's responsibility.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from spin8_triality import (
    SPIN8_DIM,
    SPIN8_PAIRS,
    TRIALITY_REPRESENTATIONS,
)


def oriented_hodge_complement(probes: torch.Tensor) -> torch.Tensor:
    """Return the oriented unit vector complementary to seven probe rows."""

    if probes.shape[-2:] != (SPIN8_DIM - 1, SPIN8_DIM):
        raise ValueError("probes must end in shape (7,8)")
    cofactors = []
    for column in range(SPIN8_DIM):
        keep = [index for index in range(SPIN8_DIM) if index != column]
        minor = probes[..., :, keep]
        sign = -1.0 if (SPIN8_DIM - 1 + column) % 2 else 1.0
        cofactors.append(sign * torch.linalg.det(minor))
    complement = torch.stack(cofactors, dim=-1)
    norm = torch.linalg.vector_norm(complement, dim=-1, keepdim=True)
    return complement / norm.clamp_min(torch.finfo(probes.dtype).tiny)


def complete_oriented_so8_frame(
    probes: torch.Tensor,
    *,
    project: bool | Literal["polar", "qr"] = True,
) -> torch.Tensor:
    """Complete seven image vectors to an oriented ``SO(8)`` action.

    Probe rows are interpreted as the images of the first seven ordered basis
    vectors.  The returned matrix stores those images as columns.  ``project``
    uses the polar factor to denoise a full-rank approximate frame while
    enforcing determinant ``+1``.
    """

    complement = oriented_hodge_complement(probes)
    action = torch.cat((probes, complement[..., None, :]), dim=-2).transpose(-1, -2)
    if project is False:
        return action
    if project == "qr":
        orthogonal, triangular = torch.linalg.qr(action)
        diagonal = torch.diagonal(triangular, dim1=-2, dim2=-1)
        signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), 1.0)
        orthogonal = orthogonal * signs[..., None, :]
        orientation = torch.linalg.det(orthogonal)
        correction = torch.ones_like(signs)
        correction[..., -1] = orientation
        return orthogonal * correction[..., None, :]
    if project is not True and project != "polar":
        raise ValueError("project must be False, True/'polar', or 'qr'")
    left, _, right_t = torch.linalg.svd(action)
    orientation = torch.linalg.det(left @ right_t)
    correction = torch.ones(
        *orientation.shape, SPIN8_DIM, dtype=action.dtype, device=action.device
    )
    correction[..., -1] = orientation
    return left @ torch.diag_embed(correction) @ right_t


def _plane_factors(
    angle: torch.Tensor,
    generators: torch.Tensor,
    representations: Sequence[str],
    pair_index: int,
) -> torch.Tensor:
    """Return one plane factor in every selected triality representation."""

    identity = torch.eye(
        SPIN8_DIM, dtype=angle.dtype, device=angle.device
    )
    factors = []
    for representation_index, representation in enumerate(representations):
        generator = generators[representation_index, pair_index]
        if representation == "vector":
            factor = (
                identity
                + torch.sin(angle)[..., None, None] * generator
                + (1.0 - torch.cos(angle))[..., None, None]
                * (generator @ generator)
            )
        else:
            factor = (
                torch.cos(0.5 * angle)[..., None, None] * identity
                + 2.0
                * torch.sin(0.5 * angle)[..., None, None]
                * generator
            )
        factors.append(factor)
    return torch.stack(factors, dim=-3)


def lift_so8_action_from_givens(
    vector_action: torch.Tensor,
    generators: torch.Tensor,
    representations: Sequence[str] = TRIALITY_REPRESENTATIONS,
    *,
    lift_sign: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lift an ``SO(8)`` matrix through a deterministic adjacent-Givens chart.

    Returns ``(actions, triangular_residual)``.  ``actions`` ends in
    ``(representations,8,8)``.  The selected vector representation reconstructs
    ``vector_action``; a negative ``lift_sign`` multiplies both half-spin
    actions by ``-1`` and leaves the vector action unchanged.
    """

    representations = tuple(representations)
    if vector_action.shape[-2:] != (SPIN8_DIM, SPIN8_DIM):
        raise ValueError("vector_action must end in shape (8,8)")
    if "vector" not in representations:
        raise ValueError("the Givens lift requires the vector representation")
    expected = (len(representations), len(SPIN8_PAIRS), SPIN8_DIM, SPIN8_DIM)
    if generators.shape != expected:
        raise ValueError(f"generators must have shape {expected}")

    leading = vector_action.shape[:-2]
    identity = torch.eye(
        SPIN8_DIM, dtype=vector_action.dtype, device=vector_action.device
    )
    lifted = identity.expand(
        *leading, len(representations), SPIN8_DIM, SPIN8_DIM
    ).clone()
    triangular = vector_action

    # Left Givens elimination: E_k ... E_1 A = I, hence
    # A = E_1^{-1} ... E_k^{-1}.  Accumulate the same inverse factors in all
    # three representations to choose a concrete Spin lift.
    for column in range(SPIN8_DIM - 1):
        for lower in range(SPIN8_DIM - 1, column, -1):
            upper = lower - 1
            x = triangular[..., upper, column]
            y = triangular[..., lower, column]
            radius = torch.hypot(x, y)
            regular = radius > 16.0 * torch.finfo(vector_action.dtype).eps
            cosine = torch.where(regular, x / radius.clamp_min(1e-30), 1.0)
            sine = torch.where(regular, y / radius.clamp_min(1e-30), 0.0)
            elimination_angle = torch.atan2(sine, cosine)

            elimination = identity.expand(*leading, SPIN8_DIM, SPIN8_DIM).clone()
            elimination[..., upper, upper] = cosine
            elimination[..., upper, lower] = sine
            elimination[..., lower, upper] = -sine
            elimination[..., lower, lower] = cosine
            triangular = elimination @ triangular

            pair_index = SPIN8_PAIRS.index((upper, lower))
            inverse_factors = _plane_factors(
                -elimination_angle,
                generators,
                representations,
                pair_index,
            )
            lifted = lifted @ inverse_factors

    if lift_sign is not None:
        if lift_sign.shape != leading:
            raise ValueError(f"lift_sign must have shape {leading}")
        signs = []
        for representation in representations:
            signs.append(
                torch.ones_like(lift_sign)
                if representation == "vector"
                else lift_sign
            )
        lifted = lifted * torch.stack(signs, dim=-1)[..., :, None, None]
    return lifted, triangular


def spin8_actions_from_seven_probes(
    probes: torch.Tensor,
    generators: torch.Tensor,
    representations: Sequence[str] = TRIALITY_REPRESENTATIONS,
    *,
    lift_sign: torch.Tensor | None = None,
    project: bool | Literal["polar", "qr"] = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compile seven ordered vector probes and one lift bit to triality actions."""

    vector_action = complete_oriented_so8_frame(probes, project=project)
    return lift_so8_action_from_givens(
        vector_action,
        generators,
        representations,
        lift_sign=lift_sign,
    )


__all__ = [
    "complete_oriented_so8_frame",
    "lift_so8_action_from_givens",
    "oriented_hodge_complement",
    "spin8_actions_from_seven_probes",
]
