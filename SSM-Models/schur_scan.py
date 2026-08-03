"""Representation-factored affine scans for symmetry-equivariant SSMs.

For real-type irreducible representations, the central construction is the
isotypic decomposition

    V = direct_sum_lambda (R**m_lambda tensor V_lambda).

For one shared group element ``g_t`` and arbitrary multiplicity-space maps
``M_t[lambda]``, a token transition acts as

    A_t = direct_sum_lambda (M_t[lambda] tensor rho_lambda(g_t)).

This family is closed under composition, so it supports an associative prefix
scan during training and a fixed-size recurrent state during inference.  The
Cl(3, 0) helpers below specialize this principle to rotor conjugation, whose
Spin(3) representation is ``1 + 3 + 3 + 1``.

For a general real representation, Schur's division algebra may instead be
``R``, ``C``, or ``H``; the complete multiplicity map then belongs to a matrix
algebra over that division algebra. The implemented Cl(3) sectors are real
type, so ordinary real matrices are complete here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


GA_DIM = 8


def pack_cl3_isotypic(multivector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack Cl(3) coefficients into trivial and vector isotypic components.

    Input shape is ``(..., channels, 8)``.  The returned shapes are
    ``(..., 2*channels)`` and ``(..., 2*channels, 3)``.  Bivectors are Hodge
    dualized from ``[e12,e13,e23]`` to ``[e23,-e13,e12]`` so they transform by
    the same proper-rotation matrix as vectors.
    """

    if multivector.ndim < 2 or multivector.shape[-1] != GA_DIM:
        raise ValueError("multivectors must have shape (..., channels, 8)")
    channels = multivector.shape[-2]
    trivial = torch.stack(
        (multivector[..., 0], multivector[..., 7]), dim=-1
    ).reshape(*multivector.shape[:-2], 2 * channels)
    dual_bivector = torch.stack(
        (
            multivector[..., 6],
            -multivector[..., 5],
            multivector[..., 4],
        ),
        dim=-1,
    )
    active = torch.stack((multivector[..., 1:4], dual_bivector), dim=-2)
    active = active.reshape(*multivector.shape[:-2], 2 * channels, 3)
    return trivial, active


def unpack_cl3_isotypic(
    trivial: torch.Tensor, active: torch.Tensor
) -> torch.Tensor:
    """Invert :func:`pack_cl3_isotypic`."""

    if trivial.shape[:-1] != active.shape[:-2] or active.shape[-1] != 3:
        raise ValueError("trivial and active isotypic shapes are incompatible")
    if trivial.shape[-1] != active.shape[-2] or trivial.shape[-1] % 2:
        raise ValueError("isotypic multiplicities must agree and be even")
    channels = trivial.shape[-1] // 2
    trivial = trivial.reshape(*trivial.shape[:-1], channels, 2)
    active = active.reshape(*active.shape[:-2], channels, 2, 3)
    vector, dual_bivector = active.unbind(dim=-2)
    output = trivial.new_zeros(*trivial.shape[:-1], GA_DIM)
    output[..., 0] = trivial[..., 0]
    output[..., 1:4] = vector
    output[..., 4] = dual_bivector[..., 2]
    output[..., 5] = -dual_bivector[..., 1]
    output[..., 6] = dual_bivector[..., 0]
    output[..., 7] = trivial[..., 1]
    return output


class Spin3IsotypicLinear(nn.Module):
    """The complete real linear commutant of Cl(3) rotor conjugation.

    ``GradeLinear`` is a valid but strict subfamily: it disallows scalar to
    pseudoscalar and vector to Hodge-dual-bivector mixing.  Schur's lemma permits
    arbitrary channel/copy mixing inside both repeated isotypic components.
    """

    def __init__(
        self, in_channels: int, out_channels: int, use_bias: bool = True
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError("channel counts must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.trivial_kernel = nn.Parameter(
            torch.empty(out_channels, 2, in_channels, 2)
        )
        self.active_kernel = nn.Parameter(
            torch.empty(out_channels, 2, in_channels, 2)
        )
        nn.init.kaiming_uniform_(self.trivial_kernel, a=5**0.5)
        nn.init.kaiming_uniform_(self.active_kernel, a=5**0.5)
        self.trivial_bias = (
            nn.Parameter(torch.zeros(out_channels, 2)) if use_bias else None
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-2:] != (self.in_channels, GA_DIM):
            raise ValueError("unexpected Spin3IsotypicLinear input shape")
        trivial, active = pack_cl3_isotypic(inputs)
        trivial = trivial.reshape(*trivial.shape[:-1], self.in_channels, 2)
        active = active.reshape(*active.shape[:-2], self.in_channels, 2, 3)
        trivial_output = torch.einsum(
            "ocid,...id->...oc", self.trivial_kernel, trivial
        )
        active_output = torch.einsum(
            "ocid,...idk->...ock", self.active_kernel, active
        )
        if self.trivial_bias is not None:
            trivial_output = trivial_output + self.trivial_bias
        return unpack_cl3_isotypic(
            trivial_output.flatten(-2), active_output.flatten(-3, -2)
        )


@dataclass(frozen=True)
class SchurAffineTransition:
    """Affine transition on the two Cl(3) isotypic components."""

    trivial_action: torch.Tensor
    active_multiplicity: torch.Tensor
    rotation: torch.Tensor
    trivial_drive: torch.Tensor
    active_drive: torch.Tensor


def apply_schur_affine(
    transition: SchurAffineTransition,
    state: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ``M0`` and ``M1 tensor R`` without materializing a Kronecker map."""

    trivial, active = state
    next_trivial = torch.einsum(
        "...ij,...j->...i", transition.trivial_action, trivial
    ) + transition.trivial_drive
    next_active = torch.einsum(
        "...ab,...bc,...dc->...ad",
        transition.active_multiplicity,
        active,
        transition.rotation,
    ) + transition.active_drive
    return next_trivial, next_active


def compose_schur_affine(
    after: SchurAffineTransition, before: SchurAffineTransition
) -> SchurAffineTransition:
    """Compose ``after(before(state))`` in the closed factored monoid."""

    transported_trivial, transported_active = apply_schur_affine(
        SchurAffineTransition(
            trivial_action=after.trivial_action,
            active_multiplicity=after.active_multiplicity,
            rotation=after.rotation,
            trivial_drive=torch.zeros_like(after.trivial_drive),
            active_drive=torch.zeros_like(after.active_drive),
        ),
        (before.trivial_drive, before.active_drive),
    )
    return SchurAffineTransition(
        trivial_action=after.trivial_action @ before.trivial_action,
        active_multiplicity=(
            after.active_multiplicity @ before.active_multiplicity
        ),
        rotation=after.rotation @ before.rotation,
        trivial_drive=after.trivial_drive + transported_trivial,
        active_drive=after.active_drive + transported_active,
    )


def associative_schur_scan(
    transition: SchurAffineTransition,
) -> SchurAffineTransition:
    """Inclusive Hillis-Steele prefix scan with logarithmic dependency depth."""

    if transition.trivial_action.ndim < 4:
        raise ValueError("transitions need batch and sequence dimensions")
    current = transition
    offset = 1
    length = transition.trivial_action.shape[1]
    while offset < length:
        after = SchurAffineTransition(
            *(value[:, offset:] for value in current.__dict__.values())
        )
        before = SchurAffineTransition(
            *(value[:, :-offset] for value in current.__dict__.values())
        )
        composed = compose_schur_affine(after, before)
        current = SchurAffineTransition(
            *(
                torch.cat((value[:, :offset], new_value), dim=1)
                for value, new_value in zip(
                    current.__dict__.values(), composed.__dict__.values()
                )
            )
        )
        offset *= 2
    return current


__all__ = [
    "GA_DIM",
    "SchurAffineTransition",
    "Spin3IsotypicLinear",
    "apply_schur_affine",
    "associative_schur_scan",
    "compose_schur_affine",
    "pack_cl3_isotypic",
    "unpack_cl3_isotypic",
]
