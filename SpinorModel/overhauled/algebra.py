"""Tensor-only Cl(3, 0) algebra and associative rotor-affine transitions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

GA_DIM = 8
BASIS_MASKS = (0, 1, 2, 4, 3, 5, 6, 7)
GRADE_SLICES = ((0, 1), (1, 4), (4, 7), (7, 8))


def _blade_product_sign(left: int, right: int) -> int:
    swaps = sum(
        (right & ((1 << bit) - 1)).bit_count()
        for bit in range(3)
        if left & (1 << bit)
    )
    return -1 if swaps % 2 else 1


def _multiplication_table() -> torch.Tensor:
    table = torch.zeros(GA_DIM, GA_DIM, GA_DIM)
    lookup = {mask: index for index, mask in enumerate(BASIS_MASKS)}
    for left_index, left_mask in enumerate(BASIS_MASKS):
        for right_index, right_mask in enumerate(BASIS_MASKS):
            output = lookup[left_mask ^ right_mask]
            table[output, left_index, right_index] = _blade_product_sign(
                left_mask, right_mask
            )
    return table


MULTIPLICATION_TABLE = _multiplication_table()
REVERSION_SIGNS = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1])


def geometric_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compute a broadcasted Cl(3, 0) geometric product."""
    if left.shape[-1] != GA_DIM or right.shape[-1] != GA_DIM:
        raise ValueError(f"multivectors must end in {GA_DIM} components")
    table = MULTIPLICATION_TABLE.to(
        device=left.device, dtype=torch.result_type(left, right)
    )
    return torch.einsum("...i,...j,kij->...k", left, right, table)


def reversion(multivector: torch.Tensor) -> torch.Tensor:
    """Reverse a Cl(3, 0) multivector."""
    if multivector.shape[-1] != GA_DIM:
        raise ValueError(f"multivectors must end in {GA_DIM} components")
    return multivector * REVERSION_SIGNS.to(multivector)


def rotor_sandwich(rotor: torch.Tensor, multivector: torch.Tensor) -> torch.Tensor:
    """Apply the grade-preserving adjoint action ``r x reverse(r)``."""
    return geometric_product(geometric_product(rotor, multivector), reversion(rotor))


def identity_rotor(reference: torch.Tensor) -> torch.Tensor:
    shape = (*reference.shape[:-1], GA_DIM)
    rotor = torch.zeros(shape, dtype=reference.dtype, device=reference.device)
    rotor[..., 0] = 1.0
    return rotor


def rotor_from_bivector(
    bivector: torch.Tensor, *, maximum_angle: float = math.pi / 2
) -> torch.Tensor:
    """Exponentiate a 3D bivector with a smooth, bounded identity chart.

    ``torch.sinc`` supplies the analytic tangent limit at zero, so a controller
    initialized at the identity still receives nonzero gradients.
    """
    if bivector.shape[-1] != 3:
        raise ValueError("bivectors must end in three components")
    if maximum_angle <= 0:
        raise ValueError("maximum_angle must be positive")
    magnitude = bivector.norm(dim=-1, keepdim=True)
    safe_magnitude = magnitude.clamp_min(torch.finfo(bivector.dtype).eps)
    angle = maximum_angle * torch.tanh(magnitude)
    angle_over_magnitude = maximum_angle * torch.where(
        magnitude > 1e-6,
        torch.tanh(magnitude) / safe_magnitude,
        1.0 - magnitude.square() / 3.0,
    )
    tangent = 0.5 * angle_over_magnitude * torch.sinc(angle / (2.0 * math.pi))
    scalar = torch.cos(angle / 2.0)
    zeros = torch.zeros_like(scalar)
    rotor = torch.cat(
        [scalar, zeros, zeros, zeros, -tangent * bivector, zeros], dim=-1
    )
    return rotor / rotor.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(rotor.dtype).tiny
    )


@dataclass(frozen=True)
class RotorAffineTransition:
    """A step ``h -> retention * Ad_rotor(h) + drive``."""

    retention: torch.Tensor
    rotor: torch.Tensor
    drive: torch.Tensor

    def validate(self) -> None:
        if self.rotor.shape[-1] != GA_DIM or self.drive.shape[-1] != GA_DIM:
            raise ValueError("rotor and drive must contain Cl(3, 0) multivectors")
        if self.retention.shape != self.drive.shape[:-1]:
            raise ValueError("retention shape must equal drive shape without blades")
        if self.rotor.shape != self.drive.shape:
            raise ValueError("rotor and drive shapes must match")


def compose_transitions(
    later: RotorAffineTransition, earlier: RotorAffineTransition
) -> RotorAffineTransition:
    """Compose ``later(earlier(h))`` in chronological order."""
    later.validate()
    earlier.validate()
    return RotorAffineTransition(
        retention=later.retention * earlier.retention,
        rotor=geometric_product(later.rotor, earlier.rotor),
        drive=(
            later.retention.unsqueeze(-1)
            * rotor_sandwich(later.rotor, earlier.drive)
            + later.drive
        ),
    )


def apply_transition(
    transition: RotorAffineTransition, state: torch.Tensor
) -> torch.Tensor:
    transition.validate()
    return (
        transition.retention.unsqueeze(-1)
        * rotor_sandwich(transition.rotor, state)
        + transition.drive
    )


def associative_scan(transition: RotorAffineTransition) -> RotorAffineTransition:
    """Inclusive Hillis--Steele scan along sequence axis 1.

    The algorithm has logarithmic dependency depth and ``O(L log L)`` work.
    It is a correctness/reference backend; a production kernel should use a
    work-efficient fused scan while preserving the same composition law.
    """
    transition.validate()
    if transition.retention.ndim < 2:
        raise ValueError("scan transitions require batch and sequence axes")
    length = transition.retention.shape[1]
    if length < 1:
        raise ValueError("cannot scan an empty sequence")
    result = transition
    offset = 1
    while offset < length:
        combined = compose_transitions(
            RotorAffineTransition(
                result.retention[:, offset:],
                result.rotor[:, offset:],
                result.drive[:, offset:],
            ),
            RotorAffineTransition(
                result.retention[:, :-offset],
                result.rotor[:, :-offset],
                result.drive[:, :-offset],
            ),
        )
        result = RotorAffineTransition(
            torch.cat((result.retention[:, :offset], combined.retention), dim=1),
            torch.cat((result.rotor[:, :offset], combined.rotor), dim=1),
            torch.cat((result.drive[:, :offset], combined.drive), dim=1),
        )
        offset *= 2
    return result


__all__ = [
    "BASIS_MASKS",
    "GA_DIM",
    "GRADE_SLICES",
    "MULTIPLICATION_TABLE",
    "RotorAffineTransition",
    "apply_transition",
    "associative_scan",
    "compose_transitions",
    "geometric_product",
    "identity_rotor",
    "reversion",
    "rotor_from_bivector",
    "rotor_sandwich",
]
