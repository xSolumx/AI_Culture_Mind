"""Optional projective-chart machinery for routing experiments.

The Riemann sphere is ``CP^1`` and ``PGL(2,C)`` acts by Möbius maps.  This is
useful as a tiny, exact projective-router control.  It is not used as the
Albert memory action: quotienting by scalar amplitude would erase information
needed by overwrite memory, and ``PGL(2,C)`` is not ``F4`` or ``E6(-26)``.
"""

from __future__ import annotations

import torch


def projective_normalize(homogeneous: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize homogeneous coordinates to unit norm without choosing a chart."""

    if homogeneous.shape[-1] < 2:
        raise ValueError("projective coordinates need at least two components")
    norm = torch.linalg.vector_norm(homogeneous, dim=-1, keepdim=True)
    return homogeneous / norm.clamp_min(eps)


def pgl_action(matrix: torch.Tensor, homogeneous: torch.Tensor) -> torch.Tensor:
    """Apply an invertible representative; scalar multiples act identically."""

    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("PGL representative must be square")
    if matrix.shape[-1] != homogeneous.shape[-1]:
        raise ValueError("matrix and homogeneous dimensions must agree")
    return projective_normalize(torch.einsum("...ij,...j->...i", matrix, homogeneous))


def complex_to_homogeneous(value: torch.Tensor) -> torch.Tensor:
    if not value.is_complex():
        raise ValueError("Riemann-sphere affine coordinates must be complex")
    return torch.stack((value, torch.ones_like(value)), dim=-1)


def mobius_action(matrix: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Apply a ``PGL(2,C)`` representative in the finite affine chart."""

    if matrix.shape[-2:] != (2, 2) or not matrix.is_complex() or not value.is_complex():
        raise ValueError("Möbius action requires complex 2x2 matrices and points")
    numerator = matrix[..., 0, 0] * value + matrix[..., 0, 1]
    denominator = matrix[..., 1, 0] * value + matrix[..., 1, 1]
    return numerator / denominator


def complex_to_riemann_sphere(value: torch.Tensor) -> torch.Tensor:
    """Inverse stereographic projection from ``C`` to the unit two-sphere."""

    if not value.is_complex():
        raise ValueError("value must be a complex tensor")
    radius_squared = value.real.square() + value.imag.square()
    denominator = 1.0 + radius_squared
    return torch.stack(
        (
            2.0 * value.real / denominator,
            2.0 * value.imag / denominator,
            (radius_squared - 1.0) / denominator,
        ),
        dim=-1,
    )


def riemann_sphere_to_complex(point: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Stereographic chart excluding the north pole ``(0,0,1)``."""

    if point.shape[-1] != 3 or point.is_complex():
        raise ValueError("sphere points must be real three-vectors")
    denominator = (1.0 - point[..., 2]).clamp_min(eps)
    return torch.complex(point[..., 0] / denominator, point[..., 1] / denominator)


__all__ = [
    "complex_to_homogeneous",
    "complex_to_riemann_sphere",
    "mobius_action",
    "pgl_action",
    "projective_normalize",
    "riemann_sphere_to_complex",
]
