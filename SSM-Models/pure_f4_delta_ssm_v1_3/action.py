"""Differentiable Lie-exponential actions for exceptional and custom algebras."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import nn

from .albert import ALBERT_DIM, build_albert_algebra

ExceptionalAlgebra = Literal[
    "identity", "g2", "spin7", "spin8", "spin9", "f4", "e6"
]
ActionGeometry = Literal["direct", "polar", "cartan"]


def exponential_action(
    coordinates: torch.Tensor, generators: torch.Tensor
) -> torch.Tensor:
    """Evaluate ``exp(sum_a coordinates[a] generators[a])``.

    This function intentionally imposes no coordinate clamp.  Stability policy
    belongs to the model/controller, while this semantic primitive represents
    the full connected one-parameter chart supplied by the generator bank.
    """

    if coordinates.shape[-1] != generators.shape[0]:
        raise ValueError("coordinate count must equal generator count")
    if generators.ndim != 3 or generators.shape[-1] != generators.shape[-2]:
        raise ValueError("generators must have shape (count,dimension,dimension)")
    return torch.matrix_exp(lie_tangent(coordinates, generators))


def lie_tangent(coordinates: torch.Tensor, generators: torch.Tensor) -> torch.Tensor:
    """Materialize a generator-bank tangent without exponentiating it."""

    if coordinates.shape[-1] != generators.shape[0]:
        raise ValueError("coordinate count must equal generator count")
    return torch.einsum("...f,fij->...ij", coordinates, generators)


def ordered_exponential_action(
    coordinates: torch.Tensor, generators: torch.Tensor
) -> torch.Tensor:
    """Evaluate an ordered product of exponential charts.

    ``coordinates`` ends in ``(factor_count,generator_count)``.  Factor zero
    acts first, so the returned column-vector action is
    ``exp(A_last) ... exp(A_0)``.  Products remove the assumption that a
    noncompact connected group is globally covered by one exponential.
    """

    if coordinates.ndim < 2 or coordinates.shape[-1] != generators.shape[0]:
        raise ValueError("coordinates must end in (factor_count,generator_count)")
    factors = exponential_action(coordinates, generators)
    dimension = generators.shape[-1]
    result = torch.eye(dimension, dtype=factors.dtype, device=factors.device)
    result = result.expand(*factors.shape[:-3], dimension, dimension)
    for index in range(factors.shape[-3]):
        result = factors[..., index, :, :] @ result
    return result


class ExceptionalAction(nn.Module):
    """A generator-bank action with built-in F4/E6 and custom-bank support."""

    def __init__(
        self,
        algebra: ExceptionalAlgebra = "e6",
        *,
        generators: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if generators is None:
            data = build_albert_algebra()
            arrays = {
                "g2": data.g2,
                "spin7": data.spin7,
                "spin8": data.spin8,
                "spin9": data.spin9,
                "f4": data.f4,
                "e6": data.e6,
            }
            array = arrays[algebra]
            generators = torch.as_tensor(array.copy(), dtype=torch.float64)
            self.algebra = algebra
        else:
            generators = torch.as_tensor(generators)
            self.algebra = "custom"
        self.geometry = "direct"
        if generators.ndim != 3 or generators.shape[-1] != generators.shape[-2]:
            raise ValueError("generators must have shape (count,dimension,dimension)")
        self.register_buffer("generators", generators, persistent=True)

    def log_operator_norm_bound(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Certified upper bound for ``log(||ordered(coordinates)||_2)``.

        Compact built-in actions are orthogonal.  For direct E6(-26), only the
        symmetric ``L(J_0)`` component can expand the Euclidean trace metric;
        the Frobenius norm bounds its logarithmic matrix norm.  Bounds add
        across ordered factors.  Custom banks deliberately return zero because
        no structure has been certified for them.
        """

        shape = coordinates.shape[:-2]
        if self.algebra != "e6":
            return coordinates.new_zeros(shape)
        content_coordinates = coordinates[..., 52:]
        content_generators = self.generators[52:].to(coordinates)
        symmetric_tangent = lie_tangent(content_coordinates, content_generators)
        return torch.linalg.matrix_norm(symmetric_tangent, ord="fro").sum(dim=-1)

    @property
    def coordinate_dim(self) -> int:
        return self.generators.shape[0]

    @property
    def representation_dim(self) -> int:
        return self.generators.shape[-1]

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return exponential_action(coordinates, self.generators.to(coordinates))

    def ordered(self, coordinates: torch.Tensor) -> torch.Tensor:
        return ordered_exponential_action(coordinates, self.generators.to(coordinates))


class IdentityAction(nn.Module):
    """Zero-coordinate 27D no-transport control."""

    algebra = "identity"
    geometry = "direct"
    coordinate_dim = 0
    representation_dim = ALBERT_DIM

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.shape[-1] != 0:
            raise ValueError("identity action has no coordinates")
        identity = torch.eye(
            self.representation_dim, dtype=coordinates.dtype, device=coordinates.device
        )
        return identity.expand(*coordinates.shape[:-1], self.representation_dim, self.representation_dim)

    def ordered(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.ndim < 2:
            raise ValueError("ordered coordinates need a factor axis")
        identity = torch.eye(
            self.representation_dim, dtype=coordinates.dtype, device=coordinates.device
        )
        return identity.expand(
            *coordinates.shape[:-2], self.representation_dim, self.representation_dim
        )

    def log_operator_norm_bound(self, coordinates: torch.Tensor) -> torch.Tensor:
        return coordinates.new_zeros(coordinates.shape[:-2])


class E6PolarAction(nn.Module):
    """Global Cartan/polar form ``K exp(P)`` for connected E6(-26).

    ``K`` is generated by the compact 52D F4 algebra and ``P`` by the 26D
    symmetric complement ``L(J_0)``.  This retains the same 78 controller
    coordinates as the direct Lie-algebra chart while making frame transport
    and content deformation separately observable.
    """

    algebra = "e6"
    geometry = "polar"

    def __init__(self) -> None:
        super().__init__()
        data = build_albert_algebra()
        self.register_buffer(
            "compact_generators", torch.tensor(data.f4, dtype=torch.float64)
        )
        self.register_buffer(
            "content_generators", torch.tensor(data.e6[52:], dtype=torch.float64)
        )

    @property
    def coordinate_dim(self) -> int:
        return 78

    @property
    def representation_dim(self) -> int:
        return ALBERT_DIM

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.shape[-1] != self.coordinate_dim:
            raise ValueError("E6 polar coordinates must have final dimension 78")
        compact_coordinates, content_coordinates = coordinates.split((52, 26), dim=-1)
        compact_tangent = lie_tangent(
            compact_coordinates, self.compact_generators.to(coordinates)
        )
        content_tangent = lie_tangent(
            content_coordinates, self.content_generators.to(coordinates)
        )
        if torch.is_grad_enabled() and coordinates.requires_grad:
            compact = torch.matrix_exp(compact_tangent)
            content = torch.matrix_exp(content_tangent)
        else:
            compact, content = torch.matrix_exp(
                torch.stack((compact_tangent, content_tangent), dim=-3)
            ).unbind(dim=-3)
        return compact @ content

    def ordered(self, coordinates: torch.Tensor) -> torch.Tensor:
        return _ordered_actions(coordinates, self.forward, self.representation_dim)

    def log_operator_norm_bound(self, coordinates: torch.Tensor) -> torch.Tensor:
        content_coordinates = coordinates[..., 52:]
        content_tangent = lie_tangent(
            content_coordinates, self.content_generators.to(coordinates)
        )
        return torch.linalg.matrix_norm(content_tangent, ord="fro").sum(dim=-1)


class E6CartanAction(nn.Module):
    """Full ``K_left A K_right`` form with an analytic rank-two radial action."""

    algebra = "e6"
    geometry = "cartan"

    def __init__(self) -> None:
        super().__init__()
        data = build_albert_algebra()
        radial = data.e6[52:54]
        off_diagonal = radial - np.einsum("fii,ij->fij", radial, np.eye(ALBERT_DIM))
        if np.max(np.abs(off_diagonal)) > 1e-14:
            raise AssertionError("the selected Albert Cartan generators must be diagonal")
        self.register_buffer(
            "compact_generators", torch.tensor(data.f4, dtype=torch.float64)
        )
        self.register_buffer(
            "radial_eigenvalues",
            torch.tensor(np.diagonal(radial, axis1=1, axis2=2).copy(), dtype=torch.float64),
        )

    @property
    def coordinate_dim(self) -> int:
        return 52 + 2 + 52

    @property
    def representation_dim(self) -> int:
        return ALBERT_DIM

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.shape[-1] != self.coordinate_dim:
            raise ValueError("E6 Cartan coordinates must have final dimension 106")
        left_coordinates, radial_coordinates, right_coordinates = coordinates.split(
            (52, 2, 52), dim=-1
        )
        generators = self.compact_generators.to(coordinates)
        left_tangent = lie_tangent(left_coordinates, generators)
        right_tangent = lie_tangent(right_coordinates, generators)
        if torch.is_grad_enabled() and coordinates.requires_grad:
            left = torch.matrix_exp(left_tangent)
            right = torch.matrix_exp(right_tangent)
        else:
            left, right = torch.matrix_exp(
                torch.stack((left_tangent, right_tangent), dim=-3)
            ).unbind(dim=-3)
        log_diagonal = torch.einsum(
            "...a,ai->...i", radial_coordinates, self.radial_eigenvalues.to(coordinates)
        )
        radial = torch.diag_embed(torch.exp(log_diagonal))
        return left @ radial @ right

    def ordered(self, coordinates: torch.Tensor) -> torch.Tensor:
        return _ordered_actions(coordinates, self.forward, self.representation_dim)

    def log_operator_norm_bound(self, coordinates: torch.Tensor) -> torch.Tensor:
        radial_coordinates = coordinates[..., 52:54]
        log_diagonal = torch.einsum(
            "...a,ai->...i",
            radial_coordinates,
            self.radial_eigenvalues.to(coordinates),
        )
        return log_diagonal.amax(dim=-1).clamp_min(0.0).sum(dim=-1)


def _ordered_actions(
    coordinates: torch.Tensor,
    constructor,
    dimension: int,
) -> torch.Tensor:
    if coordinates.ndim < 2:
        raise ValueError("ordered coordinates need a factor axis")
    result = torch.eye(dimension, dtype=coordinates.dtype, device=coordinates.device)
    result = result.expand(*coordinates.shape[:-2], dimension, dimension)
    for index in range(coordinates.shape[-2]):
        result = constructor(coordinates[..., index, :]) @ result
    return result


def build_exceptional_action(
    algebra: ExceptionalAlgebra,
    *,
    geometry: ActionGeometry = "direct",
    generators: torch.Tensor | np.ndarray | None = None,
) -> nn.Module:
    """Build an action without making geometry a hidden model assumption."""

    if generators is None and algebra == "identity":
        return IdentityAction()
    if generators is not None or algebra != "e6" or geometry == "direct":
        return ExceptionalAction(algebra, generators=generators)
    if geometry == "polar":
        return E6PolarAction()
    if geometry == "cartan":
        return E6CartanAction()
    raise ValueError(f"unknown action geometry {geometry!r}")


def unrestricted_gl_generators(
    dimension: int = ALBERT_DIM, *, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """Return the matrix-unit basis of ``gl(d)`` as an explicit falsifier.

    It is deliberately not a default: its ``d^2`` controller coordinates are
    expensive, but it provides a no-symmetry control without changing the scan.
    """

    if dimension < 1:
        raise ValueError("dimension must be positive")
    return torch.eye(dimension * dimension, dtype=dtype).reshape(
        dimension * dimension, dimension, dimension
    )


__all__ = [
    "E6CartanAction",
    "E6PolarAction",
    "ExceptionalAction",
    "IdentityAction",
    "build_exceptional_action",
    "exponential_action",
    "lie_tangent",
    "ordered_exponential_action",
    "unrestricted_gl_generators",
]
