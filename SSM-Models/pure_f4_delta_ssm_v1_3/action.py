"""Differentiable Lie-exponential actions for exceptional and custom algebras."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import nn

from .albert import ALBERT_DIM, build_albert_algebra

ExceptionalAlgebra = Literal["spin8", "spin9", "f4", "e6"]


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
    tangent = torch.einsum("...f,fij->...ij", coordinates, generators)
    return torch.matrix_exp(tangent)


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
        if generators.ndim != 3 or generators.shape[-1] != generators.shape[-2]:
            raise ValueError("generators must have shape (count,dimension,dimension)")
        self.register_buffer("generators", generators, persistent=True)

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
    "ExceptionalAction",
    "exponential_action",
    "ordered_exponential_action",
    "unrestricted_gl_generators",
]
