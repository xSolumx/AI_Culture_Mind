"""Experimental data-routed Spin(8) controller tier.

This module is deliberately separate from ``HybridMemoryLM``.  During
training, rung probabilities continuously regularize one canonical Spin(8)
coordinate bank; that soft interpolation is not claimed to be a discrete
subgroup selection.  With ``hard_eval=True``, evaluation uses exactly one
nested Spin(d) generator mask per token and channel.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations, pairwise
from typing import Literal, NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .selected_block import LowRankLinear
except ImportError:  # Support direct execution from this source directory.
    from selected_block import LowRankLinear

from pure_spin8_ssm.torch_backend import spin8_factorized_actions
from spin8_triality import TRIALITY_REPRESENTATIONS, torch_triality_generators

SPIN8_GENERATOR_PLANES = tuple(combinations(range(8), 2))
SPIN8_FACTOR_COUNT = len(SPIN8_GENERATOR_PLANES)
DEFAULT_RUNGS = (3, 4, 6, 8)

Reduction = Literal["none", "mean", "sum"]


def _validate_rungs(rungs: Sequence[int]) -> tuple[int, ...]:
    values = tuple(rungs)
    if not values:
        raise ValueError("rungs must be nonempty")
    if any(isinstance(rung, bool) or not isinstance(rung, int) for rung in values):
        raise TypeError("rungs must contain integers")
    if any(rung < 2 or rung > 8 for rung in values):
        raise ValueError("rung dimensions must be between 2 and 8")
    if any(left >= right for left, right in pairwise(values)):
        raise ValueError("rungs must be strictly increasing")
    return values


def subgroup_generator_indices(dimension: int) -> tuple[int, ...]:
    """Return global Spin(8) coordinate indices for the embedded Spin(d)."""

    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise TypeError("dimension must be an integer")
    if dimension < 2 or dimension > 8:
        raise ValueError("dimension must be between 2 and 8")
    return tuple(
        index
        for index, (left, right) in enumerate(SPIN8_GENERATOR_PLANES)
        if left < dimension and right < dimension
    )


def generator_mask_for_rung(
    dimension: int, *, device: torch.device | str | None = None
) -> torch.Tensor:
    """Return the exact 28-coordinate Boolean mask for one Spin(d) rung."""

    indices = subgroup_generator_indices(dimension)
    mask = torch.zeros(SPIN8_FACTOR_COUNT, dtype=torch.bool, device=device)
    mask[list(indices)] = True
    return mask


def rung_generator_masks(
    rungs: Sequence[int] = DEFAULT_RUNGS,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Stack the exact nested rung masks as ``(rungs, 28)``."""

    values = _validate_rungs(rungs)
    return torch.stack(
        [generator_mask_for_rung(rung, device=device) for rung in values]
    )


def cumulative_generator_gates(
    rung_probabilities: torch.Tensor,
    rungs: Sequence[int] = DEFAULT_RUNGS,
) -> torch.Tensor:
    """Mix nested masks into one cumulative gate for each generator.

    A plane receives the total probability mass of all rungs containing both
    of its coordinate axes.
    """

    values = _validate_rungs(rungs)
    if rung_probabilities.ndim < 1 or rung_probabilities.shape[-1] != len(values):
        raise ValueError("rung_probabilities must end in len(rungs)")
    if not rung_probabilities.is_floating_point():
        raise TypeError("rung_probabilities must be floating point")
    masks = rung_generator_masks(values, device=rung_probabilities.device).to(
        dtype=rung_probabilities.dtype
    )
    return rung_probabilities @ masks


def expected_factor_count(
    rung_probabilities: torch.Tensor,
    rungs: Sequence[int] = DEFAULT_RUNGS,
) -> torch.Tensor:
    """Return the expected number of active ordered factors per item."""

    values = _validate_rungs(rungs)
    if rung_probabilities.ndim < 1 or rung_probabilities.shape[-1] != len(values):
        raise ValueError("rung_probabilities must end in len(rungs)")
    counts = rung_probabilities.new_tensor([math.comb(rung, 2) for rung in values])
    return rung_probabilities @ counts


def _reduce(values: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def rung_entropy_regularizer(
    rung_probabilities: torch.Tensor, *, reduction: Reduction = "mean"
) -> torch.Tensor:
    """Entropy penalty for a rung simplex, with no module state or side effects."""

    if rung_probabilities.ndim < 1 or not rung_probabilities.is_floating_point():
        raise TypeError("rung_probabilities must be a floating-point tensor")
    tiny = torch.finfo(rung_probabilities.dtype).tiny
    entropy = -(rung_probabilities * rung_probabilities.clamp_min(tiny).log()).sum(
        dim=-1
    )
    return _reduce(entropy, reduction)


def rung_complexity_regularizer(
    rung_probabilities: torch.Tensor,
    rungs: Sequence[int] = DEFAULT_RUNGS,
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Expected active-factor penalty, measured on the unnormalized 0..28 scale."""

    return _reduce(expected_factor_count(rung_probabilities, rungs), reduction)


@dataclass(frozen=True)
class StructuredTierConfig:
    """Configuration for the isolated experimental Spin(8) controller."""

    model_dim: int
    channels: int = 1
    rungs: tuple[int, ...] = DEFAULT_RUNGS
    temperature: float = 1.0
    hard_eval: bool = True
    controller_rank: int | None = None

    def __post_init__(self) -> None:
        for name, value in (("model_dim", self.model_dim), ("channels", self.channels)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.rungs, tuple):
            raise TypeError("rungs must be a tuple")
        _validate_rungs(self.rungs)
        if isinstance(self.temperature, bool) or not isinstance(
            self.temperature, (int, float)
        ):
            raise TypeError("temperature must be a real number")
        if not math.isfinite(float(self.temperature)) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not isinstance(self.hard_eval, bool):
            raise TypeError("hard_eval must be a bool")
        if self.controller_rank is not None:
            if isinstance(self.controller_rank, bool) or not isinstance(
                self.controller_rank, int
            ):
                raise TypeError("controller_rank must be an integer or None")
            if self.controller_rank < 1:
                raise ValueError("controller_rank must be positive")
            controller_width = self.channels * SPIN8_FACTOR_COUNT
            if self.controller_rank > min(self.model_dim, controller_width):
                raise ValueError(
                    "controller_rank cannot exceed the coefficient controller dimensions"
                )


class StructuredTierOutput(NamedTuple):
    coordinates: torch.Tensor
    actions: torch.Tensor
    diagnostics: dict[str, torch.Tensor | str | bool]


class StructuredSpin8Tier(nn.Module):
    """Optional experimental per-token/per-channel Spin(8) controller tier."""

    def __init__(self, config: StructuredTierConfig) -> None:
        super().__init__()
        if not isinstance(config, StructuredTierConfig):
            raise TypeError("config must be a StructuredTierConfig")
        self.config = config
        self.rung_controller = nn.Linear(
            config.model_dim, config.channels * len(config.rungs), bias=True
        )
        coefficient_width = config.channels * SPIN8_FACTOR_COUNT
        if config.controller_rank is None:
            self.coefficient_controller: nn.Module = nn.Linear(
                config.model_dim, coefficient_width, bias=True
            )
        else:
            self.coefficient_controller = LowRankLinear(
                config.model_dim,
                coefficient_width,
                config.controller_rank,
                bias=True,
            )
        self.register_buffer(
            "generators", torch_triality_generators(TRIALITY_REPRESENTATIONS)
        )
        self.register_buffer(
            "rung_masks", rung_generator_masks(config.rungs), persistent=False
        )
        self.register_buffer(
            "rung_dimensions",
            torch.tensor(config.rungs, dtype=torch.long),
            persistent=False,
        )

    def forward(self, inputs: torch.Tensor) -> StructuredTierOutput:
        if inputs.ndim != 3 or inputs.shape[-1] != self.config.model_dim:
            raise ValueError("inputs must have shape (batch,length,model_dim)")
        batch, length, _ = inputs.shape
        rung_logits = self.rung_controller(inputs).reshape(
            batch, length, self.config.channels, len(self.config.rungs)
        )
        soft_probabilities = F.softmax(
            rung_logits / float(self.config.temperature), dim=-1
        )
        hard_selection = self.config.hard_eval and not self.training
        selected_indices = soft_probabilities.argmax(dim=-1)
        if hard_selection:
            probabilities = F.one_hot(
                selected_indices, num_classes=len(self.config.rungs)
            ).to(dtype=soft_probabilities.dtype)
        else:
            probabilities = soft_probabilities

        generator_gates = probabilities @ self.rung_masks.to(dtype=probabilities.dtype)
        raw_coordinates = self.coefficient_controller(inputs).reshape(
            batch, length, self.config.channels, SPIN8_FACTOR_COUNT
        )
        coordinates = raw_coordinates * generator_gates
        actions = spin8_factorized_actions(
            coordinates,
            self.generators.to(dtype=coordinates.dtype, device=coordinates.device),
            TRIALITY_REPRESENTATIONS,
        )
        selected_rung = self.rung_dimensions[selected_indices]
        diagnostics: dict[str, torch.Tensor | str | bool] = {
            "rung_probabilities": probabilities,
            "soft_rung_probabilities": soft_probabilities,
            "generator_gates": generator_gates,
            "expected_factor_count": expected_factor_count(
                probabilities, self.config.rungs
            ),
            "selected_rung": selected_rung,
            "hard_selection": hard_selection,
            "rung_regime": (
                "exact_rung" if hard_selection else "continuous_regularization"
            ),
            "soft_subgroup_claim": "none_soft_mixtures_are_not_discrete_subgroups",
        }
        return StructuredTierOutput(coordinates, actions, diagnostics)


__all__ = [
    "DEFAULT_RUNGS",
    "SPIN8_FACTOR_COUNT",
    "SPIN8_GENERATOR_PLANES",
    "StructuredSpin8Tier",
    "StructuredTierConfig",
    "StructuredTierOutput",
    "cumulative_generator_gates",
    "expected_factor_count",
    "generator_mask_for_rung",
    "rung_complexity_regularizer",
    "rung_entropy_regularizer",
    "rung_generator_masks",
    "subgroup_generator_indices",
]
