"""Bounded recurrent memory driven by the learned structured Spin(8) tier.

The recurrent cache has shape ``(B, C, 3, 8)`` in the canonical vector,
positive-spinor, and negative-spinor representations.  It is initialized to
zero; no learned state is hidden outside that explicit cache.  Spin(8) action
construction and affine composition are delegated to the maintained
``pure_spin8_ssm`` backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from pure_spin8_ssm.torch_backend import (
    Spin8AffineTransition,
    apply_spin8_affine,
    mask_spin8_transition,
    recurrent_spin8_scan,
    unit_ball,
    work_efficient_spin8_scan,
)

try:
    from .selected_block import LowRankLinear
    from .structured_tier import (
        DEFAULT_RUNGS,
        StructuredSpin8Tier,
        StructuredTierConfig,
    )
except ImportError:  # Support direct execution from this source directory.
    from selected_block import LowRankLinear
    from structured_tier import (
        DEFAULT_RUNGS,
        StructuredSpin8Tier,
        StructuredTierConfig,
    )


TRIALITY_SECTORS = 3
SPIN8_DIM = 8
CONTROL_SCALARS_PER_CHANNEL = 2 + TRIALITY_SECTORS * SPIN8_DIM

ScanMode = Literal["recurrent", "parallel"]
DiagnosticValue = torch.Tensor | str | bool


def _dtype_bytes(dtype: torch.dtype) -> int:
    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype")
    try:
        probe = torch.empty((), dtype=dtype)
    except (RuntimeError, TypeError) as error:
        raise TypeError("dtype must have fixed-width elements") from error
    if not probe.is_floating_point():
        raise TypeError("state dtype must be floating point")
    return probe.element_size()


@dataclass(frozen=True)
class StructuredMemoryConfig:
    """Configuration for :class:`StructuredSpin8Memory`.

    ``retention_max`` is strictly below one.  Together with the unit-ball
    drive and zero initial cache, this bounds every triality-sector norm by
    one for an arbitrarily long stream.  ``controller_rank`` constrains both
    the transport-coordinate controller in :class:`StructuredSpin8Tier` and
    the retention/write/drive controller here.
    """

    model_dim: int
    channels: int = 1
    rungs: tuple[int, ...] = DEFAULT_RUNGS
    controller_rank: int | None = None
    retention_min: float = 0.0
    retention_max: float = 0.99
    hard_eval: bool = True

    def __post_init__(self) -> None:
        for name, value in (("model_dim", self.model_dim), ("channels", self.channels)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.rungs, tuple):
            raise TypeError("rungs must be a tuple")
        if not isinstance(self.hard_eval, bool):
            raise TypeError("hard_eval must be a bool")
        for name, value in (
            ("retention_min", self.retention_min),
            ("retention_max", self.retention_max),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.retention_min <= self.retention_max < 1.0:
            raise ValueError(
                "retention bounds must satisfy 0 <= retention_min <= retention_max < 1"
            )
        if self.controller_rank is not None:
            if isinstance(self.controller_rank, bool) or not isinstance(
                self.controller_rank, int
            ):
                raise TypeError("controller_rank must be an integer or None")
            if self.controller_rank < 1:
                raise ValueError("controller_rank must be positive")
            if self.controller_rank > min(self.model_dim, self.controller_width):
                raise ValueError(
                    "controller_rank cannot exceed the controller dimensions"
                )

        # Reuse the tier's validation for exact rung ordering and dimensions.
        StructuredTierConfig(
            model_dim=self.model_dim,
            channels=self.channels,
            rungs=self.rungs,
            hard_eval=self.hard_eval,
            controller_rank=self.controller_rank,
        )

    @property
    def controller_width(self) -> int:
        return self.channels * CONTROL_SCALARS_PER_CHANNEL

    @property
    def state_shape(self) -> tuple[int, int, int]:
        return (self.channels, TRIALITY_SECTORS, SPIN8_DIM)

    @property
    def state_scalars(self) -> int:
        return self.channels * TRIALITY_SECTORS * SPIN8_DIM

    @property
    def retention_bounds(self) -> tuple[float, float]:
        return (self.retention_min, self.retention_max)

    def state_bytes(
        self, dtype: torch.dtype = torch.float32, *, batch_size: int = 1
    ) -> int:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return batch_size * self.state_scalars * _dtype_bytes(dtype)


def parallel_spin8_scan(
    transition: Spin8AffineTransition, initial_state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the maintained work-efficient affine prefix scan."""

    prefixes = work_efficient_spin8_scan(transition)
    states = apply_spin8_affine(prefixes, initial_state[:, None])
    return states, states[:, -1]


class StructuredSpin8Memory(nn.Module):
    """Bounded Spin(8) recurrent mixer with an always-live state readout.

    For every token and channel the exact transition is

    ``state' = retention * action(state) + drive``

    where ``action`` comes from :class:`StructuredSpin8Tier` and
    ``drive = (1 - retention) * write_gate * unit_ball(raw_drive)``.
    There is no read event: every post-transition state is represented by its
    smoothly squashed direction and per-sector log energy and projected to
    ``model_dim``.
    """

    scan_modes = ("recurrent", "parallel")

    def __init__(self, config: StructuredMemoryConfig) -> None:
        super().__init__()
        if not isinstance(config, StructuredMemoryConfig):
            raise TypeError("config must be a StructuredMemoryConfig")
        self.config = config
        self.tier = StructuredSpin8Tier(
            StructuredTierConfig(
                model_dim=config.model_dim,
                channels=config.channels,
                rungs=config.rungs,
                hard_eval=config.hard_eval,
                controller_rank=config.controller_rank,
            )
        )
        if config.controller_rank is None:
            self.controller: nn.Module = nn.Linear(
                config.model_dim, config.controller_width, bias=True
            )
        else:
            self.controller = LowRankLinear(
                config.model_dim,
                config.controller_width,
                config.controller_rank,
                bias=True,
            )
        readout_width = config.channels * TRIALITY_SECTORS * (SPIN8_DIM + 1)
        self.output_projection = nn.Linear(readout_width, config.model_dim, bias=False)

    @property
    def state_scalars(self) -> int:
        return self.config.state_scalars

    @property
    def cache_scalars(self) -> int:
        return self.state_scalars

    def state_bytes(
        self, dtype: torch.dtype | None = None, *, batch_size: int = 1
    ) -> int:
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return self.config.state_bytes(dtype, batch_size=batch_size)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return the explicit zero triality cache; it has no learned component."""

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        parameter = next(self.parameters())
        if device is None:
            device = parameter.device
        if dtype is None:
            dtype = parameter.dtype
        _dtype_bytes(dtype)
        return torch.zeros(
            batch_size, *self.config.state_shape, device=device, dtype=dtype
        )

    def compile_transition(
        self,
        inputs: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[Spin8AffineTransition, dict[str, DiagnosticValue]]:
        """Form the exact affine transition and its controller diagnostics."""

        mask = self._validate_sequence_inputs(inputs, valid_mask)
        return self._build_transition(inputs, mask)

    def transitions(
        self,
        inputs: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> Spin8AffineTransition:
        """Return transitions without retaining the diagnostic dictionary."""

        transition, _ = self.compile_transition(inputs, valid_mask=valid_mask)
        return transition

    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        scan_mode: ScanMode = "parallel",
        return_diagnostics: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[
            torch.Tensor,
            torch.Tensor,
            dict[str, DiagnosticValue],
        ]
    ):
        if scan_mode not in self.scan_modes:
            raise ValueError(f"scan_mode must be one of {self.scan_modes}")
        mask = self._validate_sequence_inputs(inputs, valid_mask)
        state = self._validate_state(inputs, state)
        transition, diagnostics = self._build_transition(inputs, mask)
        if scan_mode == "recurrent":
            states, final_state = recurrent_spin8_scan(transition, state)
        else:
            states, final_state = parallel_spin8_scan(transition, state)
        outputs = self.readout(states)
        if return_diagnostics:
            return outputs, final_state, diagnostics
        return outputs, final_state

    def step(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[
            torch.Tensor,
            torch.Tensor,
            dict[str, DiagnosticValue],
        ]
    ):
        """Run one recurrent token step and return the complete next cache."""

        if inputs.ndim != 2:
            raise ValueError("step inputs must have shape (B,model_dim)")
        sequence_mask = None
        if valid_mask is not None:
            if valid_mask.ndim != 1:
                raise ValueError("step valid_mask must have shape (B,)")
            sequence_mask = valid_mask[:, None]
        result = self.forward(
            inputs[:, None],
            state,
            valid_mask=sequence_mask,
            scan_mode="recurrent",
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            outputs, final_state, diagnostics = result
            batch_size = inputs.shape[0]
            squeezed = {
                name: (
                    value[:, 0]
                    if isinstance(value, torch.Tensor)
                    and value.ndim >= 2
                    and value.shape[:2] == (batch_size, 1)
                    else value
                )
                for name, value in diagnostics.items()
            }
            return outputs[:, 0], final_state, squeezed
        outputs, final_state = result
        return outputs[:, 0], final_state

    def readout(self, states: torch.Tensor) -> torch.Tensor:
        """Project smoothly bounded directions and per-sector log energies."""

        if states.ndim < 4 or states.shape[-3:] != self.config.state_shape:
            raise ValueError("states must end in (channels,3,8)")
        if not states.is_floating_point():
            raise TypeError("states must have a floating-point dtype")
        energy = states.square().sum(dim=-1, keepdim=True)
        direction = states * torch.rsqrt(1.0 + energy)
        features = torch.cat((direction, torch.log1p(energy)), dim=-1)
        return self.output_projection(features.flatten(start_dim=-3))

    def _build_transition(
        self, inputs: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[Spin8AffineTransition, dict[str, DiagnosticValue]]:
        batch, length = inputs.shape[:2]
        tier_output = self.tier(inputs)
        raw_controls = self.controller(inputs).reshape(
            batch, length, self.config.channels, CONTROL_SCALARS_PER_CHANNEL
        )
        retention_logit = raw_controls[..., 0]
        write_logit = raw_controls[..., 1]
        raw_drive = raw_controls[..., 2:].reshape(
            batch,
            length,
            self.config.channels,
            TRIALITY_SECTORS,
            SPIN8_DIM,
        )
        retention = self.config.retention_min + (
            self.config.retention_max - self.config.retention_min
        ) * torch.sigmoid(retention_logit)
        write_gate = torch.sigmoid(write_logit)
        bounded_drive = unit_ball(raw_drive)
        drive_scale = (1.0 - retention) * write_gate
        drive = drive_scale[..., None, None] * bounded_drive
        transition = mask_spin8_transition(
            Spin8AffineTransition(
                scale=retention,
                action=tier_output.actions,
                drive=drive,
            ),
            valid_mask,
        )
        diagnostics = dict(tier_output.diagnostics)
        diagnostics.update(
            {
                "transport_coordinates": tier_output.coordinates,
                "retention": retention,
                "write_gate": write_gate,
                "effective_write_gate": write_gate
                * valid_mask[..., None].to(write_gate.dtype),
                "bounded_drive": bounded_drive,
                "transition_drive": transition.drive,
                "expected_factors": tier_output.diagnostics["expected_factor_count"],
            }
        )
        return transition, diagnostics

    def _validate_sequence_inputs(
        self, inputs: torch.Tensor, valid_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if inputs.ndim != 3 or inputs.shape[0] < 1 or inputs.shape[1] < 1:
            raise ValueError("inputs must have nonempty shape (B,L,model_dim)")
        if inputs.shape[-1] != self.config.model_dim:
            raise ValueError("inputs have an incompatible model dimension")
        if not inputs.is_floating_point():
            raise TypeError("inputs must have a floating-point dtype")
        parameter = next(self.parameters())
        if inputs.dtype != parameter.dtype or inputs.device != parameter.device:
            raise ValueError("inputs must match the module parameter dtype and device")
        if not bool(torch.isfinite(inputs).all()):
            raise ValueError("inputs must be finite")
        if valid_mask is None:
            return torch.ones(inputs.shape[:2], dtype=torch.bool, device=inputs.device)
        if not isinstance(valid_mask, torch.Tensor):
            raise TypeError("valid_mask must be a torch.Tensor or None")
        if valid_mask.shape != inputs.shape[:2]:
            raise ValueError("valid_mask must have shape (B,L)")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must have dtype torch.bool")
        if valid_mask.device != inputs.device:
            raise ValueError("valid_mask must be on the input device")
        return valid_mask

    def _validate_state(
        self, inputs: torch.Tensor, state: torch.Tensor | None
    ) -> torch.Tensor:
        if state is None:
            return self.initial_state(inputs.shape[0])
        if not isinstance(state, torch.Tensor):
            raise TypeError("state must be a torch.Tensor or None")
        expected = (inputs.shape[0], *self.config.state_shape)
        if state.shape != expected:
            raise ValueError("state must have shape (B,channels,3,8)")
        if not state.is_floating_point():
            raise TypeError("state must have a floating-point dtype")
        if state.dtype != inputs.dtype or state.device != inputs.device:
            raise ValueError("state must match the input dtype and device")
        if not bool(torch.isfinite(state).all()):
            raise ValueError("state must be finite")
        return state


__all__ = [
    "CONTROL_SCALARS_PER_CHANNEL",
    "DiagnosticValue",
    "ScanMode",
    "StructuredMemoryConfig",
    "StructuredSpin8Memory",
    "parallel_spin8_scan",
]
