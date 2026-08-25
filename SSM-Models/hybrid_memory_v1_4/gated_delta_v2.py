"""Semantic Gated DeltaNet-2 candidate for Hybrid Memory v1.4.

The state has shape ``(heads, key_dim, value_dim)`` and follows

``S_t = (I - k_t (b_t * k_t)^T) D_t S_{t-1}``
``      + k_t (w_t * v_t)^T``.

Unlike Gated DeltaNet v1, the erase vector ``b`` and write vector ``w`` are
independent and channel-wise.  This is the narrow architectural change needed
to test the overwrite diagnosis from G13.  This module is a readable semantic
implementation, not a claim of parity with an optimized FLA kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .gated_delta import (
    GatedDeltaScanMode,
    _affine_prefix_scan,
    _positive_integer,
)


@dataclass(frozen=True)
class GatedDeltaV2Config:
    """Dimensions and bounded gate initialization for semantic GDN2."""

    model_dim: int
    heads: int = 4
    key_dim: int | None = None
    value_dim: int | None = None
    allow_negative_eigenvalues: bool = False
    normalize_values: bool = False
    identity_value_path: bool = False
    identity_output_gate: bool = False
    tie_query_key: bool = False
    norm_epsilon: float = 1e-6
    minimum_retention: float = 0.999
    initial_retention: float = 0.9995
    initial_erase_strength: float = 0.10
    initial_write_strength: float = 0.10

    def __post_init__(self) -> None:
        _positive_integer("model_dim", self.model_dim)
        _positive_integer("heads", self.heads)
        if self.model_dim % self.heads:
            raise ValueError("model_dim must be divisible by heads")
        for name in ("key_dim", "value_dim"):
            value = getattr(self, name)
            if value is not None:
                _positive_integer(name, value)
        for name in (
            "allow_negative_eigenvalues",
            "normalize_values",
            "identity_value_path",
            "identity_output_gate",
            "tie_query_key",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if (
            self.identity_value_path
            and self.heads * self.resolved_value_dim != self.model_dim
        ):
            raise ValueError(
                "identity_value_path requires heads * value_dim == model_dim"
            )
        if not math.isfinite(self.norm_epsilon) or self.norm_epsilon <= 0.0:
            raise ValueError("norm_epsilon must be finite and positive")
        if not 0.0 <= self.minimum_retention < 1.0:
            raise ValueError("minimum_retention must lie in [0, 1)")
        if not self.minimum_retention < self.initial_retention < 1.0:
            raise ValueError(
                "initial_retention must lie strictly between minimum_retention and 1"
            )
        erase_max = 2.0 if self.allow_negative_eigenvalues else 1.0
        if not 0.0 < self.initial_erase_strength < erase_max:
            raise ValueError("initial_erase_strength must lie inside its gate range")
        if not 0.0 < self.initial_write_strength < 1.0:
            raise ValueError("initial_write_strength must lie in (0, 1)")

    @property
    def resolved_key_dim(self) -> int:
        return self.model_dim // self.heads if self.key_dim is None else self.key_dim

    @property
    def resolved_value_dim(self) -> int:
        return (
            self.model_dim // self.heads if self.value_dim is None else self.value_dim
        )

    @property
    def state_shape(self) -> tuple[int, int, int]:
        return (self.heads, self.resolved_key_dim, self.resolved_value_dim)


class GatedDeltaV2Memory(nn.Module):
    """Channel-wise decay with independently learned erase and write gates."""

    memory_kind = "decoupled_content_addressed_fast_weight"
    supports_episode_writes = True
    claim_boundary = (
        "Semantic GDN2 recurrence in the v1.4 shell; not a fused-kernel parity, "
        "training-quality, or long-context result."
    )

    def __init__(self, config: GatedDeltaV2Config) -> None:
        super().__init__()
        if not isinstance(config, GatedDeltaV2Config):
            raise TypeError("config must be a GatedDeltaV2Config")
        self.config = config
        key_width = config.heads * config.resolved_key_dim
        value_width = config.heads * config.resolved_value_dim
        self.query_projection = nn.Linear(config.model_dim, key_width, bias=False)
        self.key_projection = (
            self.query_projection
            if config.tie_query_key
            else nn.Linear(config.model_dim, key_width, bias=False)
        )
        self.value_projection = nn.Linear(config.model_dim, value_width, bias=False)
        self.erase_projection = nn.Linear(config.model_dim, key_width, bias=True)
        self.write_projection = nn.Linear(config.model_dim, value_width, bias=True)
        self.decay_projection = nn.Linear(config.model_dim, key_width, bias=True)
        self.output_gate = nn.Linear(config.model_dim, value_width, bias=False)
        self.output_norm = nn.RMSNorm(
            config.resolved_value_dim, eps=config.norm_epsilon
        )
        self.output_projection = nn.Linear(value_width, config.model_dim, bias=False)
        self.reset_parameters()

    @property
    def state_scalars(self) -> int:
        return math.prod(self.config.state_shape)

    def state_bytes(self, dtype: torch.dtype, *, batch_size: int = 1) -> int:
        _positive_integer("batch_size", batch_size)
        probe = torch.empty((), dtype=dtype)
        if not probe.is_floating_point():
            raise TypeError("dtype must be floating point")
        return batch_size * self.state_scalars * probe.element_size()

    @staticmethod
    def _logit(ratio: float) -> float:
        return math.log(ratio / (1.0 - ratio))

    def reset_parameters(self) -> None:
        if self.config.tie_query_key:
            nn.init.orthogonal_(self.query_projection.weight)
        else:
            nn.init.normal_(self.query_projection.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.key_projection.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_gate.weight, mean=0.0, std=0.02)
        if self.config.identity_value_path:
            nn.init.eye_(self.value_projection.weight)
            nn.init.eye_(self.output_projection.weight)
        else:
            nn.init.normal_(self.value_projection.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)

        nn.init.zeros_(self.erase_projection.weight)
        erase_max = 2.0 if self.config.allow_negative_eigenvalues else 1.0
        nn.init.constant_(
            self.erase_projection.bias,
            self._logit(self.config.initial_erase_strength / erase_max),
        )
        nn.init.zeros_(self.write_projection.weight)
        nn.init.constant_(
            self.write_projection.bias,
            self._logit(self.config.initial_write_strength),
        )
        nn.init.zeros_(self.decay_projection.weight)
        retention_ratio = (
            self.config.initial_retention - self.config.minimum_retention
        ) / (1.0 - self.config.minimum_retention)
        nn.init.constant_(self.decay_projection.bias, self._logit(retention_ratio))

    def _validate(
        self,
        inputs: torch.Tensor,
        initial_state: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
        scan_mode: GatedDeltaScanMode,
    ) -> None:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a tensor")
        if inputs.ndim != 3 or inputs.shape[-1] != self.config.model_dim:
            raise ValueError("inputs must have shape (batch, length, model_dim)")
        if inputs.shape[0] < 1 or inputs.shape[1] < 1:
            raise ValueError("inputs must have nonempty batch and length dimensions")
        if not inputs.is_floating_point():
            raise TypeError("inputs must use a floating-point dtype")
        if scan_mode not in ("recurrent", "parallel"):
            raise ValueError("scan_mode must be 'recurrent' or 'parallel'")
        expected = (inputs.shape[0], *self.config.state_shape)
        if initial_state is not None:
            if not isinstance(initial_state, torch.Tensor):
                raise TypeError("initial_state must be a tensor or None")
            if initial_state.shape != expected:
                raise ValueError(f"initial_state must have shape {expected}")
            if (
                initial_state.dtype != inputs.dtype
                or initial_state.device != inputs.device
            ):
                raise ValueError("initial_state must match input dtype and device")
            if not bool(torch.isfinite(initial_state).all()):
                raise ValueError("initial_state must be finite")
        if valid_mask is not None:
            if not isinstance(valid_mask, torch.Tensor):
                raise TypeError("valid_mask must be a tensor or None")
            if valid_mask.shape != inputs.shape[:2] or valid_mask.dtype != torch.bool:
                raise ValueError("valid_mask must be bool with shape (batch, length)")
            if valid_mask.device != inputs.device:
                raise ValueError("valid_mask must be on the input device")

    def _controls(
        self, inputs: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch, length, _ = inputs.shape
        heads = self.config.heads
        key_dim = self.config.resolved_key_dim
        value_dim = self.config.resolved_value_dim
        query = self.query_projection(inputs).view(batch, length, heads, key_dim)
        key = self.key_projection(inputs).view(batch, length, heads, key_dim)
        value = self.value_projection(inputs).view(batch, length, heads, value_dim)
        query = F.normalize(query, dim=-1, eps=self.config.norm_epsilon)
        key = F.normalize(key, dim=-1, eps=self.config.norm_epsilon)
        if self.config.normalize_values:
            value = F.normalize(
                value, dim=-1, eps=self.config.norm_epsilon
            ) * math.sqrt(value_dim)
        erase = torch.sigmoid(self.erase_projection(inputs)).view(
            batch, length, heads, key_dim
        )
        if self.config.allow_negative_eigenvalues:
            erase = 2.0 * erase
        write = torch.sigmoid(self.write_projection(inputs)).view(
            batch, length, heads, value_dim
        )
        retention_unit = torch.sigmoid(self.decay_projection(inputs)).view(
            batch, length, heads, key_dim
        )
        retention = (
            self.config.minimum_retention
            + (1.0 - self.config.minimum_retention) * retention_unit
        )
        return query, key, value, erase, write, retention

    def _transitions(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        erase: torch.Tensor,
        write: torch.Tensor,
        retention: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eye = torch.eye(
            self.config.resolved_key_dim, dtype=key.dtype, device=key.device
        )
        erase_address = erase * key
        erase_operator = eye - key.unsqueeze(-1) * erase_address.unsqueeze(-2)
        decay_operator = torch.diag_embed(retention)
        transition = erase_operator @ decay_operator
        written_value = write * value
        injection = key.unsqueeze(-1) * written_value.unsqueeze(-2)
        transition = transition.transpose(1, 2)
        injection = injection.transpose(1, 2)
        if valid_mask is not None:
            valid = valid_mask[:, None, :, None, None]
            transition = torch.where(valid, transition, eye)
            injection = torch.where(valid, injection, torch.zeros_like(injection))
        return transition, injection

    @staticmethod
    def _recurrent_states(
        transition: torch.Tensor,
        injection: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = initial_state
        states = []
        for position in range(transition.shape[2]):
            state = transition[:, :, position] @ state + injection[:, :, position]
            states.append(state)
        return torch.stack(states, dim=2), state

    @staticmethod
    def _parallel_states(
        transition: torch.Tensor,
        injection: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_a, prefix_b = _affine_prefix_scan(transition, injection)
        states = prefix_a @ initial_state.unsqueeze(2) + prefix_b
        return states, states[:, :, -1]

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        scan_mode: GatedDeltaScanMode = "parallel",
        return_diagnostics: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | str]]
    ):
        self._validate(inputs, initial_state, valid_mask, scan_mode)
        query, key, value, erase, write, retention = self._controls(inputs)
        transition, injection = self._transitions(
            key, value, erase, write, retention, valid_mask
        )
        if initial_state is None:
            initial_state = inputs.new_zeros(inputs.shape[0], *self.config.state_shape)
        if scan_mode == "recurrent":
            states, final_state = self._recurrent_states(
                transition, injection, initial_state
            )
        else:
            states, final_state = self._parallel_states(
                transition, injection, initial_state
            )
        read = torch.einsum("bthk,bhtkv->bthv", query, states)
        gate = self.output_gate(inputs).view_as(read)
        gate = (
            1.0 + torch.tanh(gate) if self.config.identity_output_gate else F.silu(gate)
        )
        output = self.output_projection(
            (self.output_norm(read) * gate).flatten(start_dim=2)
        )
        if valid_mask is not None:
            output = output * valid_mask[..., None].to(output.dtype)
        if not return_diagnostics:
            return output, final_state
        diagnostics: dict[str, torch.Tensor | str] = {
            "kind": "gated_delta_v2",
            "scan_mode": scan_mode,
            "query": query,
            "key": key,
            "value": value,
            "read": read,
            "update": output,
            "erase_strength": erase,
            "write_strength": write,
            "retention": retention,
            "state_norm": states.float().square().sum(dim=(-2, -1)).sqrt(),
        }
        return output, final_state, diagnostics


__all__ = ["GatedDeltaV2Config", "GatedDeltaV2Memory"]
