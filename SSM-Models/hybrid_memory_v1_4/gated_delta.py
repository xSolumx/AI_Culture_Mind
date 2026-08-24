"""Content-addressed Gated DeltaNet memory for hybrid-memory v1.4.

This is a repository semantic implementation, not the FlashRT or FLA fused
kernel.  It follows the same recurrent state contract used by Gated DeltaNet:

``S_t = a_t S_{t-1} + beta_t k_t (v_t - k_t^T a_t S_{t-1})``

and reads ``q_t^T S_t``.  Unlike the retired selected-block primary path, the
state stores the key-to-value association itself.  No task labels or token IDs
are consulted by this layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from torch import nn
from torch.nn import functional as F

GatedDeltaScanMode: TypeAlias = Literal["recurrent", "parallel"]


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class GatedDeltaConfig:
    """Dimensions and stability bounds for one semantic Gated DeltaNet layer."""

    model_dim: int
    heads: int = 4
    key_dim: int | None = None
    value_dim: int | None = None
    allow_negative_eigenvalues: bool = False
    normalize_values: bool = False
    norm_epsilon: float = 1e-6
    minimum_retention: float = 0.90
    initial_retention: float = 0.995
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
        if type(self.allow_negative_eigenvalues) is not bool:
            raise TypeError("allow_negative_eigenvalues must be a bool")
        if type(self.normalize_values) is not bool:
            raise TypeError("normalize_values must be a bool")
        if not math.isfinite(self.norm_epsilon) or self.norm_epsilon <= 0.0:
            raise ValueError("norm_epsilon must be finite and positive")
        if not 0.0 <= self.minimum_retention < 1.0:
            raise ValueError("minimum_retention must lie in [0, 1)")
        if not self.minimum_retention < self.initial_retention < 1.0:
            raise ValueError(
                "initial_retention must lie strictly between minimum_retention and 1"
            )
        write_max = 2.0 if self.allow_negative_eigenvalues else 1.0
        if not 0.0 < self.initial_write_strength < write_max:
            raise ValueError(
                "initial_write_strength must lie strictly inside its sigmoid range"
            )

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


def _compose_affine(
    later_a: torch.Tensor,
    later_b: torch.Tensor,
    earlier_a: torch.Tensor,
    earlier_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose ``later(earlier(S))`` for left-affine matrix transitions."""

    return later_a @ earlier_a, later_a @ earlier_b + later_b


def _affine_prefix_scan(
    transition: torch.Tensor, injection: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive Hillis-Steele scan over affine matrix transitions."""

    prefix_a = transition
    prefix_b = injection
    length = transition.shape[2]
    offset = 1
    while offset < length:
        next_a = prefix_a.clone()
        next_b = prefix_b.clone()
        composed_a, composed_b = _compose_affine(
            prefix_a[:, :, offset:],
            prefix_b[:, :, offset:],
            prefix_a[:, :, :-offset],
            prefix_b[:, :, :-offset],
        )
        next_a[:, :, offset:] = composed_a
        next_b[:, :, offset:] = composed_b
        prefix_a, prefix_b = next_a, next_b
        offset *= 2
    return prefix_a, prefix_b


class GatedDeltaMemory(nn.Module):
    """Learned content-addressed fast-weight memory with exact streaming state."""

    memory_kind = "content_addressed_fast_weight"
    supports_episode_writes = True
    claim_boundary = (
        "Repository semantic Gated DeltaNet recurrence; not a fused FlashRT/FLA "
        "kernel and not by itself a model-quality result."
    )

    def __init__(self, config: GatedDeltaConfig) -> None:
        super().__init__()
        if not isinstance(config, GatedDeltaConfig):
            raise TypeError("config must be a GatedDeltaConfig")
        self.config = config
        key_width = config.heads * config.resolved_key_dim
        value_width = config.heads * config.resolved_value_dim
        self.query_projection = nn.Linear(config.model_dim, key_width, bias=False)
        self.key_projection = nn.Linear(config.model_dim, key_width, bias=False)
        self.value_projection = nn.Linear(config.model_dim, value_width, bias=False)
        self.write_projection = nn.Linear(config.model_dim, config.heads, bias=True)
        self.decay_projection = nn.Linear(config.model_dim, config.heads, bias=True)
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

    def reset_parameters(self) -> None:
        for module in (
            self.query_projection,
            self.key_projection,
            self.value_projection,
            self.output_gate,
            self.output_projection,
        ):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.write_projection.weight)
        write_max = 2.0 if self.config.allow_negative_eigenvalues else 1.0
        write_ratio = self.config.initial_write_strength / write_max
        write_bias = math.log(write_ratio / (1.0 - write_ratio))
        nn.init.constant_(self.write_projection.bias, write_bias)
        nn.init.zeros_(self.decay_projection.weight)
        retention_ratio = (
            self.config.initial_retention - self.config.minimum_retention
        ) / (1.0 - self.config.minimum_retention)
        retention_bias = math.log(retention_ratio / (1.0 - retention_ratio))
        nn.init.constant_(self.decay_projection.bias, retention_bias)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = inputs.shape
        heads = self.config.heads
        key_dim = self.config.resolved_key_dim
        value_dim = self.config.resolved_value_dim
        query = self.query_projection(inputs).view(batch, length, heads, key_dim)
        key = self.key_projection(inputs).view(batch, length, heads, key_dim)
        value = self.value_projection(inputs).view(batch, length, heads, value_dim)
        if self.config.normalize_values:
            value = F.normalize(
                value, dim=-1, eps=self.config.norm_epsilon
            ) * math.sqrt(value_dim)
        query = F.normalize(query, dim=-1, eps=self.config.norm_epsilon)
        key = F.normalize(key, dim=-1, eps=self.config.norm_epsilon)
        write = torch.sigmoid(self.write_projection(inputs))
        if self.config.allow_negative_eigenvalues:
            write = 2.0 * write
        retention_unit = torch.sigmoid(self.decay_projection(inputs))
        retention = (
            self.config.minimum_retention
            + (1.0 - self.config.minimum_retention) * retention_unit
        )
        return query, key, value, write, retention

    def _transitions(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        write: torch.Tensor,
        retention: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_outer = key.unsqueeze(-1) * key.unsqueeze(-2)
        eye = torch.eye(
            self.config.resolved_key_dim, dtype=key.dtype, device=key.device
        )
        transition = retention[..., None, None] * (
            eye - write[..., None, None] * key_outer
        )
        injection = write[..., None, None] * (key.unsqueeze(-1) * value.unsqueeze(-2))
        # Scan tensors use (batch, heads, length, ...).
        transition = transition.transpose(1, 2)
        injection = injection.transpose(1, 2)
        if valid_mask is not None:
            valid = valid_mask[:, None, :, None, None]
            transition = torch.where(valid, transition, eye)
            injection = torch.where(valid, injection, torch.zeros_like(injection))
        return transition, injection

    def _recurrent_states(
        self,
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

    def _parallel_states(
        self,
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
        query, key, value, write, retention = self._controls(inputs)
        transition, injection = self._transitions(
            key, value, write, retention, valid_mask
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
        gated = self.output_norm(read) * F.silu(self.output_gate(inputs).view_as(read))
        output = self.output_projection(gated.flatten(start_dim=2))
        if valid_mask is not None:
            output = output * valid_mask[..., None].to(output.dtype)
        if not return_diagnostics:
            return output, final_state
        diagnostics: dict[str, torch.Tensor | str] = {
            "kind": "gated_delta",
            "scan_mode": scan_mode,
            "query": query,
            "key": key,
            "value": value,
            "read": read,
            "update": output,
            "write_strength": write,
            "retention": retention,
            "state_norm": states.float().square().sum(dim=(-2, -1)).sqrt(),
        }
        return output, final_state, diagnostics


__all__ = ["GatedDeltaConfig", "GatedDeltaMemory", "GatedDeltaScanMode"]
