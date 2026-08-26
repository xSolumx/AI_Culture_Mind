"""Strict-history transactional delta memory for the G15B-T frontier.

The layer keeps one content-addressed fast-weight matrix per head.  Query and
readout controls always consume the full causal convolution view.  Edit
controls consume either that full view (the matched F control) or a structurally
computed strict-history view (the T arm).

For unit key ``k`` and effective erase/write gates ``beta,alpha in [0,1]``:

``A = (I - beta k k^T) diag(r)``
``B = k (alpha v)^T``
``S_t = A_t S_{t-1} + B_t``.

The historical product mode uses ``beta=c*e`` and ``alpha=c*w``. G15B-E's
matched mode instead forms the effective gates by adding the corresponding
logits. The shared logit is continuous and has no required binary semantics.

The symmetric erase and bounded retention are nonexpansive.  The affine law is
scan-compatible.  This module is semantic PyTorch code, not a fused-kernel or
training-quality claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .gated_delta import GatedDeltaScanMode, _affine_prefix_scan, _positive_integer

TransactionalControllerMode = Literal["full", "history"]
EffectiveEditGateMode = Literal["product", "logit_additive"]


@dataclass(frozen=True)
class TransactionalDeltaConfig:
    """Dimensions and bounded initialization for transactional delta memory."""

    model_dim: int
    heads: int = 4
    key_dim: int | None = None
    value_dim: int | None = None
    controller_mode: TransactionalControllerMode = "history"
    effective_edit_gate_mode: EffectiveEditGateMode = "product"
    normalize_values: bool = False
    identity_value_path: bool = False
    identity_output_gate: bool = False
    tie_query_key: bool = False
    norm_epsilon: float = 1e-6
    minimum_retention: float = 0.999
    initial_retention: float = 0.9995
    initial_commit_strength: float = 0.10
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
        if self.controller_mode not in ("full", "history"):
            raise ValueError("controller_mode must be 'full' or 'history'")
        if self.effective_edit_gate_mode not in ("product", "logit_additive"):
            raise ValueError(
                "effective_edit_gate_mode must be 'product' or 'logit_additive'"
            )
        for name in (
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
        for name in (
            "initial_commit_strength",
            "initial_erase_strength",
            "initial_write_strength",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1)")

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


class TransactionalDeltaMemory(nn.Module):
    """Monolithic fast weights with bounded effective erase/write decisions."""

    memory_kind = "strict_history_transactional_fast_weight"
    supports_episode_writes = True
    claim_boundary = (
        "Prospective G15B-T/G15B-E semantic recurrence; no learned-transaction, "
        "natural-text, fused-kernel, or model-promotion claim."
    )

    def __init__(self, config: TransactionalDeltaConfig) -> None:
        super().__init__()
        if not isinstance(config, TransactionalDeltaConfig):
            raise TypeError("config must be a TransactionalDeltaConfig")
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
        self.commit_projection = nn.Linear(config.model_dim, config.heads, bias=True)
        self.erase_projection = nn.Linear(config.model_dim, config.heads, bias=True)
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

        nn.init.zeros_(self.commit_projection.weight)
        nn.init.constant_(
            self.commit_projection.bias,
            self._logit(self.config.initial_commit_strength),
        )
        nn.init.zeros_(self.erase_projection.weight)
        nn.init.constant_(
            self.erase_projection.bias,
            self._logit(self.config.initial_erase_strength),
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
        full_inputs: torch.Tensor,
        history_inputs: torch.Tensor,
        initial_state: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
        scan_mode: GatedDeltaScanMode,
    ) -> None:
        for name, inputs in (
            ("full_inputs", full_inputs),
            ("history_inputs", history_inputs),
        ):
            if not isinstance(inputs, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            if inputs.ndim != 3 or inputs.shape[-1] != self.config.model_dim:
                raise ValueError(f"{name} must have shape (batch, length, model_dim)")
            if inputs.shape[0] < 1 or inputs.shape[1] < 1:
                raise ValueError(f"{name} must have nonempty batch/length")
            if not inputs.is_floating_point():
                raise TypeError(f"{name} must use a floating-point dtype")
        if full_inputs.shape != history_inputs.shape:
            raise ValueError("full_inputs and history_inputs must have the same shape")
        if (
            full_inputs.dtype != history_inputs.dtype
            or full_inputs.device != history_inputs.device
        ):
            raise ValueError("full_inputs and history_inputs must share dtype/device")
        if scan_mode not in ("recurrent", "parallel"):
            raise ValueError("scan_mode must be 'recurrent' or 'parallel'")
        expected = (full_inputs.shape[0], *self.config.state_shape)
        if initial_state is not None:
            if not isinstance(initial_state, torch.Tensor):
                raise TypeError("initial_state must be a tensor or None")
            if initial_state.shape != expected:
                raise ValueError(f"initial_state must have shape {expected}")
            if (
                initial_state.dtype != full_inputs.dtype
                or initial_state.device != full_inputs.device
            ):
                raise ValueError("initial_state must match input dtype and device")
            if not bool(torch.isfinite(initial_state).all()):
                raise ValueError("initial_state must be finite")
        if valid_mask is not None:
            if not isinstance(valid_mask, torch.Tensor):
                raise TypeError("valid_mask must be a tensor or None")
            if (
                valid_mask.shape != full_inputs.shape[:2]
                or valid_mask.dtype != torch.bool
            ):
                raise ValueError("valid_mask must be bool with shape (batch, length)")
            if valid_mask.device != full_inputs.device:
                raise ValueError("valid_mask must be on the input device")

    def _controls(
        self, full_inputs: torch.Tensor, history_inputs: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch, length, _ = full_inputs.shape
        heads = self.config.heads
        key_dim = self.config.resolved_key_dim
        value_dim = self.config.resolved_value_dim
        edit_inputs = (
            full_inputs if self.config.controller_mode == "full" else history_inputs
        )
        query = self.query_projection(full_inputs).view(batch, length, heads, key_dim)
        key = self.key_projection(edit_inputs).view(batch, length, heads, key_dim)
        value = self.value_projection(edit_inputs).view(batch, length, heads, value_dim)
        query = F.normalize(query, dim=-1, eps=self.config.norm_epsilon)
        key = F.normalize(key, dim=-1, eps=self.config.norm_epsilon)
        if self.config.normalize_values:
            value = F.normalize(
                value, dim=-1, eps=self.config.norm_epsilon
            ) * math.sqrt(value_dim)
        event_logits = self.commit_projection(edit_inputs).view(
            batch, length, heads, 1
        )
        erase_logits = self.erase_projection(edit_inputs).view(
            batch, length, heads, 1
        )
        write_logits = self.write_projection(edit_inputs).view(
            batch, length, heads, value_dim
        )
        commit = torch.sigmoid(event_logits)
        if self.config.effective_edit_gate_mode == "product":
            erase = torch.sigmoid(erase_logits)
            write = torch.sigmoid(write_logits)
        else:
            initial_commit_logit = self._logit(
                self.config.initial_commit_strength
            )
            erase_offset = (
                self._logit(
                    self.config.initial_commit_strength
                    * self.config.initial_erase_strength
                )
                - initial_commit_logit
                - self._logit(self.config.initial_erase_strength)
            )
            write_offset = (
                self._logit(
                    self.config.initial_commit_strength
                    * self.config.initial_write_strength
                )
                - initial_commit_logit
                - self._logit(self.config.initial_write_strength)
            )
            erase = torch.sigmoid(event_logits + erase_logits + erase_offset)
            write = torch.sigmoid(event_logits + write_logits + write_offset)
        retention_unit = torch.sigmoid(self.decay_projection(edit_inputs)).view(
            batch, length, heads, key_dim
        )
        retention = (
            self.config.minimum_retention
            + (1.0 - self.config.minimum_retention) * retention_unit
        )
        return query, key, value, commit, erase, write, retention

    def _transitions(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        commit: torch.Tensor,
        erase: torch.Tensor,
        write: torch.Tensor,
        retention: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eye = torch.eye(
            self.config.resolved_key_dim, dtype=key.dtype, device=key.device
        )
        if self.config.effective_edit_gate_mode == "product":
            effective_erase = commit * erase
            effective_write = commit * write
        else:
            effective_erase = erase
            effective_write = write
        erase_direction = effective_erase.sqrt() * key
        erase_operator = eye - erase_direction.unsqueeze(
            -1
        ) * erase_direction.unsqueeze(-2)
        transition = erase_operator @ torch.diag_embed(retention)
        written_value = effective_write * value
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
        full_inputs: torch.Tensor,
        history_inputs: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        scan_mode: GatedDeltaScanMode = "parallel",
        return_diagnostics: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | str]]
    ):
        self._validate(
            full_inputs, history_inputs, initial_state, valid_mask, scan_mode
        )
        query, key, value, commit, erase, write, retention = self._controls(
            full_inputs, history_inputs
        )
        transition, injection = self._transitions(
            key, value, commit, erase, write, retention, valid_mask
        )
        if initial_state is None:
            initial_state = full_inputs.new_zeros(
                full_inputs.shape[0], *self.config.state_shape
            )
        if scan_mode == "recurrent":
            states, final_state = self._recurrent_states(
                transition, injection, initial_state
            )
        else:
            states, final_state = self._parallel_states(
                transition, injection, initial_state
            )
        read = torch.einsum("bthk,bhtkv->bthv", query, states)
        gate = self.output_gate(full_inputs).view_as(read)
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
            "kind": "transactional_delta",
            "controller_mode": self.config.controller_mode,
            "effective_edit_gate_mode": self.config.effective_edit_gate_mode,
            "scan_mode": scan_mode,
            "query_vector": query,
            "key_vector": key,
            "value": value,
            "commit_strength": commit,
            "erase_strength": erase,
            "write_strength": write,
            "effective_erase_strength": (
                commit * erase
                if self.config.effective_edit_gate_mode == "product"
                else erase
            ),
            "effective_write_strength": (
                commit * write
                if self.config.effective_edit_gate_mode == "product"
                else write
            ),
            "retention": retention,
            "read": read,
            "update": output,
            "transition": transition,
            "injection": injection,
            "state_norm": states.float().square().sum(dim=(-2, -1)).sqrt(),
        }
        return output, final_state, diagnostics


__all__ = [
    "EffectiveEditGateMode",
    "TransactionalControllerMode",
    "TransactionalDeltaConfig",
    "TransactionalDeltaMemory",
]
