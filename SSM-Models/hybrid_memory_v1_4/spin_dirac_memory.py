"""Content-addressed Spin(8) fast weights with Clifford-coupled readout.

This is the v1.4.5 Spin-path candidate.  It does not use the old 24-scalar
value-only cache.  Each head stores an ``8_v -> 8_s+`` associative matrix and
applies a two-sided Spin(8) transport before a decoupled erase/write edit::

    M_t = L_t M_{t-1} R_t^T + k_t (w_t * v_t)^T
    e_t = sqrt(b_t) * k_t
    L_t = (I - e_t e_t^T) D_t R_v(g_t)
    R_t = R_s+(g_t)

The transition triples ``(L, R, U)`` form an associative semigroup, so the
semantic implementation supports exact recurrent and parallel prefix scans.
The read ``r+ = q^T M`` is coupled to ``8_s-`` by the fixed Spin(8) Clifford
map ``rho``.  This gives the Spin structure an explicit transport and
intertwiner role; it is not a claim that geometry alone solves retrieval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from pure_spin8_ssm.torch_backend import spin8_factorized_actions
from spin8_triality import (
    SPIN8_BIVECTOR_DIM,
    SPIN8_DIM,
    SPIN8_PAIRS,
    TRIALITY_REPRESENTATIONS,
    torch_triality_generators,
)
from spin8_triality_lift import triality_tensor
from torch import nn
from torch.nn import functional as F

from .gated_delta import GatedDeltaScanMode, _positive_integer

SpinTransportMode = Literal[
    "spin8", "commuting_so2", "su3_torus", "broken_spin8", "identity"
]
SpinReadoutMode = Literal["clifford", "identity"]
SpinGateMode = Literal["equivariant_scalar", "channelwise"]


@dataclass(frozen=True)
class SpinDiracConfig:
    """Configuration for the content-addressed Spin/Clifford memory."""

    model_dim: int
    heads: int = 4
    transport_mode: SpinTransportMode = "spin8"
    readout_mode: SpinReadoutMode = "clifford"
    gate_mode: SpinGateMode = "equivariant_scalar"
    tie_query_key: bool = True
    allow_negative_eigenvalues: bool = False
    bound_values: bool = True
    norm_epsilon: float = 1e-6
    minimum_retention: float = 0.999
    maximum_retention: float = 0.999999
    initial_retention: float = 0.9995
    initial_erase_strength: float = 0.10
    initial_write_strength: float = 0.10
    maximum_coordinate: float = 0.25

    def __post_init__(self) -> None:
        _positive_integer("model_dim", self.model_dim)
        _positive_integer("heads", self.heads)
        if self.transport_mode not in (
            "spin8",
            "commuting_so2",
            "su3_torus",
            "broken_spin8",
            "identity",
        ):
            raise ValueError("unknown transport_mode")
        if self.readout_mode not in ("clifford", "identity"):
            raise ValueError("unknown readout_mode")
        if self.gate_mode not in ("equivariant_scalar", "channelwise"):
            raise ValueError("unknown gate_mode")
        for name in (
            "tie_query_key",
            "allow_negative_eigenvalues",
            "bound_values",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if not math.isfinite(self.norm_epsilon) or self.norm_epsilon <= 0.0:
            raise ValueError("norm_epsilon must be finite and positive")
        if not 0.0 <= self.minimum_retention < self.maximum_retention < 1.0:
            raise ValueError("retention bounds must satisfy 0 <= minimum < maximum < 1")
        if not self.minimum_retention < self.initial_retention < self.maximum_retention:
            raise ValueError(
                "initial_retention must lie strictly between the retention bounds"
            )
        erase_max = 2.0 if self.allow_negative_eigenvalues else 1.0
        if not 0.0 < self.initial_erase_strength < erase_max:
            raise ValueError("initial_erase_strength must lie inside its gate range")
        if not 0.0 < self.initial_write_strength < 1.0:
            raise ValueError("initial_write_strength must lie in (0, 1)")
        if not math.isfinite(self.maximum_coordinate) or self.maximum_coordinate <= 0:
            raise ValueError("maximum_coordinate must be finite and positive")

    @property
    def state_shape(self) -> tuple[int, int, int]:
        return self.heads, SPIN8_DIM, SPIN8_DIM


class SpinDiracMemory(nn.Module):
    """Spin-transported, content-addressed ``8_v -> 8_s+`` fast weights."""

    memory_kind = "spin8_transported_content_addressed_fast_weight"
    supports_episode_writes = True
    claim_boundary = (
        "Semantic two-sided Spin(8) transport plus decoupled fast-weight edit; "
        "not yet a fused-kernel, language-quality, or long-context result."
    )

    def __init__(self, config: SpinDiracConfig) -> None:
        super().__init__()
        if not isinstance(config, SpinDiracConfig):
            raise TypeError("config must be a SpinDiracConfig")
        self.config = config
        width = config.heads * SPIN8_DIM
        self.query_projection = nn.Linear(config.model_dim, width, bias=False)
        self.key_projection = (
            self.query_projection
            if config.tie_query_key
            else nn.Linear(config.model_dim, width, bias=False)
        )
        self.value_projection = nn.Linear(config.model_dim, width, bias=False)
        self.coordinate_projection = nn.Linear(
            config.model_dim, config.heads * SPIN8_BIVECTOR_DIM, bias=False
        )
        gate_width = config.heads if config.gate_mode == "equivariant_scalar" else width
        self.erase_projection = nn.Linear(config.model_dim, gate_width, bias=True)
        self.write_projection = nn.Linear(config.model_dim, gate_width, bias=True)
        self.decay_projection = nn.Linear(config.model_dim, gate_width, bias=True)
        self.output_gate = nn.Linear(config.model_dim, 2 * width, bias=False)
        self.output_norm = nn.RMSNorm(2 * SPIN8_DIM, eps=config.norm_epsilon)
        self.output_projection = nn.Linear(2 * width, config.model_dim, bias=False)
        self.register_buffer(
            "generators",
            torch_triality_generators(TRIALITY_REPRESENTATIONS),
            persistent=True,
        )
        self.register_buffer("rho", triality_tensor(), persistent=True)
        commuting = torch.zeros(SPIN8_BIVECTOR_DIM, dtype=torch.bool)
        for pair in ((0, 1), (2, 3), (4, 5), (6, 7)):
            commuting[SPIN8_PAIRS.index(pair)] = True
        self.register_buffer("commuting_coordinate_mask", commuting, persistent=True)
        su3_indices = torch.tensor(
            [SPIN8_PAIRS.index(pair) for pair in ((0, 1), (2, 3), (4, 5))],
            dtype=torch.long,
        )
        self.register_buffer("su3_coordinate_indices", su3_indices, persistent=True)
        broken_permutation = torch.roll(torch.arange(SPIN8_BIVECTOR_DIM), shifts=1)
        broken_signs = torch.where(
            torch.arange(SPIN8_BIVECTOR_DIM).remainder(3) == 0,
            -torch.ones(SPIN8_BIVECTOR_DIM),
            torch.ones(SPIN8_BIVECTOR_DIM),
        )
        self.register_buffer(
            "broken_coordinate_permutation", broken_permutation, persistent=True
        )
        self.register_buffer("broken_coordinate_signs", broken_signs, persistent=True)
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
        for module in (
            self.value_projection,
            self.output_gate,
            self.output_projection,
        ):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # Identity transport is the neutral optimization start, while the
        # nonzero derivative of the exact factorization keeps the controller
        # learnable from the first step.
        nn.init.zeros_(self.coordinate_projection.weight)
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
        ) / (self.config.maximum_retention - self.config.minimum_retention)
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

    def _controls(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch, length, _ = inputs.shape
        shape = (batch, length, self.config.heads, SPIN8_DIM)
        query = F.normalize(
            self.query_projection(inputs).view(shape),
            dim=-1,
            eps=self.config.norm_epsilon,
        )
        key = F.normalize(
            self.key_projection(inputs).view(shape),
            dim=-1,
            eps=self.config.norm_epsilon,
        )
        value = self.value_projection(inputs).view(shape)
        if self.config.bound_values:
            # Smooth unit-ball projection preserves value amplitude ordering,
            # unlike unit normalization, while making every injected outer
            # product have Frobenius norm below one.
            value_norm_squared = value.float().square().sum(dim=-1, keepdim=True)
            value = value / torch.sqrt(1.0 + value_norm_squared).to(value.dtype)
        gate_shape = (
            (batch, length, self.config.heads, 1)
            if self.config.gate_mode == "equivariant_scalar"
            else shape
        )
        erase = torch.sigmoid(self.erase_projection(inputs)).view(gate_shape)
        if self.config.allow_negative_eigenvalues:
            erase = 2.0 * erase
        write = torch.sigmoid(self.write_projection(inputs)).view(gate_shape)
        retention_unit = torch.sigmoid(self.decay_projection(inputs)).view(gate_shape)
        retention = (
            self.config.minimum_retention
            + (self.config.maximum_retention - self.config.minimum_retention)
            * retention_unit
        )
        coordinate_shape = (
            batch,
            length,
            self.config.heads,
            SPIN8_BIVECTOR_DIM,
        )
        coordinates = self.config.maximum_coordinate * torch.tanh(
            self.coordinate_projection(inputs).view(coordinate_shape)
        )
        if self.config.transport_mode == "commuting_so2":
            coordinates = coordinates * self.commuting_coordinate_mask.to(coordinates)
        elif self.config.transport_mode == "su3_torus":
            # Exact rank-two SU(3) Cartan slice inside SO(2)^4:
            # (theta_1, theta_2, theta_3, theta_4)
            # = (alpha, beta, -alpha-beta, 0). Halving the two free raw
            # coordinates keeps every derived angle inside maximum_coordinate.
            constrained = torch.zeros_like(coordinates)
            first, second, third = self.su3_coordinate_indices
            alpha = 0.5 * coordinates[..., first]
            beta = 0.5 * coordinates[..., second]
            constrained[..., first] = alpha
            constrained[..., second] = beta
            constrained[..., third] = -alpha - beta
            coordinates = constrained
        elif self.config.transport_mode == "identity":
            coordinates = torch.zeros_like(coordinates)
        return query, key, value, erase, write, retention, coordinates

    def _transitions(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        erase: torch.Tensor,
        write: torch.Tensor,
        retention: torch.Tensor,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actions = spin8_factorized_actions(
            coordinates,
            self.generators.to(dtype=coordinates.dtype, device=coordinates.device),
            TRIALITY_REPRESENTATIONS,
        )
        if self.config.transport_mode == "broken_spin8":
            broken_coordinates = coordinates.index_select(
                -1, self.broken_coordinate_permutation
            ) * self.broken_coordinate_signs.to(coordinates)
            broken_actions = spin8_factorized_actions(
                broken_coordinates,
                self.generators.to(dtype=coordinates.dtype, device=coordinates.device),
                TRIALITY_REPRESENTATIONS,
            )
            positive_index = TRIALITY_REPRESENTATIONS.index("positive")
            carriers = [actions[..., index, :, :] for index in range(actions.shape[-3])]
            carriers[positive_index] = broken_actions[..., positive_index, :, :]
            actions = torch.stack(carriers, dim=-3)
        vector_action = actions[..., TRIALITY_REPRESENTATIONS.index("vector"), :, :]
        positive_action = actions[..., TRIALITY_REPRESENTATIONS.index("positive"), :, :]
        eye = torch.eye(SPIN8_DIM, dtype=key.dtype, device=key.device)
        # A symmetric rank-one erase is a contraction. With erase in [0, 1],
        # its eigenvalues lie in [0, 1]; the opt-in [0, 2] range permits sign
        # flips while keeping singular values at most one.
        erase_direction = erase.sqrt() * key
        erase_operator = eye - erase_direction.unsqueeze(
            -1
        ) * erase_direction.unsqueeze(-2)
        if self.config.gate_mode == "equivariant_scalar":
            retained_action = retention.unsqueeze(-1) * vector_action
        else:
            retained_action = torch.diag_embed(retention) @ vector_action
        left = erase_operator @ retained_action
        right = positive_action
        injection = key.unsqueeze(-1) * (write * value).unsqueeze(-2)
        # The scan uses (batch, heads, length, 8, 8).
        left = left.transpose(1, 2)
        right = right.transpose(1, 2)
        injection = injection.transpose(1, 2)
        if valid_mask is not None:
            valid = valid_mask[:, None, :, None, None]
            left = torch.where(valid, left, eye)
            right = torch.where(valid, right, eye)
            injection = torch.where(valid, injection, torch.zeros_like(injection))
        return left, right, injection, actions

    @staticmethod
    def _compose_prefix(
        left: torch.Tensor, right: torch.Tensor, injection: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix_left, prefix_right, prefix_injection = left, right, injection
        offset = 1
        while offset < left.shape[2]:
            after_left = prefix_left[:, :, offset:]
            after_right = prefix_right[:, :, offset:]
            composed_left = after_left @ prefix_left[:, :, :-offset]
            composed_right = after_right @ prefix_right[:, :, :-offset]
            composed_injection = (
                after_left
                @ prefix_injection[:, :, :-offset]
                @ after_right.transpose(-1, -2)
                + prefix_injection[:, :, offset:]
            )
            prefix_left = torch.cat((prefix_left[:, :, :offset], composed_left), dim=2)
            prefix_right = torch.cat(
                (prefix_right[:, :, :offset], composed_right), dim=2
            )
            prefix_injection = torch.cat(
                (prefix_injection[:, :, :offset], composed_injection), dim=2
            )
            offset *= 2
        return prefix_left, prefix_right, prefix_injection

    @staticmethod
    def _recurrent_states(
        left: torch.Tensor,
        right: torch.Tensor,
        injection: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = initial_state
        states = []
        for position in range(left.shape[2]):
            state = (
                left[:, :, position] @ state @ right[:, :, position].transpose(-1, -2)
                + injection[:, :, position]
            )
            states.append(state)
        return torch.stack(states, dim=2), state

    @classmethod
    def _parallel_states(
        cls,
        left: torch.Tensor,
        right: torch.Tensor,
        injection: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_left, prefix_right, prefix_injection = cls._compose_prefix(
            left, right, injection
        )
        states = (
            prefix_left @ initial_state.unsqueeze(2) @ prefix_right.transpose(-1, -2)
            + prefix_injection
        )
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
        query, key, value, erase, write, retention, coordinates = self._controls(inputs)
        left, right, injection, actions = self._transitions(
            key, value, erase, write, retention, coordinates, valid_mask
        )
        if initial_state is None:
            initial_state = inputs.new_zeros(inputs.shape[0], *self.config.state_shape)
        if scan_mode == "recurrent":
            states, final_state = self._recurrent_states(
                left, right, injection, initial_state
            )
        else:
            states, final_state = self._parallel_states(
                left, right, injection, initial_state
            )
        positive_read = torch.einsum("bthv,bhtvp->bthp", query, states)
        if self.config.readout_mode == "clifford":
            negative_read = torch.einsum(
                "...i,vji,...v->...j",
                positive_read,
                self.rho.to(positive_read),
                query,
            )
        else:
            negative_read = positive_read
        read = torch.cat((positive_read, negative_read), dim=-1)
        # The common block already has an outer learned residual gate.  Keep
        # this internal content read active at initialization rather than
        # multiplying two near-zero gates.
        gate = 1.0 + torch.tanh(self.output_gate(inputs).view_as(read))
        output = self.output_projection(
            (self.output_norm(read) * gate).flatten(start_dim=2)
        )
        if valid_mask is not None:
            output = output * valid_mask[..., None].to(output.dtype)
        if not return_diagnostics:
            return output, final_state
        diagnostics: dict[str, torch.Tensor | str] = {
            "kind": "spin_dirac",
            "scan_mode": scan_mode,
            "transport_mode": self.config.transport_mode,
            "readout_mode": self.config.readout_mode,
            "gate_mode": self.config.gate_mode,
            "query_vector": query,
            "key_vector": key,
            "value_positive": value,
            "read_positive": positive_read,
            "read_negative": negative_read,
            "erase_strength": erase,
            "write_strength": write,
            "retention": retention,
            "transport_coordinates": coordinates,
            "transport_actions": actions,
            "state_norm": states.float().square().sum(dim=(-2, -1)).sqrt(),
            "update": output,
        }
        return output, final_state, diagnostics


__all__ = [
    "SpinDiracConfig",
    "SpinDiracMemory",
    "SpinGateMode",
    "SpinReadoutMode",
    "SpinTransportMode",
]
