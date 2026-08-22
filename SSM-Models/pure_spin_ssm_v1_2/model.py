"""Pure Spin SSM v1.2: triality recurrence with local and channel mixing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import torch
from pure_spin8_ssm.torch_backend import PureSpin8SSMLayer
from torch import nn
from torch.nn import functional as F

__version__ = "1.2.0"


@dataclass(frozen=True)
class PureSpinV12Config:
    vocab_size: int = 256
    d_model: int = 128
    num_layers: int = 4
    spin_channels: int = 2
    d_conv: int = 4
    expansion: int = 2
    mixer: Literal[
        "swiglu",
        "sol_bounded_quadratic",
        "sol_self_gate",
    ] = "swiglu"
    readout: Literal["direction", "triality_invariants"] = "direction"
    multiplicity_router: Literal["none", "orthogonal_query"] = "none"
    multiplicity_angle_limit: float = 1.5707963267948966
    recurrence: Literal[
        "independent", "coupled_isotypic", "independent_block", "spin_delta"
    ] = "independent"
    delta_slots: int = 2
    recurrent_multiplicity: Literal["identity", "orthogonal"] = "identity"
    recurrent_coupling_scale: Literal["unit", "retention_step"] = "unit"
    retention_mode: Literal[
        "shared", "isotypic", "isotypic_spectrum"
    ] = "shared"
    group_schedule: tuple[int, ...] | None = None
    dropout: float = 0.0
    min_retention_logit: float = 2.0
    scan_chunk_size: int = 32
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if (
            min(
                self.vocab_size,
                self.d_model,
                self.num_layers,
                self.spin_channels,
                self.d_conv,
                self.expansion,
            )
            < 1
        ):
            raise ValueError("all model dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0,1)")
        if self.scan_chunk_size < 1:
            raise ValueError("scan_chunk_size must be positive")
        if self.mixer not in (
            "swiglu",
            "sol_bounded_quadratic",
            "sol_self_gate",
        ):
            raise ValueError("unknown channel mixer")
        if self.readout not in ("direction", "triality_invariants"):
            raise ValueError("unknown state readout")
        if self.multiplicity_router not in ("none", "orthogonal_query"):
            raise ValueError("unknown multiplicity router")
        if self.multiplicity_angle_limit < 0.0:
            raise ValueError("multiplicity_angle_limit must be nonnegative")
        if self.recurrence not in (
            "independent",
            "coupled_isotypic",
            "independent_block",
            "spin_delta",
        ):
            raise ValueError("unknown recurrence")
        if self.delta_slots != 2:
            raise ValueError("the first Spin-Delta compiler requires delta_slots=2")
        if self.recurrent_multiplicity not in ("identity", "orthogonal"):
            raise ValueError("unknown recurrent multiplicity mode")
        if self.recurrent_coupling_scale not in ("unit", "retention_step"):
            raise ValueError("unknown recurrent coupling scale")
        if self.retention_mode not in (
            "shared",
            "isotypic",
            "isotypic_spectrum",
        ):
            raise ValueError("unknown retention mode")
        if (
            self.recurrent_coupling_scale != "unit"
            and self.recurrent_multiplicity != "orthogonal"
        ):
            raise ValueError("coupling scale requires orthogonal multiplicity")
        if (
            self.recurrence == "independent"
            and self.recurrent_multiplicity != "identity"
        ):
            raise ValueError("recurrent multiplicity requires a coupled recurrence")
        if self.recurrence == "spin_delta" and self.retention_mode != "shared":
            raise ValueError("Spin-Delta currently requires shared retention")
        if (
            self.recurrence == "spin_delta"
            and self.recurrent_multiplicity != "identity"
        ):
            raise ValueError("Spin-Delta owns its slot mixing and requires identity")
        if self.group_schedule is not None:
            if len(self.group_schedule) != self.num_layers:
                raise ValueError("group_schedule must have one entry per layer")
            if any(dimension < 2 or dimension > 8 for dimension in self.group_schedule):
                raise ValueError("group_schedule dimensions must lie in [2,8]")
            if tuple(sorted(self.group_schedule)) != self.group_schedule:
                raise ValueError("group_schedule must be nondecreasing")


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, width: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            width,
            width,
            kernel_size,
            groups=width,
            padding=kernel_size - 1,
            bias=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = self.conv(inputs.transpose(1, 2))[..., : inputs.shape[1]]
        return sequence.transpose(1, 2)


class SwiGLU(nn.Module):
    def __init__(self, width: int, expansion: int) -> None:
        super().__init__()
        hidden = width * expansion
        self.input = nn.Linear(width, 2 * hidden, bias=False)
        self.output = nn.Linear(hidden, width, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value, gate = self.input(inputs).chunk(2, dim=-1)
        return self.output(value * F.silu(gate))


class SolBoundedQuadratic(nn.Module):
    """Smooth bounded quadratic gate using no exponential activation."""

    def __init__(self, width: int, expansion: int) -> None:
        super().__init__()
        hidden = width * expansion
        self.input = nn.Linear(width, 2 * hidden, bias=False)
        self.output = nn.Linear(hidden, width, bias=False)
        self.output_scale = nn.Parameter(torch.tensor(-1.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value, gate = self.input(inputs).chunk(2, dim=-1)
        bounded_gate = gate * torch.rsqrt(1.0 + gate.square())
        mixed = value * (1.0 + bounded_gate)
        return torch.sigmoid(self.output_scale) * self.output(mixed)


class SolSelfGate(nn.Module):
    """Single-projection, self-gated channel mixer.

    The map ``z -> z * (1 + z / sqrt(1 + z^2))`` keeps a direct linear path,
    adds a signed quadratic correction near the origin, and approaches slopes
    zero and two on the negative and positive tails. Unlike SwiGLU it does not
    spend a second input projection on a separate gate tensor.
    """

    def __init__(self, width: int, expansion: int) -> None:
        super().__init__()
        hidden = width * expansion
        self.input = nn.Linear(width, hidden, bias=False)
        self.output = nn.Linear(hidden, width, bias=False)
        self.output_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = self.input(inputs)
        bounded_slope = value * torch.rsqrt(1.0 + value.square())
        return torch.sigmoid(self.output_scale) * self.output(
            value * (1.0 + bounded_slope)
        )


class PureSpinV12Block(nn.Module):
    """Mamba-shaped local mixer around a triality-faithful Spin(8) cache."""

    def __init__(self, config: PureSpinV12Config, group_dimension: int = 8) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(config.d_model)
        self.input_projection = nn.Linear(
            config.d_model, 2 * config.d_model, bias=False
        )
        self.local_conv = CausalDepthwiseConv1d(config.d_model, config.d_conv)
        self.spin = PureSpin8SSMLayer(
            config.d_model,
            channels=config.spin_channels,
            action_mode="factorized",
            min_retention_logit=config.min_retention_logit,
            triality_coupling=True,
            normalize_inputs=True,
            retention_mode=config.retention_mode,
        )
        generator_pairs = tuple(combinations(range(8), 2))
        subgroup_indices = tuple(
            index
            for index, (left, right) in enumerate(generator_pairs)
            if left < group_dimension and right < group_dimension
        )
        self.group_dimension = group_dimension
        self.scan_chunk_size = config.scan_chunk_size
        self.subgroup_indices = subgroup_indices
        self.register_buffer(
            "subgroup_generators",
            self.spin.generators[:, subgroup_indices].clone(),
            persistent=False,
        )
        if group_dimension < 8:
            self.spin.coefficient_controller = nn.Linear(
                config.d_model,
                config.spin_channels * len(subgroup_indices),
            )
            nn.init.zeros_(self.spin.coefficient_controller.weight)
            nn.init.zeros_(self.spin.coefficient_controller.bias)
        self.recurrence_mode = config.recurrence
        self.recurrent_multiplicity = config.recurrent_multiplicity
        self.recurrent_coupling_scale = config.recurrent_coupling_scale
        self.retention_mode = config.retention_mode
        self.delta_slots = config.delta_slots
        if self.recurrence_mode == "spin_delta":
            # These experimental controls must not advance the random stream,
            # so every maintained parameter remains exactly paired by seed.
            with torch.random.fork_rng(devices=[]):
                self.delta_write_controller = nn.Linear(
                    config.d_model, self.spin.channels
                )
                self.delta_erase_controller = nn.Linear(
                    config.d_model, self.spin.channels
                )
                self.delta_erase_strength_controller = nn.Linear(
                    config.d_model, self.spin.channels
                )
                self.delta_query_controller = nn.Linear(
                    config.d_model, self.spin.channels
                )
            for controller in (
                self.delta_write_controller,
                self.delta_erase_controller,
                self.delta_erase_strength_controller,
                self.delta_query_controller,
            ):
                controller._pure_spin_zero_init = True
                nn.init.zeros_(controller.weight)
                nn.init.zeros_(controller.bias)
        else:
            self.delta_write_controller = None
            self.delta_erase_controller = None
            self.delta_erase_strength_controller = None
            self.delta_query_controller = None
        if self.recurrence_mode == "coupled_isotypic":
            # The replacement is an experimental branch, so constructing it
            # must not shift initialization of later parameters relative to
            # the maintained model under the same seed.
            with torch.random.fork_rng(devices=[]):
                coefficient_controller = nn.Linear(
                    config.d_model, len(subgroup_indices)
                )
            self.spin.coefficient_controller = coefficient_controller
            self.spin.coefficient_controller._pure_spin_zero_init = True
            nn.init.zeros_(self.spin.coefficient_controller.weight)
            nn.init.zeros_(self.spin.coefficient_controller.bias)
            if self.spin.channels < 2:
                raise ValueError("coupled_isotypic requires at least two channels")
        if self.recurrence_mode == "independent_block" and self.spin.channels < 2:
            raise ValueError("independent_block requires at least two channels")
        self.recurrent_pairs = tuple(combinations(range(self.spin.channels), 2))
        if self.recurrent_multiplicity == "orthogonal":
            with torch.random.fork_rng(devices=[]):
                recurrent_controller = nn.Linear(
                    config.d_model, len(self.recurrent_pairs)
                )
            self.recurrent_multiplicity_controller = recurrent_controller
            self.recurrent_multiplicity_controller._pure_spin_zero_init = True
            nn.init.zeros_(self.recurrent_multiplicity_controller.weight)
            nn.init.zeros_(self.recurrent_multiplicity_controller.bias)
        else:
            self.recurrent_multiplicity_controller = None
        self.state_norm = nn.RMSNorm(self.spin.output_size)
        self.readout_mode = config.readout
        self.multiplicity_router = config.multiplicity_router
        self.multiplicity_angle_limit = config.multiplicity_angle_limit
        self.multiplicity_pairs = tuple(combinations(range(self.spin.channels), 2))
        if self.multiplicity_router == "orthogonal_query":
            with torch.random.fork_rng(devices=[]):
                multiplicity_controller = nn.Linear(
                    config.d_model, len(self.multiplicity_pairs)
                )
            self.multiplicity_controller = multiplicity_controller
            self.multiplicity_controller._pure_spin_zero_init = True
            nn.init.zeros_(self.multiplicity_controller.weight)
            nn.init.zeros_(self.multiplicity_controller.bias)
        else:
            self.multiplicity_controller = None
        read_dimension = self.spin.output_size
        if self.readout_mode == "triality_invariants":
            # One energy per 8D triality sector and one cubic contraction per
            # channel. These scalars restore amplitude without choosing a
            # non-equivariant direction inside an irreducible Spin(8) module.
            read_dimension += 4 * self.spin.channels
        self.output_projection = nn.Linear(read_dimension, config.d_model, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))
        self.ffn_norm = nn.RMSNorm(config.d_model)
        mixer_types = {
            "swiglu": SwiGLU,
            "sol_bounded_quadratic": SolBoundedQuadratic,
            "sol_self_gate": SolSelfGate,
        }
        mixer_type = mixer_types[config.mixer]
        self.ffn = mixer_type(config.d_model, config.expansion)
        self.dropout = nn.Dropout(config.dropout)

    def _route_multiplicity(
        self, states: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        if self.multiplicity_controller is None:
            return states
        if query.shape[:-1] != states.shape[:-3]:
            raise ValueError("multiplicity query must share state leading dimensions")
        angles = self.multiplicity_angle_limit * torch.tanh(
            self.multiplicity_controller(query)
        )
        routed = [states[..., index, :, :] for index in range(self.spin.channels)]
        for factor, (left_index, right_index) in enumerate(self.multiplicity_pairs):
            cosine = torch.cos(angles[..., factor])[..., None, None]
            sine = torch.sin(angles[..., factor])[..., None, None]
            left = routed[left_index]
            right = routed[right_index]
            routed[left_index] = cosine * left - sine * right
            routed[right_index] = sine * left + cosine * right
        return torch.stack(routed, dim=-3)

    def _read_features(
        self, states: torch.Tensor, query: torch.Tensor | None = None
    ) -> torch.Tensor:
        expected = (
            self.spin.channels,
            len(self.spin.representations),
            8,
        )
        if states.shape[-3:] != expected:
            raise ValueError(f"triality states must end in {expected}")
        if self.multiplicity_controller is not None:
            if query is None:
                raise ValueError("orthogonal multiplicity routing requires a query")
            states = self._route_multiplicity(states, query)
        direction = self.state_norm(states.flatten(start_dim=-3))
        if self.readout_mode == "direction":
            return direction

        indices = {
            name: self.spin.representations.index(name)
            for name in ("vector", "positive", "negative")
        }
        vector = states[..., indices["vector"], :]
        positive = states[..., indices["positive"], :]
        negative = states[..., indices["negative"], :]
        energy = torch.log1p(states.square().mean(dim=-1))
        cubic = torch.einsum(
            "...v,...j,vji,...i->...",
            vector,
            negative,
            self.spin.rho.to(states),
            positive,
        )
        bounded_cubic = cubic * torch.rsqrt(1.0 + cubic.square())
        invariant_scalars = torch.cat(
            (energy.flatten(start_dim=-2), bounded_cubic), dim=-1
        )
        return torch.cat((direction, invariant_scalars), dim=-1)

    def forward(
        self,
        hidden: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        scan_mode: str = "compiled_controller",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value, gate = self.input_projection(self.norm(hidden)).chunk(2, dim=-1)
        value = F.silu(self.local_conv(value))
        if self.recurrence_mode == "spin_delta":
            if scan_mode not in {
                "delta_recurrent",
                "delta_parallel",
                "raw_cuda_delta",
            }:
                raise ValueError(
                    "spin_delta requires delta_recurrent, delta_parallel, "
                    "or raw_cuda_delta"
                )
            from chunk_parallel_scan import factorized_triality_actions
            from spin_delta_scan import (
                SpinDeltaTransition,
                contractive_delta_left,
                parallel_spin_delta_scan,
                read_delta_state,
                recurrent_spin_delta_scan,
                route_delta_drive,
            )

            if state is None:
                base_state = self.spin.initial_cache(value.shape[0], value)
                state = torch.stack((base_state, torch.zeros_like(base_state)), dim=2)
            normalized, scale, drive, coordinate_gate = (
                self.spin._normalized_control_fields(value, valid_mask)
            )
            factor_count = len(self.subgroup_indices)
            coordinates = self.spin.coefficient_controller(normalized).reshape(
                value.shape[0], value.shape[1], self.spin.channels, factor_count
            )
            coordinates = coordinates * coordinate_gate[..., None, None]
            assert self.delta_write_controller is not None
            assert self.delta_erase_controller is not None
            assert self.delta_erase_strength_controller is not None
            assert self.delta_query_controller is not None
            write_angle = self.multiplicity_angle_limit * torch.tanh(
                self.delta_write_controller(normalized)
            )
            erase_angle = self.multiplicity_angle_limit * torch.tanh(
                self.delta_erase_controller(normalized)
            )
            write_key = torch.stack(
                (torch.cos(write_angle), torch.sin(write_angle)), dim=-1
            )
            # The erase direction begins on the empty auxiliary slot.  This
            # preserves the maintained recurrence despite nonzero safe erase.
            erase_key = torch.stack(
                (torch.sin(erase_angle), torch.cos(erase_angle)), dim=-1
            )
            erase_strength = torch.sigmoid(
                self.delta_erase_strength_controller(normalized)
            ) * coordinate_gate[..., None]
            query_delta = torch.tanh(self.delta_query_controller(normalized))
            query = torch.stack(
                (1.0 + query_delta, 1.0 - query_delta), dim=-1
            )
            delta_left = contractive_delta_left(scale, erase_key, erase_strength)
            delta_drive = route_delta_drive(write_key, drive)
            if scan_mode == "raw_cuda_delta":
                from raw_cuda import raw_cuda_spin_delta_scan

                raw_slot_states = raw_cuda_spin_delta_scan(
                    coordinates,
                    self.subgroup_generators.to(value),
                    delta_left,
                    delta_drive,
                    state,
                )
                final_state = raw_slot_states[:, -1]
            else:
                transition = SpinDeltaTransition(
                    left=delta_left,
                    action=factorized_triality_actions(
                        coordinates, self.subgroup_generators.to(value)
                    ),
                    drive=delta_drive,
                )
                scanner = (
                    parallel_spin_delta_scan
                    if scan_mode == "delta_parallel"
                    else recurrent_spin_delta_scan
                )
                raw_slot_states, final_state = scanner(transition, state)
            raw_states = read_delta_state(raw_slot_states, query)
            states = self.spin.triality_readout(raw_states)
        elif self.recurrence_mode == "independent_block":
            if scan_mode not in {
                "block_recurrent",
                "block_parallel",
                "raw_cuda_block",
            }:
                raise ValueError(
                    "independent_block requires block_recurrent, block_parallel, "
                    "or raw_cuda_block"
                )
            from chunk_parallel_scan import factorized_triality_actions
            from coupled_isotypic_scan import (
                contractive_givens_left,
                retention_step_from_scale,
            )
            from independent_block_scan import (
                parallel_independent_block_scan,
                recurrent_independent_block_scan,
            )

            if state is None:
                state = self.spin.initial_cache(value.shape[0], value)
            normalized, scale, drive, coordinate_gate = (
                self.spin._normalized_control_fields(value, valid_mask)
            )
            factor_count = len(self.subgroup_indices)
            coordinates = self.spin.coefficient_controller(normalized).reshape(
                value.shape[0], value.shape[1], self.spin.channels, factor_count
            )
            coordinates = coordinates * coordinate_gate[..., None, None]
            if self.recurrent_multiplicity_controller is None:
                angles = value.new_zeros(*value.shape[:2], len(self.recurrent_pairs))
            else:
                angles = self.multiplicity_angle_limit * torch.tanh(
                    self.recurrent_multiplicity_controller(normalized)
                )
                angles = angles * coordinate_gate[..., None]
                if self.recurrent_coupling_scale == "retention_step":
                    angles = angles * retention_step_from_scale(scale)[..., None]
            left = contractive_givens_left(scale, angles, self.recurrent_pairs)
            if scan_mode == "raw_cuda_block":
                if self.spin.channels != 2:
                    raise ValueError(
                        "raw_cuda_block currently requires exactly two channels"
                    )
                from raw_cuda import raw_cuda_independent_block_scan

                raw_states = raw_cuda_independent_block_scan(
                    coordinates,
                    self.subgroup_generators.to(value),
                    left,
                    drive,
                    state,
                )
                final_state = raw_states[:, -1]
            else:
                actions = factorized_triality_actions(
                    coordinates, self.subgroup_generators.to(value)
                )
                scanner = (
                    parallel_independent_block_scan
                    if scan_mode == "block_parallel"
                    else recurrent_independent_block_scan
                )
                raw_states, final_state = scanner(left, actions, drive, state)
            states = self.spin.triality_readout(raw_states)
        elif self.recurrence_mode == "coupled_isotypic":
            if scan_mode not in {
                "coupled_recurrent",
                "coupled_parallel",
                "raw_cuda_coupled",
            }:
                raise ValueError(
                    "coupled_isotypic requires coupled_recurrent, "
                    "coupled_parallel, or raw_cuda_coupled"
                )
            from chunk_parallel_scan import factorized_triality_actions
            from coupled_isotypic_scan import (
                CoupledIsotypicTransition,
                contractive_givens_left,
                parallel_coupled_scan,
                recurrent_coupled_scan,
            )

            if state is None:
                state = self.spin.initial_cache(value.shape[0], value)
            normalized, scale, drive, coordinate_gate = (
                self.spin._normalized_control_fields(value, valid_mask)
            )
            coordinates = self.spin.coefficient_controller(normalized)
            coordinates = coordinates * coordinate_gate[..., None]
            if self.recurrent_multiplicity_controller is None:
                angles = value.new_zeros(*value.shape[:2], len(self.recurrent_pairs))
            else:
                angles = self.multiplicity_angle_limit * torch.tanh(
                    self.recurrent_multiplicity_controller(normalized)
                )
                angles = angles * coordinate_gate[..., None]
            left = contractive_givens_left(scale, angles, self.recurrent_pairs)
            if scan_mode == "raw_cuda_coupled":
                if self.spin.channels != 2:
                    raise ValueError(
                        "raw_cuda_coupled currently requires exactly two channels"
                    )
                from raw_cuda import raw_cuda_coupled_coordinate_scan

                raw_states = raw_cuda_coupled_coordinate_scan(
                    coordinates,
                    self.subgroup_generators.to(value),
                    left,
                    drive,
                    state,
                )
                final_state = raw_states[:, -1]
            else:
                actions = factorized_triality_actions(
                    coordinates.unsqueeze(-2), self.subgroup_generators.to(value)
                ).squeeze(-4)
                transition = CoupledIsotypicTransition(left, actions, drive)
                scanner = (
                    parallel_coupled_scan
                    if scan_mode == "coupled_parallel"
                    else recurrent_coupled_scan
                )
                raw_states, final_state = scanner(transition, state)
            states = self.spin.triality_readout(raw_states)
        elif scan_mode == "raw_cuda_controller":
            from raw_cuda import raw_cuda_controller_factorized_scan

            if self.retention_mode != "shared":
                raise ValueError(
                    "isotypic retention requires raw_cuda_hybrid, "
                    "raw_cuda_isotypic, or chunk_parallel"
                )
            if state is None:
                state = self.spin.initial_cache(value.shape[0], value)
            normalized, scale, drive, coordinate_gate = (
                self.spin._normalized_control_fields(value, valid_mask)
            )
            raw_states = raw_cuda_controller_factorized_scan(
                normalized,
                self.spin.coefficient_controller.weight,
                self.spin.coefficient_controller.bias,
                self.spin.generators.to(value),
                scale,
                drive,
                state,
                coordinate_gate,
            )
            states = self.spin.triality_readout(raw_states)
            final_state = raw_states[:, -1]
        elif scan_mode in {
            "raw_cuda_factorized",
            "raw_cuda_isotypic",
            "raw_cuda_hybrid",
            "chunk_parallel",
        }:
            from raw_cuda import (
                raw_cuda_coordinate_factorized_scan,
                raw_cuda_hybrid_coordinate_scan,
                raw_cuda_isotypic_coordinate_scan,
            )

            if state is None:
                state = self.spin.initial_cache(value.shape[0], value)
            normalized, scale, drive, coordinate_gate = (
                self.spin._normalized_control_fields(value, valid_mask)
            )
            if self.retention_mode != "shared" and scan_mode in {
                "raw_cuda_controller",
                "raw_cuda_factorized",
            }:
                raise ValueError(
                    "isotypic retention requires raw_cuda_hybrid, "
                    "raw_cuda_isotypic, or chunk_parallel"
                )
            factor_count = len(self.subgroup_indices)
            coordinates = self.spin.coefficient_controller(normalized).reshape(
                value.shape[0], value.shape[1], self.spin.channels, factor_count
            )
            coordinates = coordinates * coordinate_gate[..., None, None]
            if scan_mode == "chunk_parallel":
                from pure_spin8_ssm.torch_backend import Spin8AffineTransition

                from chunk_parallel_scan import (
                    chunk_parallel_spin8_scan,
                    factorized_triality_actions,
                )

                transition = Spin8AffineTransition(
                    scale=scale,
                    action=factorized_triality_actions(
                        coordinates, self.subgroup_generators.to(value)
                    ),
                    drive=drive,
                )
                raw_states, final_state = chunk_parallel_spin8_scan(
                    transition, state, chunk_size=self.scan_chunk_size
                )
            else:
                scan = (
                    raw_cuda_hybrid_coordinate_scan
                    if scan_mode == "raw_cuda_hybrid"
                    else (
                        raw_cuda_isotypic_coordinate_scan
                        if scan_mode == "raw_cuda_isotypic"
                        else raw_cuda_coordinate_factorized_scan
                    )
                )
                raw_states = scan(
                    coordinates,
                    self.subgroup_generators.to(value),
                    scale,
                    drive,
                    state,
                )
                final_state = raw_states[:, -1]
            states = self.spin.triality_readout(raw_states)
        else:
            if self.group_dimension != 8:
                raise ValueError(
                    "nested subgroup schedules currently require raw_cuda_factorized"
                )
            states, final_state = self.spin(
                value,
                state,
                valid_mask=valid_mask,
                scan_mode=scan_mode,
            )
        update = self.output_projection(self._read_features(states, value))
        update = update * torch.sigmoid(gate)
        hidden = hidden + torch.sigmoid(self.residual_scale) * self.dropout(update)
        return hidden + self.dropout(self.ffn(self.ffn_norm(hidden))), final_state


class PureSpinSSMV12(nn.Module):
    model_version = __version__

    def __init__(self, config: PureSpinV12Config) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        schedule = config.group_schedule or (8,) * config.num_layers
        self.blocks = nn.ModuleList(
            PureSpinV12Block(config, group_dimension) for group_dimension in schedule
        )
        self.final_norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        self.apply(_initialize_language_model_module)

    @property
    def cache_scalars(self) -> int:
        return sum(
            block.spin.cache_scalars
            * (block.delta_slots if block.recurrence_mode == "spin_delta" else 1)
            for block in self.blocks
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        states: Sequence[torch.Tensor | None] | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        scan_mode: str = "compiled_controller",
    ) -> dict[str, Any]:
        if token_ids.ndim != 2 or token_ids.shape[1] == 0:
            raise ValueError("token_ids must have nonempty shape (batch,length)")
        if states is None:
            states = [None] * len(self.blocks)
        if len(states) != len(self.blocks):
            raise ValueError("one recurrent state is required per block")
        hidden = self.embedding(token_ids)
        next_states = []
        for block, state in zip(self.blocks, states, strict=True):
            hidden, state = block(
                hidden,
                state,
                valid_mask=valid_mask,
                scan_mode=scan_mode,
            )
            next_states.append(state)
        return {"logits": self.lm_head(self.final_norm(hidden)), "states": next_states}

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any]) -> None:
        torch.save(
            {
                "format_version": 1,
                "model_type": "pure_spin_ssm_v1_2",
                "model_version": __version__,
                "config": asdict(self.config),
                "state_dict": {
                    k: v.detach().cpu() for k, v in self.state_dict().items()
                },
                "metadata": metadata,
            },
            Path(path),
        )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _initialize_language_model_module(module: nn.Module) -> None:
    if isinstance(module, (nn.Embedding, nn.Linear)):
        if getattr(module, "_pure_spin_zero_init", False):
            nn.init.zeros_(module.weight)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)


__all__ = [
    "PureSpinSSMV12",
    "PureSpinV12Config",
    "SolBoundedQuadratic",
    "SolSelfGate",
    "parameter_count",
]
