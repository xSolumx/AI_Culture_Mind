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
    group_schedule: tuple[int, ...] | None = None
    dropout: float = 0.0
    min_retention_logit: float = 2.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if min(
            self.vocab_size,
            self.d_model,
            self.num_layers,
            self.spin_channels,
            self.d_conv,
            self.expansion,
        ) < 1:
            raise ValueError("all model dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0,1)")
        if self.mixer not in (
            "swiglu",
            "sol_bounded_quadratic",
            "sol_self_gate",
        ):
            raise ValueError("unknown channel mixer")
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
        self.input_projection = nn.Linear(config.d_model, 2 * config.d_model, bias=False)
        self.local_conv = CausalDepthwiseConv1d(config.d_model, config.d_conv)
        self.spin = PureSpin8SSMLayer(
            config.d_model,
            channels=config.spin_channels,
            action_mode="factorized",
            min_retention_logit=config.min_retention_logit,
            triality_coupling=True,
            normalize_inputs=True,
        )
        generator_pairs = tuple(combinations(range(8), 2))
        subgroup_indices = tuple(
            index
            for index, (left, right) in enumerate(generator_pairs)
            if left < group_dimension and right < group_dimension
        )
        self.group_dimension = group_dimension
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
        self.state_norm = nn.RMSNorm(self.spin.output_size)
        self.output_projection = nn.Linear(self.spin.output_size, config.d_model, bias=False)
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
        if scan_mode == "raw_cuda_controller":
            from raw_cuda import raw_cuda_controller_factorized_scan

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
        elif scan_mode == "raw_cuda_factorized":
            from raw_cuda import raw_cuda_coordinate_factorized_scan

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
            raw_states = raw_cuda_coordinate_factorized_scan(
                coordinates,
                self.subgroup_generators.to(value),
                scale,
                drive,
                state,
            )
            states = self.spin.triality_readout(raw_states)
            final_state = raw_states[:, -1]
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
        update = self.output_projection(self.state_norm(states.flatten(start_dim=-3)))
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
            PureSpinV12Block(config, group_dimension)
            for group_dimension in schedule
        )
        self.final_norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        self.apply(_initialize_language_model_module)

    @property
    def cache_scalars(self) -> int:
        return sum(block.spin.cache_scalars for block in self.blocks)

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
                "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
                "metadata": metadata,
            },
            Path(path),
        )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _initialize_language_model_module(module: nn.Module) -> None:
    if isinstance(module, (nn.Embedding, nn.Linear)):
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
