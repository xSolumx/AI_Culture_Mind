"""Pure Exceptional Delta SSM v1.3 semantic model.

This is the correctness-first PyTorch oracle.  It deliberately exposes the
algebra, update rank, key tying, retention, local mixer, and channel mixer as
independent choices so an old benchmark gate cannot silently become a model
axiom.  Kernel promotion follows semantic and empirical falsification.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

from .action import IdentityAction, build_exceptional_action
from .albert import (
    ALBERT_DIM,
    albert_determinant,
    albert_determinant_via_jordan,
    albert_trace,
    jordan_product,
    octonion_structure_constants,
    orthonormal_jordan_structure_constants,
    sparse_jordan_product,
)
from .scan import (
    compile_delta_transition,
    compile_one_sided_delta_transition,
    parallel_delta_scan,
    parallel_one_sided_delta_scan,
    recurrent_delta_scan,
    recurrent_one_sided_delta_scan,
)

__version__ = "1.3.0-dev"


@dataclass(frozen=True)
class ExceptionalDeltaConfig:
    vocab_size: int = 256
    d_model: int = 96
    num_layers: int = 4
    memory_width: int = 8
    update_rank: int = 2
    action_algebra: Literal["identity", "spin8", "spin9", "f4", "e6"] = "e6"
    action_geometry: Literal["direct", "polar", "cartan"] = "direct"
    action_schedule: (
        tuple[Literal["identity", "spin8", "spin9", "f4", "e6"], ...] | None
    ) = None
    action_coordinate_scale: float = 0.02
    action_factors: int = 1
    identity_fast_path: bool = True
    d_conv: int = 4
    local_mixer: Literal["depthwise_conv", "none"] = "depthwise_conv"
    channel_mixer: Literal["swiglu", "jordan", "none"] = "jordan"
    readout_mode: Literal["auto", "vector", "albert_invariants"] = "auto"
    # The explicit cubic is algebraically equivalent, but its altered float32
    # evaluation order failed the prospective five-seed quality gate.
    albert_determinant_backend: Literal["explicit", "jordan"] = "jordan"
    albert_product_backend: Literal["sparse", "dense"] = "dense"
    expansion: int = 2
    key_parameterization: Literal[
        "independent_bounded", "tied_delta", "unconstrained"
    ] = "independent_bounded"
    retention_parameterization: Literal[
        "sigmoid", "exp_negative", "unconstrained"
    ] = "exp_negative"
    retention_bias: float = 2.0
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        dimensions = (
            self.vocab_size,
            self.d_model,
            self.num_layers,
            self.memory_width,
            self.update_rank,
            self.action_factors,
            self.expansion,
        )
        if min(dimensions) < 1:
            raise ValueError("all model dimensions must be positive")
        if self.d_conv < 1:
            raise ValueError("d_conv must be positive")
        if self.action_coordinate_scale < 0:
            raise ValueError("action_coordinate_scale must be nonnegative")
        if self.albert_product_backend not in {"sparse", "dense"}:
            raise ValueError("albert_product_backend must be sparse or dense")
        if self.albert_determinant_backend not in {"explicit", "jordan"}:
            raise ValueError("albert_determinant_backend must be explicit or jordan")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if (
            self.action_schedule is not None
            and len(self.action_schedule) != self.num_layers
        ):
            raise ValueError("action_schedule must have one entry per layer")


@dataclass(frozen=True)
class ExceptionalDeltaState:
    """Complete streaming state, including the formerly omitted conv cache."""

    memory: torch.Tensor
    convolution: torch.Tensor


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, width: int, kernel_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            width,
            width,
            kernel_size,
            groups=width,
            padding=0,
        )

    @property
    def cache_width(self) -> int:
        return self.conv.kernel_size[0] - 1

    def forward(
        self,
        inputs: torch.Tensor,
        cache: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        channels = inputs.transpose(1, 2)
        if cache is None:
            cache = channels.new_zeros(
                channels.shape[0], channels.shape[1], self.cache_width
            )
        expected = (channels.shape[0], channels.shape[1], self.cache_width)
        if cache.shape != expected:
            raise ValueError(f"convolution cache must have shape {expected}")
        if valid_mask is None:
            combined = torch.cat((cache, channels), dim=-1)
            result = self.conv(combined).transpose(1, 2)
            next_cache = (
                combined[..., -self.cache_width :]
                if self.cache_width
                else combined[..., :0]
            )
            return result, next_cache
        if valid_mask.shape != inputs.shape[:2]:
            raise ValueError("valid_mask must have shape (batch,length)")
        outputs = []
        for position in range(inputs.shape[1]):
            window = torch.cat((cache, channels[..., position : position + 1]), dim=-1)
            outputs.append(self.conv(window))
            candidate = window[..., 1:] if self.cache_width else window[..., :0]
            cache = torch.where(valid_mask[:, position, None, None], candidate, cache)
        return torch.cat(outputs, dim=-1).transpose(1, 2), cache


class SwiGLU(nn.Module):
    def __init__(self, width: int, expansion: int) -> None:
        super().__init__()
        hidden = width * expansion
        self.input = nn.Linear(width, 2 * hidden, bias=False)
        self.output = nn.Linear(hidden, width, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value, gate = self.input(inputs).chunk(2, dim=-1)
        return self.output(value * F.silu(gate))


class AlbertJordanMixer(nn.Module):
    """Pointwise bilinear mixer using the full 27D Albert product."""

    def __init__(
        self, width: int, expansion: int, backend: Literal["sparse", "dense"]
    ) -> None:
        super().__init__()
        self.copies = max(1, (width * expansion + ALBERT_DIM - 1) // ALBERT_DIM)
        hidden = self.copies * ALBERT_DIM
        self.input = nn.Linear(width, 2 * hidden, bias=False)
        self.output = nn.Linear(hidden, width, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))
        self.backend = backend
        structure = torch.tensor(
            orthonormal_jordan_structure_constants(), dtype=torch.float64
        )
        if backend == "dense":
            self.register_buffer("structure", structure, persistent=False)
            indices = (None, None, None)
            coefficients = None
        else:
            nonzero = torch.nonzero(structure, as_tuple=True)
            indices = nonzero
            coefficients = structure[nonzero]
            self.register_buffer("structure", None, persistent=False)
        self.register_buffer("output_index", indices[0], persistent=False)
        self.register_buffer("left_index", indices[1], persistent=False)
        self.register_buffer("right_index", indices[2], persistent=False)
        self.register_buffer("coefficients", coefficients, persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        left, right = self.input(inputs).chunk(2, dim=-1)
        shape = (*inputs.shape[:-1], self.copies, ALBERT_DIM)
        left, right = left.reshape(shape), right.reshape(shape)
        if self.backend == "dense":
            product = jordan_product(left, right, self.structure)
        else:
            product = sparse_jordan_product(
                left,
                right,
                self.output_index,
                self.left_index,
                self.right_index,
                self.coefficients,
            )
        return torch.sigmoid(self.residual_scale) * self.output(
            product.flatten(start_dim=-2)
        )


class AlbertInvariantReadout(nn.Module):
    """Preserve direction and expose three scale/invariant summary channels."""

    output_dim = ALBERT_DIM + 3

    def __init__(self, backend: Literal["explicit", "jordan"] = "jordan") -> None:
        super().__init__()
        self.backend = backend
        self.direction_norm = nn.RMSNorm(ALBERT_DIM)

    def forward(
        self, value: torch.Tensor, octonion_structure: torch.Tensor | None = None
    ) -> torch.Tensor:
        if value.shape[-1] != ALBERT_DIM:
            raise ValueError("Albert invariant readout requires 27D values")
        trace = albert_trace(value) / (3.0**0.5)
        log_energy = torch.log1p(value.square().mean(dim=-1))
        determinant = (
            albert_determinant(value, octonion_structure)
            if self.backend == "explicit"
            else albert_determinant_via_jordan(value, octonion_structure)
        )
        bounded_determinant = determinant * torch.rsqrt(1.0 + determinant.square())
        scalars = torch.stack((trace, log_energy, bounded_determinant), dim=-1)
        return torch.cat((self.direction_norm(value), scalars), dim=-1)


class ExceptionalDeltaBlock(nn.Module):
    def __init__(
        self,
        config: ExceptionalDeltaConfig,
        action_algebra: Literal["identity", "spin8", "spin9", "f4", "e6"] | None = None,
        generators: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.norm = nn.RMSNorm(config.d_model)
        self.input_projection = nn.Linear(config.d_model, 2 * config.d_model, bias=False)
        self.local = (
            CausalDepthwiseConv1d(config.d_model, config.d_conv)
            if config.local_mixer == "depthwise_conv"
            else None
        )
        self.action = build_exceptional_action(
            action_algebra or config.action_algebra,
            geometry=config.action_geometry,
            generators=generators,
        )
        h = config.memory_width
        r = config.update_rank
        v = self.action.representation_dim
        f = self.action.coordinate_dim
        self._segments = {
            "coordinates": config.action_factors * f,
            "retention": h,
            "write_key": r * h,
            "erase_key": r * h,
            "erase_gate": r,
            "write_value": r * v,
            "query": h,
        }
        self.controller = nn.Linear(config.d_model, sum(self._segments.values()))
        use_invariants = config.readout_mode == "albert_invariants" or (
            config.readout_mode == "auto" and v == ALBERT_DIM
        )
        if config.readout_mode == "albert_invariants" and v != ALBERT_DIM:
            raise ValueError("albert_invariants readout requires a 27D action")
        if use_invariants:
            self.read_features = AlbertInvariantReadout(
                config.albert_determinant_backend
            )
            read_dimension = self.read_features.output_dim
        else:
            self.read_features = nn.RMSNorm(v)
            read_dimension = v
        self.output_projection = nn.Linear(read_dimension, config.d_model, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))
        if config.channel_mixer == "swiglu":
            self.channel_norm = nn.RMSNorm(config.d_model)
            self.channel = SwiGLU(config.d_model, config.expansion)
        elif config.channel_mixer == "jordan":
            self.channel_norm = nn.RMSNorm(config.d_model)
            self.channel = AlbertJordanMixer(
                config.d_model, config.expansion, config.albert_product_backend
            )
        else:
            self.channel_norm = nn.Identity()
            self.channel = None
        self.dropout = nn.Dropout(config.dropout)
        if use_invariants and config.albert_determinant_backend == "explicit":
            readout_structure = torch.tensor(
                octonion_structure_constants(), dtype=torch.float64
            )
        elif use_invariants:
            readout_structure = torch.tensor(
                orthonormal_jordan_structure_constants(), dtype=torch.float64
            )
        else:
            readout_structure = None
        self.register_buffer("readout_structure", readout_structure, persistent=False)
        self._initialize_controller()

    @property
    def state_shape(self) -> tuple[int, int]:
        return self.config.memory_width, self.action.representation_dim

    @property
    def convolution_cache_scalars(self) -> int:
        if self.local is None:
            return 0
        return self.config.d_model * self.local.cache_width

    def _initialize_controller(self) -> None:
        nn.init.normal_(self.controller.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.controller.bias)
        start = self._segments["coordinates"]
        stop = start + self._segments["retention"]
        with torch.no_grad():
            self.controller.bias[start:stop].fill_(self.config.retention_bias)

    def _retention(self, raw: torch.Tensor) -> torch.Tensor:
        mode = self.config.retention_parameterization
        if mode == "sigmoid":
            return torch.sigmoid(raw)
        if mode == "exp_negative":
            return torch.exp(-F.softplus(-raw))
        return raw

    def _keys(
        self, write_key: torch.Tensor, erase_key: torch.Tensor, erase_gate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode = self.config.key_parameterization
        if mode == "unconstrained":
            return write_key, erase_key
        write_key = F.normalize(write_key, dim=-1)
        strength = torch.sigmoid(erase_gate)[..., None] / self.config.update_rank
        if mode == "tied_delta":
            return write_key, strength * write_key
        return write_key, strength * F.normalize(erase_key, dim=-1)

    def _control_fields(
        self, hidden: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        raw = self.controller(hidden)
        pieces = torch.split(raw, tuple(self._segments.values()), dim=-1)
        fields = dict(zip(self._segments, pieces, strict=True))
        batch, length = hidden.shape[:2]
        rank = self.config.update_rank
        width = self.config.memory_width
        value_dim = self.action.representation_dim
        fields["coordinates"] = (
            fields["coordinates"].reshape(
                batch, length, self.config.action_factors, self.action.coordinate_dim
            )
            * self.config.action_coordinate_scale
        )
        fields["retention"] = self._retention(fields["retention"])
        fields["write_key"] = fields["write_key"].reshape(batch, length, rank, width)
        fields["erase_key"] = fields["erase_key"].reshape(batch, length, rank, width)
        fields["erase_gate"] = fields["erase_gate"].reshape(batch, length, rank)
        fields["write_value"] = fields["write_value"].reshape(
            batch, length, rank, value_dim
        )
        fields["write_key"], fields["erase_key"] = self._keys(
            fields["write_key"], fields["erase_key"], fields["erase_gate"]
        )
        if valid_mask is not None:
            if valid_mask.shape != hidden.shape[:2]:
                raise ValueError("valid_mask must have shape (batch,length)")
            mask = valid_mask.to(dtype=hidden.dtype)
            fields["coordinates"] = fields["coordinates"] * mask[..., None, None]
            fields["retention"] = torch.where(
                valid_mask[..., None], fields["retention"], torch.ones_like(fields["retention"])
            )
            fields["write_key"] = fields["write_key"] * mask[..., None, None]
            fields["write_value"] = fields["write_value"] * mask[..., None, None]
            fields["query"] = fields["query"] * mask[..., None]
        return fields

    def forward(
        self,
        hidden: torch.Tensor,
        state: ExceptionalDeltaState | None = None,
        *,
        scan_mode: Literal["auto", "recurrent", "parallel"] = "auto",
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ExceptionalDeltaState]:
        value, gate = self.input_projection(self.norm(hidden)).chunk(2, dim=-1)
        if valid_mask is not None:
            value = value * valid_mask[..., None].to(value.dtype)
        convolution_cache = None if state is None else state.convolution
        if self.local is None:
            local_value = value
            next_convolution = value.new_zeros((value.shape[0], value.shape[-1], 0))
        else:
            local_value, next_convolution = self.local(
                value, convolution_cache, valid_mask
            )
        value = F.silu(local_value)
        fields = self._control_fields(value, valid_mask)
        use_identity_fast_path = self.config.identity_fast_path and isinstance(
            self.action, IdentityAction
        )
        if use_identity_fast_path:
            transition = compile_one_sided_delta_transition(
                fields["retention"],
                fields["write_key"],
                fields["erase_key"],
                fields["write_value"],
            )
        else:
            action = self.action.ordered(fields["coordinates"])
            transition = compile_delta_transition(
                fields["retention"],
                fields["write_key"],
                fields["erase_key"],
                fields["write_value"],
                action,
            )
        memory = None if state is None else state.memory
        if memory is None:
            memory = value.new_zeros((value.shape[0], *self.state_shape))
        if scan_mode == "auto":
            scan_mode = "parallel" if value.shape[1] > 1 else "recurrent"
        if use_identity_fast_path:
            scanner = (
                parallel_one_sided_delta_scan
                if scan_mode == "parallel"
                else recurrent_one_sided_delta_scan
            )
        else:
            scanner = parallel_delta_scan if scan_mode == "parallel" else recurrent_delta_scan
        reads, _, final_memory = scanner(transition, memory, fields["query"])
        if reads is None:
            raise AssertionError("query was supplied but scan returned no reads")
        if isinstance(self.read_features, AlbertInvariantReadout):
            read_features = self.read_features(reads, self.readout_structure)
        else:
            read_features = self.read_features(reads)
        update = self.output_projection(read_features) * torch.sigmoid(gate)
        hidden = hidden + torch.sigmoid(self.residual_scale) * self.dropout(update)
        if self.channel is not None:
            channel_input = self.channel_norm(hidden)
            channel_update = self.channel(channel_input)
            hidden = hidden + self.dropout(channel_update)
        return hidden, ExceptionalDeltaState(final_memory, next_convolution)


class ExceptionalDeltaLM(nn.Module):
    model_version = __version__

    def __init__(
        self,
        config: ExceptionalDeltaConfig,
        *,
        generator_banks: Sequence[torch.Tensor | None] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        schedule = config.action_schedule or (config.action_algebra,) * config.num_layers
        if generator_banks is None:
            generator_banks = (None,) * config.num_layers
        if len(generator_banks) != config.num_layers:
            raise ValueError("generator_banks must have one entry per layer")
        self.blocks = nn.ModuleList(
            ExceptionalDeltaBlock(config, algebra, generators)
            for algebra, generators in zip(schedule, generator_banks, strict=True)
        )
        self.final_norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        self.apply(self._initialize_module)
        # Restore the deliberately structured controller initialization after
        # the generic language-model initializer has visited every Linear.
        for block in self.blocks:
            block._initialize_controller()

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def cache_scalars(self) -> int:
        return sum(
            block.state_shape[0] * block.state_shape[1]
            + block.convolution_cache_scalars
            for block in self.blocks
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        states: Sequence[ExceptionalDeltaState | None] | None = None,
        *,
        scan_mode: Literal["auto", "recurrent", "parallel"] = "auto",
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if token_ids.ndim != 2 or token_ids.shape[1] < 1:
            raise ValueError("token_ids must have nonempty shape (batch,length)")
        if states is None:
            states = [None] * len(self.blocks)
        if len(states) != len(self.blocks):
            raise ValueError("one recurrent state is required per block")
        hidden = self.embedding(token_ids)
        next_states = []
        for block, state in zip(self.blocks, states, strict=True):
            hidden, next_state = block(
                hidden,
                state,
                scan_mode=scan_mode,
                valid_mask=valid_mask,
            )
            next_states.append(next_state)
        logits = self.lm_head(self.final_norm(hidden))
        return {"logits": logits, "states": next_states}

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any]) -> None:
        torch.save(
            {
                "format_version": 1,
                "model_type": "pure_exceptional_delta_ssm_v1_3",
                "model_version": __version__,
                "config": asdict(self.config),
                "state_dict": {
                    key: value.detach().cpu() for key, value in self.state_dict().items()
                },
                "metadata": metadata,
            },
            Path(path),
        )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = [
    "AlbertInvariantReadout",
    "AlbertJordanMixer",
    "ExceptionalDeltaBlock",
    "ExceptionalDeltaConfig",
    "ExceptionalDeltaLM",
    "ExceptionalDeltaState",
    "SwiGLU",
    "parameter_count",
]
