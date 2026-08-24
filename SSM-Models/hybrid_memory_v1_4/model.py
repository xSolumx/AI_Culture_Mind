"""Complete hybrid attention and recurrent-memory language model."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Literal, TypeAlias

import torch
from delta_product_reference import DeltaProductReferenceLayer, GatedMLP
from torch import nn
from torch.nn import functional as F

from .attention import AttentionConfig, AttentionState, CausalSelfAttention
from .gated_delta import GatedDeltaConfig, GatedDeltaMemory
from .selected_block import RouteMode, SelectedBlockConfig, SelectedBlockMemory
from .structured_memory import StructuredMemoryConfig, StructuredSpin8Memory

__version__ = "1.4.1"

LayerKind: TypeAlias = Literal[
    "attention",
    "gated_delta",
    "delta_product",
    "selected_block",
    "structured_spin8",
]
DeltaScanMode: TypeAlias = Literal["recurrent", "parallel"]
SelectedScanMode: TypeAlias = Literal[
    "physical_gather", "dense_recurrent", "dense_parallel"
]
StructuredScanMode: TypeAlias = Literal["recurrent", "parallel"]

_LAYER_KINDS = (
    "attention",
    "gated_delta",
    "delta_product",
    "selected_block",
    "structured_spin8",
)
_DELTA_SCAN_MODES = ("recurrent", "parallel")
_SELECTED_SCAN_MODES = ("physical_gather", "dense_recurrent", "dense_parallel")
_SELECTED_ROUTE_MODES = ("hard", "soft", "straight_through")
_STRUCTURED_SCAN_MODES = ("recurrent", "parallel")
_POSITION_BYTES = 8


def _positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


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
class HybridMemoryConfig:
    """Configuration with one explicit mixer choice for every model depth."""

    vocab_size: int = 256
    model_dim: int = 64
    # The v1.4.1 default pivots the primary memory from value-only selected
    # slots to content-addressed fast weights. Selected/Spin(8) tiers remain
    # explicit experimental archive/transport options.
    layer_plan: tuple[LayerKind, ...] = ("gated_delta", "attention")

    attention_heads: int = 4
    attention_window_size: int = 2048
    attention_rope_base: float = 10_000.0

    delta_heads: int = 4
    delta_num_householder: int = 4
    delta_allow_negative_eigenvalues: bool = True

    gated_delta_heads: int = 4
    gated_delta_key_dim: int | None = None
    gated_delta_value_dim: int | None = None
    gated_delta_allow_negative_eigenvalues: bool = False
    gated_delta_minimum_retention: float = 0.90
    gated_delta_initial_retention: float = 0.995
    gated_delta_initial_write_strength: float = 0.10

    selected_heads: int = 2
    selected_blocks: int = 4
    selected_slots_per_block: int = 2
    selected_value_dim: int = 16
    selected_update_rank: int = 1
    selected_controller_rank: int | None = None
    selected_retention_min: float = 0.0
    selected_retention_max: float = 0.99

    structured_channels: int = 1
    structured_rungs: tuple[int, ...] = (3, 4, 6, 8)
    structured_controller_rank: int | None = None
    structured_retention_min: float = 0.0
    structured_retention_max: float = 0.99
    structured_hard_eval: bool = True

    use_local_conv: bool = True
    conv_kernel: int = 4
    expansion: int = 2
    norm_epsilon: float = 1e-6
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "model_dim",
            "attention_heads",
            "attention_window_size",
            "delta_heads",
            "delta_num_householder",
            "gated_delta_heads",
            "selected_heads",
            "selected_blocks",
            "selected_slots_per_block",
            "selected_value_dim",
            "selected_update_rank",
            "structured_channels",
            "conv_kernel",
            "expansion",
        ):
            _positive_integer(name, getattr(self, name))
        if type(self.layer_plan) is not tuple:
            raise TypeError("layer_plan must be a tuple")
        if not self.layer_plan:
            raise ValueError("layer_plan must contain at least one layer")
        for kind in self.layer_plan:
            if kind not in _LAYER_KINDS:
                raise ValueError(f"layer_plan entries must be one of {_LAYER_KINDS}")
        if self.selected_controller_rank is not None:
            _positive_integer("selected_controller_rank", self.selected_controller_rank)
        for name in ("gated_delta_key_dim", "gated_delta_value_dim"):
            value = getattr(self, name)
            if value is not None:
                _positive_integer(name, value)
        if self.structured_controller_rank is not None:
            _positive_integer(
                "structured_controller_rank", self.structured_controller_rank
            )
        for name in (
            "use_local_conv",
            "tie_embeddings",
            "delta_allow_negative_eigenvalues",
            "gated_delta_allow_negative_eigenvalues",
            "structured_hard_eval",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if not math.isfinite(self.norm_epsilon) or self.norm_epsilon <= 0.0:
            raise ValueError("norm_epsilon must be finite and positive")
        if (
            not math.isfinite(self.attention_rope_base)
            or self.attention_rope_base <= 0.0
        ):
            raise ValueError("attention_rope_base must be finite and positive")
        if not (
            0.0 <= self.selected_retention_min <= self.selected_retention_max < 1.0
        ):
            raise ValueError(
                "selected retention bounds must satisfy 0 <= min <= max < 1"
            )
        if self.selected_update_rank > self.selected_slots_per_block:
            raise ValueError(
                "selected_update_rank cannot exceed selected_slots_per_block"
            )
        if "attention" in self.layer_plan:
            AttentionConfig(
                model_dim=self.model_dim,
                heads=self.attention_heads,
                window_size=self.attention_window_size,
                rope_base=self.attention_rope_base,
                dropout=self.dropout,
            )
        if "delta_product" in self.layer_plan and self.model_dim % self.delta_heads:
            raise ValueError("model_dim must be divisible by delta_heads")
        if "gated_delta" in self.layer_plan:
            GatedDeltaConfig(
                model_dim=self.model_dim,
                heads=self.gated_delta_heads,
                key_dim=self.gated_delta_key_dim,
                value_dim=self.gated_delta_value_dim,
                allow_negative_eigenvalues=(
                    self.gated_delta_allow_negative_eigenvalues
                ),
                norm_epsilon=self.norm_epsilon,
                minimum_retention=self.gated_delta_minimum_retention,
                initial_retention=self.gated_delta_initial_retention,
                initial_write_strength=self.gated_delta_initial_write_strength,
            )
        StructuredMemoryConfig(
            model_dim=self.model_dim,
            channels=self.structured_channels,
            rungs=self.structured_rungs,
            controller_rank=self.structured_controller_rank,
            retention_min=self.structured_retention_min,
            retention_max=self.structured_retention_max,
            hard_eval=self.structured_hard_eval,
        )

    @property
    def num_layers(self) -> int:
        return len(self.layer_plan)


@dataclass(frozen=True)
class DeltaProductState:
    """Complete DeltaProduct streaming state, including local-convolution history."""

    memory: torch.Tensor
    convolution: torch.Tensor

    @property
    def actual_bytes(self) -> int:
        return _tensor_bytes(self.memory) + _tensor_bytes(self.convolution)

    @property
    def nbytes(self) -> int:
        return self.actual_bytes


@dataclass(frozen=True)
class GatedDeltaState:
    """Complete Gated DeltaNet fast-weight and local-convolution state."""

    memory: torch.Tensor
    convolution: torch.Tensor

    @property
    def actual_bytes(self) -> int:
        return _tensor_bytes(self.memory) + _tensor_bytes(self.convolution)

    @property
    def nbytes(self) -> int:
        return self.actual_bytes


@dataclass(frozen=True)
class SelectedBlockState:
    """Complete selected-block streaming state and local-convolution history."""

    memory: torch.Tensor
    convolution: torch.Tensor

    @property
    def actual_bytes(self) -> int:
        return _tensor_bytes(self.memory) + _tensor_bytes(self.convolution)

    @property
    def nbytes(self) -> int:
        return self.actual_bytes


@dataclass(frozen=True)
class StructuredSpin8State:
    """Complete structured Spin(8) memory and local-convolution history."""

    memory: torch.Tensor
    convolution: torch.Tensor

    @property
    def actual_bytes(self) -> int:
        return _tensor_bytes(self.memory) + _tensor_bytes(self.convolution)

    @property
    def nbytes(self) -> int:
        return self.actual_bytes


LayerState: TypeAlias = (
    AttentionState
    | GatedDeltaState
    | DeltaProductState
    | SelectedBlockState
    | StructuredSpin8State
)
LayerDiagnostics: TypeAlias = dict[str, Any] | None


class CausalDepthwiseConv1d(nn.Module):
    """Causal depthwise convolution with complete arbitrary-chunk cache semantics."""

    def __init__(self, width: int, kernel_size: int) -> None:
        super().__init__()
        _positive_integer("width", width)
        _positive_integer("kernel_size", kernel_size)
        self.width = width
        self.conv = nn.Conv1d(
            width,
            width,
            kernel_size,
            groups=width,
            padding=0,
            bias=True,
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
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a tensor")
        if inputs.ndim != 3 or inputs.shape[-1] != self.width:
            raise ValueError("inputs must have shape (batch, length, width)")
        if inputs.shape[0] < 1 or inputs.shape[1] < 1:
            raise ValueError("inputs must have nonempty batch and sequence dimensions")
        if not inputs.is_floating_point():
            raise TypeError("inputs must have a floating-point dtype")
        parameter = self.conv.weight
        if inputs.dtype != parameter.dtype or inputs.device != parameter.device:
            raise ValueError("inputs must match the convolution dtype and device")

        channels = inputs.transpose(1, 2)
        expected = (channels.shape[0], channels.shape[1], self.cache_width)
        if cache is None:
            cache = channels.new_zeros(expected)
        elif not isinstance(cache, torch.Tensor):
            raise TypeError("convolution cache must be a tensor or None")
        if cache.shape != expected:
            raise ValueError(f"convolution cache must have shape {expected}")
        if not cache.is_floating_point():
            raise TypeError("convolution cache must have a floating-point dtype")
        if cache.dtype != inputs.dtype or cache.device != inputs.device:
            raise ValueError("convolution cache must match the input dtype and device")
        if not bool(torch.isfinite(cache).all()):
            raise ValueError("convolution cache must be finite")

        if valid_mask is None:
            combined = torch.cat((cache, channels), dim=-1)
            output = self.conv(combined).transpose(1, 2)
            next_cache = (
                combined[..., -self.cache_width :].clone()
                if self.cache_width
                else combined[..., :0].clone()
            )
            return output, next_cache
        if not isinstance(valid_mask, torch.Tensor):
            raise TypeError("valid_mask must be a tensor or None")
        if valid_mask.shape != inputs.shape[:2]:
            raise ValueError("valid_mask must have shape (batch, length)")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must have dtype torch.bool")
        if valid_mask.device != inputs.device:
            raise ValueError("valid_mask must be on the input device")

        outputs = []
        for position in range(inputs.shape[1]):
            window = torch.cat((cache, channels[..., position : position + 1]), dim=-1)
            outputs.append(self.conv(window))
            candidate = window[..., 1:] if self.cache_width else window[..., :0]
            cache = torch.where(valid_mask[:, position, None, None], candidate, cache)
        return torch.cat(outputs, dim=-1).transpose(1, 2), cache.clone()


class HybridMemoryBlock(nn.Module):
    """Uniform pre-norm, gated residual, and pre-norm SwiGLU layer shell."""

    def __init__(self, config: HybridMemoryConfig, kind: LayerKind) -> None:
        super().__init__()
        if kind not in _LAYER_KINDS:
            raise ValueError(f"kind must be one of {_LAYER_KINDS}")
        self.config = config
        self.kind = kind
        self.mixer_norm = nn.RMSNorm(config.model_dim, eps=config.norm_epsilon)
        self.input_projection = nn.Linear(
            config.model_dim, 2 * config.model_dim, bias=False
        )
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))
        self.ffn_norm = nn.RMSNorm(config.model_dim, eps=config.norm_epsilon)
        self.ffn = GatedMLP(config.model_dim, config.model_dim * config.expansion)
        self.dropout = nn.Dropout(config.dropout)

        if kind == "attention":
            self.mixer: nn.Module = CausalSelfAttention(
                AttentionConfig(
                    model_dim=config.model_dim,
                    heads=config.attention_heads,
                    window_size=config.attention_window_size,
                    rope_base=config.attention_rope_base,
                    dropout=config.dropout,
                )
            )
            self.local_conv = None
        elif kind == "gated_delta":
            self.mixer = GatedDeltaMemory(
                GatedDeltaConfig(
                    model_dim=config.model_dim,
                    heads=config.gated_delta_heads,
                    key_dim=config.gated_delta_key_dim,
                    value_dim=config.gated_delta_value_dim,
                    allow_negative_eigenvalues=(
                        config.gated_delta_allow_negative_eigenvalues
                    ),
                    norm_epsilon=config.norm_epsilon,
                    minimum_retention=config.gated_delta_minimum_retention,
                    initial_retention=config.gated_delta_initial_retention,
                    initial_write_strength=config.gated_delta_initial_write_strength,
                )
            )
            self.local_conv = (
                CausalDepthwiseConv1d(config.model_dim, config.conv_kernel)
                if config.use_local_conv
                else None
            )
        elif kind == "delta_product":
            self.mixer = DeltaProductReferenceLayer(
                hidden_size=config.model_dim,
                num_heads=config.delta_heads,
                num_householder=config.delta_num_householder,
                allow_negative_eigenvalues=config.delta_allow_negative_eigenvalues,
                norm_epsilon=config.norm_epsilon,
            )
            self.local_conv = (
                CausalDepthwiseConv1d(config.model_dim, config.conv_kernel)
                if config.use_local_conv
                else None
            )
        elif kind == "selected_block":
            self.mixer = SelectedBlockMemory(
                SelectedBlockConfig(
                    model_dim=config.model_dim,
                    heads=config.selected_heads,
                    blocks=config.selected_blocks,
                    slots_per_block=config.selected_slots_per_block,
                    value_dim=config.selected_value_dim,
                    update_rank=config.selected_update_rank,
                    controller_rank=config.selected_controller_rank,
                    retention_min=config.selected_retention_min,
                    retention_max=config.selected_retention_max,
                )
            )
            self.local_conv = (
                CausalDepthwiseConv1d(config.model_dim, config.conv_kernel)
                if config.use_local_conv
                else None
            )
        else:
            self.mixer = StructuredSpin8Memory(
                StructuredMemoryConfig(
                    model_dim=config.model_dim,
                    channels=config.structured_channels,
                    rungs=config.structured_rungs,
                    controller_rank=config.structured_controller_rank,
                    retention_min=config.structured_retention_min,
                    retention_max=config.structured_retention_max,
                    hard_eval=config.structured_hard_eval,
                )
            )
            self.local_conv = (
                CausalDepthwiseConv1d(config.model_dim, config.conv_kernel)
                if config.use_local_conv
                else None
            )

    @property
    def is_attention(self) -> bool:
        return self.kind == "attention"

    @property
    def convolution_cache_width(self) -> int:
        return 0 if self.local_conv is None else self.local_conv.cache_width

    def _empty_convolution(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.new_zeros(
            hidden.shape[0], self.config.model_dim, self.convolution_cache_width
        )

    @staticmethod
    def _validate_tensor_state(
        tensor: object,
        *,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype")
        if tensor.dtype != dtype or tensor.device != device:
            raise ValueError(f"{name} must match the model dtype and device")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite")

    def validate_state(
        self,
        state: LayerState,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if self.kind == "attention":
            if not isinstance(state, AttentionState):
                raise TypeError("attention layer state must be an AttentionState")
            if not isinstance(state.key_cache, torch.Tensor):
                raise TypeError("attention key cache must be a tensor")
            if state.key_cache.shape[0] != batch_size:
                raise ValueError("attention state batch size is incompatible")
            if state.key_cache.dtype != dtype or state.key_cache.device != device:
                raise ValueError(
                    "attention state must match the model dtype and device"
                )
            assert isinstance(self.mixer, CausalSelfAttention)
            self.mixer.state_actual_bytes(state)
            return

        convolution_shape = (
            batch_size,
            self.config.model_dim,
            self.convolution_cache_width,
        )
        if self.kind == "gated_delta":
            if not isinstance(state, GatedDeltaState):
                raise TypeError("gated_delta layer state must be a GatedDeltaState")
            assert isinstance(self.mixer, GatedDeltaMemory)
            memory_shape = (batch_size, *self.mixer.config.state_shape)
        elif self.kind == "delta_product":
            if not isinstance(state, DeltaProductState):
                raise TypeError("delta_product layer state must be a DeltaProductState")
            assert isinstance(self.mixer, DeltaProductReferenceLayer)
            memory_shape = (
                batch_size,
                self.mixer.num_heads,
                self.mixer.head_dim,
                self.mixer.head_dim,
            )
        elif self.kind == "selected_block":
            if not isinstance(state, SelectedBlockState):
                raise TypeError(
                    "selected_block layer state must be a SelectedBlockState"
                )
            assert isinstance(self.mixer, SelectedBlockMemory)
            memory_shape = (batch_size, *self.mixer.config.state_shape)
        else:
            if not isinstance(state, StructuredSpin8State):
                raise TypeError(
                    "structured_spin8 layer state must be a StructuredSpin8State"
                )
            assert isinstance(self.mixer, StructuredSpin8Memory)
            memory_shape = (batch_size, *self.mixer.config.state_shape)
        self._validate_tensor_state(
            state.memory,
            name=f"{self.kind} memory",
            shape=memory_shape,
            dtype=dtype,
            device=device,
        )
        self._validate_tensor_state(
            state.convolution,
            name=f"{self.kind} convolution cache",
            shape=convolution_shape,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        state: LayerState | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        delta_scan_mode: DeltaScanMode = "parallel",
        selected_scan_mode: SelectedScanMode = "physical_gather",
        selected_route_mode: RouteMode = "hard",
        structured_scan_mode: StructuredScanMode = "parallel",
        return_diagnostics: bool = False,
    ) -> (
        tuple[torch.Tensor, LayerState]
        | tuple[torch.Tensor, LayerState, LayerDiagnostics]
    ):
        if type(return_diagnostics) is not bool:
            raise TypeError("return_diagnostics must be a bool")
        if hidden.ndim != 3 or hidden.shape[-1] != self.config.model_dim:
            raise ValueError("hidden must have shape (batch, length, model_dim)")
        if state is not None:
            self.validate_state(
                state,
                batch_size=hidden.shape[0],
                dtype=hidden.dtype,
                device=hidden.device,
            )

        value, gate = self.input_projection(self.mixer_norm(hidden)).chunk(2, dim=-1)
        if valid_mask is not None and self.kind != "attention":
            value = value * valid_mask[..., None].to(value.dtype)
        diagnostics: LayerDiagnostics = None

        if self.kind == "attention":
            if valid_mask is not None:
                raise NotImplementedError(
                    "valid_mask is not supported by attention layers"
                )
            assert isinstance(self.mixer, CausalSelfAttention)
            attention_state = state if isinstance(state, AttentionState) else None
            update, next_state = self.mixer(
                value, attention_state, use_cache=True, valid_mask=None
            )
            if return_diagnostics:
                diagnostics = {
                    "kind": "attention",
                    "position": next_state.position,
                    "cache_length": next_state.cache_length,
                }
        else:
            previous_convolution = (
                None if state is None else state.convolution  # type: ignore[union-attr]
            )
            if self.local_conv is None:
                mixed_value = value
                next_convolution = self._empty_convolution(hidden)
            else:
                mixed_value, next_convolution = self.local_conv(
                    value, previous_convolution, valid_mask
                )
            mixed_value = F.silu(mixed_value)
            if self.kind == "gated_delta":
                assert isinstance(self.mixer, GatedDeltaMemory)
                memory = state.memory if isinstance(state, GatedDeltaState) else None
                if return_diagnostics:
                    update, next_memory, diagnostics = self.mixer(
                        mixed_value,
                        memory,
                        valid_mask=valid_mask,
                        scan_mode=delta_scan_mode,
                        return_diagnostics=True,
                    )
                else:
                    update, next_memory = self.mixer(
                        mixed_value,
                        memory,
                        valid_mask=valid_mask,
                        scan_mode=delta_scan_mode,
                    )
                next_state = GatedDeltaState(next_memory, next_convolution)
            elif self.kind == "delta_product":
                assert isinstance(self.mixer, DeltaProductReferenceLayer)
                memory = state.memory if isinstance(state, DeltaProductState) else None
                update, next_memory = self.mixer(
                    mixed_value,
                    memory,
                    valid_mask=valid_mask,
                    scan_mode=delta_scan_mode,
                )
                next_state = DeltaProductState(next_memory, next_convolution)
                if return_diagnostics:
                    diagnostics = {
                        "kind": "delta_product",
                        "scan_mode": delta_scan_mode,
                    }
            elif self.kind == "selected_block":
                assert isinstance(self.mixer, SelectedBlockMemory)
                memory = state.memory if isinstance(state, SelectedBlockState) else None
                if return_diagnostics:
                    update, next_memory, diagnostics = self.mixer(
                        mixed_value,
                        memory,
                        valid_mask=valid_mask,
                        scan_mode=selected_scan_mode,
                        route_mode=selected_route_mode,
                        return_diagnostics=True,
                    )
                else:
                    update, next_memory = self.mixer(
                        mixed_value,
                        memory,
                        valid_mask=valid_mask,
                        scan_mode=selected_scan_mode,
                        route_mode=selected_route_mode,
                    )
                next_state = SelectedBlockState(next_memory, next_convolution)
            else:
                assert isinstance(self.mixer, StructuredSpin8Memory)
                memory = (
                    state.memory if isinstance(state, StructuredSpin8State) else None
                )
                if return_diagnostics:
                    update, next_memory, diagnostics = self.mixer(
                        mixed_value,
                        memory,
                        valid_mask=valid_mask,
                        scan_mode=structured_scan_mode,
                        return_diagnostics=True,
                    )
                else:
                    update, next_memory = self.mixer(
                        mixed_value,
                        memory,
                        valid_mask=valid_mask,
                        scan_mode=structured_scan_mode,
                    )
                next_state = StructuredSpin8State(next_memory, next_convolution)

        hidden = hidden + torch.sigmoid(self.residual_scale) * self.dropout(
            update * torch.sigmoid(gate)
        )
        hidden = hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))
        if return_diagnostics:
            return hidden, next_state, diagnostics
        return hidden, next_state


class HybridMemoryLM(nn.Module):
    """Language model supporting full sequences, arbitrary chunks, and token steps."""

    model_version = __version__

    def __init__(self, config: HybridMemoryConfig) -> None:
        super().__init__()
        if not isinstance(config, HybridMemoryConfig):
            raise TypeError("config must be a HybridMemoryConfig")
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.blocks = nn.ModuleList(
            HybridMemoryBlock(config, kind) for kind in config.layer_plan
        )
        self.final_norm = nn.RMSNorm(config.model_dim, eps=config.norm_epsilon)
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.apply(_initialize_module)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    @property
    def layer_plan(self) -> tuple[LayerKind, ...]:
        return self.config.layer_plan

    def _validate_tokens_and_mask(
        self, token_ids: torch.Tensor, valid_mask: torch.Tensor | None
    ) -> None:
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError("token_ids must be a tensor")
        if token_ids.ndim != 2 or token_ids.shape[0] < 1 or token_ids.shape[1] < 1:
            raise ValueError("token_ids must have nonempty shape (batch, length)")
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must have dtype torch.long")
        if token_ids.device != self.embedding.weight.device:
            raise ValueError("token_ids must be on the model device")
        if valid_mask is not None:
            if not isinstance(valid_mask, torch.Tensor):
                raise TypeError("valid_mask must be a tensor or None")
            if valid_mask.shape != token_ids.shape:
                raise ValueError("valid_mask must have shape (batch, length)")
            if valid_mask.dtype != torch.bool:
                raise TypeError("valid_mask must have dtype torch.bool")
            if valid_mask.device != token_ids.device:
                raise ValueError("valid_mask must be on the token device")
            if "attention" in self.layer_plan:
                raise NotImplementedError(
                    "valid_mask is not supported when layer_plan contains attention"
                )

    def _validate_states(
        self,
        states: Sequence[LayerState] | None,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[LayerState | None, ...]:
        if states is None:
            return (None,) * len(self.blocks)
        if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
            raise TypeError(
                "states must be a sequence of complete layer states or None"
            )
        if len(states) != len(self.blocks):
            raise ValueError("states must contain one complete state per layer")
        checked: list[LayerState] = []
        for block, state in zip(self.blocks, states, strict=True):
            if state is None:
                raise TypeError("states cannot omit an individual layer cache")
            block.validate_state(
                state,
                batch_size=batch_size,
                dtype=dtype,
                device=device,
            )
            checked.append(state)
        return tuple(checked)

    def forward(
        self,
        token_ids: torch.Tensor,
        states: Sequence[LayerState] | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        delta_scan_mode: DeltaScanMode = "parallel",
        selected_scan_mode: SelectedScanMode = "physical_gather",
        selected_route_mode: RouteMode = "hard",
        structured_scan_mode: StructuredScanMode = "parallel",
        return_diagnostics: bool = False,
    ) -> dict[str, Any]:
        if type(return_diagnostics) is not bool:
            raise TypeError("return_diagnostics must be a bool")
        self._validate_tokens_and_mask(token_ids, valid_mask)
        if delta_scan_mode not in _DELTA_SCAN_MODES:
            raise ValueError(f"delta_scan_mode must be one of {_DELTA_SCAN_MODES}")
        if selected_scan_mode not in _SELECTED_SCAN_MODES:
            raise ValueError(
                f"selected_scan_mode must be one of {_SELECTED_SCAN_MODES}"
            )
        if selected_route_mode not in _SELECTED_ROUTE_MODES:
            raise ValueError(
                f"selected_route_mode must be one of {_SELECTED_ROUTE_MODES}"
            )
        if selected_scan_mode == "physical_gather" and selected_route_mode != "hard":
            raise ValueError("physical_gather requires selected_route_mode='hard'")
        if structured_scan_mode not in _STRUCTURED_SCAN_MODES:
            raise ValueError(
                f"structured_scan_mode must be one of {_STRUCTURED_SCAN_MODES}"
            )
        hidden = self.embedding(token_ids)
        layer_states = self._validate_states(
            states,
            batch_size=token_ids.shape[0],
            dtype=hidden.dtype,
            device=hidden.device,
        )
        next_states: list[LayerState] = []
        diagnostics: list[LayerDiagnostics] = []
        for block, state in zip(self.blocks, layer_states, strict=True):
            if return_diagnostics:
                hidden, next_state, layer_diagnostics = block(
                    hidden,
                    state,
                    valid_mask=valid_mask,
                    delta_scan_mode=delta_scan_mode,
                    selected_scan_mode=selected_scan_mode,
                    selected_route_mode=selected_route_mode,
                    structured_scan_mode=structured_scan_mode,
                    return_diagnostics=True,
                )
                diagnostics.append(layer_diagnostics)
            else:
                hidden, next_state = block(
                    hidden,
                    state,
                    valid_mask=valid_mask,
                    delta_scan_mode=delta_scan_mode,
                    selected_scan_mode=selected_scan_mode,
                    selected_route_mode=selected_route_mode,
                    structured_scan_mode=structured_scan_mode,
                )
            next_states.append(next_state)
        logits = self.lm_head(self.final_norm(hidden))
        output = {"logits": logits, "states": tuple(next_states)}
        if return_diagnostics:
            output["diagnostics"] = tuple(diagnostics)
        return output

    def step(
        self,
        token_ids: torch.Tensor,
        states: Sequence[LayerState] | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[LayerState, ...]]:
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError("token_ids must be a tensor")
        if token_ids.ndim != 1 or token_ids.shape[0] < 1:
            raise ValueError("step token_ids must have nonempty shape (batch,)")
        sequence_mask = None
        if valid_mask is not None:
            if not isinstance(valid_mask, torch.Tensor):
                raise TypeError("valid_mask must be a tensor or None")
            if valid_mask.ndim != 1 or valid_mask.shape != token_ids.shape:
                raise ValueError("step valid_mask must have shape (batch,)")
            sequence_mask = valid_mask[:, None]
        output = self(
            token_ids[:, None],
            states,
            valid_mask=sequence_mask,
            delta_scan_mode="recurrent",
            selected_scan_mode="physical_gather",
            structured_scan_mode="recurrent",
        )
        return output["logits"][:, 0], output["states"]

    def _state_batch_dtype(
        self, states: Sequence[LayerState]
    ) -> tuple[int, torch.dtype]:
        first = states[0]
        if isinstance(first, AttentionState):
            tensor = first.key_cache
        elif isinstance(
            first,
            (
                GatedDeltaState,
                DeltaProductState,
                SelectedBlockState,
                StructuredSpin8State,
            ),
        ):
            tensor = first.memory
        else:
            raise TypeError("states contain an unknown layer-state type")
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1:
            raise TypeError("state payloads must be batched tensors")
        return tensor.shape[0], tensor.dtype

    def state_byte_report(
        self,
        states: Sequence[LayerState] | None = None,
        *,
        batch_size: int | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, Any]:
        """Return exact logical payload bytes now and at configured capacity."""

        if states is None:
            if batch_size is None:
                raise ValueError("batch_size is required when states is None")
            _positive_integer("batch_size", batch_size)
            state_sequence: tuple[LayerState, ...] | None = None
        else:
            if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
                raise TypeError("states must be a sequence of complete layer states")
            if len(states) != len(self.blocks):
                raise ValueError("states must contain one complete state per layer")
            inferred_batch, inferred_dtype = self._state_batch_dtype(states)
            if batch_size is not None and batch_size != inferred_batch:
                raise ValueError("batch_size does not match states")
            if dtype is not None and dtype != inferred_dtype:
                raise ValueError("dtype does not match states")
            batch_size = inferred_batch
            dtype = inferred_dtype
            model_device = self.embedding.weight.device
            state_sequence = tuple(states)
            for block, state in zip(self.blocks, state_sequence, strict=True):
                block.validate_state(
                    state,
                    batch_size=batch_size,
                    dtype=dtype,
                    device=model_device,
                )
        assert batch_size is not None
        if dtype is None:
            dtype = self.embedding.weight.dtype
        element_size = _dtype_bytes(dtype)

        layers: list[dict[str, Any]] = []
        total_actual = 0
        total_capacity = 0
        for index, block in enumerate(self.blocks):
            state = None if state_sequence is None else state_sequence[index]
            if block.kind == "attention":
                assert isinstance(block.mixer, CausalSelfAttention)
                capacity_key = (
                    batch_size
                    * block.mixer.config.heads
                    * block.mixer.max_cache_length
                    * block.mixer.head_dim
                    * element_size
                )
                capacity_components = {
                    "key_cache": capacity_key,
                    "value_cache": capacity_key,
                    "position": _POSITION_BYTES,
                }
                if state is None:
                    actual_components = {
                        "key_cache": 0,
                        "value_cache": 0,
                        "position": 0,
                    }
                else:
                    assert isinstance(state, AttentionState)
                    actual_components = {
                        "key_cache": _tensor_bytes(state.key_cache),
                        "value_cache": _tensor_bytes(state.value_cache),
                        "position": _POSITION_BYTES,
                    }
            else:
                conv_capacity = (
                    batch_size
                    * self.config.model_dim
                    * block.convolution_cache_width
                    * element_size
                )
                if block.kind == "gated_delta":
                    assert isinstance(block.mixer, GatedDeltaMemory)
                    memory_capacity = (
                        batch_size * block.mixer.state_scalars * element_size
                    )
                elif block.kind == "delta_product":
                    assert isinstance(block.mixer, DeltaProductReferenceLayer)
                    memory_capacity = (
                        batch_size
                        * block.mixer.num_heads
                        * block.mixer.head_dim
                        * block.mixer.head_dim
                        * element_size
                    )
                elif block.kind == "selected_block":
                    assert isinstance(block.mixer, SelectedBlockMemory)
                    memory_capacity = (
                        batch_size * block.mixer.config.state_scalars * element_size
                    )
                else:
                    assert isinstance(block.mixer, StructuredSpin8Memory)
                    memory_capacity = (
                        batch_size * block.mixer.config.state_scalars * element_size
                    )
                capacity_components = {
                    "memory": memory_capacity,
                    "convolution": conv_capacity,
                }
                if state is None:
                    actual_components = {"memory": 0, "convolution": 0}
                else:
                    assert isinstance(
                        state,
                        (
                            GatedDeltaState,
                            DeltaProductState,
                            SelectedBlockState,
                            StructuredSpin8State,
                        ),
                    )
                    actual_components = {
                        "memory": _tensor_bytes(state.memory),
                        "convolution": _tensor_bytes(state.convolution),
                    }
            actual_bytes = sum(actual_components.values())
            capacity_bytes = sum(capacity_components.values())
            total_actual += actual_bytes
            total_capacity += capacity_bytes
            layers.append(
                {
                    "index": index,
                    "kind": block.kind,
                    "actual_bytes": actual_bytes,
                    "capacity_bytes": capacity_bytes,
                    "actual_components": actual_components,
                    "capacity_components": capacity_components,
                }
            )
        return {
            "batch_size": batch_size,
            "dtype": dtype,
            "layers": tuple(layers),
            "actual_bytes": total_actual,
            "capacity_bytes": total_capacity,
            "total_actual_bytes": total_actual,
            "total_capacity_bytes": total_capacity,
            "totals": {
                "actual_bytes": total_actual,
                "capacity_bytes": total_capacity,
            },
        }

    def state_actual_bytes(self, states: Sequence[LayerState]) -> int:
        return int(self.state_byte_report(states)["actual_bytes"])

    def state_capacity_bytes(
        self, batch_size: int, dtype: torch.dtype | None = None
    ) -> int:
        return int(
            self.state_byte_report(batch_size=batch_size, dtype=dtype)["capacity_bytes"]
        )

    def save_checkpoint(
        self, path: str | Path, metadata: dict[str, Any] | None = None
    ) -> None:
        torch.save(
            {
                "format_version": 1,
                "model_type": "hybrid_memory_v1_4",
                "model_version": __version__,
                "config": asdict(self.config),
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in self.state_dict().items()
                },
                "metadata": {} if metadata is None else metadata,
            },
            Path(path),
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device | None = None,
    ) -> tuple[HybridMemoryLM, dict[str, Any]]:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError("checkpoint payload must be a dictionary")
        if payload.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format_version")
        if payload.get("model_type") != "hybrid_memory_v1_4":
            raise ValueError("checkpoint model_type is not hybrid_memory_v1_4")
        config_payload = payload.get("config")
        state_dict = payload.get("state_dict")
        metadata = payload.get("metadata")
        if not isinstance(config_payload, dict) or not isinstance(state_dict, dict):
            raise TypeError("checkpoint config and state_dict must be dictionaries")
        if not isinstance(metadata, dict):
            raise TypeError("checkpoint metadata must be a dictionary")
        model = cls(HybridMemoryConfig(**config_payload))
        model.load_state_dict(state_dict, strict=True)
        return model, metadata


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _initialize_module(module: nn.Module) -> None:
    if isinstance(module, (nn.Embedding, nn.Linear)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)


_SUBGROUP_PAIRS = tuple(combinations(range(8), 2))


def subgroup_generator_indices(dimension: int) -> tuple[int, ...]:
    """Generator-subset indices of the Spin(3)->Spin(8) nested ladder."""

    if dimension not in (3, 4, 6, 8):
        raise ValueError("ladder dimensions are exactly 3, 4, 6, 8")
    return tuple(
        index
        for index, (left, right) in enumerate(_SUBGROUP_PAIRS)
        if left < dimension and right < dimension
    )


DeltaProductLayerState = DeltaProductState
GatedDeltaLayerState = GatedDeltaState
SelectedBlockLayerState = SelectedBlockState
StructuredSpin8LayerState = StructuredSpin8State
SwiGLU = GatedMLP


__all__ = [
    "AttentionState",
    "CausalDepthwiseConv1d",
    "DeltaProductLayerState",
    "DeltaProductState",
    "GatedDeltaLayerState",
    "GatedDeltaState",
    "HybridMemoryBlock",
    "HybridMemoryConfig",
    "HybridMemoryLM",
    "LayerState",
    "SelectedBlockLayerState",
    "SelectedBlockState",
    "StructuredSpin8LayerState",
    "StructuredSpin8State",
    "SwiGLU",
    "parameter_count",
    "subgroup_generator_indices",
]
