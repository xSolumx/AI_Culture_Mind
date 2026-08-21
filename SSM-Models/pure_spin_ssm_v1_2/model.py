"""Pure Spin SSM v1.2: triality recurrence with local and channel mixing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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


class PureSpinV12Block(nn.Module):
    """Mamba-shaped local mixer around a triality-faithful Spin(8) cache."""

    def __init__(self, config: PureSpinV12Config) -> None:
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
        self.state_norm = nn.RMSNorm(self.spin.output_size)
        self.output_projection = nn.Linear(self.spin.output_size, config.d_model, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))
        self.ffn_norm = nn.RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.expansion)
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
        self.blocks = nn.ModuleList(PureSpinV12Block(config) for _ in range(config.num_layers))
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


__all__ = ["PureSpinSSMV12", "PureSpinV12Config", "parameter_count"]
