"""Strict adapter for the official fused mamba_ssm Mamba-2 implementation."""

from __future__ import annotations

import torch
from torch import nn


class OfficialMamba2LM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 32,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba2
        except ImportError as error:
            raise RuntimeError(
                "official mamba_ssm is required; no unfused fallback is permitted"
            ) from error
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            Mamba2(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                headdim=headdim,
                use_mem_eff_path=True,
                layer_idx=index,
            )
            for index in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.RMSNorm(d_model) for _ in range(num_layers))
        self.final_norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.apply(_initialize_language_model_module)

    def forward(
        self,
        token_ids: torch.Tensor,
        *,
        inference_params=None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.embedding(token_ids)
        for norm, layer in zip(self.norms, self.layers, strict=True):
            hidden = hidden + layer(
                norm(hidden), inference_params=inference_params
            )
        return {"logits": self.lm_head(self.final_norm(hidden))}


def fused_mamba2_available() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is unavailable"
    try:
        import mamba_ssm
        from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
    except (ImportError, RuntimeError, OSError) as error:
        return False, repr(error)
    return bool(mamba_split_conv1d_scan_combined), getattr(mamba_ssm, "__version__", "unknown")


def _initialize_language_model_module(module: nn.Module) -> None:
    if isinstance(module, (nn.Embedding, nn.Linear)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
