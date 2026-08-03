"""PyTorch layers for multivectors in three-dimensional Euclidean GA.

The original prototype mixed raw tensors and Kingdon ``MultiVector`` objects,
used incompatible ``.values`` APIs, and ignored the requested number of
attention heads.  This module keeps one representation throughout: a tensor
whose final axis is ``[1, e1, e2, e3, e12, e13, e23, e123]``.
"""

from __future__ import annotations

import math

import torch
from torch import nn

GA_DIM = 8
BASIS_MASKS = (0, 1, 2, 4, 3, 5, 6, 7)


def _blade_product_sign(left: int, right: int) -> int:
    swaps = sum((right & ((1 << bit) - 1)).bit_count() for bit in range(3) if left & (1 << bit))
    return -1 if swaps % 2 else 1


def _multiplication_table() -> torch.Tensor:
    table = torch.zeros(GA_DIM, GA_DIM, GA_DIM)
    mask_to_index = {mask: index for index, mask in enumerate(BASIS_MASKS)}
    for left_index, left_mask in enumerate(BASIS_MASKS):
        for right_index, right_mask in enumerate(BASIS_MASKS):
            output_index = mask_to_index[left_mask ^ right_mask]
            table[output_index, left_index, right_index] = _blade_product_sign(
                left_mask, right_mask
            )
    return table


MULTIPLICATION_TABLE = _multiplication_table()
REVERSION_SIGNS = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1])


def geometric_product_ga3(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compute a broadcasted GA(3, 0) geometric product."""
    if left.shape[-1] != GA_DIM or right.shape[-1] != GA_DIM:
        raise ValueError(f"multivectors must have a final dimension of {GA_DIM}")
    table = MULTIPLICATION_TABLE.to(device=left.device, dtype=torch.result_type(left, right))
    return torch.einsum("...i,...j,kij->...k", left, right, table)


def reversion(multivector: torch.Tensor) -> torch.Tensor:
    """Reverse a GA(3, 0) multivector."""
    if multivector.shape[-1] != GA_DIM:
        raise ValueError(f"multivectors must have a final dimension of {GA_DIM}")
    signs = REVERSION_SIGNS.to(device=multivector.device, dtype=multivector.dtype)
    return multivector * signs


class SpinorLinearLayer(nn.Module):
    """Apply one learned multivector operator and bias."""

    def __init__(self, ga_dim_input: int = GA_DIM, ga_dim_output: int = GA_DIM):
        super().__init__()
        if ga_dim_input != GA_DIM or ga_dim_output != GA_DIM:
            raise ValueError(f"SpinorLinearLayer currently operates on {GA_DIM} blades")
        initial_operator = torch.randn(GA_DIM) * 0.02
        initial_operator[0] += 1.0
        self.weight_components = nn.Parameter(initial_operator)
        self.bias_components = nn.Parameter(torch.zeros(GA_DIM))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return geometric_product_ga3(self.weight_components, inputs) + self.bias_components


class SpinorActivation(nn.Module):
    """Component-wise GELU for the exploratory multivector network."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(inputs)


class SpinorAttentionHead(nn.Module):
    def __init__(self, ga_dim_in: int = GA_DIM):
        super().__init__()
        self.query_proj = SpinorLinearLayer(ga_dim_in, ga_dim_in)
        self.key_proj = SpinorLinearLayer(ga_dim_in, ga_dim_in)
        self.value_proj = SpinorLinearLayer(ga_dim_in, ga_dim_in)

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        queries = self.query_proj(inputs)
        keys = self.key_proj(inputs)
        values = self.value_proj(inputs)
        scores_mv = geometric_product_ga3(
            queries.unsqueeze(-2), reversion(keys).unsqueeze(-3)
        )
        scores = scores_mv[..., 0] / math.sqrt(GA_DIM)

        sequence_length = inputs.shape[-2]
        allowed = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=inputs.device,
            )
        ).unsqueeze(0)
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                allowed = allowed & attention_mask[:, None, :].bool()
            elif attention_mask.ndim == 3:
                allowed = allowed & attention_mask.bool()
            else:
                raise ValueError("attention_mask must have shape (B, S) or (B, S, S)")
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bqk,bkd->bqd", weights, values)


class SpinorTransformerBlock(nn.Module):
    def __init__(
        self,
        ga_dim: int = GA_DIM,
        num_heads: int = 1,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be positive")
        self.attention_heads = nn.ModuleList(
            SpinorAttentionHead(ga_dim) for _ in range(num_heads)
        )
        self.attention_output = SpinorLinearLayer(ga_dim, ga_dim)
        self.feed_forward = nn.Sequential(
            SpinorLinearLayer(ga_dim, ga_dim),
            SpinorActivation(),
            nn.Dropout(dropout_rate),
            SpinorLinearLayer(ga_dim, ga_dim),
        )
        self.norm1 = nn.LayerNorm(ga_dim)
        self.norm2 = nn.LayerNorm(ga_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        head_outputs = torch.stack(
            [head(inputs, attention_mask) for head in self.attention_heads], dim=0
        ).mean(dim=0)
        outputs = self.norm1(inputs + self.dropout(self.attention_output(head_outputs)))
        return self.norm2(outputs + self.dropout(self.feed_forward(outputs)))


__all__ = [
    "BASIS_MASKS",
    "GA_DIM",
    "MULTIPLICATION_TABLE",
    "SpinorActivation",
    "SpinorAttentionHead",
    "SpinorLinearLayer",
    "SpinorTransformerBlock",
    "geometric_product_ga3",
    "reversion",
]
