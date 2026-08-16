"""Unfused PyTorch reference for the DeltaProduct state-tracking baseline.

The official NeurIPS 2025 implementation delegates training to
Flash-Linear-Attention's ``GatedDeltaProductForCausalLM``.  That Triton stack is
not available in this Windows checkout.  This module preserves the relevant
state-tracking equations used by the official configuration:

* no short convolution, output gate, or forget gate;
* multiple generalized Householder/delta updates per token;
* negative eigenvalues enabled through ``beta in (0, 2)``;
* a matrix fast-weight state and conventional residual/MLP block.

It is an architecture-quality reference, not an official fused-kernel timing
result.  Source comparison is pinned in the experiment protocol.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def delta_update_affine(
    key: torch.Tensor, value: torch.Tensor, beta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``S -> A S + B`` for one normalized-key delta update."""

    if key.shape != value.shape or key.shape[-1] < 1:
        raise ValueError("key and value must have the same nonempty final dimension")
    if beta.shape != key.shape[:-1]:
        raise ValueError("beta must match key without its final dimension")
    identity = torch.eye(key.shape[-1], dtype=key.dtype, device=key.device)
    identity = identity.expand(*key.shape[:-1], key.shape[-1], key.shape[-1])
    outer_key = key.unsqueeze(-1) * key.unsqueeze(-2)
    drive = key.unsqueeze(-1) * value.unsqueeze(-2)
    return (
        identity - beta[..., None, None] * outer_key,
        beta[..., None, None] * drive,
    )


def compose_delta_transitions(
    later: tuple[torch.Tensor, torch.Tensor],
    earlier: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose two matrix-affine transitions in chronological order."""

    later_a, later_b = later
    earlier_a, earlier_b = earlier
    return (
        later_a @ earlier_a,
        later_b + later_a @ earlier_b,
    )


def compose_delta_updates_per_token(
    keys: torch.Tensor, values: torch.Tensor, betas: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose ``R`` generalized Householder updates for every token.

    Shapes are ``(batch,length,updates,heads,head_dim)`` for keys/values and
    ``(batch,length,updates,heads)`` for betas.
    """

    if keys.ndim != 5 or values.shape != keys.shape:
        raise ValueError(
            "keys and values must have shape (batch,length,updates,heads,head_dim)"
        )
    if betas.shape != keys.shape[:-1]:
        raise ValueError("betas must match keys without head_dim")
    transition = delta_update_affine(keys[:, :, 0], values[:, :, 0], betas[:, :, 0])
    for update in range(1, keys.shape[2]):
        current = delta_update_affine(
            keys[:, :, update], values[:, :, update], betas[:, :, update]
        )
        transition = compose_delta_transitions(current, transition)
    return transition


def delta_product_scan(
    transition_a: torch.Tensor,
    transition_b: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    mode: str = "parallel",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan matrix-affine DeltaProduct transitions.

    Transition/state shapes are ``(batch,length,heads,head_dim,head_dim)`` and
    ``(batch,heads,head_dim,head_dim)``.  The Hillis--Steele path is a transparent
    associative reference; the official deployment path uses FLA chunk kernels.
    """

    if (
        transition_a.ndim != 5
        or transition_a.shape != transition_b.shape
        or transition_a.shape[1] == 0
        or transition_a.shape[-1] != transition_a.shape[-2]
    ):
        raise ValueError("transitions must have nonempty matching matrix shapes")
    batch, length, heads, head_dim, _ = transition_a.shape
    if initial_state is None:
        initial_state = torch.zeros_like(transition_b[:, 0])
    elif initial_state.shape != (batch, heads, head_dim, head_dim):
        raise ValueError("initial_state has the wrong matrix-state shape")

    if mode == "recurrent":
        state = initial_state
        states = []
        for position in range(length):
            state = transition_a[:, position] @ state + transition_b[:, position]
            states.append(state)
        sequence = torch.stack(states, dim=1)
        return sequence, state
    if mode != "parallel":
        raise ValueError("mode must be 'parallel' or 'recurrent'")

    cumulative_a, cumulative_b = transition_a, transition_b
    offset = 1
    while offset < length:
        composed_a, composed_b = compose_delta_transitions(
            (cumulative_a[:, offset:], cumulative_b[:, offset:]),
            (cumulative_a[:, :-offset], cumulative_b[:, :-offset]),
        )
        cumulative_a = torch.cat((cumulative_a[:, :offset], composed_a), dim=1)
        cumulative_b = torch.cat((cumulative_b[:, :offset], composed_b), dim=1)
        offset *= 2
    sequence = cumulative_a @ initial_state[:, None] + cumulative_b
    return sequence, sequence[:, -1]


class DeltaProductReferenceLayer(nn.Module):
    """One official-style multi-update DeltaProduct layer without fused kernels."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_householder: int,
        allow_negative_eigenvalues: bool = True,
        norm_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if (
            hidden_size < 1
            or num_heads < 1
            or hidden_size % num_heads
            or num_householder < 1
        ):
            raise ValueError("hidden size, heads, and update count are invalid")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_householder = num_householder
        self.allow_negative_eigenvalues = allow_negative_eigenvalues
        self.norm_epsilon = norm_epsilon

        self.query_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_projections = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(num_householder)
        )
        self.value_projections = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(num_householder)
        )
        self.beta_projections = nn.ModuleList(
            nn.Linear(hidden_size, num_heads, bias=False)
            for _ in range(num_householder)
        )
        self.output_norm = nn.RMSNorm(self.head_dim, eps=norm_epsilon)
        self.output_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        gain = 2**-2.5
        for module in (
            self.query_projection,
            *self.key_projections,
            *self.value_projections,
            *self.beta_projections,
            self.output_projection,
        ):
            nn.init.xavier_uniform_(module.weight, gain=gain)

    def transitions(
        self, hidden_states: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states must have shape (batch,length,hidden_size)")
        if valid_mask is not None and valid_mask.shape != hidden_states.shape[:2]:
            raise ValueError("valid_mask must have shape (batch,length)")
        batch, length, _ = hidden_states.shape
        query = F.silu(self.query_projection(hidden_states)).reshape(
            batch, length, self.num_heads, self.head_dim
        )
        query = F.normalize(query, dim=-1)
        keys = []
        values = []
        betas = []
        beta_scale = 2.0 if self.allow_negative_eigenvalues else 1.0
        for key_projection, value_projection, beta_projection in zip(
            self.key_projections,
            self.value_projections,
            self.beta_projections,
        ):
            key = F.silu(key_projection(hidden_states)).reshape(
                batch, length, self.num_heads, self.head_dim
            )
            value = F.silu(value_projection(hidden_states)).reshape(
                batch, length, self.num_heads, self.head_dim
            )
            keys.append(F.normalize(key, dim=-1))
            values.append(value)
            betas.append(beta_scale * torch.sigmoid(beta_projection(hidden_states)))
        stacked_keys = torch.stack(keys, dim=2)
        stacked_values = torch.stack(values, dim=2)
        stacked_betas = torch.stack(betas, dim=2)
        transition_a, transition_b = compose_delta_updates_per_token(
            stacked_keys, stacked_values, stacked_betas
        )
        if valid_mask is not None:
            identity = torch.eye(
                self.head_dim,
                dtype=transition_a.dtype,
                device=transition_a.device,
            ).expand_as(transition_a)
            valid = valid_mask.bool()[..., None, None, None]
            transition_a = torch.where(valid, transition_a, identity)
            transition_b = torch.where(
                valid, transition_b, torch.zeros_like(transition_b)
            )
        return transition_a, transition_b, query

    def forward(
        self,
        hidden_states: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        scan_mode: str = "parallel",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transition_a, transition_b, query = self.transitions(hidden_states, valid_mask)
        states, final_state = delta_product_scan(
            transition_a, transition_b, initial_state, mode=scan_mode
        )
        outputs = torch.einsum("blhd,blhdv->blhv", query, states)
        outputs = outputs / math.sqrt(self.head_dim)
        outputs = self.output_norm(outputs).flatten(start_dim=-2)
        return self.output_projection(outputs), final_state


class GatedMLP(nn.Module):
    """Small SiLU-gated MLP matching the official residual block shape."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_projection = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_projection = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_projection = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_projection(
            F.silu(self.gate_projection(hidden_states))
            * self.up_projection(hidden_states)
        )


class DeltaProductReferenceModel(nn.Module):
    """One-layer parameter-near DeltaProduct prefix classifier."""

    source_commit = "d62241a81d07aa32b1b65e7d17377f6a7cd0a5d8"

    def __init__(
        self,
        *,
        input_vocab_size: int,
        output_size: int,
        hidden_size: int = 32,
        num_heads: int = 4,
        num_householder: int = 4,
        intermediate_size: int = 112,
        norm_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if input_vocab_size < 2 or output_size < 2 or intermediate_size < 1:
            raise ValueError("vocabulary, output, and intermediate sizes are invalid")
        self.input_vocab_size = input_vocab_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.token_embeddings = nn.Embedding(input_vocab_size, hidden_size)
        self.delta_norm = nn.RMSNorm(hidden_size, eps=norm_epsilon)
        self.delta = DeltaProductReferenceLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_householder=num_householder,
            allow_negative_eigenvalues=True,
            norm_epsilon=norm_epsilon,
        )
        self.mlp_norm = nn.RMSNorm(hidden_size, eps=norm_epsilon)
        self.mlp = GatedMLP(hidden_size, intermediate_size)
        self.final_norm = nn.RMSNorm(hidden_size, eps=norm_epsilon)
        self.output_head = nn.Linear(hidden_size, output_size, bias=False)
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_head.weight, mean=0.0, std=0.02)

    @property
    def recurrent_state_scalars(self) -> int:
        return self.delta.num_heads * self.delta.head_dim**2

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        reference = self.token_embeddings.weight
        return torch.zeros(
            batch_size,
            self.delta.num_heads,
            self.delta.head_dim,
            self.delta.head_dim,
            device=device or reference.device,
            dtype=dtype or reference.dtype,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        recurrent_state: torch.Tensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        return_recurrent_state: bool = False,
        scan_mode: str = "parallel",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if token_ids.ndim != 2 or token_ids.shape[1] == 0:
            raise ValueError("token_ids must have nonempty shape (batch,length)")
        if attention_mask is not None and attention_mask.shape != token_ids.shape:
            raise ValueError("attention_mask must match token_ids")
        hidden_states = self.token_embeddings(token_ids)
        delta_output, final_state = self.delta(
            self.delta_norm(hidden_states),
            recurrent_state,
            valid_mask=attention_mask,
            scan_mode=scan_mode,
        )
        hidden_states = hidden_states + delta_output
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        logits = self.output_head(self.final_norm(hidden_states))
        if return_recurrent_state:
            return logits, final_state
        return logits

    def step(
        self, token_ids: torch.Tensor, recurrent_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_ids.ndim != 1:
            raise ValueError("step token_ids must have shape (batch,)")
        logits, state = self(
            token_ids[:, None],
            recurrent_state,
            return_recurrent_state=True,
            scan_mode="recurrent",
        )
        return logits[:, 0], state


__all__ = [
    "DeltaProductReferenceLayer",
    "DeltaProductReferenceModel",
    "GatedMLP",
    "compose_delta_transitions",
    "compose_delta_updates_per_token",
    "delta_product_scan",
    "delta_update_affine",
]
