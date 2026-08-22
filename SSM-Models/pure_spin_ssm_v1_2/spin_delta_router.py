"""Causal low-entropy event and slot router for Spin-Delta."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class RouterOutput:
    """Differentiable hard controls and the logits used to construct them."""

    controls: torch.Tensor
    write_event_logits: torch.Tensor
    write_slot_logits: torch.Tensor
    query_event_logits: torch.Tensor
    query_slot_logits: torch.Tensor


def _straight_through_binary(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    probabilities = torch.sigmoid(logits / temperature)
    hard = (probabilities >= 0.5).to(probabilities)
    return hard.detach() - probabilities.detach() + probabilities


def _straight_through_categorical(
    logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    probabilities = torch.softmax(logits / temperature, dim=-1)
    hard = F.one_hot(probabilities.argmax(dim=-1), probabilities.shape[-1]).to(
        probabilities
    )
    return hard.detach() - probabilities.detach() + probabilities


class CausalLowEntropyRouter(nn.Module):
    """Recognize local causal event grammar and emit hard two-slot controls."""

    def __init__(
        self,
        vocab_size: int = 256,
        width: int = 32,
        kernel_size: int = 3,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if min(vocab_size, width, kernel_size) < 1:
            raise ValueError("router dimensions must be positive")
        if temperature <= 0.0:
            raise ValueError("router temperature must be positive")
        self.kernel_size = kernel_size
        self.temperature = temperature
        self.embedding = nn.Embedding(vocab_size, width)
        self.context = nn.Conv1d(
            width,
            width,
            kernel_size,
            padding=kernel_size - 1,
        )
        self.output = nn.Linear(width, 6)
        with torch.no_grad():
            # Sparse-event priors avoid destructive all-token writes at step 0.
            self.output.bias.zero_()
            self.output.bias[0] = -1.0
            self.output.bias[3] = -3.0

    def forward(self, token_ids: torch.Tensor) -> RouterOutput:
        if token_ids.ndim != 2 or token_ids.shape[1] == 0:
            raise ValueError("router token_ids must have nonempty shape (batch,length)")
        embedded = self.embedding(token_ids).transpose(1, 2)
        contextual = self.context(embedded)[..., : token_ids.shape[1]]
        logits = self.output(F.silu(contextual.transpose(1, 2)))
        write_event_logits = logits[..., 0]
        write_slot_logits = logits[..., 1:3]
        query_event_logits = logits[..., 3]
        query_slot_logits = logits[..., 4:6]
        controls = torch.cat(
            (
                _straight_through_binary(
                    write_event_logits, self.temperature
                ).unsqueeze(-1),
                _straight_through_categorical(
                    write_slot_logits, self.temperature
                ),
                _straight_through_binary(
                    query_event_logits, self.temperature
                ).unsqueeze(-1),
                _straight_through_categorical(
                    query_slot_logits, self.temperature
                ),
            ),
            dim=-1,
        )
        return RouterOutput(
            controls=controls,
            write_event_logits=write_event_logits,
            write_slot_logits=write_slot_logits,
            query_event_logits=query_event_logits,
            query_slot_logits=query_slot_logits,
        )


class RoutedSpinDelta(nn.Module):
    """Compose an autonomous causal router with an unchanged Spin-Delta core."""

    def __init__(self, core: nn.Module, router: CausalLowEntropyRouter) -> None:
        super().__init__()
        self.core = core
        self.router = router

    def forward(self, token_ids: torch.Tensor, *, scan_mode: str) -> dict[str, object]:
        routing = self.router(token_ids)
        result = self.core(
            token_ids,
            scan_mode=scan_mode,
            delta_router_controls=routing.controls,
        )
        result["router"] = routing
        return result


def _balanced_binary_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.to(torch.bool)
    positive = F.softplus(-logits[target]).mean()
    negative = F.softplus(logits[~target]).mean()
    return 0.5 * (positive + negative)


def router_supervision_loss(
    routing: RouterOutput, oracle_slots: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Balanced event and conditional slot objectives from causal grammar labels."""

    if oracle_slots.shape != (*routing.write_event_logits.shape, 2):
        raise ValueError("oracle slot labels do not match router logits")
    write_target = oracle_slots[..., 0]
    query_target = oracle_slots[..., 1]
    write_mask = write_target >= 0
    query_mask = query_target >= 0
    write_event = _balanced_binary_loss(routing.write_event_logits, write_mask)
    query_event = _balanced_binary_loss(routing.query_event_logits, query_mask)
    write_slot = F.cross_entropy(
        routing.write_slot_logits[write_mask], write_target[write_mask]
    )
    query_slot = F.cross_entropy(
        routing.query_slot_logits[query_mask], query_target[query_mask]
    )
    total = 0.25 * (write_event + query_event + write_slot + query_slot)
    return {
        "total": total,
        "write_event": write_event,
        "query_event": query_event,
        "write_slot": write_slot,
        "query_slot": query_slot,
    }
