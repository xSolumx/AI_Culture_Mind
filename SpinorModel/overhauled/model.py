"""A persistent-state selective rotor SSM language model."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .algebra import (
    GA_DIM,
    GRADE_SLICES,
    RotorAffineTransition,
    apply_transition,
    associative_scan,
    geometric_product,
    identity_rotor,
    rotor_from_bivector,
    rotor_sandwich,
)


@dataclass(frozen=True)
class SpinorSSMConfig:
    vocab_size: int
    channels: int = 8
    num_layers: int = 4
    expansion: int = 2
    dropout: float = 0.0
    min_half_life: float = 4.0
    max_half_life: float = 2048.0
    minimum_step_size: float = 1e-2
    minimum_decay_rate: float = 1e-4
    maximum_rotor_angle: float = math.pi / 2
    padding_id: int = 0
    tie_embeddings: bool = True
    scan_backend: str = "parallel"

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least two")
        if self.channels < 1 or self.num_layers < 1 or self.expansion < 1:
            raise ValueError("channels, num_layers, and expansion must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.min_half_life <= 0 or self.max_half_life < self.min_half_life:
            raise ValueError("half-life bounds must be positive and ordered")
        if self.minimum_step_size <= 0 or self.minimum_decay_rate <= 0:
            raise ValueError("step-size and decay-rate floors must be positive")
        if self.maximum_rotor_angle <= 0:
            raise ValueError("maximum_rotor_angle must be positive")
        if not 0 <= self.padding_id < self.vocab_size:
            raise ValueError("padding_id must be in the vocabulary")
        if self.scan_backend not in {"parallel", "recurrent"}:
            raise ValueError("scan_backend must be 'parallel' or 'recurrent'")


def grade_invariants(multivector: torch.Tensor) -> torch.Tensor:
    """Scalar invariants/covariants used by token-selective scalar gates."""
    if multivector.shape[-1] != GA_DIM:
        raise ValueError(f"multivectors must end in {GA_DIM} components")
    return torch.stack(
        (
            multivector[..., 0],
            multivector[..., 1:4].norm(dim=-1),
            multivector[..., 4:7].norm(dim=-1),
            multivector[..., 7],
        ),
        dim=-1,
    )


class GradeLinear(nn.Module):
    """Spin(3)-equivariant channel mixing, independent within each grade."""

    def __init__(self, in_channels: int, out_channels: int, *, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = nn.Parameter(torch.empty(4, out_channels, in_channels))
        nn.init.kaiming_uniform_(self.kernel, a=math.sqrt(5))
        self.scalar_bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-2:] != (self.in_channels, GA_DIM):
            raise ValueError(
                f"expected (..., {self.in_channels}, {GA_DIM}), got {inputs.shape}"
            )
        grades = [
            torch.einsum(
                "oi,...ic->...oc", self.kernel[index], inputs[..., start:stop]
            )
            for index, (start, stop) in enumerate(GRADE_SLICES)
        ]
        outputs = torch.cat(grades, dim=-1)
        if self.scalar_bias is not None:
            scalar = outputs[..., 0] + self.scalar_bias
            outputs = torch.cat((scalar.unsqueeze(-1), outputs[..., 1:]), dim=-1)
        return outputs


class GeometricRMSNorm(nn.Module):
    def __init__(self, channels: int, epsilon: float = 1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(channels, 1))
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        rms = inputs.square().mean(dim=(-2, -1), keepdim=True)
        return inputs * torch.rsqrt(rms + self.epsilon) * self.gain


class GeometricGatedFFN(nn.Module):
    def __init__(self, channels: int, expansion: int):
        super().__init__()
        hidden = channels * expansion
        self.input = GradeLinear(channels, hidden)
        self.gate = nn.Linear(hidden * 4, hidden)
        self.output = GradeLinear(hidden, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input(inputs)
        gates = torch.sigmoid(self.gate(grade_invariants(hidden).flatten(-2)))
        return self.output(hidden * gates.unsqueeze(-1))


class SelectiveRotorSSM(nn.Module):
    """Input-selective, BIBO-stable rotor-affine state transition."""

    def __init__(self, config: SpinorSSMConfig):
        super().__init__()
        channels = config.channels
        self.channels = channels
        self.minimum_step_size = config.minimum_step_size
        self.minimum_decay_rate = config.minimum_decay_rate
        self.maximum_rotor_angle = config.maximum_rotor_angle

        controls = channels * 4
        self.step_control = nn.Linear(controls, channels)
        self.write_control = nn.Linear(controls, channels)
        self.rotor_strength = nn.Linear(controls, channels)
        self.rotor_source = GradeLinear(channels, channels, bias=False)
        self.input_projection = GradeLinear(channels, channels)

        nn.init.zeros_(self.step_control.weight)
        nn.init.zeros_(self.step_control.bias)
        nn.init.zeros_(self.rotor_strength.weight)
        nn.init.zeros_(self.rotor_strength.bias)
        nn.init.zeros_(self.write_control.weight)
        nn.init.zeros_(self.write_control.bias)

        half_lives = torch.logspace(
            math.log10(config.min_half_life),
            math.log10(config.max_half_life),
            channels,
        )
        expected_step = config.minimum_step_size + F.softplus(torch.tensor(0.0))
        target_rates = math.log(2.0) / (half_lives * expected_step)
        free_rates = target_rates - config.minimum_decay_rate
        if torch.any(free_rates <= 0):
            raise ValueError(
                "minimum_decay_rate is too large for the requested half-lives"
            )
        self.log_rates = nn.Parameter(torch.log(torch.expm1(free_rates)))

    def transitions(
        self, inputs: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> RotorAffineTransition:
        if inputs.ndim != 4 or inputs.shape[-2:] != (self.channels, GA_DIM):
            raise ValueError(
                f"inputs must have shape (batch, length, {self.channels}, {GA_DIM})"
            )
        invariants = grade_invariants(inputs).flatten(-2)
        step = self.minimum_step_size + F.softplus(self.step_control(invariants))
        rates = self.minimum_decay_rate + F.softplus(self.log_rates)
        retention = torch.exp(-step * rates)

        strength = torch.tanh(self.rotor_strength(invariants))
        bivector = self.rotor_source(inputs)[..., 4:7] * strength.unsqueeze(-1)
        rotor = rotor_from_bivector(
            bivector, maximum_angle=self.maximum_rotor_angle
        )

        write = torch.sigmoid(self.write_control(invariants))
        candidate = self.input_projection(inputs)
        # Since 0 <= write <= 1, the innovation coefficient is bounded by
        # 1-retention. With isometric rotor transport this gives the standard
        # convex-recursion BIBO bound for bounded candidates.
        drive = (1.0 - retention).unsqueeze(-1) * write.unsqueeze(-1) * candidate

        if valid_mask is not None:
            if valid_mask.shape != inputs.shape[:2]:
                raise ValueError("valid_mask must have shape (batch, length)")
            valid = valid_mask.bool()
            retention = torch.where(valid[..., None], retention, torch.ones_like(retention))
            rotor = torch.where(
                valid[..., None, None], rotor, identity_rotor(rotor)
            )
            drive = torch.where(valid[..., None, None], drive, torch.zeros_like(drive))
        return RotorAffineTransition(retention, rotor, drive)

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        backend: str = "parallel",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transition = self.transitions(inputs, valid_mask)
        if initial_state is None:
            initial_state = torch.zeros_like(inputs[:, 0])
        if initial_state.shape != inputs.shape[:1] + inputs.shape[2:]:
            raise ValueError("initial_state must have shape (batch, channels, 8)")
        if backend == "parallel":
            prefixes = associative_scan(transition)
            sequence = apply_transition(prefixes, initial_state[:, None])
        elif backend == "recurrent":
            state = initial_state
            states = []
            for position in range(inputs.shape[1]):
                state = apply_transition(
                    RotorAffineTransition(
                        transition.retention[:, position],
                        transition.rotor[:, position],
                        transition.drive[:, position],
                    ),
                    state,
                )
                states.append(state)
            sequence = torch.stack(states, dim=1)
        else:
            raise ValueError("backend must be 'parallel' or 'recurrent'")
        return sequence, sequence[:, -1]


class SpinorSSMBlock(nn.Module):
    def __init__(self, config: SpinorSSMConfig):
        super().__init__()
        self.norm1 = GeometricRMSNorm(config.channels)
        self.ssm = SelectiveRotorSSM(config)
        self.norm2 = GeometricRMSNorm(config.channels)
        self.ffn = GeometricGatedFFN(config.channels, config.expansion)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: torch.Tensor | None,
        *,
        valid_mask: torch.Tensor | None,
        backend: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, final_state = self.ssm(
            self.norm1(inputs),
            initial_state,
            valid_mask=valid_mask,
            backend=backend,
        )
        outputs = inputs + self.dropout(sequence)
        outputs = outputs + self.dropout(self.ffn(self.norm2(outputs)))
        if valid_mask is not None:
            outputs = outputs * valid_mask[..., None, None]
        return outputs, final_state


class SpinorSSMLanguageModel(nn.Module):
    """Causal language model with one fixed Cl(3, 0) state per layer/channel."""

    def __init__(self, config: SpinorSSMConfig):
        super().__init__()
        self.config = config
        width = config.channels * GA_DIM
        self.token_embedding = nn.Embedding(
            config.vocab_size, width, padding_idx=config.padding_id
        )
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[config.padding_id].zero_()
        self.blocks = nn.ModuleList(
            SpinorSSMBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = GeometricRMSNorm(config.channels)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.output_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.output_projection = (
            None if config.tie_embeddings else nn.Linear(width, config.vocab_size, bias=False)
        )

    @property
    def recurrent_state_scalars(self) -> int:
        return self.config.num_layers * self.config.channels * GA_DIM

    def initial_states(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        parameter = self.token_embedding.weight
        return tuple(
            torch.zeros(
                batch_size,
                self.config.channels,
                GA_DIM,
                device=device or parameter.device,
                dtype=dtype or parameter.dtype,
            )
            for _ in range(self.config.num_layers)
        )

    def _decode(self, outputs: torch.Tensor) -> torch.Tensor:
        flat = self.final_norm(outputs).flatten(-2)
        if self.output_projection is None:
            logits = torch.einsum("bld,vd->blv", flat, self.token_embedding.weight)
            logits = logits / math.sqrt(flat.shape[-1])
        else:
            logits = self.output_projection(flat)
        return logits + self.output_bias

    def forward(
        self,
        token_ids: torch.Tensor,
        recurrent_states: Sequence[torch.Tensor] | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        return_recurrent_states: bool = False,
        backend: str | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if token_ids.ndim != 2 or token_ids.shape[1] < 1:
            raise ValueError("token_ids must have shape (batch, nonempty sequence)")
        if attention_mask is None:
            attention_mask = token_ids.ne(self.config.padding_id)
        if attention_mask.shape != token_ids.shape:
            raise ValueError("attention_mask must match token_ids")
        if recurrent_states is None:
            recurrent_states = (None,) * self.config.num_layers
        if len(recurrent_states) != self.config.num_layers:
            raise ValueError("one recurrent state is required per layer")

        batch, length = token_ids.shape
        outputs = self.token_embedding(token_ids).reshape(
            batch, length, self.config.channels, GA_DIM
        )
        outputs = self.embedding_dropout(outputs)
        final_states = []
        selected_backend = backend or self.config.scan_backend
        for block, state in zip(self.blocks, recurrent_states):
            outputs, state = block(
                outputs,
                state,
                valid_mask=attention_mask,
                backend=selected_backend,
            )
            final_states.append(state)
        logits = self._decode(outputs)
        if return_recurrent_states:
            return logits, tuple(final_states)
        return logits

    def step(
        self,
        token_ids: torch.Tensor,
        recurrent_states: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if token_ids.ndim != 1:
            raise ValueError("step token_ids must have shape (batch,)")
        logits, states = self.forward(
            token_ids[:, None],
            recurrent_states,
            attention_mask=torch.ones_like(token_ids[:, None], dtype=torch.bool),
            return_recurrent_states=True,
            backend="recurrent",
        )
        return logits[:, 0], states

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: int | None = None,
        stop_ids: set[int] | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Prime once, then generate with the fixed-size recurrent cache."""
        if prompt_ids.ndim != 2 or prompt_ids.shape[1] < 1:
            raise ValueError("prompt_ids must have shape (batch, nonempty sequence)")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if temperature < 0:
            raise ValueError("temperature must be nonnegative")
        self.eval()
        generated = prompt_ids
        logits, states = self.forward(
            prompt_ids,
            return_recurrent_states=True,
            backend=self.config.scan_backend,
        )
        next_logits = logits[:, -1]
        for index in range(max_new_tokens):
            if temperature == 0:
                next_token = next_logits.argmax(dim=-1)
            else:
                scores = next_logits / temperature
                if top_k is not None:
                    if not 1 <= top_k <= scores.shape[-1]:
                        raise ValueError("top_k must be in [1, vocab_size]")
                    threshold = scores.topk(top_k, dim=-1).values[:, -1:]
                    scores = scores.masked_fill(scores < threshold, -torch.inf)
                probabilities = torch.softmax(scores, dim=-1)
                next_token = torch.multinomial(
                    probabilities, 1, generator=generator
                ).squeeze(-1)
            generated = torch.cat((generated, next_token[:, None]), dim=1)
            if stop_ids is not None and all(
                int(token) in stop_ids for token in next_token
            ):
                break
            if index + 1 == max_new_tokens:
                break
            next_logits, states = self.step(next_token, states)
        return generated


__all__ = [
    "GA_DIM",
    "GeometricGatedFFN",
    "GeometricRMSNorm",
    "GradeLinear",
    "SelectiveRotorSSM",
    "SpinorSSMBlock",
    "SpinorSSMConfig",
    "SpinorSSMLanguageModel",
    "grade_invariants",
]
