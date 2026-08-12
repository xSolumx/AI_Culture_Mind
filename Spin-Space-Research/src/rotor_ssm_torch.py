"""CUDA-capable PyTorch implementation of the selective rotor SSM.

This mirrors the maintained JAX model's mathematics while using a recurrent
scan, which supports both ordinary sequence training and constant-state
streaming inference on native Windows CUDA installations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

GA_DIM = 8
GRADE_SLICES = ((0, 1), (1, 4), (4, 7), (7, 8))
BASIS_MASKS = (0, 1, 2, 4, 3, 5, 6, 7)


def _multiplication_table() -> torch.Tensor:
    table = torch.zeros(GA_DIM, GA_DIM, GA_DIM)
    lookup = {mask: index for index, mask in enumerate(BASIS_MASKS)}
    for left_index, left_mask in enumerate(BASIS_MASKS):
        for right_index, right_mask in enumerate(BASIS_MASKS):
            swaps = sum(
                (right_mask & ((1 << bit) - 1)).bit_count()
                for bit in range(3)
                if left_mask & (1 << bit)
            )
            table[lookup[left_mask ^ right_mask], left_index, right_index] = (
                -1 if swaps % 2 else 1
            )
    return table


MULTIPLICATION_TABLE = _multiplication_table()
REVERSION_SIGNS = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1])


def geometric_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape[-1] != GA_DIM or right.shape[-1] != GA_DIM:
        raise ValueError(f"multivectors must end in {GA_DIM} components")
    table = MULTIPLICATION_TABLE.to(left.device, torch.result_type(left, right))
    return torch.einsum("...i,...j,kij->...k", left, right, table)


def reversion(multivector: torch.Tensor) -> torch.Tensor:
    signs = REVERSION_SIGNS.to(multivector.device, multivector.dtype)
    return multivector * signs


def rotor_sandwich(rotor: torch.Tensor, multivector: torch.Tensor) -> torch.Tensor:
    return geometric_product(geometric_product(rotor, multivector), reversion(rotor))


def normalized_rotor(parameters: torch.Tensor) -> torch.Tensor:
    parameters = parameters / parameters.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    scalar, e12, e13, e23 = parameters.chunk(4, dim=-1)
    zeros = torch.zeros_like(scalar)
    return torch.cat(
        [scalar, zeros, zeros, zeros, e12, e13, e23, zeros], dim=-1
    )


def rotor_from_bivector(
    bivector: torch.Tensor, max_angle: float = math.pi / 2
) -> torch.Tensor:
    if bivector.shape[-1] != 3:
        raise ValueError("bivectors must end in three components")
    magnitude = bivector.norm(dim=-1, keepdim=True)
    angle = max_angle * torch.tanh(magnitude)
    regular_scale = torch.sin(angle / 2) / magnitude.clamp_min(1e-7)
    tangent_scale = torch.as_tensor(max_angle / 2, dtype=bivector.dtype, device=bivector.device)
    bivector_scale = torch.where(magnitude > 1e-7, regular_scale, tangent_scale)
    parameters = torch.cat(
        [torch.cos(angle / 2), -bivector_scale * bivector], dim=-1
    )
    return normalized_rotor(parameters)


def grade_invariants(multivector: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            multivector[..., 0],
            multivector[..., 1:4].norm(dim=-1),
            multivector[..., 4:7].norm(dim=-1),
            multivector[..., 7],
        ],
        dim=-1,
    )


class GradeLinear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = nn.Parameter(torch.empty(4, out_channels, in_channels))
        nn.init.kaiming_uniform_(self.kernel, a=math.sqrt(5))
        self.scalar_bias = (
            nn.Parameter(torch.zeros(out_channels)) if use_bias else None
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-2:] != (self.in_channels, GA_DIM):
            raise ValueError("unexpected GradeLinear input shape")
        parts = [
            torch.einsum(
                "oi,...ic->...oc", self.kernel[grade], inputs[..., start:stop]
            )
            for grade, (start, stop) in enumerate(GRADE_SLICES)
        ]
        outputs = torch.cat(parts, dim=-1)
        if self.scalar_bias is not None:
            scalar = outputs[..., 0] + self.scalar_bias
            outputs = torch.cat([scalar.unsqueeze(-1), outputs[..., 1:]], dim=-1)
        return outputs


class GeometricRMSNorm(nn.Module):
    def __init__(self, channels: int, epsilon: float = 1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(channels, 1))
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        rms = inputs.square().mean(dim=(-2, -1), keepdim=True).add(self.epsilon).sqrt()
        return inputs / rms * self.gain


class GeometricGatedFFN(nn.Module):
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden_channels = channels * expansion
        self.hidden_channels = hidden_channels
        self.input = GradeLinear(channels, hidden_channels)
        self.gate = nn.Linear(hidden_channels * 4, hidden_channels)
        self.output = GradeLinear(hidden_channels, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input(inputs)
        invariants = grade_invariants(hidden).flatten(-2)
        hidden = hidden * torch.sigmoid(self.gate(invariants)).unsqueeze(-1)
        return self.output(hidden)


class SelectiveRotorSSM(nn.Module):
    def __init__(
        self,
        channels: int,
        min_half_life: float = 4.0,
        max_half_life: float = 2048.0,
        minimum_step_size: float = 1e-2,
        minimum_decay_rate: float = 1e-4,
        max_rotor_angle: float = math.pi / 2,
    ):
        super().__init__()
        if min_half_life <= 0 or max_half_life < min_half_life:
            raise ValueError("half-life bounds must be positive and ordered")
        if minimum_step_size <= 0 or minimum_decay_rate <= 0:
            raise ValueError("step-size and decay-rate floors must be positive")
        self.channels = channels
        self.minimum_step_size = minimum_step_size
        self.minimum_decay_rate = minimum_decay_rate
        self.max_rotor_angle = max_rotor_angle
        self.step_control = nn.Linear(channels * 4, channels)
        nn.init.zeros_(self.step_control.weight)
        nn.init.zeros_(self.step_control.bias)
        self.rotor_control = nn.Linear(channels * 4, channels)
        nn.init.zeros_(self.rotor_control.weight)
        nn.init.zeros_(self.rotor_control.bias)
        self.rotor_source = GradeLinear(channels, channels, use_bias=False)
        self.input_projection = GradeLinear(channels, channels)

        half_lives = torch.logspace(
            math.log10(min_half_life), math.log10(max_half_life), channels
        )
        expected_step = minimum_step_size + math.log(2.0)
        target_rates = math.log(2.0) / (half_lives * expected_step)
        free_rates = target_rates - minimum_decay_rate
        if bool(torch.any(free_rates <= 0)):
            raise ValueError(
                "minimum_decay_rate is too large for the requested half-lives"
            )
        self.log_rates = nn.Parameter(torch.log(torch.expm1(free_rates)))

    def transitions(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        invariants = grade_invariants(inputs).flatten(-2)
        step_size = self.minimum_step_size + torch.nn.functional.softplus(
            self.step_control(invariants)
        )
        rates = self.minimum_decay_rate + torch.nn.functional.softplus(
            self.log_rates
        )
        decay = torch.exp(-step_size * rates)
        rotor_strength = torch.tanh(self.rotor_control(invariants))
        source = self.rotor_source(inputs)[..., 4:7]
        rotors = rotor_from_bivector(
            source * rotor_strength.unsqueeze(-1), self.max_rotor_angle
        )
        # Stable even when decay rounds close to one in reduced precision.
        injection_variance = -torch.expm1(-2.0 * step_size * rates)
        injection = injection_variance.clamp_min(
            torch.finfo(inputs.dtype).tiny
        ).sqrt()
        drive = injection.unsqueeze(-1) * self.input_projection(inputs)
        return decay, rotors, drive

    def forward(
        self, inputs: torch.Tensor, initial_state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decay, rotors, drive = self.transitions(inputs)
        state = torch.zeros_like(drive[:, 0]) if initial_state is None else initial_state
        states = []
        for position in range(inputs.shape[1]):
            state = (
                decay[:, position].unsqueeze(-1)
                * rotor_sandwich(rotors[:, position], state)
                + drive[:, position]
            )
            states.append(state)
        return torch.stack(states, dim=1), state


class GASSMBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        dropout_rate: float = 0.1,
        max_rotor_angle: float = math.pi / 2,
    ):
        super().__init__()
        self.norm1 = GeometricRMSNorm(channels)
        self.ssm = SelectiveRotorSSM(channels, max_rotor_angle=max_rotor_angle)
        self.norm2 = GeometricRMSNorm(channels)
        self.ffn = GeometricGatedFFN(channels, expansion)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, inputs: torch.Tensor, initial_state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, final_state = self.ssm(self.norm1(inputs), initial_state)
        outputs = inputs + self.dropout(sequence)
        outputs = outputs + self.dropout(self.ffn(self.norm2(outputs)))
        return outputs, final_state


class GASSMLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        channels: int = 8,
        num_layers: int = 4,
        expansion: int = 2,
        dropout_rate: float = 0.1,
        max_rotor_angle: float = math.pi / 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.channels = channels
        self.num_layers = num_layers
        self.token_embeddings = nn.Parameter(
            torch.empty(vocab_size, channels, GA_DIM)
        )
        nn.init.normal_(self.token_embeddings, std=0.02)
        self.blocks = nn.ModuleList(
            GASSMBlock(
                channels, expansion, dropout_rate, max_rotor_angle
            )
            for _ in range(num_layers)
        )
        self.final_norm = GeometricRMSNorm(channels)
        self.vocabulary_bias = nn.Parameter(torch.zeros(vocab_size))
        self.embedding_dropout = nn.Dropout(dropout_rate)

    def initial_states(
        self, batch_size: int, *, device: torch.device | None = None
    ) -> tuple[torch.Tensor, ...]:
        device = device or self.token_embeddings.device
        return tuple(
            torch.zeros(
                batch_size,
                self.channels,
                GA_DIM,
                device=device,
                dtype=self.token_embeddings.dtype,
            )
            for _ in range(self.num_layers)
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        recurrent_states: Sequence[torch.Tensor] | None = None,
        *,
        return_recurrent_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if recurrent_states is None:
            recurrent_states = (None,) * self.num_layers
        if len(recurrent_states) != self.num_layers:
            raise ValueError("one recurrent state is required per model layer")
        outputs = self.embedding_dropout(self.token_embeddings[token_ids])
        final_states = []
        for block, initial_state in zip(self.blocks, recurrent_states):
            outputs, final_state = block(outputs, initial_state)
            final_states.append(final_state)
        outputs = self.final_norm(outputs)
        logits = torch.einsum(
            "blci,vci->blv", outputs, self.token_embeddings
        ) / math.sqrt(self.channels * GA_DIM)
        logits = logits + self.vocabulary_bias
        if return_recurrent_states:
            return logits, tuple(final_states)
        return logits


__all__ = [
    "GA_DIM",
    "GASSMBlock",
    "GASSMLanguageModel",
    "GeometricGatedFFN",
    "GeometricRMSNorm",
    "GradeLinear",
    "SelectiveRotorSSM",
    "geometric_product",
    "grade_invariants",
    "normalized_rotor",
    "reversion",
    "rotor_from_bivector",
    "rotor_sandwich",
]
