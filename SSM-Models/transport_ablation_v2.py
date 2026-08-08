"""Matched transport families for the pure rotor SSM v2.1 experiment.

This module is deliberately outside :mod:`pure_rotor_ssm`: the pure package
contains the canonical rotor model, while this file contains falsification
baselines.  Every family shares the same bounded affine update and surrounding
block.  Only the state transport changes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from pure_rotor_ssm.torch_backend import (
    GA_DIM,
    INVARIANT_FEATURES,
    GeometricDropout,
    GeometricGatedFFN,
    GeometricRMSNorm,
    Spin3IsotypicLinear,
    bounded_multivector,
    identity_rotor,
    rotor_from_bivector,
    rotor_product,
    rotor_sandwich,
    spin3_invariant_features,
)
from torch import nn
from torch.nn import functional as F

FAMILY_NAMES = (
    "identity",
    "real_diagonal",
    "complex_phase",
    "quaternion_left",
    "rotor",
    "fixed_rotor",
    "so8",
)


def quaternion_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product in final-axis order ``[1,i,j,k]``."""

    if left.shape[-1] != 4 or right.shape[-1] != 4:
        raise ValueError("quaternions must end in four components")
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def unit_quaternion_from_vector(
    vector: torch.Tensor, max_angle: float = math.pi
) -> torch.Tensor:
    """Smooth bounded unit-quaternion chart with a finite zero tangent."""

    rotor = rotor_from_bivector(vector, max_angle)
    return torch.stack(
        (rotor[..., 0], rotor[..., 4], rotor[..., 5], rotor[..., 6]), dim=-1
    )


def _householder_so8(parameters: torch.Tensor) -> torch.Tensor:
    """Eight paired reflections giving an identity-initialized SO(8) action."""

    if parameters.shape[-2:] != (GA_DIM, GA_DIM):
        raise ValueError("SO(8) reflection parameters must end in (8,8)")
    base_indices = torch.arange(GA_DIM, device=parameters.device) // 2
    base = F.one_hot(base_indices, GA_DIM).to(parameters.dtype)
    vectors = F.normalize(base + 0.25 * torch.tanh(parameters), dim=-1)
    identity = torch.eye(
        GA_DIM, dtype=parameters.dtype, device=parameters.device
    ).expand(*parameters.shape[:-2], GA_DIM, GA_DIM)
    action = identity
    for index in range(GA_DIM):
        vector = vectors[..., index, :]
        reflection = identity - 2 * vector[..., :, None] * vector[..., None, :]
        action = reflection @ action
    return action


def identity_action(reference: torch.Tensor, family: str) -> torch.Tensor:
    """Return the family identity with ``reference`` batch/time/channel axes."""

    prefix = reference.shape[:-1]
    if family == "identity":
        return torch.ones(*prefix, 1, dtype=reference.dtype, device=reference.device)
    if family == "real_diagonal":
        return torch.ones(
            *prefix, GA_DIM, dtype=reference.dtype, device=reference.device
        )
    if family == "complex_phase":
        action = torch.zeros(
            *prefix, 4, 2, dtype=reference.dtype, device=reference.device
        )
        action[..., 0] = 1
        return action
    if family == "quaternion_left":
        action = torch.zeros(*prefix, 4, dtype=reference.dtype, device=reference.device)
        action[..., 0] = 1
        return action
    if family in ("rotor", "fixed_rotor"):
        return identity_rotor(reference)
    if family == "so8":
        return torch.eye(GA_DIM, dtype=reference.dtype, device=reference.device).expand(
            *prefix, GA_DIM, GA_DIM
        )
    raise ValueError(f"unknown transport family: {family}")


def apply_transport(
    family: str, action: torch.Tensor, state: torch.Tensor
) -> torch.Tensor:
    """Apply one family action to an eight-coordinate state."""

    if family == "identity":
        return state
    if family == "real_diagonal":
        return action * state
    if family == "complex_phase":
        pairs = state.reshape(*state.shape[:-1], 4, 2)
        real, imaginary = pairs.unbind(dim=-1)
        cosine, sine = action.unbind(dim=-1)
        return torch.stack(
            (cosine * real - sine * imaginary, sine * real + cosine * imaginary),
            dim=-1,
        ).flatten(-2)
    if family == "quaternion_left":
        copies = state.reshape(*state.shape[:-1], 2, 4)
        return quaternion_product(action.unsqueeze(-2), copies).flatten(-2)
    if family in ("rotor", "fixed_rotor"):
        return rotor_sandwich(action, state)
    if family == "so8":
        return torch.einsum("...ij,...j->...i", action, state)
    raise ValueError(f"unknown transport family: {family}")


def compose_actions(
    family: str, later: torch.Tensor, earlier: torch.Tensor
) -> torch.Tensor:
    """Compose ``later`` after ``earlier`` in chronological order."""

    if family == "identity":
        return later
    if family == "real_diagonal":
        return later * earlier
    if family == "complex_phase":
        lc, ls = later.unbind(dim=-1)
        ec, es = earlier.unbind(dim=-1)
        return torch.stack((lc * ec - ls * es, ls * ec + lc * es), dim=-1)
    if family == "quaternion_left":
        return quaternion_product(later, earlier)
    if family in ("rotor", "fixed_rotor"):
        return rotor_product(later, earlier)
    if family == "so8":
        return later @ earlier
    raise ValueError(f"unknown transport family: {family}")


def compose_affine_transitions(
    family: str,
    later: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    earlier: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    later_decay, later_action, later_drive = later
    earlier_decay, earlier_action, earlier_drive = earlier
    return (
        later_decay * earlier_decay,
        compose_actions(family, later_action, earlier_action),
        later_drive
        + later_decay.unsqueeze(-1)
        * apply_transport(family, later_action, earlier_drive),
    )


def transport_affine_scan(
    family: str,
    decay: torch.Tensor,
    actions: torch.Tensor,
    drive: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable Hillis--Steele scan for a closed transport family."""

    if decay.ndim != 3 or drive.ndim != 4 or drive.shape[:-1] != decay.shape:
        raise ValueError("transitions must have (B,L,C) and (B,L,C,8) shapes")
    if drive.shape[-1] != GA_DIM or decay.shape[1] == 0:
        raise ValueError("transitions require a nonempty eight-coordinate sequence")
    cumulative = (decay, actions, drive)
    offset = 1
    while offset < decay.shape[1]:
        earlier = tuple(value[:, :-offset] for value in cumulative)
        later = tuple(value[:, offset:] for value in cumulative)
        combined = compose_affine_transitions(family, later, earlier)
        cumulative = tuple(
            torch.cat((value[:, :offset], update), dim=1)
            for value, update in zip(cumulative, combined)
        )
        offset *= 2
    cumulative_decay, cumulative_action, cumulative_drive = cumulative
    if initial_state is None:
        states = cumulative_drive
    else:
        if initial_state.shape != drive.shape[:1] + drive.shape[2:]:
            raise ValueError("initial_state must have shape (B,C,8)")
        states = (
            cumulative_decay.unsqueeze(-1)
            * apply_transport(family, cumulative_action, initial_state[:, None])
            + cumulative_drive
        )
    return states, states[:, -1]


def transport_recurrent_scan(
    family: str,
    decay: torch.Tensor,
    actions: torch.Tensor,
    drive: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if decay.shape[1] == 0:
        raise ValueError("cannot scan an empty sequence")
    state = torch.zeros_like(drive[:, 0]) if initial_state is None else initial_state
    states = []
    for position in range(decay.shape[1]):
        state = (
            decay[:, position].unsqueeze(-1)
            * apply_transport(family, actions[:, position], state)
            + drive[:, position]
        )
        states.append(state)
    return torch.stack(states, dim=1), state


class MatchedTransportSSM(nn.Module):
    """Shared v2.1 bounded recurrence with a swappable transport action."""

    def __init__(
        self,
        channels: int,
        family: str,
        *,
        min_half_life: float = 4.0,
        max_half_life: float = 2048.0,
        minimum_step_size: float = 1e-2,
        minimum_decay_rate: float = 1e-4,
        max_action_angle: float = math.pi,
    ) -> None:
        super().__init__()
        if family not in FAMILY_NAMES:
            raise ValueError(f"family must be one of {FAMILY_NAMES}")
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = channels
        self.family = family
        self.minimum_step_size = minimum_step_size
        self.minimum_decay_rate = minimum_decay_rate
        self.max_action_angle = max_action_angle
        controls = channels * INVARIANT_FEATURES

        self.step_control = nn.Linear(controls, channels)
        self.write_control = nn.Linear(controls, channels)
        self.input_projection = Spin3IsotypicLinear(channels, channels)
        nn.init.zeros_(self.step_control.weight)
        nn.init.zeros_(self.step_control.bias)
        nn.init.zeros_(self.write_control.weight)
        nn.init.zeros_(self.write_control.bias)

        if family == "real_diagonal":
            self.action_control = nn.Linear(controls, channels * GA_DIM)
            nn.init.zeros_(self.action_control.weight)
            nn.init.constant_(self.action_control.bias, -5.0)
        elif family == "complex_phase":
            self.action_control = nn.Linear(controls, channels * 4)
            nn.init.zeros_(self.action_control.weight)
            nn.init.zeros_(self.action_control.bias)
        elif family == "quaternion_left":
            self.action_control = nn.Linear(controls, channels * 3)
            nn.init.zeros_(self.action_control.weight)
            nn.init.zeros_(self.action_control.bias)
        elif family == "rotor":
            self.rotor_control = nn.Linear(controls, channels)
            self.rotor_source = Spin3IsotypicLinear(channels, channels, use_bias=False)
            nn.init.zeros_(self.rotor_control.weight)
            nn.init.zeros_(self.rotor_control.bias)
        elif family == "fixed_rotor":
            self.fixed_bivector = nn.Parameter(torch.zeros(channels, 3))
        elif family == "so8":
            self.action_control = nn.Linear(controls, channels * GA_DIM * GA_DIM)
            nn.init.zeros_(self.action_control.weight)
            nn.init.zeros_(self.action_control.bias)

        half_lives = torch.logspace(
            math.log10(min_half_life), math.log10(max_half_life), channels
        )
        expected_step = minimum_step_size + math.log(2.0)
        target_rates = math.log(2.0) / (half_lives * expected_step)
        free_rates = target_rates - minimum_decay_rate
        if bool(torch.any(free_rates <= 0)):
            raise ValueError("decay-rate floor is too large for half-life range")
        self.log_rates = nn.Parameter(torch.log(torch.expm1(free_rates)))

    def _actions(self, inputs: torch.Tensor, invariants: torch.Tensor) -> torch.Tensor:
        prefix = inputs.shape[:3]
        if self.family == "identity":
            return torch.ones(*prefix, 1, dtype=inputs.dtype, device=inputs.device)
        if self.family == "real_diagonal":
            controls = self.action_control(invariants).reshape(*prefix, GA_DIM)
            return torch.exp(-F.softplus(controls))
        if self.family == "complex_phase":
            angles = self.max_action_angle * torch.tanh(
                self.action_control(invariants).reshape(*prefix, 4)
            )
            return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        if self.family == "quaternion_left":
            vector = self.action_control(invariants).reshape(*prefix, 3)
            return unit_quaternion_from_vector(vector, self.max_action_angle)
        if self.family == "rotor":
            strength = torch.tanh(self.rotor_control(invariants))
            source = self.rotor_source(inputs)[..., 4:7]
            return rotor_from_bivector(
                source * strength.unsqueeze(-1), self.max_action_angle
            )
        if self.family == "fixed_rotor":
            action = rotor_from_bivector(
                self.fixed_bivector.to(inputs.dtype), self.max_action_angle
            )
            return action.expand(*prefix, GA_DIM)
        if self.family == "so8":
            parameters = self.action_control(invariants).reshape(
                *prefix, GA_DIM, GA_DIM
            )
            return _householder_so8(parameters)
        raise AssertionError("unreachable family")

    def transitions(
        self,
        inputs: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        force_identity: bool = False,
        action_permutation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[-2:] != (self.channels, GA_DIM):
            raise ValueError("inputs must have shape (B,L,C,8)")
        invariants = spin3_invariant_features(inputs).flatten(-2)
        step = self.minimum_step_size + F.softplus(self.step_control(invariants))
        rates = self.minimum_decay_rate + F.softplus(self.log_rates)
        decay = torch.exp(-step * rates)
        actions = self._actions(inputs, invariants)
        write = torch.sigmoid(self.write_control(invariants))
        candidate = bounded_multivector(self.input_projection(inputs))
        drive = (1 - decay).unsqueeze(-1) * write.unsqueeze(-1) * candidate

        if force_identity:
            actions = identity_action(inputs, self.family)
        elif action_permutation is not None:
            if action_permutation.shape != (inputs.shape[1],):
                raise ValueError("action_permutation must have shape (length,)")
            actions = actions.index_select(1, action_permutation)

        if valid_mask is not None:
            if valid_mask.shape != inputs.shape[:2]:
                raise ValueError("valid_mask must have shape (B,L)")
            valid = valid_mask.bool()
            decay = torch.where(valid[..., None], decay, torch.ones_like(decay))
            action_identity = identity_action(inputs, self.family)
            action_mask = valid
            while action_mask.ndim < actions.ndim:
                action_mask = action_mask.unsqueeze(-1)
            actions = torch.where(action_mask, actions, action_identity)
            drive = torch.where(valid[..., None, None], drive, torch.zeros_like(drive))
        return decay, actions, drive

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        scan_mode: str = "parallel",
        force_identity: bool = False,
        action_permutation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transitions = self.transitions(
            inputs,
            valid_mask=valid_mask,
            force_identity=force_identity,
            action_permutation=action_permutation,
        )
        if scan_mode == "parallel":
            return transport_affine_scan(
                self.family, *transitions, initial_state=initial_state
            )
        if scan_mode == "recurrent":
            return transport_recurrent_scan(
                self.family, *transitions, initial_state=initial_state
            )
        raise ValueError("scan_mode must be 'parallel' or 'recurrent'")

    @torch.no_grad()
    def diagnostics(self, inputs: torch.Tensor) -> dict[str, float | str]:
        decay, action, drive = self.transitions(inputs)
        identity = identity_action(inputs, self.family)
        if self.family in ("rotor", "fixed_rotor", "quaternion_left"):
            magnitude = 2 * torch.acos(action[..., 0].abs().clamp(0, 1))
        elif self.family == "complex_phase":
            magnitude = torch.atan2(action[..., 1], action[..., 0]).abs()
        else:
            difference = action - identity
            magnitude = difference.flatten(start_dim=3).norm(dim=-1)
        return {
            "family": self.family,
            "mean_decay": float(decay.mean()),
            "min_decay": float(decay.min()),
            "max_decay": float(decay.max()),
            "mean_action_magnitude": float(magnitude.mean()),
            "mean_drive_norm": float(drive.norm(dim=-1).mean()),
        }


class MatchedTransportBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        family: str,
        expansion: int = 2,
        dropout_rate: float = 0.0,
        max_action_angle: float = math.pi,
    ) -> None:
        super().__init__()
        self.norm1 = GeometricRMSNorm(channels)
        self.ssm = MatchedTransportSSM(
            channels, family, max_action_angle=max_action_angle
        )
        self.norm2 = GeometricRMSNorm(channels)
        self.ffn = GeometricGatedFFN(channels, expansion)
        self.dropout = GeometricDropout(dropout_rate)

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        scan_mode: str = "parallel",
        force_identity: bool = False,
        action_permutation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, final_state = self.ssm(
            self.norm1(inputs),
            initial_state,
            valid_mask=valid_mask,
            scan_mode=scan_mode,
            force_identity=force_identity,
            action_permutation=action_permutation,
        )
        outputs = inputs + self.dropout(sequence)
        outputs = outputs + self.dropout(self.ffn(self.norm2(outputs)))
        return outputs, final_state


class MatchedTransportLanguageModel(nn.Module):
    """The v2.1 outer model with only the state transport exchanged."""

    def __init__(
        self,
        vocab_size: int,
        channels: int,
        num_layers: int,
        family: str,
        *,
        expansion: int = 2,
        dropout_rate: float = 0.0,
        max_action_angle: float = math.pi,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.channels = channels
        self.num_layers = num_layers
        self.family = family
        self.token_embeddings = nn.Parameter(torch.empty(vocab_size, channels, GA_DIM))
        nn.init.normal_(self.token_embeddings, std=0.02)
        self.blocks = nn.ModuleList(
            MatchedTransportBlock(
                channels,
                family,
                expansion,
                dropout_rate,
                max_action_angle,
            )
            for _ in range(num_layers)
        )
        self.final_norm = GeometricRMSNorm(channels)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.embedding_dropout = GeometricDropout(dropout_rate)

    @property
    def recurrent_state_scalars(self) -> int:
        return self.num_layers * self.channels * GA_DIM

    def forward(
        self,
        token_ids: torch.Tensor,
        recurrent_states: Sequence[torch.Tensor] | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        return_recurrent_states: bool = False,
        scan_mode: str = "parallel",
        force_identity: bool = False,
        action_permutation: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if token_ids.ndim != 2 or token_ids.shape[1] == 0:
            raise ValueError("token_ids must have nonempty shape (B,L)")
        if recurrent_states is None:
            recurrent_states = (None,) * self.num_layers
        if len(recurrent_states) != self.num_layers:
            raise ValueError("one recurrent state is required per layer")
        outputs = self.embedding_dropout(self.token_embeddings[token_ids])
        final_states = []
        for block, initial_state in zip(self.blocks, recurrent_states):
            outputs, final_state = block(
                outputs,
                initial_state,
                valid_mask=attention_mask,
                scan_mode=scan_mode,
                force_identity=force_identity,
                action_permutation=action_permutation,
            )
            final_states.append(final_state)
        outputs = self.final_norm(outputs)
        logits = torch.einsum("blci,vci->blv", outputs, self.token_embeddings)
        logits /= math.sqrt(self.channels * GA_DIM)
        logits = logits + self.output_bias
        if return_recurrent_states:
            return logits, tuple(final_states)
        return logits


class MatchedTransportClassifier(nn.Module):
    """Endpoint classifier used by the frozen memory diagnostics."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        channels: int,
        num_layers: int,
        family: str,
        *,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.family = family
        self.embeddings = nn.Parameter(torch.empty(vocab_size, channels, GA_DIM))
        nn.init.normal_(self.embeddings, std=0.02)
        self.blocks = nn.ModuleList(
            MatchedTransportBlock(channels, family, expansion, 0.0)
            for _ in range(num_layers)
        )
        self.final_norm = GeometricRMSNorm(channels)
        self.classifier = nn.Linear(channels * GA_DIM, num_classes)

    def forward(
        self, token_ids: torch.Tensor, *, scan_mode: str = "parallel"
    ) -> torch.Tensor:
        outputs = self.embeddings[token_ids]
        for block in self.blocks:
            outputs, _ = block(outputs, scan_mode=scan_mode)
        endpoint = self.final_norm(outputs[:, -1]).flatten(1)
        return self.classifier(endpoint)


__all__ = [
    "FAMILY_NAMES",
    "MatchedTransportBlock",
    "MatchedTransportClassifier",
    "MatchedTransportLanguageModel",
    "MatchedTransportSSM",
    "apply_transport",
    "compose_actions",
    "compose_affine_transitions",
    "identity_action",
    "quaternion_product",
    "transport_affine_scan",
    "transport_recurrent_scan",
    "unit_quaternion_from_vector",
]
