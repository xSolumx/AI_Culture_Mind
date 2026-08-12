"""Parameter-aligned recurrent transition families for controlled experiments.

Every family stores eight real values per channel and owns identically shaped
controller, drive, and decay-rate parameters.  Only the state action changes.
This makes the ladder useful for isolating commutativity, phase, geometric
structure, and input-selective rotation without changing the surrounding
sequence model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from rotor_ssm_torch import GA_DIM, rotor_from_bivector, rotor_sandwich

STATE_WIDTH = 8
FAMILY_NAMES = (
    "real_selective",
    "complex_unitary",
    "quaternion_even",
    "ga_rotor_selective",
    "ga_rotor_grade_decay",
    "hybrid_complex_ga",
    "ga_rotor_static",
)


def quaternion_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product for tensors with final order ``[1, i, j, k]``."""
    if left.shape[-1] != 4 or right.shape[-1] != 4:
        raise ValueError("quaternions must have four real components")
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


def unit_quaternion_from_bivector(
    bivector: torch.Tensor, max_angle: float = math.pi / 2
) -> torch.Tensor:
    """Exponentiate a three-coordinate imaginary quaternion with a safe tangent."""
    magnitude = bivector.norm(dim=-1, keepdim=True)
    angle = max_angle * torch.tanh(magnitude)
    regular_scale = torch.sin(angle / 2) / magnitude.clamp_min(1e-7)
    tangent_scale = torch.as_tensor(
        max_angle / 2, dtype=bivector.dtype, device=bivector.device
    )
    scale = torch.where(magnitude > 1e-7, regular_scale, tangent_scale)
    quaternion = torch.cat((torch.cos(angle / 2), -scale * bivector), dim=-1)
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-7)


class RecurrenceFamily(nn.Module):
    """Common parameter budget and stable recurrent interface."""

    action_name = "none"

    def __init__(
        self,
        channels: int,
        *,
        min_half_life: float = 4.0,
        max_half_life: float = 2048.0,
        minimum_step_size: float = 1e-2,
        minimum_decay_rate: float = 1e-4,
        max_rotation_angle: float = math.pi / 2,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if min_half_life <= 0 or max_half_life < min_half_life:
            raise ValueError("half-life bounds must be positive and ordered")
        self.channels = channels
        self.minimum_step_size = minimum_step_size
        self.minimum_decay_rate = minimum_decay_rate
        self.max_rotation_angle = max_rotation_angle
        width = channels * STATE_WIDTH

        # All families own exactly these parameter tensors in exactly this order.
        self.control_projection = nn.Linear(width, width)
        self.drive_projection = nn.Linear(width, width)
        nn.init.zeros_(self.control_projection.weight)
        nn.init.zeros_(self.control_projection.bias)

        half_lives = torch.logspace(
            math.log10(min_half_life), math.log10(max_half_life), channels
        )
        expected_step = minimum_step_size + math.log(2.0)
        target_rates = math.log(2.0) / (half_lives * expected_step)
        free_rates = target_rates - minimum_decay_rate
        if bool(torch.any(free_rates <= 0)):
            raise ValueError("decay-rate floor is too large for the half-life range")
        log_rates = torch.log(torch.expm1(free_rates))
        self.log_rates = nn.Parameter(log_rates[:, None].repeat(1, STATE_WIDTH))

    def _common_inputs(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[-2:] != (self.channels, STATE_WIDTH):
            raise ValueError(
                f"inputs must have shape (batch, length, {self.channels}, 8)"
            )
        flat = inputs.flatten(-2)
        controls = self.control_projection(flat).reshape_as(inputs)
        drive = self.drive_projection(flat).reshape_as(inputs)
        rates = self.minimum_decay_rate + torch.nn.functional.softplus(
            self.log_rates
        )
        return controls, drive, rates

    def transition_parameters(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        raise NotImplementedError

    def apply_action(
        self, action: torch.Tensor | None, state: torch.Tensor
    ) -> torch.Tensor:
        return state

    def action_magnitude(self, action: torch.Tensor | None) -> torch.Tensor:
        return torch.zeros((), device=self.log_rates.device)

    def forward(
        self, inputs: torch.Tensor, initial_state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decay, action, drive = self.transition_parameters(inputs)
        expected_state_shape = (inputs.shape[0], self.channels, STATE_WIDTH)
        if initial_state is None:
            state = torch.zeros(
                expected_state_shape, dtype=inputs.dtype, device=inputs.device
            )
        else:
            if initial_state.shape != expected_state_shape:
                raise ValueError(f"initial_state must have shape {expected_state_shape}")
            state = initial_state

        states = []
        for position in range(inputs.shape[1]):
            step_action = None if action is None else action[:, position]
            transported = self.apply_action(step_action, state)
            state = decay[:, position] * transported + drive[:, position]
            states.append(state)
        return torch.stack(states, dim=1), state

    @torch.no_grad()
    def diagnostics(self, inputs: torch.Tensor) -> dict[str, float | str]:
        decay, action, _ = self.transition_parameters(inputs)
        return {
            "action": self.action_name,
            "mean_decay": float(decay.mean()),
            "min_decay": float(decay.min()),
            "max_decay": float(decay.max()),
            "mean_action_magnitude_radians": float(self.action_magnitude(action)),
        }


class RealSelectiveRecurrence(RecurrenceFamily):
    """Commutative positive real diagonal selective SSM."""

    action_name = "positive real diagonal"

    def transition_parameters(self, inputs):
        controls, projected, rates = self._common_inputs(inputs)
        step_size = self.minimum_step_size + torch.nn.functional.softplus(controls)
        decay = torch.exp(-step_size * rates[None, None])
        drive = (1.0 - decay.square()).clamp_min(1e-6).sqrt() * projected
        return decay, None, drive


class ComplexUnitaryRecurrence(RecurrenceFamily):
    """Four commuting complex phases represented as real 2D rotations."""

    action_name = "U(1)^4 phase"

    def transition_parameters(self, inputs):
        controls, projected, rates = self._common_inputs(inputs)
        grouped_controls = controls.reshape(*controls.shape[:-1], 4, 2)
        grouped_rates = rates.reshape(self.channels, 4, 2).mean(dim=-1)
        step_size = self.minimum_step_size + torch.nn.functional.softplus(
            grouped_controls[..., 0]
        )
        grouped_decay = torch.exp(-step_size * grouped_rates[None, None])
        decay = grouped_decay.repeat_interleave(2, dim=-1)
        phase = self.max_rotation_angle * torch.tanh(grouped_controls[..., 1])
        drive = (1.0 - decay.square()).clamp_min(1e-6).sqrt() * projected
        return decay, phase, drive

    def apply_action(self, action, state):
        pairs = state.reshape(*state.shape[:-1], 4, 2)
        real, imaginary = pairs.unbind(dim=-1)
        cosine, sine = torch.cos(action), torch.sin(action)
        rotated = torch.stack(
            (cosine * real - sine * imaginary, sine * real + cosine * imaginary),
            dim=-1,
        )
        return rotated.reshape_as(state)

    def action_magnitude(self, action):
        return action.abs().mean()


class QuaternionEvenRecurrence(RecurrenceFamily):
    """Two noncommutative quaternion states with unit left multiplication."""

    action_name = "unit quaternion left action"

    def transition_parameters(self, inputs):
        controls, projected, rates = self._common_inputs(inputs)
        grouped_controls = controls.reshape(*controls.shape[:-1], 2, 4)
        grouped_rates = rates.reshape(self.channels, 2, 4).mean(dim=-1)
        step_size = self.minimum_step_size + torch.nn.functional.softplus(
            grouped_controls[..., 0]
        )
        grouped_decay = torch.exp(-step_size * grouped_rates[None, None])
        decay = grouped_decay.repeat_interleave(4, dim=-1)
        quaternions = unit_quaternion_from_bivector(
            grouped_controls[..., 1:4], self.max_rotation_angle
        )
        drive = (1.0 - decay.square()).clamp_min(1e-6).sqrt() * projected
        return decay, quaternions, drive

    def apply_action(self, action, state):
        quaternions = state.reshape(*state.shape[:-1], 2, 4)
        return quaternion_product(action, quaternions).reshape_as(state)

    def action_magnitude(self, action):
        return (2.0 * torch.acos(action[..., 0].clamp(-1.0, 1.0))).mean()


class GARotorRecurrence(RecurrenceFamily):
    """Full Cl(3,0) multivector state transported by rotor sandwiching."""

    action_name = "selective Cl(3,0) rotor sandwich"

    def __init__(
        self,
        channels: int,
        *,
        selective_rotation: bool,
        grade_decay: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(channels, **kwargs)
        self.selective_rotation = selective_rotation
        self.grade_decay = grade_decay
        if grade_decay and not selective_rotation:
            raise ValueError("grade decay is only defined for the selective rotor family")
        if grade_decay:
            self.action_name = "selective Cl(3,0) rotor with grade decay"
        if not selective_rotation:
            self.action_name = "static Cl(3,0) rotor sandwich"

    def transition_parameters(self, inputs):
        controls, projected, rates = self._common_inputs(inputs)
        if self.grade_decay:
            grade_controls = controls[..., 0:4]
            grade_rates = torch.stack(
                (
                    rates[..., 0],
                    rates[..., 1:4].mean(dim=-1),
                    rates[..., 4:7].mean(dim=-1),
                    rates[..., 7],
                ),
                dim=-1,
            )
            grade_steps = self.minimum_step_size + torch.nn.functional.softplus(
                grade_controls
            )
            grade_decay = torch.exp(-grade_steps * grade_rates[None, None])
            decay = torch.cat(
                (
                    grade_decay[..., 0:1],
                    grade_decay[..., 1:2].expand(*grade_decay.shape[:-1], 3),
                    grade_decay[..., 2:3].expand(*grade_decay.shape[:-1], 3),
                    grade_decay[..., 3:4],
                ),
                dim=-1,
            )
            bivectors = controls[..., 4:7]
        else:
            step_size = self.minimum_step_size + torch.nn.functional.softplus(
                controls[..., 0]
            )
            channel_rates = rates.mean(dim=-1)
            channel_decay = torch.exp(-step_size * channel_rates[None, None])
            decay = channel_decay[..., None].expand_as(inputs)
            bivectors = controls[..., 1:4]

        if not self.selective_rotation:
            static = self.control_projection.bias.reshape(
                self.channels, STATE_WIDTH
            )[:, 1:4]
            bivectors = static[None, None].expand(*inputs.shape[:2], -1, -1)
        rotors = rotor_from_bivector(bivectors, self.max_rotation_angle)
        drive = (1.0 - decay.square()).clamp_min(1e-6).sqrt() * projected
        return decay, rotors, drive

    def apply_action(self, action, state):
        return rotor_sandwich(action, state)

    def action_magnitude(self, action):
        return (2.0 * torch.acos(action[..., 0].clamp(-1.0, 1.0))).mean()


class HybridComplexGARecurrence(RecurrenceFamily):
    """Direct sum of complex-phase and selective Cl(3,0) channels.

    This keeps the total state and parameter budget fixed. The first half of
    channels use U(1)^4 actions and the second half use rotor sandwiches; the
    surrounding block supplies learned communication between the subspaces.
    """

    action_name = "U(1)^4 direct-sum selective Cl(3,0) rotor"

    def __init__(self, channels: int, **kwargs) -> None:
        if channels < 2 or channels % 2:
            raise ValueError("hybrid_complex_ga requires an even channel count >= 2")
        super().__init__(channels, **kwargs)
        self.split = channels // 2

    def transition_parameters(self, inputs):
        controls, projected, rates = self._common_inputs(inputs)

        complex_controls = controls[..., : self.split, :].reshape(
            *controls.shape[:2], self.split, 4, 2
        )
        complex_rates = rates[: self.split].reshape(self.split, 4, 2).mean(dim=-1)
        complex_steps = self.minimum_step_size + torch.nn.functional.softplus(
            complex_controls[..., 0]
        )
        complex_decay = torch.exp(-complex_steps * complex_rates[None, None])
        complex_decay = complex_decay.repeat_interleave(2, dim=-1)
        phases = self.max_rotation_angle * torch.tanh(complex_controls[..., 1])
        phase_action = torch.nn.functional.pad(phases, (0, 4))

        ga_controls = controls[..., self.split :, :]
        ga_rates = rates[self.split :].mean(dim=-1)
        ga_steps = self.minimum_step_size + torch.nn.functional.softplus(
            ga_controls[..., 0]
        )
        ga_decay = torch.exp(-ga_steps * ga_rates[None, None])
        ga_decay = ga_decay[..., None].expand_as(ga_controls)
        rotors = rotor_from_bivector(
            ga_controls[..., 1:4], self.max_rotation_angle
        )

        decay = torch.cat((complex_decay, ga_decay), dim=-2)
        action = torch.cat((phase_action, rotors), dim=-2)
        drive = (1.0 - decay.square()).clamp_min(1e-6).sqrt() * projected
        return decay, action, drive

    def apply_action(self, action, state):
        complex_state = state[..., : self.split, :]
        pairs = complex_state.reshape(*complex_state.shape[:-1], 4, 2)
        real, imaginary = pairs.unbind(dim=-1)
        phases = action[..., : self.split, :4]
        cosine, sine = torch.cos(phases), torch.sin(phases)
        complex_rotated = torch.stack(
            (cosine * real - sine * imaginary, sine * real + cosine * imaginary),
            dim=-1,
        ).reshape_as(complex_state)

        ga_rotated = rotor_sandwich(
            action[..., self.split :, :], state[..., self.split :, :]
        )
        return torch.cat((complex_rotated, ga_rotated), dim=-2)

    def action_magnitude(self, action):
        phase_magnitude = action[..., : self.split, :4].abs().mean()
        rotors = action[..., self.split :, :]
        rotor_magnitude = (
            2.0 * torch.acos(rotors[..., 0].clamp(-1.0, 1.0))
        ).mean()
        return 0.5 * (phase_magnitude + rotor_magnitude)


def make_recurrence(family: str, channels: int) -> RecurrenceFamily:
    if family == "real_selective":
        return RealSelectiveRecurrence(channels)
    if family == "complex_unitary":
        return ComplexUnitaryRecurrence(channels)
    if family == "quaternion_even":
        return QuaternionEvenRecurrence(channels)
    if family == "ga_rotor_selective":
        return GARotorRecurrence(channels, selective_rotation=True)
    if family == "ga_rotor_grade_decay":
        return GARotorRecurrence(
            channels, selective_rotation=True, grade_decay=True
        )
    if family == "hybrid_complex_ga":
        return HybridComplexGARecurrence(channels)
    if family == "ga_rotor_static":
        return GARotorRecurrence(channels, selective_rotation=False)
    raise ValueError(f"unknown recurrence family: {family}")


class RecurrenceBlock(nn.Module):
    """A common residual wrapper used unchanged across the transition ladder."""

    def __init__(self, family: str, channels: int, expansion: int = 2) -> None:
        super().__init__()
        width = channels * STATE_WIDTH
        self.channels = channels
        self.input_norm = nn.LayerNorm(width)
        self.recurrence = make_recurrence(family, channels)
        self.output_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, expansion * width),
            nn.SiLU(),
            nn.Linear(expansion * width, width),
        )

    def forward(self, inputs, initial_state=None):
        normalized = self.input_norm(inputs.flatten(-2)).reshape_as(inputs)
        sequence, final_state = self.recurrence(normalized, initial_state)
        outputs = inputs + sequence
        flattened = outputs.flatten(-2)
        outputs = outputs + self.feed_forward(self.output_norm(flattened)).reshape_as(
            outputs
        )
        return outputs, final_state


class RecurrenceSequenceModel(nn.Module):
    """Token model with explicit, fixed-size recurrent state for every layer."""

    def __init__(
        self,
        vocab_size: int,
        output_size: int,
        *,
        family: str,
        channels: int = 4,
        num_layers: int = 2,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        if family not in FAMILY_NAMES:
            raise ValueError(f"unknown family {family!r}")
        self.family = family
        self.channels = channels
        self.num_layers = num_layers
        width = channels * STATE_WIDTH
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.blocks = nn.ModuleList(
            RecurrenceBlock(family, channels, expansion) for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(width)
        self.output_head = nn.Linear(width, output_size)

    def initial_states(
        self, batch_size: int, *, device: torch.device | None = None
    ) -> tuple[torch.Tensor, ...]:
        device = device or self.token_embedding.weight.device
        return tuple(
            torch.zeros(
                batch_size,
                self.channels,
                STATE_WIDTH,
                device=device,
                dtype=self.token_embedding.weight.dtype,
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
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        if recurrent_states is None:
            recurrent_states = (None,) * self.num_layers
        if len(recurrent_states) != self.num_layers:
            raise ValueError("one recurrent state is required per layer")

        outputs = self.token_embedding(token_ids).reshape(
            token_ids.shape[0], token_ids.shape[1], self.channels, STATE_WIDTH
        )
        final_states = []
        for block, initial_state in zip(self.blocks, recurrent_states):
            outputs, final_state = block(outputs, initial_state)
            final_states.append(final_state)
        logits = self.output_head(self.final_norm(outputs.flatten(-2)))
        if return_recurrent_states:
            return logits, tuple(final_states)
        return logits


__all__ = [
    "FAMILY_NAMES",
    "STATE_WIDTH",
    "ComplexUnitaryRecurrence",
    "GARotorRecurrence",
    "HybridComplexGARecurrence",
    "QuaternionEvenRecurrence",
    "RealSelectiveRecurrence",
    "RecurrenceSequenceModel",
    "make_recurrence",
    "quaternion_product",
    "unit_quaternion_from_bivector",
]
