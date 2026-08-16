"""Experimental sign-sensitive ``Spin(3) ⋉ R^3`` motor scan.

Unit dual quaternions are an eight-scalar double cover of orientation-preserving
rigid motions ``SE(3)``.  They extend :mod:`pure_rotor_ssm.spin_scan` without
quotienting away the Spin center: a motor and its negative induce the same
rigid motion, but remain distinct recurrent states.

The update is a genuine group product, so recurrent inference and a parallel
prefix tree share one associative operation.  Translation is intentionally
unbounded because ``SE(3)`` is noncompact; this module must not be advertised as
the bounded affine memory from the maintained Pure Rotor v2.1 recurrence.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .spin_scan import quaternion_product, unit_quaternion

MOTOR_DIM = 8


def quaternion_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate in scalar-first ``[w,x,y,z]`` coordinates."""

    if quaternion.shape[-1] != 4:
        raise ValueError("quaternions must end in four components")
    scalar = quaternion[..., :1]
    return torch.cat((scalar, -quaternion[..., 1:]), dim=-1)


def dual_quaternion_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Associative product of dual quaternions ``q_r + eps q_d``."""

    if left.shape[-1] != MOTOR_DIM or right.shape[-1] != MOTOR_DIM:
        raise ValueError(f"dual quaternions must end in {MOTOR_DIM} components")
    left_real, left_dual = left.split(4, dim=-1)
    right_real, right_dual = right.split(4, dim=-1)
    real = quaternion_product(left_real, right_real)
    dual = quaternion_product(left_real, right_dual) + quaternion_product(
        left_dual, right_real
    )
    return torch.cat((real, dual), dim=-1)


def normalize_motor(motor: torch.Tensor) -> torch.Tensor:
    """Project onto the unit-motor constraints with identity at zero.

    A rigid motor ``q_r + eps q_d`` satisfies ``||q_r||=1`` and the Study
    condition ``<q_r,q_d>=0``.  Euclidean normalization of all eight
    coordinates is incorrect because it rescales translation.  This projection
    normalizes only ``q_r`` and removes the dual component parallel to it.
    """

    if motor.shape[-1] != MOTOR_DIM:
        raise ValueError(f"motors must end in {MOTOR_DIM} components")
    real, dual = motor.split(4, dim=-1)
    norm = torch.linalg.vector_norm(real, dim=-1, keepdim=True)
    threshold = torch.as_tensor(1e-6, dtype=motor.dtype, device=motor.device)
    safe_norm = norm.clamp_min(threshold)
    real_unit = real / safe_norm
    dual_scaled = dual / safe_norm
    study_component = (real_unit * dual_scaled).sum(dim=-1, keepdim=True)
    dual_unit = dual_scaled - study_component * real_unit
    projected = torch.cat((real_unit, dual_unit), dim=-1)
    identity = torch.zeros_like(motor)
    identity[..., 0] = 1
    return torch.where(norm > threshold, projected, identity)


def motor_from_rotation_translation(
    rotation: torch.Tensor, translation: torch.Tensor
) -> torch.Tensor:
    """Encode ``x -> R(rotation)x + translation`` as a unit motor."""

    if rotation.shape[-1] != 4 or translation.shape[-1] != 3:
        raise ValueError("rotation/translation must end in four/three components")
    if rotation.shape[:-1] != translation.shape[:-1]:
        raise ValueError("rotation and translation batch shapes must match")
    real = unit_quaternion(rotation)
    pure_translation = torch.cat(
        (torch.zeros_like(translation[..., :1]), translation), dim=-1
    )
    dual = 0.5 * quaternion_product(pure_translation, real)
    return torch.cat((real, dual), dim=-1)


def motor_translation(motor: torch.Tensor) -> torch.Tensor:
    """Extract the three-vector translation from a unit motor."""

    normalized = normalize_motor(motor)
    real, dual = normalized.split(4, dim=-1)
    pure_translation = 2.0 * quaternion_product(dual, quaternion_conjugate(real))
    return pure_translation[..., 1:]


def motor_inverse(motor: torch.Tensor) -> torch.Tensor:
    """Return the group inverse of a unit motor."""

    normalized = normalize_motor(motor)
    real, dual = normalized.split(4, dim=-1)
    real_inverse = quaternion_conjugate(real)
    dual_inverse = -quaternion_product(
        quaternion_product(real_inverse, dual), real_inverse
    )
    return torch.cat((real_inverse, dual_inverse), dim=-1)


def quaternion_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert scalar-first unit quaternions to 3 by 3 rotation matrices."""

    real = unit_quaternion(quaternion)
    w, x, y, z = real.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*real.shape[:-1], 3, 3)


def motor_to_matrix(motor: torch.Tensor) -> torch.Tensor:
    """Convert a motor to a homogeneous 4 by 4 rigid-motion matrix."""

    normalized = normalize_motor(motor)
    rotation = quaternion_rotation_matrix(normalized[..., :4])
    translation = motor_translation(normalized)
    matrix = torch.zeros(
        *normalized.shape[:-1], 4, 4, dtype=motor.dtype, device=motor.device
    )
    matrix[..., :3, :3] = rotation
    matrix[..., :3, 3] = translation
    matrix[..., 3, 3] = 1
    return matrix


def motor_transform_points(motor: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Apply motors to matching batches of final-axis 3D points."""

    if points.shape[-1] != 3 or points.shape[:-1] != motor.shape[:-1]:
        raise ValueError("points must have the same batch shape as motors and end in 3")
    normalized = normalize_motor(motor)
    rotation = quaternion_rotation_matrix(normalized[..., :4])
    return torch.einsum("...ij,...j->...i", rotation, points) + motor_translation(
        normalized
    )


def motor_prefix_scan(
    token_motors: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    valid_mask: torch.Tensor | None = None,
    mode: str = "parallel",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive unit-motor prefix products with a fixed eight-scalar cache."""

    if (
        token_motors.ndim != 4
        or token_motors.shape[1] == 0
        or token_motors.shape[-1] != MOTOR_DIM
    ):
        raise ValueError("token_motors must have nonempty shape (batch,length,lanes,8)")
    batch, length, lanes, _ = token_motors.shape
    if valid_mask is not None and valid_mask.shape != (batch, length):
        raise ValueError("valid_mask must have shape (batch,length)")
    tokens = normalize_motor(token_motors)
    if valid_mask is not None:
        identities = torch.zeros_like(tokens)
        identities[..., 0] = 1
        tokens = torch.where(valid_mask.bool()[..., None, None], tokens, identities)
    if initial_state is None:
        initial_state = torch.zeros_like(tokens[:, 0])
        initial_state[..., 0] = 1
    elif initial_state.shape != (batch, lanes, MOTOR_DIM):
        raise ValueError("initial_state must have shape (batch,lanes,8)")
    initial_state = normalize_motor(initial_state)

    if mode == "recurrent":
        state = initial_state
        states = []
        for position in range(length):
            state = normalize_motor(dual_quaternion_product(state, tokens[:, position]))
            states.append(state)
        sequence = torch.stack(states, dim=1)
        return sequence, state
    if mode != "parallel":
        raise ValueError("mode must be 'parallel' or 'recurrent'")

    prefixes = tokens
    offset = 1
    while offset < length:
        products = normalize_motor(
            dual_quaternion_product(prefixes[:, :-offset], prefixes[:, offset:])
        )
        prefixes = torch.cat((prefixes[:, :offset], products), dim=1)
        offset *= 2
    sequence = normalize_motor(
        dual_quaternion_product(initial_state[:, None], prefixes)
    )
    return sequence, sequence[:, -1]


class MotorTokenComposition(nn.Module):
    """Learn token-conditioned rigid Spin motors and scan their products."""

    def __init__(
        self,
        input_vocab_size: int,
        lanes: int,
        *,
        rotation_initialization_std: float = 0.5,
        translation_initialization_std: float = 0.05,
    ) -> None:
        super().__init__()
        if (
            input_vocab_size < 2
            or lanes < 1
            or rotation_initialization_std <= 0
            or translation_initialization_std <= 0
        ):
            raise ValueError("vocabulary, lanes, and initialization scales are invalid")
        self.input_vocab_size = input_vocab_size
        self.lanes = lanes
        self.token_rotations = nn.Parameter(torch.empty(input_vocab_size, lanes, 4))
        self.token_translations = nn.Parameter(torch.empty(input_vocab_size, lanes, 3))
        nn.init.normal_(self.token_rotations, mean=0.0, std=rotation_initialization_std)
        nn.init.normal_(
            self.token_translations, mean=0.0, std=translation_initialization_std
        )
        with torch.no_grad():
            self.token_rotations[..., 0].add_(1.0)

    @property
    def recurrent_state_scalars(self) -> int:
        return MOTOR_DIM * self.lanes

    def normalized_token_motors(self) -> torch.Tensor:
        return motor_from_rotation_translation(
            self.token_rotations, self.token_translations
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        state = torch.zeros(
            batch_size,
            self.lanes,
            MOTOR_DIM,
            device=device or self.token_rotations.device,
            dtype=dtype or self.token_rotations.dtype,
        )
        state[..., 0] = 1
        return state

    def forward(
        self,
        token_ids: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        scan_mode: str = "parallel",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_ids.ndim != 2 or token_ids.shape[1] == 0:
            raise ValueError("token_ids must have nonempty shape (batch,length)")
        if attention_mask is not None and attention_mask.shape != token_ids.shape:
            raise ValueError("attention_mask must match token_ids")
        token_motors = self.normalized_token_motors()[token_ids]
        return motor_prefix_scan(
            token_motors,
            initial_state,
            valid_mask=attention_mask,
            mode=scan_mode,
        )


class MotorCompositionClassifier(nn.Module):
    """Minimal rigid-motion composition classifier for controlled benchmarks."""

    def __init__(
        self,
        *,
        input_vocab_size: int,
        output_size: int,
        lanes: int,
        decoder_hidden: int,
    ) -> None:
        super().__init__()
        if output_size < 2 or decoder_hidden < 1:
            raise ValueError("output_size and decoder_hidden are invalid")
        self.composition = MotorTokenComposition(input_vocab_size, lanes)
        self.output_size = output_size
        self.decoder = nn.Sequential(
            nn.Linear(MOTOR_DIM * lanes, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, output_size),
        )

    @property
    def recurrent_state_scalars(self) -> int:
        return self.composition.recurrent_state_scalars

    def forward(
        self,
        token_ids: torch.Tensor,
        recurrent_state: torch.Tensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        return_recurrent_state: bool = False,
        scan_mode: str = "parallel",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        states, final_state = self.composition(
            token_ids,
            recurrent_state,
            attention_mask=attention_mask,
            scan_mode=scan_mode,
        )
        logits = self.decoder(states.flatten(start_dim=-2))
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


class DirectMotorPoseTracker(nn.Module):
    """Expose one learned motor lane directly as signed rigid pose.

    Unlike :class:`MotorCompositionClassifier`, this layer does not ask an MLP
    to rediscover dual-quaternion kinematics.  Its seven outputs are the real
    unit quaternion and the translation extracted from the scanned motor.  The
    only trainable parameters are one rotation and one translation increment
    per token.
    """

    def __init__(self, input_vocab_size: int) -> None:
        super().__init__()
        self.composition = MotorTokenComposition(input_vocab_size, lanes=1)

    @property
    def recurrent_state_scalars(self) -> int:
        return MOTOR_DIM

    def forward(
        self,
        token_ids: torch.Tensor,
        recurrent_state: torch.Tensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        return_recurrent_state: bool = False,
        scan_mode: str = "parallel",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if recurrent_state is not None and recurrent_state.ndim == 2:
            recurrent_state = recurrent_state[:, None, :]
        states, final_state = self.composition(
            token_ids,
            recurrent_state,
            attention_mask=attention_mask,
            scan_mode=scan_mode,
        )
        motors = states[:, :, 0]
        poses = torch.cat((motors[..., :4], motor_translation(motors)), dim=-1)
        if return_recurrent_state:
            return poses, final_state[:, 0]
        return poses

    def step(
        self, token_ids: torch.Tensor, recurrent_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_ids.ndim != 1 or recurrent_state.ndim != 2:
            raise ValueError(
                "step token_ids/state must have shapes (batch,) and (batch,8)"
            )
        poses, state = self(
            token_ids[:, None],
            recurrent_state,
            return_recurrent_state=True,
            scan_mode="recurrent",
        )
        return poses[:, 0], state


class DirectProductPoseTracker(nn.Module):
    """Matched ablation with state group ``Spin(3) x (R^3,+)``.

    Quaternion and translation increments are scanned independently.  It keeps
    the central Spin sign and has the same 49 trainable token scalars as
    :class:`DirectMotorPoseTracker`, but cannot rotate a local translation by
    the current orientation.  The contrast isolates the semidirect-product
    coupling in a motor from a merely signed direct-product state.
    """

    def __init__(self, input_vocab_size: int) -> None:
        super().__init__()
        from .spin_scan import SpinTokenComposition

        self.rotation = SpinTokenComposition(input_vocab_size, lanes=1)
        self.token_translations = nn.Parameter(torch.empty(input_vocab_size, 3))
        nn.init.normal_(self.token_translations, mean=0.0, std=0.05)

    @property
    def recurrent_state_scalars(self) -> int:
        return 7

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        state = torch.zeros(
            batch_size,
            7,
            device=device or self.token_translations.device,
            dtype=dtype or self.token_translations.dtype,
        )
        state[:, 0] = 1
        return state

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
        if recurrent_state is None:
            recurrent_state = self.initial_state(
                token_ids.shape[0], device=token_ids.device
            )
        elif recurrent_state.shape != (token_ids.shape[0], 7):
            raise ValueError("recurrent_state must have shape (batch,7)")
        rotations, final_rotation = self.rotation(
            token_ids,
            recurrent_state[:, None, :4],
            attention_mask=attention_mask,
            scan_mode=scan_mode,
        )
        translation_steps = self.token_translations[token_ids]
        if attention_mask is not None:
            translation_steps = torch.where(
                attention_mask.bool()[..., None],
                translation_steps,
                torch.zeros_like(translation_steps),
            )
        translations = recurrent_state[:, None, 4:] + torch.cumsum(
            translation_steps, dim=1
        )
        poses = torch.cat((rotations[:, :, 0], translations), dim=-1)
        final_state = torch.cat((final_rotation[:, 0], translations[:, -1]), dim=-1)
        if return_recurrent_state:
            return poses, final_state
        return poses

    def step(
        self, token_ids: torch.Tensor, recurrent_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_ids.ndim != 1:
            raise ValueError("step token_ids must have shape (batch,)")
        poses, state = self(
            token_ids[:, None],
            recurrent_state,
            return_recurrent_state=True,
            scan_mode="recurrent",
        )
        return poses[:, 0], state


__all__: Sequence[str] = (
    "MOTOR_DIM",
    "DirectMotorPoseTracker",
    "DirectProductPoseTracker",
    "MotorCompositionClassifier",
    "MotorTokenComposition",
    "dual_quaternion_product",
    "motor_from_rotation_translation",
    "motor_inverse",
    "motor_prefix_scan",
    "motor_to_matrix",
    "motor_transform_points",
    "motor_translation",
    "normalize_motor",
    "quaternion_conjugate",
    "quaternion_rotation_matrix",
)
