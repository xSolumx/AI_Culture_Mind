"""Pure PyTorch implementation of the selective Cl(3,0) rotor SSM.

This module contains no data loading, optimizer, checkpoint, experiment, or
reporting code. Its complete state transition is

    h_t = d_t Ad(q_t) h_(t-1) + (1-d_t) w_t z_t,

where ``0 < d_t < 1``, ``0 < w_t < 1``, ``q_t`` is a unit rotor, and
``||z_t|| < 1``. Consequently every valid step obeys

    ||h_t|| <= d_t ||h_(t-1)|| + (1-d_t),

and hence ``||h_t|| <= max(||h_0||, 1)`` in exact arithmetic. Transition
composition is associative, enabling a vectorized logarithmic-depth training
scan and a fixed-state recurrent inference path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from . import __version__

GA_DIM = 8
INVARIANT_FEATURES = 5
BASIS_MASKS = (0, 1, 2, 4, 3, 5, 6, 7)
REVERSION_SIGNS = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1])


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


def geometric_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Broadcasted Euclidean Cl(3,0) product on the final coefficient axis."""

    if left.shape[-1] != GA_DIM or right.shape[-1] != GA_DIM:
        raise ValueError(f"multivectors must end in {GA_DIM} components")
    table = MULTIPLICATION_TABLE.to(left.device, torch.result_type(left, right))
    return torch.einsum("...i,...j,kij->...k", left, right, table)


def reversion(multivector: torch.Tensor) -> torch.Tensor:
    if multivector.shape[-1] != GA_DIM:
        raise ValueError(f"multivectors must end in {GA_DIM} components")
    return multivector * REVERSION_SIGNS.to(multivector.device, multivector.dtype)


def identity_rotor(reference: torch.Tensor) -> torch.Tensor:
    rotor = torch.zeros_like(reference)
    rotor[..., 0] = 1
    return rotor


def normalized_rotor(parameters: torch.Tensor) -> torch.Tensor:
    """Embed normalized ``[scalar,e12,e13,e23]`` parameters in Cl(3,0).

    Normalization has no continuous extension at the zero vector; identity is
    the explicit deterministic fallback used there.
    """

    if parameters.shape[-1] != 4:
        raise ValueError("rotor parameters must end in four components")
    norm = parameters.norm(dim=-1, keepdim=True)
    normalized = parameters / norm.clamp_min(1e-6)
    fallback = torch.zeros_like(parameters)
    fallback[..., 0] = 1
    parameters = torch.where(norm > 1e-6, normalized, fallback)
    scalar, e12, e13, e23 = parameters.unbind(dim=-1)
    zeros = torch.zeros_like(scalar)
    return torch.stack((scalar, zeros, zeros, zeros, e12, e13, e23, zeros), dim=-1)


def rotor_from_bivector(
    bivector: torch.Tensor, max_angle: float = math.pi
) -> torch.Tensor:
    """Smooth bounded exponential chart with analytic identity derivative."""

    if bivector.shape[-1] != 3:
        raise ValueError("bivectors must end in three components")
    if not math.isfinite(max_angle) or max_angle < 0:
        raise ValueError("max_angle must be finite and nonnegative")
    angle_limit = torch.as_tensor(
        max_angle, dtype=bivector.dtype, device=bivector.device
    )
    magnitude_squared = bivector.square().sum(dim=-1, keepdim=True)
    threshold = torch.as_tensor(
        torch.finfo(bivector.dtype).eps,
        dtype=bivector.dtype,
        device=bivector.device,
    )
    safe_magnitude = magnitude_squared.clamp_min(threshold).sqrt()
    regular_angle = angle_limit * torch.tanh(safe_magnitude)
    regular_scalar = torch.cos(regular_angle / 2)
    regular_scale = torch.sin(regular_angle / 2) / safe_magnitude
    small_scalar = 1 - angle_limit.square() * magnitude_squared / 8
    small_scale = (
        angle_limit / 2 - (angle_limit / 6 + angle_limit**3 / 48) * magnitude_squared
    )
    use_regular = magnitude_squared > threshold
    parameters = torch.cat(
        (
            torch.where(use_regular, regular_scalar, small_scalar),
            -torch.where(use_regular, regular_scale, small_scale) * bivector,
        ),
        dim=-1,
    )
    return normalized_rotor(parameters)


def _quaternion_vector(rotor: torch.Tensor) -> torch.Tensor:
    return torch.stack((-rotor[..., 6], rotor[..., 5], -rotor[..., 4]), dim=-1)


def rotor_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Specialized product of even rotors in eight-coefficient storage."""

    if left.shape[-1] != GA_DIM or right.shape[-1] != GA_DIM:
        raise ValueError(f"rotors must end in {GA_DIM} components")
    left_scalar = left[..., :1]
    right_scalar = right[..., :1]
    left_vector = _quaternion_vector(left)
    right_vector = _quaternion_vector(right)
    scalar = left_scalar * right_scalar - (left_vector * right_vector).sum(
        dim=-1, keepdim=True
    )
    vector = (
        left_scalar * right_vector
        + right_scalar * left_vector
        + torch.linalg.cross(left_vector, right_vector, dim=-1)
    )
    zeros = torch.zeros_like(scalar)
    return torch.cat(
        (
            scalar,
            zeros,
            zeros,
            zeros,
            -vector[..., 2:3],
            vector[..., 1:2],
            -vector[..., 0:1],
            zeros,
        ),
        dim=-1,
    )


def specialized_rotor_sandwich(
    rotor: torch.Tensor, multivector: torch.Tensor
) -> torch.Tensor:
    """Algebraically specialized conjugation via two 3D vector actions."""

    if rotor.shape[-1] != GA_DIM or multivector.shape[-1] != GA_DIM:
        raise ValueError(f"rotors and multivectors must end in {GA_DIM} components")
    scalar = rotor[..., :1]
    quaternion_vector = _quaternion_vector(rotor)
    vector_square = quaternion_vector.square().sum(dim=-1, keepdim=True)
    norm_squared = scalar.square() + vector_square

    def rotate(vectors: torch.Tensor) -> torch.Tensor:
        dot = (quaternion_vector * vectors).sum(dim=-1, keepdim=True)
        return (
            (scalar.square() - vector_square) * vectors
            + 2 * quaternion_vector * dot
            + 2 * scalar * torch.linalg.cross(quaternion_vector, vectors, dim=-1)
        )

    vector = rotate(multivector[..., 1:4])
    dual_bivector = torch.stack(
        (multivector[..., 6], -multivector[..., 5], multivector[..., 4]),
        dim=-1,
    )
    rotated_dual = rotate(dual_bivector)
    return torch.cat(
        (
            norm_squared * multivector[..., :1],
            vector,
            rotated_dual[..., 2:3],
            -rotated_dual[..., 1:2],
            rotated_dual[..., 0:1],
            norm_squared * multivector[..., 7:8],
        ),
        dim=-1,
    )


def rotor_sandwich(rotor: torch.Tensor, multivector: torch.Tensor) -> torch.Tensor:
    """Full-Cl(3) rotor conjugation using the measured faster backend path.

    Eager CUDA executes two dense eight-coordinate products faster than the
    launch-heavy algebraic specialization on the tested RTX 2070 SUPER. CPU
    uses the specialization. Both paths implement the same exact polynomial.
    """

    if rotor.is_cuda or multivector.is_cuda:
        return geometric_product(
            geometric_product(rotor, multivector), reversion(rotor)
        )
    return specialized_rotor_sandwich(rotor, multivector)


def pack_spin3_isotypic(
    multivector: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if multivector.ndim < 2 or multivector.shape[-1] != GA_DIM:
        raise ValueError("multivectors must have shape (..., channels, 8)")
    channels = multivector.shape[-2]
    trivial = torch.stack((multivector[..., 0], multivector[..., 7]), dim=-1).reshape(
        *multivector.shape[:-2], 2 * channels
    )
    dual_bivector = torch.stack(
        (multivector[..., 6], -multivector[..., 5], multivector[..., 4]),
        dim=-1,
    )
    active = torch.stack((multivector[..., 1:4], dual_bivector), dim=-2)
    return trivial, active.reshape(*multivector.shape[:-2], 2 * channels, 3)


def unpack_spin3_isotypic(trivial: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    if trivial.shape[:-1] != active.shape[:-2] or active.shape[-1] != 3:
        raise ValueError("trivial and active isotypic shapes are incompatible")
    if trivial.shape[-1] != active.shape[-2] or trivial.shape[-1] % 2:
        raise ValueError("isotypic multiplicities must agree and be even")
    channels = trivial.shape[-1] // 2
    trivial = trivial.reshape(*trivial.shape[:-1], channels, 2)
    active = active.reshape(*active.shape[:-2], channels, 2, 3)
    vector, dual_bivector = active.unbind(dim=-2)
    return torch.stack(
        (
            trivial[..., 0],
            vector[..., 0],
            vector[..., 1],
            vector[..., 2],
            dual_bivector[..., 2],
            -dual_bivector[..., 1],
            dual_bivector[..., 0],
            trivial[..., 1],
        ),
        dim=-1,
    )


def spin3_invariant_features(multivector: torch.Tensor) -> torch.Tensor:
    """Smooth complete degree-at-most-two per-channel invariant features."""

    if multivector.shape[-1] != GA_DIM:
        raise ValueError(f"multivectors must end in {GA_DIM} components")
    vector = multivector[..., 1:4]
    dual_bivector = torch.stack(
        (multivector[..., 6], -multivector[..., 5], multivector[..., 4]),
        dim=-1,
    )
    return torch.stack(
        (
            multivector[..., 0],
            multivector[..., 7],
            vector.square().sum(dim=-1),
            dual_bivector.square().sum(dim=-1),
            (vector * dual_bivector).sum(dim=-1),
        ),
        dim=-1,
    )


def grade_invariants(multivector: torch.Tensor) -> torch.Tensor:
    """Compatibility view of the former four invariant features."""

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


def bounded_multivector(multivector: torch.Tensor) -> torch.Tensor:
    """Smoothly squash each channel to coefficient norm strictly below one."""

    norm_squared = multivector.square().sum(dim=-1, keepdim=True)
    return multivector * torch.rsqrt(1 + norm_squared)


class Spin3IsotypicLinear(nn.Module):
    """Complete real linear commutant of Cl(3) rotor conjugation."""

    def __init__(
        self, in_channels: int, out_channels: int, use_bias: bool = True
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError("channel counts must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.trivial_kernel = nn.Parameter(torch.empty(out_channels, 2, in_channels, 2))
        self.active_kernel = nn.Parameter(torch.empty(out_channels, 2, in_channels, 2))
        nn.init.kaiming_uniform_(self.trivial_kernel, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.active_kernel, a=math.sqrt(5))
        self.trivial_bias = (
            nn.Parameter(torch.zeros(out_channels, 2)) if use_bias else None
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-2:] != (self.in_channels, GA_DIM):
            raise ValueError("unexpected Spin3IsotypicLinear input shape")
        trivial, active = pack_spin3_isotypic(inputs)
        trivial = trivial.reshape(*trivial.shape[:-1], self.in_channels, 2)
        active = active.reshape(*active.shape[:-2], self.in_channels, 2, 3)
        trivial_output = torch.einsum("ocid,...id->...oc", self.trivial_kernel, trivial)
        active_output = torch.einsum("ocid,...idk->...ock", self.active_kernel, active)
        if self.trivial_bias is not None:
            trivial_output = trivial_output + self.trivial_bias
        return unpack_spin3_isotypic(
            trivial_output.flatten(-2), active_output.flatten(-3, -2)
        )


GradeLinear = Spin3IsotypicLinear


class GeometricRMSNorm(nn.Module):
    """Per-channel coefficient RMS norm with an equivariant scalar gain."""

    def __init__(self, channels: int, epsilon: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.gain = nn.Parameter(torch.ones(channels, 1))
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-2:] != (self.channels, GA_DIM):
            raise ValueError("unexpected multivector channel shape")
        rms = inputs.square().mean(dim=-1, keepdim=True)
        return inputs * torch.rsqrt(rms + self.epsilon) * self.gain


class GeometricDropout(nn.Module):
    """One dropout mask per multivector, shared across all blade coordinates."""

    def __init__(self, probability: float):
        super().__init__()
        if not 0 <= probability < 1:
            raise ValueError("dropout probability must lie in [0, 1)")
        self.probability = probability

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0:
            return inputs
        keep = 1 - self.probability
        mask = torch.empty_like(inputs[..., :1]).bernoulli_(keep).div_(keep)
        return inputs * mask


class GeometricGatedFFN(nn.Module):
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden_channels = channels * expansion
        self.hidden_channels = hidden_channels
        self.input = Spin3IsotypicLinear(channels, hidden_channels)
        self.gate = nn.Linear(hidden_channels * INVARIANT_FEATURES, hidden_channels)
        self.output = Spin3IsotypicLinear(hidden_channels, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input(inputs)
        invariants = spin3_invariant_features(hidden).flatten(-2)
        gates = torch.sigmoid(self.gate(invariants)).unsqueeze(-1)
        return self.output(hidden * gates)


def _validate_transition_shapes(
    decay: torch.Tensor, rotors: torch.Tensor, drive: torch.Tensor
) -> None:
    if rotors.shape != drive.shape or rotors.shape[:-1] != decay.shape:
        raise ValueError("decay, rotors, and drive have incompatible shapes")
    if rotors.ndim != 4 or rotors.shape[-1] != GA_DIM:
        raise ValueError("transitions must have shapes (B,L,C) and (B,L,C,8)")
    if decay.shape[1] == 0:
        raise ValueError("cannot scan an empty sequence")


def compose_transitions(
    later: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    earlier: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose ``later(earlier(h))`` in chronological order."""

    later_decay, later_rotor, later_drive = later
    earlier_decay, earlier_rotor, earlier_drive = earlier
    return (
        later_decay * earlier_decay,
        rotor_product(later_rotor, earlier_rotor),
        later_drive
        + later_decay.unsqueeze(-1) * rotor_sandwich(later_rotor, earlier_drive),
    )


def _rotor_scale(rotor: torch.Tensor) -> torch.Tensor:
    """Rotor sandwich scaling factor on trivial Cl(3,0) components."""

    if rotor.shape[-1] != GA_DIM:
        raise ValueError("rotors must end in GA_DIM components")
    quaternion_vector = _quaternion_vector(rotor)
    return rotor[..., :1].square() + quaternion_vector.square().sum(
        dim=-1, keepdim=True
    )


def _rotor_linear_map(rotor: torch.Tensor) -> torch.Tensor:
    """Build the 3x3 real-linear map associated with rotor conjugation."""

    if rotor.shape[-1] != GA_DIM:
        raise ValueError("rotors must end in GA_DIM components")
    scalar = rotor[..., 0]
    vector = _quaternion_vector(rotor)
    x, y, z = vector.unbind(dim=-1)
    xx = x.square()
    yy = y.square()
    zz = z.square()
    sx = scalar.square()

    row0 = torch.stack(
        (
            sx + xx - yy - zz,
            2.0 * (x * y - z * scalar),
            2.0 * (x * z + y * scalar),
        ),
        dim=-1,
    )
    row1 = torch.stack(
        (
            2.0 * (x * y + z * scalar),
            sx - x.square() + y.square() - z.square(),
            2.0 * (y * z - x * scalar),
        ),
        dim=-1,
    )
    row2 = torch.stack(
        (
            2.0 * (x * z - y * scalar),
            2.0 * (y * z + x * scalar),
            sx - xx - yy + z.square(),
        ),
        dim=-1,
    )
    # ``row0`` through ``row2`` are output-coordinate rows.  Keep that
    # conventional ``R[i, j]`` layout: ``_apply_schur_rotation`` and scan
    # composition apply maps as column actions, so a final-axis stack here
    # would silently transpose every rotor action.
    return torch.stack((row0, row1, row2), dim=-2)


def _apply_schur_rotation(
    rotation: torch.Tensor, vectors: torch.Tensor
) -> torch.Tensor:
    """Apply a per-step 3x3 map to stacked isotypic vectors."""

    if rotation.ndim != 5 or rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation must have shape (B,L,C,3,3)")
    if vectors.ndim not in {3, 4}:
        raise ValueError("vectors must be (B,2*C,3) or (B,L,M*C,3)")
    batch = rotation.shape[0]
    if vectors.shape[0] != batch:
        raise ValueError("vectors must match batch size")
    channels = rotation.shape[2]
    length = rotation.shape[1]
    if vectors.shape[-1] != 3:
        raise ValueError("vectors must end in three active coordinates")
    if vectors.ndim == 3:
        if vectors.shape[1] != channels * 2:
            raise ValueError("vector multiplicity must equal two per channel")
        vectors = vectors[:, None, :, :].expand(
            batch, rotation.shape[1], vectors.shape[1], vectors.shape[2]
        )
    elif vectors.shape[:2] != rotation.shape[:2]:
        raise ValueError(
            "vectors must match (batch, length) for this scan representation"
        )
    if vectors.shape[2] % channels != 0:
        raise ValueError("vector multiplicity must be channel-compatible")

    multiplicity = vectors.shape[2] // channels
    vectors = vectors.reshape(
        batch, rotation.shape[1], channels, multiplicity, vectors.shape[-1]
    )
    rotation_matrix = rotation.reshape(batch * length * channels, 3, 3)
    vectors = vectors.reshape(batch * length * channels, multiplicity, 3)
    return (
        torch.matmul(rotation_matrix, vectors.transpose(-2, -1))
        .transpose(-2, -1)
        .reshape(batch, length, channels, multiplicity, 3)
        .reshape(batch, rotation.shape[1], channels * multiplicity, 3)
    )


def compose_schur_affine_transitions(
    later: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    earlier: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Compose Schur-factored affine transitions in chronological order."""

    (
        later_trivial_scale,
        later_active_scale,
        later_rotation,
        later_trivial_drive,
        later_active_drive,
    ) = later
    (
        earlier_trivial_scale,
        earlier_active_scale,
        earlier_rotation,
        earlier_trivial_drive,
        earlier_active_drive,
    ) = earlier

    return (
        later_trivial_scale * earlier_trivial_scale,
        later_active_scale * earlier_active_scale,
        torch.matmul(later_rotation, earlier_rotation),
        later_trivial_drive + later_trivial_scale * earlier_trivial_drive,
        later_active_drive
        + later_active_scale[..., None]
        * _apply_schur_rotation(later_rotation, earlier_active_drive),
    )


def _to_schur_scan_state(
    decay: torch.Tensor, rotors: torch.Tensor, drive: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert affine transitions to Schur-factored scan representation."""

    trivial_drive, active_drive = pack_spin3_isotypic(drive)
    rotor_scale = _rotor_scale(rotors)
    scaled_decay = decay[..., None] * rotor_scale
    trivial_scale = torch.stack((scaled_decay, scaled_decay), dim=-1).reshape(
        *decay.shape[:2], -1
    )
    active_scale = torch.stack((decay, decay), dim=-1).reshape(*decay.shape[:2], -1)
    rotation = _rotor_linear_map(rotors)
    return trivial_scale, active_scale, rotation, trivial_drive, active_drive


def rotor_affine_scan_schur(
    decay: torch.Tensor,
    rotors: torch.Tensor,
    drive: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hillis--Steele scan using Schur-factored isotypic coordinates."""

    _validate_transition_shapes(decay, rotors, drive)
    cumulative = _to_schur_scan_state(decay, rotors, drive)
    offset = 1
    while offset < decay.shape[1]:
        earlier = tuple(value[:, :-offset] for value in cumulative)
        later = tuple(value[:, offset:] for value in cumulative)
        composed = compose_schur_affine_transitions(later, earlier)
        cumulative = tuple(
            torch.cat((value[:, :offset], update), dim=1)
            for value, update in zip(cumulative, composed)
        )
        offset *= 2
    (
        cumulative_trivial_scale,
        cumulative_active_scale,
        cumulative_rotation,
        cumulative_trivial_drive,
        cumulative_active_drive,
    ) = cumulative
    if initial_state is None:
        states = unpack_spin3_isotypic(
            cumulative_trivial_drive, cumulative_active_drive
        )
    else:
        if initial_state.shape != drive.shape[:1] + drive.shape[2:]:
            raise ValueError("initial_state must have shape (B,C,8)")
        initial_trivial, initial_active = pack_spin3_isotypic(initial_state)
        states = unpack_spin3_isotypic(
            cumulative_trivial_scale * initial_trivial[:, None]
            + cumulative_trivial_drive,
            cumulative_active_scale[..., None]
            * _apply_schur_rotation(cumulative_rotation, initial_active)
            + cumulative_active_drive,
        )
    return states, states[:, -1]


def rotor_transition_step(
    state: torch.Tensor,
    decay: torch.Tensor,
    rotor: torch.Tensor,
    drive: torch.Tensor,
) -> torch.Tensor:
    """Apply one damped-rotor affine transition."""

    return decay.unsqueeze(-1) * rotor_sandwich(rotor, state) + drive


def rotor_affine_scan(
    decay: torch.Tensor,
    rotors: torch.Tensor,
    drive: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable vectorized Hillis--Steele prefix scan.

    This avoids a Python loop over tokens on CUDA. It has logarithmic launch
    depth and O(L log L) work; the recurrent path remains the semantic oracle.
    """

    _validate_transition_shapes(decay, rotors, drive)
    cumulative = (decay, rotors, drive)
    offset = 1
    while offset < decay.shape[1]:
        earlier = tuple(value[:, :-offset] for value in cumulative)
        later = tuple(value[:, offset:] for value in cumulative)
        combined = compose_transitions(later, earlier)
        cumulative = tuple(
            torch.cat((value[:, :offset], update), dim=1)
            for value, update in zip(cumulative, combined)
        )
        offset *= 2
    cumulative_decay, cumulative_rotor, cumulative_drive = cumulative
    if initial_state is None:
        states = cumulative_drive
    else:
        if initial_state.shape != drive.shape[:1] + drive.shape[2:]:
            raise ValueError("initial_state must have shape (B,C,8)")
        states = (
            cumulative_decay.unsqueeze(-1)
            * rotor_sandwich(cumulative_rotor, initial_state[:, None])
            + cumulative_drive
        )
    return states, states[:, -1]


def rotor_recurrent_scan(
    decay: torch.Tensor,
    rotors: torch.Tensor,
    drive: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_transition_shapes(decay, rotors, drive)
    state = torch.zeros_like(drive[:, 0]) if initial_state is None else initial_state
    if state.shape != drive.shape[:1] + drive.shape[2:]:
        raise ValueError("initial_state must have shape (B,C,8)")
    states = []
    for position in range(decay.shape[1]):
        state = rotor_transition_step(
            state,
            decay[:, position],
            rotors[:, position],
            drive[:, position],
        )
        states.append(state)
    return torch.stack(states, dim=1), state


class SelectiveRotorSSM(nn.Module):
    """Complete-isotypic, input-selective, uniformly bounded rotor SSM."""

    def __init__(
        self,
        channels: int,
        min_half_life: float = 4.0,
        max_half_life: float = 2048.0,
        minimum_step_size: float = 1e-2,
        minimum_decay_rate: float = 1e-4,
        max_rotor_angle: float = math.pi,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if min_half_life <= 0 or max_half_life < min_half_life:
            raise ValueError("half-life bounds must be positive and ordered")
        if minimum_step_size <= 0 or minimum_decay_rate <= 0:
            raise ValueError("step-size and decay-rate floors must be positive")
        if not math.isfinite(max_rotor_angle) or max_rotor_angle < 0:
            raise ValueError("max_rotor_angle must be finite and nonnegative")
        self.channels = channels
        self.minimum_step_size = minimum_step_size
        self.minimum_decay_rate = minimum_decay_rate
        self.max_rotor_angle = max_rotor_angle

        controls = channels * INVARIANT_FEATURES
        self.step_control = nn.Linear(controls, channels)
        self.write_control = nn.Linear(controls, channels)
        self.rotor_control = nn.Linear(controls, channels)
        self.rotor_source = Spin3IsotypicLinear(channels, channels, use_bias=False)
        self.input_projection = Spin3IsotypicLinear(channels, channels)
        for controller in (
            self.step_control,
            self.write_control,
            self.rotor_control,
        ):
            nn.init.zeros_(controller.weight)
            nn.init.zeros_(controller.bias)

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
        self, inputs: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[-2:] != (self.channels, GA_DIM):
            raise ValueError(
                f"inputs must have shape (batch,length,{self.channels},{GA_DIM})"
            )
        invariants = spin3_invariant_features(inputs).flatten(-2)
        step_size = self.minimum_step_size + F.softplus(self.step_control(invariants))
        rates = self.minimum_decay_rate + F.softplus(self.log_rates)
        decay = torch.exp(-step_size * rates)

        rotor_strength = torch.tanh(self.rotor_control(invariants))
        rotor_source = self.rotor_source(inputs)[..., 4:7]
        rotors = rotor_from_bivector(
            rotor_source * rotor_strength.unsqueeze(-1), self.max_rotor_angle
        )

        write = torch.sigmoid(self.write_control(invariants))
        candidate = bounded_multivector(self.input_projection(inputs))
        drive = (1 - decay).unsqueeze(-1) * write.unsqueeze(-1) * candidate

        if valid_mask is not None:
            if valid_mask.shape != inputs.shape[:2]:
                raise ValueError("valid_mask must have shape (batch,length)")
            valid = valid_mask.bool()
            decay = torch.where(valid[..., None], decay, torch.ones_like(decay))
            rotors = torch.where(valid[..., None, None], rotors, identity_rotor(rotors))
            drive = torch.where(valid[..., None, None], drive, torch.zeros_like(drive))
        return decay, rotors, drive

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        scan_mode: str = "parallel",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decay, rotors, drive = self.transitions(inputs, valid_mask)
        if scan_mode == "parallel":
            return rotor_affine_scan(decay, rotors, drive, initial_state)
        if scan_mode == "schur_parallel":
            return rotor_affine_scan_schur(decay, rotors, drive, initial_state)
        if scan_mode == "recurrent":
            return rotor_recurrent_scan(decay, rotors, drive, initial_state)
        raise ValueError(
            "scan_mode must be 'parallel', 'schur_parallel', or 'recurrent'"
        )


class GASSMBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        dropout_rate: float = 0.1,
        max_rotor_angle: float = math.pi,
    ) -> None:
        super().__init__()
        self.norm1 = GeometricRMSNorm(channels)
        self.ssm = SelectiveRotorSSM(channels, max_rotor_angle=max_rotor_angle)
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, final_state = self.ssm(
            self.norm1(inputs),
            initial_state,
            valid_mask=valid_mask,
            scan_mode=scan_mode,
        )
        outputs = inputs + self.dropout(sequence)
        outputs = outputs + self.dropout(self.ffn(self.norm2(outputs)))
        return outputs, final_state


class GASSMLanguageModel(nn.Module):
    """Pure causal language model with fixed Cl(3,0) recurrent state."""

    model_version = __version__

    def __init__(
        self,
        vocab_size: int,
        channels: int = 8,
        num_layers: int = 4,
        expansion: int = 2,
        max_len: int = 512,
        dropout_rate: float = 0.1,
        max_rotor_angle: float = math.pi,
    ) -> None:
        super().__init__()
        if vocab_size < 2 or channels < 1 or num_layers < 1:
            raise ValueError("vocab_size, channels, and num_layers are invalid")
        self.vocab_size = vocab_size
        self.channels = channels
        self.num_layers = num_layers
        # Retained only for construction compatibility. The recurrent model
        # has no positional table and therefore no intrinsic length ceiling.
        self.max_len = max_len
        self.token_embeddings = nn.Parameter(torch.empty(vocab_size, channels, GA_DIM))
        nn.init.normal_(self.token_embeddings, std=0.02)
        self.blocks = nn.ModuleList(
            GASSMBlock(channels, expansion, dropout_rate, max_rotor_angle)
            for _ in range(num_layers)
        )
        self.final_norm = GeometricRMSNorm(channels)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.embedding_dropout = GeometricDropout(dropout_rate)

    @property
    def vocabulary_bias(self) -> torch.Tensor:
        """Compatibility alias for pre-pure checkpoints and callers."""

        return self.output_bias

    @property
    def recurrent_state_scalars(self) -> int:
        return self.num_layers * self.channels * GA_DIM

    def initial_states(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        reference = self.token_embeddings
        return tuple(
            torch.zeros(
                batch_size,
                self.channels,
                GA_DIM,
                device=device or reference.device,
                dtype=dtype or reference.dtype,
            )
            for _ in range(self.num_layers)
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        recurrent_states: Sequence[torch.Tensor] | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        return_recurrent_states: bool = False,
        scan_mode: str = "parallel",
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if token_ids.ndim != 2 or token_ids.shape[1] == 0:
            raise ValueError("token_ids must have nonempty shape (batch,length)")
        if attention_mask is not None and attention_mask.shape != token_ids.shape:
            raise ValueError("attention_mask must match token_ids")
        if recurrent_states is None:
            recurrent_states = (None,) * self.num_layers
        if len(recurrent_states) != self.num_layers:
            raise ValueError("one recurrent state is required per model layer")

        outputs = self.embedding_dropout(self.token_embeddings[token_ids])
        final_states = []
        for block, initial_state in zip(self.blocks, recurrent_states):
            outputs, final_state = block(
                outputs,
                initial_state,
                valid_mask=attention_mask,
                scan_mode=scan_mode,
            )
            final_states.append(final_state)
        outputs = self.final_norm(outputs)
        logits = torch.einsum(
            "blci,vci->blv", outputs, self.token_embeddings
        ) / math.sqrt(self.channels * GA_DIM)
        logits = logits + self.output_bias
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
        logits, states = self(
            token_ids[:, None],
            recurrent_states,
            return_recurrent_states=True,
            scan_mode="recurrent",
        )
        return logits[:, 0], states


__all__ = [
    "BASIS_MASKS",
    "GA_DIM",
    "INVARIANT_FEATURES",
    "MULTIPLICATION_TABLE",
    "GASSMBlock",
    "GASSMLanguageModel",
    "GeometricDropout",
    "GeometricGatedFFN",
    "GeometricRMSNorm",
    "GradeLinear",
    "SelectiveRotorSSM",
    "Spin3IsotypicLinear",
    "__version__",
    "bounded_multivector",
    "compose_schur_affine_transitions",
    "compose_transitions",
    "geometric_product",
    "grade_invariants",
    "identity_rotor",
    "normalized_rotor",
    "pack_spin3_isotypic",
    "reversion",
    "rotor_affine_scan",
    "rotor_affine_scan_schur",
    "rotor_from_bivector",
    "rotor_product",
    "rotor_recurrent_scan",
    "rotor_sandwich",
    "rotor_transition_step",
    "specialized_rotor_sandwich",
    "spin3_invariant_features",
    "unpack_spin3_isotypic",
]
