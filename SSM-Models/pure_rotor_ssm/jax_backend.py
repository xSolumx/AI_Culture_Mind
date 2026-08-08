"""Pure JAX/Flax implementation of the selective Cl(3,0) rotor SSM.

The recurrent transition is

    h_t = d_t Ad(q_t) h_(t-1) + (1-d_t) w_t z_t,

where ``0 < d_t < 1``, ``0 < w_t < 1``, ``q_t`` is a unit rotor, and
``||z_t|| < 1``.  Rotor conjugation is orthogonal, so every valid step obeys

    ||h_t|| <= d_t ||h_(t-1)|| + (1-d_t),

and therefore ``||h_t|| <= max(||h_0||, 1)`` in exact arithmetic.  This file
contains only algebra and model code; data, optimization, checkpoints, and
experiment reporting live outside the pure package.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from . import __version__

GA_DIM = 8
INVARIANT_FEATURES = 5
BASIS_MASKS = (0, 1, 2, 4, 3, 5, 6, 7)
REVERSION_SIGNS = jnp.asarray([1, 1, 1, 1, -1, -1, -1, -1])


def _multiplication_table() -> jax.Array:
    table = np.zeros((GA_DIM, GA_DIM, GA_DIM), dtype=np.int8)
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
    return jnp.asarray(table)


MULTIPLICATION_TABLE = _multiplication_table()


def geometric_product(left: jax.Array, right: jax.Array) -> jax.Array:
    """Broadcasted Euclidean Cl(3,0) product on the final axis."""

    if left.shape[-1] != GA_DIM or right.shape[-1] != GA_DIM:
        raise ValueError(f"multivectors must end in {GA_DIM} components")
    dtype = jnp.result_type(left, right)
    return jnp.einsum(
        "...i,...j,kij->...k",
        left,
        right,
        MULTIPLICATION_TABLE.astype(dtype),
    )


def reversion(multivector: jax.Array) -> jax.Array:
    if multivector.shape[-1] != GA_DIM:
        raise ValueError(f"multivectors must end in {GA_DIM} components")
    return multivector * REVERSION_SIGNS.astype(multivector.dtype)


def identity_rotor(reference: jax.Array) -> jax.Array:
    return jnp.zeros_like(reference).at[..., 0].set(1)


def normalized_rotor(parameters: jax.Array) -> jax.Array:
    """Embed normalized ``[scalar,e12,e13,e23]`` parameters in Cl(3,0)."""

    if parameters.shape[-1] != 4:
        raise ValueError("rotor parameters must end in four components")
    norm = jnp.linalg.norm(parameters, axis=-1, keepdims=True)
    normalized = parameters / jnp.maximum(norm, jnp.asarray(1e-6, parameters.dtype))
    fallback = jnp.zeros_like(parameters).at[..., 0].set(1)
    parameters = jnp.where(norm > 1e-6, normalized, fallback)
    scalar, e12, e13, e23 = jnp.moveaxis(parameters, -1, 0)
    zeros = jnp.zeros_like(scalar)
    return jnp.stack(
        (scalar, zeros, zeros, zeros, e12, e13, e23, zeros), axis=-1
    )


def rotor_from_bivector(
    bivector: jax.Array, max_angle: float = math.pi
) -> jax.Array:
    """Smooth bounded exponential chart with analytic identity derivative."""

    if bivector.shape[-1] != 3:
        raise ValueError("bivectors must end in three components")
    if not math.isfinite(max_angle) or max_angle < 0:
        raise ValueError("max_angle must be finite and nonnegative")
    angle_limit = jnp.asarray(max_angle, bivector.dtype)
    magnitude_squared = jnp.sum(jnp.square(bivector), axis=-1, keepdims=True)
    threshold = jnp.asarray(jnp.finfo(bivector.dtype).eps, bivector.dtype)
    safe_magnitude = jnp.sqrt(jnp.maximum(magnitude_squared, threshold))
    regular_angle = angle_limit * jnp.tanh(safe_magnitude)
    regular_scalar = jnp.cos(regular_angle / 2)
    regular_scale = jnp.sin(regular_angle / 2) / safe_magnitude
    small_scalar = 1 - jnp.square(angle_limit) * magnitude_squared / 8
    small_scale = angle_limit / 2 - (
        angle_limit / 6 + angle_limit**3 / 48
    ) * magnitude_squared
    use_regular = magnitude_squared > threshold
    parameters = jnp.concatenate(
        (
            jnp.where(use_regular, regular_scalar, small_scalar),
            -jnp.where(use_regular, regular_scale, small_scale) * bivector,
        ),
        axis=-1,
    )
    return normalized_rotor(parameters)


def _quaternion_vector(rotor: jax.Array) -> jax.Array:
    return jnp.stack((-rotor[..., 6], rotor[..., 5], -rotor[..., 4]), axis=-1)


def rotor_product(left: jax.Array, right: jax.Array) -> jax.Array:
    """Specialized product of even rotors in eight-coefficient storage."""

    if left.shape[-1] != GA_DIM or right.shape[-1] != GA_DIM:
        raise ValueError(f"rotors must end in {GA_DIM} components")
    left_scalar = left[..., :1]
    right_scalar = right[..., :1]
    left_vector = _quaternion_vector(left)
    right_vector = _quaternion_vector(right)
    scalar = left_scalar * right_scalar - jnp.sum(
        left_vector * right_vector, axis=-1, keepdims=True
    )
    vector = (
        left_scalar * right_vector
        + right_scalar * left_vector
        + jnp.cross(left_vector, right_vector)
    )
    zeros = jnp.zeros_like(scalar)
    return jnp.concatenate(
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
        axis=-1,
    )


def rotor_sandwich(rotor: jax.Array, multivector: jax.Array) -> jax.Array:
    """Optimized full-Cl(3) rotor conjugation via two 3D vector actions."""

    if rotor.shape[-1] != GA_DIM or multivector.shape[-1] != GA_DIM:
        raise ValueError(f"rotors and multivectors must end in {GA_DIM} components")
    scalar = rotor[..., :1]
    quaternion_vector = _quaternion_vector(rotor)
    vector_square = jnp.sum(jnp.square(quaternion_vector), axis=-1, keepdims=True)
    norm_squared = jnp.square(scalar) + vector_square

    def rotate(vectors: jax.Array) -> jax.Array:
        dot = jnp.sum(quaternion_vector * vectors, axis=-1, keepdims=True)
        return (
            (jnp.square(scalar) - vector_square) * vectors
            + 2 * quaternion_vector * dot
            + 2 * scalar * jnp.cross(quaternion_vector, vectors)
        )

    vector = rotate(multivector[..., 1:4])
    dual_bivector = jnp.stack(
        (multivector[..., 6], -multivector[..., 5], multivector[..., 4]),
        axis=-1,
    )
    rotated_dual = rotate(dual_bivector)
    return jnp.concatenate(
        (
            norm_squared * multivector[..., :1],
            vector,
            rotated_dual[..., 2:3],
            -rotated_dual[..., 1:2],
            rotated_dual[..., 0:1],
            norm_squared * multivector[..., 7:8],
        ),
        axis=-1,
    )


def pack_spin3_isotypic(
    multivector: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    if multivector.ndim < 2 or multivector.shape[-1] != GA_DIM:
        raise ValueError("multivectors must have shape (..., channels, 8)")
    channels = multivector.shape[-2]
    trivial = jnp.stack(
        (multivector[..., 0], multivector[..., 7]), axis=-1
    ).reshape(*multivector.shape[:-2], 2 * channels)
    dual_bivector = jnp.stack(
        (multivector[..., 6], -multivector[..., 5], multivector[..., 4]),
        axis=-1,
    )
    active = jnp.stack((multivector[..., 1:4], dual_bivector), axis=-2)
    return trivial, active.reshape(*multivector.shape[:-2], 2 * channels, 3)


def unpack_spin3_isotypic(
    trivial: jax.Array, active: jax.Array
) -> jax.Array:
    if trivial.shape[:-1] != active.shape[:-2] or active.shape[-1] != 3:
        raise ValueError("trivial and active isotypic shapes are incompatible")
    if trivial.shape[-1] != active.shape[-2] or trivial.shape[-1] % 2:
        raise ValueError("isotypic multiplicities must agree and be even")
    channels = trivial.shape[-1] // 2
    trivial = trivial.reshape(*trivial.shape[:-1], channels, 2)
    active = active.reshape(*active.shape[:-2], channels, 2, 3)
    vector, dual_bivector = active[..., 0, :], active[..., 1, :]
    output = jnp.zeros((*trivial.shape[:-1], GA_DIM), dtype=trivial.dtype)
    output = output.at[..., 0].set(trivial[..., 0])
    output = output.at[..., 1:4].set(vector)
    output = output.at[..., 4].set(dual_bivector[..., 2])
    output = output.at[..., 5].set(-dual_bivector[..., 1])
    output = output.at[..., 6].set(dual_bivector[..., 0])
    output = output.at[..., 7].set(trivial[..., 1])
    return output


def spin3_invariant_features(multivector: jax.Array) -> jax.Array:
    """Smooth degree-at-most-two per-channel invariant features."""

    if multivector.shape[-1] != GA_DIM:
        raise ValueError(f"multivectors must end in {GA_DIM} components")
    vector = multivector[..., 1:4]
    dual_bivector = jnp.stack(
        (multivector[..., 6], -multivector[..., 5], multivector[..., 4]),
        axis=-1,
    )
    return jnp.stack(
        (
            multivector[..., 0],
            multivector[..., 7],
            jnp.sum(jnp.square(vector), axis=-1),
            jnp.sum(jnp.square(dual_bivector), axis=-1),
            jnp.sum(vector * dual_bivector, axis=-1),
        ),
        axis=-1,
    )


def grade_invariants(multivector: jax.Array) -> jax.Array:
    """Compatibility view of the former four invariant features."""

    if multivector.shape[-1] != GA_DIM:
        raise ValueError(f"multivectors must end in {GA_DIM} components")
    return jnp.stack(
        (
            multivector[..., 0],
            jnp.linalg.norm(multivector[..., 1:4], axis=-1),
            jnp.linalg.norm(multivector[..., 4:7], axis=-1),
            multivector[..., 7],
        ),
        axis=-1,
    )


def bounded_multivector(multivector: jax.Array) -> jax.Array:
    """Smoothly squash each channel to coefficient norm strictly below one."""

    norm_squared = jnp.sum(jnp.square(multivector), axis=-1, keepdims=True)
    return multivector * jax.lax.rsqrt(1 + norm_squared)


class Spin3IsotypicLinear(nn.Module):
    """Complete real linear commutant of Cl(3) rotor conjugation."""

    in_channels: int
    out_channels: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        if inputs.shape[-2:] != (self.in_channels, GA_DIM):
            raise ValueError("unexpected Spin3IsotypicLinear input shape")
        trivial, active = pack_spin3_isotypic(inputs)
        trivial = trivial.reshape(*trivial.shape[:-1], self.in_channels, 2)
        active = active.reshape(*active.shape[:-2], self.in_channels, 2, 3)
        trivial_kernel = self.param(
            "trivial_kernel",
            nn.initializers.lecun_normal(),
            (self.out_channels, 2, self.in_channels, 2),
            self.param_dtype,
        ).astype(self.dtype)
        active_kernel = self.param(
            "active_kernel",
            nn.initializers.lecun_normal(),
            (self.out_channels, 2, self.in_channels, 2),
            self.param_dtype,
        ).astype(self.dtype)
        trivial_output = jnp.einsum(
            "ocid,...id->...oc", trivial_kernel, trivial.astype(self.dtype)
        )
        active_output = jnp.einsum(
            "ocid,...idk->...ock", active_kernel, active.astype(self.dtype)
        )
        if self.use_bias:
            bias = self.param(
                "trivial_bias",
                nn.initializers.zeros,
                (self.out_channels, 2),
                self.param_dtype,
            ).astype(self.dtype)
            trivial_output = trivial_output + bias
        return unpack_spin3_isotypic(
            trivial_output.reshape(*trivial_output.shape[:-2], -1),
            active_output.reshape(*active_output.shape[:-3], -1, 3),
        )


GradeLinear = Spin3IsotypicLinear


class GeometricRMSNorm(nn.Module):
    """Per-channel coefficient RMS norm with an equivariant scalar gain."""

    channels: int
    epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        if inputs.shape[-2:] != (self.channels, GA_DIM):
            raise ValueError("unexpected multivector channel shape")
        gain = self.param(
            "gain", nn.initializers.ones, (self.channels, 1), self.param_dtype
        ).astype(self.dtype)
        rms = jnp.mean(jnp.square(inputs.astype(self.dtype)), axis=-1, keepdims=True)
        return inputs.astype(self.dtype) * jax.lax.rsqrt(rms + self.epsilon) * gain


class GeometricDropout(nn.Module):
    """One dropout mask per multivector, shared across blade coordinates."""

    rate: float

    @nn.compact
    def __call__(self, inputs: jax.Array, *, deterministic: bool) -> jax.Array:
        if not 0 <= self.rate < 1:
            raise ValueError("dropout rate must lie in [0, 1)")
        return nn.Dropout(self.rate, broadcast_dims=(-1,))(
            inputs, deterministic=deterministic
        )


class GeometricGatedFFN(nn.Module):
    channels: int
    expansion: int = 2
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        hidden_channels = self.channels * self.expansion
        hidden = Spin3IsotypicLinear(
            self.channels, hidden_channels, dtype=self.dtype
        )(inputs)
        invariants = spin3_invariant_features(hidden).reshape(
            *hidden.shape[:-2], hidden_channels * INVARIANT_FEATURES
        )
        gates = nn.sigmoid(
            nn.Dense(hidden_channels, dtype=self.dtype, param_dtype=jnp.float32)(
                invariants
            )
        )[..., None]
        return Spin3IsotypicLinear(hidden_channels, self.channels, dtype=self.dtype)(
            hidden * gates
        )


def _validate_transition_shapes(
    decay: jax.Array, rotors: jax.Array, drive: jax.Array
) -> None:
    if rotors.shape != drive.shape or rotors.shape[:-1] != decay.shape:
        raise ValueError("decay, rotors, and drive have incompatible shapes")
    if rotors.ndim != 4 or rotors.shape[-1] != GA_DIM:
        raise ValueError("transitions must have shapes (B,L,C) and (B,L,C,8)")
    if decay.shape[1] == 0:
        raise ValueError("cannot scan an empty sequence")


def compose_transitions(
    earlier: tuple[jax.Array, jax.Array, jax.Array],
    later: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compose ``later(earlier(h))`` for JAX's left-to-right scan contract."""

    earlier_decay, earlier_rotor, earlier_drive = earlier
    later_decay, later_rotor, later_drive = later
    return (
        later_decay * earlier_decay,
        rotor_product(later_rotor, earlier_rotor),
        later_drive
        + later_decay[..., None]
        * rotor_sandwich(later_rotor, earlier_drive),
    )


def rotor_transition_step(
    state: jax.Array,
    decay: jax.Array,
    rotor: jax.Array,
    drive: jax.Array,
) -> jax.Array:
    """Apply one damped-rotor affine transition."""

    return decay[..., None] * rotor_sandwich(rotor, state) + drive


def rotor_affine_scan(
    decay: jax.Array,
    rotors: jax.Array,
    drive: jax.Array,
    initial_state: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Inclusive parallel prefix scan of affine rotor transitions."""

    _validate_transition_shapes(decay, rotors, drive)
    cumulative_decay, cumulative_rotor, cumulative_drive = jax.lax.associative_scan(
        compose_transitions, (decay, rotors, drive), axis=1
    )
    if initial_state is None:
        states = cumulative_drive
    else:
        if initial_state.shape != drive.shape[:1] + drive.shape[2:]:
            raise ValueError("initial_state must have shape (B,C,8)")
        states = (
            cumulative_decay[..., None]
            * rotor_sandwich(cumulative_rotor, initial_state[:, None])
            + cumulative_drive
        )
    return states, states[:, -1]


def rotor_recurrent_scan(
    decay: jax.Array,
    rotors: jax.Array,
    drive: jax.Array,
    initial_state: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Sequential scan used as the semantic and streaming oracle."""

    _validate_transition_shapes(decay, rotors, drive)
    state = jnp.zeros_like(drive[:, 0]) if initial_state is None else initial_state
    if state.shape != drive.shape[:1] + drive.shape[2:]:
        raise ValueError("initial_state must have shape (B,C,8)")

    def step(carry, transition):
        step_decay, step_rotor, step_drive = transition
        next_state = rotor_transition_step(
            carry, step_decay, step_rotor, step_drive
        )
        return next_state, next_state

    final_state, time_major = jax.lax.scan(
        step,
        state,
        tuple(jnp.moveaxis(value, 1, 0) for value in (decay, rotors, drive)),
    )
    return jnp.moveaxis(time_major, 0, 1), final_state


class SelectiveRotorSSM(nn.Module):
    """Complete-isotypic, input-selective, uniformly bounded rotor SSM."""

    channels: int
    min_half_life: float = 4.0
    max_half_life: float = 2048.0
    minimum_step_size: float = 1e-2
    minimum_decay_rate: float = 1e-4
    max_rotor_angle: float = math.pi
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        initial_state: jax.Array | None = None,
        *,
        valid_mask: jax.Array | None = None,
        scan_mode: str = "parallel",
    ) -> tuple[jax.Array, jax.Array]:
        if inputs.ndim != 4 or inputs.shape[-2:] != (self.channels, GA_DIM):
            raise ValueError(
                f"inputs must have shape (batch,length,{self.channels},{GA_DIM})"
            )
        if self.channels < 1:
            raise ValueError("channels must be positive")
        if self.min_half_life <= 0 or self.max_half_life < self.min_half_life:
            raise ValueError("half-life bounds must be positive and ordered")
        if self.minimum_step_size <= 0 or self.minimum_decay_rate <= 0:
            raise ValueError("step-size and decay-rate floors must be positive")
        if not math.isfinite(self.max_rotor_angle) or self.max_rotor_angle < 0:
            raise ValueError("max_rotor_angle must be finite and nonnegative")

        invariants = spin3_invariant_features(inputs).reshape(
            *inputs.shape[:-2], self.channels * INVARIANT_FEATURES
        )

        def zero_controller(name: str) -> jax.Array:
            return nn.Dense(
                self.channels,
                dtype=self.dtype,
                param_dtype=jnp.float32,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
                name=name,
            )(invariants)

        step_size = self.minimum_step_size + nn.softplus(
            zero_controller("step_control")
        )

        expected_step = self.minimum_step_size + math.log(2.0)
        slowest_initial_rate = math.log(2.0) / (
            self.max_half_life * expected_step
        )
        if self.minimum_decay_rate >= slowest_initial_rate:
            raise ValueError(
                "minimum_decay_rate is too large for the requested half-lives"
            )

        def rate_initializer(key, shape, dtype):
            del key
            half_lives = jnp.exp(
                jnp.linspace(
                    jnp.log(self.min_half_life),
                    jnp.log(self.max_half_life),
                    shape[0],
                    dtype=dtype,
                )
            )
            target_rates = jnp.log(2.0) / (half_lives * expected_step)
            return jnp.log(jnp.expm1(target_rates - self.minimum_decay_rate))

        log_rates = self.param(
            "log_rates", rate_initializer, (self.channels,), jnp.float32
        )
        rates = self.minimum_decay_rate + nn.softplus(log_rates).astype(self.dtype)
        decay = jnp.exp(-step_size * rates)

        rotor_strength = jnp.tanh(zero_controller("rotor_control"))
        rotor_source = Spin3IsotypicLinear(
            self.channels,
            self.channels,
            use_bias=False,
            dtype=self.dtype,
            name="rotor_source",
        )(inputs)[..., 4:7]
        rotors = rotor_from_bivector(
            rotor_source * rotor_strength[..., None], self.max_rotor_angle
        )

        write = nn.sigmoid(zero_controller("write_control"))
        candidate = bounded_multivector(
            Spin3IsotypicLinear(
                self.channels,
                self.channels,
                dtype=self.dtype,
                name="input_projection",
            )(inputs)
        )
        drive = (1 - decay)[..., None] * write[..., None] * candidate

        if valid_mask is not None:
            if valid_mask.shape != inputs.shape[:2]:
                raise ValueError("valid_mask must have shape (batch,length)")
            valid = valid_mask.astype(bool)
            decay = jnp.where(valid[..., None], decay, jnp.ones_like(decay))
            rotors = jnp.where(
                valid[..., None, None], rotors, identity_rotor(rotors)
            )
            drive = jnp.where(
                valid[..., None, None], drive, jnp.zeros_like(drive)
            )

        if scan_mode == "parallel":
            return rotor_affine_scan(decay, rotors, drive, initial_state)
        if scan_mode == "recurrent":
            return rotor_recurrent_scan(decay, rotors, drive, initial_state)
        raise ValueError("scan_mode must be 'parallel' or 'recurrent'")


class GASSMBlock(nn.Module):
    channels: int
    expansion: int = 2
    dropout_rate: float = 0.1
    max_rotor_angle: float = math.pi
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        initial_state: jax.Array | None = None,
        *,
        valid_mask: jax.Array | None = None,
        training: bool,
        scan_mode: str = "parallel",
    ) -> tuple[jax.Array, jax.Array]:
        sequence, final_state = SelectiveRotorSSM(
            self.channels,
            max_rotor_angle=self.max_rotor_angle,
            dtype=self.dtype,
            name="ssm",
        )(
            GeometricRMSNorm(self.channels, dtype=self.dtype, name="norm1")(
                inputs
            ),
            initial_state,
            valid_mask=valid_mask,
            scan_mode=scan_mode,
        )
        outputs = inputs + GeometricDropout(
            self.dropout_rate, name="state_dropout"
        )(sequence, deterministic=not training)
        feed_forward = GeometricGatedFFN(
            self.channels, self.expansion, self.dtype, name="ffn"
        )(
            GeometricRMSNorm(self.channels, dtype=self.dtype, name="norm2")(
                outputs
            )
        )
        return outputs + GeometricDropout(
            self.dropout_rate, name="ffn_dropout"
        )(feed_forward, deterministic=not training), final_state


class GASSMLanguageModel(nn.Module):
    """Pure causal language model with fixed Cl(3,0) recurrent state."""

    vocab_size: int
    channels: int = 8
    num_layers: int = 4
    expansion: int = 2
    max_len: int = 512  # Compatibility only; recurrence has no table length.
    dropout_rate: float = 0.1
    max_rotor_angle: float = math.pi
    dtype: jnp.dtype = jnp.float32

    @property
    def model_version(self) -> str:
        return __version__

    @nn.compact
    def __call__(
        self,
        token_ids: jax.Array,
        recurrent_states: Sequence[jax.Array] | None = None,
        *,
        attention_mask: jax.Array | None = None,
        training: bool,
        return_recurrent_states: bool = False,
        scan_mode: str = "parallel",
    ) -> jax.Array | tuple[jax.Array, tuple[jax.Array, ...]]:
        if token_ids.ndim != 2 or token_ids.shape[1] == 0:
            raise ValueError("token_ids must have nonempty shape (batch,length)")
        if attention_mask is not None and attention_mask.shape != token_ids.shape:
            raise ValueError("attention_mask must match token_ids")
        if self.vocab_size < 2 or self.channels < 1 or self.num_layers < 1:
            raise ValueError("vocab_size, channels, and num_layers are invalid")
        if recurrent_states is None:
            layer_states: tuple[jax.Array | None, ...] = (None,) * self.num_layers
        else:
            if len(recurrent_states) != self.num_layers:
                raise ValueError("one recurrent state is required per model layer")
            layer_states = tuple(recurrent_states)

        embeddings = self.param(
            "token_embeddings",
            nn.initializers.normal(stddev=0.02),
            (self.vocab_size, self.channels, GA_DIM),
            jnp.float32,
        ).astype(self.dtype)
        outputs = GeometricDropout(
            self.dropout_rate, name="embedding_dropout"
        )(embeddings[token_ids], deterministic=not training)
        final_states = []
        for layer_index in range(self.num_layers):
            outputs, final_state = GASSMBlock(
                self.channels,
                self.expansion,
                self.dropout_rate,
                self.max_rotor_angle,
                self.dtype,
                name=f"block_{layer_index}",
            )(
                outputs,
                layer_states[layer_index],
                valid_mask=attention_mask,
                training=training,
                scan_mode=scan_mode,
            )
            final_states.append(final_state)
        outputs = GeometricRMSNorm(
            self.channels, dtype=self.dtype, name="final_norm"
        )(outputs)
        logits = jnp.einsum("blci,vci->blv", outputs, embeddings)
        logits /= jnp.sqrt(jnp.asarray(self.channels * GA_DIM, self.dtype))
        output_bias = self.param(
            "output_bias", nn.initializers.zeros, (self.vocab_size,), jnp.float32
        )
        logits = (logits + output_bias).astype(jnp.float32)
        if return_recurrent_states:
            return logits, tuple(final_states)
        return logits

    @property
    def recurrent_state_scalars(self) -> int:
        return self.num_layers * self.channels * GA_DIM


def initialize_recurrent_states(
    model: GASSMLanguageModel, batch_size: int
) -> tuple[jax.Array, ...]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return tuple(
        jnp.zeros((batch_size, model.channels, GA_DIM), dtype=model.dtype)
        for _ in range(model.num_layers)
    )


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
    "compose_transitions",
    "geometric_product",
    "grade_invariants",
    "identity_rotor",
    "initialize_recurrent_states",
    "normalized_rotor",
    "pack_spin3_isotypic",
    "reversion",
    "rotor_affine_scan",
    "rotor_from_bivector",
    "rotor_product",
    "rotor_recurrent_scan",
    "rotor_sandwich",
    "rotor_transition_step",
    "spin3_invariant_features",
    "unpack_spin3_isotypic",
]
