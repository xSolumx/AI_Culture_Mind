"""Reusable geometric-algebra primitives for the JAX/Flax experiments.

Multivectors use the GA(3, 0) basis order
``[1, e1, e2, e3, e12, e13, e23, e123]``.  Every public operation treats the
last array axis as the eight blade coefficients and supports broadcastable
leading dimensions.
"""

from __future__ import annotations

from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

GA_DIM = 8
BASIS_MASKS = (0, 1, 2, 4, 3, 5, 6, 7)
GRADE_SLICES = ((0, 1), (1, 4), (4, 7), (7, 8))


def _blade_product_sign(left: int, right: int) -> int:
    """Return the sign of two Euclidean basis-blade bitmasks."""
    swaps = sum((right & ((1 << bit) - 1)).bit_count() for bit in range(3) if left & (1 << bit))
    return -1 if swaps % 2 else 1


def _multiplication_table() -> np.ndarray:
    table = np.zeros((GA_DIM, GA_DIM, GA_DIM), dtype=np.float32)
    mask_to_index = {mask: index for index, mask in enumerate(BASIS_MASKS)}
    for left_index, left_mask in enumerate(BASIS_MASKS):
        for right_index, right_mask in enumerate(BASIS_MASKS):
            output_index = mask_to_index[left_mask ^ right_mask]
            table[output_index, left_index, right_index] = _blade_product_sign(
                left_mask, right_mask
            )
    return table


MULTIPLICATION_TABLE = jnp.asarray(_multiplication_table())
REVERSION_SIGNS = jnp.asarray([1, 1, 1, 1, -1, -1, -1, -1], dtype=jnp.float32)


@jax.jit
def geometric_product(left: jax.Array, right: jax.Array) -> jax.Array:
    """Compute the GA(3, 0) geometric product along the final axis."""
    if left.shape[-1] != GA_DIM or right.shape[-1] != GA_DIM:
        raise ValueError(f"multivectors must have a final dimension of {GA_DIM}")
    table = MULTIPLICATION_TABLE.astype(jnp.result_type(left, right))
    return jnp.einsum("...i,...j,kij->...k", left, right, table)


@jax.jit
def reversion(multivector: jax.Array) -> jax.Array:
    """Reverse a GA(3, 0) multivector."""
    if multivector.shape[-1] != GA_DIM:
        raise ValueError(f"multivectors must have a final dimension of {GA_DIM}")
    return multivector * REVERSION_SIGNS.astype(multivector.dtype)


def scalar_product(left: jax.Array, right: jax.Array) -> jax.Array:
    """Return the scalar part of ``left * reverse(right)``."""
    return geometric_product(left, reversion(right))[..., 0]


def normalized_rotor(parameters: jax.Array) -> jax.Array:
    """Convert ``[..., scalar, e12, e13, e23]`` parameters to unit rotors."""
    if parameters.shape[-1] != 4:
        raise ValueError("rotor parameters must have a final dimension of 4")
    norm = jnp.linalg.norm(parameters, axis=-1, keepdims=True)
    normalized = parameters / jnp.maximum(norm, 1e-6)
    identity = jnp.zeros_like(parameters).at[..., 0].set(1)
    parameters = jnp.where(norm > 1e-6, normalized, identity)
    scalar, e12, e13, e23 = jnp.split(parameters, 4, axis=-1)
    zeros = jnp.zeros_like(scalar)
    return jnp.concatenate(
        [scalar, zeros, zeros, zeros, e12, e13, e23, zeros], axis=-1
    )


def rotor_from_bivector(
    bivector: jax.Array, max_angle: float = np.pi
) -> jax.Array:
    """Exponentiate a 3D bivector into a unit rotor.

    The bivector direction supplies the rotation plane while its norm is
    smoothly bounded to ``max_angle``.  The negative sign follows the usual
    ``R = exp(-B / 2)`` convention.
    """
    if bivector.shape[-1] != 3:
        raise ValueError("bivectors must have final dimension 3")
    max_angle = jnp.asarray(max_angle, bivector.dtype)
    magnitude_squared = jnp.sum(jnp.square(bivector), axis=-1, keepdims=True)
    # Avoid differentiating norm at the origin. Both output branches are smooth
    # functions of |B|^2 there; the displayed terms are their Taylor limits.
    threshold = jnp.asarray(jnp.finfo(bivector.dtype).eps, bivector.dtype)
    safe_magnitude = jnp.sqrt(jnp.maximum(magnitude_squared, threshold))
    regular_angle = max_angle * jnp.tanh(safe_magnitude)
    regular_scalar = jnp.cos(regular_angle / 2)
    regular_scale = jnp.sin(regular_angle / 2) / safe_magnitude
    small_scalar = 1 - (max_angle * max_angle / 8) * magnitude_squared
    small_scale = max_angle / 2 - (
        max_angle / 6 + max_angle**3 / 48
    ) * magnitude_squared
    use_regular = magnitude_squared > threshold
    rotor_scalar = jnp.where(use_regular, regular_scalar, small_scalar)
    bivector_scale = jnp.where(use_regular, regular_scale, small_scale)
    parameters = jnp.concatenate(
        [rotor_scalar, -bivector_scale * bivector], axis=-1
    )
    return normalized_rotor(parameters)


def rotor_product(left: jax.Array, right: jax.Array) -> jax.Array:
    """Multiply two even Cl(3) rotors without a dense 8x8 blade product.

    The public rotor representation remains ``[1,e12,e13,e23]`` embedded in
    eight coefficients. Internally it is converted to the standard quaternion
    vector ``[-e23,e13,-e12]`` for a compact Hamilton product.
    """

    if left.shape[-1] != GA_DIM or right.shape[-1] != GA_DIM:
        raise ValueError(f"rotors must have a final dimension of {GA_DIM}")
    left_scalar = left[..., :1]
    right_scalar = right[..., :1]
    left_vector = jnp.stack(
        [-left[..., 6], left[..., 5], -left[..., 4]], axis=-1
    )
    right_vector = jnp.stack(
        [-right[..., 6], right[..., 5], -right[..., 4]], axis=-1
    )
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
        [
            scalar,
            zeros,
            zeros,
            zeros,
            -vector[..., 2:3],
            vector[..., 1:2],
            -vector[..., 0:1],
            zeros,
        ],
        axis=-1,
    )


def rotor_sandwich(rotor: jax.Array, multivector: jax.Array) -> jax.Array:
    """Apply ``R x reverse(R)`` using the quaternion/vector decomposition.

    This is algebraically identical to two geometric products for an even
    rotor, including the norm-squared scaling of a non-unit rotor, but avoids
    contracting the dense multiplication table twice.
    """

    if rotor.shape[-1] != GA_DIM or multivector.shape[-1] != GA_DIM:
        raise ValueError(f"rotors and multivectors must end in {GA_DIM} components")
    scalar = rotor[..., :1]
    quaternion_vector = jnp.stack(
        [-rotor[..., 6], rotor[..., 5], -rotor[..., 4]], axis=-1
    )
    norm_squared = jnp.square(scalar) + jnp.sum(
        jnp.square(quaternion_vector), axis=-1, keepdims=True
    )

    def rotate(vectors: jax.Array) -> jax.Array:
        dot = jnp.sum(quaternion_vector * vectors, axis=-1, keepdims=True)
        return (
            (jnp.square(scalar) - jnp.sum(
                jnp.square(quaternion_vector), axis=-1, keepdims=True
            ))
            * vectors
            + 2 * quaternion_vector * dot
            + 2 * scalar * jnp.cross(quaternion_vector, vectors)
        )

    vector = rotate(multivector[..., 1:4])
    dual_bivector = jnp.stack(
        [multivector[..., 6], -multivector[..., 5], multivector[..., 4]], axis=-1
    )
    rotated_dual = rotate(dual_bivector)
    return jnp.concatenate(
        [
            norm_squared * multivector[..., :1],
            vector,
            rotated_dual[..., 2:3],
            -rotated_dual[..., 1:2],
            rotated_dual[..., 0:1],
            norm_squared * multivector[..., 7:8],
        ],
        axis=-1,
    )


def mv_relu(multivector: jax.Array) -> jax.Array:
    """Component-wise ReLU used by the exploratory neural layers."""
    return nn.relu(multivector)


def grade_invariants(multivector: jax.Array) -> jax.Array:
    """Return Spin(3)-invariant scalar summaries for each multivector.

    Scalar and pseudoscalar coefficients are invariant under proper rotations;
    vector and bivector grades contribute their Euclidean norms.
    """
    if multivector.shape[-1] != GA_DIM:
        raise ValueError(f"multivectors must have a final dimension of {GA_DIM}")
    vector_norm = jnp.linalg.norm(multivector[..., 1:4], axis=-1)
    bivector_norm = jnp.linalg.norm(multivector[..., 4:7], axis=-1)
    return jnp.stack(
        [multivector[..., 0], vector_norm, bivector_norm, multivector[..., 7]],
        axis=-1,
    )


def spin3_invariant_features(multivector: jax.Array) -> jax.Array:
    """Return smooth degree-at-most-two invariants for each Cl(3) channel.

    Unlike Euclidean norms, these polynomial features have a defined finite
    derivative when a vector or bivector grade is exactly zero. The mixed inner
    product is allowed because vectors and Hodge-dual bivectors are equivalent
    proper-rotation representations.
    """

    if multivector.shape[-1] != GA_DIM:
        raise ValueError(f"multivectors must have a final dimension of {GA_DIM}")
    vector = multivector[..., 1:4]
    dual_bivector = jnp.stack(
        [multivector[..., 6], -multivector[..., 5], multivector[..., 4]], axis=-1
    )
    return jnp.stack(
        [
            multivector[..., 0],
            multivector[..., 7],
            jnp.sum(jnp.square(vector), axis=-1),
            jnp.sum(jnp.square(dual_bivector), axis=-1),
            jnp.sum(vector * dual_bivector, axis=-1),
        ],
        axis=-1,
    )


def bounded_multivector(multivector: jax.Array) -> jax.Array:
    """Smoothly squash every channel to coefficient norm strictly below one."""

    if multivector.shape[-1] != GA_DIM:
        raise ValueError(f"multivectors must have a final dimension of {GA_DIM}")
    norm_squared = jnp.sum(jnp.square(multivector), axis=-1, keepdims=True)
    return multivector * jax.lax.rsqrt(1 + norm_squared)


class GeometricDense(nn.Module):
    """Apply learned GA(3, 0) operators to each input multivector.

    An input shaped ``(..., 8)`` produces ``(..., features, 8)``.  This is an
    operator bank, not a conventional dense layer: it deliberately preserves
    a separate multivector for each requested feature.
    """

    features: int
    activation: Callable[[jax.Array], jax.Array] | None = None
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        if inputs.shape[-1] != GA_DIM:
            raise ValueError(f"inputs must have a final dimension of {GA_DIM}")
        if self.features < 1:
            raise ValueError("features must be positive")

        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(),
            (self.features, GA_DIM),
            self.param_dtype,
        ).astype(self.dtype)
        outputs = geometric_product(kernel, inputs[..., None, :].astype(self.dtype))

        if self.use_bias:
            bias = self.param(
                "bias", nn.initializers.zeros, (self.features, GA_DIM), self.param_dtype
            ).astype(self.dtype)
            outputs = outputs + bias
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


class GeometricChannelMix(nn.Module):
    """Mix multivector channels using learned geometric-product operators."""

    in_channels: int
    out_channels: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        if inputs.shape[-2:] != (self.in_channels, GA_DIM):
            raise ValueError(
                f"expected final dimensions {(self.in_channels, GA_DIM)}, "
                f"received {inputs.shape[-2:]}"
            )
        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(),
            (self.out_channels, self.in_channels, GA_DIM),
            self.param_dtype,
        ).astype(self.dtype)
        products = geometric_product(
            kernel, inputs.astype(self.dtype)[..., None, :, :]
        )
        outputs = products.sum(axis=-2)
        if self.use_bias:
            bias = self.param(
                "bias",
                nn.initializers.zeros,
                (self.out_channels, GA_DIM),
                self.param_dtype,
            ).astype(self.dtype)
            outputs = outputs + bias
        return outputs


class GradeLinear(nn.Module):
    """Spin(3)-equivariant channel mixing that preserves blade grades.

    Each grade receives an independent scalar channel-mixing matrix, shared by
    every coordinate inside that grade.  Only the scalar grade receives a
    bias, so the operation commutes with rotor conjugation.
    """

    in_channels: int
    out_channels: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        if inputs.shape[-2:] != (self.in_channels, GA_DIM):
            raise ValueError(
                f"expected final dimensions {(self.in_channels, GA_DIM)}, "
                f"received {inputs.shape[-2:]}"
            )
        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(),
            (len(GRADE_SLICES), self.out_channels, self.in_channels),
            self.param_dtype,
        ).astype(self.dtype)
        grade_outputs = []
        for grade, (start, stop) in enumerate(GRADE_SLICES):
            grade_outputs.append(
                jnp.einsum(
                    "oi,...ic->...oc",
                    kernel[grade],
                    inputs[..., start:stop].astype(self.dtype),
                )
            )
        outputs = jnp.concatenate(grade_outputs, axis=-1)
        if self.use_bias:
            scalar_bias = self.param(
                "scalar_bias",
                nn.initializers.zeros,
                (self.out_channels,),
                self.param_dtype,
            ).astype(self.dtype)
            outputs = outputs.at[..., 0].add(scalar_bias)
        return outputs


def pack_spin3_isotypic(multivector: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Pack Cl(3) as two trivial and two vector-representation copies.

    Bivectors are Hodge-dualized from ``[e12,e13,e23]`` to
    ``[e23,-e13,e12]``. Under proper rotations this second copy transforms by
    exactly the same 3D matrix as the vector grade.
    """

    if multivector.ndim < 2 or multivector.shape[-1] != GA_DIM:
        raise ValueError("multivectors must have shape (..., channels, 8)")
    channels = multivector.shape[-2]
    trivial = jnp.stack(
        [multivector[..., 0], multivector[..., 7]], axis=-1
    ).reshape(*multivector.shape[:-2], 2 * channels)
    dual_bivector = jnp.stack(
        [multivector[..., 6], -multivector[..., 5], multivector[..., 4]],
        axis=-1,
    )
    active = jnp.stack([multivector[..., 1:4], dual_bivector], axis=-2)
    active = active.reshape(*multivector.shape[:-2], 2 * channels, 3)
    return trivial, active


def unpack_spin3_isotypic(trivial: jax.Array, active: jax.Array) -> jax.Array:
    """Invert :func:`pack_spin3_isotypic`."""

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


class Spin3IsotypicLinear(nn.Module):
    """Complete Spin(3)-equivariant linear mixing of Cl(3) channels.

    Unlike :class:`GradeLinear`, this layer includes every intertwiner between
    the repeated equivalent irreducible representations: scalar/pseudoscalar
    and vector/Hodge-dual-bivector.
    """

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


__all__ = [
    "BASIS_MASKS",
    "GA_DIM",
    "GRADE_SLICES",
    "GradeLinear",
    "GeometricChannelMix",
    "GeometricDense",
    "MULTIPLICATION_TABLE",
    "Spin3IsotypicLinear",
    "bounded_multivector",
    "geometric_product",
    "grade_invariants",
    "mv_relu",
    "normalized_rotor",
    "pack_spin3_isotypic",
    "reversion",
    "rotor_from_bivector",
    "rotor_product",
    "rotor_sandwich",
    "scalar_product",
    "spin3_invariant_features",
    "unpack_spin3_isotypic",
]
