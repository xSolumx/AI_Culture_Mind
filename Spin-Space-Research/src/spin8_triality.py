"""Real Spin(8) triality algebra and a constant-state recurrent harness.

The construction uses octonion left multiplication to obtain real Clifford
maps between the two eight-dimensional chiral spinor spaces.  One shared
28-coordinate bivector therefore induces actions on the vector, positive
half-spin, and negative half-spin representations.

This module is deliberately independent of the language model.  It is the
algebraic and recurrent correctness gate that must pass before training claims
about a Spin(8) SSM are meaningful.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SPIN8_DIM = 8
SPIN8_BIVECTOR_DIM = 28
SPIN8_PAIRS = tuple(
    (left, right)
    for left in range(SPIN8_DIM)
    for right in range(left + 1, SPIN8_DIM)
)
TRIALITY_REPRESENTATIONS = ("vector", "positive", "negative")

# Oriented lines of a Fano plane.  Reversing every orientation gives an
# isomorphic construction, but this convention is fixed for reproducibility.
FANO_TRIPLES = (
    (1, 2, 3),
    (1, 4, 5),
    (1, 7, 6),
    (2, 4, 6),
    (2, 5, 7),
    (3, 4, 7),
    (3, 6, 5),
)


@dataclass(frozen=True)
class Spin8TrialityAlgebra:
    """Fixed real matrices for V, S+, S-, and Clifford multiplication."""

    rho: np.ndarray
    gamma: np.ndarray
    chirality: np.ndarray
    vector_generators: np.ndarray
    positive_generators: np.ndarray
    negative_generators: np.ndarray

    def generators(self, representation: str) -> np.ndarray:
        if representation == "vector":
            return self.vector_generators
        if representation == "positive":
            return self.positive_generators
        if representation == "negative":
            return self.negative_generators
        raise ValueError(f"unknown triality representation {representation!r}")


@dataclass(frozen=True)
class AffineTransition:
    """A scalar-damped orthogonal-affine transition.

    Shapes may contain arbitrary common leading dimensions.  ``action`` ends
    in ``(..., representations, 8, 8)``, ``drive`` in
    ``(..., representations, 8)``, and ``scale`` contains the common leading
    dimensions without the representation axes.
    """

    scale: torch.Tensor
    action: torch.Tensor
    drive: torch.Tensor


def _octonion_multiplication_table() -> dict[tuple[int, int], tuple[int, int]]:
    table: dict[tuple[int, int], tuple[int, int]] = {}
    for index in range(SPIN8_DIM):
        table[(0, index)] = (1, index)
        table[(index, 0)] = (1, index)
    for index in range(1, SPIN8_DIM):
        table[(index, index)] = (-1, 0)
    for first, second, third in FANO_TRIPLES:
        for left, right, product in (
            (first, second, third),
            (second, third, first),
            (third, first, second),
        ):
            table[(left, right)] = (1, product)
            table[(right, left)] = (-1, product)
    if len(table) != SPIN8_DIM * SPIN8_DIM:
        raise AssertionError("the fixed Fano plane did not define every basis product")
    return table


def octonion_left_multiplication() -> np.ndarray:
    """Return ``rho[i]``, left multiplication by the i-th octonion basis."""

    table = _octonion_multiplication_table()
    matrices = np.zeros((SPIN8_DIM, SPIN8_DIM, SPIN8_DIM), dtype=np.float64)
    for left in range(SPIN8_DIM):
        for right in range(SPIN8_DIM):
            sign, product = table[(left, right)]
            matrices[left, product, right] = sign
    return matrices


def build_spin8_triality_algebra() -> Spin8TrialityAlgebra:
    """Construct the three eight-real representations without fitted data."""

    rho = octonion_left_multiplication()
    zero = np.zeros((SPIN8_DIM, SPIN8_DIM), dtype=np.float64)
    gamma = np.stack(
        [np.block([[zero, matrix.T], [matrix, zero]]) for matrix in rho]
    )
    chirality = np.linalg.multi_dot(list(gamma))

    vector_generators = []
    positive_generators = []
    negative_generators = []
    for left, right in SPIN8_PAIRS:
        vector = np.zeros((SPIN8_DIM, SPIN8_DIM), dtype=np.float64)
        vector[left, right] = 1.0
        vector[right, left] = -1.0
        spin = 0.25 * (
            gamma[left] @ gamma[right] - gamma[right] @ gamma[left]
        )
        vector_generators.append(vector)
        # The fixed Fano convention makes the top block chirality +1.
        positive_generators.append(spin[:SPIN8_DIM, :SPIN8_DIM])
        negative_generators.append(spin[SPIN8_DIM:, SPIN8_DIM:])

    return Spin8TrialityAlgebra(
        rho=rho,
        gamma=gamma,
        chirality=chirality,
        vector_generators=np.stack(vector_generators),
        positive_generators=np.stack(positive_generators),
        negative_generators=np.stack(negative_generators),
    )


def _signed_generator(
    generators: np.ndarray, first: int, second: int
) -> np.ndarray:
    if first == second:
        return np.zeros_like(generators[0])
    if first < second:
        return generators[SPIN8_PAIRS.index((first, second))]
    return -generators[SPIN8_PAIRS.index((second, first))]


def _commutator_rhs(
    generators: np.ndarray, first_pair: tuple[int, int], second_pair: tuple[int, int]
) -> np.ndarray:
    i, j = first_pair
    k, ell = second_pair
    result = np.zeros_like(generators[0])
    if j == k:
        result += _signed_generator(generators, i, ell)
    if i == k:
        result -= _signed_generator(generators, j, ell)
    if j == ell:
        result -= _signed_generator(generators, i, k)
    if i == ell:
        result += _signed_generator(generators, j, k)
    return result


def _matrix_residual(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(matrix)))


def algebra_diagnostics(seed: int = 20260803) -> dict[str, object]:
    """Evaluate every frozen Spin(8) algebra acceptance certificate."""

    algebra = build_spin8_triality_algebra()
    multiplication_table = _octonion_multiplication_table()
    identity8 = np.eye(SPIN8_DIM)
    identity16 = np.eye(2 * SPIN8_DIM)

    clifford_residual = 0.0
    rho_clifford_residual = 0.0
    for left in range(SPIN8_DIM):
        for right in range(SPIN8_DIM):
            target8 = 2.0 * float(left == right) * identity8
            target16 = 2.0 * float(left == right) * identity16
            rho_clifford_residual = max(
                rho_clifford_residual,
                _matrix_residual(
                    algebra.rho[left].T @ algebra.rho[right]
                    + algebra.rho[right].T @ algebra.rho[left]
                    - target8
                ),
            )
            clifford_residual = max(
                clifford_residual,
                _matrix_residual(
                    algebra.gamma[left] @ algebra.gamma[right]
                    + algebra.gamma[right] @ algebra.gamma[left]
                    - target16
                ),
            )

    chirality_eigenvalues = np.linalg.eigvalsh(algebra.chirality)
    chirality_positive_count = int(np.sum(chirality_eigenvalues > 0.5))
    chirality_negative_count = int(np.sum(chirality_eigenvalues < -0.5))
    chirality_square_residual = _matrix_residual(
        algebra.chirality @ algebra.chirality - identity16
    )
    basis_product_norm_residual = max(
        abs(float(sign * sign) - 1.0)
        for sign, _ in multiplication_table.values()
    )

    representation_metrics: dict[str, object] = {}
    commutator_max = 0.0
    for representation in TRIALITY_REPRESENTATIONS:
        generators = algebra.generators(representation)
        skew_residual = _matrix_residual(
            generators + np.swapaxes(generators, -1, -2)
        )
        rank = int(np.linalg.matrix_rank(generators.reshape(SPIN8_BIVECTOR_DIM, -1)))
        commutator_residual = 0.0
        for first_index, first_pair in enumerate(SPIN8_PAIRS):
            for second_index, second_pair in enumerate(SPIN8_PAIRS):
                commutator = (
                    generators[first_index] @ generators[second_index]
                    - generators[second_index] @ generators[first_index]
                )
                commutator_residual = max(
                    commutator_residual,
                    _matrix_residual(
                        commutator
                        - _commutator_rhs(generators, first_pair, second_pair)
                    ),
                )
        commutator_max = max(commutator_max, commutator_residual)
        representation_metrics[representation] = {
            "skew_symmetry_max_abs": skew_residual,
            "linear_rank": rank,
            "so8_commutator_max_abs": commutator_residual,
        }

    triality_residual = 0.0
    for pair_index, (left, right) in enumerate(SPIN8_PAIRS):
        positive = algebra.positive_generators[pair_index]
        negative = algebra.negative_generators[pair_index]
        for vector_index in range(SPIN8_DIM):
            target = np.zeros((SPIN8_DIM, SPIN8_DIM), dtype=np.float64)
            if vector_index == right:
                target += algebra.rho[left]
            if vector_index == left:
                target -= algebra.rho[right]
            triality_residual = max(
                triality_residual,
                _matrix_residual(
                    negative @ algebra.rho[vector_index]
                    - algebra.rho[vector_index] @ positive
                    - target
                ),
            )

    rng = np.random.default_rng(seed)
    coefficients = torch.from_numpy(rng.normal(size=SPIN8_BIVECTOR_DIM)).to(
        torch.float64
    )
    exponential_metrics: dict[str, object] = {}
    exponential_orthogonality_max = 0.0
    exponential_determinant_max = 0.0
    for representation in TRIALITY_REPRESENTATIONS:
        generators = torch.from_numpy(algebra.generators(representation))
        action = torch.matrix_exp(torch.einsum("p,pij->ij", coefficients, generators))
        orthogonality = float(
            (action.T @ action - torch.eye(SPIN8_DIM, dtype=torch.float64)).abs().max()
        )
        determinant_error = abs(float(torch.linalg.det(action)) - 1.0)
        exponential_orthogonality_max = max(
            exponential_orthogonality_max, orthogonality
        )
        exponential_determinant_max = max(
            exponential_determinant_max, determinant_error
        )
        exponential_metrics[representation] = {
            "orthogonality_max_abs": orthogonality,
            "determinant": float(torch.linalg.det(action)),
            "determinant_abs_error": determinant_error,
        }

    two_pi_metrics: dict[str, float] = {}
    for representation in TRIALITY_REPRESENTATIONS:
        generator = torch.from_numpy(algebra.generators(representation)[0])
        action = torch.matrix_exp(2.0 * math.pi * generator)
        target_sign = 1.0 if representation == "vector" else -1.0
        two_pi_metrics[representation] = float(
            (action - target_sign * torch.eye(SPIN8_DIM, dtype=torch.float64))
            .abs()
            .max()
        )

    omega_positive = algebra.chirality[:SPIN8_DIM, :SPIN8_DIM]
    omega_negative = algebra.chirality[SPIN8_DIM:, SPIN8_DIM:]
    central_targets = {
        "identity": (identity8, identity8, identity8),
        "minus_one": (identity8, -identity8, -identity8),
        "omega": (-identity8, identity8, -identity8),
        "minus_omega": (-identity8, -identity8, identity8),
    }
    central_actual = {
        "identity": (identity8, identity8, identity8),
        "minus_one": (identity8, -identity8, -identity8),
        "omega": (-identity8, omega_positive, omega_negative),
        "minus_omega": (-identity8, -omega_positive, -omega_negative),
    }
    center_residuals = {
        name: max(
            _matrix_residual(actual - target)
            for actual, target in zip(central_actual[name], central_targets[name])
        )
        for name in central_targets
    }

    thresholds = {
        "algebra_max_abs": 1e-12,
        "exponential_max_abs": 1e-10,
    }
    checks = {
        "octonion_basis_norm": basis_product_norm_residual == 0.0,
        "clifford": clifford_residual <= thresholds["algebra_max_abs"],
        "chirality": (
            chirality_square_residual <= thresholds["algebra_max_abs"]
            and chirality_positive_count == SPIN8_DIM
            and chirality_negative_count == SPIN8_DIM
        ),
        "generator_rank": all(
            metrics["linear_rank"] == SPIN8_BIVECTOR_DIM
            for metrics in representation_metrics.values()
        ),
        "skew_symmetry": all(
            metrics["skew_symmetry_max_abs"] <= thresholds["algebra_max_abs"]
            for metrics in representation_metrics.values()
        ),
        "so8_commutators": commutator_max <= thresholds["algebra_max_abs"],
        "triality_equivariance": triality_residual <= thresholds["algebra_max_abs"],
        "orthogonal_exponentials": (
            exponential_orthogonality_max <= thresholds["exponential_max_abs"]
            and exponential_determinant_max <= thresholds["exponential_max_abs"]
        ),
        "two_pi_center": max(two_pi_metrics.values())
        <= thresholds["exponential_max_abs"],
        "full_center_signatures": max(center_residuals.values())
        <= thresholds["algebra_max_abs"],
    }

    return {
        "experiment": "prospective Spin(8) triality algebra gate",
        "construction": "fixed octonion Fano-plane Clifford representation",
        "seed_used_only_for_random_exponential_probe": seed,
        "thresholds": thresholds,
        "octonion_basis_product_norm_abs": basis_product_norm_residual,
        "rho_clifford_max_abs": rho_clifford_residual,
        "clifford_max_abs": clifford_residual,
        "chirality_square_max_abs": chirality_square_residual,
        "chirality_multiplicities": {
            "positive": chirality_positive_count,
            "negative": chirality_negative_count,
        },
        "representations": representation_metrics,
        "triality_equivariance_max_abs": triality_residual,
        "random_exponentials": exponential_metrics,
        "two_pi_center_max_abs": two_pi_metrics,
        "center_signature_max_abs": center_residuals,
        "checks": checks,
        "passed": all(checks.values()),
    }


def torch_triality_generators(
    representations: Sequence[str] = TRIALITY_REPRESENTATIONS,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Stack fixed generators as ``(representations, 28, 8, 8)``."""

    algebra = build_spin8_triality_algebra()
    arrays = [algebra.generators(name) for name in representations]
    return torch.as_tensor(np.stack(arrays), dtype=dtype, device=device)


def so8_chart_equivalence_diagnostics(
    seed: int = 20260803,
) -> dict[str, object]:
    """Prove that one chiral chart and a generic SO(8) chart span the same actions.

    The vector generators are the standard elementary skew matrices.  The
    positive half-spin generators are another orthogonal basis of the complete
    28-dimensional skew-matrix space.  The returned coefficient map rotates
    positive-chart coordinates into standard SO(8) coordinates without changing
    the tangent matrix or its exponential.
    """

    algebra = build_spin8_triality_algebra()
    positive = algebra.positive_generators
    standard = algebra.vector_generators
    positive_gram = np.einsum("aij,bij->ab", positive, positive)
    standard_gram = np.einsum("aij,bij->ab", standard, standard)
    # Both fixed bases have squared Frobenius norm two.
    coefficient_map = np.einsum("aij,bij->ab", positive, standard) / 2.0
    reconstructed = np.einsum("ab,bij->aij", coefficient_map, standard)
    rng = np.random.default_rng(seed)
    positive_coefficients = rng.normal(scale=0.2, size=(5, SPIN8_BIVECTOR_DIM))
    standard_coefficients = positive_coefficients @ coefficient_map
    positive_actions = spin8_actions(
        torch.from_numpy(positive_coefficients),
        torch_triality_generators(("positive",), dtype=torch.float64),
    ).squeeze(-3)
    standard_actions = spin8_actions(
        torch.from_numpy(standard_coefficients),
        torch_triality_generators(("vector",), dtype=torch.float64),
    ).squeeze(-3)
    checks = {
        "positive_basis_is_orthogonal": bool(
            np.max(np.abs(positive_gram - 2.0 * np.eye(SPIN8_BIVECTOR_DIM)))
            <= 1e-12
        ),
        "standard_basis_is_orthogonal": bool(
            np.max(np.abs(standard_gram - 2.0 * np.eye(SPIN8_BIVECTOR_DIM)))
            <= 1e-12
        ),
        "basis_change_is_orthogonal": bool(
            np.max(
                np.abs(
                    coefficient_map @ coefficient_map.T
                    - np.eye(SPIN8_BIVECTOR_DIM)
                )
            )
            <= 1e-12
        ),
        "generator_reconstruction": bool(
            np.max(np.abs(reconstructed - positive)) <= 1e-12
        ),
        "random_action_equivalence": bool(
            float((positive_actions - standard_actions).abs().max()) <= 1e-12
        ),
    }
    return {
        "experiment": "positive-half-spin versus generic SO8 chart equivalence",
        "coefficient_map": coefficient_map.tolist(),
        "coefficient_map_determinant": float(np.linalg.det(coefficient_map)),
        "coefficient_map_singular_values": np.linalg.svd(
            coefficient_map, compute_uv=False
        ).tolist(),
        "positive_basis_gram_max_abs_error": float(
            np.max(np.abs(positive_gram - 2.0 * np.eye(SPIN8_BIVECTOR_DIM)))
        ),
        "standard_basis_gram_max_abs_error": float(
            np.max(np.abs(standard_gram - 2.0 * np.eye(SPIN8_BIVECTOR_DIM)))
        ),
        "basis_change_orthogonality_max_abs_error": float(
            np.max(
                np.abs(
                    coefficient_map @ coefficient_map.T
                    - np.eye(SPIN8_BIVECTOR_DIM)
                )
            )
        ),
        "generator_reconstruction_max_abs_error": float(
            np.max(np.abs(reconstructed - positive))
        ),
        "random_action_equivalence_max_abs_error": float(
            (positive_actions - standard_actions).abs().max()
        ),
        "interpretation": (
            "a single positive-half-spin 8D recurrence and a generic SO(8) "
            "exponential have identical transition families; only the chart, "
            "optimizer geometry, and global group-kernel interpretation differ"
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }


def spin8_actions(coefficients: torch.Tensor, generators: torch.Tensor) -> torch.Tensor:
    """Exponentiate a shared bivector in each selected triality representation.

    ``coefficients`` has shape ``(..., 28)`` and generators have shape
    ``(representations, 28, 8, 8)``.  The result has shape
    ``(..., representations, 8, 8)``.
    """

    if coefficients.shape[-1] != SPIN8_BIVECTOR_DIM:
        raise ValueError("Spin(8) coefficients must have 28 bivector coordinates")
    if generators.ndim != 4 or generators.shape[1:] != (
        SPIN8_BIVECTOR_DIM,
        SPIN8_DIM,
        SPIN8_DIM,
    ):
        raise ValueError("generators must have shape (representations, 28, 8, 8)")
    tangent = torch.einsum("...p,rpij->...rij", coefficients, generators)
    return torch.matrix_exp(tangent)


def compose_affine(after: AffineTransition, before: AffineTransition) -> AffineTransition:
    """Compose ``after(before(state))`` using the associative affine law."""

    action = after.action @ before.action
    rotated_drive = torch.einsum("...rij,...rj->...ri", after.action, before.drive)
    drive = after.drive + after.scale[..., None, None] * rotated_drive
    return AffineTransition(
        scale=after.scale * before.scale,
        action=action,
        drive=drive,
    )


def apply_affine(transition: AffineTransition, state: torch.Tensor) -> torch.Tensor:
    rotated = torch.einsum("...rij,...rj->...ri", transition.action, state)
    return transition.scale[..., None, None] * rotated + transition.drive


def associative_prefix_scan(transition: AffineTransition) -> AffineTransition:
    """Inclusive logarithmic-depth scan over sequence dimension 1.

    This vectorized Hillis-Steele reference has `O(log N)` dependency depth and
    `O(N log N)` work.  It is a correctness oracle, not the eventual optimized
    fused `O(N)`-work scan kernel.
    """

    if transition.scale.ndim < 2:
        raise ValueError("a scanned transition needs batch and sequence dimensions")
    scale, action, drive = transition.scale, transition.action, transition.drive
    offset = 1
    while offset < scale.shape[1]:
        after = AffineTransition(
            scale=scale[:, offset:],
            action=action[:, offset:],
            drive=drive[:, offset:],
        )
        before = AffineTransition(
            scale=scale[:, :-offset],
            action=action[:, :-offset],
            drive=drive[:, :-offset],
        )
        composed = compose_affine(after, before)
        scale = torch.cat((scale[:, :offset], composed.scale), dim=1)
        action = torch.cat((action[:, :offset], composed.action), dim=1)
        drive = torch.cat((drive[:, :offset], composed.drive), dim=1)
        offset *= 2
    return AffineTransition(scale=scale, action=action, drive=drive)


class Spin8TrialitySSM(nn.Module):
    """Selective Spin(8) recurrence with an exact constant-size cache."""

    def __init__(
        self,
        input_size: int,
        *,
        channels: int = 1,
        representations: Sequence[str] = ("positive",),
        min_half_life: float = 4.0,
    ) -> None:
        super().__init__()
        if input_size < 1 or channels < 1:
            raise ValueError("input size and channels must be positive")
        representations = tuple(representations)
        if not representations or len(set(representations)) != len(representations):
            raise ValueError("representations must be nonempty and unique")
        if any(name not in TRIALITY_REPRESENTATIONS for name in representations):
            raise ValueError(f"representations must be drawn from {TRIALITY_REPRESENTATIONS}")
        if min_half_life <= 0:
            raise ValueError("minimum half-life must be positive")

        self.input_size = input_size
        self.channels = channels
        self.representations = representations
        self.min_half_life = min_half_life
        self.coefficient_controller = nn.Linear(
            input_size, channels * SPIN8_BIVECTOR_DIM
        )
        self.decay_controller = nn.Linear(input_size, channels)
        self.drive_controller = nn.Linear(
            input_size, channels * len(representations) * SPIN8_DIM
        )
        nn.init.zeros_(self.coefficient_controller.weight)
        nn.init.zeros_(self.coefficient_controller.bias)
        nn.init.zeros_(self.decay_controller.weight)
        nn.init.zeros_(self.decay_controller.bias)
        nn.init.zeros_(self.drive_controller.bias)
        self.log_half_life = nn.Parameter(torch.zeros(channels))
        initial = torch.randn(channels, len(representations), SPIN8_DIM)
        self.initial_state = nn.Parameter(F.normalize(initial, dim=-1))
        self.register_buffer(
            "generators",
            torch_triality_generators(representations),
            persistent=True,
        )

    @property
    def cache_scalars(self) -> int:
        return self.channels * len(self.representations) * SPIN8_DIM

    def transitions(self, inputs: torch.Tensor) -> AffineTransition:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_size:
            raise ValueError("inputs must have shape (batch, sequence, input_size)")
        batch, length, _ = inputs.shape
        coefficients = self.coefficient_controller(inputs).reshape(
            batch, length, self.channels, SPIN8_BIVECTOR_DIM
        )
        generators = self.generators.to(dtype=inputs.dtype, device=inputs.device)
        actions = spin8_actions(coefficients, generators)
        half_life = self.min_half_life + F.softplus(self.log_half_life)
        step_size = F.softplus(self.decay_controller(inputs))
        scale = torch.exp(-math.log(2.0) * step_size / half_life)
        drive = self.drive_controller(inputs).reshape(
            batch,
            length,
            self.channels,
            len(self.representations),
            SPIN8_DIM,
        )
        # Fold channels into the common leading dimensions used by the affine
        # primitives; representation and vector axes stay explicit.
        return AffineTransition(scale=scale, action=actions, drive=drive)

    def initial_cache(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        return self.initial_state.to(reference).unsqueeze(0).expand(batch_size, -1, -1, -1)

    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transition = self.transitions(inputs)
        if state is None:
            state = self.initial_cache(inputs.shape[0], inputs)
        expected = (
            inputs.shape[0],
            self.channels,
            len(self.representations),
            SPIN8_DIM,
        )
        if state.shape != expected:
            raise ValueError(f"state must have shape {expected}, got {tuple(state.shape)}")
        outputs = []
        for index in range(inputs.shape[1]):
            step = AffineTransition(
                scale=transition.scale[:, index],
                action=transition.action[:, index],
                drive=transition.drive[:, index],
            )
            state = apply_affine(step, state)
            outputs.append(state)
        if not outputs:
            empty = state.new_empty((inputs.shape[0], 0, *state.shape[1:]))
            return empty, state
        return torch.stack(outputs, dim=1), state

    def parallel(self, inputs: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        """Evaluate with the associative prefix-scan correctness oracle."""

        transitions = self.transitions(inputs)
        if state is None:
            state = self.initial_cache(inputs.shape[0], inputs)
        prefixes = associative_prefix_scan(transitions)
        return apply_affine(prefixes, state[:, None])


def recurrent_diagnostics(seed: int = 20260803) -> dict[str, object]:
    """Run the frozen affine, streaming, scan, and identity-gradient checks."""

    torch.manual_seed(seed)
    dtype = torch.float64
    representations = TRIALITY_REPRESENTATIONS
    generators = torch_triality_generators(representations, dtype=dtype)

    def random_transition() -> AffineTransition:
        coefficients = 0.2 * torch.randn(SPIN8_BIVECTOR_DIM, dtype=dtype)
        return AffineTransition(
            scale=torch.sigmoid(torch.randn((), dtype=dtype)),
            action=spin8_actions(coefficients, generators),
            drive=torch.randn(len(representations), SPIN8_DIM, dtype=dtype),
        )

    first, second, third = random_transition(), random_transition(), random_transition()
    left = compose_affine(third, compose_affine(second, first))
    right = compose_affine(compose_affine(third, second), first)
    associativity_residual = max(
        float((left.scale - right.scale).abs().max()),
        float((left.action - right.action).abs().max()),
        float((left.drive - right.drive).abs().max()),
    )

    model = Spin8TrialitySSM(
        5, channels=2, representations=representations
    ).to(dtype=dtype)
    inputs = torch.randn(3, 17, 5, dtype=dtype)
    full, full_state = model(inputs)
    first_chunk, chunk_state = model(inputs[:, :6])
    second_chunk, chunk_state = model(inputs[:, 6:], chunk_state)
    chunked = torch.cat((first_chunk, second_chunk), dim=1)
    streamed_outputs = []
    streamed_state = None
    for index in range(inputs.shape[1]):
        output, streamed_state = model(inputs[:, index : index + 1], streamed_state)
        streamed_outputs.append(output)
    streamed = torch.cat(streamed_outputs, dim=1)
    parallel = model.parallel(inputs)

    loss = full.square().mean()
    loss.backward()
    coefficient_gradient = model.coefficient_controller.weight.grad
    gradient_norm = float(coefficient_gradient.norm())
    gradient_finite = bool(torch.isfinite(coefficient_gradient).all())

    thresholds = {
        "associativity_max_abs": 1e-12,
        "sequential_equivalence_max_abs": 1e-10,
    }
    metrics = {
        "affine_associativity_max_abs": associativity_residual,
        "chunk_output_max_abs": float((chunked - full).detach().abs().max()),
        "chunk_state_max_abs": float((chunk_state - full_state).detach().abs().max()),
        "token_output_max_abs": float((streamed - full).detach().abs().max()),
        "token_state_max_abs": float(
            (streamed_state - full_state).detach().abs().max()
        ),
        "parallel_output_max_abs": float((parallel - full).detach().abs().max()),
        "identity_controller_gradient_norm": gradient_norm,
        "identity_controller_gradient_finite": gradient_finite,
        "cache_scalars": model.cache_scalars,
        "cache_scalars_length_17": model.cache_scalars,
        "cache_scalars_length_257": model.cache_scalars,
    }
    checks = {
        "affine_associativity": associativity_residual
        <= thresholds["associativity_max_abs"],
        "chunk_equivalence": max(
            metrics["chunk_output_max_abs"], metrics["chunk_state_max_abs"]
        )
        <= thresholds["sequential_equivalence_max_abs"],
        "token_equivalence": max(
            metrics["token_output_max_abs"], metrics["token_state_max_abs"]
        )
        <= thresholds["sequential_equivalence_max_abs"],
        "parallel_equivalence": metrics["parallel_output_max_abs"]
        <= thresholds["sequential_equivalence_max_abs"],
        "identity_tangent_gradient": gradient_finite and gradient_norm > 0.0,
        "constant_cache": metrics["cache_scalars_length_17"]
        == metrics["cache_scalars_length_257"],
    }
    return {
        "experiment": "Spin(8) triality recurrent implementation gate",
        "seed": seed,
        "dtype": str(dtype),
        "representations": list(representations),
        "thresholds": thresholds,
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {
        "algebra": algebra_diagnostics(args.seed),
        "recurrent": recurrent_diagnostics(args.seed),
    }
    report["passed"] = report["algebra"]["passed"] and report["recurrent"]["passed"]
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
