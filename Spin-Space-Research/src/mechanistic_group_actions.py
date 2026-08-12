"""Mechanistic finite-group tests for write-free recurrent actions.

This module deliberately removes decay, affine writes, residual paths, and
token-conditioned nonlinear controllers.  A token selects one norm-preserving
linear action and the recurrent state is updated only by composing that action.
The experiment therefore asks whether SGD discovers a group representation,
not whether a larger sequence model can interpolate the training grammar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Set before the first CUDA context is created. Research seed comparisons are
# not interpretable if cuBLAS is free to select nondeterministic workspaces.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn

from compare_recurrences import (
    GROUPS,
    FiniteGroup,
    make_group_batches,
    pair_split_audit,
    parse_held_out_pairs,
    parse_input_elements,
    state_and_pair_coverage_audit,
)
from rotor_ssm_torch import GA_DIM, rotor_from_bivector, rotor_sandwich
from recurrence_families_torch import (
    quaternion_product,
    unit_quaternion_from_bivector,
)
from spin8_triality import spin8_actions, torch_triality_generators


MECHANISM_FAMILIES = (
    "pure_complex_unitary",
    "pure_ga_rotor",
    "pure_quaternion_spinor",
    "pure_spin8_positive",
    "pure_so8_exponential",
    "pure_householder",
    "pure_householder4_shared",
)


@dataclass(frozen=True)
class MechanismConfig:
    steps: int = 2000
    batch_size: int = 256
    sequence_length: int = 16
    validation_batches: int = 8
    validation_batch_size: int = 512
    channels: int = 4
    learning_rate: float = 3e-3
    final_position_loss_weight: float = 0.0
    seed: int = 0
    diagnostic_interval: int = 100
    max_rotor_angle: float = math.pi
    relation_loss_weight: float = 0.0
    relation_loss_power: float = 2.0
    relation_start_step: int = 0
    relation_ramp_steps: int = 0
    canonical_orbit_loss_weight: float = 0.0
    holonomy_loss_weight: float = 0.0
    holonomy_loss_power: float = 8.0
    holonomy_margin_weight: float = 0.0
    holonomy_margin_target: float = 0.5
    holonomy_start_step: int = 0
    holonomy_ramp_steps: int = 0
    holonomy_word_multiplier: int = 4
    holonomy_word_multipliers: tuple[int, ...] = ()
    holonomy_batch_size: int = 64
    a5_irrep_init: bool = False
    freeze_actions: bool = False


def _normalized_state(state: torch.Tensor) -> torch.Tensor:
    return state / state.norm(dim=-1, keepdim=True).clamp_min(1e-7)


def _complex_action(phases: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    pairs = state.reshape(*state.shape[:-1], 4, 2)
    real, imaginary = pairs.unbind(dim=-1)
    cosine, sine = torch.cos(phases), torch.sin(phases)
    return torch.stack(
        (cosine * real - sine * imaginary, sine * real + cosine * imaginary),
        dim=-1,
    ).reshape_as(state)


def _householder_action(vectors: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    result = state
    for reflector in vectors.unbind(dim=-2):
        unit = reflector / reflector.norm(dim=-1, keepdim=True).clamp_min(1e-7)
        result = result - 2.0 * (result * unit).sum(dim=-1, keepdim=True) * unit
    return result


class PureGroupActionModel(nn.Module):
    """A decoder over an exact, fixed-size, write-free recurrent orbit."""

    def __init__(
        self,
        vocab_size: int,
        output_size: int,
        *,
        family: str,
        channels: int = 4,
        max_rotor_angle: float = math.pi,
    ) -> None:
        super().__init__()
        if family not in MECHANISM_FAMILIES:
            raise ValueError(f"unknown mechanism family {family!r}")
        if channels < 1:
            raise ValueError("channels must be positive")
        self.family = family
        self.channels = channels
        self.vocab_size = vocab_size
        self.max_rotor_angle = max_rotor_angle

        initial = torch.randn(channels, GA_DIM)
        # Scalar and pseudoscalar coordinates are invariant under Cl(3) rotor
        # conjugation, so begin on the informative vector+bivector orbit.
        initial[:, 0] = 0.0
        initial[:, 7] = 0.0
        self.initial_orbit_state = nn.Parameter(_normalized_state(initial))
        # Construct the common decoder before family-specific parameters so
        # identical seeds give every identity-initialized family the same
        # initial function even when raw action parameter counts differ.
        self.output_head = nn.Linear(channels * GA_DIM, output_size)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

        if family == "pure_complex_unitary":
            self.action_parameters = nn.Parameter(
                torch.zeros(vocab_size, channels, 4)
            )
        elif family in ("pure_ga_rotor", "pure_quaternion_spinor"):
            self.action_parameters = nn.Parameter(
                torch.zeros(vocab_size, channels, 3)
            )
        elif family in ("pure_spin8_positive", "pure_so8_exponential"):
            # Unconstrained tangent updates are exponentiated jointly in the
            # fixed positive half-spin representation.  No token is normalized
            # independently; later family-level retraction must act on the
            # complete token-action family if the task supplies group structure.
            self.action_parameters = nn.Parameter(
                torch.zeros(vocab_size, channels, 28)
            )
            representation = (
                "positive" if family == "pure_spin8_positive" else "vector"
            )
            self.register_buffer(
                "spin8_generators",
                torch_triality_generators((representation,)),
            )
        elif family == "pure_householder":
            # Two reflections express one arbitrary plane rotation.  Identical
            # initial reflectors make every token action exactly the identity.
            base = _normalized_state(torch.randn(vocab_size, channels, 1, GA_DIM))
            self.action_parameters = nn.Parameter(base.repeat(1, 1, 2, 1))
        else:
            # A faithful Q8 left action is an isoclinic SO(4) rotation with
            # rank(I-A)=4, so it needs four reflections. Apply one learned O(4)
            # action to both quaternion blocks, matching the spinor's shared
            # action structure while retaining a generic capable chart.
            base = _normalized_state(
                torch.randn(vocab_size, channels, 2, GA_DIM // 2)
            )
            self.action_parameters = nn.Parameter(
                base.repeat_interleave(2, dim=-2)
            )

    def initial_state(self, batch_size: int) -> torch.Tensor:
        state = _normalized_state(self.initial_orbit_state)
        return state.unsqueeze(0).expand(batch_size, -1, -1)

    def token_actions(self, token_ids: torch.Tensor) -> torch.Tensor:
        parameters = self.action_parameters[token_ids]
        if self.family == "pure_complex_unitary":
            return math.pi * torch.tanh(parameters)
        if self.family == "pure_ga_rotor":
            return rotor_from_bivector(parameters, self.max_rotor_angle)
        if self.family == "pure_quaternion_spinor":
            return unit_quaternion_from_bivector(
                parameters, self.max_rotor_angle
            )
        if self.family in ("pure_spin8_positive", "pure_so8_exponential"):
            generators = self.spin8_generators.to(
                dtype=parameters.dtype, device=parameters.device
            )
            return spin8_actions(parameters, generators).squeeze(-3)
        return parameters

    def apply_action(self, action: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self.family == "pure_complex_unitary":
            return _complex_action(action, state)
        if self.family == "pure_ga_rotor":
            return rotor_sandwich(action, state)
        if self.family == "pure_quaternion_spinor":
            spinors = state.reshape(*state.shape[:-1], 2, 4)
            return quaternion_product(action.unsqueeze(-2), spinors).reshape_as(state)
        if self.family in ("pure_spin8_positive", "pure_so8_exponential"):
            return torch.einsum("...ij,...j->...i", action, state)
        if self.family == "pure_householder4_shared":
            spinors = state.reshape(*state.shape[:-1], 2, 4)
            transformed = _householder_action(action.unsqueeze(-3), spinors)
            return transformed.reshape_as(state)
        return _householder_action(action, state)

    def step(self, token_ids: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.apply_action(self.token_actions(token_ids), state)

    def decode(self, states: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * self.output_head(states.flatten(-2))

    def forward(
        self,
        token_ids: torch.Tensor,
        recurrent_state: torch.Tensor | None = None,
        *,
        return_recurrent_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        if recurrent_state is None:
            state = self.initial_state(token_ids.shape[0])
        else:
            expected = (token_ids.shape[0], self.channels, GA_DIM)
            if recurrent_state.shape != expected:
                raise ValueError(f"recurrent_state must have shape {expected}")
            state = recurrent_state
        # Token actions are input-independent in this mechanistic model. Build
        # their small matrices once, then reuse them at every sequence position.
        # This is exactly the same linear action as ``step`` but substantially
        # cheaper than expanding Clifford products token by token.
        action_matrices = self.action_matrices()
        states = []
        for position in range(token_ids.shape[1]):
            selected = action_matrices[token_ids[:, position]]
            state = torch.einsum("bcij,bcj->bci", selected, state)
            states.append(state)
        logits = self.decode(torch.stack(states, dim=1))
        if return_recurrent_state:
            return logits, state
        return logits

    def action_matrices(self) -> torch.Tensor:
        """Return matrices with shape ``(token, channel, output, input)``."""
        if self.family in ("pure_spin8_positive", "pure_so8_exponential"):
            token_ids = torch.arange(
                self.vocab_size,
                dtype=torch.long,
                device=self.initial_orbit_state.device,
            )
            return self.token_actions(token_ids)
        matrices = []
        basis = torch.eye(
            GA_DIM,
            dtype=self.initial_orbit_state.dtype,
            device=self.initial_orbit_state.device,
        ).unsqueeze(1).expand(-1, self.channels, -1)
        for token in range(self.vocab_size):
            token_ids = torch.full(
                (GA_DIM,), token, dtype=torch.long, device=basis.device
            )
            transformed = self.step(token_ids, basis)
            matrices.append(transformed.permute(1, 2, 0))
        return torch.stack(matrices)


def canonical_group_words(
    group: FiniteGroup, input_elements: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    """Breadth-first canonical generator word for every reachable element."""
    words: list[tuple[int, ...] | None] = [None] * group.order
    words[0] = ()
    queue: deque[int] = deque([0])
    while queue:
        element = queue.popleft()
        prefix = words[element]
        assert prefix is not None
        for token, generator in enumerate(input_elements):
            product = int(group.table[element, generator])
            if words[product] is None:
                words[product] = prefix + (token,)
                queue.append(product)
    if any(word is None for word in words):
        reached = sum(word is not None for word in words)
        raise ValueError(
            f"input alphabet generates only {reached}/{group.order} elements of {group.key}"
        )
    return tuple(word for word in words if word is not None)


def _element_inverses(group: FiniteGroup) -> np.ndarray:
    inverses = []
    for element in range(group.order):
        candidates = np.flatnonzero(group.table[element] == 0)
        if len(candidates) != 1:
            raise ValueError(f"{group.key} element {element} has no unique inverse")
        inverses.append(int(candidates[0]))
    return np.asarray(inverses, dtype=np.int64)


def _element_orders(group: FiniteGroup) -> np.ndarray:
    orders = []
    for element in range(group.order):
        product = 0
        for order in range(1, group.order + 1):
            product = int(group.table[product, element])
            if product == 0:
                orders.append(order)
                break
        else:
            raise ValueError(f"could not find the order of {group.key} element {element}")
    return np.asarray(orders, dtype=np.int64)


def a5_orthogonal_irrep(group: FiniteGroup, branch: int = 0) -> np.ndarray:
    """Construct either exact real 3D A5 irrep from the regular representation.

    The degree-three character projects the 60D left-regular representation
    onto its 9D isotypic component. A symmetric right-regular operator, which
    commutes with every left action, then selects one invariant 3D copy.
    """
    if group.key != "a5" or group.order != 60:
        raise ValueError("the character construction is defined only for A5")
    if branch not in (0, 1):
        raise ValueError("A5 degree-three irrep branch must be 0 or 1")
    inverses = _element_inverses(group)
    orders = _element_orders(group)

    unused = set(range(group.order))
    conjugacy_classes: list[list[int]] = []
    while unused:
        element = min(unused)
        conjugates = {
            int(group.table[group.table[h, element], inverses[h]])
            for h in range(group.order)
        }
        conjugacy_classes.append(sorted(conjugates))
        unused -= conjugates

    character = np.zeros(group.order, dtype=np.float64)
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    phi_conjugate = (1.0 - math.sqrt(5.0)) / 2.0
    five_cycle_values = (phi, phi_conjugate)
    if branch == 1:
        five_cycle_values = tuple(reversed(five_cycle_values))
    five_cycle_values = iter(five_cycle_values)
    for conjugacy_class in conjugacy_classes:
        order = int(orders[conjugacy_class[0]])
        value = {1: 3.0, 2: -1.0, 3: 0.0}.get(order)
        if order == 5:
            value = next(five_cycle_values)
        if value is None:
            raise ValueError(f"unexpected A5 conjugacy class of order {order}")
        character[conjugacy_class] = value

    left_regular = np.zeros((group.order, group.order, group.order), dtype=np.float64)
    right_regular = np.zeros_like(left_regular)
    for element in range(group.order):
        for state in range(group.order):
            left_regular[element, group.table[element, state], state] = 1.0
            right_regular[element, group.table[state, element], state] = 1.0

    projector = (3.0 / group.order) * np.einsum(
        "g,gij->ij", character[inverses], left_regular
    )
    eigenvalues, eigenvectors = np.linalg.eigh(projector)
    isotypic_basis = eigenvectors[:, eigenvalues > 0.5]
    if isotypic_basis.shape != (group.order, 9):
        raise RuntimeError("A5 character projector did not produce rank nine")

    coefficients = np.random.default_rng(2).normal(size=group.order)
    commuting = np.einsum(
        "g,gij->ij",
        coefficients,
        right_regular + right_regular.transpose(0, 2, 1),
    )
    restricted = isotypic_basis.T @ commuting @ isotypic_basis
    _, multiplicity_vectors = np.linalg.eigh(restricted)
    irrep_basis = isotypic_basis @ multiplicity_vectors[:, :3]
    representation = np.stack(
        [irrep_basis.T @ action @ irrep_basis for action in left_regular]
    )
    return representation


def _rotation_matrix_to_bivector(
    matrix: np.ndarray, max_rotor_angle: float
) -> np.ndarray:
    """Invert ``rotor_from_bivector`` for a proper 3D rotation matrix."""
    trace = float(np.trace(matrix))
    cosine = float(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-10:
        return np.zeros(3, dtype=np.float64)
    if not angle < max_rotor_angle:
        raise ValueError("rotation angle must be strictly below max_rotor_angle")
    axis = np.asarray(
        (
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ),
        dtype=np.float64,
    ) / (2.0 * math.sin(angle))
    # The Cl(3) basis is [e12,e13,e23], while the quaternion vector basis is
    # [e23,-e13,e12]. ``rotor_from_bivector`` also carries a leading minus.
    bivector_direction = np.asarray((axis[2], -axis[1], axis[0]))
    magnitude = np.arctanh(angle / max_rotor_angle)
    return magnitude * bivector_direction


def _rotation_matrix_to_householders(matrix: np.ndarray) -> np.ndarray:
    """Factor a proper 3D rotation as ``H(v) @ H(u)``."""
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    axis_candidates = np.flatnonzero(np.abs(eigenvalues - 1.0) < 1e-7)
    if len(axis_candidates) != 1:
        raise ValueError("proper 3D rotation must have one real fixed axis")
    axis = np.real(eigenvectors[:, axis_candidates[0]])
    axis /= np.linalg.norm(axis)
    coordinate = np.eye(3)[np.argmin(np.abs(axis))]
    first = np.cross(axis, coordinate)
    first /= np.linalg.norm(first)
    first_reflection = np.eye(3) - 2.0 * np.outer(first, first)
    second_reflection = matrix @ first_reflection
    second_reflection = 0.5 * (second_reflection + second_reflection.T)
    values, vectors = np.linalg.eigh(second_reflection)
    second = vectors[:, np.argmin(values)]
    reconstructed = (
        np.eye(3) - 2.0 * np.outer(second, second)
    ) @ first_reflection
    if not np.allclose(reconstructed, matrix, rtol=1e-7, atol=1e-7):
        raise RuntimeError("two-reflection factorization failed")
    result = np.zeros((2, GA_DIM), dtype=np.float64)
    result[0, :3] = first
    result[1, :3] = second
    return result


def initialize_from_a5_irrep(
    model: PureGroupActionModel,
    group: FiniteGroup,
    input_elements: tuple[int, ...],
) -> None:
    """Initialize rotor or Householder actions with the exact A5 anti-representation."""
    if model.family not in ("pure_ga_rotor", "pure_householder"):
        raise ValueError(
            "A5 irrep initialization requires pure_ga_rotor or pure_householder"
        )
    representation = a5_orthogonal_irrep(group)
    inverses = _element_inverses(group)
    if model.family == "pure_ga_rotor":
        parameters = np.stack(
            [
                _rotation_matrix_to_bivector(
                    representation[inverses[element]], model.max_rotor_angle
                )
                for element in input_elements
            ]
        )
    else:
        parameters = np.stack(
            [
                _rotation_matrix_to_householders(
                    representation[inverses[element]]
                )
                for element in input_elements
            ]
        )
    values = torch.as_tensor(
        parameters,
        dtype=model.action_parameters.dtype,
        device=model.action_parameters.device,
    ).unsqueeze(1).expand(-1, model.channels, *parameters.shape[1:])
    with torch.no_grad():
        model.action_parameters.copy_(values)


def _compose_word(action_matrices: torch.Tensor, word: tuple[int, ...]) -> torch.Tensor:
    channels = action_matrices.shape[1]
    result = torch.eye(
        GA_DIM, dtype=action_matrices.dtype, device=action_matrices.device
    ).expand(channels, -1, -1)
    for token in word:
        result = action_matrices[token] @ result
    return result


def _power_mean_squared_error(
    squared_errors: torch.Tensor, power: float
) -> torch.Tensor:
    """Stable squared-unit generalized mean with an exact zero value.

    For powers above two, ``mean(x**(p/2))**(2/p)`` has a singular outer
    derivative at an all-zero identity initialization. The shifted form keeps
    that boundary finite without changing nonzero experiment-scale values.
    """
    epsilon = squared_errors.new_tensor(1e-12)
    exponent = 2.0 / power
    moment = squared_errors.pow(power / 2.0).mean()
    return (moment + epsilon).pow(exponent) - epsilon.pow(exponent)


def algebraic_objectives(
    model: PureGroupActionModel,
    group: FiniteGroup,
    input_elements: tuple[int, ...],
    relation_loss_power: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Cayley-closure and faithful-canonical-orbit objectives.

    This uses the known finite-group table and is therefore an algebraic upper-
    bound objective, not ordinary sequence supervision. At power two, the
    closure normalization makes its square root directly comparable to
    ``cayley_edge_relation_rms``; higher powers emphasize the error tail while
    preserving squared-error units. Canonical-orbit cross entropy prevents the
    trivial all-identity representation from minimizing closure by itself.
    """
    actions = model.action_matrices()
    words = canonical_group_words(group, input_elements)
    operators = torch.stack([_compose_word(actions, word) for word in words])
    squared_errors = []
    for token, generator in enumerate(input_elements):
        products = torch.as_tensor(
            group.table[:, generator], dtype=torch.long, device=operators.device
        )
        difference = actions[token].unsqueeze(0) @ operators - operators[products]
        squared_errors.append(difference.square().sum(dim=(-2, -1)) / GA_DIM)
    squared_errors_tensor = torch.stack(squared_errors)
    # p=2 recovers mean squared Frobenius error exactly. Higher powers retain
    # squared-error units while concentrating gradient on the tail, which is
    # more relevant to worst-case error accumulation under long products.
    relation_loss = _power_mean_squared_error(
        squared_errors_tensor, relation_loss_power
    )
    initial = _normalized_state(model.initial_orbit_state)
    prototype_states = torch.einsum("gcij,cj->gci", operators, initial)
    canonical_logits = model.decode(prototype_states)
    canonical_targets = torch.arange(group.order, device=canonical_logits.device)
    canonical_loss = nn.functional.cross_entropy(canonical_logits, canonical_targets)
    return relation_loss, canonical_loss


def cayley_relation_loss(
    model: PureGroupActionModel,
    group: FiniteGroup,
    input_elements: tuple[int, ...],
) -> torch.Tensor:
    return algebraic_objectives(model, group, input_elements)[0]


def path_holonomy_objectives(
    model: PureGroupActionModel,
    group: FiniteGroup,
    input_elements: tuple[int, ...],
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    word_multiplier: int = 4,
    batch_size: int = 64,
    loss_power: float = 8.0,
    margin_target: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compare long alternate paths with canonical operators of the same label.

    Holonomy alone admits the all-identity solution. ``separation_loss`` must
    therefore be used together with ordinary supervised task loss; it includes
    canonical and alternate-path margins so non-canonical paths cannot collapse
    invisibly behind separated canonical prototypes.
    """
    if word_multiplier < 1 or batch_size < 1:
        raise ValueError("holonomy word multiplier and batch size must be positive")
    if loss_power < 2:
        raise ValueError("holonomy loss power must be at least 2")
    if margin_target <= 0:
        raise ValueError("holonomy margin target must be positive")
    selected_count = min(batch_size, tokens.shape[0])
    base_tokens = tokens[:selected_count]
    base_targets = targets[:selected_count, -1]
    path_segments = []
    segment_targets = []
    for segment in range(word_multiplier):
        path_segments.append(torch.roll(base_tokens, shifts=-segment, dims=0))
        segment_targets.append(torch.roll(base_targets, shifts=-segment, dims=0))
    path_tokens = torch.cat(path_segments, dim=1)
    table = torch.as_tensor(group.table, dtype=torch.long, device=tokens.device)
    path_targets = torch.zeros(selected_count, dtype=torch.long, device=tokens.device)
    for target in segment_targets:
        path_targets = table[path_targets, target]

    actions = model.action_matrices()
    operators = torch.stack(
        [
            _compose_word(actions, word)
            for word in canonical_group_words(group, input_elements)
        ]
    )
    path_operators = torch.eye(
        GA_DIM, dtype=actions.dtype, device=actions.device
    ).expand(selected_count, model.channels, -1, -1)
    for position in range(path_tokens.shape[1]):
        path_operators = actions[path_tokens[:, position]] @ path_operators
    canonical_path_operators = operators[path_targets]
    difference = path_operators - canonical_path_operators
    if model.family == "pure_ga_rotor":
        active_indices = torch.arange(1, GA_DIM - 1, device=actions.device)
    else:
        active_indices = torch.arange(GA_DIM, device=actions.device)
    squared_path_errors = (
        difference[..., active_indices].square().sum(dim=(-2, -1))
        / len(active_indices)
    )
    holonomy_loss = _power_mean_squared_error(squared_path_errors, loss_power)

    initial = _normalized_state(model.initial_orbit_state)
    canonical_states = torch.einsum("gcij,cj->gci", operators, initial)
    path_states = torch.einsum("bcij,cj->bci", path_operators, initial)
    canonical_flat = canonical_states.flatten(1)
    path_flat = path_states.flatten(1)
    canonical_squared_distances = torch.cdist(
        canonical_flat, canonical_flat
    ).square()
    off_diagonal = ~torch.eye(
        group.order, dtype=torch.bool, device=actions.device
    )
    margin_squared = margin_target * margin_target
    canonical_margin_loss = nn.functional.relu(
        margin_squared - canonical_squared_distances[off_diagonal]
    ).mean()

    path_to_canonical_squared = torch.cdist(path_flat, canonical_flat).square()
    positive_squared = path_to_canonical_squared.gather(
        1, path_targets.unsqueeze(1)
    ).squeeze(1)
    negative_squared = path_to_canonical_squared.clone()
    negative_squared.scatter_(1, path_targets.unsqueeze(1), torch.inf)
    nearest_negative_squared = negative_squared.min(dim=1).values
    alternate_path_margin_loss = nn.functional.relu(
        margin_squared + positive_squared - nearest_negative_squared
    ).mean()
    separation_loss = canonical_margin_loss + alternate_path_margin_loss
    diagnostics = {
        "path_holonomy_rms": squared_path_errors.mean().sqrt(),
        "path_holonomy_max": squared_path_errors.max().sqrt(),
        "canonical_minimum_margin": canonical_squared_distances[off_diagonal]
        .min()
        .sqrt(),
        "alternate_mean_target_distance": positive_squared.mean().sqrt(),
        "alternate_minimum_nearest_negative_margin": (
            nearest_negative_squared - positive_squared
        ).min(),
    }
    return holonomy_loss, separation_loss, diagnostics


@torch.no_grad()
def representation_diagnostics(
    model: PureGroupActionModel,
    group: FiniteGroup,
    input_elements: tuple[int, ...],
) -> dict[str, object]:
    """Measure whether token actions form the requested finite-group action."""
    actions = model.action_matrices().double()
    words = canonical_group_words(group, input_elements)
    operators = torch.stack([_compose_word(actions, word) for word in words])
    identity = torch.eye(GA_DIM, dtype=operators.dtype, device=operators.device)
    orthogonality = actions.transpose(-1, -2) @ actions - identity
    orthogonality_errors = orthogonality.square().sum(dim=(-2, -1)).sqrt() / math.sqrt(
        GA_DIM
    )

    edge_errors = []
    for token, generator in enumerate(input_elements):
        products = torch.as_tensor(
            group.table[:, generator], dtype=torch.long, device=operators.device
        )
        difference = actions[token].unsqueeze(0) @ operators - operators[products]
        edge_errors.append(
            difference.square().sum(dim=(-2, -1)).sqrt() / math.sqrt(GA_DIM)
        )
    edge_errors_tensor = torch.stack(edge_errors)

    products = torch.as_tensor(
        group.table, dtype=torch.long, device=operators.device
    )
    composed = operators.unsqueeze(0) @ operators.unsqueeze(1)
    homomorphism_difference = composed - operators[products]
    homomorphism = (
        homomorphism_difference.square().sum(dim=(-2, -1)).sqrt()
        / math.sqrt(GA_DIM)
    )
    if model.family == "pure_ga_rotor":
        channel_homomorphism_difference = homomorphism_difference[..., 1:7, 1:7]
        channel_homomorphism_dimension = 6
    else:
        channel_homomorphism_difference = homomorphism_difference
        channel_homomorphism_dimension = GA_DIM
    channel_homomorphism = (
        channel_homomorphism_difference.square().sum(dim=(-2, -1)).sqrt()
        / math.sqrt(channel_homomorphism_dimension)
    )
    per_channel_homomorphism_rms = [
        float(value)
        for value in channel_homomorphism.square().mean(dim=(0, 1)).sqrt()
    ]
    per_channel_homomorphism_max = [
        float(value)
        for value in channel_homomorphism.amax(dim=(0, 1))
    ]

    initial = _normalized_state(model.initial_orbit_state).double()
    prototype_states = torch.einsum("gcij,cj->gci", operators, initial)
    prototypes = prototype_states.flatten(1)
    distances = torch.cdist(prototypes, prototypes)
    off_diagonal = distances[~torch.eye(group.order, dtype=torch.bool, device=distances.device)]

    centered_decoder_weight = (
        model.output_head.weight.double()
        - model.output_head.weight.double().mean(dim=0, keepdim=True)
    ).reshape(group.order, model.channels, GA_DIM)
    decoder_contrast_norms = centered_decoder_weight.square().sum(dim=(0, 2)).sqrt()
    decoder_contrast_energy = decoder_contrast_norms.square()
    decoder_contrast_energy_fraction = (
        decoder_contrast_energy / decoder_contrast_energy.sum().clamp_min(1e-12)
    )
    centered_prototype_states = prototype_states - prototype_states.mean(
        dim=0, keepdim=True
    )
    channel_orbit_variation_energy = centered_prototype_states.square().sum(
        dim=(0, 2)
    )
    channel_orbit_variation_energy_fraction = (
        channel_orbit_variation_energy
        / channel_orbit_variation_energy.sum().clamp_min(1e-12)
    )
    only_channel_canonical_accuracy = []
    leave_one_channel_out_canonical_accuracy = []
    canonical_targets = torch.arange(group.order, device=prototypes.device)
    for channel in range(model.channels):
        only_channel = torch.zeros_like(prototype_states)
        only_channel[:, channel] = prototype_states[:, channel]
        without_channel = prototype_states.clone()
        without_channel[:, channel] = 0.0
        only_channel_canonical_accuracy.append(
            float(
                (
                    model.decode(only_channel.to(model.initial_orbit_state.dtype)).argmax(-1)
                    == canonical_targets
                ).float().mean()
            )
        )
        leave_one_channel_out_canonical_accuracy.append(
            float(
                (
                    model.decode(without_channel.to(model.initial_orbit_state.dtype)).argmax(-1)
                    == canonical_targets
                ).float().mean()
            )
        )

    inverse = []
    for element in range(group.order):
        candidates = np.flatnonzero(group.table[element] == 0)
        if len(candidates) != 1:
            raise ValueError(f"{group.key} element {element} has no unique inverse")
        inverse.append(int(candidates[0]))
    inverse_indices = torch.as_tensor(
        inverse, dtype=torch.long, device=operators.device
    )
    round_trip = operators[inverse_indices] @ operators
    returned = torch.einsum("gcij,cj->gci", round_trip, initial)
    identity_drift = (returned - initial).flatten(1).norm(dim=-1) / initial.norm().clamp_min(1e-12)

    orbit_edge_errors = []
    orbit_edge_correct = 0
    orbit_edge_total = 0
    for token, generator in enumerate(input_elements):
        edge_products = torch.as_tensor(
            group.table[:, generator], dtype=torch.long, device=operators.device
        )
        transitioned = torch.einsum(
            "cij,gcj->gci", actions[token], prototype_states
        )
        orbit_edge_errors.append(
            (transitioned - prototype_states[edge_products]).flatten(1).norm(dim=-1)
            / initial.norm().clamp_min(1e-12)
        )
        predictions = model.decode(
            transitioned.to(model.initial_orbit_state.dtype)
        ).argmax(dim=-1)
        orbit_edge_correct += int((predictions == edge_products).sum())
        orbit_edge_total += group.order
    orbit_edge_error = torch.stack(orbit_edge_errors)

    composed_orbit = torch.einsum(
        "hcij,gcj->ghci", operators, prototype_states
    )
    expected_orbit = prototype_states[products]
    orbit_homomorphism = (
        (composed_orbit - expected_orbit).flatten(2).norm(dim=-1)
        / initial.norm().clamp_min(1e-12)
    )
    orbit_predictions = model.decode(
        composed_orbit.to(model.initial_orbit_state.dtype)
    ).argmax(dim=-1)
    decoded_homomorphism_accuracy = float(
        (orbit_predictions == products).float().mean()
    )

    full_operators = torch.stack(
        [torch.block_diag(*operator.unbind(dim=0)) for operator in operators]
    )
    full_difference = (
        full_operators.unsqueeze(0) @ full_operators.unsqueeze(1)
        - full_operators[products]
    )
    state_dimension = prototypes.shape[1]

    # Report every restricted error as RMS output error per input-subspace
    # dimension.  This makes the full, reachable, complementary, and fixed
    # sectors comparable even when their ranks differ.
    common_fixed_constraint = torch.cat(
        [operator - torch.eye(state_dimension, dtype=operator.dtype, device=operator.device)
         for operator in full_operators[
             torch.as_tensor(input_elements, dtype=torch.long, device=operators.device)
         ]],
        dim=0,
    )
    _, fixed_singular_values, fixed_right_vectors = torch.linalg.svd(
        common_fixed_constraint, full_matrices=True
    )
    fixed_tolerance = (
        max(common_fixed_constraint.shape)
        * torch.finfo(common_fixed_constraint.dtype).eps
        * fixed_singular_values.max().clamp_min(1.0)
    )
    # Operators are generated by the float32 training model and promoted to
    # float64 for diagnostics; use a source-precision-aware numerical nullspace.
    fixed_tolerance = torch.maximum(
        fixed_tolerance,
        fixed_singular_values.max().clamp_min(1.0) * 1e-6,
    )
    fixed_rank = state_dimension - int((fixed_singular_values > fixed_tolerance).sum())
    if fixed_rank:
        fixed_basis = fixed_right_vectors[-fixed_rank:].transpose(0, 1)
        fixed_residual = (
            (full_difference @ fixed_basis).square().sum(dim=(-2, -1)).sqrt()
            / math.sqrt(fixed_rank)
        )
        fixed_rms: float | None = float(fixed_residual.square().mean().sqrt())
        fixed_max: float | None = float(fixed_residual.max())
    else:
        fixed_rms = None
        fixed_max = None

    cl3_invariant_dimension = 0
    cl3_invariant_rms: float | None = None
    cl3_invariant_max: float | None = None
    if model.family == "pure_ga_rotor":
        invariant_indices = [
            channel * GA_DIM + grade_index
            for channel in range(model.channels)
            for grade_index in (0, GA_DIM - 1)
        ]
        cl3_invariant_dimension = len(invariant_indices)
        invariant_basis = torch.eye(
            state_dimension, dtype=full_difference.dtype, device=full_difference.device
        )[:, invariant_indices]
        invariant_residual = (
            (full_difference @ invariant_basis).square().sum(dim=(-2, -1)).sqrt()
            / math.sqrt(cl3_invariant_dimension)
        )
        cl3_invariant_rms = float(invariant_residual.square().mean().sqrt())
        cl3_invariant_max = float(invariant_residual.max())

    _, singular_values, right_vectors = torch.linalg.svd(
        prototypes, full_matrices=True
    )
    tolerance = (
        max(prototypes.shape)
        * torch.finfo(prototypes.dtype).eps
        * singular_values.max().clamp_min(1.0)
    )
    tolerance = torch.maximum(
        tolerance, singular_values.max().clamp_min(1.0) * 1e-6
    )
    reachable_rank = int((singular_values > tolerance).sum())
    reachable_basis = right_vectors[:reachable_rank].transpose(0, 1)
    reachable_residual = (
        (full_difference @ reachable_basis).square().sum(dim=(-2, -1)).sqrt()
        / math.sqrt(reachable_rank)
    )
    complement_rank = prototypes.shape[1] - reachable_rank
    if complement_rank:
        complement_basis = right_vectors[reachable_rank:].transpose(0, 1)
        complement_residual = (
            (full_difference @ complement_basis).square().sum(dim=(-2, -1)).sqrt()
            / math.sqrt(complement_rank)
        )
        complement_rms: float | None = float(
            complement_residual.square().mean().sqrt()
        )
        complement_max: float | None = float(complement_residual.max())
    else:
        complement_rms = None
        complement_max = None

    centered_prototypes = prototypes - prototypes.mean(dim=0, keepdim=True)
    _, variation_singular_values, variation_right_vectors = torch.linalg.svd(
        centered_prototypes, full_matrices=True
    )
    variation_tolerance = (
        max(centered_prototypes.shape)
        * torch.finfo(centered_prototypes.dtype).eps
        * variation_singular_values.max().clamp_min(1.0)
    )
    variation_tolerance = torch.maximum(
        variation_tolerance,
        variation_singular_values.max().clamp_min(1.0) * 1e-6,
    )
    variation_rank = int((variation_singular_values > variation_tolerance).sum())
    if variation_rank:
        variation_basis = variation_right_vectors[:variation_rank].transpose(0, 1)
        variation_residual = (
            (full_difference @ variation_basis).square().sum(dim=(-2, -1)).sqrt()
            / math.sqrt(variation_rank)
        )
        variation_rms: float | None = float(
            variation_residual.square().mean().sqrt()
        )
        variation_max: float | None = float(variation_residual.max())
    else:
        variation_rms = None
        variation_max = None

    commutator_values = []
    commutator_pairs = []
    commutator_values_by_channel = []
    for left in range(len(input_elements)):
        for right in range(left + 1, len(input_elements)):
            commutator = actions[right] @ actions[left] - actions[left] @ actions[right]
            values_by_channel = (
                commutator.square().sum(dim=(-2, -1)).sqrt() / math.sqrt(GA_DIM)
            )
            value = values_by_channel.mean()
            commutator_values.append(value)
            commutator_pairs.append((left, right))
            commutator_values_by_channel.append(values_by_channel)
    if commutator_values:
        commutators = torch.stack(commutator_values)
        maximum_index = int(commutators.argmax())
        commutator_separation = float(commutators[maximum_index])
        commutator_mean = float(commutators.mean())
        commutator_pair = commutator_pairs[maximum_index]
        commutators_by_channel = torch.stack(commutator_values_by_channel)
        per_channel_maximum_commutator = [
            float(value) for value in commutators_by_channel.max(dim=0).values
        ]
        pairwise_commutators_by_channel = [
            {
                "token_pair": list(pair),
                "per_channel": [float(value) for value in values],
            }
            for pair, values in zip(
                commutator_pairs, commutator_values_by_channel, strict=True
            )
        ]
    else:
        commutator_separation = 0.0
        commutator_mean = 0.0
        commutator_pair = (-1, -1)
        per_channel_maximum_commutator = []
        pairwise_commutators_by_channel = []

    # Exact short relators expose coherent finite-order defects that can be
    # invisible at a handful of sequence lengths.  Derive the orders from the
    # actual finite-group table rather than assuming a particular A5
    # presentation or multiplication convention.
    identity_element = next(
        index
        for index in range(group.order)
        if np.array_equal(group.table[index], np.arange(group.order))
    )

    def element_order(element: int) -> int:
        product = identity_element
        for order in range(1, group.order + 1):
            product = int(group.table[product, element])
            if product == identity_element:
                return order
        raise RuntimeError(f"element {element} has no finite order")

    relator_sector = slice(1, 7) if model.family == "pure_ga_rotor" else slice(None)
    relator_dimension = 6 if model.family == "pure_ga_rotor" else GA_DIM
    relator_identity = torch.eye(
        relator_dimension, dtype=actions.dtype, device=actions.device
    )
    finite_order_relators = []
    for token, element in enumerate(input_elements):
        order = element_order(element)
        residual = (
            torch.linalg.matrix_power(actions[token], order)[
                ..., relator_sector, relator_sector
            ]
            - relator_identity
        ).square().sum(dim=(-2, -1)).sqrt() / math.sqrt(relator_dimension)
        finite_order_relators.append(
            {
                "tokens": [token],
                "group_product_order": order,
                "mean_rms": float(residual.mean()),
                "maximum_channel_rms": float(residual.max()),
                "per_channel_rms": [float(value) for value in residual],
            }
        )
    for left in range(len(input_elements)):
        for right in range(left + 1, len(input_elements)):
            product_element = int(
                group.table[input_elements[left], input_elements[right]]
            )
            order = element_order(product_element)
            product_action = actions[right] @ actions[left]
            residual = (
                torch.linalg.matrix_power(product_action, order)[
                    ..., relator_sector, relator_sector
                ]
                - relator_identity
            ).square().sum(dim=(-2, -1)).sqrt() / math.sqrt(relator_dimension)
            finite_order_relators.append(
                {
                    "tokens": [left, right],
                    "group_product_order": order,
                    "mean_rms": float(residual.mean()),
                    "maximum_channel_rms": float(residual.max()),
                    "per_channel_rms": [float(value) for value in residual],
                }
            )

    decoded = model.decode(prototype_states.to(model.initial_orbit_state.dtype))
    canonical_accuracy = float(
        (decoded.argmax(dim=-1) == torch.arange(group.order, device=decoded.device))
        .float()
        .mean()
    )
    identity_matrix = torch.eye(
        GA_DIM, dtype=actions.dtype, device=actions.device
    )
    action_displacement = (
        (actions - identity_matrix).square().sum(dim=(-2, -1)).sqrt()
        / math.sqrt(GA_DIM)
    )
    action_metrics: dict[str, object] = {
        "mean_action_displacement_from_identity": float(action_displacement.mean()),
        "max_action_displacement_from_identity": float(action_displacement.max()),
        "per_token_mean_action_displacement": [
            float(value) for value in action_displacement.mean(dim=-1)
        ],
    }
    if model.family == "pure_ga_rotor":
        token_ids = torch.arange(model.vocab_size, device=actions.device)
        rotors = model.token_actions(token_ids)
        angles = 2.0 * torch.acos(rotors[..., 0].clamp(-1.0, 1.0))
        # Cl(3) bivectors occupy blade coordinates e12/e13/e23 (4:7) in the
        # repository's canonical blade ordering.  Report that actual 3-vector
        # axis, not the surrounding structurally-zero even/odd coordinates.
        bivector_coordinates = rotors[..., 4:7]
        bivector_norms = bivector_coordinates.norm(dim=-1, keepdim=True)
        axes = torch.where(
            bivector_norms > 1e-8,
            bivector_coordinates / bivector_norms.clamp_min(1e-8),
            torch.zeros_like(bivector_coordinates),
        )
        generator_angle_ratios = []
        if model.vocab_size >= 3:
            for channel in range(model.channels):
                numerator = float(angles[0, channel])
                denominator = float(angles[2, channel])
                ratio = numerator / denominator if denominator > 1e-8 else None
                nearest = None
                if ratio is not None:
                    candidates = [
                        (abs(ratio - p / q), p, q)
                        for p in range(1, 9)
                        for q in range(1, 9)
                    ]
                    error, numerator_integer, denominator_integer = min(candidates)
                    nearest = {
                        "numerator": numerator_integer,
                        "denominator": denominator_integer,
                        "value": numerator_integer / denominator_integer,
                        "absolute_error": error,
                    }
                generator_angle_ratios.append(
                    {
                        "channel": channel,
                        "token_0_over_token_2": ratio,
                        "nearest_positive_rational_with_terms_at_most_8": nearest,
                        "axis_dot_product": float(
                            torch.dot(axes[0, channel], axes[2, channel])
                        ),
                    }
                )
        action_metrics.update(
            {
                "mean_rotor_angle_radians": float(angles.mean()),
                "max_rotor_angle_radians": float(angles.max()),
                "per_token_mean_rotor_angle_radians": [
                    float(value) for value in angles.mean(dim=-1)
                ],
                "per_token_per_channel_rotor_angle_radians": angles.tolist(),
                "per_token_per_channel_rotor_axis": axes.tolist(),
                "generator_angle_ratio_diagnostics": generator_angle_ratios,
            }
        )
    elif model.family == "pure_quaternion_spinor":
        token_ids = torch.arange(model.vocab_size, device=actions.device)
        quaternions = model.token_actions(token_ids)
        angles = 2.0 * torch.acos(quaternions[..., 0].clamp(-1.0, 1.0))
        vector = quaternions[..., 1:4]
        vector_norm = vector.norm(dim=-1, keepdim=True)
        axes = torch.where(
            vector_norm > 1e-8,
            vector / vector_norm.clamp_min(1e-8),
            torch.zeros_like(vector),
        )
        spinor_metrics: dict[str, object] = {
            "mean_spinor_action_angle_radians": float(angles.mean()),
            "max_spinor_action_angle_radians": float(angles.max()),
            "per_token_per_channel_spinor_angle_radians": angles.tolist(),
            "per_token_per_channel_spinor_axis": axes.tolist(),
        }
        if model.vocab_size >= 4:
            target_center = torch.zeros_like(quaternions[0])
            target_center[..., 0] = -1.0
            squares = torch.stack(
                [quaternion_product(quaternions[index], quaternions[index])
                 for index in range(4)]
            )
            inverse_pair_residual = torch.stack(
                (
                    quaternions[0] + quaternions[1],
                    quaternions[2] + quaternions[3],
                )
            )
            square_residual = squares - target_center
            generator_anticommutator = (
                quaternion_product(quaternions[0], quaternions[2])
                + quaternion_product(quaternions[2], quaternions[0])
            )
            spinor_metrics.update(
                {
                    "q8_inverse_pair_antipodal_rms": float(
                        inverse_pair_residual.square().mean().sqrt()
                    ),
                    "q8_generator_square_to_minus_identity_rms": float(
                        square_residual.square().mean().sqrt()
                    ),
                    "q8_generator_anticommutator_rms": float(
                        generator_anticommutator.square().mean().sqrt()
                    ),
                    "per_channel_q8_inverse_pair_antipodal_rms": [
                        float(value)
                        for value in inverse_pair_residual.square()
                        .mean(dim=(0, 2)).sqrt()
                    ],
                    "per_channel_q8_generator_square_to_minus_identity_rms": [
                        float(value)
                        for value in square_residual.square()
                        .mean(dim=(0, 2)).sqrt()
                    ],
                    "per_channel_q8_generator_anticommutator_rms": [
                        float(value)
                        for value in generator_anticommutator.square()
                        .mean(dim=-1).sqrt()
                    ],
                }
            )
        action_metrics.update(spinor_metrics)
    return {
        "canonical_max_word_length": max(map(len, words)),
        "operator_orthogonality_rms": float(orthogonality_errors.square().mean().sqrt()),
        "operator_orthogonality_max": float(orthogonality_errors.max()),
        "cayley_edge_relation_rms": float(edge_errors_tensor.square().mean().sqrt()),
        "cayley_edge_relation_max": float(edge_errors_tensor.max()),
        "linear_homomorphism_rms": float(homomorphism.square().mean().sqrt()),
        "linear_homomorphism_max": float(homomorphism.max()),
        "per_channel_active_homomorphism_rms": per_channel_homomorphism_rms,
        "per_channel_active_homomorphism_max": per_channel_homomorphism_max,
        "identity_word_state_drift_rms": float(identity_drift.square().mean().sqrt()),
        "identity_word_state_drift_max": float(identity_drift.max()),
        "orbit_cayley_edge_rms": float(orbit_edge_error.square().mean().sqrt()),
        "orbit_cayley_edge_max": float(orbit_edge_error.max()),
        "decoded_cayley_edge_accuracy": orbit_edge_correct / orbit_edge_total,
        "orbit_homomorphism_rms": float(orbit_homomorphism.square().mean().sqrt()),
        "orbit_homomorphism_max": float(orbit_homomorphism.max()),
        # Explicit name: this samples the error operator on canonical state
        # directions. It is not an orthogonal projection onto the orbit span.
        "canonical_orbit_directional_homomorphism_rms": float(
            orbit_homomorphism.square().mean().sqrt()
        ),
        "canonical_orbit_directional_homomorphism_max": float(
            orbit_homomorphism.max()
        ),
        "decoded_homomorphism_accuracy": decoded_homomorphism_accuracy,
        "reachable_linear_span_rank": reachable_rank,
        "state_dimension": state_dimension,
        "reachable_span_homomorphism_rms": float(
            reachable_residual.square().mean().sqrt()
        ),
        "reachable_span_homomorphism_max": float(reachable_residual.max()),
        "orthogonal_complement_dimension": complement_rank,
        "orthogonal_complement_homomorphism_rms": complement_rms,
        "orthogonal_complement_homomorphism_max": complement_max,
        "orbit_variation_span_rank": variation_rank,
        "orbit_variation_span_homomorphism_rms": variation_rms,
        "orbit_variation_span_homomorphism_max": variation_max,
        "common_fixed_subspace_dimension": fixed_rank,
        "common_fixed_constraint_min_singular_value": float(
            fixed_singular_values.min()
        ),
        "common_fixed_constraint_rank_tolerance": float(fixed_tolerance),
        "common_fixed_subspace_homomorphism_rms": fixed_rms,
        "common_fixed_subspace_homomorphism_max": fixed_max,
        "cl3_invariant_grade_dimension": cl3_invariant_dimension,
        "cl3_invariant_grade_homomorphism_rms": cl3_invariant_rms,
        "cl3_invariant_grade_homomorphism_max": cl3_invariant_max,
        "generator_commutator_separation": commutator_separation,
        "mean_generator_pair_commutator_separation": commutator_mean,
        "maximum_commutator_token_pair": list(commutator_pair),
        "per_channel_maximum_generator_commutator_separation": (
            per_channel_maximum_commutator
        ),
        "pairwise_generator_commutator_separation_by_channel": (
            pairwise_commutators_by_channel
        ),
        "finite_order_relator_residuals": finite_order_relators,
        "prototype_minimum_margin": float(off_diagonal.min()),
        "prototype_median_distance": float(off_diagonal.median()),
        "canonical_prototype_accuracy": canonical_accuracy,
        "per_channel_decoder_contrast_norm": [
            float(value) for value in decoder_contrast_norms
        ],
        "per_channel_decoder_contrast_energy_fraction": [
            float(value) for value in decoder_contrast_energy_fraction
        ],
        "per_channel_orbit_variation_energy_fraction": [
            float(value) for value in channel_orbit_variation_energy_fraction
        ],
        "only_channel_canonical_accuracy": only_channel_canonical_accuracy,
        "leave_one_channel_out_canonical_accuracy": (
            leave_one_channel_out_canonical_accuracy
        ),
        **action_metrics,
    }


@torch.no_grad()
def channel_ablation_final_accuracy(
    model: PureGroupActionModel,
    batches_by_length: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
    device: torch.device,
) -> dict[str, object]:
    """Measure causal channel sufficiency and necessity at every test length."""
    model.eval()
    output: dict[str, object] = {}
    for length, batches in batches_by_length.items():
        only_correct = [0] * model.channels
        without_correct = [0] * model.channels
        examples = 0
        for tokens, targets in batches:
            tokens = tokens.to(device)
            final_targets = targets[:, -1].to(device)
            _, final_state = model(tokens, return_recurrent_state=True)
            for channel in range(model.channels):
                only_channel = torch.zeros_like(final_state)
                only_channel[:, channel] = final_state[:, channel]
                without_channel = final_state.clone()
                without_channel[:, channel] = 0.0
                only_correct[channel] += int(
                    (model.decode(only_channel).argmax(-1) == final_targets).sum()
                )
                without_correct[channel] += int(
                    (model.decode(without_channel).argmax(-1) == final_targets).sum()
                )
            examples += len(tokens)
        output[str(length)] = {
            "only_channel_final_position_accuracy": [
                correct / examples for correct in only_correct
            ],
            "leave_one_channel_out_final_position_accuracy": [
                correct / examples for correct in without_correct
            ],
        }
    return output


@torch.no_grad()
def evaluate(
    model: PureGroupActionModel,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    prefix_correct = 0
    final_correct = 0
    prefix_examples = 0
    final_examples = 0
    norm_errors = []
    for tokens, targets in batches:
        tokens, targets = tokens.to(device), targets.to(device)
        logits, state = model(tokens, return_recurrent_state=True)
        total_loss += float(
            nn.functional.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), reduction="sum"
            )
        )
        predictions = logits.argmax(dim=-1)
        prefix_correct += int((predictions == targets).sum())
        final_correct += int((predictions[:, -1] == targets[:, -1]).sum())
        prefix_examples += targets.numel()
        final_examples += targets.shape[0]
        initial_norm = model.initial_state(tokens.shape[0]).norm(dim=-1)
        norm_errors.append(float((state.norm(dim=-1) - initial_norm).abs().max()))
    return (
        total_loss / prefix_examples,
        prefix_correct / prefix_examples,
        final_correct / final_examples,
        max(norm_errors),
    )


@torch.no_grad()
def streaming_equivalence(
    model: PureGroupActionModel, tokens: torch.Tensor
) -> dict[str, float]:
    full_logits, full_state = model(tokens, return_recurrent_state=True)
    split = max(1, tokens.shape[1] // 2)
    first_logits, state = model(tokens[:, :split], return_recurrent_state=True)
    second_logits, chunked_state = model(
        tokens[:, split:], state, return_recurrent_state=True
    )
    chunked_logits = torch.cat((first_logits, second_logits), dim=1)
    streamed = []
    state = None
    for position in range(tokens.shape[1]):
        logits, state = model(
            tokens[:, position : position + 1], state, return_recurrent_state=True
        )
        streamed.append(logits)
    streamed_logits = torch.cat(streamed, dim=1)
    return {
        "chunked_logit_max_abs_error": float((chunked_logits - full_logits).abs().max()),
        "streaming_logit_max_abs_error": float((streamed_logits - full_logits).abs().max()),
        "chunked_state_max_abs_error": float((chunked_state - full_state).abs().max()),
        "streaming_state_max_abs_error": float((state - full_state).abs().max()),
    }


def _parameter_fingerprint(model: nn.Module) -> tuple[str, str]:
    shapes = json.dumps(
        [(name, list(parameter.shape)) for name, parameter in model.named_parameters()],
        separators=(",", ":"),
    ).encode("utf-8")
    values = hashlib.sha256()
    for name, parameter in model.named_parameters():
        values.update(name.encode("utf-8"))
        values.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return hashlib.sha256(shapes).hexdigest(), values.hexdigest()


def parameter_gradient_diagnostics(
    model: PureGroupActionModel,
) -> dict[str, float]:
    def norm(parameter: torch.Tensor) -> float:
        if parameter.grad is None:
            return 0.0
        return float(parameter.grad.detach().float().norm())

    output_gradients = [
        parameter.grad.detach().float().norm().square()
        for parameter in model.output_head.parameters()
        if parameter.grad is not None
    ]
    return {
        "action_gradient_norm": norm(model.action_parameters),
        "initial_state_gradient_norm": norm(model.initial_orbit_state),
        "decoder_gradient_norm": (
            float(torch.stack(output_gradients).sum().sqrt())
            if output_gradients
            else 0.0
        ),
        "logit_scale_gradient_abs": (
            float(model.logit_scale.grad.detach().abs())
            if model.logit_scale.grad is not None
            else 0.0
        ),
    }


@torch.no_grad()
def trajectory_action_diagnostics(
    model: PureGroupActionModel,
) -> dict[str, float | list[float]]:
    """Cheap action diagnostics for dense optimization trajectories.

    Full A5 representation diagnostics are intentionally reserved for the
    endpoints: evaluating all 3,600 products at every logged training step
    obscures the optimization signal and dominates runtime.
    """
    actions = model.action_matrices().double()
    identity = torch.eye(GA_DIM, dtype=actions.dtype, device=actions.device)
    displacement = (
        (actions - identity).square().sum(dim=(-2, -1)).sqrt()
        / math.sqrt(GA_DIM)
    )
    initial = _normalized_state(model.initial_orbit_state).double()
    moved = torch.einsum("tcij,cj->tci", actions, initial)
    state_displacement = (moved - initial).norm(dim=-1)
    commutators = []
    for left in range(model.vocab_size):
        for right in range(left + 1, model.vocab_size):
            difference = (
                actions[right] @ actions[left]
                - actions[left] @ actions[right]
            )
            commutators.append(
                difference.square().sum(dim=(-2, -1)).sqrt().mean()
                / math.sqrt(GA_DIM)
            )
    result: dict[str, float | list[float]] = {
        "mean_action_displacement_from_identity": float(displacement.mean()),
        "max_action_displacement_from_identity": float(displacement.max()),
        "mean_initial_state_displacement": float(state_displacement.mean()),
        "max_initial_state_displacement": float(state_displacement.max()),
        "maximum_token_commutator": (
            float(torch.stack(commutators).max()) if commutators else 0.0
        ),
        "per_token_mean_action_displacement": [
            float(value) for value in displacement.mean(dim=-1)
        ],
    }
    if model.family == "pure_ga_rotor":
        token_ids = torch.arange(model.vocab_size, device=actions.device)
        rotors = model.token_actions(token_ids)
        angles = 2.0 * torch.acos(rotors[..., 0].clamp(-1.0, 1.0))
        result.update(
            {
                "mean_rotor_angle_radians": float(angles.mean()),
                "max_rotor_angle_radians": float(angles.max()),
                "per_token_mean_rotor_angle_radians": [
                    float(value) for value in angles.mean(dim=-1)
                ],
            }
        )
    return result


def run_variant(
    family: str,
    group: FiniteGroup,
    input_elements: tuple[int, ...],
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    validation_batches: list[tuple[torch.Tensor, torch.Tensor]],
    generalization_batches: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
    config: MechanismConfig,
    device: torch.device,
    checkpoint_output: Path | None = None,
) -> dict:
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = PureGroupActionModel(
        len(input_elements),
        group.order,
        family=family,
        channels=config.channels,
        max_rotor_angle=config.max_rotor_angle,
    ).to(device)
    if config.a5_irrep_init:
        initialize_from_a5_irrep(model, group, input_elements)
    if config.freeze_actions:
        model.action_parameters.requires_grad_(False)
    shape_hash, initial_hash = _parameter_fingerprint(model)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=1e-4,
    )
    initial_metrics = evaluate(model, validation_batches, device)
    initial_mechanism = representation_diagnostics(model, group, input_elements)
    trajectory = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    model.train()
    for step, (tokens, targets) in enumerate(train_batches, start=1):
        tokens, targets = tokens.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        prefix_task_loss = nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.flatten()
        )
        final_position_task_loss = nn.functional.cross_entropy(
            logits[:, -1], targets[:, -1]
        )
        task_loss = (
            prefix_task_loss
            + config.final_position_loss_weight * final_position_task_loss
        )
        algebraic_active = (
            (config.relation_loss_weight and step > config.relation_start_step)
            or config.canonical_orbit_loss_weight
        )
        if algebraic_active:
            relation_loss, canonical_orbit_loss = algebraic_objectives(
                model,
                group,
                input_elements,
                relation_loss_power=config.relation_loss_power,
            )
        else:
            relation_loss = task_loss.new_zeros(())
            canonical_orbit_loss = task_loss.new_zeros(())
        holonomy_active = bool(
            (config.holonomy_loss_weight or config.holonomy_margin_weight)
            and step > config.holonomy_start_step
        )
        if holonomy_active:
            holonomy_multipliers = (
                config.holonomy_word_multipliers
                or (config.holonomy_word_multiplier,)
            )
            holonomy_multiplier = holonomy_multipliers[
                (step - config.holonomy_start_step - 1)
                % len(holonomy_multipliers)
            ]
            holonomy_loss, holonomy_margin_loss, holonomy_diagnostics = (
                path_holonomy_objectives(
                    model,
                    group,
                    input_elements,
                    tokens,
                    targets,
                    word_multiplier=holonomy_multiplier,
                    batch_size=config.holonomy_batch_size,
                    loss_power=config.holonomy_loss_power,
                    margin_target=config.holonomy_margin_target,
                )
            )
            holonomy_ramp = (
                min(
                    1.0,
                    (step - config.holonomy_start_step)
                    / config.holonomy_ramp_steps,
                )
                if config.holonomy_ramp_steps
                else 1.0
            )
        else:
            holonomy_loss = task_loss.new_zeros(())
            holonomy_margin_loss = task_loss.new_zeros(())
            holonomy_diagnostics = {}
            holonomy_ramp = 0.0
            holonomy_multiplier = config.holonomy_word_multiplier
        if config.relation_loss_weight and step > config.relation_start_step:
            if config.relation_ramp_steps:
                ramp = min(
                    1.0,
                    (step - config.relation_start_step) / config.relation_ramp_steps,
                )
            else:
                ramp = 1.0
            effective_relation_weight = config.relation_loss_weight * ramp
        else:
            effective_relation_weight = 0.0
        loss = (
            task_loss
            + effective_relation_weight * relation_loss
            + config.canonical_orbit_loss_weight * canonical_orbit_loss
            + holonomy_ramp * config.holonomy_loss_weight * holonomy_loss
            + holonomy_ramp
            * config.holonomy_margin_weight
            * holonomy_margin_loss
        )
        loss.backward()
        record_diagnostic = bool(
            config.diagnostic_interval
            and (
                step == 1
                or step % config.diagnostic_interval == 0
                or step == config.steps
            )
        )
        gradient_components = (
            parameter_gradient_diagnostics(model) if record_diagnostic else {}
        )
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == config.steps:
            print(
                f"{family} step={step}/{config.steps} "
                f"loss={float(loss.detach()):.6f} "
                f"task={float(task_loss.detach()):.6f} "
                f"relation={float(relation_loss.detach()):.6f} "
                f"canonical={float(canonical_orbit_loss.detach()):.6f}"
            )
        if record_diagnostic:
            detached_logits = logits.detach()
            predictions = detached_logits.argmax(dim=-1)
            correct_logits = detached_logits.gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)
            competing_logits = detached_logits.clone()
            competing_logits.scatter_(-1, targets.unsqueeze(-1), -torch.inf)
            correct_class_margin = correct_logits - competing_logits.max(dim=-1).values
            trajectory[str(step)] = {
                "loss": float(loss.detach()),
                "task_loss": float(task_loss.detach()),
                "prefix_task_loss": float(prefix_task_loss.detach()),
                "final_position_task_loss": float(final_position_task_loss.detach()),
                "relation_loss": float(relation_loss.detach()),
                "canonical_orbit_loss": float(canonical_orbit_loss.detach()),
                "holonomy_loss": float(holonomy_loss.detach()),
                "holonomy_margin_loss": float(holonomy_margin_loss.detach()),
                "effective_relation_weight": effective_relation_weight,
                "effective_holonomy_loss_weight": (
                    holonomy_ramp * config.holonomy_loss_weight
                ),
                "effective_holonomy_margin_weight": (
                    holonomy_ramp * config.holonomy_margin_weight
                ),
                "holonomy": {
                    "word_multiplier": holonomy_multiplier,
                    **{
                        key: float(value.detach())
                        for key, value in holonomy_diagnostics.items()
                    },
                },
                "preclip_gradient_norm": float(gradient_norm),
                "training_prefix_accuracy": float(
                    (predictions == targets).float().mean()
                ),
                "training_final_position_accuracy": float(
                    (predictions[:, -1] == targets[:, -1]).float().mean()
                ),
                "training_median_correct_class_margin": float(
                    correct_class_margin.median()
                ),
                "training_final_position_median_correct_class_margin": float(
                    correct_class_margin[:, -1].median()
                ),
                **gradient_components,
                "mechanism": trajectory_action_diagnostics(model),
            }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    final_loss, prefix_accuracy, final_accuracy, norm_error = evaluate(
        model, validation_batches, device
    )
    length_generalization = {}
    for length, batches in generalization_batches.items():
        loss, prefix, final, length_norm_error = evaluate(model, batches, device)
        length_generalization[str(length)] = {
            "validation_loss": loss,
            "prefix_accuracy": prefix,
            "final_position_accuracy": final,
            "maximum_state_norm_error": length_norm_error,
        }
    with torch.no_grad():
        final_holonomy_by_multiplier = {}
        final_multipliers = sorted(
            set(
                (config.holonomy_word_multiplier,)
                + config.holonomy_word_multipliers
            )
        )
        for multiplier in final_multipliers:
            scale_loss, scale_margin_loss, scale_metrics = path_holonomy_objectives(
                model,
                group,
                input_elements,
                validation_batches[0][0].to(device),
                validation_batches[0][1].to(device),
                word_multiplier=multiplier,
                batch_size=config.holonomy_batch_size,
                loss_power=config.holonomy_loss_power,
                margin_target=config.holonomy_margin_target,
            )
            final_holonomy_by_multiplier[str(multiplier)] = {
                "loss": float(scale_loss),
                "separation_loss": float(scale_margin_loss),
                **{key: float(value) for key, value in scale_metrics.items()},
            }
        final_holonomy_metrics = final_holonomy_by_multiplier[
            str(config.holonomy_word_multiplier)
        ]
    probe = validation_batches[0][0][:8].to(device)
    if checkpoint_output is not None:
        checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "family": family,
                "group": group.key,
                "input_elements": input_elements,
                "config": asdict(config),
                "state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
            },
            checkpoint_output,
        )
    return {
        "family": family,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_shape_sha256": shape_hash,
        "initial_parameter_sha256": initial_hash,
        "state_scalars_per_sequence": config.channels * GA_DIM,
        "has_decay": False,
        "has_affine_write": False,
        "has_residual_or_feed_forward_path": False,
        "a5_irrep_initialization": config.a5_irrep_init,
        "actions_frozen": config.freeze_actions,
        "initial_validation_loss": initial_metrics[0],
        "initial_validation_prefix_accuracy": initial_metrics[1],
        "initial_validation_final_position_accuracy": initial_metrics[2],
        "initial_mechanism_diagnostics": initial_mechanism,
        "final_validation_loss": final_loss,
        "final_validation_prefix_accuracy": prefix_accuracy,
        "final_validation_final_position_accuracy": final_accuracy,
        "maximum_validation_state_norm_error": norm_error,
        "length_generalization": length_generalization,
        "mechanism_diagnostics": representation_diagnostics(model, group, input_elements),
        "channel_ablation_by_length": channel_ablation_final_accuracy(
            model, generalization_batches, device
        ),
        "path_holonomy_diagnostics": final_holonomy_metrics,
        "path_holonomy_diagnostics_by_multiplier": final_holonomy_by_multiplier,
        "training_trajectory": trajectory,
        "streaming_equivalence": streaming_equivalence(model, probe),
        "elapsed_seconds": elapsed,
        "steps_per_second": config.steps / elapsed,
        "peak_cuda_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=512)
    parser.add_argument("--eval-lengths", nargs="*", type=int)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--final-position-loss-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diagnostic-interval", type=int, default=100)
    parser.add_argument("--max-rotor-angle", type=float, default=math.pi)
    parser.add_argument("--relation-loss-weight", type=float, default=0.0)
    parser.add_argument("--relation-loss-power", type=float, default=2.0)
    parser.add_argument("--relation-start-step", type=int, default=0)
    parser.add_argument("--relation-ramp-steps", type=int, default=0)
    parser.add_argument("--canonical-orbit-loss-weight", type=float, default=0.0)
    parser.add_argument("--holonomy-loss-weight", type=float, default=0.0)
    parser.add_argument("--holonomy-loss-power", type=float, default=8.0)
    parser.add_argument("--holonomy-margin-weight", type=float, default=0.0)
    parser.add_argument("--holonomy-margin-target", type=float, default=0.5)
    parser.add_argument("--holonomy-start-step", type=int, default=0)
    parser.add_argument("--holonomy-ramp-steps", type=int, default=0)
    parser.add_argument("--holonomy-word-multiplier", type=int, default=4)
    parser.add_argument("--holonomy-word-multipliers", nargs="*", type=int)
    parser.add_argument("--holonomy-batch-size", type=int, default=64)
    parser.add_argument("--a5-irrep-init", action="store_true")
    parser.add_argument("--freeze-actions", action="store_true")
    parser.add_argument("--group", choices=tuple(GROUPS), default="a5")
    parser.add_argument("--input-elements", nargs="*", default=[])
    parser.add_argument("--held-out-pairs", nargs="*", default=[])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--families", nargs="+", choices=MECHANISM_FAMILIES, default=list(MECHANISM_FAMILIES)
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.use_deterministic_algorithms(True)
    if min(args.steps, args.batch_size, args.seq_len, args.channels) < 1:
        raise ValueError("steps, batch size, sequence length, and channels must be positive")
    if args.diagnostic_interval < 0:
        raise ValueError("diagnostic interval cannot be negative")
    if min(
        args.relation_loss_weight,
        args.relation_start_step,
        args.relation_ramp_steps,
        args.canonical_orbit_loss_weight,
        args.final_position_loss_weight,
        args.holonomy_loss_weight,
        args.holonomy_margin_weight,
        args.holonomy_start_step,
        args.holonomy_ramp_steps,
    ) < 0:
        raise ValueError("loss weights and objective scheduling cannot be negative")
    if args.max_rotor_angle <= 0:
        raise ValueError("max rotor angle must be positive")
    if args.relation_loss_power < 2:
        raise ValueError("relation loss power must be at least 2")
    if args.holonomy_loss_power < 2:
        raise ValueError("holonomy loss power must be at least 2")
    if args.holonomy_margin_target <= 0:
        raise ValueError("holonomy margin target must be positive")
    if min(args.holonomy_word_multiplier, args.holonomy_batch_size) < 1:
        raise ValueError("holonomy word multiplier and batch size must be positive")
    if args.holonomy_word_multipliers and min(args.holonomy_word_multipliers) < 1:
        raise ValueError("every holonomy word multiplier must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.freeze_actions and not args.a5_irrep_init:
        raise ValueError("--freeze-actions requires --a5-irrep-init")
    if args.a5_irrep_init and (
        args.group != "a5"
        or len(args.families) != 1
        or args.families[0] not in ("pure_ga_rotor", "pure_householder")
    ):
        raise ValueError(
            "--a5-irrep-init requires A5 and one rotor or Householder family"
        )
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    group = GROUPS[args.group]
    input_elements = parse_input_elements(args.input_elements, group)
    canonical_group_words(group, input_elements)
    held_out_group_pairs = parse_held_out_pairs(args.held_out_pairs, group)
    input_lookup = {element: token for token, element in enumerate(input_elements)}
    try:
        held_out_pairs = tuple(
            (input_lookup[left], input_lookup[right]) for left, right in held_out_group_pairs
        )
    except KeyError as error:
        raise ValueError("every held-out pair element must be in the input alphabet") from error
    config = MechanismConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.seq_len,
        validation_batches=args.validation_batches,
        validation_batch_size=args.validation_batch_size,
        channels=args.channels,
        learning_rate=args.learning_rate,
        final_position_loss_weight=args.final_position_loss_weight,
        seed=args.seed,
        diagnostic_interval=args.diagnostic_interval,
        max_rotor_angle=args.max_rotor_angle,
        relation_loss_weight=args.relation_loss_weight,
        relation_loss_power=args.relation_loss_power,
        relation_start_step=args.relation_start_step,
        relation_ramp_steps=args.relation_ramp_steps,
        canonical_orbit_loss_weight=args.canonical_orbit_loss_weight,
        holonomy_loss_weight=args.holonomy_loss_weight,
        holonomy_loss_power=args.holonomy_loss_power,
        holonomy_margin_weight=args.holonomy_margin_weight,
        holonomy_margin_target=args.holonomy_margin_target,
        holonomy_start_step=args.holonomy_start_step,
        holonomy_ramp_steps=args.holonomy_ramp_steps,
        holonomy_word_multiplier=args.holonomy_word_multiplier,
        holonomy_word_multipliers=tuple(args.holonomy_word_multipliers or ()),
        holonomy_batch_size=args.holonomy_batch_size,
        a5_irrep_init=args.a5_irrep_init,
        freeze_actions=args.freeze_actions,
    )
    train_batches = make_group_batches(
        group, config.steps, config.batch_size, config.sequence_length, config.seed + 1000,
        input_elements=input_elements, held_out_pairs=held_out_pairs,
    )
    validation_batches = make_group_batches(
        group, config.validation_batches, config.validation_batch_size,
        config.sequence_length, 91_337, input_elements=input_elements,
        held_out_pairs=held_out_pairs, require_held_out_pair=bool(held_out_pairs),
    )
    # Sparse checkpoint lengths twice concealed real interior failures.  Dense
    # multiples of the training length are therefore the default evaluation
    # protocol; ``--eval-lengths`` remains available for deliberately cheaper
    # smoke tests and backwards-compatible reproductions.
    evaluation_lengths = args.eval_lengths or sorted(
        {2, 4, 8, *range(config.sequence_length, 16 * config.sequence_length + 1,
                         config.sequence_length)}
    )
    generalization_batches = {
        length: make_group_batches(
            group, config.validation_batches, config.validation_batch_size, length,
            91_337 + length, input_elements=input_elements, held_out_pairs=held_out_pairs,
            require_held_out_pair=bool(held_out_pairs),
        )
        for length in evaluation_lengths
    }
    results = []
    for family in args.families:
        checkpoint_output = args.checkpoint_output
        if checkpoint_output is not None and len(args.families) > 1:
            checkpoint_output = checkpoint_output.with_name(
                f"{checkpoint_output.stem}_{family}{checkpoint_output.suffix}"
            )
        results.append(
            run_variant(
                family,
                group,
                input_elements,
                train_batches,
                validation_batches,
                generalization_batches,
                config,
                device,
                checkpoint_output=checkpoint_output,
            )
        )
    report = {
        "experiment": f"write-free norm-preserving {group.key.upper()} group-action gate",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "torch_version": torch.__version__,
        "config": asdict(config),
        "group": {"key": group.key, "name": group.name, "order": group.order},
        "input_alphabet": [
            {"token_index": token, "group_index": element, "element": group.elements[element]}
            for token, element in enumerate(input_elements)
        ],
        "held_out_transition_pairs": [
            {"left": group.elements[left], "right": group.elements[right]}
            for left, right in held_out_group_pairs
        ],
        "data_split_audit": (
            {
                "training": pair_split_audit(train_batches, held_out_pairs),
                "validation": pair_split_audit(validation_batches, held_out_pairs),
                "generalization": {
                    str(length): pair_split_audit(batches, held_out_pairs)
                    for length, batches in generalization_batches.items()
                },
            }
            if held_out_pairs else None
        ),
        "language_coverage_audit": {
            "training": state_and_pair_coverage_audit(
                train_batches,
                input_order=len(input_elements),
                group_order=group.order,
            ),
            "validation": state_and_pair_coverage_audit(
                validation_batches,
                input_order=len(input_elements),
                group_order=group.order,
            ),
        },
        "evaluation_lengths": evaluation_lengths,
        "comparison_contract": {
            "same_training_batches": True,
            "same_validation_batches": True,
            "same_state_width": True,
            "all_actions_norm_preserving": True,
            "no_affine_writes": True,
            "parameter_counts_are_reported_not_matched": True,
        },
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
