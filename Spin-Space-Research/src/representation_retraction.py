"""Group-table-derived exact representation compilation and joint retraction.

The compiler deliberately does not use a character table or hand-supplied
irreducible matrices.  A generic symmetric element of the right-regular group
algebra commutes with the exact left-regular action.  Its generic eigenspaces
therefore split the regular representation into irreducible invariant copies.
Learned token actions select the nearest copy and one global conjugation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from compare_recurrences import FiniteGroup


@dataclass(frozen=True)
class RepresentationCandidate:
    actions: np.ndarray
    character: np.ndarray
    invariance_rms: float
    homomorphism_rms: float


@dataclass(frozen=True)
class CompiledRepresentation:
    token_actions: np.ndarray
    group_actions: np.ndarray
    candidate_index: int
    alignment: np.ndarray
    alignment_rms: float
    runner_up_rms: float
    character: np.ndarray
    invariance_rms: float
    homomorphism_rms: float


def element_inverses(group: FiniteGroup) -> np.ndarray:
    identity_products = group.table == 0
    if not np.all(identity_products.sum(axis=1) == 1):
        raise ValueError("group table does not provide one inverse per element")
    return identity_products.argmax(axis=1)


def nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(matrix)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return rotation


def align_representation(
    learned: np.ndarray, candidate: np.ndarray, seed: int
) -> tuple[np.ndarray, float]:
    """Fit one global SO(3) conjugation without representation-specific data."""

    identity = np.eye(3)
    intertwiner = np.concatenate(
        [
            np.kron(identity, left) - np.kron(right.T, identity)
            for left, right in zip(learned, candidate)
        ],
        axis=0,
    )
    _, _, right_singular = np.linalg.svd(intertwiner)
    linear_guess = right_singular[-1].reshape(3, 3, order="F")
    guesses = [nearest_rotation(linear_guess), identity]
    generator = np.random.default_rng(seed)
    guesses.extend(Rotation.random(8, random_state=generator).as_matrix())

    def objective(rotvec: np.ndarray) -> float:
        change = Rotation.from_rotvec(rotvec).as_matrix()
        residual = learned - change[None] @ candidate @ change.T[None]
        return float(np.mean(np.square(residual)))

    best_matrix = identity
    best_value = math.inf
    for guess in guesses:
        result = minimize(
            objective,
            Rotation.from_matrix(guess).as_rotvec(),
            method="BFGS",
            options={"maxiter": 1_000, "gtol": 1e-12},
        )
        if result.fun < best_value:
            best_value = float(result.fun)
            best_matrix = Rotation.from_rotvec(result.x).as_matrix()
    return best_matrix, math.sqrt(best_value)


def token_commutator_max(actions: np.ndarray) -> float:
    values = []
    for left in range(len(actions)):
        for right in range(left + 1, len(actions)):
            difference = actions[right] @ actions[left] - actions[left] @ actions[right]
            values.append(np.linalg.norm(difference) / math.sqrt(actions.shape[-1]))
    return max(values, default=0.0)


def regular_actions(group: FiniteGroup) -> tuple[np.ndarray, np.ndarray]:
    """Return commuting left and right regular permutation actions."""

    order = group.order
    columns = np.arange(order)
    left = np.zeros((order, order, order), dtype=np.float64)
    right = np.zeros_like(left)
    for element in range(order):
        left[element, group.table[element], columns] = 1.0
        right[element, group.table[:, element], columns] = 1.0
    commutator = left[1] @ right[1] - right[1] @ left[1]
    if np.max(np.abs(commutator)) > 1e-12:
        raise RuntimeError("left and right regular actions do not commute")
    return left, right


def _eigenvalue_clusters(values: np.ndarray, tolerance: float) -> list[np.ndarray]:
    clusters: list[list[int]] = []
    for index, value in enumerate(values):
        if not clusters or abs(value - values[clusters[-1][0]]) > tolerance:
            clusters.append([index])
        else:
            clusters[-1].append(index)
    return [np.asarray(cluster, dtype=np.int64) for cluster in clusters]


def _homomorphism_rms(actions: np.ndarray, table: np.ndarray) -> float:
    composed = actions[:, None] @ actions[None, :]
    return float(np.sqrt(np.mean(np.square(composed - actions[table]))))


def regular_irrep_candidates(
    group: FiniteGroup,
    dimension: int,
    *,
    seed: int = 7_301,
    eigenvalue_tolerance: float = 1e-8,
    deduplication_tolerance: float = 1e-7,
) -> tuple[RepresentationCandidate, ...]:
    """Extract exact ``dimension``-D irreducible copies from the group table.

    A deterministic random self-adjoint element of the right-regular algebra
    has, generically, one eigenvalue for each multiplicity coordinate.  Every
    corresponding eigenspace is invariant under the commuting left action and
    has the irrep dimension.  Equivalent multiplicity copies are deduplicated
    by their full character vector.
    """

    left, right = regular_actions(group)
    inverses = element_inverses(group)
    generator = np.random.default_rng(seed)
    raw = generator.normal(size=group.order)
    weights = 0.5 * (raw + raw[inverses])
    commuting_operator = np.einsum("g,gij->ij", weights, right)
    commuting_operator = 0.5 * (commuting_operator + commuting_operator.T)
    values, vectors = np.linalg.eigh(commuting_operator)
    candidates: list[RepresentationCandidate] = []
    for cluster in _eigenvalue_clusters(values, eigenvalue_tolerance):
        if len(cluster) != dimension:
            continue
        basis = vectors[:, cluster]
        actions = np.einsum("ia,gij,jb->gab", basis, left, basis)
        projected = np.einsum("ia,gij->gaj", basis, left)
        reconstructed = np.einsum("gab,ib->gai", actions, basis)
        invariance_rms = float(
            np.sqrt(np.mean(np.square(projected - reconstructed)))
        )
        character = np.trace(actions, axis1=-2, axis2=-1)
        if any(
            np.max(np.abs(character - existing.character))
            <= deduplication_tolerance
            for existing in candidates
        ):
            continue
        homomorphism_rms = _homomorphism_rms(actions, group.table)
        if invariance_rms > 1e-8 or homomorphism_rms > 1e-8:
            continue
        candidates.append(
            RepresentationCandidate(
                actions=actions,
                character=character,
                invariance_rms=invariance_rms,
                homomorphism_rms=homomorphism_rms,
            )
        )
    if not candidates:
        raise RuntimeError(
            f"regular representation exposed no {dimension}D invariant copies"
        )
    return tuple(candidates)


def compile_nearest_representation(
    learned_token_actions: np.ndarray,
    group: FiniteGroup,
    input_elements: tuple[int, ...],
    *,
    dimension: int = 3,
    seed: int = 7_301,
    candidates: tuple[RepresentationCandidate, ...] | None = None,
) -> CompiledRepresentation:
    """Jointly retract a learned token family to the nearest exact irrep copy."""

    learned = np.stack([nearest_rotation(matrix) for matrix in learned_token_actions])
    inverses = element_inverses(group)
    if candidates is None:
        candidates = regular_irrep_candidates(group, dimension, seed=seed)
    fits: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    for index, candidate in enumerate(candidates):
        token_actions = np.stack(
            [candidate.actions[inverses[element]] for element in input_elements]
        )
        alignment, rms = align_representation(
            learned, token_actions, seed=seed + 101 * (index + 1)
        )
        fits.append((rms, index, alignment, token_actions))
    fits.sort(key=lambda item: item[0])
    rms, index, alignment, token_actions = fits[0]
    candidate = candidates[index]
    aligned_tokens = alignment[None] @ token_actions @ alignment.T[None]
    aligned_group = alignment[None] @ candidate.actions @ alignment.T[None]
    return CompiledRepresentation(
        token_actions=aligned_tokens,
        group_actions=aligned_group,
        candidate_index=index,
        alignment=alignment,
        alignment_rms=float(rms),
        runner_up_rms=float(fits[1][0]) if len(fits) > 1 else math.inf,
        character=candidate.character,
        invariance_rms=candidate.invariance_rms,
        homomorphism_rms=candidate.homomorphism_rms,
    )


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def local_joint_conjugacy_retraction(
    ambient_targets: np.ndarray,
    exact_token_actions: np.ndarray,
    *,
    iterations: int = 3,
) -> tuple[np.ndarray, float, float]:
    """Retract independent ambient updates through one shared SO(3) conjugation.

    Each iteration projects the ambient family displacement onto the three
    tangent directions ``[Omega, A_s]`` and then applies the resulting *single*
    conjugation to every token.  Exact mixed relations are preserved by
    construction throughout.
    """

    if iterations < 1:
        raise ValueError("iterations must be positive")
    targets = np.stack([nearest_rotation(matrix) for matrix in ambient_targets])
    current = np.asarray(exact_token_actions, dtype=np.float64).copy()
    total_tangent_norm = 0.0
    basis = np.eye(3)
    for _ in range(iterations):
        jacobian = np.stack(
            [
                np.concatenate(
                    [(_skew(axis) @ action - action @ _skew(axis)).ravel()
                     for action in current]
                )
                for axis in basis
            ],
            axis=1,
        )
        residual = (targets - current).ravel()
        tangent, *_ = np.linalg.lstsq(jacobian, residual, rcond=None)
        total_tangent_norm += float(np.linalg.norm(tangent))
        change = Rotation.from_rotvec(tangent).as_matrix()
        current = change[None] @ current @ change.T[None]
    projection_rms = float(np.sqrt(np.mean(np.square(targets - current))))
    return current, projection_rms, total_tangent_norm
