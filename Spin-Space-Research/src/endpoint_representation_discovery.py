"""Recover a finite multiplication table from endpoint-clustered learned actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compare_recurrences import FiniteGroup


@dataclass(frozen=True)
class EndpointManifoldRecovery:
    group: FiniteGroup
    input_elements: tuple[int, ...]
    identity_label: int
    label_to_element: np.ndarray
    element_to_label: np.ndarray
    centers: np.ndarray
    class_consistency_rms: float
    minimum_center_separation: float
    multiplication_rms: float
    multiplication_max: float
    minimum_assignment_gap: float


def nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(matrix)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return rotation


def word_action_products(
    token_rotations: np.ndarray, words: np.ndarray
) -> np.ndarray:
    """Compose column-vector actions in the same order as the recurrence."""
    batch = words.shape[0]
    products = np.broadcast_to(np.eye(3), (batch, 3, 3)).copy()
    for position in range(words.shape[1]):
        products = token_rotations[words[:, position]] @ products
    return products


def endpoint_rotation_centers(
    token_rotations: np.ndarray,
    words: np.ndarray,
    endpoint_labels: np.ndarray,
    *,
    state_count: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    products = word_action_products(token_rotations, words)
    centers = np.empty((state_count, 3, 3), dtype=np.float64)
    counts = np.bincount(endpoint_labels, minlength=state_count)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"endpoint corpus misses labels {missing}")
    for label in range(state_count):
        centers[label] = nearest_rotation(products[endpoint_labels == label].mean(axis=0))
    residual = products - centers[endpoint_labels]
    consistency_rms = float(np.sqrt(np.mean(np.square(residual))))
    return centers, consistency_rms, counts


def _validate_table(table: np.ndarray) -> None:
    order = len(table)
    expected = np.arange(order)
    if not all(np.array_equal(np.sort(row), expected) for row in table):
        raise ValueError("nearest-center products do not form row permutations")
    if not all(np.array_equal(np.sort(column), expected) for column in table.T):
        raise ValueError("nearest-center products do not form column permutations")
    for left in range(order):
        for middle in range(order):
            lhs = table[table[left, middle]]
            rhs = table[left, table[middle]]
            if not np.array_equal(lhs, rhs):
                raise ValueError("nearest-center multiplication is not associative")


def recover_endpoint_manifold(
    token_rotations: np.ndarray,
    words: np.ndarray,
    endpoint_labels: np.ndarray,
    *,
    state_count: int,
) -> EndpointManifoldRecovery:
    """Cluster words by endpoint, then close the class centers under products."""
    centers_by_label, consistency_rms, _ = endpoint_rotation_centers(
        token_rotations,
        words,
        endpoint_labels,
        state_count=state_count,
    )
    pairwise = np.sqrt(
        np.mean(
            np.square(centers_by_label[:, None] - centers_by_label[None, :]),
            axis=(2, 3),
        )
    )
    np.fill_diagonal(pairwise, np.inf)
    minimum_separation = float(np.min(pairwise))

    identity_label = int(
        np.argmin(np.mean(np.square(centers_by_label - np.eye(3)), axis=(1, 2)))
    )
    element_to_label = np.asarray(
        [identity_label]
        + [label for label in range(state_count) if label != identity_label],
        dtype=np.int64,
    )
    label_to_element = np.empty(state_count, dtype=np.int64)
    label_to_element[element_to_label] = np.arange(state_count)
    centers = centers_by_label[element_to_label]

    table = np.empty((state_count, state_count), dtype=np.int64)
    nearest_residuals = []
    assignment_gaps = []
    for left in range(state_count):
        for right in range(state_count):
            # A word for left followed by a word for right acts as R_right R_left.
            product = centers[right] @ centers[left]
            distances = np.sqrt(np.mean(np.square(centers - product), axis=(1, 2)))
            ordering = np.argsort(distances)
            table[left, right] = int(ordering[0])
            nearest_residuals.append(float(distances[ordering[0]]))
            assignment_gaps.append(float(distances[ordering[1]] - distances[ordering[0]]))
    _validate_table(table)
    if not np.array_equal(table[0], np.arange(state_count)):
        raise ValueError("recovered identity is not a left identity")
    if not np.array_equal(table[:, 0], np.arange(state_count)):
        raise ValueError("recovered identity is not a right identity")

    token_elements = []
    for rotation in token_rotations:
        distances = np.sqrt(np.mean(np.square(centers - rotation), axis=(1, 2)))
        token_elements.append(int(np.argmin(distances)))
    generated = {0}
    frontier = [0]
    while frontier:
        state = frontier.pop()
        for token in token_elements:
            target = int(table[state, token])
            if target not in generated:
                generated.add(target)
                frontier.append(target)
    if len(generated) != state_count:
        raise ValueError(
            f"recovered token actions generate {len(generated)}/{state_count} elements"
        )

    residuals = np.asarray(nearest_residuals)
    group = FiniteGroup(
        key="endpoint_recovered",
        name="endpoint-recovered finite action",
        elements=tuple(f"e{index}" for index in range(state_count)),
        table=table,
    )
    return EndpointManifoldRecovery(
        group=group,
        input_elements=tuple(token_elements),
        identity_label=identity_label,
        label_to_element=label_to_element,
        element_to_label=element_to_label,
        centers=centers,
        class_consistency_rms=consistency_rms,
        minimum_center_separation=minimum_separation,
        multiplication_rms=float(np.sqrt(np.mean(np.square(residuals)))),
        multiplication_max=float(np.max(residuals)),
        minimum_assignment_gap=float(np.min(assignment_gaps)),
    )
