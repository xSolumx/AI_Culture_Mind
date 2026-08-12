"""Exact congruence-lattice certificates for small deterministic actions.

The geometric compiler discovers a finite transition action from continuous
states.  Once that finite action is available, its congruences are a purely
combinatorial object: an equivalence relation is a transition congruence when
every token maps equivalent states to equivalent states.  For small state sets
we can enumerate the complete partition lattice, eliminating metric clustering
and local-optimizer ambiguity from this *post-discovery* certificate.

This module deliberately does not claim that the finite action itself is the
canonical quotient of the original continuous system.  Without observations or
another separating prior, both the universal and discrete partitions are always
congruences.  That identifiability boundary is part of the returned audit.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from latent_group_discovery import TransitionEvidence


@dataclass(frozen=True)
class ActionCongruence:
    """One canonical transition-stable partition and its quotient action."""

    labels: tuple[int, ...]
    block_sizes: tuple[int, ...]
    quotient_next_states: tuple[tuple[int, ...], ...]
    quotient_is_regular: bool
    generated_permutation_order: int | None

    @property
    def block_count(self) -> int:
        return len(self.block_sizes)

    def to_dict(self) -> dict[str, object]:
        return {
            "labels": list(self.labels),
            "block_count": self.block_count,
            "block_sizes": list(self.block_sizes),
            "quotient_next_states": [list(row) for row in self.quotient_next_states],
            "quotient_is_regular": self.quotient_is_regular,
            "generated_permutation_order": self.generated_permutation_order,
        }


def canonical_partition(labels: Sequence[int]) -> tuple[int, ...]:
    """Rename blocks by first occurrence, yielding a restricted-growth tuple."""

    mapping: dict[int, int] = {}
    canonical = []
    for value in labels:
        integer = int(value)
        if integer not in mapping:
            mapping[integer] = len(mapping)
        canonical.append(mapping[integer])
    return tuple(canonical)


def set_partitions(state_count: int) -> Iterator[tuple[int, ...]]:
    """Yield every set partition exactly once as a restricted-growth tuple."""

    if state_count < 1:
        raise ValueError("state_count must be positive")
    labels = [0] * state_count

    def extend(position: int, maximum: int) -> Iterator[tuple[int, ...]]:
        if position == state_count:
            yield tuple(labels)
            return
        for block in range(maximum + 2):
            labels[position] = block
            yield from extend(position + 1, max(maximum, block))

    yield from extend(1, 0)


def _validated_action(next_states: np.ndarray) -> np.ndarray:
    action = np.asarray(next_states, dtype=np.int64)
    if action.ndim != 2 or min(action.shape) < 1:
        raise ValueError("next_states must have shape (states, tokens)")
    if action.min() < 0 or action.max() >= action.shape[0]:
        raise ValueError("next_states contains an out-of-range state")
    return action


def is_transition_congruence(
    next_states: np.ndarray, labels: Sequence[int]
) -> bool:
    """Return whether ``labels`` is stable under every token transition."""

    action = _validated_action(next_states)
    partition = np.asarray(canonical_partition(labels), dtype=np.int64)
    if partition.shape != (action.shape[0],):
        raise ValueError("partition length must equal the state count")
    for block in range(int(partition.max()) + 1):
        members = np.flatnonzero(partition == block)
        target_blocks = partition[action[members]]
        if np.any(target_blocks != target_blocks[:1]):
            return False
    return True


def quotient_action(next_states: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    """Construct the deterministic action induced on congruence blocks."""

    action = _validated_action(next_states)
    partition = np.asarray(canonical_partition(labels), dtype=np.int64)
    if not is_transition_congruence(action, partition):
        raise ValueError("partition is not a transition congruence")
    block_count = int(partition.max()) + 1
    quotient = np.empty((block_count, action.shape[1]), dtype=np.int64)
    for block in range(block_count):
        representative = int(np.flatnonzero(partition == block)[0])
        quotient[block] = partition[action[representative]]
    return quotient


def _regularity(quotient: np.ndarray) -> tuple[bool, int | None]:
    evidence = TransitionEvidence(*quotient.shape)
    evidence.next_states[:] = quotient
    evidence.counts[:] = 1
    try:
        recovered = evidence.recover(base_state=0)
    except ValueError:
        return False, None
    return True, recovered.group.order


def enumerate_action_congruences(
    next_states: np.ndarray, *, maximum_states: int = 10
) -> tuple[ActionCongruence, ...]:
    """Exhaustively enumerate all transition congruences of a small action."""

    action = _validated_action(next_states)
    if action.shape[0] > maximum_states:
        raise ValueError(
            f"exact partition enumeration is capped at {maximum_states} states; "
            f"received {action.shape[0]}"
        )
    congruences = []
    for labels in set_partitions(action.shape[0]):
        if not is_transition_congruence(action, labels):
            continue
        quotient = quotient_action(action, labels)
        regular, order = _regularity(quotient)
        counts = np.bincount(np.asarray(labels), minlength=quotient.shape[0])
        congruences.append(
            ActionCongruence(
                labels=labels,
                block_sizes=tuple(int(value) for value in counts),
                quotient_next_states=tuple(
                    tuple(int(value) for value in row) for row in quotient
                ),
                quotient_is_regular=regular,
                generated_permutation_order=order,
            )
        )
    return tuple(congruences)


def exact_congruence_lattice_audit(next_states: np.ndarray) -> dict[str, object]:
    """Return an exhaustive certificate and the observation-free boundary."""

    action = _validated_action(next_states)
    congruences = enumerate_action_congruences(action)
    counts = Counter(item.block_count for item in congruences)
    regular_counts = Counter(
        item.block_count for item in congruences if item.quotient_is_regular
    )
    universal = (0,) * action.shape[0]
    discrete = tuple(range(action.shape[0]))
    return {
        "state_count": int(action.shape[0]),
        "token_count": int(action.shape[1]),
        "enumerated_set_partitions": sum(1 for _ in set_partitions(action.shape[0])),
        "transition_congruence_count": len(congruences),
        "congruence_count_by_block_count": {
            str(key): counts[key] for key in sorted(counts)
        },
        "regular_quotient_count_by_block_count": {
            str(key): regular_counts[key] for key in sorted(regular_counts)
        },
        "universal_partition_is_congruence": is_transition_congruence(
            action, universal
        ),
        "discrete_partition_is_congruence": is_transition_congruence(
            action, discrete
        ),
        "observation_free_unique_nontrivial_quotient_identifiable": False,
        "identifiability_reason": (
            "transition closure alone always accepts both the universal and "
            "discrete partitions; selecting an intermediate finite quotient "
            "requires observations or an explicit geometric/complexity prior"
        ),
        "congruences": [item.to_dict() for item in congruences],
    }
