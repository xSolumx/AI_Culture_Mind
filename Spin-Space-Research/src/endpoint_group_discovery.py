"""Recover a regular finite action using complete-word endpoint labels only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from compare_recurrences import FiniteGroup
from latent_group_discovery import RecoveredPermutationGroup, TransitionEvidence


Word = tuple[int, ...]
EndpointQuery = Callable[[Word], int]


@dataclass(frozen=True)
class EndpointRecoveryReport:
    identity_label: int
    passive_samples: int
    passive_word_length: int
    samples_until_all_states: int
    passive_unique_labels: int
    inverse_pair_queries: int
    extension_queries: int
    total_active_queries: int
    total_endpoint_labels: int
    inferred_inverse_tokens: tuple[int, ...]
    queried_tokens: tuple[int, ...]
    observed_transition_edges: int
    completed_transition_edges: int


class GroupEndpointOracle:
    """Environment boundary: return a final label, never a prefix trace."""

    def __init__(
        self, group: FiniteGroup, input_elements: tuple[int, ...]
    ) -> None:
        self._group = group
        self._input_elements = input_elements
        self.query_count = 0

    def query(self, word: Word) -> int:
        self.query_count += 1
        state = 0
        for token in word:
            if not 0 <= token < len(self._input_elements):
                raise ValueError("endpoint query contains an invalid token")
            state = int(self._group.table[state, self._input_elements[token]])
        return state


def passive_representatives(
    query: EndpointQuery,
    *,
    token_count: int,
    state_count: int,
    samples: int,
    word_length: int,
    seed: int,
) -> tuple[dict[int, Word], int]:
    """Collect one complete-word representative for each anonymous label."""

    if min(token_count, state_count, samples, word_length) < 1:
        raise ValueError("passive endpoint parameters must be positive")
    generator = np.random.default_rng(seed)
    representatives: dict[int, Word] = {}
    samples_until_all_states = 0
    for index in range(samples):
        word = tuple(
            int(token)
            for token in generator.integers(
                0, token_count, size=word_length, dtype=np.int64
            )
        )
        label = int(query(word))
        if not 0 <= label < state_count:
            raise ValueError("endpoint oracle returned an out-of-range label")
        representatives.setdefault(label, word)
        if len(representatives) == state_count and samples_until_all_states == 0:
            samples_until_all_states = index + 1
    if len(representatives) != state_count:
        raise ValueError(
            f"passive endpoint corpus covers {len(representatives)}/{state_count} labels"
        )
    return representatives, samples_until_all_states


def infer_four_token_inverse_matching(
    query: EndpointQuery, identity_label: int
) -> tuple[int, ...]:
    """Identify one pair with three endpoint queries; matching forces the other."""

    candidates = [
        other for other in (1, 2, 3) if query((0, other)) == identity_label
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"token 0 has {len(candidates)} endpoint-identified inverse candidates"
        )
    partner = candidates[0]
    remaining = [token for token in range(4) if token not in (0, partner)]
    inverse = np.empty(4, dtype=np.int64)
    inverse[0] = partner
    inverse[partner] = 0
    inverse[remaining[0]] = remaining[1]
    inverse[remaining[1]] = remaining[0]
    return tuple(int(value) for value in inverse)


def endpoint_transition_cover(
    query: EndpointQuery,
    representatives: dict[int, Word],
    inverse_tokens: tuple[int, ...],
    *,
    state_count: int,
    omit_extension: tuple[int, int] | None = None,
) -> tuple[TransitionEvidence, tuple[int, ...]]:
    """Query one complete token action per inverse family, then invert it."""

    token_count = len(inverse_tokens)
    queried_tokens = tuple(
        token for token, inverse in enumerate(inverse_tokens) if token < inverse
    )
    evidence = TransitionEvidence(state_count, token_count)
    for label, word in sorted(representatives.items()):
        for token in queried_tokens:
            if omit_extension == (label, token):
                continue
            endpoint = int(query(word + (token,)))
            existing = evidence.next_states[label, token]
            if existing >= 0 and existing != endpoint:
                raise ValueError("endpoint extensions imply a nondeterministic action")
            evidence.next_states[label, token] = endpoint
            evidence.counts[label, token] += 1
    evidence.complete_with_inverse_tokens(inverse_tokens)
    return evidence, queried_tokens


def recover_from_endpoint_queries(
    query: EndpointQuery,
    *,
    state_count: int,
    token_count: int,
    passive_samples: int = 1_024,
    passive_word_length: int = 16,
    seed: int = 0,
) -> tuple[RecoveredPermutationGroup, EndpointRecoveryReport]:
    """Discover and complete an anonymous action without any prefix labels."""

    if token_count != 4:
        raise ValueError("the current endpoint matching protocol requires four tokens")
    identity_label = int(query(()))
    representatives, samples_until_all_states = passive_representatives(
        query,
        token_count=token_count,
        state_count=state_count,
        samples=passive_samples,
        word_length=passive_word_length,
        seed=seed,
    )
    inverse_tokens = infer_four_token_inverse_matching(query, identity_label)
    evidence, queried_tokens = endpoint_transition_cover(
        query,
        representatives,
        inverse_tokens,
        state_count=state_count,
    )
    recovered = evidence.recover(base_state=identity_label)
    observed_edges = int(np.sum(evidence.counts > 0))
    report = EndpointRecoveryReport(
        identity_label=identity_label,
        passive_samples=passive_samples,
        passive_word_length=passive_word_length,
        samples_until_all_states=samples_until_all_states,
        passive_unique_labels=len(representatives),
        inverse_pair_queries=3,
        extension_queries=state_count * len(queried_tokens),
        total_active_queries=1 + 3 + state_count * len(queried_tokens),
        total_endpoint_labels=(
            passive_samples + 1 + 3 + state_count * len(queried_tokens)
        ),
        inferred_inverse_tokens=inverse_tokens,
        queried_tokens=queried_tokens,
        observed_transition_edges=observed_edges,
        completed_transition_edges=state_count * token_count - observed_edges,
    )
    return recovered, report
