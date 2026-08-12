"""Recover a finite permutation group from labeled transition evidence only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from compare_recurrences import FiniteGroup


@dataclass(frozen=True)
class RecoveredPermutationGroup:
    group: FiniteGroup
    input_elements: tuple[int, ...]
    token_permutations: np.ndarray
    group_permutations: np.ndarray
    state_to_element: np.ndarray
    element_to_state: np.ndarray
    evidence_counts: np.ndarray

    def translate_targets(self, targets: torch.Tensor) -> torch.Tensor:
        mapping = torch.as_tensor(
            self.state_to_element, dtype=torch.long, device=targets.device
        )
        return mapping[targets]


def exact_inverse_tokens(next_states: np.ndarray) -> tuple[int, ...]:
    """Infer distinct inverse tokens from a complete permutation action."""

    transitions = np.asarray(next_states, dtype=np.int64)
    if transitions.ndim != 2 or np.any(transitions < 0):
        raise ValueError("inverse-token audit needs a complete transition table")
    state_count, token_count = transitions.shape
    expected = np.arange(state_count)
    inverses = []
    for token in range(token_count):
        candidates = [
            other
            for other in range(token_count)
            if other != token
            and np.array_equal(transitions[transitions[:, token], other], expected)
            and np.array_equal(transitions[transitions[:, other], token], expected)
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"token {token} has {len(candidates)} distinct inverse candidates"
            )
        inverses.append(candidates[0])
    return tuple(inverses)


def inverse_cover_partial_evidence(
    full: "TransitionEvidence",
    *,
    calibration_fraction: float = 0.0,
    calibration_pairs_total: int | None = None,
    seed: int,
) -> tuple["TransitionEvidence", dict[str, object]]:
    """Keep one orientation of every reverse edge plus calibrated pairs.

    The masking environment may inspect the complete action to construct a
    balanced reverse-edge cover. The returned learner evidence does *not*
    contain the inverse-token pairing: it must recover that shared involution
    from the bidirectionally visible calibration pairs.
    """

    if not full.complete:
        raise ValueError("inverse-cover masking needs complete source evidence")
    if not 0.0 <= calibration_fraction <= 1.0:
        raise ValueError("calibration_fraction must lie in [0, 1]")
    inverse_tokens = exact_inverse_tokens(full.next_states)
    rng = np.random.default_rng(seed)
    families: dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int]]]] = {}
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for source in range(full.state_count):
        for token in range(full.token_count):
            target = int(full.next_states[source, token])
            inverse = inverse_tokens[token]
            edge = (source, token)
            reverse = (target, inverse)
            pair = tuple(sorted((edge, reverse)))
            if pair in seen:
                continue
            seen.add(pair)
            family = tuple(sorted((token, inverse)))
            families.setdefault(family, []).append(pair)

    if calibration_pairs_total is not None:
        if not 0 <= calibration_pairs_total <= len(seen):
            raise ValueError("calibration_pairs_total is out of range")
        all_pair_keys = [
            (family, index)
            for family, pairs in sorted(families.items())
            for index in range(len(pairs))
        ]
        order = rng.permutation(len(all_pair_keys))
        calibrated_keys = {
            all_pair_keys[int(index)]
            for index in order[:calibration_pairs_total]
        }
    else:
        calibrated_keys = set()
        for family, pairs in sorted(families.items()):
            order = rng.permutation(len(pairs))
            calibrated_count = int(round(calibration_fraction * len(pairs)))
            calibrated_keys.update(
                (family, int(index)) for index in order[:calibrated_count]
            )

    partial = TransitionEvidence(full.state_count, full.token_count)
    calibration_by_family: dict[str, int] = {}
    for family, pairs in sorted(families.items()):
        family_key = "+".join(map(str, family))
        calibration_by_family[family_key] = 0
        for index, pair in enumerate(pairs):
            calibrated = (family, index) in calibrated_keys
            calibration_by_family[family_key] += int(calibrated)
            selected = pair if calibrated else (pair[int(rng.integers(2))],)
            for source, token in selected:
                partial.next_states[source, token] = full.next_states[source, token]
                partial.counts[source, token] = full.counts[source, token]

    observed_edges = int(np.sum(partial.next_states >= 0))
    return partial, {
        "true_inverse_tokens_for_audit_only": list(inverse_tokens),
        "inverse_families": {
            "+".join(map(str, family)): len(pairs)
            for family, pairs in sorted(families.items())
        },
        "calibration_pairs": len(calibrated_keys),
        "calibration_pairs_by_inverse_family": calibration_by_family,
        "observed_edges": observed_edges,
        "total_edges": int(full.next_states.size),
        "observed_fraction": observed_edges / full.next_states.size,
        "mask_seed": seed,
    }


def random_partial_evidence(
    full: "TransitionEvidence", *, observed_edges: int, seed: int
) -> "TransitionEvidence":
    """Uniform equal-budget mask used only as a negative control."""

    if not full.complete:
        raise ValueError("random masking needs complete source evidence")
    if not 0 <= observed_edges <= full.next_states.size:
        raise ValueError("observed_edges is out of range")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(full.next_states.size, size=observed_edges, replace=False)
    partial = TransitionEvidence(full.state_count, full.token_count)
    rows, columns = np.unravel_index(chosen, full.next_states.shape)
    partial.next_states[rows, columns] = full.next_states[rows, columns]
    partial.counts[rows, columns] = full.counts[rows, columns]
    return partial


class TransitionEvidence:
    """Accumulate deterministic ``state --token--> state`` observations."""

    def __init__(self, state_count: int, token_count: int) -> None:
        if min(state_count, token_count) < 1:
            raise ValueError("state and token counts must be positive")
        self.state_count = state_count
        self.token_count = token_count
        self.next_states = np.full((state_count, token_count), -1, dtype=np.int64)
        self.counts = np.zeros_like(self.next_states)

    def observe(self, tokens: torch.Tensor, prefix_labels: torch.Tensor) -> None:
        if tokens.shape != prefix_labels.shape or tokens.ndim != 2:
            raise ValueError("tokens and prefix labels must share shape (batch, length)")
        if tokens.shape[1] < 2:
            return
        token_values = tokens[:, 1:].detach().cpu().numpy().ravel()
        previous = prefix_labels[:, :-1].detach().cpu().numpy().ravel()
        following = prefix_labels[:, 1:].detach().cpu().numpy().ravel()
        if (
            token_values.min(initial=0) < 0
            or token_values.max(initial=0) >= self.token_count
            or previous.min(initial=0) < 0
            or previous.max(initial=0) >= self.state_count
            or following.min(initial=0) < 0
            or following.max(initial=0) >= self.state_count
        ):
            raise ValueError("transition evidence contains an out-of-range label")
        for state, token, next_state in zip(previous, token_values, following):
            existing = self.next_states[state, token]
            if existing >= 0 and existing != next_state:
                raise ValueError(
                    f"nondeterministic transition for state={state}, token={token}: "
                    f"observed {existing} and {next_state}"
                )
            self.next_states[state, token] = next_state
            self.counts[state, token] += 1

    @property
    def coverage(self) -> float:
        return float(np.mean(self.next_states >= 0))

    @property
    def complete(self) -> bool:
        return bool(np.all(self.next_states >= 0))

    def recover(
        self,
        *,
        base_state: int = 0,
        generator_order: tuple[int, ...] | None = None,
    ) -> RecoveredPermutationGroup:
        if not self.complete:
            missing = int(np.sum(self.next_states < 0))
            raise ValueError(f"transition table is incomplete: {missing} missing edges")
        if not 0 <= base_state < self.state_count:
            raise ValueError("base state is out of range")
        if generator_order is None:
            generator_order = tuple(range(self.token_count))
        if tuple(sorted(generator_order)) != tuple(range(self.token_count)):
            raise ValueError("generator_order must be a permutation of token indices")
        token_permutations = self.next_states.T.copy()
        expected = np.arange(self.state_count)
        for token, permutation in enumerate(token_permutations):
            if not np.array_equal(np.sort(permutation), expected):
                raise ValueError(f"token {token} is not a permutation of latent states")

        identity = expected.copy()
        permutations = [identity]
        index = {identity.tobytes(): 0}
        cursor = 0
        while cursor < len(permutations):
            current = permutations[cursor]
            cursor += 1
            for token in generator_order:
                generator = token_permutations[token]
                # Apply ``current`` and then the token generator.
                product = generator[current]
                key = product.tobytes()
                if key not in index:
                    index[key] = len(permutations)
                    permutations.append(product)
                    if len(permutations) > self.state_count:
                        raise ValueError(
                            "generated permutation group is larger than the labeled state set"
                        )
        if len(permutations) != self.state_count:
            raise ValueError(
                f"action is not regular/transitive: generated {len(permutations)} "
                f"permutations for {self.state_count} states"
            )
        group_permutations = np.stack(permutations)
        input_elements = tuple(
            index[permutation.tobytes()] for permutation in token_permutations
        )
        table = np.empty((self.state_count, self.state_count), dtype=np.int64)
        for left, left_permutation in enumerate(group_permutations):
            for right, right_permutation in enumerate(group_permutations):
                # Product left*right means apply left, then right.
                product = right_permutation[left_permutation]
                try:
                    table[left, right] = index[product.tobytes()]
                except KeyError as error:
                    raise ValueError("generated permutations are not closed") from error

        element_to_state = group_permutations[:, base_state]
        if not np.array_equal(np.sort(element_to_state), expected):
            raise ValueError("permutation action is not regular on the chosen base state")
        state_to_element = np.empty(self.state_count, dtype=np.int64)
        state_to_element[element_to_state] = expected
        for state in range(self.state_count):
            element = state_to_element[state]
            for token, token_element in enumerate(input_elements):
                observed_next = self.next_states[state, token]
                predicted_element = table[element, token_element]
                if element_to_state[predicted_element] != observed_next:
                    raise RuntimeError("recovered Cayley table does not reproduce evidence")

        group = FiniteGroup(
            key="latent",
            name="transition-inferred finite permutation group",
            elements=tuple(f"latent_{index}" for index in range(self.state_count)),
            table=table,
        )
        return RecoveredPermutationGroup(
            group=group,
            input_elements=input_elements,
            token_permutations=token_permutations,
            group_permutations=group_permutations,
            state_to_element=state_to_element,
            element_to_state=element_to_state,
            evidence_counts=self.counts.copy(),
        )

    def complete_with_inverse_tokens(
        self, inverse_tokens: tuple[int, ...]
    ) -> None:
        """Complete a reverse cover using an externally inferred involution."""

        if len(inverse_tokens) != self.token_count:
            raise ValueError("inverse-token mapping has the wrong length")
        if any(
            inverse == token
            or not 0 <= inverse < self.token_count
            or inverse_tokens[inverse] != token
            for token, inverse in enumerate(inverse_tokens)
        ):
            raise ValueError("inverse-token mapping must be a fixed-point-free involution")
        completed = self.next_states.copy()
        inferred = completed < 0
        for token, inverse in enumerate(inverse_tokens):
            for source in range(self.state_count):
                target = completed[source, token]
                if target < 0:
                    continue
                reverse = completed[target, inverse]
                if reverse >= 0 and reverse != source:
                    raise ValueError("inverse propagation contradicts observed transition")
                if reverse < 0:
                    completed[target, inverse] = source
        expected = np.arange(self.state_count)
        if np.any(completed < 0) or not all(
            np.array_equal(np.sort(completed[:, token]), expected)
            for token in range(self.token_count)
        ):
            missing = int(np.sum(completed < 0))
            raise ValueError(
                f"inverse propagation does not complete permutation actions: "
                f"{missing} missing edges"
            )
        self.next_states[:] = completed
        self.counts[inferred] = 0

    def infer_inverse_pairs_and_complete(
        self, *, minimum_total_support: int = 0
    ) -> tuple[int, ...]:
        """Complete hidden reverse edges after inferring inverse token pairs.

        This deliberately handles only even token counts with distinct inverse
        partners. It does not assume which tokens are paired. Candidate perfect
        matching is propagated across the *whole* action family and retained
        only if it produces complete permutations without contradiction.
        Observed two-step identities break ties but are not required when
        global completion feasibility already selects a unique matching.
        """

        if self.token_count % 2:
            raise ValueError("distinct inverse-pair completion needs an even token count")
        if minimum_total_support < 0:
            raise ValueError("minimum total support must be non-negative")

        def matchings(tokens: tuple[int, ...]):
            if not tokens:
                yield ()
                return
            first = tokens[0]
            for offset in range(1, len(tokens)):
                second = tokens[offset]
                remaining = tokens[1:offset] + tokens[offset + 1 :]
                for tail in matchings(remaining):
                    yield ((first, second),) + tail

        candidates = []
        for pairing in matchings(tuple(range(self.token_count))):
            inverse_tokens = np.empty(self.token_count, dtype=np.int64)
            for left, right in pairing:
                inverse_tokens[left] = right
                inverse_tokens[right] = left
            total_support = 0
            valid = True
            for left, right in pairing:
                pair_support = 0
                for source in range(self.state_count):
                    target = self.next_states[source, left]
                    if target >= 0 and self.next_states[target, right] >= 0:
                        if self.next_states[target, right] != source:
                            valid = False
                            break
                        pair_support += 1
                    target = self.next_states[source, right]
                    if target >= 0 and self.next_states[target, left] >= 0:
                        if self.next_states[target, left] != source:
                            valid = False
                            break
                        pair_support += 1
                total_support += pair_support
            if not valid or total_support < minimum_total_support:
                continue

            completed = self.next_states.copy()
            changed = True
            while valid and changed:
                changed = False
                for token, inverse in enumerate(inverse_tokens):
                    for source in range(self.state_count):
                        target = completed[source, token]
                        if target < 0:
                            continue
                        reverse = completed[target, inverse]
                        if reverse >= 0 and reverse != source:
                            valid = False
                            break
                        if reverse < 0:
                            completed[target, inverse] = source
                            changed = True
                    if not valid:
                        break
            expected = np.arange(self.state_count)
            complete_permutations = valid and np.all(completed >= 0) and all(
                np.array_equal(np.sort(completed[:, token]), expected)
                for token in range(self.token_count)
            )
            if complete_permutations:
                candidates.append((total_support, pairing, completed))

        if not candidates:
            raise ValueError(
                "no inverse-token matching completes the observed action family"
            )
        candidates.sort(reverse=True, key=lambda item: item[0])
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            raise ValueError("inverse-token matching is ambiguous")
        _, pairing, completed = candidates[0]
        inverse_tokens = np.empty(self.token_count, dtype=np.int64)
        for left, right in pairing:
            inverse_tokens[left] = right
            inverse_tokens[right] = left
        inferred = self.next_states < 0
        self.next_states[:] = completed
        self.counts[inferred] = 0
        return tuple(int(value) for value in inverse_tokens)
