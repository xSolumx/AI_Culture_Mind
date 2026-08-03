"""Construct adversarial reverse-cover masks via an exact 2-SAT reduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_recurrences import GROUPS, make_group_batches, parse_input_elements
from latent_group_discovery import TransitionEvidence, exact_inverse_tokens


INPUT_LABELS = ("23145", "31245", "23451", "51234")
MATCHINGS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))


def full_evidence() -> TransitionEvidence:
    group = GROUPS["a5"]
    inputs = parse_input_elements(INPUT_LABELS, group)
    result = TransitionEvidence(group.order, len(inputs))
    for tokens, targets in make_group_batches(
        group, 4, 512, 16, 18_901, input_elements=inputs
    ):
        result.observe(tokens, targets)
    if not result.complete:
        raise RuntimeError("source evidence is incomplete")
    return result


def reverse_pair_variables(
    transitions: np.ndarray, inverse_tokens: tuple[int, ...]
) -> tuple[
    list[tuple[tuple[int, int], tuple[int, int]]],
    dict[tuple[int, int], tuple[int, bool]],
]:
    pairs = []
    coordinate_literals = {}
    seen = set()
    for source in range(transitions.shape[0]):
        for token in range(transitions.shape[1]):
            reverse = (
                int(transitions[source, token]),
                inverse_tokens[token],
            )
            pair = tuple(sorted(((source, token), reverse)))
            if pair in seen:
                continue
            seen.add(pair)
            variable = len(pairs)
            pairs.append(pair)
            coordinate_literals[pair[0]] = (variable, False)
            coordinate_literals[pair[1]] = (variable, True)
    return pairs, coordinate_literals


def solve_2sat(
    clauses: list[tuple[tuple[int, bool], tuple[int, bool]]], variables: int
) -> list[bool] | None:
    adjacency = [[] for _ in range(2 * variables)]
    reverse = [[] for _ in range(2 * variables)]

    def node(literal: tuple[int, bool]) -> int:
        variable, positive = literal
        return 2 * variable + int(positive)

    def negate(literal: tuple[int, bool]) -> tuple[int, bool]:
        return literal[0], not literal[1]

    for left, right in clauses:
        for source, target in ((negate(left), right), (negate(right), left)):
            source_node, target_node = node(source), node(target)
            adjacency[source_node].append(target_node)
            reverse[target_node].append(source_node)

    visited = [False] * (2 * variables)
    order = []

    def visit(vertex: int) -> None:
        visited[vertex] = True
        for target in adjacency[vertex]:
            if not visited[target]:
                visit(target)
        order.append(vertex)

    for vertex in range(2 * variables):
        if not visited[vertex]:
            visit(vertex)

    component = [-1] * (2 * variables)

    def assign(vertex: int, label: int) -> None:
        component[vertex] = label
        for target in reverse[vertex]:
            if component[target] < 0:
                assign(target, label)

    for vertex in reversed(order):
        if component[vertex] < 0:
            assign(vertex, vertex)
    if any(
        component[2 * variable] == component[2 * variable + 1]
        for variable in range(variables)
    ):
        return None
    solution = [
        component[2 * variable + 1] > component[2 * variable]
        for variable in range(variables)
    ]
    if not all(
        solution[left[0]] == left[1] or solution[right[0]] == right[1]
        for left, right in clauses
    ):
        solution = [not value for value in solution]
    if not all(
        solution[left[0]] == left[1] or solution[right[0]] == right[1]
        for left, right in clauses
    ):
        raise RuntimeError("2-SAT assignment does not satisfy its clauses")
    return solution


def matching_inverse(matching) -> np.ndarray:
    result = np.empty(4, dtype=np.int64)
    for left, right in matching:
        result[left] = right
        result[right] = left
    return result


def adversarial_solution(
    transitions: np.ndarray,
    pairs,
    coordinate_literals,
    matching,
) -> tuple[list[bool] | None, int]:
    inverse = matching_inverse(matching)
    clauses = []
    wrong_identity_support = 0
    for source in range(transitions.shape[0]):
        for token in range(transitions.shape[1]):
            inverse_token = int(inverse[token])
            predecessor = int(np.flatnonzero(
                transitions[:, inverse_token] == source
            )[0])
            observed = coordinate_literals[(source, token)]
            counterpart = coordinate_literals[(predecessor, inverse_token)]
            # Candidate completion needs at least one of these directions.
            clauses.append((observed, counterpart))
            if int(transitions[source, token]) == predecessor:
                wrong_identity_support += 1
            else:
                # Observing both would directly contradict the candidate.
                clauses.append(
                    (
                        (observed[0], not observed[1]),
                        (counterpart[0], not counterpart[1]),
                    )
                )
    return solve_2sat(clauses, len(pairs)), wrong_identity_support


def evidence_from_solution(
    full: TransitionEvidence, pairs, solution: list[bool]
) -> TransitionEvidence:
    partial = TransitionEvidence(full.state_count, full.token_count)
    for variable, choice in enumerate(solution):
        source, token = pairs[variable][int(choice)]
        partial.next_states[source, token] = full.next_states[source, token]
        partial.counts[source, token] = full.counts[source, token]
    return partial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    full = full_evidence()
    true_inverse = exact_inverse_tokens(full.next_states)
    pairs, coordinate_literals = reverse_pair_variables(
        full.next_states, true_inverse
    )
    results = []
    for matching in MATCHINGS:
        solution, identity_support = adversarial_solution(
            full.next_states, pairs, coordinate_literals, matching
        )
        learner_outcome = None
        exact_replay = False
        calibration_resolution_passes = 0
        if solution is not None:
            partial = evidence_from_solution(full, pairs, solution)
            try:
                inferred = partial.infer_inverse_pairs_and_complete()
                learner_outcome = {"status": "completed", "inferred": list(inferred)}
                exact_replay = bool(
                    np.array_equal(partial.next_states, full.next_states)
                )
            except ValueError as error:
                learner_outcome = {"status": "refused", "error": str(error)}
            if tuple(matching) != MATCHINGS[0]:
                for calibrated_variable, pair in enumerate(pairs):
                    calibrated = evidence_from_solution(full, pairs, solution)
                    for source, token in pair:
                        calibrated.next_states[source, token] = full.next_states[
                            source, token
                        ]
                        calibrated.counts[source, token] = full.counts[
                            source, token
                        ]
                    try:
                        inferred = calibrated.infer_inverse_pairs_and_complete()
                        calibration_resolution_passes += int(
                            inferred == true_inverse
                            and np.array_equal(
                                calibrated.next_states, full.next_states
                            )
                        )
                    except ValueError:
                        pass
        results.append(
            {
                "matching": [list(pair) for pair in matching],
                "is_true_matching": tuple(matching) == MATCHINGS[0],
                "two_sat_feasible": solution is not None,
                "full_action_two_step_identity_count": identity_support,
                "orientation_bits": (
                    [int(value) for value in solution]
                    if solution is not None
                    else None
                ),
                "learner_outcome": learner_outcome,
                "exact_true_action_replay": exact_replay,
                "one_calibration_resolution": (
                    {
                        "choices": len(pairs),
                        "exact_recovery_passes": calibration_resolution_passes,
                        "all_choices_pass": calibration_resolution_passes == len(pairs),
                    }
                    if not tuple(matching) == MATCHINGS[0]
                    else None
                ),
            }
        )
    report = {
        "experiment": "adversarial exact-half inverse-cover identifiability",
        "states": full.state_count,
        "tokens": full.token_count,
        "reverse_pair_variables": len(pairs),
        "observed_edges_per_mask": len(pairs),
        "true_inverse_tokens": list(true_inverse),
        "results": results,
        "universal_exact_half_identifiability_falsified": any(
            result["two_sat_feasible"]
            and not result["is_true_matching"]
            for result in results
        ),
        "safe_failure_on_ambiguous_masks": all(
            result["learner_outcome"]["status"] == "refused"
            for result in results
            if not result["is_true_matching"] and result["two_sat_feasible"]
        ),
        "one_calibration_resolves_every_adversarial_choice": all(
            result["one_calibration_resolution"]["all_choices_pass"]
            for result in results
            if not result["is_true_matching"] and result["two_sat_feasible"]
        ),
        "one_calibration_pair_resolution": (
            "The true matching has a two-step identity and every wrong matching "
            "has zero such support; one revealed reverse pair gives the true "
            "matching a strict score advantage and forces the other token pair."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
