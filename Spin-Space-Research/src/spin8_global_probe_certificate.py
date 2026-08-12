"""Exact global certificate for one five-probe Spin(8) triality sensor.

The earlier five-probe harness established full differential rank.  This module
uses the invariant triality products instead: any Spin(8) element fixing the
input probes must fix their complete triality closure.  For the canonical
five-probe tuple below that closure contains an exact basis of all three
eight-dimensional representations, so its global stabilizer is trivial.

The four-probe control has a three-dimensional exact Lie annihilator.  Its
three displayed generators close as su(2), hence exponentiating any of them
produces a continuous family of globally indistinguishable actions.

This proves an explicit global five-versus-four separation.  It does not, by
itself, classify every generic probe allocation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import sympy as sp

from spin8_triality import SPIN8_PAIRS, build_spin8_triality_algebra

DIMENSION = 8
REPRESENTATIONS = ("vector", "positive", "negative")
FIVE_POSITIVE_INDICES = (0, 1, 2, 4)
FOUR_POSITIVE_INDICES = (0, 1, 2)


def _standard_basis(index: int) -> tuple[int, ...]:
    return tuple(1 if coordinate == index else 0 for coordinate in range(DIMENSION))


def _exact_algebra() -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Return the integral triality tensor and twice each generator family."""

    algebra = build_spin8_triality_algebra()
    rounded_rho = np.rint(algebra.rho).astype(np.int64)
    if not np.array_equal(algebra.rho, rounded_rho):
        raise AssertionError("the maintained triality tensor is not integral")

    scaled_families = []
    for family in (
        algebra.vector_generators,
        algebra.positive_generators,
        algebra.negative_generators,
    ):
        scaled = np.rint(2.0 * family).astype(np.int64)
        if not np.array_equal(2.0 * family, scaled):
            raise AssertionError("a maintained Spin(8) generator is not half-integral")
        scaled_families.append(scaled)
    return rounded_rho, tuple(scaled_families)


def _rank(vectors: Iterable[tuple[int, ...]]) -> int:
    rows = list(vectors)
    return 0 if not rows else int(sp.Matrix(rows).rank())


def _add_independent(basis: list[tuple[int, ...]], vector: Iterable[int]) -> bool:
    candidate = tuple(int(value) for value in vector)
    if not any(candidate):
        return False
    if _rank([*basis, candidate]) == len(basis):
        return False
    basis.append(candidate)
    return True


def _closure_certificate(positive_indices: tuple[int, ...]) -> dict[str, object]:
    rho, _ = _exact_algebra()
    vector_basis = [_standard_basis(0)]
    positive_basis = [_standard_basis(index) for index in positive_indices]
    negative_basis: list[tuple[int, ...]] = []
    history = [[_rank(vector_basis), _rank(positive_basis), _rank(negative_basis)]]

    while True:
        changed = False
        old_vector = list(vector_basis)
        old_positive = list(positive_basis)
        old_negative = list(negative_basis)

        for vector in old_vector:
            clifford = sum(
                (int(vector[index]) * rho[index] for index in range(DIMENSION)),
                start=np.zeros((DIMENSION, DIMENSION), dtype=np.int64),
            )
            for positive in old_positive:
                changed |= _add_independent(negative_basis, clifford @ positive)
            for negative in old_negative:
                changed |= _add_independent(positive_basis, clifford.T @ negative)

        for positive in old_positive:
            for negative in old_negative:
                bound_vector = [
                    int(np.asarray(negative) @ rho[index] @ np.asarray(positive))
                    for index in range(DIMENSION)
                ]
                changed |= _add_independent(vector_basis, bound_vector)

        ranks = [_rank(vector_basis), _rank(positive_basis), _rank(negative_basis)]
        if ranks != history[-1]:
            history.append(ranks)
        if not changed:
            break

    bases = {
        "vector": [list(row) for row in vector_basis],
        "positive": [list(row) for row in positive_basis],
        "negative": [list(row) for row in negative_basis],
    }
    determinants = {
        representation: (str(sp.Matrix(rows).det()) if len(rows) == DIMENSION else None)
        for representation, rows in bases.items()
    }
    return {
        "positive_probe_indices": list(positive_indices),
        "rank_history_vector_positive_negative": history,
        "final_ranks": history[-1],
        "bases": bases,
        "basis_determinants": determinants,
    }


def _coordinate_support_closure(
    positive_indices: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Close coordinate supports using the signed permutation triality tensor."""

    rho, _ = _exact_algebra()
    vector = {0}
    positive = set(positive_indices)
    negative: set[int] = set()
    while True:
        before = (set(vector), set(positive), set(negative))
        for vector_index in list(vector):
            for positive_index in list(positive):
                outputs = np.flatnonzero(rho[vector_index, :, positive_index])
                if len(outputs) != 1:
                    raise AssertionError("coordinate Clifford product is not monomial")
                negative.add(int(outputs[0]))
            for negative_index in list(negative):
                outputs = np.flatnonzero(rho[vector_index, negative_index, :])
                if len(outputs) != 1:
                    raise AssertionError("transpose Clifford product is not monomial")
                positive.add(int(outputs[0]))
        for positive_index in list(positive):
            for negative_index in list(negative):
                outputs = np.flatnonzero(rho[:, negative_index, positive_index])
                if len(outputs) != 1:
                    raise AssertionError("coordinate triality binding is not monomial")
                vector.add(int(outputs[0]))
        after = (vector, positive, negative)
        if after == before:
            return (
                tuple(sorted(vector)),
                tuple(sorted(positive)),
                tuple(sorted(negative)),
            )


def _gf2_rank(words: Iterable[int]) -> int:
    pivots: dict[int, int] = {}
    for original in words:
        word = int(original)
        while word:
            pivot = word.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = word
                break
            word ^= pivots[pivot]
    return len(pivots)


def _coordinate_atlas_certificate() -> dict[str, object]:
    rows = []
    exceptional_masks = []
    for indices in itertools.combinations(range(DIMENSION), 4):
        closure = _coordinate_support_closure(indices)
        closure_size = [len(support) for support in closure]
        rows.append(
            {
                "positive_indices": list(indices),
                "closure_sizes": closure_size,
            }
        )
        if closure_size == [4, 4, 4]:
            exceptional_masks.append(sum(1 << index for index in indices))
        elif closure_size != [8, 8, 8]:
            raise AssertionError(f"unexpected coordinate closure size {closure_size}")

    code = {0, (1 << DIMENSION) - 1, *exceptional_masks}
    xor_closed = all(left ^ right in code for left in code for right in code)
    weights = {weight: 0 for weight in range(DIMENSION + 1)}
    for word in code:
        weights[word.bit_count()] += 1
    nonzero_weights = {str(weight): count for weight, count in weights.items() if count}
    self_orthogonal = all(
        ((left & right).bit_count() % 2) == 0 for left in code for right in code
    )
    code_dimension = _gf2_rank(code)
    minimum_nonzero_weight = min(word.bit_count() for word in code if word)
    triple_counts = []
    for triple in itertools.combinations(range(DIMENSION), 3):
        triple_set = set(triple)
        triple_counts.append(
            sum(
                triple_set.issubset(
                    {index for index in range(DIMENSION) if mask & (1 << index)}
                )
                for mask in exceptional_masks
            )
        )

    return {
        "all_coordinate_four_subsets": rows,
        "full_closure_count": sum(row["closure_sizes"] == [8, 8, 8] for row in rows),
        "exceptional_closure_count": len(exceptional_masks),
        "exceptional_supports": [
            [index for index in range(DIMENSION) if mask & (1 << index)]
            for mask in exceptional_masks
        ],
        "binary_code": {
            "word_count": len(code),
            "dimension": code_dimension,
            "xor_closed": xor_closed,
            "self_orthogonal": self_orthogonal,
            "self_dual": self_orthogonal and code_dimension * 2 == DIMENSION,
            "doubly_even": all(word.bit_count() % 4 == 0 for word in code),
            "minimum_nonzero_weight": minimum_nonzero_weight,
            "weight_enumerator": nonzero_weights,
        },
        "steiner_s_3_4_8": {
            "triple_count": len(triple_counts),
            "minimum_blocks_per_triple": min(triple_counts),
            "maximum_blocks_per_triple": max(triple_counts),
            "every_triple_occurs_once": set(triple_counts) == {1},
        },
    }


def _primitive_integer_vector(vector: sp.Matrix) -> tuple[int, ...]:
    denominators = [sp.denom(value) for value in vector]
    scale = int(sp.ilcm(*[int(value) for value in denominators]))
    integers = [int(value * scale) for value in vector]
    divisor = math.gcd(*[abs(value) for value in integers if value])
    integers = [value // divisor for value in integers]
    first = next(value for value in integers if value)
    if first < 0:
        integers = [-value for value in integers]
    return tuple(integers)


def _annihilator_certificate(positive_indices: tuple[int, ...]) -> dict[str, object]:
    _, scaled_families = _exact_algebra()
    probes = (
        (0, _standard_basis(0)),
        *((1, _standard_basis(index)) for index in positive_indices),
    )
    columns = []
    for generator_index in range(len(SPIN8_PAIRS)):
        column: list[int] = []
        for representation, probe in probes:
            column.extend(
                int(value)
                for value in scaled_families[representation][generator_index]
                @ np.asarray(probe, dtype=np.int64)
            )
        columns.append(column)
    matrix = sp.Matrix(columns).T
    nullspace = [_primitive_integer_vector(vector) for vector in matrix.nullspace()]
    return {
        "scaled_constraint_matrix_shape": list(matrix.shape),
        "rank": int(matrix.rank()),
        "nullity": len(SPIN8_PAIRS) - int(matrix.rank()),
        "primitive_nullspace_coefficients": [list(vector) for vector in nullspace],
        "primitive_nullspace_terms": [
            [
                {"plane": list(SPIN8_PAIRS[index]), "coefficient": coefficient}
                for index, coefficient in enumerate(vector)
                if coefficient
            ]
            for vector in nullspace
        ],
    }


def _su2_certificate(nullspace: list[list[int]]) -> dict[str, object]:
    _, scaled_families = _exact_algebra()
    coefficients = [sp.Matrix(vector) for vector in nullspace]
    generators: list[list[sp.Matrix]] = []
    for coefficient in coefficients:
        representation_generators = []
        for family in scaled_families:
            matrix = sp.zeros(DIMENSION)
            for index, value in enumerate(coefficient):
                matrix += sp.Rational(int(value), 2) * sp.Matrix(family[index])
            representation_generators.append(matrix)
        generators.append(representation_generators)

    # Relations in the deterministic nullspace order.
    expected = {(0, 1): (2, 2), (0, 2): (1, -2), (1, 2): (0, 2)}
    relations = []
    for (left, right), (target, scale) in expected.items():
        exact = all(
            generators[left][representation] * generators[right][representation]
            - generators[right][representation] * generators[left][representation]
            == scale * generators[target][representation]
            for representation in range(3)
        )
        relations.append(
            {
                "left": left,
                "right": right,
                "target": target,
                "scale": scale,
                "all_three_representations_exact": exact,
            }
        )

    observed = [(0, _standard_basis(0))] + [
        (1, _standard_basis(index)) for index in FOUR_POSITIVE_INDICES
    ]
    annihilation = []
    hidden_motion = []
    for generator in generators:
        annihilation.append(
            all(
                generator[representation] * sp.Matrix(probe) == sp.zeros(DIMENSION, 1)
                for representation, probe in observed
            )
        )
        hidden_motion.append(
            [int(value) for value in generator[1] * sp.Matrix(_standard_basis(4))]
        )

    return {
        "commutator_relations": relations,
        "annihilates_all_four_observed_probes": annihilation,
        "motion_of_withheld_positive_e4": hidden_motion,
        "every_generator_moves_withheld_probe": all(any(row) for row in hidden_motion),
    }


def run() -> dict[str, object]:
    five_closure = _closure_certificate(FIVE_POSITIVE_INDICES)
    four_closure = _closure_certificate(FOUR_POSITIVE_INDICES)
    five_annihilator = _annihilator_certificate(FIVE_POSITIVE_INDICES)
    four_annihilator = _annihilator_certificate(FOUR_POSITIVE_INDICES)
    su2 = _su2_certificate(four_annihilator["primitive_nullspace_coefficients"])
    coordinate_atlas = _coordinate_atlas_certificate()

    five_full = bool(
        five_closure["final_ranks"] == [8, 8, 8]
        and all(
            value in {"-1", "1"}
            for value in five_closure["basis_determinants"].values()
        )
        and five_annihilator["rank"] == 28
        and five_annihilator["nullity"] == 0
    )
    four_continuous = bool(
        four_closure["final_ranks"] == [4, 4, 4]
        and four_annihilator["rank"] == 25
        and four_annihilator["nullity"] == 3
        and all(su2["annihilates_all_four_observed_probes"])
        and su2["every_generator_moves_withheld_probe"]
        and all(
            relation["all_three_representations_exact"]
            for relation in su2["commutator_relations"]
        )
    )
    hamming_atlas = bool(
        coordinate_atlas["full_closure_count"] == 56
        and coordinate_atlas["exceptional_closure_count"] == 14
        and coordinate_atlas["binary_code"]
        == {
            "word_count": 16,
            "dimension": 4,
            "xor_closed": True,
            "self_orthogonal": True,
            "self_dual": True,
            "doubly_even": True,
            "minimum_nonzero_weight": 4,
            "weight_enumerator": {"0": 1, "4": 14, "8": 1},
        }
        and coordinate_atlas["steiner_s_3_4_8"]["every_triple_occurs_once"]
    )
    return {
        "experiment": "exact global Spin8 triality five-probe certificate",
        "canonical_probe_allocation": {"vector": 1, "positive": 4, "negative": 0},
        "five_probe_triality_closure": five_closure,
        "five_probe_lie_annihilator": five_annihilator,
        "four_probe_triality_closure": four_closure,
        "four_probe_lie_annihilator": four_annihilator,
        "four_probe_su2_certificate": su2,
        "coordinate_four_probe_atlas": coordinate_atlas,
        "claims": {
            "explicit_five_probe_global_stabilizer_is_trivial": five_full,
            "explicit_four_probe_tuple_has_continuous_su2_stabilizer": four_continuous,
            "all_generic_five_probe_allocations_classified": False,
            "universal_four_probe_insufficiency_proved": False,
            "coordinate_exceptional_supports_form_extended_hamming_8_4_4": hamming_atlas,
        },
        "proof_boundary": (
            "The invariant closure proves global uniqueness for the displayed exact tuple; "
            "it does not classify every generic tuple or every four-probe allocation."
        ),
        "passed": five_full and four_continuous and hamming_atlas,
    }


def verify_report(report: dict[str, object]) -> bool:
    return report == run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run()
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
