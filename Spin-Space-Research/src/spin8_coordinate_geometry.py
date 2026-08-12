"""Exact binary classification of coordinate Spin(8) triality sensors.

The maintained octonionic triality tensor is monomial in its coordinate
bases.  Its support law is

    rho[i, k, j] != 0  iff  k = i XOR j.

Give the three triality representations the three nonzero colours in F_2^2
and the eight coordinates their labels in F_2^3.  A coordinate probe is then
a five-bit word.  Triality contraction between two different representations
is exactly addition of those words.

This module exhausts every multiview coordinate sensor with four or five
probes.  It compares actual tensor closure with the predicted binary span,
and computes exact rational Lie ranks for every distinct closure.  The result
is a sharp coordinate theorem: four probes never globally identify the
action; five do exactly when their five labels form a basis of F_2^5.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import sympy as sp

from spin8_triality import SPIN8_PAIRS, build_spin8_triality_algebra

DIMENSION = 8
REPRESENTATIONS = ("vector", "positive", "negative")
# The nonzero elements of F_2^2.  The property needed below is
# 1 XOR 2 == 3 and its permutations.
COLOURS = (1, 2, 3)

Probe = tuple[int, int]
Closure = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]


def _exact_algebra() -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    algebra = build_spin8_triality_algebra()
    rho = np.rint(algebra.rho).astype(np.int64)
    if not np.array_equal(rho, algebra.rho):
        raise AssertionError("the maintained triality tensor is not integral")
    families = tuple(
        np.rint(2.0 * family).astype(np.int64)
        for family in (
            algebra.vector_generators,
            algebra.positive_generators,
            algebra.negative_generators,
        )
    )
    if not all(
        np.array_equal(2.0 * family, scaled)
        for family, scaled in zip(
            (
                algebra.vector_generators,
                algebra.positive_generators,
                algebra.negative_generators,
            ),
            families,
            strict=True,
        )
    ):
        raise AssertionError("a maintained generator is not half-integral")
    return rho, families


def _word(probe: Probe) -> int:
    representation, coordinate = probe
    return (COLOURS[representation] << 3) | coordinate


def _gf2_span(words: Iterable[int]) -> set[int]:
    span = {0}
    for word in words:
        span |= {existing ^ int(word) for existing in tuple(span)}
    return span


def _gf2_rank(words: Iterable[int]) -> int:
    return (len(_gf2_span(words))).bit_length() - 1


def _support_law_certificate(rho: np.ndarray) -> dict[str, object]:
    signs = []
    failures = []
    for vector_index in range(DIMENSION):
        row = []
        for positive_index in range(DIMENSION):
            outputs = np.flatnonzero(rho[vector_index, :, positive_index])
            expected = vector_index ^ positive_index
            if len(outputs) != 1 or int(outputs[0]) != expected:
                failures.append(
                    [
                        vector_index,
                        positive_index,
                        outputs.tolist(),
                        expected,
                    ]
                )
                row.append(0)
            else:
                row.append(int(rho[vector_index, expected, positive_index]))
        signs.append(row)
    return {
        "checked_products": DIMENSION * DIMENSION,
        "support_is_coordinate_xor": not failures,
        "failures": failures,
        "nonzero_sign_table": signs,
    }


def _actual_closure(probes: Iterable[Probe], rho: np.ndarray) -> Closure:
    supports = [set() for _ in REPRESENTATIONS]
    for representation, coordinate in probes:
        supports[representation].add(coordinate)

    while True:
        before = tuple(frozenset(support) for support in supports)
        # V x S+ -> S-
        for vector_index in tuple(supports[0]):
            for positive_index in tuple(supports[1]):
                output = vector_index ^ positive_index
                if rho[vector_index, output, positive_index] == 0:
                    raise AssertionError("V x S+ XOR support law failed")
                supports[2].add(output)
            # V x S- -> S+
            for negative_index in tuple(supports[2]):
                output = vector_index ^ negative_index
                if rho[vector_index, negative_index, output] == 0:
                    raise AssertionError("V x S- XOR support law failed")
                supports[1].add(output)
        # S+ x S- -> V
        for positive_index in tuple(supports[1]):
            for negative_index in tuple(supports[2]):
                output = positive_index ^ negative_index
                if rho[output, negative_index, positive_index] == 0:
                    raise AssertionError("S+ x S- XOR support law failed")
                supports[0].add(output)
        if before == tuple(frozenset(support) for support in supports):
            return tuple(tuple(sorted(support)) for support in supports)  # type: ignore[return-value]


def _span_closure(probes: Iterable[Probe]) -> Closure:
    span = _gf2_span(_word(probe) for probe in probes)
    return tuple(
        tuple(
            sorted(
                word & 0b111 for word in span if (word >> 3) == COLOURS[representation]
            )
        )
        for representation in range(3)
    )  # type: ignore[return-value]


def _allocation(probes: Iterable[Probe]) -> tuple[int, int, int]:
    rows = tuple(probes)
    return tuple(
        sum(representation == target for representation, _ in rows)
        for target in range(3)
    )  # type: ignore[return-value]


def _constraint_matrix(
    closure: Closure, scaled_families: tuple[np.ndarray, ...]
) -> sp.Matrix:
    columns = []
    for generator_index in range(len(SPIN8_PAIRS)):
        column: list[int] = []
        for representation, coordinates in enumerate(closure):
            for coordinate in coordinates:
                column.extend(
                    int(value)
                    for value in scaled_families[representation][
                        generator_index, :, coordinate
                    ]
                )
        columns.append(column)
    return sp.Matrix(columns).T


def _exact_closure_rank_atlas(
    closures: Iterable[Closure], scaled_families: tuple[np.ndarray, ...]
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[Closure]] = defaultdict(list)
    for closure in sorted(set(closures)):
        matrix = _constraint_matrix(closure, scaled_families)
        rank = int(matrix.rank())
        grouped[(len(closure[0]), rank)].append(closure)
    return [
        {
            "coordinates_per_representation": coordinates,
            "exact_constraint_rank": rank,
            "exact_lie_nullity": len(SPIN8_PAIRS) - rank,
            "distinct_closures": len(rows),
            "representative": [list(support) for support in rows[0]],
        }
        for (coordinates, rank), rows in sorted(grouped.items())
    ]


def _enumerate_probe_count(
    probe_count: int, rho: np.ndarray
) -> tuple[dict[str, object], set[Closure]]:
    slots = tuple(
        (representation, coordinate)
        for representation in range(3)
        for coordinate in range(DIMENSION)
    )
    aggregate: Counter[tuple[tuple[int, int, int], int, tuple[int, int, int]]] = (
        Counter()
    )
    closures: set[Closure] = set()
    mismatches = []
    evaluated = 0

    for probes in itertools.combinations(slots, probe_count):
        allocation = _allocation(probes)
        if sum(count > 0 for count in allocation) < 2:
            continue
        evaluated += 1
        binary_rank = _gf2_rank(_word(probe) for probe in probes)
        actual = _actual_closure(probes, rho)
        predicted = _span_closure(probes)
        if actual != predicted:
            mismatches.append(
                {
                    "probes": [list(probe) for probe in probes],
                    "actual": [list(support) for support in actual],
                    "predicted": [list(support) for support in predicted],
                }
            )
        closures.add(actual)
        signature = tuple(len(support) for support in actual)
        aggregate[(allocation, binary_rank, signature)] += 1

    rows = [
        {
            "allocation_vector_positive_negative": list(allocation),
            "binary_rank": binary_rank,
            "closure_coordinates_per_representation": list(signature),
            "sensor_count": count,
        }
        for (allocation, binary_rank, signature), count in sorted(aggregate.items())
    ]
    rank_counts: Counter[int] = Counter()
    for (_, binary_rank, _), count in aggregate.items():
        rank_counts[binary_rank] += count
    return (
        {
            "probe_count": probe_count,
            "multiview_sensors_evaluated": evaluated,
            "actual_equals_binary_span_count": evaluated - len(mismatches),
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
            "binary_rank_counts": {
                str(rank): count for rank, count in sorted(rank_counts.items())
            },
            "full_triality_closure_count": rank_counts[5],
            "aggregate_by_allocation_rank_and_closure": rows,
            "distinct_closures": len(closures),
        },
        closures,
    )


def _representative_stabilizer_chain(
    rho: np.ndarray, scaled_families: tuple[np.ndarray, ...]
) -> dict[str, object]:
    # Rank-three example: V0, S+0, S-0, S-1.  The redundant S-0 is included
    # to keep the example at four probes; its closure is two coordinates per
    # representation.
    examples = {
        "binary_rank_3": ((0, 0), (1, 0), (2, 0), (2, 1)),
        "binary_rank_4": ((0, 0), (1, 0), (1, 1), (1, 2)),
        "binary_rank_5": ((0, 0), (1, 0), (1, 1), (1, 2), (1, 4)),
    }
    rows = {}
    for name, probes in examples.items():
        closure = _actual_closure(probes, rho)
        matrix = _constraint_matrix(closure, scaled_families)
        rank = int(matrix.rank())
        nullspace = matrix.nullspace()
        lie_type: dict[str, object]
        if nullspace:
            vector_basis = []
            for coefficients in nullspace:
                generator = sp.zeros(DIMENSION)
                for index, coefficient in enumerate(coefficients):
                    generator += coefficient * sp.Matrix(scaled_families[0][index])
                vector_basis.append(generator)
            flattened = sp.Matrix.hstack(
                *(
                    generator.reshape(DIMENSION * DIMENSION, 1)
                    for generator in vector_basis
                )
            )
            brackets: dict[tuple[int, int], sp.Matrix] = {}
            for left in range(len(vector_basis)):
                for right in range(len(vector_basis)):
                    bracket = (
                        vector_basis[left] * vector_basis[right]
                        - vector_basis[right] * vector_basis[left]
                    ).reshape(DIMENSION * DIMENSION, 1)
                    solution, parameters = flattened.gauss_jordan_solve(bracket)
                    if parameters.rows:
                        raise AssertionError("stabilizer basis is not independent")
                    brackets[(left, right)] = solution
            adjoint = []
            for left in range(len(vector_basis)):
                adjoint.append(
                    sp.Matrix.hstack(
                        *(brackets[(left, right)] for right in range(len(vector_basis)))
                    )
                )
            derived_rank = int(
                sp.Matrix.hstack(*brackets.values()).rank() if brackets else 0
            )
            centre_system = sp.Matrix.vstack(
                *(
                    sp.Matrix.hstack(
                        *(brackets[(left, right)] for left in range(len(vector_basis)))
                    )
                    for right in range(len(vector_basis))
                )
            )
            centre_dimension = len(vector_basis) - int(centre_system.rank())
            killing = sp.Matrix(
                [
                    [
                        sp.trace(adjoint[left] * adjoint[right])
                        for right in range(len(adjoint))
                    ]
                    for left in range(len(adjoint))
                ]
            )
            leading_signs = [
                sp.sign(killing[:order, :order].det())
                for order in range(1, killing.rows + 1)
            ]
            negative_definite = all(
                sign == (-1) ** order
                for order, sign in enumerate(leading_signs, start=1)
            )
            dimension = len(vector_basis)
            if dimension == 8 and derived_rank == 8 and centre_dimension == 0:
                classification = "compact semisimple A2, hence su(3)"
            elif dimension == 3 and derived_rank == 3 and centre_dimension == 0:
                classification = "compact simple A1, hence su(2)"
            else:
                classification = "unclassified"
            lie_type = {
                "dimension": dimension,
                "derived_algebra_rank": derived_rank,
                "centre_dimension": centre_dimension,
                "killing_form_determinant": str(killing.det()),
                "killing_form_leading_principal_minor_signs": [
                    int(sign) for sign in leading_signs
                ],
                "killing_form_negative_definite": negative_definite,
                "classification": classification,
            }
        else:
            lie_type = {
                "dimension": 0,
                "derived_algebra_rank": 0,
                "centre_dimension": 0,
                "killing_form_determinant": "1",
                "killing_form_leading_principal_minor_signs": [],
                "killing_form_negative_definite": True,
                "classification": "trivial Lie stabilizer",
            }
        rows[name] = {
            "probes": [list(probe) for probe in probes],
            "closure": [list(support) for support in closure],
            "exact_constraint_rank": rank,
            "exact_lie_nullity": len(SPIN8_PAIRS) - rank,
            "exact_lie_type_certificate": lie_type,
        }
    rows["interpretation"] = {
        "binary_rank_3": "8-dimensional compact stabilizer, the classical SU(3) stage",
        "binary_rank_4": "3-dimensional compact stabilizer, the classical SU(2) stage",
        "binary_rank_5": "zero-dimensional Lie stabilizer and full invariant closure",
    }
    return rows


def run() -> dict[str, object]:
    rho, scaled_families = _exact_algebra()
    support_law = _support_law_certificate(rho)
    four, four_closures = _enumerate_probe_count(4, rho)
    five, five_closures = _enumerate_probe_count(5, rho)
    all_closures = four_closures | five_closures
    lie_atlas = _exact_closure_rank_atlas(all_closures, scaled_families)
    stabilizer_chain = _representative_stabilizer_chain(rho, scaled_families)

    four_never_full = four["full_triality_closure_count"] == 0
    five_iff_basis = bool(
        five["full_triality_closure_count"]
        == five["binary_rank_counts"].get("5", 0)
        == 21504
    )
    exact_lie_pattern = lie_atlas == [
        {
            "coordinates_per_representation": 2,
            "exact_constraint_rank": 20,
            "exact_lie_nullity": 8,
            "distinct_closures": 112,
            "representative": lie_atlas[0]["representative"],
        },
        {
            "coordinates_per_representation": 4,
            "exact_constraint_rank": 25,
            "exact_lie_nullity": 3,
            "distinct_closures": 28,
            "representative": lie_atlas[1]["representative"],
        },
        {
            "coordinates_per_representation": 8,
            "exact_constraint_rank": 28,
            "exact_lie_nullity": 0,
            "distinct_closures": 1,
            "representative": lie_atlas[2]["representative"],
        },
    ]
    passed = bool(
        support_law["support_is_coordinate_xor"]
        and four["mismatch_count"] == 0
        and five["mismatch_count"] == 0
        and four_never_full
        and five_iff_basis
        and exact_lie_pattern
        and stabilizer_chain["binary_rank_3"]["exact_lie_type_certificate"][
            "classification"
        ]
        == "compact semisimple A2, hence su(3)"
        and stabilizer_chain["binary_rank_4"]["exact_lie_type_certificate"][
            "classification"
        ]
        == "compact simple A1, hence su(2)"
    )
    return {
        "experiment": "exact F2^5 coordinate geometry of Spin8 triality probes",
        "support_law": support_law,
        "four_probe_atlas": four,
        "five_probe_atlas": five,
        "exact_lie_rank_by_distinct_closure": lie_atlas,
        "representative_stabilizer_chain": stabilizer_chain,
        "claims": {
            "coordinate_probe_is_five_bit_word": support_law[
                "support_is_coordinate_xor"
            ],
            "every_multiview_four_and_five_probe_closure_equals_binary_span": bool(
                four["mismatch_count"] == five["mismatch_count"] == 0
            ),
            "four_coordinate_probes_never_globally_identify": four_never_full,
            "five_coordinate_probes_identify_iff_labels_are_an_F2_basis": five_iff_basis,
            "continuous_noncoordinate_sensor_space_classified": False,
        },
        "proof_boundary": (
            "This is an exhaustive exact theorem for coordinate probes in the fixed "
            "triality bases. It does not classify arbitrary continuous probes."
        ),
        "passed": passed,
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
