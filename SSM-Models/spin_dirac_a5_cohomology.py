"""Exact low-degree cochain contraction for the binary icosahedral group.

For a finite group G over a field in which |G| is invertible, normalized
averaging contracts positive-degree inhomogeneous group cochains.  This module
does not merely quote that theorem: it builds the exact 120-element 2.A5 table
over Q(sqrt(5)) and checks the universal coefficient cancellations in degrees
one and two for every output group tuple.

The result applies to every Q(sqrt(5))-linear G-module, including the adjoint
so(n) modules in the Spin(3), Spin(8), ..., Spin(12) ladder.  It closes H1 and
H2 for the group-cohomological deformation complex while keeping raw
presentation-relator syzygies logically separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from spin_dirac_a5_rigidity import (
    FIELD,
    enumerate_exact_binary_icosahedral_group,
    exact_quaternion_multiply,
    exact_quaternion_rotation,
)

OperatorTag = tuple[str, int]
CochainKey = tuple[int, ...]


def build_exact_group_table() -> tuple[
    tuple[tuple[Any, Any, Any, Any], ...], tuple[tuple[int, ...], ...]
]:
    """Return exact group elements and their deterministic multiplication table."""

    group = enumerate_exact_binary_icosahedral_group()
    lookup = {element: index for index, element in enumerate(group)}
    table = tuple(
        tuple(lookup[exact_quaternion_multiply(left, right)] for right in group)
        for left in group
    )
    return group, table


def _is_zero_matrix(matrix: Any) -> bool:
    return all(value == FIELD.zero for value in matrix.to_list_flat())


def group_table_diagnostics(
    group: tuple[tuple[Any, Any, Any, Any], ...],
    table: tuple[tuple[int, ...], ...],
) -> dict[str, object]:
    """Check the complete finite group and exact vector-action contracts."""

    order = len(group)
    identity = 0
    right_rows_are_permutations = all(
        sorted(row) == list(range(order)) for row in table
    )
    left_columns_are_permutations = all(
        sorted(table[row][column] for row in range(order)) == list(range(order))
        for column in range(order)
    )
    identity_ok = all(
        table[identity][element] == element and table[element][identity] == element
        for element in range(order)
    )
    inverse_ok = all(
        any(
            table[element][candidate] == identity
            and table[candidate][element] == identity
            for candidate in range(order)
        )
        for element in range(order)
    )
    associativity_ok = all(
        table[table[first][second]][third] == table[first][table[second][third]]
        for first in range(order)
        for second in range(order)
        for third in range(order)
    )
    unit_norm_ok = all(
        sum(coordinate * coordinate for coordinate in quaternion) == FIELD.one
        for quaternion in group
    )
    rotations = tuple(exact_quaternion_rotation(quaternion) for quaternion in group)
    action_homomorphism_ok = all(
        _is_zero_matrix(
            rotations[left].matmul(rotations[right]) - rotations[table[left][right]]
        )
        for left in range(order)
        for right in range(order)
    )
    table_sha256 = hashlib.sha256(
        json.dumps(table, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    checks = {
        "order_is_120": order == 120,
        "identity": identity_ok,
        "inverses": inverse_ok,
        "associativity": associativity_ok,
        "right_rows_are_permutations": right_rows_are_permutations,
        "left_columns_are_permutations": left_columns_are_permutations,
        "unit_quaternions": unit_norm_ok,
        "exact_vector_action_homomorphism": action_homomorphism_ok,
    }
    return {
        "order": order,
        "identity_index": identity,
        "multiplication_table_sha256": table_sha256,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _add_term(
    coefficients: defaultdict[tuple[CochainKey, OperatorTag], int],
    key: CochainKey,
    operator: OperatorTag,
    value: int,
) -> None:
    term = (key, operator)
    coefficients[term] += value
    if coefficients[term] == 0:
        del coefficients[term]


def degree_one_homotopy_certificate(
    table: tuple[tuple[int, ...], ...],
) -> dict[str, object]:
    """Verify ``d h1 + h2 d = id`` on all formal one-cochain outputs."""

    order = len(table)
    identity_operator = ("identity", 0)
    maximum_unexpected_terms = 0
    expected_coefficient_minimum = order
    expected_coefficient_maximum = order
    passed = True
    for group_element in range(order):
        action = ("action", group_element)
        coefficients: defaultdict[tuple[CochainKey, OperatorTag], int] = defaultdict(
            int
        )
        for averaged in range(order):
            # d h1 with h1(f) = -|G|^-1 sum_k f(k)
            _add_term(coefficients, (averaged,), action, -1)
            _add_term(coefficients, (averaged,), identity_operator, 1)
            # h2 d with h2(c)(g) = |G|^-1 sum_k c(g,k)
            _add_term(coefficients, (averaged,), action, 1)
            _add_term(
                coefficients,
                (table[group_element][averaged],),
                identity_operator,
                -1,
            )
            _add_term(coefficients, (group_element,), identity_operator, 1)
        expected = {((group_element,), identity_operator): order}
        unexpected = sum(
            1
            for term, coefficient in coefficients.items()
            if term not in expected or coefficient != expected[term]
        )
        maximum_unexpected_terms = max(maximum_unexpected_terms, unexpected)
        observed = coefficients.get(((group_element,), identity_operator), 0)
        expected_coefficient_minimum = min(expected_coefficient_minimum, observed)
        expected_coefficient_maximum = max(expected_coefficient_maximum, observed)
        passed &= coefficients == expected
    return {
        "degree": 1,
        "outputs_checked": order,
        "scaled_identity_coefficient": order,
        "observed_identity_coefficient_range": [
            expected_coefficient_minimum,
            expected_coefficient_maximum,
        ],
        "maximum_unexpected_terms": maximum_unexpected_terms,
        "homotopy": "h1=-average_last, h2=+average_last",
        "passed": bool(passed),
    }


def degree_two_homotopy_certificate(
    table: tuple[tuple[int, ...], ...],
) -> dict[str, object]:
    """Verify ``d h2 + h3 d = id`` on all formal two-cochain outputs."""

    order = len(table)
    identity_operator = ("identity", 0)
    maximum_unexpected_terms = 0
    expected_coefficient_minimum = order
    expected_coefficient_maximum = order
    passed = True
    for first in range(order):
        action = ("action", first)
        for second in range(order):
            product = table[first][second]
            coefficients: defaultdict[tuple[CochainKey, OperatorTag], int] = (
                defaultdict(int)
            )
            for averaged in range(order):
                # d h2 for h2(c)(g) = |G|^-1 sum_k c(g,k)
                _add_term(coefficients, (second, averaged), action, 1)
                _add_term(coefficients, (product, averaged), identity_operator, -1)
                _add_term(coefficients, (first, averaged), identity_operator, 1)
                # h3 d for h3(c)(g,h) = -|G|^-1 sum_k c(g,h,k)
                _add_term(coefficients, (second, averaged), action, -1)
                _add_term(coefficients, (product, averaged), identity_operator, 1)
                _add_term(
                    coefficients,
                    (first, table[second][averaged]),
                    identity_operator,
                    -1,
                )
                _add_term(coefficients, (first, second), identity_operator, 1)
            expected = {((first, second), identity_operator): order}
            unexpected = sum(
                1
                for term, coefficient in coefficients.items()
                if term not in expected or coefficient != expected[term]
            )
            maximum_unexpected_terms = max(maximum_unexpected_terms, unexpected)
            observed = coefficients.get(((first, second), identity_operator), 0)
            expected_coefficient_minimum = min(expected_coefficient_minimum, observed)
            expected_coefficient_maximum = max(expected_coefficient_maximum, observed)
            passed &= coefficients == expected
    return {
        "degree": 2,
        "outputs_checked": order * order,
        "scaled_identity_coefficient": order,
        "observed_identity_coefficient_range": [
            expected_coefficient_minimum,
            expected_coefficient_maximum,
        ],
        "maximum_unexpected_terms": maximum_unexpected_terms,
        "homotopy": "h2=+average_last, h3=-average_last",
        "passed": bool(passed),
    }


def diagnostics() -> dict[str, object]:
    group, table = build_exact_group_table()
    group_report = group_table_diagnostics(group, table)
    degree_one = degree_one_homotopy_certificate(table)
    degree_two = degree_two_homotopy_certificate(table)
    passed = group_report["passed"] and degree_one["passed"] and degree_two["passed"]
    return {
        "schema_version": 1,
        "experiment": "exact low-degree 2.A5 group-cochain contraction",
        "field": "Q(sqrt(5))",
        "group": group_report,
        "contraction": {
            "formula": "h_n(f)(g1,...,g(n-1))=(-1)^n |G|^-1 sum_k f(g1,...,g(n-1),k)",
            "degree_one": degree_one,
            "degree_two": degree_two,
        },
        "consequences": {
            "h1_dimension_for_every_linear_module": 0,
            "h2_dimension_for_every_linear_module": 0,
            "applies_to": [
                "the adjoint so(n) modules for n=3,8,9,10,11,12",
                "the fixed 2.A5 Spin-ladder embeddings",
            ],
        },
        "claim_scope": {
            "computer_assisted_exact": [
                "the exact 120-element group table is associative",
                "the exact quaternion vector action is a homomorphism",
                "d h1 + h2 d is identity on all formal degree-one outputs",
                "d h2 + h3 d is identity on all formal degree-two outputs",
                "H1 and H2 vanish over Q(sqrt(5)) for every linear module",
            ],
            "not_claimed": [
                "classification of global representation components",
                "absence of stabilizers in the quotient stack",
                "that the raw three-relator presentation cokernel is H2",
                "an ML or SSM advantage",
            ],
        },
        "passed": bool(passed),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = diagnostics()
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
