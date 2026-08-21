"""Exact theorem certificate for the topological closure of the mixed system.

Let H be generated in the maintained eight-dimensional real carrier by the
signed-monomial octonion-operator normalizer generators and the golden vector
pair.  The companion mixed-closure certificate supplies an explicit
infinite-order element W.  This module proves that the closure of H is SO(8).

The proof is finite exact linear algebra:

* the monomial normalizer acts on so(8) as two non-isomorphic irreducibles,
  V_7 plus V_21, realized by Clifford grades one and two;
* any nonzero Lie algebra of the compact closure is a normalizer-invariant
  sum of those two pieces;
* V_7 is not a Lie algebra, and W does not normalize V_21;
* infinitude rules out the zero Lie algebra.

No floating-point quantity participates in acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

from octonion_operator_groups import (
    _automorphism_group_certificate,
    _operator_group_certificate,
    closure,
    matrix_from_monomial,
)
from spin8_mixed_monomial_golden_closure import (
    ONE,
    ZERO,
    Matrix,
    Pair,
    _add,
    _cyclotomic_factor_gate,
    _identity,
    _mixed_generators,
    _multiply,
    _product,
    _sympy_matrix,
)


def _transpose(matrix: Matrix) -> Matrix:
    return tuple(tuple(matrix[column][row] for column in range(len(matrix))) for row in range(len(matrix)))


def _trace_inner(left: Matrix, right: Matrix) -> Pair:
    """Frobenius inner product tr(left^T right) in Q(sqrt(5))."""

    total = ZERO
    for row in range(len(left)):
        for column in range(len(left)):
            total = _add(total, _multiply(left[row][column], right[row][column]))
    return total


def _pair_string(value: Pair) -> str:
    if value[1] == 0:
        return str(value[0])
    if value[0] == 0:
        return f"{value[1]}*sqrt(5)"
    return f"{value[0]} + ({value[1]})*sqrt(5)"


def _integer_matrix(matrix: Matrix) -> np.ndarray:
    result = np.asarray(
        [[int(value[0]) for value in row] for row in matrix], dtype=np.int64
    )
    if any(value[1] != 0 for row in matrix for value in row):
        raise ValueError("expected an integral signed-monomial matrix")
    return result


def _signed_action_on_grade_one(
    conjugator: np.ndarray, grade_one: tuple[np.ndarray, ...]
) -> np.ndarray:
    """Return the 7-by-7 signed permutation for Clifford grade one."""

    action = np.zeros((7, 7), dtype=np.int64)
    for source, basis in enumerate(grade_one):
        image = conjugator @ basis @ conjugator.T
        matches = [
            (target, sign)
            for target, candidate in enumerate(grade_one)
            for sign in (-1, 1)
            if np.array_equal(image, sign * candidate)
        ]
        if len(matches) != 1:
            raise AssertionError("normalizer failed to preserve Clifford grade one")
        target, sign = matches[0]
        action[target, source] = sign
    return action


def _exact_rank(matrices: tuple[np.ndarray, ...]) -> int:
    columns = [sp.Matrix(matrix.reshape(-1, 1).tolist()) for matrix in matrices]
    return int(sp.Matrix.hstack(*columns).rank())


def _is_special_orthogonal(matrix: Matrix) -> bool:
    return bool(
        _product(_transpose(matrix), matrix) == _identity(len(matrix))
        and sp.simplify(_sympy_matrix(matrix).det() - 1) == 0
    )


def _character_report(
    normalizer: set[tuple[int, ...]], grade_one: tuple[np.ndarray, ...]
) -> dict[str, object]:
    characters_seven: list[int] = []
    characters_twenty_one: list[int] = []
    for element in normalizer:
        action = _signed_action_on_grade_one(matrix_from_monomial(element), grade_one)
        character_seven = int(np.trace(action))
        character_twenty_one = (
            character_seven * character_seven - int(np.trace(action @ action))
        ) // 2
        characters_seven.append(character_seven)
        characters_twenty_one.append(character_twenty_one)
    order = len(normalizer)
    seven_norm_sum = sum(value * value for value in characters_seven)
    twenty_one_norm_sum = sum(value * value for value in characters_twenty_one)
    cross_sum = sum(
        left * right for left, right in zip(characters_seven, characters_twenty_one, strict=True)
    )
    if seven_norm_sum % order or twenty_one_norm_sum % order or cross_sum % order:
        raise AssertionError("character inner products were not integral")
    return {
        "normalizer_order": order,
        "grade_one_character_inner_product": seven_norm_sum // order,
        "grade_two_character_inner_product": twenty_one_norm_sum // order,
        "cross_character_inner_product": cross_sum // order,
        "grade_one_character_values": {
            str(key): value
            for key, value in sorted(Counter(characters_seven).items())
        },
        "grade_two_character_values": {
            str(key): value
            for key, value in sorted(Counter(characters_twenty_one).items())
        },
    }


def certificate() -> dict[str, object]:
    """Build the exact SO(8)-closure theorem certificate."""

    operator_report, _, operator_generators = _operator_group_certificate()
    automorphism_report, _, automorphism_generators = _automorphism_group_certificate()
    normalizer_generators = (*operator_generators, *automorphism_generators)
    normalizer = closure(normalizer_generators, maximum_order=21504)
    components, names, mixed_generators = _mixed_generators()
    if tuple(names[:9]) != (
        "left_operator_e1",
        "left_operator_e2",
        "left_operator_e3",
        "left_operator_e4",
        "left_operator_e5",
        "left_operator_e6",
        "left_operator_e7",
        "fano_automorphism_a",
        "fano_automorphism_b",
    ):
        raise AssertionError("mixed generator convention changed")

    grade_one_pair = tuple(mixed_generators[:7])
    grade_two_pair = tuple(
        _product(grade_one_pair[left], grade_one_pair[right])
        for left in range(7)
        for right in range(left + 1, 7)
    )
    grade_one = tuple(_integer_matrix(matrix) for matrix in grade_one_pair)
    grade_two = tuple(_integer_matrix(matrix) for matrix in grade_two_pair)
    grade_one_rank = _exact_rank(grade_one)
    grade_two_rank = _exact_rank(grade_two)
    full_rank = _exact_rank((*grade_one, *grade_two))
    skew_symmetric = all(
        np.array_equal(matrix.T, -matrix) for matrix in (*grade_one, *grade_two)
    )
    grade_orthogonal = all(
        _trace_inner(left, right) == ZERO
        for left in grade_one_pair
        for right in grade_two_pair
    )
    grade_one_norms_are_eight = all(
        _trace_inner(basis, basis) == (8, 0) for basis in grade_one_pair
    )
    character = _character_report(normalizer, grade_one)

    # This is exactly the already identified non-cyclotomic word, written in
    # maintained generator order: golden_b * fano_b * left_operator_e1.
    witness = _product(
        mixed_generators[10], _product(mixed_generators[8], mixed_generators[0])
    )
    cyclotomic_gate = _cyclotomic_factor_gate(witness)
    bivector_14 = _product(grade_one_pair[0], grade_one_pair[3])
    conjugated_bivector = _product(
        witness, _product(bivector_14, _transpose(witness))
    )
    grade_one_projection = [
        _pair_string(
            (
                _trace_inner(basis, conjugated_bivector)[0] / 8,
                _trace_inner(basis, conjugated_bivector)[1] / 8,
            )
        )
        for basis in grade_one_pair
    ]
    nonzero_projection = any(value != "0" for value in grade_one_projection)
    bracket = _product(grade_one_pair[0], grade_one_pair[1])
    twice_bivector = tuple(
        tuple((2 * value[0], 2 * value[1]) for value in row)
        for row in bracket
    )
    reverse_bracket = _product(grade_one_pair[1], grade_one_pair[0])
    # Evaluate [L1,L2] with ordinary associative matrix products.
    commutator = tuple(
        tuple(
            (
                bracket[row][column][0] - reverse_bracket[row][column][0],
                bracket[row][column][1] - reverse_bracket[row][column][1],
            )
            for column in range(8)
        )
        for row in range(8)
    )
    checks = {
        "operator_input_passed": bool(operator_report["passed"]),
        "automorphism_input_passed": bool(automorphism_report["passed"]),
        "normalizer_order_is_21504": len(normalizer) == 21504,
        "all_stated_generators_are_in_so8": all(
            _is_special_orthogonal(generator) for generator in mixed_generators
        ),
        "grade_one_dimension_is_7": grade_one_rank == 7,
        "grade_two_dimension_is_21": grade_two_rank == 21,
        "grades_span_so8": full_rank == 28 and skew_symmetric,
        "grade_decomposition_is_frobenius_orthogonal": grade_orthogonal,
        "grade_one_frobenius_norms_are_eight": grade_one_norms_are_eight,
        "grade_one_is_irreducible": character["grade_one_character_inner_product"] == 1,
        "grade_two_is_irreducible": character["grade_two_character_inner_product"] == 1,
        "grades_are_nonisomorphic": character["cross_character_inner_product"] == 0,
        "grade_one_is_not_lie_closed": commutator == twice_bivector,
        "infinite_word_has_noncyclotomic_factor": bool(
            cyclotomic_gate["has_non_cyclotomic_factor"]
        ),
        "infinite_word_does_not_normalize_grade_two": nonzero_projection,
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "experiment": "exact SO(8) topological-closure theorem for the mixed monomial--golden system",
        "generator_system": {
            "dimension": 8,
            "coefficient_field": "Q(sqrt(5))",
            "monomial_normalizer_order": len(normalizer),
            "mixed_generator_names": list(names),
            "infinite_word": "golden_vector_b*(fano_automorphism_b*(left_operator_e1))",
        },
        "clifford_grade_decomposition": {
            "grade_one_dimension": grade_one_rank,
            "grade_two_dimension": grade_two_rank,
            "combined_dimension": full_rank,
            "ambient_so8_dimension": 28,
            "character_orthogonality": character,
            "grade_one_bracket_witness": "[L_e1,L_e2] = 2 L_e1 L_e2 in grade two",
        },
        "infinite_word_gate": cyclotomic_gate,
        "grade_two_non_normalization_witness": {
            "tested_bivector": "L_e1*L_e4",
            "grade_one_frobenius_coefficients": grade_one_projection,
            "interpretation": "a nonzero grade-one coefficient proves conjugation leaves the grade-two subspace",
        },
        "proof": [
            "the mixed group is a subgroup of SO(8), and the non-cyclotomic witness makes its compact closure infinite",
            "an infinite closed subgroup of a compact Lie group has nonzero Lie algebra",
            "the identity-component Lie algebra is normalizer-invariant and the exact character calculation leaves only 0, V7, V21, or V7+V21 as possible invariant subspaces",
            "V7 is not a Lie subalgebra because its displayed bracket lies in V21",
            "V21 cannot be the identity-component Lie algebra because the displayed group element fails to normalize V21",
            "therefore the closure Lie algebra is so(8), and a closed subgroup of connected SO(8) with this Lie algebra is SO(8)",
        ],
        "proved_conclusion": {
            "statement": "the topological closure of the stated mixed monomial--golden subgroup is SO(8)",
            "equivalent_statement": "the stated mixed subgroup is dense in SO(8)",
        },
        "claim_scope": {
            "proved_exactly": [
                "the SO(8) topological-closure theorem for the stated eleven-generator mixed system",
                "the grade-one plus grade-two normalizer-module decomposition and character irreducibility checks",
            ],
            "not_claimed": [
                "that the dense subgroup is a previously unknown abstract group",
                "a learned-model accuracy, memory, or kernel-speed advantage",
                "a closure theorem for differently parameterized Spin(8) layers without these fixed generators",
            ],
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["certificate_sha256_without_self_hash"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    report = certificate()
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
