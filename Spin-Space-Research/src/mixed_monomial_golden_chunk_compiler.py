"""Exact every-prefix chunk compiler for labelled N-H-N words.

For a labelled word ``left * middle * right`` acting on a column state, the
causal primitive order is right, middle, left.  One 24-by-8 stacked operator
maps the chunk input to all three prefix states:

``[right; middle*right; left*middle*right] @ x``.

This module constructs every such operator over Q(sqrt(5)), verifies that its
last block equals the exact endpoint macro dictionary, and records the small
finite-table memory envelope.  Runtime timing remains a separate empirical
gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import mixed_monomial_golden_closure as closure
import mixed_monomial_golden_macro_compiler as macro
import mixed_monomial_golden_mixing as low
import octonion_operator_groups as monomial
import spin8_triality_2a5_closure as golden

FieldMatrix = low.FieldMatrix
PrefixOperator = tuple[tuple[low.FieldElement, ...], ...]


@dataclass(frozen=True)
class CompiledPrefixTable:
    """Flat deterministic table of labelled 24-by-8 prefix operators."""

    operators: tuple[PrefixOperator, ...]
    monomial_count: int
    golden_count: int

    @property
    def labelled_triple_count(self) -> int:
        return self.monomial_count * self.golden_count * self.monomial_count

    def flat_index(self, left: int, middle: int, right: int) -> int:
        return (
            (left * self.golden_count + middle) * self.monomial_count + right
        )


def compile_prefix_table(
    monomial_steps: tuple[FieldMatrix, ...],
    golden_steps: tuple[FieldMatrix, ...],
) -> CompiledPrefixTable:
    operators = []
    for left in monomial_steps:
        for middle in golden_steps:
            for right in monomial_steps:
                second = golden._matrix_product(middle, right)
                endpoint = golden._matrix_product(left, second)
                operators.append((*right, *second, *endpoint))
    return CompiledPrefixTable(
        operators=tuple(operators),
        monomial_count=len(monomial_steps),
        golden_count=len(golden_steps),
    )


def _operator_strings(operator: PrefixOperator) -> list[list[str]]:
    return [
        [low._field_string(value) for value in row]
        for row in operator
    ]


def _operator_sequence_hash(table: CompiledPrefixTable) -> str:
    canonical = json.dumps(
        [_operator_strings(operator) for operator in table.operators],
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _view_certificate(
    monomial_steps: tuple[FieldMatrix, ...],
    golden_steps: tuple[FieldMatrix, ...],
) -> dict[str, object]:
    table = compile_prefix_table(monomial_steps, golden_steps)
    endpoint_dictionary = macro.compile_macro_dictionary(
        monomial_steps, golden_steps
    )
    first_blocks_match = True
    second_blocks_match = True
    endpoint_blocks_match = True
    for left_index, left in enumerate(monomial_steps):
        for middle_index, middle in enumerate(golden_steps):
            for right_index, right in enumerate(monomial_steps):
                operator = table.operators[
                    table.flat_index(left_index, middle_index, right_index)
                ]
                second = golden._matrix_product(middle, right)
                endpoint_position = endpoint_dictionary.lookup[left_index][
                    middle_index
                ][right_index]
                endpoint = endpoint_dictionary.matrices[endpoint_position]
                first_blocks_match &= operator[:8] == right
                second_blocks_match &= operator[8:16] == second
                endpoint_blocks_match &= operator[16:24] == endpoint

    distinct_count = len(set(table.operators))
    checks = {
        "table_has_one_operator_per_labelled_triple": (
            len(table.operators) == table.labelled_triple_count
        ),
        "every_prefix_operator_is_exactly_distinct": (
            distinct_count == table.labelled_triple_count
        ),
        "first_block_is_right_action": first_blocks_match,
        "second_block_is_middle_times_right": second_blocks_match,
        "third_block_matches_exact_macro_endpoint": endpoint_blocks_match,
    }
    return {
        "monomial_symmetric_label_count": len(monomial_steps),
        "golden_symmetric_label_count": len(golden_steps),
        "labelled_triple_count": table.labelled_triple_count,
        "prefix_operator_shape": [24, 8],
        "distinct_exact_prefix_operator_count": distinct_count,
        "exact_prefix_operator_sequence_sha256": _operator_sequence_hash(table),
        "runtime_storage_bytes": {
            "float32_full_labelled_prefix_table": (
                table.labelled_triple_count * 24 * 8 * 4
            ),
            "float16_full_labelled_prefix_table": (
                table.labelled_triple_count * 24 * 8 * 2
            ),
        },
        "causal_contract": {
            "matrix_word": "left * middle * right",
            "primitive_application_order": ["right", "middle", "left"],
            "output_blocks": [
                "right",
                "middle * right",
                "left * middle * right",
            ],
            "emits_every_primitive_prefix": True,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def certificate() -> dict[str, object]:
    """Return the exact every-prefix chunk-compiler certificate."""

    operator_report, _, operator_generators = monomial._operator_group_certificate()
    automorphism_report, _, automorphism_generators = (
        monomial._automorphism_group_certificate()
    )
    normalizer_generators = (*operator_generators, *automorphism_generators)
    monomial_steps = tuple(
        closure._field_matrix_from_monomial(step)
        for step in closure._symmetric_monomial_steps(normalizer_generators)
    )
    triality_matrices = golden._spin8_action_matrices()
    view_reports = {}
    for view, indices in closure.VIEW_GENERATOR_INDICES.items():
        pair = tuple(triality_matrices[index] for index in indices)
        golden_steps = closure._symmetric_golden_steps(pair)
        view_reports[view] = _view_certificate(monomial_steps, golden_steps)

    checks = {
        "operator_group_input_passed": bool(operator_report["passed"]),
        "signed_fano_input_passed": bool(automorphism_report["passed"]),
        "all_three_prefix_tables_passed": all(
            report["passed"] for report in view_reports.values()
        ),
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "experiment": "exact every-prefix N-H-N chunk compiler",
        "field": "Q(sqrt(5))",
        "views": view_reports,
        "checks": checks,
        "claim_scope": {
            "proved": [
                "one exact 24-by-8 labelled operator emits all three causal prefix states for every fixed N-H-N word",
                "the third block agrees with the exact endpoint macro dictionary for every labelled triple",
                "the displayed full-table byte counts for float32 and float16 storage",
            ],
            "not_claimed": [
                "one stacked 24-by-8 application is faster than three 8-by-8 primitive applications",
                "the table applies to arbitrary learned continuous transitions",
                "float16 preserves the exact field identities or spectral theorem",
                "an end-to-end SSM training or inference advantage",
            ],
        },
        "passed": all(checks.values()),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["certificate_sha256_without_self_hash"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return payload


def runtime_prefix_tables() -> dict[str, CompiledPrefixTable]:
    """Compile and return every view table for the empirical benchmark."""

    _, _, operator_generators = monomial._operator_group_certificate()
    _, _, automorphism_generators = monomial._automorphism_group_certificate()
    normalizer_generators = (*operator_generators, *automorphism_generators)
    monomial_steps = tuple(
        closure._field_matrix_from_monomial(step)
        for step in closure._symmetric_monomial_steps(normalizer_generators)
    )
    triality_matrices = golden._spin8_action_matrices()
    result = {}
    for view, indices in closure.VIEW_GENERATOR_INDICES.items():
        pair = tuple(triality_matrices[index] for index in indices)
        golden_steps = closure._symmetric_golden_steps(pair)
        result[view] = compile_prefix_table(monomial_steps, golden_steps)
    return result


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
