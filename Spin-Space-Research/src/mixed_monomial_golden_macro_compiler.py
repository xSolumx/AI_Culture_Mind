"""Exact compiler for the finite monomial-golden-monomial macro walks.

The higher-weight certificate studies the symmetric probability distribution
obtained from independent choices ``n1 ~ N``, ``h ~ H``, ``n2 ~ N``.  Its
representation operator is ``M_N M_H M_N``.  Many labelled triples produce
the same exact 8-by-8 matrix, so a runtime implementation should store one
matrix per exact value plus a small labelled-triple lookup table.

This module compiles those dictionaries over Q(sqrt(5)), preserves
multiplicity, verifies inverse symmetry, and proves that weighted dictionary
averaging reproduces ``M_N M_H M_N`` exactly.  It does not benchmark hardware;
timing is deliberately kept in a separate empirical artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import sympy as sp

import mixed_monomial_golden_closure as closure
import mixed_monomial_golden_higher_weight as higher
import mixed_monomial_golden_mixing as low
import octonion_operator_groups as monomial
import spin8_triality_2a5_closure as golden

FieldMatrix = low.FieldMatrix


@dataclass(frozen=True)
class CompiledMacroDictionary:
    """Exact macro matrices, multiplicities, and labelled-triple lookup."""

    matrices: tuple[FieldMatrix, ...]
    multiplicities: tuple[int, ...]
    lookup: tuple[tuple[tuple[int, ...], ...], ...]
    monomial_count: int
    golden_count: int

    @property
    def labelled_triple_count(self) -> int:
        return self.monomial_count * self.golden_count * self.monomial_count


def compile_macro_dictionary(
    monomial_steps: tuple[FieldMatrix, ...],
    golden_steps: tuple[FieldMatrix, ...],
) -> CompiledMacroDictionary:
    """Compile exact ``n1*h*n2`` products in deterministic label order."""

    matrices: list[FieldMatrix] = []
    positions: dict[FieldMatrix, int] = {}
    multiplicities: list[int] = []
    lookup: list[tuple[tuple[int, ...], ...]] = []
    for left in monomial_steps:
        left_rows = []
        for middle in golden_steps:
            left_middle = golden._matrix_product(left, middle)
            row = []
            for right in monomial_steps:
                product = golden._matrix_product(left_middle, right)
                position = positions.get(product)
                if position is None:
                    position = len(matrices)
                    positions[product] = position
                    matrices.append(product)
                    multiplicities.append(0)
                multiplicities[position] += 1
                row.append(position)
            left_rows.append(tuple(row))
        lookup.append(tuple(left_rows))
    return CompiledMacroDictionary(
        matrices=tuple(matrices),
        multiplicities=tuple(multiplicities),
        lookup=tuple(lookup),
        monomial_count=len(monomial_steps),
        golden_count=len(golden_steps),
    )


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _dictionary_matrix_hash(dictionary: CompiledMacroDictionary) -> str:
    return _canonical_hash(
        [low._matrix_strings(matrix) for matrix in dictionary.matrices]
    )


def _lookup_hash(dictionary: CompiledMacroDictionary) -> str:
    return _canonical_hash(dictionary.lookup)


def _multiplicity_hash(dictionary: CompiledMacroDictionary) -> str:
    return _canonical_hash(dictionary.multiplicities)


def _weighted_dictionary_mean(
    dictionary: CompiledMacroDictionary,
) -> FieldMatrix:
    total = dictionary.labelled_triple_count
    return higher._matrix_linear_combination(
        tuple(
            (
                low._field(sp.Rational(multiplicity, total)),
                matrix,
            )
            for matrix, multiplicity in zip(
                dictionary.matrices,
                dictionary.multiplicities,
                strict=True,
            )
        )
    )


def _inverse_symmetry(dictionary: CompiledMacroDictionary) -> bool:
    positions = {
        matrix: position for position, matrix in enumerate(dictionary.matrices)
    }
    return all(
        (inverse_position := positions.get(golden._transpose(matrix))) is not None
        and dictionary.multiplicities[inverse_position] == multiplicity
        for matrix, multiplicity in zip(
            dictionary.matrices,
            dictionary.multiplicities,
            strict=True,
        )
    )


def _source_is_so8(matrix: FieldMatrix) -> bool:
    return bool(
        golden._matrix_product(golden._transpose(matrix), matrix)
        == golden._identity_matrix(8)
        and golden._determinant_is_one(matrix)
    )


def _view_certificate(
    monomial_steps: tuple[FieldMatrix, ...],
    golden_steps: tuple[FieldMatrix, ...],
) -> tuple[dict[str, object], CompiledMacroDictionary]:
    dictionary = compile_macro_dictionary(monomial_steps, golden_steps)
    monomial_mean = low._mean_matrix(monomial_steps)
    golden_mean = low._mean_matrix(golden_steps)
    expected_mean = higher._sandwich_mean(monomial_mean, golden_mean)
    compiled_mean = _weighted_dictionary_mean(dictionary)
    histogram = Counter(dictionary.multiplicities)
    distinct_count = len(dictionary.matrices)
    maximum_index = distinct_count - 1
    lookup_index_bytes = 2 if maximum_index < 2**16 else 4
    checks = {
        "all_source_steps_are_exactly_in_SO8": all(
            _source_is_so8(matrix)
            for matrix in (*monomial_steps, *golden_steps)
        ),
        "lookup_shape_matches_labelled_alphabets": (
            len(dictionary.lookup) == len(monomial_steps)
            and all(len(rows) == len(golden_steps) for rows in dictionary.lookup)
            and all(
                len(row) == len(monomial_steps)
                for rows in dictionary.lookup
                for row in rows
            )
        ),
        "multiplicities_sum_to_all_labelled_triples": (
            sum(dictionary.multiplicities) == dictionary.labelled_triple_count
        ),
        "weighted_dictionary_is_inverse_symmetric": _inverse_symmetry(dictionary),
        "weighted_dictionary_mean_equals_MN_MH_MN": compiled_mean == expected_mean,
        "dictionary_fits_unsigned_sixteen_bit_lookup": maximum_index < 2**16,
    }
    report = {
        "monomial_symmetric_label_count": len(monomial_steps),
        "golden_symmetric_label_count": len(golden_steps),
        "labelled_triple_count": dictionary.labelled_triple_count,
        "distinct_exact_matrix_count": distinct_count,
        "deduplication_ratio": str(
            sp.Rational(distinct_count, dictionary.labelled_triple_count)
        ),
        "multiplicity_histogram": {
            str(multiplicity): count
            for multiplicity, count in sorted(histogram.items())
        },
        "minimum_multiplicity": min(dictionary.multiplicities),
        "maximum_multiplicity": max(dictionary.multiplicities),
        "exact_matrix_sequence_sha256": _dictionary_matrix_hash(dictionary),
        "labelled_lookup_sha256": _lookup_hash(dictionary),
        "multiplicity_sequence_sha256": _multiplicity_hash(dictionary),
        "runtime_storage_bytes": {
            "float32_matrix_table": distinct_count * 8 * 8 * 4,
            "float16_matrix_table": distinct_count * 8 * 8 * 2,
            "uint16_labelled_lookup": (
                dictionary.labelled_triple_count * lookup_index_bytes
            ),
            "float32_table_plus_uint16_lookup": (
                distinct_count * 8 * 8 * 4
                + dictionary.labelled_triple_count * lookup_index_bytes
            ),
        },
        "sampling_contract": (
            "sample labelled triples uniformly and use the lookup, or sample "
            "distinct matrices according to stored multiplicities; uniform "
            "sampling over distinct matrices is a different measure"
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }
    return report, dictionary


def certificate() -> dict[str, object]:
    """Return the deterministic exact macro-compiler certificate."""

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
        report, _ = _view_certificate(monomial_steps, golden_steps)
        view_reports[view] = report

    checks = {
        "operator_group_input_passed": bool(operator_report["passed"]),
        "signed_fano_input_passed": bool(automorphism_report["passed"]),
        "monomial_symmetric_label_count_is_seventeen": len(monomial_steps) == 17,
        "all_three_macro_dictionaries_passed": all(
            report["passed"] for report in view_reports.values()
        ),
        "half_spin_dictionary_combinatorics_agree": (
            view_reports["positive_half_spin"]["labelled_triple_count"]
            == view_reports["negative_half_spin"]["labelled_triple_count"]
            and view_reports["positive_half_spin"]["distinct_exact_matrix_count"]
            == view_reports["negative_half_spin"]["distinct_exact_matrix_count"]
            and view_reports["positive_half_spin"]["multiplicity_histogram"]
            == view_reports["negative_half_spin"]["multiplicity_histogram"]
        ),
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "experiment": "exact finite N-H-N macro dictionary compiler",
        "field": "Q(sqrt(5))",
        "matrix_shape": [8, 8],
        "views": view_reports,
        "checks": checks,
        "claim_scope": {
            "proved": [
                "the exact distinct dictionary sizes and multiplicity histograms for all three fixed views",
                "multiplicity-weighted inverse symmetry of every compiled dictionary",
                "exact equality between weighted dictionary averaging and M_N M_H M_N",
                "the displayed matrix-table and lookup byte counts for the stated scalar and index formats",
            ],
            "not_claimed": [
                "uniform sampling over distinct matrices has the certified sandwich spectrum",
                "lookup plus one matrix application is faster than three primitive applications",
                "float16 preserves the exact algebra or the certified spectral bounds",
                "an SSM accuracy, training, or end-to-end inference advantage",
            ],
        },
        "passed": all(checks.values()),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["certificate_sha256_without_self_hash"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return payload


def runtime_dictionaries() -> dict[str, CompiledMacroDictionary]:
    """Compile and return the three exact dictionaries for benchmarking."""

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
        result[view] = compile_macro_dictionary(monomial_steps, golden_steps)
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
