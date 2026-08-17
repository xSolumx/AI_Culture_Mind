"""Exact higher-weight bottleneck and compiled-sandwich certificates.

The low-degree mixed-walk certificate covers the defining 8, adjoint 28, and
traceless-symmetric 35 representations.  This continuation constructs
``exterior^3`` (dimension 56) and the two Hodge-star eigenspaces of
``exterior^4`` (dimensions 35+35) exactly over Q(sqrt(5)).

One Hodge sector exposes a real obstruction: the maintained monomial subgroup
fixes a unique Cayley four-form.  The uniform union walk therefore mixes that
sector much more slowly than the other displayed representations.  A symmetric
three-letter macro distribution ``N * H * N`` (monomial, golden, monomial)
breaks the bottleneck more strongly per compiled macro-step.  The module
certifies that comparison by exact LDL^T positivity and exact Rayleigh
witnesses.  It makes no full L2(SO(8)), primitive-operation efficiency, or ML
quality claim.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections.abc import Sequence
from pathlib import Path

import sympy as sp

import mixed_monomial_golden_closure as closure
import mixed_monomial_golden_mixing as low
import octonion_operator_groups as monomial
import spin8_triality_2a5_closure as golden

FieldMatrix = low.FieldMatrix
FieldElement = low.FieldElement

REPRESENTATION_NAMES = (
    "defining_8",
    "adjoint_28",
    "traceless_symmetric_35",
    "exterior_3_56",
    "hodge_plus_35",
    "hodge_minus_35",
)

ORIGINAL_RADIUS_BOUNDS = {
    "vector": {
        "defining_8": sp.Rational(1, 3),
        "adjoint_28": sp.Rational(5, 8),
        "traceless_symmetric_35": sp.Rational(1, 3),
        "exterior_3_56": sp.Rational(7, 25),
        "hodge_plus_35": sp.Rational(1, 3),
        "hodge_minus_35": sp.Rational(24, 25),
    },
    "positive_half_spin": {
        "defining_8": sp.Rational(1, 5),
        "adjoint_28": sp.Rational(3, 4),
        "traceless_symmetric_35": sp.Rational(3, 8),
        "exterior_3_56": sp.Rational(1, 5),
        "hodge_plus_35": sp.Rational(3, 8),
        "hodge_minus_35": sp.Rational(99, 100),
    },
    "negative_half_spin": {
        "defining_8": sp.Rational(1, 5),
        "adjoint_28": sp.Rational(3, 4),
        "traceless_symmetric_35": sp.Rational(3, 8),
        "exterior_3_56": sp.Rational(1, 5),
        "hodge_plus_35": sp.Rational(3, 8),
        "hodge_minus_35": sp.Rational(99, 100),
    },
}

SANDWICH_RADIUS_BOUNDS = {
    "vector": {
        "defining_8": sp.Rational(1, 25),
        "adjoint_28": sp.Rational(1, 2),
        "traceless_symmetric_35": sp.Rational(1, 10),
        "exterior_3_56": sp.Rational(1, 25),
        "hodge_plus_35": sp.Rational(1, 10),
        "hodge_minus_35": sp.Rational(17, 20),
    },
    "positive_half_spin": {
        "defining_8": sp.Rational(1, 100),
        "adjoint_28": sp.Rational(2, 5),
        "traceless_symmetric_35": sp.Rational(1, 10),
        "exterior_3_56": sp.Rational(1, 100),
        "hodge_plus_35": sp.Rational(1, 10),
        "hodge_minus_35": sp.Rational(97, 100),
    },
    "negative_half_spin": {
        "defining_8": sp.Rational(1, 100),
        "adjoint_28": sp.Rational(2, 5),
        "traceless_symmetric_35": sp.Rational(1, 10),
        "exterior_3_56": sp.Rational(1, 100),
        "hodge_plus_35": sp.Rational(1, 10),
        "hodge_minus_35": sp.Rational(97, 100),
    },
}

CAYLEY_COEFFICIENTS = {
    (0, 1, 2, 3): -1,
    (0, 1, 4, 5): -1,
    (0, 1, 6, 7): 1,
    (0, 2, 4, 6): -1,
    (0, 2, 5, 7): -1,
    (0, 3, 4, 7): -1,
    (0, 3, 5, 6): 1,
}

RAYLEIGH_WITNESSES = {
    "vector": {
        (0, 1, 2, 3): 5,
        (0, 1, 4, 5): 4,
        (0, 2, 4, 6): 1,
    },
    "positive_half_spin": {
        (0, 1, 2, 3): 10,
        (0, 1, 4, 5): 10,
        (0, 1, 6, 7): 1,
        (0, 2, 4, 6): 7,
        (0, 2, 5, 7): 1,
        (0, 3, 4, 7): -2,
        (0, 3, 5, 6): 5,
    },
    "negative_half_spin": {
        (0, 1, 2, 3): 10,
        (0, 1, 4, 5): 10,
        (0, 1, 6, 7): 1,
        (0, 2, 4, 6): 7,
        (0, 2, 5, 7): 1,
        (0, 3, 4, 7): -2,
        (0, 3, 5, 6): 5,
    },
}


def _permutations_with_sign(
    degree: int,
) -> tuple[tuple[tuple[int, ...], int], ...]:
    result = []
    for permutation in itertools.permutations(range(degree)):
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(degree)
            for right in range(left + 1, degree)
        )
        result.append((permutation, -1 if inversions % 2 else 1))
    return tuple(result)


def _minor_determinant(
    matrix: FieldMatrix,
    rows: tuple[int, ...],
    columns: tuple[int, ...],
    permutations: Sequence[tuple[tuple[int, ...], int]],
) -> FieldElement:
    total = golden.FIELD.zero
    for permutation, sign in permutations:
        term = golden.FIELD.one
        for row_index, column_index in enumerate(permutation):
            value = matrix[rows[row_index]][columns[column_index]]
            if value == golden.FIELD.zero:
                term = golden.FIELD.zero
                break
            term *= value
        total += sign * term
    return total


def _exterior_power(matrix: FieldMatrix, degree: int) -> FieldMatrix:
    indices = tuple(itertools.combinations(range(len(matrix)), degree))
    permutations = _permutations_with_sign(degree)
    return tuple(
        tuple(
            _minor_determinant(matrix, rows, columns, permutations)
            for columns in indices
        )
        for rows in indices
    )


def _complement(index: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(value for value in range(8) if value not in index)


def _orientation_sign(
    index: tuple[int, ...], complement: tuple[int, ...]
) -> int:
    concatenated = (*index, *complement)
    inversions = sum(
        concatenated[left] > concatenated[right]
        for left in range(8)
        for right in range(left + 1, 8)
    )
    return -1 if inversions % 2 else 1


def _hodge_basis() -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    representatives = []
    signs = []
    for index in itertools.combinations(range(8), 4):
        complement = _complement(index)
        if index < complement:
            representatives.append(index)
            signs.append(_orientation_sign(index, complement))
    if len(representatives) != 35:
        raise AssertionError("unexpected Hodge-pair count")
    return tuple(representatives), tuple(signs)


def _hodge_four_blocks(
    matrix: FieldMatrix,
) -> tuple[FieldMatrix, FieldMatrix, bool]:
    full = _exterior_power(matrix, 4)
    all_indices = tuple(itertools.combinations(range(8), 4))
    positions = {index: position for position, index in enumerate(all_indices)}
    representatives, signs = _hodge_basis()

    blocks = []
    invariant = True
    for eigenvalue in (1, -1):
        block = tuple(
            tuple(
                full[positions[target]][positions[source]]
                + eigenvalue
                * signs[column]
                * full[positions[target]][positions[_complement(source)]]
                for column, source in enumerate(representatives)
            )
            for target in representatives
        )
        blocks.append(block)
        for column, source in enumerate(representatives):
            source_complement = _complement(source)
            for row, target in enumerate(representatives):
                target_complement = _complement(target)
                complement_coefficient = (
                    full[positions[target_complement]][positions[source]]
                    + eigenvalue
                    * signs[column]
                    * full[positions[target_complement]][
                        positions[source_complement]
                    ]
                )
                invariant &= complement_coefficient == (
                    eigenvalue * signs[row] * block[row][column]
                )
    return blocks[0], blocks[1], invariant


def _representations(matrix: FieldMatrix) -> tuple[dict[str, FieldMatrix], bool]:
    hodge_plus, hodge_minus, hodge_invariant = _hodge_four_blocks(matrix)
    return (
        {
            "defining_8": matrix,
            "adjoint_28": low._wedge_square(matrix),
            "traceless_symmetric_35": low._traceless_symmetric_square(matrix),
            "exterior_3_56": _exterior_power(matrix, 3),
            "hodge_plus_35": hodge_plus,
            "hodge_minus_35": hodge_minus,
        },
        hodge_invariant,
    )


def _matrix_linear_combination(
    matrices: Sequence[tuple[FieldElement, FieldMatrix]],
) -> FieldMatrix:
    dimension = len(matrices[0][1])
    return tuple(
        tuple(
            sum(
                (
                    coefficient * matrix[row][column]
                    for coefficient, matrix in matrices
                ),
                golden.FIELD.zero,
            )
            for column in range(dimension)
        )
        for row in range(dimension)
    )


def _label_union_mean(
    monomial_mean: FieldMatrix,
    monomial_count: int,
    golden_mean: FieldMatrix,
    golden_count: int,
) -> FieldMatrix:
    total = monomial_count + golden_count
    return _matrix_linear_combination(
        (
            (_field_fraction(monomial_count, total), monomial_mean),
            (_field_fraction(golden_count, total), golden_mean),
        )
    )


def _field_fraction(numerator: int, denominator: int) -> FieldElement:
    return low._field(sp.Rational(numerator, denominator))


def _sandwich_mean(
    monomial_mean: FieldMatrix, golden_mean: FieldMatrix
) -> FieldMatrix:
    return low._matrix_product(
        low._matrix_product(monomial_mean, golden_mean), monomial_mean
    )


def _vector_from_sparse(
    coefficients: dict[tuple[int, ...], int]
) -> tuple[FieldElement, ...]:
    representatives, _ = _hodge_basis()
    return tuple(low._field(coefficients.get(index, 0)) for index in representatives)


def _matrix_vector_product(
    matrix: FieldMatrix, vector: Sequence[FieldElement]
) -> tuple[FieldElement, ...]:
    return tuple(
        sum(
            (matrix[row][column] * vector[column] for column in range(len(vector))),
            golden.FIELD.zero,
        )
        for row in range(len(matrix))
    )


def _rayleigh_quotient(
    matrix: FieldMatrix, vector: Sequence[FieldElement]
) -> FieldElement:
    image = _matrix_vector_product(matrix, vector)
    numerator = sum(
        (left * right for left, right in zip(vector, image, strict=True)),
        golden.FIELD.zero,
    )
    denominator = sum(
        (value * value for value in vector), golden.FIELD.zero
    )
    return numerator / denominator


def _sparse_vector_report(
    coefficients: dict[tuple[int, ...], int]
) -> list[dict[str, object]]:
    return [
        {"four_index": list(index), "coefficient": coefficient}
        for index, coefficient in coefficients.items()
    ]


def _fixed_cayley_certificate(
    monomial_representations: Sequence[dict[str, FieldMatrix]],
) -> dict[str, object]:
    vector = _vector_from_sparse(CAYLEY_COEFFICIENTS)
    fixed_by_every_step = all(
        _matrix_vector_product(report["hodge_minus_35"], vector) == vector
        for report in monomial_representations
    )
    identity = sp.eye(35)
    equations = sp.Matrix.vstack(
        *(
            low._field_matrix_to_sympy(report["hodge_minus_35"]) - identity
            for report in monomial_representations
        )
    )
    rank = equations.rank()
    return {
        "orientation_convention": "e_0 wedge e_1 wedge ... wedge e_7 is positive",
        "hodge_sector": "minus",
        "sparse_reduced_basis_coefficients": _sparse_vector_report(
            CAYLEY_COEFFICIENTS
        ),
        "reduced_basis_norm_squared": sum(
            value * value for value in CAYLEY_COEFFICIENTS.values()
        ),
        "fixed_space_equation_rows": equations.rows,
        "fixed_space_equation_columns": equations.cols,
        "fixed_space_equation_rank": rank,
        "fixed_space_dimension": 35 - rank,
        "fixed_by_all_seventeen_monomial_steps": fixed_by_every_step,
        "passed": bool(fixed_by_every_step and rank == 34),
    }


def _representation_metric(name: str, gram: FieldMatrix) -> FieldMatrix | None:
    return gram if name == "traceless_symmetric_35" else None


def _measure_certificate(
    means: dict[str, FieldMatrix],
    bounds: dict[str, sp.Rational],
    gram: FieldMatrix,
) -> dict[str, object]:
    representations = {
        name: low._ldlt_radius_certificate(
            means[name], bounds[name], gram=_representation_metric(name, gram)
        )
        for name in REPRESENTATION_NAMES
    }
    return {
        "representations": representations,
        "all_six_representation_bounds_passed": all(
            report["passed"] for report in representations.values()
        ),
        "band_radius_upper_bound": str(max(bounds.values())),
        "band_gap_lower_bound": str(1 - max(bounds.values())),
    }


def _bottleneck_witness_certificate(
    view: str, original_hodge_minus: FieldMatrix
) -> dict[str, object]:
    coefficients = RAYLEIGH_WITNESSES[view]
    vector = _vector_from_sparse(coefficients)
    quotient = _rayleigh_quotient(original_hodge_minus, vector)
    threshold = sp.Rational(19, 20) if view == "vector" else sp.Rational(39, 40)
    exceeds_threshold = low._field_sign(quotient - low._field(threshold)) > 0
    exact_gap = golden.FIELD.one - quotient
    return {
        "representation": "hodge_minus_35",
        "sparse_reduced_basis_coefficients": _sparse_vector_report(coefficients),
        "reduced_basis_norm_squared": sum(value * value for value in coefficients.values()),
        "exact_rayleigh_quotient": low._field_string(quotient),
        "exact_original_band_gap_upper_bound_from_witness": low._field_string(
            exact_gap
        ),
        "comparison_threshold": str(threshold),
        "rayleigh_quotient_exceeds_threshold": exceeds_threshold,
        "coarse_original_band_gap_upper_bound": str(1 - threshold),
        "passed": exceeds_threshold,
    }


def certificate() -> dict[str, object]:
    """Return the exact higher-weight and sandwich-improvement certificate."""

    operator_report, _, operator_generators = monomial._operator_group_certificate()
    automorphism_report, _, automorphism_generators = (
        monomial._automorphism_group_certificate()
    )
    normalizer_generators = (*operator_generators, *automorphism_generators)
    monomial_steps = closure._symmetric_monomial_steps(normalizer_generators)
    monomial_matrices = tuple(
        closure._field_matrix_from_monomial(step) for step in monomial_steps
    )
    monomial_representations_with_checks = tuple(
        _representations(matrix) for matrix in monomial_matrices
    )
    monomial_representations = tuple(
        report for report, _ in monomial_representations_with_checks
    )
    monomial_hodge_checks = tuple(
        check for _, check in monomial_representations_with_checks
    )
    monomial_means = {
        name: low._mean_matrix(
            tuple(report[name] for report in monomial_representations)
        )
        for name in REPRESENTATION_NAMES
    }
    cayley = _fixed_cayley_certificate(monomial_representations)
    gram = low._traceless_gram()
    triality_matrices = golden._spin8_action_matrices()

    view_reports: dict[str, object] = {}
    for view, indices in closure.VIEW_GENERATOR_INDICES.items():
        pair = tuple(triality_matrices[index] for index in indices)
        golden_steps = closure._symmetric_golden_steps(pair)
        golden_representations_with_checks = tuple(
            _representations(matrix) for matrix in golden_steps
        )
        golden_representations = tuple(
            report for report, _ in golden_representations_with_checks
        )
        golden_hodge_checks = tuple(
            check for _, check in golden_representations_with_checks
        )
        golden_means = {
            name: low._mean_matrix(
                tuple(report[name] for report in golden_representations)
            )
            for name in REPRESENTATION_NAMES
        }
        original_means = {
            name: _label_union_mean(
                monomial_means[name],
                len(monomial_steps),
                golden_means[name],
                len(golden_steps),
            )
            for name in REPRESENTATION_NAMES
        }
        sandwich_means = {
            name: _sandwich_mean(monomial_means[name], golden_means[name])
            for name in REPRESENTATION_NAMES
        }
        original = _measure_certificate(
            original_means, ORIGINAL_RADIUS_BOUNDS[view], gram
        )
        sandwich = _measure_certificate(
            sandwich_means, SANDWICH_RADIUS_BOUNDS[view], gram
        )
        witness = _bottleneck_witness_certificate(
            view, original_means["hodge_minus_35"]
        )
        quotient = _rayleigh_quotient(
            original_means["hodge_minus_35"],
            _vector_from_sparse(RAYLEIGH_WITNESSES[view]),
        )
        original_gap_upper = golden.FIELD.one - quotient
        sandwich_gap_lower = low._field(
            1 - max(SANDWICH_RADIUS_BOUNDS[view].values())
        )
        ratio = sandwich_gap_lower / original_gap_upper
        conservative_factor = (
            sp.Rational(3, 1)
            if view == "vector"
            else sp.Rational(56, 25)
        )
        factor_passed = low._field_sign(
            ratio - low._field(conservative_factor)
        ) >= 0
        exact_factor_lower_source = low._field_string(ratio)

        checks = {
            "every_hodge_split_is_exactly_invariant": all(
                (*monomial_hodge_checks, *golden_hodge_checks)
            ),
            "original_six_representation_band_passed": bool(
                original["all_six_representation_bounds_passed"]
            ),
            "sandwich_six_representation_band_passed": bool(
                sandwich["all_six_representation_bounds_passed"]
            ),
            "hodge_minus_bottleneck_witness_passed": bool(witness["passed"]),
            "compiled_macro_band_improvement_factor_passed": factor_passed,
        }
        view_reports[view] = {
            "golden_symmetric_label_count": len(golden_steps),
            "original_uniform_label_measure": original,
            "monomial_golden_monomial_sandwich_measure": {
                **sandwich,
                "operator_formula": "M_N M_H M_N",
                "primitive_word_length": 3,
                "distribution_is_symmetric": True,
                "precompiled_macro_support_upper_bound": (
                    len(monomial_steps) * len(golden_steps) * len(monomial_steps)
                ),
            },
            "bottleneck_rayleigh_witness": witness,
            "certified_macro_step_band_gap_improvement_factor": (
                f"> {conservative_factor}"
            ),
            "improvement_factor_exact_comparison_source": exact_factor_lower_source,
            "checks": checks,
            "passed": all(checks.values()),
        }

    checks = {
        "operator_group_input_passed": bool(operator_report["passed"]),
        "signed_fano_input_passed": bool(automorphism_report["passed"]),
        "monomial_symmetric_step_count_is_seventeen": len(monomial_steps) == 17,
        "unique_monomial_fixed_cayley_line_certified": bool(cayley["passed"]),
        "all_three_view_certificates_passed": all(
            report["passed"] for report in view_reports.values()
        ),
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "experiment": (
            "mixed monomial/golden higher-weight bottleneck and compiled "
            "sandwich improvement"
        ),
        "field": "Q(sqrt(5))",
        "orientation": "standard ordered basis e0,...,e7",
        "representations": {
            "band": list(REPRESENTATION_NAMES),
            "dimensions": [8, 28, 35, 56, 35, 35],
            "hodge_basis": (
                "for I<I_complement, e_I plus/minus orientation_sign(I,Ic)*e_Ic"
            ),
        },
        "monomial_cayley_fixed_line": cayley,
        "views": view_reports,
        "checks": checks,
        "claim_scope": {
            "proved": [
                "the exact displayed radius bounds on the six-representation band for the original and sandwich measures",
                "the maintained monomial group has a unique fixed line in the orientation-labelled Hodge-minus four-form sector",
                "the fixed sparse Rayleigh witnesses make Hodge-minus the certified obstruction to the original band gap",
                "per compiled macro-step the N-H-N distribution improves the certified six-representation band gap by more than 3x in the vector view and more than 56/25x in both half-spin views",
            ],
            "not_claimed": [
                "a spectral gap on the full mean-zero L2(SO(8))",
                "optimality among word distributions or generator weights",
                "an improvement per primitive matrix multiplication; one sandwich macro consumes three primitive letters unless precompiled",
                "a sequence-model accuracy, training, memory, or kernel-speed advantage",
            ],
        },
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
