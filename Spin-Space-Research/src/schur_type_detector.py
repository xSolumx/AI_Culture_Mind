"""Exact Schur-type detection for supplied real representation generators.

Given exact real matrices generating a representation, this module solves the
simultaneous commutant and tests whether it is a real division algebra.  Under
the explicit assumption that the representation is completely reducible, a
division commutant certifies irreducibility and identifies the real Schur type
as ``R``, ``C``, or ``H``.

The dimension test is deliberately insufficient on its own.  The detector
also verifies an exact negative-square complex structure or a positive-
definite quaternionic imaginary norm and multiplication table.  Thus a split
two-dimensional algebra and a repeated irreducible block are rejected rather
than mislabelled.  General decomposition of a reducible representation into
isotypic summands remains outside this module.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import sympy as sp
from exact_real_scalar_field import field_from_extension

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "schur_type_detection_20260811.json"


def _flatten(matrix: sp.Matrix) -> sp.Matrix:
    return sp.Matrix(list(matrix))


def _matrix_equal(left: sp.Matrix, right: sp.Matrix) -> bool:
    difference = (sp.Matrix(left) - sp.Matrix(right)).applyfunc(sp.simplify)
    return difference == sp.zeros(*difference.shape)


def _primitive_matrix(
    matrix: sp.Matrix, scalar_extension: sp.Expr | None = None
) -> sp.Matrix:
    """Choose a deterministic projective representative over the field."""

    return field_from_extension(scalar_extension).projective_matrix_normal_form(matrix)


def _scalar_multiple(matrix: sp.Matrix) -> sp.Expr | None:
    if matrix.rows != matrix.cols:
        return None
    scalar = sp.factor(sp.trace(matrix) / matrix.rows)
    if _matrix_equal(matrix, scalar * sp.eye(matrix.rows)):
        return scalar
    return None


def _select_independent(
    matrices: Iterable[sp.Matrix], scalar_extension: sp.Expr | None = None
) -> list[sp.Matrix]:
    selected: list[sp.Matrix] = []
    rank = 0
    for matrix in matrices:
        candidate = _primitive_matrix(matrix, scalar_extension)
        if not any(candidate):
            continue
        columns = [_flatten(value) for value in (*selected, candidate)]
        new_rank = sp.Matrix.hstack(*columns).rank()
        if new_rank > rank:
            selected.append(candidate)
            rank = new_rank
    return selected


def exact_commutant_basis(
    generators: Iterable[sp.Matrix], *, scalar_extension: sp.Expr | None = None
) -> list[sp.Matrix]:
    """Return a deterministic exact basis of simultaneous commuting matrices."""

    matrices = [sp.Matrix(generator) for generator in generators]
    if not matrices:
        raise ValueError("at least one representation generator is required")
    dimension = matrices[0].rows
    if dimension == 0 or matrices[0].cols != dimension:
        raise ValueError("representation generators must be nonempty square matrices")
    if any(matrix.shape != (dimension, dimension) for matrix in matrices):
        raise ValueError("all representation generators must have the same shape")
    field = field_from_extension(scalar_extension)
    if not field.contains_all(entry for matrix in matrices for entry in matrix):
        if scalar_extension is None:
            raise ValueError("exact rational representation generators are required")
        raise ValueError(
            f"exact representation generators over the declared field {field.name} "
            "are required"
        )

    identity = sp.eye(dimension)
    constraints = sp.Matrix.vstack(
        *(
            sp.kronecker_product(identity, generator.T)
            - sp.kronecker_product(generator, identity)
            for generator in matrices
        )
    )
    return [
        _primitive_matrix(
            sp.Matrix(dimension, dimension, list(vector)), scalar_extension
        )
        for vector in field.nullspace(constraints)
    ]


def _traceless_basis(
    commutant: Iterable[sp.Matrix],
    dimension: int,
    scalar_extension: sp.Expr | None = None,
) -> list[sp.Matrix]:
    identity = sp.eye(dimension)
    candidates = [
        matrix - sp.trace(matrix) * identity / dimension for matrix in commutant
    ]
    return _select_independent(candidates, scalar_extension)


def _imaginary_inner(left: sp.Matrix, right: sp.Matrix) -> sp.Expr | None:
    anticommutator = (left * right + right * left) / 2
    scalar = _scalar_multiple(anticommutator)
    return None if scalar is None else sp.factor(-scalar)


def _coordinates(matrix: sp.Matrix, basis: list[sp.Matrix]) -> list[sp.Expr]:
    design = sp.Matrix.hstack(*(_flatten(value) for value in basis))
    solution, parameters = design.gauss_jordan_solve(_flatten(matrix))
    if parameters.rows:
        raise ValueError("division basis coordinates were not unique")
    return [sp.factor(value) for value in solution]


def _multiplication_table(basis: list[sp.Matrix]) -> list[list[list[sp.Expr]]]:
    return [[_coordinates(left * right, basis) for right in basis] for left in basis]


def _positive_definite_exact(
    matrix: sp.Matrix, scalar_extension: sp.Expr | None = None
) -> bool:
    if matrix != matrix.T:
        return False
    field = field_from_extension(scalar_extension)
    return all(
        field.sign(sp.factor(matrix[:size, :size].det())) > 0
        for size in range(1, matrix.rows + 1)
    )


@dataclass(frozen=True)
class SchurTypeCertificate:
    """Exact classification and extracted division-algebra basis."""

    real_dimension: int
    generator_count: int
    commutant_dimension: int
    schur_type: str | None
    completely_reducible_assumed: bool
    division_basis: tuple[sp.Matrix, ...]
    basis_labels: tuple[str, ...]
    multiplication_table: tuple[tuple[tuple[sp.Expr, ...], ...], ...]
    exact_gates: dict[str, bool]
    structure_constants: dict[str, sp.Expr]
    rejection_reason: str | None

    @property
    def classified_irreducible(self) -> bool:
        return bool(
            self.completely_reducible_assumed
            and self.schur_type is not None
            and all(self.exact_gates.values())
        )


def detect_schur_type(
    generators: Iterable[sp.Matrix],
    *,
    assume_completely_reducible: bool,
    scalar_extension: sp.Expr | None = None,
) -> SchurTypeCertificate:
    """Detect ``R``, ``C``, or ``H`` type and extract an exact basis.

    Complete reducibility is an explicit logical input.  For representations
    of finite or compact groups it is standard after choosing an invariant
    inner product, but the matrix equations alone do not establish it for an
    arbitrary nonsemisimple algebra representation.
    """

    matrices = [sp.Matrix(generator) for generator in generators]
    field = field_from_extension(scalar_extension)
    commutant = exact_commutant_basis(matrices, scalar_extension=scalar_extension)
    dimension = matrices[0].rows
    commutant_dimension = len(commutant)
    identity = sp.eye(dimension)
    common_gates = {
        "basis_commutes_with_all_generators": all(
            _matrix_equal(basis * generator, generator * basis)
            for basis in commutant
            for generator in matrices
        ),
        "identity_in_commutant": all(
            _matrix_equal(identity * generator, generator * identity)
            for generator in matrices
        ),
    }

    if not assume_completely_reducible:
        return SchurTypeCertificate(
            real_dimension=dimension,
            generator_count=len(matrices),
            commutant_dimension=commutant_dimension,
            schur_type=None,
            completely_reducible_assumed=False,
            division_basis=(),
            basis_labels=(),
            multiplication_table=(),
            exact_gates=common_gates,
            structure_constants={},
            rejection_reason=(
                "complete reducibility was not supplied, so a division "
                "commutant cannot certify irreducibility"
            ),
        )

    if commutant_dimension == 1:
        basis = [identity]
        table = _multiplication_table(basis)
        gates = {
            **common_gates,
            "commutant_is_scalar": _matrix_equal(
                commutant[0], commutant[0][0, 0] * identity
            ),
            "division_basis_spans_commutant": True,
            "multiplication_table_closes_exactly": table == [[[sp.Integer(1)]]],
        }
        return SchurTypeCertificate(
            real_dimension=dimension,
            generator_count=len(matrices),
            commutant_dimension=1,
            schur_type="real" if all(gates.values()) else None,
            completely_reducible_assumed=True,
            division_basis=tuple(basis),
            basis_labels=("1",),
            multiplication_table=tuple(
                tuple(tuple(row) for row in plane) for plane in table
            ),
            exact_gates=gates,
            structure_constants={},
            rejection_reason=(
                None if all(gates.values()) else "scalar commutant gate failed"
            ),
        )

    imaginary = _traceless_basis(commutant, dimension, scalar_extension)
    if commutant_dimension == 2 and len(imaginary) == 1:
        complex_structure = imaginary[0]
        square_scalar = _scalar_multiple(complex_structure**2)
        negative_square = bool(
            square_scalar is not None and field.sign(square_scalar) < 0
        )
        basis = [identity, complex_structure]
        basis_rank = sp.Matrix.hstack(*(_flatten(value) for value in basis)).rank()
        table = _multiplication_table(basis)
        gates = {
            **common_gates,
            "traceless_complement_dimension_one": True,
            "complex_structure_square_is_negative_scalar": negative_square,
            "division_basis_spans_commutant": basis_rank == 2,
            "multiplication_table_closes_exactly": True,
        }
        classified = all(gates.values())
        return SchurTypeCertificate(
            real_dimension=dimension,
            generator_count=len(matrices),
            commutant_dimension=2,
            schur_type="complex" if classified else None,
            completely_reducible_assumed=True,
            division_basis=tuple(basis) if classified else (),
            basis_labels=("1", "j") if classified else (),
            multiplication_table=(
                tuple(tuple(tuple(row) for row in plane) for plane in table)
                if classified
                else ()
            ),
            exact_gates=gates,
            structure_constants=(
                {"j_square": sp.factor(square_scalar)}
                if square_scalar is not None
                else {}
            ),
            rejection_reason=(
                None
                if classified
                else "two-dimensional commutant is split or nondivision, not C"
            ),
        )

    if commutant_dimension == 4 and len(imaginary) == 3:
        gram_entries = [
            [_imaginary_inner(left, right) for right in imaginary] for left in imaginary
        ]
        anticommutators_scalar = all(
            value is not None for row in gram_entries for value in row
        )
        gram = sp.Matrix(gram_entries) if anticommutators_scalar else sp.zeros(3, 3)
        norm_positive = anticommutators_scalar and _positive_definite_exact(
            gram, scalar_extension
        )

        basis: list[sp.Matrix] = []
        table: list[list[list[sp.Expr]]] = []
        structure: dict[str, sp.Expr] = {}
        quaternion_relations = False
        if norm_positive:
            first = imaginary[0]
            first_norm = _imaginary_inner(first, first)
            second = None
            for candidate in imaginary[1:]:
                cross = _imaginary_inner(first, candidate)
                assert first_norm is not None and cross is not None
                orthogonal = _primitive_matrix(
                    first_norm * candidate - cross * first, scalar_extension
                )
                if any(orthogonal):
                    second = orthogonal
                    break
            if second is not None:
                second_norm = _imaginary_inner(second, second)
                cross = _imaginary_inner(first, second)
                third = first * second
                basis = [identity, first, second, third]
                basis_rank = sp.Matrix.hstack(
                    *(_flatten(value) for value in basis)
                ).rank()
                table = _multiplication_table(basis)
                quaternion_relations = bool(
                    first_norm is not None
                    and second_norm is not None
                    and field.sign(first_norm) > 0
                    and field.sign(second_norm) > 0
                    and cross == 0
                    and _matrix_equal(first**2, -first_norm * identity)
                    and _matrix_equal(second**2, -second_norm * identity)
                    and _matrix_equal(first * second, third)
                    and _matrix_equal(second * first, -third)
                    and _matrix_equal(third**2, -first_norm * second_norm * identity)
                    and basis_rank == 4
                )
                structure = {
                    "i_square": -first_norm,
                    "j_square": -second_norm,
                    "k_square": -first_norm * second_norm,
                }

        gates = {
            **common_gates,
            "traceless_complement_dimension_three": True,
            "imaginary_anticommutators_are_scalar": anticommutators_scalar,
            "imaginary_norm_is_positive_definite": norm_positive,
            "quaternion_relations_hold_exactly": quaternion_relations,
            "division_basis_spans_commutant": len(basis) == 4,
            "multiplication_table_closes_exactly": bool(table),
        }
        classified = all(gates.values())
        return SchurTypeCertificate(
            real_dimension=dimension,
            generator_count=len(matrices),
            commutant_dimension=4,
            schur_type="quaternion" if classified else None,
            completely_reducible_assumed=True,
            division_basis=tuple(basis) if classified else (),
            basis_labels=("1", "i", "j", "k") if classified else (),
            multiplication_table=(
                tuple(tuple(tuple(row) for row in plane) for plane in table)
                if classified
                else ()
            ),
            exact_gates=gates,
            structure_constants=structure,
            rejection_reason=(
                None
                if classified
                else "four-dimensional commutant failed quaternion division gates"
            ),
        )

    return SchurTypeCertificate(
        real_dimension=dimension,
        generator_count=len(matrices),
        commutant_dimension=commutant_dimension,
        schur_type=None,
        completely_reducible_assumed=True,
        division_basis=(),
        basis_labels=(),
        multiplication_table=(),
        exact_gates=common_gates,
        structure_constants={},
        rejection_reason=(
            f"commutant dimension {commutant_dimension} is not an irreducible "
            "real division-algebra dimension"
        ),
    )


def _left_quaternion_matrix(coordinates: tuple[int, int, int, int]) -> sp.Matrix:
    w, x, y, z = coordinates
    return sp.Matrix(
        [
            [w, -x, -y, -z],
            [x, w, -z, y],
            [y, z, w, -x],
            [z, -y, x, w],
        ]
    )


def canonical_examples() -> dict[str, list[sp.Matrix]]:
    """Return exact generator sets for canonical positive controls."""

    so3 = [
        sp.Matrix([[0, 0, 0], [0, 0, -1], [0, 1, 0]]),
        sp.Matrix([[0, 0, 1], [0, 0, 0], [-1, 0, 0]]),
        sp.Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 0]]),
    ]
    complex_generator = sp.Matrix([[0, -1], [1, 0]])
    quaternion = [
        _left_quaternion_matrix((0, 1, 0, 0)),
        _left_quaternion_matrix((0, 0, 1, 0)),
    ]
    return {
        "real_so3_vector": so3,
        "complex_u1_realification": [complex_generator],
        "quaternion_su2_spinor_realification": quaternion,
    }


def rejection_examples() -> dict[str, list[sp.Matrix]]:
    complex_generator = sp.Matrix([[0, -1], [1, 0]])
    return {
        "split_two_line_representation": [sp.diag(1, -1)],
        "doubled_complex_irrep": [sp.diag(complex_generator, complex_generator)],
    }


def rational_conjugacy_examples() -> dict[str, list[sp.Matrix]]:
    """Conjugate every positive control by a non-orthogonal rational basis."""

    changes = {
        "real_so3_vector": sp.Matrix([[1, 1, 0], [0, 1, 1], [0, 0, 1]]),
        "complex_u1_realification": sp.Matrix([[1, 1], [0, 1]]),
        "quaternion_su2_spinor_realification": sp.Matrix(
            [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
        ),
    }
    return {
        name: [
            changes[name] * generator * changes[name].inv() for generator in generators
        ]
        for name, generators in canonical_examples().items()
    }


def _matrix_json(matrix: sp.Matrix) -> list[list[str]]:
    return [
        [str(sp.factor(matrix[row, column])) for column in range(matrix.cols)]
        for row in range(matrix.rows)
    ]


def certificate_json(certificate: SchurTypeCertificate) -> dict[str, object]:
    return {
        "real_dimension": certificate.real_dimension,
        "generator_count": certificate.generator_count,
        "commutant_dimension": certificate.commutant_dimension,
        "schur_type": certificate.schur_type,
        "completely_reducible_assumed": certificate.completely_reducible_assumed,
        "classified_irreducible": certificate.classified_irreducible,
        "basis_labels": list(certificate.basis_labels),
        "division_basis_matrices": [
            _matrix_json(matrix) for matrix in certificate.division_basis
        ],
        "multiplication_table_coefficients": [
            [[str(sp.factor(value)) for value in coefficients] for coefficients in row]
            for row in certificate.multiplication_table
        ],
        "structure_constants": {
            key: str(sp.factor(value))
            for key, value in certificate.structure_constants.items()
        },
        "exact_gates": certificate.exact_gates,
        "rejection_reason": certificate.rejection_reason,
    }


def diagnostics() -> dict[str, object]:
    expected = {
        "real_so3_vector": "real",
        "complex_u1_realification": "complex",
        "quaternion_su2_spinor_realification": "quaternion",
    }
    positive = {
        name: detect_schur_type(
            generators,
            assume_completely_reducible=True,
        )
        for name, generators in canonical_examples().items()
    }
    rejections = {
        name: detect_schur_type(
            generators,
            assume_completely_reducible=True,
        )
        for name, generators in rejection_examples().items()
    }
    conjugated = {
        name: detect_schur_type(
            generators,
            assume_completely_reducible=True,
        )
        for name, generators in rational_conjugacy_examples().items()
    }
    missing_assumption = detect_schur_type(
        canonical_examples()["complex_u1_realification"],
        assume_completely_reducible=False,
    )
    passed = bool(
        all(
            certificate.classified_irreducible
            and certificate.schur_type == expected[name]
            for name, certificate in positive.items()
        )
        and all(
            not certificate.classified_irreducible and certificate.schur_type is None
            for certificate in rejections.values()
        )
        and all(
            certificate.classified_irreducible
            and certificate.schur_type == expected[name]
            for name, certificate in conjugated.items()
        )
        and not missing_assumption.classified_irreducible
        and missing_assumption.schur_type is None
    )
    return {
        "schema_version": 1,
        "claim_scope": (
            "exact Schur-type detection and division-basis extraction for "
            "supplied rational generators under complete reducibility"
        ),
        "positive_controls": {
            name: certificate_json(certificate)
            for name, certificate in positive.items()
        },
        "rejection_controls": {
            name: certificate_json(certificate)
            for name, certificate in rejections.items()
        },
        "rational_conjugacy_controls": {
            name: certificate_json(certificate)
            for name, certificate in conjugated.items()
        },
        "missing_complete_reducibility_assumption": certificate_json(
            missing_assumption
        ),
        "reducible_isotypic_decomposition_implemented_in_this_module": False,
        "companion_reducible_isotypic_decomposition_available": True,
        "floating_point_noisy_detection_claimed": False,
        "sequence_model_superiority_claimed": False,
        "passed": passed,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = diagnostics()
    encoded = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "SchurTypeCertificate",
    "canonical_examples",
    "certificate_json",
    "detect_schur_type",
    "diagnostics",
    "exact_commutant_basis",
    "rational_conjugacy_examples",
    "rejection_examples",
]
