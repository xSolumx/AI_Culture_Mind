"""Exact Q(sqrt(2))-to-rational bridge for the Spin(9) quotient slice.

The Grassmann normal-slice certificate constructs its six-dimensional
stabilizer action in a geometrically natural basis over ``Q(sqrt(2))``.  The
reducible isotypic compiler deliberately accepts rational generators only.
This module closes that concrete interface gap without weakening either
contract:

1. normalize the stabilizer basis to the standard so(3) brackets;
2. solve the full algebraic intertwiner space to ``V1 + V5`` exactly;
3. identify the supported ``Sym_0(3)`` coefficient basis with the same
   rational spin-two module; and
4. conjugate the concrete ``V1 + V5 + V5`` action to rational coordinates
   before invoking the certified reducible compiler.

This is an exact certificate at the Cayley-null slice.  It is not a global
slice chart, a finite-radius determinant inequality, or a new local Hessian
theorem.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import sympy as sp

from reducible_isotypic_decomposition import (
    ReducibleIsotypicCertificate,
    decompose_reducible_representation,
    spin9_slice_fixture,
    spin_two_generators,
)
from reducible_isotypic_decomposition import (
    certificate_json as reducible_certificate_json,
)
from schur_type_detector import canonical_examples
from spin9_grassmann_slice import GrassmannSliceData, construct_slice_data

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "spin9_slice_isotypic_bridge_20260811.json"


def _flatten(matrix: sp.Matrix) -> sp.Matrix:
    return sp.Matrix(list(matrix))


def _matrix_json(matrix: sp.Matrix) -> list[list[str]]:
    return [
        [sp.sstr(sp.factor(matrix[row, column])) for column in range(matrix.cols)]
        for row in range(matrix.rows)
    ]


def _in_q_sqrt_two(values: Iterable[sp.Expr]) -> bool:
    field = sp.QQ.algebraic_field(sp.sqrt(2))
    try:
        for value in values:
            field.from_sympy(sp.simplify(value))
    except (sp.polys.polyerrors.CoercionFailed, ValueError):
        return False
    return True


def exact_algebraic_intertwiner_basis(
    source_generators: Iterable[sp.Matrix],
    target_generators: Iterable[sp.Matrix],
) -> tuple[sp.Matrix, ...]:
    """Solve ``T A_i = B_i T`` exactly over the entries' algebraic field."""

    source = tuple(sp.Matrix(value) for value in source_generators)
    target = tuple(sp.Matrix(value) for value in target_generators)
    if not source or len(source) != len(target):
        raise ValueError("source and target need the same positive generator count")
    source_dimension = source[0].rows
    target_dimension = target[0].rows
    if any(value.shape != (source_dimension, source_dimension) for value in source):
        raise ValueError("source generators must be square with one dimension")
    if any(value.shape != (target_dimension, target_dimension) for value in target):
        raise ValueError("target generators must be square with one dimension")
    constraints = sp.Matrix.vstack(
        *(
            sp.kronecker_product(sp.eye(target_dimension), left.T)
            - sp.kronecker_product(right, sp.eye(source_dimension))
            for left, right in zip(source, target, strict=True)
        )
    )
    return tuple(
        sp.Matrix(target_dimension, source_dimension, list(vector))
        for vector in constraints.nullspace()
    )


def supported_sym0_basis() -> tuple[sp.Matrix, ...]:
    """Return the exact coefficient basis used by the local-Hessian verifier."""

    return (
        sp.diag(1, -1, 0),
        sp.diag(1, 1, -2),
        sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
        sp.Matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
        sp.Matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
    )


def _sym0_generators(basis: tuple[sp.Matrix, ...]) -> tuple[sp.Matrix, ...]:
    design = sp.Matrix.hstack(*(_flatten(value) for value in basis))
    output = []
    for generator in canonical_examples()["real_so3_vector"]:
        columns = []
        for matrix in basis:
            image = generator * matrix - matrix * generator
            solution, parameters = design.gauss_jordan_solve(_flatten(image))
            if parameters.rows:
                raise ValueError("Sym_0(3) action coordinates were not unique")
            columns.append(solution)
        output.append(sp.Matrix.hstack(*columns))
    return tuple(output)


def _supported_to_canonical_change() -> sp.Matrix:
    canonical_basis = (
        sp.diag(1, -1, 0),
        sp.diag(0, 1, -1),
        sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
        sp.Matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
        sp.Matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
    )
    canonical_design = sp.Matrix.hstack(*(_flatten(value) for value in canonical_basis))
    supported_design = sp.Matrix.hstack(
        *(_flatten(value) for value in supported_sym0_basis())
    )
    solution, parameters = canonical_design.gauss_jordan_solve(supported_design)
    if parameters.rows:
        raise ValueError("supported coefficient-basis change was not unique")
    return solution


@dataclass(frozen=True)
class Spin9SliceBridgeCertificate:
    slice_data: GrassmannSliceData
    standardized_slice_generators: tuple[sp.Matrix, ...]
    canonical_slice_generators: tuple[sp.Matrix, ...]
    trivial_projector: sp.Matrix
    spin_two_projector: sp.Matrix
    intertwiner_basis: tuple[sp.Matrix, ...]
    slice_change: sp.Matrix
    supported_generators: tuple[sp.Matrix, ...]
    supported_change: sp.Matrix
    concrete_generators: tuple[sp.Matrix, ...]
    extended_change: sp.Matrix
    rational_generators: tuple[sp.Matrix, ...]
    transformed_curve_coordinates: sp.Matrix
    reducible_certificate: ReducibleIsotypicCertificate
    exact_gates: dict[str, bool]

    @property
    def certified(self) -> bool:
        return bool(
            self.reducible_certificate.certified and all(self.exact_gates.values())
        )


def build_certificate() -> Spin9SliceBridgeCertificate:
    """Construct and verify the concrete algebraic-to-rational bridge."""

    data = construct_slice_data()
    raw = data.slice_actions
    sqrt_two = sp.sqrt(2)
    source = (
        (raw[0] / sqrt_two).applyfunc(sp.simplify),
        raw[1],
        (raw[2] / sqrt_two).applyfunc(sp.simplify),
    )
    spin_two = tuple(spin_two_generators())
    target = tuple(sp.diag(sp.zeros(1), generator) for generator in spin_two)

    intertwiners = exact_algebraic_intertwiner_basis(source, target)
    rank_one = tuple(value for value in intertwiners if value.rank() == 1)
    rank_five = tuple(value for value in intertwiners if value.rank() == 5)
    if len(rank_one) != 1 or len(rank_five) != 1:
        raise ValueError("expected one rank-one and one rank-five intertwiner")
    slice_change = (rank_one[0] + rank_five[0]).applyfunc(sp.factor)

    supported_generators = _sym0_generators(supported_sym0_basis())
    supported_change = _supported_to_canonical_change()
    concrete = tuple(
        sp.diag(source_generator, supported_generator)
        for source_generator, supported_generator in zip(
            source, supported_generators, strict=True
        )
    )
    extended_change = sp.diag(slice_change, supported_change)
    rational = tuple(spin9_slice_fixture())
    transformed = tuple(
        (extended_change * generator * extended_change.inv()).applyfunc(sp.simplify)
        for generator in concrete
    )
    transformed_curve = (slice_change * data.curve_coordinates).applyfunc(sp.simplify)

    reducible = decompose_reducible_representation(
        rational,
        assume_completely_reducible=True,
    )
    block_signatures = sorted(
        (
            block.schur_type,
            block.multiplicity,
            block.irreducible_dimension,
            block.commutant_dimension,
        )
        for block in reducible.blocks
    )

    casimir = data.casimir
    trivial_projector = (sp.eye(6) - casimir / 6).applyfunc(sp.simplify)
    spin_two_projector = (casimir / 6).applyfunc(sp.simplify)
    supported_norms = tuple(
        int(sp.trace(matrix.T * matrix)) for matrix in supported_sym0_basis()
    )
    raw_brackets = (
        raw[0] * raw[1] - raw[1] * raw[0] == raw[2],
        raw[0] * raw[2] - raw[2] * raw[0] == -2 * raw[1],
        raw[1] * raw[2] - raw[2] * raw[1] == raw[0],
    )
    standard_brackets = (
        source[0] * source[1] - source[1] * source[0] == source[2],
        source[0] * source[2] - source[2] * source[0] == -source[1],
        source[1] * source[2] - source[2] * source[1] == source[0],
    )
    intertwiner_identities = tuple(
        all(
            value * left == right * value
            for left, right in zip(source, target, strict=True)
        )
        for value in intertwiners
    )
    supported_identities = tuple(
        supported_change * left == right * supported_change
        for left, right in zip(supported_generators, spin_two, strict=True)
    )
    metric_skew = tuple(
        sp.simplify(action.T * data.normal_metric + data.normal_metric * action)
        == sp.zeros(6)
        for action in source
    )
    gates = {
        "raw_stabilizer_brackets": all(raw_brackets),
        "standard_so3_brackets": all(standard_brackets),
        "slice_actions_are_metric_skew": all(metric_skew),
        "slice_actions_lie_in_q_sqrt_two": _in_q_sqrt_two(
            entry for matrix in source for entry in matrix
        ),
        "casimir_equals_standard_sum_of_squares": casimir
        == -sum((generator**2 for generator in source), sp.zeros(6)),
        "casimir_projectors_are_complementary": bool(
            trivial_projector**2 == trivial_projector
            and spin_two_projector**2 == spin_two_projector
            and trivial_projector * spin_two_projector == sp.zeros(6)
            and trivial_projector + spin_two_projector == sp.eye(6)
        ),
        "casimir_projector_ranks_are_one_and_five": (
            trivial_projector.rank(),
            spin_two_projector.rank(),
        )
        == (1, 5),
        "intertwiner_space_dimension_is_two": len(intertwiners) == 2,
        "intertwiner_ranks_are_one_and_five": sorted(
            value.rank() for value in intertwiners
        )
        == [1, 5],
        "intertwiner_identities": all(intertwiner_identities),
        "slice_change_lies_in_q_sqrt_two": _in_q_sqrt_two(slice_change),
        "slice_change_is_invertible": slice_change.det() != 0,
        "slice_change_conjugates_to_v1_plus_v5": all(
            slice_change * left * slice_change.inv() == right
            for left, right in zip(source, target, strict=True)
        ),
        "curve_maps_to_nonzero_trivial_coordinate": bool(
            transformed_curve[0] != 0 and transformed_curve[1:, :] == sp.zeros(5, 1)
        ),
        "supported_basis_norms_match_local_hessian": supported_norms == (2, 6, 2, 2, 2),
        "supported_basis_change_is_rational_and_invertible": bool(
            all(entry.is_Rational for entry in supported_change)
            and supported_change.det() != 0
        ),
        "supported_basis_intertwines_spin_two": all(supported_identities),
        "extended_change_conjugates_to_rational_fixture": transformed == rational,
        "rational_fixture_entries_are_rational": all(
            entry.is_Rational for matrix in rational for entry in matrix
        ),
        "reducible_compiler_certified": reducible.certified,
        "compiler_recovers_v1_plus_two_v5": block_signatures
        == [("real", 1, 1, 1), ("real", 2, 5, 4)],
    }

    return Spin9SliceBridgeCertificate(
        slice_data=data,
        standardized_slice_generators=source,
        canonical_slice_generators=target,
        trivial_projector=trivial_projector,
        spin_two_projector=spin_two_projector,
        intertwiner_basis=intertwiners,
        slice_change=slice_change,
        supported_generators=supported_generators,
        supported_change=supported_change,
        concrete_generators=concrete,
        extended_change=extended_change,
        rational_generators=rational,
        transformed_curve_coordinates=transformed_curve,
        reducible_certificate=reducible,
        exact_gates=gates,
    )


def certificate_json(certificate: Spin9SliceBridgeCertificate) -> dict[str, object]:
    data = certificate.slice_data
    return {
        "schema_version": 1,
        "claim_scope": (
            "exact Q(sqrt(2))-to-rational isotypic bridge at the Cayley-null "
            "Spin(9) slice; no global quotient or determinant-optimality claim"
        ),
        "scalar_field": "Q(sqrt(2))",
        "normal_metric": _matrix_json(data.normal_metric),
        "raw_slice_generators": [_matrix_json(value) for value in data.slice_actions],
        "standardized_slice_generators": [
            _matrix_json(value) for value in certificate.standardized_slice_generators
        ],
        "casimir": _matrix_json(data.casimir),
        "trivial_projector": _matrix_json(certificate.trivial_projector),
        "spin_two_projector": _matrix_json(certificate.spin_two_projector),
        "intertwiner_space_dimension": len(certificate.intertwiner_basis),
        "intertwiner_ranks": [value.rank() for value in certificate.intertwiner_basis],
        "intertwiner_basis": [
            _matrix_json(value) for value in certificate.intertwiner_basis
        ],
        "slice_change": _matrix_json(certificate.slice_change),
        "slice_change_determinant": sp.sstr(sp.factor(certificate.slice_change.det())),
        "transformed_curve_coordinates": _matrix_json(
            certificate.transformed_curve_coordinates
        ),
        "supported_sym0_basis": [
            _matrix_json(value) for value in supported_sym0_basis()
        ],
        "supported_change": _matrix_json(certificate.supported_change),
        "supported_change_determinant": sp.sstr(
            sp.factor(certificate.supported_change.det())
        ),
        "extended_change_determinant": sp.sstr(
            sp.factor(certificate.extended_change.det())
        ),
        "rationalized_branching": "V1 + 2*V5",
        "reducible_compiler": reducible_certificate_json(
            certificate.reducible_certificate
        ),
        "exact_gates": certificate.exact_gates,
        "global_grassmann_quotient_solved": False,
        "finite_radius_coupled_determinant_solved": False,
        "passed": certificate.certified,
    }


def diagnostics() -> dict[str, object]:
    return certificate_json(build_certificate())


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
    "Spin9SliceBridgeCertificate",
    "build_certificate",
    "certificate_json",
    "diagnostics",
    "exact_algebraic_intertwiner_basis",
    "supported_sym0_basis",
]
