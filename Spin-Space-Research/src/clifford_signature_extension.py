"""Exact Cl(3,0) -> Cl^0(1,4) bridge inside the Spin(9) Clifford system.

The maintained Spin(9) involutions give a faithful 16-real-dimensional model
of ``Cl(1,4)`` after a signature-changing product construction.  Its even
algebra contains a concrete copy of ``Cl(3,0)``.  This module certifies blade
ranks, central volume sectors, real Schur types, restriction multiplicities,
and the simultaneous Spin(8) branching controls.

No equality between the 8-dimensional algebra ``Cl(3,0)`` and the
32-dimensional algebra ``Cl(1,4)`` is claimed.  The established map is an
injective algebra homomorphism into the 16-dimensional even subalgebra.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import sympy as sp
from algebraic_isotypic_decomposition import algebraic_change
from exact_real_scalar_field import Q_SQRT_TWO
from reducible_isotypic_decomposition import (
    cl3_spin3_fixture,
    decompose_reducible_representation,
    exact_intertwiner_basis,
)
from schur_type_detector import detect_schur_type, exact_commutant_basis
from spin9_dirac_clifford import build_spin9_clifford_system

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "clifford_signature_extension_20260811.json"


def _flatten(matrix: sp.Matrix) -> sp.Matrix:
    return sp.Matrix(list(matrix))


def _matrix_equal(left: sp.Matrix, right: sp.Matrix) -> bool:
    difference = (sp.Matrix(left) - sp.Matrix(right)).applyfunc(sp.simplify)
    return difference == sp.zeros(*difference.shape)


def _matrix_json(matrix: sp.Matrix) -> list[list[str]]:
    return [
        [sp.sstr(sp.factor(matrix[row, column])) for column in range(matrix.cols)]
        for row in range(matrix.rows)
    ]


def clifford_1_4_generators() -> tuple[sp.Matrix, ...]:
    """Return five exact generators with signature ``(+----)``."""

    involutions = tuple(
        sp.Matrix(value) for value in build_spin9_clifford_system().involutions
    )
    positive = involutions[0]
    negative = tuple(positive * involutions[index] for index in range(1, 5))
    return (positive, *negative)


def clifford_blades(generators: tuple[sp.Matrix, ...]) -> tuple[sp.Matrix, ...]:
    """Return ordered blade images indexed by binary subset mask."""

    dimension = generators[0].rows
    blades = []
    for mask in range(1 << len(generators)):
        value = sp.eye(dimension)
        for index, generator in enumerate(generators):
            if mask & (1 << index):
                value = value * generator
        blades.append(value)
    return tuple(blades)


def _span_rank(matrices: tuple[sp.Matrix, ...] | list[sp.Matrix]) -> int:
    return Q_SQRT_TWO.rank(sp.Matrix.hstack(*(_flatten(value) for value in matrices)))


def _projector_restriction(
    generators: tuple[sp.Matrix, ...] | list[sp.Matrix], projector: sp.Matrix
) -> tuple[sp.Matrix, ...]:
    basis = sp.Matrix.hstack(*projector.columnspace())
    left_inverse = (basis.T * basis).inv() * basis.T
    return tuple(
        (left_inverse * generator * basis).applyfunc(sp.simplify)
        for generator in generators
    )


def _commutant_dimension(generators: tuple[sp.Matrix, ...]) -> int:
    return len(exact_commutant_basis(generators, scalar_extension=sp.sqrt(2)))


def _spin8_modules() -> dict[str, tuple[sp.Matrix, ...]]:
    system = build_spin9_clifford_system()
    pair_to_index = {pair: index for index, pair in enumerate(system.generator_pairs)}
    adjacent = [pair_to_index[(index, index + 1)] for index in range(7)]
    vector = tuple(
        sp.Matrix(system.vector_generators[index])[:8, :8] for index in adjacent
    )
    spin = tuple(
        sp.Matrix(system.doubled_spin_generators[index]) / 2 for index in adjacent
    )
    return {
        "8v": vector,
        "8+": tuple(value[:8, :8] for value in spin),
        "8-": tuple(value[8:, 8:] for value in spin),
    }


def _spin8_full_generators() -> tuple[list[int], dict[str, tuple[sp.Matrix, ...]]]:
    system = build_spin9_clifford_system()
    indices = [
        index for index, pair in enumerate(system.generator_pairs) if pair[1] < 8
    ]
    vector = tuple(
        sp.Matrix(system.vector_generators[index])[:8, :8] for index in indices
    )
    spin = tuple(
        sp.Matrix(system.doubled_spin_generators[index]) / 2 for index in indices
    )
    return indices, {
        "8v": vector,
        "8+": tuple(value[:8, :8] for value in spin),
        "8-": tuple(value[8:, 8:] for value in spin),
    }


def _lie_closure_dimension(generators: tuple[sp.Matrix, ...]) -> int:
    basis: list[sp.Matrix] = []

    def add(candidate: sp.Matrix) -> bool:
        if not any(candidate):
            return False
        old_rank = _span_rank(basis) if basis else 0
        new_rank = _span_rank([*basis, candidate])
        if new_rank == old_rank:
            return False
        basis.append(candidate)
        return True

    for generator in generators:
        add(generator)
    changed = True
    while changed:
        changed = False
        snapshot = tuple(basis)
        for left, right in itertools.combinations(snapshot, 2):
            if add(left * right - right * left):
                changed = True
    return len(basis)


def diagnostics() -> dict[str, object]:
    generators = clifford_1_4_generators()
    dimension = generators[0].rows
    identity = sp.eye(dimension)
    signature = (1, -1, -1, -1, -1)
    square_gates = tuple(
        _matrix_equal(generator**2, sign * identity)
        for generator, sign in zip(generators, signature, strict=True)
    )
    anticommuting = all(
        _matrix_equal(left * right + right * left, sp.zeros(dimension))
        for index, left in enumerate(generators)
        for right in generators[index + 1 :]
    )

    blades = clifford_blades(generators)
    even_blades = tuple(
        blade for mask, blade in enumerate(blades) if mask.bit_count() % 2 == 0
    )
    volume = blades[-1]
    central_projectors = ((identity + volume) / 2, (identity - volume) / 2)
    full_sector_generators = tuple(
        _projector_restriction(generators, projector)
        for projector in central_projectors
    )
    full_sector_schur = tuple(
        detect_schur_type(
            sector,
            assume_completely_reducible=True,
            scalar_extension=sp.sqrt(2),
        )
        for sector in full_sector_generators
    )
    full_cross_dimension = len(
        exact_intertwiner_basis(
            full_sector_generators[0],
            full_sector_generators[1],
            scalar_extension=sp.sqrt(2),
        )
    )

    even_generators = tuple(generators[0] * generators[index] for index in range(1, 5))
    even_sector_generators = tuple(
        _projector_restriction(even_generators, projector)
        for projector in central_projectors
    )
    even_sector_schur = tuple(
        detect_schur_type(
            sector,
            assume_completely_reducible=True,
            scalar_extension=sp.sqrt(2),
        )
        for sector in even_sector_generators
    )
    even_cross_dimension = len(
        exact_intertwiner_basis(
            even_sector_generators[0],
            even_sector_generators[1],
            scalar_extension=sp.sqrt(2),
        )
    )

    embedded_cl3_generators = even_generators[:3]
    embedded_cl3_blades = clifford_blades(embedded_cl3_generators)
    cl3_sector_generators = tuple(
        _projector_restriction(embedded_cl3_generators, projector)
        for projector in central_projectors
    )
    cl3_sector_certificates = tuple(
        decompose_reducible_representation(
            sector,
            assume_completely_reducible=True,
            scalar_extension=sp.sqrt(2),
        )
        for sector in cl3_sector_generators
    )
    cl3_cross_dimension = len(
        exact_intertwiner_basis(
            cl3_sector_generators[0],
            cl3_sector_generators[1],
            scalar_extension=sp.sqrt(2),
        )
    )
    full_commutant_dimension = _commutant_dimension(generators)
    even_commutant_dimension = _commutant_dimension(even_generators)
    cl3_commutant_dimension = _commutant_dimension(embedded_cl3_generators)
    full_blade_rank = _span_rank(blades)
    even_blade_rank = _span_rank(even_blades)
    cl3_blade_rank = _span_rank(embedded_cl3_blades)

    spin8_adjacent = _spin8_modules()
    spin8_indices, _spin8_full = _spin8_full_generators()
    spin8_schur = {
        name: detect_schur_type(
            module,
            assume_completely_reducible=True,
            scalar_extension=sp.sqrt(2),
        )
        for name, module in spin8_adjacent.items()
    }
    spin8_pairwise = {
        left_name: {
            right_name: len(
                exact_intertwiner_basis(
                    left,
                    right,
                    scalar_extension=sp.sqrt(2),
                )
            )
            for right_name, right in spin8_adjacent.items()
        }
        for left_name, left in spin8_adjacent.items()
    }
    spin8_algebraic_schur = {}
    spin8_algebraic_modules = {}
    for name, module in spin8_adjacent.items():
        change = algebraic_change(8)
        inverse = change.inv()
        conjugated = tuple(
            (change * generator * inverse).applyfunc(sp.factor) for generator in module
        )
        spin8_algebraic_modules[name] = conjugated
        spin8_algebraic_schur[name] = detect_schur_type(
            conjugated,
            assume_completely_reducible=True,
            scalar_extension=sp.sqrt(2),
        )
    spin8_algebraic_pairwise = {
        left_name: {
            right_name: len(
                exact_intertwiner_basis(
                    left,
                    right,
                    scalar_extension=sp.sqrt(2),
                )
            )
            for right_name, right in spin8_algebraic_modules.items()
        }
        for left_name, left in spin8_algebraic_modules.items()
    }
    system = build_spin9_clifford_system()
    chirality = sp.Matrix(system.involutions[8])
    chirality_projectors = ((identity + chirality) / 2, (identity - chirality) / 2)
    spin8_spin_generators = tuple(
        sp.Matrix(system.doubled_spin_generators[index]) / 2 for index in spin8_indices
    )
    spin8_lie_closure_dimension = _lie_closure_dimension(spin8_adjacent["8v"])

    maintained_cl3 = decompose_reducible_representation(
        cl3_spin3_fixture(), assume_completely_reducible=True
    )
    maintained_cl3_signatures = sorted(
        [block.schur_type, block.multiplicity, block.irreducible_dimension]
        for block in maintained_cl3.blocks
    )
    cl3_sector_signatures = [
        sorted(
            [block.schur_type, block.multiplicity, block.irreducible_dimension]
            for block in certificate.blocks
        )
        for certificate in cl3_sector_certificates
    ]

    gates = {
        "cl_1_4_signature_squares": all(square_gates),
        "cl_1_4_generators_anticommute": anticommuting,
        "full_cl_1_4_blade_rank_is_32": full_blade_rank == 32,
        "even_cl_1_4_blade_rank_is_16": even_blade_rank == 16,
        "volume_is_central_involution": bool(
            _matrix_equal(volume**2, identity)
            and all(
                _matrix_equal(volume * value, value * volume) for value in generators
            )
        ),
        "volume_projectors_are_complementary_rank_eight": bool(
            all(_matrix_equal(value**2, value) for value in central_projectors)
            and _matrix_equal(
                central_projectors[0] * central_projectors[1], sp.zeros(16)
            )
            and sum(Q_SQRT_TWO.rank(value) for value in central_projectors) == 16
            and all(Q_SQRT_TWO.rank(value) == 8 for value in central_projectors)
        ),
        "full_cl_1_4_sectors_are_inequivalent_quaternionic": bool(
            all(
                value.classified_irreducible and value.schur_type == "quaternion"
                for value in full_sector_schur
            )
            and full_cross_dimension == 0
            and full_commutant_dimension == 8
        ),
        "even_cl_1_4_is_two_equivalent_quaternionic_modules": bool(
            all(
                value.classified_irreducible and value.schur_type == "quaternion"
                for value in even_sector_schur
            )
            and even_cross_dimension == 4
            and even_commutant_dimension == 16
        ),
        "embedded_cl_3_0_generators_are_positive_and_anticommuting": bool(
            all(_matrix_equal(value**2, identity) for value in embedded_cl3_generators)
            and all(
                _matrix_equal(left * right + right * left, sp.zeros(16))
                for index, left in enumerate(embedded_cl3_generators)
                for right in embedded_cl3_generators[index + 1 :]
            )
        ),
        "embedded_cl_3_0_blade_rank_is_eight": cl3_blade_rank == 8,
        "embedded_cl_3_0_lies_in_even_cl_1_4": _span_rank(
            [*even_blades, *embedded_cl3_blades]
        )
        == 16,
        "embedded_cl_3_0_is_four_complex_spinor_copies": bool(
            all(certificate.certified for certificate in cl3_sector_certificates)
            and cl3_sector_signatures == [[["complex", 2, 4]], [["complex", 2, 4]]]
            and cl3_cross_dimension == 8
            and cl3_commutant_dimension == 32
        ),
        "adjacent_spin8_generators_close_to_dimension_28": (
            spin8_lie_closure_dimension == 28
        ),
        "spin8_triality_irreps_are_real_and_pairwise_inequivalent": bool(
            all(
                certificate.classified_irreducible and certificate.schur_type == "real"
                for certificate in spin8_schur.values()
            )
            and spin8_pairwise
            == {
                "8v": {"8v": 1, "8+": 0, "8-": 0},
                "8+": {"8v": 0, "8+": 1, "8-": 0},
                "8-": {"8v": 0, "8+": 0, "8-": 1},
            }
        ),
        "spin8_triality_survives_q_sqrt_two_conjugacy": bool(
            all(
                certificate.classified_irreducible and certificate.schur_type == "real"
                for certificate in spin8_algebraic_schur.values()
            )
            and spin8_algebraic_pairwise == spin8_pairwise
        ),
        "spin9_spinor_restricts_to_spin8_chiral_halves": bool(
            all(Q_SQRT_TWO.rank(value) == 8 for value in chirality_projectors)
            and all(
                _matrix_equal(chirality * generator, generator * chirality)
                for generator in spin8_spin_generators
            )
        ),
        "maintained_cl3_conjugation_state_remains_two_v0_plus_two_v1": (
            maintained_cl3.certified
            and maintained_cl3_signatures == [["real", 2, 1], ["real", 2, 3]]
        ),
    }

    return {
        "schema_version": 1,
        "claim_scope": (
            "exact Clifford-algebra embedding and real-module branching inside "
            "the maintained Spin(9) system; no equality of Cl(3,0) and Cl(1,4), "
            "and no model-quality consequence"
        ),
        "cl_1_4_signature": [1, -1, -1, -1, -1],
        "cl_1_4_generator_shape": [5, 16, 16],
        "cl_1_4_blade_rank": full_blade_rank,
        "cl_1_4_even_blade_rank": even_blade_rank,
        "volume_matrix": _matrix_json(volume),
        "volume_projector_ranks": [
            Q_SQRT_TWO.rank(value) for value in central_projectors
        ],
        "full_sector_schur_types": [value.schur_type for value in full_sector_schur],
        "full_sector_intertwiner_dimension": full_cross_dimension,
        "full_commutant_dimension": full_commutant_dimension,
        "even_sector_schur_types": [value.schur_type for value in even_sector_schur],
        "even_sector_intertwiner_dimension": even_cross_dimension,
        "even_commutant_dimension": even_commutant_dimension,
        "embedded_cl_3_0_blade_rank": cl3_blade_rank,
        "embedded_cl_3_0_sector_signatures": cl3_sector_signatures,
        "embedded_cl_3_0_cross_intertwiner_dimension": cl3_cross_dimension,
        "embedded_cl_3_0_commutant_dimension": cl3_commutant_dimension,
        "spin8_generator_count": len(spin8_indices),
        "spin8_adjacent_generator_count": len(spin8_adjacent["8v"]),
        "spin8_lie_closure_dimension": spin8_lie_closure_dimension,
        "spin8_triality_schur_types": {
            name: certificate.schur_type for name, certificate in spin8_schur.items()
        },
        "spin8_triality_pairwise_intertwiner_dimensions": spin8_pairwise,
        "spin8_algebraic_pairwise_intertwiner_dimensions": spin8_algebraic_pairwise,
        "maintained_cl3_conjugation_isotypic_signatures": maintained_cl3_signatures,
        "dimension_ledger": {
            "Cl(3,0)": 8,
            "Cl^0(1,4)": 16,
            "Cl(1,4)": 32,
        },
        "exact_gates": gates,
        "cl_3_0_equals_cl_1_4_claimed": False,
        "maintained_cl3_model_embedded_as_same_state_model": False,
        "spin8_or_spin9_model_advantage_claimed": False,
        "passed": all(gates.values()),
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
    "clifford_1_4_generators",
    "clifford_blades",
    "diagnostics",
]
