"""Exact algebraic-scalar extension gates for real isotypic decomposition.

This module promotes ``Q(sqrt(2))`` as the first non-rational arithmetic field
accepted by the maintained Schur and reducible compilers.  It includes a
genuine splitting obstruction, nonsplit division controls, algebraic basis
conjugacies, and direct compilation of the concrete Spin(9) slice before any
rationalizing change of coordinates.

The word "field" here describes certificate arithmetic.  The represented
vector spaces and Schur types remain real.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp
from exact_real_scalar_field import Q_SQRT_TWO
from reducible_isotypic_decomposition import (
    ReducibleIsotypicCertificate,
    decompose_reducible_representation,
)
from reducible_isotypic_decomposition import (
    certificate_json as reducible_certificate_json,
)
from schur_type_detector import (
    SchurTypeCertificate,
    canonical_examples,
    detect_schur_type,
)
from schur_type_detector import certificate_json as schur_certificate_json
from spin9_slice_isotypic_bridge import build_certificate as build_spin9_bridge

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "algebraic_isotypic_decomposition_20260811.json"


def quadratic_split_generator() -> sp.Matrix:
    """Return a rational operator with eigenprojectors only over Q(sqrt(2))."""

    return sp.Matrix([[0, 2], [1, 0]])


def negative_square_generator() -> sp.Matrix:
    """Return the nonsplit complex-type control with square ``-2 I``."""

    return sp.Matrix([[0, -2], [1, 0]])


def algebraic_change(dimension: int) -> sp.Matrix:
    """Return a deterministic dense-enough Q(sqrt(2)) basis change."""

    change = sp.eye(dimension)
    for index in range(dimension - 1):
        change[index, index + 1] = sp.sqrt(2)
    if dimension > 2:
        change[0, -1] = 1 + sp.sqrt(2)
    return change


def algebraically_conjugate(
    generators: list[sp.Matrix] | tuple[sp.Matrix, ...],
) -> tuple[sp.Matrix, ...]:
    dimension = generators[0].rows
    change = algebraic_change(dimension)
    inverse = change.inv()
    return tuple(
        (change * generator * inverse).applyfunc(sp.factor) for generator in generators
    )


def _block_signatures(
    certificate: ReducibleIsotypicCertificate,
) -> list[tuple[str, int, int, int]]:
    return sorted(
        (
            block.schur_type,
            block.multiplicity,
            block.irreducible_dimension,
            block.commutant_dimension,
        )
        for block in certificate.blocks
    )


def _schur_summary(certificate: SchurTypeCertificate) -> dict[str, object]:
    return schur_certificate_json(certificate)


def diagnostics() -> dict[str, object]:
    split_generator = quadratic_split_generator()
    rational_attempt = decompose_reducible_representation(
        [split_generator],
        assume_completely_reducible=True,
    )
    extension_split = decompose_reducible_representation(
        [split_generator],
        assume_completely_reducible=True,
        scalar_extension=sp.sqrt(2),
    )
    nonsplit = decompose_reducible_representation(
        [negative_square_generator()],
        assume_completely_reducible=True,
        scalar_extension=sp.sqrt(2),
    )

    expected_schur = {
        "real_so3_vector": "real",
        "complex_u1_realification": "complex",
        "quaternion_su2_spinor_realification": "quaternion",
    }
    conjugated_schur = {
        name: detect_schur_type(
            algebraically_conjugate(generators),
            assume_completely_reducible=True,
            scalar_extension=sp.sqrt(2),
        )
        for name, generators in canonical_examples().items()
    }

    spin9_bridge = build_spin9_bridge()
    spin9_slice = decompose_reducible_representation(
        spin9_bridge.standardized_slice_generators,
        assume_completely_reducible=True,
        scalar_extension=sp.sqrt(2),
    )
    spin9_full = decompose_reducible_representation(
        spin9_bridge.concrete_generators,
        assume_completely_reducible=True,
        scalar_extension=sp.sqrt(2),
    )

    mismatch_rejected = False
    try:
        decompose_reducible_representation(
            [sp.Matrix([[sp.sqrt(3)]])],
            assume_completely_reducible=True,
            scalar_extension=sp.sqrt(2),
        )
    except ValueError as error:
        mismatch_rejected = "declared field" in str(error)

    split_projectors = [summand.projector for summand in extension_split.summands]
    expected_projectors = [
        sp.eye(2) / 2 + split_generator * sp.sqrt(2) / 4,
        sp.eye(2) / 2 - split_generator * sp.sqrt(2) / 4,
    ]
    projector_match = all(
        any(projector == expected for expected in expected_projectors)
        for projector in split_projectors
    )
    order_controls = {
        sp.sstr(value): Q_SQRT_TWO.sign(value)
        for value in (
            sp.sqrt(2) - 1,
            1 - sp.sqrt(2),
            3 - 2 * sp.sqrt(2),
            7 - 5 * sp.sqrt(2),
            sp.Integer(0),
        )
    }

    gates = {
        "field_degree_is_two": Q_SQRT_TWO.degree == 2,
        "field_defining_polynomial_is_x_squared_minus_two": (
            Q_SQRT_TWO.defining_polynomial == sp.Symbol("x") ** 2 - 2
        ),
        "exact_order_controls": list(order_controls.values()) == [1, -1, 1, -1, 0],
        "rational_compiler_refuses_missing_split": bool(
            not rational_attempt.certified
            and rational_attempt.unresolved_projector_ranks == (2,)
        ),
        "quadratic_extension_exposes_split": extension_split.certified,
        "quadratic_projectors_match_closed_form": projector_match,
        "quadratic_split_recovers_two_real_lines": _block_signatures(extension_split)
        == [("real", 1, 1, 1), ("real", 1, 1, 1)],
        "negative_square_control_remains_complex_irreducible": (
            nonsplit.certified and _block_signatures(nonsplit) == [("complex", 1, 2, 2)]
        ),
        "algebraic_conjugacies_preserve_real_schur_types": all(
            certificate.classified_irreducible
            and certificate.schur_type == expected_schur[name]
            for name, certificate in conjugated_schur.items()
        ),
        "undeclared_sqrt_three_is_rejected": mismatch_rejected,
        "concrete_spin9_slice_compiles_directly_over_quadratic_field": (
            spin9_slice.certified
            and _block_signatures(spin9_slice) == [("real", 1, 1, 1), ("real", 1, 5, 1)]
        ),
        "concrete_spin9_full_quotient_compiles_directly": (
            spin9_full.certified
            and _block_signatures(spin9_full) == [("real", 1, 1, 1), ("real", 2, 5, 4)]
        ),
    }

    return {
        "schema_version": 1,
        "claim_scope": (
            "exact real Schur and reducible decomposition using arithmetic in "
            "Q(sqrt(2)); generic algebraic fields and automatic field discovery "
            "remain open"
        ),
        "scalar_field": {
            "name": Q_SQRT_TWO.name,
            "degree": Q_SQRT_TWO.degree,
            "primitive_element": sp.sstr(Q_SQRT_TWO.primitive_element),
            "defining_polynomial": sp.sstr(Q_SQRT_TWO.defining_polynomial),
            "real_embedding": "sqrt(2) > 0",
        },
        "exact_order_controls": order_controls,
        "rational_obstruction": reducible_certificate_json(rational_attempt),
        "quadratic_split": reducible_certificate_json(extension_split),
        "nonsplit_complex_control": reducible_certificate_json(nonsplit),
        "algebraic_schur_conjugacy_controls": {
            name: _schur_summary(certificate)
            for name, certificate in conjugated_schur.items()
        },
        "spin9_concrete_slice": reducible_certificate_json(spin9_slice),
        "spin9_concrete_full_quotient": reducible_certificate_json(spin9_full),
        "exact_gates": gates,
        "automatic_field_discovery_implemented": False,
        "generic_number_field_signs_implemented": False,
        "noisy_numeric_input_supported": False,
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
    "algebraic_change",
    "algebraically_conjugate",
    "diagnostics",
    "negative_square_generator",
    "quadratic_split_generator",
]
