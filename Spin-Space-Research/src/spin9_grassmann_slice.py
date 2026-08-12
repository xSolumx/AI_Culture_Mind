"""Exact normal-slice certificate for the symmetric Spin(9) three-plane.

At the Cayley-null plane, the Spin(9) orbit has dimension 33 inside the
39-dimensional Grassmannian G_3(R^16).  This module constructs the exact
six-dimensional normal slice and proves that its stabilizer representation is
the direct sum of a trivial line and the five-dimensional spin-2 irrep.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import sympy as sp

from spin9_dirac_clifford import build_spin9_clifford_system
from spin9_three_spinor_conditioning import _symbolic_template
from spin9_three_spinor_symmetry import _canonical_integer_vector


@dataclass(frozen=True)
class GrassmannSliceData:
    """Exact matrices underlying the Cayley-null normal-slice certificate."""

    spinor_frame: sp.Matrix
    projector: sp.Matrix
    orbit: sp.Matrix
    horizontal_constraint: sp.Matrix
    normal_basis: sp.Matrix
    normal_metric: sp.Matrix
    stabilizer_coordinates: tuple[sp.Matrix, ...]
    stabilizer: tuple[sp.Matrix, ...]
    slice_actions: tuple[sp.Matrix, ...]
    casimir: sp.Matrix
    curve_coordinates: sp.Matrix


def _horizontal_constraint(spinor_frame: sp.Matrix) -> sp.Matrix:
    """Linear map encoding B^T Z=0 on row-major 16-by-3 matrices Z."""

    symbols = sp.symbols("z0:48")
    variation = sp.Matrix(16, 3, symbols)
    equations = spinor_frame.T * variation
    return sp.Matrix(
        [[sp.diff(equation, symbol) for symbol in symbols] for equation in equations]
    )


def construct_slice_data() -> GrassmannSliceData:
    """Return the exact matrices used by the normal-slice theorem.

    This is a reusable algebraic API: downstream certificates can inspect and
    transform the concrete stabilizer representation without parsing the JSON
    diagnostics or reconstructing a second, potentially drifting copy of the
    slice calculation.
    """

    c, spinors, _, substitution = _symbolic_template()
    spinor_frame = sp.Matrix.hstack(*spinors).subs(substitution).subs(c, 0)
    projector = sp.simplify(spinor_frame * spinor_frame.T)
    system = build_spin9_clifford_system()
    generators = [sp.Matrix(matrix) / 2 for matrix in system.doubled_spin_generators]

    orbit = sp.Matrix.hstack(
        *[
            sp.simplify((sp.eye(16) - projector) * generator * spinor_frame).reshape(
                48, 1
            )
            for generator in generators
        ]
    )
    horizontal_constraint = _horizontal_constraint(spinor_frame)
    normal_basis = sp.Matrix.hstack(
        *sp.Matrix.vstack(orbit.T, horizontal_constraint).nullspace()
    )
    normal_metric = sp.simplify(normal_basis.T * normal_basis)

    stabilizer_coordinates = tuple(
        _canonical_integer_vector(vector) for vector in orbit.nullspace()
    )
    stabilizer = tuple(
        sum(
            (coordinates[index] * generators[index] for index in range(36)),
            sp.zeros(16),
        )
        for coordinates in stabilizer_coordinates
    )

    slice_actions = []
    for element in stabilizer:
        induced_plane_action = sp.simplify(spinor_frame.T * element * spinor_frame)
        columns = []
        for index in range(normal_basis.cols):
            variation = sp.Matrix(16, 3, list(normal_basis[:, index]))
            transformed = sp.simplify(
                element * variation - variation * induced_plane_action
            )
            coordinates = sp.simplify(
                normal_metric.inv() * normal_basis.T * transformed.reshape(48, 1)
            )
            columns.append(coordinates)
        slice_actions.append(sp.Matrix.hstack(*columns))

    casimir = sp.simplify(
        -(slice_actions[0] ** 2 + 2 * slice_actions[1] ** 2 + slice_actions[2] ** 2) / 2
    )

    u = sp.symbols("u", positive=True)
    radical = sp.sqrt((3 - u**2) * (1 + u**2))
    curve_frame = sp.zeros(16, 3)
    curve_frame[0, 0] = 1
    curve_frame[1, 1] = 1
    curve_frame[8, 1] = u
    curve_frame[2, 2] = -2
    curve_frame[11, 2] = u * (1 - u**2)
    curve_frame[12, 2] = u * radical
    curve_projector = sp.simplify(
        curve_frame * (curve_frame.T * curve_frame).inv() * curve_frame.T
    )
    projector_derivative = sp.diff(curve_projector, u).subs(u, 1)
    curve_tangent = sp.simplify(projector_derivative * spinor_frame)
    curve_coordinates = sp.simplify(
        normal_metric.inv() * normal_basis.T * curve_tangent.reshape(48, 1)
    )

    return GrassmannSliceData(
        spinor_frame=spinor_frame,
        projector=projector,
        orbit=orbit,
        horizontal_constraint=horizontal_constraint,
        normal_basis=normal_basis,
        normal_metric=normal_metric,
        stabilizer_coordinates=stabilizer_coordinates,
        stabilizer=stabilizer,
        slice_actions=tuple(slice_actions),
        casimir=casimir,
        curve_coordinates=curve_coordinates,
    )


def diagnostics() -> dict[str, object]:
    """Reconstruct the exact V1+V5 Grassmann normal-slice theorem."""

    data = construct_slice_data()
    orbit = data.orbit
    horizontal_constraint = data.horizontal_constraint
    normal_basis = data.normal_basis
    normal_metric = data.normal_metric
    slice_actions = data.slice_actions

    metric_skew_identities = [
        sp.simplify(action.T * normal_metric + normal_metric * action) == sp.zeros(6)
        for action in slice_actions
    ]
    casimir = data.casimir
    casimir_eigenvalues = {
        int(eigenvalue): int(multiplicity)
        for eigenvalue, multiplicity in casimir.eigenvals().items()
    }

    curve_coordinates = data.curve_coordinates

    report: dict[str, object] = {
        "schema_version": 1,
        "claim_scope": "exact normal-slice representation at c=0",
        "grassmann_dimension": 39,
        "spin9_dimension": 36,
        "plane_stabilizer_dimension": 3,
        "orbit_rank": int(orbit.rank()),
        "horizontal_constraint_rank": int(horizontal_constraint.rank()),
        "normal_slice_dimension": int(normal_basis.cols),
        "slice_actions_metric_skew": metric_skew_identities,
        "slice_casimir_eigenvalue_multiplicities": casimir_eigenvalues,
        "slice_branching": "V1 + V5",
        "curve_tangent_coordinates": [sp.sstr(value) for value in curve_coordinates],
        "curve_tangent_nonzero": curve_coordinates != sp.zeros(6, 1),
        "curve_tangent_is_trivial_slice_direction": sp.simplify(
            casimir * curve_coordinates
        )
        == sp.zeros(6, 1),
        "local_invariant_degrees": [1, 2, 3],
        "global_grassmann_quotient_solved": False,
    }
    report["passed"] = bool(
        report["orbit_rank"] == 33
        and report["horizontal_constraint_rank"] == 9
        and report["normal_slice_dimension"] == 6
        and all(metric_skew_identities)
        and casimir_eigenvalues == {6: 5, 0: 1}
        and report["curve_tangent_nonzero"]
        and report["curve_tangent_is_trivial_slice_direction"]
        and not report["global_grassmann_quotient_solved"]
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional JSON artifact path")
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
