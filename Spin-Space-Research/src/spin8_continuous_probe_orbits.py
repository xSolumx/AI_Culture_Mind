"""Exact prerequisites for the continuous Spin(8) probe-orbit theorem.

Four unit probes have four allocation types up to triality permutation.  This
module gives algebraically independent invariant functions and exact orbit-rank
representatives for every type.  The compact principal-orbit theorem then
upgrades the calculation: every four-probe stabilizer has positive dimension.

For five probes, an exact full triality closure is exhibited in every mixed
allocation type.  Those points have trivial global stabilizer, hence the
principal stabilizer of each mixed family is trivial and its free stratum is
open and dense.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import sympy as sp

from spin8_coordinate_geometry import _actual_closure, _exact_algebra
from spin8_triality import SPIN8_PAIRS

DIMENSION = 8
Probe = tuple[int, int]

FOUR_PROBE_REPRESENTATIVES: dict[str, tuple[Probe, ...]] = {
    "4_0_0": ((0, 0), (0, 1), (0, 2), (0, 3)),
    "3_1_0": ((0, 0), (0, 1), (0, 2), (1, 0)),
    "2_2_0": ((0, 0), (0, 1), (1, 0), (1, 2)),
    "2_1_1": ((0, 0), (0, 1), (1, 0), (2, 2)),
}

FIVE_PROBE_REPRESENTATIVES: dict[str, tuple[Probe, ...]] = {
    "4_1_0": ((0, 0), (0, 1), (0, 2), (0, 4), (1, 0)),
    "3_2_0": ((0, 0), (0, 1), (0, 2), (1, 0), (1, 4)),
    "3_1_1": ((0, 0), (0, 1), (0, 2), (1, 0), (2, 4)),
    "2_2_1": ((0, 0), (0, 1), (1, 0), (1, 2), (2, 4)),
}


def _probe_constraint_matrix(
    probes: Iterable[Probe], scaled_families: tuple[np.ndarray, ...]
) -> sp.Matrix:
    rows = tuple(probes)
    columns = []
    for generator_index in range(len(SPIN8_PAIRS)):
        column: list[int] = []
        for representation, coordinate in rows:
            column.extend(
                int(value)
                for value in scaled_families[representation][
                    generator_index, :, coordinate
                ]
            )
        columns.append(column)
    return sp.Matrix(columns).T


def _lie_type_certificate(
    constraint: sp.Matrix, scaled_vector_family: np.ndarray
) -> dict[str, object]:
    nullspace = constraint.nullspace()
    basis = []
    for coefficients in nullspace:
        generator = sp.zeros(DIMENSION)
        for index, coefficient in enumerate(coefficients):
            generator += coefficient * sp.Matrix(scaled_vector_family[index])
        basis.append(generator)
    if not basis:
        return {
            "dimension": 0,
            "derived_algebra_rank": 0,
            "centre_dimension": 0,
            "killing_form_negative_definite": True,
            "classification": "trivial Lie algebra",
        }

    flattened = sp.Matrix.hstack(
        *(generator.reshape(DIMENSION * DIMENSION, 1) for generator in basis)
    )
    brackets = {}
    for left in range(len(basis)):
        for right in range(len(basis)):
            bracket = (basis[left] * basis[right] - basis[right] * basis[left]).reshape(
                DIMENSION * DIMENSION, 1
            )
            solution, parameters = flattened.gauss_jordan_solve(bracket)
            if parameters.rows:
                raise AssertionError("stabilizer basis is not independent")
            brackets[(left, right)] = solution
    adjoint = [
        sp.Matrix.hstack(*(brackets[(left, right)] for right in range(len(basis))))
        for left in range(len(basis))
    ]
    derived_rank = int(sp.Matrix.hstack(*brackets.values()).rank())
    centre_system = sp.Matrix.vstack(
        *(
            sp.Matrix.hstack(*(brackets[(left, right)] for left in range(len(basis))))
            for right in range(len(basis))
        )
    )
    centre_dimension = len(basis) - int(centre_system.rank())
    killing = sp.Matrix(
        [
            [sp.trace(adjoint[left] * adjoint[right]) for right in range(len(basis))]
            for left in range(len(basis))
        ]
    )
    leading_signs = [
        int(sp.sign(killing[:order, :order].det()))
        for order in range(1, len(basis) + 1)
    ]
    negative_definite = all(
        sign == (-1) ** order for order, sign in enumerate(leading_signs, start=1)
    )
    if len(basis) == 3 and derived_rank == 3 and centre_dimension == 0:
        classification = "compact simple A1, hence su(2)"
    elif len(basis) == 6 and derived_rank == 6 and centre_dimension == 0:
        classification = "compact semisimple A1+A1, hence spin(4)"
    else:
        classification = "unclassified"
    return {
        "dimension": len(basis),
        "derived_algebra_rank": derived_rank,
        "centre_dimension": centre_dimension,
        "killing_form_determinant": str(killing.det()),
        "killing_form_leading_principal_minor_signs": leading_signs,
        "killing_form_negative_definite": negative_definite,
        "classification": classification,
    }


def _tangent_row(probes: tuple[Probe, ...], gradient: np.ndarray) -> list[int]:
    # Every representative is a coordinate unit vector.  Its tangent basis is
    # the seven other coordinate directions.
    return [
        int(gradient[probe_index, coordinate])
        for probe_index, (_, radial_coordinate) in enumerate(probes)
        for coordinate in range(DIMENSION)
        if coordinate != radial_coordinate
    ]


def _gram_gradient(probes: tuple[Probe, ...], left: int, right: int) -> np.ndarray:
    gradient = np.zeros((len(probes), DIMENSION), dtype=np.int64)
    gradient[left, probes[right][1]] = 1
    gradient[right, probes[left][1]] = 1
    return gradient


def _triality_gradient(
    probes: tuple[Probe, ...],
    vector_index: int,
    positive_index: int,
    negative_index: int,
    rho: np.ndarray,
) -> np.ndarray:
    gradient = np.zeros((len(probes), DIMENSION), dtype=np.int64)
    vector_coordinate = probes[vector_index][1]
    positive_coordinate = probes[positive_index][1]
    negative_coordinate = probes[negative_index][1]
    gradient[vector_index] = [
        rho[coordinate, negative_coordinate, positive_coordinate]
        for coordinate in range(DIMENSION)
    ]
    gradient[positive_index] = rho[vector_coordinate, negative_coordinate, :]
    gradient[negative_index] = rho[vector_coordinate, :, positive_coordinate]
    return gradient


def _quartic_gradient(
    probes: tuple[Probe, ...],
    first_vector: int,
    second_vector: int,
    first_positive: int,
    second_positive: int,
    rho: np.ndarray,
) -> np.ndarray:
    """Gradient of <mu(v1,p1), mu(v2,p2)> at a coordinate tuple."""

    gradient = np.zeros((len(probes), DIMENSION), dtype=np.int64)
    v1 = probes[first_vector][1]
    v2 = probes[second_vector][1]
    p1 = probes[first_positive][1]
    p2 = probes[second_positive][1]
    n1 = rho[v1, :, p1]
    n2 = rho[v2, :, p2]
    gradient[first_vector] = [rho[index, :, p1] @ n2 for index in range(8)]
    gradient[first_positive] = rho[v1].T @ n2
    gradient[second_vector] = [n1 @ rho[index, :, p2] for index in range(8)]
    gradient[second_positive] = rho[v2].T @ n1
    return gradient


def _four_probe_invariant_gradients(
    name: str, probes: tuple[Probe, ...], rho: np.ndarray
) -> tuple[list[str], list[np.ndarray]]:
    if name == "4_0_0":
        pairs = tuple(itertools.combinations(range(4), 2))
        return (
            [f"gram_{left}_{right}" for left, right in pairs],
            [_gram_gradient(probes, left, right) for left, right in pairs],
        )
    if name == "3_1_0":
        pairs = tuple(itertools.combinations(range(3), 2))
        return (
            [f"gram_{left}_{right}" for left, right in pairs],
            [_gram_gradient(probes, left, right) for left, right in pairs],
        )
    if name == "2_2_0":
        return (
            ["vector_gram", "positive_gram", "quartic_triality_contraction"],
            [
                _gram_gradient(probes, 0, 1),
                _gram_gradient(probes, 2, 3),
                _quartic_gradient(probes, 0, 1, 2, 3, rho),
            ],
        )
    if name == "2_1_1":
        return (
            ["vector_gram", "triality_scalar_0", "triality_scalar_1"],
            [
                _gram_gradient(probes, 0, 1),
                _triality_gradient(probes, 0, 2, 3, rho),
                _triality_gradient(probes, 1, 2, 3, rho),
            ],
        )
    raise KeyError(name)


def _infinitesimal_triality_certificate(
    rho: np.ndarray, scaled_families: tuple[np.ndarray, ...]
) -> dict[str, object]:
    maximum = 0
    nonzero_generators = []
    for generator_index in range(len(SPIN8_PAIRS)):
        vector = scaled_families[0][generator_index]
        positive = scaled_families[1][generator_index]
        negative = scaled_families[2][generator_index]
        residual = (
            np.einsum("vnp,va->anp", rho, vector)
            + np.einsum("vnp,pc->vnc", rho, positive)
            + np.einsum("vnp,nb->vbp", rho, negative)
        )
        current = int(np.max(np.abs(residual)))
        maximum = max(maximum, current)
        if current:
            nonzero_generators.append(generator_index)
    skew = all(
        np.array_equal(generator + generator.T, np.zeros_like(generator))
        for family in scaled_families
        for generator in family
    )
    return {
        "all_84_generators_exactly_skew": skew,
        "triality_generators_checked": len(SPIN8_PAIRS),
        "maximum_integral_infinitesimal_triality_residual": maximum,
        "nonzero_residual_generators": nonzero_generators,
        "passed": skew and maximum == 0,
    }


def _four_probe_certificate(
    rho: np.ndarray, scaled_families: tuple[np.ndarray, ...]
) -> list[dict[str, object]]:
    rows = []
    for name, probes in FOUR_PROBE_REPRESENTATIVES.items():
        invariant_names, gradients = _four_probe_invariant_gradients(name, probes, rho)
        tangent_jacobian = sp.Matrix(
            [_tangent_row(probes, gradient) for gradient in gradients]
        )
        constraint = _probe_constraint_matrix(probes, scaled_families)
        action_rank = int(constraint.rank())
        invariant_rank = int(tangent_jacobian.rank())
        rows.append(
            {
                "allocation_up_to_triality": [int(value) for value in name.split("_")],
                "probes": [list(probe) for probe in probes],
                "invariants": invariant_names,
                "tangent_space_dimension": 4 * (DIMENSION - 1),
                "exact_invariant_jacobian_rank": invariant_rank,
                "exact_action_rank": action_rank,
                "exact_stabilizer_dimension": len(SPIN8_PAIRS) - action_rank,
                "exact_stabilizer_lie_type": _lie_type_certificate(
                    constraint, scaled_families[0]
                ),
                "rank_saturates_invariant_bound": action_rank
                == 4 * (DIMENSION - 1) - invariant_rank,
            }
        )
    return rows


def _five_probe_certificate(
    rho: np.ndarray, scaled_families: tuple[np.ndarray, ...]
) -> dict[str, object]:
    mixed_rows = []
    for name, probes in FIVE_PROBE_REPRESENTATIVES.items():
        closure = _actual_closure(probes, rho)
        constraint = _probe_constraint_matrix(probes, scaled_families)
        mixed_rows.append(
            {
                "allocation_up_to_triality": [int(value) for value in name.split("_")],
                "probes": [list(probe) for probe in probes],
                "closure_sizes": [len(support) for support in closure],
                "exact_action_rank": int(constraint.rank()),
                "global_stabilizer_trivial_by_full_closure": all(
                    support == tuple(range(DIMENSION)) for support in closure
                ),
            }
        )

    single_view = tuple((0, coordinate) for coordinate in range(5))
    single_constraint = _probe_constraint_matrix(single_view, scaled_families)
    return {
        "mixed_allocation_representatives": mixed_rows,
        "single_view_control": {
            "allocation": [5, 0, 0],
            "probes": [list(probe) for probe in single_view],
            "exact_action_rank": int(single_constraint.rank()),
            "exact_stabilizer_dimension": len(SPIN8_PAIRS)
            - int(single_constraint.rank()),
            "interpretation": "generic stabilizer Spin(3), so one-view sensing is insufficient",
        },
    }


def run() -> dict[str, object]:
    rho, scaled_families = _exact_algebra()
    invariance = _infinitesimal_triality_certificate(rho, scaled_families)
    four = _four_probe_certificate(rho, scaled_families)
    five = _five_probe_certificate(rho, scaled_families)

    four_exact = all(
        row["rank_saturates_invariant_bound"]
        and row["exact_stabilizer_dimension"]
        == (6 if row["allocation_up_to_triality"] == [4, 0, 0] else 3)
        and row["exact_stabilizer_lie_type"]["classification"]
        == (
            "compact semisimple A1+A1, hence spin(4)"
            if row["allocation_up_to_triality"] == [4, 0, 0]
            else "compact simple A1, hence su(2)"
        )
        for row in four
    )
    five_exact = all(
        row["closure_sizes"] == [8, 8, 8]
        and row["exact_action_rank"] == len(SPIN8_PAIRS)
        and row["global_stabilizer_trivial_by_full_closure"]
        for row in five["mixed_allocation_representatives"]
    )
    claims = {
        "every_continuous_four_probe_sensor_has_positive_dimensional_stabilizer": four_exact,
        "generic_mixed_four_probe_stabilizer_lie_algebra_is_su2": four_exact,
        "generic_single_view_four_probe_stabilizer_lie_algebra_is_spin4": four_exact,
        "every_mixed_five_probe_allocation_has_open_dense_free_stratum": five_exact,
        "single_view_five_probe_sensing_is_sufficient": False,
    }
    return {
        "experiment": "continuous Spin8 triality probe orbit theorem",
        "exact_invariance_prerequisites": invariance,
        "four_probe_allocation_certificates": four,
        "five_probe_allocation_certificates": five,
        "principal_orbit_argument": {
            "step_1": (
                "Exact tangent Jacobian rank proves the displayed invariant functions "
                "are algebraically independent in each four-probe allocation."
            ),
            "step_2": (
                "Invariant differentials annihilate orbit tangents, bounding principal "
                "orbit codimension by the invariant count."
            ),
            "step_3": (
                "Exact representatives saturate the bound, fixing the principal "
                "stabilizer dimension at six for one view and three for mixed views."
            ),
            "step_4": (
                "For a compact group action every isotropy group contains a conjugate "
                "of the principal isotropy group, so special tuples cannot have a "
                "smaller stabilizer."
            ),
            "step_5": (
                "A full-closure point in every mixed five-probe allocation has trivial "
                "global isotropy; therefore each mixed principal isotropy is trivial."
            ),
        },
        "proof_layers": {
            "exact_repository_premises": (
                "Invariant Jacobians, action ranks, stabilizer Lie algebras, "
                "triality closure, and explicit trivial-isotropy witnesses are "
                "computed in exact arithmetic."
            ),
            "classical_global_input": (
                "For a smooth compact-group action, the principal stratum is "
                "open and dense and every isotropy group contains a conjugate "
                "of the principal isotropy group."
            ),
            "deduction": (
                "The exact four-probe representatives saturate the invariant "
                "codimension bound, fixing the principal stabilizer dimension. "
                "The exact mixed five-probe free points force trivial principal "
                "isotropy in each mixed allocation."
            ),
        },
        "claims": claims,
        "proof_boundary": (
            "The theorem concerns unit probes and shared Spin(8) action. It classifies "
            "principal stabilizer dimension and proves universal four-probe "
            "insufficiency; it does not enumerate every nonprincipal five-probe orbit."
        ),
        "passed": invariance["passed"] and four_exact and five_exact,
    }


def verify_report(report: dict[str, object]) -> bool:
    return report == run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run()
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
