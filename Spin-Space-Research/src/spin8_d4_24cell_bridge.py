"""Exact D4/24-cell bridge and triality-projector geometry audit.

There are three unrelated conventions commonly denoted ``D4`` in this
repository's neighbourhood: the D4 root system, the Dynkin type of Spin(8),
and the order-eight dihedral benchmark.  This module certifies the precise
bridge between the first two and explicitly separates both from the benchmark.

The three minuscule weight orbits 8v, 8s+, and 8s- form a standard 24-cell.
The maintained coordinate probes are not those four-dimensional vertices:
their observation projectors are rank-seven points of Gr(7, 28).  We certify
their own exact, coloured tight-fusion-frame geometry as well.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp

from spin8_triality import build_spin8_triality_algebra


def _weight_orbits_scaled_by_two() -> dict[str, set[tuple[int, ...]]]:
    vector: set[tuple[int, ...]] = set()
    for coordinate in range(4):
        for sign in (-1, 1):
            value = [0] * 4
            value[coordinate] = 2 * sign
            vector.add(tuple(value))

    spinors = {
        parity: {
            signs
            for signs in itertools.product((-1, 1), repeat=4)
            if sum(sign < 0 for sign in signs) % 2 == parity
        }
        for parity in (0, 1)
    }
    return {"8v": vector, "8s_plus": spinors[0], "8s_minus": spinors[1]}


def _sphere_moment_s3(exponents: tuple[int, ...]) -> Fraction:
    if any(exponent % 2 for exponent in exponents):
        return Fraction(0)
    half = tuple(exponent // 2 for exponent in exponents)
    total = sum(half)
    numerator = 1
    for value in half:
        for odd in range(1, 2 * value, 2):
            numerator *= odd
    denominator = 1
    for index in range(total):
        denominator *= 4 + 2 * index
    return Fraction(numerator, denominator)


def _weight_certificate() -> dict[str, object]:
    orbits = _weight_orbits_scaled_by_two()
    names = ("8v", "8s_plus", "8s_minus")
    weights = sorted(set().union(*(orbits[name] for name in names)))

    hadamard = np.array(
        (
            (1, 1, 1, 1),
            (1, 1, -1, -1),
            (1, -1, 1, -1),
            (-1, 1, 1, -1),
        ),
        dtype=np.int64,
    )

    def image(points: set[tuple[int, ...]]) -> set[tuple[int, ...]]:
        result = set()
        for point in points:
            numerator = hadamard @ np.asarray(point, dtype=np.int64)
            if np.any(numerator % 2):
                raise AssertionError(
                    "triality image left the half-integral weight lattice"
                )
            result.add(tuple(int(value) for value in numerator // 2))
        return result

    cycle = {
        "8v": "8s_minus",
        "8s_minus": "8s_plus",
        "8s_plus": "8v",
    }
    cycle_passed = all(
        image(orbits[source]) == orbits[target] for source, target in cycle.items()
    )

    inner_products: Counter[Fraction] = Counter()
    for left_index, left in enumerate(weights):
        for right in weights[left_index + 1 :]:
            inner_products[Fraction(sum(a * b for a, b in zip(left, right)), 4)] += 1

    moment_failures = []
    monomial_count = 0
    for exponents in itertools.product(range(6), repeat=4):
        degree = sum(exponents)
        if degree > 5:
            continue
        monomial_count += 1
        observed = sum(
            Fraction(
                np.prod([point[axis] ** exponents[axis] for axis in range(4)]),
                2**degree,
            )
            for point in weights
        )
        expected = len(weights) * _sphere_moment_s3(exponents)
        if observed != expected:
            moment_failures.append(
                {
                    "exponents": list(exponents),
                    "observed": str(observed),
                    "expected": str(expected),
                }
            )

    passed = bool(
        [len(orbits[name]) for name in names] == [8, 8, 8]
        and len(weights) == 24
        and all(sum(value * value for value in point) == 4 for point in weights)
        and np.array_equal(hadamard @ hadamard.T, 4 * np.eye(4, dtype=np.int64))
        and np.array_equal(
            hadamard @ hadamard @ hadamard, 8 * np.eye(4, dtype=np.int64)
        )
        and cycle_passed
        and inner_products
        == Counter(
            {
                Fraction(-1): 12,
                Fraction(-1, 2): 96,
                Fraction(0): 72,
                Fraction(1, 2): 96,
            }
        )
        and not moment_failures
    )
    return {
        "coordinate_scale": "stored integer coordinates equal twice each unit weight",
        "orbit_sizes": {name: len(orbits[name]) for name in names},
        "union_size": len(weights),
        "pair_inner_product_distribution": {
            str(value): count for value, count in sorted(inner_products.items())
        },
        "order_three_triality_numerator": hadamard.tolist(),
        "triality_map": "T=H/2",
        "triality_cycle": cycle,
        "triality_cycle_exact": cycle_passed,
        "spherical_monomials_checked_through_degree_five": monomial_count,
        "spherical_five_design_failures": moment_failures,
        "interpretation": (
            "The union of the three minuscule D4 weight orbits is a standard "
            "24-cell realization, and an exact order-three orthogonal map cycles "
            "the vector and two chiral-spinor octets."
        ),
        "passed": passed,
    }


def _projector_certificate() -> dict[str, object]:
    algebra = build_spin8_triality_algebra()
    families_float = (
        algebra.vector_generators,
        algebra.positive_generators,
        algebra.negative_generators,
    )
    families = tuple(np.rint(2 * family).astype(np.int64) for family in families_float)
    exact_half_integral = all(
        np.array_equal(2 * source, scaled)
        for source, scaled in zip(families_float, families, strict=True)
    )
    if not exact_half_integral:
        raise AssertionError(
            "maintained triality generators are not exactly half-integral"
        )

    # P4 is four times the actual observation projector P.  Keeping this
    # integer scaling makes every certificate below exact.
    projectors = []
    labels = []
    for view, family in enumerate(families):
        for coordinate in range(8):
            jacobian_twice = family[:, :, coordinate].T
            projectors.append(jacobian_twice.T @ jacobian_twice)
            labels.append((view, coordinate))

    identity = np.eye(28, dtype=np.int64)
    projector_law = all(
        np.array_equal(projector @ projector, 4 * projector)
        and int(np.trace(projector)) == 28
        for projector in projectors
    )
    per_view_tight = all(
        np.array_equal(sum(projectors[8 * view : 8 * (view + 1)]), 8 * identity)
        for view in range(3)
    )
    full_tight = np.array_equal(sum(projectors), 24 * identity)

    overlap_numerators: Counter[tuple[str, int]] = Counter()
    same_intersection_law = True
    cross_isoclinic_law = True
    for left_index, left in enumerate(projectors):
        for right_index in range(left_index + 1, len(projectors)):
            right = projectors[right_index]
            same_view = labels[left_index][0] == labels[right_index][0]
            kind = "same_view" if same_view else "cross_view"
            overlap_numerator = int(np.trace(left @ right))
            overlap_numerators[(kind, overlap_numerator)] += 1
            compression = left @ right @ left
            if same_view:
                same_intersection_law &= bool(
                    np.array_equal(compression @ compression, 64 * compression)
                    and int(np.trace(compression)) == 64
                )
            else:
                cross_isoclinic_law &= bool(np.array_equal(compression, 4 * left))

    expected_overlaps = Counter(
        {
            ("same_view", 16): 84,
            ("cross_view", 28): 192,
        }
    )

    # Tightness is not confined to the coordinate vertices.  Because P_r(x)
    # is quadratic in x, summing it over any orthonormal basis contracts the
    # basis second moment to the identity.  The rational rotation below is an
    # exact non-coordinate witness, kept in the artifact as a regression test.
    rational_basis = [sp.eye(8)[:, coordinate] for coordinate in range(8)]
    rational_basis[0] = sp.Matrix(
        [sp.Rational(3, 5), sp.Rational(4, 5), 0, 0, 0, 0, 0, 0]
    )
    rational_basis[1] = sp.Matrix(
        [-sp.Rational(4, 5), sp.Rational(3, 5), 0, 0, 0, 0, 0, 0]
    )
    rational_gram = sp.Matrix.hstack(*rational_basis).T * sp.Matrix.hstack(
        *rational_basis
    )

    def exact_p4(family: np.ndarray, state: sp.Matrix) -> sp.Matrix:
        jacobian_twice = sum(
            (
                state[coordinate] * sp.Matrix(family[:, :, coordinate]).T
                for coordinate in range(8)
            ),
            sp.zeros(8, 28),
        )
        return jacobian_twice.T * jacobian_twice

    rational_view_sum = sum(
        (exact_p4(families[0], state) for state in rational_basis), sp.zeros(28)
    )
    rational_nonvertex_tight = bool(
        rational_gram == sp.eye(8) and rational_view_sum == 8 * sp.eye(28)
    )

    # This coloured frame is tight, but it is not spectrally optimal as an
    # uncoloured Grassmannian packing. Same-view pairs intersect, so spectral
    # coherence is one, whereas a generic finite collection of 7-planes in
    # R^28 is pairwise transverse and has coherence strictly below one. Its
    # minimum chordal distance is also strictly below the fusion-frame simplex
    # upper bound. Missing that upper bound does not, without an attainability
    # theorem, prove chordal non-optimality.
    chordal_distance_squared = Fraction(7) - Fraction(7, 4)
    chordal_simplex_bound_squared = Fraction(24 * 7 * (28 - 7), 28 * 23)
    attains_chordal_simplex_bound = bool(
        chordal_distance_squared == chordal_simplex_bound_squared
    )
    passed = bool(
        exact_half_integral
        and len(projectors) == 24
        and projector_law
        and per_view_tight
        and full_tight
        and overlap_numerators == expected_overlaps
        and same_intersection_law
        and cross_isoclinic_law
        and rational_nonvertex_tight
        and not attains_chordal_simplex_bound
    )
    return {
        "ambient_space": "rank-seven orthogonal projectors in R^28",
        "projector_count": len(projectors),
        "integer_scaling": "P4=4P",
        "all_projectors_rank_seven": projector_law,
        "per_view_tight_fusion_frame_identity": "sum_(x in basis(view)) P(x)=2 I_28",
        "per_view_tight_fusion_frame_exact": per_view_tight,
        "all_views_tight_fusion_frame_identity": "sum_(all 24 probes) P=6 I_28",
        "all_views_tight_fusion_frame_exact": full_tight,
        "pair_overlap_distribution": {
            "same_view": {"trace_PQ": "1", "unordered_pairs": 84},
            "cross_view": {"trace_PQ": "7/4", "unordered_pairs": 192},
        },
        "same_view_distinct_subspaces_intersect_in_one_line": same_intersection_law,
        "cross_view_isoclinic_identity": "PQP=(1/4)P",
        "cross_view_squared_cosine": "1/4",
        "cross_view_isoclinic_exact": cross_isoclinic_law,
        "continuous_nonvertex_deformation": {
            "rational_basis_first_two_vectors": [
                [str(value) for value in rational_basis[index]] for index in range(2)
            ],
            "basis_gram_is_identity": rational_gram == sp.eye(8),
            "deformed_view_sum_identity": "sum P(x_j)=2 I_28",
            "deformed_view_sum_exact": rational_nonvertex_tight,
            "general_reason": (
                "P_r(x) is quadratic in x, so every orthonormal basis has the "
                "same second moment and the same projector sum. Tightness has a "
                "continuous orthonormal-basis deformation family; it is not a "
                "vertex-rigidity theorem."
            ),
        },
        "standard_grassmannian_packing_audit": {
            "spectral_coherence_squared": "1",
            "reason_for_spectral_coherence": (
                "distinct same-view subspaces intersect in one line"
            ),
            "minimum_chordal_distance_squared": str(chordal_distance_squared),
            "chordal_simplex_bound_squared": str(chordal_simplex_bound_squared),
            "meets_chordal_simplex_bound": attains_chordal_simplex_bound,
            "spectrally_optimal": False,
            "spectral_nonoptimality_reason": (
                "The configuration has coherence one because some pairs "
                "intersect. Generic finite collections of 7-planes in R^28 "
                "are pairwise transverse and therefore have maximum "
                "coherence strictly below one."
            ),
            "chordal_optimality_status": "open from this audit",
            "interpretation": (
                "The full 24-sensor coloured frame is not equi-isoclinic and "
                "is not spectrally optimal. It does not attain the chordal "
                "simplex bound, but that fact alone does not settle chordal "
                "optimality."
            ),
        },
        "interpretation": (
            "The maintained 24 coordinate sensors form a three-colour tight "
            "fusion-frame configuration in Gr(7,28). This is not the Euclidean "
            "24-cell vertex set, despite the shared D4/triality origin and count 24."
        ),
        "passed": passed,
    }


def run() -> dict[str, object]:
    weights = _weight_certificate()
    projectors = _projector_certificate()
    return {
        "experiment": "exact D4 24-cell and Spin8 triality-projector bridge audit",
        "terminology": {
            "D4_root_system": "24 roots / regular 24-cell in four dimensions",
            "Dynkin_D4": "type of so(8), whose outer automorphism group is triality S3",
            "benchmark_D4": "legacy label for the dihedral group of order eight",
        },
        "minuscule_weight_24cell": weights,
        "coordinate_sensor_projectors": projectors,
        "non_equivalence": (
            "The minuscule weights are 24 points on S^3. The sensor objects are "
            "24 rank-seven subspaces of R^28. The exact bridge is representation-"
            "theoretic; an isometry between these two configurations is neither "
            "defined nor claimed."
        ),
        "scientific_boundary": (
            "This certificate does not imply universal optimality of either "
            "configuration and does not prove global five-query D-optimality."
        ),
        "passed": bool(weights["passed"] and projectors["passed"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run()
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("D4/24-cell bridge certificate failed")


if __name__ == "__main__":
    main()
