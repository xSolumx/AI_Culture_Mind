"""Exact closure of the three triality views of the binary icosahedral group.

The maintained Spin(8) algebra supplies vector, positive-half-spin, and
negative-half-spin actions for one shared Spin(3) subgroup.  Restricting that
subgroup to the exact binary icosahedral generators gives six 8-by-8 matrices
over Q(sqrt(5)).  This module determines the group generated when the three
matrix views are placed in one common carrier space.

The closure has 864,000 elements, so it is not enumerated as dense matrices.
Instead, the matrices preserve two exact 120-point spanning orbits.  Their
faithful permutation action on the disjoint union of those orbits is handled
by deterministic Schreier--Sims arithmetic.  Floating-point values and the
stored artifact do not participate in acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from sympy import QQ, sqrt
from sympy.combinatorics import Permutation, PermutationGroup

from spin8_triality import build_spin8_triality_algebra

FIELD = QQ.algebraic_field(sqrt(5))
GENERATOR_NAMES = (
    "vector_a",
    "vector_b",
    "positive_a",
    "positive_b",
    "negative_a",
    "negative_b",
)

FieldElement = Any
Quaternion = tuple[FieldElement, FieldElement, FieldElement, FieldElement]
Vector4 = tuple[FieldElement, FieldElement, FieldElement, FieldElement]
Matrix4 = tuple[tuple[FieldElement, ...], ...]
Matrix8 = tuple[tuple[FieldElement, ...], ...]


def _field(value: sp.Expr | int) -> FieldElement:
    return FIELD.from_sympy(sp.expand(value))


def _field_string(value: FieldElement) -> str:
    return str(sp.factor(FIELD.to_sympy(value)))


def _identity_quaternion() -> Quaternion:
    return (FIELD.one, FIELD.zero, FIELD.zero, FIELD.zero)


def _minus_identity_quaternion() -> Quaternion:
    return (-FIELD.one, FIELD.zero, FIELD.zero, FIELD.zero)


def quaternion_multiply(left: Quaternion, right: Quaternion) -> Quaternion:
    """Hamilton multiplication over the exact quadratic field."""

    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def quaternion_inverse(value: Quaternion) -> Quaternion:
    return (value[0], -value[1], -value[2], -value[3])


def quaternion_power(value: Quaternion, exponent: int) -> Quaternion:
    result = _identity_quaternion()
    for _ in range(exponent):
        result = quaternion_multiply(result, value)
    return result


def icosahedral_quaternion_generators() -> tuple[Quaternion, Quaternion]:
    """Return exact lifts satisfying a^2=b^3=(ab)^5=-1."""

    root5 = _field(sqrt(5))
    half = FIELD.convert(QQ(1, 2))
    phi = half * (FIELD.one + root5)
    first = (FIELD.zero, FIELD.zero, FIELD.zero, FIELD.one)
    second = (half, half * (phi - FIELD.one), FIELD.zero, -half * phi)
    return first, second


def binary_icosahedral_group() -> tuple[Quaternion, ...]:
    """Enumerate the exact 120 unit icosians in deterministic BFS order."""

    first, second = icosahedral_quaternion_generators()
    steps = (first, second, quaternion_inverse(first), quaternion_inverse(second))
    identity = _identity_quaternion()
    reached = {identity}
    queue = [identity]
    offset = 0
    while offset < len(queue):
        current = queue[offset]
        offset += 1
        for step in steps:
            candidate = quaternion_multiply(current, step)
            if candidate in reached:
                continue
            reached.add(candidate)
            queue.append(candidate)
            if len(queue) > 256:
                raise RuntimeError("binary-icosahedral closure exceeded its cap")
    return tuple(queue)


def _sympy_quaternion(value: Quaternion) -> tuple[sp.Expr, ...]:
    return tuple(FIELD.to_sympy(coordinate) for coordinate in value)


def _matrix_from_sympy(matrix: sp.Matrix) -> Matrix8:
    if matrix.rows != 8 or matrix.cols != 8:
        raise ValueError("expected an 8-by-8 matrix")
    return tuple(
        tuple(_field(matrix[row, column]) for column in range(8))
        for row in range(8)
    )


def _spin8_action_matrices() -> tuple[Matrix8, ...]:
    """Construct the six common-carrier triality matrices exactly."""

    algebra = build_spin8_triality_algebra()
    rounded_gamma = np.rint(algebra.gamma).astype(np.int64)
    if not np.array_equal(algebra.gamma, rounded_gamma):
        raise ValueError("maintained Spin(8) gamma matrices are not integral")
    gamma = tuple(sp.Matrix(matrix) for matrix in rounded_gamma)
    first, second = icosahedral_quaternion_generators()

    def spin_action(value: Quaternion, positive: bool) -> sp.Matrix:
        w, x, y, z = _sympy_quaternion(value)
        full = (
            w * sp.eye(16)
            + x * (-gamma[1] * gamma[2])
            + y * (-gamma[2] * gamma[0])
            + z * (-gamma[0] * gamma[1])
        )
        return full[:8, :8] if positive else full[8:, 8:]

    def vector_action(value: Quaternion) -> sp.Matrix:
        w, x, y, z = _sympy_quaternion(value)
        rotation = sp.Matrix(
            (
                (
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - w * z),
                    2 * (x * z + w * y),
                ),
                (
                    2 * (x * y + w * z),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - w * x),
                ),
                (
                    2 * (x * z - w * y),
                    2 * (y * z + w * x),
                    1 - 2 * (x * x + y * y),
                ),
            )
        )
        return sp.diag(rotation, sp.eye(5))

    matrices = (
        vector_action(first),
        vector_action(second),
        spin_action(first, True),
        spin_action(second, True),
        spin_action(first, False),
        spin_action(second, False),
    )
    return tuple(_matrix_from_sympy(matrix) for matrix in matrices)


def _subblock(matrix: Matrix8, block: int) -> Matrix4:
    offset = 4 * block
    return tuple(
        tuple(matrix[offset + row][offset + column] for column in range(4))
        for row in range(4)
    )


def _identity_matrix(dimension: int) -> tuple[tuple[FieldElement, ...], ...]:
    return tuple(
        tuple(
            FIELD.one if row == column else FIELD.zero
            for column in range(dimension)
        )
        for row in range(dimension)
    )


def _matrix_product(
    left: tuple[tuple[FieldElement, ...], ...],
    right: tuple[tuple[FieldElement, ...], ...],
) -> tuple[tuple[FieldElement, ...], ...]:
    dimension = len(left)
    if dimension != len(right) or any(
        len(row) != dimension for row in (*left, *right)
    ):
        raise ValueError("matrix product requires equal square matrices")
    return tuple(
        tuple(
            sum(
                (
                    left[row][offset] * right[offset][column]
                    for offset in range(dimension)
                ),
                FIELD.zero,
            )
            for column in range(dimension)
        )
        for row in range(dimension)
    )


def _matrix_power(
    matrix: tuple[tuple[FieldElement, ...], ...], exponent: int
) -> tuple[tuple[FieldElement, ...], ...]:
    result = _identity_matrix(len(matrix))
    for _ in range(exponent):
        result = _matrix_product(result, matrix)
    return result


def _transpose(
    matrix: tuple[tuple[FieldElement, ...], ...],
) -> tuple[tuple[FieldElement, ...], ...]:
    return tuple(tuple(value) for value in zip(*matrix, strict=True))


def _negate(
    matrix: tuple[tuple[FieldElement, ...], ...],
) -> tuple[tuple[FieldElement, ...], ...]:
    return tuple(tuple(-value for value in row) for row in matrix)


def _determinant_is_one(matrix: Matrix8) -> bool:
    sympy_matrix = sp.Matrix(
        [[FIELD.to_sympy(value) for value in row] for row in matrix]
    )
    return bool(sp.simplify(sympy_matrix.det() - 1) == 0)


def _apply(matrix: Matrix4, vector: Vector4) -> Vector4:
    return tuple(
        sum(
            (matrix[row][column] * vector[column] for column in range(4)),
            FIELD.zero,
        )
        for row in range(4)
    )


def _block_diagonal(matrix: Matrix8) -> bool:
    return all(
        matrix[row][column] == FIELD.zero
        for row in range(8)
        for column in range(8)
        if (row < 4) != (column < 4)
    )


def _orbit(generators: Sequence[Matrix4]) -> tuple[Vector4, ...]:
    seed = (FIELD.one, FIELD.zero, FIELD.zero, FIELD.zero)
    reached = {seed}
    queue = [seed]
    offset = 0
    while offset < len(queue):
        current = queue[offset]
        offset += 1
        for generator in generators:
            candidate = _apply(generator, current)
            if candidate in reached:
                continue
            reached.add(candidate)
            queue.append(candidate)
            if len(queue) > 512:
                raise RuntimeError("triality block orbit exceeded its cap")
    return tuple(queue)


def _permutations_on_orbit(
    generators: Sequence[Matrix4], orbit: tuple[Vector4, ...]
) -> tuple[Permutation, ...]:
    lookup = {value: index for index, value in enumerate(orbit)}
    return tuple(
        Permutation([lookup[_apply(generator, value)] for value in orbit])
        for generator in generators
    )


def _combined_permutations(
    first: Sequence[Permutation], second: Sequence[Permutation]
) -> tuple[Permutation, ...]:
    if len(first) != len(second):
        raise ValueError("block generator counts disagree")
    first_degree = first[0].size
    second_degree = second[0].size
    return tuple(
        Permutation(
            [left(index) for index in range(first_degree)]
            + [
                first_degree + right(index)
                for index in range(second_degree)
            ]
        )
        for left, right in zip(first, second, strict=True)
    )


def _generated_elements(group: PermutationGroup) -> set[Permutation]:
    return set(group.generate_schreier_sims())


def _permutation_sha256(permutations: Iterable[Permutation], degree: int) -> str:
    payload = [[permutation(index) for index in range(degree)] for permutation in permutations]
    canonical = json.dumps(payload, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _orbit_inner_products(orbit: tuple[Vector4, ...]) -> dict[str, int]:
    distribution: Counter[str] = Counter()
    for left_index, left in enumerate(orbit):
        for right in orbit[left_index + 1 :]:
            value = sum(
                (left[index] * right[index] for index in range(4)), FIELD.zero
            )
            distribution[_field_string(value)] += 1
    return dict(sorted(distribution.items()))


def _quaternion_certificate() -> dict[str, object]:
    group = binary_icosahedral_group()
    lookup = {value: index for index, value in enumerate(group)}
    first, second = icosahedral_quaternion_generators()
    minus_one = _minus_identity_quaternion()
    left_permutations = tuple(
        Permutation(
            [lookup[quaternion_multiply(generator, value)] for value in group]
        )
        for generator in (first, second)
    )
    permutation_group = PermutationGroup(left_permutations)
    checks = {
        "group_order_is_120": len(group) == 120,
        "a_squared_is_minus_one": quaternion_power(first, 2) == minus_one,
        "b_cubed_is_minus_one": quaternion_power(second, 3) == minus_one,
        "ab_fifth_is_minus_one": quaternion_power(
            quaternion_multiply(first, second), 5
        )
        == minus_one,
        "left_regular_image_order_is_120": permutation_group.order() == 120,
        "derived_subgroup_is_whole_group": (
            permutation_group.derived_subgroup().order() == 120
        ),
        "center_order_is_two": permutation_group.center().order() == 2,
    }
    return {
        "order": len(group),
        "structure": "2.A5 = SL(2,5)",
        "presentation": "a^2=b^3=(ab)^5=-1",
        "permutation_generator_sha256": _permutation_sha256(
            left_permutations, len(group)
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }


def certificate() -> dict[str, object]:
    """Return the exact triality-closure classification certificate."""

    quaternion_report = _quaternion_certificate()
    matrices = _spin8_action_matrices()
    blocks = tuple(
        tuple(_subblock(matrix, block) for matrix in matrices) for block in range(2)
    )
    orbits = tuple(_orbit(block) for block in blocks)
    block_permutations = tuple(
        _permutations_on_orbit(block, orbit)
        for block, orbit in zip(blocks, orbits, strict=True)
    )
    block_groups = tuple(PermutationGroup(permutations) for permutations in block_permutations)
    combined_permutations = _combined_permutations(*block_permutations)
    combined_group = PermutationGroup(combined_permutations)

    first_lookup = {value: index for index, value in enumerate(orbits[0])}
    second_lookup = {value: index for index, value in enumerate(orbits[1])}
    degree = len(orbits[0]) + len(orbits[1])
    first_block_sign = Permutation(
        [
            first_lookup[tuple(-coordinate for coordinate in value)]
            for value in orbits[0]
        ]
        + list(range(len(orbits[0]), degree))
    )
    second_block_sign = Permutation(
        list(range(len(orbits[0])))
        + [
            len(orbits[0])
            + second_lookup[tuple(-coordinate for coordinate in value)]
            for value in orbits[1]
        ]
    )
    identity_permutation = Permutation(list(range(degree)))
    expected_center = {
        identity_permutation,
        first_block_sign,
        second_block_sign,
        first_block_sign * second_block_sign,
    }
    actual_center = _generated_elements(combined_group.center())

    quaternion_vertices = set(binary_icosahedral_group())
    # This fixed coordinate exchange matches the maintained Clifford convention.
    mapped_vertices = {
        (value[0], value[1], value[3], value[2]) for value in quaternion_vertices
    }
    orbit_matches = [set(orbit) == mapped_vertices for orbit in orbits]
    orbit_spans = [
        all(
            tuple(
                FIELD.one if index == coordinate else FIELD.zero
                for index in range(4)
            )
            in set(orbit)
            for coordinate in range(4)
        )
        for orbit in orbits
    ]

    positive_groups = tuple(
        PermutationGroup(permutations[2:4]) for permutations in block_permutations
    )
    negative_groups = tuple(
        PermutationGroup(permutations[4:6]) for permutations in block_permutations
    )
    positive_negative_groups = tuple(
        PermutationGroup(permutations[2:6]) for permutations in block_permutations
    )
    positive_vector_groups = tuple(
        PermutationGroup(permutations[:4]) for permutations in block_permutations
    )
    first_positive_elements = _generated_elements(positive_groups[0])
    first_negative_elements = _generated_elements(negative_groups[0])
    second_positive_elements = _generated_elements(positive_groups[1])
    second_negative_elements = _generated_elements(negative_groups[1])
    first_commuting_factor = block_groups[0].centralizer(positive_groups[0])
    first_commuting_factor_elements = _generated_elements(first_commuting_factor)
    first_central_product = PermutationGroup(
        [*positive_groups[0].generators, *first_commuting_factor.generators]
    )

    first_block_checks = {
        "block_group_order_is_7200": block_groups[0].order() == 7200,
        "positive_image_order_is_120": positive_groups[0].order() == 120,
        "negative_image_order_is_120": negative_groups[0].order() == 120,
        "positive_and_negative_images_coincide": (
            first_positive_elements == first_negative_elements
        ),
        "commuting_factor_order_is_120": first_commuting_factor.order() == 120,
        "commuting_factor_is_perfect": (
            first_commuting_factor.derived_subgroup().order() == 120
        ),
        "commuting_factor_center_order_is_two": (
            first_commuting_factor.center().order() == 2
        ),
        "commuting_factor_intersection_order_is_two": len(
            first_positive_elements.intersection(first_commuting_factor_elements)
        )
        == 2,
        "central_product_order_is_7200": first_central_product.order() == 7200,
        "positive_and_vector_images_generate_first_block": (
            positive_vector_groups[0].order() == block_groups[0].order()
        ),
        "negative_image_adds_no_new_first_block_elements": (
            positive_negative_groups[0].order() == positive_groups[0].order()
        ),
        "orbit_is_exact_binary_icosahedral_root_set": orbit_matches[0],
        "orbit_spans_first_block": orbit_spans[0],
        "center_order_is_two": block_groups[0].center().order() == 2,
        "derived_subgroup_is_whole_group": (
            block_groups[0].derived_subgroup().order() == 7200
        ),
    }
    second_block_checks = {
        "block_group_order_is_120": block_groups[1].order() == 120,
        "positive_image_order_is_120": positive_groups[1].order() == 120,
        "negative_image_order_is_120": negative_groups[1].order() == 120,
        "positive_and_negative_images_coincide": (
            second_positive_elements == second_negative_elements
        ),
        "orbit_is_exact_binary_icosahedral_root_set": orbit_matches[1],
        "orbit_spans_second_block": orbit_spans[1],
        "center_order_is_two": block_groups[1].center().order() == 2,
        "derived_subgroup_is_whole_group": (
            block_groups[1].derived_subgroup().order() == 120
        ),
    }

    projected_product_order = block_groups[0].order() * block_groups[1].order()
    full_checks = {
        "all_six_generators_are_block_diagonal": all(
            _block_diagonal(matrix) for matrix in matrices
        ),
        "all_six_generators_are_exactly_orthogonal": all(
            _matrix_product(_transpose(matrix), matrix) == _identity_matrix(8)
            for matrix in matrices
        ),
        "all_six_generators_have_determinant_one": all(
            _determinant_is_one(matrix) for matrix in matrices
        ),
        "vector_generators_satisfy_a5_relations": (
            _matrix_power(matrices[0], 2) == _identity_matrix(8)
            and _matrix_power(matrices[1], 3) == _identity_matrix(8)
            and _matrix_power(_matrix_product(matrices[0], matrices[1]), 5)
            == _identity_matrix(8)
        ),
        "positive_generators_satisfy_binary_relations": (
            _matrix_power(matrices[2], 2) == _negate(_identity_matrix(8))
            and _matrix_power(matrices[3], 3) == _negate(_identity_matrix(8))
            and _matrix_power(_matrix_product(matrices[2], matrices[3]), 5)
            == _negate(_identity_matrix(8))
        ),
        "negative_generators_satisfy_binary_relations": (
            _matrix_power(matrices[4], 2) == _negate(_identity_matrix(8))
            and _matrix_power(matrices[5], 3) == _negate(_identity_matrix(8))
            and _matrix_power(_matrix_product(matrices[4], matrices[5]), 5)
            == _negate(_identity_matrix(8))
        ),
        "two_orbits_have_size_120": [len(orbit) for orbit in orbits] == [120, 120],
        "disjoint_orbit_action_is_faithful": all(orbit_spans),
        "full_group_order_is_864000": combined_group.order() == 864000,
        "order_equals_product_of_block_projection_orders": (
            combined_group.order() == projected_product_order
        ),
        "center_order_is_four": combined_group.center().order() == 4,
        "center_is_independent_block_signs": actual_center == expected_center,
        "derived_subgroup_is_whole_group": (
            combined_group.derived_subgroup().order() == combined_group.order()
        ),
        "group_is_nonsolvable": not bool(combined_group.is_solvable),
    }

    first_report = {
        "orbit_order": len(orbits[0]),
        "orbit_inner_product_distribution": _orbit_inner_products(orbits[0]),
        "permutation_group_order": block_groups[0].order(),
        "positive_image_order": positive_groups[0].order(),
        "negative_image_order": negative_groups[0].order(),
        "commuting_binary_icosahedral_factor_order": (
            first_commuting_factor.order()
        ),
        "central_factor_intersection_order": len(
            first_positive_elements.intersection(first_commuting_factor_elements)
        ),
        "center_order": block_groups[0].center().order(),
        "derived_subgroup_order": block_groups[0].derived_subgroup().order(),
        "structure": "(2.A5 x 2.A5)/C2_diagonal",
        "geometric_name": "orientation-preserving H4 / 600-cell rotation group",
        "generator_permutation_sha256": _permutation_sha256(
            block_permutations[0], len(orbits[0])
        ),
        "checks": first_block_checks,
        "passed": all(first_block_checks.values()),
    }
    second_report = {
        "orbit_order": len(orbits[1]),
        "orbit_inner_product_distribution": _orbit_inner_products(orbits[1]),
        "permutation_group_order": block_groups[1].order(),
        "positive_image_order": positive_groups[1].order(),
        "negative_image_order": negative_groups[1].order(),
        "center_order": block_groups[1].center().order(),
        "derived_subgroup_order": block_groups[1].derived_subgroup().order(),
        "structure": "2.A5 = SL(2,5)",
        "generator_permutation_sha256": _permutation_sha256(
            block_permutations[1], len(orbits[1])
        ),
        "checks": second_block_checks,
        "passed": all(second_block_checks.values()),
    }
    full_report = {
        "faithful_permutation_degree": combined_group.degree,
        "orbit_orders": sorted(len(orbit) for orbit in combined_group.orbits()),
        "generator_orders": {
            name: int(permutation.order())
            for name, permutation in zip(
                GENERATOR_NAMES, combined_permutations, strict=True
            )
        },
        "order": combined_group.order(),
        "center_order": combined_group.center().order(),
        "center_structure": "C2 x C2 independent signs on the two H4 blocks",
        "center_block_signatures": ["++", "+-", "-+", "--"],
        "derived_subgroup_order": combined_group.derived_subgroup().order(),
        "block_projection_orders": [group.order() for group in block_groups],
        "structure": "((2.A5 x 2.A5)/C2_diagonal) x 2.A5",
        "generator_permutation_sha256": _permutation_sha256(
            combined_permutations, combined_group.degree
        ),
        "checks": full_checks,
        "passed": all(full_checks.values()),
    }
    passed = bool(
        quaternion_report["passed"]
        and first_report["passed"]
        and second_report["passed"]
        and full_report["passed"]
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "experiment": "exact triality closure of the binary icosahedral embedding",
        "coefficient_field": "Q(sqrt(5))",
        "generator_names": list(GENERATOR_NAMES),
        "binary_icosahedral_input": quaternion_report,
        "first_four_dimensional_block": first_report,
        "second_four_dimensional_block": second_report,
        "full_triality_closure": full_report,
        "claim_scope": {
            "proved_exactly": [
                "the maintained common-carrier triality generators preserve two four-dimensional blocks",
                "both spanning 120-point orbits are the binary-icosahedral H4 root set",
                "the first block image is the central product of two binary icosahedral groups",
                "the second block image is one binary icosahedral group",
                "the full generated group is their direct product of order 864000",
            ],
            "external_nomenclature": [
                "the 120-point H4 root system is the 600-cell vertex set",
                "its orientation-preserving symmetry group has structure (2I x 2I)/C2_diagonal",
            ],
            "not_claimed": [
                "discovery of a previously unknown abstract finite group",
                "classification of all 2.A5 embeddings in Spin(8)",
                "an irreducible eight-dimensional closure",
                "an ML-quality or kernel-speed advantage",
            ],
        },
        "passed": passed,
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
