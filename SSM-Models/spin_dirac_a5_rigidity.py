"""Exact tangent-rigidity certificates for the icosahedral Spin ladder.

The vector generators lie in ``SO(3, Q(sqrt(5)))`` and satisfy the exact
``(2, 3, 5)`` presentation of A5.  Their lifts to Spin(n) satisfy the same
relations with the central target ``-1``.  Because Spin(n) -> SO(n) is a local
covering, the relation Jacobian and infinitesimal conjugacy ranks can be
computed in the vector quotient without losing spinorial tangent data.

For each requested n, this module constructs the exact relation Jacobian over
Q(sqrt(5)), the infinitesimal conjugacy map, and finite-field pivot-minor
certificates.  Exact composition ``J C = 0`` supplies the rank upper bound;
nonzero modular minors supply the matching lower bound.  Equality proves that
every infinitesimal deformation is conjugacy at this embedding.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spin_dirac_a5_ladder import LADDER_DIMENSIONS
from sympy import QQ, sqrt, sqrt_mod
from sympy.polys.matrices import DomainMatrix

FIELD = QQ.algebraic_field(sqrt(5))
CERTIFICATE_PRIMES = (1_000_039, 1_000_081, 1_000_099)


@dataclass(frozen=True)
class ExactA5TangentSystem:
    """Exact generators and linearized maps at one embedded A5 action."""

    dimension: int
    a: DomainMatrix
    b: DomainMatrix
    relation_jacobian: DomainMatrix
    conjugacy_map: DomainMatrix
    pairs: tuple[tuple[int, int], ...]


def _matrix(rows: list[list[Any]]) -> DomainMatrix:
    return DomainMatrix(
        rows,
        (len(rows), len(rows[0]) if rows else 0),
        FIELD,
        fmt="dense",
    )


def _zero(dimension: int) -> DomainMatrix:
    return _matrix([[FIELD.zero for _ in range(dimension)] for _ in range(dimension)])


def _identity(dimension: int) -> DomainMatrix:
    return _matrix(
        [
            [FIELD.one if row == column else FIELD.zero for column in range(dimension)]
            for row in range(dimension)
        ]
    )


def _matrix_power(matrix: DomainMatrix, exponent: int) -> DomainMatrix:
    result = _identity(matrix.shape[0])
    for _ in range(exponent):
        result = result.matmul(matrix)
    return result


def _is_zero(matrix: DomainMatrix) -> bool:
    return all(value == FIELD.zero for value in matrix.to_list_flat())


def generator_pairs(dimension: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (left, right)
        for left in range(dimension)
        for right in range(left + 1, dimension)
    )


def skew_basis(dimension: int) -> tuple[DomainMatrix, ...]:
    matrices = []
    for left, right in generator_pairs(dimension):
        rows = [[FIELD.zero for _ in range(dimension)] for _ in range(dimension)]
        rows[left][right] = FIELD.one
        rows[right][left] = -FIELD.one
        matrices.append(_matrix(rows))
    return tuple(matrices)


def exact_icosahedral_generators(dimension: int) -> tuple[DomainMatrix, DomainMatrix]:
    """Return the fixed A5 vector generators embedded in SO(n)."""

    if dimension < 3:
        raise ValueError("the icosahedral vector action requires dimension at least 3")
    root5 = FIELD.from_sympy(sqrt(5))
    quarter = FIELD.convert(QQ(1, 4))
    half = FIELD.convert(QQ(1, 2))
    a3 = [
        [-FIELD.one, FIELD.zero, FIELD.zero],
        [FIELD.zero, -FIELD.one, FIELD.zero],
        [FIELD.zero, FIELD.zero, FIELD.one],
    ]
    b3 = [
        [quarter * (1 - root5), quarter * (1 + root5), -half],
        [-quarter * (1 + root5), -half, quarter * (1 - root5)],
        [-half, quarter * (-1 + root5), quarter * (1 + root5)],
    ]

    def embed(block: list[list[Any]]) -> DomainMatrix:
        rows = [
            [FIELD.one if row == column else FIELD.zero for column in range(dimension)]
            for row in range(dimension)
        ]
        for row in range(3):
            for column in range(3):
                rows[row][column] = block[row][column]
        return _matrix(rows)

    return embed(a3), embed(b3)


def exact_quaternion_multiply(
    left: tuple[Any, Any, Any, Any],
    right: tuple[Any, Any, Any, Any],
) -> tuple[Any, Any, Any, Any]:
    """Hamilton product over Q(sqrt(5))."""

    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def exact_quaternion_power(
    quaternion: tuple[Any, Any, Any, Any], exponent: int
) -> tuple[Any, Any, Any, Any]:
    value = (FIELD.one, FIELD.zero, FIELD.zero, FIELD.zero)
    for _ in range(exponent):
        value = exact_quaternion_multiply(value, quaternion)
    return value


def exact_icosahedral_quaternion_generators() -> tuple[
    tuple[Any, Any, Any, Any], tuple[Any, Any, Any, Any]
]:
    root5 = FIELD.from_sympy(sqrt(5))
    half = FIELD.convert(QQ(1, 2))
    phi = half * (FIELD.one + root5)
    a = (FIELD.zero, FIELD.zero, FIELD.zero, FIELD.one)
    b = (half, half * (phi - FIELD.one), FIELD.zero, -half * phi)
    return a, b


def exact_quaternion_rotation(quaternion: tuple[Any, Any, Any, Any]) -> DomainMatrix:
    w, x, y, z = quaternion
    two = FIELD.convert(QQ(2))
    return _matrix(
        [
            [
                FIELD.one - two * (y * y + z * z),
                two * (x * y - w * z),
                two * (x * z + w * y),
            ],
            [
                two * (x * y + w * z),
                FIELD.one - two * (x * x + z * z),
                two * (y * z - w * x),
            ],
            [
                two * (x * z - w * y),
                two * (y * z + w * x),
                FIELD.one - two * (x * x + y * y),
            ],
        ]
    )


def enumerate_exact_binary_icosahedral_group() -> tuple[tuple[Any, Any, Any, Any], ...]:
    """Return the deterministic exact BFS order of the generated group."""

    a, b = exact_icosahedral_quaternion_generators()
    identity = (FIELD.one, FIELD.zero, FIELD.zero, FIELD.zero)
    inverses = tuple(
        (quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3])
        for quaternion in (a, b)
    )
    steps = (a, b, *inverses)
    reached = {identity}
    queue = [identity]
    offset = 0
    while offset < len(queue):
        current = queue[offset]
        offset += 1
        for step in steps:
            candidate = exact_quaternion_multiply(current, step)
            if candidate not in reached:
                reached.add(candidate)
                queue.append(candidate)
                if len(reached) > 256:
                    raise RuntimeError(
                        "exact quaternion enumeration exceeded its safety cap"
                    )
    return tuple(queue)


def exact_binary_icosahedral_certificate() -> dict[str, object]:
    """Enumerate 2.A5 and its A5 projective quotient with exact field elements."""

    group = enumerate_exact_binary_icosahedral_group()
    reached = set(group)
    a, b = exact_icosahedral_quaternion_generators()
    minus_one = (-FIELD.one, FIELD.zero, FIELD.zero, FIELD.zero)
    projective_classes = {
        frozenset(
            (
                quaternion,
                tuple(-coordinate for coordinate in quaternion),
            )
        )
        for quaternion in reached
    }
    vector_a, vector_b = exact_icosahedral_generators(3)
    checks = {
        "a_squared_to_minus_one": exact_quaternion_power(a, 2) == minus_one,
        "b_cubed_to_minus_one": exact_quaternion_power(b, 3) == minus_one,
        "ab_fifth_to_minus_one": exact_quaternion_power(
            exact_quaternion_multiply(a, b), 5
        )
        == minus_one,
        "minus_one_is_present": minus_one in reached,
        "a_vector_projection_matches": _is_zero(
            exact_quaternion_rotation(a) - vector_a
        ),
        "b_vector_projection_matches": _is_zero(
            exact_quaternion_rotation(b) - vector_b
        ),
    }
    passed = (
        all(checks.values()) and len(reached) == 120 and len(projective_classes) == 60
    )
    return {
        "field": "Q(sqrt(5))",
        "binary_group_order": len(reached),
        "projective_group_order": len(projective_classes),
        "checks": checks,
        "passed": bool(passed),
    }


def _skew_coordinates(
    matrix: DomainMatrix, pairs: tuple[tuple[int, int], ...]
) -> list[Any]:
    if not _is_zero(matrix + matrix.transpose()):
        raise AssertionError("a relation differential left the skew tangent space")
    rows = matrix.to_list()
    return [rows[left][right] for left, right in pairs]


def relation_differentials(
    a: DomainMatrix,
    b: DomainMatrix,
    x: DomainMatrix,
    y: DomainMatrix,
    *,
    b_squared: DomainMatrix | None = None,
    c: DomainMatrix | None = None,
    c_powers: tuple[DomainMatrix, ...] | None = None,
) -> tuple[DomainMatrix, DomainMatrix, DomainMatrix]:
    """Differentiate A^2, B^3, and (AB)^5 under A exp(tX), B exp(tY)."""

    dimension = a.shape[0]
    b_squared = _matrix_power(b, 2) if b_squared is None else b_squared
    c = a.matmul(b) if c is None else c
    c_powers = (
        tuple(_matrix_power(c, exponent) for exponent in range(5))
        if c_powers is None
        else c_powers
    )
    da = a.matmul(x)
    db = b.matmul(y)
    da2 = da.matmul(a) + a.matmul(da)
    db3 = db.matmul(b_squared) + b.matmul(db).matmul(b) + b_squared.matmul(db)
    dc = da.matmul(b) + a.matmul(db)
    dc5 = _zero(dimension)
    for offset in range(5):
        dc5 += c_powers[offset].matmul(dc).matmul(c_powers[4 - offset])
    return da2, db3, dc5


def build_exact_tangent_system(dimension: int) -> ExactA5TangentSystem:
    """Construct exact relation and conjugacy matrices over Q(sqrt(5))."""

    a, b = exact_icosahedral_generators(dimension)
    pairs = generator_pairs(dimension)
    basis = skew_basis(dimension)
    zero = _zero(dimension)
    b_squared = _matrix_power(b, 2)
    c = a.matmul(b)
    c_powers = tuple(_matrix_power(c, exponent) for exponent in range(5))
    relation_columns: list[list[Any]] = []
    for variable in range(2 * len(pairs)):
        x = basis[variable] if variable < len(pairs) else zero
        y = basis[variable - len(pairs)] if variable >= len(pairs) else zero
        differentials = relation_differentials(
            a,
            b,
            x,
            y,
            b_squared=b_squared,
            c=c,
            c_powers=c_powers,
        )
        relation_columns.append(
            [
                coordinate
                for differential in differentials
                for coordinate in _skew_coordinates(differential, pairs)
            ]
        )
    relation_jacobian = _matrix(
        [
            [relation_columns[column][row] for column in range(2 * len(pairs))]
            for row in range(3 * len(pairs))
        ]
    )

    conjugacy_columns: list[list[Any]] = []
    for z in basis:
        x = a.transpose().matmul(z).matmul(a) - z
        y = b.transpose().matmul(z).matmul(b) - z
        conjugacy_columns.append(
            _skew_coordinates(x, pairs) + _skew_coordinates(y, pairs)
        )
    conjugacy_map = _matrix(
        [
            [conjugacy_columns[column][row] for column in range(len(pairs))]
            for row in range(2 * len(pairs))
        ]
    )
    return ExactA5TangentSystem(
        dimension=dimension,
        a=a,
        b=b,
        relation_jacobian=relation_jacobian,
        conjugacy_map=conjugacy_map,
        pairs=pairs,
    )


def _rational_mod(value: Any, prime: int) -> int:
    numerator = int(value.numerator) % prime
    denominator = int(value.denominator) % prime
    return numerator * pow(denominator, -1, prime) % prime


def _field_element_mod(value: Any, prime: int, root5: int) -> int:
    coefficients = value.rep
    if not coefficients:
        return 0
    if len(coefficients) == 1:
        return _rational_mod(coefficients[0], prime)
    if len(coefficients) == 2:
        linear, constant = coefficients
        return (
            _rational_mod(linear, prime) * root5 + _rational_mod(constant, prime)
        ) % prime
    raise ValueError("unexpected algebraic-field representative")


def modular_matrix(matrix: DomainMatrix, prime: int, root5: int) -> list[list[int]]:
    return [
        [_field_element_mod(value, prime, root5) for value in row]
        for row in matrix.to_list()
    ]


def _modular_determinant(matrix: list[list[int]], prime: int) -> int:
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("determinant requires a square matrix")
    work = [row.copy() for row in matrix]
    determinant = 1
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if work[row][column] % prime),
            None,
        )
        if pivot is None:
            return 0
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            determinant = -determinant
        pivot_value = work[column][column] % prime
        determinant = determinant * pivot_value % prime
        inverse = pow(pivot_value, -1, prime)
        for row in range(column + 1, size):
            factor = work[row][column] * inverse % prime
            if factor:
                work[row] = [
                    (left - factor * right) % prime
                    for left, right in zip(work[row], work[column], strict=True)
                ]
    return determinant % prime


def modular_pivot_certificate(matrix: DomainMatrix, prime: int) -> dict[str, object]:
    """Return a nonzero pivot minor after specializing sqrt(5) in F_p."""

    root5 = sqrt_mod(5, prime)
    if root5 is None:
        raise ValueError(f"5 is not a square modulo {prime}")
    work = modular_matrix(matrix, prime, int(root5))
    original = [row.copy() for row in work]
    row_labels = list(range(len(work)))
    columns = len(work[0]) if work else 0
    pivot_rows: list[int] = []
    pivot_columns: list[int] = []
    rank = 0
    for column in range(columns):
        pivot = next(
            (row for row in range(rank, len(work)) if work[row][column] % prime),
            None,
        )
        if pivot is None:
            continue
        if pivot != rank:
            work[rank], work[pivot] = work[pivot], work[rank]
            row_labels[rank], row_labels[pivot] = row_labels[pivot], row_labels[rank]
        pivot_rows.append(row_labels[rank])
        pivot_columns.append(column)
        inverse = pow(work[rank][column] % prime, -1, prime)
        work[rank] = [value * inverse % prime for value in work[rank]]
        for row in range(len(work)):
            if row == rank:
                continue
            factor = work[row][column] % prime
            if factor:
                work[row] = [
                    (left - factor * right) % prime
                    for left, right in zip(work[row], work[rank], strict=True)
                ]
        rank += 1
        if rank == len(work):
            break
    minor = [[original[row][column] for column in pivot_columns] for row in pivot_rows]
    determinant = _modular_determinant(minor, prime)
    if rank and determinant == 0:
        raise AssertionError("pivot elimination produced a singular minor")
    return {
        "prime": prime,
        "sqrt5_root": int(root5),
        "rank": rank,
        "pivot_rows": pivot_rows,
        "pivot_columns": pivot_columns,
        "pivot_minor_determinant_mod_prime": determinant,
    }


def stage_diagnostics(dimension: int) -> dict[str, object]:
    """Certify infinitesimal rigidity for one Spin(n) embedding."""

    system = build_exact_tangent_system(dimension)
    a, b = system.a, system.b
    c = a.matmul(b)
    identity = _identity(dimension)
    lie_dimension = len(system.pairs)
    centralizer_dimension = (dimension - 3) * (dimension - 4) // 2
    orbit_dimension = lie_dimension - centralizer_dimension
    expected_relation_rank = 2 * lie_dimension - orbit_dimension
    exact_checks = {
        "a_squared": _is_zero(_matrix_power(a, 2) - identity),
        "b_cubed": _is_zero(_matrix_power(b, 3) - identity),
        "ab_fifth": _is_zero(_matrix_power(c, 5) - identity),
        "a_orthogonal": _is_zero(a.transpose().matmul(a) - identity),
        "b_orthogonal": _is_zero(b.transpose().matmul(b) - identity),
        "relation_after_conjugacy_zero": _is_zero(
            system.relation_jacobian.matmul(system.conjugacy_map)
        ),
    }
    untouched_columns = [
        index
        for index, (left, right) in enumerate(system.pairs)
        if left >= 3 and right >= 3
    ]
    conjugacy_rows = system.conjugacy_map.to_list()
    exact_checks["explicit_so_n_minus_3_kernel"] = all(
        conjugacy_rows[row][column] == FIELD.zero
        for column in untouched_columns
        for row in range(2 * lie_dimension)
    )

    relation_certificates = [
        modular_pivot_certificate(system.relation_jacobian, prime)
        for prime in CERTIFICATE_PRIMES
    ]
    conjugacy_certificates = [
        modular_pivot_certificate(system.conjugacy_map, prime)
        for prime in CERTIFICATE_PRIMES
    ]
    relation_ranks = [certificate["rank"] for certificate in relation_certificates]
    conjugacy_ranks = [certificate["rank"] for certificate in conjugacy_certificates]
    passed = (
        all(exact_checks.values())
        and len(untouched_columns) == centralizer_dimension
        and relation_ranks == [expected_relation_rank] * len(CERTIFICATE_PRIMES)
        and conjugacy_ranks == [orbit_dimension] * len(CERTIFICATE_PRIMES)
    )
    return {
        "dimension": dimension,
        "field": "Q(sqrt(5))",
        "lie_dimension": lie_dimension,
        "relation_jacobian_shape": list(system.relation_jacobian.shape),
        "conjugacy_map_shape": list(system.conjugacy_map.shape),
        "expected_centralizer_dimension": centralizer_dimension,
        "explicit_centralizer_basis_dimension": len(untouched_columns),
        "certified_conjugacy_rank": orbit_dimension,
        "certified_relation_rank": expected_relation_rank,
        "certified_relation_kernel_dimension": orbit_dimension,
        "h1_dimension": 0,
        "exact_checks": exact_checks,
        "relation_rank_certificates": relation_certificates,
        "conjugacy_rank_certificates": conjugacy_certificates,
        "rank_argument": (
            "im(C) subset ker(J) gives the exact upper bound; each nonzero "
            "finite-field pivot minor gives the matching lower bound"
        ),
        "passed": bool(passed),
    }


def diagnostics(
    dimensions: tuple[int, ...] = LADDER_DIMENSIONS,
) -> dict[str, object]:
    binary_group = exact_binary_icosahedral_certificate()
    stages = {str(dimension): stage_diagnostics(dimension) for dimension in dimensions}
    return {
        "schema_version": 1,
        "experiment": "exact A5/2.A5 representation-tangent rigidity along the Spin ladder",
        "ladder": list(dimensions),
        "coefficient_field": "Q(sqrt(5))",
        "certificate_primes": list(CERTIFICATE_PRIMES),
        "exact_binary_icosahedral_group": binary_group,
        "claim_scope": {
            "computer_assisted_exact": [
                "the fixed A5 vector generators satisfy the (2,3,5) relations",
                "the relation Jacobian annihilates infinitesimal conjugacy",
                "the centralizer is exactly so(n-3) on every listed rung",
                "the relation kernel equals the conjugacy image on every listed rung",
                "H1 vanishes for the listed fixed embeddings",
            ],
            "bridge_to_spin": (
                "the lifted 2.A5 relations have central target -1; the local covering "
                "Spin(n)->SO(n) identifies their tangent calculations"
            ),
            "not_claimed": [
                "classification of every 2.A5 embedding in Spin(n)",
                "global uniqueness up to Spin(n) conjugacy",
                "a derived representation-stack or obstruction theorem",
                "an ML or SSM advantage",
            ],
        },
        "stages": stages,
        "passed": binary_group["passed"]
        and all(stage["passed"] for stage in stages.values()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        choices=LADDER_DIMENSIONS,
        default=list(LADDER_DIMENSIONS),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = diagnostics(tuple(args.dimensions))
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
