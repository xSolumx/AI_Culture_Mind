"""Exact local-Hessian certificate at the Spin(9) three-spinor optimum.

The algebraic point contains several nested square roots.  Converting them to
one primitive element is needlessly expensive, so this module evaluates the
Hessian in the explicit quadratic tower Q(sqrt(241), d, b, z).  All arithmetic
is exact over fractions.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import TypeAlias

import sympy as sp

from spin9_dirac_clifford import build_spin9_clifford_system

Element: TypeAlias = tuple[Fraction, ...]
Matrix: TypeAlias = list[list[Element]]

# Extension order: q=sqrt(241), d, b, z.  Each square belongs to the previous
# subfield, which makes recursive conjugation an exact and efficient inverse.
SQUARES: dict[int, Element] = {
    1: (Fraction(241),),
    2: (Fraction(7, 48), Fraction(1, 48)),
    3: (
        Fraction(41, 48),
        Fraction(-1, 48),
        Fraction(0),
        Fraction(0),
    ),
    4: (
        Fraction(-247, 32),
        Fraction(17, 32),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
    ),
}


def _zero(length: int = 16) -> Element:
    return (Fraction(0),) * length


def _rational(value: int | Fraction) -> Element:
    return (Fraction(value),) + _zero(15)


def _generator(index: int) -> Element:
    values = [Fraction(0)] * 16
    values[index] = Fraction(1)
    return tuple(values)


def _add(left: Element, right: Element) -> Element:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _neg(value: Element) -> Element:
    return tuple(-entry for entry in value)


def _sub(left: Element, right: Element) -> Element:
    return _add(left, _neg(right))


@cache
def _multiply(left: Element, right: Element) -> Element:
    length = len(left)
    if length == 1:
        return (left[0] * right[0],)
    half = length // 2
    level = length.bit_length() - 1
    left_low, left_high = left[:half], left[half:]
    right_low, right_high = right[:half], right[half:]
    low = _add(
        _multiply(left_low, right_low),
        _multiply(_multiply(left_high, right_high), SQUARES[level]),
    )
    high = _add(_multiply(left_low, right_high), _multiply(left_high, right_low))
    return low + high


@cache
def _inverse(value: Element) -> Element:
    length = len(value)
    if length == 1:
        if value[0] == 0:
            raise ZeroDivisionError("zero field element")
        return (1 / value[0],)
    half = length // 2
    level = length.bit_length() - 1
    low, high = value[:half], value[half:]
    denominator = _sub(
        _multiply(low, low), _multiply(_multiply(high, high), SQUARES[level])
    )
    denominator_inverse = _inverse(denominator)
    return _multiply(low, denominator_inverse) + _neg(
        _multiply(high, denominator_inverse)
    )


def _divide(left: Element, right: Element) -> Element:
    return _multiply(left, _inverse(right))


def _scale(value: Element, scalar: int | Fraction) -> Element:
    factor = Fraction(scalar)
    return tuple(factor * entry for entry in value)


def _sum_elements(values: object, length: int = 16) -> Element:
    total = _zero(length)
    for value in values:  # type: ignore[union-attr]
        total = _add(total, value)
    return total


def _zeros(rows: int, columns: int) -> Matrix:
    return [[_zero() for _ in range(columns)] for _ in range(rows)]


def _identity(size: int) -> Matrix:
    result = _zeros(size, size)
    for index in range(size):
        result[index][index] = _rational(1)
    return result


def _matrix_add(left: Matrix, right: Matrix) -> Matrix:
    return [
        [_add(a, b) for a, b in zip(row_left, row_right, strict=True)]
        for row_left, row_right in zip(left, right, strict=True)
    ]


def _matrix_sub(left: Matrix, right: Matrix) -> Matrix:
    return [
        [_sub(a, b) for a, b in zip(row_left, row_right, strict=True)]
        for row_left, row_right in zip(left, right, strict=True)
    ]


def _transpose(matrix: Matrix) -> Matrix:
    return [list(row) for row in zip(*matrix, strict=True)]


def _matrix_multiply(left: Matrix, right: Matrix) -> Matrix:
    right_transpose = _transpose(right)
    return [
        [
            _sum_elements(
                (_multiply(a, b) for a, b in zip(row, column, strict=True)),
            )
            for column in right_transpose
        ]
        for row in left
    ]


def _trace(matrix: Matrix) -> Element:
    return _sum_elements(matrix[index][index] for index in range(len(matrix)))


def _frobenius_inner(left: Matrix, right: Matrix) -> Element:
    return _sum_elements(
        _multiply(a, b)
        for row_left, row_right in zip(left, right, strict=True)
        for a, b in zip(row_left, row_right, strict=True)
    )


def _matrix_inverse(matrix: Matrix) -> Matrix:
    size = len(matrix)
    augmented = [row[:] + identity for row, identity in zip(matrix, _identity(size))]
    for column in range(size):
        pivot = next(
            row for row in range(column, size) if augmented[row][column] != _zero()
        )
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_inverse = _inverse(augmented[column][column])
        augmented[column] = [
            _multiply(entry, pivot_inverse) for entry in augmented[column]
        ]
        for row in range(size):
            if row == column or augmented[row][column] == _zero():
                continue
            factor = augmented[row][column]
            augmented[row] = [
                _sub(entry, _multiply(factor, pivot_entry))
                for entry, pivot_entry in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return [row[size:] for row in augmented]


def _linear_solve(matrix: Matrix, right: list[Element]) -> list[Element]:
    """Solve a nonsingular exact linear system without constructing its inverse."""

    size = len(matrix)
    augmented = [row[:] + [value] for row, value in zip(matrix, right, strict=True)]
    for column in range(size):
        pivot = next(
            row for row in range(column, size) if augmented[row][column] != _zero()
        )
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_inverse = _inverse(augmented[column][column])
        augmented[column] = [
            _multiply(entry, pivot_inverse) for entry in augmented[column]
        ]
        for row in range(size):
            if row == column or augmented[row][column] == _zero():
                continue
            factor = augmented[row][column]
            augmented[row] = [
                _sub(entry, _multiply(factor, pivot_entry))
                for entry, pivot_entry in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return [row[-1] for row in augmented]


def _linear_solve_many(
    matrix: Matrix, rights: list[list[Element]]
) -> list[list[Element]]:
    """Solve several exact systems with one shared elimination."""

    size = len(matrix)
    augmented = [
        row[:] + [right[index] for right in rights] for index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = next(
            row for row in range(column, size) if augmented[row][column] != _zero()
        )
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_inverse = _inverse(augmented[column][column])
        augmented[column] = [
            _multiply(entry, pivot_inverse) for entry in augmented[column]
        ]
        for row in range(size):
            if row == column or augmented[row][column] == _zero():
                continue
            factor = augmented[row][column]
            augmented[row] = [
                _sub(entry, _multiply(factor, pivot_entry))
                for entry, pivot_entry in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return [[row[size + offset] for row in augmented] for offset in range(len(rights))]


def _outer(left: list[Element], right: list[Element]) -> Matrix:
    return [[_multiply(a, b) for b in right] for a in left]


def _matrix_scale(matrix: Matrix, scalar: Element) -> Matrix:
    return [[_multiply(value, scalar) for value in row] for row in matrix]


def _frame() -> Matrix:
    one = _rational(1)
    q = _generator(1)
    d = _generator(2)
    b = _generator(4)
    z = _generator(8)
    y = _multiply(_scale(_sub(_rational(15), q), Fraction(1, 8)), b)
    frame = _zeros(16, 3)
    frame[0][0] = one
    frame[1][1] = d
    frame[8][1] = b
    frame[2][2] = _neg(d)
    frame[11][2] = y
    frame[12][2] = z
    return frame


def _information_products() -> list[list[list[tuple[int, int, Fraction]]]]:
    system = build_spin9_clifford_system()
    generators = [sp.Matrix(matrix) / 2 for matrix in system.doubled_spin_generators]
    products: list[list[list[tuple[int, int, Fraction]]]] = []
    for left in generators:
        row = []
        for right in generators:
            product = left.T * right
            entries = []
            for i in range(16):
                for j in range(16):
                    value = product[j, i]
                    if value:
                        entries.append(
                            (
                                i,
                                j,
                                Fraction(int(sp.numer(value)), int(sp.denom(value))),
                            )
                        )
            row.append(entries)
        products.append(row)
    return products


def _spin9_generators() -> list[Matrix]:
    system = build_spin9_clifford_system()
    result = []
    for matrix in system.doubled_spin_generators:
        result.append(
            [
                [
                    _rational(Fraction(int(sp.numer(value)), 2 * int(sp.denom(value))))
                    for value in row
                ]
                for row in sp.Matrix(matrix).tolist()
            ]
        )
    return result


def _information(
    frame_operator: Matrix,
    products: list[list[list[tuple[int, int, Fraction]]]],
) -> Matrix:
    result = _zeros(36, 36)
    for left in range(36):
        for right in range(36):
            result[left][right] = _sum_elements(
                (
                    _scale(frame_operator[i][j], coefficient)
                    for i, j, coefficient in products[left][right]
                ),
            )
    return result


BLOCKS = (
    (6, 12, 14, 17, 21, 34),
    (0, 5, 7, 13, 15, 16, 22, 26, 33, 35),
    (1, 4, 9, 10, 19, 23, 25, 27, 29, 31),
    (2, 3, 8, 11, 18, 20, 24, 28, 30, 32),
)


def _extract(matrix: Matrix, rows: tuple[int, ...], columns: tuple[int, ...]) -> Matrix:
    return [[matrix[row][column] for column in columns] for row in rows]


def _linear_trace(inverses: dict[tuple[int, ...], Matrix], value: Matrix) -> Element:
    return _sum_elements(
        (
            _trace(_matrix_multiply(inverses[block], _extract(value, block, block)))
            for block in BLOCKS
        ),
    )


def _quadratic_trace(
    inverses: dict[tuple[int, ...], Matrix], left: Matrix, right: Matrix
) -> Element:
    total = _zero()
    for block_left in BLOCKS:
        for block_right in BLOCKS:
            term = _matrix_multiply(
                inverses[block_left], _extract(left, block_left, block_right)
            )
            term = _matrix_multiply(term, inverses[block_right])
            term = _matrix_multiply(term, _extract(right, block_right, block_left))
            total = _add(total, _trace(term))
    return total


def _supported_directions(frame: Matrix) -> tuple[list[Matrix], list[int]]:
    coefficient_matrices = []
    first = [[Fraction(0) for _ in range(3)] for _ in range(3)]
    first[0][0], first[1][1] = Fraction(1), Fraction(-1)
    coefficient_matrices.append(first)
    second = [[Fraction(0) for _ in range(3)] for _ in range(3)]
    second[0][0], second[1][1], second[2][2] = (
        Fraction(1),
        Fraction(1),
        Fraction(-2),
    )
    coefficient_matrices.append(second)
    for row, column in ((0, 1), (0, 2), (1, 2)):
        coefficient = [[Fraction(0) for _ in range(3)] for _ in range(3)]
        coefficient[row][column] = coefficient[column][row] = Fraction(1)
        coefficient_matrices.append(coefficient)

    directions = []
    for coefficient in coefficient_matrices:
        embedded = [
            [
                _sum_elements(
                    (
                        _scale(frame[i][index], coefficient[index][column])
                        for index in range(3)
                    ),
                )
                for column in range(3)
            ]
            for i in range(16)
        ]
        directions.append(_matrix_multiply(embedded, _transpose(frame)))
    return directions, [2, 6, 2, 2, 2]


def _base_quadratic(value: Element) -> tuple[Fraction, Fraction]:
    if any(value[index] for index in range(2, 16)):
        raise AssertionError(f"value did not descend to Q(sqrt(241)): {value}")
    return value[0], value[1]


def _quadratic_string(value: Element) -> str:
    rational, radical = _base_quadratic(value)
    return f"({rational}) + ({radical})*sqrt(241)"


def _quadratic_sign(value: Element) -> int:
    """Return the exact sign of an element of Q(sqrt(241))."""

    rational, radical = _base_quadratic(value)
    if radical == 0:
        return (rational > 0) - (rational < 0)
    if rational == 0:
        return (radical > 0) - (radical < 0)
    if rational > 0 and radical > 0:
        return 1
    if rational < 0 and radical < 0:
        return -1
    comparison = rational * rational - 241 * radical * radical
    if comparison == 0:
        return 0
    if rational > 0:  # rational - |radical| sqrt(241)
        return (comparison > 0) - (comparison < 0)
    # radical sqrt(241) - |rational|
    return (comparison < 0) - (comparison > 0)


def _tower_coefficients(value: Element) -> list[str]:
    """Serialize a quadratic-tower element without pretending it is quadratic."""

    return [str(coefficient) for coefficient in value]


def diagnostics() -> dict[str, object]:
    """Evaluate the complete exact local Hessian certificate."""

    frame = _frame()
    projector = _matrix_multiply(frame, _transpose(frame))
    products = _information_products()
    information = _information(projector, products)
    block_of = {
        index: block_index
        for block_index, block in enumerate(BLOCKS)
        for index in block
    }
    information_block_diagonal = all(
        information[row][column] == _zero()
        for row in range(36)
        for column in range(36)
        if block_of[row] != block_of[column]
    )
    frame_orthonormal = _matrix_multiply(_transpose(frame), frame) == _identity(3)
    inverses = {
        block: _matrix_inverse(_extract(information, block, block)) for block in BLOCKS
    }

    d = _generator(2)
    b = _generator(4)
    q = _generator(1)
    c = _scale(_add(_rational(-17), q), Fraction(1, 24))
    one_plus_c = _add(_rational(1), c)
    one_minus_c = _sub(_rational(1), c)
    one_plus_two_c = _add(_rational(1), _scale(c, 2))

    # Start from a simple horizontal variation, then remove its exact component
    # along the symmetric curve.  The result is the pure V5 orientation vector.
    orientation_seed = _zeros(16, 3)
    orientation_seed[9][0] = _rational(1)
    orientation_seed[1][1] = _neg(b)
    orientation_seed[8][1] = d
    curve_variation = _zeros(16, 3)
    curve_variation[1][1] = _divide(_rational(1), _scale(d, 4))
    curve_variation[8][1] = _neg(_divide(_rational(1), _scale(b, 4)))
    curve_variation[2][2] = _neg(curve_variation[1][1])
    y_prime = _sub(
        _divide(b, _multiply(one_plus_c, one_plus_c)),
        _divide(c, _scale(_multiply(b, one_plus_c), 4)),
    )
    curve_variation[11][2] = y_prime
    z_log_derivative = _sub(
        _sub(
            _divide(_rational(-1), _scale(one_minus_c, 2)),
            _divide(_rational(1), one_plus_c),
        ),
        _neg(_divide(_rational(1), one_plus_two_c)),
    )
    curve_variation[12][2] = _multiply(_generator(8), z_log_derivative)
    spin9_generators = _spin9_generators()
    projector_complement = _matrix_sub(_identity(16), projector)
    orbit_directions_all = [
        _matrix_multiply(
            projector_complement, _matrix_multiply(generator_matrix, frame)
        )
        for generator_matrix in spin9_generators
    ]
    # These pivot columns are certified at c=0 by the normal-slice theorem and
    # remain independent at c*; nonsingularity is rechecked by this solve.
    orbit_pivots = (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        29,
        32,
        33,
        34,
        35,
    )
    orbit_directions = [orbit_directions_all[index] for index in orbit_pivots]
    orbit_gram = [
        [_frobenius_inner(left, right) for right in orbit_directions]
        for left in orbit_directions
    ]
    orbit_rights = [
        [_frobenius_inner(direction, value) for direction in orbit_directions]
        for value in (orientation_seed, curve_variation)
    ]
    seed_orbit_coefficients, curve_orbit_coefficients = _linear_solve_many(
        orbit_gram, orbit_rights
    )

    def remove_orbit(value: Matrix, coefficients: list[Element]) -> Matrix:
        result = [row[:] for row in value]
        for coefficient, direction in zip(coefficients, orbit_directions, strict=True):
            result = _matrix_sub(result, _matrix_scale(direction, coefficient))
        return result

    orientation_quotient_seed = remove_orbit(orientation_seed, seed_orbit_coefficients)
    curve_quotient_variation = remove_orbit(curve_variation, curve_orbit_coefficients)
    curve_norm_squared = _frobenius_inner(
        curve_quotient_variation, curve_quotient_variation
    )
    seed_curve_inner = _frobenius_inner(
        orientation_quotient_seed, curve_quotient_variation
    )
    curve_projection = _divide(seed_curve_inner, curve_norm_squared)
    orientation = _matrix_sub(
        orientation_quotient_seed,
        _matrix_scale(curve_quotient_variation, curve_projection),
    )
    orientation_curve_inner = _frobenius_inner(orientation, curve_quotient_variation)
    orientation_horizontal = _matrix_multiply(_transpose(frame), orientation)
    orientation_orbit_inner_products = [
        _frobenius_inner(orientation, direction) for direction in orbit_directions_all
    ]
    curve_orbit_inner_products = [
        _frobenius_inner(curve_quotient_variation, direction)
        for direction in orbit_directions_all
    ]
    orientation_grassmann_norm_squared = _frobenius_inner(orientation, orientation)
    generator = _matrix_sub(
        _matrix_multiply(orientation, _transpose(frame)),
        _matrix_multiply(frame, _transpose(orientation)),
    )
    first_plane = _matrix_sub(
        _matrix_multiply(generator, projector),
        _matrix_multiply(projector, generator),
    )
    second_plane = _matrix_sub(
        _matrix_multiply(generator, first_plane),
        _matrix_multiply(first_plane, generator),
    )
    first_information = _information(first_plane, products)
    second_information = _information(second_plane, products)
    orientation_raw = _sub(
        _linear_trace(inverses, second_information),
        _quadratic_trace(inverses, first_information, first_information),
    )
    orientation_eigenvalue = _divide(
        orientation_raw, orientation_grassmann_norm_squared
    )
    orientation_norm_squared = _trace(
        _matrix_multiply(_transpose(first_plane), first_plane)
    )
    orientation_first_variation = _linear_trace(inverses, first_information)

    supported, supported_norms = _supported_directions(frame)
    supported_information = [
        _information(direction, products) for direction in supported
    ]
    supported_hessian_raw = [
        [
            _neg(_quadratic_trace(inverses, left, right))
            for right in supported_information
        ]
        for left in supported_information
    ]
    supported_eigenvalue = _scale(
        supported_hessian_raw[0][0], Fraction(1, supported_norms[0])
    )
    supported_isotropic = all(
        (
            supported_hessian_raw[row][column] == _zero()
            if row != column
            else supported_hessian_raw[row][column]
            == _scale(supported_eigenvalue, supported_norms[row])
        )
        for row in range(5)
        for column in range(5)
    )
    supported_first_variations = [
        _linear_trace(inverses, value) for value in supported_information
    ]
    supported_actual_norms = [
        _trace(_matrix_multiply(_transpose(direction), direction))
        for direction in supported
    ]

    cross_norm_squared = _zero()
    cross_values = []
    for direction, direction_information, norm_squared in zip(
        supported, supported_information, supported_norms, strict=True
    ):
        mixed_second = _matrix_sub(
            _matrix_multiply(generator, direction),
            _matrix_multiply(direction, generator),
        )
        raw_cross = _sub(
            _linear_trace(inverses, _information(mixed_second, products)),
            _quadratic_trace(inverses, first_information, direction_information),
        )
        cross_values.append(raw_cross)
        cross_norm_squared = _add(
            cross_norm_squared,
            _scale(
                _multiply(raw_cross, raw_cross),
                Fraction(1, norm_squared),
            ),
        )
    cross_norm_squared = _divide(cross_norm_squared, orientation_grassmann_norm_squared)

    multiplicity_determinant = _sub(
        _multiply(orientation_eigenvalue, supported_eigenvalue),
        cross_norm_squared,
    )

    expected_orientation = _scale(_add(_rational(-706), _scale(q, -8)), Fraction(1, 15))
    expected_supported = _scale(
        _add(_rational(-7717), _scale(q, 127)), Fraction(1, 600)
    )
    expected_cross_squared = _scale(
        _add(_rational(449999), _scale(q, -28919)), Fraction(1, 2250)
    )
    expected_determinant = _scale(
        _add(_rational(7563), _scale(q, 195)), Fraction(1, 20)
    )
    expected_curve_eigenvalue = _scale(
        _add(_rational(-11809), _scale(q, -137)), Fraction(1, 540)
    )
    curve_first_derivative = _add(
        _add(
            _divide(_rational(-10), one_minus_c),
            _divide(_rational(5), _add(c, _rational(2))),
        ),
        _divide(_rational(6), one_plus_two_c),
    )
    curve_eigenvalue = _sum_elements(
        (
            _neg(_divide(_rational(10), _multiply(one_minus_c, one_minus_c))),
            _neg(
                _divide(
                    _rational(5),
                    _multiply(_add(c, _rational(2)), _add(c, _rational(2))),
                )
            ),
            _neg(_divide(_rational(12), _multiply(one_plus_two_c, one_plus_two_c))),
        )
    )

    report: dict[str, object] = {
        "schema_version": 1,
        "claim_scope": "exact Hessian at c*=(-17+sqrt(241))/24",
        "field_basis": "Q(sqrt(241), d, b, z), dimension 16",
        "frame_orthonormal": frame_orthonormal,
        "information_block_diagonal": information_block_diagonal,
        "orientation_v5_hessian_eigenvalue": _quadratic_string(orientation_eigenvalue),
        "supported_v5_hessian_eigenvalue": _quadratic_string(supported_eigenvalue),
        "v5_cross_coupling_norm_squared": _quadratic_string(cross_norm_squared),
        "v5_multiplicity_determinant": _quadratic_string(multiplicity_determinant),
        "curve_v1_hessian_eigenvalue_in_c_coordinate": _quadratic_string(
            curve_eigenvalue
        ),
        "orientation_identity": orientation_eigenvalue == expected_orientation,
        "supported_identity": supported_eigenvalue == expected_supported,
        "cross_squared_identity": cross_norm_squared == expected_cross_squared,
        "multiplicity_determinant_identity": multiplicity_determinant
        == expected_determinant,
        "curve_stationary_identity": curve_first_derivative == _zero(),
        "curve_hessian_identity": curve_eigenvalue == expected_curve_eigenvalue,
        "supported_basis_norms_squared": supported_norms,
        "orientation_norm_squared": _quadratic_string(orientation_norm_squared),
        "orientation_grassmann_norm_squared": _quadratic_string(
            orientation_grassmann_norm_squared
        ),
        "curve_grassmann_norm_squared": _quadratic_string(curve_norm_squared),
        "orientation_curve_inner_zero": orientation_curve_inner == _zero(),
        "orientation_horizontal": orientation_horizontal == _zeros(3, 3),
        "orientation_spin9_orbit_normal": all(
            value == _zero() for value in orientation_orbit_inner_products
        ),
        "curve_spin9_orbit_normal": all(
            value == _zero() for value in curve_orbit_inner_products
        ),
        "orbit_gram_nonsingular": True,
        "supported_actual_norms_squared": [
            _quadratic_string(value) for value in supported_actual_norms
        ],
        "orientation_first_variation_zero": orientation_first_variation == _zero(),
        "supported_first_variations_zero": all(
            value == _zero() for value in supported_first_variations
        ),
        "supported_v5_hessian_isotropic": supported_isotropic,
        "tower_basis": [
            "1",
            "q",
            "d",
            "qd",
            "b",
            "qb",
            "db",
            "qdb",
            "z",
            "qz",
            "dz",
            "qdz",
            "bz",
            "qbz",
            "dbz",
            "qdbz",
        ],
        "raw_cross_values_tower_coefficients": [
            _tower_coefficients(value) for value in cross_values
        ],
        "orientation_negative": _quadratic_sign(orientation_eigenvalue) < 0,
        "supported_negative": _quadratic_sign(supported_eigenvalue) < 0,
        "multiplicity_determinant_positive": _quadratic_sign(multiplicity_determinant)
        > 0,
        "curve_negative": _quadratic_sign(curve_eigenvalue) < 0,
        "local_hessian_negative_modulo_spin9": False,
        "global_maximum_claimed": False,
    }
    report["local_hessian_negative_modulo_spin9"] = bool(
        report["orientation_first_variation_zero"]
        and report["supported_first_variations_zero"]
        and report["supported_v5_hessian_isotropic"]
        and report["orientation_negative"]
        and report["supported_negative"]
        and report["multiplicity_determinant_positive"]
        and report["curve_negative"]
    )
    report["passed"] = bool(
        report["orientation_identity"]
        and report["frame_orthonormal"]
        and report["information_block_diagonal"]
        and report["supported_identity"]
        and report["cross_squared_identity"]
        and report["multiplicity_determinant_identity"]
        and report["curve_stationary_identity"]
        and report["curve_hessian_identity"]
        and report["orientation_curve_inner_zero"]
        and report["orientation_horizontal"]
        and report["orientation_spin9_orbit_normal"]
        and report["curve_spin9_orbit_normal"]
        and report["orbit_gram_nonsingular"]
        and report["supported_actual_norms_squared"]
        == [f"({value}) + (0)*sqrt(241)" for value in supported_norms]
        and report["orientation_first_variation_zero"]
        and report["supported_first_variations_zero"]
        and report["supported_v5_hessian_isotropic"]
        and report["orientation_negative"]
        and report["supported_negative"]
        and report["multiplicity_determinant_positive"]
        and report["curve_negative"]
        and report["local_hessian_negative_modulo_spin9"]
        and not report["global_maximum_claimed"]
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
