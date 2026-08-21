"""Exact-data construction of the real Albert algebra and its Lie algebras.

The Albert algebra is ``H_3(O)``: three-by-three Hermitian octonionic matrices
with the Jordan product ``x o y = (xy + yx) / 2``.  Raw octonion products are
never used as scan composition.  They are used only to construct the finite
dimensional bilinear product and its associative linear operators.

The generated operator spaces are

``f4 = Der(J)`` (dimension 52), and
``e6(-26) = Der(J) + L(J_0)`` (dimension 78).

All structure coefficients are dyadic rationals.  NumPy float64 is used as a
compact carrier after the integer/dyadic construction; the tests reconstruct
the scaled integer tensors and verify the advertised ranks and identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np
import torch

OCTONION_DIM = 8
ALBERT_DIM = 27
ALBERT_TRACEFREE_DIM = 26
F4_DIM = 52
E6_DIM = 78
SPIN8_DIM = 28
SPIN9_DIM = 36

FANO_TRIPLES = (
    (1, 2, 3),
    (1, 4, 5),
    (1, 7, 6),
    (2, 4, 6),
    (2, 5, 7),
    (3, 4, 7),
    (3, 6, 5),
)


def octonion_structure_constants() -> np.ndarray:
    """Return integer ``c[k,i,j]`` with ``e_i e_j = c[k,i,j] e_k``."""

    constants = np.zeros((OCTONION_DIM,) * 3, dtype=np.int64)
    for index in range(OCTONION_DIM):
        constants[index, 0, index] = 1
        constants[index, index, 0] = 1
    for index in range(1, OCTONION_DIM):
        constants[0, index, index] = -1
    for first, second, third in FANO_TRIPLES:
        for left, right, product in (
            (first, second, third),
            (second, third, first),
            (third, first, second),
        ):
            constants[product, left, right] = 1
            constants[product, right, left] = -1
    if np.count_nonzero(constants) != OCTONION_DIM**2:
        raise AssertionError("the Fano convention must define every basis product")
    return constants


def _octonion_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.einsum(
        "kij,...i,...j->...k", octonion_structure_constants(), left, right
    )


def _octonion_conjugate(value: np.ndarray) -> np.ndarray:
    result = value.copy()
    result[..., 1:] *= -1
    return result


def coordinates_to_hermitian(coordinates: np.ndarray) -> np.ndarray:
    """Map final-axis 27-vectors to ``(...,3,3,8)`` Hermitian matrices."""

    coordinates = np.asarray(coordinates)
    if coordinates.shape[-1] != ALBERT_DIM:
        raise ValueError("Albert coordinates must have final dimension 27")
    matrix = np.zeros((*coordinates.shape[:-1], 3, 3, 8), dtype=coordinates.dtype)
    for index in range(3):
        matrix[..., index, index, 0] = coordinates[..., index]
    for start, left, right in ((3, 0, 1), (11, 0, 2), (19, 1, 2)):
        value = coordinates[..., start : start + OCTONION_DIM]
        matrix[..., left, right, :] = value
        matrix[..., right, left, :] = _octonion_conjugate(value)
    return matrix


def hermitian_to_coordinates(matrix: np.ndarray) -> np.ndarray:
    """Inverse of :func:`coordinates_to_hermitian` on Hermitian matrices."""

    if matrix.shape[-3:] != (3, 3, OCTONION_DIM):
        raise ValueError("matrix must end in (3,3,8)")
    coordinates = np.empty((*matrix.shape[:-3], ALBERT_DIM), dtype=matrix.dtype)
    coordinates[..., :3] = np.stack(
        [matrix[..., index, index, 0] for index in range(3)], axis=-1
    )
    coordinates[..., 3:11] = matrix[..., 0, 1, :]
    coordinates[..., 11:19] = matrix[..., 0, 2, :]
    coordinates[..., 19:27] = matrix[..., 1, 2, :]
    return coordinates


def _octonion_matrix_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.zeros_like(left)
    for row in range(3):
        for column in range(3):
            for inner in range(3):
                result[..., row, column, :] += _octonion_product(
                    left[..., row, inner, :], right[..., inner, column, :]
                )
    return result


def jordan_product_numpy(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Evaluate one Albert Jordan product in standard 27 coordinates."""

    left_matrix = coordinates_to_hermitian(left)
    right_matrix = coordinates_to_hermitian(right)
    product = 0.5 * (
        _octonion_matrix_product(left_matrix, right_matrix)
        + _octonion_matrix_product(right_matrix, left_matrix)
    )
    return hermitian_to_coordinates(product)


@lru_cache(maxsize=1)
def jordan_structure_constants() -> np.ndarray:
    """Return ``C[k,i,j]`` for the Albert Jordan product."""

    basis = np.eye(ALBERT_DIM, dtype=np.float64)
    constants = np.empty((ALBERT_DIM, ALBERT_DIM, ALBERT_DIM), dtype=np.float64)
    for left in range(ALBERT_DIM):
        for right in range(ALBERT_DIM):
            constants[:, left, right] = jordan_product_numpy(
                basis[left], basis[right]
            )
    scaled = constants * 2.0
    if not np.array_equal(scaled, np.rint(scaled)):
        raise AssertionError("Albert structure constants must be half-integral")
    constants.setflags(write=False)
    return constants


def _raw_left_multiplications() -> np.ndarray:
    constants = jordan_structure_constants()
    return np.transpose(constants, (1, 0, 2)).copy()


def trace_metric() -> np.ndarray:
    """Coordinate Gram matrix for ``tr(x o y)``."""

    return np.diag([1.0, 1.0, 1.0] + [2.0] * 24)


def orthonormal_to_raw_basis() -> np.ndarray:
    """Columns map Euclidean orthonormal coordinates to standard coordinates."""

    return np.diag([1.0, 1.0, 1.0] + [1.0 / np.sqrt(2.0)] * 24)


def tracefree_to_raw_basis() -> np.ndarray:
    """Columns form an orthonormal basis of the 26D trace-free Albert module."""

    basis = np.zeros((ALBERT_DIM, ALBERT_TRACEFREE_DIM), dtype=np.float64)
    basis[:3, 0] = (1.0, -1.0, 0.0)
    basis[:3, 0] /= np.sqrt(2.0)
    basis[:3, 1] = (1.0, 1.0, -2.0)
    basis[:3, 1] /= np.sqrt(6.0)
    for index in range(24):
        basis[3 + index, 2 + index] = 1.0 / np.sqrt(2.0)
    return basis


def _to_orthonormal_operator(raw_operator: np.ndarray) -> np.ndarray:
    basis = orthonormal_to_raw_basis()
    inverse = np.diag([1.0, 1.0, 1.0] + [np.sqrt(2.0)] * 24)
    return inverse @ raw_operator @ basis


def _independent_operator_indices(
    candidates: list[tuple[tuple[int, int], np.ndarray]],
    target_rank: int,
    *,
    tolerance: float = 1e-10,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    selected: list[tuple[int, int]] = []
    operators: list[np.ndarray] = []
    orthonormal_columns: list[np.ndarray] = []
    for label, candidate in candidates:
        residual = candidate.reshape(-1).astype(np.float64)
        for column in orthonormal_columns:
            residual -= np.dot(column, residual) * column
        norm = np.linalg.norm(residual)
        if norm <= tolerance:
            continue
        orthonormal_columns.append(residual / norm)
        selected.append(label)
        operators.append(candidate)
        if len(operators) == target_rank:
            break
    if len(operators) != target_rank:
        raise AssertionError(
            f"operator span has rank {len(operators)}, expected {target_rank}"
        )
    return selected, np.stack(operators)


def _nullspace(matrix: np.ndarray, expected_dimension: int) -> np.ndarray:
    """Deterministic-sign numerical nullspace for subgroup alignment."""

    _, singular_values, right = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.sum(singular_values > 1e-10))
    basis = right[rank:]
    if basis.shape[0] != expected_dimension:
        raise AssertionError(
            f"stabilizer dimension {basis.shape[0]}, expected {expected_dimension}"
        )
    for row in basis:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1
    return basis


@dataclass(frozen=True)
class AlbertAlgebra:
    """Cached numerical carrier for exact Albert/F4/E6 structure data."""

    structure: np.ndarray
    left_multiplications_raw: np.ndarray
    f4_pairs: tuple[tuple[int, int], ...]
    f4_raw: np.ndarray
    f4: np.ndarray
    f4_tracefree: np.ndarray
    spin8: np.ndarray
    spin9: np.ndarray
    e6: np.ndarray
    tracefree_basis_raw: np.ndarray
    identity_orthonormal: np.ndarray

    def torch_structure(self, reference: torch.Tensor) -> torch.Tensor:
        # ``structure`` is read-only to protect the cached exact-data carrier;
        # copy before handing memory ownership to Torch.
        return torch.tensor(self.structure, dtype=reference.dtype, device=reference.device)

    def torch_generators(self, algebra: str, reference: torch.Tensor) -> torch.Tensor:
        if algebra == "f4":
            generators = self.f4
        elif algebra == "spin8":
            generators = self.spin8
        elif algebra == "spin9":
            generators = self.spin9
        elif algebra == "f4_tracefree":
            generators = self.f4_tracefree
        elif algebra == "e6":
            generators = self.e6
        else:
            raise ValueError(f"unknown exceptional algebra {algebra!r}")
        return torch.as_tensor(generators, dtype=reference.dtype, device=reference.device)


@lru_cache(maxsize=1)
def build_albert_algebra() -> AlbertAlgebra:
    """Construct the 27D Albert product, 52D F4, and 78D E6(-26)."""

    structure = jordan_structure_constants()
    left_raw = _raw_left_multiplications()
    candidates: list[tuple[tuple[int, int], np.ndarray]] = []
    for left, right in combinations(range(ALBERT_DIM), 2):
        derivation = left_raw[left] @ left_raw[right] - left_raw[right] @ left_raw[left]
        candidates.append(((left, right), derivation))
    pairs, f4_raw = _independent_operator_indices(candidates, F4_DIM)
    f4 = np.stack([_to_orthonormal_operator(operator) for operator in f4_raw])

    # F4 acts on primitive idempotents transitively with Spin(9) stabilizer.
    # Fixing an ordered diagonal Jordan frame leaves Spin(8).  These subgroup
    # bases are derived from the same 52 generators, so every inclusion is an
    # executable restriction rather than an unrelated representation claim.
    diagonal_idempotents = np.eye(ALBERT_DIM, dtype=np.float64)[:3]
    spin9_constraints = np.stack(
        [generator @ diagonal_idempotents[0] for generator in f4], axis=1
    )
    spin9_coefficients = _nullspace(spin9_constraints, SPIN9_DIM)
    spin9 = np.einsum("af,fij->aij", spin9_coefficients, f4)
    spin8_constraints = np.concatenate(
        [
            np.stack([generator @ point for generator in f4], axis=1)
            for point in diagonal_idempotents[:2]
        ],
        axis=0,
    )
    spin8_coefficients = _nullspace(spin8_constraints, SPIN8_DIM)
    spin8 = np.einsum("af,fij->aij", spin8_coefficients, f4)

    tracefree_basis = tracefree_to_raw_basis()
    metric = trace_metric()
    f4_tracefree = np.einsum(
        "ai,fab,bj->fij", tracefree_basis, metric @ f4_raw, tracefree_basis
    )

    symmetric_generators = []
    for column in tracefree_basis.T:
        raw = np.einsum("i,ijk->jk", column, left_raw)
        symmetric_generators.append(_to_orthonormal_operator(raw))
    e6 = np.concatenate([f4, np.stack(symmetric_generators)], axis=0)

    identity_raw = np.zeros(ALBERT_DIM, dtype=np.float64)
    identity_raw[:3] = 1.0
    raw_to_orthonormal = np.diag([1.0, 1.0, 1.0] + [np.sqrt(2.0)] * 24)
    identity_orthonormal = raw_to_orthonormal @ identity_raw

    return AlbertAlgebra(
        structure=structure,
        left_multiplications_raw=left_raw,
        f4_pairs=tuple(pairs),
        f4_raw=f4_raw,
        f4=f4,
        f4_tracefree=f4_tracefree,
        spin8=spin8,
        spin9=spin9,
        e6=e6,
        tracefree_basis_raw=tracefree_basis,
        identity_orthonormal=identity_orthonormal,
    )


def raw_to_orthonormal(coordinates: torch.Tensor) -> torch.Tensor:
    if coordinates.shape[-1] != ALBERT_DIM:
        raise ValueError("Albert coordinates must have final dimension 27")
    scale = coordinates.new_tensor([1.0, 1.0, 1.0] + [2.0**0.5] * 24)
    return coordinates * scale


def orthonormal_to_raw(coordinates: torch.Tensor) -> torch.Tensor:
    if coordinates.shape[-1] != ALBERT_DIM:
        raise ValueError("Albert coordinates must have final dimension 27")
    scale = coordinates.new_tensor([1.0, 1.0, 1.0] + [2.0**-0.5] * 24)
    return coordinates * scale


def jordan_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Differentiable Albert product in orthonormal 27 coordinates."""

    if left.shape != right.shape or left.shape[-1] != ALBERT_DIM:
        raise ValueError("Albert operands must share a final dimension of 27")
    left_raw = orthonormal_to_raw(left)
    right_raw = orthonormal_to_raw(right)
    structure = build_albert_algebra().torch_structure(left)
    product_raw = torch.einsum("kij,...i,...j->...k", structure, left_raw, right_raw)
    return raw_to_orthonormal(product_raw)


def albert_trace(value: torch.Tensor) -> torch.Tensor:
    raw = orthonormal_to_raw(value)
    return raw[..., :3].sum(dim=-1)


def albert_determinant(value: torch.Tensor) -> torch.Tensor:
    """Cubic norm from the power-associative Jordan trace identity."""

    square = jordan_product(value, value)
    cube = jordan_product(square, value)
    trace = albert_trace(value)
    trace_square = albert_trace(square)
    trace_cube = albert_trace(cube)
    return (
        trace.pow(3) - 3.0 * trace * trace_square + 2.0 * trace_cube
    ) / 6.0


__all__ = [
    "ALBERT_DIM",
    "ALBERT_TRACEFREE_DIM",
    "E6_DIM",
    "F4_DIM",
    "SPIN8_DIM",
    "SPIN9_DIM",
    "AlbertAlgebra",
    "albert_determinant",
    "albert_trace",
    "build_albert_algebra",
    "coordinates_to_hermitian",
    "hermitian_to_coordinates",
    "jordan_product",
    "jordan_product_numpy",
    "jordan_structure_constants",
    "octonion_structure_constants",
    "orthonormal_to_raw",
    "raw_to_orthonormal",
    "trace_metric",
    "tracefree_to_raw_basis",
]
