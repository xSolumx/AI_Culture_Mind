from __future__ import annotations

import numpy as np
import torch

from .action import (
    E6CartanAction,
    E6PolarAction,
    exponential_action,
    ordered_exponential_action,
)
from .albert import (
    E6_DIM,
    F4_DIM,
    SPIN8_DIM,
    SPIN9_DIM,
    albert_determinant,
    build_albert_algebra,
    jordan_product_numpy,
    jordan_structure_constants,
)


def _span_residual(candidate: np.ndarray, basis: np.ndarray) -> float:
    flat_basis = basis.reshape(basis.shape[0], -1).T
    coefficients = np.linalg.lstsq(flat_basis, candidate.reshape(-1), rcond=None)[0]
    reconstruction = flat_basis @ coefficients
    return float(np.max(np.abs(reconstruction - candidate.reshape(-1))))


def test_albert_structure_is_half_integral_and_jordan_commutative() -> None:
    structure = jordan_structure_constants()
    np.testing.assert_array_equal(2.0 * structure, np.rint(2.0 * structure))
    np.testing.assert_array_equal(structure, structure.swapaxes(1, 2))
    rng = np.random.default_rng(20260821)
    left = rng.integers(-2, 3, size=27).astype(np.float64)
    right = rng.integers(-2, 3, size=27).astype(np.float64)
    np.testing.assert_array_equal(
        jordan_product_numpy(left, right), jordan_product_numpy(right, left)
    )


def test_f4_derivations_have_exact_dimension_and_obey_leibniz() -> None:
    algebra = build_albert_algebra()
    assert algebra.f4.shape == (F4_DIM, 27, 27)
    assert np.linalg.matrix_rank(algebra.f4.reshape(F4_DIM, -1)) == F4_DIM
    assert np.max(np.abs(algebra.f4 + algebra.f4.swapaxes(1, 2))) < 1e-14
    rng = np.random.default_rng(17)
    left = rng.integers(-2, 3, size=27).astype(np.float64)
    right = rng.integers(-2, 3, size=27).astype(np.float64)
    for derivation in algebra.f4_raw[::7]:
        expected = derivation @ jordan_product_numpy(left, right)
        actual = jordan_product_numpy(derivation @ left, right) + jordan_product_numpy(
            left, derivation @ right
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-13)


def test_executable_subgroup_ladder_has_expected_dimensions_and_closure() -> None:
    algebra = build_albert_algebra()
    assert algebra.spin8.shape[0] == SPIN8_DIM
    assert algebra.spin9.shape[0] == SPIN9_DIM
    for generators, dimension in (
        (algebra.spin8, SPIN8_DIM),
        (algebra.spin9, SPIN9_DIM),
    ):
        assert np.linalg.matrix_rank(generators.reshape(dimension, -1)) == dimension
        for left, right in ((0, 1), (2, 7), (dimension - 2, dimension - 1)):
            commutator = generators[left] @ generators[right] - generators[right] @ generators[left]
            assert _span_residual(commutator, generators) < 2e-13
        for generator in generators[::5]:
            assert _span_residual(generator, algebra.f4) < 2e-13
    diagonal = np.eye(27)[:3]
    assert np.max(np.abs(np.einsum("fij,pj->fpi", algebra.spin8, diagonal))) < 1e-13
    assert np.max(np.abs(np.einsum("fij,j->fi", algebra.spin9, diagonal[0]))) < 1e-13


def test_f4_and_e6_exponentials_preserve_the_albert_cubic() -> None:
    algebra = build_albert_algebra()
    assert algebra.e6.shape == (E6_DIM, 27, 27)
    assert np.linalg.matrix_rank(algebra.e6.reshape(E6_DIM, -1)) == E6_DIM
    torch.manual_seed(19)
    value = torch.randn(4, 27, dtype=torch.float64)
    for generators in (algebra.f4, algebra.e6):
        coordinates = 0.03 * torch.randn(len(generators), dtype=torch.float64)
        action = exponential_action(
            coordinates, torch.tensor(generators, dtype=torch.float64)
        )
        transformed = value @ action.T
        torch.testing.assert_close(
            albert_determinant(transformed),
            albert_determinant(value),
            rtol=2e-12,
            atol=2e-12,
        )
    f4_action = exponential_action(
        0.03 * torch.randn(F4_DIM, dtype=torch.float64),
        torch.tensor(algebra.f4, dtype=torch.float64),
    )
    torch.testing.assert_close(
        f4_action.T @ f4_action,
        torch.eye(27, dtype=torch.float64),
        rtol=2e-12,
        atol=2e-12,
    )
    e6_action = exponential_action(
        0.03 * torch.randn(E6_DIM, dtype=torch.float64),
        torch.tensor(algebra.e6, dtype=torch.float64),
    )
    assert float((e6_action.T @ e6_action - torch.eye(27, dtype=torch.float64)).abs().max()) > 1e-4


def test_tracefree_f4_restriction_is_26_dimensional_and_skew() -> None:
    algebra = build_albert_algebra()
    assert algebra.f4_tracefree.shape == (52, 26, 26)
    assert np.linalg.matrix_rank(algebra.f4_tracefree.reshape(52, -1)) == 52
    assert np.max(np.abs(algebra.f4_tracefree + algebra.f4_tracefree.swapaxes(1, 2))) < 1e-13


def test_ordered_exponentials_match_explicit_chronological_product() -> None:
    algebra = build_albert_algebra()
    generators = torch.tensor(algebra.e6, dtype=torch.float64)
    torch.manual_seed(29)
    coordinates = 0.01 * torch.randn(3, E6_DIM, dtype=torch.float64)
    actual = ordered_exponential_action(coordinates, generators)
    factors = [exponential_action(row, generators) for row in coordinates]
    expected = factors[2] @ factors[1] @ factors[0]
    torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)


def test_polar_and_cartan_e6_actions_preserve_the_cubic_and_differentiate() -> None:
    torch.manual_seed(37)
    value = torch.randn(3, 27, dtype=torch.float64)
    for action in (E6PolarAction(), E6CartanAction()):
        coordinates = (
            0.02 * torch.randn(2, action.coordinate_dim, dtype=torch.float64)
        ).requires_grad_()
        matrix = action(coordinates)
        transformed = torch.einsum("bij,bj->bi", matrix, value[:2])
        torch.testing.assert_close(
            albert_determinant(transformed),
            albert_determinant(value[:2]),
            rtol=3e-12,
            atol=3e-12,
        )
        transformed.square().mean().backward()
        assert torch.isfinite(coordinates.grad).all()
        assert float(coordinates.grad.abs().max()) > 0.0


def test_cartan_radial_action_is_positive_diagonal_and_rank_two() -> None:
    action = E6CartanAction()
    assert action.coordinate_dim == 106
    assert torch.linalg.matrix_rank(action.radial_eigenvalues) == 2
    coordinates = torch.zeros(1, 106, dtype=torch.float64)
    coordinates[0, 52:54] = torch.tensor((0.2, -0.1), dtype=torch.float64)
    matrix = action(coordinates)[0]
    torch.testing.assert_close(matrix, torch.diag(torch.diagonal(matrix)))
    assert torch.all(torch.diagonal(matrix) > 0.0)
