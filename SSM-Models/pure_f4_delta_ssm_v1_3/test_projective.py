from __future__ import annotations

import torch

from .projective import (
    complex_to_riemann_sphere,
    mobius_action,
    pgl_action,
    riemann_sphere_to_complex,
)


def test_riemann_sphere_roundtrip_and_unit_norm() -> None:
    value = torch.tensor([1.0 + 2.0j, -0.5 + 0.25j], dtype=torch.complex128)
    sphere = complex_to_riemann_sphere(value)
    torch.testing.assert_close(torch.linalg.vector_norm(sphere, dim=-1), torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(riemann_sphere_to_complex(sphere), value)


def test_pgl_action_is_invariant_under_nonzero_scalar_representative() -> None:
    matrix = torch.tensor(
        [[1.0 + 1.0j, 2.0 - 0.5j], [0.25j, 1.0 - 1.0j]],
        dtype=torch.complex128,
    )
    point = torch.tensor([2.0 - 1.0j, 1.0 + 0.0j], dtype=torch.complex128)
    first = pgl_action(matrix, point)
    second = pgl_action((3.0 - 2.0j) * matrix, point)
    # Unit normalization leaves a global phase; compare the projective wedge.
    assert abs(first[0] * second[1] - first[1] * second[0]) < 1e-12


def test_mobius_formula_matches_homogeneous_action() -> None:
    matrix = torch.tensor(
        [[1.0 + 0.5j, -0.25j], [0.1 + 0.2j, 1.0 - 0.3j]],
        dtype=torch.complex128,
    )
    value = torch.tensor(0.5 - 0.75j, dtype=torch.complex128)
    transformed = pgl_action(matrix, torch.stack((value, torch.ones_like(value))))
    affine = transformed[0] / transformed[1]
    torch.testing.assert_close(affine, mobius_action(matrix, value))
