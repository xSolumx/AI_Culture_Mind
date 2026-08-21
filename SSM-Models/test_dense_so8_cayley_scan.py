"""Structural contracts for the experimental dense SO(8) Cayley scan."""

from __future__ import annotations

import unittest

import torch

from pure_rotor_ssm.dense_so8_cayley_scan import (
    DenseSO8CayleySSMLayer,
    SO8_LIE_DIM,
    bounded_so8_cayley_affine_scan,
    cayley_so8,
    so8_clifford_lie_basis,
    so8_tangent_matrix,
)


class DenseSO8CayleyAlgebraTests(unittest.TestCase):
    def test_clifford_basis_spans_so8_and_cayley_is_special_orthogonal(self) -> None:
        basis = so8_clifford_lie_basis(dtype=torch.float64)
        self.assertEqual(tuple(basis.shape), (28, 8, 8))
        self.assertEqual(int(torch.linalg.matrix_rank(basis.reshape(28, -1))), 28)
        torch.testing.assert_close(basis.transpose(-1, -2), -basis, rtol=0, atol=0)

        torch.manual_seed(20260821)
        tangent = so8_tangent_matrix(torch.randn(3, 5, SO8_LIE_DIM, dtype=torch.float64))
        rotations = cayley_so8(tangent)
        identity = torch.eye(8, dtype=torch.float64).expand_as(rotations)
        torch.testing.assert_close(
            rotations.transpose(-1, -2) @ rotations, identity, rtol=2e-13, atol=2e-13
        )
        torch.testing.assert_close(
            torch.linalg.det(rotations), torch.ones(3, 5, dtype=torch.float64),
            rtol=2e-13, atol=2e-13,
        )


class DenseSO8CayleyScanTests(unittest.TestCase):
    def _inputs(self, length: int) -> tuple[torch.Tensor, ...]:
        torch.manual_seed(73 + length)
        coordinates = 0.2 * torch.randn(2, length, 3, SO8_LIE_DIM, dtype=torch.float64)
        retention = torch.sigmoid(torch.randn(2, length, 3, dtype=torch.float64))
        write = torch.sigmoid(torch.randn(2, length, 3, dtype=torch.float64))
        values = torch.randn(2, length, 3, 8, dtype=torch.float64)
        initial = 0.2 * torch.randn(2, 3, 8, dtype=torch.float64)
        return coordinates, retention, write, values, initial

    def test_scan_cache_mask_and_state_bound(self) -> None:
        coordinates, retention, write, values, initial = self._inputs(23)
        mask = torch.ones(2, 23, dtype=torch.bool)
        mask[:, 9:12] = False
        recurrent, final = bounded_so8_cayley_affine_scan(
            coordinates, retention, write, values, initial, valid_mask=mask, mode="recurrent"
        )
        for mode in ("work_efficient", "hillis_steele"):
            parallel, parallel_final = bounded_so8_cayley_affine_scan(
                coordinates, retention, write, values, initial, valid_mask=mask, mode=mode
            )
            torch.testing.assert_close(parallel, recurrent, rtol=3e-12, atol=4e-14)
            torch.testing.assert_close(parallel_final, final, rtol=3e-12, atol=4e-14)

        first, cache = bounded_so8_cayley_affine_scan(
            coordinates[:, :8], retention[:, :8], write[:, :8], values[:, :8], initial,
            valid_mask=mask[:, :8], mode="work_efficient"
        )
        second, cache = bounded_so8_cayley_affine_scan(
            coordinates[:, 8:], retention[:, 8:], write[:, 8:], values[:, 8:], cache,
            valid_mask=mask[:, 8:], mode="work_efficient"
        )
        torch.testing.assert_close(
            torch.cat((first, second), dim=1), recurrent, rtol=3e-12, atol=4e-14
        )
        initial_bound = float(torch.linalg.vector_norm(initial, dim=-1).max())
        self.assertLessEqual(
            float(torch.linalg.vector_norm(recurrent, dim=-1).max()),
            max(initial_bound, 1.0) + 2e-12,
        )

    def test_first_order_gradients_match_recurrent(self) -> None:
        coordinates, _, _, values, initial = self._inputs(11)
        retention_logits = torch.randn(2, 11, 3, dtype=torch.float64)
        write_logits = torch.randn(2, 11, 3, dtype=torch.float64)
        weights = torch.randn(2, 11, 3, 8, dtype=torch.float64)

        def gradients(mode: str) -> tuple[torch.Tensor, ...]:
            local_coordinates = coordinates.clone().requires_grad_(True)
            local_retention = retention_logits.clone().requires_grad_(True)
            local_write = write_logits.clone().requires_grad_(True)
            local_values = values.clone().requires_grad_(True)
            local_initial = initial.clone().requires_grad_(True)
            states, _ = bounded_so8_cayley_affine_scan(
                local_coordinates,
                torch.sigmoid(local_retention),
                torch.sigmoid(local_write),
                local_values,
                local_initial,
                mode=mode,
            )
            return torch.autograd.grad(
                (states * weights).sum(),
                (local_coordinates, local_retention, local_write, local_values, local_initial),
            )

        recurrent = gradients("recurrent")
        parallel = gradients("work_efficient")
        for actual, expected in zip(parallel, recurrent, strict=True):
            torch.testing.assert_close(actual, expected, rtol=5e-11, atol=5e-12)

    def test_layer_streaming_contract(self) -> None:
        torch.manual_seed(17)
        layer = DenseSO8CayleySSMLayer(6, 2, tangent_initialization_scale=0.15).double()
        inputs = torch.randn(3, 13, 6, dtype=torch.float64)
        full, final = layer(inputs, return_recurrent_state=True, scan_mode="work_efficient")
        state = layer.initial_state(3, dtype=torch.float64)
        streamed = []
        for position in range(inputs.shape[1]):
            output, state = layer.step(inputs[:, position], state)
            streamed.append(output)
        torch.testing.assert_close(
            torch.stack(streamed, dim=1), full, rtol=3e-12, atol=4e-14
        )
        torch.testing.assert_close(state, final, rtol=3e-12, atol=4e-14)
        self.assertEqual(layer.recurrent_state_scalars, 16)
        self.assertEqual(layer.transition_control_scalars, 56)


if __name__ == "__main__":
    unittest.main()
