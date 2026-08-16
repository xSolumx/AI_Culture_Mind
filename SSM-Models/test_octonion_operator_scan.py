"""Contracts for the associative octonion multiplication-operator lift."""

from __future__ import annotations

import unittest

import sympy as sp
import torch
from pure_rotor_ssm.octonion_operator_scan import (
    OCTONION_DIM,
    OctonionOperatorSSMLayer,
    bounded_octonion_affine_scan,
    octonion_left_lie_coordinate_matrix,
    octonion_left_multiplication_matrix,
    octonion_operator_prefix_scan,
    octonion_product,
    octonion_right_multiplication_matrix,
    octonion_state_scan,
    scan_composition_counts,
    unit_octonion,
)
from spin8_triality import octonion_left_multiplication


def basis(index: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    value = torch.zeros(OCTONION_DIM, dtype=dtype)
    value[index] = 1
    return value


class OctonionOperatorAlgebraTests(unittest.TestCase):
    def test_fixed_fano_convention_matches_triality_algebra(self) -> None:
        current = octonion_left_multiplication_matrix(
            torch.eye(OCTONION_DIM, dtype=torch.float64)
        )
        canonical = torch.as_tensor(octonion_left_multiplication())
        torch.testing.assert_close(current, canonical, rtol=0, atol=0)

    def test_left_and_right_operators_reproduce_raw_products(self) -> None:
        torch.manual_seed(20260816)
        left = torch.randn(5, OCTONION_DIM, dtype=torch.float64)
        right = torch.randn(5, OCTONION_DIM, dtype=torch.float64)
        raw = octonion_product(left, right)
        through_left = torch.einsum(
            "bij,bj->bi", octonion_left_multiplication_matrix(left), right
        )
        through_right = torch.einsum(
            "bij,bj->bi", octonion_right_multiplication_matrix(right), left
        )
        torch.testing.assert_close(through_left, raw, rtol=0, atol=1e-14)
        torch.testing.assert_close(through_right, raw, rtol=0, atol=1e-14)

    def test_associator_is_preserved_in_operator_lift(self) -> None:
        e1, e2, e4 = basis(1), basis(2), basis(4)
        left_parenthesized = octonion_product(octonion_product(e1, e2), e4)
        right_parenthesized = octonion_product(e1, octonion_product(e2, e4))
        associator = left_parenthesized - right_parenthesized
        self.assertEqual(float(torch.linalg.vector_norm(associator)), 2.0)
        self.assertEqual(float(associator[7]), 2.0)

        lifted = octonion_left_multiplication_matrix(e1) @ (
            octonion_left_multiplication_matrix(e2)
        )
        collapsed = octonion_left_multiplication_matrix(octonion_product(e1, e2))
        torch.testing.assert_close(lifted @ e4, right_parenthesized, rtol=0, atol=0)
        torch.testing.assert_close(collapsed @ e4, left_parenthesized, rtol=0, atol=0)
        self.assertGreater(float(torch.linalg.matrix_norm(lifted - collapsed)), 0)

        l1 = octonion_left_multiplication_matrix(e1)
        l2 = octonion_left_multiplication_matrix(e2)
        l4 = octonion_left_multiplication_matrix(e4)
        torch.testing.assert_close((l1 @ l2) @ l4, l1 @ (l2 @ l4), rtol=0, atol=0)

    def test_unit_operators_are_special_orthogonal_and_keep_center(self) -> None:
        torch.manual_seed(3)
        units = unit_octonion(torch.randn(32, OCTONION_DIM, dtype=torch.float64))
        identity = torch.eye(OCTONION_DIM, dtype=torch.float64)
        for operator in (
            octonion_left_multiplication_matrix(units),
            octonion_right_multiplication_matrix(units),
        ):
            torch.testing.assert_close(
                operator.transpose(-1, -2) @ operator,
                identity.expand_as(operator),
                rtol=0,
                atol=2e-15,
            )
            torch.testing.assert_close(
                torch.linalg.det(operator),
                torch.ones(32, dtype=torch.float64),
                rtol=0,
                atol=3e-15,
            )
        minus_one = -basis(0)
        torch.testing.assert_close(
            octonion_left_multiplication_matrix(minus_one),
            -identity,
            rtol=0,
            atol=0,
        )

    def test_left_generators_have_exact_full_so8_lie_closure(self) -> None:
        coordinates = octonion_left_lie_coordinate_matrix()
        exact = sp.Matrix(coordinates.tolist())
        self.assertEqual(exact.rank(), 28)
        self.assertEqual(exact.det(), -(2**49))


class OctonionOperatorScanTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(41)
        self.tokens = torch.randn(2, 19, 3, OCTONION_DIM, dtype=torch.float64)
        self.initial = unit_octonion(
            torch.randn(2, 3, OCTONION_DIM, dtype=torch.float64)
        )

    def test_all_state_scan_paths_and_both_sides_agree(self) -> None:
        for side in ("left", "right"):
            recurrent, recurrent_final = octonion_state_scan(
                self.tokens, self.initial, side=side, mode="recurrent"
            )
            for mode in ("work_efficient", "hillis_steele"):
                parallel, parallel_final = octonion_state_scan(
                    self.tokens, self.initial, side=side, mode=mode
                )
                torch.testing.assert_close(parallel, recurrent, rtol=1e-12, atol=2e-14)
                torch.testing.assert_close(
                    parallel_final, recurrent_final, rtol=1e-12, atol=2e-14
                )

    def test_operator_scan_padding_and_cache_continuation(self) -> None:
        mask = torch.ones(2, 19, dtype=torch.bool)
        mask[:, 8:11] = False
        full, full_final = octonion_operator_prefix_scan(
            self.tokens, valid_mask=mask, mode="work_efficient"
        )
        first, cache = octonion_operator_prefix_scan(
            self.tokens[:, :7], valid_mask=mask[:, :7], mode="work_efficient"
        )
        second, cache = octonion_operator_prefix_scan(
            self.tokens[:, 7:],
            cache,
            valid_mask=mask[:, 7:],
            mode="work_efficient",
        )
        torch.testing.assert_close(
            torch.cat((first, second), dim=1), full, rtol=1e-12, atol=2e-14
        )
        torch.testing.assert_close(cache, full_final, rtol=1e-12, atol=2e-14)
        torch.testing.assert_close(full[:, 8], full[:, 7], rtol=1e-14, atol=1e-15)
        torch.testing.assert_close(full[:, 9], full[:, 7], rtol=1e-14, atol=1e-15)

    def test_state_scan_first_order_gradients_match_recurrence(self) -> None:
        weights = torch.randn(2, 19, 3, OCTONION_DIM, dtype=torch.float64)

        def gradients(mode: str) -> tuple[torch.Tensor, torch.Tensor]:
            tokens = self.tokens.clone().requires_grad_(True)
            initial = self.initial.clone().requires_grad_(True)
            states, _ = octonion_state_scan(tokens, initial, mode=mode)
            loss = (states * weights).sum()
            token_gradient, initial_gradient = torch.autograd.grad(
                loss, (tokens, initial)
            )
            return token_gradient, initial_gradient

        recurrent = gradients("recurrent")
        parallel = gradients("work_efficient")
        torch.testing.assert_close(parallel[0], recurrent[0], rtol=2e-11, atol=2e-12)
        torch.testing.assert_close(parallel[1], recurrent[1], rtol=2e-11, atol=2e-12)

    def test_work_efficient_tree_reduces_long_composition_count(self) -> None:
        counts = scan_composition_counts(4096)
        self.assertEqual(counts["work_efficient"], 12286)
        self.assertEqual(counts["hillis_steele"], 45057)
        self.assertLess(counts["work_efficient"], counts["hillis_steele"])


class BoundedOctonionLayerTests(unittest.TestCase):
    def _inputs(
        self, length: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(91 + length)
        tokens = torch.randn(2, length, 2, OCTONION_DIM, dtype=torch.float64)
        retention = torch.sigmoid(torch.randn(2, length, 2, dtype=torch.float64))
        write = torch.sigmoid(torch.randn(2, length, 2, dtype=torch.float64))
        values = torch.randn(2, length, 2, OCTONION_DIM, dtype=torch.float64)
        initial = 0.2 * torch.randn(2, 2, OCTONION_DIM, dtype=torch.float64)
        return tokens, retention, write, values, initial

    def test_affine_paths_cache_and_norm_bound(self) -> None:
        tokens, retention, write, values, initial = self._inputs(31)
        recurrent, recurrent_final = bounded_octonion_affine_scan(
            tokens, retention, write, values, initial, mode="recurrent"
        )
        for mode in ("work_efficient", "hillis_steele"):
            parallel, parallel_final = bounded_octonion_affine_scan(
                tokens, retention, write, values, initial, mode=mode
            )
            torch.testing.assert_close(parallel, recurrent, rtol=2e-12, atol=3e-14)
            torch.testing.assert_close(
                parallel_final, recurrent_final, rtol=2e-12, atol=3e-14
            )

        first, cache = bounded_octonion_affine_scan(
            tokens[:, :13],
            retention[:, :13],
            write[:, :13],
            values[:, :13],
            initial,
            mode="work_efficient",
        )
        second, cache = bounded_octonion_affine_scan(
            tokens[:, 13:],
            retention[:, 13:],
            write[:, 13:],
            values[:, 13:],
            cache,
            mode="work_efficient",
        )
        torch.testing.assert_close(
            torch.cat((first, second), dim=1), recurrent, rtol=2e-12, atol=3e-14
        )
        initial_bound = float(torch.linalg.vector_norm(initial, dim=-1).max())
        self.assertLessEqual(
            float(torch.linalg.vector_norm(recurrent, dim=-1).max()),
            max(initial_bound, 1.0) + 1e-12,
        )

    def test_long_recurrent_state_obeys_stabilization_theorem(self) -> None:
        tokens, retention, write, values, initial = self._inputs(2048)
        states, _ = bounded_octonion_affine_scan(
            tokens, retention, write, values, initial, mode="recurrent"
        )
        initial_bound = torch.linalg.vector_norm(initial, dim=-1).max()
        expected = torch.maximum(initial_bound, torch.ones_like(initial_bound))
        self.assertLessEqual(
            float(torch.linalg.vector_norm(states, dim=-1).max()),
            float(expected) + 2e-12,
        )

    def test_affine_first_order_gradients_match_recurrence(self) -> None:
        tokens, _, _, values, initial = self._inputs(17)
        retention_logits = torch.randn(2, 17, 2, dtype=torch.float64)
        write_logits = torch.randn(2, 17, 2, dtype=torch.float64)
        weights = torch.randn(2, 17, 2, OCTONION_DIM, dtype=torch.float64)

        def gradients(mode: str) -> tuple[torch.Tensor, ...]:
            local_tokens = tokens.clone().requires_grad_(True)
            local_retention = retention_logits.clone().requires_grad_(True)
            local_write = write_logits.clone().requires_grad_(True)
            local_values = values.clone().requires_grad_(True)
            local_initial = initial.clone().requires_grad_(True)
            states, _ = bounded_octonion_affine_scan(
                local_tokens,
                torch.sigmoid(local_retention),
                torch.sigmoid(local_write),
                local_values,
                local_initial,
                mode=mode,
            )
            return torch.autograd.grad(
                (states * weights).sum(),
                (
                    local_tokens,
                    local_retention,
                    local_write,
                    local_values,
                    local_initial,
                ),
            )

        recurrent = gradients("recurrent")
        parallel = gradients("work_efficient")
        for actual, expected in zip(parallel, recurrent):
            torch.testing.assert_close(actual, expected, rtol=5e-11, atol=5e-12)

    def test_trainable_layer_shapes_streaming_and_gradients(self) -> None:
        torch.manual_seed(7)
        layer = OctonionOperatorSSMLayer(12, 3).double()
        inputs = torch.randn(2, 11, 12, dtype=torch.float64, requires_grad=True)
        outputs, final_state = layer(inputs, return_recurrent_state=True)
        self.assertEqual(outputs.shape, inputs.shape)
        self.assertEqual(final_state.shape, (2, 3, OCTONION_DIM))
        self.assertEqual(layer.recurrent_state_scalars, 24)
        loss = outputs.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in layer.parameters()
            )
        )

        state = layer.initial_state(2, dtype=torch.float64)
        rows = []
        for position in range(inputs.shape[1]):
            row, state = layer.step(inputs.detach()[:, position], state)
            rows.append(row)
        streamed = torch.stack(rows, dim=1)
        recurrent = layer(inputs.detach(), scan_mode="recurrent")
        torch.testing.assert_close(streamed, recurrent, rtol=1e-12, atol=2e-14)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_parallel_backward_is_finite(self) -> None:
        torch.manual_seed(12)
        layer = OctonionOperatorSSMLayer(16, 4).cuda()
        inputs = torch.randn(3, 127, 16, device="cuda", requires_grad=True)
        outputs = layer(inputs)
        outputs.square().mean().backward()
        self.assertTrue(torch.isfinite(outputs).all())
        self.assertTrue(torch.isfinite(inputs.grad).all())


if __name__ == "__main__":
    unittest.main()
