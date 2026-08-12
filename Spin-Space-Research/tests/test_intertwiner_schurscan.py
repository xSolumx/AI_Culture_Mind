from __future__ import annotations

import unittest

import torch

from intertwiner_schurscan import (
    associative_matrix_scan,
    bilinear_contract,
    diagnostics,
    feedback_degree_growth,
    recurrent_intertwiner_scan,
    scan_composition_counts,
    scan_dependency_depths,
    so3_cross_product_tensor,
    staged_intertwiner_scan,
    work_efficient_affine_prefixes,
    work_efficient_associative_matrix_scan,
)


class IntertwinerSchurScanTests(unittest.TestCase):
    @staticmethod
    def _random_problem(
        *, batch: int, length: int, dimension: int, dtype: torch.dtype, seed: int
    ) -> tuple[torch.Tensor, ...]:
        generator = torch.Generator().manual_seed(seed)

        def actions() -> torch.Tensor:
            raw = torch.randn(
                batch, length, dimension, dimension, dtype=dtype, generator=generator
            )
            skew = raw - raw.transpose(-1, -2)
            return 0.997 * torch.matrix_exp(0.035 * skew)

        return (
            actions(),
            0.01
            * torch.randn(batch, length, dimension, dtype=dtype, generator=generator),
            actions(),
            0.01
            * torch.randn(batch, length, dimension, dtype=dtype, generator=generator),
            actions(),
            0.006
            * torch.randn(batch, length, dimension, dtype=dtype, generator=generator),
            torch.randn(batch, dimension, dtype=dtype, generator=generator),
            torch.randn(batch, dimension, dtype=dtype, generator=generator),
            torch.randn(batch, dimension, dtype=dtype, generator=generator),
            so3_cross_product_tensor(dtype=dtype),
        )

    def test_generic_triangular_scan_and_lift(self) -> None:
        report = diagnostics()
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["parallel_scan_stages"], 2)
        self.assertEqual(report["streaming_cache_scalars"], 9)
        self.assertEqual(report["homogeneous_proof_lift_scalars"], 19)
        self.assertLessEqual(report["staged_recurrent_max_abs_error"], 1e-11)
        self.assertLessEqual(report["hillis_recurrent_max_abs_error"], 1e-11)
        self.assertLessEqual(report["backend_max_abs_difference"], 1e-11)
        self.assertLessEqual(report["affine_tree_recurrent_max_abs_error"], 1e-11)
        self.assertLessEqual(report["lifted_recurrent_max_abs_error"], 1e-11)
        self.assertLessEqual(report["equivariance_max_abs_error"], 1e-11)

    def test_ordered_matrix_scan_at_irregular_lengths(self) -> None:
        for dtype, tolerance in ((torch.float64, 2e-11), (torch.float32, 3e-5)):
            for length in (1, 2, 3, 5, 7, 9, 17, 31, 64, 127):
                with self.subTest(dtype=dtype, length=length):
                    generator = torch.Generator().manual_seed(1_000 + length)
                    matrices = torch.eye(dtype=dtype, n=4).reshape(1, 1, 4, 4)
                    matrices = matrices + 0.05 * torch.randn(
                        2, length, 4, 4, dtype=dtype, generator=generator
                    )
                    sequential = []
                    prefix = torch.eye(dtype=dtype, n=4).expand(2, -1, -1)
                    for position in range(length):
                        prefix = matrices[:, position] @ prefix
                        sequential.append(prefix)
                    expected = torch.stack(sequential, dim=1)
                    work_efficient = work_efficient_associative_matrix_scan(matrices)
                    hillis = associative_matrix_scan(matrices)
                    self.assertLess(
                        float((work_efficient - expected).abs().max()), tolerance
                    )
                    self.assertLess(float((hillis - expected).abs().max()), tolerance)

    def test_affine_tree_matches_recurrence_and_gradients(self) -> None:
        torch.manual_seed(3)
        dtype = torch.float64
        batch, length, dimension = 2, 7, 3
        base_action = torch.eye(dtype=dtype, n=dimension).reshape(
            1, 1, dimension, dimension
        ) + 0.03 * torch.randn(batch, length, dimension, dimension, dtype=dtype)
        base_drive = 0.05 * torch.randn(batch, length, dimension, dtype=dtype)
        weights = torch.randn(batch, length, dimension, dtype=dtype)

        def tree_loss(action: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
            prefix_action, prefix_drive = work_efficient_affine_prefixes(action, drive)
            return (prefix_drive * weights).sum() + 0.01 * prefix_action.square().sum()

        def recurrent_loss(action: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
            prefix_action = torch.eye(dtype=dtype, n=dimension).expand(batch, -1, -1)
            prefix_drive = torch.zeros(batch, dimension, dtype=dtype)
            loss = torch.zeros((), dtype=dtype)
            for position in range(length):
                prefix_drive = (
                    torch.einsum("bij,bj->bi", action[:, position], prefix_drive)
                    + drive[:, position]
                )
                prefix_action = action[:, position] @ prefix_action
                loss = loss + (prefix_drive * weights[:, position]).sum()
                loss = loss + 0.01 * prefix_action.square().sum()
            return loss

        gradients = []
        losses = []
        for loss_function in (tree_loss, recurrent_loss):
            action = base_action.clone().requires_grad_()
            drive = base_drive.clone().requires_grad_()
            loss = loss_function(action, drive)
            gradients.append(torch.autograd.grad(loss, (action, drive)))
            losses.append(loss)

        self.assertLess(float((losses[0] - losses[1]).detach().abs()), 2e-12)
        for tree_gradient, recurrent_gradient in zip(gradients[0], gradients[1]):
            self.assertLess(
                float((tree_gradient - recurrent_gradient).abs().max()), 2e-12
            )

    def test_full_intertwiner_gradient_parity(self) -> None:
        problem = self._random_problem(
            batch=1, length=7, dimension=3, dtype=torch.float64, seed=81
        )
        weights = [
            torch.randn(problem[0].shape[:2], dtype=torch.float64) for _ in range(3)
        ]

        def differentiate(kind: str) -> tuple[torch.Tensor, ...]:
            differentiable = tuple(value.clone().requires_grad_() for value in problem)
            if kind == "tree":
                output = staged_intertwiner_scan(*differentiable)
            else:
                output = recurrent_intertwiner_scan(*differentiable)
            loss = sum(
                (component.square().sum(dim=-1) * weight).sum()
                for component, weight in zip(output, weights)
            )
            return torch.autograd.grad(loss, differentiable)

        tree_gradients = differentiate("tree")
        recurrent_gradients = differentiate("recurrent")
        for tree_gradient, recurrent_gradient in zip(
            tree_gradients, recurrent_gradients
        ):
            self.assertLess(
                float((tree_gradient - recurrent_gradient).abs().max()), 2e-10
            )

    def test_long_horizon_contract_and_operation_counts(self) -> None:
        problem = self._random_problem(
            batch=1, length=2_048, dimension=3, dtype=torch.float64, seed=2_048
        )
        tree = staged_intertwiner_scan(*problem)
        recurrent = recurrent_intertwiner_scan(*problem)
        absolute_error = max(
            float((left - right).abs().max()) for left, right in zip(tree, recurrent)
        )
        scale = max(float(value.abs().max()) for value in recurrent)
        self.assertLess(absolute_error / max(scale, 1.0), 2e-12)
        self.assertEqual(scan_composition_counts(64)["hillis_steele"], 321)
        self.assertEqual(scan_composition_counts(64)["work_efficient"], 190)
        self.assertEqual(scan_dependency_depths(64)["hillis_steele"], 6)
        self.assertEqual(scan_dependency_depths(64)["work_efficient"], 13)
        self.assertLess(
            scan_composition_counts(2_048)["work_efficient"],
            scan_composition_counts(2_048)["hillis_steele"],
        )

    def test_malformed_contract_and_empty_scans_are_rejected(self) -> None:
        u = torch.ones(2, 3)
        v = torch.ones(2, 3)
        with self.assertRaisesRegex(ValueError, "beta must have shape"):
            bilinear_contract(u, v, torch.ones(3, 3))
        with self.assertRaisesRegex(ValueError, "scan length must be positive"):
            work_efficient_associative_matrix_scan(torch.empty(1, 0, 3, 3))

    def test_so3_control_is_actual_cross_product(self) -> None:
        dtype = torch.float64
        beta = so3_cross_product_tensor(dtype=dtype)
        u = torch.tensor([[1.0, 2.0, -1.0]], dtype=dtype)
        v = torch.tensor([[0.5, -2.0, 3.0]], dtype=dtype)
        self.assertTrue(
            torch.equal(bilinear_contract(u, v, beta), torch.linalg.cross(u, v))
        )

    def test_feedback_degree_obstruction(self) -> None:
        growth = feedback_degree_growth(8)
        self.assertEqual(growth["triangular"], [2] * 8)
        self.assertEqual(growth["feedback_into_one_source"], list(range(2, 10)))
        self.assertEqual(
            growth["feedback_into_both_sources"],
            [2, 4, 8, 16, 32, 64, 128, 256],
        )


if __name__ == "__main__":
    unittest.main()
