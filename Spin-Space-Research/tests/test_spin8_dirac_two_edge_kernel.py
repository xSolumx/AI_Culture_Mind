"""Tests for the exact two-edge boundary-kernel scalar reduction."""

from __future__ import annotations

import unittest
from pathlib import Path

import sympy as sp
import torch

from spin8_dirac_two_edge_kernel import (
    TorchChannels,
    _sample_points,
    exact_channels,
    exact_even_i2_curvatures,
    exact_local_kernel_certificate,
    exact_quadratic_schur_counterexample,
    load_sectors,
)


class Spin8DiracTwoEdgeKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).parents[1]
        cls.sectors = load_sectors(
            root / "artifacts/spin8_dirac_two_edge_all_sectors_coefficients_20260806.json"
        )

    def test_orthonormal_equality_line_annihilates_odd_derivative(self) -> None:
        for z in (sp.Rational(0), sp.Rational(1, 5), sp.Rational(4, 5), sp.Rational(1)):
            lambdas, mus = exact_channels(self.sectors, (0, 0, 0, 0, z))
            self.assertEqual(lambdas, (0, 0, 0, 0))
            self.assertEqual(mus, (0, 0, 0, 0))

    def test_float_evaluator_matches_exact_interior_values(self) -> None:
        evaluator = TorchChannels(self.sectors, torch.device("cpu"))
        point = (
            sp.Rational(1, 7),
            sp.Rational(2, 9),
            sp.Rational(3, 11),
            sp.Rational(4, 13),
            sp.Rational(5, 17),
        )
        expected_lambda, expected_mu = exact_channels(self.sectors, point)
        observed_lambda, observed_mu = evaluator(
            torch.tensor([[float(value) for value in point]], dtype=torch.float64)
        )
        torch.testing.assert_close(
            observed_lambda[0],
            torch.tensor(
                [float(value) for value in expected_lambda], dtype=torch.float64
            ),
            rtol=2e-12,
            atol=2e-12,
        )
        torch.testing.assert_close(
            observed_mu[0],
            torch.tensor([float(value) for value in expected_mu], dtype=torch.float64),
            rtol=2e-12,
            atol=2e-12,
        )

    def test_exact_local_kernel_certificate(self) -> None:
        report = exact_local_kernel_certificate(self.sectors)
        self.assertTrue(report["passed"])
        self.assertEqual(report["eg_block_determinant"], "4*(z - 9)**3*(z - 1)")
        self.assertTrue(report["odd_quadratic_vanishes_on_degenerate_tangent"])
        self.assertTrue(report["endpoint_null_is_lifted_quartically"])
        self.assertTrue(report["new_edge_matches_a_direction_stiffness"])

    def test_float_curvature_matches_exact_interior_values(self) -> None:
        evaluator = TorchChannels(self.sectors, torch.device("cpu"))
        point = tuple(
            map(
                sp.Rational,
                ("1/7", "2/9", "3/11", "4/13", "5/17"),
            )
        )
        expected = exact_even_i2_curvatures(self.sectors, point)
        observed = evaluator.curvatures(
            torch.tensor([[float(value) for value in point]], dtype=torch.float64)
        )
        torch.testing.assert_close(
            observed[0],
            torch.tensor([float(value) for value in expected], dtype=torch.float64),
            rtol=2e-12,
            atol=2e-12,
        )

    def test_exact_counterexample_rejects_only_quadratic_schur_strategy(self) -> None:
        report = exact_quadratic_schur_counterexample(self.sectors)
        self.assertTrue(report["passed"])
        self.assertTrue(all(value < 0 for value in report["decimal_residuals"]))
        self.assertIn("does not falsify", report["scope_boundary"])

    def test_sampler_advances_instead_of_repeating_batches(self) -> None:
        random = torch.Generator(device="cpu")
        random.manual_seed(17)
        first = _sample_points(32, random, torch.device("cpu"))
        second = _sample_points(32, random, torch.device("cpu"))
        self.assertFalse(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
