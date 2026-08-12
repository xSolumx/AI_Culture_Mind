"""Contracts for the complete finite-edge radical elimination."""

from __future__ import annotations

import unittest
from pathlib import Path

import sympy as sp
import torch

from spin8_dirac_two_edge_finite import (
    _sample_six_cube,
    exact_finite_components,
    exact_radical_elimination_certificate,
    torch_finite_certificates,
)
from spin8_dirac_two_edge_kernel import TorchChannels, load_sectors


class Spin8DiracTwoEdgeFiniteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).parents[1]
        cls.sectors = load_sectors(
            root / "artifacts/spin8_dirac_two_edge_all_sectors_coefficients_20260806.json"
        )

    def test_exact_radical_elimination_and_degrees(self) -> None:
        report = exact_radical_elimination_certificate(self.sectors)
        self.assertTrue(report["passed"])
        self.assertEqual(report["maximum_center_degree"], 6)
        self.assertEqual(report["maximum_square_margin_degree"], 12)

    def test_torch_center_and_square_margin_match_exact(self) -> None:
        evaluator = TorchChannels(self.sectors, torch.device("cpu"))
        point = tuple(
            map(sp.Rational, ("25/169", "9/25", "49/289", "64/361", "9/49"))
        )
        y = sp.Rational(3, 5)
        observed_center, observed_square = torch_finite_certificates(
            evaluator,
            torch.tensor([[float(value) for value in point]], dtype=torch.float64),
            torch.tensor([float(y)], dtype=torch.float64),
        )
        for channel in range(4):
            exact = exact_finite_components(self.sectors, point, channel, y)
            self.assertAlmostEqual(float(exact[4]), float(observed_center[0, channel]), 10)
            self.assertAlmostEqual(float(exact[5]), float(observed_square[0, channel]), 8)

    def test_six_cube_sampler_advances(self) -> None:
        random = torch.Generator(device="cpu")
        random.manual_seed(31)
        first = _sample_six_cube(32, random, torch.device("cpu"))
        second = _sample_six_cube(32, random, torch.device("cpu"))
        self.assertFalse(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
