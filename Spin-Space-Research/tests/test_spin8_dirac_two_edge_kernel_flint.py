"""FLINT replay contract for the two-edge exact local kernel jet."""

from __future__ import annotations

import unittest
from pathlib import Path

from spin8_dirac_two_edge_kernel_flint import FLINT_AVAILABLE, run


@unittest.skipUnless(FLINT_AVAILABLE, "python-flint is not installed")
class Spin8DiracTwoEdgeKernelFlintTests(unittest.TestCase):
    def test_flint_replays_sympy_jet(self) -> None:
        root = Path(__file__).parents[1]
        report = run(
            root
            / "artifacts/spin8_dirac_two_edge_all_sectors_coefficients_20260806.json",
            flint_threads=1,
        )
        self.assertTrue(report["passed"])
        self.assertTrue(report["checks"]["eg_block_factorization_matches"])


if __name__ == "__main__":
    unittest.main()
