from __future__ import annotations

import unittest

import numpy as np

from spin9_dirac_clifford import build_spin9_clifford_system
from spin9_three_spinor_global_screen import (
    algebraic_candidate,
    candidate_log_determinant,
)


class Spin9ThreeSpinorGlobalScreenTests(unittest.TestCase):
    def test_candidate_matches_direct_information(self) -> None:
        c, spinors = algebraic_candidate()
        np.testing.assert_allclose(spinors @ spinors.T, np.eye(3), atol=1e-14)
        system = build_spin9_clifford_system()
        generators = system.doubled_spin_generators.astype(np.float64) / 2.0
        observations = np.einsum("aij,rj->rai", generators, spinors)
        information = np.einsum("rai,rbi->ab", observations, observations)
        sign, logdet = np.linalg.slogdet(information)
        self.assertEqual(sign, 1.0)
        self.assertAlmostEqual(logdet, candidate_log_determinant(c), places=12)

        hopf = np.einsum("ri,kij,rj->rk", spinors, system.involutions, spinors)
        expected = np.eye(3) + c * (np.ones((3, 3)) - np.eye(3))
        np.testing.assert_allclose(hopf @ hopf.T, expected, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
