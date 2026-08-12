from __future__ import annotations

import unittest

from spin9_three_spinor_conditioning import diagnostics


class Spin9ThreeSpinorConditioningTests(unittest.TestCase):
    def test_exact_symmetric_family_certificate(self) -> None:
        report = diagnostics()
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["information_block_sizes"], [6, 10, 10, 10])
        self.assertEqual(report["trace"], "27")
        self.assertEqual(report["unique_maximizer"], "(-17 + sqrt(241))/24")
        self.assertEqual(report["spectral_factor_degrees"], [2, 2, 4])
        self.assertEqual(report["spectral_factor_multiplicities"], [7, 5, 3])
        self.assertFalse(report["complete_equiangular_locus_claimed"])
        self.assertFalse(report["global_all_triples_optimality_claimed"])


if __name__ == "__main__":
    unittest.main()
