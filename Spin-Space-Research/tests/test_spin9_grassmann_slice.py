from __future__ import annotations

import unittest

from spin9_grassmann_slice import diagnostics


class Spin9GrassmannSliceTests(unittest.TestCase):
    def test_exact_normal_slice_branching(self) -> None:
        report = diagnostics()
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["orbit_rank"], 33)
        self.assertEqual(report["normal_slice_dimension"], 6)
        self.assertEqual(
            report["slice_casimir_eigenvalue_multiplicities"], {6: 5, 0: 1}
        )
        self.assertEqual(report["slice_branching"], "V1 + V5")
        self.assertTrue(report["curve_tangent_is_trivial_slice_direction"])
        self.assertEqual(report["local_invariant_degrees"], [1, 2, 3])
        self.assertFalse(report["global_grassmann_quotient_solved"])


if __name__ == "__main__":
    unittest.main()
