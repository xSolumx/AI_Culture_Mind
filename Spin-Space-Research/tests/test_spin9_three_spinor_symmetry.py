from __future__ import annotations

import unittest

from spin9_three_spinor_symmetry import diagnostics


class Spin9ThreeSpinorSymmetryTests(unittest.TestCase):
    def test_exact_cayley_null_stabilizer_and_branching(self) -> None:
        report = diagnostics()
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["plane_stabilizer_dimension"], 3)
        self.assertEqual(report["pointwise_stabilizer_dimension"], 0)
        self.assertEqual(
            report["casimir_eigenvalue_multiplicities"], {12: 14, 6: 10, 2: 12}
        )
        self.assertEqual(report["adjoint_branching"], "2*V7 + 2*V5 + 4*V3")
        curve = report["curve_stabilizer_certificate"]
        self.assertEqual(curve["symbolic_nullity"], 3)
        self.assertEqual(curve["induced_action_span_rank"], 3)
        self.assertTrue(curve["rank_witness_identity"])
        self.assertTrue(curve["curve_plane_stabilizer_is_full_so3"])
        self.assertFalse(report["whole_curve_fixed_subgroup_claimed"])


if __name__ == "__main__":
    unittest.main()
