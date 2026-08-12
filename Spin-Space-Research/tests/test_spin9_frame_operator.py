from __future__ import annotations

import unittest

from spin9_frame_operator import diagnostics


class Spin9FrameOperatorTests(unittest.TestCase):
    def test_exact_frame_operator_certificate(self) -> None:
        report = diagnostics()
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["symmetric_basis_size"], 136)
        self.assertEqual(set(report["information_map_ranks"].values()), {127})
        self.assertEqual(report["information_map_kernel_dimension"], 9)
        self.assertFalse(report["exact_three_probe_attains_approximate_optimum"])
        self.assertFalse(report["global_exact_three_probe_optimum_claimed"])


if __name__ == "__main__":
    unittest.main()
