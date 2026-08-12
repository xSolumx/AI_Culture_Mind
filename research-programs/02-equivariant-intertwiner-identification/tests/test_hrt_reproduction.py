import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hrt_reproduction import (
    build_report,
    finite_weyl_composition_residual,
    finite_weyl_matrix,
    gaussian,
    rank_diagnostic,
    shift_columns,
    symmetric_configuration,
)


class HRTReproductionTests(unittest.TestCase):
    def test_symmetric_family_has_expected_cardinality(self) -> None:
        self.assertEqual(len(symmetric_configuration(1)), 5)
        self.assertEqual(len(symmetric_configuration(3)), 9)

    def test_sampled_gaussian_shifts_are_full_rank(self) -> None:
        points = [(0.0, 0.0), (0.0, 1.0), (0.73, -0.41), (1.17, 0.22)]
        _, columns = shift_columns(points, gaussian, grid_size=4096)
        diagnostic = rank_diagnostic(columns)
        self.assertTrue(diagnostic["full_column_rank"])

    def test_report_is_explicitly_numerical_only(self) -> None:
        report = build_report(grid_size=2048)
        self.assertEqual(report["status"], "numerical-sanity-check-only")
        self.assertIn("the arbitrary-L2 HRT theorem", report["claim_boundary"]["does_not_establish"])

    def test_finite_weyl_projective_composition(self) -> None:
        residual = finite_weyl_composition_residual(32, (0, 1), (1, 0))
        self.assertLess(residual, 1e-12)
        self.assertTrue(np.allclose(finite_weyl_matrix(32, 0, 0), np.eye(32)))


if __name__ == "__main__":
    unittest.main()
