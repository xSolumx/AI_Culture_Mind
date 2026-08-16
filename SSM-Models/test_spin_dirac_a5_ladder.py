from __future__ import annotations

import unittest

import numpy as np
from spin_dirac_a5_ladder import (
    LADDER_DIMENSIONS,
    QUATERNION_TOLERANCE,
    branching_diagnostics,
    build_clifford_ladder,
    clifford_stage_diagnostics,
    diagnostics,
    icosahedral_spin_generators,
    quaternion_multiply,
    spin_matrix_from_quaternion,
)


class SpinDiracA5LadderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stages = build_clifford_ladder()
        cls.report = diagnostics()

    def test_exact_clifford_ladder_and_branching(self) -> None:
        self.assertEqual(tuple(self.stages), LADDER_DIMENSIONS)
        expected_spinor_dimensions = {3: 2, 8: 16, 9: 16, 10: 32, 11: 32, 12: 64}
        for dimension, spinor_dimension in expected_spinor_dimensions.items():
            stage = self.stages[dimension]
            self.assertEqual(
                stage.gamma.shape, (dimension, spinor_dimension, spinor_dimension)
            )
            stage_report = clifford_stage_diagnostics(stage)
            self.assertTrue(stage_report["passed"])
            self.assertEqual(stage_report["clifford_max_abs"], 0.0)
            self.assertEqual(stage_report["spin_vector_covariance_max_abs"], 0.0)
        self.assertTrue(branching_diagnostics(self.stages)["passed"])

    def test_a5_projection_and_binary_lift_keep_distinct_centers(self) -> None:
        a5 = self.report["a5"]
        self.assertTrue(a5["passed"])
        self.assertEqual(
            a5["abstract_a5_permutation_certificate"]["generated_group_order"], 60
        )
        self.assertEqual(a5["binary_lift"]["generated_group_order_numerical"], 120)
        self.assertEqual(a5["binary_lift"]["projective_group_order_numerical"], 60)
        for stage_report in a5["spinor_restrictions"].values():
            self.assertTrue(stage_report["passed"])
            self.assertLessEqual(
                max(stage_report["lifted_presentation_residuals"].values()),
                QUATERNION_TOLERANCE,
            )
            self.assertEqual(stage_report["central_minus_one_visibility_max_abs"], 0.0)
        expected_centralizers = {3: 0, 8: 10, 9: 15, 10: 21, 11: 28, 12: 36}
        centralizers = a5["vector_centralizers"]
        self.assertTrue(centralizers["passed"])
        for dimension, expected in expected_centralizers.items():
            self.assertEqual(
                centralizers["stages"][str(dimension)][
                    "centralizer_dimension_numerical"
                ],
                expected,
            )

    def test_spin3_restriction_is_isotypic_and_triality_stays_exceptional(self) -> None:
        restrictions = self.report["a5"]["spinor_restrictions"]
        expected_multiplicities = {3: 1, 8: 8, 9: 8, 10: 16, 11: 16, 12: 32}
        for dimension, multiplicity in expected_multiplicities.items():
            stage_report = restrictions[str(dimension)]
            self.assertEqual(
                stage_report["spin3_fundamental_multiplicity"], multiplicity
            )
            self.assertLessEqual(
                stage_report["spin3_isotypic_character_max_abs"],
                QUATERNION_TOLERANCE,
            )
            self.assertEqual(stage_report["triality_dimension_match"], dimension == 8)

    def test_quaternion_formula_is_a_spin_representation(self) -> None:
        a, b = icosahedral_spin_generators()
        ab = quaternion_multiply(a, b)
        for stage in self.stages.values():
            expected = spin_matrix_from_quaternion(stage, ab)
            observed = spin_matrix_from_quaternion(
                stage, a
            ) @ spin_matrix_from_quaternion(stage, b)
            np.testing.assert_allclose(
                observed, expected, rtol=0.0, atol=QUATERNION_TOLERANCE
            )


if __name__ == "__main__":
    unittest.main()
