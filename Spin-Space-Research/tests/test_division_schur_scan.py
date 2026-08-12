from __future__ import annotations

import json
import unittest

import torch

from division_schur_scan import (
    DEFAULT_OUTPUT,
    DivisionSchurTransition,
    apply_division_schur,
    compose_division_schur,
    diagnostics,
    division_product,
)


class DivisionSchurScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = diagnostics()
        cls.artifact = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))

    def test_exact_commutants_scan_and_gradients(self) -> None:
        self.assertTrue(self.report["passed"], self.report)
        expected_dimensions = {"complex": 2, "quaternion": 4}
        for algebra, dimension in expected_dimensions.items():
            row = self.report["algebras"][algebra]
            exact = row["exact_commutant"]
            numerical = row["numerical_scan"]
            self.assertEqual(exact["exact_commutant_dimension"], dimension)
            self.assertEqual(exact["right_multiplication_basis_rank"], dimension)
            self.assertTrue(exact["complete_right_division_algebra_basis"])
            self.assertTrue(exact["right_basis_commutes_with_left_generators"])
            self.assertLess(numerical["associativity_max_error"], 1e-12)
            self.assertLess(numerical["scan_recurrent_max_error"], 1e-11)
            self.assertLess(numerical["gradient_max_error"], 1e-10)

    def test_quaternionic_multiplicity_order_is_not_reversed(self) -> None:
        dtype = torch.float64
        one = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype)
        i = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=dtype)
        j = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=dtype)
        k = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=dtype)
        zero = torch.zeros(1, 4, dtype=dtype)
        before = DivisionSchurTransition(
            algebra="quaternion",
            left_action=one,
            right_multiplicity=i.reshape(1, 1, 4),
            drive=zero,
        )
        after = DivisionSchurTransition(
            algebra="quaternion",
            left_action=one,
            right_multiplicity=j.reshape(1, 1, 4),
            drive=zero,
        )

        composed = compose_division_schur(after, before)
        direct = apply_division_schur(
            after,
            apply_division_schur(before, one.reshape(1, 4)),
        )

        torch.testing.assert_close(
            composed.right_multiplicity.reshape(4),
            k,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(direct.reshape(4), k, rtol=0, atol=0)
        torch.testing.assert_close(
            division_product(j, i, "quaternion"),
            -k,
            rtol=0,
            atol=0,
        )

    def test_factored_actions_match_materialized_real_matrices(self) -> None:
        torch.manual_seed(91)
        dtype = torch.float64
        for algebra, dimension in (("complex", 2), ("quaternion", 4)):
            multiplicity = 2
            left = torch.randn(dimension, dtype=dtype)
            left = left / left.norm()
            right = torch.randn(
                multiplicity,
                multiplicity,
                dimension,
                dtype=dtype,
            )
            transition = DivisionSchurTransition(
                algebra=algebra,
                left_action=left,
                right_multiplicity=right,
                drive=torch.zeros(multiplicity, dimension, dtype=dtype),
            )
            width = multiplicity * dimension
            basis = torch.eye(width, dtype=dtype).reshape(
                width,
                multiplicity,
                dimension,
            )
            materialized = apply_division_schur(transition, basis).reshape(
                width,
                width,
            ).T
            state = torch.randn(multiplicity, dimension, dtype=dtype)
            factored = apply_division_schur(transition, state)
            dense = (materialized @ state.flatten()).reshape_as(state)
            torch.testing.assert_close(factored, dense, rtol=1e-12, atol=1e-12)

    def test_artifact_and_claim_boundary(self) -> None:
        self.assertEqual(self.artifact, self.report)
        self.assertFalse(
            self.report["automatic_arbitrary_representation_decomposition_implemented"]
        )
        self.assertFalse(self.report["sequence_model_superiority_claimed"])


if __name__ == "__main__":
    unittest.main()
