from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from spin9_v5_cartan_certificate import (
    DEFAULT_COEFFICIENT_ARTIFACT,
    DEFAULT_RAY_ARTIFACT,
    bernstein_atlas_certificate,
    coefficient_bound_certificate,
    invariant_gap_coefficients,
)
from spin9_v5_cartan_reconstruction import (
    IDENTITY_PRIMES,
    MONOMIALS,
    load_coefficients,
    validate_polynomial_identity_mod_prime,
    validate_rational_coefficients,
)


class Spin9V5CartanCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.coefficients = load_coefficients(DEFAULT_COEFFICIENT_ARTIFACT)

    def test_reconstruction_artifact_and_unused_prime(self) -> None:
        self.assertEqual(len(self.coefficients), 631)
        self.assertEqual(len(self.coefficients), len(MONOMIALS))
        self.assertTrue(all(value.denominator == 1 for value in self.coefficients))
        self.assertTrue(all(value != 0 for value in self.coefficients))
        self.assertTrue(validate_rational_coefficients(self.coefficients))

        payload = json.loads(DEFAULT_COEFFICIENT_ARTIFACT.read_text(encoding="utf-8"))
        self.assertTrue(payload["passed"])
        self.assertFalse(payload["characteristic_zero_identity_certified"])
        self.assertEqual(payload["maximum_numerator_digits"], 27)

    def test_strict_six_cell_bernstein_atlas(self) -> None:
        gap = invariant_gap_coefficients(self.coefficients)
        self.assertEqual(len(gap), 631)

        atlas = bernstein_atlas_certificate(self.coefficients)
        self.assertEqual(atlas["power_multidegree"], [84, 84])
        self.assertEqual(atlas["power_term_count"], 1849)
        self.assertEqual(atlas["native_negative_coefficient_count"], 1792)
        self.assertEqual(atlas["native_zero_coefficient_count"], 85)
        self.assertEqual(atlas["split_count"], 5)
        self.assertEqual(atlas["leaf_count"], 6)
        self.assertTrue(atlas["all_leaf_controls_strictly_positive"])
        self.assertTrue(
            all(
                int(leaf["minimum_scaled_coefficient"]) > 0
                and leaf["zero_coefficient_count"] == 0
                for leaf in atlas["leaves"]
            )
        )

    def test_characteristic_zero_coefficient_bound(self) -> None:
        bound = coefficient_bound_certificate(self.coefficients)
        self.assertEqual(len(bound["information_row_l1_norms"]), 36)
        self.assertEqual(bound["base_information_determinant"], "17179869184")
        self.assertTrue(bound["two_square_root_embeddings_checked_per_prime"])
        self.assertTrue(bound["prime_product_exceeds_twice_residual_bound"])
        self.assertGreater(
            math.prod(IDENTITY_PRIMES),
            2 * int(bound["residual_coefficient_bound"]),
        )

    def test_full_twenty_prime_raw_identity(self) -> None:
        for prime in IDENTITY_PRIMES:
            with self.subTest(prime=prime):
                self.assertTrue(
                    validate_polynomial_identity_mod_prime(
                        self.coefficients,
                        prime,
                    )
                )

    def test_promoted_artifact_and_claim_boundaries(self) -> None:
        artifact = (
            Path(__file__).parents[1]
            / "artifacts"
            / "spin9_v5_cartan_certificate_20260811.json"
        )
        report = json.loads(artifact.read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])
        self.assertTrue(report["characteristic_zero_identity"]["passed"])
        self.assertTrue(report["all_v5_shapes_certified"])
        self.assertTrue(
            report["symmetric_candidate_comparison"][
                "candidate_beats_entire_pure_v5_family"
            ]
        )
        self.assertFalse(report["full_v1_plus_v5_slice_certified"])
        self.assertFalse(report["global_grassmann_quotient_certified"])
        self.assertFalse(report["global_rank_three_optimum_certified"])

        ray = json.loads(DEFAULT_RAY_ARTIFACT.read_text(encoding="utf-8"))
        self.assertTrue(
            ray["symmetric_candidate_comparison"]["ratio_exceeds_101_over_100"][
                "positive"
            ]
        )


if __name__ == "__main__":
    unittest.main()
