from __future__ import annotations

import unittest

import sympy as sp

from spin9_v5_ray_certificate import _quadratic_field_sign_witness, diagnostics


class Spin9V5RayCertificateTests(unittest.TestCase):
    def test_quadratic_sign_witness_combines_split_denominators(self) -> None:
        q = sp.sqrt(241)
        c_star = (q - 17) / 24
        candidate = sp.simplify(
            (1 - c_star) ** 10
            * (c_star + 2) ** 5
            * (2 * c_star + 1) ** 3
            / 32
        )
        split_denominator_gap = sp.expand(candidate - sp.Rational(26, 25))

        witness = _quadratic_field_sign_witness(split_denominator_gap, 241)

        self.assertTrue(witness["positive"], witness)
        self.assertGreater(int(witness["denominator"]), 1)

        exact_sign_cases = (
            (sp.Rational(3, 5), True),
            (-sp.Rational(3, 5), False),
            (sp.sqrt(2) - 1, True),
            (1 - sp.sqrt(2), False),
            (2 - sp.sqrt(2), True),
            (sp.sqrt(2) - 2, False),
        )
        for value, expected in exact_sign_cases:
            self.assertEqual(
                _quadratic_field_sign_witness(value, 2)["positive"],
                expected,
            )

    def test_exact_complete_ray_bounds(self) -> None:
        report = diagnostics()
        self.assertTrue(report["passed"], report)

        zero = report["zero_cubic_ray"]
        self.assertEqual(zero["gap_quotient_real_root_count"], 0)
        self.assertEqual(zero["information_block_sizes"], [16, 20])
        self.assertTrue(zero["numerator_is_rational"])

        axis = report["axisymmetric_ray"]
        self.assertEqual(axis["upper_gap_norm_degree"], 160)
        self.assertEqual(axis["upper_gap_norm_real_root_count"], 0)
        self.assertEqual(axis["information_block_sizes"], [6, 10, 10, 10])
        self.assertFalse(axis["cayley_null_global_ray_maximum"])
        self.assertTrue(axis["cayley_null_challenger_sign_witness"]["positive"])

        comparison = report["symmetric_candidate_comparison"]
        self.assertTrue(comparison["ratio_exceeds_101_over_100"]["positive"])
        self.assertTrue(comparison["candidate_beats_both_complete_rays"])

        self.assertFalse(report["all_v5_shapes_certified"])
        self.assertFalse(report["global_grassmann_quotient_certified"])
        self.assertFalse(report["global_rank_three_optimum_certified"])


if __name__ == "__main__":
    unittest.main()
