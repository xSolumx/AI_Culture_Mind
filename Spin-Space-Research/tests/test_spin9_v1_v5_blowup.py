from __future__ import annotations

import json
import math
import unittest
from fractions import Fraction

import sympy as sp

from spin9_v1_v5_blowup import (
    COMPACT_DEGREE,
    DEFAULT_COEFFICIENT_ARTIFACT,
    ROOT,
    UPPER_DENOMINATOR,
    UPPER_NUMERATOR,
    certificate,
    compact_bernstein_controls,
    expected_factorization,
)

ARTIFACT = ROOT / "artifacts" / "spin9_v1_v5_blowup_20260811.json"


class Spin9V1V5BlowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = certificate(DEFAULT_COEFFICIENT_ARTIFACT)
        cls.artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_exact_boundary_certificate_replays(self) -> None:
        self.assertTrue(self.report["passed"], self.report)
        self.assertEqual(
            self.report["bound"],
            "R_A(d,w), R_B(d,w) < 26/25 on R^2",
        )
        self.assertTrue(
            self.report["symmetric_candidate_comparison"]
            ["candidate_exceeds_26_over_25"]["positive"]
        )

        families = {row["family"]: row for row in self.report["families"]}
        self.assertEqual(set(families), {"A", "B"})
        self.assertEqual(families["A"]["boundary_numerator_term_count"], 209)
        self.assertEqual(families["B"]["boundary_numerator_term_count"], 401)
        for row in families.values():
            self.assertEqual(row["boundary_numerator_maximum_total_degree"], 28)
            self.assertTrue(row["extracted_factorization_identity_passed"])
            self.assertTrue(row["all_quadrants_strictly_positive"])
            self.assertTrue(
                all(
                    chart["atlas"]["all_leaf_controls_strictly_positive"]
                    for chart in row["quadrant_charts"]
                )
            )

    def test_checked_artifact_and_claim_boundary(self) -> None:
        self.assertEqual(self.artifact, self.report)
        self.assertEqual(
            self.report["family_rows_sha256"],
            "7e2c7254a15cf35feea034c1801d8199097b0879c4e93d7c53fdd1304dcb3145",
        )
        self.assertFalse(
            self.report["raw_characteristic_zero_coupled_identity_certified"]
        )
        self.assertTrue(
            self.report["raw_characteristic_zero_boundary_identities_certified"]
        )
        self.assertFalse(self.report["global_coupled_determinant_theorem_claimed"])

    def test_projective_bernstein_transform_is_exact(self) -> None:
        d, w = sp.symbols("d w", real=True)
        t = Fraction(2, 5)
        a = Fraction(3, 7)
        rho = t / (1 - t)
        binomials = [math.comb(COMPACT_DEGREE, index) for index in range(29)]
        for family in ("A", "B"):
            normalized_numerator, norm, _ = expected_factorization(family)
            gap = sp.expand(
                UPPER_NUMERATOR * sp.denom(normalized_numerator) * norm**14
                - UPPER_DENOMINATOR * sp.numer(normalized_numerator)
            )
            for d_sign, w_sign in ((-1, 1), (1, -1)):
                controls, scale = compact_bernstein_controls(
                    gap,
                    d_sign,
                    w_sign,
                )
                bernstein_value = sum(
                    Fraction(int(controls[i, j]), scale)
                    * binomials[i]
                    * t**i
                    * (1 - t) ** (COMPACT_DEGREE - i)
                    * binomials[j]
                    * a**j
                    * (1 - a) ** (COMPACT_DEGREE - j)
                    for i in range(29)
                    for j in range(29)
                )
                direct_value = sp.Rational(
                    gap.subs(
                        {
                            d: d_sign * a * rho,
                            w: w_sign * (1 - a) * rho,
                        }
                    )
                    * (1 - t) ** COMPACT_DEGREE
                )
                self.assertEqual(sp.Rational(bernstein_value), direct_value)


if __name__ == "__main__":
    unittest.main()
