from __future__ import annotations

import json
import os
import unittest

from spin9_v1_v5_char0 import (
    DEFAULT_COEFFICIENT_ARTIFACT,
    DEFAULT_OUTPUT,
    IDENTITY_DEGREE,
    SHAPE_MONOMIALS,
    WEIGHT_GATES,
    _direction_pairs,
    coefficient_bound_certificate,
    validate_identity_mod_prime,
)
from spin9_v1_v5_reconstruction import load_coefficients


class Spin9V1V5CharacteristicZeroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.coefficients = load_coefficients(DEFAULT_COEFFICIENT_ARTIFACT)
        cls.report = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))

    def test_coefficient_bound_replays(self) -> None:
        bound = coefficient_bound_certificate(self.coefficients)
        self.assertEqual(bound, self.report["coefficient_bound"])
        self.assertEqual(
            bound["base_information_block_determinants"],
            [65_536, 262_144],
        )
        self.assertTrue(bound["information_entries_in_Z_sqrt2"])
        self.assertGreater(
            int(bound["identity_prime_product"]),
            2 * int(bound["residual_coefficient_bound"]),
        )

    def test_graded_prime_receipts_are_complete(self) -> None:
        primes = self.report["coefficient_bound"]["identity_primes"]
        rows = self.report["prime_rows"]
        self.assertEqual([row["prime"] for row in rows], primes)
        self.assertEqual(len(_direction_pairs()), len(SHAPE_MONOMIALS))
        expected_points_per_embedding = 316_243
        for row in rows:
            with self.subTest(prime=row["prime"]):
                self.assertTrue(row["passed"])
                self.assertEqual(row["raw_point_count"], 632_486)
                self.assertEqual(
                    [gate["maximum_weight"] for gate in row["graded_rank_gates"]],
                    list(WEIGHT_GATES),
                )
                self.assertTrue(
                    all(
                        gate["rank"] == gate["direction_count"]
                        for gate in row["graded_rank_gates"]
                    )
                )
                self.assertEqual(
                    row["graded_rank_gates"][-1]["direction_count"],
                    len(SHAPE_MONOMIALS),
                )
                roots = [embedding["sqrt2_root"] for embedding in row["embeddings"]]
                self.assertEqual(sum(roots) % row["prime"], 0)
                self.assertTrue(
                    all(root**2 % row["prime"] == 2 for root in roots)
                )
                self.assertTrue(
                    all(
                        embedding["passed"]
                        and embedding["raw_point_count"]
                        == expected_points_per_embedding
                        for embedding in row["embeddings"]
                    )
                )

    def test_promoted_characteristic_zero_scope(self) -> None:
        self.assertEqual(self.report["raw_total_degree_bound"], IDENTITY_DEGREE)
        self.assertTrue(self.report["full_required_prime_set_checked"])
        self.assertTrue(self.report["characteristic_zero_identity_certified"])
        self.assertTrue(self.report["passed"])

    @unittest.skipUnless(
        os.environ.get("SPIN9_FULL_CHAR0_REPLAY") == "1",
        "set SPIN9_FULL_CHAR0_REPLAY=1 for the 22-prime raw determinant replay",
    )
    def test_full_raw_identity_replay(self) -> None:
        for prime in self.report["coefficient_bound"]["identity_primes"]:
            with self.subTest(prime=prime):
                row = validate_identity_mod_prime((prime, self.coefficients))
                expected = next(
                    stored
                    for stored in self.report["prime_rows"]
                    if stored["prime"] == prime
                )
                self.assertEqual(row, expected)


if __name__ == "__main__":
    unittest.main()
