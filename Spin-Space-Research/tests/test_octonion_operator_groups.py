import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from octonion_operator_groups import (
    certificate,
    compose,
    matrix_from_monomial,
    monomial_determinant,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "octonion_operator_groups_20260817.json"
EXPECTED_ARTIFACT_SHA256 = (
    "835d0535d7827834ecd6b707984a7f3ae0eeac651f6fec4c3ff6f4ba74796a92"
)


class OctonionOperatorGroupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = certificate()
        cls.stored = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_complete_exact_replay_matches_artifact(self) -> None:
        self.assertEqual(self.report, self.stored)
        self.assertTrue(self.report["passed"])

    def test_operator_group_is_plus_extraspecial(self) -> None:
        group = self.report["left_operator_group"]
        self.assertEqual(group["order"], 128)
        self.assertEqual(group["structure"], "2_+^(1+6)")
        self.assertEqual(group["center_order"], 2)
        self.assertEqual(group["derived_subgroup_order"], 2)
        self.assertEqual(group["square_identity_count"], 72)
        self.assertEqual(group["ordered_volume_product"], list(range(-1, -9, -1)))
        self.assertEqual(
            group["element_order_distribution"], {"1": 1, "2": 71, "4": 56}
        )

    def test_signed_fano_extension_is_nonsplit(self) -> None:
        group = self.report["signed_basis_automorphism_group"]
        self.assertEqual(group["order"], 1344)
        self.assertEqual(group["projected_fano_permutation_order"], 168)
        self.assertEqual(group["projection_kernel_order"], 8)
        self.assertFalse(group["extension_splits"])
        self.assertFalse(group["split_diagnostic"]["split_complement_found"])
        self.assertEqual(
            group["split_diagnostic"]["lift_pair_generated_order_distribution"],
            {"1344": 64},
        )

    def test_generated_normalizer_is_split_and_perfect(self) -> None:
        group = self.report["generated_operator_normalizer"]
        self.assertEqual(group["order"], 21504)
        self.assertEqual(group["structure"], "2_+^(1+6):PSL(2,7)")
        self.assertTrue(group["extension_splits"])
        self.assertEqual(group["center_order"], 2)
        self.assertEqual(group["derived_subgroup_order"], 21504)
        self.assertEqual(group["determinant_distribution"], {"1": 21504})
        diagnostic = group["split_diagnostic"]
        self.assertEqual(diagnostic["order_two_lifts"], 24)
        self.assertEqual(diagnostic["order_three_lifts"], 16)
        self.assertEqual(diagnostic["lift_pairs_with_product_order_seven"], 192)
        self.assertEqual(diagnostic["lift_pairs_with_commutator_order_four"], 64)

    def test_compact_composition_is_exact_matrix_multiplication(self) -> None:
        left = (3, 4, 1, 2, -8, -7, -6, -5)
        right = (1, 3, 5, 7, 2, -4, -6, 8)
        product = compose(left, right)
        self.assertTrue(
            np.array_equal(
                matrix_from_monomial(product),
                matrix_from_monomial(left) @ matrix_from_monomial(right),
            )
        )
        self.assertEqual(monomial_determinant(left), 1)
        self.assertEqual(monomial_determinant(right), 1)

    def test_artifact_hash_and_claim_boundary(self) -> None:
        digest = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_ARTIFACT_SHA256)
        nonclaims = self.report["claim_scope"]["not_claimed"]
        self.assertIn(
            "that any abstract group in this report was previously unknown",
            nonclaims,
        )
        self.assertIn("a new simple group", nonclaims)


if __name__ == "__main__":
    unittest.main()
