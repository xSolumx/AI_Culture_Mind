import hashlib
import json
import unittest
from pathlib import Path

from spin8_triality_2a5_closure import certificate

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "spin8_triality_2a5_closure_20260817.json"
EXPECTED_ARTIFACT_SHA256 = (
    "ff238d047c94362136d70a9c57ae41c832d63bf15fb6263d0019456a9351cfc9"
)


class Spin8TrialityBinaryIcosahedralClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = certificate()
        cls.stored = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_exact_replay_matches_artifact(self) -> None:
        self.assertEqual(self.report, self.stored)
        self.assertTrue(self.report["passed"])

    def test_binary_icosahedral_input_is_exact_and_perfect(self) -> None:
        source = self.report["binary_icosahedral_input"]
        self.assertEqual(source["order"], 120)
        self.assertEqual(source["structure"], "2.A5 = SL(2,5)")
        self.assertTrue(source["checks"]["a_squared_is_minus_one"])
        self.assertTrue(source["checks"]["b_cubed_is_minus_one"])
        self.assertTrue(source["checks"]["ab_fifth_is_minus_one"])
        self.assertTrue(source["checks"]["derived_subgroup_is_whole_group"])

    def test_first_block_is_the_600_cell_rotation_group(self) -> None:
        block = self.report["first_four_dimensional_block"]
        self.assertEqual(block["orbit_order"], 120)
        self.assertEqual(block["permutation_group_order"], 7200)
        self.assertEqual(block["positive_image_order"], 120)
        self.assertEqual(block["commuting_binary_icosahedral_factor_order"], 120)
        self.assertEqual(block["central_factor_intersection_order"], 2)
        self.assertEqual(block["center_order"], 2)
        self.assertEqual(block["derived_subgroup_order"], 7200)
        self.assertTrue(block["checks"]["central_product_order_is_7200"])
        self.assertTrue(block["checks"]["positive_and_negative_images_coincide"])
        self.assertTrue(
            block["checks"]["orbit_is_exact_binary_icosahedral_root_set"]
        )

    def test_second_block_is_one_binary_icosahedral_image(self) -> None:
        block = self.report["second_four_dimensional_block"]
        self.assertEqual(block["orbit_order"], 120)
        self.assertEqual(block["permutation_group_order"], 120)
        self.assertEqual(block["center_order"], 2)
        self.assertEqual(block["derived_subgroup_order"], 120)
        self.assertTrue(block["checks"]["positive_and_negative_images_coincide"])

    def test_full_closure_is_the_direct_product_and_has_klein_center(self) -> None:
        group = self.report["full_triality_closure"]
        self.assertEqual(group["order"], 864000)
        self.assertEqual(group["block_projection_orders"], [7200, 120])
        self.assertEqual(group["faithful_permutation_degree"], 240)
        self.assertEqual(group["orbit_orders"], [120, 120])
        self.assertEqual(group["center_order"], 4)
        self.assertEqual(group["derived_subgroup_order"], 864000)
        self.assertEqual(
            group["structure"],
            "((2.A5 x 2.A5)/C2_diagonal) x 2.A5",
        )
        self.assertEqual(group["center_block_signatures"], ["++", "+-", "-+", "--"])
        self.assertTrue(group["checks"]["order_equals_product_of_block_projection_orders"])
        self.assertTrue(group["checks"]["center_is_independent_block_signs"])
        self.assertTrue(group["checks"]["all_six_generators_are_exactly_orthogonal"])
        self.assertTrue(group["checks"]["all_six_generators_have_determinant_one"])

    def test_claim_boundary_and_artifact_hash(self) -> None:
        digest = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_ARTIFACT_SHA256)
        nonclaims = self.report["claim_scope"]["not_claimed"]
        self.assertIn(
            "discovery of a previously unknown abstract finite group", nonclaims
        )
        self.assertIn("an irreducible eight-dimensional closure", nonclaims)
        self.assertIn("an ML-quality or kernel-speed advantage", nonclaims)


if __name__ == "__main__":
    unittest.main()
