"""Compact tests for the extended-core adaptive dominance atlas."""

from __future__ import annotations

import unittest
from fractions import Fraction
from pathlib import Path

from spin8_dirac_endpoint_octet_core_dominance_atlas import (
    _box_intervals,
    verify_report,
)

ROOT = Path(__file__).resolve().parents[1]
COEFFICIENT_DIR = ROOT / "artifacts" / "spin8_dirac_unrestricted_coefficients_20260807"
ARTIFACT = (
    ROOT / "artifacts" / "spin8_dirac_endpoint_octet_core_dominance_atlas_20260816.json"
)


class EndpointOctetCoreDominanceAtlasTests(unittest.TestCase):
    def test_affine_path_intervals_refine_exactly(self) -> None:
        self.assertEqual(
            _box_intervals("00001"),
            [
                ["1/8", "1/2"],
                ["1/8", "1/2"],
                ["1/8", "1/2"],
                ["1/8", "1/2"],
                ["1/2", "7/8"],
            ],
        )
        self.assertEqual(
            _box_intervals("00001/00001/00001/00001/00001"),
            [
                ["1/8", "19/128"],
                ["1/8", "19/128"],
                ["1/8", "19/128"],
                ["1/8", "19/128"],
                ["109/128", "7/8"],
            ],
        )

    def test_stored_adaptive_atlas_is_a_complete_compact_replay(self) -> None:
        verification = verify_report(ARTIFACT, coefficient_dir=COEFFICIENT_DIR)
        self.assertIs(verification["verified"], True)
        self.assertEqual(verification["failures"], [])
        self.assertIs(
            verification["tree_verification"]["complete_prefix_tree_cover"], True
        )
        self.assertEqual(verification["certified_leaf_count"], 2140)
        self.assertEqual(verification["rejected_basis_node_count"], 68)
        self.assertGreater(Fraction(verification["minimum_physical_gap_lower"]), 0)
        self.assertIs(
            verification["full_transform_replay_required_for_independent_sign_check"],
            True,
        )


if __name__ == "__main__":
    unittest.main()
