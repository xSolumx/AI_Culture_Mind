"""Regression tests for exact spinor branching over the global component atlas."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from spin_dirac_a5_spinors import branching_atlas

ARTIFACT = (
    Path(__file__).parent
    / "experiments"
    / "artifacts"
    / "spin_dirac_a5_spinors_20260816.json"
)


class SpinDiracA5SpinorTests(unittest.TestCase):
    """Falsify block branching, chirality, and Clifford-identity regressions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.live = branching_atlas()
        cls.saved = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_quaternionic_base_blocks_come_from_su2_weights(self) -> None:
        blocks = self.live["base_spinor_blocks"]
        self.assertEqual(
            blocks["2_H"],
            {
                "parity": "even",
                "plus": {"1": 2},
                "minus": {"2": 1},
                "su2_vector_highest_weight": 1,
                "su2_plus_decomposition": {"0": 2},
                "su2_minus_decomposition": {"1": 1},
            },
        )
        self.assertEqual(blocks["4_H"]["plus"], {"1": 3, "5": 1})
        self.assertEqual(blocks["4_H"]["minus"], {"4_spin": 2})
        self.assertEqual(blocks["6_H"]["plus"], {"1": 4, "4_vector": 2, "5": 4})
        self.assertEqual(blocks["6_H"]["minus"], {"4_spin": 2, "6_spin": 4})
        self.assertEqual(
            blocks["6_H"]["su2_plus_decomposition"],
            {"0": 4, "4": 2, "8": 2},
        )
        self.assertEqual(
            blocks["6_H"]["su2_minus_decomposition"],
            {"3": 1, "5": 3, "9": 1},
        )

    def test_all_245_branchings_pass_independent_clifford_identities(self) -> None:
        self.assertTrue(self.live["passed"])
        self.assertTrue(self.live["verification"]["all_checks_pass"])
        self.assertEqual(self.live["verification"]["components_checked"], 245)
        expected = {
            "3": (3, 3, 1, 0, 0),
            "8": (25, 32, 16, 7, 7),
            "9": (32, 32, 16, 0, 0),
            "10": (42, 42, 21, 0, 0),
            "11": (59, 59, 29, 0, 0),
            "12": (84, 98, 44, 14, 14),
        }
        for dimension, counts in expected.items():
            record = self.live["by_dimension"][dimension]
            self.assertEqual(
                (
                    record["orthogonal_types"],
                    record["spin_conjugacy_components"],
                    record["types_with_invariant_spinors"],
                    record["orientation_split_types"],
                    record["orientation_splits_distinguished_by_chiral_character"],
                ),
                counts,
            )
            self.assertTrue(
                all(
                    all(branching["checks"].values()) for branching in record["records"]
                )
            )

    def test_fixed_spin3_ladder_remains_defining_spinor_isotypic(self) -> None:
        expected_multiplicities = {3: 1, 8: 4, 9: 8, 10: 8, 11: 16, 12: 16}
        for dimension, multiplicity in expected_multiplicities.items():
            fixed_id = "1*3_R" if dimension == 3 else f"{dimension - 3}*1_R+1*3_R"
            branching = next(
                record
                for record in self.live["by_dimension"][str(dimension)]["records"]
                if record["id"] == fixed_id
            )
            if dimension % 2:
                self.assertEqual(branching["spinor"], {"2": multiplicity})
                self.assertEqual(branching["invariant_spinors"], 0)
            else:
                self.assertEqual(
                    branching["orientation_A"]["plus"], {"2": multiplicity}
                )
                self.assertEqual(
                    branching["orientation_A"]["minus"], {"2": multiplicity}
                )
                self.assertIsNone(branching["orientation_B"])

    def test_orientation_split_is_detected_by_chiral_branching(self) -> None:
        pure_quaternionic = next(
            record
            for record in self.live["by_dimension"]["8"]["records"]
            if record["id"] == "1*4_H"
        )
        self.assertTrue(pure_quaternionic["orientation_split"])
        self.assertTrue(
            pure_quaternionic["chirality_character_distinguishes_orientations"]
        )
        self.assertEqual(
            pure_quaternionic["orientation_A"],
            {
                "plus": {"1": 3, "5": 1},
                "minus": {"4_spin": 2},
            },
        )
        self.assertEqual(
            pure_quaternionic["orientation_B"],
            {
                "plus": {"4_spin": 2},
                "minus": {"1": 3, "5": 1},
            },
        )

    def test_artifact_matches_full_live_replay_and_keeps_geometric_boundary(
        self,
    ) -> None:
        self.assertEqual(self.saved, self.live)
        self.assertIn(
            "a geometric Dirac spectrum without a manifold and Dirac operator",
            self.live["claim_scope"]["not_claimed"],
        )
        self.assertIn(
            "that invariant spinors are automatically zero modes on an arbitrary geometry",
            self.live["claim_scope"]["not_claimed"],
        )


if __name__ == "__main__":
    unittest.main()
