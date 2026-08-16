"""Regression tests for the exact binary-icosahedral Spin-component atlas."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from spin_dirac_a5_components import diagnostics

ARTIFACT = (
    Path(__file__).parent
    / "experiments"
    / "artifacts"
    / "spin_dirac_a5_components_20260816.json"
)


class SpinDiracA5ComponentTests(unittest.TestCase):
    """Falsify character, real-type, lift, and component-count regressions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.live = diagnostics()
        cls.saved = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_exact_character_and_mckay_certificates(self) -> None:
        certificate = self.live["character_certificate"]
        self.assertTrue(certificate["passed"])
        self.assertTrue(all(certificate["exact_checks"].values()))
        self.assertEqual(certificate["derived_subgroup_order"], 120)
        self.assertEqual(certificate["degree_square_sum"], 120)
        self.assertEqual(
            {
                record["label"]: (record["size"], record["element_order"])
                for record in certificate["conjugacy_classes"]
            },
            {
                "1A": (1, 1),
                "2A": (1, 2),
                "3A": (20, 3),
                "4A": (30, 4),
                "5A": (12, 5),
                "5B": (12, 5),
                "6A": (20, 6),
                "10A": (12, 10),
                "10B": (12, 10),
            },
        )
        character_types = {
            record["key"]: (
                record["complex_dimension"],
                record["frobenius_schur_indicator"],
                record["center_eigenvalue"],
            )
            for record in certificate["characters"]
        }
        self.assertEqual(
            character_types,
            {
                "1": (1, 1, 1),
                "2": (2, -1, -1),
                "2_prime": (2, -1, -1),
                "3": (3, 1, 1),
                "3_prime": (3, 1, 1),
                "4_vector": (4, 1, 1),
                "4_spin": (4, -1, -1),
                "5": (5, 1, 1),
                "6_spin": (6, -1, -1),
            },
        )
        self.assertEqual(
            {
                record["key"]: (
                    record["real_dimension"],
                    record["division_algebra"],
                    record["center_eigenvalue"],
                )
                for record in certificate["real_irreps"]
            },
            {
                "1_R": (1, "R", 1),
                "3_R": (3, "R", 1),
                "3p_R": (3, "R", 1),
                "4_R": (4, "R", 1),
                "5_R": (5, "R", 1),
                "2_H": (4, "H", -1),
                "2p_H": (4, "H", -1),
                "4_H": (8, "H", -1),
                "6_H": (12, "H", -1),
            },
        )
        self.assertEqual(
            certificate["tensor_by_defining_2"],
            {
                "1": {"2": 1},
                "2": {"1": 1, "3": 1},
                "2_prime": {"4_vector": 1},
                "3": {"2": 1, "4_spin": 1},
                "3_prime": {"6_spin": 1},
                "4_vector": {"2_prime": 1, "6_spin": 1},
                "4_spin": {"3": 1, "5": 1},
                "5": {"4_spin": 1, "6_spin": 1},
                "6_spin": {"3_prime": 1, "4_vector": 1, "5": 1},
            },
        )
        self.assertEqual(certificate["mckay_graph"]["type"], "affine_E8")
        self.assertEqual(certificate["mckay_graph"]["arm_lengths"], [1, 2, 5])
        self.assertTrue(certificate["mckay_graph"]["passed"])

    def test_global_component_counts_and_fixed_ladder(self) -> None:
        atlas = self.live["component_atlas"]
        self.assertTrue(atlas["passed"])
        expected_counts = {
            "3": (3, 3, 0, 3, 3),
            "8": (25, 32, 7, 13, 14),
            "9": (32, 32, 0, 18, 18),
            "10": (42, 42, 0, 22, 22),
            "11": (59, 59, 0, 27, 27),
            "12": (84, 98, 14, 35, 36),
        }
        for dimension, expected in expected_counts.items():
            record = atlas["by_dimension"][dimension]
            self.assertEqual(
                (
                    record["orthogonal_isomorphism_types"],
                    record["spin_conjugacy_components"],
                    record["orientation_split_types"],
                    record["a5_projecting_types"],
                    record["a5_projecting_spin_components"],
                ),
                expected,
            )
            self.assertTrue(all(record["checks"].values()))

            n = int(dimension)
            fixed_id = "1*3_R" if n == 3 else f"{n - 3}*1_R+1*3_R"
            fixed = next(
                component
                for component in record["components"]
                if component["id"] == fixed_id
            )
            self.assertTrue(fixed["a5_projecting"])
            self.assertEqual(fixed["spin_center_scalar_sign"], -1)
            self.assertEqual(fixed["homomorphism_kernel"], "trivial")
            self.assertEqual(fixed["spin_conjugacy_components"], 1)
            self.assertEqual(fixed["centralizer_lie_dimension"], (n - 3) * (n - 4) // 2)

    def test_orientation_splits_and_center_images_are_explicit(self) -> None:
        atlas = self.live["component_atlas"]["by_dimension"]

        pure_vector_four = next(
            component
            for component in atlas["8"]["components"]
            if component["id"] == "2*4_R"
        )
        self.assertEqual(pure_vector_four["spin_conjugacy_components"], 2)
        self.assertEqual(pure_vector_four["spin_center_scalar_sign"], 1)
        self.assertEqual(
            pure_vector_four["spin_center_images_across_oriented_components"],
            ["+1"],
        )

        pure_quaternionic = next(
            component
            for component in atlas["8"]["components"]
            if component["id"] == "1*4_H"
        )
        self.assertEqual(pure_quaternionic["spin_conjugacy_components"], 2)
        self.assertEqual(
            pure_quaternionic["spin_center_images_across_oriented_components"],
            ["+volume_8", "-volume_8"],
        )
        self.assertEqual(pure_quaternionic["homomorphism_kernel"], "trivial")

        triple_vector_four = next(
            component
            for component in atlas["12"]["components"]
            if component["id"] == "3*4_R"
        )
        self.assertEqual(triple_vector_four["spin_conjugacy_components"], 2)
        self.assertEqual(triple_vector_four["spin_center_scalar_sign"], -1)

    def test_artifact_is_a_full_exact_replay(self) -> None:
        self.assertEqual(self.saved, self.live)
        self.assertTrue(self.saved["passed"])
        theorem_input = self.saved["spin_lift_theorem_input"]
        self.assertEqual(theorem_input["h1_with_z2"], 0)
        self.assertEqual(theorem_input["h2_with_z2"], 0)
        self.assertIn("standard theorem input", theorem_input["status"])
        self.assertIn(
            "that characteristic-zero H2 proves the mod-2 Spin-lifting statement",
            self.saved["claim_scope"]["not_claimed"],
        )


if __name__ == "__main__":
    unittest.main()
