from __future__ import annotations

import json
import unittest

import sympy as sp
from clifford_signature_extension import (
    DEFAULT_OUTPUT,
    clifford_1_4_generators,
    clifford_blades,
    diagnostics,
)


class CliffordSignatureExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = diagnostics()
        cls.artifact = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))

    def test_diagnostics_and_stored_artifact_pass(self) -> None:
        self.assertTrue(self.report["passed"])
        self.assertEqual(self.report, self.artifact)
        self.assertTrue(all(self.report["exact_gates"].values()))

    def test_cl_1_4_signature_is_exact(self) -> None:
        generators = clifford_1_4_generators()
        identity = sp.eye(16)
        for generator, sign in zip(generators, (1, -1, -1, -1, -1), strict=True):
            self.assertEqual(generator**2, sign * identity)
        for index, left in enumerate(generators):
            for right in generators[index + 1 :]:
                self.assertEqual(left * right + right * left, sp.zeros(16))

    def test_full_even_and_embedded_dimensions_are_not_conflated(self) -> None:
        self.assertEqual(
            self.report["dimension_ledger"],
            {"Cl(3,0)": 8, "Cl^0(1,4)": 16, "Cl(1,4)": 32},
        )
        self.assertEqual(self.report["cl_1_4_blade_rank"], 32)
        self.assertEqual(self.report["cl_1_4_even_blade_rank"], 16)
        self.assertEqual(self.report["embedded_cl_3_0_blade_rank"], 8)
        self.assertFalse(self.report["cl_3_0_equals_cl_1_4_claimed"])

    def test_volume_element_separates_full_clifford_components(self) -> None:
        self.assertEqual(self.report["volume_projector_ranks"], [8, 8])
        self.assertEqual(
            self.report["full_sector_schur_types"],
            ["quaternion", "quaternion"],
        )
        self.assertEqual(self.report["full_sector_intertwiner_dimension"], 0)
        self.assertEqual(self.report["full_commutant_dimension"], 8)

    def test_even_algebra_has_two_equivalent_quaternionic_modules(self) -> None:
        self.assertEqual(
            self.report["even_sector_schur_types"],
            ["quaternion", "quaternion"],
        )
        self.assertEqual(self.report["even_sector_intertwiner_dimension"], 4)
        self.assertEqual(self.report["even_commutant_dimension"], 16)

    def test_embedded_cl3_has_four_complex_spinor_copies(self) -> None:
        self.assertEqual(
            self.report["embedded_cl_3_0_sector_signatures"],
            [[["complex", 2, 4]], [["complex", 2, 4]]],
        )
        self.assertEqual(self.report["embedded_cl_3_0_cross_intertwiner_dimension"], 8)
        self.assertEqual(self.report["embedded_cl_3_0_commutant_dimension"], 32)

    def test_spin8_triality_is_explicit_and_pairwise_inequivalent(self) -> None:
        expected = {
            "8v": {"8v": 1, "8+": 0, "8-": 0},
            "8+": {"8v": 0, "8+": 1, "8-": 0},
            "8-": {"8v": 0, "8+": 0, "8-": 1},
        }
        self.assertEqual(self.report["spin8_lie_closure_dimension"], 28)
        self.assertEqual(
            self.report["spin8_triality_schur_types"],
            {"8v": "real", "8+": "real", "8-": "real"},
        )
        self.assertEqual(
            self.report["spin8_triality_pairwise_intertwiner_dimensions"], expected
        )
        self.assertEqual(
            self.report["spin8_algebraic_pairwise_intertwiner_dimensions"], expected
        )

    def test_maintained_cl3_state_claim_remains_separate(self) -> None:
        self.assertEqual(
            self.report["maintained_cl3_conjugation_isotypic_signatures"],
            [["real", 2, 1], ["real", 2, 3]],
        )
        self.assertFalse(
            self.report["maintained_cl3_model_embedded_as_same_state_model"]
        )
        self.assertFalse(self.report["spin8_or_spin9_model_advantage_claimed"])

    def test_blade_constructor_has_one_image_per_abstract_blade(self) -> None:
        self.assertEqual(len(clifford_blades(clifford_1_4_generators())), 32)


if __name__ == "__main__":
    unittest.main()
