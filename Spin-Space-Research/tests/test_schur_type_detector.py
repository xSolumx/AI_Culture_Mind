from __future__ import annotations

import json
import unittest

import sympy as sp

from schur_type_detector import (
    DEFAULT_OUTPUT,
    canonical_examples,
    detect_schur_type,
    diagnostics,
    exact_commutant_basis,
    rational_conjugacy_examples,
    rejection_examples,
)


class SchurTypeDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = diagnostics()
        cls.artifact = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))

    def test_canonical_real_complex_quaternion_types(self) -> None:
        expected = {
            "real_so3_vector": ("real", 1),
            "complex_u1_realification": ("complex", 2),
            "quaternion_su2_spinor_realification": ("quaternion", 4),
        }
        for name, generators in canonical_examples().items():
            certificate = detect_schur_type(
                generators,
                assume_completely_reducible=True,
            )
            schur_type, dimension = expected[name]
            self.assertEqual(certificate.schur_type, schur_type)
            self.assertEqual(certificate.commutant_dimension, dimension)
            self.assertTrue(certificate.classified_irreducible)
            self.assertTrue(all(certificate.exact_gates.values()))

    def test_extracted_multiplication_table_reconstructs_products(self) -> None:
        for generators in canonical_examples().values():
            certificate = detect_schur_type(
                generators,
                assume_completely_reducible=True,
            )
            for left_index, left in enumerate(certificate.division_basis):
                for right_index, right in enumerate(certificate.division_basis):
                    coefficients = certificate.multiplication_table[left_index][
                        right_index
                    ]
                    reconstructed = sp.zeros(left.rows, left.cols)
                    for coefficient, basis in zip(
                        coefficients,
                        certificate.division_basis,
                        strict=True,
                    ):
                        reconstructed += coefficient * basis
                    self.assertEqual(left * right, reconstructed)

    def test_complex_and_quaternion_relations_are_exact(self) -> None:
        complex_certificate = detect_schur_type(
            canonical_examples()["complex_u1_realification"],
            assume_completely_reducible=True,
        )
        one, imaginary = complex_certificate.division_basis
        self.assertEqual(imaginary**2, -one)

        quaternion_certificate = detect_schur_type(
            canonical_examples()["quaternion_su2_spinor_realification"],
            assume_completely_reducible=True,
        )
        one, i, j, k = quaternion_certificate.division_basis
        self.assertEqual(i**2, -one)
        self.assertEqual(j**2, -one)
        self.assertEqual(k**2, -one)
        self.assertEqual(i * j, k)
        self.assertEqual(j * i, -k)

    def test_rational_change_of_basis_preserves_detected_type(self) -> None:
        expected = {
            "real_so3_vector": "real",
            "complex_u1_realification": "complex",
            "quaternion_su2_spinor_realification": "quaternion",
        }
        for name, conjugated in rational_conjugacy_examples().items():
            certificate = detect_schur_type(
                conjugated,
                assume_completely_reducible=True,
            )
            self.assertEqual(certificate.schur_type, expected[name])
            self.assertTrue(certificate.classified_irreducible)

    def test_split_and_repeated_representations_are_rejected(self) -> None:
        certificates = {
            name: detect_schur_type(
                generators,
                assume_completely_reducible=True,
            )
            for name, generators in rejection_examples().items()
        }
        split = certificates["split_two_line_representation"]
        self.assertEqual(split.commutant_dimension, 2)
        self.assertFalse(
            split.exact_gates["complex_structure_square_is_negative_scalar"]
        )
        self.assertIsNone(split.schur_type)

        repeated = certificates["doubled_complex_irrep"]
        self.assertEqual(repeated.commutant_dimension, 8)
        self.assertIsNone(repeated.schur_type)
        self.assertFalse(repeated.classified_irreducible)

    def test_complete_reducibility_is_a_required_logical_input(self) -> None:
        certificate = detect_schur_type(
            canonical_examples()["complex_u1_realification"],
            assume_completely_reducible=False,
        )
        self.assertIsNone(certificate.schur_type)
        self.assertFalse(certificate.classified_irreducible)
        self.assertIn("complete reducibility", certificate.rejection_reason or "")

    def test_invalid_generator_sets_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            exact_commutant_basis([])
        with self.assertRaisesRegex(ValueError, "same shape"):
            exact_commutant_basis([sp.eye(2), sp.eye(3)])
        with self.assertRaisesRegex(ValueError, "exact rational"):
            exact_commutant_basis([sp.Matrix([[sp.sqrt(2)]])])

    def test_artifact_replays_exactly_and_keeps_claim_boundaries(self) -> None:
        self.assertTrue(self.report["passed"])
        self.assertEqual(self.artifact, self.report)
        self.assertFalse(
            self.report["reducible_isotypic_decomposition_implemented_in_this_module"]
        )
        self.assertTrue(
            self.report["companion_reducible_isotypic_decomposition_available"]
        )
        self.assertFalse(self.report["floating_point_noisy_detection_claimed"])
        self.assertFalse(self.report["sequence_model_superiority_claimed"])


if __name__ == "__main__":
    unittest.main()
