from __future__ import annotations

import json
import unittest

import sympy as sp
from algebraic_isotypic_decomposition import (
    DEFAULT_OUTPUT,
    algebraically_conjugate,
    diagnostics,
    negative_square_generator,
    quadratic_split_generator,
)
from exact_real_scalar_field import Q_SQRT_TWO, ExactRealScalarField
from reducible_isotypic_decomposition import (
    decompose_reducible_representation,
    transform_to_isotypic_coordinates,
)
from schur_type_detector import canonical_examples, detect_schur_type


class AlgebraicIsotypicDecompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = diagnostics()
        cls.artifact = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))

    def test_diagnostics_and_stored_artifact_pass(self) -> None:
        self.assertTrue(self.report["passed"])
        self.assertEqual(self.report, self.artifact)
        self.assertTrue(all(self.report["exact_gates"].values()))

    def test_quadratic_field_membership_and_exact_order(self) -> None:
        self.assertTrue(Q_SQRT_TWO.contains(3 - 2 * sp.sqrt(2)))
        self.assertFalse(Q_SQRT_TWO.contains(sp.sqrt(3)))
        controls = {
            sp.sqrt(2) - 1: 1,
            1 - sp.sqrt(2): -1,
            3 - 2 * sp.sqrt(2): 1,
            7 - 5 * sp.sqrt(2): -1,
            sp.Integer(0): 0,
        }
        for value, expected in controls.items():
            with self.subTest(value=value):
                self.assertEqual(Q_SQRT_TWO.sign(value), expected)

    def test_unsupported_extensions_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "quadratic extensions"):
            ExactRealScalarField(sp.real_root(2, 3))
        with self.assertRaisesRegex(ValueError, "actually rational"):
            ExactRealScalarField(sp.sqrt(4))
        with self.assertRaisesRegex(ValueError, "positive root"):
            ExactRealScalarField(-sp.sqrt(2))

    def test_extension_is_genuinely_needed_for_split_projectors(self) -> None:
        generator = quadratic_split_generator()
        rational = decompose_reducible_representation(
            [generator], assume_completely_reducible=True
        )
        quadratic = decompose_reducible_representation(
            [generator],
            assume_completely_reducible=True,
            scalar_extension=sp.sqrt(2),
        )
        self.assertFalse(rational.certified)
        self.assertEqual(rational.unresolved_projector_ranks, (2,))
        self.assertTrue(quadratic.certified)
        expected = {
            tuple(sp.eye(2) / 2 + generator * sp.sqrt(2) / 4),
            tuple(sp.eye(2) / 2 - generator * sp.sqrt(2) / 4),
        }
        self.assertEqual(
            {tuple(summand.projector) for summand in quadratic.summands}, expected
        )
        transformed = transform_to_isotypic_coordinates([generator], quadratic)
        self.assertEqual(set(transformed[0].diagonal()), {sp.sqrt(2), -sp.sqrt(2)})

    def test_negative_square_is_not_falsely_split(self) -> None:
        certificate = decompose_reducible_representation(
            [negative_square_generator()],
            assume_completely_reducible=True,
            scalar_extension=sp.sqrt(2),
        )
        self.assertTrue(certificate.certified)
        self.assertEqual(len(certificate.blocks), 1)
        block = certificate.blocks[0]
        self.assertEqual(
            (block.schur_type, block.multiplicity, block.irreducible_dimension),
            ("complex", 1, 2),
        )

    def test_real_complex_quaternion_types_survive_algebraic_conjugacy(self) -> None:
        expected = {
            "real_so3_vector": "real",
            "complex_u1_realification": "complex",
            "quaternion_su2_spinor_realification": "quaternion",
        }
        for name, generators in canonical_examples().items():
            with self.subTest(name=name):
                certificate = detect_schur_type(
                    algebraically_conjugate(generators),
                    assume_completely_reducible=True,
                    scalar_extension=sp.sqrt(2),
                )
                self.assertTrue(certificate.classified_irreducible)
                self.assertEqual(certificate.schur_type, expected[name])

    def test_spin9_concrete_modules_compile_without_rationalization(self) -> None:
        slice_rows = self.report["spin9_concrete_slice"]["isotypic_blocks"]
        full_rows = self.report["spin9_concrete_full_quotient"]["isotypic_blocks"]
        self.assertEqual(
            sorted(
                (row["multiplicity"], row["irreducible_dimension"])
                for row in slice_rows
            ),
            [(1, 1), (1, 5)],
        )
        self.assertEqual(
            sorted(
                (row["multiplicity"], row["irreducible_dimension"]) for row in full_rows
            ),
            [(1, 1), (2, 5)],
        )

    def test_field_mismatch_is_rejected_before_decomposition(self) -> None:
        with self.assertRaisesRegex(ValueError, "declared field"):
            decompose_reducible_representation(
                [sp.Matrix([[sp.sqrt(3)]])],
                assume_completely_reducible=True,
                scalar_extension=sp.sqrt(2),
            )


if __name__ == "__main__":
    unittest.main()
