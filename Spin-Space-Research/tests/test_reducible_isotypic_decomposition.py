from __future__ import annotations

import json
import unittest

import sympy as sp

from reducible_isotypic_decomposition import (
    DEFAULT_OUTPUT,
    aligned_isotypic_basis,
    canonical_reducible_examples,
    certificate_json,
    cl3_spin3_fixture,
    decompose_reducible_representation,
    diagnostics,
    exact_algebra_center_basis,
    exact_intertwiner_basis,
    rationally_conjugate_representation,
    repeated_representation,
    spin9_slice_fixture,
    spin_two_generators,
    transform_to_isotypic_coordinates,
)
from schur_type_detector import canonical_examples


class ReducibleIsotypicDecompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = diagnostics()
        cls.artifact = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
        cls.examples = canonical_reducible_examples()

    def test_diagnostics_and_stored_artifact_pass(self) -> None:
        self.assertTrue(self.report["passed"])
        self.assertTrue(self.artifact["passed"])
        self.assertEqual(self.report, self.artifact)
        self.assertEqual(
            self.report["expected_block_signatures"],
            self.report["observed_block_signatures"],
        )

    def test_all_positive_controls_reconstruct_exactly(self) -> None:
        for name, generators in self.examples.items():
            with self.subTest(name=name):
                certificate = decompose_reducible_representation(
                    generators,
                    assume_completely_reducible=True,
                )
                self.assertTrue(certificate.certified)
                self.assertTrue(all(certificate.exact_gates.values()))
                self.assertFalse(certificate.unresolved_projector_ranks)
                self.assertIsNone(certificate.rejection_reason)
                self.assertEqual(
                    sum(block.real_dimension for block in certificate.blocks),
                    certificate.real_dimension,
                )
                self.assertEqual(
                    sum(
                        block.expected_commutant_dimension
                        for block in certificate.blocks
                    ),
                    certificate.commutant_dimension,
                )

    def test_repeated_real_complex_quaternion_blocks(self) -> None:
        expected = {
            "doubled_real_so3": ("real", 2, 3, 4),
            "doubled_complex_u1": ("complex", 2, 2, 8),
            "doubled_quaternion_su2": ("quaternion", 2, 4, 16),
        }
        for name, signature in expected.items():
            with self.subTest(name=name):
                certificate = decompose_reducible_representation(
                    self.examples[name],
                    assume_completely_reducible=True,
                )
                self.assertEqual(len(certificate.blocks), 1)
                block = certificate.blocks[0]
                self.assertEqual(
                    (
                        block.schur_type,
                        block.multiplicity,
                        block.irreducible_dimension,
                        block.commutant_dimension,
                    ),
                    signature,
                )
                division_dimension = {"real": 1, "complex": 2, "quaternion": 4}[
                    block.schur_type
                ]
                self.assertEqual(
                    block.commutant_dimension,
                    block.multiplicity**2 * division_dimension,
                )
                self.assertTrue(all(block.exact_gates.values()))
                self.assertEqual(
                    block.center_dimension,
                    2 if block.schur_type == "complex" else 1,
                )
                self.assertEqual(
                    block.aligned_basis.rank(),
                    block.real_dimension,
                )
                self.assertEqual(
                    len(block.aligned_commutant_basis),
                    block.expected_commutant_dimension,
                )
                self.assertEqual(
                    len(block.aligned_center_basis),
                    block.expected_center_dimension,
                )

    def test_mixed_types_have_zero_cross_intertwiners(self) -> None:
        certificate = decompose_reducible_representation(
            self.examples["mixed_real_complex_quaternion"],
            assume_completely_reducible=True,
        )
        self.assertTrue(certificate.certified)
        self.assertEqual(
            sorted(block.schur_type for block in certificate.blocks),
            ["complex", "quaternion", "real"],
        )
        self.assertEqual(certificate.commutant_dimension, 1 + 2 + 4)
        self.assertTrue(
            certificate.exact_gates["inequivalent_blocks_have_zero_intertwiner_space"]
        )

    def test_cl3_fixture_is_two_trivial_plus_two_vector(self) -> None:
        certificate = decompose_reducible_representation(
            cl3_spin3_fixture(),
            assume_completely_reducible=True,
        )
        signatures = sorted(
            (
                block.schur_type,
                block.multiplicity,
                block.irreducible_dimension,
                block.commutant_dimension,
            )
            for block in certificate.blocks
        )
        self.assertEqual(
            signatures,
            [("real", 2, 1, 4), ("real", 2, 3, 4)],
        )
        self.assertEqual(certificate.commutant_dimension, 8)
        self.assertTrue(certificate.certified)

    def test_spin9_slice_fixture_is_v1_plus_two_v5(self) -> None:
        generators = spin_two_generators()
        self.assertEqual(
            -sum((generator**2 for generator in generators), sp.zeros(5)),
            6 * sp.eye(5),
        )
        certificate = decompose_reducible_representation(
            spin9_slice_fixture(),
            assume_completely_reducible=True,
        )
        signatures = sorted(
            (
                block.schur_type,
                block.multiplicity,
                block.irreducible_dimension,
                block.commutant_dimension,
            )
            for block in certificate.blocks
        )
        self.assertEqual(
            signatures,
            [("real", 1, 1, 1), ("real", 2, 5, 4)],
        )
        self.assertTrue(certificate.certified)

    def test_center_is_split_before_multiplicity_space(self) -> None:
        for name in (
            "cl3_two_trivial_plus_two_vector",
            "spin9_v1_plus_two_v5",
        ):
            with self.subTest(name=name):
                certificate = decompose_reducible_representation(
                    self.examples[name],
                    assume_completely_reducible=True,
                )
                sources = [row.source for row in certificate.split_witnesses]
                self.assertEqual(sources[0], "central_search")
                self.assertIn("multiplicity_search", sources)
                self.assertTrue(
                    certificate.exact_gates["central_stage_projectors_are_central"]
                )

    def test_rational_conjugacy_does_not_change_isotypic_signatures(self) -> None:
        pairs = [
            ("cl3_two_trivial_plus_two_vector", "cl3_rational_conjugacy"),
            ("spin9_v1_plus_two_v5", "spin9_slice_rational_conjugacy"),
        ]
        for original, conjugated in pairs:
            with self.subTest(pair=(original, conjugated)):
                left = decompose_reducible_representation(
                    self.examples[original],
                    assume_completely_reducible=True,
                )
                right = decompose_reducible_representation(
                    self.examples[conjugated],
                    assume_completely_reducible=True,
                )
                left_signature = sorted(
                    (
                        block.schur_type,
                        block.multiplicity,
                        block.irreducible_dimension,
                    )
                    for block in left.blocks
                )
                right_signature = sorted(
                    (
                        block.schur_type,
                        block.multiplicity,
                        block.irreducible_dimension,
                    )
                    for block in right.blocks
                )
                self.assertEqual(left_signature, right_signature)
                self.assertTrue(right.certified)

    def test_rational_conjugacy_preserves_all_division_types(self) -> None:
        for name in (
            "doubled_real_so3",
            "doubled_complex_u1",
            "doubled_quaternion_su2",
            "mixed_real_complex_quaternion",
        ):
            with self.subTest(name=name):
                original = decompose_reducible_representation(
                    self.examples[name],
                    assume_completely_reducible=True,
                )
                conjugated = decompose_reducible_representation(
                    rationally_conjugate_representation(self.examples[name]),
                    assume_completely_reducible=True,
                )

                def signature(certificate):
                    return sorted(
                        (
                            block.schur_type,
                            block.multiplicity,
                            block.irreducible_dimension,
                            block.commutant_dimension,
                            block.center_dimension,
                        )
                        for block in certificate.blocks
                    )

                self.assertEqual(signature(original), signature(conjugated))
                self.assertTrue(conjugated.certified)

    def test_compiler_basis_repeats_one_reference_action_per_block(self) -> None:
        for name in (
            "doubled_complex_u1",
            "doubled_quaternion_su2",
            "cl3_rational_conjugacy",
            "spin9_slice_rational_conjugacy",
        ):
            with self.subTest(name=name):
                generators = self.examples[name]
                certificate = decompose_reducible_representation(
                    generators,
                    assume_completely_reducible=True,
                )
                basis = aligned_isotypic_basis(certificate)
                self.assertEqual(
                    basis.shape,
                    (certificate.real_dimension, certificate.real_dimension),
                )
                transformed = transform_to_isotypic_coordinates(generators, certificate)
                for generator_index, generator in enumerate(transformed):
                    expected = sp.diag(
                        *(
                            sp.diag(
                                *(
                                    certificate.summands[
                                        block.summand_indices[0]
                                    ].restricted_generators[generator_index]
                                    for _ in range(block.multiplicity)
                                )
                            )
                            for block in certificate.blocks
                        )
                    )
                    self.assertEqual(generator, expected)

    def test_irreducible_inputs_are_valid_multiplicity_one_blocks(self) -> None:
        expected = {
            "real_so3_vector": "real",
            "complex_u1_realification": "complex",
            "quaternion_su2_spinor_realification": "quaternion",
        }
        for name, generators in canonical_examples().items():
            with self.subTest(name=name):
                certificate = decompose_reducible_representation(
                    generators,
                    assume_completely_reducible=True,
                )
                self.assertTrue(certificate.certified)
                self.assertEqual(len(certificate.summands), 1)
                self.assertEqual(len(certificate.blocks), 1)
                self.assertEqual(certificate.blocks[0].multiplicity, 1)
                self.assertEqual(certificate.blocks[0].schur_type, expected[name])

    def test_exact_intertwiner_dimensions_recover_division_algebras(self) -> None:
        expected = {
            "real_so3_vector": 1,
            "complex_u1_realification": 2,
            "quaternion_su2_spinor_realification": 4,
        }
        examples = canonical_examples()
        for name, dimension in expected.items():
            with self.subTest(name=name):
                self.assertEqual(
                    len(exact_intertwiner_basis(examples[name], examples[name])),
                    dimension,
                )

    def test_exact_commutant_centers_distinguish_complex_type(self) -> None:
        expected = {
            "doubled_real_so3": 1,
            "doubled_complex_u1": 2,
            "doubled_quaternion_su2": 1,
        }
        for name, center_dimension in expected.items():
            with self.subTest(name=name):
                certificate = decompose_reducible_representation(
                    self.examples[name],
                    assume_completely_reducible=True,
                )
                block = certificate.blocks[0]
                self.assertEqual(block.center_dimension, center_dimension)
                self.assertEqual(
                    len(
                        exact_algebra_center_basis(
                            exact_intertwiner_basis(
                                self.examples[name],
                                self.examples[name],
                            )
                        )
                    ),
                    center_dimension,
                )

    def test_supplied_commutant_idempotent_is_audited(self) -> None:
        generators = repeated_representation(canonical_examples()["real_so3_vector"], 2)
        witness = sp.diag(sp.eye(3), sp.zeros(3))
        certificate = decompose_reducible_representation(
            generators,
            assume_completely_reducible=True,
            splitting_witnesses=[witness],
            max_candidates_per_leaf=1,
        )
        self.assertTrue(certificate.certified)
        self.assertEqual(certificate.split_witnesses[0].source, "supplied")
        self.assertEqual(certificate.split_witnesses[0].child_ranks, (3, 3))

    def test_missing_assumption_and_bounded_search_refuse(self) -> None:
        generators = self.examples["doubled_real_so3"]
        missing = decompose_reducible_representation(
            generators,
            assume_completely_reducible=False,
        )
        self.assertFalse(missing.certified)
        self.assertIn("complete reducibility", missing.rejection_reason or "")
        bounded = decompose_reducible_representation(
            generators,
            assume_completely_reducible=True,
            max_candidates_per_leaf=0,
        )
        self.assertFalse(bounded.certified)
        self.assertEqual(bounded.unresolved_projector_ranks, (6,))
        self.assertIn("rational idempotent search", bounded.rejection_reason or "")
        with self.assertRaisesRegex(ValueError, "certified"):
            aligned_isotypic_basis(bounded)

    def test_invalid_inputs_and_witnesses_are_rejected(self) -> None:
        generators = self.examples["doubled_real_so3"]
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            decompose_reducible_representation(
                generators,
                assume_completely_reducible=True,
                max_candidates_per_leaf=-1,
            )
        with self.assertRaisesRegex(ValueError, "match"):
            decompose_reducible_representation(
                generators,
                assume_completely_reducible=True,
                splitting_witnesses=[sp.eye(2)],
            )
        with self.assertRaisesRegex(ValueError, "exact rational"):
            decompose_reducible_representation(
                generators,
                assume_completely_reducible=True,
                splitting_witnesses=[sp.sqrt(2) * sp.eye(6)],
            )
        with self.assertRaisesRegex(ValueError, "exact commutant"):
            decompose_reducible_representation(
                generators,
                assume_completely_reducible=True,
                splitting_witnesses=[sp.diag(1, 0, 0, 0, 0, 0)],
            )

    def test_json_certificate_is_deterministic(self) -> None:
        certificate = decompose_reducible_representation(
            self.examples["spin9_v1_plus_two_v5"],
            assume_completely_reducible=True,
        )
        self.assertEqual(certificate_json(certificate), certificate_json(certificate))
        self.assertTrue(certificate_json(certificate)["certified"])


if __name__ == "__main__":
    unittest.main()
