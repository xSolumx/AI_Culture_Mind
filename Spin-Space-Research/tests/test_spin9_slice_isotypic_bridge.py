from __future__ import annotations

import json
import unittest

import sympy as sp

from spin9_grassmann_slice import construct_slice_data
from spin9_slice_isotypic_bridge import (
    DEFAULT_OUTPUT,
    build_certificate,
    certificate_json,
    exact_algebraic_intertwiner_basis,
    supported_sym0_basis,
)


class Spin9SliceIsotypicBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.certificate = build_certificate()
        cls.report = certificate_json(cls.certificate)
        cls.artifact = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))

    def test_certificate_and_stored_artifact_pass(self) -> None:
        self.assertTrue(self.certificate.certified)
        self.assertTrue(self.report["passed"])
        self.assertEqual(self.report, self.artifact)
        self.assertTrue(all(self.report["exact_gates"].values()))

    def test_concrete_slice_has_exact_q_sqrt_two_intertwiner(self) -> None:
        certificate = self.certificate
        self.assertEqual(len(certificate.intertwiner_basis), 2)
        self.assertEqual(
            sorted(value.rank() for value in certificate.intertwiner_basis),
            [1, 5],
        )
        self.assertEqual(sp.factor(certificate.slice_change.det()), -sp.Rational(1, 4))
        for source, target in zip(
            certificate.standardized_slice_generators,
            certificate.canonical_slice_generators,
            strict=True,
        ):
            self.assertEqual(
                certificate.slice_change * source * certificate.slice_change.inv(),
                target,
            )

    def test_curve_is_exactly_the_trivial_coordinate(self) -> None:
        transformed = self.certificate.transformed_curve_coordinates
        self.assertEqual(transformed[0], 3 * sp.sqrt(2) / 8)
        self.assertEqual(transformed[1:, :], sp.zeros(5, 1))

    def test_supported_basis_matches_local_hessian_convention(self) -> None:
        norms = tuple(
            int(sp.trace(value.T * value)) for value in supported_sym0_basis()
        )
        self.assertEqual(norms, (2, 6, 2, 2, 2))
        self.assertEqual(self.certificate.supported_change.det(), 2)
        self.assertEqual(self.certificate.extended_change.det(), -sp.Rational(1, 2))

    def test_concrete_v1_plus_two_v5_rationalizes_before_compilation(self) -> None:
        certificate = self.certificate
        transformed = tuple(
            certificate.extended_change * generator * certificate.extended_change.inv()
            for generator in certificate.concrete_generators
        )
        self.assertEqual(transformed, certificate.rational_generators)
        self.assertTrue(
            all(
                entry.is_Rational
                for generator in certificate.rational_generators
                for entry in generator
            )
        )
        signatures = sorted(
            (
                block.schur_type,
                block.multiplicity,
                block.irreducible_dimension,
                block.commutant_dimension,
            )
            for block in certificate.reducible_certificate.blocks
        )
        self.assertEqual(
            signatures,
            [("real", 1, 1, 1), ("real", 2, 5, 4)],
        )

    def test_slice_constructor_is_the_shared_source_of_truth(self) -> None:
        data = construct_slice_data()
        self.assertEqual(data.normal_metric, sp.diag(2, 4, 4, 2, 8, 4))
        self.assertEqual(data.casimir.eigenvals(), {0: 1, 6: 5})
        self.assertEqual(
            data.curve_coordinates, self.certificate.slice_data.curve_coordinates
        )

    def test_algebraic_intertwiner_api_refuses_bad_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "same positive generator count"):
            exact_algebraic_intertwiner_basis([sp.eye(2)], [])
        with self.assertRaisesRegex(ValueError, "source generators must be square"):
            exact_algebraic_intertwiner_basis([sp.zeros(2, 3)], [sp.eye(2)])
        with self.assertRaisesRegex(ValueError, "target generators must be square"):
            exact_algebraic_intertwiner_basis([sp.eye(2)], [sp.zeros(2, 3)])

    def test_artifact_keeps_global_claims_open(self) -> None:
        self.assertFalse(self.report["global_grassmann_quotient_solved"])
        self.assertFalse(self.report["finite_radius_coupled_determinant_solved"])


if __name__ == "__main__":
    unittest.main()
