from __future__ import annotations

import json
import unittest

import sympy as sp

from spin9_v1_v5_reconstruction import (
    COUPLED_MONOMIALS,
    DEFAULT_V5_ARTIFACT,
    ROOT,
    VALIDATION_PRIME,
    _build_coupled_system,
    _coupled_numerator_value,
    _v1_boundary_check,
    _v5_boundary_check,
    load_coefficients,
    validate_rational_coefficients,
)

ARTIFACT = ROOT / "artifacts" / "spin9_v1_v5_reconstruction_20260811.json"
SCREEN_ARTIFACT = ROOT / "artifacts" / "spin9_v1_v5_screen_20260811.json"


class Spin9V1V5ReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.coefficients = load_coefficients(ARTIFACT)
        cls.payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_artifact_contract_and_claim_boundary(self) -> None:
        self.assertEqual(len(self.coefficients), 18_600)
        self.assertEqual(len(self.coefficients), len(COUPLED_MONOMIALS))
        self.assertTrue(all(value.denominator == 1 for value in self.coefficients))
        self.assertTrue(all(value != 0 for value in self.coefficients))
        self.assertEqual(self.payload["maximum_numerator_digits"], 33)
        self.assertTrue(self.payload["passed"])
        self.assertTrue(self.payload["unused_prime_both_sqrt2_embeddings_passed"])
        self.assertFalse(self.payload["characteristic_zero_identity_certified"])
        self.assertFalse(self.payload["global_coupled_positivity_certified"])

    def test_exact_gram_invariant_identity(self) -> None:
        x, u, v = sp.symbols("x u v", real=True)
        p = 3 * u**2 + v**2
        y = sp.sqrt(2) * u * (v**2 - u**2)
        s = x / sp.sqrt(2)
        factored = (
            (1 + 4 * (s - u) ** 2 + 4 * v**2)
            * (1 + 4 * s**2 + 4 * s * u - 4 * s * v + 10 * u**2 + 4 * u * v + 2 * v**2)
            * (1 + 4 * s**2 + 4 * s * u + 4 * s * v + 10 * u**2 - 4 * u * v + 2 * v**2)
        )
        invariant = (
            1
            + 8 * p
            + 20 * p**2
            + 16 * p**3
            - 16 * y**2
            + 6 * x**2
            + 24 * x**2 * p
            + 8 * x**2 * p**2
            + 12 * x**4
            + 16 * x**4 * p
            + 8 * x**6
            + 24 * x * y
            + 80 * x * p * y
            + 80 * x**3 * y
        )
        self.assertEqual(sp.factor(factored - invariant), 0)

    def test_exact_special_boundaries(self) -> None:
        self.assertTrue(DEFAULT_V5_ARTIFACT.exists())
        self.assertTrue(_v1_boundary_check(self.coefficients))
        self.assertTrue(_v5_boundary_check(self.coefficients))

    def test_numerical_screen_retains_its_claim_boundary(self) -> None:
        report = json.loads(SCREEN_ARTIFACT.read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])
        self.assertTrue(report["no_candidate_counterexample_found"])
        self.assertTrue(
            report["pointwise_v5_monotonicity_falsifier"]["strict_improvement"]
        )
        self.assertFalse(report["global_optimality_proved"])

    def test_raw_point_controls_under_both_embeddings(self) -> None:
        prime = VALIDATION_PRIME
        root = int(sp.sqrt_mod(2, prime))
        for sqrt2 in (root, (-root) % prime):
            system = _build_coupled_system(prime, sqrt2=sqrt2)
            inverse_sqrt2 = pow(sqrt2, -1, prime)
            coefficient_residues = tuple(
                value.numerator * pow(value.denominator, -1, prime) % prime
                for value in self.coefficients
            )
            for x_value, u_value, v_value in ((0, 0, 0), (1, 2, 5), (7, 11, 13)):
                p_value = (3 * u_value**2 + v_value**2) % prime
                y_value = (sqrt2 * u_value * (v_value**2 - u_value**2)) % prime
                predicted = (
                    sum(
                        coefficient
                        * pow(x_value, x_power, prime)
                        * pow(p_value, p_power, prime)
                        * pow(y_value, y_power, prime)
                        for (x_power, p_power, y_power), coefficient in zip(
                            COUPLED_MONOMIALS,
                            coefficient_residues,
                            strict=True,
                        )
                    )
                    % prime
                )
                observed = _coupled_numerator_value(
                    system,
                    x_value * inverse_sqrt2,
                    u_value,
                    v_value,
                )
                self.assertEqual(observed, predicted)

    def test_full_unused_prime_reconstruction_both_embeddings(self) -> None:
        self.assertTrue(validate_rational_coefficients(self.coefficients))


if __name__ == "__main__":
    unittest.main()
