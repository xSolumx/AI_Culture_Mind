from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spin9_dirac_clifford import (
    SPINOR_WITNESSES,
    build_spin9_clifford_system,
    diagnostics,
    dirac_operator,
    even_spin_transition,
    exact_hopf_residual,
    hurwitz_radon_number,
    modular_pivot_certificate,
    modular_rank,
    observation_matrix,
)


class Spin9DiracCliffordTests(unittest.TestCase):
    def test_exact_diagnostics(self) -> None:
        report = diagnostics()
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["involution_shape"], [9, 16, 16])
        self.assertEqual(report["generator_shape"], [36, 16, 16])
        self.assertEqual(report["hopf_polynomial_nonzero_coefficients"], 0)

    def test_hurwitz_radon_boundary(self) -> None:
        self.assertEqual(hurwitz_radon_number(8), 8)
        self.assertEqual(hurwitz_radon_number(16), 9)
        self.assertEqual(hurwitz_radon_number(32), 10)

    def test_third_probe_must_add_a_new_direction(self) -> None:
        system = build_spin9_clifford_system()
        generic = observation_matrix(system.doubled_spin_generators, SPINOR_WITNESSES)
        dependent = observation_matrix(
            system.doubled_spin_generators,
            [
                SPINOR_WITNESSES[0],
                SPINOR_WITNESSES[1],
                SPINOR_WITNESSES[0] + 2 * SPINOR_WITNESSES[1],
            ],
        )
        self.assertEqual(modular_rank(generic, 1_000_003), 36)
        self.assertEqual(modular_rank(dependent, 1_000_003), 28)
        certificate = modular_pivot_certificate(generic, 1_000_003)
        self.assertEqual(certificate["rank"], 36)
        self.assertEqual(len(certificate["pivot_rows"]), 36)
        self.assertNotEqual(certificate["pivot_minor_determinant_mod_prime"], 0)

    def test_hopf_identity_is_polynomial_not_sampled(self) -> None:
        system = build_spin9_clifford_system()
        self.assertEqual(exact_hopf_residual(system.involutions), {})

    def test_even_transition_is_orthogonal(self) -> None:
        first = np.arange(1.0, 10.0)
        second = np.asarray([2.0, -1.0, 3.0, 4.0, -2.0, 1.0, 5.0, -3.0, 2.0])
        transition = even_spin_transition(first, second)
        self.assertTrue(np.allclose(transition.T @ transition, np.eye(16), atol=2e-15))
        state = np.linspace(-1.0, 1.0, 16)
        self.assertAlmostEqual(
            float(np.linalg.norm(transition @ state)),
            float(np.linalg.norm(state)),
            places=13,
        )

    def test_dirac_square(self) -> None:
        address = np.asarray([1, 2, -1, 0, 3, -2, 1, 4, -3], dtype=np.float64)
        operator = dirac_operator(address)
        expected = float(address @ address) * np.eye(16)
        self.assertTrue(np.array_equal(operator @ operator, expected))

    def test_cli_artifact_replays(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "spin9.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "spin9_dirac_clifford",
                    "--output",
                    str(output),
                ],
                cwd=root,
                env={**dict(__import__("os").environ), "PYTHONPATH": "src"},
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(json.loads(output.read_text(encoding="utf-8"))["passed"])


if __name__ == "__main__":
    unittest.main()
