"""Tests for the independent python-flint arithmetic cross-verifier."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from spin8_flint_crosscheck import FLINT_AVAILABLE, run


@unittest.skipUnless(FLINT_AVAILABLE, "python-flint exact extra is not installed")
class Spin8FlintCrosscheckTests(unittest.TestCase):
    def test_sympy_and_flint_exact_arithmetic_agree(self) -> None:
        report = run(flint_threads=2)
        self.assertTrue(report["passed"])
        self.assertTrue(
            report["matrix_certificate"]["characteristic_coefficients_match"]
        )
        self.assertTrue(
            report["weight_polynomial_certificate"][
                "coefficients_match_sympy_factorization"
            ]
        )
        self.assertEqual(report["rank_boundary_certificate"]["flint_rank"], 25)

    def test_stored_artifact_replays_exactly(self) -> None:
        artifact = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "spin8_flint_crosscheck_20260806.json"
        )
        stored = json.loads(artifact.read_text(encoding="utf-8"))
        fresh = run(flint_threads=stored["flint_threads"])
        self.assertEqual(stored, fresh)


if __name__ == "__main__":
    unittest.main()
