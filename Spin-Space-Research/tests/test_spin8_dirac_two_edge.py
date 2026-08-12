"""Contracts for the preregistered second-residual Dirac bridge."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import torch

from spin8_dirac_one_edge import _projector
from spin8_dirac_star import rational_circle
from spin8_dirac_two_edge import (
    ANCHORS,
    exact_normalized_determinant,
    exact_sign_symmetry_certificate,
    verify_anchor_report,
)
from spin8_dirac_two_edge_attack import log_advantage
from spin8_dirac_two_edge_degree import verify_degree_report
from spin8_triality import torch_triality_generators


class Spin8DiracTwoEdgeTests(unittest.TestCase):
    def test_common_triality_symmetry_allows_exactly_eight_sectors(self) -> None:
        report = exact_sign_symmetry_certificate()
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["induced_sign_group"]), 8)
        self.assertEqual(len(report["walsh_annihilator"]), 8)

    def test_two_exact_anchors_activate_the_complete_allowed_support(self) -> None:
        artifact = (
            Path(__file__).parents[1]
            / "artifacts"
            / "spin8_dirac_two_edge_anchor_20260806.json"
        )
        report = json.loads(artifact.read_text(encoding="utf-8"))
        self.assertTrue(verify_anchor_report(report))
        self.assertEqual(report["exact_determinant_count"], 128)
        self.assertTrue(report["supports_match_between_anchors"])
        self.assertTrue(
            all(row["support_equals_symmetry_annihilator"] for row in report["anchors"])
        )

        first = report["determinant_rows"][0]
        direct = exact_normalized_determinant(
            ANCHORS[first["anchor_index"]], tuple(first["signs"])
        )
        self.assertEqual(str(direct), first["normalized_determinant"])

    def test_float_falsifier_matches_an_exact_anchor(self) -> None:
        generators = torch_triality_generators(dtype=torch.float64)
        basis_zero = torch.zeros(1, 8, dtype=torch.float64)
        basis_zero[0, 0] = 1
        fixed = _projector(generators, 0, basis_zero) + _projector(
            generators, 1, basis_zero
        )
        parameters = torch.tensor(
            [[float(rational_circle(value)[0]) for value in ANCHORS[0]]],
            dtype=torch.float64,
        )
        observed = float(log_advantage(parameters, generators, fixed)[0])
        direct = exact_normalized_determinant(ANCHORS[0], (1, 1, 1, 1, 1, 1))
        cayley = rational_circle(ANCHORS[0][-1])[0]
        target = (1 - cayley**2) ** 3 * (9 - cayley**2) ** 2
        expected = math.log(float(direct / target))
        self.assertAlmostEqual(observed, expected, places=13)

    def test_exact_degree_artifact_replays_its_acceptance_predicate(self) -> None:
        artifact = (
            Path(__file__).parents[1]
            / "artifacts"
            / "spin8_dirac_two_edge_degree_20260806.json"
        )
        report = json.loads(artifact.read_text(encoding="utf-8"))
        self.assertTrue(verify_degree_report(report))
        self.assertEqual(report["exact_determinant_count"], 2736)
        self.assertEqual(report["slice_count"], 144)
        self.assertEqual(
            sum(row["confirmation_nodes_passed"] for row in report["slice_rows"]),
            576,
        )
        self.assertEqual(
            report["post_degree_square_discovery"],
            {
                "all_nontrivial_residuals_are_exact_squares": True,
                "exact_square_slice_count": 126,
                "nontrivial_slice_count": 126,
                "status": ("exact on every audited slice; global factorization open"),
            },
        )


if __name__ == "__main__":
    unittest.main()
