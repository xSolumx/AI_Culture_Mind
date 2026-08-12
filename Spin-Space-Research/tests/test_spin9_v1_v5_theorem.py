from __future__ import annotations

import json
import unittest

from spin9_v1_v5_theorem import DEFAULT_OUTPUT, certificate


class Spin9V1V5TheoremTests(unittest.TestCase):
    def test_combined_theorem_replays(self) -> None:
        artifact = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
        self.assertEqual(certificate(), artifact)
        self.assertTrue(artifact["passed"])
        self.assertTrue(artifact["finite_radius_coupled_slice_bound_certified"])
        self.assertTrue(artifact["global_on_stated_slice"])

    def test_unproved_global_claims_remain_false(self) -> None:
        artifact = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
        self.assertFalse(
            artifact["algebraic_symmetric_candidate_optimality_certified"]
        )
        self.assertFalse(artifact["second_v5_copy_certified"])
        self.assertFalse(artifact["global_grassmann_quotient_certified"])
        self.assertFalse(artifact["global_rank_three_optimum_certified"])


if __name__ == "__main__":
    unittest.main()
