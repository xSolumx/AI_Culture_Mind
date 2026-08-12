from __future__ import annotations

import json
import unittest

from spin9_v1_candidate_line import DEFAULT_OUTPUT, certificate


class Spin9V1CandidateLineTests(unittest.TestCase):
    def test_complete_pure_v1_line_theorem_replays(self) -> None:
        artifact = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
        self.assertEqual(certificate(), artifact)
        self.assertTrue(artifact["passed"])
        self.assertTrue(artifact["pure_v1_global_candidate_optimality_certified"])

    def test_four_graph_preimages_are_exactly_classified(self) -> None:
        artifact = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
        fiber = artifact["graph_fiber"]
        self.assertEqual(fiber["octic_real_root_count_by_exact_sturm"], 4)
        self.assertEqual(len(fiber["isolating_intervals"]), 4)
        self.assertTrue(
            fiber["all_real_octic_roots_are_the_four_candidate_preimages"]
        )

    def test_mixed_and_unrestricted_claims_remain_open(self) -> None:
        artifact = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
        self.assertFalse(artifact["coupled_p_positive_candidate_optimality_certified"])
        self.assertFalse(artifact["global_grassmann_quotient_certified"])


if __name__ == "__main__":
    unittest.main()
