import json
import unittest
from pathlib import Path

import torch

from analyze_matched_retrieval_campaign import analyze
from benchmark_matched_memory_cores import forward_functions, make_problem


class MatchedRetrievalCampaignTests(unittest.TestCase):
    def test_state_matched_memory_cores_agree_in_float64(self) -> None:
        problem = make_problem(
            batch=2,
            length=31,
            dtype=torch.float64,
            device=torch.device("cpu"),
            seed=20260810,
        )
        functions = forward_functions(problem)
        with torch.no_grad():
            direct = functions["direct_slot_hybrid"](problem.values)
            triality = functions["triality_slot_hybrid"](problem.values)
            delta = functions["delta_chunkwise"](problem.values)
        torch.testing.assert_close(triality, direct, rtol=2e-11, atol=2e-11)
        torch.testing.assert_close(delta, direct, rtol=2e-11, atol=2e-11)

    def test_frozen_synthesis_preserves_claim_boundaries(self) -> None:
        artifacts = Path("artifacts")

        def load(name: str) -> dict[str, object]:
            return json.loads((artifacts / name).read_text(encoding="utf-8"))

        report = analyze(
            load("matched_learned_retrieval_task_a_seeds0_9.json"),
            load("matched_memory_cores_cuda_rtx2070s_20260810.json"),
            load("spin8_blind_alias_action_seeds0_9.json"),
            load("intertwiner_schurscan_equivariant_identification_20260810.json"),
        )
        self.assertTrue(report["task_a"]["implementation"]["passed"])
        self.assertTrue(
            report["task_a"]["hard_routing_paired_verdict"]["decision_rule_supported"]
        )
        self.assertFalse(
            report["task_a"]["direct_triality_effects"][
                "triality_memory_law_advantage_supported"
            ]
        )
        self.assertFalse(
            report["task_b"]["delta_action_row"][
                "task_b_decision_rule_fully_empirically_closed"
            ]
        )
        self.assertFalse(report["programme_verdict"]["dirac_gram_prerequisite"])

    def test_prospective_task_b_replication_closes_only_the_action_row(self) -> None:
        artifacts = Path("artifacts")

        def load(name: str) -> dict[str, object]:
            return json.loads((artifacts / name).read_text(encoding="utf-8"))

        report = analyze(
            load("matched_learned_retrieval_task_a_seeds0_9.json"),
            load("matched_memory_cores_cuda_rtx2070s_20260810.json"),
            load("spin8_blind_alias_action_seeds0_9.json"),
            load("intertwiner_schurscan_equivariant_identification_20260810.json"),
            load("task_b_paired_action_replication_seeds20_29.json"),
        )
        self.assertTrue(
            report["task_b"]["delta_action_row"][
                "task_b_decision_rule_fully_empirically_closed"
            ]
        )
        self.assertTrue(
            report["task_b"]["claim_boundary"][
                "spin8_shared_representation_prior_supported"
            ]
        )
        self.assertFalse(
            report["task_b"]["claim_boundary"][
                "triality_specific_memory_update_supported"
            ]
        )


if __name__ == "__main__":
    unittest.main()
