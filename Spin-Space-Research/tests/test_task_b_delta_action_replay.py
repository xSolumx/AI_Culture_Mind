from __future__ import annotations

import unittest

import torch

from aggregate_task_b_delta_action_replay import aggregate
from spin8_blind_alias_action import negative_calibration_basis
from spin8_blind_shared_action import sample_teacher
from spin8_continuous_alias import FrozenSlotPolicy
from spin8_triality import torch_triality_generators
from task_b_delta_action_replay import (
    paired_scan_parity,
    paired_sequence_evaluation,
)


class TaskBDeltaActionReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dtype = torch.float64
        generators = torch_triality_generators(dtype=dtype)
        cls.teacher = sample_teacher(seed=101, generators=generators)
        cls.policy = FrozenSlotPolicy("oracle_both", None, None)
        cls.basis = negative_calibration_basis(
            101, dtype=dtype, device=torch.device("cpu")
        )

    def test_direct_and_delta_match_on_one_hot_events(self) -> None:
        report = paired_sequence_evaluation(
            self.teacher.actions,
            self.teacher.actions,
            self.policy,
            seed=101,
            length=64,
            batch_size=12,
        )
        self.assertLess(report["direct_delta_max_state_error"], 1e-12)
        self.assertLess(report["direct_delta_max_prediction_error"], 1e-12)
        self.assertGreater(report["delta"]["minimum_query_cosine"], 1 - 1e-12)

    def test_parallel_recurrent_and_cross_memory_parity(self) -> None:
        report = paired_scan_parity(
            self.teacher.actions, self.policy, seed=101
        )
        self.assertEqual(report["streaming_state_scalars"], 64)
        self.assertLess(report["direct_parallel_recurrent_max_error"], 1e-12)
        self.assertLess(report["delta_parallel_recurrent_max_error"], 1e-12)
        self.assertLess(report["direct_delta_parallel_max_error"], 1e-12)

    def test_aggregate_enforces_complete_unique_seed_cohort(self) -> None:
        def row(seed: int, *, win: bool = True) -> dict[str, object]:
            return {
                "experiment": "Task-B independent-action delta replay",
                "seed": seed,
                "dense": True,
                "decision": {
                    "implementation_passed": True,
                    "representation_prior_win": win,
                    "independent_delta_length2048_mean_cosine": 0.5,
                    "shared_direct_length2048_mean_cosine": 1.0,
                    "maximum_direct_delta_state_error": 0.0,
                    "maximum_direct_delta_prediction_error": 0.0,
                    "minimum_write_route_agreement": 1.0,
                    "minimum_query_route_agreement": 1.0,
                    "minimum_oracle_delta_query_cosine": 1.0,
                },
            }

        report = aggregate([row(0), row(1)], expected_seeds=[0, 1])
        self.assertTrue(
            report["summary"]["task_b_decision_rule_fully_empirically_closed"]
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            aggregate([row(0), row(0)], expected_seeds=[0, 1])


if __name__ == "__main__":
    unittest.main()
