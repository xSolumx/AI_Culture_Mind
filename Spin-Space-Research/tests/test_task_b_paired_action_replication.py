from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from aggregate_task_b_paired_action_replication import aggregate
from task_b_paired_action_replication import verify_retained_parameters

ROOT = Path(__file__).resolve().parents[1]


class TaskBPairedActionReplicationTests(unittest.TestCase):
    def test_retained_coordinates_reconstruct_actions(self) -> None:
        report = json.loads(
            (
                ROOT
                / "artifacts"
                / "task_b_paired_action_replication_seed20.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(verify_retained_parameters(report)["passed"])

        tampered = copy.deepcopy(report)
        tampered["learned_parameters"]["shared"]["coordinates"][0][0] += 0.1
        self.assertFalse(verify_retained_parameters(tampered)["passed"])

    def test_aggregate_applies_eight_of_ten_rule(self) -> None:
        def row(seed: int, *, win: bool) -> dict[str, object]:
            return {
                "experiment": "Task-B prospective paired-action replication",
                "seed": seed,
                "dense": True,
                "retained_parameter_verification": {
                    "passed": True,
                    "maximum_metric_difference": 0.0,
                },
                "decision": {
                    "implementation_passed": True,
                    "representation_prior_win": win,
                    "shared_delta_length2048_mean_cosine": 1.0,
                    "independent_delta_length2048_mean_cosine": 0.5,
                    "routing_matched_independent_delta_length2048_mean_cosine": 0.5,
                    "maximum_direct_delta_error": 0.0,
                },
            }

        rows = [row(seed, win=seed < 8) for seed in range(10)]
        report = aggregate(
            rows, expected_seeds=list(range(10)), verify_parameters=False
        )
        self.assertTrue(
            report["summary"]["task_b_decision_rule_fully_empirically_closed"]
        )
        rows[-1]["decision"]["implementation_passed"] = False
        report = aggregate(
            rows, expected_seeds=list(range(10)), verify_parameters=False
        )
        self.assertFalse(
            report["summary"]["task_b_decision_rule_fully_empirically_closed"]
        )


if __name__ == "__main__":
    unittest.main()
