from __future__ import annotations

import unittest

from aggregate_large_slot_semantic_hierarchy import aggregate


class AggregateLargeSlotSemanticHierarchyTests(unittest.TestCase):
    @staticmethod
    def row(seed: int, *, shared: bool, hierarchy: bool) -> dict[str, object]:
        return {
            "experiment": "large-slot overlapping-semantic hierarchy",
            "seed": seed,
            "retained_parameter_verification": {
                "passed": True,
                "maximum_difference": 0.0,
            },
            "decision": {
                "implementation_passed": True,
                "shared_router_completion_passed": shared,
                "hierarchical_routing_passed": hierarchy,
                "shared_heldout_hard_accuracy": 0.9,
                "independent_heldout_hard_accuracy": 0.1,
                "independent_observed_hard_accuracy": 0.9,
                "mean_block_improvement_over_dense": {
                    "direct": 0.1,
                    "delta": 0.05,
                },
                "maximum_canonicalization_error": 1e-15,
                "maximum_hard_direct_delta_state_error": 0.0,
                "maximum_hard_direct_delta_prediction_error": 0.0,
            },
        }

    def test_eight_of_ten_rule_and_seed_integrity(self) -> None:
        rows = [
            self.row(seed, shared=seed < 8, hierarchy=seed < 8)
            for seed in range(10)
        ]
        report = aggregate(
            rows, expected_seeds=list(range(10)), verify_parameters=False
        )
        self.assertTrue(report["summary"]["shared_router_completion_supported"])
        self.assertTrue(report["summary"]["hierarchical_routing_supported"])
        rows[-1]["decision"]["implementation_passed"] = False
        report = aggregate(
            rows, expected_seeds=list(range(10)), verify_parameters=False
        )
        self.assertFalse(report["summary"]["shared_router_completion_supported"])
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            aggregate(rows[:-1], expected_seeds=list(range(10)), verify_parameters=False)


if __name__ == "__main__":
    unittest.main()
