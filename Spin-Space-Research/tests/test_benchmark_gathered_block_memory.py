from __future__ import annotations

import unittest

import torch

from benchmark_gathered_block_memory import (
    benchmark,
    correctness,
    cyclic_orders,
    make_problem,
)


class GatheredBlockMemoryBenchmarkTests(unittest.TestCase):
    def test_timing_orders_rotate_across_blocks(self) -> None:
        variants = ("a", "b", "c")
        self.assertEqual(
            cyclic_orders(variants, rotation=1, timing_blocks=4),
            [
                ("b", "c", "a"),
                ("c", "a", "b"),
                ("a", "b", "c"),
                ("b", "c", "a"),
            ],
        )

    def test_gathered_matches_masked_full_for_both_laws(self) -> None:
        problem = make_problem(
            batch=5,
            slots=64,
            device=torch.device("cpu"),
            dtype=torch.float64,
            seed=11,
        )
        report = correctness(problem)
        self.assertTrue(report["passed"], report)
        for row in report["rows"].values():
            self.assertLessEqual(row["maximum_state_error"], 1e-12)
            self.assertLessEqual(row["maximum_prediction_error"], 1e-12)

    def test_small_cpu_benchmark_preserves_state_contract(self) -> None:
        report = benchmark(
            device=torch.device("cpu"),
            slots=(64,),
            batches=(2,),
            warmup=1,
            timing_blocks=2,
            inner_calls=2,
            seed=12,
        )
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["rows"][0]["logical_state_scalars"], 2 * 64 * 8)
        self.assertEqual(
            report["router_parameters_at_max_slots"]["three_independent"],
            3 * report["router_parameters_at_max_slots"]["shared"],
        )
        self.assertEqual(len(report["rows"][0]["timing_block_orders"]), 2)


if __name__ == "__main__":
    unittest.main()
