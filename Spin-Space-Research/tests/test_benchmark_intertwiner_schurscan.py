from __future__ import annotations

import unittest

import torch

from benchmark_intertwiner_schurscan import run_benchmark


class IntertwinerSchurScanBenchmarkTests(unittest.TestCase):
    def test_cpu_smoke_records_contract_and_correctness(self) -> None:
        report = run_benchmark(
            device=torch.device("cpu"),
            dtype=torch.float64,
            batch=1,
            lengths=[3],
            backward_max_length=0,
            lift_max_length=3,
            warmup=1,
            repeats=2,
            seed=19,
        )
        self.assertEqual(report["state_dimensions"]["streaming"], 24)
        self.assertEqual(report["state_dimensions"]["homogeneous_full_proof_lift"], 89)
        self.assertFalse(
            report["interpretation_contract"][
                "algebraic_equivalence_implies_bitwise_float_equality"
            ]
        )
        comparable = [row for row in report["forward"] if row["comparable_semantics"]]
        self.assertEqual(len(comparable), 5)
        for row in comparable:
            self.assertLessEqual(
                row["error_vs_recurrent_same_dtype"]["max_abs_over_reference_max"],
                2e-14,
                row,
            )
            self.assertGreater(row["timing"]["median_ms"], 0)
        self.assertEqual(
            comparable[0]["scan_dependency_depth"],
            {"hillis_steele": 2, "work_efficient": 5},
        )
        self.assertEqual(report["forward_backward"], [])


if __name__ == "__main__":
    unittest.main()
