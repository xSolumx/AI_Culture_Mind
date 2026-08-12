from __future__ import annotations

import unittest

import torch

from benchmark_schurscan_memory_scanners import (
    dense_affine_states,
    dense_homogeneous_states,
    materialize_dense_transition,
    random_slot_problem,
    recurrent_states,
    run_benchmark,
)
from intertwiner_schurscan import homogeneous_affine_matrix


class SchurScanMemoryScannerBenchmarkTests(unittest.TestCase):
    def test_dense_compiler_preserves_structured_slot_recurrence(self) -> None:
        transition, initial = random_slot_problem(
            batch=2,
            length=7,
            dtype=torch.float64,
            device=torch.device("cpu"),
            seed=4,
        )
        action, drive = materialize_dense_transition(transition)
        expected = recurrent_states(transition, initial)
        affine = dense_affine_states(action, drive, initial)
        homogeneous = dense_homogeneous_states(
            homogeneous_affine_matrix(action, drive), initial
        )
        torch.testing.assert_close(affine, expected, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(homogeneous, expected, rtol=1e-11, atol=1e-11)

    def test_small_cpu_benchmark_passes_all_correctness_gates(self) -> None:
        report = run_benchmark(
            device=torch.device("cpu"),
            dtype=torch.float64,
            batch=1,
            lengths=(3, 5),
            warmup=0,
            repeats=1,
        )
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["protocol"]["state_scalars"], 64)
        self.assertFalse(
            report["claim_boundary"]["absolute_memory_architecture_winner_established"]
        )


if __name__ == "__main__":
    unittest.main()
