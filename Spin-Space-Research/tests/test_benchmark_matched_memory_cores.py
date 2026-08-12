from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from benchmark_matched_memory_cores import (
    VARIANTS,
    benchmark,
    end_to_end_forward_functions,
    make_problem,
    triality_forward,
)


class MatchedMemoryCoreBenchmarkTests(unittest.TestCase):
    def test_triality_tensor_is_precomputed_outside_timed_forward(self) -> None:
        problem = make_problem(
            batch=1,
            length=5,
            dtype=torch.float64,
            device=torch.device("cpu"),
            seed=3,
        )
        self.assertEqual(problem.rho.device, problem.values.device)
        self.assertEqual(problem.rho.dtype, problem.values.dtype)
        with patch(
            "benchmark_matched_memory_cores.triality_tensor",
            side_effect=AssertionError("rho was reconstructed inside forward"),
        ):
            output = triality_forward(problem, problem.values)
        self.assertEqual(output.shape, (1, 5, 8))

    def test_end_to_end_variants_differentiate_all_trainable_inputs(self) -> None:
        problem = make_problem(
            batch=1,
            length=5,
            dtype=torch.float64,
            device=torch.device("cpu"),
            seed=4,
        )
        functions = end_to_end_forward_functions(
            problem,
            slot_backends={
                "direct_slot_hybrid": "hillis_steele",
                "triality_slot_hybrid": "hillis_steele",
            },
            delta_chunk_sizes={
                "delta_chunkwise": 4,
                "fast_weight_chunkwise": 4,
            },
            encoder_temperature=0.15,
        )
        self.assertEqual(2 * problem.encoder_initial_weight.numel(), 384)
        for name in VARIANTS:
            with self.subTest(name=name):
                values = problem.values.detach().clone().requires_grad_(True)
                write_weight = (
                    problem.encoder_initial_weight.detach().clone().requires_grad_(True)
                )
                query_weight = (
                    problem.encoder_initial_weight.detach().clone().requires_grad_(True)
                )
                functions[name](
                    values, write_weight, query_weight
                ).square().mean().backward()
                for tensor in (values, write_weight, query_weight):
                    self.assertIsNotNone(tensor.grad)
                    self.assertTrue(bool(torch.isfinite(tensor.grad).all()))

    def test_small_replicated_cpu_benchmark_passes(self) -> None:
        report = benchmark(
            device=torch.device("cpu"),
            dtype=torch.float64,
            batch=1,
            lengths=(3,),
            warmup=0,
            repeats=1,
            backward_repeats=1,
            replications=1,
            tuning_repeats=1,
        )
        self.assertTrue(report["passed"], report)
        self.assertTrue(report["protocol"]["end_to_end_address_encoding_timed"])
        self.assertEqual(
            report["protocol"]["end_to_end_encoder_parameters"],
            {name: 384 for name in VARIANTS},
        )
        row = report["rows"][0]
        self.assertIn("core", row)
        self.assertIn("end_to_end", row)
        self.assertEqual(
            row["core"]["timing_orders"]["within_replication"],
            "cyclic rotation each repeat",
        )
        self.assertTrue(
            all(
                diagnostic["all_finite"]
                for diagnostic in row["end_to_end"]["gradient_diagnostics"].values()
            )
        )

    def test_frozen_selection_skips_in_process_tuning(self) -> None:
        selection = {
            "selection_rule": "unit-test frozen selection",
            "inputs": [],
            "selections": {
                "3": {
                    "slot_backends": {
                        "direct_slot_hybrid": "hillis_steele",
                        "triality_slot_hybrid": "hillis_steele",
                    },
                    "delta_chunk_sizes": {
                        "delta_chunkwise": 4,
                        "fast_weight_chunkwise": 4,
                    },
                    "diagnostics": {},
                }
            },
        }
        report = benchmark(
            device=torch.device("cpu"),
            dtype=torch.float64,
            batch=1,
            lengths=(3,),
            warmup=0,
            repeats=1,
            backward_repeats=1,
            replications=1,
            tuning_repeats=99,
            frozen_selection=selection,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["protocol"]["selection_mode"],
            "externally_frozen_disjoint_tuning",
        )
        self.assertEqual(report["protocol"]["tuning_repeats_in_this_process"], 0)


if __name__ == "__main__":
    unittest.main()
