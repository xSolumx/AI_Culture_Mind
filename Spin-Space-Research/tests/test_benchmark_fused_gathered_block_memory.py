from __future__ import annotations

import unittest

import torch

from benchmark_fused_gathered_block_memory import (
    TRITON_AVAILABLE,
    correctness,
    recurrent_correctness,
)
from benchmark_gathered_block_memory import make_problem


@unittest.skipUnless(
    torch.cuda.is_available() and TRITON_AVAILABLE,
    "CUDA and triton-windows required",
)
class FusedGatheredBlockMemoryTests(unittest.TestCase):
    def test_triton_matches_eager_gathered(self) -> None:
        for slots in (64, 256, 1024, 4096):
            with self.subTest(slots=slots):
                problem = make_problem(
                    batch=7,
                    slots=slots,
                    device=torch.device("cuda"),
                    dtype=torch.float32,
                    seed=slots,
                )
                report = correctness(problem)
                self.assertTrue(report["passed"], report)

    def test_recurrent_trajectory_matches_eager_gathered(self) -> None:
        problem = make_problem(
            batch=7,
            slots=64,
            device=torch.device("cuda"),
            dtype=torch.float32,
            seed=257,
        )
        report = recurrent_correctness(problem, steps=33)
        self.assertTrue(report["passed"], report)


if __name__ == "__main__":
    unittest.main()
