"""Tests for bounded exact-stage subprocess execution."""

from __future__ import annotations

import os
import sys
import unittest

from spin8_resource_limits import bounded_environment, run_bounded


class Spin8ResourceLimitTests(unittest.TestCase):
    def test_thread_environment_leaves_cpu_headroom(self) -> None:
        environment = bounded_environment(6)
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            self.assertEqual(environment[name], "6")
        self.assertEqual(environment["SYMPY_GROUND_TYPES"], "flint")

    def test_bounded_child_records_affinity_and_memory(self) -> None:
        workers = min(
            2, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 2
        )
        report = run_bounded(
            [sys.executable, "-c", "print('bounded child ok')"],
            workers=workers,
            memory_gib=0.25,
        )
        self.assertTrue(report["passed"])
        self.assertFalse(report["memory_limit_exceeded"])
        self.assertLess(report["peak_process_tree_rss_gib"], 0.25)
        self.assertEqual(len(report["cpu_affinity"]), workers)


if __name__ == "__main__":
    unittest.main()
