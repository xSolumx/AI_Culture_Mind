import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from collatz_inverse_frontier import (
    accelerated_odd_step,
    bounded_descent_path_merge,
    hard_records,
    inverse_chain,
    inverse_source,
    scan_alpha,
    scan_alpha_reference,
    scan_alpha_with_stats,
)


class CollatzInverseFrontierTests(unittest.TestCase):
    def test_accelerated_step_and_early_orbit(self) -> None:
        self.assertEqual(accelerated_odd_step(33), (25, 2))
        self.assertEqual(accelerated_odd_step(25), (19, 2))

    def test_inverse_22_reconstructs_33_from_19(self) -> None:
        self.assertEqual(inverse_source((2, 2), 19), 33)
        self.assertEqual(inverse_chain((2, 2), 19), (33, 25, 19))

    def test_inverse_14_formula_and_residue_control(self) -> None:
        self.assertEqual(inverse_source((1, 4), 19), 67)
        self.assertEqual(inverse_chain((1, 4), 19), (67, 101, 19))
        for target in range(1, 500, 2):
            admissible = inverse_chain((1, 4), target) is not None
            self.assertEqual(admissible, target % 18 == 1)

    def test_ascending_source_scan_reproduces_alpha_19(self) -> None:
        alpha = scan_alpha(100, 10_000)
        self.assertEqual(alpha[19], 33)
        records = hard_records(alpha)
        self.assertTrue(any(row == {"target": 19, "alpha": 33} for row in records))

    def test_memoized_scan_matches_reference(self) -> None:
        reference = scan_alpha_reference(501, 20_000)
        optimized, stats = scan_alpha_with_stats(501, 20_000)
        self.assertEqual(optimized, reference)
        self.assertGreater(stats["global_path_merges"], 0)
        self.assertGreater(stats["cached_transitions"], 0)

    def test_memoized_scan_stops_after_all_bounded_targets(self) -> None:
        optimized, stats = scan_alpha_with_stats(101, 100_000)
        self.assertEqual(len(optimized), 51)
        self.assertTrue(stats["stopped_when_complete"])
        self.assertLess(stats["sources_scanned"], 100_000 // 6)

    def test_step_budget_preserves_reference_failure_boundary(self) -> None:
        with self.assertRaises(RuntimeError):
            scan_alpha(101, 100, max_steps=1)

    def test_cache_cap_preserves_alpha_minimality(self) -> None:
        capped, stats = scan_alpha_with_stats(
            501, 20_000, max_cached_states=10
        )
        reference = scan_alpha_reference(501, 20_000)
        self.assertEqual(capped, reference)
        self.assertTrue(stats["path_merge_disabled_by_cache_cap"])

    def test_bounded_descent_path_merge_is_finite_only(self) -> None:
        result = bounded_descent_path_merge(10_000)
        self.assertTrue(result["all_certified"])
        self.assertEqual(result["certified_count"], 10_000)


if __name__ == "__main__":
    unittest.main()
