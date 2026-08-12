from __future__ import annotations

import copy
import unittest

from spin8_continuous_probe_orbits import run, verify_report


class Spin8ContinuousProbeOrbitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run()

    def test_all_four_probe_allocations_have_exact_invariant_bound(self) -> None:
        report = self.report
        self.assertTrue(report["passed"], report)
        self.assertTrue(report["exact_invariance_prerequisites"]["passed"])
        rows = report["four_probe_allocation_certificates"]
        self.assertEqual(
            [row["allocation_up_to_triality"] for row in rows],
            [[4, 0, 0], [3, 1, 0], [2, 2, 0], [2, 1, 1]],
        )
        self.assertEqual([row["exact_action_rank"] for row in rows], [22, 25, 25, 25])
        self.assertEqual(
            [row["exact_invariant_jacobian_rank"] for row in rows], [6, 3, 3, 3]
        )
        self.assertTrue(all(row["rank_saturates_invariant_bound"] for row in rows))
        self.assertEqual(
            [row["exact_stabilizer_lie_type"]["classification"] for row in rows],
            [
                "compact semisimple A1+A1, hence spin(4)",
                "compact simple A1, hence su(2)",
                "compact simple A1, hence su(2)",
                "compact simple A1, hence su(2)",
            ],
        )

    def test_every_mixed_five_probe_allocation_has_a_global_free_point(self) -> None:
        rows = self.report["five_probe_allocation_certificates"][
            "mixed_allocation_representatives"
        ]
        self.assertEqual(
            [row["allocation_up_to_triality"] for row in rows],
            [[4, 1, 0], [3, 2, 0], [3, 1, 1], [2, 2, 1]],
        )
        self.assertTrue(all(row["closure_sizes"] == [8, 8, 8] for row in rows))
        self.assertTrue(all(row["exact_action_rank"] == 28 for row in rows))
        self.assertTrue(
            all(row["global_stabilizer_trivial_by_full_closure"] for row in rows)
        )
        self.assertIn("classical_global_input", self.report["proof_layers"])
        self.assertIn(
            "principal stratum", self.report["proof_layers"]["classical_global_input"]
        )
        single = self.report["five_probe_allocation_certificates"][
            "single_view_control"
        ]
        self.assertEqual(single["exact_stabilizer_dimension"], 3)
        self.assertFalse(
            self.report["claims"]["single_view_five_probe_sensing_is_sufficient"]
        )

    def test_verifier_recomputes_the_certificate(self) -> None:
        self.assertTrue(verify_report(self.report))
        tampered = copy.deepcopy(self.report)
        tampered["four_probe_allocation_certificates"][1]["exact_action_rank"] = 26
        self.assertFalse(verify_report(tampered))


if __name__ == "__main__":
    unittest.main()
