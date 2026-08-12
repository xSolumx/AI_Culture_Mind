from __future__ import annotations

import copy
import unittest

from spin8_coordinate_geometry import run, verify_report


class Spin8CoordinateGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run()

    def test_exact_binary_coordinate_classification(self) -> None:
        report = self.report
        self.assertTrue(report["passed"], report)
        self.assertTrue(report["support_law"]["support_is_coordinate_xor"])
        self.assertEqual(report["support_law"]["checked_products"], 64)

        four = report["four_probe_atlas"]
        self.assertEqual(four["multiview_sensors_evaluated"], 10416)
        self.assertEqual(four["mismatch_count"], 0)
        self.assertEqual(four["binary_rank_counts"], {"3": 1680, "4": 8736})
        self.assertEqual(four["full_triality_closure_count"], 0)

        five = report["five_probe_atlas"]
        self.assertEqual(five["multiview_sensors_evaluated"], 42336)
        self.assertEqual(five["mismatch_count"], 0)
        self.assertEqual(five["binary_rank_counts"], {"3": 672, "4": 20160, "5": 21504})
        self.assertEqual(five["full_triality_closure_count"], 21504)

        self.assertEqual(
            [
                (
                    row["coordinates_per_representation"],
                    row["exact_constraint_rank"],
                    row["exact_lie_nullity"],
                    row["distinct_closures"],
                )
                for row in report["exact_lie_rank_by_distinct_closure"]
            ],
            [(2, 20, 8, 112), (4, 25, 3, 28), (8, 28, 0, 1)],
        )

    def test_exact_compact_stabilizer_chain(self) -> None:
        chain = self.report["representative_stabilizer_chain"]
        self.assertEqual(
            chain["binary_rank_3"]["exact_lie_type_certificate"]["classification"],
            "compact semisimple A2, hence su(3)",
        )
        self.assertEqual(
            chain["binary_rank_4"]["exact_lie_type_certificate"]["classification"],
            "compact simple A1, hence su(2)",
        )
        for rank in ("binary_rank_3", "binary_rank_4"):
            certificate = chain[rank]["exact_lie_type_certificate"]
            self.assertEqual(certificate["centre_dimension"], 0)
            self.assertEqual(
                certificate["derived_algebra_rank"], certificate["dimension"]
            )
            self.assertTrue(certificate["killing_form_negative_definite"])

    def test_verifier_recomputes_full_report(self) -> None:
        self.assertTrue(verify_report(self.report))
        tampered = copy.deepcopy(self.report)
        tampered["five_probe_atlas"]["binary_rank_counts"]["5"] -= 1
        self.assertFalse(verify_report(tampered))


if __name__ == "__main__":
    unittest.main()
