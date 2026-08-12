from __future__ import annotations

import copy
import unittest

from spin8_global_probe_certificate import run, verify_report


class GlobalFiveProbeCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run()

    def test_exact_global_five_probe_and_four_probe_su2_certificates(self) -> None:
        report = self.report
        self.assertTrue(report["passed"], report)
        self.assertEqual(
            report["five_probe_triality_closure"]["final_ranks"], [8, 8, 8]
        )
        self.assertEqual(
            report["five_probe_triality_closure"]["basis_determinants"],
            {"vector": "-1", "positive": "-1", "negative": "-1"},
        )
        self.assertEqual(report["five_probe_lie_annihilator"]["nullity"], 0)
        self.assertEqual(report["four_probe_lie_annihilator"]["nullity"], 3)
        self.assertTrue(
            report["claims"]["explicit_five_probe_global_stabilizer_is_trivial"]
        )
        self.assertTrue(
            report["claims"]["explicit_four_probe_tuple_has_continuous_su2_stabilizer"]
        )
        self.assertFalse(
            report["claims"]["all_generic_five_probe_allocations_classified"]
        )
        self.assertFalse(report["claims"]["universal_four_probe_insufficiency_proved"])
        self.assertTrue(
            report["claims"][
                "coordinate_exceptional_supports_form_extended_hamming_8_4_4"
            ]
        )
        atlas = report["coordinate_four_probe_atlas"]
        self.assertEqual(atlas["full_closure_count"], 56)
        self.assertEqual(atlas["exceptional_closure_count"], 14)
        self.assertEqual(
            atlas["binary_code"]["weight_enumerator"], {"0": 1, "4": 14, "8": 1}
        )
        self.assertTrue(atlas["binary_code"]["self_dual"])
        self.assertEqual(atlas["binary_code"]["minimum_nonzero_weight"], 4)
        self.assertTrue(atlas["steiner_s_3_4_8"]["every_triple_occurs_once"])

    def test_verifier_recomputes_instead_of_trusting_passed(self) -> None:
        self.assertTrue(verify_report(self.report))
        tampered = copy.deepcopy(self.report)
        tampered["five_probe_triality_closure"]["final_ranks"] = [8, 8, 7]
        self.assertFalse(verify_report(tampered))


if __name__ == "__main__":
    unittest.main()
