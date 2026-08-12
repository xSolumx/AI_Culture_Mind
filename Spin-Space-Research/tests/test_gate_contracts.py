from __future__ import annotations

import unittest

from spin8_gate_contracts import GATES, validate_gate_contracts


class GateContractTests(unittest.TestCase):
    def test_every_maintained_suite_has_a_valid_evidence_contract(self) -> None:
        self.assertEqual(validate_gate_contracts(), [])

    def test_open_gates_cannot_be_mistaken_for_proved_gates(self) -> None:
        by_id = {gate.gate_id: gate for gate in GATES}
        for gate_id in (
            "global_five_query_exact_design",
            "unrestricted_dirac_gram",
            "triality_specific_ml_advantage",
        ):
            gate = by_id[gate_id]
            self.assertEqual(gate.status, "open")
            self.assertNotIn("positivity_certificate", gate.evidence_layers)
            self.assertTrue(any("Open:" in item for item in gate.limitations))

    def test_frozen_two_edge_atlas_is_not_reopened_by_stale_history(self) -> None:
        gate = next(
            item for item in GATES if item.gate_id == "two_edge_global_positivity"
        )
        self.assertEqual(gate.status, "proved_exact")
        self.assertIn("positivity_certificate", gate.evidence_layers)
        self.assertIn(
            "artifacts/spin8_dirac_two_edge_atlas_20260807.json", gate.artifacts
        )

    def test_spin9_coupled_finite_radius_gate_is_promoted(self) -> None:
        gate = next(
            item
            for item in GATES
            if item.gate_id == "spin9_coupled_v1_v5_finite_radius"
        )
        self.assertEqual(gate.status, "proved_exact")
        self.assertIn("positivity_certificate", gate.evidence_layers)
        self.assertIn(
            "artifacts/spin9_v1_v5_theorem_20260812.json",
            gate.artifacts,
        )
        self.assertTrue(any("second supported V5" in row for row in gate.limitations))

    def test_hybrid_cayley_gate_names_its_nonlocal_proof_input(self) -> None:
        gate = next(
            item
            for item in GATES
            if item.gate_id == "balanced_cayley_information_family"
        )
        self.assertEqual(gate.status, "proved_hybrid")
        self.assertIn("external_theorem", gate.evidence_layers)
        self.assertTrue(gate.external_inputs)
        self.assertTrue(
            any("Local Lie-rank" in limitation for limitation in gate.limitations)
        )

    def test_empirical_gates_disclose_replay_boundary(self) -> None:
        empirical = [gate for gate in GATES if gate.status == "empirical"]
        self.assertGreaterEqual(len(empirical), 2)
        for gate in empirical:
            self.assertIn("raw_artifact", gate.evidence_layers)
            self.assertTrue(gate.artifacts)
            self.assertEqual(gate.replay_tier, "artifact_only_empirical")


if __name__ == "__main__":
    unittest.main()
