"""Exact split, oracle, and model tests for the multi-relation benchmark."""

from __future__ import annotations

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

import torch
from benchmark_spin_multirelation_2a5 import (
    ALL_ORACLES,
    MultiRelationConfig,
    build_models,
    evaluate_relation_pairs,
    logits_for,
    make_multirelation_task,
    make_relation_pair_batches,
    make_training_batches,
    parameter_count,
    relation_pair_evaluation_audit,
    shortest_legal_state_witnesses,
    training_split_audit,
)
from pdssm_group_actions import ExactRegularPD


class SpinMultiRelationBinaryA5Tests(unittest.TestCase):
    def test_frozen_pilot_artifact_contract_and_registered_gates(self) -> None:
        artifact_path = (
            Path(__file__).parent
            / "experiments"
            / "artifacts"
            / "spin_2a5_multirelation_pilot300.json"
        )
        artifact_bytes = artifact_path.read_bytes()
        self.assertEqual(
            hashlib.sha256(artifact_bytes).hexdigest(),
            "054527e8c3e30d64df30217c2128616e82e7f2025278c200cdbde611647fe6d4",
        )
        report = json.loads(artifact_bytes)
        self.assertEqual(report["seeds"], [0])
        self.assertEqual(report["coordinates"], ["e", "a", "b"])
        self.assertEqual(len(report["results"]), 15)
        self.assertTrue(
            report["integrity"]["same_input_training_schedule_across_coordinates"]
        )
        self.assertTrue(
            report["integrity"]["same_input_evaluation_schedule_across_coordinates"]
        )
        rows = {
            (row["coordinate_label"], row["name"]): row for row in report["results"]
        }
        for coordinate in report["coordinates"]:
            spin = rows[(coordinate, "spin_quaternion_scan")]
            for relation in ("a_squared", "b_cubed", "ab_fifth"):
                for length in (64, 128):
                    key = f"{relation}__early_L{length}"
                    spin_metrics = spin["final_relation_metrics"][key]
                    self.assertGreater(
                        spin_metrics["post_relation_central_margin_accuracy"], 0.75
                    )
                    spin_exact = spin_metrics["post_relation_exact_accuracy"]
                    for candidate in report["candidates"]:
                        if candidate != "spin_quaternion_scan":
                            alternative = rows[(coordinate, candidate)]
                            self.assertGreater(
                                spin_exact,
                                alternative["final_relation_metrics"][key][
                                    "post_relation_exact_accuracy"
                                ],
                            )

    def test_all_conjugated_presentations_and_legal_languages(self) -> None:
        expected_inputs = {
            "e": (1, 2, 4, 0),
            "a": (1, 17, 19, 0),
            "b": (24, 2, 4, 0),
        }
        expected_distribution = {
            0: 1,
            1: 3,
            2: 6,
            3: 11,
            4: 15,
            5: 19,
            6: 17,
            7: 18,
            8: 18,
            9: 8,
            10: 4,
        }
        for coordinate, inputs in expected_inputs.items():
            task = make_multirelation_task(coordinate)
            self.assertEqual(task.binary.input_elements, inputs)
            self.assertEqual(task.binary.group.order, 120)
            self.assertEqual(
                task.binary.group_table_sha256,
                ("ee26aff5719e54cf28eccc7a0259c79eeb27c7e06bb18134bf7ca910773f58c9"),
            )
            witnesses = shortest_legal_state_witnesses(task)
            self.assertEqual(len(witnesses), 120)
            self.assertEqual(
                Counter(map(len, witnesses.values())), expected_distribution
            )

    def test_training_forbids_all_relations_but_realizes_every_state(self) -> None:
        config = MultiRelationConfig(
            steps=8,
            batch_size=16,
            training_length=16,
            validation_batches=1,
            validation_pairs_per_batch=4,
            evaluation_lengths=(16,),
        )
        input_hashes = set()
        for coordinate in ("e", "a", "b"):
            task = make_multirelation_task(coordinate)
            audit = training_split_audit(task, make_training_batches(task, config))
            self.assertEqual(
                audit["forbidden_relation_occurrences"],
                {"a_squared": 0, "b_cubed": 0, "ab_fifth": 0},
            )
            self.assertEqual(audit["observed_binary_target_states"], 120)
            self.assertEqual(audit["observed_projective_target_states"], 60)
            self.assertEqual(audit["observed_center_bits"], [0, 1])
            self.assertTrue(
                audit["exact_training_language_coverage"]["all_binary_states_reachable"]
            )
            self.assertEqual(
                audit["exact_training_language_coverage"][
                    "maximum_shortest_witness_length"
                ],
                10,
            )
            input_hashes.add(audit["input_schedule_sha256"])
        self.assertEqual(len(input_hashes), 1)

    def test_relation_pairs_are_exact_and_coordinate_identical(self) -> None:
        config = MultiRelationConfig(
            steps=1,
            batch_size=120,
            training_length=16,
            validation_batches=1,
            validation_pairs_per_batch=4,
            evaluation_lengths=(16,),
        )
        hashes: dict[str, set[str]] = {}
        for coordinate in ("e", "a", "b"):
            task = make_multirelation_task(coordinate)
            for relation in task.relations:
                for position in ("early", "late"):
                    key = f"{relation.key}/{position}"
                    batches = make_relation_pair_batches(
                        task, relation, config, 16, position
                    )
                    audit = relation_pair_evaluation_audit(task, relation, batches)
                    self.assertTrue(audit["passed"])
                    self.assertEqual(
                        audit["exact_central_partner_checks"],
                        audit["post_relation_pair_positions"],
                    )
                    self.assertEqual(
                        audit["projective_match_checks"],
                        audit["post_relation_pair_positions"],
                    )
                    self.assertEqual(
                        audit["forced_block_checks"],
                        2 * audit["paired_sequences"],
                    )
                    hashes.setdefault(key, set()).add(audit["input_schedule_sha256"])
        self.assertTrue(all(len(values) == 1 for values in hashes.values()))

    def test_parameter_near_models_and_delta_dispatch(self) -> None:
        task = make_multirelation_task("e")
        config = MultiRelationConfig(
            steps=1,
            batch_size=2,
            training_length=16,
            validation_batches=1,
            validation_pairs_per_batch=2,
            evaluation_lengths=(16,),
        )
        torch.manual_seed(2026)
        models = build_models(task.binary, config)
        counts = {name: parameter_count(model) for name, model in models.items()}
        self.assertEqual(
            counts,
            {
                "pure_rotor": 29370,
                "identity_rotation_ablation": 29370,
                "spin_quaternion_scan": 29624,
                "mamba2_transformers": 29300,
                "delta_product_reference": 29288,
            },
        )
        self.assertLess(
            (max(counts.values()) - min(counts.values())) / max(counts.values()),
            0.02,
        )
        tokens = torch.randint(0, 4, (2, 16))
        for name, model in models.items():
            logits = logits_for(
                name,
                model.eval(),
                tokens,
                rotor_scan_mode="parallel",
                quaternion_scan_mode="parallel",
                delta_scan_mode="parallel",
            )
            self.assertEqual(tuple(logits.shape), (2, 16, 120))
            self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_oracle_contracts_hold_for_every_relation(self) -> None:
        task = make_multirelation_task("e")
        config = MultiRelationConfig(
            steps=1,
            batch_size=120,
            training_length=16,
            validation_batches=1,
            validation_pairs_per_batch=4,
            evaluation_lengths=(16,),
            evaluation_microbatch_size=3,
        )
        exact_pd = ExactRegularPD(task.binary.group, task.binary.input_elements)
        for relation in task.relations:
            batches = make_relation_pair_batches(task, relation, config, 16, "early")
            for name in ALL_ORACLES:
                model = exact_pd if name == "exact_regular_pd_oracle" else None
                metrics = evaluate_relation_pairs(
                    name,
                    model,
                    batches,
                    task.binary,
                    torch.device("cpu"),
                    config,
                    rotor_scan_mode="parallel",
                    quaternion_scan_mode="parallel",
                    delta_scan_mode="parallel",
                )
                if name == "projective_a5_oracle":
                    self.assertEqual(metrics["post_relation_projective_accuracy"], 1.0)
                    self.assertEqual(metrics["post_relation_exact_accuracy"], 0.5)
                    self.assertEqual(
                        metrics["post_relation_central_margin_accuracy"], 0.5
                    )
                else:
                    self.assertEqual(metrics["post_relation_exact_accuracy"], 1.0)
                    self.assertEqual(metrics["post_relation_center_bit_accuracy"], 1.0)
                    self.assertEqual(metrics["paired_final_exact_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
