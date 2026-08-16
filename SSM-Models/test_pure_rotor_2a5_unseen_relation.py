"""Tests for the frozen-checkpoint unseen ``2.A5`` relation evaluator."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import torch
from benchmark_pure_rotor_2a5 import (
    ORACLE_CANDIDATES,
    binary_icosahedral_task,
    evaluate_central_pairs,
)
from evaluate_pure_rotor_2a5_unseen_relation import (
    _config_for_seed,
    _word_product,
    make_unseen_relation_batches,
    select_unseen_relation_words,
    unseen_relation_audit,
)


class UnseenRelationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = (
            Path(__file__).parent
            / "experiments"
            / "artifacts"
            / "pure_rotor_2a5_center_pilot300.json"
        )
        cls.source = json.loads(path.read_bytes())
        cls.task = binary_icosahedral_task()
        cls.selection = select_unseen_relation_words(cls.task, cls.source)

    def test_selection_replays_frozen_words_and_exact_products(self) -> None:
        self.assertEqual(self.selection["length"], 11)
        self.assertEqual(
            self.selection["identity_word"],
            [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        )
        self.assertEqual(
            self.selection["center_word"],
            [0, 1, 1, 0, 1, 1, 1, 0, 2, 2, 0],
        )
        self.assertEqual(
            _word_product(self.task, tuple(self.selection["identity_word"])), 0
        )
        self.assertEqual(
            _word_product(self.task, tuple(self.selection["center_word"])),
            self.task.center_index,
        )
        for counts in self.selection["training_occurrences"].values():
            self.assertEqual(set(counts.values()), {0})

    def test_pair_audit_and_oracle_contracts(self) -> None:
        config = _config_for_seed(self.source, 0)
        identity_word = tuple(self.selection["identity_word"])
        center_word = tuple(self.selection["center_word"])
        batches = make_unseen_relation_batches(
            self.task,
            config,
            length=16,
            relation_position="early",
            identity_word=identity_word,
            center_word=center_word,
        )
        audit = unseen_relation_audit(
            self.task,
            batches,
            identity_word=identity_word,
            center_word=center_word,
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(len(audit["evaluation_schedule_sha256"]), 64)
        metrics = {
            name: evaluate_central_pairs(
                name,
                None,
                batches,
                self.task,
                torch.device("cpu"),
                "parallel",
                "parallel",
                16,
            )
            for name in ORACLE_CANDIDATES
        }
        for name in ("exact_table_oracle", "float64_quaternion_oracle"):
            self.assertEqual(metrics[name]["post_relation_exact_accuracy"], 1.0)
            self.assertEqual(
                metrics[name]["post_relation_central_margin_accuracy"], 1.0
            )
        projective = metrics["projective_a5_oracle"]
        self.assertEqual(projective["post_relation_projective_accuracy"], 1.0)
        self.assertEqual(projective["post_relation_exact_accuracy"], 0.5)
        self.assertEqual(projective["post_relation_central_margin_accuracy"], 0.5)

    def test_completed_exploratory_artifact_contract(self) -> None:
        path = (
            Path(__file__).parent
            / "experiments"
            / "artifacts"
            / "pure_rotor_2a5_unseen_relation_exploratory.json"
        )
        artifact_bytes = path.read_bytes()
        self.assertEqual(
            hashlib.sha256(artifact_bytes).hexdigest(),
            "6580fb4a27cf9a77d19b39ba95fa614026903e81558a605d2e8dcd37c85a3b81",
        )
        report = json.loads(artifact_bytes)
        self.assertEqual(len(report["results"]), 12)
        self.assertEqual(
            report["task"]["selection"], json.loads(json.dumps(self.selection))
        )
        for audits in report["task"]["evaluation_audits"].values():
            self.assertTrue(all(audit["passed"] for audit in audits.values()))
        spin_rows = [
            row for row in report["results"] if row["name"] == "spin_quaternion_scan"
        ]
        self.assertEqual(len(spin_rows), 3)
        for row in spin_rows:
            for length in (16, 64, 128):
                self.assertGreater(
                    row["metrics"][f"early_L{length}"][
                        "post_relation_central_margin_accuracy"
                    ],
                    0.75,
                )


if __name__ == "__main__":
    unittest.main()
