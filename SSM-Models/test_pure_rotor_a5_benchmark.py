from __future__ import annotations

import unittest

import torch
from benchmark_pure_rotor_a5 import (
    A5BenchmarkConfig,
    a5_presentation_task,
    build_models,
    evaluate,
    logits_for,
    make_evaluation_batches,
    make_training_batches,
    parameter_count,
    task_split_audit,
)


class PureRotorA5BenchmarkTests(unittest.TestCase):
    def test_requested_a5_presentation_and_split_are_valid(self) -> None:
        task = a5_presentation_task()
        presentation = task.presentation
        self.assertTrue(presentation["a_squared_is_identity"])
        self.assertTrue(presentation["b_cubed_is_identity"])
        self.assertTrue(presentation["ab_to_fifth_is_identity"])
        self.assertEqual(presentation["generated_subgroup_order"], 60)
        self.assertEqual(task.input_symbols, ("a", "b", "b_inverse"))

        config = A5BenchmarkConfig(
            steps=4,
            batch_size=32,
            training_length=16,
            validation_batches=2,
            validation_batch_size=32,
            evaluation_lengths=(2, 16),
        )
        training = make_training_batches(task, config)
        evaluations = {
            length: make_evaluation_batches(task, config, length)
            for length in config.evaluation_lengths
        }
        audit = task_split_audit(task, training, evaluations)
        self.assertEqual(audit["training_pair_audit"]["pair_occurrences"], 0)
        self.assertEqual(
            audit["training_coverage"]["observed_group_states"], task.group.order
        )
        self.assertTrue(
            audit["exact_training_language_coverage"][
                "all_group_states_reachable_within_training_length"
            ]
        )
        self.assertLessEqual(
            audit["exact_training_language_coverage"][
                "maximum_shortest_witness_length"
            ],
            config.training_length,
        )
        self.assertEqual(len(audit["training_schedule_sha256"]), 64)
        for evaluation in audit["evaluation_pair_audits"].values():
            self.assertEqual(
                evaluation["sequences_with_pair"], evaluation["total_sequences"]
            )

    def test_parameter_near_candidates_emit_a5_prefix_logits(self) -> None:
        task = a5_presentation_task()
        config = A5BenchmarkConfig(
            steps=1,
            batch_size=2,
            training_length=3,
            validation_batches=1,
            validation_batch_size=2,
            evaluation_lengths=(2,),
        )
        torch.manual_seed(2718)
        models = build_models(task, config)
        counts = {name: parameter_count(model) for name, model in models.items()}
        self.assertEqual(counts["pure_rotor"], counts["identity_rotation_ablation"])
        self.assertLess(
            abs(counts["pure_rotor"] - counts["mamba2_transformers"])
            / max(counts["pure_rotor"], counts["mamba2_transformers"]),
            0.05,
        )
        tokens = torch.randint(0, len(task.input_elements), (2, 3))
        for name, model in models.items():
            logits = logits_for(name, model.eval(), tokens, "parallel")
            self.assertEqual(tuple(logits.shape), (2, 3, task.group.order))
            self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_mamba_shape_constraint_is_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "2\\*hidden_size"):
            build_models(
                a5_presentation_task(), A5BenchmarkConfig(mamba_hidden_size=31)
            )

    def test_evaluation_microbatching_preserves_rotor_metrics(self) -> None:
        task = a5_presentation_task()
        config = A5BenchmarkConfig(
            steps=1,
            batch_size=2,
            training_length=2,
            validation_batches=1,
            validation_batch_size=4,
            evaluation_lengths=(2,),
        )
        torch.manual_seed(314)
        model = build_models(task, config)["pure_rotor"].eval()
        batches = make_evaluation_batches(task, config, 2)
        whole = evaluate(
            "pure_rotor", model, batches, torch.device("cpu"), "parallel", 4
        )
        split = evaluate(
            "pure_rotor", model, batches, torch.device("cpu"), "parallel", 1
        )
        self.assertAlmostEqual(whole["prefix_nll"], split["prefix_nll"], places=6)
        self.assertEqual(whole["all_prefix_accuracy"], split["all_prefix_accuracy"])
        self.assertEqual(
            whole["final_position_accuracy"], split["final_position_accuracy"]
        )


if __name__ == "__main__":
    unittest.main()
