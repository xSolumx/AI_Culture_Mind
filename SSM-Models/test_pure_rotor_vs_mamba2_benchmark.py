from __future__ import annotations

import unittest

import torch
from benchmark_pure_rotor_vs_mamba2 import (
    BenchmarkConfig,
    build_models,
    decoder_tying_metadata,
    identity_ablation_metadata,
    logits_for,
    parameter_count,
)


class PureRotorMamba2BenchmarkTests(unittest.TestCase):
    def test_matched_candidates_emit_byte_logits(self) -> None:
        config = BenchmarkConfig(steps=1, batch_size=2, sequence_length=7)
        torch.manual_seed(117)
        models = build_models(config)
        counts = {name: parameter_count(model) for name, model in models.items()}
        self.assertEqual(counts["pure_rotor"], counts["identity_rotation_ablation"])
        self.assertLess(
            abs(counts["pure_rotor"] - counts["mamba2_transformers"])
            / max(counts["pure_rotor"], counts["mamba2_transformers"]),
            0.05,
        )
        decoder_metadata = decoder_tying_metadata(
            models["pure_rotor"], models["mamba2_transformers"]
        )
        self.assertTrue(decoder_metadata["pure_rotor_input_output_embeddings_tied"])
        self.assertTrue(decoder_metadata["mamba2_input_output_embeddings_tied"])
        tokens = torch.randint(0, 256, (2, 7))
        for name, model in models.items():
            logits = logits_for(name, model.eval(), tokens, "parallel")
            self.assertEqual(tuple(logits.shape), (2, 7, 256))
        self.assertTrue(bool(torch.isfinite(logits).all()))

        identity_metadata = identity_ablation_metadata(
            models["identity_rotation_ablation"]
        )
        self.assertTrue(
            identity_metadata["raw_parameter_match_is_not_effective_capacity_match"]
        )
        self.assertGreater(
            identity_metadata["disabled_rotor_controller_parameter_count"], 0
        )
        self.assertLess(
            identity_metadata[
                "effective_parameter_count_if_rotation_is_fixed_identity"
            ],
            identity_metadata["raw_parameter_count"],
        )

    def test_mamba_shape_constraint_is_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "2\\*hidden_size"):
            build_models(BenchmarkConfig(mamba_hidden_size=95))


if __name__ == "__main__":
    unittest.main()
