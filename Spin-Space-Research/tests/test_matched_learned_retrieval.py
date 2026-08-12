import unittest

import torch

from matched_learned_retrieval import (
    evaluate_overwrite_depth,
    evaluate_stream_length,
    perturb_route,
    perturb_unit,
)
from spin8_continuous_alias import AliasWorld, FrozenKeyPolicy, FrozenSlotPolicy


class MatchedLearnedRetrievalTests(unittest.TestCase):
    def test_unit_perturbation_is_normalized_and_reproducible(self) -> None:
        value = torch.eye(8, dtype=torch.float64)[:4]
        left, observed_left = perturb_unit(
            value, 0.1, generator=torch.Generator().manual_seed(17)
        )
        right, observed_right = perturb_unit(
            value, 0.1, generator=torch.Generator().manual_seed(17)
        )
        torch.testing.assert_close(left, right, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            left.norm(dim=-1), torch.ones(4, dtype=torch.float64)
        )
        self.assertGreater(observed_left, 0.0)
        self.assertEqual(observed_left, observed_right)

    def test_route_perturbation_stays_on_simplex(self) -> None:
        route = torch.eye(8, dtype=torch.float64)[:4]
        perturbed, observed = perturb_route(
            route, 0.2, generator=torch.Generator().manual_seed(19)
        )
        self.assertGreaterEqual(float(perturbed.min()), 0.0)
        torch.testing.assert_close(
            perturbed.sum(dim=-1), torch.ones(4, dtype=torch.float64)
        )
        self.assertGreater(observed, 0.0)

    def test_oracle_memory_rows_and_chunk_scan_are_exact(self) -> None:
        world = AliasWorld.create(101, dtype=torch.float64, device=torch.device("cpu"))
        slot_policy = FrozenSlotPolicy("oracle_both", None, None)
        # Dotting an alias against the orthonormal centers recovers its ideal
        # eight-dimensional semantic key before normalization.
        key_policy = FrozenKeyPolicy(world.centers, world.centers)
        for transport in (False, True):
            with self.subTest(transport=transport):
                row = evaluate_overwrite_depth(
                    slot_policy,
                    key_policy,
                    seed=101,
                    overwrite_depth=2,
                    radius=0.75,
                    perturbation_norm=0.0,
                    transport=transport,
                    batch_size=2,
                    chunk_size=16,
                )
                for name in (
                    "direct_slot_joint",
                    "triality_slot_joint",
                    "delta_chunk_joint",
                    "delta_chunk_oracle",
                ):
                    self.assertGreaterEqual(
                        row["metrics"][name]["minimum_query_cosine"],
                        1.0 - 1e-12,
                    )
                self.assertLess(
                    row["diagnostics"]["learned_delta_chunk_recurrent_max_abs_error"],
                    1e-12,
                )
                self.assertLess(
                    row["diagnostics"]["direct_triality_prediction_max_abs_gap"],
                    1e-12,
                )
                self.assertLess(
                    row["metrics"]["fast_weight_joint"]["mean_query_cosine"],
                    0.95,
                )

    def test_long_stream_has_matched_hot_and_cold_query_cohorts(self) -> None:
        world = AliasWorld.create(101, dtype=torch.float64, device=torch.device("cpu"))
        row = evaluate_stream_length(
            FrozenSlotPolicy("oracle_both", None, None),
            FrozenKeyPolicy(world.centers, world.centers),
            seed=101,
            length=64,
            radius=0.75,
            perturbation_norm=0.0,
            transport=True,
            batch_size=8,
            chunk_size=16,
        )
        self.assertEqual(row["query_count"], 256)
        self.assertEqual(row["hot_query_count"], 128)
        self.assertEqual(row["cold_query_count"], 128)
        for name in (
            "direct_slot_joint",
            "triality_slot_joint",
            "delta_chunk_joint",
            "delta_chunk_oracle",
        ):
            self.assertGreaterEqual(
                row["metrics"][name]["minimum_query_cosine"], 1.0 - 1e-12
            )
        self.assertLess(
            row["diagnostics"]["oracle_delta_chunk_recurrent_max_abs_error"],
            1e-12,
        )


if __name__ == "__main__":
    unittest.main()
