import unittest

import torch

from hierarchical_matched_retrieval import (
    ROUTE_STRATEGIES,
    evaluate_overwrite_depth,
    evaluate_stream_length,
    transform_route,
    transformed_triality_gauge_diagnostic,
)
from spin8_continuous_alias import FrozenSlotPolicy


class HierarchicalMatchedRetrievalTests(unittest.TestCase):
    def test_route_transforms_have_frozen_support_and_simplex_contract(self) -> None:
        route = torch.softmax(
            torch.randn(7, 8, generator=torch.Generator().manual_seed(11)), dim=-1
        )
        expected_support = {"dense_soft": 8, "block_top1": 2, "hard_top1": 1}
        for strategy in ROUTE_STRATEGIES:
            with self.subTest(strategy=strategy):
                transformed = transform_route(route, strategy)
                self.assertGreaterEqual(float(transformed.min()), 0.0)
                torch.testing.assert_close(
                    transformed.sum(dim=-1), torch.ones(7), atol=1e-7, rtol=1e-7
                )
                self.assertTrue(
                    bool(((transformed > 0).sum(dim=-1) == expected_support[strategy]).all())
                )
                torch.testing.assert_close(
                    transformed.argmax(dim=-1), route.argmax(dim=-1)
                )

    def test_exact_routes_preserve_triality_gauge(self) -> None:
        gaps = transformed_triality_gauge_diagnostic(101, steps=11, batch_size=2)
        for strategy, gap in gaps.items():
            with self.subTest(strategy=strategy):
                self.assertLess(gap, 1e-11)

    def test_oracle_overwrite_rows_and_hard_update_laws_agree(self) -> None:
        row = evaluate_overwrite_depth(
            FrozenSlotPolicy("oracle_both", None, None),
            seed=101,
            overwrite_depth=2,
            radius=0.75,
            perturbation_norm=0.0,
            transport=True,
            batch_size=2,
            chunk_size=16,
        )
        for name in ("direct_oracle", "delta_oracle"):
            self.assertGreaterEqual(
                row["metrics"][name]["minimum_query_cosine"], 1.0 - 1e-12
            )
        self.assertLess(
            row["diagnostics"]["maximum_delta_chunk_recurrent_abs_error"], 1e-12
        )
        self.assertLess(
            row["diagnostics"]["hard_direct_delta_prediction_max_abs_gap"], 1e-12
        )

    def test_stream_cohorts_and_hard_update_laws_agree(self) -> None:
        row = evaluate_stream_length(
            FrozenSlotPolicy("oracle_both", None, None),
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
        self.assertLess(
            row["diagnostics"]["hard_direct_delta_prediction_max_abs_gap"], 1e-12
        )


if __name__ == "__main__":
    unittest.main()
