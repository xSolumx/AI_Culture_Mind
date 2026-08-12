from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from large_slot_semantic_hierarchy import (
    ACTION_WORDS,
    BLOCKS,
    ROUTE_STRATEGIES,
    SLOTS,
    SLOTS_PER_BLOCK,
    FrozenRouter,
    SemanticWorld,
    _memory_predictions,
    _memory_write,
    canonicalize_aliases,
    transport_aliases,
)
from spin8_blind_shared_action import sample_teacher
from spin8_triality import torch_triality_generators


class LargeSlotSemanticHierarchyTests(unittest.TestCase):
    def test_world_has_overlapping_fine_blocks(self) -> None:
        world = SemanticWorld.create(103)
        gram = world.centers @ world.centers.transpose(0, 1)
        same_block = []
        other_block = []
        for left in range(SLOTS):
            for right in range(left + 1, SLOTS):
                target = (
                    same_block
                    if left // SLOTS_PER_BLOCK == right // SLOTS_PER_BLOCK
                    else other_block
                )
                target.append(float(gram[left, right]))
        self.assertGreater(sum(same_block) / len(same_block), 0.55)
        self.assertLess(abs(sum(other_block) / len(other_block)), 0.1)

    def test_spin8_transport_canonicalizes_every_view(self) -> None:
        generators = torch_triality_generators(dtype=torch.float64)
        actions = sample_teacher(seed=103, generators=generators).actions
        canonical = F.normalize(
            torch.randn(12, 8, generator=torch.Generator().manual_seed(4)), dim=-1
        ).to(torch.float64)
        words = torch.arange(12).remainder(ACTION_WORDS)
        views = torch.arange(12).remainder(3)
        raw = transport_aliases(canonical, words, views, actions)
        recovered = canonicalize_aliases(raw, words, views, actions)
        self.assertLess(float((canonical - recovered).abs().max()), 2e-14)

    def test_route_support_and_hard_memory_parity(self) -> None:
        generator = torch.Generator().manual_seed(7)
        router = FrozenRouter(
            "shared",
            torch.randn(BLOCKS, 8, generator=generator, dtype=torch.float64),
            torch.randn(SLOTS, 8, generator=generator, dtype=torch.float64),
        )
        aliases = F.normalize(
            torch.randn(5, 8, generator=generator, dtype=torch.float64), dim=-1
        )
        views = torch.arange(5).remainder(3)
        expected_support = {
            "dense_soft": SLOTS,
            "block_top1": SLOTS_PER_BLOCK,
            "hard_top1": 1,
        }
        routes = {}
        for strategy in ROUTE_STRATEGIES:
            route = router.routes(aliases, views, strategy)
            routes[strategy] = route
            self.assertTrue(
                bool(
                    (
                        (route > 0).sum(dim=-1)
                        == expected_support[strategy]
                    ).all()
                )
            )
            torch.testing.assert_close(
                route.sum(dim=-1), torch.ones(5, dtype=torch.float64)
            )

        hard = routes["hard_top1"][None, None]
        direct = torch.zeros(1, 1, 5, SLOTS, 8, dtype=torch.float64)
        delta = torch.zeros_like(direct)
        values = torch.randn(5, 8, generator=generator, dtype=torch.float64)
        direct, delta = _memory_write(direct, delta, hard, values)
        self.assertEqual(float((direct - delta).abs().max()), 0.0)
        direct_prediction, delta_prediction = _memory_predictions(
            direct, delta, hard
        )
        self.assertEqual(
            float((direct_prediction - delta_prediction).abs().max()), 0.0
        )


if __name__ == "__main__":
    unittest.main()
