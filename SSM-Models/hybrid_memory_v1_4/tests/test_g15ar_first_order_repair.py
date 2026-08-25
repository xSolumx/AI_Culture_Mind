"""Contracts for the prospectively frozen G15A-R learning repair."""

from __future__ import annotations

import torch

from hybrid_memory_v1_4.g15al_learned_coordinate_cohort import (
    TokenCoordinateController,
)
from hybrid_memory_v1_4.g15ar_first_order_repair import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_RECIPES,
    DEVELOPMENT_SEEDS,
    SELECTION_ORDER,
    VALIDATION_MILESTONES,
    _build_optimizer,
    _learning_rate,
    _ordered_inverse_batch,
    _select_recipe,
    _singleton_batch,
    _training_batch,
    quality_config,
)
from hybrid_memory_v1_4.optimizers import (
    BlockScalarSecondMomentAdamW,
    ScalarSecondMomentAdamW,
)


def test_quality_protocol_seeds_recipes_and_budget_are_frozen() -> None:
    config = quality_config()
    assert config.development_seeds == DEVELOPMENT_SEEDS == (2237, 2239, 2243)
    assert config.confirmation_seeds == CONFIRMATION_SEEDS == (2251, 2267, 2273)
    assert config.updates == 600
    assert config.batch_size == 16
    assert config.validation_milestones == VALIDATION_MILESTONES
    assert DEVELOPMENT_RECIPES == (
        "G-fixed/random",
        "G-decay/random",
        "B-decay/random",
        "G-decay/curriculum",
        "B-decay/curriculum",
    )
    assert SELECTION_ORDER == DEVELOPMENT_RECIPES[1:]


def test_learning_rate_schedule_has_exact_frozen_boundaries() -> None:
    assert _learning_rate("G-fixed/random", 600) == 0.05
    assert _learning_rate("G-decay/random", 1) == 0.05
    assert _learning_rate("G-decay/random", 100) == 0.05
    assert _learning_rate("G-decay/random", 101) == 0.01
    assert _learning_rate("G-decay/random", 300) == 0.01
    assert _learning_rate("G-decay/random", 301) == 0.002
    assert _learning_rate("G-decay/random", 600) == 0.002


def test_curriculum_batches_are_balanced_singletons_then_both_inverse_orders() -> None:
    singleton = _singleton_batch(2237)
    inverse = _ordered_inverse_batch(2237)
    assert singleton.token_ids.shape == (16, 16)
    assert inverse.token_ids.shape == (16, 16)
    assert torch.all((singleton.token_ids != 0).sum(dim=1) == 1)
    assert torch.all((inverse.token_ids != 0).sum(dim=1) == 2)
    assert torch.allclose(
        inverse.exact_coordinates.sum(dim=1),
        torch.zeros_like(inverse.exact_coordinates[:, 0]),
    )
    for pair in range(8):
        first = inverse.token_ids[2 * pair, 1:3]
        second = inverse.token_ids[2 * pair + 1, 1:3]
        assert torch.equal(first, second.flip(0))


def test_training_batch_switches_at_the_frozen_curriculum_boundaries() -> None:
    config = quality_config()
    singleton = _training_batch(
        "G-decay/curriculum",
        seed=2237,
        update=100,
        stage="development",
        config=config,
    )
    inverse = _training_batch(
        "G-decay/curriculum",
        seed=2237,
        update=101,
        stage="development",
        config=config,
    )
    random = _training_batch(
        "G-decay/curriculum",
        seed=2237,
        update=201,
        stage="development",
        config=config,
    )
    assert torch.all((singleton.token_ids != 0).sum(dim=1) == 1)
    assert torch.all((inverse.token_ids != 0).sum(dim=1) == 2)
    assert int((random.token_ids != 0).sum(dim=1).min()) >= 2
    assert int((random.token_ids != 0).sum(dim=1).max()) <= 6


def test_optimizer_recipe_changes_only_second_moment_structure() -> None:
    global_controller = TokenCoordinateController()
    block_controller = TokenCoordinateController()
    global_optimizer = _build_optimizer("G-decay/random", global_controller)
    block_optimizer = _build_optimizer("B-decay/random", block_controller)
    assert isinstance(global_optimizer, ScalarSecondMomentAdamW)
    assert not isinstance(global_optimizer, BlockScalarSecondMomentAdamW)
    assert isinstance(block_optimizer, BlockScalarSecondMomentAdamW)


def _recipe_result(error: float) -> dict[str, object]:
    return {
        "evaluation": {
            str(length): {
                "mean_relative_frobenius_error": error,
                "p95_relative_frobenius_error": error,
                "maximum_relative_frobenius_error": error,
            }
            for length in (64, 256, 1024)
        }
    }


def test_selection_uses_all_seeds_and_frozen_least_intervention_order() -> None:
    reports = []
    for seed in DEVELOPMENT_SEEDS:
        reports.append(
            {
                "seed": seed,
                "recipes": {
                    "G-fixed/random": _recipe_result(0.01),
                    "G-decay/random": _recipe_result(0.04),
                    "B-decay/random": _recipe_result(0.03),
                    "G-decay/curriculum": _recipe_result(0.02),
                    "B-decay/curriculum": _recipe_result(0.01),
                },
            }
        )
    selection = _select_recipe(reports)
    assert selection["selected_recipe"] == "G-decay/random"
    assert selection["candidates"]["G-fixed/random"]["selectable"] is False
    reports[0]["recipes"]["G-decay/random"] = _recipe_result(0.06)
    selection = _select_recipe(reports)
    assert selection["selected_recipe"] == "B-decay/random"
