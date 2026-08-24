from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hybrid_memory_v1_4.combined_validation import (
    COMBINED_SCHEDULE,
    schedule_label_counts,
)
from hybrid_memory_v1_4.successor_screen import (
    RETENTION_SAFE_INITIAL,
    RETENTION_SAFE_MINIMUM,
    _retention_safe_config,
)
from hybrid_memory_v1_4.upstream_learning_comparison import _build_model


def test_g9_combined_schedule_and_label_budget_are_frozen() -> None:
    assert [phase.updates for phase in COMBINED_SCHEDULE] == [
        1200,
        1200,
        1400,
        1300,
        600,
    ]
    assert [phase.length for phase in COMBINED_SCHEDULE] == [16, 24, 48, 96, 512]
    assert schedule_label_counts() == (1_292_800, 1_408_000)


def test_retention_safe_successor_changes_only_decay_bounds() -> None:
    config = _retention_safe_config()
    assert config.gated_delta_minimum_retention == RETENTION_SAFE_MINIMUM
    assert config.gated_delta_initial_retention == RETENTION_SAFE_INITIAL
    model = _build_model("hybrid_v1_4_5", torch.device("cpu"))
    mixer = model.blocks[0].mixer
    inputs = torch.randn(2, 7, config.model_dim)
    *_, retention = mixer._controls(inputs)
    assert bool((retention >= RETENTION_SAFE_MINIMUM).all())
    assert bool((retention < 1.0).all())
