from __future__ import annotations

import pytest

from hybrid_memory_v1_4.compute_matched_frontier import (
    CALIBRATION_EXPANSIONS,
    CALIBRATION_REPEATS,
    CALIBRATION_WARMUPS,
    CALIBRATION_WIDTHS,
    _config,
)


def test_calibration_grid_and_budget_are_frozen() -> None:
    assert CALIBRATION_WIDTHS == tuple(range(24, 97, 4))
    assert CALIBRATION_EXPANSIONS == tuple(range(1, 7))
    assert CALIBRATION_WARMUPS == 5
    assert CALIBRATION_REPEATS == 15


def test_compute_candidate_configuration_tracks_width() -> None:
    config = _config(512, 48, 5)
    assert config.vocab_size == 512
    assert config.model_dim == 48
    assert config.expansion == 5
    assert config.gated_delta_key_dim == 24
    assert config.gated_delta_value_dim == 12


def test_invalid_rope_width_is_rejected() -> None:
    with pytest.raises(ValueError, match="head_dim must be even"):
        _config(512, 28, 2)
