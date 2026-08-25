from __future__ import annotations

from dataclasses import replace

import torch

from hybrid_memory_v1_4.natural_text_frontier import (
    DEVELOPMENT_SEEDS,
    TARGET_PARAMETER_COUNT,
    _batch,
    _parameter_matched_config,
)
from hybrid_memory_v1_4.tokenization import RawByteTokenizer


def test_weighted_batch_keeps_target_byte_accounting() -> None:
    stream = RawByteTokenizer().encode("a" * 2048)
    inputs, targets, byte_lengths = _batch(
        stream,
        namespace="test",
        batch_index=0,
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert inputs.shape == targets.shape == byte_lengths.shape == (2, 256)
    assert int(byte_lengths.sum()) == targets.numel()


def test_parameter_matching_is_deterministic_and_close() -> None:
    config, report = _parameter_matched_config(512)
    second_config, second_report = _parameter_matched_config(512)
    assert config == second_config
    assert report == second_report
    assert config.vocab_size == 512
    assert report["target"] == TARGET_PARAMETER_COUNT
    assert report["relative_difference"] < 0.05
    assert config == replace(
        config,
        model_dim=report["model_dim"],
        expansion=report["expansion"],
    )


def test_development_seeds_are_frozen_and_distinct() -> None:
    assert DEVELOPMENT_SEEDS == (1823, 1829)
