from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hybrid_memory_v1_4.natural_text_screen import (
    DATA_SEED,
    EVAL_UPDATES,
    MODEL_NAMES,
    SNAPSHOT_SHA256,
    TRAIN_UPDATES,
    _batch,
    _offset,
)


def test_g11_protocol_constants_are_frozen() -> None:
    assert MODEL_NAMES == (
        "hybrid_v1_4_5",
        "transformers_mamba2",
        "transformers_olmo_hybrid",
    )
    assert DATA_SEED == 1817
    assert TRAIN_UPDATES == 2000
    assert EVAL_UPDATES == (0, 500, 1000, 2000)
    assert SNAPSHOT_SHA256 == (
        "ebe7c7c948f3e59781097ffa64e214da15364d61b38622d33dd076d40471adc6"
    )


def test_g11_offsets_and_batches_are_deterministic() -> None:
    assert _offset("train", 3, 10_000) == _offset("train", 3, 10_000)
    assert _offset("train", 3, 10_000) != _offset("validation", 3, 10_000)
    stream = torch.arange(1024, dtype=torch.long) % 256
    left = _batch(
        stream,
        namespace="test",
        batch_index=2,
        batch_size=3,
        device=torch.device("cpu"),
    )
    right = _batch(
        stream,
        namespace="test",
        batch_index=2,
        batch_size=3,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(left[0], right[0])
    torch.testing.assert_close(left[1], right[1])
    torch.testing.assert_close(left[0][:, 1:], left[1][:, :-1])


def test_retained_snapshot_matches_g11_hash() -> None:
    snapshot = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "tinystories_snapshot_rows_0_1999_train_0_255_validation_2026-08-25.json"
    )
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == SNAPSHOT_SHA256
