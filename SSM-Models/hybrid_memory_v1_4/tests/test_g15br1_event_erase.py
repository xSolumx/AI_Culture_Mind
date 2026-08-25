from __future__ import annotations

import pytest
import torch

from hybrid_memory_v1_4.g15b_interleaved_cohort import build_model
from hybrid_memory_v1_4.g15b_interleaved_tasks import generate_interleaved_batch
from hybrid_memory_v1_4.g15br1_event_erase import (
    INTERVENTIONS,
    event_erase_forward,
)


def _batch():
    return generate_interleaved_batch("overwrite", 2, 64, 2, 4, 2, seed=260827)


def test_event_erase_changes_only_erase_and_preserves_learned_write_tail() -> None:
    model = build_model("I", 23, torch.device("cpu")).eval()
    batch = _batch()
    with torch.no_grad():
        learned = event_erase_forward(model, batch, "learned")
        soft = event_erase_forward(model, batch, "soft_event_erase")
        exact = event_erase_forward(model, batch, "exact_event_erase")

    preserved = (
        "query_vector",
        "key_vector",
        "value_positive",
        "write_strength",
        "retention",
        "transport_coordinates",
    )
    for result in (soft, exact):
        for name in preserved:
            assert torch.equal(learned["controls"][name], result["controls"][name])

    soft_erase = soft["controls"]["erase_strength"]
    learned_write = learned["controls"]["write_strength"]
    event = batch.write_event_mask[..., None, None].expand_as(soft_erase)
    assert bool(soft_erase[~event].eq(0).all())
    assert torch.equal(
        soft_erase[event],
        learned_write[event],
    )
    assert torch.equal(
        exact["controls"]["erase_strength"].gt(0).any(dim=(-1, -2)),
        batch.write_event_mask,
    )
    after = batch.write_positions + 1
    rows = torch.arange(batch.batch_size)[:, None].expand_as(after)
    assert torch.equal(
        learned["controls"]["write_strength"][rows, after],
        exact["controls"]["write_strength"][rows, after],
    )


def test_unknown_event_erase_intervention_fails_closed() -> None:
    model = build_model("I", 23, torch.device("cpu"))
    with pytest.raises(ValueError, match="unknown G15B-R1 intervention"):
        event_erase_forward(model, _batch(), "tied_delta")  # type: ignore[arg-type]


def test_r1_intervention_set_is_frozen() -> None:
    assert INTERVENTIONS == (
        "learned",
        "soft_event_erase",
        "exact_event_erase",
    )
