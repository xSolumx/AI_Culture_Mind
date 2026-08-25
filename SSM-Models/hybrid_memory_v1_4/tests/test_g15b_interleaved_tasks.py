from __future__ import annotations

import pytest
import torch

from hybrid_memory_v1_4.g15b_interleaved_tasks import (
    ITEM_TOKEN,
    PAYLOAD_START,
    SELECT_TOKEN,
    generate_interleaved_batch,
    oracle_direct_read_accuracy,
)
from hybrid_memory_v1_4.spin_dirac_memory import SpinDiracConfig, SpinDiracMemory


@pytest.mark.parametrize("task", ("mqar", "overwrite", "selective", "needle"))
def test_g15b_batches_are_deterministic_fresh_and_causally_valid(task: str) -> None:
    kwargs = {
        "task": task,
        "batch_size": 4,
        "length": 128,
        "live_keys": 4,
        "max_writes": 8,
        "queries": 4,
        "seed": 2309,
        "needle_distance": 64 if task == "needle" else None,
    }
    first = generate_interleaved_batch(**kwargs)
    second = generate_interleaved_batch(**kwargs)
    fresh = generate_interleaved_batch(**{**kwargs, "seed": 2311})

    assert first.fingerprint() == second.fingerprint()
    assert first.fingerprint() != fresh.fingerprint()
    assert first.token_ids.shape == (4, 128)
    assert bool((first.query_keys != first.targets).all())
    assert int(first.token_ids.min()) >= 1
    assert int(first.targets.min()) >= PAYLOAD_START


def test_g15b_overwrite_contains_changed_pre_and_post_overwrite_queries() -> None:
    batch = generate_interleaved_batch("overwrite", 8, 128, 4, 8, 4, seed=2317)
    for row in range(batch.batch_size):
        found = False
        for left in range(batch.queries):
            for right in range(left + 1, batch.queries):
                if (
                    batch.query_keys[row, left] == batch.query_keys[row, right]
                    and batch.targets[row, left] != batch.targets[row, right]
                ):
                    found = True
        assert found
    assert bool(batch.overwrite_mask.any(dim=1).all())


def test_g15b_selective_items_are_not_valid_write_events() -> None:
    batch = generate_interleaved_batch("selective", 6, 128, 4, 8, 4, seed=2321)
    item_markers = batch.token_ids == ITEM_TOKEN
    select_markers = batch.token_ids == SELECT_TOKEN
    assert bool(item_markers.any())
    assert bool(select_markers.any())
    item_value_positions = item_markers.nonzero() + torch.tensor((0, 2))
    selected_value_positions = select_markers.nonzero() + torch.tensor((0, 2))
    assert not bool(
        batch.write_event_mask[
            item_value_positions[:, 0], item_value_positions[:, 1]
        ].any()
    )
    assert bool(
        batch.write_event_mask[
            selected_value_positions[:, 0], selected_value_positions[:, 1]
        ].all()
    )


def test_g15b_needle_distance_is_exact_and_query_position_varies() -> None:
    batch = generate_interleaved_batch(
        "needle",
        16,
        512,
        8,
        24,
        8,
        seed=2333,
        needle_distance=448,
    )
    assert bool(batch.needle_distances.eq(448).all())
    assert batch.query_positions.unique().numel() > 1


@pytest.mark.parametrize(
    ("transport", "readout"),
    (
        ("identity", "identity"),
        ("commuting_so2", "clifford"),
        ("spin8", "clifford"),
    ),
)
@pytest.mark.parametrize("task", ("mqar", "overwrite", "selective", "needle"))
def test_g15b_actual_memory_has_an_exact_oracle_control_ceiling(
    task: str, transport: str, readout: str
) -> None:
    batch = generate_interleaved_batch(
        task,
        3,
        96,
        4,
        8,
        4,
        seed=2341,
        needle_distance=64 if task == "needle" else None,
    )
    memory = SpinDiracMemory(
        SpinDiracConfig(
            model_dim=64,
            heads=4,
            transport_mode=transport,
            readout_mode=readout,
        )
    ).double()
    accuracy, residual = oracle_direct_read_accuracy(batch, memory)
    assert accuracy == 1.0
    assert residual < 1e-10


def test_g15b_rejects_more_than_eight_live_keys() -> None:
    with pytest.raises(ValueError, match="at most eight"):
        generate_interleaved_batch("mqar", 2, 64, 9, 9, 2, seed=1)


def test_g15b_device_transfer_does_not_repeat_causal_replay() -> None:
    batch = generate_interleaved_batch("mqar", 2, 64, 2, 4, 4, seed=7)
    transferred = batch.to("cpu")
    assert transferred._skip_validation
    assert transferred.fingerprint() == batch.fingerprint()
