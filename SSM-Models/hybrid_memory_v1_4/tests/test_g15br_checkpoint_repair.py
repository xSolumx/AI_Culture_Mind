from __future__ import annotations

import pytest
import torch

from hybrid_memory_v1_4.g15b_interleaved_cohort import (
    build_model,
    complete_control_forward,
)
from hybrid_memory_v1_4.g15b_interleaved_tasks import generate_interleaved_batch
from hybrid_memory_v1_4.g15br_checkpoint_repair import (
    INTERVENTIONS,
    local_write_event_mask,
    repair_control_forward,
    temporal_observability_witness,
)


def _batch(task: str = "overwrite"):
    return generate_interleaved_batch(
        task,
        2,
        64,
        2,
        4,
        2,
        seed=260826,
        needle_distance=32 if task == "needle" else None,
    )


@pytest.mark.parametrize("task", ("mqar", "overwrite", "selective", "needle"))
def test_repaired_write_target_is_exactly_local(task: str) -> None:
    batch = _batch(task)
    assert torch.equal(local_write_event_mask(batch.token_ids), batch.write_event_mask)


def test_rejected_collision_target_has_constructive_local_alias() -> None:
    model = build_model("I", 23, torch.device("cpu")).eval()
    witness = temporal_observability_witness(model)
    assert witness["local_windows_equal"]
    assert witness["collision_labels"] == [True, False]
    assert witness["labels_differ"]
    assert witness["maximum_control_residual"] <= 5e-7
    assert witness["passed"]


def test_learned_path_exactly_replays_frozen_g15b_forward() -> None:
    model = build_model("I", 23, torch.device("cpu")).eval()
    batch = _batch("selective")
    with torch.no_grad():
        frozen = complete_control_forward(model, batch, "learned")
        repaired = repair_control_forward(model, batch, "learned")
    assert torch.equal(frozen["logits"], repaired["logits"])


def test_repair_interventions_preserve_gauge_and_change_only_edit_timing() -> None:
    model = build_model("I", 23, torch.device("cpu")).eval()
    batch = _batch()
    with torch.no_grad():
        learned = repair_control_forward(model, batch, "learned")
        soft = repair_control_forward(model, batch, "soft_delta")
        collision = repair_control_forward(model, batch, "exact_collision_timing")
        exact = repair_control_forward(model, batch, "exact_delta_timing")

    preserved = (
        "query_vector",
        "key_vector",
        "value_positive",
        "retention",
        "transport_coordinates",
    )
    for result in (soft, collision, exact):
        for name in preserved:
            assert torch.equal(
                learned["controls"][name],
                result["controls"][name],
            )

    assert torch.equal(
        soft["controls"]["erase_strength"],
        soft["controls"]["write_strength"],
    )
    collision_write = collision["controls"]["write_strength"].gt(0).any(dim=(-1, -2))
    collision_erase = collision["controls"]["erase_strength"].gt(0).any(dim=(-1, -2))
    exact_write = exact["controls"]["write_strength"].gt(0).any(dim=(-1, -2))
    exact_erase = exact["controls"]["erase_strength"].gt(0).any(dim=(-1, -2))
    assert torch.equal(collision_write, batch.write_event_mask)
    assert torch.equal(collision_erase, batch.erase_event_mask)
    assert torch.equal(exact_write, batch.write_event_mask)
    assert torch.equal(exact_erase, batch.write_event_mask)


def test_unknown_repair_intervention_fails_closed() -> None:
    model = build_model("I", 23, torch.device("cpu"))
    with pytest.raises(ValueError, match="unknown G15B-R0 intervention"):
        repair_control_forward(model, _batch(), "oracle_basis")  # type: ignore[arg-type]


def test_intervention_set_is_frozen() -> None:
    assert INTERVENTIONS == (
        "learned",
        "soft_delta",
        "exact_collision_timing",
        "exact_delta_timing",
    )
