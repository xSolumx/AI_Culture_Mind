from __future__ import annotations

import copy

import pytest
import torch

from hybrid_memory_v1_4 import g15br3_logical_component as r3
from hybrid_memory_v1_4.g15b_interleaved_cohort import build_model
from hybrid_memory_v1_4.g15b_interleaved_tasks import (
    ROLE_FILLER,
    generate_interleaved_batch,
)
from hybrid_memory_v1_4.g15br2_collision_erase import overwrite_query_strata


def _batch(task: str = "overwrite"):
    return generate_interleaved_batch(task, 2, 128, 8, 24, 8, seed=4)


def test_component_ownership_is_exclusive_and_resets_every_valid_write() -> None:
    batch = _batch()
    ownership, reset = r3.logical_component_ownership(batch)
    assert ownership.shape == batch.token_ids.shape
    assert reset.shape == (batch.batch_size, 9, batch.length)
    assert int(reset.sum()) == int(batch.write_event_mask.sum())
    matches = batch.write_keys[..., None].eq(batch.live_keys[:, None, :])
    key_index = matches.to(torch.int64).argmax(-1) + 1
    assert torch.equal(ownership.gather(1, batch.write_positions), key_index)
    assert torch.equal(ownership.gather(1, batch.write_positions + 1), key_index)
    # The compact grammar can place the next event marker in the frozen tail.
    assert bool(batch.roles.gather(1, batch.write_positions + 1).ne(ROLE_FILLER).any())


def test_constructed_guard_populates_every_causal_stratum() -> None:
    batch = r3.generate_component_guard_batch(16, 128, seed=31)
    strata = overwrite_query_strata(batch)
    assert [int(strata[name].sum()) for name in r3.STRATA] == [32, 48, 48]
    assert (
        batch.fingerprint()
        == r3.generate_component_guard_batch(16, 128, seed=31).fingerprint()
    )


def test_decomposed_replays_match_monolithic_controls() -> None:
    model = build_model("I", 29, torch.device("cpu")).eval()
    batch = _batch()
    with torch.no_grad():
        learned = r3.logical_component_forward(model, batch, "learned")
        learned_replay = r3.logical_component_forward(
            model, batch, "learned_decomposed_replay"
        )
        erase_free = r3._erase_free_monolithic_forward(model, batch)
        erase_free_replay = r3.logical_component_forward(
            model, batch, "erase_free_no_reset"
        )
    assert torch.allclose(
        learned["logits"], learned_replay["logits"], atol=5e-5, rtol=0.0
    )
    assert torch.allclose(
        erase_free["logits"], erase_free_replay["logits"], atol=5e-5, rtol=0.0
    )
    assert learned_replay["component_capacity"] == {
        "base_state_scalars_per_sequence": 256,
        "logical_components_per_sequence": 9,
        "expanded_state_scalars_per_sequence": 2304,
    }


def test_first_write_reset_is_noop_but_overwrite_reset_changes_state() -> None:
    model = build_model("I", 29, torch.device("cpu")).eval()
    with torch.no_grad():
        mqar = _batch("mqar")
        no_reset = r3.logical_component_forward(model, mqar, "erase_free_no_reset")
        lww = r3.logical_component_forward(model, mqar, "erase_free_lww")
        assert torch.equal(no_reset["logits"], lww["logits"])

        overwrite = _batch("overwrite")
        no_reset = r3.logical_component_forward(model, overwrite, "erase_free_no_reset")
        lww = r3.logical_component_forward(model, overwrite, "erase_free_lww")
        assert not torch.equal(no_reset["logits"], lww["logits"])
        for name in r3.PRESERVED_CONTROL_NAMES:
            assert torch.equal(no_reset["controls"][name], lww["controls"][name])


def _seed_report() -> dict:
    cells = {}
    for task, learned, no_reset, lww in (
        ("mqar", 0.90, 0.90, 0.89),
        ("overwrite", 0.50, 0.50, 0.65),
        ("overwrite_guard", 0.50, 0.50, 0.65),
        ("selective", 0.90, 0.90, 0.89),
        ("needle", 1.00, 1.00, 1.00),
    ):
        cell = {
            "interventions": {
                "learned": {"query_accuracy": learned},
                "learned_decomposed_replay": {"query_accuracy": learned},
                "erase_free_no_reset": {"query_accuracy": no_reset},
                "erase_free_lww": {"query_accuracy": lww},
            },
        }
        if task != "overwrite_guard":
            cell.update(
                {
                    "baseline_query_accuracy_absolute_residual": 0.0,
                    "baseline_exact_episode_accuracy_absolute_residual": 0.0,
                    "baseline_bits_per_query_absolute_residual": 0.0,
                    "no_erase_query_accuracy_absolute_residual": 0.0,
                }
            )
        if task in ("overwrite", "overwrite_guard"):
            cell["query_strata"] = {}
            for stratum in r3.STRATA:
                same = stratum == "after_same_key_overwrite"
                cell["query_strata"][stratum] = {
                    "query_decisions": (
                        16
                        if task == "overwrite_guard"
                        else (16 if stratum != "after_unrelated_overwrite_only" else 0)
                    ),
                    "accuracy": {
                        "learned": 0.50 if same else 0.80,
                        "learned_decomposed_replay": 0.50 if same else 0.80,
                        "erase_free_no_reset": 0.50 if same else 0.80,
                        "erase_free_lww": 0.65 if same else 0.80,
                    },
                }
                if task == "overwrite" and stratum == "after_unrelated_overwrite_only":
                    cell["query_strata"][stratum]["accuracy"] = {
                        name: None for name in r3.INTERVENTIONS
                    }
        cells[f"{task}:L128"] = cell
    return {
        "observability_witness": {"passed": True},
        "evaluation": {
            "cells": cells,
            "runtime_integrity": {
                "model_forward_maximum_absolute_logit_residual": 0.0,
                "learned_decomposition_maximum_absolute_logit_residual": 1e-6,
                "learned_decomposition_maximum_absolute_state_residual": 1e-7,
                "learned_decomposition_query_predictions_equal": True,
                "erase_free_decomposition_maximum_absolute_logit_residual": 1e-6,
                "erase_free_decomposition_maximum_absolute_state_residual": 1e-7,
                "erase_free_decomposition_query_predictions_equal": True,
                "preserved_controls_bitwise_equal": True,
                "local_decoder_batches_checked": 1,
                "collision_mask_batches_checked": 1,
                "component_assignment_batches_checked": 1,
            },
        },
    }


def test_adjudication_requires_populated_guard_and_both_controls() -> None:
    reports = [_seed_report() for _ in range(3)]
    assert r3._adjudicate(reports)["passed"] is True

    missing = copy.deepcopy(reports)
    for report in missing:
        row = report["evaluation"]["cells"]["overwrite_guard:L128"]
        row["query_strata"]["after_unrelated_overwrite_only"]["query_decisions"] = 0
        row["query_strata"]["after_unrelated_overwrite_only"]["accuracy"] = {
            name: None for name in r3.INTERVENTIONS
        }
    assert r3._adjudicate(missing)["passed"] is False


def test_frozen_r2_hash_and_contract() -> None:
    assert r3._sha256(r3.R2_ARTIFACT) == r3.EXPECTED_R2_SHA256
    report, actual = r3._validate_r2(r3.R2_ARTIFACT)
    assert actual == r3.EXPECTED_R2_SHA256
    assert report["adjudication"]["passed"] is False


def test_cpu_checkpoint_evaluation_smoke_records_both_replay_proofs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(r3, "EVALUATION_LENGTHS", (128,))
    model = build_model("I", 29, torch.device("cpu")).eval()
    report = r3.evaluate_checkpoint(model, seed=29, decisions=16, batch_cap=2)
    integrity = report["runtime_integrity"]
    assert integrity["model_forward_maximum_absolute_logit_residual"] == 0.0
    assert integrity["learned_decomposition_maximum_absolute_logit_residual"] <= 5e-4
    assert integrity["learned_decomposition_maximum_absolute_state_residual"] <= 2e-6
    assert integrity["erase_free_decomposition_maximum_absolute_logit_residual"] <= 5e-4
    assert integrity["erase_free_decomposition_maximum_absolute_state_residual"] <= 2e-6
    assert integrity["learned_decomposition_query_predictions_equal"] is True
    assert integrity["erase_free_decomposition_query_predictions_equal"] is True
    assert integrity["preserved_controls_bitwise_equal"] is True
    guard = report["cells"]["overwrite_guard:L128"]["query_strata"]
    assert all(guard[name]["query_decisions"] > 0 for name in r3.STRATA)


def test_unknown_component_intervention_fails_closed() -> None:
    model = build_model("I", 29, torch.device("cpu"))
    with pytest.raises(ValueError, match="unknown G15B-R3 intervention"):
        r3.logical_component_forward(model, _batch(), "collision")  # type: ignore[arg-type]
