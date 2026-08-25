from __future__ import annotations

import copy

import pytest
import torch

from hybrid_memory_v1_4 import g15br1_event_erase as r1
from hybrid_memory_v1_4.g15b_interleaved_cohort import build_model
from hybrid_memory_v1_4.g15b_interleaved_tasks import generate_interleaved_batch
from hybrid_memory_v1_4.g15br1_event_erase import (
    EXPECTED_G15B_SHA256,
    EXPECTED_R0_SHA256,
    INTERVENTIONS,
    _adjudicate,
    _sha256,
    _validate_quality_artifact,
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


def _adjudication_seed_report() -> dict:
    cells = {}
    for task, learned, soft, exact in (
        ("mqar", 0.90, 0.885, 0.885),
        ("overwrite", 0.50, 0.625, 0.625),
        ("selective", 0.90, 0.885, 0.885),
        ("needle", 1.00, 1.00, 1.00),
    ):
        cells[f"{task}:L128"] = {
            "interventions": {
                "learned": {"query_accuracy": learned},
                "soft_event_erase": {"query_accuracy": soft},
                "exact_event_erase": {"query_accuracy": exact},
            },
            "baseline_query_accuracy_absolute_residual": 0.0,
            "baseline_exact_episode_accuracy_absolute_residual": 0.0,
            "baseline_bits_per_query_absolute_residual": 0.0,
        }
    return {
        "observability_witness": {"passed": True},
        "evaluation": {
            "cells": cells,
            "prototype_cross": {
                "mean_absolute_off_diagonal_cosine": 0.25,
                "maximum_absolute_off_diagonal_cosine": 0.75,
            },
            "runtime_integrity": {
                "model_forward_maximum_absolute_logit_residual": 0.0,
                "preserved_controls_bitwise_equal": True,
                "local_decoder_batches_checked": 1,
            },
        },
    }


def test_adjudication_enforces_integrity_and_soft_tie_break() -> None:
    reports = [_adjudication_seed_report() for _ in range(3)]
    accepted = _adjudicate(reports)
    assert accepted["passed"] is True
    assert accepted["selected_mode"] == "soft_event_erase"

    broken = copy.deepcopy(reports)
    broken[0]["evaluation"]["runtime_integrity"]["preserved_controls_bitwise_equal"] = (
        False
    )
    rejected = _adjudicate(broken)
    assert rejected["runtime_integrity_passed"] is False
    assert rejected["passed"] is False


def test_frozen_artifact_hashes_and_quality_contract() -> None:
    assert _sha256(r1.PARENT_ARTIFACT) == EXPECTED_G15B_SHA256
    assert _sha256(r1.R0_ARTIFACT) == EXPECTED_R0_SHA256
    valid = {
        "mode": "quality",
        "evidentiary": True,
        "git_status_at_start": [],
        "protocol": {"seeds": list(r1.QUALITY_SEEDS)},
        "environment": {"device": "cuda", "compute_capability": [7, 5]},
    }
    _validate_quality_artifact(valid, name="test")
    for mutation in (
        {"mode": "smoke"},
        {"git_status_at_start": ["M file"]},
        {"protocol": {"seeds": [2309]}},
        {"environment": {"device": "cuda", "compute_capability": [8, 0]}},
    ):
        invalid = copy.deepcopy(valid)
        invalid.update(mutation)
        with pytest.raises(ValueError):
            _validate_quality_artifact(invalid, name="test")


def test_cpu_checkpoint_evaluation_smoke_records_runtime_integrity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(r1, "EVALUATION_LENGTHS", (128,))
    model = build_model("I", 23, torch.device("cpu")).eval()
    report = r1.evaluate_checkpoint(model, seed=23, decisions=16, batch_cap=2)
    integrity = report["runtime_integrity"]
    assert integrity["model_forward_maximum_absolute_logit_residual"] == 0.0
    assert integrity["preserved_controls_bitwise_equal"] is True
    assert integrity["local_decoder_batches_checked"] > 0
    assert set(report["cells"]) == {
        "mqar:L128",
        "overwrite:L128",
        "selective:L128",
        "needle:L128",
    }
