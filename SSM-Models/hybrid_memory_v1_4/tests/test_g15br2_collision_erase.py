from __future__ import annotations

import copy

import pytest
import torch

from hybrid_memory_v1_4 import g15br2_collision_erase as r2
from hybrid_memory_v1_4.g15b_interleaved_cohort import build_model
from hybrid_memory_v1_4.g15b_interleaved_tasks import generate_interleaved_batch


def _batch(task: str = "overwrite"):
    return generate_interleaved_batch(task, 2, 128, 8, 24, 8, seed=260828)


def test_collision_erase_changes_only_erase_and_preserves_write_tail() -> None:
    model = build_model("I", 29, torch.device("cpu")).eval()
    batch = _batch()
    with torch.no_grad():
        learned = r2.collision_erase_forward(model, batch, "learned")
        soft = r2.collision_erase_forward(model, batch, "soft_collision_erase")
        exact = r2.collision_erase_forward(model, batch, "exact_collision_erase")

    for result in (soft, exact):
        for name in r2.PRESERVED_CONTROL_NAMES:
            assert torch.equal(learned["controls"][name], result["controls"][name])
    collision = batch.erase_event_mask[..., None, None].expand_as(
        soft["controls"]["erase_strength"]
    )
    assert bool(soft["controls"]["erase_strength"][~collision].eq(0).all())
    assert torch.equal(
        soft["controls"]["erase_strength"][collision],
        learned["controls"]["write_strength"][collision],
    )
    assert torch.equal(
        exact["controls"]["erase_strength"].gt(0).any(dim=(-1, -2)),
        batch.erase_event_mask,
    )
    rows = torch.arange(batch.batch_size)[:, None].expand_as(batch.write_positions)
    after = batch.write_positions + 1
    valid_after = after < batch.length
    assert torch.equal(
        learned["controls"]["write_strength"][rows[valid_after], after[valid_after]],
        exact["controls"]["write_strength"][rows[valid_after], after[valid_after]],
    )


def test_overwrite_query_strata_are_mutually_exclusive_and_complete() -> None:
    batch = _batch()
    strata = r2.overwrite_query_strata(batch)
    assert tuple(strata) == r2.STRATA
    assigned = sum(mask.to(torch.int8) for mask in strata.values())
    assert torch.equal(assigned, torch.ones_like(assigned))
    assert int(strata["after_same_key_overwrite"].sum()) > 0


def _seed_report() -> dict:
    cells = {}
    for task, learned, soft, exact in (
        ("mqar", 0.90, 0.885, 0.885),
        ("overwrite", 0.50, 0.625, 0.625),
        ("selective", 0.90, 0.885, 0.885),
        ("needle", 1.00, 1.00, 1.00),
    ):
        cell = {
            "interventions": {
                "learned": {"query_accuracy": learned},
                "soft_collision_erase": {"query_accuracy": soft},
                "exact_collision_erase": {"query_accuracy": exact},
            },
            "baseline_query_accuracy_absolute_residual": 0.0,
            "baseline_exact_episode_accuracy_absolute_residual": 0.0,
            "baseline_bits_per_query_absolute_residual": 0.0,
        }
        if task == "overwrite":
            cell["query_strata"] = {
                stratum: {
                    "accuracy": {
                        "learned": 0.50,
                        "soft_collision_erase": (
                            0.625 if stratum == "after_same_key_overwrite" else 0.50
                        ),
                        "exact_collision_erase": (
                            0.625 if stratum == "after_same_key_overwrite" else 0.50
                        ),
                    }
                }
                for stratum in r2.STRATA
            }
        cells[f"{task}:L128"] = cell
    return {
        "observability_witness": {"passed": True},
        "evaluation": {
            "cells": cells,
            "runtime_integrity": {
                "model_forward_maximum_absolute_logit_residual": 0.0,
                "preserved_controls_bitwise_equal": True,
                "local_decoder_batches_checked": 1,
                "collision_mask_batches_checked": 1,
            },
        },
    }


def test_adjudication_enforces_strata_and_integrity() -> None:
    reports = [_seed_report() for _ in range(3)]
    accepted = r2._adjudicate(reports)
    assert accepted["passed"] is True
    assert accepted["selected_mode"] == "soft_collision_erase"

    damaged = copy.deepcopy(reports)
    for report in damaged:
        row = report["evaluation"]["cells"]["overwrite:L128"]["query_strata"]
        row["after_unrelated_overwrite_only"]["accuracy"]["soft_collision_erase"] = 0.40
        row["after_unrelated_overwrite_only"]["accuracy"]["exact_collision_erase"] = (
            0.40
        )
    rejected = r2._adjudicate(damaged)
    assert rejected["passed"] is False
    assert rejected["decision"].endswith("separate erase address")


def test_frozen_r1_hash_and_contract() -> None:
    assert r2._sha256(r2.R1_ARTIFACT) == r2.EXPECTED_R1_SHA256
    report, actual = r2._validate_r1(r2.R1_ARTIFACT)
    assert actual == r2.EXPECTED_R1_SHA256
    assert report["adjudication"]["passed"] is False


def test_cpu_checkpoint_evaluation_smoke_records_strata_and_integrity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(r2, "EVALUATION_LENGTHS", (128,))
    model = build_model("I", 29, torch.device("cpu")).eval()
    report = r2.evaluate_checkpoint(model, seed=29, decisions=16, batch_cap=2)
    integrity = report["runtime_integrity"]
    assert integrity["model_forward_maximum_absolute_logit_residual"] == 0.0
    assert integrity["preserved_controls_bitwise_equal"] is True
    assert integrity["collision_mask_batches_checked"] > 0
    strata = report["cells"]["overwrite:L128"]["query_strata"]
    assert tuple(strata) == r2.STRATA
    assert sum(row["query_decisions"] for row in strata.values()) == 16


def test_unknown_collision_intervention_fails_closed() -> None:
    model = build_model("I", 29, torch.device("cpu"))
    with pytest.raises(ValueError, match="unknown G15B-R2 intervention"):
        r2.collision_erase_forward(model, _batch(), "event")  # type: ignore[arg-type]
