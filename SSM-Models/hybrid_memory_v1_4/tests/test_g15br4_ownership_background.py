from __future__ import annotations

import copy

import pytest
import torch

from hybrid_memory_v1_4 import g15br3_logical_component as r3
from hybrid_memory_v1_4 import g15br4_ownership_background as r4
from hybrid_memory_v1_4.g15b_interleaved_cohort import build_model
from hybrid_memory_v1_4.g15b_interleaved_tasks import generate_interleaved_batch


def _batch(task: str = "overwrite"):
    return generate_interleaved_batch(task, 2, 128, 8, 24, 8, seed=17)


def test_local_query_decoder_matches_batch_positions() -> None:
    batch = _batch()
    expected = torch.zeros_like(batch.token_ids, dtype=torch.bool)
    expected.scatter_(1, batch.query_positions, True)
    assert torch.equal(r4.local_query_position_mask(batch.token_ids), expected)


def test_value_only_and_value_tail_ownership_share_resets() -> None:
    batch = _batch()
    v_owner, v_reset = r4.factor_ownership(batch, "v")
    vt_owner, vt_reset = r4.factor_ownership(batch, "vt")
    assert torch.equal(v_reset, vt_reset)
    rows = torch.arange(batch.batch_size)[:, None].expand_as(batch.write_positions)
    tails = batch.write_positions + 1
    valid = tails < batch.length
    assert not bool(v_owner[rows[valid], tails[valid]].any())
    assert bool(vt_owner[rows[valid], tails[valid]].gt(0).all())
    with pytest.raises(ValueError, match="unknown ownership mode"):
        r4.factor_ownership(batch, "filler")  # type: ignore[arg-type]


def test_factorial_reference_arms_replay_r3_and_background_identity() -> None:
    model = build_model("I", 29, torch.device("cpu")).eval()
    batch = _batch()
    with torch.no_grad():
        result = r4.factorial_forwards(model, batch)
        learned = r3.logical_component_forward(model, batch, "learned")
        no_reset = r3.logical_component_forward(model, batch, "erase_free_no_reset")
        lww = r3.logical_component_forward(model, batch, "erase_free_lww")
    assert torch.equal(result["logits"]["learned"], learned["logits"])
    assert torch.equal(
        result["logits"]["erase_free_no_reset_bgplus"], no_reset["logits"]
    )
    assert torch.equal(result["logits"]["vt_lww_bgplus"], lww["logits"])
    for reads in result["reads"].values():
        assert reads["background_relation_residual"] <= 1e-5


def test_first_writes_make_lww_and_no_reset_equal() -> None:
    model = build_model("I", 31, torch.device("cpu")).eval()
    batch = _batch("mqar")
    with torch.no_grad():
        result = r4.factorial_forwards(model, batch)["logits"]
    for arm, control in r4.MATCHING_CONTROL.items():
        assert torch.equal(result[arm].argmax(-1), result[control].argmax(-1))


def test_fp64_contract_passes_recurrent_parallel_and_background_checks() -> None:
    model = build_model("I", 37, torch.device("cpu")).eval()
    with torch.no_grad():
        report = r4.fp64_algebraic_contract(model, _batch())
    assert report["passed"] is True
    assert report["maximum_residual"] <= 1e-10


def _synthetic_seed_report(seed: int, *, value_passes: bool) -> dict:
    cells = {}
    for length in r4.EVALUATION_LENGTHS:
        for task in ("mqar", "overwrite", "overwrite_guard", "selective", "needle"):
            learned = 0.50 if task == "overwrite" else 1.0
            control = (
                0.40
                if task == "overwrite"
                else (0.80 if task == "overwrite_guard" else learned)
            )
            interventions = {}
            for name in r4.INTERVENTIONS:
                if name == "learned":
                    accuracy = learned
                elif name in (
                    "erase_free_no_reset_bgplus",
                    "v_no_reset_bgminus",
                    "vt_no_reset_bgminus",
                ):
                    accuracy = control
                elif name in r4.VALUE_ONLY_ARMS and not value_passes:
                    accuracy = learned
                else:
                    accuracy = 0.65 if task == "overwrite" else 1.0
                interventions[name] = {
                    "query_accuracy": accuracy,
                    "exact_episode_accuracy": accuracy,
                    "bits_per_query": 0.1,
                }
            cell = {
                "interventions": interventions,
                "r3_reference_replay": {
                    name: {
                        "query_accuracy": 0.0,
                        "exact_episode_accuracy": 0.0,
                        "bits_per_query": 0.0,
                    }
                    for name in ("learned", "erase_free_no_reset", "erase_free_lww")
                },
            }
            if task in ("overwrite", "overwrite_guard"):
                cell["query_strata"] = {}
                for stratum in r4.STRATA:
                    stratum_accuracy = {
                        name: values["query_accuracy"]
                        for name, values in interventions.items()
                    }
                    if (
                        task == "overwrite"
                        and stratum == "after_unrelated_overwrite_only"
                    ):
                        stratum_accuracy = {name: None for name in r4.INTERVENTIONS}
                    cell["query_strata"][stratum] = {
                        "query_decisions": (
                            0
                            if task == "overwrite"
                            and stratum == "after_unrelated_overwrite_only"
                            else 16
                        ),
                        "accuracy": stratum_accuracy,
                    }
            cells[f"{task}:L{length}"] = cell
    integrity = r4._new_integrity()
    integrity.update(
        {
            "local_write_batches_checked": 1,
            "local_query_batches_checked": 1,
            "ownership_batches_checked": 1,
            "ordinary_model_forward_maximum_absolute_logit_residual": 0.0,
            "fp64_algebraic_contract": {"passed": True},
            "finite_logits": True,
            "preserved_controls_bitwise_equal": True,
        }
    )
    return {
        "seed": seed,
        "observability_witness": {"passed": True},
        "evaluation": {"cells": cells, "runtime_integrity": integrity},
    }


def test_adjudication_requires_value_only_arm_for_training() -> None:
    tail_only = [
        _synthetic_seed_report(seed, value_passes=False) for seed in r4.QUALITY_SEEDS
    ]
    report = r4._adjudicate(tail_only)
    assert report["passed"] is False
    assert report["passed_tail_arms"]
    assert "ambiguous" in report["decision"]

    value = copy.deepcopy(
        [_synthetic_seed_report(seed, value_passes=True) for seed in r4.QUALITY_SEEDS]
    )
    report = r4._adjudicate(value)
    assert report["passed"] is True
    assert report["selected_training_law"] == "value-only slots with shared background"


def test_frozen_r3_parent_hash_and_contract() -> None:
    report, actual = r4._validate_r3(r4.R3_ARTIFACT)
    assert actual == r4.EXPECTED_R3_SHA256
    assert report["adjudication"]["passed"] is False
    assert report["adjudication"]["post_same_key_improved"] is True
