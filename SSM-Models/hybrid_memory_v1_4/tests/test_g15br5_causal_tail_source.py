from __future__ import annotations

import copy

import torch

from hybrid_memory_v1_4 import g15br3_logical_component as r3
from hybrid_memory_v1_4 import g15br5_causal_tail_source as r5
from hybrid_memory_v1_4.g15b_interleaved_cohort import build_model
from hybrid_memory_v1_4.g15b_interleaved_tasks import generate_interleaved_batch


def _batch(task: str = "overwrite"):
    return generate_interleaved_batch(task, 2, 128, 8, 24, 8, seed=43)


def test_local_completed_write_tail_decoder_matches_audit_positions() -> None:
    batch = _batch()
    expected = torch.zeros_like(batch.token_ids, dtype=torch.bool)
    positions = batch.write_positions + 1
    valid = positions < batch.length
    rows = torch.arange(batch.batch_size)[:, None].expand_as(positions)
    expected[rows[valid], positions[valid]] = True
    assert torch.equal(r5.local_completed_write_tail_mask(batch.token_ids), expected)


def test_source_ownership_is_disjoint_and_uses_common_resets() -> None:
    batch = _batch()
    value_owner, tail_owner, reset = r5.source_ownership(batch)
    assert not bool((value_owner.gt(0) & tail_owner.gt(0)).any())
    assert int(value_owner.gt(0).sum()) == int(batch.write_event_mask.sum())
    assert int(tail_owner.gt(0).sum()) == int(
        (batch.write_positions + 1 < batch.length).sum()
    )
    assert int(reset.sum()) == int(batch.write_event_mask.sum())


def test_source_split_reconstructs_full_and_preserves_reference_forward() -> None:
    model = build_model("I", 47, torch.device("cpu")).eval()
    batch = _batch()
    with torch.no_grad():
        result = r5.source_forwards(model, batch)
        learned = r3.logical_component_forward(model, batch, "learned")
    assert torch.equal(result["logits"]["learned"], learned["logits"])
    assert result["shared_full_transition_controls"] is True
    assert (
        len(
            {
                reads["shared_transition_tensor_ids"]
                for reads in result["reads"].values()
            }
        )
        == 1
    )
    for reads in result["reads"].values():
        assert reads["injection_sum_residual"] <= 5e-6
        assert reads["assignment_residual"] <= 5e-6
        assert reads["background_relation_residual"] <= 1e-5
    for source in ("h", "c", "b"):
        assert result["reads"][(source, False)]["state_residual"] <= 5e-6


def test_first_writes_make_lww_and_no_reset_predictions_equal() -> None:
    model = build_model("I", 53, torch.device("cpu")).eval()
    with torch.no_grad():
        logits = r5.source_forwards(model, _batch("mqar"))["logits"]
    for arm, control in r5.MATCHING_CONTROL.items():
        assert torch.equal(logits[arm].argmax(-1), logits[control].argmax(-1))


def test_convolution_source_and_fp64_contracts_pass() -> None:
    model = build_model("I", 59, torch.device("cpu")).eval()
    batch = _batch()
    with torch.no_grad():
        structure = r5.convolution_structure_contract(model, batch)
        locality = r5.source_locality_witness(model, batch)
        algebra = r5.fp64_algebraic_contract(model, batch)
    assert structure["passed"] is True
    assert structure["maximum_residual"] <= 2e-6
    assert locality["passed"] is True
    assert locality["maximum_invariance_residual"] <= 5e-7
    assert locality["maximum_future_token_source_residual"] <= 5e-7
    assert locality["minimum_nondegenerate_effect"] >= 1e-6
    assert algebra["passed"] is True
    assert algebra["independent_fp64_injections_used"] is True
    assert algebra["maximum_residual"] <= 1e-10


def _synthetic_seed_report(
    seed: int,
    *,
    history_bgplus_passes: bool,
    history_bgminus_passes: bool,
    bias_matches_history: bool = False,
) -> dict:
    cells = {}
    for length in r5.EVALUATION_LENGTHS:
        for task in ("mqar", "overwrite", "overwrite_guard", "selective", "needle"):
            learned = 0.50 if task == "overwrite" else 1.0
            control = (
                0.30
                if task == "overwrite"
                else (0.80 if task == "overwrite_guard" else learned)
            )
            interventions = {}
            for name in r5.INTERVENTIONS:
                if name == "learned":
                    accuracy = learned
                elif name in (
                    "erase_free_no_reset_bgplus",
                    "h_no_reset_bgminus",
                    "c_no_reset_bgminus",
                    "b_no_reset_bgminus",
                ):
                    accuracy = control
                elif (name == "h_lww_bgplus" and history_bgplus_passes) or (
                    name == "h_lww_bgminus" and history_bgminus_passes
                ):
                    accuracy = 0.70 if task == "overwrite" else 1.0
                elif name in r5.BIAS_ARMS:
                    accuracy = (
                        0.68
                        if task == "overwrite" and bias_matches_history
                        else (0.55 if task == "overwrite" else 0.90)
                    )
                else:
                    accuracy = learned
                interventions[name] = {
                    "query_accuracy": accuracy,
                    "exact_episode_accuracy": accuracy,
                    "bits_per_query": 0.1,
                }
            cell = {
                "interventions": interventions,
                "r4_reference_replay": {
                    "learned:query_accuracy": 0.0,
                    "learned:exact_episode_accuracy": 0.0,
                    "learned:bits_per_query": 0.0,
                    "erase_free_no_reset_bgplus:query_accuracy": 0.0,
                    "erase_free_no_reset_bgplus:exact_episode_accuracy": 0.0,
                    "erase_free_no_reset_bgplus:bits_per_query": 0.0,
                },
                "r4_batch_fingerprint_match": True,
            }
            if task in ("overwrite", "overwrite_guard"):
                cell["query_strata"] = {}
                for stratum in r5.STRATA:
                    stratum_accuracy = {
                        name: values["query_accuracy"]
                        for name, values in interventions.items()
                    }
                    if (
                        task == "overwrite"
                        and stratum == "after_unrelated_overwrite_only"
                    ):
                        stratum_accuracy = {name: None for name in r5.INTERVENTIONS}
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
    integrity = r5._new_integrity()
    integrity.update(
        {
            "local_write_batches_checked": 1,
            "local_tail_batches_checked": 1,
            "local_query_batches_checked": 1,
            "source_assignment_batches_checked": 1,
            "ordinary_model_forward_maximum_absolute_logit_residual": 0.0,
            "convolution_structure_contract": {"passed": True},
            "source_locality_witness": {"passed": True},
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


def test_adjudication_requires_background_free_bias_separated_history() -> None:
    bgplus_only = [
        _synthetic_seed_report(
            seed, history_bgplus_passes=True, history_bgminus_passes=False
        )
        for seed in r5.QUALITY_SEEDS
    ]
    report = r5._adjudicate(bgplus_only)
    assert report["passed"] is False
    assert "residual" in report["decision"]

    background_free = copy.deepcopy(
        [
            _synthetic_seed_report(
                seed, history_bgplus_passes=True, history_bgminus_passes=True
            )
            for seed in r5.QUALITY_SEEDS
        ]
    )
    report = r5._adjudicate(background_free)
    assert report["passed"] is True
    assert report["history_background_free_passed"] is True

    bias_confounded = [
        _synthetic_seed_report(
            seed,
            history_bgplus_passes=True,
            history_bgminus_passes=True,
            bias_matches_history=True,
        )
        for seed in r5.QUALITY_SEEDS
    ]
    report = r5._adjudicate(bias_confounded)
    assert report["passed"] is False
    assert report["history_background_free_passed"] is False


def test_runtime_integrity_fails_closed_on_recorded_residuals_and_transition_drift() -> (
    None
):
    reports = [
        _synthetic_seed_report(
            seed, history_bgplus_passes=True, history_bgminus_passes=True
        )
        for seed in r5.QUALITY_SEEDS
    ]
    assert r5._runtime_integrity_passed(reports) is True

    for field, key in (
        ("no_reset_state_residual", "h"),
        ("background_relation_maximum_absolute_read_residual", "h_lww"),
    ):
        mutated = copy.deepcopy(reports)
        mutated[0]["evaluation"]["runtime_integrity"][field][key] = 1e9
        assert r5._runtime_integrity_passed(mutated) is False

    mutated = copy.deepcopy(reports)
    mutated[0]["evaluation"]["runtime_integrity"][
        "shared_full_transition_controls"
    ] = False
    assert r5._runtime_integrity_passed(mutated) is False


def test_frozen_r4_parent_hash_and_contract() -> None:
    report, actual = r5._validate_r4(r5.R4_ARTIFACT)
    assert actual == r5.EXPECTED_R4_SHA256
    assert report["adjudication"]["passed"] is False
    assert report["adjudication"]["passed_value_arms"] == []
