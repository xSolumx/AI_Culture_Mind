"""Contracts for the prospectively frozen G15B-E training cohort."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hybrid_memory_v1_4.g15b_interleaved_tasks import generate_interleaved_batch
from hybrid_memory_v1_4.g15be_effective_edit_cohort import (
    ARMS,
    EVALUATION_LENGTHS,
    EVALUATION_TASKS,
    INTERVENTIONS,
    QUALITY_SEEDS,
    _adjudicate,
    _diagnostics,
    _enforce_execution_eligibility,
    _event_mask,
    _model_state_sha256,
    _tensor,
    build_model,
    commissioned_losses,
    effective_edit_intervention_forward,
    frozen_config,
    run_preflight,
)
from hybrid_memory_v1_4.model import parameter_count


def _batch() -> object:
    return generate_interleaved_batch(
        "overwrite", 3, 128, 4, 8, 4, seed=2481
    )


def test_product_and_additive_training_arms_are_exactly_matched() -> None:
    models = {arm: build_model(arm, 2481, torch.device("cpu")) for arm in ARMS}
    assert parameter_count(models["P"]) == parameter_count(models["A"])
    assert models["P"].state_capacity_bytes(
        1, torch.float32
    ) == models["A"].state_capacity_bytes(1, torch.float32)
    assert _model_state_sha256(models["P"]) == _model_state_sha256(models["A"])
    assert (
        models["P"].config.transactional_effective_edit_gate_mode == "product"
    )
    assert (
        models["A"].config.transactional_effective_edit_gate_mode
        == "logit_additive"
    )


def test_primary_loss_reaches_both_effective_gate_graphs() -> None:
    batch = _batch()
    assert hasattr(batch, "token_ids")
    for arm in ARMS:
        model = build_model(arm, 2483, torch.device("cpu"))
        output = model(batch.token_ids, return_diagnostics=True)
        loss, components = commissioned_losses(output, batch)
        assert set(components) == {
            "retrieval",
            "reverse_binding",
            "query_to_value_position_address",
        }
        loss.backward()
        parameters = dict(model.named_parameters())
        for suffix in (
            "commit_projection.weight",
            "erase_projection.weight",
            "write_projection.weight",
        ):
            name = next(name for name in parameters if name.endswith(suffix))
            gradient = parameters[name].grad
            assert gradient is not None
            assert torch.isfinite(gradient).all()
            assert torch.count_nonzero(gradient) > 0


def test_interventions_operate_on_effective_edits_and_reconstruct() -> None:
    batch = _batch()
    model = build_model("A", 2489, torch.device("cpu")).eval()
    with torch.no_grad():
        ordinary = model(batch.token_ids)["logits"]
        replay = effective_edit_intervention_forward(
            model, batch, "learned_reconstruction"
        )
        torch.testing.assert_close(replay["logits"], ordinary, rtol=0.0, atol=0.0)
        event = _event_mask(batch, dtype=torch.float32).bool()
        event_zero = effective_edit_intervention_forward(
            model, batch, "valid_event_edit_zero"
        )
        event_only = effective_edit_intervention_forward(
            model, batch, "valid_event_only"
        )
        non_event_only = effective_edit_intervention_forward(
            model, batch, "non_event_only"
        )
        erase_zero = effective_edit_intervention_forward(
            model, batch, "erase_zero"
        )
        permuted = effective_edit_intervention_forward(
            model, batch, "permuted_write_binding"
        )
    assert torch.count_nonzero(event_zero["effective_erase"][event.expand_as(event_zero["effective_erase"])]) == 0
    assert torch.count_nonzero(event_zero["effective_write"][event.expand_as(event_zero["effective_write"])]) == 0
    assert torch.count_nonzero(event_only["effective_erase"][~event.expand_as(event_only["effective_erase"])]) == 0
    assert torch.count_nonzero(event_only["effective_write"][~event.expand_as(event_only["effective_write"])]) == 0
    assert torch.count_nonzero(non_event_only["effective_erase"][event.expand_as(non_event_only["effective_erase"])]) == 0
    assert torch.count_nonzero(non_event_only["effective_write"][event.expand_as(non_event_only["effective_write"])]) == 0
    assert torch.count_nonzero(erase_zero["effective_erase"]) == 0
    for row in range(batch.batch_size):
        positions = batch.write_positions[row]
        torch.testing.assert_close(
            permuted["written_payload"][row, positions],
            replay["written_payload"][row, positions.roll(1)],
            rtol=0.0,
            atol=0.0,
        )


def test_diagnostics_expose_only_bounded_effective_gate_metrics() -> None:
    batch = _batch()
    model = build_model("A", 2491, torch.device("cpu"))
    diagnostics = _diagnostics(model(batch.token_ids, return_diagnostics=True))
    for name in ("effective_erase_strength", "effective_write_strength"):
        gate = _tensor(diagnostics, name)
        assert bool(((gate > 0.0) & (gate < 1.0)).all())


def test_phase1_preflight_passes_on_semantic_cpu_path() -> None:
    report = run_preflight(torch.device("cpu"))
    assert report["passed"] is True
    assert all(report["checks"].values())


def _passing_cell(task: str, length: int, decisions: int) -> dict[str, object]:
    cell: dict[str, object] = {
        "task": task,
        "length": length,
        "query_decisions": decisions,
        "query_accuracy": 1.0,
        "exact_episode_accuracy": 1.0,
        "bits_per_query": 0.0,
        "query_address_top1": 1.0,
    }
    if task in ("overwrite", "overwrite_guard"):
        counts = (
            (decisions // 4, decisions // 4, decisions // 2)
            if task == "overwrite_guard"
            else (decisions // 2, 0, decisions // 2)
        )
        cell["query_strata"] = {
            name: {
                "query_decisions": count,
                "accuracy": 1.0 if count else None,
            }
            for name, count in zip(
                (
                    "before_any_overwrite",
                    "after_unrelated_overwrite_only",
                    "after_same_key_overwrite",
                ),
                counts,
                strict=True,
            )
        }
    return cell


def _passing_intervention_cell(task: str, length: int) -> dict[str, object]:
    decisions = 512
    cell = _passing_cell(task, length, decisions)
    strata = cell.get(
        "query_strata",
        {
            "before_any_overwrite": {"query_decisions": 0, "accuracy": None},
            "after_unrelated_overwrite_only": {
                "query_decisions": 0,
                "accuracy": None,
            },
            "after_same_key_overwrite": {"query_decisions": 0, "accuracy": None},
        },
    )
    interventions = {}
    for name in INTERVENTIONS:
        accuracy = 1.0 if name in ("learned_reconstruction", "valid_event_only") else 0.0
        if name == "erase_zero" and task == "mqar":
            accuracy = 1.0
        intervention_strata = deepcopy(strata)
        if name == "erase_zero" and task == "overwrite":
            intervention_strata["after_same_key_overwrite"]["accuracy"] = 0.0
        interventions[name] = {
            "query_accuracy": accuracy,
            "drop_from_learned": 1.0 - accuracy,
            "query_strata": intervention_strata,
        }
    cell["interventions"] = interventions
    cell["reconstruction_maximum_absolute_logit_residual"] = 0.0
    cell["reconstruction_query_predictions_equal"] = True
    return cell


def _passing_reports(tmp_path: Path) -> list[dict[str, object]]:
    reports = []
    for seed in QUALITY_SEEDS:
        for arm in ARMS:
            checkpoint = tmp_path / f"{arm}-{seed}.pt"
            checkpoint.write_bytes(f"{arm}-{seed}".encode())
            import hashlib

            checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            cells = {
                f"{task}:L{length}": _passing_cell(task, length, 2048)
                for task in EVALUATION_TASKS
                for length in EVALUATION_LENGTHS
            }
            if arm == "P":
                for length in (128, 512, 1024):
                    overwrite = cells[f"overwrite:L{length}"]
                    overwrite["query_accuracy"] = 0.95
                    overwrite["query_strata"]["after_same_key_overwrite"][
                        "accuracy"
                    ] = 0.95
            interventions = (
                {
                    f"{task}:L{length}": _passing_intervention_cell(task, length)
                    for task in ("mqar", "overwrite", "selective")
                    for length in (512, 1024)
                }
                if arm == "A"
                else {}
            )
            reports.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                    "parameters": 67033,
                    "active_parameters": 67033,
                    "state_bytes_per_sequence_fp32": 4864,
                    "initial_parameter_sha256": f"initial-{seed}",
                    "training_updates": 3400,
                    "training_tokens": 13_926_400,
                    "training_schedule_sha256": f"train-{seed}",
                    "train_evaluation_hash_intersection": [],
                    "evaluation": {
                        "cells": cells,
                        "intervention_cells": interventions,
                        "standard_evaluation_schedule_sha256": f"eval-{seed}",
                        "boundary_batch_sha256": f"boundary-{seed}",
                        "boundary_audit": {"passed": True},
                    },
                }
            )
    return reports


def test_adjudicator_fails_closed_on_repaired_gate_edges(tmp_path: Path) -> None:
    config = frozen_config("quality")
    baseline = _passing_reports(tmp_path)
    assert _adjudicate(config, baseline)["passed"] is True

    improved_event_only = deepcopy(baseline)
    cell = improved_event_only[1]["evaluation"]["intervention_cells"]["mqar:L512"]
    cell["interventions"]["valid_event_only"]["drop_from_learned"] = -0.50
    result = _adjudicate(config, improved_event_only)
    assert result["checks"]["A:2481:mqar:event_only:L512"] is False

    missing_mqar_use = deepcopy(baseline)
    cell = missing_mqar_use[1]["evaluation"]["intervention_cells"]["mqar:L512"]
    cell["interventions"]["memory_zero"]["drop_from_learned"] = 0.0
    result = _adjudicate(config, missing_mqar_use)
    assert result["checks"]["A:2481:mqar:memory_zero:L512"] is False

    truncated = deepcopy(baseline)
    truncated[1]["evaluation"]["cells"]["mqar:L128"]["query_decisions"] = 1
    result = _adjudicate(config, truncated)
    assert result["checks"]["decisions:A:2481:mqar:L128"] is False


def test_product_decision_uses_its_full_absolute_gate_vector(tmp_path: Path) -> None:
    reports = _passing_reports(tmp_path)
    product = reports[0]
    product["evaluation"]["cells"]["overwrite_guard:L128"]["query_strata"][
        "after_same_key_overwrite"
    ]["accuracy"] = 0.0
    result = _adjudicate(frozen_config("quality"), reports)
    assert result["passed"] is True
    assert result["product_absolute_quality_passed"] is False
    assert result["decision"].startswith("A passes and P fails")


def test_dirty_execution_is_always_ineligible() -> None:
    clean = {"passed": True, "eligible_for_promotion": True, "decision": "pass"}
    assert _enforce_execution_eligibility(clean, []) == clean
    dirty = _enforce_execution_eligibility(clean, ["?? artifact.json"])
    assert dirty["passed"] is False
    assert dirty["eligible_for_promotion"] is False
    assert "dirty" in dirty["decision"]
