"""Contracts for the frozen G15B-D residual-delta cohort."""

from __future__ import annotations

import hashlib
import sys
from copy import deepcopy
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hybrid_memory_v1_4.g15bd_residual_delta_cohort import (
    ARMS,
    _adjudicate,
    frozen_config,
    run_preflight,
)
from hybrid_memory_v1_4.g15be_effective_edit_cohort import (
    EVALUATION_LENGTHS,
    EVALUATION_TASKS,
    INTERVENTIONS,
    QUALITY_SEEDS,
    build_model,
)
from hybrid_memory_v1_4.model import parameter_count


def _cell(task: str, length: int, decisions: int) -> dict[str, object]:
    cell: dict[str, object] = {
        "task": task,
        "length": length,
        "query_decisions": decisions,
        "query_accuracy": 1.0,
        "exact_episode_accuracy": 1.0,
        "bits_per_query": 0.0,
        "query_address_top1": 1.0,
        "state_norm_maximum": 1.0,
        "effective_gate_statistics": {
            "event_erase": {"mean": 0.50},
            "event_write": {"mean": 0.50},
            "non_event_erase": {"mean": 0.01},
            "non_event_write": {"mean": 0.01},
        },
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


def _intervention_cell(task: str, length: int) -> dict[str, object]:
    cell = _cell(task, length, 512)
    empty_strata = {
        "before_any_overwrite": {"query_decisions": 0, "accuracy": None},
        "after_unrelated_overwrite_only": {
            "query_decisions": 0,
            "accuracy": None,
        },
        "after_same_key_overwrite": {"query_decisions": 0, "accuracy": None},
    }
    strata = cell.get("query_strata", empty_strata)
    interventions = {}
    for name in INTERVENTIONS:
        accuracy = 1.0 if name in ("learned_reconstruction", "valid_event_only") else 0.0
        interventions[name] = {
            "query_accuracy": accuracy,
            "drop_from_learned": 1.0 - accuracy,
            "query_strata": deepcopy(strata),
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
            cells = {
                f"{task}:L{length}": _cell(task, length, 2048)
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
                    f"{task}:L{length}": _intervention_cell(task, length)
                    for task in ("mqar", "overwrite", "selective")
                    for length in (512, 1024)
                }
                if arm == "D"
                else {}
            )
            reports.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": hashlib.sha256(
                        checkpoint.read_bytes()
                    ).hexdigest(),
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


def test_g15bd_models_and_preflight_are_matched() -> None:
    assert frozen_config("smoke").evaluation_batch_cap == 2
    assert frozen_config("quality").evaluation_batch_cap == 4
    models = {
        arm: build_model(arm, 2581, torch.device("cpu"))  # type: ignore[arg-type]
        for arm in ARMS
    }
    assert parameter_count(models["P"]) == parameter_count(models["D"])
    assert models["P"].state_capacity_bytes(
        1, torch.float32
    ) == models["D"].state_capacity_bytes(1, torch.float32)
    assert (
        models["D"].config.transactional_effective_edit_gate_mode
        == "residual_delta"
    )
    preflight = run_preflight(torch.device("cpu"))
    assert preflight["passed"] is True
    assert all(preflight["checks"].values())


def test_g15bd_adjudicator_passes_complete_frozen_vector(tmp_path: Path) -> None:
    result = _adjudicate(frozen_config("quality"), _passing_reports(tmp_path))
    assert result["passed"] is True
    assert result["product_absolute_quality_passed"] is True
    assert result["delta_absolute_and_causal_passed"] is True
    assert all(result["shared_integrity_checks"].values())
    assert all(result["delta_absolute_and_causal_checks"].values())
    assert all(result["comparative_checks"].values())


def test_g15bd_adjudicator_fails_closed_on_locality_and_norms(
    tmp_path: Path,
) -> None:
    reports = _passing_reports(tmp_path)
    event_only = deepcopy(reports)
    cell = event_only[1]["evaluation"]["intervention_cells"]["mqar:L512"]
    cell["interventions"]["valid_event_only"]["drop_from_learned"] = 0.021
    result = _adjudicate(frozen_config("quality"), event_only)
    assert result["delta_absolute_and_causal_checks"][
        "D:2481:mqar:event_only:L512"
    ] is False
    assert result["passed"] is False

    diffuse = deepcopy(reports)
    cell = diffuse[1]["evaluation"]["intervention_cells"]["overwrite:L1024"]
    cell["effective_gate_statistics"]["non_event_write"]["mean"] = 0.051
    result = _adjudicate(frozen_config("quality"), diffuse)
    assert result["delta_absolute_and_causal_checks"][
        "D:2481:non_event_edit_mean:L1024"
    ] is False

    inflated = deepcopy(reports)
    inflated[1]["evaluation"]["cells"]["overwrite:L512"][
        "state_norm_maximum"
    ] = 1.251
    result = _adjudicate(frozen_config("quality"), inflated)
    assert result["comparative_checks"][
        "D:state_norm:2481:overwrite:L512"
    ] is False


def test_g15bd_delta_decision_is_independent_of_product_quality(
    tmp_path: Path,
) -> None:
    reports = _passing_reports(tmp_path)
    reports[0]["evaluation"]["cells"]["overwrite:L128"]["query_accuracy"] = 0.50
    result = _adjudicate(frozen_config("quality"), reports)
    assert result["product_absolute_quality_passed"] is False
    assert result["delta_absolute_and_causal_passed"] is True
    assert result["passed"] is True
    assert result["decision"].startswith("D passes and P fails")


def test_g15bd_adjudicator_rejects_incomplete_decisions(tmp_path: Path) -> None:
    reports = _passing_reports(tmp_path)
    reports[1]["evaluation"]["cells"]["mqar:L128"]["query_decisions"] = 1
    result = _adjudicate(frozen_config("quality"), reports)
    assert result["shared_integrity_checks"][
        "decisions:D:2481:mqar:L128"
    ] is False
    assert result["passed"] is False
