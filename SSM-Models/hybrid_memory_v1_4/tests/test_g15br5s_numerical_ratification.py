from __future__ import annotations

import copy
import json

import pytest
import torch

from hybrid_memory_v1_4 import g15br3_logical_component as r3
from hybrid_memory_v1_4 import g15br5_causal_tail_source as r5
from hybrid_memory_v1_4 import g15br5s_numerical_ratification as r5s
from hybrid_memory_v1_4.g15b_interleaved_cohort import build_model
from hybrid_memory_v1_4.g15b_interleaved_tasks import generate_interleaved_batch


def test_sealed_r5_validation_and_numerical_audit_pass() -> None:
    report, actual = r5s._validate_r5(r5s.R5_ARTIFACT)
    assert actual == r5s.EXPECTED_R5_SHA256
    assert report["adjudication"]["passed"] is False
    assert report["adjudication"]["performance_passed"]["h_lww_bgminus"] is True
    audit = r5s._audit_sealed_r5(report)
    assert audit["passed"] is True
    assert audit["zero_discrete_and_learned_replay_passed"] is True
    assert audit["no_reset_bpq_replay_maximum_absolute_residual"] <= 1e-6
    assert audit["fp64_maximum_absolute_residual"] <= 1e-10


def test_sealed_audit_fails_on_discrete_drift() -> None:
    report, _ = r5s._validate_r5(r5s.R5_ARTIFACT)
    mutated = copy.deepcopy(report)
    first = mutated["seed_reports"][0]["evaluation"]["cells"]["mqar:L128"]
    first["r4_reference_replay"]["learned:query_accuracy"] = 1e-12
    assert r5s._audit_sealed_r5(mutated)["passed"] is False


def test_validator_rejects_changed_history_performance_status(
    tmp_path, monkeypatch
) -> None:
    report, _ = r5s._validate_r5(r5s.R5_ARTIFACT)
    mutated = copy.deepcopy(report)
    mutated["adjudication"]["performance_passed"]["h_lww_bgminus"] = False
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    monkeypatch.setattr(r5s, "_sha256", lambda _path: r5s.EXPECTED_R5_SHA256)
    with pytest.raises(ValueError, match="performance-positive"):
        r5s._validate_r5(path)


def test_sealed_r5_source_hash_audit_fails_closed_on_mutation() -> None:
    report, _ = r5s._validate_r5(r5s.R5_ARTIFACT)
    assert r5s._audit_r5_source_hashes(report)["passed"] is True
    mutated = copy.deepcopy(report)
    first = next(iter(mutated["source_files"]))
    mutated["source_files"][first] = "0" * 64
    assert r5s._audit_r5_source_hashes(mutated)["passed"] is False


def test_scaled_tensor_residual_has_fixed_accept_reject_boundary() -> None:
    reference = torch.tensor([0.0, 1.0, 10.0], dtype=torch.float32)
    inside = reference + 32.0 * r5s.FP32_EPSILON * torch.tensor([1.0, 1.0, 10.0])
    outside = reference + 96.0 * r5s.FP32_EPSILON * torch.tensor([1.0, 1.0, 10.0])
    assert r5s._scaled_tensor_residual(reference, inside)["passed"] is True
    assert r5s._scaled_tensor_residual(reference, outside)["passed"] is False


def test_fresh_fingerprints_are_deterministic_and_namespace_disjoint() -> None:
    common = {
        "checkpoint_seed": 2309,
        "task": "mqar",
        "length": 128,
        "decisions": 16,
        "batch_cap": 2,
    }
    first = r5s._cohort_fingerprints("g15br5s-stability", **common)
    second = r5s._cohort_fingerprints("g15br5s-stability", **common)
    original = r5s._cohort_fingerprints("g15b-eval", **common)
    assert first["fingerprints"] == second["fingerprints"]
    assert first["aggregate_sha256"] == second["aggregate_sha256"]
    assert first["fingerprint_set"].isdisjoint(original["fingerprint_set"])


def test_fresh_source_decomposition_matches_direct_categorical_path() -> None:
    model = build_model("I", 71, torch.device("cpu")).eval()
    batch = generate_interleaved_batch("overwrite", 2, 128, 8, 24, 8, seed=73)
    with torch.no_grad():
        result = r5.source_forwards(model, batch)
        monolithic = r3._erase_free_monolithic_forward(model, batch)
        hidden, outer_gate, mixed, mixer = result["forward_context"]
        reference = monolithic["logits"]
        reference_prediction = reference.argmax(-1)
        for source in r5s.SOURCES:
            reads = result["reads"][(source, False)]
            candidate = r3._finish_forward(
                model, hidden, outer_gate, mixed, reads["full_read"], mixer
            )
            assert r5s._scaled_tensor_residual(reference, candidate)["passed"] is True
            assert torch.equal(reference_prediction, candidate.argmax(-1))
            assert reads["state_residual"] <= r5s.ABSOLUTE_COMPONENT_BOUND
            assert reads["background_relation_residual"] <= r5s.ABSOLUTE_COMPONENT_BOUND
            assert reads["assignment_residual"] == 0.0

        controls = [result["component_controls"][name] for name in r3.CONTROL_NAMES]
        transition = r5s._independent_transition_contract(
            result["forward_context"][3], controls, result["full_injection"]
        )
        assert transition["passed"] is True
        changed = result["full_injection"].clone()
        changed[0, 0, 0, 0, 0] += 1e-4
        assert (
            r5s._independent_transition_contract(
                result["forward_context"][3], controls, changed
            )["passed"]
            is False
        )


def test_source_gate_has_frozen_absolute_and_categorical_boundaries() -> None:
    metrics = {
        "state_maximum_absolute_residual": 5e-6,
        "background_relation_maximum_absolute_residual": 5e-6,
        "injection_sum_maximum_absolute_residual": 5e-6,
        "source_assignment_maximum_absolute_residual": 0.0,
        "logit_maximum_scaled_allowance_ratio": 1.0,
        "finite_logits": True,
    }
    assert r5s._source_gate(metrics, categorical_equal=True, bpq_absolute_residual=1e-6)
    changed = dict(metrics)
    changed["state_maximum_absolute_residual"] = 5.0001e-6
    assert not r5s._source_gate(
        changed, categorical_equal=True, bpq_absolute_residual=1e-6
    )
    assert not r5s._source_gate(
        metrics, categorical_equal=False, bpq_absolute_residual=1e-6
    )


def _seed_report(seed: int, *, passed: bool = True, omit_cell: bool = False):
    cells = {
        f"{task}:L{length}": {"query_decisions": 512}
        for task in r5s.TASKS
        for length in r5s.EVALUATION_LENGTHS
    }
    if omit_cell:
        cells.pop(next(iter(cells)))
    return {
        "checkpoint_seed": seed,
        "checkpoint_matches_sealed_r5": True,
        "evaluation": {"passed": passed, "cells": cells},
    }


def test_adjudication_cannot_pass_smoke_or_failed_fresh_cohort() -> None:
    sealed = {"passed": True}
    source_hashes = {"passed": True}
    passing = [_seed_report(2309)]
    for cell in passing[0]["evaluation"]["cells"].values():
        cell["query_decisions"] = 16
    smoke = r5s._adjudicate(
        mode="smoke",
        sealed_audit=sealed,
        seed_reports=passing,
        source_hash_audit=source_hashes,
        clean_start=True,
        exact_sm75=True,
    )
    assert smoke["passed"] is False
    assert "smoke" in smoke["decision"]
    failed = r5s._adjudicate(
        mode="quality",
        sealed_audit=sealed,
        seed_reports=[
            _seed_report(2309),
            _seed_report(2311, passed=False),
            _seed_report(2333),
        ],
        source_hash_audit=source_hashes,
        clean_start=True,
        exact_sm75=True,
    )
    assert failed["passed"] is False


def test_adjudication_fails_closed_on_missing_cell() -> None:
    result = r5s._adjudicate(
        mode="quality",
        sealed_audit={"passed": True},
        seed_reports=[
            _seed_report(2309),
            _seed_report(2311, omit_cell=True),
            _seed_report(2333),
        ],
        source_hash_audit={"passed": True},
        clean_start=True,
        exact_sm75=True,
    )
    assert result["fresh_cohort_complete"] is False
    assert result["passed"] is False
