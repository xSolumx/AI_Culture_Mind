"""G15B-R5-S prospective numerical ratification on fresh stability batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

from . import g15br3_logical_component as r3
from . import g15br5_causal_tail_source as r5
from .g15b_interleaved_cohort import (
    EVALUATION_LENGTHS,
    NEEDLE_DISTANCES,
    _evaluation_batch_size,
    _gather_time,
    _sha256,
    _stable_seed,
)
from .g15b_interleaved_tasks import generate_interleaved_batch
from .g15br_checkpoint_repair import (
    PARENT_ARTIFACT,
    QUALITY_SEEDS,
    _expected_identity,
    _git_provenance,
    _load_checkpoint,
    _sync,
    local_write_event_mask,
)
from .model import HybridMemoryLM

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "G15BR5S_NUMERICAL_RATIFICATION_PROTOCOL_2026-08-26.md"
R5_ARTIFACT = ROOT / "artifacts/g15br5_causal_tail_source_sm75_2026-08-26.json"
EXPECTED_R5_SHA256 = "ba627fe34e8dd29458fc1321b52c98242838c3b56e2abdc7e44c749f50aaa313"
EXPECTED_R5_COMMIT = "e039e499b44b8e9bbb1108eb456c051a4702ba4e"
SOURCES = ("h", "c", "b")
TASKS = ("mqar", "overwrite", "overwrite_guard", "selective", "needle")
FP32_EPSILON = torch.finfo(torch.float32).eps
ABSOLUTE_COMPONENT_BOUND = 5e-6
SCALED_LOGIT_MULTIPLIER = 64.0
BPQ_ABSOLUTE_BOUND = 1e-6


def _validate_r5(path: Path) -> tuple[dict[str, Any], str]:
    actual_sha256 = _sha256(path)
    if actual_sha256 != EXPECTED_R5_SHA256:
        raise ValueError("R5 artifact hash does not match the frozen input")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not (
        report.get("mode") == "quality"
        and report.get("evidentiary") is True
        and report.get("git_commit_at_start") == EXPECTED_R5_COMMIT
        and report.get("git_status_at_start") == []
        and report.get("environment", {}).get("compute_capability") == [7, 5]
        and report.get("protocol", {}).get("optimizer_updates") == 0
        and report.get("protocol", {}).get("seeds") == list(QUALITY_SEEDS)
    ):
        raise ValueError("R5 is not the sealed clean exact-SM75 quality artifact")
    adjudication = report.get("adjudication", {})
    performance = adjudication.get("performance_passed", {})
    if not (
        adjudication.get("passed") is False
        and adjudication.get("passed_arms") == []
        and adjudication.get("r4_reference_replay_passed") is False
        and adjudication.get("runtime_integrity_passed") is False
        and performance.get("h_lww_bgminus") is True
        and all(
            passed is (arm == "h_lww_bgminus") for arm, passed in performance.items()
        )
        and len(adjudication.get("arm_checks", {}).get("h_lww_bgminus", {})) == 132
        and all(adjudication["arm_checks"]["h_lww_bgminus"].values())
        and adjudication.get("decision")
        == "stop retained-checkpoint tail repair; no bias-separated background-free history source passes"
    ):
        raise ValueError("R5 does not have the frozen performance-positive formal fail")
    return report, actual_sha256


def _audit_sealed_r5(report: dict[str, Any]) -> dict[str, Any]:
    replay_by_kind: dict[str, list[float]] = {}
    fingerprint_passed = True
    for seed_report in report["seed_reports"]:
        for cell in seed_report["evaluation"]["cells"].values():
            fingerprint_passed = fingerprint_passed and bool(
                cell.get("r4_batch_fingerprint_match")
            )
            for name, residual in cell["r4_reference_replay"].items():
                replay_by_kind.setdefault(name, []).append(float(residual))

    required_zero = (
        "learned:query_accuracy",
        "learned:exact_episode_accuracy",
        "learned:bits_per_query",
        "erase_free_no_reset_bgplus:query_accuracy",
        "erase_free_no_reset_bgplus:exact_episode_accuracy",
    )
    zero_replay_passed = all(
        max(replay_by_kind.get(name, [math.inf])) == 0.0 for name in required_zero
    )
    bpq_values = replay_by_kind.get(
        "erase_free_no_reset_bgplus:bits_per_query", [math.inf]
    )
    bpq_maximum = max(bpq_values)
    known_replay_names = set(required_zero) | {
        "erase_free_no_reset_bgplus:bits_per_query"
    }
    replay_schema_passed = set(replay_by_kind) == known_replay_names

    maximum_preactivation = 0.0
    maximum_injection_sum = 0.0
    maximum_assignment = 0.0
    maximum_state = 0.0
    maximum_background_relation = 0.0
    maximum_fp64 = 0.0
    runtime_booleans_passed = True
    seed_rows: dict[str, Any] = {}
    for seed_report in report["seed_reports"]:
        integrity = seed_report["evaluation"]["runtime_integrity"]
        maximum_preactivation = max(
            maximum_preactivation,
            float(integrity["preactivation_reconstruction_maximum_absolute_residual"]),
        )
        maximum_injection_sum = max(
            maximum_injection_sum,
            *map(float, integrity["injection_sum_maximum_absolute_residual"].values()),
        )
        maximum_assignment = max(
            maximum_assignment,
            *map(
                float, integrity["source_assignment_maximum_absolute_residual"].values()
            ),
        )
        maximum_state = max(
            maximum_state,
            *map(float, integrity["no_reset_state_residual"].values()),
        )
        maximum_background_relation = max(
            maximum_background_relation,
            *map(
                float,
                integrity[
                    "background_relation_maximum_absolute_read_residual"
                ].values(),
            ),
        )
        fp64 = float(integrity["fp64_algebraic_contract"]["maximum_residual"])
        maximum_fp64 = max(maximum_fp64, fp64)
        seed_boolean = (
            integrity["ordinary_model_forward_maximum_absolute_logit_residual"] == 0.0
            and all(integrity["no_reset_query_predictions_equal"].values())
            and all(integrity["lww_no_reset_no_overwrite_predictions_equal"].values())
            and all(
                integrity["lww_no_reset_before_overwrite_predictions_equal"].values()
            )
            and integrity["preserved_controls_bitwise_equal"]
            and integrity["shared_full_transition_controls"]
            and integrity["finite_logits"]
            and integrity["convolution_structure_contract"]["passed"]
            and integrity["source_locality_witness"]["passed"]
            and integrity["fp64_algebraic_contract"]["passed"]
        )
        runtime_booleans_passed = runtime_booleans_passed and seed_boolean
        seed_rows[str(seed_report["seed"])] = {
            "runtime_boolean_contracts_passed": seed_boolean,
            "fp64_maximum_residual": fp64,
        }

    passed = (
        fingerprint_passed
        and replay_schema_passed
        and zero_replay_passed
        and bpq_maximum <= BPQ_ABSOLUTE_BOUND
        and runtime_booleans_passed
        and maximum_preactivation <= 2e-6
        and maximum_injection_sum <= ABSOLUTE_COMPONENT_BOUND
        and maximum_assignment == 0.0
        and maximum_state <= ABSOLUTE_COMPONENT_BOUND
        and maximum_background_relation <= ABSOLUTE_COMPONENT_BOUND
        and maximum_fp64 <= 1e-10
    )
    return {
        "passed": passed,
        "r4_batch_fingerprints_passed": fingerprint_passed,
        "replay_schema_passed": replay_schema_passed,
        "zero_discrete_and_learned_replay_passed": zero_replay_passed,
        "no_reset_bpq_replay_maximum_absolute_residual": bpq_maximum,
        "runtime_boolean_contracts_passed": runtime_booleans_passed,
        "preactivation_maximum_absolute_residual": maximum_preactivation,
        "injection_sum_maximum_absolute_residual": maximum_injection_sum,
        "source_assignment_maximum_absolute_residual": maximum_assignment,
        "no_reset_state_maximum_absolute_residual": maximum_state,
        "background_relation_maximum_absolute_residual": maximum_background_relation,
        "fp64_maximum_absolute_residual": maximum_fp64,
        "seeds": seed_rows,
    }


def _audit_r5_source_hashes(report: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for relative, expected in report["source_files"].items():
        path = ROOT / relative
        actual = _sha256(path) if path.is_file() else None
        rows[relative] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "passed": actual == expected,
        }
    return {
        "files": rows,
        "passed": bool(rows) and all(row["passed"] for row in rows.values()),
    }


def _scaled_tensor_residual(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float | bool]:
    if reference.shape != candidate.shape:
        raise ValueError("scaled comparison tensors must have the same shape")
    difference = (candidate - reference).abs()
    symmetric_scale = torch.maximum(reference.abs(), candidate.abs()).clamp_min(1.0)
    allowance = SCALED_LOGIT_MULTIPLIER * FP32_EPSILON * symmetric_scale
    maximum_absolute = float(difference.max())
    maximum_ratio = float((difference / allowance).max())
    return {
        "maximum_absolute_residual": maximum_absolute,
        "maximum_scaled_allowance_ratio": maximum_ratio,
        "passed": maximum_ratio <= 1.0,
    }


def _cell_batch(
    task: str,
    *,
    batch_size: int,
    length: int,
    seed: int,
    device: torch.device,
):
    if task == "overwrite_guard":
        return r3.generate_component_guard_batch(batch_size, length, seed=seed).to(
            device
        )
    return generate_interleaved_batch(
        task,  # type: ignore[arg-type]
        batch_size,
        length,
        8,
        24,
        8,
        seed=seed,
        needle_distance=NEEDLE_DISTANCES[length] if task == "needle" else None,
    ).to(device)


def _cohort_fingerprints(
    namespace: str,
    *,
    checkpoint_seed: int,
    task: str,
    length: int,
    decisions: int,
    batch_cap: int,
) -> dict[str, Any]:
    generation_task = "overwrite" if task == "overwrite_guard" else task
    batch_size = _evaluation_batch_size(
        generation_task, decisions=decisions, cap=batch_cap
    )
    per_batch = batch_size * (1 if task == "needle" else 8)
    if decisions % per_batch:
        raise ValueError("decisions must contain complete fingerprint batches")
    fingerprints: list[str] = []
    digest = hashlib.sha256()
    total = 0
    batch_index = 0
    while total < decisions:
        batch_seed = _stable_seed(namespace, checkpoint_seed, task, length, batch_index)
        batch = _cell_batch(
            task,
            batch_size=batch_size,
            length=length,
            seed=batch_seed,
            device=torch.device("cpu"),
        )
        fingerprint = batch.fingerprint()
        fingerprints.append(fingerprint)
        digest.update(fingerprint.encode())
        total += batch.targets.numel()
        batch_index += 1
    return {
        "fingerprints": fingerprints,
        "fingerprint_set": set(fingerprints),
        "aggregate_sha256": digest.hexdigest(),
        "query_decisions": total,
    }


def _independent_transition_contract(
    mixer: Any,
    controls: list[torch.Tensor],
    used_full_injection: torch.Tensor,
) -> dict[str, Any]:
    first = mixer._transitions(*controls[1:], None)
    second = mixer._transitions(*controls[1:], None)
    left_equal = torch.equal(first[0], second[0])
    right_equal = torch.equal(first[1], second[1])
    injection_equal = torch.equal(first[2], second[2]) and torch.equal(
        first[2], used_full_injection
    )
    return {
        "left_bitwise_equal": left_equal,
        "right_bitwise_equal": right_equal,
        "full_injection_bitwise_equal": injection_equal,
        "left_maximum_absolute_residual": float((first[0] - second[0]).abs().max()),
        "right_maximum_absolute_residual": float((first[1] - second[1]).abs().max()),
        "injection_maximum_absolute_residual": max(
            float((first[2] - second[2]).abs().max()),
            float((first[2] - used_full_injection).abs().max()),
        ),
        "passed": left_equal and right_equal and injection_equal,
    }


def _common_fp64_worst_batch_diagnostics(
    model: HybridMemoryLM, batch: Any
) -> dict[str, Any]:
    context = r5._source_context(model, batch.token_ids)
    mixer = context["mixer"]
    controls = list(context["learned_controls"])
    controls[3] = torch.zeros_like(controls[3])
    query, key, value, erase, write, retention, coordinates = controls
    left, right, full_injection, _ = mixer._transitions(
        key, value, erase, write, retention, coordinates, None
    )
    batch_size, heads, length, width, _ = full_injection.shape
    zero32 = full_injection.new_zeros(batch_size, heads, width, width)
    monolithic32, _ = mixer._parallel_states(left, right, full_injection, zero32)
    left64 = left.double()
    right64 = right.double()
    full64 = full_injection.double()
    zero64 = full64.new_zeros(batch_size, heads, width, width)
    monolithic64, _ = mixer._parallel_states(left64, right64, full64, zero64)
    value_owner, tail_owner, reset_mask = r5.source_ownership(batch)
    components = reset_mask.shape[1]
    sources = {
        "h": r5._injection_from_mixed(mixer, context["history_mixed"]),
        "c": r5._injection_from_mixed(mixer, context["current_mixed"]),
        "b": r5._injection_from_mixed(mixer, context["bias_mixed"]),
    }
    query64 = query.double()
    monolithic_read32 = r3._read_from_states(mixer, query, monolithic32)
    monolithic_read64 = r3._read_from_states(mixer, query64, monolithic64)
    state_scale = max(1.0, float(monolithic64.abs().max()))
    read_scale = max(1.0, float(monolithic_read64.abs().max()))
    source_rows: dict[str, Any] = {}
    for source, source_injection in sources.items():
        component_injection32, injection_sum32 = r5._split_component_injection(
            full_injection,
            source_injection,
            value_owner,
            tail_owner,
            components,
        )
        left_component32 = left[:, None].expand(
            batch_size, components, heads, length, width, width
        )
        right_component32 = right[:, None].expand_as(left_component32)
        flat_shape = (batch_size * components, heads, length, width, width)
        component32, _ = mixer._parallel_states(
            left_component32.reshape(flat_shape),
            right_component32.reshape(flat_shape),
            component_injection32.reshape(flat_shape),
            full_injection.new_zeros(batch_size * components, heads, width, width),
        )
        decomposed32 = component32.reshape(
            batch_size, components, heads, length, width, width
        ).sum(dim=1)
        decomposed_read32 = r3._read_from_states(mixer, query, decomposed32)
        state_mono_to_fp64 = float((monolithic32.double() - monolithic64).abs().max())
        state_decomp_to_fp64 = float((decomposed32.double() - monolithic64).abs().max())
        read_mono_to_fp64 = float(
            (monolithic_read32.double() - monolithic_read64).abs().max()
        )
        read_decomp_to_fp64 = float(
            (decomposed_read32.double() - monolithic_read64).abs().max()
        )
        source_rows[source] = {
            "state_reference_maximum_magnitude": state_scale,
            "read_reference_maximum_magnitude": read_scale,
            "fp32_monolithic_state_to_common_fp64_absolute_residual": state_mono_to_fp64,
            "fp32_decomposed_state_to_common_fp64_absolute_residual": state_decomp_to_fp64,
            "fp32_monolithic_state_to_common_fp64_normalized_residual": state_mono_to_fp64
            / state_scale,
            "fp32_decomposed_state_to_common_fp64_normalized_residual": state_decomp_to_fp64
            / state_scale,
            "fp32_monolithic_read_to_common_fp64_absolute_residual": read_mono_to_fp64,
            "fp32_decomposed_read_to_common_fp64_absolute_residual": read_decomp_to_fp64,
            "fp32_monolithic_read_to_common_fp64_normalized_residual": read_mono_to_fp64
            / read_scale,
            "fp32_decomposed_read_to_common_fp64_normalized_residual": read_decomp_to_fp64
            / read_scale,
            "fp32_component_injection_sum_maximum_absolute_residual": injection_sum32,
        }
    explicit_contract = r5.fp64_algebraic_contract(model, batch)
    maximum_fp64 = float(explicit_contract["maximum_residual"])
    return {
        "common_reference": "FP64 scan/read of the scored FP32 transitions and injections",
        "sources": source_rows,
        "independently_recomputed_fp64_contract": explicit_contract,
        "maximum_fp64_algebraic_residual": maximum_fp64,
        "passed": explicit_contract["passed"] and maximum_fp64 <= 1e-10,
    }


def _source_gate(
    metrics: dict[str, float | bool],
    *,
    categorical_equal: bool,
    bpq_absolute_residual: float,
) -> bool:
    """Apply only the prospective R5-S source thresholds."""

    return bool(
        float(metrics["state_maximum_absolute_residual"]) <= ABSOLUTE_COMPONENT_BOUND
        and float(metrics["background_relation_maximum_absolute_residual"])
        <= ABSOLUTE_COMPONENT_BOUND
        and float(metrics["injection_sum_maximum_absolute_residual"])
        <= ABSOLUTE_COMPONENT_BOUND
        and float(metrics["source_assignment_maximum_absolute_residual"]) == 0.0
        and float(metrics["logit_maximum_scaled_allowance_ratio"]) <= 1.0
        and categorical_equal
        and bpq_absolute_residual <= BPQ_ABSOLUTE_BOUND
        and bool(metrics["finite_logits"])
    )


def _batch_severity(
    *,
    reads: dict[str, Any],
    comparison: dict[str, float | bool],
    bpq_absolute_residual: float,
    categorical_equal: bool,
    finite: bool,
    transition_passed: bool,
) -> tuple[float, dict[str, float | bool]]:
    """Rank a batch by its largest normalized prospective-gate residual."""

    components: dict[str, float | bool] = {
        "state_bound_ratio": float(reads["state_residual"]) / ABSOLUTE_COMPONENT_BOUND,
        "background_relation_bound_ratio": float(reads["background_relation_residual"])
        / ABSOLUTE_COMPONENT_BOUND,
        "injection_sum_bound_ratio": float(reads["injection_sum_residual"])
        / ABSOLUTE_COMPONENT_BOUND,
        "source_assignment_exact": float(reads["assignment_residual"]) == 0.0,
        "logit_scaled_allowance_ratio": float(
            comparison["maximum_scaled_allowance_ratio"]
        ),
        "bpq_absolute_bound_ratio": bpq_absolute_residual / BPQ_ABSOLUTE_BOUND,
        "categorical_equal": categorical_equal,
        "finite": finite,
        "transition_passed": transition_passed,
    }
    score = max(
        float(components["state_bound_ratio"]),
        float(components["background_relation_bound_ratio"]),
        float(components["injection_sum_bound_ratio"]),
        float(components["logit_scaled_allowance_ratio"]),
        float(components["bpq_absolute_bound_ratio"]),
    )
    if not (
        bool(components["source_assignment_exact"])
        and categorical_equal
        and finite
        and transition_passed
    ):
        score = math.inf
    return score, components


@torch.no_grad()
def evaluate_checkpoint(
    model: HybridMemoryLM,
    *,
    checkpoint_seed: int,
    decisions: int,
    batch_cap: int,
    r5_cells: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    device = model.embedding.weight.device
    cells: dict[str, Any] = {}
    for task in TASKS:
        for length in EVALUATION_LENGTHS:
            cell_name = f"{task}:L{length}"
            if cell_name not in r5_cells:
                raise ValueError(f"sealed R5 cell is missing: {cell_name}")
            generation_task = "overwrite" if task == "overwrite_guard" else task
            batch_size = _evaluation_batch_size(
                generation_task, decisions=decisions, cap=batch_cap
            )
            per_batch = batch_size * (1 if task == "needle" else 8)
            if decisions % per_batch:
                raise ValueError("decisions must contain complete evaluation batches")
            original = _cohort_fingerprints(
                "g15b-eval",
                checkpoint_seed=checkpoint_seed,
                task=task,
                length=length,
                decisions=4096,
                batch_cap=16,
            )
            sealed_r5_fingerprint = r5_cells[cell_name]["batch_fingerprint_sha256"]
            original_digest_reproduced = (
                original["aggregate_sha256"] == sealed_r5_fingerprint
            )
            fresh_first = _cohort_fingerprints(
                "g15br5s-stability",
                checkpoint_seed=checkpoint_seed,
                task=task,
                length=length,
                decisions=decisions,
                batch_cap=batch_cap,
            )
            fresh_second = _cohort_fingerprints(
                "g15br5s-stability",
                checkpoint_seed=checkpoint_seed,
                task=task,
                length=length,
                decisions=decisions,
                batch_cap=batch_cap,
            )
            fresh_generation_deterministic = (
                fresh_first["fingerprints"] == fresh_second["fingerprints"]
                and fresh_first["aggregate_sha256"] == fresh_second["aggregate_sha256"]
            )
            individual_fingerprints_disjoint = original["fingerprint_set"].isdisjoint(
                fresh_first["fingerprint_set"]
            )
            total = 0
            episodes = 0
            batch_index = 0
            fingerprint_digest = hashlib.sha256()
            actual_fingerprints: list[str] = []
            reference_correct = 0
            reference_episode_correct = 0
            reference_nll = 0.0
            reference_finite = True
            source_correct = {source: 0 for source in SOURCES}
            source_episode_correct = {source: 0 for source in SOURCES}
            source_nll = {source: 0.0 for source in SOURCES}
            source_metrics = {
                source: {
                    "state_maximum_absolute_residual": 0.0,
                    "background_relation_maximum_absolute_residual": 0.0,
                    "injection_sum_maximum_absolute_residual": 0.0,
                    "source_assignment_maximum_absolute_residual": 0.0,
                    "logit_maximum_absolute_residual": 0.0,
                    "logit_maximum_scaled_allowance_ratio": 0.0,
                    "query_predictions_equal": True,
                    "finite_logits": True,
                }
                for source in SOURCES
            }
            transition_contract = {
                "batches_checked": 0,
                "left_maximum_absolute_residual": 0.0,
                "right_maximum_absolute_residual": 0.0,
                "injection_maximum_absolute_residual": 0.0,
                "passed": True,
            }
            worst_batch: dict[str, Any] = {
                "normalized_severity": -1.0,
                "batch_index": None,
                "source": None,
                "components": None,
                "fingerprint": None,
            }
            local_batches_checked = 0
            while total < decisions:
                batch_seed = _stable_seed(
                    "g15br5s-stability",
                    checkpoint_seed,
                    task,
                    length,
                    batch_index,
                )
                batch = _cell_batch(
                    task,
                    batch_size=batch_size,
                    length=length,
                    seed=batch_seed,
                    device=device,
                )
                batch_fingerprint = batch.fingerprint()
                actual_fingerprints.append(batch_fingerprint)
                fingerprint_digest.update(batch_fingerprint.encode())
                if not torch.equal(
                    local_write_event_mask(batch.token_ids), batch.write_event_mask
                ):
                    raise RuntimeError("fresh write mask is not locally observable")
                if not torch.equal(
                    r5.local_completed_write_tail_mask(batch.token_ids),
                    r5._audit_tail_mask(batch),
                ):
                    raise RuntimeError("fresh tail mask is not locally observable")
                expected_query = torch.zeros_like(batch.write_event_mask)
                expected_query.scatter_(1, batch.query_positions, True)
                if not torch.equal(
                    r5.r4.local_query_position_mask(batch.token_ids), expected_query
                ):
                    raise RuntimeError("fresh query mask is not locally observable")
                local_batches_checked += 1

                result = r5.source_forwards(model, batch)
                if not result["shared_full_transition_controls"]:
                    raise RuntimeError("source arms did not share full transitions")
                controls = [
                    result["component_controls"][name] for name in r3.CONTROL_NAMES
                ]
                transition = _independent_transition_contract(
                    result["forward_context"][3],
                    controls,
                    result["full_injection"],
                )
                transition_contract["batches_checked"] += 1
                transition_contract["left_maximum_absolute_residual"] = max(
                    float(transition_contract["left_maximum_absolute_residual"]),
                    float(transition["left_maximum_absolute_residual"]),
                )
                transition_contract["right_maximum_absolute_residual"] = max(
                    float(transition_contract["right_maximum_absolute_residual"]),
                    float(transition["right_maximum_absolute_residual"]),
                )
                transition_contract["injection_maximum_absolute_residual"] = max(
                    float(transition_contract["injection_maximum_absolute_residual"]),
                    float(transition["injection_maximum_absolute_residual"]),
                )
                transition_contract["passed"] = bool(
                    transition_contract["passed"]
                ) and bool(transition["passed"])
                monolithic = r3._erase_free_monolithic_forward(model, batch)
                reference_finite = reference_finite and bool(
                    torch.isfinite(monolithic["logits"]).all()
                )
                reference_query_logits = _gather_time(
                    monolithic["logits"], batch.query_positions
                )
                reference_prediction = reference_query_logits.argmax(-1)
                reference_match = reference_prediction.eq(batch.targets)
                reference_correct += int(reference_match.sum())
                reference_episode_correct += int(reference_match.all(dim=1).sum())
                reference_nll += float(
                    F.cross_entropy(
                        reference_query_logits.flatten(0, 1),
                        batch.targets.flatten(),
                        reduction="sum",
                    )
                )
                hidden, outer_gate, mixed, mixer = result["forward_context"]
                for source in SOURCES:
                    reads = result["reads"][(source, False)]
                    candidate_logits = r3._finish_forward(
                        model,
                        hidden,
                        outer_gate,
                        mixed,
                        reads["full_read"],
                        mixer,
                    )
                    comparison = _scaled_tensor_residual(
                        monolithic["logits"], candidate_logits
                    )
                    selected = _gather_time(candidate_logits, batch.query_positions)
                    prediction = selected.argmax(-1)
                    match = prediction.eq(batch.targets)
                    source_correct[source] += int(match.sum())
                    source_episode_correct[source] += int(match.all(dim=1).sum())
                    source_nll[source] += float(
                        F.cross_entropy(
                            selected.flatten(0, 1),
                            batch.targets.flatten(),
                            reduction="sum",
                        )
                    )
                    batch_reference_nll = float(
                        F.cross_entropy(
                            reference_query_logits.flatten(0, 1),
                            batch.targets.flatten(),
                            reduction="sum",
                        )
                    )
                    batch_candidate_nll = float(
                        F.cross_entropy(
                            selected.flatten(0, 1),
                            batch.targets.flatten(),
                            reduction="sum",
                        )
                    )
                    batch_bpq_residual = (
                        abs(batch_candidate_nll - batch_reference_nll)
                        / batch.targets.numel()
                        / math.log(2.0)
                    )
                    batch_categorical_equal = torch.equal(
                        reference_prediction, prediction
                    )
                    batch_finite = bool(torch.isfinite(candidate_logits).all())
                    severity, severity_components = _batch_severity(
                        reads=reads,
                        comparison=comparison,
                        bpq_absolute_residual=batch_bpq_residual,
                        categorical_equal=batch_categorical_equal,
                        finite=batch_finite,
                        transition_passed=bool(transition["passed"]),
                    )
                    if severity > float(worst_batch["normalized_severity"]):
                        worst_batch = {
                            "normalized_severity": severity,
                            "batch_index": batch_index,
                            "source": source,
                            "components": severity_components,
                            "fingerprint": batch_fingerprint,
                        }
                    metrics = source_metrics[source]
                    metrics["state_maximum_absolute_residual"] = max(
                        float(metrics["state_maximum_absolute_residual"]),
                        float(reads["state_residual"]),
                    )
                    metrics["background_relation_maximum_absolute_residual"] = max(
                        float(metrics["background_relation_maximum_absolute_residual"]),
                        float(reads["background_relation_residual"]),
                    )
                    metrics["injection_sum_maximum_absolute_residual"] = max(
                        float(metrics["injection_sum_maximum_absolute_residual"]),
                        float(reads["injection_sum_residual"]),
                    )
                    metrics["source_assignment_maximum_absolute_residual"] = max(
                        float(metrics["source_assignment_maximum_absolute_residual"]),
                        float(reads["assignment_residual"]),
                    )
                    metrics["logit_maximum_absolute_residual"] = max(
                        float(metrics["logit_maximum_absolute_residual"]),
                        float(comparison["maximum_absolute_residual"]),
                    )
                    metrics["logit_maximum_scaled_allowance_ratio"] = max(
                        float(metrics["logit_maximum_scaled_allowance_ratio"]),
                        float(comparison["maximum_scaled_allowance_ratio"]),
                    )
                    metrics["query_predictions_equal"] = bool(
                        metrics["query_predictions_equal"]
                    ) and torch.equal(reference_prediction, prediction)
                    metrics["finite_logits"] = (
                        bool(metrics["finite_logits"]) and batch_finite
                    )

                total += batch.targets.numel()
                episodes += batch.batch_size
                batch_index += 1

            fingerprint = fingerprint_digest.hexdigest()
            actual_generation_matches_plan = (
                actual_fingerprints == fresh_first["fingerprints"]
                and fingerprint == fresh_first["aggregate_sha256"]
            )
            if worst_batch["batch_index"] is None:
                raise RuntimeError("fresh cohort did not retain a worst batch")
            worst_seed = _stable_seed(
                "g15br5s-stability",
                checkpoint_seed,
                task,
                length,
                int(worst_batch["batch_index"]),
            )
            regenerated_worst = _cell_batch(
                task,
                batch_size=batch_size,
                length=length,
                seed=worst_seed,
                device=device,
            )
            worst_batch_regenerated = (
                regenerated_worst.fingerprint() == worst_batch["fingerprint"]
            )
            common_fp64 = _common_fp64_worst_batch_diagnostics(model, regenerated_worst)
            reference_accuracy = reference_correct / total
            reference_episode_accuracy = reference_episode_correct / episodes
            reference_bpq = reference_nll / total / math.log(2.0)
            source_reports: dict[str, Any] = {}
            for source in SOURCES:
                metrics = source_metrics[source]
                accuracy = source_correct[source] / total
                episode_accuracy = source_episode_correct[source] / episodes
                bpq = source_nll[source] / total / math.log(2.0)
                bpq_absolute = abs(bpq - reference_bpq)
                bpq_relative = bpq_absolute / max(1.0, abs(bpq), abs(reference_bpq))
                categorical_equal = (
                    bool(metrics["query_predictions_equal"])
                    and accuracy == reference_accuracy
                    and episode_accuracy == reference_episode_accuracy
                )
                passed = _source_gate(
                    metrics,
                    categorical_equal=categorical_equal,
                    bpq_absolute_residual=bpq_absolute,
                )
                source_reports[source] = {
                    **metrics,
                    "query_accuracy": accuracy,
                    "exact_episode_accuracy": episode_accuracy,
                    "bits_per_query": bpq,
                    "bits_per_query_absolute_residual": bpq_absolute,
                    "bits_per_query_relative_residual": bpq_relative,
                    "categorical_metrics_equal": categorical_equal,
                    "passed": passed,
                }
            cell_passed = (
                original_digest_reproduced
                and fresh_generation_deterministic
                and individual_fingerprints_disjoint
                and actual_generation_matches_plan
                and worst_batch_regenerated
                and local_batches_checked > 0
                and bool(transition_contract["passed"])
                and reference_finite
                and all(report["passed"] for report in source_reports.values())
                and bool(common_fp64["passed"])
            )
            cells[cell_name] = {
                "task": task,
                "length": length,
                "query_decisions": total,
                "fingerprint_audit": {
                    "sealed_r5_aggregate_sha256": sealed_r5_fingerprint,
                    "reconstructed_original_aggregate_sha256": original[
                        "aggregate_sha256"
                    ],
                    "original_aggregate_digest_reproduced": original_digest_reproduced,
                    "original_individual_fingerprint_count": len(
                        original["fingerprints"]
                    ),
                    "fresh_aggregate_sha256": fingerprint,
                    "fresh_individual_fingerprint_count": len(actual_fingerprints),
                    "fresh_generation_deterministic_twice": fresh_generation_deterministic,
                    "fresh_individual_fingerprints_disjoint_from_original": individual_fingerprints_disjoint,
                    "actual_generation_matches_cpu_plan": actual_generation_matches_plan,
                },
                "local_batches_checked": local_batches_checked,
                "reference": {
                    "query_accuracy": reference_accuracy,
                    "exact_episode_accuracy": reference_episode_accuracy,
                    "bits_per_query": reference_bpq,
                    "finite_logits": reference_finite,
                },
                "sources": source_reports,
                "independent_transition_contract": transition_contract,
                "worst_normalized_batch": {
                    **worst_batch,
                    "regenerated_fingerprint_equal": worst_batch_regenerated,
                    "common_fp64_diagnostics": common_fp64,
                },
                "passed": cell_passed,
            }
    return {
        "cells": cells,
        "passed": all(cell["passed"] for cell in cells.values()),
    }


def _adjudicate(
    *,
    mode: Literal["smoke", "quality"],
    sealed_audit: dict[str, Any],
    seed_reports: list[dict[str, Any]],
    source_hash_audit: dict[str, Any],
    clean_start: bool,
    exact_sm75: bool,
) -> dict[str, Any]:
    expected_cells = {
        f"{task}:L{length}" for task in TASKS for length in EVALUATION_LENGTHS
    }
    expected_seeds = set(QUALITY_SEEDS if mode == "quality" else QUALITY_SEEDS[:1])
    expected_decisions = 512 if mode == "quality" else 16
    complete = {
        row["checkpoint_seed"] for row in seed_reports
    } == expected_seeds and all(
        set(row["evaluation"]["cells"]) == expected_cells
        and all(
            cell["query_decisions"] == expected_decisions
            for cell in row["evaluation"]["cells"].values()
        )
        for row in seed_reports
    )
    checkpoint_hashes_passed = bool(seed_reports) and all(
        row["checkpoint_matches_sealed_r5"] for row in seed_reports
    )
    fresh_passed = complete and all(row["evaluation"]["passed"] for row in seed_reports)
    quality_provenance = (
        clean_start
        and exact_sm75
        and source_hash_audit["passed"]
        and checkpoint_hashes_passed
    )
    passed = (
        mode == "quality"
        and sealed_audit["passed"]
        and fresh_passed
        and quality_provenance
    )
    if passed:
        decision = (
            "R5 formal fail remains unchanged; no semantic divergence was detected "
            "above the prospective engineering bounds on the smaller fresh cohort; "
            "support drafting a separate fresh training protocol"
        )
    elif mode == "smoke":
        decision = "smoke only; no numerical ratification or training authorization"
    else:
        decision = (
            "stop retained-checkpoint tail repair; R5 numerical stability did not "
            "ratify prospectively"
        )
    return {
        "sealed_r5_audit": sealed_audit,
        "sealed_r5_source_hash_audit": source_hash_audit,
        "fresh_cohort_complete": complete,
        "checkpoint_hashes_match_sealed_r5": checkpoint_hashes_passed,
        "fresh_stability_cohort_passed": fresh_passed,
        "quality_provenance_passed": quality_provenance,
        "passed": passed,
        "decision": decision,
    }


def run(
    *,
    mode: Literal["smoke", "quality"],
    device: torch.device,
    parent_path: Path,
    r5_path: Path,
    checkpoint_directory: Path,
    commit: str,
    status_at_start: list[str],
) -> dict[str, Any]:
    parent, parent_sha256 = r3._validate_parent(parent_path)
    r5_report, r5_sha256 = _validate_r5(r5_path)
    if r5_report["parent_g15b_sha256"] != parent_sha256:
        raise ValueError("R5 and R5-S do not bind the same G15B parent")
    expected = _expected_identity(parent)
    sealed_audit = _audit_sealed_r5(r5_report)
    source_hash_audit = _audit_r5_source_hashes(r5_report)
    r5_seeds = {row["seed"]: row for row in r5_report["seed_reports"]}
    seeds = QUALITY_SEEDS if mode == "quality" else QUALITY_SEEDS[:1]
    decisions = 512 if mode == "quality" else 16
    batch_cap = 16 if mode == "quality" else 2
    started = time.perf_counter()
    seed_reports = []
    for checkpoint_seed in seeds:
        checkpoint_path = checkpoint_directory / f"g15b_I_seed{checkpoint_seed}.pt"
        model, _ = _load_checkpoint(
            checkpoint_path,
            seed=checkpoint_seed,
            expected=expected[checkpoint_seed],
            device=device,
        )
        _sync(device)
        evaluation_started = time.perf_counter()
        evaluation = evaluate_checkpoint(
            model,
            checkpoint_seed=checkpoint_seed,
            decisions=decisions,
            batch_cap=batch_cap,
            r5_cells=r5_seeds[checkpoint_seed]["evaluation"]["cells"],
        )
        _sync(device)
        seed_reports.append(
            {
                "checkpoint_seed": checkpoint_seed,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "sealed_r5_checkpoint_sha256": r5_seeds[checkpoint_seed][
                    "checkpoint_sha256"
                ],
                "checkpoint_matches_sealed_r5": _sha256(checkpoint_path)
                == r5_seeds[checkpoint_seed]["checkpoint_sha256"],
                "evaluation_wall_seconds": time.perf_counter() - evaluation_started,
                "evaluation": evaluation,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    exact_sm75 = device.type == "cuda" and (
        torch.cuda.get_device_capability(device) == (7, 5)
    )
    adjudication = _adjudicate(
        mode=mode,
        sealed_audit=sealed_audit,
        seed_reports=seed_reports,
        source_hash_audit=source_hash_audit,
        clean_start=not status_at_start,
        exact_sm75=exact_sm75,
    )
    source_paths = (
        Path(__file__),
        PROTOCOL,
        ROOT / "g15br5_causal_tail_source.py",
        ROOT / "g15br3_logical_component.py",
        ROOT / "g15b_interleaved_cohort.py",
        ROOT / "g15b_interleaved_tasks.py",
        ROOT / "spin_dirac_memory.py",
        ROOT / "model.py",
    )
    return {
        "schema_version": 1,
        "experiment": "G15B-R5-S numerical ratification",
        "mode": mode,
        "evidentiary": mode == "quality" and not status_at_start and exact_sm75,
        "git_commit_at_start": commit,
        "git_status_at_start": status_at_start,
        "elapsed_wall_seconds": time.perf_counter() - started,
        "parent_g15b_artifact": str(parent_path),
        "parent_g15b_sha256": parent_sha256,
        "parent_r5_artifact": str(r5_path),
        "parent_r5_sha256": r5_sha256,
        "protocol": {
            "checkpoint_seeds": list(seeds),
            "fresh_batch_namespace": "g15br5s-stability",
            "reconstructed_original_batch_namespace": "g15b-eval",
            "reconstructed_original_decisions_per_cell": 4096,
            "evaluation_decisions_per_cell": decisions,
            "evaluation_batch_cap": batch_cap,
            "tasks": list(TASKS),
            "lengths": list(EVALUATION_LENGTHS),
            "sources": list(SOURCES),
            "absolute_component_bound": ABSOLUTE_COMPONENT_BOUND,
            "scaled_logit_multiplier_fp32_epsilon": SCALED_LOGIT_MULTIPLIER,
            "bpq_absolute_bound": BPQ_ABSOLUTE_BOUND,
            "fp64_algebraic_bound": 1e-10,
            "optimizer_updates": 0,
        },
        "source_files": {
            str(path.relative_to(ROOT)): _sha256(path) for path in source_paths
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "compute_capability": (
                list(torch.cuda.get_device_capability(device))
                if device.type == "cuda"
                else None
            ),
        },
        "seed_reports": seed_reports,
        "adjudication": adjudication,
        "explicit_nonclaims": [
            "R5 remains a formally failed frozen result",
            "no parameter is trained or updated",
            "no R5 performance gate is recomputed",
            "the 64-epsilon allowance is an engineering tolerance, not a theorem",
            "the 5e-6 component ceiling is an engineering tolerance, not a derived forward-error theorem",
            "the smaller fresh cohort cannot localize or replay the original R5 worst state/read batch",
            "a pass is consistent with but cannot prove FP32 reduction order caused the original R5 maxima",
            "a pass supports only drafting a separately frozen training protocol",
            "no generic association, natural-text, scaling, efficiency, Spin, or model promotion follows",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "quality"), required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--parent-artifact", type=Path, default=PARENT_ARTIFACT)
    parser.add_argument("--r5-artifact", type=Path, default=R5_ARTIFACT)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    device = torch.device(args.device)
    commit, status = _git_provenance()
    if args.mode == "quality":
        if status:
            raise RuntimeError("G15B-R5-S quality requires a clean git tree at start")
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("G15B-R5-S quality requires CUDA")
        if torch.cuda.get_device_capability(device) != (7, 5):
            raise RuntimeError("G15B-R5-S quality is frozen to SM75")
    report = run(
        mode=args.mode,
        device=device,
        parent_path=args.parent_artifact,
        r5_path=args.r5_artifact,
        checkpoint_directory=args.checkpoint_directory,
        commit=commit,
        status_at_start=status,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _sha256(args.output),
                "decision": report["adjudication"]["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "ABSOLUTE_COMPONENT_BOUND",
    "BPQ_ABSOLUTE_BOUND",
    "EXPECTED_R5_SHA256",
    "SCALED_LOGIT_MULTIPLIER",
    "_audit_r5_source_hashes",
    "_audit_sealed_r5",
    "_batch_severity",
    "_cohort_fingerprints",
    "_common_fp64_worst_batch_diagnostics",
    "_independent_transition_contract",
    "_scaled_tensor_residual",
    "_source_gate",
    "_validate_r5",
    "evaluate_checkpoint",
]
