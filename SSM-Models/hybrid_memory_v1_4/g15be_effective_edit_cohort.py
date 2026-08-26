"""Frozen G15B-E product versus logit-additive effective-edit cohort."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

from .g15b_interleaved_cohort import (
    _evaluation_batch_size,
    _gather_time,
    _sha256,
    _stable_seed,
)
from .g15b_interleaved_tasks import (
    InterleavedBatch,
    TaskName,
    VOCAB_SIZE,
    generate_interleaved_batch,
)
from .g15br2_collision_erase import overwrite_query_strata
from .g15br3_logical_component import STRATA, generate_component_guard_batch
from .g15bt_transactional_cohort import (
    CohortConfig,
    Phase,
    QUALITY_PHASES,
    SMOKE_PHASES,
    _evaluation_batch,
    generate_boundary_batch,
)
from .model import GatedDeltaState, HybridMemoryConfig, HybridMemoryLM, parameter_count
from .optimizers import HarmonicMuonAdamW
from .transactional_delta import TransactionalDeltaMemory

ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[1]
PROTOCOL = ROOT / "G15BE_EFFECTIVE_EDIT_PROTOCOL_2026-08-26.md"
PHASE0_ARTIFACT = ROOT / "artifacts/g15be_phase0_qualification_sm75_2026-08-26.json"
EXPECTED_PHASE0_SHA256 = (
    "41b49a6de9a74a563c4dc6f3c0571d8d9b0c7fd8fd95405232be22807d88936b"
)
QUALITY_SEEDS = (2481, 2483, 2489)
ARMS = ("P", "A")
Arm = Literal["P", "A", "D"]
TASK_CYCLE: tuple[TaskName, ...] = (
    "mqar",
    "overwrite",
    "overwrite",
    "selective",
    "needle",
)
EVALUATION_TASKS = ("mqar", "overwrite", "overwrite_guard", "selective", "needle")
EVALUATION_LENGTHS = (128, 512, 1024, 2048)
INTERVENTIONS = (
    "learned_reconstruction",
    "memory_zero",
    "valid_event_edit_zero",
    "erase_zero",
    "permuted_write_binding",
    "valid_event_only",
    "non_event_only",
)
FP32_ROUNDOFF_MULTIPLIER = 128.0


def frozen_config(mode: Literal["smoke", "quality"]) -> CohortConfig:
    if mode == "quality":
        return CohortConfig("quality", QUALITY_SEEDS, QUALITY_PHASES, 2048, 8, 512, 4)
    if mode == "smoke":
        return CohortConfig("smoke", (29,), SMOKE_PHASES, 16, 2, 16, 2)
    raise ValueError("mode must be 'smoke' or 'quality'")


def build_model(arm: Arm, seed: int, device: torch.device) -> HybridMemoryLM:
    if arm not in (*ARMS, "D"):
        raise ValueError(f"unknown effective-edit arm {arm!r}")
    torch.manual_seed(seed)
    gate_mode = {
        "P": "product",
        "A": "logit_additive",
        "D": "residual_delta",
    }[arm]
    model = HybridMemoryLM(
        HybridMemoryConfig(
            vocab_size=VOCAB_SIZE,
            model_dim=64,
            layer_plan=("transactional_delta",),
            gated_delta_heads=4,
            gated_delta_key_dim=16,
            gated_delta_value_dim=16,
            gated_delta_normalize_values=False,
            gated_delta_identity_value_path=False,
            gated_delta_identity_output_gate=False,
            gated_delta_tie_query_key=False,
            gated_delta_minimum_retention=0.999,
            gated_delta_initial_retention=0.9995,
            gated_delta_initial_erase_strength=0.10,
            gated_delta_initial_write_strength=0.10,
            transactional_controller_mode="full",
            transactional_initial_commit_strength=0.10,
            transactional_effective_edit_gate_mode=gate_mode,  # type: ignore[arg-type]
            use_local_conv=True,
            conv_kernel=4,
            expansion=2,
            dropout=0.0,
            tie_embeddings=True,
        )
    )
    return model.to(device)


def _diagnostics(output: dict[str, Any]) -> dict[str, Any]:
    rows = output.get("diagnostics")
    if not isinstance(rows, tuple) or len(rows) != 1:
        raise RuntimeError("G15B-E requires exactly one diagnostic block")
    row = rows[0]
    if not isinstance(row, dict) or row.get("kind") != "transactional_delta":
        raise RuntimeError("G15B-E did not receive transactional diagnostics")
    return row


def _tensor(diagnostics: dict[str, Any], name: str) -> torch.Tensor:
    value = diagnostics.get(name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"missing tensor diagnostic {name!r}")
    return value


def _model_state_sha256(model: HybridMemoryLM) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _write_address_scores(
    diagnostics: dict[str, Any], batch: InterleavedBatch
) -> torch.Tensor:
    key = _tensor(diagnostics, "key_vector")
    query = _tensor(diagnostics, "query_vector")
    edit_keys = _gather_time(key, batch.write_positions)
    prototypes = []
    for key_index in range(batch.live_keys.shape[1]):
        selected = batch.live_keys[:, key_index : key_index + 1]
        mask = (batch.write_keys == selected).to(edit_keys.dtype)
        count = mask.sum(dim=1)
        queried = (batch.query_key_indices == key_index).any(dim=1)
        if bool(((count == 0) & queried).any()):
            raise RuntimeError("a queried key has no value-position prototype")
        prototype = torch.einsum("bw,bwhd->bhd", mask, edit_keys)
        prototypes.append(
            F.normalize(prototype / count.clamp_min(1.0)[:, None, None], dim=-1)
        )
    prototype_tensor = torch.stack(prototypes, dim=1)
    query_vectors = _gather_time(query, batch.query_positions)
    return torch.einsum(
        "bqhd,bkhd->bqk", query_vectors, prototype_tensor
    ) / query.shape[2]


def commissioned_losses(
    output: dict[str, Any], batch: InterleavedBatch
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = output.get("logits")
    if not isinstance(logits, torch.Tensor):
        raise TypeError("model output is missing logits")
    diagnostics = _diagnostics(output)
    query_logits = _gather_time(logits, batch.query_positions)
    reverse_logits = _gather_time(logits, batch.write_positions)
    retrieval = F.cross_entropy(query_logits.flatten(0, 1), batch.targets.flatten())
    reverse = F.cross_entropy(reverse_logits.flatten(0, 1), batch.write_keys.flatten())
    address = F.cross_entropy(
        (_write_address_scores(diagnostics, batch) / 0.10).flatten(0, 1),
        batch.query_key_indices.flatten(),
    )
    components = {
        "retrieval": retrieval,
        "reverse_binding": reverse,
        "query_to_value_position_address": address,
    }
    return retrieval + 0.25 * reverse + 0.25 * address, components


def _event_mask(batch: InterleavedBatch, *, dtype: torch.dtype) -> torch.Tensor:
    return batch.write_event_mask[..., None, None].to(dtype)


def _effective_controls(
    mixer: TransactionalDeltaMemory,
    full: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    query, key, value, shared, erase, write, retention = mixer._controls(full, full)
    if mixer.config.effective_edit_gate_mode == "product":
        erase = shared * erase
        write = shared * write
    return query, key, value, erase, write, retention


def _execute_effective_transitions(
    mixer: TransactionalDeltaMemory,
    transition: torch.Tensor,
    injection: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    scan_mode: Literal["recurrent", "parallel"],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute an intervention transition in the mixer's native state layout."""

    if mixer.config.effective_edit_gate_mode != "residual_delta":
        executor = (
            mixer._recurrent_states
            if scan_mode == "recurrent"
            else mixer._parallel_states
        )
        return executor(transition, injection, initial_state)
    batch, heads, key_dim, value_dim = initial_state.shape
    scan_initial = (
        initial_state.permute(0, 1, 3, 2)
        .contiguous()
        .view(batch, heads * value_dim, key_dim, 1)
    )
    executor = (
        mixer._recurrent_states
        if scan_mode == "recurrent"
        else mixer._parallel_states
    )
    scan_states, scan_final = executor(transition, injection, scan_initial)
    scan_states = scan_states.squeeze(-1)
    scan_final = scan_final.squeeze(-1)
    length = scan_states.shape[2]
    states = (
        scan_states.view(batch, heads, value_dim, length, key_dim)
        .permute(0, 1, 3, 4, 2)
        .contiguous()
    )
    final_state = (
        scan_final.view(batch, heads, value_dim, key_dim)
        .permute(0, 1, 3, 2)
        .contiguous()
    )
    return states, final_state


@torch.no_grad()
def effective_edit_intervention_forward(
    model: HybridMemoryLM,
    batch: InterleavedBatch,
    intervention: str,
) -> dict[str, Any]:
    if intervention not in INTERVENTIONS:
        raise ValueError(f"unknown intervention {intervention!r}")
    block = model.blocks[0]
    mixer = block.mixer
    convolution = block.local_conv
    if not isinstance(mixer, TransactionalDeltaMemory) or convolution is None:
        raise TypeError("effective-edit block is incomplete")
    hidden = model.embedding(batch.token_ids)
    value, outer_gate = block.input_projection(block.mixer_norm(hidden)).chunk(2, -1)
    full_pre, _, _ = convolution.full_and_strict_history(value)
    full = F.silu(full_pre)
    query, key, projected_value, erase, write, retention = _effective_controls(
        mixer, full
    )
    event = _event_mask(batch, dtype=erase.dtype)
    if intervention == "valid_event_edit_zero":
        erase = erase * (1.0 - event)
        write = write * (1.0 - event)
    elif intervention == "erase_zero":
        erase = torch.zeros_like(erase)
    elif intervention == "valid_event_only":
        erase = erase * event
        write = write * event
    elif intervention == "non_event_only":
        erase = erase * (1.0 - event)
        write = write * (1.0 - event)
    elif intervention == "permuted_write_binding":
        projected_value = projected_value.clone()
        if mixer.config.effective_edit_gate_mode != "residual_delta":
            write = write.clone()
        for row in range(batch.batch_size):
            positions = batch.write_positions[row]
            if positions.numel() > 1:
                projected_value[row, positions] = projected_value[
                    row, positions.roll(1)
                ]
                if mixer.config.effective_edit_gate_mode != "residual_delta":
                    write[row, positions] = write[row, positions.roll(1)]
    unit_event = torch.ones_like(erase)
    transition, injection = mixer._transitions(
        key,
        projected_value,
        unit_event,
        erase,
        write,
        retention,
        None,
    )
    states, final_state = _execute_effective_transitions(
        mixer,
        transition,
        injection,
        full.new_zeros(full.shape[0], *mixer.config.state_shape),
        scan_mode="parallel",
    )
    read = torch.einsum("bthk,bhtkv->bthv", query, states)
    output_gate = F.silu(mixer.output_gate(full).view_as(read))
    update = mixer.output_projection(
        (mixer.output_norm(read) * output_gate).flatten(start_dim=2)
    )
    if intervention == "memory_zero":
        update = torch.zeros_like(update)
    hidden = hidden + torch.sigmoid(block.residual_scale) * block.dropout(
        update * torch.sigmoid(outer_gate)
    )
    hidden = hidden + block.dropout(block.ffn(block.ffn_norm(hidden)))
    return {
        "logits": model.lm_head(model.final_norm(hidden)),
        "final_state": final_state,
        "effective_erase": erase,
        "effective_write": write,
        "written_payload": write * projected_value,
    }


def _batch_for_update(
    phase: Phase,
    *,
    seed: int,
    global_update: int,
    namespace: str = "g15be-train",
) -> InterleavedBatch:
    task = TASK_CYCLE[global_update % len(TASK_CYCLE)]
    return generate_interleaved_batch(
        task,
        phase.batch_size,
        phase.length,
        phase.live_keys,
        phase.max_writes,
        phase.queries,
        seed=_stable_seed(namespace, seed, global_update, task),
    )


def _evaluation_batch_for_task(
    task: str, *, batch_size: int, length: int, seed: int
) -> InterleavedBatch:
    if task == "overwrite_guard":
        return generate_component_guard_batch(batch_size, length, seed=seed)
    return _evaluation_batch(task, batch_size=batch_size, length=length, seed=seed)


def _gate_accumulator() -> dict[str, float]:
    rows: dict[str, float] = {}
    for prefix in ("event", "non_event"):
        for gate in ("erase", "write"):
            rows[f"{prefix}_{gate}_sum"] = 0.0
            rows[f"{prefix}_{gate}_sum_square"] = 0.0
            rows[f"{prefix}_{gate}_count"] = 0.0
            rows[f"{prefix}_{gate}_minimum"] = math.inf
            rows[f"{prefix}_{gate}_maximum"] = -math.inf
    return rows


def _accumulate_gates(
    rows: dict[str, float], diagnostics: dict[str, Any], batch: InterleavedBatch
) -> None:
    erase = _tensor(diagnostics, "effective_erase_strength")
    write = _tensor(diagnostics, "effective_write_strength")
    event = batch.write_event_mask[..., None, None].expand_as(erase)
    write_event = batch.write_event_mask[..., None, None].expand_as(write)
    for prefix, mask in (("event", event), ("non_event", ~event)):
        selected = erase[mask]
        rows[f"{prefix}_erase_sum"] += float(selected.sum())
        rows[f"{prefix}_erase_sum_square"] += float(selected.square().sum())
        rows[f"{prefix}_erase_count"] += int(selected.numel())
        rows[f"{prefix}_erase_minimum"] = min(
            rows[f"{prefix}_erase_minimum"], float(selected.min())
        )
        rows[f"{prefix}_erase_maximum"] = max(
            rows[f"{prefix}_erase_maximum"], float(selected.max())
        )
    for prefix, mask in (("event", write_event), ("non_event", ~write_event)):
        selected = write[mask]
        rows[f"{prefix}_write_sum"] += float(selected.sum())
        rows[f"{prefix}_write_sum_square"] += float(selected.square().sum())
        rows[f"{prefix}_write_count"] += int(selected.numel())
        rows[f"{prefix}_write_minimum"] = min(
            rows[f"{prefix}_write_minimum"], float(selected.min())
        )
        rows[f"{prefix}_write_maximum"] = max(
            rows[f"{prefix}_write_maximum"], float(selected.max())
        )


def _gate_statistics(rows: dict[str, float]) -> dict[str, dict[str, float]]:
    statistics = {}
    for prefix in ("event", "non_event"):
        for gate in ("erase", "write"):
            name = f"{prefix}_{gate}"
            count = rows[f"{name}_count"]
            mean = rows[f"{name}_sum"] / max(1.0, count)
            variance = max(
                0.0,
                rows[f"{name}_sum_square"] / max(1.0, count) - mean * mean,
            )
            statistics[name] = {
                "count": count,
                "mean": mean,
                "standard_deviation": math.sqrt(variance),
                "minimum": rows[f"{name}_minimum"],
                "maximum": rows[f"{name}_maximum"],
            }
    return statistics


@torch.no_grad()
def _evaluate_cell(
    model: HybridMemoryLM,
    task: str,
    length: int,
    *,
    seed: int,
    decisions: int,
    batch_cap: int,
    namespace: str,
    interventions: bool,
) -> tuple[dict[str, Any], set[str]]:
    generation_task: TaskName = "overwrite" if task == "overwrite_guard" else task  # type: ignore[assignment]
    batch_size = _evaluation_batch_size(
        generation_task, decisions=decisions, cap=batch_cap
    )
    per_batch = batch_size * (1 if task == "needle" else 8)
    if decisions % per_batch:
        raise ValueError("evaluation decisions must contain complete batches")
    correct = 0
    episodes = 0
    nll = 0.0
    address_correct = 0
    stratum_total = {name: 0 for name in STRATA}
    stratum_correct = {name: 0 for name in STRATA}
    intervention_correct = {name: 0 for name in INTERVENTIONS}
    intervention_stratum_correct = {
        name: {stratum: 0 for stratum in STRATA} for name in INTERVENTIONS
    }
    reconstruction_residual = 0.0
    reconstruction_predictions_equal = True
    fingerprints: set[str] = set()
    gate_rows = _gate_accumulator()
    state_norm_maximum = 0.0
    read_maximum = 0.0
    total = 0
    episode_total = 0
    batch_index = 0
    model.eval()
    while total < decisions:
        batch = _evaluation_batch_for_task(
            task,
            batch_size=batch_size,
            length=length,
            seed=_stable_seed(namespace, seed, task, length, batch_index),
        ).to(model.embedding.weight.device)
        fingerprints.add(batch.fingerprint())
        output = model(batch.token_ids, return_diagnostics=True)
        diagnostics = _diagnostics(output)
        query_logits = _gather_time(output["logits"], batch.query_positions)
        matches = query_logits.argmax(-1) == batch.targets
        correct += int(matches.sum())
        episodes += int(matches.all(dim=1).sum())
        nll += float(
            F.cross_entropy(
                query_logits.flatten(0, 1), batch.targets.flatten(), reduction="sum"
            )
        )
        address_correct += int(
            (
                _write_address_scores(diagnostics, batch).argmax(-1)
                == batch.query_key_indices
            ).sum()
        )
        _accumulate_gates(gate_rows, diagnostics, batch)
        state_norm_maximum = max(
            state_norm_maximum, float(_tensor(diagnostics, "state_norm").max())
        )
        read_maximum = max(
            read_maximum, float(_tensor(diagnostics, "read").abs().max())
        )
        strata = (
            overwrite_query_strata(batch)
            if task in ("overwrite", "overwrite_guard")
            else {}
        )
        for name, mask in strata.items():
            stratum_total[name] += int(mask.sum())
            stratum_correct[name] += int((matches & mask).sum())
        if interventions:
            ordinary_predictions = query_logits.argmax(-1)
            for name in INTERVENTIONS:
                intervention = effective_edit_intervention_forward(model, batch, name)
                selected = _gather_time(intervention["logits"], batch.query_positions)
                intervention_matches = selected.argmax(-1) == batch.targets
                intervention_correct[name] += int(intervention_matches.sum())
                for stratum, mask in strata.items():
                    intervention_stratum_correct[name][stratum] += int(
                        (intervention_matches & mask).sum()
                    )
                if name == "learned_reconstruction":
                    reconstruction_residual = max(
                        reconstruction_residual,
                        float((selected - query_logits).abs().max()),
                    )
                    reconstruction_predictions_equal = (
                        reconstruction_predictions_equal
                        and torch.equal(selected.argmax(-1), ordinary_predictions)
                    )
        total += batch.targets.numel()
        episode_total += batch.batch_size
        batch_index += 1
    cell: dict[str, Any] = {
        "task": task,
        "length": length,
        "query_decisions": total,
        "query_accuracy": correct / total,
        "exact_episode_accuracy": episodes / episode_total,
        "bits_per_query": nll / total / math.log(2.0),
        "query_address_top1": address_correct / total,
        "effective_gate_statistics": _gate_statistics(gate_rows),
        "state_norm_maximum": state_norm_maximum,
        "read_maximum_absolute": read_maximum,
        "evaluation_batch_fingerprints": len(fingerprints),
    }
    if task in ("overwrite", "overwrite_guard"):
        cell["query_strata"] = {
            name: {
                "query_decisions": stratum_total[name],
                "accuracy": (
                    stratum_correct[name] / stratum_total[name]
                    if stratum_total[name]
                    else None
                ),
            }
            for name in STRATA
        }
    if interventions:
        cell["interventions"] = {
            name: {
                "query_accuracy": intervention_correct[name] / total,
                "drop_from_learned": (correct - intervention_correct[name]) / total,
                "query_strata": {
                    stratum: {
                        "query_decisions": stratum_total[stratum],
                        "accuracy": (
                            intervention_stratum_correct[name][stratum]
                            / stratum_total[stratum]
                            if stratum_total[stratum]
                            else None
                        ),
                    }
                    for stratum in STRATA
                },
            }
            for name in INTERVENTIONS
        }
        cell["reconstruction_maximum_absolute_logit_residual"] = (
            reconstruction_residual
        )
        cell["reconstruction_query_predictions_equal"] = (
            reconstruction_predictions_equal
        )
    return cell, fingerprints


def _state_residual(
    left: tuple[object, ...], right: tuple[object, ...]
) -> float:
    maximum = 0.0
    for left_state, right_state in zip(left, right, strict=True):
        if not isinstance(left_state, GatedDeltaState) or not isinstance(
            right_state, GatedDeltaState
        ):
            raise TypeError("G15B-E boundary requires GatedDeltaState")
        maximum = max(
            maximum,
            float((left_state.memory - right_state.memory).abs().max()),
            float((left_state.convolution - right_state.convolution).abs().max()),
        )
    return maximum


def _scaled_bound(reference_absmax: float, length: int) -> float:
    return (
        FP32_ROUNDOFF_MULTIPLIER
        * torch.finfo(torch.float32).eps
        * length
        * max(1.0, reference_absmax)
    )


@torch.no_grad()
def _boundary_audit(model: HybridMemoryLM, *, seed: int, batch_size: int) -> tuple[dict[str, Any], str]:
    batch = generate_boundary_batch(batch_size, 128, seed=seed).to(
        model.embedding.weight.device
    )
    full = model(batch.token_ids, delta_scan_mode="recurrent")
    query_logits = _gather_time(full["logits"], batch.query_positions)
    accuracy = float((query_logits.argmax(-1) == batch.targets).float().mean())
    pieces = []
    chunk_state = None
    for start, stop in ((0, 7), (7, 31), (31, 64), (64, 97), (97, 128)):
        output = model(
            batch.token_ids[:, start:stop],
            chunk_state,
            delta_scan_mode="parallel",
        )
        pieces.append(output["logits"])
        chunk_state = output["states"]
    chunk_logits = torch.cat(pieces, dim=1)
    valid_mask = torch.ones_like(batch.token_ids, dtype=torch.bool)
    valid_mask[:, 1::7] &= batch.write_event_mask[:, 1::7]
    masked = model(batch.token_ids, valid_mask=valid_mask, delta_scan_mode="recurrent")
    step_logits = []
    step_state = None
    for position in range(batch.length):
        logits, step_state = model.step(
            batch.token_ids[:, position],
            step_state,
            valid_mask=valid_mask[:, position],
        )
        step_logits.append(logits[:, None])
    step_tensor = torch.cat(step_logits, dim=1)
    if chunk_state is None or step_state is None:
        raise RuntimeError("boundary streaming state missing")
    chunk_logit_residual = float((chunk_logits - full["logits"]).abs().max())
    chunk_state_residual = _state_residual(chunk_state, full["states"])
    step_logit_residual = float((step_tensor - masked["logits"]).abs().max())
    step_state_residual = _state_residual(step_state, masked["states"])
    logit_absmax = float(full["logits"].abs().max())
    state_absmax = max(
        float(state.memory.abs().max())
        for state in full["states"]
        if isinstance(state, GatedDeltaState)
    )
    logit_bound = _scaled_bound(logit_absmax, batch.length)
    state_bound = _scaled_bound(state_absmax, batch.length)
    predictions_equal = torch.equal(
        chunk_logits.argmax(-1), full["logits"].argmax(-1)
    ) and torch.equal(step_tensor.argmax(-1), masked["logits"].argmax(-1))
    return {
        "query_accuracy": accuracy,
        "chunk_logit_maximum_absolute_residual": chunk_logit_residual,
        "chunk_state_maximum_absolute_residual": chunk_state_residual,
        "masked_step_logit_maximum_absolute_residual": step_logit_residual,
        "masked_step_state_maximum_absolute_residual": step_state_residual,
        "logit_bound": logit_bound,
        "state_bound": state_bound,
        "predictions_equal": predictions_equal,
        "passed": chunk_logit_residual <= logit_bound
        and step_logit_residual <= logit_bound
        and chunk_state_residual <= state_bound
        and step_state_residual <= state_bound
        and predictions_equal,
    }, batch.fingerprint()


@torch.no_grad()
def _evaluate(
    model: HybridMemoryLM,
    config: CohortConfig,
    *,
    seed: int,
    arm: Arm,
    namespace: str = "g15be",
) -> tuple[dict[str, Any], set[str]]:
    cells = {}
    fingerprints: set[str] = set()
    for task in EVALUATION_TASKS:
        for length in EVALUATION_LENGTHS:
            cell, hashes = _evaluate_cell(
                model,
                task,
                length,
                seed=seed,
                decisions=config.evaluation_decisions,
                batch_cap=config.evaluation_batch_cap,
                namespace=f"{namespace}-eval",
                interventions=False,
            )
            cells[f"{task}:L{length}"] = cell
            fingerprints.update(hashes)
    intervention_cells = {}
    intervention_hashes: set[str] = set()
    if arm in ("A", "D"):
        for task in ("mqar", "overwrite", "selective"):
            for length in (512, 1024):
                cell, hashes = _evaluate_cell(
                    model,
                    task,
                    length,
                    seed=seed,
                    decisions=config.intervention_decisions,
                    batch_cap=config.intervention_batch_cap,
                    namespace=f"{namespace}-intervention",
                    interventions=True,
                )
                intervention_cells[f"{task}:L{length}"] = cell
                intervention_hashes.update(hashes)
    if fingerprints & intervention_hashes:
        raise RuntimeError("standard and intervention fingerprints overlap")
    boundary, boundary_hash = _boundary_audit(
        model,
        seed=_stable_seed(f"{namespace}-boundary", seed),
        batch_size=min(config.evaluation_batch_cap, 8),
    )
    if boundary_hash in fingerprints or boundary_hash in intervention_hashes:
        raise RuntimeError("boundary fingerprint overlaps evaluation")
    return {
        "cells": cells,
        "intervention_cells": intervention_cells,
        "standard_evaluation_schedule_sha256": hashlib.sha256(
            "\n".join(sorted(fingerprints)).encode()
        ).hexdigest(),
        "intervention_evaluation_schedule_sha256": hashlib.sha256(
            "\n".join(sorted(intervention_hashes)).encode()
        ).hexdigest(),
        "boundary_batch_sha256": boundary_hash,
        "boundary_audit": boundary,
    }, fingerprints | intervention_hashes | {boundary_hash}


def _optimizer_partition(model: HybridMemoryLM) -> tuple[HarmonicMuonAdamW, dict[str, Any]]:
    optimizer = HarmonicMuonAdamW(model, lr=0.003, weight_decay=0.01)
    assigned = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    names = [
        name
        for group in optimizer.partition_report()
        if group["role"] == "memory_controls"
        for name in group["names"]
    ]
    required = ("commit_projection", "erase_projection", "write_projection")
    report = {
        "assigned_tensors": len(assigned),
        "trainable_tensors": len(trainable),
        "memory_control_names": names,
        "groups": optimizer.partition_report(),
        "passed": len(assigned) == len(trainable)
        and len({id(parameter) for parameter in assigned}) == len(trainable)
        and all(any(required_name in name for name in names) for required_name in required),
    }
    return optimizer, report


def run_preflight(device: torch.device) -> dict[str, Any]:
    phase0_sha = _sha256(PHASE0_ARTIFACT)
    phase0 = json.loads(PHASE0_ARTIFACT.read_text(encoding="utf-8"))
    phase0_sources = phase0.get("source_files", {})
    core_source_paths = (
        ROOT / "transactional_delta.py",
        ROOT / "model.py",
    )
    core_source_matches = {
        str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"): (
            phase0_sources.get(
                str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/")
            )
            == _sha256(path)
        )
        for path in core_source_paths
    }
    models = {arm: build_model(arm, 29, device) for arm in ARMS}
    counts = {arm: parameter_count(model) for arm, model in models.items()}
    active = {
        arm: sum(p.numel() for p in model.parameters() if p.requires_grad)
        for arm, model in models.items()
    }
    state_bytes = {
        arm: model.state_capacity_bytes(1, torch.float32)
        for arm, model in models.items()
    }
    hashes = {arm: _model_state_sha256(model) for arm, model in models.items()}
    batch = generate_interleaved_batch(
        "overwrite", 2, 128, 4, 8, 4, seed=_stable_seed("g15be-preflight")
    ).to(device)
    gradients = {}
    reconstruction = {}
    for arm, model in models.items():
        model.train()
        model.zero_grad(set_to_none=True)
        output = model(batch.token_ids, return_diagnostics=True)
        loss, _ = commissioned_losses(output, batch)
        loss.backward()
        gradients[arm] = all(
            p.grad is not None
            and bool(torch.isfinite(p.grad).all())
            and bool(torch.count_nonzero(p.grad))
            for p in model.parameters()
            if p.requires_grad
        )
        model.eval()
        with torch.no_grad():
            ordinary = model(batch.token_ids)["logits"]
            replay = effective_edit_intervention_forward(
                model, batch, "learned_reconstruction"
            )["logits"]
        reconstruction[arm] = {
            "maximum_absolute_logit_residual": float((ordinary - replay).abs().max()),
            "predictions_equal": torch.equal(
                ordinary.argmax(-1), replay.argmax(-1)
            ),
        }
    optimizer, partition = _optimizer_partition(models["A"])
    del optimizer
    checks = {
        "sealed_phase0": phase0_sha == EXPECTED_PHASE0_SHA256
        and phase0.get("adjudication", {}).get("passed") is True,
        "phase0_core_sources_unchanged": all(core_source_matches.values()),
        "matched_parameters": len(set(counts.values())) == 1
        and len(set(active.values())) == 1
        and len(set(state_bytes.values())) == 1
        and len(set(hashes.values())) == 1,
        "finite_nonzero_gradients": all(gradients.values()),
        "optimizer_partition": partition["passed"],
        "learned_reconstruction": all(
            row["maximum_absolute_logit_residual"] <= 5e-4
            and row["predictions_equal"]
            for row in reconstruction.values()
        ),
    }
    return {
        "phase0_sha256": phase0_sha,
        "phase0_core_source_matches": core_source_matches,
        "parameter_counts": counts,
        "active_parameter_counts": active,
        "state_bytes_per_sequence_fp32": state_bytes,
        "initial_parameter_sha256": hashes,
        "gradients_finite_nonzero": gradients,
        "optimizer_partition": partition,
        "learned_reconstruction": reconstruction,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _set_learning_rate(optimizer: HarmonicMuonAdamW, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _train_arm(
    arm: Arm,
    config: CohortConfig,
    *,
    seed: int,
    device: torch.device,
    checkpoint_directory: Path,
    experiment_name: str = "G15B-E effective edit",
    namespace: str = "g15be",
    checkpoint_prefix: str = "g15be",
) -> dict[str, Any]:
    model = build_model(arm, seed, device)
    initial_hash = _model_state_sha256(model)
    optimizer = HarmonicMuonAdamW(
        model, lr=config.phases[0].learning_rate, weight_decay=0.01
    )
    schedule_hash = hashlib.sha256()
    train_fingerprints: set[str] = set()
    loss_samples = []
    global_update = 0
    tokens_seen = 0
    phase_execution = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    for phase_index, phase in enumerate(config.phases):
        _set_learning_rate(optimizer, phase.learning_rate)
        task_counts = {task: 0 for task in set(TASK_CYCLE)}
        phase_tokens = 0
        phase_queries = 0
        for phase_update in range(phase.updates):
            batch = _batch_for_update(
                phase,
                seed=seed,
                global_update=global_update,
                namespace=f"{namespace}-train",
            ).to(device)
            fingerprint = batch.fingerprint()
            train_fingerprints.add(fingerprint)
            schedule_hash.update(fingerprint.encode())
            tokens_seen += batch.token_ids.numel()
            phase_tokens += batch.token_ids.numel()
            phase_queries += batch.targets.numel()
            task_counts[batch.task] += 1
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(batch.token_ids, return_diagnostics=True)
            loss, components = commissioned_losses(output, batch)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    f"nonfinite {experiment_name} loss at update {global_update}"
                )
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError(
                    f"nonfinite {experiment_name} gradient at update {global_update}"
                )
            optimizer.step()
            if phase_update == 0 or phase_update + 1 == phase.updates or global_update % 100 == 0:
                loss_samples.append(
                    {
                        "global_update": global_update + 1,
                        "phase": phase_index + 1,
                        "phase_update": phase_update + 1,
                        "task": batch.task,
                        "learning_rate": phase.learning_rate,
                        "total": float(loss.detach()),
                        "gradient_norm": float(gradient_norm.detach()),
                        "components": {
                            name: float(value.detach())
                            for name, value in components.items()
                        },
                    }
                )
            global_update += 1
        phase_execution.append(
            {
                "phase": phase_index + 1,
                "declared": asdict(phase),
                "task_batches": task_counts,
                "tokens": phase_tokens,
                "query_decisions": phase_queries,
                "optimizer_group_learning_rates": sorted(
                    {float(group["lr"]) for group in optimizer.param_groups}
                ),
            }
        )
    _sync(device)
    training_seconds = time.perf_counter() - started
    evaluation_started = time.perf_counter()
    evaluation, evaluation_hashes = _evaluate(
        model, config, seed=seed, arm=arm, namespace=namespace
    )
    _sync(device)
    evaluation_seconds = time.perf_counter() - evaluation_started
    intersection = train_fingerprints & evaluation_hashes
    if intersection:
        raise RuntimeError("training and evaluation fingerprints overlap")
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_directory / f"{checkpoint_prefix}_{arm.lower()}_seed{seed}.pt"
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(
        {
            "schema_version": 1,
            "experiment": experiment_name,
            "arm": arm,
            "seed": seed,
            "cohort": asdict(config),
            "model_config": asdict(model.config),
            "model_state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "training_schedule_sha256": schedule_hash.hexdigest(),
            "evaluation": evaluation,
        },
        temporary,
    )
    os.replace(temporary, checkpoint)
    report = {
        "arm": arm,
        "seed": seed,
        "parameters": parameter_count(model),
        "active_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "state_bytes_per_sequence_fp32": model.state_capacity_bytes(1, torch.float32),
        "initial_parameter_sha256": initial_hash,
        "training_updates": global_update,
        "training_tokens": tokens_seen,
        "training_schedule_sha256": schedule_hash.hexdigest(),
        "training_batch_fingerprints": len(train_fingerprints),
        "evaluation_batch_fingerprints": len(evaluation_hashes),
        "train_evaluation_hash_intersection": sorted(intersection),
        "loss_samples": loss_samples,
        "phase_execution": phase_execution,
        "training_wall_seconds": training_seconds,
        "mean_synchronized_step_seconds": training_seconds / max(1, global_update),
        "training_tokens_per_second": tokens_seen / max(training_seconds, 1e-12),
        "evaluation_wall_seconds": evaluation_seconds,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "optimizer_partition": optimizer.partition_report(),
        "evaluation": evaluation,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
    }
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def _finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return True


def _adjudicate(config: CohortConfig, reports: list[dict[str, Any]]) -> dict[str, Any]:
    if config.mode == "smoke":
        return {
            "passed": False,
            "eligible_for_promotion": False,
            "decision": "smoke execution completed; run the frozen quality cohort",
        }
    by_arm_seed = {(row["arm"], row["seed"]): row for row in reports}
    expected = {(arm, seed) for arm in ARMS for seed in QUALITY_SEEDS}
    checks: dict[str, bool] = {
        "complete_seed_arm_matrix": set(by_arm_seed) == expected
        and len(reports) == len(expected)
    }
    if not checks["complete_seed_arm_matrix"]:
        return {
            "passed": False,
            "eligible_for_promotion": False,
            "checks": checks,
            "decision": "stop G15B-E after an incomplete Phase-1 cohort",
        }
    expected_cells = {
        f"{task}:L{length}"
        for task in EVALUATION_TASKS
        for length in EVALUATION_LENGTHS
    }
    expected_interventions = {
        f"{task}:L{length}"
        for task in ("mqar", "overwrite", "selective")
        for length in (512, 1024)
    }
    for (arm, seed), row in by_arm_seed.items():
        checkpoint = Path(row["checkpoint"])
        checks[f"provenance:{arm}:{seed}:checkpoint"] = checkpoint.is_file() and _sha256(checkpoint) == row["checkpoint_sha256"]
        checks[f"numerical:{arm}:{seed}:finite"] = _finite_tree(row)
        checks[f"boundary:{arm}:{seed}"] = row["evaluation"]["boundary_audit"]["passed"]
        checks[f"complete:{arm}:{seed}:ordinary"] = set(
            row["evaluation"]["cells"]
        ) == expected_cells
        checks[f"complete:{arm}:{seed}:interventions"] = set(
            row["evaluation"]["intervention_cells"]
        ) == (expected_interventions if arm == "A" else set())
        for cell_name, cell in row["evaluation"]["cells"].items():
            checks[f"decisions:{arm}:{seed}:{cell_name}"] = (
                cell["query_decisions"] == config.evaluation_decisions
            )
            if cell["task"] in ("overwrite", "overwrite_guard"):
                checks[f"strata_partition:{arm}:{seed}:{cell_name}"] = sum(
                    stratum["query_decisions"]
                    for stratum in cell["query_strata"].values()
                ) == cell["query_decisions"]
        for cell_name, cell in row["evaluation"]["intervention_cells"].items():
            checks[f"intervention_decisions:{arm}:{seed}:{cell_name}"] = (
                cell["query_decisions"] == config.intervention_decisions
            )
            if cell["task"] == "overwrite":
                checks[f"intervention_strata_partition:{arm}:{seed}:{cell_name}"] = sum(
                    stratum["query_decisions"]
                    for stratum in cell["query_strata"].values()
                ) == cell["query_decisions"]
    for seed in QUALITY_SEEDS:
        paired = [by_arm_seed[(arm, seed)] for arm in ARMS]
        for name, getter in (
            ("initial", lambda row: row["initial_parameter_sha256"]),
            ("training", lambda row: row["training_schedule_sha256"]),
            ("evaluation", lambda row: row["evaluation"]["standard_evaluation_schedule_sha256"]),
            ("boundary", lambda row: row["evaluation"]["boundary_batch_sha256"]),
            ("parameters", lambda row: (row["parameters"], row["active_parameters"], row["state_bytes_per_sequence_fp32"])),
        ):
            checks[f"paired:{seed}:{name}"] = len({getter(row) for row in paired}) == 1
        checks[f"paired:{seed}:budget"] = all(
            row["training_updates"] == 3400 and row["training_tokens"] == 13_926_400
            for row in paired
        )
        checks[f"paired:{seed}:fingerprints"] = all(
            not row["train_evaluation_hash_intersection"] for row in paired
        )
    for seed in QUALITY_SEEDS:
        cells = by_arm_seed[("A", seed)]["evaluation"]["cells"]
        for length in (128, 512, 1024):
            overwrite = cells[f"overwrite:L{length}"]
            checks[f"A:{seed}:overwrite:L{length}"] = overwrite["query_accuracy"] >= 0.93
            post_same = overwrite["query_strata"]["after_same_key_overwrite"]["accuracy"]
            checks[f"A:{seed}:post_same:L{length}"] = post_same is not None and post_same >= 0.92
            for task in ("mqar", "selective"):
                checks[f"A:{seed}:{task}:L{length}"] = cells[f"{task}:L{length}"]["query_accuracy"] >= 0.98
        for length in EVALUATION_LENGTHS:
            checks[f"A:{seed}:needle:L{length}"] = cells[f"needle:L{length}"]["query_accuracy"] == 1.0
            guard = cells[f"overwrite_guard:L{length}"]
            checks[f"A:{seed}:guard:L{length}"] = guard["query_accuracy"] >= 0.99 and all(
                guard["query_strata"][name]["query_decisions"] > 0
                and guard["query_strata"][name]["accuracy"] is not None
                and guard["query_strata"][name]["accuracy"] >= 0.99
                for name in STRATA
            )
        interventions = by_arm_seed[("A", seed)]["evaluation"]["intervention_cells"]
        for length in (512, 1024):
            for task in ("mqar", "overwrite", "selective"):
                cell = interventions[f"{task}:L{length}"]
                for name in ("memory_zero", "valid_event_edit_zero", "permuted_write_binding"):
                    checks[f"A:{seed}:{task}:{name}:L{length}"] = cell["interventions"][name]["drop_from_learned"] >= 0.50
                checks[f"A:{seed}:{task}:event_only:L{length}"] = abs(cell["interventions"]["valid_event_only"]["drop_from_learned"]) <= 0.02
                checks[f"A:{seed}:{task}:non_event_only:L{length}"] = cell["interventions"]["non_event_only"]["query_accuracy"] <= 0.50
                checks[f"A:{seed}:{task}:reconstruction:L{length}"] = cell["reconstruction_maximum_absolute_logit_residual"] <= 5e-4 and cell["reconstruction_query_predictions_equal"]
            overwrite = interventions[f"overwrite:L{length}"]
            learned_post = overwrite["query_strata"]["after_same_key_overwrite"]["accuracy"]
            erase_post = overwrite["interventions"]["erase_zero"]["query_strata"]["after_same_key_overwrite"]["accuracy"]
            checks[f"A:{seed}:erase_post:L{length}"] = learned_post is not None and erase_post is not None and learned_post - erase_post >= 0.20
            mqar = interventions[f"mqar:L{length}"]
            checks[f"A:{seed}:erase_mqar:L{length}"] = abs(
                mqar["interventions"]["erase_zero"]["drop_from_learned"]
            ) <= 0.02
    for length in (128, 512, 1024):
        means = {
            arm: sum(
                by_arm_seed[(arm, seed)]["evaluation"]["cells"][f"overwrite:L{length}"]["query_accuracy"]
                for seed in QUALITY_SEEDS
            ) / len(QUALITY_SEEDS)
            for arm in ARMS
        }
        checks[f"A:mean_minus_P:L{length}"] = means["A"] - means["P"] >= 0.02
        for arm in ARMS:
            checks[f"{arm}:worst_seed:L{length}"] = all(
                means[arm] - by_arm_seed[(arm, seed)]["evaluation"]["cells"][f"overwrite:L{length}"]["query_accuracy"] <= 0.03
                for seed in QUALITY_SEEDS
            )
    product_specific_checks: dict[str, bool] = {}
    for seed in QUALITY_SEEDS:
        cells = by_arm_seed[("P", seed)]["evaluation"]["cells"]
        for length in (128, 512, 1024):
            overwrite = cells[f"overwrite:L{length}"]
            post_same = overwrite["query_strata"]["after_same_key_overwrite"]["accuracy"]
            product_specific_checks[f"P:{seed}:overwrite:L{length}"] = (
                overwrite["query_accuracy"] >= 0.93
            )
            product_specific_checks[f"P:{seed}:post_same:L{length}"] = (
                post_same is not None and post_same >= 0.92
            )
            product_specific_checks[f"P:{seed}:mqar_selective:L{length}"] = all(
                cells[f"{task}:L{length}"]["query_accuracy"] >= 0.98
                for task in ("mqar", "selective")
            )
        product_specific_checks[f"P:{seed}:needle_guard"] = all(
            cells[f"needle:L{length}"]["query_accuracy"] == 1.0
            and cells[f"overwrite_guard:L{length}"]["query_accuracy"] >= 0.99
            and all(
                cells[f"overwrite_guard:L{length}"]["query_strata"][name][
                    "query_decisions"
                ]
                > 0
                and cells[f"overwrite_guard:L{length}"]["query_strata"][name][
                    "accuracy"
                ]
                is not None
                and cells[f"overwrite_guard:L{length}"]["query_strata"][name][
                    "accuracy"
                ]
                >= 0.99
                for name in STRATA
            )
            for length in EVALUATION_LENGTHS
        )
    shared_integrity_checks = {
        name: value
        for name, value in checks.items()
        if not name.startswith(("A:", "P:"))
    }
    product_specific_checks.update(
        {
            name: value
            for name, value in checks.items()
            if name.startswith("P:")
        }
    )
    additive_specific_checks = {
        name: value
        for name, value in checks.items()
        if name.startswith("A:") and not name.startswith("A:mean_minus_P:")
    }
    comparative_checks = {
        name: value
        for name, value in checks.items()
        if name.startswith("A:mean_minus_P:")
    }
    product_checks = {**shared_integrity_checks, **product_specific_checks}
    additive_checks = {**shared_integrity_checks, **additive_specific_checks}
    p_passed = all(product_checks.values())
    a_absolute_passed = all(additive_checks.values())
    passed = a_absolute_passed and all(comparative_checks.values())
    if passed and p_passed:
        decision = "both pass; run a fresh compiled-efficiency comparison"
    elif passed:
        decision = "A passes and P fails; freeze a separate natural-text identity protocol"
    elif a_absolute_passed and p_passed:
        decision = "both pass absolute gates, but A lacks the frozen comparative margin"
    elif a_absolute_passed:
        decision = "A passes absolute gates but lacks the frozen comparative margin; no promotion"
    elif p_passed:
        decision = "P passes while A fails; reject the additive parameterization"
    else:
        decision = "both effective-edit arms fail; test a frozen residual-delta write law"
    return {
        "passed": passed,
        "eligible_for_promotion": passed,
        "product_absolute_quality_passed": p_passed,
        "product_diagnostic_checks": product_checks,
        "additive_absolute_and_causal_passed": a_absolute_passed,
        "additive_absolute_and_causal_checks": additive_checks,
        "comparative_checks": comparative_checks,
        "shared_integrity_checks": shared_integrity_checks,
        "checks": checks,
        "decision": decision,
    }


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        text=True,
        encoding="utf-8",
    ).strip()


def _enforce_execution_eligibility(
    adjudication: dict[str, Any], status: list[str]
) -> dict[str, Any]:
    if not status:
        return adjudication
    return {
        **adjudication,
        "passed": False,
        "eligible_for_promotion": False,
        "decision": "non-evidentiary dirty execution; rerun from a clean commit",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "quality"), required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-commit")
    parser.add_argument("--allow-dirty", action="store_true")
    arguments = parser.parse_args()
    config = frozen_config(arguments.mode)
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain").splitlines()
    if arguments.expected_commit and commit != arguments.expected_commit:
        raise RuntimeError("HEAD does not match --expected-commit")
    if status and not arguments.allow_dirty:
        raise RuntimeError("G15B-E cohort requires a clean checkout")
    device = torch.device(arguments.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("G15B-E requires the declared CUDA device")
    if torch.cuda.get_device_capability(device) != (7, 5):
        raise RuntimeError("G15B-E requires exact compute capability (7, 5)")
    release = platform.release().lower()
    if platform.system() != "Linux" or not (
        "microsoft" in release or "wsl" in release
    ):
        raise RuntimeError("G15B-E evidentiary execution requires WSL2 Linux")
    preflight = run_preflight(device)
    if not preflight["passed"]:
        raise RuntimeError("G15B-E preflight failed")
    reports = []
    for seed in config.seeds:
        for arm in ARMS:
            reports.append(
                _train_arm(
                    arm,  # type: ignore[arg-type]
                    config,
                    seed=seed,
                    device=device,
                    checkpoint_directory=arguments.checkpoint_dir,
                )
            )
    adjudication = _enforce_execution_eligibility(
        _adjudicate(config, reports), status
    )
    report = {
        "schema_version": 1,
        "experiment": "G15B-E effective edit",
        "mode": config.mode,
        "evidentiary": not status,
        "git_commit_at_start": commit,
        "git_status_at_start": status,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        },
        "protocol": asdict(config),
        "source_files": {
            str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"): _sha256(path)
            for path in (
                PROTOCOL,
                Path(__file__).resolve(),
                ROOT / "transactional_delta.py",
                ROOT / "model.py",
                ROOT / "optimizers.py",
                ROOT / "g15b_interleaved_tasks.py",
                ROOT / "g15b_interleaved_cohort.py",
                ROOT / "g15bt_transactional_cohort.py",
                ROOT / "g15br2_collision_erase.py",
                ROOT / "g15br3_logical_component.py",
            )
        },
        "phase0_artifact_sha256": _sha256(PHASE0_ARTIFACT),
        "preflight": preflight,
        "seed_arm_reports": reports,
        "adjudication": adjudication,
    }
    arguments.artifact.parent.mkdir(parents=True, exist_ok=True)
    arguments.artifact.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(adjudication, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ARMS",
    "QUALITY_SEEDS",
    "build_model",
    "commissioned_losses",
    "effective_edit_intervention_forward",
    "frozen_config",
    "run_preflight",
]
