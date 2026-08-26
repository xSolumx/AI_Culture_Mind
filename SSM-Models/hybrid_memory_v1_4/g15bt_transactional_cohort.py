"""Frozen G15B-T matched transactional-controller training cohort."""

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

from .g15b_interleaved_cohort import (
    _balanced_bce,
    _classification_report,
    _evaluation_batch_size,
    _gather_time,
    _merge_counts,
    _sha256,
    _stable_seed,
)
from .g15b_interleaved_tasks import (
    PAYLOAD_COUNT,
    PAYLOAD_START,
    QUERY_TOKEN,
    ROLE_FILLER,
    ROLE_QUERY_KEY,
    ROLE_QUERY_MARKER,
    ROLE_WRITE_KEY,
    ROLE_WRITE_MARKER,
    ROLE_WRITE_VALUE,
    VOCAB_SIZE,
    WRITE_TOKEN,
    InterleavedBatch,
    TaskName,
    generate_interleaved_batch,
)
from .g15br2_collision_erase import overwrite_query_strata
from .g15br3_logical_component import STRATA, generate_component_guard_batch
from .model import HybridMemoryConfig, HybridMemoryLM, parameter_count
from .optimizers import HarmonicMuonAdamW
from .transactional_delta import TransactionalDeltaMemory

ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[1]
PROTOCOL = ROOT / "G15BT_TRANSACTIONAL_DELTA_PROTOCOL_2026-08-26.md"
PHASE0_ARTIFACT = ROOT / "artifacts/g15bt_phase0_qualification_sm75_2026-08-26.json"
EXPECTED_PHASE0_SHA256 = (
    "0b4683ad3b66f7dc010e03737550873cd695d46ade897f21e094d38f4ece2438"
)
QUALITY_SEEDS = (2381, 2383, 2389)
ARMS = ("F", "T", "T-AUX")
Arm = Literal["F", "T", "T-AUX"]
TASK_CYCLE: tuple[TaskName, ...] = (
    "mqar",
    "overwrite",
    "overwrite",
    "selective",
    "needle",
)
EVALUATION_TASKS = ("mqar", "overwrite", "overwrite_guard", "selective", "needle")
EVALUATION_LENGTHS = (128, 512, 1024, 2048)
NEEDLE_DISTANCES = {128: 64, 512: 448, 1024: 960, 2048: 1984}
INTERVENTIONS = (
    "learned_reconstruction",
    "commit_zero",
    "memory_zero",
    "permuted_history",
    "commit_shift_minus_one",
    "commit_shift_plus_one",
    "erase_zero",
    "bias_only_history",
)


@dataclass(frozen=True)
class Phase:
    length: int
    live_keys: int
    max_writes: int
    queries: int
    batch_size: int
    updates: int
    learning_rate: float


QUALITY_PHASES = (
    Phase(128, 4, 8, 4, 32, 1000, 0.003),
    Phase(256, 8, 16, 8, 16, 1200, 0.003),
    Phase(512, 8, 24, 8, 8, 800, 0.001),
    Phase(1024, 8, 24, 8, 4, 400, 0.001),
)
SMOKE_PHASES = tuple(
    Phase(
        phase.length,
        phase.live_keys,
        phase.max_writes,
        phase.queries,
        min(2, phase.batch_size),
        1,
        phase.learning_rate,
    )
    for phase in QUALITY_PHASES
)


@dataclass(frozen=True)
class CohortConfig:
    mode: Literal["smoke", "quality"]
    seeds: tuple[int, ...]
    phases: tuple[Phase, ...]
    evaluation_decisions: int
    evaluation_batch_cap: int
    intervention_decisions: int
    intervention_batch_cap: int


def frozen_config(mode: Literal["smoke", "quality"]) -> CohortConfig:
    if mode == "quality":
        return CohortConfig("quality", QUALITY_SEEDS, QUALITY_PHASES, 2048, 8, 512, 4)
    if mode == "smoke":
        return CohortConfig("smoke", (23,), SMOKE_PHASES, 16, 2, 16, 2)
    raise ValueError("mode must be 'smoke' or 'quality'")


def build_model(arm: Arm, seed: int, device: torch.device) -> HybridMemoryLM:
    if arm not in ARMS:
        raise ValueError(f"unknown G15B-T arm {arm!r}")
    torch.manual_seed(seed)
    controller_mode = "full" if arm == "F" else "history"
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
            transactional_controller_mode=controller_mode,  # type: ignore[arg-type]
            transactional_initial_commit_strength=0.10,
            use_local_conv=True,
            conv_kernel=4,
            expansion=2,
            dropout=0.0,
            tie_embeddings=True,
        )
    )
    return model.to(device)


def _transactional_diagnostics(output: dict[str, Any]) -> dict[str, Any]:
    rows = output.get("diagnostics")
    if not isinstance(rows, tuple) or len(rows) != 1:
        raise RuntimeError("G15B-T requires exactly one diagnostic block")
    row = rows[0]
    if not isinstance(row, dict) or row.get("kind") != "transactional_delta":
        raise RuntimeError("G15B-T did not receive transactional diagnostics")
    return row


def _tensor(diagnostics: dict[str, Any], name: str) -> torch.Tensor:
    value = diagnostics.get(name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"missing tensor diagnostic {name!r}")
    return value


def commit_positions(batch: InterleavedBatch) -> torch.Tensor:
    return batch.write_positions + 1


def valid_commit_mask(batch: InterleavedBatch) -> torch.Tensor:
    return commit_positions(batch) < batch.length


def commit_event_mask(batch: InterleavedBatch) -> torch.Tensor:
    mask = torch.zeros_like(batch.write_event_mask)
    positions = commit_positions(batch)
    valid = valid_commit_mask(batch)
    rows = torch.arange(batch.batch_size, device=mask.device)[:, None].expand_as(
        positions
    )
    mask[rows[valid], positions[valid]] = True
    if int(mask.sum()) != int(valid.sum()):
        raise RuntimeError("commit target does not match the valid-tail count")
    return mask


def _address_scores(
    diagnostics: dict[str, Any], batch: InterleavedBatch
) -> torch.Tensor:
    key = _tensor(diagnostics, "key_vector")
    query = _tensor(diagnostics, "query_vector")
    positions = commit_positions(batch)
    valid = valid_commit_mask(batch)
    edit_keys = _gather_time(key, positions.clamp_max(batch.length - 1))
    prototypes = []
    for key_index in range(batch.live_keys.shape[1]):
        selected = batch.live_keys[:, key_index : key_index + 1]
        mask = ((batch.write_keys == selected) & valid).to(edit_keys.dtype)
        count = mask.sum(dim=1)
        queried = (batch.query_key_indices == key_index).any(dim=1)
        if bool(((count == 0) & queried).any()):
            raise RuntimeError("a queried key has no valid commit prototype")
        prototype = torch.einsum("bw,bwhd->bhd", mask, edit_keys)
        prototypes.append(
            F.normalize(prototype / count.clamp_min(1.0)[:, None, None], dim=-1)
        )
    prototype_tensor = torch.stack(prototypes, dim=1)
    query_vectors = _gather_time(query, batch.query_positions)
    return (
        torch.einsum("bqhd,bkhd->bqk", query_vectors, prototype_tensor) / query.shape[2]
    )


def commissioned_losses(
    output: dict[str, Any], batch: InterleavedBatch, *, arm: Arm
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = output.get("logits")
    if not isinstance(logits, torch.Tensor):
        raise TypeError("model output is missing logits")
    diagnostics = _transactional_diagnostics(output)
    query_logits = _gather_time(logits, batch.query_positions)
    reverse_logits = _gather_time(logits, batch.write_positions)
    retrieval = F.cross_entropy(query_logits.flatten(0, 1), batch.targets.flatten())
    reverse = F.cross_entropy(reverse_logits.flatten(0, 1), batch.write_keys.flatten())
    address = F.cross_entropy(
        (_address_scores(diagnostics, batch) / 0.10).flatten(0, 1),
        batch.query_key_indices.flatten(),
    )
    components = {
        "retrieval": retrieval,
        "reverse_binding": reverse,
        "query_to_commit_address": address,
    }
    total = retrieval + 0.25 * reverse + 0.25 * address
    if arm == "T-AUX":
        commit = _tensor(diagnostics, "commit_strength")
        commit_loss = _balanced_bce(commit, commit_event_mask(batch)[..., None, None])
        components["balanced_commit"] = commit_loss
        total = total + 0.25 * commit_loss
    return total, components


def _binary_counts(probability: torch.Tensor, target: torch.Tensor) -> dict[str, int]:
    prediction = probability >= 0.5
    target = target.expand_as(prediction)
    return {
        "tp": int((prediction & target).sum()),
        "fp": int((prediction & ~target).sum()),
        "fn": int((~prediction & target).sum()),
        "tn": int((~prediction & ~target).sum()),
    }


def _set_learning_rate(optimizer: HarmonicMuonAdamW, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _batch_for_update(
    phase: Phase, *, seed: int, global_update: int
) -> InterleavedBatch:
    task = TASK_CYCLE[global_update % len(TASK_CYCLE)]
    return generate_interleaved_batch(
        task,
        phase.batch_size,
        phase.length,
        phase.live_keys,
        phase.max_writes,
        phase.queries,
        seed=_stable_seed("g15bt-train", seed, global_update, task),
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _model_state_sha256(model: HybridMemoryLM) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def transactional_intervention_forward(
    model: HybridMemoryLM,
    token_ids: torch.Tensor,
    intervention: str,
) -> dict[str, Any]:
    if intervention not in INTERVENTIONS:
        raise ValueError(f"unknown intervention {intervention!r}")
    if model.layer_plan != ("transactional_delta",):
        raise ValueError("intervention forward requires one transactional block")
    block = model.blocks[0]
    mixer = block.mixer
    convolution = block.local_conv
    if not isinstance(mixer, TransactionalDeltaMemory) or convolution is None:
        raise TypeError("transactional block is incomplete")

    hidden = model.embedding(token_ids)
    value, outer_gate = block.input_projection(block.mixer_norm(hidden)).chunk(2, -1)
    full_pre, history_pre, _ = convolution.full_and_strict_history(value)
    full = F.silu(full_pre)
    history = F.silu(history_pre)
    if intervention == "permuted_history":
        history = history.roll(1, dims=0)
    elif intervention == "bias_only_history":
        bias = convolution.conv.bias
        if bias is None:
            raise RuntimeError("bias-only intervention requires convolution bias")
        history = F.silu(bias.view(1, 1, -1).expand_as(history))

    query, key, projected_value, commit, erase, write, retention = mixer._controls(
        full, history
    )
    if intervention == "commit_zero":
        commit = torch.zeros_like(commit)
    elif intervention == "erase_zero":
        erase = torch.zeros_like(erase)
    elif intervention in ("commit_shift_minus_one", "commit_shift_plus_one"):
        shifted = torch.zeros_like(commit)
        if intervention == "commit_shift_minus_one":
            shifted[:, :-1] = commit[:, 1:]
        else:
            shifted[:, 1:] = commit[:, :-1]
        commit = shifted

    transition, injection = mixer._transitions(
        key, projected_value, commit, erase, write, retention, None
    )
    states, final_state = mixer._parallel_states(
        transition,
        injection,
        full.new_zeros(full.shape[0], *mixer.config.state_shape),
    )
    read = torch.einsum("bthk,bhtkv->bthv", query, states)
    output_gate = mixer.output_gate(full).view_as(read)
    output_gate = F.silu(output_gate)
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
        "controls": {
            "query": query,
            "key": key,
            "value": projected_value,
            "commit": commit,
            "erase": erase,
            "write": write,
            "retention": retention,
        },
    }


def _evaluation_batch(
    task: str,
    *,
    batch_size: int,
    length: int,
    seed: int,
) -> InterleavedBatch:
    if task == "overwrite_guard":
        return generate_component_guard_batch(batch_size, length, seed=seed)
    return generate_interleaved_batch(
        task,  # type: ignore[arg-type]
        batch_size,
        length,
        8,
        24,
        8,
        seed=seed,
        needle_distance=NEEDLE_DISTANCES[length] if task == "needle" else None,
    )


def generate_boundary_batch(
    batch_size: int, length: int, *, seed: int
) -> InterleavedBatch:
    """Construct zero-gap write-query and write-write-query transactions."""

    if batch_size < 1 or length < 64:
        raise ValueError("boundary batch requires batch_size >= 1 and length >= 64")
    first = 4
    consecutive = 20
    immediate = length // 2
    delayed = immediate + 16
    if delayed + 2 >= length:
        raise RuntimeError("boundary layout exceeds the sequence")
    rows_tokens = []
    rows_roles = []
    live_keys = []
    targets = []
    query_positions = []
    query_keys = []
    query_indices = []
    write_positions = []
    write_keys = []
    write_values = []
    needle_distances = []
    for row in range(batch_size):
        generator = torch.Generator(device="cpu").manual_seed(
            _stable_seed("g15bt-boundary-row", seed, row)
        )
        permutation = torch.randperm(PAYLOAD_COUNT, generator=generator).tolist()
        key_a, key_b, value_a, old_b, new_b, newest_b = [
            PAYLOAD_START + value for value in permutation[:6]
        ]
        tokens = torch.randint(
            PAYLOAD_START,
            VOCAB_SIZE,
            (length,),
            generator=generator,
            dtype=torch.long,
        )
        roles = torch.full((length,), ROLE_FILLER, dtype=torch.long)
        events = (
            (first, WRITE_TOKEN, key_a, value_a),
            (consecutive, WRITE_TOKEN, key_b, old_b),
            (consecutive + 3, WRITE_TOKEN, key_b, new_b),
            (immediate, WRITE_TOKEN, key_b, newest_b),
        )
        for start, marker, key, value in events:
            tokens[start : start + 3] = torch.tensor((marker, key, value))
            roles[start : start + 3] = torch.tensor(
                (ROLE_WRITE_MARKER, ROLE_WRITE_KEY, ROLE_WRITE_VALUE)
            )
        queries = (
            (consecutive + 20, key_b, new_b),
            (immediate + 3, key_a, value_a),
            (delayed, key_b, newest_b),
        )
        for start, key, _ in queries:
            tokens[start : start + 2] = torch.tensor((QUERY_TOKEN, key))
            roles[start : start + 2] = torch.tensor((ROLE_QUERY_MARKER, ROLE_QUERY_KEY))
        row_writes = [first + 2, consecutive + 2, consecutive + 5, immediate + 2]
        row_queries = [consecutive + 21, immediate + 4, delayed + 1]
        rows_tokens.append(tokens)
        rows_roles.append(roles)
        live_keys.append([key_a, key_b])
        targets.append([new_b, value_a, newest_b])
        query_positions.append(row_queries)
        query_keys.append([key_b, key_a, key_b])
        query_indices.append([1, 0, 1])
        write_positions.append(row_writes)
        write_keys.append([key_a, key_b, key_b, key_b])
        write_values.append([value_a, old_b, new_b, newest_b])
        needle_distances.append(
            [
                row_queries[0] - row_writes[2],
                row_queries[1] - row_writes[0],
                row_queries[2] - row_writes[3],
            ]
        )
    write_position_tensor = torch.tensor(write_positions, dtype=torch.long)
    overwrite = torch.tensor([[False, False, True, True]] * batch_size)
    write_mask = torch.zeros(batch_size, length, dtype=torch.bool)
    write_mask.scatter_(1, write_position_tensor, True)
    erase_mask = torch.zeros_like(write_mask)
    erase_mask.scatter_(1, write_position_tensor, overwrite)
    return InterleavedBatch(
        task="overwrite",
        token_ids=torch.stack(rows_tokens),
        targets=torch.tensor(targets),
        query_positions=torch.tensor(query_positions),
        query_keys=torch.tensor(query_keys),
        query_key_indices=torch.tensor(query_indices),
        live_keys=torch.tensor(live_keys),
        write_positions=write_position_tensor,
        write_keys=torch.tensor(write_keys),
        write_values=torch.tensor(write_values),
        overwrite_mask=overwrite,
        write_event_mask=write_mask,
        erase_event_mask=erase_mask,
        roles=torch.stack(rows_roles),
        needle_distances=torch.tensor(needle_distances),
        seed=seed,
    )


def _history_views(
    model: HybridMemoryLM, token_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, TransactionalDeltaMemory]:
    block = model.blocks[0]
    if block.local_conv is None or not isinstance(
        block.mixer, TransactionalDeltaMemory
    ):
        raise TypeError("history audit requires one transactional block")
    hidden = model.embedding(token_ids)
    value, _ = block.input_projection(block.mixer_norm(hidden)).chunk(2, -1)
    full, history, _ = block.local_conv.full_and_strict_history(value)
    return F.silu(full), F.silu(history), block.mixer


@torch.no_grad()
def _trained_causal_audit(
    model: HybridMemoryLM, batch: InterleavedBatch
) -> dict[str, Any]:
    tokens = batch.token_ids
    position = int(commit_positions(batch)[0, 0])
    current_changed = tokens.clone()
    current_changed[:, position] = (
        (current_changed[:, position] - PAYLOAD_START + 17) % PAYLOAD_COUNT
    ) + PAYLOAD_START
    history_changed = tokens.clone()
    history_position = int(batch.write_positions[0, 0])
    history_changed[:, history_position] = (
        (history_changed[:, history_position] - PAYLOAD_START + 19) % PAYLOAD_COUNT
    ) + PAYLOAD_START
    full, history, mixer = _history_views(model, tokens)
    current_full, current_history, _ = _history_views(model, current_changed)
    _, prior_history, _ = _history_views(model, history_changed)
    controls = mixer._controls(
        full[:, position : position + 1], history[:, position : position + 1]
    )
    current_controls = mixer._controls(
        current_full[:, position : position + 1],
        current_history[:, position : position + 1],
    )
    prior_controls = mixer._controls(
        full[:, position : position + 1],
        prior_history[:, position : position + 1],
    )
    edit_equal = all(
        torch.equal(left, right)
        for left, right in zip(controls[1:], current_controls[1:], strict=True)
    )
    prior_effect = max(
        float((left - right).abs().max())
        for left, right in zip(controls[1:], prior_controls[1:], strict=True)
    )
    transition, injection = mixer._transitions(*controls[1:], None)
    changed_transition, changed_injection = mixer._transitions(
        *current_controls[1:], None
    )
    return {
        "position": position,
        "current_history_maximum_absolute_residual": float(
            (history[:, position] - current_history[:, position]).abs().max()
        ),
        "current_edit_controls_bit_identical": edit_equal,
        "current_transition_bit_identical": torch.equal(transition, changed_transition),
        "current_injection_bit_identical": torch.equal(injection, changed_injection),
        "prior_edit_control_effect": prior_effect,
        "passed": edit_equal
        and torch.equal(transition, changed_transition)
        and torch.equal(injection, changed_injection)
        and prior_effect > 0.0,
    }


@torch.no_grad()
def _boundary_audit(
    model: HybridMemoryLM, *, seed: int, batch_size: int
) -> tuple[dict[str, Any], str]:
    batch = generate_boundary_batch(batch_size, 128, seed=seed).to(
        model.embedding.weight.device
    )
    full = model(batch.token_ids, delta_scan_mode="recurrent")
    query_logits = _gather_time(full["logits"], batch.query_positions)
    accuracy = float((query_logits.argmax(-1) == batch.targets).float().mean())
    cut_points = sorted(
        {
            int(position)
            for position in commit_positions(batch)[0].detach().cpu().tolist()
        }
        | {batch.length}
    )
    pieces = []
    states = None
    start = 0
    for stop in cut_points:
        if stop <= start:
            continue
        output = model(
            batch.token_ids[:, start:stop], states, delta_scan_mode="parallel"
        )
        pieces.append(output["logits"])
        states = output["states"]
        start = stop
    chunk_logits = torch.cat(pieces, dim=1)
    chunk_residual = float((chunk_logits - full["logits"]).abs().max())
    chunk_predictions_equal = torch.equal(
        chunk_logits.argmax(-1), full["logits"].argmax(-1)
    )

    valid_mask = torch.ones_like(batch.token_ids, dtype=torch.bool)
    filler = batch.roles == ROLE_FILLER
    valid_mask[:, 1::7] &= ~filler[:, 1::7]
    masked_full = model(
        batch.token_ids, valid_mask=valid_mask, delta_scan_mode="recurrent"
    )
    step_logits = []
    step_states = None
    for position in range(batch.length):
        logits, step_states = model.step(
            batch.token_ids[:, position],
            step_states,
            valid_mask=valid_mask[:, position],
        )
        step_logits.append(logits[:, None])
    masked_step_logits = torch.cat(step_logits, dim=1)
    masked_residual = float((masked_step_logits - masked_full["logits"]).abs().max())
    masked_predictions_equal = torch.equal(
        masked_step_logits.argmax(-1), masked_full["logits"].argmax(-1)
    )
    compact_logit_residual = 0.0
    compact_state_residual = 0.0
    compact_predictions_equal = True
    masked_state = masked_full["states"][0]
    for row in range(batch.batch_size):
        selected = valid_mask[row]
        compact = model(
            batch.token_ids[row : row + 1, selected],
            delta_scan_mode="recurrent",
        )
        selected_masked_logits = masked_full["logits"][row : row + 1, selected]
        compact_logit_residual = max(
            compact_logit_residual,
            float((selected_masked_logits - compact["logits"]).abs().max()),
        )
        compact_predictions_equal = compact_predictions_equal and torch.equal(
            selected_masked_logits.argmax(-1), compact["logits"].argmax(-1)
        )
        compact_state = compact["states"][0]
        compact_state_residual = max(
            compact_state_residual,
            float(
                (masked_state.memory[row : row + 1] - compact_state.memory).abs().max()
            ),
            float(
                (masked_state.convolution[row : row + 1] - compact_state.convolution)
                .abs()
                .max()
            ),
        )
    tail_roles = batch.roles.gather(1, commit_positions(batch))
    tail_role_support = {
        "filler": int((tail_roles == ROLE_FILLER).sum()),
        "write_marker": int((tail_roles == ROLE_WRITE_MARKER).sum()),
        "query_marker": int((tail_roles == ROLE_QUERY_MARKER).sum()),
    }
    tail_role_support_passed = all(tail_role_support.values())
    causal = _trained_causal_audit(model, batch)
    return {
        "query_accuracy": accuracy,
        "chunk_boundary_maximum_absolute_logit_residual": chunk_residual,
        "chunk_boundary_predictions_equal": chunk_predictions_equal,
        "masked_step_maximum_absolute_logit_residual": masked_residual,
        "masked_step_predictions_equal": masked_predictions_equal,
        "masked_compact_maximum_absolute_logit_residual": compact_logit_residual,
        "masked_compact_maximum_absolute_state_residual": compact_state_residual,
        "masked_compact_predictions_equal": compact_predictions_equal,
        "commit_tail_role_support": tail_role_support,
        "commit_tail_role_support_passed": tail_role_support_passed,
        "causal_intervention": causal,
        "passed": chunk_residual <= 5e-4
        and chunk_predictions_equal
        and masked_residual <= 5e-4
        and masked_predictions_equal
        and compact_logit_residual <= 5e-4
        and compact_state_residual <= 5e-4
        and compact_predictions_equal
        and tail_role_support_passed
        and causal["passed"],
    }, batch.fingerprint()


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
    commit_counts: dict[str, int] = {}
    stratum_total = {name: 0 for name in STRATA}
    stratum_correct = {name: 0 for name in STRATA}
    intervention_correct = {name: 0 for name in INTERVENTIONS}
    intervention_stratum_correct = {
        intervention: {stratum: 0 for stratum in STRATA}
        for intervention in INTERVENTIONS
    }
    reconstruction_residual = 0.0
    reconstruction_predictions_equal = True
    fingerprints: set[str] = set()
    total = 0
    episode_total = 0
    batch_index = 0
    model.eval()
    while total < decisions:
        batch_seed = _stable_seed(namespace, seed, task, length, batch_index)
        batch = _evaluation_batch(
            task, batch_size=batch_size, length=length, seed=batch_seed
        ).to(model.embedding.weight.device)
        fingerprints.add(batch.fingerprint())
        output = model(batch.token_ids, return_diagnostics=True)
        diagnostics = _transactional_diagnostics(output)
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
                _address_scores(diagnostics, batch).argmax(-1)
                == batch.query_key_indices
            ).sum()
        )
        _merge_counts(
            commit_counts,
            _binary_counts(
                _tensor(diagnostics, "commit_strength"),
                commit_event_mask(batch)[..., None, None],
            ),
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
                intervention = transactional_intervention_forward(
                    model, batch.token_ids, name
                )
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

    cell = {
        "task": task,
        "length": length,
        "query_decisions": total,
        "query_accuracy": correct / total,
        "exact_episode_accuracy": episodes / episode_total,
        "bits_per_query": nll / total / math.log(2.0),
        "query_address_top1": address_correct / total,
        "completed_tail_commit": _classification_report(commit_counts),
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
        cell["reconstruction_maximum_absolute_logit_residual"] = reconstruction_residual
        cell["reconstruction_query_predictions_equal"] = (
            reconstruction_predictions_equal
        )
    return cell, fingerprints


def _evaluate(
    model: HybridMemoryLM, config: CohortConfig, *, seed: int, arm: Arm
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
                namespace="g15bt-eval",
                interventions=False,
            )
            cells[f"{task}:L{length}"] = cell
            fingerprints.update(hashes)
    intervention_cells = {}
    intervention_hashes: set[str] = set()
    if arm in ("T", "T-AUX"):
        for task in ("mqar", "overwrite", "overwrite_guard"):
            for length in (512, 1024):
                cell, hashes = _evaluate_cell(
                    model,
                    task,
                    length,
                    seed=seed,
                    decisions=config.intervention_decisions,
                    batch_cap=config.intervention_batch_cap,
                    namespace="g15bt-intervention",
                    interventions=True,
                )
                intervention_cells[f"{task}:L{length}"] = cell
                intervention_hashes.update(hashes)
    if fingerprints & intervention_hashes:
        raise RuntimeError("standard and intervention evaluation fingerprints overlap")
    boundary, boundary_fingerprint = _boundary_audit(
        model,
        seed=_stable_seed("g15bt-boundary", seed),
        batch_size=min(config.evaluation_batch_cap, 8),
    )
    if (
        boundary_fingerprint in fingerprints
        or boundary_fingerprint in intervention_hashes
    ):
        raise RuntimeError("boundary and ordinary evaluation fingerprints overlap")
    standard_schedule_sha256 = hashlib.sha256(
        "\n".join(sorted(fingerprints)).encode()
    ).hexdigest()
    intervention_schedule_sha256 = hashlib.sha256(
        "\n".join(sorted(intervention_hashes)).encode()
    ).hexdigest()
    return {
        "cells": cells,
        "intervention_cells": intervention_cells,
        "standard_evaluation_schedule_sha256": standard_schedule_sha256,
        "intervention_evaluation_schedule_sha256": intervention_schedule_sha256,
        "boundary_batch_sha256": boundary_fingerprint,
        "intervention_batch_fingerprints": len(intervention_hashes),
        "boundary_audit": boundary,
    }, fingerprints | intervention_hashes | {boundary_fingerprint}


def _optimizer_partition(model: HybridMemoryLM) -> dict[str, Any]:
    optimizer = HarmonicMuonAdamW(model, lr=0.003, weight_decay=0.01)
    assigned = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    commit_names = [
        name
        for group in optimizer.partition_report()
        if group["role"] == "memory_controls"
        for name in group["names"]
        if "commit_projection" in name
    ]
    passed = (
        len(assigned) == len(trainable)
        and len({id(parameter) for parameter in assigned}) == len(trainable)
        and bool(commit_names)
    )
    report = {
        "assigned_tensors": len(assigned),
        "trainable_tensors": len(trainable),
        "commit_control_names": commit_names,
        "groups": optimizer.partition_report(),
        "passed": passed,
    }
    del optimizer
    return report


def run_preflight(device: torch.device) -> dict[str, Any]:
    phase0_sha = _sha256(PHASE0_ARTIFACT)
    phase0 = json.loads(PHASE0_ARTIFACT.read_text(encoding="utf-8"))
    models = {arm: build_model(arm, 23, device) for arm in ARMS}
    counts = {arm: parameter_count(model) for arm, model in models.items()}
    active = {
        arm: sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        for arm, model in models.items()
    }
    states = {
        arm: model.state_capacity_bytes(1, torch.float32)
        for arm, model in models.items()
    }
    shapes = {
        arm: {
            name: tuple(parameter.shape) for name, parameter in model.named_parameters()
        }
        for arm, model in models.items()
    }
    initial_hashes = {arm: _model_state_sha256(model) for arm, model in models.items()}
    batch = generate_interleaved_batch(
        "overwrite", 2, 128, 4, 8, 4, seed=_stable_seed("g15bt-preflight")
    ).to(device)
    losses = {}
    gradients = {}
    for arm, model in models.items():
        model.train()
        model.zero_grad(set_to_none=True)
        output = model(batch.token_ids, return_diagnostics=True)
        loss, _ = commissioned_losses(output, batch, arm=arm)  # type: ignore[arg-type]
        loss.backward()
        finite_nonzero = all(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and bool(torch.count_nonzero(parameter.grad))
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        losses[arm] = float(loss.detach())
        gradients[arm] = finite_nonzero
    models["T"].eval()
    with torch.no_grad():
        ordinary = models["T"](batch.token_ids)["logits"]
        reconstructed = transactional_intervention_forward(
            models["T"], batch.token_ids, "learned_reconstruction"
        )["logits"]
    reconstruction_residual = float((ordinary - reconstructed).abs().max())
    reconstruction_equal = torch.equal(ordinary.argmax(-1), reconstructed.argmax(-1))
    partition = _optimizer_partition(models["T"])
    paired = (
        len(set(counts.values())) == 1
        and len(set(active.values())) == 1
        and len(set(states.values())) == 1
        and shapes["F"] == shapes["T"] == shapes["T-AUX"]
        and initial_hashes["F"] == initial_hashes["T"] == initial_hashes["T-AUX"]
    )
    checks = {
        "sealed_phase0": phase0_sha == EXPECTED_PHASE0_SHA256
        and phase0.get("adjudication", {}).get("passed") is True,
        "matched_arms": paired,
        "finite_nonzero_gradients": all(gradients.values()),
        "optimizer_partition": partition["passed"],
        "learned_reconstruction": reconstruction_residual <= 5e-4
        and reconstruction_equal,
    }
    return {
        "phase0_sha256": phase0_sha,
        "parameter_counts": counts,
        "active_parameter_counts": active,
        "state_bytes_per_sequence_fp32": states,
        "initial_parameter_sha256": initial_hashes,
        "losses": losses,
        "gradients_finite_nonzero": gradients,
        "optimizer_partition": partition,
        "learned_reconstruction_maximum_absolute_logit_residual": reconstruction_residual,
        "learned_reconstruction_predictions_equal": reconstruction_equal,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _train_arm(
    arm: Arm,
    config: CohortConfig,
    *,
    seed: int,
    device: torch.device,
    checkpoint_directory: Path,
) -> dict[str, Any]:
    model = build_model(arm, seed, device)
    initial_parameter_sha256 = _model_state_sha256(model)
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
            batch = _batch_for_update(phase, seed=seed, global_update=global_update).to(
                device
            )
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
            loss, components = commissioned_losses(output, batch, arm=arm)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite G15B-T loss at update {global_update}")
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError(
                    f"nonfinite G15B-T gradient at update {global_update}"
                )
            optimizer.step()
            if (
                phase_update == 0
                or phase_update + 1 == phase.updates
                or global_update % 100 == 0
            ):
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
    evaluation, evaluation_hashes = _evaluate(model, config, seed=seed, arm=arm)
    _sync(device)
    evaluation_seconds = time.perf_counter() - evaluation_started
    intersection = train_fingerprints & evaluation_hashes
    if intersection:
        raise RuntimeError("training and evaluation fingerprints overlap")
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        checkpoint_directory / f"g15bt_{arm.replace('-', '').lower()}_seed{seed}.pt"
    )
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(
        {
            "schema_version": 1,
            "experiment": "G15B-T transactional controller",
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
        "active_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "state_bytes_per_sequence_fp32": model.state_capacity_bytes(1, torch.float32),
        "initial_parameter_sha256": initial_parameter_sha256,
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
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
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


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        text=True,
        encoding="utf-8",
    ).strip()


def _finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return True


def _arm_phase1_checks(
    arm: Arm, by_arm_seed: dict[tuple[str, int], dict[str, Any]]
) -> dict[str, bool]:
    """Apply the complete frozen absolute and causal gate vector to one arm."""

    checks: dict[str, bool] = {}
    for seed in QUALITY_SEEDS:
        arm_cells = by_arm_seed[(arm, seed)]["evaluation"]["cells"]
        f_cells = by_arm_seed[("F", seed)]["evaluation"]["cells"]
        boundary = by_arm_seed[(arm, seed)]["evaluation"]["boundary_audit"]
        checks[f"{arm}:{seed}:boundary_accuracy"] = boundary["query_accuracy"] >= 0.90
        checks[f"{arm}:{seed}:boundary_execution"] = boundary["passed"]
        for length in (128, 512, 1024):
            overwrite = arm_cells[f"overwrite:L{length}"]
            checks[f"{arm}:{seed}:overwrite:L{length}"] = (
                overwrite["query_accuracy"] >= 0.90
            )
            checks[f"{arm}:{seed}:post_same:L{length}"] = (
                overwrite["query_strata"]["after_same_key_overwrite"]["accuracy"]
                >= 0.90
            )
            for task in ("mqar", "overwrite", "selective"):
                task_cell = arm_cells[f"{task}:L{length}"]
                checks[f"{arm}:{seed}:{task}:address:L{length}"] = (
                    task_cell["query_address_top1"] >= 0.95
                )
                checks[f"{arm}:{seed}:{task}:commit:L{length}"] = (
                    task_cell["completed_tail_commit"]["f1"] >= 0.95
                )
            for task in ("mqar", "selective"):
                arm_cell = arm_cells[f"{task}:L{length}"]
                f_cell = f_cells[f"{task}:L{length}"]
                checks[f"{arm}:{seed}:{task}:versus_F:L{length}"] = (
                    arm_cell["query_accuracy"] >= f_cell["query_accuracy"] - 0.02
                )
        for length in EVALUATION_LENGTHS:
            checks[f"{arm}:{seed}:needle:L{length}"] = (
                arm_cells[f"needle:L{length}"]["query_accuracy"] == 1.0
            )
            guard = arm_cells[f"overwrite_guard:L{length}"]
            checks[f"{arm}:{seed}:guard:L{length}"] = guard[
                "query_accuracy"
            ] >= 0.99 and all(
                guard["query_strata"][stratum]["query_decisions"] > 0
                and guard["query_strata"][stratum]["accuracy"] is not None
                and guard["query_strata"][stratum]["accuracy"] >= 0.99
                for stratum in STRATA
            )
        intervention_cells = by_arm_seed[(arm, seed)]["evaluation"][
            "intervention_cells"
        ]
        for length in (512, 1024):
            overwrite = intervention_cells[f"overwrite:L{length}"]
            for name in (
                "commit_zero",
                "memory_zero",
                "permuted_history",
                "bias_only_history",
            ):
                checks[f"{arm}:{seed}:{name}:L{length}"] = (
                    overwrite["interventions"][name]["drop_from_learned"] >= 0.25
                )
            for name in ("commit_shift_minus_one", "commit_shift_plus_one"):
                checks[f"{arm}:{seed}:{name}:L{length}"] = (
                    overwrite["interventions"][name]["drop_from_learned"] >= 0.10
                )
            learned_post = overwrite["query_strata"]["after_same_key_overwrite"][
                "accuracy"
            ]
            erase_zero = overwrite["interventions"]["erase_zero"]["query_strata"][
                "after_same_key_overwrite"
            ]["accuracy"]
            checks[f"{arm}:{seed}:erase_zero_post:L{length}"] = (
                learned_post is not None
                and erase_zero is not None
                and learned_post - erase_zero >= 0.10
            )
            mqar = intervention_cells[f"mqar:L{length}"]
            checks[f"{arm}:{seed}:erase_zero_mqar:L{length}"] = (
                mqar["interventions"]["erase_zero"]["drop_from_learned"] <= 0.02
            )
            for cell_name in (
                f"mqar:L{length}",
                f"overwrite:L{length}",
                f"overwrite_guard:L{length}",
            ):
                cell = intervention_cells[cell_name]
                checks[f"{arm}:{seed}:reconstruction:{cell_name}"] = (
                    cell["reconstruction_maximum_absolute_logit_residual"] <= 5e-4
                    and cell["reconstruction_query_predictions_equal"]
                )
    for length in EVALUATION_LENGTHS:
        arm_overwrite_mean = sum(
            by_arm_seed[(arm, seed)]["evaluation"]["cells"][f"overwrite:L{length}"][
                "query_accuracy"
            ]
            for seed in QUALITY_SEEDS
        ) / len(QUALITY_SEEDS)
        f_overwrite_mean = sum(
            by_arm_seed[("F", seed)]["evaluation"]["cells"][f"overwrite:L{length}"][
                "query_accuracy"
            ]
            for seed in QUALITY_SEEDS
        ) / len(QUALITY_SEEDS)
        checks[f"{arm}:mean_minus_F:overwrite:L{length}"] = (
            arm_overwrite_mean - f_overwrite_mean >= 0.05
        )
        if length <= 1024:
            for task in ("mqar", "selective"):
                arm_mean = sum(
                    by_arm_seed[(arm, seed)]["evaluation"]["cells"][
                        f"{task}:L{length}"
                    ]["query_accuracy"]
                    for seed in QUALITY_SEEDS
                ) / len(QUALITY_SEEDS)
                checks[f"{arm}:mean:{task}:L{length}"] = arm_mean >= 0.95
    return checks


def _adjudicate(config: CohortConfig, reports: list[dict[str, Any]]) -> dict[str, Any]:
    if config.mode == "smoke":
        return {
            "passed": False,
            "eligible_for_promotion": False,
            "decision": "smoke execution completed; run the frozen quality cohort",
        }
    expected_rows = {(arm, seed) for arm in ARMS for seed in QUALITY_SEEDS}
    by_arm_seed = {(row["arm"], row["seed"]): row for row in reports}
    if len(reports) != len(expected_rows) or set(by_arm_seed) != expected_rows:
        return {
            "passed": False,
            "eligible_for_promotion": False,
            "checks": {"complete_seed_arm_matrix": False},
            "diagnostic_t_aux": {
                "passed": False,
                "checks": {},
                "interpretation": "incomplete cohort",
            },
            "decision": "stop G15B-T after an incomplete Phase-1 cohort",
        }
    checks: dict[str, bool] = {}
    expected_cells = {
        f"{task}:L{length}"
        for task in EVALUATION_TASKS
        for length in EVALUATION_LENGTHS
    }
    expected_interventions = {
        f"{task}:L{length}"
        for task in ("mqar", "overwrite", "overwrite_guard")
        for length in (512, 1024)
    }
    checks["complete_seed_arm_matrix"] = True
    for (arm, seed), row in by_arm_seed.items():
        evaluation = row["evaluation"]
        checks[f"complete:{arm}:{seed}:ordinary_cells"] = (
            set(evaluation["cells"]) == expected_cells
        )
        checks[f"complete:{arm}:{seed}:intervention_cells"] = set(
            evaluation["intervention_cells"]
        ) == (expected_interventions if arm in ("T", "T-AUX") else set())
        checkpoint = Path(row["checkpoint"])
        checks[f"provenance:{arm}:{seed}:checkpoint"] = (
            checkpoint.is_file()
            and len(row["checkpoint_sha256"]) == 64
            and _sha256(checkpoint) == row["checkpoint_sha256"]
        )
        checks[f"numerical:{arm}:{seed}:finite_report"] = _finite_tree(row)
    for seed in QUALITY_SEEDS:
        seed_rows = [by_arm_seed[(arm, seed)] for arm in ARMS]
        checks[f"paired:{seed}:initial_parameters"] = (
            len({row["initial_parameter_sha256"] for row in seed_rows}) == 1
        )
        checks[f"paired:{seed}:training_schedule"] = (
            len({row["training_schedule_sha256"] for row in seed_rows}) == 1
        )
        checks[f"paired:{seed}:standard_evaluation_schedule"] = (
            len(
                {
                    row["evaluation"]["standard_evaluation_schedule_sha256"]
                    for row in seed_rows
                }
            )
            == 1
        )
        checks[f"paired:{seed}:boundary_batch"] = (
            len({row["evaluation"]["boundary_batch_sha256"] for row in seed_rows}) == 1
        )
        checks[f"paired:{seed}:parameters"] = (
            len({row["parameters"] for row in seed_rows}) == 1
            and len({row["active_parameters"] for row in seed_rows}) == 1
            and len({row["state_bytes_per_sequence_fp32"] for row in seed_rows}) == 1
        )
        checks[f"paired:{seed}:updates_and_tokens"] = all(
            row["training_updates"] == 3400 and row["training_tokens"] == 13_926_400
            for row in seed_rows
        )
        checks[f"paired:{seed}:fingerprints"] = all(
            not row["train_evaluation_hash_intersection"] for row in seed_rows
        )
    t_checks = _arm_phase1_checks("T", by_arm_seed)
    t_aux_checks = _arm_phase1_checks("T-AUX", by_arm_seed)
    checks.update(t_checks)
    passed = all(checks.values())
    t_aux_passed = all(t_aux_checks.values())
    if passed:
        decision = "freeze a separate natural-text and scaling protocol"
    elif t_aux_passed:
        decision = (
            "stop before geometry: topology is sufficient, but the LM-only "
            "objective did not identify transactional commit timing"
        )
    else:
        decision = "stop G15B-T after the constructed Phase-1 failure"
    return {
        "passed": passed,
        "eligible_for_promotion": passed,
        "checks": checks,
        "diagnostic_t_aux": {
            "passed": t_aux_passed,
            "eligible_for_promotion": False,
            "checks": t_aux_checks,
        },
        "decision": decision,
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
        raise RuntimeError("G15B-T cohort requires a clean checkout")
    device = torch.device(arguments.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("G15B-T requires the declared CUDA device")
    if torch.cuda.get_device_capability(device) != (7, 5):
        raise RuntimeError("G15B-T requires exact compute capability (7, 5)")
    preflight = run_preflight(device)
    if not preflight["passed"]:
        raise RuntimeError("G15B-T preflight failed")

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
    adjudication = _adjudicate(config, reports)
    report = {
        "schema_version": 1,
        "experiment": "G15B-T transactional controller",
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
    "commit_event_mask",
    "commit_positions",
    "frozen_config",
    "run_preflight",
    "transactional_intervention_forward",
    "valid_commit_mask",
]
