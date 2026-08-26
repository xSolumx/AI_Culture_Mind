"""G15B-R3 exact logical-component replacement checkpoint diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

from .g15b_interleaved_cohort import (
    EVALUATION_LENGTHS,
    NEEDLE_DISTANCES,
    _evaluation_batch_size,
    _gather_time,
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
    generate_interleaved_batch,
)
from .g15br1_event_erase import PRESERVED_CONTROL_NAMES
from .g15br2_collision_erase import (
    EXPECTED_G15B_SHA256,
    EXPECTED_R0_SHA256,
    EXPECTED_R1_SHA256,
    R0_ARTIFACT,
    R1_ARTIFACT,
    _validate_quality_artifact,
    _validate_r1,
    overwrite_query_strata,
)
from .g15br_checkpoint_repair import (
    PARENT_ARTIFACT,
    QUALITY_SEEDS,
    ROOT,
    ROLE_NAMES,
    _expected_identity,
    _git_provenance,
    _hidden_controls,
    _load_checkpoint,
    _sync,
    local_write_event_mask,
    temporal_observability_witness,
)
from .model import HybridMemoryLM

PROTOCOL = ROOT / "G15BR3_LOGICAL_COMPONENT_PROTOCOL_2026-08-26.md"
R2_ARTIFACT = ROOT / "artifacts/g15br2_collision_erase_sm75_2026-08-26.json"
EXPECTED_R2_SHA256 = "90652fe7034e5901b968eb5d139f02eb8bc714b0417c0889e16a2fdd6b7cf924"
INTERVENTIONS = (
    "learned",
    "learned_decomposed_replay",
    "erase_free_no_reset",
    "erase_free_lww",
)
Intervention = Literal[
    "learned",
    "learned_decomposed_replay",
    "erase_free_no_reset",
    "erase_free_lww",
]
STRATA = (
    "before_any_overwrite",
    "after_unrelated_overwrite_only",
    "after_same_key_overwrite",
)
CONTROL_NAMES = (
    "query_vector",
    "key_vector",
    "value_positive",
    "erase_strength",
    "write_strength",
    "retention",
    "transport_coordinates",
)


def logical_component_ownership(
    batch: InterleavedBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign background/value-plus-tail injections and exact reset events.

    Component zero is background. Components one through K correspond to the
    order of ``batch.live_keys``.
    """

    matches = batch.write_keys[..., None].eq(batch.live_keys[:, None, :])
    if not bool(matches.sum(dim=-1).eq(1).all()):
        raise RuntimeError("every write key must match exactly one live key")
    key_index = matches.to(torch.int64).argmax(dim=-1) + 1
    tail_positions = batch.write_positions + 1
    if not bool((tail_positions < batch.length).all()):
        raise RuntimeError("every logical write tail must remain in range")
    positions = torch.cat((batch.write_positions, tail_positions), dim=1)
    owners = torch.cat((key_index, key_index), dim=1)
    assignment_count = torch.zeros_like(batch.token_ids, dtype=torch.int64)
    assignment_count.scatter_add_(1, positions, torch.ones_like(positions))
    if bool((assignment_count > 1).any()):
        raise RuntimeError("logical write programs overlap")

    ownership = torch.zeros_like(batch.token_ids, dtype=torch.int64)
    ownership.scatter_(1, positions, owners)
    components = batch.live_keys.shape[1] + 1
    if int(ownership.min()) < 0 or int(ownership.max()) >= components:
        raise RuntimeError("logical component owner is out of range")

    reset_mask = torch.zeros(
        batch.batch_size,
        components,
        batch.length,
        dtype=torch.bool,
        device=batch.token_ids.device,
    )
    rows = torch.arange(batch.batch_size, device=batch.token_ids.device)[:, None]
    reset_mask[
        rows.expand_as(key_index).flatten(),
        key_index.flatten(),
        batch.write_positions.flatten(),
    ] = True
    if int(reset_mask.sum()) != int(batch.write_event_mask.sum()):
        raise RuntimeError("component reset mask does not match valid-write events")
    return ownership, reset_mask


def _read_from_states(
    mixer: Any,
    query: torch.Tensor,
    states: torch.Tensor,
) -> torch.Tensor:
    positive = torch.einsum("bthv,bhtvp->bthp", query, states)
    if mixer.config.readout_mode == "clifford":
        negative = torch.einsum(
            "...i,vji,...v->...j",
            positive,
            mixer.rho.to(positive),
            query,
        )
    else:
        negative = positive
    return torch.cat((positive, negative), dim=-1)


def _component_read(
    mixer: Any,
    controls: list[torch.Tensor],
    ownership: torch.Tensor,
    reset_mask: torch.Tensor,
    *,
    replace: bool,
) -> tuple[torch.Tensor, dict[str, int], float | None]:
    query, key, value, erase, write, retention, coordinates = controls
    left, right, injection, _ = mixer._transitions(
        key,
        value,
        erase,
        write,
        retention,
        coordinates,
        None,
    )
    batch, heads, length, width, _ = injection.shape
    components = int(ownership.max()) + 1
    expected_components = reset_mask.shape[1]
    if components > expected_components:
        raise RuntimeError("ownership references an unavailable component")
    components = expected_components

    weights = F.one_hot(ownership, num_classes=components).to(injection.dtype)
    if not bool(weights.sum(dim=-1).eq(1).all()):
        raise RuntimeError("each token injection must have exactly one owner")
    component_injection = (
        injection[:, None] * weights.permute(0, 2, 1)[:, :, None, :, None, None]
    )
    left_component = left[:, None].expand(
        batch, components, heads, length, width, width
    )
    right_component = right[:, None].expand_as(left_component)
    if replace:
        left_component = torch.where(
            reset_mask[:, :, None, :, None, None],
            torch.zeros((), dtype=left.dtype, device=left.device),
            left_component,
        )

    flat_shape = (batch * components, heads, length, width, width)
    states, _ = mixer._parallel_states(
        left_component.reshape(flat_shape),
        right_component.reshape(flat_shape),
        component_injection.reshape(flat_shape),
        injection.new_zeros(batch * components, heads, width, width),
    )
    summed_states = states.reshape(batch, components, heads, length, width, width).sum(
        dim=1
    )
    state_residual = None
    if not replace:
        monolithic_states, _ = mixer._parallel_states(
            left,
            right,
            injection,
            injection.new_zeros(batch, heads, width, width),
        )
        state_residual = float((monolithic_states - summed_states).abs().max())
    read = _read_from_states(mixer, query, summed_states)
    return (
        read,
        {
            "base_state_scalars_per_sequence": heads * width * width,
            "logical_components_per_sequence": components,
            "expanded_state_scalars_per_sequence": components * heads * width * width,
        },
        state_residual,
    )


def _finish_forward(
    model: HybridMemoryLM,
    hidden: torch.Tensor,
    outer_gate: torch.Tensor,
    mixed: torch.Tensor,
    read: torch.Tensor,
    mixer: Any,
) -> torch.Tensor:
    output_gate = 1.0 + torch.tanh(mixer.output_gate(mixed).view_as(read))
    update = mixer.output_projection(
        (mixer.output_norm(read) * output_gate).flatten(start_dim=2)
    )
    block = model.blocks[0]
    hidden = hidden + torch.sigmoid(block.residual_scale) * block.dropout(
        update * torch.sigmoid(outer_gate)
    )
    hidden = hidden + block.dropout(block.ffn(block.ffn_norm(hidden)))
    return model.lm_head(model.final_norm(hidden))


def logical_component_forward(
    model: HybridMemoryLM,
    batch: InterleavedBatch,
    intervention: Intervention,
) -> dict[str, Any]:
    """Run one frozen R3 state-law intervention."""

    if intervention not in INTERVENTIONS:
        raise ValueError(f"unknown G15B-R3 intervention {intervention!r}")
    hidden, outer_gate, mixed, mixer, controls = _hidden_controls(
        model, batch.token_ids
    )
    ownership, reset_mask = logical_component_ownership(batch)
    capacity = None
    component_state_residual = None
    if intervention == "learned":
        read, _ = mixer.forward_controls(*controls, scan_mode="parallel")
    else:
        if intervention in ("erase_free_no_reset", "erase_free_lww"):
            controls[3] = torch.zeros_like(controls[3])
        read, capacity, component_state_residual = _component_read(
            mixer,
            controls,
            ownership,
            reset_mask,
            replace=intervention == "erase_free_lww",
        )
    return {
        "logits": _finish_forward(model, hidden, outer_gate, mixed, read, mixer),
        "controls": {
            name: tensor for name, tensor in zip(CONTROL_NAMES, controls, strict=True)
        },
        "component_capacity": capacity,
        "component_state_residual": component_state_residual,
    }


def _erase_free_monolithic_forward(
    model: HybridMemoryLM,
    batch: InterleavedBatch,
) -> dict[str, Any]:
    hidden, outer_gate, mixed, mixer, controls = _hidden_controls(
        model, batch.token_ids
    )
    controls[3] = torch.zeros_like(controls[3])
    read, _ = mixer.forward_controls(*controls, scan_mode="parallel")
    return {
        "logits": _finish_forward(model, hidden, outer_gate, mixed, read, mixer),
        "controls": {
            name: tensor for name, tensor in zip(CONTROL_NAMES, controls, strict=True)
        },
    }


def _new_integrity() -> dict[str, Any]:
    return {
        "local_decoder_batches_checked": 0,
        "collision_mask_batches_checked": 0,
        "component_assignment_batches_checked": 0,
        "model_forward_maximum_absolute_logit_residual": 0.0,
        "learned_decomposition_maximum_absolute_logit_residual": 0.0,
        "learned_decomposition_maximum_absolute_state_residual": 0.0,
        "learned_decomposition_query_predictions_equal": True,
        "erase_free_decomposition_maximum_absolute_logit_residual": 0.0,
        "erase_free_decomposition_maximum_absolute_state_residual": 0.0,
        "erase_free_decomposition_query_predictions_equal": True,
        "tail_role_counts": {name: 0 for name in ROLE_NAMES.values()},
        "component_capacities": {},
        "preserved_controls": {
            name: {"bitwise_equal": True, "maximum_absolute_residual": 0.0}
            for name in PRESERVED_CONTROL_NAMES
        },
    }


def generate_component_guard_batch(
    batch_size: int,
    length: int,
    *,
    seed: int,
) -> InterleavedBatch:
    """Build a deterministic overwrite guard with every causal query stratum."""

    if batch_size < 1 or length < 64:
        raise ValueError("component guard requires batch_size >= 1 and length >= 64")
    widths = (3, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2)
    extra = length - sum(widths)
    gaps = [extra // (len(widths) + 1)] * (len(widths) + 1)
    for index in range(extra % len(gaps)):
        gaps[index] += 1
    starts = []
    cursor = gaps[0]
    for index, width in enumerate(widths):
        starts.append(cursor)
        cursor += width + gaps[index + 1]
    if cursor != length:
        raise RuntimeError("component guard gap accounting failed")

    rows_tokens = []
    rows_roles = []
    rows_live_keys = []
    rows_targets = []
    rows_query_positions = []
    rows_query_keys = []
    rows_query_indices = []
    rows_write_positions = []
    rows_write_keys = []
    rows_write_values = []
    rows_distances = []
    for row in range(batch_size):
        generator = torch.Generator(device="cpu").manual_seed(
            _stable_seed("g15br3-guard", seed, length, row)
        )
        permutation = torch.randperm(PAYLOAD_COUNT, generator=generator).tolist()
        keys = [PAYLOAD_START + value for value in permutation[:8]]
        old_a, value_b, new_a = [PAYLOAD_START + value for value in permutation[8:11]]
        a_index = row % len(keys)
        b_index = (row + 1) % len(keys)
        key_a, key_b = keys[a_index], keys[b_index]
        tokens = torch.randint(
            PAYLOAD_START,
            VOCAB_SIZE,
            (length,),
            generator=generator,
            dtype=torch.long,
        )
        roles = torch.full((length,), ROLE_FILLER, dtype=torch.long)

        write_specs = (
            (starts[0], key_a, old_a),
            (starts[1], key_b, value_b),
            (starts[4], key_a, new_a),
        )
        for start, key, value in write_specs:
            tokens[start : start + 3] = torch.tensor(
                (WRITE_TOKEN, key, value), dtype=torch.long
            )
            roles[start : start + 3] = torch.tensor(
                (ROLE_WRITE_MARKER, ROLE_WRITE_KEY, ROLE_WRITE_VALUE),
                dtype=torch.long,
            )

        query_specs = (
            (starts[2], key_a, a_index, old_a),
            (starts[3], key_b, b_index, value_b),
            (starts[5], key_b, b_index, value_b),
            (starts[6], key_a, a_index, new_a),
            (starts[7], key_b, b_index, value_b),
            (starts[8], key_a, a_index, new_a),
            (starts[9], key_b, b_index, value_b),
            (starts[10], key_a, a_index, new_a),
        )
        for start, key, _, target in query_specs:
            tokens[start : start + 2] = torch.tensor(
                (QUERY_TOKEN, key), dtype=torch.long
            )
            roles[start : start + 2] = torch.tensor(
                (ROLE_QUERY_MARKER, ROLE_QUERY_KEY), dtype=torch.long
            )
            query_position = start + 1
            for local in range(max(0, query_position - 3), query_position + 1):
                if roles[local] == ROLE_FILLER and int(tokens[local]) == target:
                    tokens[local] = PAYLOAD_START + (
                        (target - PAYLOAD_START + 1) % PAYLOAD_COUNT
                    )

        write_positions = [start + 2 for start, _, _ in write_specs]
        query_positions = [start + 1 for start, _, _, _ in query_specs]
        latest = {key_a: write_positions[0], key_b: write_positions[1]}
        distances = []
        for query_position, (_, key, _, _) in zip(
            query_positions, query_specs, strict=True
        ):
            if query_position > write_positions[2] and key == key_a:
                latest[key_a] = write_positions[2]
            distances.append(query_position - latest[key])

        rows_tokens.append(tokens)
        rows_roles.append(roles)
        rows_live_keys.append(keys)
        rows_targets.append([target for _, _, _, target in query_specs])
        rows_query_positions.append(query_positions)
        rows_query_keys.append([key for _, key, _, _ in query_specs])
        rows_query_indices.append([index for _, _, index, _ in query_specs])
        rows_write_positions.append(write_positions)
        rows_write_keys.append([key_a, key_b, key_a])
        rows_write_values.append([old_a, value_b, new_a])
        rows_distances.append(distances)

    token_ids = torch.stack(rows_tokens)
    roles = torch.stack(rows_roles)
    write_positions = torch.tensor(rows_write_positions, dtype=torch.long)
    overwrite = torch.tensor([[False, False, True]] * batch_size, dtype=torch.bool)
    write_event_mask = torch.zeros(batch_size, length, dtype=torch.bool)
    write_event_mask.scatter_(1, write_positions, True)
    erase_event_mask = torch.zeros_like(write_event_mask)
    erase_event_mask.scatter_(1, write_positions, overwrite)
    return InterleavedBatch(
        task="overwrite",
        token_ids=token_ids,
        targets=torch.tensor(rows_targets, dtype=torch.long),
        query_positions=torch.tensor(rows_query_positions, dtype=torch.long),
        query_keys=torch.tensor(rows_query_keys, dtype=torch.long),
        query_key_indices=torch.tensor(rows_query_indices, dtype=torch.long),
        live_keys=torch.tensor(rows_live_keys, dtype=torch.long),
        write_positions=write_positions,
        write_keys=torch.tensor(rows_write_keys, dtype=torch.long),
        write_values=torch.tensor(rows_write_values, dtype=torch.long),
        overwrite_mask=overwrite,
        write_event_mask=write_event_mask,
        erase_event_mask=erase_event_mask,
        roles=roles,
        needle_distances=torch.tensor(rows_distances, dtype=torch.long),
        seed=seed,
    )


@torch.no_grad()
def evaluate_checkpoint(
    model: HybridMemoryLM,
    *,
    seed: int,
    decisions: int,
    batch_cap: int,
) -> dict[str, Any]:
    model.eval()
    device = model.embedding.weight.device
    cells: dict[str, Any] = {}
    integrity = _new_integrity()
    for task in ("mqar", "overwrite", "overwrite_guard", "selective", "needle"):
        for length in EVALUATION_LENGTHS:
            batch_size = _evaluation_batch_size(
                "overwrite" if task == "overwrite_guard" else task,
                decisions=decisions,
                cap=batch_cap,
            )
            per_batch = batch_size * (1 if task == "needle" else 8)
            if decisions % per_batch:
                raise ValueError("decisions must contain complete evaluation batches")
            correct = {name: 0 for name in INTERVENTIONS}
            episodes = {name: 0 for name in INTERVENTIONS}
            nll_sum = {name: 0.0 for name in INTERVENTIONS}
            stratum_correct = {
                name: {stratum: 0 for stratum in STRATA} for name in INTERVENTIONS
            }
            stratum_total = {stratum: 0 for stratum in STRATA}
            total = 0
            episode_total = 0
            batch_index = 0
            while total < decisions:
                batch_seed = _stable_seed("g15b-eval", seed, task, length, batch_index)
                if task == "overwrite_guard":
                    batch = generate_component_guard_batch(
                        batch_size, length, seed=batch_seed
                    ).to(device)
                else:
                    batch = generate_interleaved_batch(
                        task,  # type: ignore[arg-type]
                        batch_size,
                        length,
                        8,
                        24,
                        8,
                        seed=batch_seed,
                        needle_distance=(
                            NEEDLE_DISTANCES[length] if task == "needle" else None
                        ),
                    ).to(device)
                if not torch.equal(
                    local_write_event_mask(batch.token_ids), batch.write_event_mask
                ):
                    raise RuntimeError("valid-write target is not locally observable")
                integrity["local_decoder_batches_checked"] += 1
                expected_collisions = torch.zeros_like(batch.erase_event_mask)
                expected_collisions.scatter_(
                    1, batch.write_positions, batch.overwrite_mask
                )
                if not torch.equal(expected_collisions, batch.erase_event_mask):
                    raise RuntimeError(
                        "collision mask does not match overwrite history"
                    )
                integrity["collision_mask_batches_checked"] += 1
                ownership, reset_mask = logical_component_ownership(batch)
                if int(reset_mask.sum()) != int(batch.write_event_mask.sum()):
                    raise RuntimeError("valid-write reset mask lost an event")
                tail_roles = batch.roles.gather(1, batch.write_positions + 1)
                for role, name in ROLE_NAMES.items():
                    integrity["tail_role_counts"][name] += int(
                        tail_roles.eq(role).sum()
                    )
                integrity["component_assignment_batches_checked"] += 1

                results = {
                    intervention: logical_component_forward(model, batch, intervention)
                    for intervention in INTERVENTIONS
                }
                erase_free_monolithic = _erase_free_monolithic_forward(model, batch)
                learned_result = results["learned"]
                ordinary_logits = model(batch.token_ids)["logits"]
                integrity["model_forward_maximum_absolute_logit_residual"] = max(
                    float(integrity["model_forward_maximum_absolute_logit_residual"]),
                    float((ordinary_logits - learned_result["logits"]).abs().max()),
                )
                learned_replay_logits = results["learned_decomposed_replay"]["logits"]
                integrity["learned_decomposition_maximum_absolute_logit_residual"] = (
                    max(
                        float(
                            integrity[
                                "learned_decomposition_maximum_absolute_logit_residual"
                            ]
                        ),
                        float(
                            (learned_result["logits"] - learned_replay_logits)
                            .abs()
                            .max()
                        ),
                    )
                )
                integrity["learned_decomposition_maximum_absolute_state_residual"] = (
                    max(
                        float(
                            integrity[
                                "learned_decomposition_maximum_absolute_state_residual"
                            ]
                        ),
                        float(
                            results["learned_decomposed_replay"][
                                "component_state_residual"
                            ]
                        ),
                    )
                )
                erase_free_logits = results["erase_free_no_reset"]["logits"]
                integrity[
                    "erase_free_decomposition_maximum_absolute_logit_residual"
                ] = max(
                    float(
                        integrity[
                            "erase_free_decomposition_maximum_absolute_logit_residual"
                        ]
                    ),
                    float(
                        (erase_free_monolithic["logits"] - erase_free_logits)
                        .abs()
                        .max()
                    ),
                )
                integrity[
                    "erase_free_decomposition_maximum_absolute_state_residual"
                ] = max(
                    float(
                        integrity[
                            "erase_free_decomposition_maximum_absolute_state_residual"
                        ]
                    ),
                    float(results["erase_free_no_reset"]["component_state_residual"]),
                )
                capacity = results["learned_decomposed_replay"]["component_capacity"]
                capacity_key = str(capacity["logical_components_per_sequence"])
                prior_capacity = integrity["component_capacities"].get(capacity_key)
                if prior_capacity is None:
                    integrity["component_capacities"][capacity_key] = capacity
                elif prior_capacity != capacity:
                    raise RuntimeError(
                        "component capacity changed for a live-key count"
                    )

                strata = (
                    overwrite_query_strata(batch)
                    if task in ("overwrite", "overwrite_guard")
                    else {}
                )
                for stratum, mask in strata.items():
                    stratum_total[stratum] += int(mask.sum())
                query_predictions: dict[str, torch.Tensor] = {}
                for intervention, result in results.items():
                    selected_logits = _gather_time(
                        result["logits"], batch.query_positions
                    )
                    prediction = selected_logits.argmax(-1)
                    query_predictions[intervention] = prediction
                    match = prediction == batch.targets
                    correct[intervention] += int(match.sum())
                    episodes[intervention] += int(match.all(dim=1).sum())
                    nll_sum[intervention] += float(
                        F.cross_entropy(
                            selected_logits.flatten(0, 1),
                            batch.targets.flatten(),
                            reduction="sum",
                        )
                    )
                    for stratum, mask in strata.items():
                        stratum_correct[intervention][stratum] += int(
                            (match & mask).sum()
                        )
                    if intervention == "learned":
                        continue
                    for name in PRESERVED_CONTROL_NAMES:
                        learned_control = learned_result["controls"][name]
                        candidate_control = result["controls"][name]
                        report = integrity["preserved_controls"][name]
                        report["bitwise_equal"] = bool(
                            report["bitwise_equal"]
                        ) and torch.equal(learned_control, candidate_control)
                        report["maximum_absolute_residual"] = max(
                            float(report["maximum_absolute_residual"]),
                            float((learned_control - candidate_control).abs().max()),
                        )
                monolithic_prediction = _gather_time(
                    erase_free_monolithic["logits"], batch.query_positions
                ).argmax(-1)
                integrity["learned_decomposition_query_predictions_equal"] = bool(
                    integrity["learned_decomposition_query_predictions_equal"]
                ) and torch.equal(
                    query_predictions["learned"],
                    query_predictions["learned_decomposed_replay"],
                )
                integrity["erase_free_decomposition_query_predictions_equal"] = bool(
                    integrity["erase_free_decomposition_query_predictions_equal"]
                ) and torch.equal(
                    monolithic_prediction,
                    query_predictions["erase_free_no_reset"],
                )
                total += batch.targets.numel()
                episode_total += batch.batch_size
                batch_index += 1

            cell: dict[str, Any] = {
                "task": task,
                "length": length,
                "query_decisions": total,
                "interventions": {
                    name: {
                        "query_accuracy": correct[name] / total,
                        "exact_episode_accuracy": episodes[name] / episode_total,
                        "bits_per_query": nll_sum[name] / total / math.log(2.0),
                    }
                    for name in INTERVENTIONS
                },
            }
            if task in ("overwrite", "overwrite_guard"):
                cell["query_strata"] = {
                    stratum: {
                        "query_decisions": stratum_total[stratum],
                        "accuracy": {
                            name: (
                                stratum_correct[name][stratum] / stratum_total[stratum]
                                if stratum_total[stratum]
                                else None
                            )
                            for name in INTERVENTIONS
                        },
                    }
                    for stratum in STRATA
                }
            cells[f"{task}:L{length}"] = cell
    integrity["preserved_controls_bitwise_equal"] = all(
        row["bitwise_equal"] for row in integrity["preserved_controls"].values()
    )
    return {"cells": cells, "runtime_integrity": integrity}


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot calculate an empty mean")
    return sum(values) / len(values)


def _adjudicate(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    cell_names = list(seed_reports[0]["evaluation"]["cells"])
    replay_residuals = [
        residual
        for report in seed_reports
        for cell in report["evaluation"]["cells"].values()
        if "baseline_query_accuracy_absolute_residual" in cell
        for residual in (
            cell["baseline_query_accuracy_absolute_residual"],
            cell["baseline_exact_episode_accuracy_absolute_residual"],
            cell["baseline_bits_per_query_absolute_residual"],
            cell["no_erase_query_accuracy_absolute_residual"],
        )
    ]
    means: dict[str, Any] = {}
    for cell_name in cell_names:
        rows = [report["evaluation"]["cells"][cell_name] for report in seed_reports]
        accuracy = {
            intervention: _mean(
                [row["interventions"][intervention]["query_accuracy"] for row in rows]
            )
            for intervention in INTERVENTIONS
        }
        cell_report: dict[str, Any] = {
            "mean_query_accuracy": accuracy,
            "erase_free_lww_minus_learned": (
                accuracy["erase_free_lww"] - accuracy["learned"]
            ),
            "erase_free_lww_minus_erase_free_no_reset": (
                accuracy["erase_free_lww"] - accuracy["erase_free_no_reset"]
            ),
            "erase_free_no_reset_minus_learned": (
                accuracy["erase_free_no_reset"] - accuracy["learned"]
            ),
        }
        if cell_name.startswith(("overwrite:", "overwrite_guard:")):
            cell_report["query_strata"] = {}
            for stratum in STRATA:
                stratum_accuracy: dict[str, float | None] = {}
                for intervention in INTERVENTIONS:
                    values = [
                        row["query_strata"][stratum]["accuracy"][intervention]
                        for row in rows
                        if row["query_strata"][stratum]["accuracy"][intervention]
                        is not None
                    ]
                    stratum_accuracy[intervention] = _mean(values) if values else None
                learned = stratum_accuracy["learned"]
                no_reset = stratum_accuracy["erase_free_no_reset"]
                replacement = stratum_accuracy["erase_free_lww"]
                cell_report["query_strata"][stratum] = {
                    "query_decisions": sum(
                        row["query_strata"][stratum]["query_decisions"] for row in rows
                    ),
                    "mean_accuracy": stratum_accuracy,
                    "erase_free_lww_minus_learned": (
                        replacement - learned
                        if replacement is not None and learned is not None
                        else None
                    ),
                    "erase_free_lww_minus_erase_free_no_reset": (
                        replacement - no_reset
                        if replacement is not None and no_reset is not None
                        else None
                    ),
                }
        means[cell_name] = cell_report

    replay_passed = max(replay_residuals, default=math.inf) <= 1e-12
    witness_passed = all(row["observability_witness"]["passed"] for row in seed_reports)
    runtime_integrity_passed = all(
        report["evaluation"]["runtime_integrity"][
            "model_forward_maximum_absolute_logit_residual"
        ]
        == 0.0
        and report["evaluation"]["runtime_integrity"][
            "learned_decomposition_maximum_absolute_logit_residual"
        ]
        <= 5e-4
        and report["evaluation"]["runtime_integrity"][
            "learned_decomposition_maximum_absolute_state_residual"
        ]
        <= 2e-6
        and report["evaluation"]["runtime_integrity"][
            "learned_decomposition_query_predictions_equal"
        ]
        and report["evaluation"]["runtime_integrity"][
            "erase_free_decomposition_maximum_absolute_logit_residual"
        ]
        <= 5e-4
        and report["evaluation"]["runtime_integrity"][
            "erase_free_decomposition_maximum_absolute_state_residual"
        ]
        <= 2e-6
        and report["evaluation"]["runtime_integrity"][
            "erase_free_decomposition_query_predictions_equal"
        ]
        and report["evaluation"]["runtime_integrity"][
            "preserved_controls_bitwise_equal"
        ]
        and report["evaluation"]["runtime_integrity"]["local_decoder_batches_checked"]
        > 0
        and report["evaluation"]["runtime_integrity"]["collision_mask_batches_checked"]
        > 0
        and report["evaluation"]["runtime_integrity"][
            "component_assignment_batches_checked"
        ]
        > 0
        for report in seed_reports
    )
    checks: dict[str, bool] = {}
    stratum_checks: dict[str, bool] = {}
    for name, row in means.items():
        task = name.split(":", 1)[0]
        if task in ("overwrite", "overwrite_guard"):
            checks[f"{name}:versus_learned"] = (
                row["erase_free_lww_minus_learned"] >= 0.10
            )
            checks[f"{name}:versus_erase_free_no_reset"] = (
                row["erase_free_lww_minus_erase_free_no_reset"] >= 0.10
            )
            for stratum, stratum_row in row["query_strata"].items():
                learned_delta = stratum_row["erase_free_lww_minus_learned"]
                no_reset_delta = stratum_row["erase_free_lww_minus_erase_free_no_reset"]
                if stratum == "after_same_key_overwrite":
                    stratum_checks[f"{name}:{stratum}:versus_learned"] = (
                        learned_delta is not None and learned_delta >= 0.10
                    )
                    stratum_checks[f"{name}:{stratum}:versus_no_reset"] = (
                        no_reset_delta is not None and no_reset_delta >= 0.10
                    )
                elif task == "overwrite_guard":
                    stratum_checks[f"{name}:{stratum}:populated"] = (
                        stratum_row["query_decisions"] > 0
                    )
                    stratum_checks[f"{name}:{stratum}:versus_no_reset"] = (
                        no_reset_delta is not None and no_reset_delta >= -0.02
                    )
                    stratum_checks[f"{name}:{stratum}:versus_learned"] = (
                        learned_delta is not None and learned_delta >= -0.02
                    )
        elif task in ("mqar", "selective"):
            checks[name] = row["erase_free_lww_minus_learned"] >= -0.02
        else:
            checks[name] = row["mean_query_accuracy"]["erase_free_lww"] >= 0.999
    passed = (
        replay_passed
        and witness_passed
        and runtime_integrity_passed
        and all(checks.values())
        and all(stratum_checks.values())
    )
    post_same_improved = all(
        row["query_strata"]["after_same_key_overwrite"][
            "erase_free_lww_minus_erase_free_no_reset"
        ]
        is not None
        and row["query_strata"]["after_same_key_overwrite"][
            "erase_free_lww_minus_erase_free_no_reset"
        ]
        >= 0.10
        for name, row in means.items()
        if name.startswith(("overwrite:", "overwrite_guard:"))
    )
    if passed:
        decision = (
            "support a separately frozen fresh explicit-slot/occupancy state-law "
            "screen; do not revive G15C or the token-local controller"
        )
    elif post_same_improved:
        decision = (
            "do not train yet; inspect write-tail ownership and background/component "
            "coupling"
        )
    else:
        decision = (
            "reject this frozen post-hoc two-token ownership/reset construction; "
            "do not infer why the decoder fails or reject fresh slot training"
        )
    return {
        "baseline_replay_maximum_absolute_residual": max(
            replay_residuals, default=math.inf
        ),
        "baseline_replay_passed": replay_passed,
        "observability_witness_passed": witness_passed,
        "runtime_integrity_passed": runtime_integrity_passed,
        "three_seed_means": means,
        "erase_free_lww_checks": checks,
        "stratum_checks": stratum_checks,
        "post_same_key_improved": post_same_improved,
        "passed": passed,
        "decision": decision,
    }


def _validate_r2(path: Path) -> tuple[dict[str, Any], str]:
    actual_sha256 = _sha256(path)
    if actual_sha256 != EXPECTED_R2_SHA256:
        raise ValueError("R2 artifact hash does not match the frozen input")
    report = json.loads(path.read_text(encoding="utf-8"))
    _validate_quality_artifact(report, name="R2")
    if report.get("parent_g15b_sha256") != EXPECTED_G15B_SHA256:
        raise ValueError("R2 does not bind the frozen G15B artifact")
    if report.get("parent_r0_sha256") != EXPECTED_R0_SHA256:
        raise ValueError("R2 does not bind the frozen R0 artifact")
    if report.get("parent_r1_sha256") != EXPECTED_R1_SHA256:
        raise ValueError("R2 does not bind the frozen R1 artifact")
    if report.get("adjudication", {}).get("passed") is not False:
        raise ValueError("R3 requires the failed R2 adjudication")
    if report["adjudication"].get("runtime_integrity_passed") is not True:
        raise ValueError("R3 requires R2 runtime integrity")
    return report, actual_sha256


def run(
    *,
    mode: Literal["smoke", "quality"],
    device: torch.device,
    parent_path: Path,
    r0_path: Path,
    r1_path: Path,
    r2_path: Path,
    checkpoint_directory: Path,
    commit: str,
    status_at_start: list[str],
) -> dict[str, Any]:
    parent, parent_sha256 = _validate_parent(parent_path)
    _, r0_sha256 = _validate_r0(r0_path, parent_sha256=parent_sha256)
    _, r1_sha256 = _validate_r1(r1_path)
    _, r2_sha256 = _validate_r2(r2_path)
    expected = _expected_identity(parent)
    seeds = QUALITY_SEEDS if mode == "quality" else QUALITY_SEEDS[:1]
    decisions = 4096 if mode == "quality" else 16
    batch_cap = 16 if mode == "quality" else 2
    seed_reports = []
    started = time.perf_counter()
    for seed in seeds:
        checkpoint_path = checkpoint_directory / f"g15b_I_seed{seed}.pt"
        model, checkpoint = _load_checkpoint(
            checkpoint_path,
            seed=seed,
            expected=expected[seed],
            device=device,
        )
        witness = temporal_observability_witness(model)
        _sync(device)
        evaluation_started = time.perf_counter()
        evaluation = evaluate_checkpoint(
            model, seed=seed, decisions=decisions, batch_cap=batch_cap
        )
        _sync(device)
        for name, cell in evaluation["cells"].items():
            if name.startswith("overwrite_guard:"):
                continue
            recorded = checkpoint["evaluation"]["cells"][name]
            replayed = cell["interventions"]["learned"]
            for metric in (
                "query_accuracy",
                "exact_episode_accuracy",
                "bits_per_query",
            ):
                cell[f"recorded_g15b_{metric}"] = recorded[metric]
                cell[f"baseline_{metric}_absolute_residual"] = abs(
                    replayed[metric] - recorded[metric]
                )
            recorded_no_erase = recorded["interventions"]["no_erase"]["accuracy"]
            replayed_no_erase = cell["interventions"]["erase_free_no_reset"][
                "query_accuracy"
            ]
            cell["recorded_g15b_no_erase_query_accuracy"] = recorded_no_erase
            cell["no_erase_query_accuracy_absolute_residual"] = abs(
                replayed_no_erase - recorded_no_erase
            )
        seed_reports.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "observability_witness": witness,
                "evaluation_wall_seconds": time.perf_counter() - evaluation_started,
                "evaluation": evaluation,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    adjudication = _adjudicate(seed_reports)
    source_paths = (
        Path(__file__),
        PROTOCOL,
        ROOT / "g15br2_collision_erase.py",
        ROOT / "g15br1_event_erase.py",
        ROOT / "g15br_checkpoint_repair.py",
        ROOT / "g15b_interleaved_cohort.py",
        ROOT / "g15b_interleaved_tasks.py",
        ROOT / "spin_dirac_memory.py",
        ROOT / "model.py",
    )
    return {
        "schema_version": 1,
        "experiment": "G15B-R3 exact logical-component replacement diagnostic",
        "mode": mode,
        "evidentiary": mode == "quality" and not status_at_start,
        "git_commit_at_start": commit,
        "git_status_at_start": status_at_start,
        "elapsed_wall_seconds": time.perf_counter() - started,
        "parent_g15b_artifact": str(parent_path),
        "parent_g15b_sha256": parent_sha256,
        "parent_r0_artifact": str(r0_path),
        "parent_r0_sha256": r0_sha256,
        "parent_r1_artifact": str(r1_path),
        "parent_r1_sha256": r1_sha256,
        "parent_r2_artifact": str(r2_path),
        "parent_r2_sha256": r2_sha256,
        "protocol": {
            "seeds": list(seeds),
            "evaluation_decisions_per_cell": decisions,
            "evaluation_batch_cap": batch_cap,
            "tasks": [
                "mqar",
                "overwrite",
                "overwrite_guard",
                "selective",
                "needle",
            ],
            "lengths": list(EVALUATION_LENGTHS),
            "interventions": list(INTERVENTIONS),
            "write_program_positions": ["write_value", "one_token_tail"],
            "overwrite_query_strata": list(STRATA),
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
            "no parameter is trained or updated",
            "component ownership uses commissioned task metadata",
            "the constructed guard is deterministic mechanism evidence, not generalization",
            "the expanded component state is not parameter, state, compute, or wall-time matched",
            "replayed held-out G15B cells are not fresh generalization evidence",
            "no G15C, token-local controller, natural-text, optimizer, Spin, scaling, or model-family promotion follows",
        ],
    }


def _validate_parent(path: Path) -> tuple[dict[str, Any], str]:
    actual_sha256 = _sha256(path)
    if actual_sha256 != EXPECTED_G15B_SHA256:
        raise ValueError("G15B artifact hash does not match the frozen input")
    report = json.loads(path.read_text(encoding="utf-8"))
    _validate_quality_artifact(report, name="G15B")
    if report.get("adjudication", {}).get("passed") is not False:
        raise ValueError("R3 requires the failed G15B adjudication")
    return report, actual_sha256


def _validate_r0(path: Path, *, parent_sha256: str) -> tuple[dict[str, Any], str]:
    actual_sha256 = _sha256(path)
    if actual_sha256 != EXPECTED_R0_SHA256:
        raise ValueError("R0 artifact hash does not match the frozen input")
    report = json.loads(path.read_text(encoding="utf-8"))
    _validate_quality_artifact(report, name="R0")
    if report.get("parent_artifact_sha256") != parent_sha256:
        raise ValueError("R0 does not bind the frozen G15B artifact")
    if report.get("adjudication", {}).get("passed") is not False:
        raise ValueError("R3 requires the failed R0 adjudication")
    return report, actual_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "quality"), required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--parent-artifact", type=Path, default=PARENT_ARTIFACT)
    parser.add_argument("--r0-artifact", type=Path, default=R0_ARTIFACT)
    parser.add_argument("--r1-artifact", type=Path, default=R1_ARTIFACT)
    parser.add_argument("--r2-artifact", type=Path, default=R2_ARTIFACT)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    device = torch.device(args.device)
    commit, status = _git_provenance()
    if args.mode == "quality":
        if status:
            raise RuntimeError("G15B-R3 quality requires a clean git tree at start")
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("G15B-R3 quality requires CUDA")
        if torch.cuda.get_device_capability(device) != (7, 5):
            raise RuntimeError("G15B-R3 quality is frozen to SM75")
    report = run(
        mode=args.mode,
        device=device,
        parent_path=args.parent_artifact,
        r0_path=args.r0_artifact,
        r1_path=args.r1_artifact,
        r2_path=args.r2_artifact,
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
    "INTERVENTIONS",
    "STRATA",
    "evaluate_checkpoint",
    "generate_component_guard_batch",
    "logical_component_forward",
    "logical_component_ownership",
]
