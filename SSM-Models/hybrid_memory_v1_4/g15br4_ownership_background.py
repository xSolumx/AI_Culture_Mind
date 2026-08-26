"""G15B-R4 frozen ownership/background retained-checkpoint diagnostic."""

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
from .g15b_interleaved_cohort import (
    EVALUATION_LENGTHS,
    NEEDLE_DISTANCES,
    _evaluation_batch_size,
    _gather_time,
    _sha256,
    _stable_seed,
)
from .g15b_interleaved_tasks import (
    QUERY_TOKEN,
    InterleavedBatch,
    generate_interleaved_batch,
)
from .g15br1_event_erase import PRESERVED_CONTROL_NAMES
from .g15br2_collision_erase import overwrite_query_strata
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

PROTOCOL = ROOT / "G15BR4_OWNERSHIP_BACKGROUND_PROTOCOL_2026-08-26.md"
R3_ARTIFACT = ROOT / "artifacts/g15br3_logical_component_sm75_2026-08-26.json"
EXPECTED_R3_SHA256 = "0fe54b8ce38868d67a7ecb0cb888f2279d8809c2bbaf3ccbda678326ff808959"

INTERVENTIONS = (
    "learned",
    "erase_free_no_reset_bgplus",
    "v_no_reset_bgminus",
    "vt_no_reset_bgminus",
    "v_lww_bgplus",
    "v_lww_bgminus",
    "vt_lww_bgplus",
    "vt_lww_bgminus",
)
LWW_ARMS = (
    "v_lww_bgplus",
    "v_lww_bgminus",
    "vt_lww_bgplus",
    "vt_lww_bgminus",
)
VALUE_ONLY_ARMS = ("v_lww_bgplus", "v_lww_bgminus")
TAIL_ARMS = ("vt_lww_bgplus", "vt_lww_bgminus")
STRATA = r3.STRATA
OwnershipMode = Literal["v", "vt"]

MATCHING_CONTROL = {
    "v_lww_bgplus": "erase_free_no_reset_bgplus",
    "v_lww_bgminus": "v_no_reset_bgminus",
    "vt_lww_bgplus": "erase_free_no_reset_bgplus",
    "vt_lww_bgminus": "vt_no_reset_bgminus",
}


def local_query_position_mask(token_ids: torch.Tensor) -> torch.Tensor:
    """Decode query-key positions using current and immediately prior tokens."""

    if token_ids.ndim != 2 or token_ids.shape[1] < 1:
        raise ValueError("token_ids must have nonempty shape (batch,length)")
    mask = torch.zeros_like(token_ids, dtype=torch.bool)
    if token_ids.shape[1] > 1:
        mask[:, 1:] = token_ids[:, :-1].eq(QUERY_TOKEN)
    return mask


def factor_ownership(
    batch: InterleavedBatch,
    mode: OwnershipMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exclusive V or VT ownership and reset every valid write."""

    if mode == "vt":
        return r3.logical_component_ownership(batch)
    if mode != "v":
        raise ValueError(f"unknown ownership mode {mode!r}")
    matches = batch.write_keys[..., None].eq(batch.live_keys[:, None, :])
    if not bool(matches.sum(dim=-1).eq(1).all()):
        raise RuntimeError("every write key must match exactly one live key")
    key_index = matches.to(torch.int64).argmax(dim=-1) + 1
    ownership = torch.zeros_like(batch.token_ids, dtype=torch.int64)
    ownership.scatter_(1, batch.write_positions, key_index)
    components = batch.live_keys.shape[1] + 1
    reset_mask = torch.zeros(
        batch.batch_size,
        components,
        batch.length,
        dtype=torch.bool,
        device=batch.token_ids.device,
    )
    rows = torch.arange(batch.batch_size, device=batch.token_ids.device)[:, None]
    rows = rows.expand_as(key_index)
    reset_mask[rows.flatten(), key_index.flatten(), batch.write_positions.flatten()] = (
        True
    )
    if int(reset_mask.sum()) != int(batch.write_event_mask.sum()):
        raise RuntimeError("component reset mask does not match valid writes")
    return ownership, reset_mask


def _component_factor_reads(
    mixer: Any,
    controls: list[torch.Tensor],
    ownership: torch.Tensor,
    reset_mask: torch.Tensor,
    *,
    replace: bool,
) -> dict[str, Any]:
    query, key, value, erase, write, retention, coordinates = controls
    left, right, injection, _ = mixer._transitions(
        key, value, erase, write, retention, coordinates, None
    )
    batch, heads, length, width, _ = injection.shape
    components = reset_mask.shape[1]
    if int(ownership.min()) < 0 or int(ownership.max()) >= components:
        raise RuntimeError("ownership references an unavailable component")
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
    component_states = states.reshape(batch, components, heads, length, width, width)
    full_states = component_states.sum(dim=1)
    key_states = component_states[:, 1:].sum(dim=1)
    background_states = component_states[:, 0]
    full_read = r3._read_from_states(mixer, query, full_states)
    key_read = r3._read_from_states(mixer, query, key_states)
    background_read = r3._read_from_states(mixer, query, background_states)
    relation_residual = float((full_read - key_read - background_read).abs().max())
    state_residual = None
    if not replace:
        monolithic_states, _ = mixer._parallel_states(
            left,
            right,
            injection,
            injection.new_zeros(batch, heads, width, width),
        )
        state_residual = float((monolithic_states - full_states).abs().max())
    return {
        "full_read": full_read,
        "key_read": key_read,
        "background_read": background_read,
        "background_relation_residual": relation_residual,
        "state_residual": state_residual,
        "capacity": {
            "base_state_scalars_per_sequence": heads * width * width,
            "logical_components_per_sequence": components,
            "expanded_state_scalars_per_sequence": components * heads * width * width,
        },
    }


def _query_background_read(
    reads: dict[str, Any], query_mask: torch.Tensor, *, include_background: bool
) -> torch.Tensor:
    if include_background:
        return reads["full_read"]
    return torch.where(
        query_mask[..., None, None], reads["key_read"], reads["full_read"]
    )


def factorial_forwards(
    model: HybridMemoryLM,
    batch: InterleavedBatch,
) -> dict[str, Any]:
    """Compute all frozen R4 arms while sharing learned controls."""

    hidden, outer_gate, mixed, mixer, learned_controls = _hidden_controls(
        model, batch.token_ids
    )
    learned_read, _ = mixer.forward_controls(*learned_controls, scan_mode="parallel")
    component_controls = list(learned_controls)
    component_controls[3] = torch.zeros_like(component_controls[3])
    query_mask = local_query_position_mask(batch.token_ids)

    reads: dict[tuple[str, bool], dict[str, Any]] = {}
    for ownership_mode in ("v", "vt"):
        ownership, reset_mask = factor_ownership(batch, ownership_mode)
        reads[(ownership_mode, False)] = _component_factor_reads(
            mixer,
            component_controls,
            ownership,
            reset_mask,
            replace=False,
        )
        reads[(ownership_mode, True)] = _component_factor_reads(
            mixer,
            component_controls,
            ownership,
            reset_mask,
            replace=True,
        )

    selected_reads = {
        "learned": learned_read,
        "erase_free_no_reset_bgplus": reads[("vt", False)]["full_read"],
        "v_no_reset_bgminus": _query_background_read(
            reads[("v", False)], query_mask, include_background=False
        ),
        "vt_no_reset_bgminus": _query_background_read(
            reads[("vt", False)], query_mask, include_background=False
        ),
        "v_lww_bgplus": reads[("v", True)]["full_read"],
        "v_lww_bgminus": _query_background_read(
            reads[("v", True)], query_mask, include_background=False
        ),
        "vt_lww_bgplus": reads[("vt", True)]["full_read"],
        "vt_lww_bgminus": _query_background_read(
            reads[("vt", True)], query_mask, include_background=False
        ),
    }
    logits = {
        name: r3._finish_forward(
            model, hidden, outer_gate, mixed, selected_reads[name], mixer
        )
        for name in INTERVENTIONS
    }
    return {
        "logits": logits,
        "learned_controls": {
            name: tensor
            for name, tensor in zip(r3.CONTROL_NAMES, learned_controls, strict=True)
        },
        "component_controls": {
            name: tensor
            for name, tensor in zip(r3.CONTROL_NAMES, component_controls, strict=True)
        },
        "forward_context": (hidden, outer_gate, mixed, mixer),
        "query_mask": query_mask,
        "reads": reads,
    }


def fp64_algebraic_contract(
    model: HybridMemoryLM,
    batch: InterleavedBatch,
) -> dict[str, Any]:
    """Verify component algebra in FP64 independently of scored logits."""

    _, _, _, mixer, controls = _hidden_controls(model, batch.token_ids)
    controls[3] = torch.zeros_like(controls[3])
    query, key, value, erase, write, retention, coordinates = controls
    left, right, injection, _ = mixer._transitions(
        key, value, erase, write, retention, coordinates, None
    )
    query64 = query.double()
    left64 = left.double()
    right64 = right.double()
    injection64 = injection.double()
    batch_size, heads, length, width, _ = injection64.shape
    zero = injection64.new_zeros(batch_size, heads, width, width)
    monolithic_parallel, _ = mixer._parallel_states(left64, right64, injection64, zero)
    monolithic_recurrent, _ = mixer._recurrent_states(
        left64, right64, injection64, zero
    )
    maximum_recurrent_parallel = float(
        (monolithic_parallel - monolithic_recurrent).abs().max()
    )
    maximum_component_sum = 0.0
    maximum_component_recurrent_parallel = 0.0
    maximum_background_relation = 0.0
    mode_reports: dict[str, Any] = {}
    for ownership_mode in ("v", "vt"):
        ownership, reset_mask = factor_ownership(batch, ownership_mode)
        components = reset_mask.shape[1]
        weights = F.one_hot(ownership, num_classes=components).to(injection64)
        component_injection = (
            injection64[:, None] * weights.permute(0, 2, 1)[:, :, None, :, None, None]
        )
        left_component = left64[:, None].expand(
            batch_size, components, heads, length, width, width
        )
        right_component = right64[:, None].expand_as(left_component)
        mode_report: dict[str, float] = {}
        for replace in (False, True):
            selected_left = left_component
            if replace:
                selected_left = torch.where(
                    reset_mask[:, :, None, :, None, None],
                    torch.zeros((), dtype=left64.dtype, device=left64.device),
                    left_component,
                )
            flat_shape = (
                batch_size * components,
                heads,
                length,
                width,
                width,
            )
            flat_zero = injection64.new_zeros(
                batch_size * components, heads, width, width
            )
            parallel, _ = mixer._parallel_states(
                selected_left.reshape(flat_shape),
                right_component.reshape(flat_shape),
                component_injection.reshape(flat_shape),
                flat_zero,
            )
            recurrent, _ = mixer._recurrent_states(
                selected_left.reshape(flat_shape),
                right_component.reshape(flat_shape),
                component_injection.reshape(flat_shape),
                flat_zero,
            )
            parity = float((parallel - recurrent).abs().max())
            maximum_component_recurrent_parallel = max(
                maximum_component_recurrent_parallel, parity
            )
            component_states = parallel.reshape(
                batch_size, components, heads, length, width, width
            )
            full_states = component_states.sum(dim=1)
            if not replace:
                sum_residual = float((full_states - monolithic_parallel).abs().max())
                maximum_component_sum = max(maximum_component_sum, sum_residual)
                mode_report["no_reset_component_sum_residual"] = sum_residual
            full_read = r3._read_from_states(mixer, query64, full_states)
            key_read = r3._read_from_states(
                mixer, query64, component_states[:, 1:].sum(dim=1)
            )
            background_read = r3._read_from_states(
                mixer, query64, component_states[:, 0]
            )
            relation = float((full_read - key_read - background_read).abs().max())
            maximum_background_relation = max(maximum_background_relation, relation)
            prefix = "lww" if replace else "no_reset"
            mode_report[f"{prefix}_recurrent_parallel_residual"] = parity
            mode_report[f"{prefix}_background_relation_residual"] = relation
        mode_reports[ownership_mode] = mode_report
    maximum_residual = max(
        maximum_recurrent_parallel,
        maximum_component_sum,
        maximum_component_recurrent_parallel,
        maximum_background_relation,
    )
    return {
        "dtype": "float64",
        "batch_size": batch.batch_size,
        "length": batch.length,
        "monolithic_recurrent_parallel_residual": maximum_recurrent_parallel,
        "maximum_component_sum_residual": maximum_component_sum,
        "maximum_component_recurrent_parallel_residual": (
            maximum_component_recurrent_parallel
        ),
        "maximum_background_relation_residual": maximum_background_relation,
        "maximum_residual": maximum_residual,
        "passed": maximum_residual <= 1e-10,
        "ownership_modes": mode_reports,
    }


def _new_integrity() -> dict[str, Any]:
    return {
        "local_write_batches_checked": 0,
        "local_query_batches_checked": 0,
        "ownership_batches_checked": 0,
        "ordinary_model_forward_maximum_absolute_logit_residual": 0.0,
        "no_reset_state_residual": {"v": 0.0, "vt": 0.0},
        "no_reset_query_predictions_equal": {"v": True, "vt": True},
        "background_relation_maximum_absolute_read_residual": {
            "v_no_reset": 0.0,
            "v_lww": 0.0,
            "vt_no_reset": 0.0,
            "vt_lww": 0.0,
        },
        "lww_no_reset_no_overwrite_predictions_equal": {arm: True for arm in LWW_ARMS},
        "lww_no_reset_before_overwrite_predictions_equal": {
            arm: True for arm in LWW_ARMS
        },
        "preserved_controls": {
            name: {"bitwise_equal": True, "maximum_absolute_residual": 0.0}
            for name in PRESERVED_CONTROL_NAMES
        },
        "tail_role_counts": {name: 0 for name in ROLE_NAMES.values()},
        "writes_without_in_range_tail": 0,
        "component_capacities": {},
        "fp64_algebraic_contract": None,
        "finite_logits": True,
    }


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
            fingerprint_digest = hashlib.sha256()
            while total < decisions:
                batch_seed = _stable_seed("g15b-eval", seed, task, length, batch_index)
                if task == "overwrite_guard":
                    batch = r3.generate_component_guard_batch(
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
                fingerprint_digest.update(batch.fingerprint().encode())
                if not torch.equal(
                    local_write_event_mask(batch.token_ids), batch.write_event_mask
                ):
                    raise RuntimeError("valid-write target is not locally observable")
                integrity["local_write_batches_checked"] += 1
                expected_query = torch.zeros_like(batch.write_event_mask)
                expected_query.scatter_(1, batch.query_positions, True)
                if not torch.equal(
                    local_query_position_mask(batch.token_ids), expected_query
                ):
                    raise RuntimeError("query positions are not locally observable")
                integrity["local_query_batches_checked"] += 1
                v_ownership, v_reset = factor_ownership(batch, "v")
                vt_ownership, vt_reset = factor_ownership(batch, "vt")
                tail_positions = batch.write_positions + 1
                valid_tail = tail_positions < batch.length
                rows = torch.arange(batch.batch_size, device=device)[:, None]
                rows = rows.expand_as(tail_positions)
                if bool(
                    v_ownership[rows[valid_tail], tail_positions[valid_tail]].any()
                ):
                    raise RuntimeError("value-only ownership captured a write tail")
                if int(v_reset.sum()) != int(vt_reset.sum()) or not torch.equal(
                    v_reset, vt_reset
                ):
                    raise RuntimeError("ownership modes disagree on reset events")
                integrity["ownership_batches_checked"] += 1

                tail_roles = batch.roles[rows[valid_tail], tail_positions[valid_tail]]
                integrity["writes_without_in_range_tail"] += int((~valid_tail).sum())
                for role, name in ROLE_NAMES.items():
                    integrity["tail_role_counts"][name] += int(
                        tail_roles.eq(role).sum()
                    )

                if integrity["fp64_algebraic_contract"] is None:
                    integrity["fp64_algebraic_contract"] = fp64_algebraic_contract(
                        model, batch
                    )

                result = factorial_forwards(model, batch)
                logits = result["logits"]
                integrity["finite_logits"] = bool(integrity["finite_logits"]) and all(
                    bool(torch.isfinite(value).all()) for value in logits.values()
                )
                ordinary_logits = model(batch.token_ids)["logits"]
                integrity["ordinary_model_forward_maximum_absolute_logit_residual"] = (
                    max(
                        float(
                            integrity[
                                "ordinary_model_forward_maximum_absolute_logit_residual"
                            ]
                        ),
                        float((ordinary_logits - logits["learned"]).abs().max()),
                    )
                )
                erase_free_monolithic = r3._erase_free_monolithic_forward(model, batch)
                monolithic_prediction = _gather_time(
                    erase_free_monolithic["logits"], batch.query_positions
                ).argmax(-1)
                hidden, outer_gate, mixed, mixer = result["forward_context"]
                for ownership_mode in ("v", "vt"):
                    reads = result["reads"][(ownership_mode, False)]
                    integrity["no_reset_state_residual"][ownership_mode] = max(
                        float(integrity["no_reset_state_residual"][ownership_mode]),
                        float(reads["state_residual"]),
                    )
                    no_reset_full_logits = r3._finish_forward(
                        model,
                        hidden,
                        outer_gate,
                        mixed,
                        reads["full_read"],
                        mixer,
                    )
                    full_prediction = _gather_time(
                        no_reset_full_logits, batch.query_positions
                    ).argmax(-1)
                    integrity["no_reset_query_predictions_equal"][ownership_mode] = (
                        bool(
                            integrity["no_reset_query_predictions_equal"][
                                ownership_mode
                            ]
                        )
                        and torch.equal(monolithic_prediction, full_prediction)
                    )
                    for replace in (False, True):
                        key = f"{ownership_mode}_{'lww' if replace else 'no_reset'}"
                        relation = result["reads"][(ownership_mode, replace)][
                            "background_relation_residual"
                        ]
                        integrity["background_relation_maximum_absolute_read_residual"][
                            key
                        ] = max(
                            float(
                                integrity[
                                    "background_relation_maximum_absolute_read_residual"
                                ][key]
                            ),
                            float(relation),
                        )
                capacity = result["reads"][("vt", False)]["capacity"]
                capacity_key = str(capacity["logical_components_per_sequence"])
                prior_capacity = integrity["component_capacities"].get(capacity_key)
                if prior_capacity is None:
                    integrity["component_capacities"][capacity_key] = capacity
                elif prior_capacity != capacity:
                    raise RuntimeError("component capacity changed")
                for name in PRESERVED_CONTROL_NAMES:
                    learned_control = result["learned_controls"][name]
                    candidate_control = result["component_controls"][name]
                    report = integrity["preserved_controls"][name]
                    report["bitwise_equal"] = bool(report["bitwise_equal"]) and (
                        torch.equal(learned_control, candidate_control)
                    )
                    report["maximum_absolute_residual"] = max(
                        float(report["maximum_absolute_residual"]),
                        float((learned_control - candidate_control).abs().max()),
                    )

                strata = (
                    overwrite_query_strata(batch)
                    if task in ("overwrite", "overwrite_guard")
                    else {}
                )
                for stratum, mask in strata.items():
                    stratum_total[stratum] += int(mask.sum())
                predictions: dict[str, torch.Tensor] = {}
                for intervention in INTERVENTIONS:
                    selected_logits = _gather_time(
                        logits[intervention], batch.query_positions
                    )
                    prediction = selected_logits.argmax(-1)
                    predictions[intervention] = prediction
                    match = prediction.eq(batch.targets)
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
                for arm in LWW_ARMS:
                    control = MATCHING_CONTROL[arm]
                    if task not in ("overwrite", "overwrite_guard"):
                        integrity["lww_no_reset_no_overwrite_predictions_equal"][
                            arm
                        ] = bool(
                            integrity["lww_no_reset_no_overwrite_predictions_equal"][
                                arm
                            ]
                        ) and torch.equal(
                            predictions[arm], predictions[control]
                        )
                    elif strata:
                        before = strata["before_any_overwrite"]
                        integrity["lww_no_reset_before_overwrite_predictions_equal"][
                            arm
                        ] = bool(
                            integrity[
                                "lww_no_reset_before_overwrite_predictions_equal"
                            ][arm]
                        ) and torch.equal(
                            predictions[arm][before], predictions[control][before]
                        )
                total += batch.targets.numel()
                episode_total += batch.batch_size
                batch_index += 1

            cell: dict[str, Any] = {
                "task": task,
                "length": length,
                "query_decisions": total,
                "batch_fingerprint_sha256": fingerprint_digest.hexdigest(),
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


def _three_seed_means(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    cell_names = list(seed_reports[0]["evaluation"]["cells"])
    means: dict[str, Any] = {}
    for cell_name in cell_names:
        rows = [report["evaluation"]["cells"][cell_name] for report in seed_reports]
        accuracy = {
            intervention: _mean(
                [row["interventions"][intervention]["query_accuracy"] for row in rows]
            )
            for intervention in INTERVENTIONS
        }
        cell_report: dict[str, Any] = {"mean_query_accuracy": accuracy}
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
                cell_report["query_strata"][stratum] = {
                    "query_decisions": sum(
                        row["query_strata"][stratum]["query_decisions"] for row in rows
                    ),
                    "mean_accuracy": stratum_accuracy,
                }
        means[cell_name] = cell_report
    return means


def _runtime_integrity_passed(seed_reports: list[dict[str, Any]]) -> bool:
    for report in seed_reports:
        integrity = report["evaluation"]["runtime_integrity"]
        if not (
            integrity["ordinary_model_forward_maximum_absolute_logit_residual"] == 0.0
            and all(integrity["no_reset_query_predictions_equal"].values())
            and all(integrity["lww_no_reset_no_overwrite_predictions_equal"].values())
            and all(
                integrity["lww_no_reset_before_overwrite_predictions_equal"].values()
            )
            and integrity["preserved_controls_bitwise_equal"]
            and integrity["finite_logits"]
            and integrity["fp64_algebraic_contract"] is not None
            and integrity["fp64_algebraic_contract"]["passed"]
            and integrity["local_write_batches_checked"] > 0
            and integrity["local_query_batches_checked"] > 0
            and integrity["ownership_batches_checked"] > 0
        ):
            return False
    return True


def _arm_checks(
    arm: str,
    seed_reports: list[dict[str, Any]],
    means: dict[str, Any],
) -> dict[str, bool]:
    control = MATCHING_CONTROL[arm]
    checks: dict[str, bool] = {}
    for length in EVALUATION_LENGTHS:
        overwrite_name = f"overwrite:L{length}"
        overwrite = means[overwrite_name]
        accuracy = overwrite["mean_query_accuracy"]
        checks[f"{overwrite_name}:aggregate:versus_learned"] = (
            accuracy[arm] - accuracy["learned"] >= 0.10
        )
        checks[f"{overwrite_name}:aggregate:versus_control"] = (
            accuracy[arm] - accuracy[control] >= 0.10
        )
        same = overwrite["query_strata"]["after_same_key_overwrite"]["mean_accuracy"]
        checks[f"{overwrite_name}:post_same:versus_learned"] = (
            same[arm] is not None
            and same["learned"] is not None
            and same[arm] - same["learned"] >= 0.10
        )
        checks[f"{overwrite_name}:post_same:versus_control"] = (
            same[arm] is not None
            and same[control] is not None
            and same[arm] - same[control] >= 0.10
        )
        before = overwrite["query_strata"]["before_any_overwrite"]["mean_accuracy"]
        checks[f"{overwrite_name}:before:versus_learned"] = (
            before[arm] is not None
            and before["learned"] is not None
            and before[arm] - before["learned"] >= -0.02
        )
        checks[f"{overwrite_name}:before:versus_control"] = (
            before[arm] is not None
            and before[control] is not None
            and before[arm] - before[control] >= -0.02
        )
        for report in seed_reports:
            cell = report["evaluation"]["cells"][overwrite_name]
            seed = report["seed"]
            seed_accuracy = cell["interventions"]
            checks[f"{overwrite_name}:seed{seed}:aggregate:versus_learned"] = (
                seed_accuracy[arm]["query_accuracy"]
                - seed_accuracy["learned"]["query_accuracy"]
                >= 0.05
            )
            checks[f"{overwrite_name}:seed{seed}:aggregate:versus_control"] = (
                seed_accuracy[arm]["query_accuracy"]
                - seed_accuracy[control]["query_accuracy"]
                >= 0.05
            )
            seed_same = cell["query_strata"]["after_same_key_overwrite"]["accuracy"]
            checks[f"{overwrite_name}:seed{seed}:post_same:versus_learned"] = (
                seed_same[arm] is not None
                and seed_same["learned"] is not None
                and seed_same[arm] - seed_same["learned"] >= 0.05
            )
            checks[f"{overwrite_name}:seed{seed}:post_same:versus_control"] = (
                seed_same[arm] is not None
                and seed_same[control] is not None
                and seed_same[arm] - seed_same[control] >= 0.05
            )

        guard_name = f"overwrite_guard:L{length}"
        guard = means[guard_name]
        guard_accuracy = guard["mean_query_accuracy"]
        checks[f"{guard_name}:aggregate:absolute"] = guard_accuracy[arm] >= 0.995
        checks[f"{guard_name}:aggregate:versus_learned"] = (
            guard_accuracy[arm] - guard_accuracy["learned"] >= -0.005
        )
        checks[f"{guard_name}:aggregate:versus_control"] = (
            guard_accuracy[arm] - guard_accuracy[control] >= 0.10
        )
        guard_same = guard["query_strata"]["after_same_key_overwrite"]
        checks[f"{guard_name}:post_same:populated"] = guard_same["query_decisions"] > 0
        guard_same_accuracy = guard_same["mean_accuracy"]
        checks[f"{guard_name}:post_same:absolute"] = (
            guard_same_accuracy[arm] is not None and guard_same_accuracy[arm] >= 0.995
        )
        checks[f"{guard_name}:post_same:versus_learned"] = (
            guard_same_accuracy[arm] is not None
            and guard_same_accuracy["learned"] is not None
            and guard_same_accuracy[arm] - guard_same_accuracy["learned"] >= -0.005
        )
        checks[f"{guard_name}:post_same:versus_control"] = (
            guard_same_accuracy[arm] is not None
            and guard_same_accuracy[control] is not None
            and guard_same_accuracy[arm] - guard_same_accuracy[control] >= 0.10
        )
        for stratum in ("before_any_overwrite", "after_unrelated_overwrite_only"):
            row = guard["query_strata"][stratum]
            row_accuracy = row["mean_accuracy"]
            checks[f"{guard_name}:{stratum}:populated"] = row["query_decisions"] > 0
            checks[f"{guard_name}:{stratum}:absolute"] = (
                row_accuracy[arm] is not None and row_accuracy[arm] >= 0.98
            )
            checks[f"{guard_name}:{stratum}:versus_learned"] = (
                row_accuracy[arm] is not None
                and row_accuracy["learned"] is not None
                and row_accuracy[arm] - row_accuracy["learned"] >= -0.02
            )
            checks[f"{guard_name}:{stratum}:versus_control"] = (
                row_accuracy[arm] is not None
                and row_accuracy[control] is not None
                and row_accuracy[arm] - row_accuracy[control] >= -0.02
            )

        for task in ("mqar", "selective"):
            row = means[f"{task}:L{length}"]["mean_query_accuracy"]
            checks[f"{task}:L{length}:safety"] = row[arm] - row["learned"] >= -0.02
        needle = means[f"needle:L{length}"]["mean_query_accuracy"]
        checks[f"needle:L{length}:safety"] = needle[arm] >= 0.999
    return checks


def _adjudicate(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    means = _three_seed_means(seed_reports)
    replay_residuals = [
        value
        for report in seed_reports
        for cell in report["evaluation"]["cells"].values()
        for reference in cell.get("r3_reference_replay", {}).values()
        for value in reference.values()
    ]
    replay_passed = max(replay_residuals, default=math.inf) <= 1e-12
    witness_passed = all(row["observability_witness"]["passed"] for row in seed_reports)
    runtime_passed = _runtime_integrity_passed(seed_reports)
    checks = {arm: _arm_checks(arm, seed_reports, means) for arm in LWW_ARMS}
    performance_passed = {
        arm: all(arm_checks.values()) for arm, arm_checks in checks.items()
    }
    passed_arms = [
        arm
        for arm in LWW_ARMS
        if performance_passed[arm]
        and replay_passed
        and witness_passed
        and runtime_passed
    ]
    passed_value_arms = [arm for arm in VALUE_ONLY_ARMS if arm in passed_arms]
    passed_tail_arms = [arm for arm in TAIL_ARMS if arm in passed_arms]
    selected_training_law = None
    if "v_lww_bgplus" in passed_value_arms:
        selected_training_law = "value-only slots with shared background"
    elif "v_lww_bgminus" in passed_value_arms:
        selected_training_law = (
            "value-only slots with protected background-free query reads"
        )
    if selected_training_law is not None:
        decision = (
            "support a separately frozen fresh explicit-slot training screen with "
            f"{selected_training_law}"
        )
    elif passed_tail_arms:
        decision = (
            "do not train; passing behavior remains dependent on ambiguous "
            "value-plus-tail ownership"
        )
    else:
        decision = (
            "reject this frozen post-hoc ownership/background family; do not "
            "reject fresh slot, GDN2/KDA, or dual-address training"
        )
    return {
        "r3_reference_replay_maximum_absolute_residual": max(
            replay_residuals, default=math.inf
        ),
        "r3_reference_replay_passed": replay_passed,
        "observability_witness_passed": witness_passed,
        "runtime_integrity_passed": runtime_passed,
        "three_seed_means": means,
        "arm_checks": checks,
        "performance_passed": performance_passed,
        "passed_arms": passed_arms,
        "passed_value_arms": passed_value_arms,
        "passed_tail_arms": passed_tail_arms,
        "selected_training_law": selected_training_law,
        "passed": selected_training_law is not None,
        "decision": decision,
    }


def _validate_r3(path: Path) -> tuple[dict[str, Any], str]:
    actual_sha256 = _sha256(path)
    if actual_sha256 != EXPECTED_R3_SHA256:
        raise ValueError("R3 artifact hash does not match the frozen input")
    report = json.loads(path.read_text(encoding="utf-8"))
    r3._validate_quality_artifact(report, name="R3")
    adjudication = report.get("adjudication", {})
    if adjudication.get("passed") is not False:
        raise ValueError("R4 requires the failed R3 adjudication")
    if adjudication.get("post_same_key_improved") is not True:
        raise ValueError("R4 requires R3 post-same-key mechanism improvement")
    if adjudication.get("decision") != (
        "do not train yet; inspect write-tail ownership and background/component "
        "coupling"
    ):
        raise ValueError("R3 does not select the frozen R4 question")
    return report, actual_sha256


def _r3_seed_map(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result = {int(row["seed"]): row for row in report["seed_reports"]}
    if set(result) != set(QUALITY_SEEDS):
        raise ValueError("R3 seed reports do not match the frozen quality seeds")
    return result


def run(
    *,
    mode: Literal["smoke", "quality"],
    device: torch.device,
    parent_path: Path,
    r3_path: Path,
    checkpoint_directory: Path,
    commit: str,
    status_at_start: list[str],
) -> dict[str, Any]:
    parent, parent_sha256 = r3._validate_parent(parent_path)
    r3_report, r3_sha256 = _validate_r3(r3_path)
    r3_seeds = _r3_seed_map(r3_report)
    expected = _expected_identity(parent)
    seeds = QUALITY_SEEDS if mode == "quality" else QUALITY_SEEDS[:1]
    decisions = 4096 if mode == "quality" else 16
    batch_cap = 16 if mode == "quality" else 2
    seed_reports = []
    started = time.perf_counter()
    replay_mapping = {
        "learned": "learned",
        "erase_free_no_reset": "erase_free_no_reset_bgplus",
        "erase_free_lww": "vt_lww_bgplus",
    }
    for seed in seeds:
        checkpoint_path = checkpoint_directory / f"g15b_I_seed{seed}.pt"
        model, _ = _load_checkpoint(
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
        r3_cells = r3_seeds[seed]["evaluation"]["cells"]
        for name, cell in evaluation["cells"].items():
            cell["r3_reference_replay"] = {}
            for r3_intervention, r4_intervention in replay_mapping.items():
                reference = r3_cells[name]["interventions"][r3_intervention]
                replay = cell["interventions"][r4_intervention]
                cell["r3_reference_replay"][r3_intervention] = {
                    metric: abs(replay[metric] - reference[metric])
                    for metric in (
                        "query_accuracy",
                        "exact_episode_accuracy",
                        "bits_per_query",
                    )
                }
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
        ROOT / "g15br3_logical_component.py",
        ROOT / "g15b_interleaved_cohort.py",
        ROOT / "g15b_interleaved_tasks.py",
        ROOT / "spin_dirac_memory.py",
        ROOT / "model.py",
    )
    return {
        "schema_version": 1,
        "experiment": "G15B-R4 ownership/background factorial diagnostic",
        "mode": mode,
        "evidentiary": mode == "quality" and not status_at_start,
        "git_commit_at_start": commit,
        "git_status_at_start": status_at_start,
        "elapsed_wall_seconds": time.perf_counter() - started,
        "parent_g15b_artifact": str(parent_path),
        "parent_g15b_sha256": parent_sha256,
        "parent_r3_artifact": str(r3_path),
        "parent_r3_sha256": r3_sha256,
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
            "ownership_modes": ["value_only", "value_plus_tail"],
            "query_background_modes": ["included", "excluded"],
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
            "logical component ownership uses commissioned task metadata",
            "value-plus-tail ownership remains semantically ambiguous",
            "background exclusion does not select an oracle queried slot",
            "replayed G15B cells are not fresh generalization evidence",
            "expanded component state is not parameter, state, compute, or wall-time matched",
            "no G15C, natural-text, optimizer, tokenizer, Spin, scaling, or model-family promotion follows",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "quality"), required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--parent-artifact", type=Path, default=PARENT_ARTIFACT)
    parser.add_argument("--r3-artifact", type=Path, default=R3_ARTIFACT)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    device = torch.device(args.device)
    commit, status = _git_provenance()
    if args.mode == "quality":
        if status:
            raise RuntimeError("G15B-R4 quality requires a clean git tree at start")
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("G15B-R4 quality requires CUDA")
        if torch.cuda.get_device_capability(device) != (7, 5):
            raise RuntimeError("G15B-R4 quality is frozen to SM75")
    report = run(
        mode=args.mode,
        device=device,
        parent_path=args.parent_artifact,
        r3_path=args.r3_artifact,
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
    "LWW_ARMS",
    "factor_ownership",
    "factorial_forwards",
    "fp64_algebraic_contract",
    "local_query_position_mask",
]
