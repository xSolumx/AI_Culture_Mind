"""G15B-R5 frozen causal tail-source retained-checkpoint diagnostic."""

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
from . import g15br4_ownership_background as r4
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
    SELECT_TOKEN,
    WRITE_TOKEN,
    InterleavedBatch,
    generate_interleaved_batch,
)
from .g15br1_event_erase import PRESERVED_CONTROL_NAMES
from .g15br2_collision_erase import overwrite_query_strata
from .g15br_checkpoint_repair import (
    PARENT_ARTIFACT,
    QUALITY_SEEDS,
    ROLE_NAMES,
    ROOT,
    _expected_identity,
    _git_provenance,
    _hidden_controls,
    _load_checkpoint,
    _sync,
    local_write_event_mask,
    temporal_observability_witness,
)
from .model import HybridMemoryLM

PROTOCOL = ROOT / "G15BR5_CAUSAL_TAIL_SOURCE_PROTOCOL_2026-08-26.md"
R4_ARTIFACT = ROOT / "artifacts/g15br4_ownership_background_sm75_2026-08-26.json"
EXPECTED_R4_SHA256 = "921d45e3c492e172fae62064120e9e051dca2965bacc44891268b135d8cef26e"

INTERVENTIONS = (
    "learned",
    "erase_free_no_reset_bgplus",
    "h_no_reset_bgminus",
    "c_no_reset_bgminus",
    "b_no_reset_bgminus",
    "h_lww_bgplus",
    "h_lww_bgminus",
    "c_lww_bgplus",
    "c_lww_bgminus",
    "b_lww_bgplus",
    "b_lww_bgminus",
)
LWW_ARMS = (
    "h_lww_bgplus",
    "h_lww_bgminus",
    "c_lww_bgplus",
    "c_lww_bgminus",
    "b_lww_bgplus",
    "b_lww_bgminus",
)
HISTORY_ARMS = ("h_lww_bgplus", "h_lww_bgminus")
CURRENT_ARMS = ("c_lww_bgplus", "c_lww_bgminus")
BIAS_ARMS = ("b_lww_bgplus", "b_lww_bgminus")
MATCHING_CONTROL = {
    "h_lww_bgplus": "erase_free_no_reset_bgplus",
    "h_lww_bgminus": "h_no_reset_bgminus",
    "c_lww_bgplus": "erase_free_no_reset_bgplus",
    "c_lww_bgminus": "c_no_reset_bgminus",
    "b_lww_bgplus": "erase_free_no_reset_bgplus",
    "b_lww_bgminus": "b_no_reset_bgminus",
}
STRATA = r3.STRATA
SourceMode = Literal["h", "c", "b"]


def local_completed_write_tail_mask(token_ids: torch.Tensor) -> torch.Tensor:
    """Decode t+1 after a complete local write/select triple from tokens only."""

    if token_ids.ndim != 2 or token_ids.shape[1] < 1:
        raise ValueError("token_ids must have nonempty shape (batch,length)")
    mask = torch.zeros_like(token_ids, dtype=torch.bool)
    if token_ids.shape[1] < 4:
        return mask
    marker = token_ids[:, :-3].eq(WRITE_TOKEN) | token_ids[:, :-3].eq(SELECT_TOKEN)
    key = token_ids[:, 1:-2].ge(PAYLOAD_START) & token_ids[:, 1:-2].lt(
        PAYLOAD_START + PAYLOAD_COUNT
    )
    value = token_ids[:, 2:-1].ge(PAYLOAD_START) & token_ids[:, 2:-1].lt(
        PAYLOAD_START + PAYLOAD_COUNT
    )
    mask[:, 3:] = marker & key & value
    return mask


def _audit_tail_mask(batch: InterleavedBatch) -> torch.Tensor:
    mask = torch.zeros_like(batch.token_ids, dtype=torch.bool)
    positions = batch.write_positions + 1
    valid = positions < batch.length
    rows = torch.arange(batch.batch_size, device=batch.token_ids.device)[:, None]
    rows = rows.expand_as(positions)
    mask[rows[valid], positions[valid]] = True
    return mask


def _preactivations_from_expanded(block: Any, expanded: torch.Tensor) -> dict[str, Any]:
    """Split a width-four depthwise convolution before its nonlinearity."""

    if block.local_conv is None:
        raise ValueError("G15B-R5 requires the frozen width-four local convolution")
    conv = block.local_conv.conv
    if (
        conv.kernel_size != (4,)
        or conv.stride != (1,)
        or conv.dilation != (1,)
        or conv.groups != block.local_conv.width
        or conv.in_channels != block.local_conv.width
        or conv.out_channels != block.local_conv.width
    ):
        raise RuntimeError("G15B-R5 requires exact width-four depthwise convolution")
    full_preactivation, _ = block.local_conv(expanded, None, None)
    weight = conv.weight[:, 0, :]
    bias_parameter = conv.bias
    if bias_parameter is None:
        raise RuntimeError("G15B-R5 requires the frozen convolution bias")
    bias = bias_parameter.view(1, 1, -1)
    channels = expanded.transpose(1, 2)
    history_weight = conv.weight.clone()
    history_weight[:, :, -1] = 0.0
    cache = channels.new_zeros(
        channels.shape[0], channels.shape[1], block.local_conv.cache_width
    )
    padded = torch.cat((cache, channels), dim=-1)
    history = F.conv1d(
        padded,
        history_weight,
        bias=None,
        groups=conv.groups,
    ).transpose(1, 2)
    current = expanded * weight[:, -1].view(1, 1, -1)
    reconstruction_residual = float(
        (full_preactivation - (bias + history + current)).abs().max()
    )
    return {
        "bias": bias,
        "full_preactivation": full_preactivation,
        "history_preactivation": bias + history,
        "current_preactivation": bias + current,
        "bias_preactivation": bias.expand_as(full_preactivation),
        "preactivation_reconstruction_residual": reconstruction_residual,
    }


def _source_context(model: HybridMemoryLM, token_ids: torch.Tensor) -> dict[str, Any]:
    """Expose exact full/history/current/bias convolution source inputs."""

    if model.config.layer_plan != ("spin_dirac",) or len(model.blocks) != 1:
        raise RuntimeError("G15B-R5 requires the exact one-block Spin-Dirac shell")
    hidden, outer_gate, mixed, mixer, learned_controls = _hidden_controls(
        model, token_ids
    )
    block = model.blocks[0]
    expanded, replay_outer_gate = block.input_projection(
        block.mixer_norm(hidden)
    ).chunk(2, dim=-1)
    if not torch.equal(outer_gate, replay_outer_gate):
        raise RuntimeError("outer-gate replay changed")
    split = _preactivations_from_expanded(block, expanded)
    full_mixed = F.silu(split["full_preactivation"])
    if not torch.equal(full_mixed, mixed):
        raise RuntimeError("local-convolution replay changed")
    return {
        "hidden": hidden,
        "outer_gate": outer_gate,
        "mixed": mixed,
        "mixer": mixer,
        "learned_controls": learned_controls,
        "expanded": expanded,
        "full_preactivation": split["full_preactivation"],
        "history_preactivation": split["history_preactivation"],
        "current_preactivation": split["current_preactivation"],
        "bias_preactivation": split["bias_preactivation"],
        "history_mixed": F.silu(split["history_preactivation"]),
        "current_mixed": F.silu(split["current_preactivation"]),
        "bias_mixed": F.silu(split["bias_preactivation"]),
        "preactivation_reconstruction_residual": split[
            "preactivation_reconstruction_residual"
        ],
    }


def _injection_from_mixed(mixer: Any, mixed: torch.Tensor) -> torch.Tensor:
    controls = mixer._controls(mixed, None)
    _, key, value, _, write, _, _ = controls
    return (key.unsqueeze(-1) * (write * value).unsqueeze(-2)).transpose(1, 2)


def source_ownership(
    batch: InterleavedBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return value owner, tail owner, and the common reset mask."""

    value_owner, reset_mask = r4.factor_ownership(batch, "v")
    vt_owner, vt_reset = r4.factor_ownership(batch, "vt")
    if not torch.equal(reset_mask, vt_reset):
        raise RuntimeError("V and VT reset masks changed")
    tail_owner = torch.where(value_owner.eq(0), vt_owner, torch.zeros_like(vt_owner))
    if bool((value_owner.gt(0) & tail_owner.gt(0)).any()):
        raise RuntimeError("value and tail ownership overlap")
    if not torch.equal(tail_owner.gt(0), _audit_tail_mask(batch)):
        raise RuntimeError("tail ownership does not match the audit tail mask")
    return value_owner, tail_owner, reset_mask


def _split_component_injection(
    full_injection: torch.Tensor,
    source_injection: torch.Tensor,
    value_owner: torch.Tensor,
    tail_owner: torch.Tensor,
    components: int,
) -> tuple[torch.Tensor, float]:
    """Put source in the key component and exact full-source in background."""

    if full_injection.shape != source_injection.shape:
        raise ValueError("full and source injections must have the same shape")
    batch, heads, length, width, _ = full_injection.shape
    background_owner = torch.zeros_like(value_owner)
    background_weights = F.one_hot(background_owner, num_classes=components).to(
        full_injection.dtype
    )
    component = (
        full_injection[:, None]
        * background_weights.permute(0, 2, 1)[:, :, None, :, None, None]
    )
    for owner, injection in (
        (value_owner, full_injection),
        (tail_owner, source_injection),
    ):
        active = owner.gt(0)
        owner_weights = F.one_hot(owner, num_classes=components).to(injection.dtype)
        delta = (owner_weights - background_weights) * active[..., None].to(
            injection.dtype
        )
        component = component + (
            injection[:, None] * delta.permute(0, 2, 1)[:, :, None, :, None, None]
        )
    expected = (batch, components, heads, length, width, width)
    if component.shape != expected:
        raise RuntimeError(
            f"component injection has shape {component.shape}, not {expected}"
        )
    residual = float((component.sum(dim=1) - full_injection).abs().max())
    return component, residual


def _assignment_residual(
    component: torch.Tensor,
    full_injection: torch.Tensor,
    source_injection: torch.Tensor,
    value_owner: torch.Tensor,
    tail_owner: torch.Tensor,
) -> float:
    """Verify full value writes and exact source/residual tail assignments."""

    residuals = [0.0]
    for owner, expected_key, expected_background in (
        (value_owner, full_injection, torch.zeros_like(full_injection)),
        (tail_owner, source_injection, full_injection - source_injection),
    ):
        active = owner.gt(0)
        rows, positions = active.nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        owners = owner[rows, positions]
        observed_key = component[rows, owners, :, positions]
        observed_background = component[rows, 0, :, positions]
        residuals.append(
            float((observed_key - expected_key[rows, :, positions]).abs().max())
        )
        residuals.append(
            float(
                (observed_background - expected_background[rows, :, positions])
                .abs()
                .max()
            )
        )
    return max(residuals)


def _component_source_reads(
    mixer: Any,
    controls: list[torch.Tensor],
    shared_transitions: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    source_injection: torch.Tensor,
    value_owner: torch.Tensor,
    tail_owner: torch.Tensor,
    reset_mask: torch.Tensor,
    *,
    replace: bool,
) -> dict[str, Any]:
    query = controls[0]
    left, right, full_injection = shared_transitions
    batch, heads, length, width, _ = full_injection.shape
    components = reset_mask.shape[1]
    component_injection, injection_sum_residual = _split_component_injection(
        full_injection,
        source_injection,
        value_owner,
        tail_owner,
        components,
    )
    assignment_residual = _assignment_residual(
        component_injection,
        full_injection,
        source_injection,
        value_owner,
        tail_owner,
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
        full_injection.new_zeros(batch * components, heads, width, width),
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
            full_injection,
            full_injection.new_zeros(batch, heads, width, width),
        )
        state_residual = float((monolithic_states - full_states).abs().max())
    return {
        "full_read": full_read,
        "key_read": key_read,
        "background_read": background_read,
        "background_relation_residual": relation_residual,
        "injection_sum_residual": injection_sum_residual,
        "assignment_residual": assignment_residual,
        "state_residual": state_residual,
        "shared_transition_tensor_ids": tuple(
            id(tensor) for tensor in shared_transitions
        ),
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


def source_forwards(model: HybridMemoryLM, batch: InterleavedBatch) -> dict[str, Any]:
    """Compute the frozen history/current source arms from shared controls."""

    context = _source_context(model, batch.token_ids)
    mixer = context["mixer"]
    learned_controls = context["learned_controls"]
    learned_read, _ = mixer.forward_controls(*learned_controls, scan_mode="parallel")
    component_controls = list(learned_controls)
    component_controls[3] = torch.zeros_like(component_controls[3])
    _, key, value, erase, write, retention, coordinates = component_controls
    left, right, full_injection, _ = mixer._transitions(
        key, value, erase, write, retention, coordinates, None
    )
    shared_transitions = (left, right, full_injection)
    shared_transition_tensor_ids = tuple(id(tensor) for tensor in shared_transitions)
    query_mask = r4.local_query_position_mask(batch.token_ids)
    value_owner, tail_owner, reset_mask = source_ownership(batch)
    source_injections = {
        "h": _injection_from_mixed(mixer, context["history_mixed"]),
        "c": _injection_from_mixed(mixer, context["current_mixed"]),
        "b": _injection_from_mixed(mixer, context["bias_mixed"]),
    }
    reads: dict[tuple[str, bool], dict[str, Any]] = {}
    for source in ("h", "c", "b"):
        for replace in (False, True):
            reads[(source, replace)] = _component_source_reads(
                mixer,
                component_controls,
                shared_transitions,
                source_injections[source],
                value_owner,
                tail_owner,
                reset_mask,
                replace=replace,
            )
    shared_full_transition_controls = all(
        row["shared_transition_tensor_ids"] == shared_transition_tensor_ids
        for row in reads.values()
    )
    selected_reads = {
        "learned": learned_read,
        "erase_free_no_reset_bgplus": reads[("h", False)]["full_read"],
        "h_no_reset_bgminus": _query_background_read(
            reads[("h", False)], query_mask, include_background=False
        ),
        "c_no_reset_bgminus": _query_background_read(
            reads[("c", False)], query_mask, include_background=False
        ),
        "b_no_reset_bgminus": _query_background_read(
            reads[("b", False)], query_mask, include_background=False
        ),
        "h_lww_bgplus": reads[("h", True)]["full_read"],
        "h_lww_bgminus": _query_background_read(
            reads[("h", True)], query_mask, include_background=False
        ),
        "c_lww_bgplus": reads[("c", True)]["full_read"],
        "c_lww_bgminus": _query_background_read(
            reads[("c", True)], query_mask, include_background=False
        ),
        "b_lww_bgplus": reads[("b", True)]["full_read"],
        "b_lww_bgminus": _query_background_read(
            reads[("b", True)], query_mask, include_background=False
        ),
    }
    logits = {
        name: r3._finish_forward(
            model,
            context["hidden"],
            context["outer_gate"],
            context["mixed"],
            selected_reads[name],
            mixer,
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
        "forward_context": (
            context["hidden"],
            context["outer_gate"],
            context["mixed"],
            mixer,
        ),
        "query_mask": query_mask,
        "reads": reads,
        "source_injections": source_injections,
        "full_injection": full_injection,
        "shared_full_transition_controls": shared_full_transition_controls,
        "preactivation_reconstruction_residual": context[
            "preactivation_reconstruction_residual"
        ],
    }


def _explicit_injection(mixer: Any, mixed: torch.Tensor) -> torch.Tensor:
    """Evaluate only the key/value/write injection projections in mixed dtype."""

    def linear(layer: Any) -> torch.Tensor:
        return F.linear(
            mixed,
            layer.weight.to(dtype=mixed.dtype, device=mixed.device),
            (
                None
                if layer.bias is None
                else layer.bias.to(dtype=mixed.dtype, device=mixed.device)
            ),
        )

    batch, length, _ = mixed.shape
    shape = (batch, length, mixer.config.heads, 8)
    key = F.normalize(
        linear(mixer.key_projection).view(shape),
        dim=-1,
        eps=mixer.config.norm_epsilon,
    )
    value = linear(mixer.value_projection).view(shape)
    if mixer.config.bound_values:
        norm_squared = value.square().sum(dim=-1, keepdim=True)
        value = value / torch.sqrt(1.0 + norm_squared)
    gate_shape = (
        (batch, length, mixer.config.heads, 1)
        if mixer.config.gate_mode == "equivariant_scalar"
        else shape
    )
    write = torch.sigmoid(linear(mixer.write_projection)).view(gate_shape)
    return (key.unsqueeze(-1) * (write * value).unsqueeze(-2)).transpose(1, 2)


def convolution_structure_contract(
    model: HybridMemoryLM, batch: InterleavedBatch
) -> dict[str, Any]:
    """Check tap orientation, full/chunk parity, and preactivation reconstruction."""

    context = _source_context(model, batch.token_ids)
    block = model.blocks[0]
    if block.local_conv is None:
        raise RuntimeError("missing local convolution")
    conv = block.local_conv.conv
    expanded = context["expanded"]
    full, _ = block.local_conv(expanded, None, None)
    cut = max(1, expanded.shape[1] // 3)
    first, cache = block.local_conv(expanded[:, :cut], None, None)
    second, _ = block.local_conv(expanded[:, cut:], cache, None)
    chunk_residual = float((full - torch.cat((first, second), dim=1)).abs().max())

    target = 4
    impulse_residuals: list[float] = []
    bias_parameter = conv.bias
    if bias_parameter is None:
        raise RuntimeError("missing convolution bias")
    for tap in range(4):
        impulse = expanded.new_zeros(1, 7, block.local_conv.width)
        source_position = target - (3 - tap)
        impulse[:, source_position] = 1.0
        output, _ = block.local_conv(impulse, None, None)
        observed = output[0, target] - bias_parameter
        expected = conv.weight[:, 0, tap]
        impulse_residuals.append(float((observed - expected).abs().max()))
    maximum_impulse = max(impulse_residuals)
    maximum = max(
        float(context["preactivation_reconstruction_residual"]),
        chunk_residual,
        maximum_impulse,
    )
    return {
        "kernel_size": list(conv.kernel_size),
        "stride": list(conv.stride),
        "dilation": list(conv.dilation),
        "groups": conv.groups,
        "expanded_width": block.local_conv.width,
        "full_sequence_reconstruction_residual": context[
            "preactivation_reconstruction_residual"
        ],
        "arbitrary_chunk_residual": chunk_residual,
        "per_tap_impulse_residuals": impulse_residuals,
        "maximum_residual": maximum,
        "passed": maximum <= 2e-6,
    }


def source_locality_witness(
    model: HybridMemoryLM, batch: InterleavedBatch
) -> dict[str, Any]:
    """Prove intended source invariances and reject degenerate attribution."""

    tails = _audit_tail_mask(batch)
    positions = tails[0].nonzero(as_tuple=False).flatten()
    positions = positions[positions + 1 < batch.length]
    if positions.numel() == 0:
        raise RuntimeError("source-locality witness requires a nonfinal write tail")
    position = int(positions[0])
    context = _source_context(model, batch.token_ids[:1])
    block = model.blocks[0]
    mixer = context["mixer"]
    expanded = context["expanded"].clone()
    direction = torch.linspace(
        -1.0,
        1.0,
        expanded.shape[-1],
        dtype=expanded.dtype,
        device=expanded.device,
    ).view(1, -1)

    current_changed_expanded = expanded.clone()
    current_changed_expanded[:, position] += 0.75 * direction
    history_changed_expanded = expanded.clone()
    history_changed_expanded[:, position - 1] -= 0.75 * direction
    future_changed_expanded = expanded.clone()
    future_changed_expanded[:, position + 1] += 0.75 * direction
    base = _preactivations_from_expanded(block, expanded)
    current_changed = _preactivations_from_expanded(block, current_changed_expanded)
    history_changed = _preactivations_from_expanded(block, history_changed_expanded)
    future_changed = _preactivations_from_expanded(block, future_changed_expanded)

    def injection(split: dict[str, Any], source: str) -> torch.Tensor:
        return _injection_from_mixed(mixer, F.silu(split[f"{source}_preactivation"]))[
            :, :, position
        ]

    invariance = {
        "history_preactivation_under_current_perturbation": float(
            (
                base["history_preactivation"][:, position]
                - current_changed["history_preactivation"][:, position]
            )
            .abs()
            .max()
        ),
        "history_injection_under_current_perturbation": float(
            (injection(base, "history") - injection(current_changed, "history"))
            .abs()
            .max()
        ),
        "current_preactivation_under_history_perturbation": float(
            (
                base["current_preactivation"][:, position]
                - history_changed["current_preactivation"][:, position]
            )
            .abs()
            .max()
        ),
        "current_injection_under_history_perturbation": float(
            (injection(base, "current") - injection(history_changed, "current"))
            .abs()
            .max()
        ),
        "history_preactivation_under_future_perturbation": float(
            (
                base["history_preactivation"][:, position]
                - future_changed["history_preactivation"][:, position]
            )
            .abs()
            .max()
        ),
        "current_preactivation_under_future_perturbation": float(
            (
                base["current_preactivation"][:, position]
                - future_changed["current_preactivation"][:, position]
            )
            .abs()
            .max()
        ),
    }
    effects = {
        "current_preactivation_current_perturbation": float(
            (
                base["current_preactivation"][:, position]
                - current_changed["current_preactivation"][:, position]
            )
            .abs()
            .max()
        ),
        "current_injection_current_perturbation": float(
            (injection(base, "current") - injection(current_changed, "current"))
            .abs()
            .max()
        ),
        "history_preactivation_history_perturbation": float(
            (
                base["history_preactivation"][:, position]
                - history_changed["history_preactivation"][:, position]
            )
            .abs()
            .max()
        ),
        "history_injection_history_perturbation": float(
            (injection(base, "history") - injection(history_changed, "history"))
            .abs()
            .max()
        ),
        "history_injection_above_bias": float(
            (injection(base, "history") - injection(base, "bias")).abs().max()
        ),
        "current_injection_above_bias": float(
            (injection(base, "current") - injection(base, "bias")).abs().max()
        ),
    }
    changed_tokens = batch.token_ids[:1].clone()
    future_token = int(changed_tokens[0, position + 1])
    changed_tokens[0, position + 1] = PAYLOAD_START + (
        (future_token - PAYLOAD_START + 1) % PAYLOAD_COUNT
    )
    tail_prefix_invariant = torch.equal(
        local_completed_write_tail_mask(batch.token_ids[:1])[:, : position + 1],
        local_completed_write_tail_mask(changed_tokens)[:, : position + 1],
    )
    changed_token_context = _source_context(model, changed_tokens)
    future_token_source_residuals = {
        source: float(
            (
                context[f"{source}_mixed"][:, position]
                - changed_token_context[f"{source}_mixed"][:, position]
            )
            .abs()
            .max()
        )
        for source in ("history", "current", "bias")
    }
    maximum_future_token_source_residual = max(future_token_source_residuals.values())
    maximum_invariance = max(invariance.values())
    minimum_effect = min(effects.values())
    return {
        "position": position,
        "invariance_residuals": invariance,
        "nondegenerate_effects": effects,
        "maximum_invariance_residual": maximum_invariance,
        "minimum_nondegenerate_effect": minimum_effect,
        "future_token_tail_prefix_invariant": tail_prefix_invariant,
        "future_token_source_residuals": future_token_source_residuals,
        "maximum_future_token_source_residual": maximum_future_token_source_residual,
        "passed": (
            maximum_invariance <= 5e-7
            and minimum_effect >= 1e-6
            and tail_prefix_invariant
            and maximum_future_token_source_residual <= 5e-7
        ),
    }


def fp64_algebraic_contract(
    model: HybridMemoryLM, batch: InterleavedBatch
) -> dict[str, Any]:
    """Verify source/residual and decomposed affine scans independently in FP64."""

    context = _source_context(model, batch.token_ids)
    block = model.blocks[0]
    if block.local_conv is None:
        raise RuntimeError("missing local convolution")
    expanded64 = context["expanded"].double()
    channels64 = expanded64.transpose(1, 2)
    cache64 = channels64.new_zeros(
        channels64.shape[0], channels64.shape[1], block.local_conv.cache_width
    )
    padded64 = torch.cat((cache64, channels64), dim=-1)
    weight64 = block.local_conv.conv.weight.double()
    bias_parameter = block.local_conv.conv.bias
    if bias_parameter is None:
        raise RuntimeError("missing local-convolution bias")
    bias64 = bias_parameter.double().view(1, 1, -1)
    full_preactivation64 = F.conv1d(
        padded64,
        weight64,
        bias=bias_parameter.double(),
        groups=block.local_conv.width,
    ).transpose(1, 2)
    history_weight64 = weight64.clone()
    history_weight64[:, :, -1] = 0.0
    history64 = F.conv1d(
        padded64,
        history_weight64,
        bias=None,
        groups=block.local_conv.width,
    ).transpose(1, 2)
    current64 = expanded64 * weight64[:, 0, -1].view(1, 1, -1)
    preactivation_residual = float(
        (full_preactivation64 - (bias64 + history64 + current64)).abs().max()
    )
    mixer = context["mixer"]
    explicit_full64 = _explicit_injection(mixer, F.silu(full_preactivation64))
    explicit_sources64 = {
        "h": _explicit_injection(mixer, F.silu(bias64 + history64)),
        "c": _explicit_injection(mixer, F.silu(bias64 + current64)),
        "b": _explicit_injection(mixer, F.silu(bias64.expand_as(full_preactivation64))),
    }
    explicit_residual_identity = max(
        float((source + (explicit_full64 - source) - explicit_full64).abs().max())
        for source in explicit_sources64.values()
    )

    controls = list(context["learned_controls"])
    controls[3] = torch.zeros_like(controls[3])
    query, key, value, erase, write, retention, coordinates = controls
    left, right, full_injection, _ = mixer._transitions(
        key, value, erase, write, retention, coordinates, None
    )
    independently_projected_full = _injection_from_mixed(mixer, context["mixed"])
    independent_full_injection_residual = float(
        (full_injection - independently_projected_full).abs().max()
    )
    left64 = left.double()
    right64 = right.double()
    # The scored path above is intentionally FP32.  The algebraic contract must
    # instead consume the independently reconstructed convolution and Phi
    # projections, rather than merely casting scored injections to FP64.
    injection64 = explicit_full64
    query64 = query.double()
    zero = injection64.new_zeros(injection64.shape[0], injection64.shape[1], 8, 8)
    monolithic_parallel, _ = mixer._parallel_states(left64, right64, injection64, zero)
    monolithic_recurrent, _ = mixer._recurrent_states(
        left64, right64, injection64, zero
    )
    maximum_recurrent_parallel = float(
        (monolithic_parallel - monolithic_recurrent).abs().max()
    )
    value_owner, tail_owner, reset_mask = source_ownership(batch)
    maximum_component_sum = 0.0
    maximum_component_recurrent_parallel = 0.0
    maximum_background_relation = 0.0
    maximum_injection_sum = 0.0
    source_reports: dict[str, Any] = {}
    fp64_scored_projection_deltas = {
        "full": float((explicit_full64 - full_injection.double()).abs().max()),
        **{
            source: float(
                (
                    explicit_sources64[source]
                    - _injection_from_mixed(
                        mixer,
                        context[
                            {
                                "h": "history_mixed",
                                "c": "current_mixed",
                                "b": "bias_mixed",
                            }[source]
                        ],
                    ).double()
                )
                .abs()
                .max()
            )
            for source in ("h", "c", "b")
        },
    }
    batch_size, heads, length, width, _ = injection64.shape
    components = reset_mask.shape[1]
    for source_name, source_injection in explicit_sources64.items():
        component_injection, injection_sum = _split_component_injection(
            injection64,
            source_injection,
            value_owner,
            tail_owner,
            components,
        )
        maximum_injection_sum = max(maximum_injection_sum, injection_sum)
        left_component = left64[:, None].expand(
            batch_size, components, heads, length, width, width
        )
        right_component = right64[:, None].expand_as(left_component)
        report: dict[str, float] = {"injection_sum_residual": injection_sum}
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
                report["no_reset_component_sum_residual"] = sum_residual
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
            report[f"{prefix}_recurrent_parallel_residual"] = parity
            report[f"{prefix}_background_relation_residual"] = relation
        source_reports[source_name] = report
    maximum = max(
        preactivation_residual,
        explicit_residual_identity,
        maximum_recurrent_parallel,
        maximum_component_sum,
        maximum_component_recurrent_parallel,
        maximum_background_relation,
        maximum_injection_sum,
        independent_full_injection_residual,
    )
    return {
        "dtype": "float64",
        "preactivation_reconstruction_residual": preactivation_residual,
        "explicit_source_residual_identity": explicit_residual_identity,
        "monolithic_recurrent_parallel_residual": maximum_recurrent_parallel,
        "maximum_component_sum_residual": maximum_component_sum,
        "maximum_component_recurrent_parallel_residual": (
            maximum_component_recurrent_parallel
        ),
        "maximum_background_relation_residual": maximum_background_relation,
        "maximum_injection_sum_residual": maximum_injection_sum,
        "independent_full_injection_residual": (independent_full_injection_residual),
        "independent_fp64_injections_used": True,
        "fp64_scored_projection_maximum_absolute_deltas": (
            fp64_scored_projection_deltas
        ),
        "maximum_residual": maximum,
        "passed": maximum <= 1e-10,
        "sources": source_reports,
    }


def _new_tail_statistics() -> dict[str, Any]:
    return {
        source: {
            name: {
                "count": 0,
                "source_norm_sum": 0.0,
                "residual_norm_sum": 0.0,
                "full_norm_sum": 0.0,
                "source_bias_cosine_sum": 0.0,
            }
            for name in ROLE_NAMES.values()
        }
        for source in ("h", "c", "b")
    }


def _accumulate_tail_statistics(
    statistics: dict[str, Any],
    batch: InterleavedBatch,
    full_injection: torch.Tensor,
    source_injections: dict[str, torch.Tensor],
) -> None:
    positions = batch.write_positions + 1
    valid = positions < batch.length
    rows = torch.arange(batch.batch_size, device=batch.token_ids.device)[:, None]
    rows = rows.expand_as(positions)
    selected_rows = rows[valid]
    selected_positions = positions[valid]
    roles = batch.roles[selected_rows, selected_positions]
    full = full_injection[selected_rows, :, selected_positions].flatten(start_dim=1)
    bias = source_injections["b"][selected_rows, :, selected_positions].flatten(
        start_dim=1
    )
    for source_name, source_injection in source_injections.items():
        source = source_injection[selected_rows, :, selected_positions].flatten(
            start_dim=1
        )
        residual = full - source
        source_norm = source.norm(dim=-1)
        residual_norm = residual.norm(dim=-1)
        full_norm = full.norm(dim=-1)
        cosine = F.cosine_similarity(source, bias, dim=-1, eps=1e-12)
        for role, role_name in ROLE_NAMES.items():
            mask = roles.eq(role)
            count = int(mask.sum())
            if not count:
                continue
            row = statistics[source_name][role_name]
            row["count"] += count
            row["source_norm_sum"] += float(source_norm[mask].sum())
            row["residual_norm_sum"] += float(residual_norm[mask].sum())
            row["full_norm_sum"] += float(full_norm[mask].sum())
            row["source_bias_cosine_sum"] += float(cosine[mask].sum())


def _finish_tail_statistics(statistics: dict[str, Any]) -> dict[str, Any]:
    for sources in statistics.values():
        for row in sources.values():
            count = row["count"]
            if count:
                row["mean_source_norm"] = row.pop("source_norm_sum") / count
                row["mean_residual_norm"] = row.pop("residual_norm_sum") / count
                row["mean_full_norm"] = row.pop("full_norm_sum") / count
                row["mean_source_bias_cosine"] = (
                    row.pop("source_bias_cosine_sum") / count
                )
            else:
                row["mean_source_norm"] = None
                row["mean_residual_norm"] = None
                row["mean_full_norm"] = None
                row["mean_source_bias_cosine"] = None
                row.pop("source_norm_sum")
                row.pop("residual_norm_sum")
                row.pop("full_norm_sum")
                row.pop("source_bias_cosine_sum")
    return statistics


def _new_integrity() -> dict[str, Any]:
    return {
        "local_write_batches_checked": 0,
        "local_tail_batches_checked": 0,
        "local_query_batches_checked": 0,
        "source_assignment_batches_checked": 0,
        "ordinary_model_forward_maximum_absolute_logit_residual": 0.0,
        "preactivation_reconstruction_maximum_absolute_residual": 0.0,
        "no_reset_state_residual": {source: 0.0 for source in ("h", "c", "b")},
        "no_reset_query_predictions_equal": {
            source: True for source in ("h", "c", "b")
        },
        "injection_sum_maximum_absolute_residual": {
            f"{source}_{'lww' if replace else 'no_reset'}": 0.0
            for source in ("h", "c", "b")
            for replace in (False, True)
        },
        "source_assignment_maximum_absolute_residual": {
            f"{source}_{'lww' if replace else 'no_reset'}": 0.0
            for source in ("h", "c", "b")
            for replace in (False, True)
        },
        "background_relation_maximum_absolute_read_residual": {
            f"{source}_{'lww' if replace else 'no_reset'}": 0.0
            for source in ("h", "c", "b")
            for replace in (False, True)
        },
        "lww_no_reset_no_overwrite_predictions_equal": {arm: True for arm in LWW_ARMS},
        "lww_no_reset_before_overwrite_predictions_equal": {
            arm: True for arm in LWW_ARMS
        },
        "preserved_controls": {
            name: {"bitwise_equal": True, "maximum_absolute_residual": 0.0}
            for name in PRESERVED_CONTROL_NAMES
        },
        "shared_full_transition_controls": True,
        "tail_role_counts": {name: 0 for name in ROLE_NAMES.values()},
        "tail_source_statistics": _new_tail_statistics(),
        "writes_without_in_range_tail": 0,
        "component_capacities": {},
        "convolution_structure_contract": None,
        "source_locality_witness": None,
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
            generation_task = "overwrite" if task == "overwrite_guard" else task
            batch_size = _evaluation_batch_size(
                generation_task, decisions=decisions, cap=batch_cap
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
                        generation_task,  # type: ignore[arg-type]
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
                if not torch.equal(
                    local_completed_write_tail_mask(batch.token_ids),
                    _audit_tail_mask(batch),
                ):
                    raise RuntimeError("write-tail target is not locally observable")
                integrity["local_tail_batches_checked"] += 1
                expected_query = torch.zeros_like(batch.write_event_mask)
                expected_query.scatter_(1, batch.query_positions, True)
                if not torch.equal(
                    r4.local_query_position_mask(batch.token_ids), expected_query
                ):
                    raise RuntimeError("query positions are not locally observable")
                integrity["local_query_batches_checked"] += 1
                _value_owner, tail_owner, reset_mask = source_ownership(batch)
                if int(reset_mask.sum()) != int(batch.write_event_mask.sum()):
                    raise RuntimeError("reset count changed")
                if not torch.equal(tail_owner.gt(0), _audit_tail_mask(batch)):
                    raise RuntimeError("tail owner changed")
                integrity["source_assignment_batches_checked"] += 1

                tail_positions = batch.write_positions + 1
                valid_tail = tail_positions < batch.length
                rows = torch.arange(batch.batch_size, device=device)[:, None]
                rows = rows.expand_as(tail_positions)
                tail_roles = batch.roles[rows[valid_tail], tail_positions[valid_tail]]
                integrity["writes_without_in_range_tail"] += int((~valid_tail).sum())
                for role, name in ROLE_NAMES.items():
                    integrity["tail_role_counts"][name] += int(
                        tail_roles.eq(role).sum()
                    )

                if integrity["convolution_structure_contract"] is None:
                    integrity["convolution_structure_contract"] = (
                        convolution_structure_contract(model, batch)
                    )
                    integrity["source_locality_witness"] = source_locality_witness(
                        model, batch
                    )
                    integrity["fp64_algebraic_contract"] = fp64_algebraic_contract(
                        model, batch
                    )

                result = source_forwards(model, batch)
                logits = result["logits"]
                integrity["finite_logits"] = bool(integrity["finite_logits"]) and all(
                    bool(torch.isfinite(value).all()) for value in logits.values()
                )
                integrity["shared_full_transition_controls"] = bool(
                    integrity["shared_full_transition_controls"]
                ) and bool(result["shared_full_transition_controls"])
                integrity["preactivation_reconstruction_maximum_absolute_residual"] = (
                    max(
                        float(
                            integrity[
                                "preactivation_reconstruction_maximum_absolute_residual"
                            ]
                        ),
                        float(result["preactivation_reconstruction_residual"]),
                    )
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
                for source in ("h", "c", "b"):
                    no_reset_reads = result["reads"][(source, False)]
                    integrity["no_reset_state_residual"][source] = max(
                        float(integrity["no_reset_state_residual"][source]),
                        float(no_reset_reads["state_residual"]),
                    )
                    no_reset_logits = r3._finish_forward(
                        model,
                        hidden,
                        outer_gate,
                        mixed,
                        no_reset_reads["full_read"],
                        mixer,
                    )
                    no_reset_prediction = _gather_time(
                        no_reset_logits, batch.query_positions
                    ).argmax(-1)
                    integrity["no_reset_query_predictions_equal"][source] = bool(
                        integrity["no_reset_query_predictions_equal"][source]
                    ) and torch.equal(monolithic_prediction, no_reset_prediction)
                    for replace in (False, True):
                        key_name = f"{source}_{'lww' if replace else 'no_reset'}"
                        reads = result["reads"][(source, replace)]
                        for field, integrity_field in (
                            (
                                "injection_sum_residual",
                                "injection_sum_maximum_absolute_residual",
                            ),
                            (
                                "assignment_residual",
                                "source_assignment_maximum_absolute_residual",
                            ),
                            (
                                "background_relation_residual",
                                "background_relation_maximum_absolute_read_residual",
                            ),
                        ):
                            integrity[integrity_field][key_name] = max(
                                float(integrity[integrity_field][key_name]),
                                float(reads[field]),
                            )
                capacity = result["reads"][("h", False)]["capacity"]
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
                _accumulate_tail_statistics(
                    integrity["tail_source_statistics"],
                    batch,
                    result["full_injection"],
                    result["source_injections"],
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
                    else:
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
    integrity["tail_source_statistics"] = _finish_tail_statistics(
        integrity["tail_source_statistics"]
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
        maximum_fp32_injection = max(
            *integrity["injection_sum_maximum_absolute_residual"].values(),
            *integrity["source_assignment_maximum_absolute_residual"].values(),
        )
        maximum_no_reset_state = max(integrity["no_reset_state_residual"].values())
        maximum_background_relation = max(
            integrity["background_relation_maximum_absolute_read_residual"].values()
        )
        if not (
            integrity["ordinary_model_forward_maximum_absolute_logit_residual"] == 0.0
            and integrity["preactivation_reconstruction_maximum_absolute_residual"]
            <= 2e-6
            and maximum_fp32_injection <= 5e-6
            and maximum_no_reset_state <= 2e-6
            and maximum_background_relation <= 2e-6
            and all(integrity["no_reset_query_predictions_equal"].values())
            and all(integrity["lww_no_reset_no_overwrite_predictions_equal"].values())
            and all(
                integrity["lww_no_reset_before_overwrite_predictions_equal"].values()
            )
            and integrity["preserved_controls_bitwise_equal"]
            and integrity["shared_full_transition_controls"]
            and integrity["finite_logits"]
            and integrity["convolution_structure_contract"] is not None
            and integrity["convolution_structure_contract"]["passed"]
            and integrity["source_locality_witness"] is not None
            and integrity["source_locality_witness"]["passed"]
            and integrity["fp64_algebraic_contract"] is not None
            and integrity["fp64_algebraic_contract"]["passed"]
            and integrity["local_write_batches_checked"] > 0
            and integrity["local_tail_batches_checked"] > 0
            and integrity["local_query_batches_checked"] > 0
            and integrity["source_assignment_batches_checked"] > 0
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
    bias_arm = "b_lww_bgminus" if arm.endswith("bgminus") else "b_lww_bgplus"
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
        if arm in HISTORY_ARMS:
            checks[f"{overwrite_name}:aggregate:versus_bias"] = (
                accuracy[arm] - accuracy[bias_arm] >= 0.05
            )
            checks[f"{overwrite_name}:post_same:versus_bias"] = (
                same[arm] is not None
                and same[bias_arm] is not None
                and same[arm] - same[bias_arm] >= 0.05
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
            if arm in HISTORY_ARMS:
                checks[f"{overwrite_name}:seed{seed}:aggregate:versus_bias"] = (
                    seed_accuracy[arm]["query_accuracy"]
                    - seed_accuracy[bias_arm]["query_accuracy"]
                    >= 0.02
                )
                checks[f"{overwrite_name}:seed{seed}:post_same:versus_bias"] = (
                    seed_same[arm] is not None
                    and seed_same[bias_arm] is not None
                    and seed_same[arm] - seed_same[bias_arm] >= 0.02
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
        for value in cell.get("r4_reference_replay", {}).values()
    ]
    replay_passed = max(replay_residuals, default=math.inf) <= 1e-12
    fingerprint_passed = all(
        cell.get("r4_batch_fingerprint_match") is True
        for report in seed_reports
        for cell in report["evaluation"]["cells"].values()
    )
    observability_passed = all(
        row["observability_witness"]["passed"] for row in seed_reports
    )
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
        and fingerprint_passed
        and observability_passed
        and runtime_passed
    ]
    passed_history_arms = [arm for arm in HISTORY_ARMS if arm in passed_arms]
    passed_current_arms = [arm for arm in CURRENT_ARMS if arm in passed_arms]
    passed_bias_arms = [arm for arm in BIAS_ARMS if arm in passed_arms]
    selected_training_law = None
    if "h_lww_bgminus" in passed_history_arms:
        selected_training_law = (
            "history-derived pending-write injection with a protected "
            "background-free transaction read"
        )
        decision = (
            "support a separately frozen fresh pending-write/commit training "
            "screen with a mandatory protected-read control"
        )
    elif "h_lww_bgplus" in passed_history_arms:
        decision = (
            "do not train; history evidence remains coupled to full residual "
            "background at query time"
        )
    elif passed_current_arms and not passed_history_arms:
        decision = (
            "stop retained-checkpoint tail repair; passing association depends "
            "on the new token rather than a clean write-history source"
        )
    else:
        decision = (
            "stop retained-checkpoint tail repair; no bias-separated "
            "background-free history source passes"
        )
    return {
        "r4_reference_replay_maximum_absolute_residual": max(
            replay_residuals, default=math.inf
        ),
        "r4_reference_replay_passed": replay_passed,
        "r4_batch_fingerprints_passed": fingerprint_passed,
        "observability_witness_passed": observability_passed,
        "runtime_integrity_passed": runtime_passed,
        "three_seed_means": means,
        "arm_checks": checks,
        "performance_passed": performance_passed,
        "passed_arms": passed_arms,
        "passed_history_arms": passed_history_arms,
        "passed_current_arms": passed_current_arms,
        "passed_bias_arms": passed_bias_arms,
        "history_background_free_passed": ("h_lww_bgminus" in passed_history_arms),
        "non_unique_background_free_source": (
            "h_lww_bgminus" in passed_history_arms
            and "c_lww_bgminus" in passed_current_arms
        ),
        "selected_training_law": selected_training_law,
        "passed": selected_training_law is not None,
        "decision": decision,
    }


def _validate_r4(path: Path) -> tuple[dict[str, Any], str]:
    actual_sha256 = _sha256(path)
    if actual_sha256 != EXPECTED_R4_SHA256:
        raise ValueError("R4 artifact hash does not match the frozen input")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not (
        report.get("mode") == "quality"
        and report.get("evidentiary") is True
        and report.get("git_status_at_start") == []
        and report.get("environment", {}).get("compute_capability") == [7, 5]
    ):
        raise ValueError("R4 is not the sealed exact-SM75 quality artifact")
    adjudication = report.get("adjudication", {})
    if not (
        adjudication.get("passed") is False
        and adjudication.get("passed_value_arms") == []
        and adjudication.get("passed_tail_arms") == ["vt_lww_bgplus", "vt_lww_bgminus"]
        and adjudication.get("decision")
        == (
            "do not train; passing behavior remains dependent on ambiguous "
            "value-plus-tail ownership"
        )
    ):
        raise ValueError("R4 does not select the frozen R5 question")
    return report, actual_sha256


def _r4_seed_map(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result = {int(row["seed"]): row for row in report["seed_reports"]}
    if set(result) != set(QUALITY_SEEDS):
        raise ValueError("R4 seed reports do not match the frozen quality seeds")
    return result


def run(
    *,
    mode: Literal["smoke", "quality"],
    device: torch.device,
    parent_path: Path,
    r4_path: Path,
    checkpoint_directory: Path,
    commit: str,
    status_at_start: list[str],
) -> dict[str, Any]:
    parent, parent_sha256 = r3._validate_parent(parent_path)
    r4_report, r4_sha256 = _validate_r4(r4_path)
    r4_seeds = _r4_seed_map(r4_report)
    expected = _expected_identity(parent)
    seeds = QUALITY_SEEDS if mode == "quality" else QUALITY_SEEDS[:1]
    decisions = 4096 if mode == "quality" else 16
    batch_cap = 16 if mode == "quality" else 2
    seed_reports = []
    started = time.perf_counter()
    replay_mapping = {
        "learned": "learned",
        "erase_free_no_reset_bgplus": "erase_free_no_reset_bgplus",
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
        if mode == "quality":
            r4_cells = r4_seeds[seed]["evaluation"]["cells"]
            for name, cell in evaluation["cells"].items():
                reference_cell = r4_cells[name]
                cell["r4_batch_fingerprint_match"] = (
                    cell["batch_fingerprint_sha256"]
                    == reference_cell["batch_fingerprint_sha256"]
                )
                cell["r4_reference_replay"] = {}
                for r4_intervention, r5_intervention in replay_mapping.items():
                    reference = reference_cell["interventions"][r4_intervention]
                    replay = cell["interventions"][r5_intervention]
                    for metric in (
                        "query_accuracy",
                        "exact_episode_accuracy",
                        "bits_per_query",
                    ):
                        cell["r4_reference_replay"][f"{r4_intervention}:{metric}"] = (
                            abs(replay[metric] - reference[metric])
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
        ROOT / "g15br4_ownership_background.py",
        ROOT / "g15br3_logical_component.py",
        ROOT / "g15b_interleaved_cohort.py",
        ROOT / "g15b_interleaved_tasks.py",
        ROOT / "spin_dirac_memory.py",
        ROOT / "model.py",
    )
    return {
        "schema_version": 1,
        "experiment": "G15B-R5 causal tail-source decomposition diagnostic",
        "mode": mode,
        "evidentiary": mode == "quality" and not status_at_start,
        "git_commit_at_start": commit,
        "git_status_at_start": status_at_start,
        "elapsed_wall_seconds": time.perf_counter() - started,
        "parent_g15b_artifact": str(parent_path),
        "parent_g15b_sha256": parent_sha256,
        "parent_r4_artifact": str(r4_path),
        "parent_r4_sha256": r4_sha256,
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
            "tail_sources": ["strict_history", "current_token", "bias_only"],
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
            "logical key components still use commissioned task metadata",
            "history attribution concerns injection conditional on the unchanged full-token transition",
            "the signed residual is an attribution device, not a bounded write law",
            "BG+ arms remain coupled to current-token and nonlinear residual background",
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
    parser.add_argument("--r4-artifact", type=Path, default=R4_ARTIFACT)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    device = torch.device(args.device)
    commit, status = _git_provenance()
    if args.mode == "quality":
        if status:
            raise RuntimeError("G15B-R5 quality requires a clean git tree at start")
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("G15B-R5 quality requires CUDA")
        if torch.cuda.get_device_capability(device) != (7, 5):
            raise RuntimeError("G15B-R5 quality is frozen to SM75")
    report = run(
        mode=args.mode,
        device=device,
        parent_path=args.parent_artifact,
        r4_path=args.r4_artifact,
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
    "BIAS_ARMS",
    "CURRENT_ARMS",
    "HISTORY_ARMS",
    "INTERVENTIONS",
    "LWW_ARMS",
    "convolution_structure_contract",
    "fp64_algebraic_contract",
    "local_completed_write_tail_mask",
    "source_forwards",
    "source_locality_witness",
    "source_ownership",
]
