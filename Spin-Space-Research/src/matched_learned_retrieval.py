"""Matched learned-address overwrite benchmark for Program 03.

This module implements Task A from the frozen matched-retrieval protocol.  It
uses one event stream for direct slots, triality-bound slots, learned-key
delta memory, oracle-key delta memory, and additive fast weights.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from pathlib import Path

import torch
from torch.nn import functional as F

from schurscan_delta_memory import (
    DeltaTransition,
    compose_delta,
    delta_read,
    delta_write_transitions,
    recurrent_delta_states,
    scanned_delta_states,
    value_transport_transitions,
)
from spin8_continuous_alias import (
    TEST_RADIUS,
    AliasWorld,
    FrozenKeyPolicy,
    FrozenSlotPolicy,
    alias_diagnostics,
    key_diagnostics,
    train_key_policy,
    train_slot_policy,
)
from spin8_learned_address import DIMENSION, KEYS, SLOTS, teacher_actions
from spin8_triality_lift import (
    triality_bind,
    triality_tensor,
    triality_unbind_negative,
)
from spin8_triality_memory import random_unit

DTYPE = torch.float64
PRIMARY_VARIANTS = (
    "direct_slot_joint",
    "triality_slot_joint",
    "delta_chunk_joint",
    "delta_chunk_oracle",
    "fast_weight_joint",
)


def perturb_unit(
    value: torch.Tensor,
    norm: float,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float]:
    if norm < 0:
        raise ValueError("perturbation norm must be nonnegative")
    if norm == 0:
        return value, 0.0
    noise = torch.randn(
        value.shape, generator=generator, dtype=value.dtype, device=value.device
    )
    noise = noise - (noise * value).sum(dim=-1, keepdim=True) * value
    noise = F.normalize(noise, dim=-1)
    perturbed = F.normalize(value + norm * noise, dim=-1)
    observed = float((perturbed - value).norm(dim=-1).mean())
    return perturbed, observed


def perturb_route(
    route: torch.Tensor,
    norm: float,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float]:
    if norm < 0:
        raise ValueError("perturbation norm must be nonnegative")
    if norm == 0:
        return route, 0.0
    noise = torch.randn(
        route.shape, generator=generator, dtype=route.dtype, device=route.device
    )
    noise = noise - noise.mean(dim=-1, keepdim=True)
    noise = F.normalize(noise, dim=-1)
    perturbed = (route + norm * noise).clamp_min(0.0)
    total = perturbed.sum(dim=-1, keepdim=True)
    fallback = total <= torch.finfo(route.dtype).eps
    perturbed = perturbed / total.clamp_min(torch.finfo(route.dtype).eps)
    perturbed = torch.where(fallback, route, perturbed)
    observed = float((perturbed - route).norm(dim=-1).mean())
    return perturbed, observed


def concatenate_transitions(
    transitions: list[DeltaTransition],
) -> DeltaTransition:
    if not transitions:
        raise ValueError("at least one transition is required")
    return DeltaTransition(
        key_action=torch.cat([row.key_action for row in transitions], dim=1),
        value_action=torch.cat([row.value_action for row in transitions], dim=1),
        drive=torch.cat([row.drive for row in transitions], dim=1),
    )


def retrieval_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float | int]:
    cosine = F.cosine_similarity(prediction, target, dim=-1)
    relative = (prediction - target).square().sum(dim=-1) / target.square().sum(
        dim=-1
    ).clamp_min(torch.finfo(target.dtype).tiny)
    return {
        "queries": int(cosine.numel()),
        "mean_query_cosine": float(cosine.mean()),
        "minimum_query_cosine": float(cosine.min()),
        "mean_relative_squared_error": float(relative.mean()),
        "maximum_relative_squared_error": float(relative.max()),
    }


@torch.no_grad()
def evaluate_overwrite_depth(
    slot_policy: FrozenSlotPolicy,
    key_policy: FrozenKeyPolicy,
    *,
    seed: int,
    overwrite_depth: int,
    radius: float,
    perturbation_norm: float,
    transport: bool,
    batch_size: int = 32,
    chunk_size: int = 16,
) -> dict[str, object]:
    if overwrite_depth < 1:
        raise ValueError("overwrite_depth must be positive")
    device = torch.device("cpu")
    world = AliasWorld.create(seed, dtype=DTYPE, device=device)
    actions = teacher_actions(seed, dtype=DTYPE, device=device)
    vector_actions, positive_actions, negative_actions = actions.unbind(dim=1)
    rho = triality_tensor(dtype=DTYPE)
    generator = torch.Generator().manual_seed(
        810_000
        + 101 * seed
        + 17 * overwrite_depth
        + round(1000 * radius)
        + round(10_000 * perturbation_norm)
        + int(transport)
    )
    alias_generator = torch.Generator().manual_seed(
        820_000
        + 103 * seed
        + 19 * overwrite_depth
        + round(1000 * radius)
        + round(10_000 * perturbation_norm)
        + int(transport)
    )
    perturb_generator = torch.Generator().manual_seed(
        830_000
        + 107 * seed
        + 23 * overwrite_depth
        + round(1000 * radius)
        + round(10_000 * perturbation_norm)
        + int(transport)
    )

    direct = torch.zeros(batch_size, SLOTS, DIMENSION, dtype=DTYPE)
    triality = torch.zeros_like(direct)
    fast_weight = torch.zeros(batch_size, DIMENSION, DIMENSION, dtype=DTYPE)
    truth = torch.zeros(batch_size, KEYS, DIMENSION, dtype=DTYPE)
    geometric_keys = random_unit(
        (batch_size, KEYS, DIMENSION),
        generator=generator,
        dtype=DTYPE,
    )
    learned_delta_steps: list[DeltaTransition] = []
    oracle_delta_steps: list[DeltaTransition] = []
    route_perturbations: list[float] = []
    key_perturbations: list[float] = []
    ones = torch.ones(batch_size, 1, dtype=DTYPE)

    for round_index in range(overwrite_depth + 1):
        for label in range(KEYS):
            labels = torch.full((batch_size,), label, dtype=torch.long)
            aliases = world.sample(labels, radius=radius, generator=alias_generator)
            route = slot_policy.routes(aliases, labels, side="write")
            route, route_change = perturb_route(
                route, perturbation_norm, generator=perturb_generator
            )
            learned_key = key_policy.keys(aliases, side="write")
            learned_key, key_change = perturb_unit(
                learned_key, perturbation_norm, generator=perturb_generator
            )
            value = random_unit(
                (batch_size, DIMENSION),
                generator=generator,
                dtype=DTYPE,
            )
            payload = triality_bind(geometric_keys[:, label], value, rho)
            direct = (1.0 - route[..., None]) * direct + route[..., None] * value[
                :, None
            ]
            triality = (1.0 - route[..., None]) * triality + route[..., None] * payload[
                :, None
            ]
            fast_weight = fast_weight + learned_key[..., None] * value[:, None, :]
            learned_delta_steps.append(
                delta_write_transitions(learned_key[:, None], value[:, None], ones)
            )
            oracle_key = F.one_hot(labels, KEYS).to(dtype=DTYPE)
            oracle_delta_steps.append(
                delta_write_transitions(oracle_key[:, None], value[:, None], ones)
            )
            truth[:, label] = value
            route_perturbations.append(route_change)
            key_perturbations.append(key_change)

        if transport and round_index < overwrite_depth:
            tokens = torch.randint(actions.shape[0], (batch_size,), generator=generator)
            vector = vector_actions[tokens]
            positive = positive_actions[tokens]
            negative = negative_actions[tokens]
            direct = torch.einsum("bij,bhj->bhi", negative, direct)
            triality = torch.einsum("bij,bhj->bhi", vector, triality)
            fast_weight = torch.einsum("bkv,bwv->bkw", fast_weight, negative)
            truth = torch.einsum("bij,bkj->bki", negative, truth)
            geometric_keys = torch.einsum("bij,bkj->bki", positive, geometric_keys)
            learned_delta_steps.append(
                value_transport_transitions(negative[:, None], key_dimension=DIMENSION)
            )
            oracle_delta_steps.append(
                value_transport_transitions(negative[:, None], key_dimension=DIMENSION)
            )

    learned_transition = concatenate_transitions(learned_delta_steps)
    oracle_transition = concatenate_transitions(oracle_delta_steps)
    initial = torch.zeros(batch_size, DIMENSION, DIMENSION, dtype=DTYPE)
    learned_states = scanned_delta_states(
        learned_transition,
        initial,
        backend="chunkwise",
        chunk_size=chunk_size,
    )
    oracle_states = scanned_delta_states(
        oracle_transition,
        initial,
        backend="chunkwise",
        chunk_size=chunk_size,
    )
    learned_recurrent = recurrent_delta_states(learned_transition, initial)
    oracle_recurrent = recurrent_delta_states(oracle_transition, initial)
    learned_delta = learned_states[:, -1]
    oracle_delta = oracle_states[:, -1]

    predictions: dict[str, list[torch.Tensor]] = {name: [] for name in PRIMARY_VARIANTS}
    targets: list[torch.Tensor] = []
    for label in range(KEYS):
        labels = torch.full((batch_size,), label, dtype=torch.long)
        aliases = world.sample(labels, radius=radius, generator=alias_generator)
        route = slot_policy.routes(aliases, labels, side="query")
        route, route_change = perturb_route(
            route, perturbation_norm, generator=perturb_generator
        )
        learned_query = key_policy.keys(aliases, side="query")
        learned_query, key_change = perturb_unit(
            learned_query, perturbation_norm, generator=perturb_generator
        )
        candidates = triality_unbind_negative(
            geometric_keys[:, label, None], triality, rho
        )
        predictions["direct_slot_joint"].append((route[..., None] * direct).sum(dim=1))
        predictions["triality_slot_joint"].append(
            (route[..., None] * candidates).sum(dim=1)
        )
        predictions["delta_chunk_joint"].append(
            delta_read(learned_delta, learned_query)
        )
        predictions["delta_chunk_oracle"].append(
            delta_read(oracle_delta, F.one_hot(labels, KEYS).to(dtype=DTYPE))
        )
        predictions["fast_weight_joint"].append(delta_read(fast_weight, learned_query))
        targets.append(truth[:, label])
        route_perturbations.append(route_change)
        key_perturbations.append(key_change)

    target = torch.stack(targets, dim=1).flatten(0, 1)
    metrics = {
        name: retrieval_metrics(torch.stack(rows, dim=1).flatten(0, 1), target)
        for name, rows in predictions.items()
    }
    direct_prediction = torch.stack(predictions["direct_slot_joint"], dim=1)
    triality_prediction = torch.stack(predictions["triality_slot_joint"], dim=1)
    return {
        "overwrite_depth": overwrite_depth,
        "radius": radius,
        "requested_perturbation_norm": perturbation_norm,
        "mean_observed_route_perturbation": sum(route_perturbations)
        / len(route_perturbations),
        "mean_observed_key_perturbation": sum(key_perturbations)
        / len(key_perturbations),
        "transport": transport,
        "batch_size": batch_size,
        "query_count": batch_size * KEYS,
        "transition_length": int(learned_transition.drive.shape[1]),
        "chunk_size": chunk_size,
        "state_scalars": {name: 64 for name in PRIMARY_VARIANTS},
        "metrics": metrics,
        "diagnostics": {
            "learned_delta_chunk_recurrent_max_abs_error": float(
                (learned_states - learned_recurrent).abs().max()
            ),
            "oracle_delta_chunk_recurrent_max_abs_error": float(
                (oracle_states - oracle_recurrent).abs().max()
            ),
            "direct_triality_prediction_max_abs_gap": float(
                (direct_prediction - triality_prediction).abs().max()
            ),
        },
    }


@torch.no_grad()
def evaluate_stream_length(
    slot_policy: FrozenSlotPolicy,
    key_policy: FrozenKeyPolicy,
    *,
    seed: int,
    length: int,
    radius: float,
    perturbation_norm: float,
    transport: bool,
    batch_size: int = 8,
    chunk_size: int = 64,
) -> dict[str, object]:
    """Evaluate a shared long stream with hot and cold address cohorts.

    The first eight tokens initialize every semantic key.  Later tokens only
    overwrite four hot keys, while queries continue to cover all eight keys.
    Cold-key queries therefore measure long retention under interference; hot
    queries measure repeated overwrite.  When enabled, a supplied value action
    is applied before every write and folded into the same delta transition, so
    ``length`` remains a token count rather than an internal-operation count.
    """

    if length < 64:
        raise ValueError("stream length must be at least 64")
    if batch_size * (length - 32) < 256:
        raise ValueError("every stream cell must contain at least 256 queries")
    device = torch.device("cpu")
    world = AliasWorld.create(seed, dtype=DTYPE, device=device)
    actions = teacher_actions(seed, dtype=DTYPE, device=device)
    vector_actions, positive_actions, negative_actions = actions.unbind(dim=1)
    rho = triality_tensor(dtype=DTYPE)
    base_seed = (
        840_000
        + 109 * seed
        + 29 * length
        + round(1000 * radius)
        + round(10_000 * perturbation_norm)
        + int(transport)
    )
    generator = torch.Generator().manual_seed(base_seed)
    alias_generator = torch.Generator().manual_seed(base_seed + 10_000)
    perturb_generator = torch.Generator().manual_seed(base_seed + 20_000)

    direct = torch.zeros(batch_size, SLOTS, DIMENSION, dtype=DTYPE)
    triality = torch.zeros_like(direct)
    fast_weight = torch.zeros(batch_size, DIMENSION, DIMENSION, dtype=DTYPE)
    truth = torch.zeros(batch_size, KEYS, DIMENSION, dtype=DTYPE)
    geometric_keys = random_unit(
        (batch_size, KEYS, DIMENSION), generator=generator, dtype=DTYPE
    )
    permutations = torch.stack(
        [torch.randperm(KEYS, generator=generator) for _ in range(batch_size)]
    )
    batch_index = torch.arange(batch_size)
    batch_offsets = torch.arange(batch_size)
    ones = torch.ones(batch_size, 1, dtype=DTYPE)

    learned_delta_steps: list[DeltaTransition] = []
    oracle_delta_steps: list[DeltaTransition] = []
    learned_queries: list[torch.Tensor] = []
    oracle_queries: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    query_is_cold: list[torch.Tensor] = []
    direct_predictions: list[torch.Tensor] = []
    triality_predictions: list[torch.Tensor] = []
    fast_weight_predictions: list[torch.Tensor] = []
    route_perturbations: list[float] = []
    key_perturbations: list[float] = []

    for position in range(length):
        if position < KEYS:
            write_labels = permutations[:, position]
        else:
            hot_column = (position - KEYS + batch_offsets) % (KEYS // 2)
            write_labels = permutations[batch_index, hot_column]
        query_column = (3 * position + batch_offsets) % KEYS
        query_labels = permutations[batch_index, query_column]

        if transport:
            tokens = torch.randint(actions.shape[0], (batch_size,), generator=generator)
            vector = vector_actions[tokens]
            positive = positive_actions[tokens]
            negative = negative_actions[tokens]
            direct = torch.einsum("bij,bhj->bhi", negative, direct)
            triality = torch.einsum("bij,bhj->bhi", vector, triality)
            fast_weight = torch.einsum("bkv,bwv->bkw", fast_weight, negative)
            truth = torch.einsum("bij,bkj->bki", negative, truth)
            geometric_keys = torch.einsum("bij,bkj->bki", positive, geometric_keys)
            transport_step = value_transport_transitions(
                negative[:, None], key_dimension=DIMENSION
            )

        write_alias = world.sample(
            write_labels, radius=radius, generator=alias_generator
        )
        write_route = slot_policy.routes(write_alias, write_labels, side="write")
        write_route, route_change = perturb_route(
            write_route, perturbation_norm, generator=perturb_generator
        )
        learned_write_key = key_policy.keys(write_alias, side="write")
        learned_write_key, key_change = perturb_unit(
            learned_write_key,
            perturbation_norm,
            generator=perturb_generator,
        )
        value = random_unit((batch_size, DIMENSION), generator=generator, dtype=DTYPE)
        geometric_write_key = geometric_keys[batch_index, write_labels]
        payload = triality_bind(geometric_write_key, value, rho)
        direct = (1.0 - write_route[..., None]) * direct + write_route[
            ..., None
        ] * value[:, None]
        triality = (1.0 - write_route[..., None]) * triality + write_route[
            ..., None
        ] * payload[:, None]
        fast_weight = fast_weight + learned_write_key[..., None] * value[:, None, :]
        truth[batch_index, write_labels] = value
        oracle_write_key = F.one_hot(write_labels, KEYS).to(dtype=DTYPE)
        learned_write = delta_write_transitions(
            learned_write_key[:, None], value[:, None], ones
        )
        oracle_write = delta_write_transitions(
            oracle_write_key[:, None], value[:, None], ones
        )
        if transport:
            learned_write = compose_delta(learned_write, transport_step)
            oracle_write = compose_delta(oracle_write, transport_step)
        learned_delta_steps.append(learned_write)
        oracle_delta_steps.append(oracle_write)

        query_alias = world.sample(
            query_labels, radius=radius, generator=alias_generator
        )
        query_route = slot_policy.routes(query_alias, query_labels, side="query")
        query_route, query_route_change = perturb_route(
            query_route, perturbation_norm, generator=perturb_generator
        )
        learned_query = key_policy.keys(query_alias, side="query")
        learned_query, query_key_change = perturb_unit(
            learned_query,
            perturbation_norm,
            generator=perturb_generator,
        )
        candidates = triality_unbind_negative(
            geometric_keys[batch_index, query_labels, None], triality, rho
        )
        direct_predictions.append((query_route[..., None] * direct).sum(dim=1))
        triality_predictions.append((query_route[..., None] * candidates).sum(dim=1))
        fast_weight_predictions.append(delta_read(fast_weight, learned_query))
        learned_queries.append(learned_query)
        oracle_queries.append(F.one_hot(query_labels, KEYS).to(dtype=DTYPE))
        targets.append(truth[batch_index, query_labels])
        # The first four columns in each per-batch permutation define hot keys.
        query_is_cold.append(query_column >= (KEYS // 2))
        route_perturbations.extend((route_change, query_route_change))
        key_perturbations.extend((key_change, query_key_change))

    learned_transition = concatenate_transitions(learned_delta_steps)
    oracle_transition = concatenate_transitions(oracle_delta_steps)
    initial = torch.zeros(batch_size, DIMENSION, DIMENSION, dtype=DTYPE)
    learned_states = scanned_delta_states(
        learned_transition,
        initial,
        backend="chunkwise",
        chunk_size=chunk_size,
    )
    oracle_states = scanned_delta_states(
        oracle_transition,
        initial,
        backend="chunkwise",
        chunk_size=chunk_size,
    )
    learned_recurrent = recurrent_delta_states(learned_transition, initial)
    oracle_recurrent = recurrent_delta_states(oracle_transition, initial)

    prediction_tensors = {
        "direct_slot_joint": torch.stack(direct_predictions, dim=1),
        "triality_slot_joint": torch.stack(triality_predictions, dim=1),
        "delta_chunk_joint": delta_read(
            learned_states, torch.stack(learned_queries, dim=1)
        ),
        "delta_chunk_oracle": delta_read(
            oracle_states, torch.stack(oracle_queries, dim=1)
        ),
        "fast_weight_joint": torch.stack(fast_weight_predictions, dim=1),
    }
    target_tensor = torch.stack(targets, dim=1)
    cold_tensor = torch.stack(query_is_cold, dim=1)
    report_mask = torch.arange(length)[None] >= 32
    report_mask = report_mask.expand(batch_size, -1)
    hot_mask = report_mask & ~cold_tensor
    cold_mask = report_mask & cold_tensor

    def cohort_metrics(mask: torch.Tensor) -> dict[str, dict[str, float | int]]:
        return {
            name: retrieval_metrics(prediction[mask], target_tensor[mask])
            for name, prediction in prediction_tensors.items()
        }

    return {
        "length": length,
        "radius": radius,
        "requested_perturbation_norm": perturbation_norm,
        "mean_observed_route_perturbation": sum(route_perturbations)
        / len(route_perturbations),
        "mean_observed_key_perturbation": sum(key_perturbations)
        / len(key_perturbations),
        "transport": transport,
        "batch_size": batch_size,
        "query_count": int(report_mask.sum()),
        "hot_query_count": int(hot_mask.sum()),
        "cold_query_count": int(cold_mask.sum()),
        "chunk_size": chunk_size,
        "state_scalars": {name: 64 for name in PRIMARY_VARIANTS},
        "metrics": cohort_metrics(report_mask),
        "hot_metrics": cohort_metrics(hot_mask),
        "cold_metrics": cohort_metrics(cold_mask),
        "diagnostics": {
            "learned_delta_chunk_recurrent_max_abs_error": float(
                (learned_states - learned_recurrent).abs().max()
            ),
            "oracle_delta_chunk_recurrent_max_abs_error": float(
                (oracle_states - oracle_recurrent).abs().max()
            ),
            "direct_triality_prediction_max_abs_gap": float(
                (
                    prediction_tensors["direct_slot_joint"]
                    - prediction_tensors["triality_slot_joint"]
                )
                .abs()
                .max()
            ),
        },
    }


def run_seed(
    seed: int,
    *,
    device: torch.device,
    steps_per_stage: int,
    depths: Iterable[int],
    radii: Iterable[float],
    perturbations: Iterable[float],
    transports: Iterable[bool],
    batch_size: int,
    chunk_size: int,
    lengths: Iterable[int],
    length_radii: Iterable[float],
    length_perturbations: Iterable[float],
    length_batch_size: int,
    sample_budgets: Iterable[int],
) -> dict[str, object]:
    start = time.perf_counter()
    slot_training = train_slot_policy(
        "learned_both_joint",
        seed=seed,
        device=device,
        steps_per_stage=steps_per_stage,
    )
    slot_seconds = time.perf_counter() - start
    start = time.perf_counter()
    key_training = train_key_policy(
        seed=seed,
        device=device,
        steps_per_stage=steps_per_stage,
    )
    key_seconds = time.perf_counter() - start
    overwrite_cells = [
        evaluate_overwrite_depth(
            slot_training.policy,
            key_training.policy,
            seed=seed,
            overwrite_depth=depth,
            radius=radius,
            perturbation_norm=perturbation,
            transport=transport,
            batch_size=batch_size,
            chunk_size=chunk_size,
        )
        for depth in depths
        for radius in radii
        for perturbation in perturbations
        for transport in transports
    ]
    stream_cells = [
        evaluate_stream_length(
            slot_training.policy,
            key_training.policy,
            seed=seed,
            length=length,
            radius=radius,
            perturbation_norm=perturbation,
            transport=transport,
            batch_size=length_batch_size,
            chunk_size=chunk_size,
        )
        for length in lengths
        for radius in length_radii
        for perturbation in length_perturbations
        for transport in transports
    ]
    world = AliasWorld.create(seed, dtype=DTYPE, device=torch.device("cpu"))
    oracle_slot = FrozenSlotPolicy("oracle_both", None, None)
    ideal_key = FrozenKeyPolicy(world.centers, world.centers)
    oracle_capacity_cells = [
        evaluate_overwrite_depth(
            oracle_slot,
            ideal_key,
            seed=seed,
            overwrite_depth=16,
            radius=0.75,
            perturbation_norm=0.0,
            transport=transport,
            batch_size=batch_size,
            chunk_size=chunk_size,
        )
        for transport in transports
    ]
    sample_efficiency = []
    for budget in sample_budgets:
        if budget < 1:
            raise ValueError("sample-efficiency budgets must be positive")
        if budget == steps_per_stage:
            budget_slot = slot_training
            budget_key = key_training
            budget_slot_seconds = slot_seconds
            budget_key_seconds = key_seconds
        else:
            budget_start = time.perf_counter()
            budget_slot = train_slot_policy(
                "learned_both_joint",
                seed=seed,
                device=device,
                steps_per_stage=budget,
            )
            budget_slot_seconds = time.perf_counter() - budget_start
            budget_start = time.perf_counter()
            budget_key = train_key_policy(
                seed=seed,
                device=device,
                steps_per_stage=budget,
            )
            budget_key_seconds = time.perf_counter() - budget_start
        budget_cells = [
            evaluate_overwrite_depth(
                budget_slot.policy,
                budget_key.policy,
                seed=seed,
                overwrite_depth=4,
                radius=0.75,
                perturbation_norm=0.10,
                transport=transport,
                batch_size=batch_size,
                chunk_size=chunk_size,
            )
            for transport in transports
        ]
        sample_efficiency.append(
            {
                "steps_per_stage": budget,
                "examples_per_encoder_family": 3 * budget * 128,
                "slot_seconds": budget_slot_seconds,
                "key_seconds": budget_key_seconds,
                "slot_final_loss": budget_slot.final_loss,
                "key_final_loss": budget_key.final_loss,
                "cells": budget_cells,
            }
        )
    return {
        "seed": seed,
        "training": {
            "steps_per_stage": steps_per_stage,
            "stages": 3,
            "examples_per_step": 128,
            "total_examples_per_encoder_family": 3 * steps_per_stage * 128,
            "slot_encoder_parameters": 2 * SLOTS * 24,
            "key_encoder_parameters": 2 * DIMENSION * 24,
            "slot_seconds": slot_seconds,
            "key_seconds": key_seconds,
            "slot_final_loss": slot_training.final_loss,
            "slot_final_endpoint_loss": slot_training.final_endpoint_loss,
            "slot_final_balance_loss": slot_training.final_balance_loss,
            "key_final_loss": key_training.final_loss,
            "key_final_endpoint_loss": key_training.final_endpoint_loss,
            "key_final_whitening_loss": key_training.final_whitening_loss,
        },
        "alias_diagnostics": {
            "slot": alias_diagnostics(
                slot_training.policy, seed=seed, radius=TEST_RADIUS
            ),
            "key": key_diagnostics(key_training.policy, seed=seed, radius=TEST_RADIUS),
        },
        "oracle_hard_route_capacity_cells": oracle_capacity_cells,
        "sample_efficiency": sample_efficiency,
        "overwrite_cells": overwrite_cells,
        "stream_cells": stream_cells,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    maximum_chunk_error = max(
        max(
            max(
                float(value)
                for key, value in cell["diagnostics"].items()
                if "chunk" in key
            )
            for cell in row["overwrite_cells"] + row["stream_cells"]
        )
        for row in rows
    )
    minimum_cosines = {
        cohort: {
            name: min(
                float(cell[metric_key][name]["mean_query_cosine"])
                for row in rows
                for cell in row[cell_key]
            )
            for name in PRIMARY_VARIANTS
        }
        for cohort, cell_key, metric_key in (
            ("overwrite", "overwrite_cells", "metrics"),
            ("stream_all", "stream_cells", "metrics"),
            ("stream_hot", "stream_cells", "hot_metrics"),
            ("stream_cold", "stream_cells", "cold_metrics"),
        )
    }
    oracle_delta_passes = sum(
        all(
            float(cell["metrics"]["delta_chunk_oracle"]["mean_query_cosine"])
            >= 1.0 - 1e-10
            for cell in row["overwrite_cells"] + row["stream_cells"]
        )
        for row in rows
    )
    hard_route_gauge_passes = sum(
        all(
            float(cell["diagnostics"]["direct_triality_prediction_max_abs_gap"]) < 1e-10
            for cell in row["oracle_hard_route_capacity_cells"]
        )
        for row in rows
    )
    budgets = sorted(
        {
            int(budget_row["steps_per_stage"])
            for row in rows
            for budget_row in row["sample_efficiency"]
        }
    )

    def sample_values(budget: int, name: str) -> list[float]:
        return [
            float(cell["metrics"][name]["mean_query_cosine"])
            for row in rows
            for budget_row in row["sample_efficiency"]
            if int(budget_row["steps_per_stage"]) == budget
            for cell in budget_row["cells"]
        ]

    sample_efficiency = {
        str(budget): {
            name: {
                "mean_query_cosine": sum(sample_values(budget, name))
                / len(sample_values(budget, name)),
                "minimum_query_cosine": min(sample_values(budget, name)),
            }
            for name in PRIMARY_VARIANTS
        }
        for budget in budgets
    }
    return {
        "seeds": len(rows),
        "minimum_mean_cosine_by_cohort_and_variant": minimum_cosines,
        "oracle_delta_all_cell_passes": oracle_delta_passes,
        "hard_route_direct_triality_gauge_passes": hard_route_gauge_passes,
        "sample_efficiency_by_steps_per_stage": sample_efficiency,
        "maximum_chunk_recurrent_abs_error": maximum_chunk_error,
        "implementation_gate_passed": (
            maximum_chunk_error < 1e-9
            and oracle_delta_passes == len(rows)
            and hard_route_gauge_passes == len(rows)
        ),
        "claim_boundary": {
            "triality_memory_law_advantage_established": False,
            "fused_delta_kernel_compared": False,
            "task_b_cross_view_result_included": False,
        },
    }


def run(
    seeds: list[int],
    *,
    device: torch.device,
    steps_per_stage: int,
    depths: tuple[int, ...],
    radii: tuple[float, ...],
    perturbations: tuple[float, ...],
    transports: tuple[bool, ...],
    batch_size: int,
    chunk_size: int,
    lengths: tuple[int, ...],
    length_radii: tuple[float, ...],
    length_perturbations: tuple[float, ...],
    length_batch_size: int,
    sample_budgets: tuple[int, ...],
) -> dict[str, object]:
    rows = [
        run_seed(
            seed,
            device=device,
            steps_per_stage=steps_per_stage,
            depths=depths,
            radii=radii,
            perturbations=perturbations,
            transports=transports,
            batch_size=batch_size,
            chunk_size=chunk_size,
            lengths=lengths,
            length_radii=length_radii,
            length_perturbations=length_perturbations,
            length_batch_size=length_batch_size,
            sample_budgets=sample_budgets,
        )
        for seed in seeds
    ]
    return {
        "experiment": "matched learned-address overwrite and recall",
        "protocol": "MATCHED_LEARNED_RETRIEVAL_PREREGISTRATION.md Task A",
        "device": str(device),
        "dtype": str(DTYPE),
        "seeds": seeds,
        "grid": {
            "overwrite_depths": depths,
            "radii": radii,
            "perturbations": perturbations,
            "transports": transports,
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "stream_lengths": lengths,
            "stream_radii": length_radii,
            "stream_perturbations": length_perturbations,
            "stream_batch_size": length_batch_size,
            "sample_efficiency_steps_per_stage": sample_budgets,
        },
        "results": rows,
        "summary": summarize(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--steps-per-stage", type=int, default=300)
    parser.add_argument("--depths", type=int, nargs="+", default=(1, 2, 4, 8, 16))
    parser.add_argument("--radii", type=float, nargs="+", default=(0.35, 0.55, 0.75))
    parser.add_argument(
        "--perturbations",
        type=float,
        nargs="+",
        default=(0.0, 0.02, 0.05, 0.10, 0.20),
    )
    parser.add_argument(
        "--transports", choices=("both", "false", "true"), default="both"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=(64, 256, 1024, 2048, 4096)
    )
    parser.add_argument("--length-radii", type=float, nargs="+", default=(0.75,))
    parser.add_argument(
        "--length-perturbations",
        type=float,
        nargs="+",
        default=(0.0, 0.10, 0.20),
    )
    parser.add_argument("--length-batch-size", type=int, default=8)
    parser.add_argument(
        "--sample-budgets", type=int, nargs="+", default=(25, 75, 150, 300)
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/matched_learned_retrieval_task_a_seeds0_9.json"),
    )
    args = parser.parse_args()
    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    transports = (
        (False, True) if args.transports == "both" else (args.transports == "true",)
    )
    report = run(
        args.seeds,
        device=torch.device(device_name),
        steps_per_stage=args.steps_per_stage,
        depths=tuple(args.depths),
        radii=tuple(args.radii),
        perturbations=tuple(args.perturbations),
        transports=transports,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        lengths=tuple(args.lengths),
        length_radii=tuple(args.length_radii),
        length_perturbations=tuple(args.length_perturbations),
        length_batch_size=args.length_batch_size,
        sample_budgets=tuple(dict.fromkeys(args.sample_budgets)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
