"""NSA-inspired hierarchical routing for matched addressed memories.

The experiment deliberately keeps the existing learned categorical router and
compares three parameter-free post-encoding transforms.  Direct slots and the
corrective DeltaRule consume the same transformed route, isolating routing
from the recurrent update law.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from pathlib import Path

import torch
from torch.nn import functional as F

from matched_learned_retrieval import (
    DTYPE,
    concatenate_transitions,
    perturb_route,
    retrieval_metrics,
)
from schurscan_delta_memory import (
    DeltaTransition,
    compose_delta,
    delta_read,
    delta_write_transitions,
    recurrent_delta_states,
    scanned_delta_states,
    value_transport_transitions,
)
from spin8_continuous_alias import AliasWorld, FrozenSlotPolicy, train_slot_policy
from spin8_learned_address import DIMENSION, KEYS, SLOTS, teacher_actions
from spin8_triality_lift import (
    triality_bind,
    triality_tensor,
    triality_unbind_negative,
)
from spin8_triality_memory import random_unit

ROUTE_STRATEGIES = ("dense_soft", "block_top1", "hard_top1")
MEMORY_KINDS = ("direct", "delta")
PRIMARY_VARIANTS = tuple(
    f"{kind}_{strategy}" for strategy in ROUTE_STRATEGIES for kind in MEMORY_KINDS
) + ("direct_oracle", "delta_oracle")
SELECTED_SLOTS = {"dense_soft": 8, "block_top1": 2, "hard_top1": 1}


def transform_route(route: torch.Tensor, strategy: str) -> torch.Tensor:
    """Apply a frozen parameter-free dense, block, or hard route transform."""

    if route.shape[-1] != SLOTS:
        raise ValueError(f"route must have final dimension {SLOTS}")
    if strategy == "dense_soft":
        return route
    if strategy == "block_top1":
        blocks = route.reshape(*route.shape[:-1], SLOTS // 2, 2)
        block_index = blocks.sum(dim=-1).argmax(dim=-1)
        block_mask = F.one_hot(block_index, SLOTS // 2).to(route)[..., :, None]
        selected = (blocks * block_mask).reshape_as(route)
        return selected / selected.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(route.dtype).tiny
        )
    if strategy == "hard_top1":
        return F.one_hot(route.argmax(dim=-1), SLOTS).to(route)
    raise ValueError(f"unknown route strategy: {strategy}")


def route_access_profile(strategy: str) -> dict[str, float | int]:
    if strategy not in SELECTED_SLOTS:
        raise ValueError(f"unknown route strategy: {strategy}")
    selected = SELECTED_SLOTS[strategy]
    return {
        "selected_slots": selected,
        "selected_fraction": selected / SLOTS,
        "ideal_float32_value_payload_bytes_per_access": selected * DIMENSION * 4,
    }


def _key(route: torch.Tensor) -> torch.Tensor:
    return F.normalize(route, dim=-1)


def _metrics(
    predictions: dict[str, torch.Tensor], target: torch.Tensor
) -> dict[str, dict[str, float | int]]:
    return {
        name: retrieval_metrics(prediction, target)
        for name, prediction in predictions.items()
    }


def _scan_delta(
    steps: dict[str, list[DeltaTransition]],
    *,
    batch_size: int,
    chunk_size: int,
) -> tuple[dict[str, torch.Tensor], float]:
    initial = torch.zeros(batch_size, KEYS, DIMENSION, dtype=DTYPE)
    states: dict[str, torch.Tensor] = {}
    maximum_error = 0.0
    for name, rows in steps.items():
        transition = concatenate_transitions(rows)
        scanned = scanned_delta_states(
            transition,
            initial,
            backend="chunkwise",
            chunk_size=chunk_size,
        )
        recurrent = recurrent_delta_states(transition, initial)
        maximum_error = max(maximum_error, float((scanned - recurrent).abs().max()))
        states[name] = scanned
    return states, maximum_error


@torch.no_grad()
def transformed_triality_gauge_diagnostic(
    seed: int, *, steps: int = 23, batch_size: int = 4
) -> dict[str, float]:
    """Replay every transform on exact routes in direct and triality gauges."""

    generator = torch.Generator().manual_seed(870_000 + seed)
    rho = triality_tensor(dtype=DTYPE)
    actions = teacher_actions(seed, dtype=DTYPE, device=torch.device("cpu"))
    vector_actions, positive_actions, negative_actions = actions.unbind(dim=1)
    keys = random_unit(
        (batch_size, KEYS, DIMENSION), generator=generator, dtype=DTYPE
    )
    gaps: dict[str, float] = {}
    for strategy in ROUTE_STRATEGIES:
        direct = torch.zeros(batch_size, SLOTS, DIMENSION, dtype=DTYPE)
        bound = torch.zeros_like(direct)
        moving_keys = keys.clone()
        for position in range(steps):
            token = torch.randint(actions.shape[0], (batch_size,), generator=generator)
            vector = vector_actions[token]
            positive = positive_actions[token]
            negative = negative_actions[token]
            direct = torch.einsum("bij,bhj->bhi", negative, direct)
            bound = torch.einsum("bij,bhj->bhi", vector, bound)
            moving_keys = torch.einsum("bij,bkj->bki", positive, moving_keys)
            labels = (position + torch.arange(batch_size)) % KEYS
            route = transform_route(
                F.one_hot(labels, SLOTS).to(dtype=DTYPE), strategy
            )
            value = random_unit(
                (batch_size, DIMENSION), generator=generator, dtype=DTYPE
            )
            payload = triality_bind(
                moving_keys[torch.arange(batch_size), labels], value, rho
            )
            direct = (1.0 - route[..., None]) * direct + route[..., None] * value[
                :, None
            ]
            bound = (1.0 - route[..., None]) * bound + route[..., None] * payload[
                :, None
            ]
        candidates = triality_unbind_negative(
            moving_keys[:, :, None, :], bound[:, None, :, :], rho
        )
        slot_index = torch.arange(SLOTS)
        recovered = candidates[:, slot_index, slot_index]
        gaps[strategy] = float((recovered - direct).abs().max())
    return gaps


@torch.no_grad()
def evaluate_overwrite_depth(
    policy: FrozenSlotPolicy,
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
    world = AliasWorld.create(seed, dtype=DTYPE, device=torch.device("cpu"))
    actions = teacher_actions(seed, dtype=DTYPE, device=torch.device("cpu"))
    _, _, negative_actions = actions.unbind(dim=1)
    base_seed = (
        880_000
        + 101 * seed
        + 17 * overwrite_depth
        + round(1000 * radius)
        + round(10_000 * perturbation_norm)
        + int(transport)
    )
    generator = torch.Generator().manual_seed(base_seed)
    alias_generator = torch.Generator().manual_seed(base_seed + 10_000)
    perturb_generator = torch.Generator().manual_seed(base_seed + 20_000)
    direct = {
        strategy: torch.zeros(batch_size, SLOTS, DIMENSION, dtype=DTYPE)
        for strategy in ROUTE_STRATEGIES
    }
    direct_oracle = torch.zeros(batch_size, SLOTS, DIMENSION, dtype=DTYPE)
    truth = torch.zeros(batch_size, KEYS, DIMENSION, dtype=DTYPE)
    delta_steps = {strategy: [] for strategy in ROUTE_STRATEGIES}
    delta_steps["oracle"] = []
    ones = torch.ones(batch_size, 1, dtype=DTYPE)
    route_changes: list[float] = []

    for round_index in range(overwrite_depth + 1):
        for label in range(KEYS):
            labels = torch.full((batch_size,), label, dtype=torch.long)
            alias = world.sample(labels, radius=radius, generator=alias_generator)
            base_route = policy.routes(alias, labels, side="write")
            base_route, change = perturb_route(
                base_route, perturbation_norm, generator=perturb_generator
            )
            value = random_unit(
                (batch_size, DIMENSION), generator=generator, dtype=DTYPE
            )
            for strategy in ROUTE_STRATEGIES:
                route = transform_route(base_route, strategy)
                direct[strategy] = (
                    (1.0 - route[..., None]) * direct[strategy]
                    + route[..., None] * value[:, None]
                )
                delta_steps[strategy].append(
                    delta_write_transitions(_key(route)[:, None], value[:, None], ones)
                )
            oracle = F.one_hot(labels, SLOTS).to(dtype=DTYPE)
            direct_oracle = (
                (1.0 - oracle[..., None]) * direct_oracle
                + oracle[..., None] * value[:, None]
            )
            delta_steps["oracle"].append(
                delta_write_transitions(oracle[:, None], value[:, None], ones)
            )
            truth[:, label] = value
            route_changes.append(change)

        if transport and round_index < overwrite_depth:
            tokens = torch.randint(actions.shape[0], (batch_size,), generator=generator)
            negative = negative_actions[tokens]
            truth = torch.einsum("bij,bkj->bki", negative, truth)
            direct_oracle = torch.einsum("bij,bhj->bhi", negative, direct_oracle)
            transition = value_transport_transitions(
                negative[:, None], key_dimension=DIMENSION
            )
            for strategy in ROUTE_STRATEGIES:
                direct[strategy] = torch.einsum(
                    "bij,bhj->bhi", negative, direct[strategy]
                )
                delta_steps[strategy].append(transition)
            delta_steps["oracle"].append(transition)

    states, scan_error = _scan_delta(
        delta_steps, batch_size=batch_size, chunk_size=chunk_size
    )
    final_states = {name: value[:, -1] for name, value in states.items()}
    prediction_rows = {name: [] for name in PRIMARY_VARIANTS}
    targets: list[torch.Tensor] = []
    for label in range(KEYS):
        labels = torch.full((batch_size,), label, dtype=torch.long)
        alias = world.sample(labels, radius=radius, generator=alias_generator)
        base_route = policy.routes(alias, labels, side="query")
        base_route, change = perturb_route(
            base_route, perturbation_norm, generator=perturb_generator
        )
        for strategy in ROUTE_STRATEGIES:
            route = transform_route(base_route, strategy)
            prediction_rows[f"direct_{strategy}"].append(
                (route[..., None] * direct[strategy]).sum(dim=1)
            )
            prediction_rows[f"delta_{strategy}"].append(
                delta_read(final_states[strategy], _key(route))
            )
        oracle = F.one_hot(labels, SLOTS).to(dtype=DTYPE)
        prediction_rows["direct_oracle"].append(
            (oracle[..., None] * direct_oracle).sum(dim=1)
        )
        prediction_rows["delta_oracle"].append(
            delta_read(final_states["oracle"], oracle)
        )
        targets.append(truth[:, label])
        route_changes.append(change)

    predictions = {
        name: torch.stack(rows, dim=1).flatten(0, 1)
        for name, rows in prediction_rows.items()
    }
    target = torch.stack(targets, dim=1).flatten(0, 1)
    return {
        "overwrite_depth": overwrite_depth,
        "radius": radius,
        "requested_perturbation_norm": perturbation_norm,
        "mean_observed_route_perturbation": sum(route_changes) / len(route_changes),
        "transport": transport,
        "batch_size": batch_size,
        "query_count": int(target.shape[0]),
        "state_scalars": {name: 64 for name in PRIMARY_VARIANTS},
        "route_access": {
            strategy: route_access_profile(strategy)
            for strategy in ROUTE_STRATEGIES
        },
        "metrics": _metrics(predictions, target),
        "diagnostics": {
            "maximum_delta_chunk_recurrent_abs_error": scan_error,
            "hard_direct_delta_prediction_max_abs_gap": float(
                (predictions["direct_hard_top1"] - predictions["delta_hard_top1"])
                .abs()
                .max()
            ),
            "oracle_direct_delta_prediction_max_abs_gap": float(
                (predictions["direct_oracle"] - predictions["delta_oracle"])
                .abs()
                .max()
            ),
        },
    }


@torch.no_grad()
def evaluate_stream_length(
    policy: FrozenSlotPolicy,
    *,
    seed: int,
    length: int,
    radius: float,
    perturbation_norm: float,
    transport: bool,
    batch_size: int = 8,
    chunk_size: int = 64,
) -> dict[str, object]:
    if length < 64:
        raise ValueError("stream length must be at least 64")
    if batch_size * (length - 32) < 256:
        raise ValueError("every stream cell must contain at least 256 queries")
    world = AliasWorld.create(seed, dtype=DTYPE, device=torch.device("cpu"))
    actions = teacher_actions(seed, dtype=DTYPE, device=torch.device("cpu"))
    _, _, negative_actions = actions.unbind(dim=1)
    base_seed = (
        890_000
        + 109 * seed
        + 29 * length
        + round(1000 * radius)
        + round(10_000 * perturbation_norm)
        + int(transport)
    )
    generator = torch.Generator().manual_seed(base_seed)
    alias_generator = torch.Generator().manual_seed(base_seed + 10_000)
    perturb_generator = torch.Generator().manual_seed(base_seed + 20_000)
    direct = {
        strategy: torch.zeros(batch_size, SLOTS, DIMENSION, dtype=DTYPE)
        for strategy in ROUTE_STRATEGIES
    }
    direct_oracle = torch.zeros(batch_size, SLOTS, DIMENSION, dtype=DTYPE)
    truth = torch.zeros(batch_size, KEYS, DIMENSION, dtype=DTYPE)
    permutations = torch.stack(
        [torch.randperm(KEYS, generator=generator) for _ in range(batch_size)]
    )
    batch_index = torch.arange(batch_size)
    batch_offsets = torch.arange(batch_size)
    ones = torch.ones(batch_size, 1, dtype=DTYPE)
    delta_steps = {strategy: [] for strategy in ROUTE_STRATEGIES}
    delta_steps["oracle"] = []
    query_keys = {strategy: [] for strategy in ROUTE_STRATEGIES}
    oracle_queries: list[torch.Tensor] = []
    direct_predictions = {strategy: [] for strategy in ROUTE_STRATEGIES}
    oracle_direct_predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    query_is_cold: list[torch.Tensor] = []
    route_changes: list[float] = []

    for position in range(length):
        if position < KEYS:
            write_labels = permutations[:, position]
        else:
            hot_column = (position - KEYS + batch_offsets) % (KEYS // 2)
            write_labels = permutations[batch_index, hot_column]
        query_column = (3 * position + batch_offsets) % KEYS
        query_labels = permutations[batch_index, query_column]

        transition: DeltaTransition | None = None
        if transport:
            tokens = torch.randint(actions.shape[0], (batch_size,), generator=generator)
            negative = negative_actions[tokens]
            truth = torch.einsum("bij,bkj->bki", negative, truth)
            direct_oracle = torch.einsum("bij,bhj->bhi", negative, direct_oracle)
            transition = value_transport_transitions(
                negative[:, None], key_dimension=DIMENSION
            )
            for strategy in ROUTE_STRATEGIES:
                direct[strategy] = torch.einsum(
                    "bij,bhj->bhi", negative, direct[strategy]
                )

        write_alias = world.sample(
            write_labels, radius=radius, generator=alias_generator
        )
        base_write = policy.routes(write_alias, write_labels, side="write")
        base_write, change = perturb_route(
            base_write, perturbation_norm, generator=perturb_generator
        )
        value = random_unit((batch_size, DIMENSION), generator=generator, dtype=DTYPE)
        for strategy in ROUTE_STRATEGIES:
            route = transform_route(base_write, strategy)
            direct[strategy] = (
                (1.0 - route[..., None]) * direct[strategy]
                + route[..., None] * value[:, None]
            )
            write = delta_write_transitions(_key(route)[:, None], value[:, None], ones)
            delta_steps[strategy].append(
                compose_delta(write, transition) if transition is not None else write
            )
        oracle = F.one_hot(write_labels, SLOTS).to(dtype=DTYPE)
        direct_oracle = (
            (1.0 - oracle[..., None]) * direct_oracle
            + oracle[..., None] * value[:, None]
        )
        oracle_write = delta_write_transitions(oracle[:, None], value[:, None], ones)
        delta_steps["oracle"].append(
            compose_delta(oracle_write, transition)
            if transition is not None
            else oracle_write
        )
        truth[batch_index, write_labels] = value

        query_alias = world.sample(
            query_labels, radius=radius, generator=alias_generator
        )
        base_query = policy.routes(query_alias, query_labels, side="query")
        base_query, query_change = perturb_route(
            base_query, perturbation_norm, generator=perturb_generator
        )
        for strategy in ROUTE_STRATEGIES:
            route = transform_route(base_query, strategy)
            direct_predictions[strategy].append(
                (route[..., None] * direct[strategy]).sum(dim=1)
            )
            query_keys[strategy].append(_key(route))
        oracle_query = F.one_hot(query_labels, SLOTS).to(dtype=DTYPE)
        oracle_direct_predictions.append(
            (oracle_query[..., None] * direct_oracle).sum(dim=1)
        )
        oracle_queries.append(oracle_query)
        targets.append(truth[batch_index, query_labels])
        query_is_cold.append(query_column >= (KEYS // 2))
        route_changes.extend((change, query_change))

    states, scan_error = _scan_delta(
        delta_steps, batch_size=batch_size, chunk_size=chunk_size
    )
    predictions: dict[str, torch.Tensor] = {}
    for strategy in ROUTE_STRATEGIES:
        predictions[f"direct_{strategy}"] = torch.stack(
            direct_predictions[strategy], dim=1
        )
        predictions[f"delta_{strategy}"] = delta_read(
            states[strategy], torch.stack(query_keys[strategy], dim=1)
        )
    predictions["direct_oracle"] = torch.stack(oracle_direct_predictions, dim=1)
    predictions["delta_oracle"] = delta_read(
        states["oracle"], torch.stack(oracle_queries, dim=1)
    )
    target = torch.stack(targets, dim=1)
    cold = torch.stack(query_is_cold, dim=1)
    report_mask = (torch.arange(length)[None] >= 32).expand(batch_size, -1)
    hot_mask = report_mask & ~cold
    cold_mask = report_mask & cold

    def cohort(mask: torch.Tensor) -> dict[str, dict[str, float | int]]:
        return _metrics(
            {name: prediction[mask] for name, prediction in predictions.items()},
            target[mask],
        )

    return {
        "length": length,
        "radius": radius,
        "requested_perturbation_norm": perturbation_norm,
        "mean_observed_route_perturbation": sum(route_changes) / len(route_changes),
        "transport": transport,
        "batch_size": batch_size,
        "query_count": int(report_mask.sum()),
        "hot_query_count": int(hot_mask.sum()),
        "cold_query_count": int(cold_mask.sum()),
        "state_scalars": {name: 64 for name in PRIMARY_VARIANTS},
        "route_access": {
            strategy: route_access_profile(strategy)
            for strategy in ROUTE_STRATEGIES
        },
        "metrics": cohort(report_mask),
        "hot_metrics": cohort(hot_mask),
        "cold_metrics": cohort(cold_mask),
        "diagnostics": {
            "maximum_delta_chunk_recurrent_abs_error": scan_error,
            "hard_direct_delta_prediction_max_abs_gap": float(
                (
                    predictions["direct_hard_top1"]
                    - predictions["delta_hard_top1"]
                )
                .abs()
                .max()
            ),
            "oracle_direct_delta_prediction_max_abs_gap": float(
                (predictions["direct_oracle"] - predictions["delta_oracle"])
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
    perturbations: Iterable[float],
    transports: Iterable[bool],
    lengths: Iterable[int],
    overwrite_batch_size: int,
    stream_batch_size: int,
    chunk_size: int,
) -> dict[str, object]:
    start = time.perf_counter()
    training = train_slot_policy(
        "learned_both_joint",
        seed=seed,
        device=device,
        steps_per_stage=steps_per_stage,
    )
    training_seconds = time.perf_counter() - start
    overwrite_cells = [
        evaluate_overwrite_depth(
            training.policy,
            seed=seed,
            overwrite_depth=16,
            radius=0.75,
            perturbation_norm=perturbation,
            transport=transport,
            batch_size=overwrite_batch_size,
            chunk_size=chunk_size,
        )
        for perturbation in perturbations
        for transport in transports
    ]
    stream_cells = [
        evaluate_stream_length(
            training.policy,
            seed=seed,
            length=length,
            radius=0.75,
            perturbation_norm=perturbation,
            transport=transport,
            batch_size=stream_batch_size,
            chunk_size=chunk_size,
        )
        for length in lengths
        for perturbation in perturbations
        for transport in transports
    ]
    return {
        "seed": seed,
        "training": {
            "steps_per_stage": steps_per_stage,
            "encoder_parameters": 2 * SLOTS * 24,
            "seconds": training_seconds,
            "final_loss": training.final_loss,
            "final_endpoint_loss": training.final_endpoint_loss,
            "final_balance_loss": training.final_balance_loss,
        },
        "transformed_triality_gauge_max_abs_gap": transformed_triality_gauge_diagnostic(
            seed
        ),
        "overwrite_cells": overwrite_cells,
        "stream_cells": stream_cells,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    cells = [cell for row in rows for cell in row["overwrite_cells"]]
    stream_cells = [cell for row in rows for cell in row["stream_cells"]]
    all_cells = cells + stream_cells
    maximum_scan_error = max(
        float(cell["diagnostics"]["maximum_delta_chunk_recurrent_abs_error"])
        for cell in all_cells
    )
    maximum_hard_gap = max(
        float(cell["diagnostics"]["hard_direct_delta_prediction_max_abs_gap"])
        for cell in all_cells
    )
    maximum_oracle_gap = max(
        float(cell["diagnostics"]["oracle_direct_delta_prediction_max_abs_gap"])
        for cell in all_cells
    )
    maximum_triality_gap = max(
        float(gap)
        for row in rows
        for gap in row["transformed_triality_gauge_max_abs_gap"].values()
    )

    def values(name: str, metric: str = "metrics") -> list[float]:
        return [
            float(cell[metric][name]["mean_query_cosine"])
            for cell in all_cells
            if metric in cell
        ]

    mean_cosines = {
        name: sum(values(name)) / len(values(name)) for name in PRIMARY_VARIANTS
    }
    minimum_cosines = {
        name: min(values(name)) for name in PRIMARY_VARIANTS
    }
    paired_improvements = {
        kind: {
            strategy: sum(
                float(cell["metrics"][f"{kind}_{strategy}"]["mean_query_cosine"])
                - float(cell["metrics"][f"{kind}_dense_soft"]["mean_query_cosine"])
                for cell in all_cells
            )
            / len(all_cells)
            for strategy in ("block_top1", "hard_top1")
        }
        for kind in MEMORY_KINDS
    }
    oracle_exact = all(
        float(cell["metrics"][name]["minimum_query_cosine"]) >= 1.0 - 1e-10
        for cell in all_cells
        for name in ("direct_oracle", "delta_oracle")
    )
    return {
        "seeds": len(rows),
        "cells": len(all_cells),
        "mean_query_cosine_across_cells": mean_cosines,
        "minimum_cell_mean_query_cosine": minimum_cosines,
        "mean_paired_improvement_over_dense_soft": paired_improvements,
        "maximum_delta_chunk_recurrent_abs_error": maximum_scan_error,
        "maximum_hard_direct_delta_prediction_abs_gap": maximum_hard_gap,
        "maximum_oracle_direct_delta_prediction_abs_gap": maximum_oracle_gap,
        "maximum_transformed_triality_gauge_abs_gap": maximum_triality_gap,
        "oracle_rows_exact": oracle_exact,
        "implementation_gate_passed": (
            maximum_scan_error < 1e-9
            and maximum_hard_gap < 1e-10
            and maximum_oracle_gap < 1e-10
            and maximum_triality_gap < 1e-10
            and oracle_exact
        ),
        "claim_boundary": {
            "routing_improvement_is_spin8_specific": False,
            "new_overwrite_law_established": False,
            "measured_sparse_cuda_speedup_included": False,
        },
    }


def run(
    seeds: list[int],
    *,
    device: torch.device,
    steps_per_stage: int = 300,
    perturbations: tuple[float, ...] = (0.0, 0.1, 0.2),
    transports: tuple[bool, ...] = (False, True),
    lengths: tuple[int, ...] = (256, 1024, 4096),
    overwrite_batch_size: int = 32,
    stream_batch_size: int = 8,
    chunk_size: int = 64,
) -> dict[str, object]:
    rows = [
        run_seed(
            seed,
            device=device,
            steps_per_stage=steps_per_stage,
            perturbations=perturbations,
            transports=transports,
            lengths=lengths,
            overwrite_batch_size=overwrite_batch_size,
            stream_batch_size=stream_batch_size,
            chunk_size=chunk_size,
        )
        for seed in seeds
    ]
    return {
        "experiment": "NSA-inspired hierarchical matched retrieval",
        "protocol": "HIERARCHICAL_MATCHED_RETRIEVAL_PREREGISTRATION.md",
        "device": str(device),
        "dtype": str(DTYPE),
        "seeds": seeds,
        "grid": {
            "overwrite_depth": 16,
            "radius": 0.75,
            "perturbations": perturbations,
            "transports": transports,
            "stream_lengths": lengths,
            "overwrite_batch_size": overwrite_batch_size,
            "stream_batch_size": stream_batch_size,
            "chunk_size": chunk_size,
            "route_strategies": ROUTE_STRATEGIES,
        },
        "results": rows,
        "summary": summarize(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--steps-per-stage", type=int, default=300)
    parser.add_argument(
        "--perturbations", type=float, nargs="+", default=(0.0, 0.1, 0.2)
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=(256, 1024, 4096))
    parser.add_argument(
        "--transports", choices=("both", "false", "true"), default="both"
    )
    parser.add_argument("--overwrite-batch-size", type=int, default=32)
    parser.add_argument("--stream-batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/hierarchical_matched_retrieval_seeds0_9.json"),
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
        perturbations=tuple(args.perturbations),
        transports=transports,
        lengths=tuple(args.lengths),
        overwrite_batch_size=args.overwrite_batch_size,
        stream_batch_size=args.stream_batch_size,
        chunk_size=args.chunk_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
