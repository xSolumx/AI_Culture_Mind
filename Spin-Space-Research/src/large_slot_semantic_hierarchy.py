"""Large-slot overlapping-semantic hierarchy and shared-router campaign."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from spin8_blind_shared_action import sample_teacher
from spin8_triality import spin8_actions, torch_triality_generators

SLOTS = 64
BLOCKS = 8
SLOTS_PER_BLOCK = 8
VIEWS = 3
DIMENSION = 8
VALUE_DIMENSION = 8
ACTION_WORDS = 4
TRAIN_ACTION_WORDS = 3
TEMPERATURE = 0.35
ROUTER_FAMILIES = ("shared", "independent")
ROUTE_STRATEGIES = ("dense_soft", "block_top1", "hard_top1")
RADII = (0.20, 0.40, 0.60)
LENGTHS = (256, 1024, 2048)
QUERY_COHORTS = ("observed", "heldout")


def _random_unit(
    shape: tuple[int, ...], *, generator: torch.Generator
) -> torch.Tensor:
    return F.normalize(
        torch.randn(*shape, generator=generator, dtype=torch.float64), dim=-1
    )


@dataclass(frozen=True)
class SemanticWorld:
    block_prototypes: torch.Tensor
    centers: torch.Tensor

    @classmethod
    def create(cls, seed: int) -> SemanticWorld:
        generator = torch.Generator().manual_seed(1_030_000 + seed)
        raw = torch.randn(
            DIMENSION, DIMENSION, generator=generator, dtype=torch.float64
        )
        orthogonal, _ = torch.linalg.qr(raw)
        prototypes = orthogonal.transpose(0, 1).contiguous()
        offsets = torch.randn(
            BLOCKS,
            SLOTS_PER_BLOCK,
            DIMENSION,
            generator=generator,
            dtype=torch.float64,
        )
        offsets = offsets - torch.einsum(
            "bsd,bd->bs", offsets, prototypes
        )[..., None] * prototypes[:, None, :]
        offsets = F.normalize(offsets, dim=-1)
        centers = F.normalize(prototypes[:, None, :] + 0.60 * offsets, dim=-1)
        return cls(prototypes, centers.reshape(SLOTS, DIMENSION))

    def sample(
        self,
        labels: torch.Tensor,
        *,
        radius: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        noise = _random_unit((*labels.shape, DIMENSION), generator=generator)
        return F.normalize(self.centers[labels] + radius * noise, dim=-1)


def transport_aliases(
    canonical: torch.Tensor,
    words: torch.Tensor,
    views: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    selected = actions[words, views]
    return torch.einsum("...ij,...j->...i", selected, canonical)


def canonicalize_aliases(
    raw: torch.Tensor,
    words: torch.Tensor,
    views: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    selected = actions[words, views]
    return torch.einsum("...ji,...j->...i", selected, raw)


def observed_views(labels: torch.Tensor, choices: torch.Tensor) -> torch.Tensor:
    heldout = labels.remainder(VIEWS)
    return (heldout + 1 + choices.remainder(VIEWS - 1)).remainder(VIEWS)


class HierarchicalRouter(nn.Module):
    def __init__(self, family: str, *, seed: int) -> None:
        super().__init__()
        if family not in ROUTER_FAMILIES:
            raise ValueError(f"unknown router family: {family}")
        self.family = family
        generator = torch.Generator().manual_seed(1_040_000 + seed)
        prefix = () if family == "shared" else (VIEWS,)
        self.coarse_weight = nn.Parameter(
            0.1
            * torch.randn(
                *prefix,
                BLOCKS,
                DIMENSION,
                generator=generator,
                dtype=torch.float64,
            )
        )
        self.fine_weight = nn.Parameter(
            0.1
            * torch.randn(
                *prefix,
                SLOTS,
                DIMENSION,
                generator=generator,
                dtype=torch.float64,
            )
        )

    def logits(
        self, aliases: torch.Tensor, views: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coarse = F.normalize(self.coarse_weight, dim=-1)
        fine = F.normalize(self.fine_weight, dim=-1)
        if self.family == "independent":
            coarse = coarse[views]
            fine = fine[views]
            return (
                torch.einsum("...d,...bd->...b", aliases, coarse) / TEMPERATURE,
                torch.einsum("...d,...hd->...h", aliases, fine) / TEMPERATURE,
            )
        return (
            torch.einsum("...d,bd->...b", aliases, coarse) / TEMPERATURE,
            torch.einsum("...d,hd->...h", aliases, fine) / TEMPERATURE,
        )


@dataclass(frozen=True)
class FrozenRouter:
    family: str
    coarse_weight: torch.Tensor
    fine_weight: torch.Tensor

    def logits(
        self, aliases: torch.Tensor, views: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coarse = F.normalize(self.coarse_weight.to(aliases), dim=-1)
        fine = F.normalize(self.fine_weight.to(aliases), dim=-1)
        if self.family == "independent":
            return (
                torch.einsum("...d,...bd->...b", aliases, coarse[views])
                / TEMPERATURE,
                torch.einsum("...d,...hd->...h", aliases, fine[views])
                / TEMPERATURE,
            )
        return (
            torch.einsum("...d,bd->...b", aliases, coarse) / TEMPERATURE,
            torch.einsum("...d,hd->...h", aliases, fine) / TEMPERATURE,
        )

    def routes(
        self, aliases: torch.Tensor, views: torch.Tensor, strategy: str
    ) -> torch.Tensor:
        coarse_logits, fine_logits = self.logits(aliases, views)
        if strategy == "dense_soft":
            return F.softmax(fine_logits, dim=-1)
        block = coarse_logits.argmax(dim=-1)
        local_indices = block[..., None] * SLOTS_PER_BLOCK + torch.arange(
            SLOTS_PER_BLOCK, device=aliases.device
        )
        local_logits = fine_logits.gather(-1, local_indices)
        if strategy == "block_top1":
            local = F.softmax(local_logits, dim=-1)
        elif strategy == "hard_top1":
            local = F.one_hot(
                local_logits.argmax(dim=-1), SLOTS_PER_BLOCK
            ).to(aliases)
        else:
            raise ValueError(f"unknown route strategy: {strategy}")
        return torch.zeros_like(fine_logits).scatter(-1, local_indices, local)


@dataclass(frozen=True)
class TrainedRouters:
    shared: FrozenRouter
    independent: FrozenRouter
    report: dict[str, object]


def train_routers(
    world: SemanticWorld,
    actions: torch.Tensor,
    *,
    seed: int,
    steps: int,
    batch_size: int,
) -> TrainedRouters:
    shared = HierarchicalRouter("shared", seed=seed)
    independent = HierarchicalRouter("independent", seed=seed)
    optimizers = {
        "shared": torch.optim.Adam(shared.parameters(), lr=0.03),
        "independent": torch.optim.Adam(independent.parameters(), lr=0.03),
    }
    models = {"shared": shared, "independent": independent}
    generator = torch.Generator().manual_seed(1_050_000 + seed)
    trajectory: list[dict[str, float | int]] = []
    final_losses: dict[str, float] = {}
    start = time.perf_counter()
    for step in range(steps):
        labels = torch.randint(SLOTS, (batch_size,), generator=generator)
        choices = torch.randint(VIEWS - 1, (batch_size,), generator=generator)
        views = observed_views(labels, choices)
        words = torch.randint(
            TRAIN_ACTION_WORDS, (batch_size,), generator=generator
        )
        radius = (0.10, 0.30, 0.50)[step % 3]
        canonical = world.sample(labels, radius=radius, generator=generator)
        raw = transport_aliases(canonical, words, views, actions)
        recovered = canonicalize_aliases(raw, words, views, actions)
        for family, model in models.items():
            optimizers[family].zero_grad(set_to_none=True)
            coarse, fine = model.logits(recovered, views)
            loss = F.cross_entropy(fine, labels) + F.cross_entropy(
                coarse, labels.div(SLOTS_PER_BLOCK, rounding_mode="floor")
            )
            loss.backward()
            optimizers[family].step()
            final_losses[family] = float(loss.detach())
        if step % 100 == 0 or step + 1 == steps:
            trajectory.append(
                {
                    "step": step + 1,
                    "shared_loss": final_losses["shared"],
                    "independent_loss": final_losses["independent"],
                }
            )
    return TrainedRouters(
        shared=FrozenRouter(
            "shared",
            shared.coarse_weight.detach().cpu(),
            shared.fine_weight.detach().cpu(),
        ),
        independent=FrozenRouter(
            "independent",
            independent.coarse_weight.detach().cpu(),
            independent.fine_weight.detach().cpu(),
        ),
        report={
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": 0.03,
            "temperature": TEMPERATURE,
            "shared_parameters": sum(p.numel() for p in shared.parameters()),
            "independent_parameters": sum(
                p.numel() for p in independent.parameters()
            ),
            "seconds": time.perf_counter() - start,
            "trajectory": trajectory,
        },
    )


def _router_payload(router: FrozenRouter) -> dict[str, object]:
    return {
        "family": router.family,
        "coarse_weight": router.coarse_weight.tolist(),
        "fine_weight": router.fine_weight.tolist(),
    }


def _router_from_payload(payload: dict[str, object]) -> FrozenRouter:
    return FrozenRouter(
        str(payload["family"]),
        torch.tensor(payload["coarse_weight"], dtype=torch.float64),
        torch.tensor(payload["fine_weight"], dtype=torch.float64),
    )


@torch.no_grad()
def router_diagnostics(
    routers: dict[str, FrozenRouter],
    world: SemanticWorld,
    actions: torch.Tensor,
    *,
    seed: int,
    repeats: int = 8,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(1_060_000 + seed)
    results: dict[str, object] = {}
    max_canonicalization_error = 0.0
    base_labels = torch.arange(SLOTS).repeat_interleave(repeats)
    for radius in RADII:
        for cohort in QUERY_COHORTS:
            labels = base_labels.clone()
            if cohort == "heldout":
                views = labels.remainder(VIEWS)
            else:
                choices = torch.randint(
                    VIEWS - 1, labels.shape, generator=generator
                )
                views = observed_views(labels, choices)
            words = torch.full_like(labels, ACTION_WORDS - 1)
            canonical = world.sample(labels, radius=radius, generator=generator)
            raw = transport_aliases(canonical, words, views, actions)
            recovered = canonicalize_aliases(raw, words, views, actions)
            max_canonicalization_error = max(
                max_canonicalization_error,
                float((canonical - recovered).abs().max()),
            )
            cell: dict[str, object] = {}
            for family, router in routers.items():
                coarse_logits, _ = router.logits(recovered, views)
                family_metrics: dict[str, object] = {
                    "coarse_accuracy": float(
                        (
                            coarse_logits.argmax(dim=-1)
                            == labels.div(SLOTS_PER_BLOCK, rounding_mode="floor")
                        )
                        .double()
                        .mean()
                    )
                }
                for strategy in ROUTE_STRATEGIES:
                    route = router.routes(recovered, views, strategy)
                    entropy = -(route * route.clamp_min(1e-30).log()).sum(dim=-1)
                    family_metrics[strategy] = {
                        "accuracy": float(
                            (route.argmax(dim=-1) == labels).double().mean()
                        ),
                        "mean_entropy": float(entropy.mean()),
                        "mean_true_route_mass": float(
                            route.gather(-1, labels[:, None]).mean()
                        ),
                    }
                cell[family] = family_metrics
            results[f"radius_{radius:.2f}_{cohort}"] = cell
    return {
        "heldout_action_word": ACTION_WORDS - 1,
        "maximum_canonicalization_error": max_canonicalization_error,
        "cells": results,
    }


def _stack_routes(
    routers: dict[str, FrozenRouter],
    aliases: torch.Tensor,
    views: torch.Tensor,
) -> torch.Tensor:
    return torch.stack(
        [
            torch.stack(
                [router.routes(aliases, views, strategy) for strategy in ROUTE_STRATEGIES]
            )
            for router in routers.values()
        ]
    )


def _memory_write(
    direct: torch.Tensor,
    delta: torch.Tensor,
    routes: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    drive = routes[..., None] * values[None, None, :, None, :]
    direct = (1.0 - routes[..., None]) * direct + drive
    keys = F.normalize(routes, dim=-1)
    prediction = torch.einsum("rsbh,rsbhv->rsbv", keys, delta)
    delta = delta + keys[..., None] * (
        values[None, None] - prediction
    )[..., None, :]
    return direct, delta


def _memory_predictions(
    direct: torch.Tensor, delta: torch.Tensor, routes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = F.normalize(routes, dim=-1)
    return (
        torch.einsum("rsbh,rsbhv->rsbv", routes, direct),
        torch.einsum("rsbh,rsbhv->rsbv", keys, delta),
    )


def _metric_snapshot(
    cosine_sums: torch.Tensor,
    squared_error_sums: torch.Tensor,
    target_energy_sum: float,
    minima: torch.Tensor,
    count: int,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for router_index, family in enumerate(ROUTER_FAMILIES):
        family_rows: dict[str, object] = {}
        for strategy_index, strategy in enumerate(ROUTE_STRATEGIES):
            for memory_index, memory in enumerate(("direct", "delta")):
                family_rows[f"{memory}_{strategy}"] = {
                    "mean_query_cosine": float(
                        cosine_sums[
                            router_index, strategy_index, memory_index
                        ]
                        / count
                    ),
                    "minimum_query_cosine": float(
                        minima[router_index, strategy_index, memory_index]
                    ),
                    "mean_relative_squared_error": float(
                        squared_error_sums[
                            router_index, strategy_index, memory_index
                        ]
                        / target_energy_sum
                    ),
                    "query_count": count,
                }
        result[family] = family_rows
    return result


@torch.no_grad()
def evaluate_stream(
    routers: dict[str, FrozenRouter],
    world: SemanticWorld,
    actions: torch.Tensor,
    *,
    seed: int,
    radius: float,
    query_cohort: str,
    batch_size: int,
) -> dict[str, object]:
    if query_cohort not in QUERY_COHORTS:
        raise ValueError(f"unknown query cohort: {query_cohort}")
    generator = torch.Generator().manual_seed(
        1_070_000 + seed + round(100 * radius) + 10_000 * QUERY_COHORTS.index(query_cohort)
    )
    direct = torch.zeros(
        len(ROUTER_FAMILIES),
        len(ROUTE_STRATEGIES),
        batch_size,
        SLOTS,
        VALUE_DIMENSION,
        dtype=torch.float64,
    )
    delta = torch.zeros_like(direct)
    reference = torch.zeros(
        batch_size, SLOTS, VALUE_DIMENSION, dtype=torch.float64
    )
    oracle_direct = torch.zeros_like(reference)
    oracle_delta = torch.zeros_like(reference)
    maximum_hard_state_gap = 0.0
    maximum_hard_prediction_gap = 0.0
    maximum_canonicalization_error = 0.0
    minimum_oracle_cosine = 1.0
    maximum_oracle_direct_delta_gap = 0.0

    def oracle_write(labels: torch.Tensor, values: torch.Tensor) -> None:
        nonlocal oracle_direct, oracle_delta
        route = F.one_hot(labels, SLOTS).to(dtype=torch.float64)
        oracle_direct = (
            (1.0 - route[..., None]) * oracle_direct
            + route[..., None] * values[:, None, :]
        )
        prediction = torch.einsum("bh,bhv->bv", route, oracle_delta)
        oracle_delta = oracle_delta + route[..., None] * (
            values - prediction
        )[:, None, :]

    def routes_for(labels: torch.Tensor, views: torch.Tensor) -> torch.Tensor:
        nonlocal maximum_canonicalization_error
        words = torch.full_like(labels, ACTION_WORDS - 1)
        canonical = world.sample(labels, radius=radius, generator=generator)
        raw = transport_aliases(canonical, words, views, actions)
        recovered = canonicalize_aliases(raw, words, views, actions)
        maximum_canonicalization_error = max(
            maximum_canonicalization_error,
            float((canonical - recovered).abs().max()),
        )
        return _stack_routes(routers, recovered, views)

    batch_index = torch.arange(batch_size)
    for label in range(SLOTS):
        labels = torch.full((batch_size,), label, dtype=torch.long)
        choices = torch.randint(VIEWS - 1, labels.shape, generator=generator)
        views = observed_views(labels, choices)
        values = _random_unit((batch_size, VALUE_DIMENSION), generator=generator)
        routes = routes_for(labels, views)
        direct, delta = _memory_write(direct, delta, routes, values)
        oracle_write(labels, values)
        reference[:, label] = values

    cosine_sums = torch.zeros(
        len(ROUTER_FAMILIES), len(ROUTE_STRATEGIES), 2, dtype=torch.float64
    )
    squared_error_sums = torch.zeros_like(cosine_sums)
    minima = torch.full_like(cosine_sums, math.inf)
    target_energy_sum = 0.0
    snapshots: dict[str, object] = {}
    for position in range(1, max(LENGTHS) + 1):
        write_labels = torch.randint(SLOTS, (batch_size,), generator=generator)
        write_choices = torch.randint(
            VIEWS - 1, write_labels.shape, generator=generator
        )
        write_views = observed_views(write_labels, write_choices)
        values = _random_unit((batch_size, VALUE_DIMENSION), generator=generator)
        write_routes = routes_for(write_labels, write_views)
        direct, delta = _memory_write(direct, delta, write_routes, values)
        oracle_write(write_labels, values)
        reference[batch_index, write_labels] = values
        maximum_hard_state_gap = max(
            maximum_hard_state_gap,
            float((direct[:, 2] - delta[:, 2]).abs().max()),
        )

        query_labels = torch.randint(SLOTS, (batch_size,), generator=generator)
        if query_cohort == "heldout":
            query_views = query_labels.remainder(VIEWS)
        else:
            query_choices = torch.randint(
                VIEWS - 1, query_labels.shape, generator=generator
            )
            query_views = observed_views(query_labels, query_choices)
        query_routes = routes_for(query_labels, query_views)
        direct_prediction, delta_prediction = _memory_predictions(
            direct, delta, query_routes
        )
        maximum_hard_prediction_gap = max(
            maximum_hard_prediction_gap,
            float(
                (direct_prediction[:, 2] - delta_prediction[:, 2]).abs().max()
            ),
        )
        predictions = torch.stack((direct_prediction, delta_prediction), dim=2)
        target = reference[batch_index, query_labels]
        oracle_route = F.one_hot(query_labels, SLOTS).to(dtype=torch.float64)
        oracle_direct_prediction = torch.einsum(
            "bh,bhv->bv", oracle_route, oracle_direct
        )
        oracle_delta_prediction = torch.einsum(
            "bh,bhv->bv", oracle_route, oracle_delta
        )
        minimum_oracle_cosine = min(
            minimum_oracle_cosine,
            float(
                F.cosine_similarity(oracle_direct_prediction, target, dim=-1).min()
            ),
            float(
                F.cosine_similarity(oracle_delta_prediction, target, dim=-1).min()
            ),
        )
        maximum_oracle_direct_delta_gap = max(
            maximum_oracle_direct_delta_gap,
            float((oracle_direct_prediction - oracle_delta_prediction).abs().max()),
        )
        cosine = F.cosine_similarity(
            predictions, target[None, None, None], dim=-1
        )
        squared_error = (
            predictions - target[None, None, None]
        ).square().sum(dim=-1)
        cosine_sums += cosine.sum(dim=-1)
        squared_error_sums += squared_error.sum(dim=-1)
        minima = torch.minimum(minima, cosine.amin(dim=-1))
        target_energy_sum += float(target.square().sum())
        if position in LENGTHS:
            snapshots[str(position)] = _metric_snapshot(
                cosine_sums,
                squared_error_sums,
                target_energy_sum,
                minima,
                position * batch_size,
            )
    return {
        "radius": radius,
        "query_cohort": query_cohort,
        "batch_size": batch_size,
        "snapshots": snapshots,
        "diagnostics": {
            "maximum_canonicalization_error": maximum_canonicalization_error,
            "maximum_hard_direct_delta_state_error": maximum_hard_state_gap,
            "maximum_hard_direct_delta_prediction_error": (
                maximum_hard_prediction_gap
            ),
            "oracle_minimum_query_cosine": minimum_oracle_cosine,
            "maximum_oracle_direct_delta_prediction_error": (
                maximum_oracle_direct_delta_gap
            ),
        },
    }


def _router_decision(diagnostics: dict[str, object]) -> dict[str, float | bool]:
    cells = diagnostics["cells"]
    keys = [
        f"radius_{radius:.2f}_{cohort}"
        for radius in (0.20, 0.40)
        for cohort in QUERY_COHORTS
    ]
    shared_heldout = [
        float(cells[key]["shared"]["hard_top1"]["accuracy"])
        for key in keys
        if key.endswith("heldout")
    ]
    independent_heldout = [
        float(cells[key]["independent"]["hard_top1"]["accuracy"])
        for key in keys
        if key.endswith("heldout")
    ]
    independent_observed = [
        float(cells[key]["independent"]["hard_top1"]["accuracy"])
        for key in keys
        if key.endswith("observed")
    ]
    shared_mean = sum(shared_heldout) / len(shared_heldout)
    independent_mean = sum(independent_heldout) / len(independent_heldout)
    observed_mean = sum(independent_observed) / len(independent_observed)
    return {
        "shared_heldout_hard_accuracy": shared_mean,
        "independent_heldout_hard_accuracy": independent_mean,
        "independent_observed_hard_accuracy": observed_mean,
        "shared_minus_independent_heldout_accuracy": shared_mean - independent_mean,
        "shared_router_completion_passed": (
            shared_mean >= 0.85
            and shared_mean - independent_mean >= 0.25
            and observed_mean >= 0.85
        ),
    }


def _hierarchy_decision(streams: list[dict[str, object]]) -> dict[str, object]:
    selected = [
        stream
        for stream in streams
        if float(stream["radius"]) in (0.20, 0.40)
    ]
    improvements: dict[str, float] = {}
    for memory in ("direct", "delta"):
        differences = []
        for stream in selected:
            shared = stream["snapshots"][str(max(LENGTHS))]["shared"]
            differences.append(
                float(shared[f"{memory}_block_top1"]["mean_query_cosine"])
                - float(shared[f"{memory}_dense_soft"]["mean_query_cosine"])
            )
        improvements[memory] = sum(differences) / len(differences)
    return {
        "mean_block_improvement_over_dense": improvements,
        "hierarchical_routing_passed": all(value > 0.0 for value in improvements.values()),
    }


def _replay_payload(
    routers: dict[str, FrozenRouter],
    world: SemanticWorld,
    actions: torch.Tensor,
    *,
    seed: int,
) -> dict[str, object]:
    return router_diagnostics(
        routers, world, actions, seed=seed + 90_000, repeats=2
    )


def _maximum_nested_difference(left: object, right: object) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return math.inf
        return max(
            (_maximum_nested_difference(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return math.inf
        return max(
            (_maximum_nested_difference(a, b) for a, b in zip(left, right, strict=True)),
            default=0.0,
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else math.inf


def verify_retained_parameters(report: dict[str, object]) -> dict[str, float | bool]:
    seed = int(report["seed"])
    payload = report["retained_parameters"]
    world = SemanticWorld(
        torch.tensor(payload["block_prototypes"], dtype=torch.float64),
        torch.tensor(payload["semantic_centers"], dtype=torch.float64),
    )
    actions = torch.tensor(payload["spin8_actions"], dtype=torch.float64)
    routers = {
        family: _router_from_payload(payload[family]) for family in ROUTER_FAMILIES
    }
    expected_world = SemanticWorld.create(seed)
    teacher = sample_teacher(
        seed=seed,
        generators=torch_triality_generators(dtype=torch.float64),
    )
    reconstruction_error = max(
        float((world.centers - expected_world.centers).abs().max()),
        float(
            (world.block_prototypes - expected_world.block_prototypes).abs().max()
        ),
        float((actions - teacher.actions).abs().max()),
        float(
            (
                actions
                - spin8_actions(
                    torch.tensor(payload["teacher_coordinates"], dtype=torch.float64),
                    torch_triality_generators(dtype=torch.float64),
                )
            ).abs().max()
        ),
    )
    replay = _replay_payload(routers, world, actions, seed=seed)
    replay_difference = _maximum_nested_difference(
        replay, report["retained_parameter_replay"]
    )
    maximum_difference = max(reconstruction_error, replay_difference)
    return {
        "maximum_difference": maximum_difference,
        "passed": maximum_difference <= 1e-12,
    }


def run_seed(
    seed: int,
    *,
    training_steps: int = 800,
    training_batch_size: int = 512,
    stream_batch_size: int = 8,
) -> dict[str, object]:
    generators = torch_triality_generators(dtype=torch.float64)
    teacher = sample_teacher(seed=seed, generators=generators)
    actions = teacher.actions.detach().cpu()
    world = SemanticWorld.create(seed)
    trained = train_routers(
        world,
        actions,
        seed=seed,
        steps=training_steps,
        batch_size=training_batch_size,
    )
    routers = {"shared": trained.shared, "independent": trained.independent}
    diagnostics = router_diagnostics(routers, world, actions, seed=seed)
    streams = [
        evaluate_stream(
            routers,
            world,
            actions,
            seed=seed,
            radius=radius,
            query_cohort=cohort,
            batch_size=stream_batch_size,
        )
        for radius in RADII
        for cohort in QUERY_COHORTS
    ]
    router_decision = _router_decision(diagnostics)
    hierarchy_decision = _hierarchy_decision(streams)
    maximum_canonicalization_error = max(
        float(diagnostics["maximum_canonicalization_error"]),
        max(
            float(stream["diagnostics"]["maximum_canonicalization_error"])
            for stream in streams
        ),
    )
    maximum_hard_state_error = max(
        float(stream["diagnostics"]["maximum_hard_direct_delta_state_error"])
        for stream in streams
    )
    maximum_hard_prediction_error = max(
        float(
            stream["diagnostics"]["maximum_hard_direct_delta_prediction_error"]
        )
        for stream in streams
    )
    implementation_passed = (
        maximum_canonicalization_error <= 1e-10
        and maximum_hard_state_error <= 1e-10
        and maximum_hard_prediction_error <= 1e-10
        and all(
            float(stream["diagnostics"]["oracle_minimum_query_cosine"])
            >= 1.0 - 1e-10
            for stream in streams
        )
    )
    retained_parameters = {
        "block_prototypes": world.block_prototypes.tolist(),
        "semantic_centers": world.centers.tolist(),
        "teacher_coordinates": teacher.coefficients.detach().cpu().tolist(),
        "spin8_actions": actions.tolist(),
        "shared": _router_payload(trained.shared),
        "independent": _router_payload(trained.independent),
    }
    replay = _replay_payload(routers, world, actions, seed=seed)
    report = {
        "experiment": "large-slot overlapping-semantic hierarchy",
        "protocol": "LARGE_SLOT_SEMANTIC_HIERARCHY_PREREGISTRATION.md",
        "seed": seed,
        "device": "cpu",
        "dtype": "float64",
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "world": {
            "slots": SLOTS,
            "blocks": BLOCKS,
            "slots_per_block": SLOTS_PER_BLOCK,
            "views": VIEWS,
            "action_words": ACTION_WORDS,
            "heldout_action_word": ACTION_WORDS - 1,
            "radii": RADII,
            "lengths": LENGTHS,
        },
        "training": trained.report,
        "router_diagnostics": diagnostics,
        "streams": streams,
        "retained_parameters": retained_parameters,
        "retained_parameter_replay": replay,
        "decision": {
            "implementation_passed": implementation_passed,
            **router_decision,
            **hierarchy_decision,
            "maximum_canonicalization_error": maximum_canonicalization_error,
            "maximum_hard_direct_delta_state_error": maximum_hard_state_error,
            "maximum_hard_direct_delta_prediction_error": (
                maximum_hard_prediction_error
            ),
        },
    }
    verification = verify_retained_parameters(report)
    report["retained_parameter_verification"] = verification
    report["decision"]["implementation_passed"] = implementation_passed and bool(
        verification["passed"]
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--training-steps", type=int, default=800)
    parser.add_argument("--training-batch-size", type=int, default=512)
    parser.add_argument("--stream-batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        report = json.loads(args.verify.read_text(encoding="utf-8"))
        verification = verify_retained_parameters(report)
        print(json.dumps(verification, indent=2))
        raise SystemExit(0 if verification["passed"] else 1)
    if args.seed is None or args.output is None:
        parser.error("--seed and --output are required unless --verify is used")
    report = run_seed(
        args.seed,
        training_steps=args.training_steps,
        training_batch_size=args.training_batch_size,
        stream_batch_size=args.stream_batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
