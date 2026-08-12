"""Continuous-alias routing gate for scan-compatible Spin(8) memory."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from spin8_learned_address import (
    ACTION_TOKENS,
    DIMENSION,
    KEYS,
    SLOTS,
    teacher_actions,
)
from spin8_triality_lift import triality_bind, triality_tensor, triality_unbind_negative
from spin8_triality_memory import SlotTransition, apply_slot, associative_slot_scan


ALIAS_DIMENSION = 24
DENSE_LENGTHS = (16, 32, 64, 128, 256, 512, 1024, 2048)
TRAIN_RADII = (0.05, 0.10, 0.15)
VALIDATION_RADIUS = 0.22
TEST_RADIUS = 0.35
SLOT_VARIANTS = (
    "oracle_both",
    "oracle_write_learned_query",
    "learned_write_oracle_query",
    "learned_both_independent",
    "learned_both_joint",
    "learned_both_joint_untrained",
)


def random_unit(
    shape: tuple[int, ...], *, generator: torch.Generator, dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return F.normalize(
        torch.randn(*shape, generator=generator, dtype=dtype, device=device), dim=-1
    )


@dataclass(frozen=True)
class AliasWorld:
    centers: torch.Tensor

    @classmethod
    def create(
        cls, seed: int, *, dtype: torch.dtype, device: torch.device
    ) -> "AliasWorld":
        # CPU construction is the canonical world definition. CPU and CUDA
        # generators do not emit the same stream for an equal integer seed.
        generator = torch.Generator().manual_seed(610_000 + seed)
        matrix = torch.randn(
            ALIAS_DIMENSION,
            KEYS,
            generator=generator,
            dtype=torch.float64,
            device="cpu",
        )
        basis, _ = torch.linalg.qr(matrix, mode="reduced")
        return cls(centers=basis.T.contiguous().to(device=device, dtype=dtype))

    def sample(
        self,
        labels: torch.Tensor,
        *,
        radius: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        noise = torch.randn(
            *labels.shape,
            ALIAS_DIMENSION,
            generator=generator,
            dtype=self.centers.dtype,
            device=self.centers.device,
        )
        # Nuisance is exactly orthogonal to the entire semantic center span.
        coordinates = torch.einsum("...d,kd->...k", noise, self.centers)
        noise = noise - torch.einsum("...k,kd->...d", coordinates, self.centers)
        noise = F.normalize(noise, dim=-1)
        return F.normalize(self.centers[labels] + radius * noise, dim=-1)


class AliasEncoders(nn.Module):
    def __init__(self, *, seed: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        generator = torch.Generator(device=device).manual_seed(620_000 + seed)
        scale = ALIAS_DIMENSION**-0.5
        self.write_weight = nn.Parameter(
            scale
            * torch.randn(
                SLOTS,
                ALIAS_DIMENSION,
                generator=generator,
                dtype=dtype,
                device=device,
            )
        )
        self.query_weight = nn.Parameter(
            scale
            * torch.randn(
                SLOTS,
                ALIAS_DIMENSION,
                generator=generator,
                dtype=dtype,
                device=device,
            )
        )

    def routes(self, aliases: torch.Tensor, *, side: str, temperature: float) -> torch.Tensor:
        weight = self.write_weight if side == "write" else self.query_weight
        return F.softmax(torch.einsum("...d,hd->...h", aliases, weight) / temperature, dim=-1)


@dataclass(frozen=True)
class FrozenSlotPolicy:
    variant: str
    write_weight: torch.Tensor | None
    query_weight: torch.Tensor | None
    temperature: float = 0.03

    def routes(
        self, aliases: torch.Tensor, labels: torch.Tensor, *, side: str
    ) -> torch.Tensor:
        oracle = self.variant == "oracle_both" or (
            self.variant == "oracle_write_learned_query" and side == "write"
        ) or (
            self.variant == "learned_write_oracle_query" and side == "query"
        )
        if oracle:
            return F.one_hot(labels, SLOTS).to(dtype=aliases.dtype)
        weight = self.write_weight if side == "write" else self.query_weight
        if weight is None:
            raise RuntimeError(f"missing learned {side} encoder")
        return F.softmax(
            torch.einsum("...d,hd->...h", aliases, weight.to(aliases)) / self.temperature,
            dim=-1,
        )


class VectorKeyEncoders(nn.Module):
    def __init__(self, *, seed: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        generator = torch.Generator(device=device).manual_seed(625_000 + seed)
        scale = ALIAS_DIMENSION**-0.5
        self.write_weight = nn.Parameter(
            scale
            * torch.randn(
                DIMENSION,
                ALIAS_DIMENSION,
                generator=generator,
                dtype=dtype,
                device=device,
            )
        )
        self.query_weight = nn.Parameter(
            scale
            * torch.randn(
                DIMENSION,
                ALIAS_DIMENSION,
                generator=generator,
                dtype=dtype,
                device=device,
            )
        )

    def keys(self, aliases: torch.Tensor, *, side: str) -> torch.Tensor:
        weight = self.write_weight if side == "write" else self.query_weight
        return F.normalize(torch.einsum("...d,hd->...h", aliases, weight), dim=-1)


@dataclass(frozen=True)
class FrozenKeyPolicy:
    write_weight: torch.Tensor
    query_weight: torch.Tensor

    def keys(self, aliases: torch.Tensor, *, side: str) -> torch.Tensor:
        weight = self.write_weight if side == "write" else self.query_weight
        return F.normalize(
            torch.einsum("...d,hd->...h", aliases, weight.to(aliases)), dim=-1
        )


def balanced_labels(repeats: int, *, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    labels = torch.arange(KEYS, device=device).repeat(repeats)
    return labels[torch.randperm(labels.numel(), generator=generator, device=device)]


def slot_endpoint_loss(
    write_routes: torch.Tensor,
    query_routes: torch.Tensor,
    *,
    joint: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    overlap = (write_routes * query_routes).sum(dim=-1)
    endpoint = (1.0 - overlap).square().mean()
    vertex = 0.5 * (
        (write_routes * (1.0 - write_routes)).mean()
        + (query_routes * (1.0 - query_routes)).mean()
    )
    uniform = 1.0 / SLOTS
    balance = (
        (write_routes.mean(dim=0) - uniform).square().sum()
        + (query_routes.mean(dim=0) - uniform).square().sum()
    )
    loss = endpoint + 0.02 * vertex + (8.0 * balance if joint else 0.0)
    return loss, {"endpoint": endpoint, "vertex": vertex, "balance": balance}


@dataclass(frozen=True)
class TrainedSlotPolicy:
    policy: FrozenSlotPolicy
    initial_loss: float
    final_loss: float
    final_endpoint_loss: float
    final_balance_loss: float


@dataclass(frozen=True)
class TrainedKeyPolicy:
    policy: FrozenKeyPolicy
    initial_loss: float
    final_loss: float
    final_endpoint_loss: float
    final_whitening_loss: float


def train_slot_policy(
    variant: str,
    *,
    seed: int,
    device: torch.device,
    steps_per_stage: int = 300,
    repeats: int = 16,
) -> TrainedSlotPolicy:
    if variant not in SLOT_VARIANTS:
        raise ValueError(f"unknown slot variant: {variant}")
    dtype = torch.float64
    world = AliasWorld.create(seed, dtype=dtype, device=device)
    model = AliasEncoders(seed=seed, dtype=dtype, device=device)
    if variant in {"oracle_both", "learned_both_joint_untrained"}:
        policy = FrozenSlotPolicy(
            variant=variant,
            write_weight=None if variant == "oracle_both" else model.write_weight.detach().cpu(),
            query_weight=None if variant == "oracle_both" else model.query_weight.detach().cpu(),
        )
        return TrainedSlotPolicy(policy, 0.0, 0.0, 0.0, 0.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.06)
    generator = torch.Generator(device=device).manual_seed(630_000 + seed)
    joint = variant == "learned_both_joint"
    initial_loss = 0.0
    final_loss = 0.0
    final_terms: dict[str, torch.Tensor] = {}
    total_steps = len(TRAIN_RADII) * steps_per_stage
    for stage, radius in enumerate(TRAIN_RADII):
        for step in range(steps_per_stage):
            index = stage * steps_per_stage + step
            fraction = index / max(1, total_steps - 1)
            temperature = 0.50 * (0.07 / 0.50) ** fraction
            labels = balanced_labels(repeats, generator=generator, device=device)
            write_alias = world.sample(labels, radius=radius, generator=generator)
            query_alias = world.sample(labels, radius=radius, generator=generator)
            learned_write = model.routes(write_alias, side="write", temperature=temperature)
            learned_query = model.routes(query_alias, side="query", temperature=temperature)
            oracle = F.one_hot(labels, SLOTS).to(dtype=dtype)
            write_routes = (
                oracle if variant == "oracle_write_learned_query" else learned_write
            )
            query_routes = (
                oracle if variant == "learned_write_oracle_query" else learned_query
            )
            loss, terms = slot_endpoint_loss(write_routes, query_routes, joint=joint)
            if index == 0:
                initial_loss = float(loss.detach())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
            final_terms = terms
    policy = FrozenSlotPolicy(
        variant=variant,
        write_weight=model.write_weight.detach().cpu(),
        query_weight=model.query_weight.detach().cpu(),
    )
    return TrainedSlotPolicy(
        policy=policy,
        initial_loss=initial_loss,
        final_loss=final_loss,
        final_endpoint_loss=float(final_terms["endpoint"].detach()),
        final_balance_loss=float(final_terms["balance"].detach()),
    )


def key_endpoint_loss(
    write_keys: torch.Tensor, query_keys: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    overlap = (write_keys * query_keys).sum(dim=-1)
    endpoint = (1.0 - overlap).square().mean()
    identity = torch.eye(DIMENSION, dtype=write_keys.dtype, device=write_keys.device)
    write_second = write_keys.T @ write_keys / write_keys.shape[0]
    query_second = query_keys.T @ query_keys / query_keys.shape[0]
    whitening = 0.5 * (
        (DIMENSION * write_second - identity).square().mean()
        + (DIMENSION * query_second - identity).square().mean()
    )
    return endpoint + 4.0 * whitening, {
        "endpoint": endpoint,
        "whitening": whitening,
    }


def train_key_policy(
    *,
    seed: int,
    device: torch.device,
    steps_per_stage: int = 300,
    repeats: int = 16,
) -> TrainedKeyPolicy:
    dtype = torch.float64
    world = AliasWorld.create(seed, dtype=dtype, device=device)
    model = VectorKeyEncoders(seed=seed, dtype=dtype, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
    generator = torch.Generator(device=device).manual_seed(635_000 + seed)
    initial_loss = 0.0
    final_loss = 0.0
    final_terms: dict[str, torch.Tensor] = {}
    index = 0
    for radius in TRAIN_RADII:
        for _ in range(steps_per_stage):
            labels = balanced_labels(repeats, generator=generator, device=device)
            write_alias = world.sample(labels, radius=radius, generator=generator)
            query_alias = world.sample(labels, radius=radius, generator=generator)
            write_keys = model.keys(write_alias, side="write")
            query_keys = model.keys(query_alias, side="query")
            loss, terms = key_endpoint_loss(write_keys, query_keys)
            if index == 0:
                initial_loss = float(loss.detach())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
            final_terms = terms
            index += 1
    return TrainedKeyPolicy(
        policy=FrozenKeyPolicy(
            model.write_weight.detach().cpu(), model.query_weight.detach().cpu()
        ),
        initial_loss=initial_loss,
        final_loss=final_loss,
        final_endpoint_loss=float(final_terms["endpoint"].detach()),
        final_whitening_loss=float(final_terms["whitening"].detach()),
    )


@torch.no_grad()
def alias_diagnostics(
    policy: FrozenSlotPolicy,
    *,
    seed: int,
    radius: float,
    samples_per_class: int = 256,
) -> dict[str, object]:
    dtype = torch.float64
    device = torch.device("cpu")
    world = AliasWorld.create(seed, dtype=dtype, device=device)
    center_labels = torch.arange(KEYS)
    write_center = policy.routes(world.centers, center_labels, side="write")
    query_center = policy.routes(world.centers, center_labels, side="query")
    write_slots = write_center.argmax(dim=-1)
    query_slots = query_center.argmax(dim=-1)
    generator = torch.Generator().manual_seed(640_000 + seed + round(1000 * radius))
    labels = torch.arange(KEYS).repeat_interleave(samples_per_class)
    aliases = world.sample(labels, radius=radius, generator=generator)
    write_routes = policy.routes(aliases, labels, side="write")
    query_routes = policy.routes(aliases, labels, side="query")
    write_agreement = (write_routes.argmax(dim=-1) == write_slots[labels]).double().mean()
    query_agreement = (query_routes.argmax(dim=-1) == query_slots[labels]).double().mean()
    cross_center = (write_slots == query_slots).double().mean()
    entropy = lambda routes: -(routes.clamp_min(1e-300).log() * routes).sum(dim=-1).mean()
    return {
        "radius": radius,
        "write_center_slots": [int(value) for value in write_slots],
        "query_center_slots": [int(value) for value in query_slots],
        "write_center_collisions": KEYS - int(torch.unique(write_slots).numel()),
        "query_center_collisions": KEYS - int(torch.unique(query_slots).numel()),
        "center_cross_encoder_agreement": float(cross_center),
        "write_alias_agreement": float(write_agreement),
        "query_alias_agreement": float(query_agreement),
        "write_center_column_residual": float((write_center.sum(dim=0) - 1.0).abs().max()),
        "query_center_column_residual": float((query_center.sum(dim=0) - 1.0).abs().max()),
        "mean_write_alias_entropy": float(entropy(write_routes)),
        "mean_query_alias_entropy": float(entropy(query_routes)),
    }


@torch.no_grad()
def alias_world_audit(seed: int) -> dict[str, float]:
    dtype = torch.float64
    cpu = AliasWorld.create(seed, dtype=dtype, device=torch.device("cpu"))
    gram = cpu.centers @ cpu.centers.T
    generator = torch.Generator().manual_seed(645_000 + seed)
    labels = torch.arange(KEYS).repeat_interleave(64)
    radius_errors = []
    for radius in (*TRAIN_RADII, VALIDATION_RADIUS, TEST_RADIUS):
        aliases = cpu.sample(labels, radius=radius, generator=generator)
        cosine = (aliases * cpu.centers[labels]).sum(dim=-1)
        expected = 1.0 / (1.0 + radius**2) ** 0.5
        radius_errors.append(float((cosine - expected).abs().max()))
    device_error = 0.0
    if torch.cuda.is_available():
        cuda = AliasWorld.create(seed, dtype=dtype, device=torch.device("cuda"))
        device_error = float((cuda.centers.cpu() - cpu.centers).abs().max())
    return {
        "center_gram_max_error": float(
            (gram - torch.eye(KEYS, dtype=dtype)).abs().max()
        ),
        "radius_cosine_max_error": max(radius_errors),
        "cross_device_center_max_error": device_error,
    }


@torch.no_grad()
def key_diagnostics(
    policy: FrozenKeyPolicy,
    *,
    seed: int,
    radius: float,
    samples_per_class: int = 256,
) -> dict[str, float]:
    dtype = torch.float64
    world = AliasWorld.create(seed, dtype=dtype, device=torch.device("cpu"))
    center_write = policy.keys(world.centers, side="write")
    center_query = policy.keys(world.centers, side="query")
    generator = torch.Generator().manual_seed(646_000 + seed + round(1000 * radius))
    labels = torch.arange(KEYS).repeat_interleave(samples_per_class)
    aliases = world.sample(labels, radius=radius, generator=generator)
    write = policy.keys(aliases, side="write")
    query = policy.keys(aliases, side="query")
    identity = torch.eye(KEYS, dtype=dtype)
    return {
        "radius": radius,
        "write_center_gram_max_error": float(
            (center_write @ center_write.T - identity).abs().max()
        ),
        "query_center_gram_max_error": float(
            (center_query @ center_query.T - identity).abs().max()
        ),
        "minimum_center_cross_encoder_cosine": float(
            (center_write * center_query).sum(dim=-1).min()
        ),
        "minimum_write_alias_center_cosine": float(
            (write * center_write[labels]).sum(dim=-1).min()
        ),
        "minimum_query_alias_center_cosine": float(
            (query * center_query[labels]).sum(dim=-1).min()
        ),
    }


@torch.no_grad()
def evaluate_slot_sequences(
    policy: FrozenSlotPolicy,
    *,
    memory_kind: str,
    seed: int,
    length: int,
    radius: float,
    batch_size: int,
) -> dict[str, float | int]:
    if memory_kind not in {"triality", "direct"}:
        raise ValueError(f"unknown memory kind: {memory_kind}")
    dtype = torch.float64
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(650_000 + seed + 17 * length)
    alias_generator = torch.Generator().manual_seed(660_000 + seed + 19 * length)
    world = AliasWorld.create(seed, dtype=dtype, device=device)
    rho = triality_tensor(dtype=dtype)
    actions = teacher_actions(seed, dtype=dtype, device=device)
    vector_actions, positive_actions, negative_actions = actions.unbind(dim=1)
    geometric_keys = random_unit(
        (batch_size, KEYS, DIMENSION), generator=generator, dtype=dtype, device=device
    )
    values = torch.zeros(batch_size, KEYS, DIMENSION, dtype=dtype)
    memory = torch.zeros(batch_size, SLOTS, DIMENSION, dtype=dtype)
    cosines: list[torch.Tensor] = []
    errors: list[torch.Tensor] = []

    def write(batch_index: torch.Tensor, labels: torch.Tensor) -> None:
        nonlocal memory, values
        count = int(batch_index.numel())
        if count == 0:
            return
        aliases = world.sample(labels, radius=radius, generator=alias_generator)
        address = policy.routes(aliases, labels, side="write")
        value = random_unit(
            (count, DIMENSION), generator=generator, dtype=dtype, device=device
        )
        key = geometric_keys[batch_index, labels]
        payload = triality_bind(key, value, rho) if memory_kind == "triality" else value
        selected = memory[batch_index]
        memory[batch_index] = (
            (1.0 - address[..., None]) * selected
            + address[..., None] * payload[:, None, :]
        )
        values[batch_index, labels] = value

    all_batches = torch.arange(batch_size)
    for label in range(KEYS):
        write(all_batches, torch.full((batch_size,), label, dtype=torch.long))

    for _ in range(max(0, length - KEYS)):
        event = torch.rand(batch_size, generator=generator)
        rotate_batch = torch.nonzero(event < 0.35, as_tuple=False).flatten()
        write_batch = torch.nonzero((event >= 0.35) & (event < 0.70), as_tuple=False).flatten()
        query_batch = torch.nonzero(event >= 0.70, as_tuple=False).flatten()
        if rotate_batch.numel():
            token = torch.randint(ACTION_TOKENS, (rotate_batch.numel(),), generator=generator)
            vector = vector_actions[token]
            positive = positive_actions[token]
            negative = negative_actions[token]
            memory_action = vector if memory_kind == "triality" else negative
            memory[rotate_batch] = torch.einsum(
                "bij,bhj->bhi", memory_action, memory[rotate_batch]
            )
            geometric_keys[rotate_batch] = torch.einsum(
                "bij,bkj->bki", positive, geometric_keys[rotate_batch]
            )
            values[rotate_batch] = torch.einsum(
                "bij,bkj->bki", negative, values[rotate_batch]
            )
        write_labels = torch.randint(KEYS, (write_batch.numel(),), generator=generator)
        write(write_batch, write_labels)
        if query_batch.numel():
            labels = torch.randint(KEYS, (query_batch.numel(),), generator=generator)
            aliases = world.sample(labels, radius=radius, generator=alias_generator)
            address = policy.routes(aliases, labels, side="query")
            if memory_kind == "triality":
                candidates = triality_unbind_negative(
                    geometric_keys[query_batch, labels, None, :], memory[query_batch], rho
                )
            else:
                candidates = memory[query_batch]
            prediction = (address[..., None] * candidates).sum(dim=1)
            target = values[query_batch, labels]
            cosines.append(F.cosine_similarity(prediction, target, dim=-1))
            errors.append(
                (prediction - target).square().sum(dim=-1)
                / target.square().sum(dim=-1).clamp_min(1e-30)
            )
    cosine = torch.cat(cosines)
    error = torch.cat(errors)
    return {
        "length": length,
        "queries": int(cosine.numel()),
        "mean_query_cosine": float(cosine.mean()),
        "minimum_query_cosine": float(cosine.min()),
        "mean_relative_squared_error": float(error.mean()),
        "maximum_relative_squared_error": float(error.max()),
    }


@torch.no_grad()
def slot_scan_parity(
    policy: FrozenSlotPolicy, *, memory_kind: str, seed: int, radius: float
) -> dict[str, float | int]:
    dtype = torch.float64
    device = torch.device("cpu")
    batch, length = 3, 31
    generator = torch.Generator().manual_seed(670_000 + seed)
    alias_generator = torch.Generator().manual_seed(680_000 + seed)
    world = AliasWorld.create(seed, dtype=dtype, device=device)
    rho = triality_tensor(dtype=dtype)
    actions = teacher_actions(seed, dtype=dtype, device=device)
    vector_actions, positive_actions, negative_actions = actions.unbind(dim=1)
    geometric_keys = random_unit(
        (batch, KEYS, DIMENSION), generator=generator, dtype=dtype, device=device
    )
    retention_steps: list[torch.Tensor] = []
    action_steps: list[torch.Tensor] = []
    drive_steps: list[torch.Tensor] = []
    eye = torch.eye(DIMENSION, dtype=dtype).expand(batch, -1, -1)
    for position in range(length):
        if position % 3 == 1:
            token = torch.randint(ACTION_TOKENS, (batch,), generator=generator)
            vector, positive, negative = (
                vector_actions[token], positive_actions[token], negative_actions[token]
            )
            retention_steps.append(torch.ones(batch, SLOTS, dtype=dtype))
            action_steps.append(vector if memory_kind == "triality" else negative)
            drive_steps.append(torch.zeros(batch, SLOTS, DIMENSION, dtype=dtype))
            geometric_keys = torch.einsum("bij,bkj->bki", positive, geometric_keys)
        else:
            labels = torch.randint(KEYS, (batch,), generator=generator)
            aliases = world.sample(labels, radius=radius, generator=alias_generator)
            address = policy.routes(aliases, labels, side="write")
            value = random_unit(
                (batch, DIMENSION), generator=generator, dtype=dtype, device=device
            )
            key = geometric_keys[torch.arange(batch), labels]
            payload = triality_bind(key, value, rho) if memory_kind == "triality" else value
            retention_steps.append(1.0 - address)
            action_steps.append(eye)
            drive_steps.append(address[..., None] * payload[:, None, :])
    transition = SlotTransition(
        torch.stack(retention_steps, dim=1),
        torch.stack(action_steps, dim=1),
        torch.stack(drive_steps, dim=1),
    )
    initial = torch.zeros(batch, SLOTS, DIMENSION, dtype=dtype)
    parallel = apply_slot(associative_slot_scan(transition), initial[:, None])
    state = initial
    recurrent: list[torch.Tensor] = []
    for position in range(length):
        state = apply_slot(
            SlotTransition(
                transition.retention[:, position],
                transition.action[:, position],
                transition.drive[:, position],
            ),
            state,
        )
        recurrent.append(state)
    return {
        "length": length,
        "streaming_state_scalars": SLOTS * DIMENSION,
        "parallel_recurrent_max_error": float(
            (parallel - torch.stack(recurrent, dim=1)).abs().max()
        ),
    }


@torch.no_grad()
def evaluate_key_sequences(
    policy: FrozenKeyPolicy,
    *,
    update_kind: str,
    seed: int,
    length: int,
    radius: float,
    batch_size: int,
) -> dict[str, float | int]:
    if update_kind not in {"delta", "fast_weight"}:
        raise ValueError(f"unknown key-memory update: {update_kind}")
    dtype = torch.float64
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(690_000 + seed + 17 * length)
    alias_generator = torch.Generator().manual_seed(700_000 + seed + 19 * length)
    world = AliasWorld.create(seed, dtype=dtype, device=device)
    negative_actions = teacher_actions(seed, dtype=dtype, device=device)[:, 2]
    values = torch.zeros(batch_size, KEYS, DIMENSION, dtype=dtype)
    memory = torch.zeros(batch_size, DIMENSION, DIMENSION, dtype=dtype)
    cosines: list[torch.Tensor] = []
    errors: list[torch.Tensor] = []

    def write(batch_index: torch.Tensor, labels: torch.Tensor) -> None:
        nonlocal memory, values
        count = int(batch_index.numel())
        if count == 0:
            return
        aliases = world.sample(labels, radius=radius, generator=alias_generator)
        key = policy.keys(aliases, side="write")
        value = random_unit(
            (count, DIMENSION), generator=generator, dtype=dtype, device=device
        )
        selected = memory[batch_index]
        if update_kind == "delta":
            old = torch.einsum("bi,bij->bj", key, selected)
            drive_value = value - old
        else:
            drive_value = value
        memory[batch_index] = selected + key[..., None] * drive_value[:, None, :]
        values[batch_index, labels] = value

    all_batches = torch.arange(batch_size)
    for label in range(KEYS):
        write(all_batches, torch.full((batch_size,), label, dtype=torch.long))
    for _ in range(max(0, length - KEYS)):
        event = torch.rand(batch_size, generator=generator)
        rotate_batch = torch.nonzero(event < 0.35, as_tuple=False).flatten()
        write_batch = torch.nonzero((event >= 0.35) & (event < 0.70), as_tuple=False).flatten()
        query_batch = torch.nonzero(event >= 0.70, as_tuple=False).flatten()
        if rotate_batch.numel():
            token = torch.randint(ACTION_TOKENS, (rotate_batch.numel(),), generator=generator)
            negative = negative_actions[token]
            memory[rotate_batch] = torch.einsum(
                "bij,bhj->bhi", negative, memory[rotate_batch]
            )
            values[rotate_batch] = torch.einsum(
                "bij,bkj->bki", negative, values[rotate_batch]
            )
        write_labels = torch.randint(KEYS, (write_batch.numel(),), generator=generator)
        write(write_batch, write_labels)
        if query_batch.numel():
            labels = torch.randint(KEYS, (query_batch.numel(),), generator=generator)
            aliases = world.sample(labels, radius=radius, generator=alias_generator)
            key = policy.keys(aliases, side="query")
            prediction = torch.einsum("bi,bij->bj", key, memory[query_batch])
            target = values[query_batch, labels]
            cosines.append(F.cosine_similarity(prediction, target, dim=-1))
            errors.append(
                (prediction - target).square().sum(dim=-1)
                / target.square().sum(dim=-1).clamp_min(1e-30)
            )
    cosine = torch.cat(cosines)
    error = torch.cat(errors)
    return {
        "length": length,
        "queries": int(cosine.numel()),
        "mean_query_cosine": float(cosine.mean()),
        "minimum_query_cosine": float(cosine.min()),
        "mean_relative_squared_error": float(error.mean()),
        "maximum_relative_squared_error": float(error.max()),
    }


def affine_prefix_scan(
    action: torch.Tensor, drive: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    offset = 1
    while offset < action.shape[1]:
        after_action = action[:, offset:]
        before_action = action[:, :-offset]
        composed_action = after_action @ before_action
        composed_drive = torch.einsum(
            "...ij,...j->...i", after_action, drive[:, :-offset]
        ) + drive[:, offset:]
        action = torch.cat((action[:, :offset], composed_action), dim=1)
        drive = torch.cat((drive[:, :offset], composed_drive), dim=1)
        offset *= 2
    return action, drive


@torch.no_grad()
def key_scan_parity(
    policy: FrozenKeyPolicy, *, update_kind: str, seed: int, radius: float
) -> dict[str, float | int]:
    dtype = torch.float64
    batch, length, width = 2, 23, DIMENSION * DIMENSION
    generator = torch.Generator().manual_seed(710_000 + seed)
    alias_generator = torch.Generator().manual_seed(720_000 + seed)
    world = AliasWorld.create(seed, dtype=dtype, device=torch.device("cpu"))
    negative_actions = teacher_actions(
        seed, dtype=dtype, device=torch.device("cpu")
    )[:, 2]
    actions: list[torch.Tensor] = []
    drives: list[torch.Tensor] = []
    identity_key = torch.eye(DIMENSION, dtype=dtype)
    identity_state = torch.eye(width, dtype=dtype).expand(batch, -1, -1)
    for position in range(length):
        if position % 3 == 1:
            token = torch.randint(ACTION_TOKENS, (batch,), generator=generator)
            negative = negative_actions[token]
            actions.append(
                torch.stack(
                    [torch.kron(identity_key, matrix) for matrix in negative], dim=0
                )
            )
            drives.append(torch.zeros(batch, width, dtype=dtype))
        else:
            labels = torch.randint(KEYS, (batch,), generator=generator)
            aliases = world.sample(labels, radius=radius, generator=alias_generator)
            key = policy.keys(aliases, side="write")
            value = random_unit(
                (batch, DIMENSION),
                generator=generator,
                dtype=dtype,
                device=torch.device("cpu"),
            )
            if update_kind == "delta":
                projection = identity_key[None] - key[..., None] * key[:, None, :]
                actions.append(
                    torch.stack(
                        [torch.kron(matrix, identity_key) for matrix in projection],
                        dim=0,
                    )
                )
            else:
                actions.append(identity_state)
            drives.append((key[..., None] * value[:, None, :]).flatten(1))
    action = torch.stack(actions, dim=1)
    drive = torch.stack(drives, dim=1)
    prefix_action, prefix_drive = affine_prefix_scan(action, drive)
    initial = torch.zeros(batch, width, dtype=dtype)
    parallel = torch.einsum("blij,bj->bli", prefix_action, initial) + prefix_drive
    state = initial
    recurrent: list[torch.Tensor] = []
    for position in range(length):
        state = torch.einsum("bij,bj->bi", action[:, position], state) + drive[:, position]
        recurrent.append(state)
    return {
        "length": length,
        "streaming_state_scalars": width,
        "parallel_recurrent_max_error": float(
            (parallel - torch.stack(recurrent, dim=1)).abs().max()
        ),
    }


def slot_variant_report(
    trained: TrainedSlotPolicy,
    *,
    seed: int,
    memory_kind: str,
    dense: bool,
) -> dict[str, object]:
    lengths = DENSE_LENGTHS if dense else (32, 128, 512, 2048)
    validation = alias_diagnostics(
        trained.policy, seed=seed, radius=VALIDATION_RADIUS
    )
    test = alias_diagnostics(trained.policy, seed=seed, radius=TEST_RADIUS)
    evaluation = [
        evaluate_slot_sequences(
            trained.policy,
            memory_kind=memory_kind,
            seed=seed,
            length=length,
            radius=TEST_RADIUS,
            batch_size=160 if length <= 16 else 48,
        )
        for length in lengths
    ]
    parity = slot_scan_parity(
        trained.policy, memory_kind=memory_kind, seed=seed, radius=TEST_RADIUS
    )
    passed = (
        int(test["write_center_collisions"]) == 0
        and int(test["query_center_collisions"]) == 0
        and float(test["center_cross_encoder_agreement"]) == 1.0
        and float(test["write_alias_agreement"]) >= 0.99
        and float(test["query_alias_agreement"]) >= 0.99
        and min(float(row["mean_query_cosine"]) for row in evaluation) >= 0.995
        and min(float(row["minimum_query_cosine"]) for row in evaluation) >= 0.98
        and max(float(row["mean_relative_squared_error"]) for row in evaluation) < 1e-3
        and float(parity["parallel_recurrent_max_error"]) < 1e-9
        and int(parity["streaming_state_scalars"]) == 64
    )
    return {
        "memory_kind": memory_kind,
        "training": {
            "initial_loss": trained.initial_loss,
            "final_loss": trained.final_loss,
            "final_endpoint_loss": trained.final_endpoint_loss,
            "final_balance_loss": trained.final_balance_loss,
        },
        "validation_aliases": validation,
        "test_aliases": test,
        "evaluation": evaluation,
        "scan": parity,
        "passed": passed,
    }


def key_variant_report(
    trained: TrainedKeyPolicy,
    *,
    seed: int,
    update_kind: str,
    dense: bool,
) -> dict[str, object]:
    lengths = DENSE_LENGTHS if dense else (32, 128, 512, 2048)
    validation = key_diagnostics(
        trained.policy, seed=seed, radius=VALIDATION_RADIUS
    )
    test = key_diagnostics(trained.policy, seed=seed, radius=TEST_RADIUS)
    evaluation = [
        evaluate_key_sequences(
            trained.policy,
            update_kind=update_kind,
            seed=seed,
            length=length,
            radius=TEST_RADIUS,
            batch_size=160 if length <= 16 else 48,
        )
        for length in lengths
    ]
    parity = key_scan_parity(
        trained.policy, update_kind=update_kind, seed=seed, radius=TEST_RADIUS
    )
    passed = (
        float(test["write_center_gram_max_error"]) < 1e-3
        and float(test["query_center_gram_max_error"]) < 1e-3
        and float(test["minimum_center_cross_encoder_cosine"]) >= 0.999
        and float(test["minimum_write_alias_center_cosine"]) >= 0.99
        and float(test["minimum_query_alias_center_cosine"]) >= 0.99
        and min(float(row["mean_query_cosine"]) for row in evaluation) >= 0.995
        and min(float(row["minimum_query_cosine"]) for row in evaluation) >= 0.98
        and max(float(row["mean_relative_squared_error"]) for row in evaluation) < 1e-3
        and float(parity["parallel_recurrent_max_error"]) < 1e-9
        and int(parity["streaming_state_scalars"]) == 64
    )
    return {
        "update_kind": update_kind,
        "training": {
            "initial_loss": trained.initial_loss,
            "final_loss": trained.final_loss,
            "final_endpoint_loss": trained.final_endpoint_loss,
            "final_whitening_loss": trained.final_whitening_loss,
        },
        "validation_aliases": validation,
        "test_aliases": test,
        "evaluation": evaluation,
        "scan": parity,
        "passed": passed,
    }


def run_seed(
    seed: int, *, device: torch.device, steps_per_stage: int, dense: bool
) -> dict[str, object]:
    trained = {
        variant: train_slot_policy(
            variant, seed=seed, device=device, steps_per_stage=steps_per_stage
        )
        for variant in SLOT_VARIANTS
    }
    reports = {
        variant: slot_variant_report(
            result, seed=seed, memory_kind="triality", dense=dense
        )
        for variant, result in trained.items()
    }
    reports["direct_joint"] = slot_variant_report(
        trained["learned_both_joint"], seed=seed, memory_kind="direct", dense=dense
    )
    trained_key = train_key_policy(
        seed=seed, device=device, steps_per_stage=steps_per_stage
    )
    oracle_world = AliasWorld.create(
        seed, dtype=torch.float64, device=torch.device("cpu")
    )
    oracle_key = TrainedKeyPolicy(
        policy=FrozenKeyPolicy(oracle_world.centers, oracle_world.centers),
        initial_loss=0.0,
        final_loss=0.0,
        final_endpoint_loss=0.0,
        final_whitening_loss=0.0,
    )
    reports["delta_oracle"] = key_variant_report(
        oracle_key, seed=seed, update_kind="delta", dense=dense
    )
    reports["fast_weight_oracle"] = key_variant_report(
        oracle_key, seed=seed, update_kind="fast_weight", dense=dense
    )
    reports["delta_joint"] = key_variant_report(
        trained_key, seed=seed, update_kind="delta", dense=dense
    )
    reports["fast_weight_joint"] = key_variant_report(
        trained_key, seed=seed, update_kind="fast_weight", dense=dense
    )
    return {
        "seed": seed,
        "world_audit": alias_world_audit(seed),
        "variants": reports,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    names = tuple(rows[0]["variants"])
    passes = {
        name: sum(bool(row["variants"][name]["passed"]) for row in rows)
        for name in names
    }
    joint_beats_independent = 0
    joint_beats_untrained = 0
    for row in rows:
        joint = row["variants"]["learned_both_joint"]
        independent = row["variants"]["learned_both_independent"]
        untrained = row["variants"]["learned_both_joint_untrained"]
        if (
            joint["test_aliases"]["write_center_collisions"]
            < independent["test_aliases"]["write_center_collisions"]
            or joint["evaluation"][-1]["mean_query_cosine"]
            > independent["evaluation"][-1]["mean_query_cosine"]
        ):
            joint_beats_independent += 1
        if (
            joint["evaluation"][-1]["mean_query_cosine"]
            > untrained["evaluation"][-1]["mean_query_cosine"]
        ):
            joint_beats_untrained += 1
    gates = {
        "oracle": passes["oracle_both"] == len(rows),
        "one_sided_write_oracle": passes["oracle_write_learned_query"] >= 8,
        "one_sided_query_oracle": passes["learned_write_oracle_query"] >= 8,
        "joint_reliability": passes["learned_both_joint"] >= 8,
        "joint_beats_independent": joint_beats_independent >= 8,
        "joint_beats_untrained": joint_beats_untrained == len(rows),
        "world_contract": max(
            max(float(value) for value in row["world_audit"].values()) for row in rows
        )
        < 1e-12,
    }
    return {
        "seeds": len(rows),
        "passes": passes,
        "joint_beats_independent": joint_beats_independent,
        "joint_beats_untrained": joint_beats_untrained,
        "gates": gates,
        "passed": all(gates.values()),
    }


def run(
    seeds: list[int], *, device: torch.device, steps_per_stage: int, dense: bool
) -> dict[str, object]:
    rows = [
        run_seed(seed, device=device, steps_per_stage=steps_per_stage, dense=dense)
        for seed in seeds
    ]
    return {
        "experiment": "Spin(8) continuous-alias content routing",
        "alias_dimension": ALIAS_DIMENSION,
        "train_radii": TRAIN_RADII,
        "validation_radius": VALIDATION_RADIUS,
        "test_radius": TEST_RADIUS,
        "steps_per_stage": steps_per_stage,
        "device": str(device),
        "results": rows,
        "summary": summarize(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--steps-per-stage", type=int, default=300)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/spin8_continuous_alias_seeds0_9.json"),
    )
    args = parser.parse_args()
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    report = run(
        args.seeds,
        device=torch.device(device_name),
        steps_per_stage=args.steps_per_stage,
        dense=args.dense,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
