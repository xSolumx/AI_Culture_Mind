"""Blind latent-slot routing for associative Spin(8) triality memory.

Training exposes one logical key per episode.  Mixed-key sequences are held
out, so an independently normalized router can solve training while silently
colliding in latent slots.  A jointly Sinkhorn-normalized family supplies the
missing global constraint without making the recurrence state-dependent.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from spin8_triality import SPIN8_BIVECTOR_DIM, spin8_actions, torch_triality_generators
from spin8_triality_lift import triality_bind, triality_tensor, triality_unbind_negative
from spin8_triality_memory import (
    SlotTransition,
    apply_slot,
    associative_slot_scan,
)


KEYS = 8
SLOTS = 8
DIMENSION = 8
ACTION_TOKENS = 4
EVALUATION_LENGTHS = (32, 128, 512, 2048)
DENSE_LENGTHS = (16, 32, 64, 128, 256, 512, 1024, 2048)


def random_unit(
    shape: tuple[int, ...], *, generator: torch.Generator, dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return F.normalize(
        torch.randn(*shape, generator=generator, dtype=dtype, device=device),
        dim=-1,
    )


def log_sinkhorn(logits: torch.Tensor, temperature: float, iterations: int = 64) -> torch.Tensor:
    """Return a square doubly-stochastic family without tokenwise projection."""

    log_weights = logits / temperature
    for _ in range(iterations):
        log_weights = log_weights - torch.logsumexp(log_weights, dim=-1, keepdim=True)
        log_weights = log_weights - torch.logsumexp(log_weights, dim=-2, keepdim=True)
    return log_weights.exp()


class AddressFamily(nn.Module):
    def __init__(self, kind: str, *, seed: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        if kind not in {"independent", "joint"}:
            raise ValueError(f"unknown address family: {kind}")
        generator = torch.Generator(device=device).manual_seed(seed)
        self.kind = kind
        self.logits = nn.Parameter(
            0.05 * torch.randn(KEYS, SLOTS, generator=generator, dtype=dtype, device=device)
        )

    def forward(self, temperature: float) -> torch.Tensor:
        if self.kind == "independent":
            return F.softmax(self.logits / temperature, dim=-1)
        return log_sinkhorn(self.logits, temperature)


def single_key_endpoint_loss(
    routes: torch.Tensor,
    *,
    horizon: int,
    generator: torch.Generator,
    batch_size: int,
) -> torch.Tensor:
    """Exact co-moving-frame loss for independent single-key episodes.

    Orthogonal transport cancels from squared endpoint error, and triality
    bind/unbind is an isometry for a unit key.  Each logical key is therefore
    vectorized as its own episode; no mixed-key state occurs in this loss.
    """

    dtype, device = routes.dtype, routes.device
    state = torch.zeros(batch_size, KEYS, SLOTS, DIMENSION, dtype=dtype, device=device)
    route = routes[None, :, :, None]
    losses: list[torch.Tensor] = []
    writes = max(2, horizon // 4)
    for _ in range(writes):
        value = random_unit(
            (batch_size, KEYS, DIMENSION), generator=generator, dtype=dtype, device=device
        )
        state = (1.0 - route) * state + route * value[:, :, None, :]
        prediction = (route * state).sum(dim=2)
        losses.append((prediction - value).square().mean())
    endpoint = torch.stack(losses).mean()
    discreteness = (routes * (1.0 - routes)).mean()
    return endpoint + 0.02 * discreteness


@dataclass(frozen=True)
class TrainedRouter:
    routes: torch.Tensor
    initial_loss: float
    final_loss: float
    rounded_collisions: int


def route_statistics(routes: torch.Tensor) -> dict[str, float | int | list[int]]:
    rounded = routes.argmax(dim=-1)
    collisions = KEYS - int(torch.unique(rounded).numel())
    entropy = -(routes.clamp_min(torch.finfo(routes.dtype).tiny).log() * routes).sum(dim=-1)
    return {
        "rounded_slots": [int(value) for value in rounded.cpu()],
        "rounded_collisions": collisions,
        "maximum_row_sum_residual": float((routes.sum(dim=-1) - 1.0).abs().max()),
        "maximum_column_sum_residual": float((routes.sum(dim=-2) - 1.0).abs().max()),
        "mean_row_entropy": float(entropy.mean()),
        "maximum_off_vertex_mass": float((1.0 - routes.max(dim=-1).values).max()),
    }


def train_router(
    kind: str,
    *,
    seed: int,
    device: torch.device,
    steps_per_stage: int = 250,
    batch_size: int = 64,
) -> TrainedRouter:
    dtype = torch.float64
    model = AddressFamily(kind, seed=seed, dtype=dtype, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
    generator = torch.Generator(device=device).manual_seed(100_000 + seed)
    initial_loss = float(
        single_key_endpoint_loss(
            model(0.40), horizon=8, generator=generator, batch_size=batch_size
        ).detach()
    )
    final_loss = initial_loss
    for stage, horizon in enumerate((8, 16, 32)):
        for step in range(steps_per_stage):
            fraction = (stage * steps_per_stage + step) / max(1, 3 * steps_per_stage - 1)
            temperature = 0.40 * (0.08 / 0.40) ** fraction
            routes = model(temperature)
            loss = single_key_endpoint_loss(
                routes, horizon=horizon, generator=generator, batch_size=batch_size
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
    routes = model(0.03).detach().to(device="cpu", dtype=torch.float64)
    statistics = route_statistics(routes)
    return TrainedRouter(
        routes=routes,
        initial_loss=initial_loss,
        final_loss=final_loss,
        rounded_collisions=int(statistics["rounded_collisions"]),
    )


def teacher_actions(seed: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(200_000 + seed)
    coefficients = 0.28 * torch.randn(
        ACTION_TOKENS,
        SPIN8_BIVECTOR_DIM,
        generator=generator,
        dtype=dtype,
        device=device,
    )
    return spin8_actions(coefficients, torch_triality_generators(dtype=dtype, device=device))


@torch.no_grad()
def evaluate_mixed_sequences(
    routes: torch.Tensor,
    *,
    kind: str,
    length: int,
    seed: int,
    batch_size: int = 48,
) -> dict[str, float | int]:
    if kind not in {"triality", "direct"}:
        raise ValueError(f"unknown memory kind: {kind}")
    device = torch.device("cpu")
    dtype = torch.float64
    routes = routes.to(device=device, dtype=dtype)
    generator = torch.Generator().manual_seed(300_000 + seed + 17 * length)
    rho = triality_tensor(dtype=dtype)
    actions = teacher_actions(seed, dtype=dtype, device=device)
    vector_actions, positive_actions, negative_actions = actions.unbind(dim=1)
    keys = random_unit(
        (batch_size, KEYS, DIMENSION), generator=generator, dtype=dtype, device=device
    )
    values = torch.zeros(batch_size, KEYS, DIMENSION, dtype=dtype)
    memory = torch.zeros(batch_size, SLOTS, DIMENSION, dtype=dtype)
    query_cosines: list[torch.Tensor] = []
    query_errors: list[torch.Tensor] = []

    def write(batch_index: torch.Tensor, logical_key: torch.Tensor) -> None:
        nonlocal memory, values
        count = int(batch_index.numel())
        if count == 0:
            return
        new_value = random_unit(
            (count, DIMENSION), generator=generator, dtype=dtype, device=device
        )
        address = routes[logical_key]
        selected_key = keys[batch_index, logical_key]
        payload = (
            triality_bind(selected_key, new_value, rho)
            if kind == "triality"
            else new_value
        )
        selected_memory = memory[batch_index]
        memory[batch_index] = (
            (1.0 - address[..., None]) * selected_memory
            + address[..., None] * payload[:, None, :]
        )
        values[batch_index, logical_key] = new_value

    # Prospective tests always begin with a full logical prefill.  Slot labels
    # remain hidden; only the key IDs and endpoint values are observable.
    all_batches = torch.arange(batch_size)
    for logical_key_value in range(KEYS):
        write(all_batches, torch.full((batch_size,), logical_key_value, dtype=torch.long))

    for _ in range(max(0, length - KEYS)):
        event = torch.rand(batch_size, generator=generator)
        rotate_mask = event < 0.35
        write_mask = (event >= 0.35) & (event < 0.70)
        query_mask = event >= 0.70

        rotate_batch = torch.nonzero(rotate_mask, as_tuple=False).flatten()
        if rotate_batch.numel():
            token = torch.randint(
                ACTION_TOKENS, (rotate_batch.numel(),), generator=generator
            )
            vector = vector_actions[token]
            positive = positive_actions[token]
            negative = negative_actions[token]
            memory_action = vector if kind == "triality" else negative
            memory[rotate_batch] = torch.einsum(
                "bij,bhj->bhi", memory_action, memory[rotate_batch]
            )
            keys[rotate_batch] = torch.einsum(
                "bij,bkj->bki", positive, keys[rotate_batch]
            )
            values[rotate_batch] = torch.einsum(
                "bij,bkj->bki", negative, values[rotate_batch]
            )

        write_batch = torch.nonzero(write_mask, as_tuple=False).flatten()
        write_key = torch.randint(KEYS, (write_batch.numel(),), generator=generator)
        write(write_batch, write_key)

        query_batch = torch.nonzero(query_mask, as_tuple=False).flatten()
        if query_batch.numel():
            query_key = torch.randint(KEYS, (query_batch.numel(),), generator=generator)
            address = routes[query_key]
            if kind == "triality":
                candidates = triality_unbind_negative(
                    keys[query_batch, query_key, None, :], memory[query_batch], rho
                )
            else:
                candidates = memory[query_batch]
            prediction = (address[..., None] * candidates).sum(dim=1)
            target = values[query_batch, query_key]
            cosine = F.cosine_similarity(prediction, target, dim=-1)
            relative_error = (prediction - target).square().sum(dim=-1) / target.square().sum(dim=-1).clamp_min(1e-30)
            query_cosines.append(cosine)
            query_errors.append(relative_error)

    cosines = torch.cat(query_cosines)
    errors = torch.cat(query_errors)
    return {
        "length": length,
        "queries": int(cosines.numel()),
        "mean_query_cosine": float(cosines.mean()),
        "minimum_query_cosine": float(cosines.min()),
        "mean_relative_squared_error": float(errors.mean()),
        "maximum_relative_squared_error": float(errors.max()),
    }


@torch.no_grad()
def scan_parity(
    routes: torch.Tensor, *, kind: str, seed: int, length: int = 31
) -> dict[str, float | int]:
    dtype = torch.float64
    device = torch.device("cpu")
    routes = routes.to(dtype=dtype, device=device)
    generator = torch.Generator().manual_seed(400_000 + seed)
    rho = triality_tensor(dtype=dtype)
    actions = teacher_actions(seed, dtype=dtype, device=device)
    vector_actions, positive_actions, negative_actions = actions.unbind(dim=1)
    batch = 3
    keys = random_unit(
        (batch, KEYS, DIMENSION), generator=generator, dtype=dtype, device=device
    )
    retention_steps: list[torch.Tensor] = []
    action_steps: list[torch.Tensor] = []
    drive_steps: list[torch.Tensor] = []
    eye = torch.eye(DIMENSION, dtype=dtype).expand(batch, -1, -1)
    for position in range(length):
        if position % 3 == 1:
            token = torch.randint(ACTION_TOKENS, (batch,), generator=generator)
            vector = vector_actions[token]
            positive = positive_actions[token]
            negative = negative_actions[token]
            retention_steps.append(torch.ones(batch, SLOTS, dtype=dtype))
            action_steps.append(vector if kind == "triality" else negative)
            drive_steps.append(torch.zeros(batch, SLOTS, DIMENSION, dtype=dtype))
            keys = torch.einsum("bij,bkj->bki", positive, keys)
        else:
            logical_key = torch.randint(KEYS, (batch,), generator=generator)
            address = routes[logical_key]
            value = random_unit(
                (batch, DIMENSION), generator=generator, dtype=dtype, device=device
            )
            selected_key = keys[torch.arange(batch), logical_key]
            payload = (
                triality_bind(selected_key, value, rho)
                if kind == "triality"
                else value
            )
            retention_steps.append(1.0 - address)
            action_steps.append(eye)
            drive_steps.append(address[..., None] * payload[:, None, :])
    transition = SlotTransition(
        retention=torch.stack(retention_steps, dim=1),
        action=torch.stack(action_steps, dim=1),
        drive=torch.stack(drive_steps, dim=1),
    )
    initial = torch.zeros(batch, SLOTS, DIMENSION, dtype=dtype)
    prefixes = associative_slot_scan(transition)
    parallel = apply_slot(prefixes, initial[:, None])
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
    recurrent_states = torch.stack(recurrent, dim=1)
    return {
        "length": length,
        "streaming_state_scalars": SLOTS * DIMENSION,
        "parallel_recurrent_max_error": float((parallel - recurrent_states).abs().max()),
    }


@torch.no_grad()
def rotation_norm_drift(seed: int, *, length: int = 2048) -> dict[str, float]:
    dtype = torch.float64
    generator = torch.Generator().manual_seed(500_000 + seed)
    actions = teacher_actions(seed, dtype=dtype, device=torch.device("cpu"))
    state = random_unit((3, DIMENSION), generator=generator, dtype=dtype, device=torch.device("cpu"))
    maximum = torch.zeros(3, dtype=dtype)
    for _ in range(length):
        token = int(torch.randint(ACTION_TOKENS, (), generator=generator))
        state = torch.einsum("rij,rj->ri", actions[token], state)
        maximum = torch.maximum(maximum, state.norm(dim=-1).log().abs())
    return {
        "vector_max_abs_log_norm_drift": float(maximum[0]),
        "positive_max_abs_log_norm_drift": float(maximum[1]),
        "negative_max_abs_log_norm_drift": float(maximum[2]),
    }


def variant_report(
    routes: torch.Tensor,
    *,
    kind: str,
    seed: int,
    dense: bool,
) -> dict[str, object]:
    lengths = DENSE_LENGTHS if dense else EVALUATION_LENGTHS
    evaluations = [
        evaluate_mixed_sequences(
            routes,
            kind=kind,
            length=length,
            seed=seed,
            batch_size=160 if length <= 16 else 48,
        )
        for length in lengths
    ]
    statistics = route_statistics(routes)
    parity = scan_parity(routes, kind=kind, seed=seed)
    passed = (
        int(statistics["rounded_collisions"]) == 0
        and float(statistics["maximum_row_sum_residual"]) < 1e-6
        and float(statistics["maximum_column_sum_residual"]) < 1e-6
        and min(float(row["mean_query_cosine"]) for row in evaluations) >= 0.995
        and min(float(row["minimum_query_cosine"]) for row in evaluations) >= 0.98
        and max(float(row["mean_relative_squared_error"]) for row in evaluations) < 1e-3
        and float(parity["parallel_recurrent_max_error"]) < 1e-9
    )
    return {
        "memory_kind": kind,
        "routes": statistics,
        "evaluation": evaluations,
        "scan": parity,
        "passed": passed,
    }


def run_seed(
    seed: int,
    *,
    device: torch.device,
    steps_per_stage: int,
    dense: bool,
) -> dict[str, object]:
    untrained_model = AddressFamily(
        "joint", seed=seed, dtype=torch.float64, device=torch.device("cpu")
    )
    untrained_routes = untrained_model(0.03).detach()
    independent = train_router(
        "independent", seed=seed, device=device, steps_per_stage=steps_per_stage
    )
    joint = train_router(
        "joint", seed=seed, device=device, steps_per_stage=steps_per_stage
    )
    oracle_routes = torch.eye(KEYS, dtype=torch.float64)
    variants = {
        "triality_oracle": variant_report(
            oracle_routes, kind="triality", seed=seed, dense=dense
        ),
        "triality_independent": variant_report(
            independent.routes, kind="triality", seed=seed, dense=dense
        ),
        "triality_joint": variant_report(
            joint.routes, kind="triality", seed=seed, dense=dense
        ),
        "triality_joint_untrained": variant_report(
            untrained_routes, kind="triality", seed=seed, dense=dense
        ),
        "direct_joint": variant_report(
            joint.routes, kind="direct", seed=seed, dense=dense
        ),
    }
    variants["triality_independent"]["training"] = {
        "initial_loss": independent.initial_loss,
        "final_loss": independent.final_loss,
    }
    variants["triality_joint"]["training"] = {
        "initial_loss": joint.initial_loss,
        "final_loss": joint.final_loss,
    }
    return {
        "seed": seed,
        "rotation_norm_drift": rotation_norm_drift(seed),
        "variants": variants,
    }


def summarize(seeds: list[dict[str, object]]) -> dict[str, object]:
    joint_passes = sum(bool(row["variants"]["triality_joint"]["passed"]) for row in seeds)
    direct_passes = sum(bool(row["variants"]["direct_joint"]["passed"]) for row in seeds)
    oracle_passes = sum(bool(row["variants"]["triality_oracle"]["passed"]) for row in seeds)
    untrained_passes = sum(
        bool(row["variants"]["triality_joint_untrained"]["passed"]) for row in seeds
    )
    joint_beats_independent = 0
    joint_beats_untrained = 0
    collision_wins = 0
    for row in seeds:
        joint = row["variants"]["triality_joint"]
        independent = row["variants"]["triality_independent"]
        joint_last = joint["evaluation"][-1]["mean_query_cosine"]
        independent_last = independent["evaluation"][-1]["mean_query_cosine"]
        untrained_last = row["variants"]["triality_joint_untrained"]["evaluation"][-1][
            "mean_query_cosine"
        ]
        if joint_last > independent_last:
            joint_beats_independent += 1
        if joint_last > untrained_last:
            joint_beats_untrained += 1
        if joint["routes"]["rounded_collisions"] < independent["routes"]["rounded_collisions"]:
            collision_wins += 1
    gates = {
        "oracle_numerical": oracle_passes == len(seeds),
        "joint_reliability": joint_passes >= max(1, (8 * len(seeds) + 9) // 10),
        "joint_collision_advantage": collision_wins >= max(1, (8 * len(seeds) + 9) // 10),
        "joint_length2048_advantage": joint_beats_independent >= max(1, (8 * len(seeds) + 9) // 10),
        "training_beats_untrained": joint_beats_untrained == len(seeds),
        "untrained_does_not_pass": untrained_passes <= max(0, len(seeds) // 5),
    }
    return {
        "seeds": len(seeds),
        "triality_oracle_passes": oracle_passes,
        "triality_joint_passes": joint_passes,
        "direct_joint_passes": direct_passes,
        "triality_joint_untrained_passes": untrained_passes,
        "joint_fewer_collisions_than_independent": collision_wins,
        "joint_beats_independent_at_longest_length": joint_beats_independent,
        "joint_beats_untrained_at_longest_length": joint_beats_untrained,
        "gates": gates,
        "passed": all(gates.values()),
    }


def run(
    seeds: list[int], *, device: torch.device, steps_per_stage: int, dense: bool
) -> dict[str, object]:
    rows = [
        run_seed(
            seed, device=device, steps_per_stage=steps_per_stage, dense=dense
        )
        for seed in seeds
    ]
    return {
        "experiment": "Spin(8) learned latent-slot addressing",
        "training_contract": "single-key endpoint episodes only",
        "evaluation_contract": "unseen mixed-key overwrite/rotate/query sequences",
        "device": str(device),
        "steps_per_curriculum_stage": steps_per_stage,
        "dense_lengths": dense,
        "results": rows,
        "summary": summarize(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--steps-per-stage", type=int, default=250)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/spin8_learned_address_seeds0_9.json"),
    )
    args = parser.parse_args()
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    device = torch.device(device_name)
    report = run(
        args.seeds,
        device=device,
        steps_per_stage=args.steps_per_stage,
        dense=args.dense,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
