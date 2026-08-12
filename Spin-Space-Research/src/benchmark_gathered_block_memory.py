"""Benchmark actual gathered-block direct and delta recurrent memory access."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

SLOTS_PER_BLOCK = 8
ALIAS_DIMENSION = 8
VALUE_DIMENSION = 8
TEMPERATURE = 0.35
UPDATE_LAWS = ("direct", "delta")
IMPLEMENTATIONS = ("dense_full", "block_masked_full", "block_gathered")
VARIANTS = tuple(
    f"{law}_{implementation}"
    for law in UPDATE_LAWS
    for implementation in IMPLEMENTATIONS
)


@dataclass(frozen=True)
class GatherProblem:
    state: torch.Tensor
    write_alias: torch.Tensor
    query_alias: torch.Tensor
    value: torch.Tensor
    coarse_weight: torch.Tensor
    fine_weight: torch.Tensor

    @property
    def batch(self) -> int:
        return self.state.shape[0]

    @property
    def blocks(self) -> int:
        return self.state.shape[1]

    @property
    def slots(self) -> int:
        return self.blocks * SLOTS_PER_BLOCK


def make_problem(
    *,
    batch: int,
    slots: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> GatherProblem:
    if slots % SLOTS_PER_BLOCK:
        raise ValueError("slots must be divisible by slots per block")
    blocks = slots // SLOTS_PER_BLOCK
    generator = torch.Generator().manual_seed(seed)

    def sample(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=dtype).to(device)

    return GatherProblem(
        state=0.1
        * sample(batch, blocks, SLOTS_PER_BLOCK, VALUE_DIMENSION),
        write_alias=F.normalize(sample(batch, ALIAS_DIMENSION), dim=-1),
        query_alias=F.normalize(sample(batch, ALIAS_DIMENSION), dim=-1),
        value=F.normalize(sample(batch, VALUE_DIMENSION), dim=-1),
        coarse_weight=F.normalize(
            sample(blocks, ALIAS_DIMENSION), dim=-1
        ),
        fine_weight=F.normalize(
            sample(blocks, SLOTS_PER_BLOCK, ALIAS_DIMENSION), dim=-1
        ),
    )


def _dense_route(alias: torch.Tensor, fine_weight: torch.Tensor) -> torch.Tensor:
    flat = fine_weight.reshape(-1, ALIAS_DIMENSION)
    return F.softmax(alias @ flat.transpose(0, 1) / TEMPERATURE, dim=-1)


def _local_route(
    alias: torch.Tensor,
    coarse_weight: torch.Tensor,
    fine_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    block = (alias @ coarse_weight.transpose(0, 1)).argmax(dim=-1)
    selected_weight = fine_weight[block]
    local = F.softmax(
        torch.einsum("bd,bhd->bh", alias, selected_weight) / TEMPERATURE,
        dim=-1,
    )
    return block, local


def _expand_local_route(
    block: torch.Tensor, local: torch.Tensor, *, slots: int
) -> torch.Tensor:
    indices = block[:, None] * SLOTS_PER_BLOCK + torch.arange(
        SLOTS_PER_BLOCK, device=block.device
    )
    return torch.zeros(
        block.shape[0], slots, device=local.device, dtype=local.dtype
    ).scatter(-1, indices, local)


def _update_full(
    state: torch.Tensor,
    route: torch.Tensor,
    value: torch.Tensor,
    law: str,
) -> None:
    flat = state.view(state.shape[0], -1, VALUE_DIMENSION)
    if law == "direct":
        flat.mul_(1.0 - route[..., None]).add_(route[..., None] * value[:, None])
        return
    if law == "delta":
        key = F.normalize(route, dim=-1)
        prediction = torch.einsum("bh,bhv->bv", key, flat)
        flat.add_(key[..., None] * (value - prediction)[:, None])
        return
    raise ValueError(f"unknown update law: {law}")


def _read_full(state: torch.Tensor, route: torch.Tensor, law: str) -> torch.Tensor:
    flat = state.view(state.shape[0], -1, VALUE_DIMENSION)
    key = route if law == "direct" else F.normalize(route, dim=-1)
    return torch.einsum("bh,bhv->bv", key, flat)


def _update_local(
    selected: torch.Tensor,
    route: torch.Tensor,
    value: torch.Tensor,
    law: str,
) -> torch.Tensor:
    if law == "direct":
        return (1.0 - route[..., None]) * selected + route[..., None] * value[:, None]
    if law == "delta":
        key = F.normalize(route, dim=-1)
        prediction = torch.einsum("bh,bhv->bv", key, selected)
        return selected + key[..., None] * (value - prediction)[:, None]
    raise ValueError(f"unknown update law: {law}")


def _read_local(
    selected: torch.Tensor, route: torch.Tensor, law: str
) -> torch.Tensor:
    key = route if law == "direct" else F.normalize(route, dim=-1)
    return torch.einsum("bh,bhv->bv", key, selected)


@torch.no_grad()
def memory_step(
    state: torch.Tensor,
    problem: GatherProblem,
    *,
    law: str,
    implementation: str,
) -> torch.Tensor:
    if implementation == "dense_full":
        write_route = _dense_route(problem.write_alias, problem.fine_weight)
        query_route = _dense_route(problem.query_alias, problem.fine_weight)
        _update_full(state, write_route, problem.value, law)
        return _read_full(state, query_route, law)

    write_block, write_local = _local_route(
        problem.write_alias, problem.coarse_weight, problem.fine_weight
    )
    query_block, query_local = _local_route(
        problem.query_alias, problem.coarse_weight, problem.fine_weight
    )
    if implementation == "block_masked_full":
        write_route = _expand_local_route(
            write_block, write_local, slots=problem.slots
        )
        query_route = _expand_local_route(
            query_block, query_local, slots=problem.slots
        )
        _update_full(state, write_route, problem.value, law)
        return _read_full(state, query_route, law)
    if implementation == "block_gathered":
        batch_index = torch.arange(problem.batch, device=state.device)
        selected = state[batch_index, write_block]
        updated = _update_local(selected, write_local, problem.value, law)
        state[batch_index, write_block] = updated
        query_state = state[batch_index, query_block]
        return _read_local(query_state, query_local, law)
    raise ValueError(f"unknown implementation: {implementation}")


@torch.no_grad()
def correctness(problem: GatherProblem) -> dict[str, object]:
    rows: dict[str, object] = {}
    passed = True
    for law in UPDATE_LAWS:
        masked_state = problem.state.clone()
        gathered_state = problem.state.clone()
        masked_prediction = memory_step(
            masked_state,
            problem,
            law=law,
            implementation="block_masked_full",
        )
        gathered_prediction = memory_step(
            gathered_state,
            problem,
            law=law,
            implementation="block_gathered",
        )
        state_error = float((masked_state - gathered_state).abs().max())
        prediction_error = float(
            (masked_prediction - gathered_prediction).abs().max()
        )
        finite = bool(
            torch.isfinite(masked_state).all()
            and torch.isfinite(gathered_state).all()
            and torch.isfinite(masked_prediction).all()
            and torch.isfinite(gathered_prediction).all()
        )
        rows[law] = {
            "maximum_state_error": state_error,
            "maximum_prediction_error": prediction_error,
            "all_finite": finite,
        }
        passed = passed and state_error <= 1e-5 and prediction_error <= 1e-5 and finite
    return {"passed": passed, "rows": rows}


def cyclic_orders(
    variants: tuple[str, ...], *, rotation: int, timing_blocks: int
) -> list[tuple[str, ...]]:
    if not variants:
        raise ValueError("variants must not be empty")
    if timing_blocks < 1:
        raise ValueError("timing_blocks must be positive")
    return [
        variants[offset:] + variants[:offset]
        for block in range(timing_blocks)
        for offset in ((rotation + block) % len(variants),)
    ]


def _time_variants(
    problem: GatherProblem,
    *,
    variants: tuple[str, ...],
    rotation: int,
    warmup: int,
    timing_blocks: int,
    inner_calls: int,
) -> tuple[dict[str, list[float]], list[tuple[str, ...]]]:
    states = {variant: problem.state.clone() for variant in variants}
    for variant in variants:
        law, implementation = variant.split("_", maxsplit=1)
        for _ in range(warmup):
            memory_step(
                states[variant],
                problem,
                law=law,
                implementation=implementation,
            )
    if problem.state.is_cuda:
        torch.cuda.synchronize(problem.state.device)
    samples = {variant: [] for variant in variants}
    orders = cyclic_orders(
        variants, rotation=rotation, timing_blocks=timing_blocks
    )
    for order in orders:
        for variant in order:
            law, implementation = variant.split("_", maxsplit=1)
            state = states[variant]
            if state.is_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(inner_calls):
                    memory_step(
                        state,
                        problem,
                        law=law,
                        implementation=implementation,
                    )
                end.record()
                end.synchronize()
                elapsed = float(start.elapsed_time(end)) / inner_calls
            else:
                start_time = time.perf_counter()
                for _ in range(inner_calls):
                    memory_step(
                        state,
                        problem,
                        law=law,
                        implementation=implementation,
                    )
                elapsed = (
                    1000.0 * (time.perf_counter() - start_time) / inner_calls
                )
            samples[variant].append(elapsed)
    return samples, orders


def _incremental_memory(
    problem: GatherProblem, *, law: str, implementation: str
) -> int | None:
    if not problem.state.is_cuda:
        return None
    state = problem.state.clone()
    torch.cuda.synchronize(state.device)
    torch.cuda.reset_peak_memory_stats(state.device)
    baseline = torch.cuda.memory_allocated(state.device)
    memory_step(state, problem, law=law, implementation=implementation)
    torch.cuda.synchronize(state.device)
    return int(torch.cuda.max_memory_allocated(state.device) - baseline)


def _summary(samples: list[float]) -> dict[str, object]:
    ordered = sorted(samples)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else 0.5 * (ordered[middle - 1] + ordered[middle])
    )
    return {
        "median_ms": median,
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
        "raw_ms": samples,
    }


def _nvidia_smi() -> dict[str, str] | None:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    name, driver, memory = [part.strip() for part in output.split(",", maxsplit=2)]
    return {"name": name, "driver": driver, "memory_total": memory}


def benchmark(
    *,
    device: torch.device,
    slots: tuple[int, ...],
    batches: tuple[int, ...],
    warmup: int,
    timing_blocks: int,
    inner_calls: int,
    seed: int,
) -> dict[str, object]:
    dtype = torch.float32
    rows = []
    all_passed = True
    for slot_count in slots:
        for batch in batches:
            problem = make_problem(
                batch=batch,
                slots=slot_count,
                device=device,
                dtype=dtype,
                seed=seed + slot_count + 10_000 * batch,
            )
            parity = correctness(problem)
            all_passed = all_passed and bool(parity["passed"])
            timing: dict[str, object] = {}
            rotation = (slot_count + batch + seed) % len(VARIANTS)
            samples_by_variant, timing_orders = _time_variants(
                problem,
                variants=VARIANTS,
                rotation=rotation,
                warmup=warmup,
                timing_blocks=timing_blocks,
                inner_calls=inner_calls,
            )
            for variant in VARIANTS:
                law, implementation = variant.split("_", maxsplit=1)
                timing[variant] = {
                    **_summary(samples_by_variant[variant]),
                    "incremental_peak_bytes": _incremental_memory(
                        problem, law=law, implementation=implementation
                    ),
                }
            rows.append(
                {
                    "slots": slot_count,
                    "blocks": slot_count // SLOTS_PER_BLOCK,
                    "batch": batch,
                    "logical_state_scalars": (
                        batch * slot_count * VALUE_DIMENSION
                    ),
                    "correctness": parity,
                    "timing_order": timing_orders[0],
                    "timing_block_orders": timing_orders,
                    "timing": timing,
                }
            )
    shared_parameters = max(slots) * ALIAS_DIMENSION + (
        max(slots) // SLOTS_PER_BLOCK
    ) * ALIAS_DIMENSION
    return {
        "experiment": "actual gathered-block recurrent memory benchmark",
        "protocol": "GATHERED_BLOCK_MEMORY_BENCHMARK_PREREGISTRATION.md",
        "passed": all_passed,
        "device": str(device),
        "dtype": str(dtype),
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nvidia_smi": _nvidia_smi(),
        },
        "grid": {
            "slots": slots,
            "batches": batches,
            "slots_per_block": SLOTS_PER_BLOCK,
            "alias_dimension": ALIAS_DIMENSION,
            "value_dimension": VALUE_DIMENSION,
            "warmup": warmup,
            "timing_blocks": timing_blocks,
            "inner_calls": inner_calls,
        },
        "router_parameters_at_max_slots": {
            "shared": shared_parameters,
            "three_independent": 3 * shared_parameters,
            "shared_float32_bytes": 4 * shared_parameters,
            "three_independent_float32_bytes": 12 * shared_parameters,
        },
        "rows": rows,
        "claim_boundary": {
            "inference_only": True,
            "fused_training_kernel": False,
            "model_level_quality": False,
            "triality_specific_capacity": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--slots", nargs="+", type=int, default=(64, 256, 1024, 4096))
    parser.add_argument("--batches", nargs="+", type=int, default=(1, 16, 64))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--timing-blocks", type=int, default=25)
    parser.add_argument("--inner-calls", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1_080_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    report = benchmark(
        device=torch.device(args.device),
        slots=tuple(args.slots),
        batches=tuple(args.batches),
        warmup=args.warmup,
        timing_blocks=args.timing_blocks,
        inner_calls=args.inner_calls,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "rows": len(report["rows"])}, indent=2))


if __name__ == "__main__":
    main()
