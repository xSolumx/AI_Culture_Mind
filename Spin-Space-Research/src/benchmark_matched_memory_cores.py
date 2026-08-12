"""Blocked, state-matched benchmark for the local memory implementations.

The report separates a pre-encoded recurrence-core tier from an end-to-end
tier with matched 384-parameter write/query encoders.  Input generation stays
outside both tiers.  Timing orders rotate between repetitions, scan schedules
and delta chunk sizes are tuned on a disjoint problem, and training timings
differentiate value-only core gradients from full encoder/value gradients.

This remains an eager PyTorch comparison.  A fused DeltaNet kernel is measured
by the separate Linux/FLA benchmark because Triton is not supported natively
by this Windows environment.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from schurscan_delta_memory import (
    DeltaTransition,
    compose_delta,
    delta_read,
    delta_write_transitions,
    scanned_delta_states,
    value_transport_transitions,
)
from spin8_continuous_alias import ALIAS_DIMENSION, AliasWorld
from spin8_learned_address import DIMENSION, KEYS, SLOTS, teacher_actions
from spin8_triality_lift import (
    triality_bind,
    triality_tensor,
    triality_unbind_negative,
)
from spin8_triality_memory import (
    SlotTransition,
    packed_homogeneous_slot_scan,
)

VARIANTS = (
    "direct_slot_hybrid",
    "triality_slot_hybrid",
    "delta_chunkwise",
    "fast_weight_chunkwise",
)

SLOT_BACKENDS = ("hillis_steele", "work_efficient")
DELTA_CHUNK_SIZES = (16, 32, 64, 128, 256)


@dataclass(frozen=True)
class TimingSummary:
    median_ms: float
    minimum_ms: float
    p05_ms: float
    p20_ms: float
    p80_ms: float
    p95_ms: float
    maximum_ms: float
    mean_ms: float
    stdev_ms: float
    coefficient_of_variation: float
    median_absolute_deviation_ms: float
    repeats: int


@dataclass(frozen=True)
class MatchedProblem:
    routes: torch.Tensor
    values: torch.Tensor
    keys: torch.Tensor
    query_routes: torch.Tensor
    query_keys: torch.Tensor
    vector_actions: torch.Tensor
    negative_actions: torch.Tensor
    write_geometric_keys: torch.Tensor
    query_geometric_keys: torch.Tensor
    write_aliases: torch.Tensor
    query_aliases: torch.Tensor
    encoder_initial_weight: torch.Tensor
    rho: torch.Tensor


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize(samples: list[float]) -> TimingSummary:
    median = statistics.median(samples)
    mean = statistics.fmean(samples)
    stdev = statistics.pstdev(samples)
    return TimingSummary(
        median_ms=median,
        minimum_ms=min(samples),
        p05_ms=_percentile(samples, 0.05),
        p20_ms=_percentile(samples, 0.2),
        p80_ms=_percentile(samples, 0.8),
        p95_ms=_percentile(samples, 0.95),
        maximum_ms=max(samples),
        mean_ms=mean,
        stdev_ms=stdev,
        coefficient_of_variation=stdev / mean,
        median_absolute_deviation_ms=statistics.median(
            abs(sample - median) for sample in samples
        ),
        repeats=len(samples),
    )


def make_problem(
    *,
    batch: int,
    length: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> MatchedProblem:
    generator = torch.Generator().manual_seed(seed + length)
    labels = torch.randint(KEYS, (batch, length), generator=generator)
    query_labels = torch.randint(KEYS, (batch, length), generator=generator)
    tokens = torch.randint(4, (batch, length), generator=generator)
    values = F.normalize(
        torch.randn(batch, length, DIMENSION, generator=generator, dtype=torch.float64),
        dim=-1,
    )
    routes = F.one_hot(labels, SLOTS).to(dtype=torch.float64)
    query_routes = F.one_hot(query_labels, SLOTS).to(dtype=torch.float64)
    keys = F.one_hot(labels, KEYS).to(dtype=torch.float64)
    query_keys = F.one_hot(query_labels, KEYS).to(dtype=torch.float64)
    alias_world = AliasWorld.create(
        seed, dtype=torch.float64, device=torch.device("cpu")
    )
    alias_generator = torch.Generator().manual_seed(90_000 + seed + length)
    write_aliases = alias_world.sample(labels, radius=0.35, generator=alias_generator)
    query_aliases = alias_world.sample(
        query_labels, radius=0.35, generator=alias_generator
    )
    action_table = teacher_actions(
        seed, dtype=torch.float64, device=torch.device("cpu")
    )
    selected = action_table[tokens]
    vector_actions = selected[:, :, 0]
    positive_actions = selected[:, :, 1]
    negative_actions = selected[:, :, 2]
    geometric = F.normalize(
        torch.randn(batch, KEYS, DIMENSION, generator=generator, dtype=torch.float64),
        dim=-1,
    )
    batch_index = torch.arange(batch)
    write_geometric_keys = []
    query_geometric_keys = []
    for position in range(length):
        geometric = torch.einsum(
            "bij,bkj->bki", positive_actions[:, position], geometric
        )
        write_geometric_keys.append(geometric[batch_index, labels[:, position]])
        query_geometric_keys.append(geometric[batch_index, query_labels[:, position]])

    def move(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(dtype=dtype, device=device)

    return MatchedProblem(
        routes=move(routes),
        values=move(values),
        keys=move(keys),
        query_routes=move(query_routes),
        query_keys=move(query_keys),
        vector_actions=move(vector_actions),
        negative_actions=move(negative_actions),
        write_geometric_keys=move(torch.stack(write_geometric_keys, dim=1)),
        query_geometric_keys=move(torch.stack(query_geometric_keys, dim=1)),
        write_aliases=move(write_aliases),
        query_aliases=move(query_aliases),
        encoder_initial_weight=move(alias_world.centers),
        rho=triality_tensor(dtype=dtype).to(device=device),
    )


def slot_backend(length: int) -> str:
    return "hillis_steele" if length < 1024 else "work_efficient"


def direct_forward(
    problem: MatchedProblem,
    values: torch.Tensor,
    *,
    routes: torch.Tensor | None = None,
    query_routes: torch.Tensor | None = None,
    backend: str | None = None,
) -> torch.Tensor:
    routes = problem.routes if routes is None else routes
    query_routes = problem.query_routes if query_routes is None else query_routes
    transition = SlotTransition(
        retention=1.0 - routes,
        action=problem.negative_actions,
        drive=routes[..., None] * values[:, :, None],
    )
    initial = torch.zeros(
        values.shape[0], SLOTS, DIMENSION, dtype=values.dtype, device=values.device
    )
    states = packed_homogeneous_slot_scan(
        transition,
        initial,
        backend=slot_backend(values.shape[1]) if backend is None else backend,
    )
    return (query_routes[..., None] * states).sum(dim=2)


def triality_forward(
    problem: MatchedProblem,
    values: torch.Tensor,
    *,
    routes: torch.Tensor | None = None,
    query_routes: torch.Tensor | None = None,
    backend: str | None = None,
) -> torch.Tensor:
    routes = problem.routes if routes is None else routes
    query_routes = problem.query_routes if query_routes is None else query_routes
    payload = triality_bind(problem.write_geometric_keys, values, problem.rho)
    transition = SlotTransition(
        retention=1.0 - routes,
        action=problem.vector_actions,
        drive=routes[..., None] * payload[:, :, None],
    )
    initial = torch.zeros(
        values.shape[0], SLOTS, DIMENSION, dtype=values.dtype, device=values.device
    )
    states = packed_homogeneous_slot_scan(
        transition,
        initial,
        backend=slot_backend(values.shape[1]) if backend is None else backend,
    )
    candidates = triality_unbind_negative(
        problem.query_geometric_keys[:, :, None], states, problem.rho
    )
    return (query_routes[..., None] * candidates).sum(dim=2)


def delta_forward(
    problem: MatchedProblem,
    values: torch.Tensor,
    *,
    keys: torch.Tensor | None = None,
    query_keys: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    keys = problem.keys if keys is None else keys
    query_keys = problem.query_keys if query_keys is None else query_keys
    beta = torch.ones(values.shape[:2], dtype=values.dtype, device=values.device)
    write = delta_write_transitions(keys, values, beta)
    transport = value_transport_transitions(
        problem.negative_actions, key_dimension=KEYS
    )
    transition = compose_delta(write, transport)
    initial = torch.zeros(
        values.shape[0], KEYS, DIMENSION, dtype=values.dtype, device=values.device
    )
    states = scanned_delta_states(
        transition, initial, backend="chunkwise", chunk_size=chunk_size
    )
    return delta_read(states, query_keys)


def fast_weight_forward(
    problem: MatchedProblem,
    values: torch.Tensor,
    *,
    keys: torch.Tensor | None = None,
    query_keys: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    keys = problem.keys if keys is None else keys
    query_keys = problem.query_keys if query_keys is None else query_keys
    identity = torch.eye(KEYS, dtype=values.dtype, device=values.device).reshape(
        1, 1, KEYS, KEYS
    )
    transition = DeltaTransition(
        key_action=identity.expand(values.shape[0], values.shape[1], -1, -1),
        value_action=problem.negative_actions,
        drive=keys[..., None] * values[:, :, None, :],
    )
    initial = torch.zeros(
        values.shape[0], KEYS, DIMENSION, dtype=values.dtype, device=values.device
    )
    states = scanned_delta_states(
        transition, initial, backend="chunkwise", chunk_size=chunk_size
    )
    return delta_read(states, query_keys)


def core_forward_functions(
    problem: MatchedProblem,
    *,
    slot_backends: dict[str, str],
    delta_chunk_sizes: dict[str, int],
) -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    return {
        "direct_slot_hybrid": lambda values: direct_forward(
            problem,
            values,
            backend=slot_backends["direct_slot_hybrid"],
        ),
        "triality_slot_hybrid": lambda values: triality_forward(
            problem,
            values,
            backend=slot_backends["triality_slot_hybrid"],
        ),
        "delta_chunkwise": lambda values: delta_forward(
            problem,
            values,
            chunk_size=delta_chunk_sizes["delta_chunkwise"],
        ),
        "fast_weight_chunkwise": lambda values: fast_weight_forward(
            problem,
            values,
            chunk_size=delta_chunk_sizes["fast_weight_chunkwise"],
        ),
    }


def forward_functions(
    problem: MatchedProblem,
) -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    """Compatibility wrapper using the maintained pre-tuning defaults."""

    backend = slot_backend(problem.values.shape[1])
    return core_forward_functions(
        problem,
        slot_backends={
            "direct_slot_hybrid": backend,
            "triality_slot_hybrid": backend,
        },
        delta_chunk_sizes={
            "delta_chunkwise": 64,
            "fast_weight_chunkwise": 64,
        },
    )


def _slot_addresses(
    problem: MatchedProblem,
    write_weight: torch.Tensor,
    query_weight: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    write_logits = problem.write_aliases @ write_weight.transpose(-1, -2)
    query_logits = problem.query_aliases @ query_weight.transpose(-1, -2)
    return (
        F.softmax(write_logits / temperature, dim=-1),
        F.softmax(query_logits / temperature, dim=-1),
    )


def _delta_addresses(
    problem: MatchedProblem,
    write_weight: torch.Tensor,
    query_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        F.normalize(problem.write_aliases @ write_weight.transpose(-1, -2), dim=-1),
        F.normalize(problem.query_aliases @ query_weight.transpose(-1, -2), dim=-1),
    )


def end_to_end_forward_functions(
    problem: MatchedProblem,
    *,
    slot_backends: dict[str, str],
    delta_chunk_sizes: dict[str, int],
    encoder_temperature: float,
) -> dict[
    str,
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
]:
    def direct(
        values: torch.Tensor,
        write_weight: torch.Tensor,
        query_weight: torch.Tensor,
    ) -> torch.Tensor:
        routes, query_routes = _slot_addresses(
            problem,
            write_weight,
            query_weight,
            temperature=encoder_temperature,
        )
        return direct_forward(
            problem,
            values,
            routes=routes,
            query_routes=query_routes,
            backend=slot_backends["direct_slot_hybrid"],
        )

    def triality(
        values: torch.Tensor,
        write_weight: torch.Tensor,
        query_weight: torch.Tensor,
    ) -> torch.Tensor:
        routes, query_routes = _slot_addresses(
            problem,
            write_weight,
            query_weight,
            temperature=encoder_temperature,
        )
        return triality_forward(
            problem,
            values,
            routes=routes,
            query_routes=query_routes,
            backend=slot_backends["triality_slot_hybrid"],
        )

    def delta(
        values: torch.Tensor,
        write_weight: torch.Tensor,
        query_weight: torch.Tensor,
    ) -> torch.Tensor:
        keys, query_keys = _delta_addresses(problem, write_weight, query_weight)
        return delta_forward(
            problem,
            values,
            keys=keys,
            query_keys=query_keys,
            chunk_size=delta_chunk_sizes["delta_chunkwise"],
        )

    def fast_weight(
        values: torch.Tensor,
        write_weight: torch.Tensor,
        query_weight: torch.Tensor,
    ) -> torch.Tensor:
        keys, query_keys = _delta_addresses(problem, write_weight, query_weight)
        return fast_weight_forward(
            problem,
            values,
            keys=keys,
            query_keys=query_keys,
            chunk_size=delta_chunk_sizes["fast_weight_chunkwise"],
        )

    return {
        "direct_slot_hybrid": direct,
        "triality_slot_hybrid": triality,
        "delta_chunkwise": delta,
        "fast_weight_chunkwise": fast_weight,
    }


def _time(
    function: Callable[[], torch.Tensor | None],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> TimingSummary:
    for _ in range(warmup):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            function()
            stop.record()
            stop.synchronize()
            samples.append(float(start.elapsed_time(stop)))
    else:
        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            function()
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
    return _summarize(samples)


def _time_interleaved(
    functions: dict[str, Callable[[], torch.Tensor | None]],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    replications: int,
) -> tuple[
    dict[str, TimingSummary],
    list[list[str]],
    dict[str, list[float]],
]:
    """Time alternatives with cyclic/reversed order balancing.

    A fixed variant order can systematically reward later kernels after GPU
    clocks and allocator caches have warmed.  Every alternative appears at
    every cyclic order position, and odd replications reverse the cycle.
    """

    names = list(functions)
    if not names or repeats < 1 or replications < 1 or warmup < 0:
        raise ValueError("timing requires functions, positive repeats/replications")

    for warmup_index in range(warmup):
        offset = warmup_index % len(names)
        order = names[offset:] + names[:offset]
        for name in order:
            output = functions[name]()
            del output
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    samples = {name: [] for name in names}
    reported_orders: list[list[str]] = []
    for replication in range(replications):
        base = list(reversed(names)) if replication % 2 else names
        reported_orders.append(base)
        for repeat_index in range(repeats):
            offset = (replication + repeat_index) % len(base)
            order = base[offset:] + base[:offset]
            for name in order:
                function = functions[name]
                if device.type == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    stop = torch.cuda.Event(enable_timing=True)
                    start.record()
                    output = function()
                    stop.record()
                    stop.synchronize()
                    samples[name].append(float(start.elapsed_time(stop)))
                else:
                    start_ns = time.perf_counter_ns()
                    output = function()
                    samples[name].append(
                        (time.perf_counter_ns() - start_ns) / 1_000_000
                    )
                del output
    return (
        {name: _summarize(values) for name, values in samples.items()},
        reported_orders,
        samples,
    )


def _clear_tensor_grads(tensors: Sequence[torch.Tensor]) -> None:
    for tensor in tensors:
        tensor.grad = None


def _peak_memory(
    function: Callable[[], torch.Tensor | None],
    *,
    device: torch.device,
    clear_before: Callable[[], None] | None = None,
) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    if clear_before is not None:
        clear_before()
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    output = function()
    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    del output
    if clear_before is not None:
        clear_before()
    return {
        "baseline_allocated_bytes": int(baseline_allocated),
        "baseline_reserved_bytes": int(baseline_reserved),
        "incremental_peak_allocated_bytes": int(
            max(0, peak_allocated - baseline_allocated)
        ),
        "incremental_peak_reserved_bytes": int(
            max(0, peak_reserved - baseline_reserved)
        ),
        "absolute_peak_allocated_bytes": int(peak_allocated),
        "absolute_peak_reserved_bytes": int(peak_reserved),
    }


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual - expected).abs().max()) / max(
        1.0, float(expected.abs().max())
    )


def _metadata(device: torch.device) -> dict[str, object]:
    result: dict[str, object] = {
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_logical_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "fla_installed": importlib.util.find_spec("fla") is not None,
        "triton_installed": importlib.util.find_spec("triton") is not None,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_bytes": properties.total_memory,
                "gpu_compute_capability": list(
                    torch.cuda.get_device_capability(device)
                ),
            }
        )
    return result


def _nvidia_smi_snapshot() -> dict[str, str] | None:
    query = (
        "driver_version,pstate,temperature.gpu,power.draw,utilization.gpu,"
        "memory.used,clocks.current.sm,clocks.current.memory"
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    names = query.split(",")
    if len(values) != len(names):
        return {"raw": completed.stdout.strip()}
    return dict(zip(names, values))


def _tune_implementations(
    problem: MatchedProblem,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, str], dict[str, int], dict[str, object]]:
    slot_backends: dict[str, str] = {}
    delta_chunk_sizes: dict[str, int] = {}
    report: dict[str, object] = {}

    slot_variants: dict[
        str, Callable[[MatchedProblem, torch.Tensor, str], torch.Tensor]
    ] = {
        "direct_slot_hybrid": lambda p, v, backend: direct_forward(
            p, v, backend=backend
        ),
        "triality_slot_hybrid": lambda p, v, backend: triality_forward(
            p, v, backend=backend
        ),
    }
    for variant, implementation in slot_variants.items():
        candidates: dict[str, Callable[[], torch.Tensor | None]] = {}
        for backend in SLOT_BACKENDS:

            def call(
                implementation: Callable[
                    [MatchedProblem, torch.Tensor, str], torch.Tensor
                ] = implementation,
                backend: str = backend,
            ) -> torch.Tensor:
                with torch.no_grad():
                    return implementation(problem, problem.values, backend)

            candidates[backend] = call
        timings, orders, samples = _time_interleaved(
            candidates,
            device=device,
            warmup=warmup,
            repeats=repeats,
            replications=1,
        )
        winner = min(timings, key=lambda name: timings[name].median_ms)
        slot_backends[variant] = winner
        report[variant] = {
            "selected": winner,
            "candidate_ms": {
                name: asdict(summary) for name, summary in timings.items()
            },
            "candidate_samples_ms": samples,
            "timing_orders": orders,
        }

    delta_variants: dict[
        str, Callable[[MatchedProblem, torch.Tensor, int], torch.Tensor]
    ] = {
        "delta_chunkwise": lambda p, v, chunk_size: delta_forward(
            p, v, chunk_size=chunk_size
        ),
        "fast_weight_chunkwise": lambda p, v, chunk_size: fast_weight_forward(
            p, v, chunk_size=chunk_size
        ),
    }
    for variant, implementation in delta_variants.items():
        candidates = {}
        for chunk_size in DELTA_CHUNK_SIZES:

            def call(
                implementation: Callable[
                    [MatchedProblem, torch.Tensor, int], torch.Tensor
                ] = implementation,
                chunk_size: int = chunk_size,
            ) -> torch.Tensor:
                with torch.no_grad():
                    return implementation(problem, problem.values, chunk_size)

            candidates[str(chunk_size)] = call
        timings, orders, samples = _time_interleaved(
            candidates,
            device=device,
            warmup=warmup,
            repeats=repeats,
            replications=1,
        )
        winner = min(timings, key=lambda name: timings[name].median_ms)
        delta_chunk_sizes[variant] = int(winner)
        report[variant] = {
            "selected": int(winner),
            "candidate_ms": {
                name: asdict(summary) for name, summary in timings.items()
            },
            "candidate_samples_ms": samples,
            "timing_orders": orders,
        }
    return slot_backends, delta_chunk_sizes, report


def _tensor_bytes(problem: MatchedProblem) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in vars(problem).values()
        if isinstance(value, torch.Tensor)
    )


def _gradient_diagnostic(
    call: Callable[[], torch.Tensor | None],
    tensors: Sequence[torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, object]:
    call()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gradients = [tensor.grad for tensor in tensors]
    result = {
        "all_present": all(gradient is not None for gradient in gradients),
        "all_finite": all(
            gradient is not None and bool(torch.isfinite(gradient).all())
            for gradient in gradients
        ),
        "maximum_absolute_gradient": max(
            (
                float(gradient.detach().abs().max())
                for gradient in gradients
                if gradient is not None
            ),
            default=0.0,
        ),
        "tensor_count": len(tensors),
    }
    _clear_tensor_grads(tensors)
    return result


def _tier_measurements(
    functions: dict[str, Callable[..., torch.Tensor]],
    *,
    problem: MatchedProblem,
    end_to_end: bool,
    device: torch.device,
    warmup: int,
    repeats: int,
    backward_repeats: int,
    replications: int,
) -> dict[str, object]:
    forward_calls: dict[str, Callable[[], torch.Tensor | None]] = {}
    backward_calls: dict[str, Callable[[], torch.Tensor | None]] = {}
    gradient_tensors: dict[str, tuple[torch.Tensor, ...]] = {}

    for name in VARIANTS:
        function = functions[name]
        if end_to_end:
            static_weight = problem.encoder_initial_weight

            def forward_call(
                function: Callable[..., torch.Tensor] = function,
                static_weight: torch.Tensor = static_weight,
            ) -> torch.Tensor:
                with torch.no_grad():
                    return function(
                        problem.values,
                        static_weight,
                        static_weight,
                    )

            grad_values = problem.values.detach().clone().requires_grad_(True)
            grad_write = (
                problem.encoder_initial_weight.detach().clone().requires_grad_(True)
            )
            grad_query = (
                problem.encoder_initial_weight.detach().clone().requires_grad_(True)
            )
            tensors = (grad_values, grad_write, grad_query)

            def backward_call(
                function: Callable[..., torch.Tensor] = function,
                tensors: tuple[torch.Tensor, ...] = tensors,
            ) -> None:
                _clear_tensor_grads(tensors)
                output = function(*tensors)
                output.square().mean().backward()

        else:

            def forward_call(
                function: Callable[..., torch.Tensor] = function,
            ) -> torch.Tensor:
                with torch.no_grad():
                    return function(problem.values)

            grad_values = problem.values.detach().clone().requires_grad_(True)
            tensors = (grad_values,)

            def backward_call(
                function: Callable[..., torch.Tensor] = function,
                tensors: tuple[torch.Tensor, ...] = tensors,
            ) -> None:
                _clear_tensor_grads(tensors)
                output = function(tensors[0])
                output.square().mean().backward()

        forward_calls[name] = forward_call
        backward_calls[name] = backward_call
        gradient_tensors[name] = tensors

    forward_timings, forward_orders, forward_samples = _time_interleaved(
        forward_calls,
        device=device,
        warmup=warmup,
        repeats=repeats,
        replications=replications,
    )
    backward_timings, backward_orders, backward_samples = _time_interleaved(
        backward_calls,
        device=device,
        warmup=max(1, warmup // 2),
        repeats=backward_repeats,
        replications=replications,
    )

    forward_memory = {
        name: _peak_memory(call, device=device) for name, call in forward_calls.items()
    }
    backward_memory = {
        name: _peak_memory(
            backward_calls[name],
            device=device,
            clear_before=lambda tensors=gradient_tensors[name]: _clear_tensor_grads(
                tensors
            ),
        )
        for name in VARIANTS
    }
    gradient_diagnostics = {
        name: _gradient_diagnostic(
            backward_calls[name], gradient_tensors[name], device=device
        )
        for name in VARIANTS
    }
    tokens = problem.values.shape[0] * problem.values.shape[1]
    return {
        "forward_ms": {
            name: asdict(summary) for name, summary in forward_timings.items()
        },
        "forward_samples_ms": forward_samples,
        "forward_backward_ms": {
            name: asdict(summary) for name, summary in backward_timings.items()
        },
        "forward_backward_samples_ms": backward_samples,
        "forward_tokens_per_second": {
            name: 1000.0 * tokens / summary.median_ms
            for name, summary in forward_timings.items()
        },
        "forward_cuda_memory": forward_memory,
        "forward_backward_cuda_memory": backward_memory,
        "gradient_diagnostics": gradient_diagnostics,
        "timing_orders": {
            "forward_replication_base_orders": forward_orders,
            "forward_backward_replication_base_orders": backward_orders,
            "within_replication": "cyclic rotation each repeat",
        },
    }


def benchmark(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch: int,
    lengths: Iterable[int],
    warmup: int,
    repeats: int,
    backward_repeats: int,
    replications: int = 5,
    tuning_repeats: int = 7,
    encoder_temperature: float = 0.15,
    seed: int = 20260810,
    frozen_selection: dict[str, object] | None = None,
) -> dict[str, object]:
    lengths = tuple(lengths)
    device_metadata = _metadata(device)
    if device.type == "cuda":
        device_metadata["nvidia_smi_before"] = _nvidia_smi_snapshot()
    rows = []
    tolerance = 2e-10 if dtype == torch.float64 else 2e-3
    for length in lengths:
        if frozen_selection is None:
            tuning_problem = make_problem(
                batch=batch,
                length=length,
                dtype=dtype,
                device=device,
                seed=seed + 1_000_000,
            )
            slot_backends, delta_chunk_sizes, tuning = _tune_implementations(
                tuning_problem,
                device=device,
                warmup=max(1, warmup // 2),
                repeats=tuning_repeats,
            )
            del tuning_problem
        else:
            selection = frozen_selection["selections"][str(length)]
            slot_backends = {
                name: str(value) for name, value in selection["slot_backends"].items()
            }
            delta_chunk_sizes = {
                name: int(value)
                for name, value in selection["delta_chunk_sizes"].items()
            }
            tuning = {
                "mode": "externally_frozen",
                "selection_rule": frozen_selection["selection_rule"],
                "selection_diagnostics": selection["diagnostics"],
            }
        problem = make_problem(
            batch=batch,
            length=length,
            dtype=dtype,
            device=device,
            seed=seed,
        )
        core_functions = core_forward_functions(
            problem,
            slot_backends=slot_backends,
            delta_chunk_sizes=delta_chunk_sizes,
        )
        end_to_end_functions = end_to_end_forward_functions(
            problem,
            slot_backends=slot_backends,
            delta_chunk_sizes=delta_chunk_sizes,
            encoder_temperature=encoder_temperature,
        )
        with torch.no_grad():
            core_outputs = {
                name: function(problem.values)
                for name, function in core_functions.items()
            }
            end_to_end_outputs = {
                name: function(
                    problem.values,
                    problem.encoder_initial_weight,
                    problem.encoder_initial_weight,
                )
                for name, function in end_to_end_functions.items()
            }
        core_equivalence = {
            "triality_vs_direct": _relative_error(
                core_outputs["triality_slot_hybrid"],
                core_outputs["direct_slot_hybrid"],
            ),
            "delta_vs_direct": _relative_error(
                core_outputs["delta_chunkwise"], core_outputs["direct_slot_hybrid"]
            ),
        }
        end_to_end_cross_variant_difference = {
            "triality_vs_direct": _relative_error(
                end_to_end_outputs["triality_slot_hybrid"],
                end_to_end_outputs["direct_slot_hybrid"],
            )
        }
        if max(core_equivalence.values()) >= tolerance:
            raise AssertionError(
                "matched implementation correctness failed at "
                f"length {length}: core={core_equivalence}"
            )
        core_measurements = _tier_measurements(
            core_functions,
            problem=problem,
            end_to_end=False,
            device=device,
            warmup=warmup,
            repeats=repeats,
            backward_repeats=backward_repeats,
            replications=replications,
        )
        end_to_end_measurements = _tier_measurements(
            end_to_end_functions,
            problem=problem,
            end_to_end=True,
            device=device,
            warmup=warmup,
            repeats=repeats,
            backward_repeats=backward_repeats,
            replications=replications,
        )
        gradients_passed = all(
            bool(diagnostic[gate])
            for tier in (core_measurements, end_to_end_measurements)
            for diagnostic in tier["gradient_diagnostics"].values()
            for gate in ("all_present", "all_finite")
        )
        rows.append(
            {
                "length": length,
                "batch": batch,
                "tuning": tuning,
                "selected_implementations": {
                    "slot_backends": slot_backends,
                    "delta_chunk_sizes": delta_chunk_sizes,
                },
                "static_problem_tensor_bytes": _tensor_bytes(problem),
                "core": {
                    "equivalence_relative_error": core_equivalence,
                    **core_measurements,
                },
                "end_to_end": {
                    "cross_variant_relative_difference_not_a_correctness_gate": (
                        end_to_end_cross_variant_difference
                    ),
                    **end_to_end_measurements,
                },
                "correctness_passed": gradients_passed,
            }
        )
    if device.type == "cuda":
        device_metadata["nvidia_smi_after"] = _nvidia_smi_snapshot()
    return {
        "experiment": "blocked state-matched local memory implementation benchmark",
        "device_metadata": device_metadata,
        "protocol": {
            "dtype": str(dtype).removeprefix("torch."),
            "batch": batch,
            "lengths": list(lengths),
            "warmup": warmup,
            "forward_repeats": repeats,
            "forward_backward_repeats": backward_repeats,
            "same_process_timing_blocks": replications,
            "independent_process_runs": 1,
            "samples_per_forward_variant_and_length": repeats * replications,
            "samples_per_backward_variant_and_length": backward_repeats * replications,
            "selection_mode": (
                "in_process_disjoint_tuning"
                if frozen_selection is None
                else "externally_frozen_disjoint_tuning"
            ),
            "tuning_repeats_in_this_process": (
                tuning_repeats if frozen_selection is None else 0
            ),
            "problem_seed": seed,
            "tuning_problem_seed": seed + 1_000_000,
            "frozen_selection_inputs": (
                None if frozen_selection is None else frozen_selection["inputs"]
            ),
            "state_scalars": {name: 64 for name in VARIANTS},
            "input_generation_timed": False,
            "core_address_encoding_timed": False,
            "end_to_end_address_encoding_timed": True,
            "end_to_end_encoder_parameters": {name: 384 for name in VARIANTS},
            "alias_dimension": ALIAS_DIMENSION,
            "encoder_temperature": encoder_temperature,
            "transition_construction_timed": True,
            "scan_and_read_timed": True,
            "triality_bind_unbind_timed": True,
            "core_backward_targets": "stored values",
            "end_to_end_backward_targets": (
                "stored values plus 192-parameter write encoder plus "
                "192-parameter query encoder"
            ),
            "timing_order": "cyclically rotated; odd replications reversed",
            "tf32_disabled": not torch.backends.cuda.matmul.allow_tf32,
        },
        "rows": rows,
        "claim_boundary": {
            "eager_pytorch_only": True,
            "fused_delta_kernel_compared": False,
            "compact_wy_compared": False,
            "quality_campaign_separate": True,
            "absolute_architecture_winner_established": False,
        },
        "passed": all(bool(row["correctness_passed"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=(64, 256, 1024, 2048, 4096)
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--backward-repeats", type=int, default=5)
    parser.add_argument("--replications", type=int, default=5)
    parser.add_argument("--tuning-repeats", type=int, default=7)
    parser.add_argument("--encoder-temperature", type=float, default=0.15)
    parser.add_argument("--selection-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.backends.cuda.matmul.allow_tf32 = False
    frozen_selection = (
        None
        if args.selection_config is None
        else json.loads(args.selection_config.read_text(encoding="utf-8"))
    )
    report = benchmark(
        device=torch.device(args.device),
        dtype=dtype,
        batch=args.batch,
        lengths=tuple(args.lengths),
        warmup=args.warmup,
        repeats=args.repeats,
        backward_repeats=args.backward_repeats,
        replications=args.replications,
        tuning_repeats=args.tuning_repeats,
        encoder_temperature=args.encoder_temperature,
        frozen_selection=frozen_selection,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
