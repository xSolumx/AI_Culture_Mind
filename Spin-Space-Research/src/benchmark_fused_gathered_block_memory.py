"""Benchmark a one-kernel Triton gathered-block recurrent memory step."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import torch

from benchmark_gathered_block_memory import (
    ALIAS_DIMENSION,
    SLOTS_PER_BLOCK,
    UPDATE_LAWS,
    VALUE_DIMENSION,
    GatherProblem,
    cyclic_orders,
    make_problem,
    memory_step,
)

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - optional environment boundary
    TRITON_AVAILABLE = False

    class _MissingTriton:
        @staticmethod
        def jit(function):
            return function

    class _MissingTritonLanguage:
        constexpr = int

    triton = _MissingTriton()
    tl = _MissingTritonLanguage()

IMPLEMENTATIONS = ("eager_dense", "eager_gathered", "triton_fused_gathered")
VARIANTS = tuple(
    f"{law}_{implementation}"
    for law in UPDATE_LAWS
    for implementation in IMPLEMENTATIONS
)


@triton.jit
def _fused_gathered_step(
    state_ptr,
    write_alias_ptr,
    query_alias_ptr,
    value_ptr,
    coarse_weight_ptr,
    fine_weight_ptr,
    output_ptr,
    BLOCKS: tl.constexpr,
    BLOCK_PAD: tl.constexpr,
    LAW: tl.constexpr,
):
    pid = tl.program_id(0)
    block_offsets = tl.arange(0, BLOCK_PAD)
    block_mask = block_offsets < BLOCKS
    write_block_scores = tl.zeros([BLOCK_PAD], dtype=tl.float32)
    query_block_scores = tl.zeros([BLOCK_PAD], dtype=tl.float32)
    for dimension in tl.static_range(0, 8):
        coarse = tl.load(
            coarse_weight_ptr + block_offsets * 8 + dimension,
            mask=block_mask,
            other=0.0,
        )
        write_alias = tl.load(
            write_alias_ptr + pid * 8 + dimension
        )
        query_alias = tl.load(
            query_alias_ptr + pid * 8 + dimension
        )
        write_block_scores += coarse * write_alias
        query_block_scores += coarse * query_alias
    negative_infinity = float("-inf")
    write_block_scores = tl.where(
        block_mask, write_block_scores, negative_infinity
    )
    query_block_scores = tl.where(
        block_mask, query_block_scores, negative_infinity
    )
    write_block = tl.argmax(write_block_scores, axis=0)
    query_block = tl.argmax(query_block_scores, axis=0)

    slot_offsets = tl.arange(0, 8)
    write_fine_scores = tl.zeros([8], dtype=tl.float32)
    query_fine_scores = tl.zeros([8], dtype=tl.float32)
    for dimension in tl.static_range(0, 8):
        write_weight = tl.load(
            fine_weight_ptr
            + ((write_block * 8 + slot_offsets) * 8)
            + dimension
        )
        query_weight = tl.load(
            fine_weight_ptr
            + ((query_block * 8 + slot_offsets) * 8)
            + dimension
        )
        write_alias = tl.load(
            write_alias_ptr + pid * 8 + dimension
        )
        query_alias = tl.load(
            query_alias_ptr + pid * 8 + dimension
        )
        write_fine_scores += write_weight * write_alias / 0.35
        query_fine_scores += query_weight * query_alias / 0.35
    write_fine_scores -= tl.max(write_fine_scores, axis=0)
    query_fine_scores -= tl.max(query_fine_scores, axis=0)
    write_route = tl.exp(write_fine_scores)
    query_route = tl.exp(query_fine_scores)
    write_route /= tl.sum(write_route, axis=0)
    query_route /= tl.sum(query_route, axis=0)

    value_offsets = tl.arange(0, 8)
    state_write_offsets = (
        (
            (pid * BLOCKS + write_block) * 8
            + slot_offsets[:, None]
        )
        * 8
        + value_offsets[None, :]
    )
    selected = tl.load(state_ptr + state_write_offsets)
    value = tl.load(
        value_ptr + pid * 8 + value_offsets
    )[None, :]
    if LAW == 0:
        updated = (
            (1.0 - write_route[:, None]) * selected
            + write_route[:, None] * value
        )
    else:
        write_key = write_route / tl.sqrt(tl.sum(write_route * write_route, axis=0))
        old_prediction = tl.sum(write_key[:, None] * selected, axis=0)
        updated = selected + write_key[:, None] * (value - old_prediction[None, :])
    tl.store(state_ptr + state_write_offsets, updated)

    state_query_offsets = (
        (
            (pid * BLOCKS + query_block) * 8
            + slot_offsets[:, None]
        )
        * 8
        + value_offsets[None, :]
    )
    query_state = tl.load(state_ptr + state_query_offsets)
    query_state = tl.where(write_block == query_block, updated, query_state)
    if LAW == 0:
        prediction = tl.sum(query_route[:, None] * query_state, axis=0)
    else:
        query_key = query_route / tl.sqrt(tl.sum(query_route * query_route, axis=0))
        prediction = tl.sum(query_key[:, None] * query_state, axis=0)
    tl.store(output_ptr + pid * 8 + value_offsets, prediction)


@torch.no_grad()
def triton_step(
    state: torch.Tensor, problem: GatherProblem, *, law: str
) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "fused gathered benchmark requires the optional triton-windows package"
        )
    if not state.is_cuda or state.dtype != torch.float32:
        raise ValueError("Triton gathered step requires CUDA float32 tensors")
    if law not in UPDATE_LAWS:
        raise ValueError(f"unknown update law: {law}")
    output = torch.empty(
        problem.batch, VALUE_DIMENSION, device=state.device, dtype=state.dtype
    )
    block_pad = triton.next_power_of_2(problem.blocks)
    _fused_gathered_step[(problem.batch,)](
        state,
        problem.write_alias,
        problem.query_alias,
        problem.value,
        problem.coarse_weight,
        problem.fine_weight,
        output,
        BLOCKS=problem.blocks,
        BLOCK_PAD=block_pad,
        LAW=0 if law == "direct" else 1,
        num_warps=4,
    )
    return output


@torch.no_grad()
def correctness(problem: GatherProblem) -> dict[str, object]:
    rows = {}
    passed = True
    for law in UPDATE_LAWS:
        eager_state = problem.state.clone()
        triton_state = problem.state.clone()
        eager_prediction = memory_step(
            eager_state,
            problem,
            law=law,
            implementation="block_gathered",
        )
        triton_prediction = triton_step(triton_state, problem, law=law)
        torch.cuda.synchronize(problem.state.device)
        state_error = float((eager_state - triton_state).abs().max())
        prediction_error = float((eager_prediction - triton_prediction).abs().max())
        finite = bool(
            torch.isfinite(triton_state).all()
            and torch.isfinite(triton_prediction).all()
        )
        rows[law] = {
            "maximum_state_error": state_error,
            "maximum_prediction_error": prediction_error,
            "all_finite": finite,
        }
        passed = passed and state_error <= 1e-5 and prediction_error <= 1e-5 and finite
    return {"passed": passed, "rows": rows}


@torch.no_grad()
def recurrent_correctness(
    problem: GatherProblem, *, steps: int = 257
) -> dict[str, object]:
    """Compare fused and eager state trajectories, not only one isolated step."""
    if steps < 1:
        raise ValueError("steps must be positive")
    rows = {}
    passed = True
    for law in UPDATE_LAWS:
        eager_state = problem.state.clone()
        triton_state = problem.state.clone()
        eager_prediction = None
        triton_prediction = None
        for _ in range(steps):
            eager_prediction = memory_step(
                eager_state,
                problem,
                law=law,
                implementation="block_gathered",
            )
            triton_prediction = triton_step(triton_state, problem, law=law)
        torch.cuda.synchronize(problem.state.device)
        state_error = float((eager_state - triton_state).abs().max())
        prediction_error = float(
            (eager_prediction - triton_prediction).abs().max()
        )
        finite = bool(
            torch.isfinite(triton_state).all()
            and torch.isfinite(triton_prediction).all()
        )
        rows[law] = {
            "steps": steps,
            "maximum_state_error": state_error,
            "maximum_prediction_error": prediction_error,
            "all_finite": finite,
        }
        passed = passed and state_error <= 1e-5 and prediction_error <= 1e-5 and finite
    return {"passed": passed, "rows": rows}


def _call_variant(
    state: torch.Tensor,
    problem: GatherProblem,
    *,
    law: str,
    implementation: str,
) -> torch.Tensor:
    if implementation == "triton_fused_gathered":
        return triton_step(state, problem, law=law)
    eager_implementation = (
        "dense_full" if implementation == "eager_dense" else "block_gathered"
    )
    return memory_step(
        state, problem, law=law, implementation=eager_implementation
    )


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
            _call_variant(
                states[variant],
                problem,
                law=law,
                implementation=implementation,
            )
    torch.cuda.synchronize(problem.state.device)
    samples = {variant: [] for variant in variants}
    orders = cyclic_orders(
        variants, rotation=rotation, timing_blocks=timing_blocks
    )
    for order in orders:
        for variant in order:
            law, implementation = variant.split("_", maxsplit=1)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(inner_calls):
                _call_variant(
                    states[variant],
                    problem,
                    law=law,
                    implementation=implementation,
                )
            end.record()
            end.synchronize()
            samples[variant].append(
                float(start.elapsed_time(end)) / inner_calls
            )
    return samples, orders


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


def _incremental_memory(
    problem: GatherProblem, *, law: str, implementation: str
) -> int:
    state = problem.state.clone()
    torch.cuda.synchronize(state.device)
    torch.cuda.reset_peak_memory_stats(state.device)
    baseline = torch.cuda.memory_allocated(state.device)
    _call_variant(state, problem, law=law, implementation=implementation)
    torch.cuda.synchronize(state.device)
    return int(torch.cuda.max_memory_allocated(state.device) - baseline)


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
    slots: tuple[int, ...],
    batches: tuple[int, ...],
    warmup: int,
    timing_blocks: int,
    inner_calls: int,
    seed: int,
) -> dict[str, object]:
    device = torch.device("cuda")
    rows = []
    passed = True
    for slot_count in slots:
        for batch in batches:
            problem = make_problem(
                batch=batch,
                slots=slot_count,
                device=device,
                dtype=torch.float32,
                seed=seed + slot_count + 10_000 * batch,
            )
            parity = correctness(problem)
            passed = passed and bool(parity["passed"])
            timing = {}
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
                    "batch": batch,
                    "logical_state_scalars": batch
                    * slot_count
                    * VALUE_DIMENSION,
                    "correctness": parity,
                    "timing_order": timing_orders[0],
                    "timing_block_orders": timing_orders,
                    "timing": timing,
                }
            )
    return {
        "experiment": "fused gathered-block recurrent memory benchmark",
        "protocol": "FUSED_GATHERED_BLOCK_MEMORY_PREREGISTRATION.md",
        "passed": passed,
        "device": "cuda",
        "dtype": "torch.float32",
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton_windows": importlib.metadata.version("triton-windows"),
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
        "rows": rows,
        "claim_boundary": {
            "inference_only": True,
            "training_kernel": False,
            "model_level_quality": False,
            "hardware_general": False,
            "triality_specific_capacity": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", nargs="+", type=int, default=(64, 256, 1024, 4096))
    parser.add_argument("--batches", nargs="+", type=int, default=(1, 16, 64))
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--timing-blocks", type=int, default=25)
    parser.add_argument("--inner-calls", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1_090_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    report = benchmark(
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
