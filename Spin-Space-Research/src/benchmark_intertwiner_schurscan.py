"""Reproducible correctness and performance benchmark for Intertwiner SchurScan.

This harness distinguishes four questions that are easy to conflate:

* recurrent correctness;
* scan-tree arithmetic work;
* eager PyTorch wall-clock performance;
* proof-lift overhead.

It deliberately does not call an eager Python scan a production kernel.  A
future fused CUDA/Triton implementation should be added as a separate backend
and evaluated under the same contract.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from intertwiner_schurscan import (
    lift_layout,
    lifted_intertwiner_scan,
    recurrent_intertwiner_scan,
    scan_composition_counts,
    scan_dependency_depths,
    staged_intertwiner_scan,
    work_efficient_affine_prefixes,
)
from spin8_triality_lift import triality_tensor

TensorTuple = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
Problem = tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class TimingSummary:
    median_ms: float
    minimum_ms: float
    p20_ms: float
    p80_ms: float
    mean_ms: float
    stdev_ms: float
    repeats: int


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty sequence")
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summarize(samples_ms: list[float]) -> TimingSummary:
    return TimingSummary(
        median_ms=statistics.median(samples_ms),
        minimum_ms=min(samples_ms),
        p20_ms=_percentile(samples_ms, 0.2),
        p80_ms=_percentile(samples_ms, 0.8),
        mean_ms=statistics.fmean(samples_ms),
        stdev_ms=statistics.pstdev(samples_ms),
        repeats=len(samples_ms),
    )


def _consume(output: TensorTuple | torch.Tensor) -> torch.Tensor:
    if isinstance(output, tuple):
        return sum(component[:, -1].sum() for component in output)
    return output[:, -1].sum()


def _time_callable(
    function: Callable[[], TensorTuple | torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> TimingSummary:
    for _ in range(warmup):
        _consume(function())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            _consume(function())
            stop.record()
            stop.synchronize()
            samples.append(float(start.elapsed_time(stop)))
    else:
        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            _consume(function())
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
    return _summarize(samples)


def _random_problem(
    *,
    batch: int,
    length: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> Problem:
    # Construct canonically on CPU so the same seed defines the same mathematical
    # problem on CPU and CUDA.  Input construction is outside all timed regions.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dimension = 8

    def actions() -> torch.Tensor:
        raw = torch.randn(
            batch,
            length,
            dimension,
            dimension,
            dtype=dtype,
            generator=generator,
        )
        skew = raw - raw.transpose(-1, -2)
        return (0.99 * torch.matrix_exp(0.018 * skew)).to(device)

    def drive(scale: float) -> torch.Tensor:
        return (
            scale
            * torch.randn(batch, length, dimension, dtype=dtype, generator=generator)
        ).to(device)

    def initial() -> torch.Tensor:
        return torch.randn(batch, dimension, dtype=dtype, generator=generator).to(
            device
        )

    return (
        actions(),
        drive(0.002),
        actions(),
        drive(0.002),
        actions(),
        drive(0.001),
        initial(),
        initial(),
        initial(),
        triality_tensor(dtype=dtype, device=device),
    )


def _direct_three_stream_scan(problem: Problem) -> TensorTuple:
    outputs = []
    for action, drive, initial in (
        (problem[0], problem[1], problem[6]),
        (problem[2], problem[3], problem[7]),
        (problem[4], problem[5], problem[8]),
    ):
        prefix_action, prefix_drive = work_efficient_affine_prefixes(action, drive)
        outputs.append(
            torch.einsum("blij,bj->bli", prefix_action, initial) + prefix_drive
        )
    return outputs[0], outputs[1], outputs[2]


def _lifted_components(problem: Problem) -> TensorTuple:
    lifted = lifted_intertwiner_scan(*problem, scan_backend="work_efficient")
    layout = lift_layout(8, 8, 8)
    return (
        lifted[..., layout.u],
        lifted[..., layout.v],
        lifted[..., layout.w],
    )


def _errors(candidate: TensorTuple, reference: TensorTuple) -> dict[str, float]:
    absolute = max(
        float((left - right).abs().max().detach().cpu())
        for left, right in zip(candidate, reference)
    )
    scale = max(float(right.abs().max().detach().cpu()) for right in reference)
    return {
        "max_abs": absolute,
        "max_abs_over_reference_max": absolute
        / max(scale, torch.finfo(reference[0].dtype).tiny),
    }


def _peak_memory_bytes(
    function: Callable[[], TensorTuple | torch.Tensor], device: torch.device
) -> int | None:
    if device.type != "cuda":
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    output = function()
    _consume(output)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    del output
    return int(max(0, peak - baseline))


def _benchmark_forward_variant(
    *,
    name: str,
    function: Callable[[], TensorTuple | torch.Tensor],
    reference: TensorTuple,
    device: torch.device,
    batch: int,
    length: int,
    warmup: int,
    repeats: int,
    comparable_output: bool,
) -> dict[str, object]:
    with torch.inference_mode():
        candidate = function()
        if isinstance(candidate, tuple) and comparable_output:
            error = _errors(candidate, reference)
        else:
            error = None
        peak_memory = _peak_memory_bytes(function, device)
        timing = _time_callable(function, device=device, warmup=warmup, repeats=repeats)
    tokens = batch * length
    return {
        "variant": name,
        "timing": asdict(timing),
        "tokens_per_second_at_median": tokens / (timing.median_ms / 1_000),
        "peak_incremental_cuda_bytes": peak_memory,
        "error_vs_recurrent_same_dtype": error,
        "comparable_semantics": comparable_output,
    }


def _gradient_problem(problem: Problem) -> Problem:
    return tuple(value.detach().clone().requires_grad_() for value in problem)


def _time_forward_backward(
    function: Callable[..., TensorTuple],
    problem: Problem,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> TimingSummary:
    differentiable = _gradient_problem(problem)

    def step() -> torch.Tensor:
        for value in differentiable:
            value.grad = None
        output = function(*differentiable)
        loss = sum(component.square().mean() for component in output)
        loss.backward()
        return loss

    for _ in range(warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            step()
            stop.record()
            stop.synchronize()
            samples.append(float(start.elapsed_time(stop)))
    else:
        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            step()
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
    return _summarize(samples)


def _peak_forward_backward_memory_bytes(
    function: Callable[..., TensorTuple],
    problem: Problem,
    *,
    device: torch.device,
) -> int | None:
    if device.type != "cuda":
        return None
    differentiable = _gradient_problem(problem)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    output = function(*differentiable)
    loss = sum(component.square().mean() for component in output)
    loss.backward()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    del output, loss, differentiable
    return int(max(0, peak - baseline))


def _device_metadata(device: torch.device) -> dict[str, object]:
    try:
        triton_version: str | None = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        try:
            triton_version = importlib.metadata.version("triton-windows")
        except importlib.metadata.PackageNotFoundError:
            triton_version = None
    metadata: dict[str, object] = {
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_logical_count": os.cpu_count(),
        "torch_cpu_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "triton_python_package": triton_version,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        metadata.update(
            {
                "gpu_name": properties.name,
                "gpu_compute_capability": list(
                    torch.cuda.get_device_capability(device)
                ),
                "gpu_total_memory_bytes": properties.total_memory,
            }
        )
    return metadata


def run_benchmark(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch: int,
    lengths: Iterable[int],
    backward_max_length: int,
    lift_max_length: int,
    warmup: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    lengths = tuple(lengths)
    rows = []
    backward_rows = []
    for length in lengths:
        problem = _random_problem(
            batch=batch,
            length=length,
            dtype=dtype,
            device=device,
            seed=seed + length,
        )
        with torch.inference_mode():
            reference = recurrent_intertwiner_scan(*problem)

        variants: list[tuple[str, Callable[[], TensorTuple | torch.Tensor], bool]] = [
            (
                "recurrent_eager",
                lambda problem=problem: recurrent_intertwiner_scan(*problem),
                True,
            ),
            (
                "staged_hillis_steele_homogeneous",
                lambda problem=problem: staged_intertwiner_scan(
                    *problem, scan_backend="hillis_steele"
                ),
                True,
            ),
            (
                "staged_work_efficient_affine",
                lambda problem=problem: staged_intertwiner_scan(
                    *problem, scan_backend="work_efficient_affine"
                ),
                True,
            ),
            (
                "staged_work_efficient_homogeneous",
                lambda problem=problem: staged_intertwiner_scan(
                    *problem, scan_backend="work_efficient"
                ),
                True,
            ),
            (
                "direct_three_stream_work_efficient_control",
                lambda problem=problem: _direct_three_stream_scan(problem),
                False,
            ),
        ]
        if length <= lift_max_length:
            variants.append(
                (
                    "single_stage_89d_proof_lift",
                    lambda problem=problem: _lifted_components(problem),
                    True,
                )
            )
        for name, function, comparable in variants:
            row = _benchmark_forward_variant(
                name=name,
                function=function,
                reference=reference,
                device=device,
                batch=batch,
                length=length,
                warmup=warmup,
                repeats=repeats,
                comparable_output=comparable,
            )
            row.update(
                {
                    "length": length,
                    "batch": batch,
                    "dtype": str(dtype).removeprefix("torch."),
                    "scan_compositions": scan_composition_counts(length),
                    "scan_dependency_depth": scan_dependency_depths(length),
                }
            )
            rows.append(row)

        if length <= backward_max_length:
            for name, function in (
                (
                    "recurrent_eager",
                    recurrent_intertwiner_scan,
                ),
                (
                    "staged_hillis_steele_homogeneous",
                    lambda *values: staged_intertwiner_scan(
                        *values, scan_backend="hillis_steele"
                    ),
                ),
                (
                    "staged_work_efficient_affine",
                    lambda *values: staged_intertwiner_scan(
                        *values, scan_backend="work_efficient_affine"
                    ),
                ),
                (
                    "staged_work_efficient_homogeneous",
                    lambda *values: staged_intertwiner_scan(
                        *values, scan_backend="work_efficient"
                    ),
                ),
            ):
                timing = _time_forward_backward(
                    function,
                    problem,
                    device=device,
                    warmup=max(1, warmup // 2),
                    repeats=max(3, repeats // 2),
                )
                peak_memory = _peak_forward_backward_memory_bytes(
                    function,
                    problem,
                    device=device,
                )
                backward_rows.append(
                    {
                        "variant": name,
                        "length": length,
                        "batch": batch,
                        "dtype": str(dtype).removeprefix("torch."),
                        "timing": asdict(timing),
                        "tokens_per_second_at_median": batch
                        * length
                        / (timing.median_ms / 1_000),
                        "peak_incremental_cuda_bytes": peak_memory,
                        "gradient_scope": "actions, drives, initial states, and beta",
                    }
                )
    layout = lift_layout(8, 8, 8)
    return {
        "experiment": "Intertwiner SchurScan eager PyTorch benchmark",
        "seed": seed,
        "environment": _device_metadata(device),
        "configuration": {
            "dtype": str(dtype).removeprefix("torch."),
            "batch": batch,
            "lengths": list(lengths),
            "warmup": warmup,
            "repeats": repeats,
            "backward_max_length": backward_max_length,
            "lift_max_length": lift_max_length,
            "cpu_thread_cap": torch.get_num_threads(),
        },
        "state_dimensions": {
            "streaming": 24,
            "staged_affine_monoid_per_stream": 72,
            "homogeneous_full_proof_lift": layout.dimension,
            "triality_tensor_coefficients": 512,
        },
        "interpretation_contract": {
            "wall_clock_scope": "eager PyTorch tensor program; no fused scan kernel",
            "recurrent_baseline": "Python-loop eager recurrence, not a fused recurrent kernel",
            "direct_control": "same state width but omits the bilinear triality drive",
            "proof_lift": "same recurrence semantics, but output is the 89D lifted state",
            "timed_input_generation": False,
            "algebraic_equivalence_implies_bitwise_float_equality": False,
        },
        "forward": rows,
        "forward_backward": backward_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[16, 64, 256, 1024, 2048]
    )
    parser.add_argument("--backward-max-length", type=int, default=256)
    parser.add_argument("--lift-max-length", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()

    if args.batch < 1 or min(args.lengths) < 1:
        parser.error("batch and every length must be positive")
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(min(args.threads, 6))
    torch.set_num_interop_threads(1)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    report = run_benchmark(
        device=device,
        dtype=dtype,
        batch=args.batch,
        lengths=args.lengths,
        backward_max_length=args.backward_max_length,
        lift_max_length=args.lift_max_length,
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
