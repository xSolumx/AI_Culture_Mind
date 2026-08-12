"""Benchmark structured and dense eager prefix scans for slot memory."""

from __future__ import annotations

import argparse
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
    homogeneous_affine_matrix,
    scan_composition_counts,
    scan_dependency_depths,
    work_efficient_affine_prefixes,
    work_efficient_associative_matrix_scan,
)
from spin8_triality_memory import (
    SlotTransition,
    apply_slot,
    associative_slot_scan,
    pack_slot_homogeneous_matrices,
    packed_homogeneous_slot_scan,
    packed_homogeneous_slot_states,
    work_efficient_slot_scan,
)

SLOTS = 8
DIMENSION = 8
STATE_WIDTH = SLOTS * DIMENSION


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
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize(samples: list[float]) -> TimingSummary:
    return TimingSummary(
        median_ms=statistics.median(samples),
        minimum_ms=min(samples),
        p20_ms=_percentile(samples, 0.2),
        p80_ms=_percentile(samples, 0.8),
        mean_ms=statistics.fmean(samples),
        stdev_ms=statistics.pstdev(samples),
        repeats=len(samples),
    )


def random_slot_problem(
    *,
    batch: int,
    length: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[SlotTransition, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(
        batch, length, DIMENSION, DIMENSION, dtype=torch.float64, generator=generator
    )
    skew = raw - raw.transpose(-1, -2)
    action = 0.995 * torch.matrix_exp(0.018 * skew)
    retention = 0.90 + 0.099 * torch.rand(
        batch, length, SLOTS, dtype=torch.float64, generator=generator
    )
    drive = 0.01 * torch.randn(
        batch,
        length,
        SLOTS,
        DIMENSION,
        dtype=torch.float64,
        generator=generator,
    )
    initial = torch.randn(
        batch, SLOTS, DIMENSION, dtype=torch.float64, generator=generator
    )
    transition = SlotTransition(
        retention.to(dtype=dtype, device=device),
        action.to(dtype=dtype, device=device),
        drive.to(dtype=dtype, device=device),
    )
    return transition, initial.to(dtype=dtype, device=device)


def materialize_dense_transition(
    transition: SlotTransition,
) -> tuple[torch.Tensor, torch.Tensor]:
    slot_identity = torch.eye(
        SLOTS,
        dtype=transition.action.dtype,
        device=transition.action.device,
    )
    dense_action = torch.einsum(
        "blh,blij,hk->blhikj",
        transition.retention,
        transition.action,
        slot_identity,
    ).reshape(
        transition.retention.shape[0],
        transition.retention.shape[1],
        STATE_WIDTH,
        STATE_WIDTH,
    )
    return dense_action, transition.drive.flatten(-2)


def recurrent_states(transition: SlotTransition, initial: torch.Tensor) -> torch.Tensor:
    state = initial
    outputs = []
    for position in range(transition.retention.shape[1]):
        state = apply_slot(
            SlotTransition(
                transition.retention[:, position],
                transition.action[:, position],
                transition.drive[:, position],
            ),
            state,
        )
        outputs.append(state)
    return torch.stack(outputs, dim=1)


def slot_scan_states(
    transition: SlotTransition,
    initial: torch.Tensor,
    *,
    work_efficient: bool,
) -> torch.Tensor:
    scan = work_efficient_slot_scan if work_efficient else associative_slot_scan
    return apply_slot(scan(transition), initial[:, None])


def dense_affine_states(
    dense_action: torch.Tensor,
    dense_drive: torch.Tensor,
    initial: torch.Tensor,
) -> torch.Tensor:
    prefix_action, prefix_drive = work_efficient_affine_prefixes(
        dense_action, dense_drive
    )
    flat = torch.einsum("blij,bj->bli", prefix_action, initial.flatten(1))
    return (flat + prefix_drive).reshape(
        initial.shape[0], dense_drive.shape[1], SLOTS, DIMENSION
    )


def dense_homogeneous_states(
    homogeneous: torch.Tensor, initial: torch.Tensor
) -> torch.Tensor:
    prefixes = work_efficient_associative_matrix_scan(homogeneous)
    one = torch.ones(initial.shape[0], 1, dtype=initial.dtype, device=initial.device)
    initial_h = torch.cat((one, initial.flatten(1)), dim=-1)
    flat = torch.einsum("blij,bj->bli", prefixes, initial_h)[..., 1:]
    return flat.reshape(initial.shape[0], homogeneous.shape[1], SLOTS, DIMENSION)


def dense_affine_end_to_end(
    transition: SlotTransition, initial: torch.Tensor
) -> torch.Tensor:
    action, drive = materialize_dense_transition(transition)
    return dense_affine_states(action, drive, initial)


def dense_homogeneous_end_to_end(
    transition: SlotTransition, initial: torch.Tensor
) -> torch.Tensor:
    action, drive = materialize_dense_transition(transition)
    homogeneous = homogeneous_affine_matrix(action, drive)
    return dense_homogeneous_states(homogeneous, initial)


def _consume(output: torch.Tensor) -> torch.Tensor:
    return output[:, -1].sum()


def _time_callable(
    function: Callable[[], torch.Tensor],
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


def _peak_incremental_memory(
    function: Callable[[], torch.Tensor], *, device: torch.device
) -> int | None:
    if device.type != "cuda":
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    output = function()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    del output
    return int(max(0, peak - baseline))


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = float(expected.abs().max())
    return float((actual - expected).abs().max()) / max(scale, 1.0)


def _device_metadata(device: torch.device) -> dict[str, object]:
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


@torch.inference_mode()
def run_benchmark(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch: int,
    lengths: Iterable[int],
    warmup: int,
    repeats: int,
    seed: int = 20260810,
) -> dict[str, object]:
    lengths = tuple(lengths)
    methods = (
        "slot_hillis_scan_only",
        "slot_work_efficient_scan_only",
        "slot_local_homogeneous_hillis_scan_only",
        "slot_local_homogeneous_work_efficient_scan_only",
        "dense_affine_work_efficient_scan_only",
        "dense_homogeneous_work_efficient_scan_only",
        "slot_local_homogeneous_hillis_end_to_end",
        "slot_local_homogeneous_work_efficient_end_to_end",
        "dense_affine_work_efficient_end_to_end",
        "dense_homogeneous_work_efficient_end_to_end",
    )
    rows = []
    tolerance = 2e-10 if dtype == torch.float64 else 5e-5
    for length_index, length in enumerate(lengths):
        transition, initial = random_slot_problem(
            batch=batch,
            length=length,
            dtype=dtype,
            device=device,
            seed=seed + length,
        )
        dense_action, dense_drive = materialize_dense_transition(transition)
        homogeneous = homogeneous_affine_matrix(dense_action, dense_drive)
        local_homogeneous = pack_slot_homogeneous_matrices(transition)
        reference = recurrent_states(transition, initial)
        callables: dict[str, Callable[[], torch.Tensor]] = {
            "slot_hillis_scan_only": lambda transition=transition, initial=initial: slot_scan_states(
                transition, initial, work_efficient=False
            ),
            "slot_work_efficient_scan_only": lambda transition=transition, initial=initial: slot_scan_states(
                transition, initial, work_efficient=True
            ),
            "slot_local_homogeneous_hillis_scan_only": lambda local_homogeneous=local_homogeneous, initial=initial: packed_homogeneous_slot_states(
                local_homogeneous, initial, backend="hillis_steele"
            ),
            "slot_local_homogeneous_work_efficient_scan_only": lambda local_homogeneous=local_homogeneous, initial=initial: packed_homogeneous_slot_states(
                local_homogeneous, initial, backend="work_efficient"
            ),
            "dense_affine_work_efficient_scan_only": lambda dense_action=dense_action, dense_drive=dense_drive, initial=initial: dense_affine_states(
                dense_action, dense_drive, initial
            ),
            "dense_homogeneous_work_efficient_scan_only": lambda homogeneous=homogeneous, initial=initial: dense_homogeneous_states(
                homogeneous, initial
            ),
            "slot_local_homogeneous_hillis_end_to_end": lambda transition=transition, initial=initial: packed_homogeneous_slot_scan(
                transition, initial, backend="hillis_steele"
            ),
            "slot_local_homogeneous_work_efficient_end_to_end": lambda transition=transition, initial=initial: packed_homogeneous_slot_scan(
                transition, initial, backend="work_efficient"
            ),
            "dense_affine_work_efficient_end_to_end": lambda transition=transition, initial=initial: dense_affine_end_to_end(
                transition, initial
            ),
            "dense_homogeneous_work_efficient_end_to_end": lambda transition=transition, initial=initial: dense_homogeneous_end_to_end(
                transition, initial
            ),
        }
        errors = {
            name: _relative_error(function(), reference)
            for name, function in callables.items()
        }
        passed = all(value < tolerance for value in errors.values())
        if not passed:
            raise AssertionError(
                f"correctness gate failed at length {length}: {errors}"
            )

        order = (
            methods[length_index % len(methods) :]
            + methods[: length_index % len(methods)]
        )
        timings: dict[str, object] = {}
        memory: dict[str, int | None] = {}
        for name in order:
            timings[name] = asdict(
                _time_callable(
                    callables[name],
                    device=device,
                    warmup=warmup,
                    repeats=repeats,
                )
            )
            memory[name] = _peak_incremental_memory(callables[name], device=device)
        medians = {name: float(timings[name]["median_ms"]) for name in methods}
        rows.append(
            {
                "length": length,
                "batch": batch,
                "timing_order": list(order),
                "maximum_relative_errors": errors,
                "correctness_passed": passed,
                "timings_ms": timings,
                "incremental_cuda_memory_bytes": memory,
                "fastest_median_backend": min(medians, key=medians.get),
                "composition_counts": scan_composition_counts(length),
                "dependency_depths": scan_dependency_depths(length),
            }
        )

    long_rows = [row for row in rows if int(row["length"]) >= 256]
    winners = [str(row["fastest_median_backend"]) for row in long_rows]
    global_winner = winners[0] if winners and len(set(winners)) == 1 else None
    return {
        "experiment": "SchurScan structured-slot memory scanner benchmark",
        "device_metadata": _device_metadata(device),
        "protocol": {
            "dtype": str(dtype).removeprefix("torch."),
            "batch": batch,
            "slots": SLOTS,
            "coordinates_per_slot": DIMENSION,
            "state_scalars": STATE_WIDTH,
            "lengths": list(lengths),
            "warmup": warmup,
            "repeats": repeats,
            "seed": seed,
            "input_generation_timed": False,
            "dense_materialization_in_scan_only_rows": False,
            "dense_materialization_in_end_to_end_rows": True,
            "tf32_disabled": not torch.backends.cuda.matmul.allow_tf32,
        },
        "rows": rows,
        "global_local_winner_at_lengths_ge_256": global_winner,
        "claim_boundary": {
            "identical_slot_recurrence_all_rows": True,
            "triality_bind_unbind_timed": False,
            "fused_kernel_comparison": False,
            "faithful_modern_delta_kernel_comparison": False,
            "absolute_memory_architecture_winner_established": False,
        },
        "passed": all(bool(row["correctness_passed"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lengths", type=int, nargs="+", default=[64, 256, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = run_benchmark(
        device=torch.device(args.device),
        dtype=dtype,
        batch=args.batch,
        lengths=args.lengths,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
