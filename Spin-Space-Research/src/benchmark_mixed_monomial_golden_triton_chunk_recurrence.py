"""Benchmark the register-resident exact compiled chunk recurrence.

The candidate is one Triton program per batch sequence.  It walks exact N-H-N
dictionary labels serially, keeps the eight-state in registers, emits every
causal prefix, and provides an initial-state reverse kernel.  Controls include
the maintained work-efficient endpoint scan with preselected operators,
realistic eager indexing, the indexed local-prefix Triton kernel, and an eager
sequential recurrence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import triton

import benchmark_mixed_monomial_golden_chunk as chunk_benchmark
import mixed_monomial_golden_chunk_compiler as chunk_compiler
import mixed_monomial_golden_parallel_chunk_scan as parallel
import mixed_monomial_golden_triton_chunk_recurrence as recurrence

DEFAULT_BATCH_SIZES = (1, 64, 1024)
DEFAULT_CHUNK_COUNTS = (4, 16, 64)
EXACT_CHUNK_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "mixed_monomial_golden_chunk_compiler_20260817.json"
)
VARIANTS = (
    "parallel_preselected_eager",
    "parallel_indexed_eager",
    "parallel_indexed_triton_local",
    "sequential_indexed_eager",
    "fused_triton_recurrence",
)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _synchronize() -> None:
    torch.cuda.synchronize()


def _time_callable(
    function: Callable[[], torch.Tensor], repeats: int
) -> tuple[dict[str, float | int], torch.Tensor]:
    warmups = max(5, min(10, repeats // 4))
    output = function()
    for _ in range(warmups):
        output = function()
    _synchronize()
    samples = []
    for _ in range(repeats):
        _synchronize()
        start = time.perf_counter_ns()
        output = function()
        _synchronize()
        samples.append((time.perf_counter_ns() - start) / 1000)
    return (
        {
            "repeats": repeats,
            "warmups": warmups,
            "median_microseconds": statistics.median(samples),
            "p10_microseconds": _percentile(samples, 0.10),
            "p90_microseconds": _percentile(samples, 0.90),
            "minimum_microseconds": min(samples),
        },
        output,
    )


def _repeat_counts(batch_size: int, chunk_count: int) -> tuple[int, int]:
    work = batch_size * chunk_count
    if work <= 64:
        return 100, 30
    if work <= 1024:
        return 50, 20
    if work <= 16384:
        return 20, 10
    return 10, 10


def _speedup(baseline: dict[str, Any], candidate: dict[str, Any]) -> float:
    return float(
        baseline["median_microseconds"] / candidate["median_microseconds"]
    )


def _initial_state_gradient(
    function: Callable[[torch.Tensor], torch.Tensor],
    initial_base: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    initial = initial_base.detach().clone().requires_grad_(True)
    outputs = function(initial)
    loss = (outputs * weights).sum() / outputs.numel()
    (gradient,) = torch.autograd.grad(loss, initial)
    return outputs.detach(), gradient.detach()


def _benchmark_view(
    view: str,
    exact_table: chunk_compiler.CompiledPrefixTable,
    batch_sizes: Sequence[int],
    chunk_counts: Sequence[int],
    forward_tolerance: float,
    gradient_tolerance: float,
) -> dict[str, object]:
    prefix_table = torch.from_numpy(
        chunk_benchmark._float32_prefix_table(exact_table)
    ).cuda()
    _synchronize()
    rows = []
    all_parity = []

    for batch_size in batch_sizes:
        for chunk_count in chunk_counts:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                20_260_818 + 100_000 * batch_size + chunk_count
            )
            left_index = torch.randint(
                exact_table.monomial_count,
                (batch_size, chunk_count),
                generator=generator,
            ).cuda()
            middle_index = torch.randint(
                exact_table.golden_count,
                (batch_size, chunk_count),
                generator=generator,
            ).cuda()
            right_index = torch.randint(
                exact_table.monomial_count,
                (batch_size, chunk_count),
                generator=generator,
            ).cuda()
            selected = prefix_table[left_index, middle_index, right_index]
            endpoint = selected[..., 16:24, :]
            initial_base = torch.randn(
                batch_size, 8, generator=generator
            ).cuda()
            weights = torch.randn(
                batch_size, 3 * chunk_count, 8, generator=generator
            ).cuda()

            def parallel_preselected_eager(
                initial: torch.Tensor = initial_base,
                endpoint: torch.Tensor = endpoint,
                selected: torch.Tensor = selected,
            ) -> torch.Tensor:
                return parallel.compiled_parallel_states(
                    endpoint, selected, initial
                )

            def parallel_indexed_eager(
                initial: torch.Tensor = initial_base,
                endpoint: torch.Tensor = endpoint,
                left_index: torch.Tensor = left_index,
                middle_index: torch.Tensor = middle_index,
                right_index: torch.Tensor = right_index,
            ) -> torch.Tensor:
                return parallel.compiled_parallel_indexed_states(
                    endpoint,
                    prefix_table,
                    left_index,
                    middle_index,
                    right_index,
                    initial,
                    local_backend="eager",
                )

            def parallel_indexed_triton_local(
                initial: torch.Tensor = initial_base,
                endpoint: torch.Tensor = endpoint,
                left_index: torch.Tensor = left_index,
                middle_index: torch.Tensor = middle_index,
                right_index: torch.Tensor = right_index,
            ) -> torch.Tensor:
                return parallel.compiled_parallel_indexed_states(
                    endpoint,
                    prefix_table,
                    left_index,
                    middle_index,
                    right_index,
                    initial,
                    local_backend="triton",
                )

            def sequential_indexed_eager(
                initial: torch.Tensor = initial_base,
                left_index: torch.Tensor = left_index,
                middle_index: torch.Tensor = middle_index,
                right_index: torch.Tensor = right_index,
            ) -> torch.Tensor:
                return recurrence.indexed_chunk_recurrence(
                    prefix_table,
                    left_index,
                    middle_index,
                    right_index,
                    initial,
                    backend="eager",
                )

            def fused_triton_recurrence(
                initial: torch.Tensor = initial_base,
                left_index: torch.Tensor = left_index,
                middle_index: torch.Tensor = middle_index,
                right_index: torch.Tensor = right_index,
            ) -> torch.Tensor:
                return recurrence.indexed_chunk_recurrence(
                    prefix_table,
                    left_index,
                    middle_index,
                    right_index,
                    initial,
                    backend="triton",
                )

            functions = {
                "parallel_preselected_eager": parallel_preselected_eager,
                "parallel_indexed_eager": parallel_indexed_eager,
                "parallel_indexed_triton_local": parallel_indexed_triton_local,
                "sequential_indexed_eager": sequential_indexed_eager,
                "fused_triton_recurrence": fused_triton_recurrence,
            }
            with torch.inference_mode():
                outputs = {
                    name: function() for name, function in functions.items()
                }
            reference = outputs["parallel_preselected_eager"]
            forward_errors = {
                name: float((output - reference).abs().max())
                for name, output in outputs.items()
            }
            backward = {
                name: _initial_state_gradient(
                    function, initial_base, weights
                )
                for name, function in functions.items()
            }
            gradient_reference = backward["parallel_preselected_eager"][1]
            gradient_errors = {
                name: float((result[1] - gradient_reference).abs().max())
                for name, result in backward.items()
            }
            parity_passed = bool(
                max(forward_errors.values()) <= forward_tolerance
                and max(gradient_errors.values()) <= gradient_tolerance
            )
            all_parity.append(parity_passed)

            forward_repeats, backward_repeats = _repeat_counts(
                batch_size, chunk_count
            )
            with torch.inference_mode():
                forward_timings = {
                    name: _time_callable(function, forward_repeats)[0]
                    for name, function in functions.items()
                }
            backward_functions = {
                name: (
                    lambda function=function,
                    initial_base=initial_base,
                    weights=weights: _initial_state_gradient(
                        function, initial_base, weights
                    )[1]
                )
                for name, function in functions.items()
            }
            backward_timings = {
                name: _time_callable(function, backward_repeats)[0]
                for name, function in backward_functions.items()
            }

            def phase_report(
                timings: dict[str, dict[str, float | int]],
                errors: dict[str, float],
            ) -> dict[str, object]:
                candidate = timings["fused_triton_recurrence"]
                return {
                    "timings": timings,
                    "fused_speedup_vs_parallel_preselected_eager": _speedup(
                        timings["parallel_preselected_eager"], candidate
                    ),
                    "fused_speedup_vs_parallel_indexed_eager": _speedup(
                        timings["parallel_indexed_eager"], candidate
                    ),
                    "fused_speedup_vs_parallel_indexed_triton_local": _speedup(
                        timings["parallel_indexed_triton_local"], candidate
                    ),
                    "fused_speedup_vs_sequential_indexed_eager": _speedup(
                        timings["sequential_indexed_eager"], candidate
                    ),
                    "max_abs_error_vs_parallel_preselected_eager": errors,
                }

            rows.append(
                {
                    "batch_size": batch_size,
                    "chunk_count": chunk_count,
                    "primitive_sequence_length": 3 * chunk_count,
                    "forward": phase_report(forward_timings, forward_errors),
                    "forward_plus_initial_state_backward": phase_report(
                        backward_timings, gradient_errors
                    ),
                    "float32_forward_backward_parity_passed": parity_passed,
                }
            )

    return {
        "view": view,
        "prefix_table_shape": list(prefix_table.shape),
        "prefix_table_bytes": prefix_table.numel() * prefix_table.element_size(),
        "results": rows,
        "all_float32_forward_backward_parity_checks_passed": all(all_parity),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benchmark(
    batch_sizes: Sequence[int], chunk_counts: Sequence[int]
) -> dict[str, object]:
    """Run the bounded exact-table CUDA benchmark."""

    if not recurrence.triton_is_available():
        raise RuntimeError("this benchmark requires Triton and CUDA")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(20_260_818)
    forward_tolerance = 1e-4
    gradient_tolerance = 5e-5
    tables = chunk_compiler.runtime_prefix_tables()
    reports = []
    for view, table in tables.items():
        reports.append(
            _benchmark_view(
                view,
                table,
                batch_sizes,
                chunk_counts,
                forward_tolerance,
                gradient_tolerance,
            )
        )
        torch.cuda.empty_cache()
    parity_passed = all(
        report["all_float32_forward_backward_parity_checks_passed"]
        for report in reports
    )
    properties = torch.cuda.get_device_properties(0)
    return {
        "schema_version": 1,
        "experiment": "register-resident exact compiled chunk recurrence benchmark",
        "recorded_at": datetime.now().astimezone().isoformat(),
        "exact_chunk_compiler_artifact": EXACT_CHUNK_ARTIFACT.name,
        "exact_chunk_compiler_artifact_sha256": _sha256(EXACT_CHUNK_ARTIFACT),
        "algorithm": {
            "candidate": (
                "one Triton program per batch sequence walks C exact labelled "
                "24x8 operators, keeps the 8-state in registers, and emits 3C states"
            ),
            "parallel_control": (
                "eager PyTorch work-efficient scan over C endpoint matrices, "
                "followed by parallel local-prefix expansion"
            ),
            "depth_tradeoff": (
                "candidate has serial depth C inside each program; parallel control "
                "has logarithmic endpoint-tree depth but multiple kernel launches"
            ),
            "backward": (
                "custom reverse recurrence for initial-state gradients only"
            ),
        },
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "triton": triton.__version__,
            "cuda_device": {
                "name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "total_memory_bytes": properties.total_memory,
            },
        },
        "settings": {
            "device": "cuda",
            "batch_sizes": list(batch_sizes),
            "chunk_counts": list(chunk_counts),
            "primitive_lengths": [3 * value for value in chunk_counts],
            "variants": list(VARIANTS),
            "torch_cpu_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "cuda_synchronized_each_complete_call": True,
            "dtype": "float32",
            "index_dtype": "int64",
            "seed": 20_260_818,
            "forward_parity_tolerance_max_abs": forward_tolerance,
            "initial_gradient_parity_tolerance_max_abs": gradient_tolerance,
            "backward_scope": "gradient with respect to initial state",
        },
        "results": reports,
        "checks": {
            "triton_forward_and_backward_kernels_compiled_and_executed": True,
            "all_float32_forward_backward_parity_checks_passed": parity_passed,
            "all_result_grids_are_complete": all(
                len(report["results"]) == len(batch_sizes) * len(chunk_counts)
                for report in reports
            ),
        },
        "claim_scope": {
            "validated_implementation": [
                "one-kernel exact labelled chunk recurrence on CUDA float32",
                "custom initial-state reverse recurrence on CUDA float32",
                "automatic eager fallback for CPU, unsupported dtype, or trainable operator tables",
            ],
            "empirical": [
                "the recorded fused-recurrence, eager-recurrence, and parallel-control timing distributions on this workstation",
            ],
            "not_claimed": [
                "the candidate is a parallel prefix scan",
                "prefix-table gradients in the Triton path",
                "gradients through discrete label selection",
                "full model backward, optimizer-step, or end-to-end SSM throughput",
                "the timing ranking generalizes beyond the recorded workstation or sequence grid",
                "an SSM accuracy advantage",
            ],
        },
        "passed": parity_passed,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--batch-sizes",
        default=",".join(str(value) for value in DEFAULT_BATCH_SIZES),
    )
    parser.add_argument(
        "--chunk-counts",
        default=",".join(str(value) for value in DEFAULT_CHUNK_COUNTS),
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    batch_sizes = tuple(
        int(value) for value in arguments.batch_sizes.split(",") if value.strip()
    )
    chunk_counts = tuple(
        int(value) for value in arguments.chunk_counts.split(",") if value.strip()
    )
    report = benchmark(batch_sizes, chunk_counts)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
