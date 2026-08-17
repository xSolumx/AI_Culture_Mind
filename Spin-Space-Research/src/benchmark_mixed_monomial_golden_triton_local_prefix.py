"""Benchmark fused indexed local-prefix expansion on the exact N-H-N tables.

Two scopes are timed on CUDA:

1. the isolated labelled-table lookup plus 24-by-8 local expansion;
2. the full compiled two-stage scan, whose endpoint tree remains eager
   PyTorch while the selected local expansion uses either eager indexing or
   the custom Triton kernel.

Each scope has a realistic indexed eager control and an optimistic eager
control whose 24-by-8 operators were gathered before the timed region.
Forward and forward-plus-state-backward timings are recorded separately.
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
import mixed_monomial_golden_triton_local_prefix as fused

DEFAULT_BATCH_SIZES = (1, 64, 1024)
DEFAULT_CHUNK_COUNTS = (4, 16, 64)
EXACT_CHUNK_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "mixed_monomial_golden_chunk_compiler_20260817.json"
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


def _state_gradient(
    function: Callable[[torch.Tensor], torch.Tensor],
    state_base: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = state_base.detach().clone().requires_grad_(True)
    outputs = function(state)
    loss = (outputs * weights).sum() / outputs.numel()
    (gradient,) = torch.autograd.grad(loss, state)
    return outputs.detach(), gradient.detach()


def _timings(
    variants: dict[str, Callable[[], torch.Tensor]], repeats: int
) -> dict[str, dict[str, float | int]]:
    return {
        name: _time_callable(function, repeats)[0]
        for name, function in variants.items()
    }


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
                20_260_817 + 100_000 * batch_size + chunk_count
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
            incoming_base = torch.randn(
                batch_size, chunk_count, 8, generator=generator
            ).cuda()
            initial_base = torch.randn(
                batch_size, 8, generator=generator
            ).cuda()
            weights = torch.randn(
                batch_size, 3 * chunk_count, 8, generator=generator
            ).cuda()

            def local_preselected(
                incoming: torch.Tensor = incoming_base,
                selected: torch.Tensor = selected,
                batch_size: int = batch_size,
                chunk_count: int = chunk_count,
            ) -> torch.Tensor:
                packed = (selected @ incoming[..., None]).squeeze(-1)
                return packed.reshape(batch_size, 3 * chunk_count, 8)

            def local_indexed_eager(
                incoming: torch.Tensor = incoming_base,
                left_index: torch.Tensor = left_index,
                middle_index: torch.Tensor = middle_index,
                right_index: torch.Tensor = right_index,
            ) -> torch.Tensor:
                return fused.indexed_local_prefix_states(
                    prefix_table,
                    left_index,
                    middle_index,
                    right_index,
                    incoming,
                    backend="eager",
                )

            def local_indexed_triton(
                incoming: torch.Tensor = incoming_base,
                left_index: torch.Tensor = left_index,
                middle_index: torch.Tensor = middle_index,
                right_index: torch.Tensor = right_index,
            ) -> torch.Tensor:
                return fused.indexed_local_prefix_states(
                    prefix_table,
                    left_index,
                    middle_index,
                    right_index,
                    incoming,
                    backend="triton",
                )

            def full_preselected(
                initial: torch.Tensor = initial_base,
                endpoint: torch.Tensor = endpoint,
                selected: torch.Tensor = selected,
            ) -> torch.Tensor:
                return parallel.compiled_parallel_states(
                    endpoint, selected, initial
                )

            def full_indexed_eager(
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

            def full_indexed_triton(
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

            local_functions = {
                "preselected_eager": local_preselected,
                "indexed_eager": local_indexed_eager,
                "indexed_triton": local_indexed_triton,
            }
            full_functions = {
                "preselected_eager": full_preselected,
                "indexed_eager": full_indexed_eager,
                "indexed_triton": full_indexed_triton,
            }
            with torch.inference_mode():
                local_outputs = {
                    name: function()
                    for name, function in local_functions.items()
                }
                full_outputs = {
                    name: function()
                    for name, function in full_functions.items()
                }

            local_forward_errors = {
                name: float(
                    (output - local_outputs["preselected_eager"])
                    .abs()
                    .max()
                )
                for name, output in local_outputs.items()
            }
            full_forward_errors = {
                name: float(
                    (output - full_outputs["preselected_eager"]).abs().max()
                )
                for name, output in full_outputs.items()
            }
            local_backward = {
                name: _state_gradient(function, incoming_base, weights)
                for name, function in local_functions.items()
            }
            full_backward = {
                name: _state_gradient(function, initial_base, weights)
                for name, function in full_functions.items()
            }
            local_gradient_errors = {
                name: float(
                    (result[1] - local_backward["preselected_eager"][1])
                    .abs()
                    .max()
                )
                for name, result in local_backward.items()
            }
            full_gradient_errors = {
                name: float(
                    (result[1] - full_backward["preselected_eager"][1])
                    .abs()
                    .max()
                )
                for name, result in full_backward.items()
            }
            parity_passed = bool(
                max(*local_forward_errors.values(), *full_forward_errors.values())
                <= forward_tolerance
                and max(
                    *local_gradient_errors.values(),
                    *full_gradient_errors.values(),
                )
                <= gradient_tolerance
            )
            all_parity.append(parity_passed)

            forward_repeats, backward_repeats = _repeat_counts(
                batch_size, chunk_count
            )
            with torch.inference_mode():
                local_forward_timings = _timings(
                    local_functions, forward_repeats
                )
                full_forward_timings = _timings(
                    full_functions, forward_repeats
                )

            local_backward_functions = {
                name: (
                    lambda function=function,
                    incoming_base=incoming_base,
                    weights=weights: _state_gradient(
                        function, incoming_base, weights
                    )[1]
                )
                for name, function in local_functions.items()
            }
            full_backward_functions = {
                name: (
                    lambda function=function,
                    initial_base=initial_base,
                    weights=weights: _state_gradient(
                        function, initial_base, weights
                    )[1]
                )
                for name, function in full_functions.items()
            }
            local_backward_timings = _timings(
                local_backward_functions, backward_repeats
            )
            full_backward_timings = _timings(
                full_backward_functions, backward_repeats
            )

            def timing_scope(
                forward: dict[str, dict[str, float | int]],
                backward: dict[str, dict[str, float | int]],
                forward_errors: dict[str, float],
                gradient_errors: dict[str, float],
            ) -> dict[str, object]:
                return {
                    "forward": {
                        "timings": forward,
                        "triton_speedup_vs_indexed_eager": _speedup(
                            forward["indexed_eager"],
                            forward["indexed_triton"],
                        ),
                        "triton_speedup_vs_preselected_eager": _speedup(
                            forward["preselected_eager"],
                            forward["indexed_triton"],
                        ),
                        "max_abs_error_vs_preselected_eager": forward_errors,
                    },
                    "forward_plus_state_backward": {
                        "timings": backward,
                        "triton_speedup_vs_indexed_eager": _speedup(
                            backward["indexed_eager"],
                            backward["indexed_triton"],
                        ),
                        "triton_speedup_vs_preselected_eager": _speedup(
                            backward["preselected_eager"],
                            backward["indexed_triton"],
                        ),
                        "state_gradient_max_abs_error_vs_preselected_eager": (
                            gradient_errors
                        ),
                    },
                }

            rows.append(
                {
                    "batch_size": batch_size,
                    "chunk_count": chunk_count,
                    "primitive_sequence_length": 3 * chunk_count,
                    "local_expansion_only": timing_scope(
                        local_forward_timings,
                        local_backward_timings,
                        local_forward_errors,
                        local_gradient_errors,
                    ),
                    "full_two_stage_scan": timing_scope(
                        full_forward_timings,
                        full_backward_timings,
                        full_forward_errors,
                        full_gradient_errors,
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

    if not fused.triton_is_available():
        raise RuntimeError("this benchmark requires Triton and CUDA")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(20_260_817)
    forward_tolerance = 1e-4
    gradient_tolerance = 5e-5
    tables = chunk_compiler.runtime_prefix_tables()
    reports = [
        _benchmark_view(
            view,
            table,
            batch_sizes,
            chunk_counts,
            forward_tolerance,
            gradient_tolerance,
        )
        for view, table in tables.items()
    ]
    parity_passed = all(
        report["all_float32_forward_backward_parity_checks_passed"]
        for report in reports
    )
    properties = torch.cuda.get_device_properties(0)
    return {
        "schema_version": 1,
        "experiment": "fused indexed exact local-prefix expansion benchmark",
        "recorded_at": datetime.now().astimezone().isoformat(),
        "exact_chunk_compiler_artifact": EXACT_CHUNK_ARTIFACT.name,
        "exact_chunk_compiler_artifact_sha256": _sha256(EXACT_CHUNK_ARTIFACT),
        "algorithm": {
            "local_kernel": (
                "one Triton program per chunk fuses a three-index exact-table "
                "lookup with one 24x8 matrix-vector product"
            ),
            "full_scan": (
                "eager PyTorch work-efficient endpoint scan followed by the "
                "fused indexed local-prefix kernel"
            ),
            "backward": (
                "custom Triton transpose matvec for incoming-state gradients; "
                "endpoint-tree autograd remains PyTorch"
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
            "torch_cpu_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "cuda_synchronized_each_complete_call": True,
            "dtype": "float32",
            "index_dtype": "int64",
            "seed": 20_260_817,
            "forward_parity_tolerance_max_abs": forward_tolerance,
            "state_gradient_parity_tolerance_max_abs": gradient_tolerance,
            "backward_scope": (
                "incoming-state gradient for isolated kernel and initial-state "
                "gradient for full scan"
            ),
        },
        "results": reports,
        "checks": {
            "triton_kernel_compiled_and_executed": True,
            "all_float32_forward_backward_parity_checks_passed": parity_passed,
            "all_result_grids_are_complete": all(
                len(report["results"]) == len(batch_sizes) * len(chunk_counts)
                for report in reports
            ),
        },
        "claim_scope": {
            "validated_implementation": [
                "indexed 24x8 local-prefix forward on CUDA float32",
                "incoming-state transpose-matvec backward on CUDA float32",
                "integration after the eager work-efficient endpoint scan",
                "automatic eager fallback for CPU, unsupported dtype, or trainable operator tables",
            ],
            "empirical": [
                "the recorded isolated-kernel and full-scan timing distributions on this workstation",
            ],
            "not_claimed": [
                "prefix-table gradients in the Triton path",
                "a fused endpoint matrix scan",
                "gradients through discrete label selection",
                "full model backward, optimizer-step, or end-to-end SSM throughput",
                "the timing ranking generalizes beyond the recorded workstation",
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
