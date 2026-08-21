"""Benchmark the two-stage compiled parallel scan, including state backward.

The primitive control performs the maintained work-efficient ordered scan over
all ``3C`` matrices.  The compiled path scans only ``C`` endpoint matrices and
then applies the selected 24-by-8 local-prefix operators in parallel.  Forward
and forward-plus-initial-state-backward timings are recorded separately.

This is a PyTorch composition benchmark, not a fused Triton/CUDA kernel.
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

import benchmark_mixed_monomial_golden_chunk as chunk_benchmark
import benchmark_mixed_monomial_golden_macro as macro_benchmark
import mixed_monomial_golden_chunk_compiler as chunk_compiler
import mixed_monomial_golden_closure as closure
import mixed_monomial_golden_parallel_chunk_scan as parallel
import octonion_operator_groups as monomial
import spin8_triality_2a5_closure as golden

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


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_callable(
    function: Callable[[], torch.Tensor],
    device: torch.device,
    repeats: int,
) -> tuple[dict[str, float | int], torch.Tensor]:
    warmups = max(5, min(10, repeats // 4))
    output = function()
    for _ in range(warmups):
        output = function()
    _synchronize(device)
    samples = []
    for _ in range(repeats):
        _synchronize(device)
        start = time.perf_counter_ns()
        output = function()
        _synchronize(device)
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
    initial: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial.detach().clone().requires_grad_(True)
    outputs = function(state)
    loss = (outputs * weights).sum() / outputs.numel()
    (gradient,) = torch.autograd.grad(loss, state)
    return outputs.detach(), gradient.detach()


def _benchmark_view_device(
    view: str,
    prefix_table_exact: chunk_compiler.CompiledPrefixTable,
    monomial_steps: tuple[chunk_compiler.FieldMatrix, ...],
    golden_steps: tuple[chunk_compiler.FieldMatrix, ...],
    device: torch.device,
    batch_sizes: Sequence[int],
    chunk_counts: Sequence[int],
    forward_tolerance: float,
    gradient_tolerance: float,
) -> dict[str, object]:
    monomial_table = torch.from_numpy(
        macro_benchmark._float32_matrices(monomial_steps)
    ).to(device)
    golden_table = torch.from_numpy(
        macro_benchmark._float32_matrices(golden_steps)
    ).to(device)
    prefix_table = torch.from_numpy(
        chunk_benchmark._float32_prefix_table(prefix_table_exact)
    ).to(device)
    rows = []
    all_parity = []

    for batch_size in batch_sizes:
        for chunk_count in chunk_counts:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                20_260_817 + 100_000 * batch_size + chunk_count
            )
            left_index = torch.randint(
                len(monomial_steps),
                (batch_size, chunk_count),
                generator=generator,
            ).to(device)
            middle_index = torch.randint(
                len(golden_steps),
                (batch_size, chunk_count),
                generator=generator,
            ).to(device)
            right_index = torch.randint(
                len(monomial_steps),
                (batch_size, chunk_count),
                generator=generator,
            ).to(device)
            left = monomial_table[left_index]
            middle = golden_table[middle_index]
            right = monomial_table[right_index]
            local_prefix = prefix_table[left_index, middle_index, right_index]
            endpoint = local_prefix[..., 16:24, :]
            initial = torch.randn(
                batch_size, 8, generator=generator
            ).to(device)
            weights = torch.randn(
                batch_size,
                3 * chunk_count,
                8,
                generator=generator,
            ).to(device)

            def recurrent_forward(
                state: torch.Tensor = initial,
                left: torch.Tensor = left,
                middle: torch.Tensor = middle,
                right: torch.Tensor = right,
            ) -> torch.Tensor:
                return parallel.primitive_recurrent_states(
                    left, middle, right, state
                )

            def primitive_forward(
                state: torch.Tensor = initial,
                left: torch.Tensor = left,
                middle: torch.Tensor = middle,
                right: torch.Tensor = right,
            ) -> torch.Tensor:
                return parallel.primitive_parallel_states(
                    left,
                    middle,
                    right,
                    state,
                    backend="work_efficient",
                )

            def compiled_forward(
                state: torch.Tensor = initial,
                endpoint: torch.Tensor = endpoint,
                local_prefix: torch.Tensor = local_prefix,
            ) -> torch.Tensor:
                return parallel.compiled_parallel_states(
                    endpoint,
                    local_prefix,
                    state,
                    backend="work_efficient",
                )

            with torch.inference_mode():
                recurrent_reference = recurrent_forward()
                primitive_reference = primitive_forward()
                compiled_reference = compiled_forward()
            primitive_forward_error = float(
                (recurrent_reference - primitive_reference).abs().max()
            )
            compiled_forward_error = float(
                (recurrent_reference - compiled_reference).abs().max()
            )

            primitive_outputs, primitive_gradient = _initial_state_gradient(
                primitive_forward, initial, weights
            )
            compiled_outputs, compiled_gradient = _initial_state_gradient(
                compiled_forward, initial, weights
            )
            backward_output_error = float(
                (primitive_outputs - compiled_outputs).abs().max()
            )
            initial_gradient_error = float(
                (primitive_gradient - compiled_gradient).abs().max()
            )
            parity_passed = bool(
                max(
                    primitive_forward_error,
                    compiled_forward_error,
                    backward_output_error,
                )
                <= forward_tolerance
                and initial_gradient_error <= gradient_tolerance
            )
            all_parity.append(parity_passed)

            forward_repeats, backward_repeats = _repeat_counts(
                batch_size, chunk_count
            )
            with torch.inference_mode():
                primitive_forward_timing, _ = _time_callable(
                    primitive_forward, device, forward_repeats
                )
                compiled_forward_timing, _ = _time_callable(
                    compiled_forward, device, forward_repeats
                )

            def primitive_forward_backward(
                function: Callable[[torch.Tensor], torch.Tensor] = primitive_forward,
                initial: torch.Tensor = initial,
                weights: torch.Tensor = weights,
            ) -> torch.Tensor:
                _, gradient = _initial_state_gradient(
                    function, initial, weights
                )
                return gradient

            def compiled_forward_backward(
                function: Callable[[torch.Tensor], torch.Tensor] = compiled_forward,
                initial: torch.Tensor = initial,
                weights: torch.Tensor = weights,
            ) -> torch.Tensor:
                _, gradient = _initial_state_gradient(
                    function, initial, weights
                )
                return gradient

            primitive_backward_timing, _ = _time_callable(
                primitive_forward_backward, device, backward_repeats
            )
            compiled_backward_timing, _ = _time_callable(
                compiled_forward_backward, device, backward_repeats
            )

            rows.append(
                {
                    "batch_size": batch_size,
                    "chunk_count": chunk_count,
                    "primitive_sequence_length": 3 * chunk_count,
                    "composition_counts": parallel.scan_composition_counts(
                        chunk_count
                    ),
                    "forward": {
                        "primitive_work_efficient_scan": primitive_forward_timing,
                        "compiled_two_stage_scan": compiled_forward_timing,
                        "compiled_speedup_vs_primitive": _speedup(
                            primitive_forward_timing, compiled_forward_timing
                        ),
                        "primitive_vs_recurrent_max_abs_error": (
                            primitive_forward_error
                        ),
                        "compiled_vs_recurrent_max_abs_error": (
                            compiled_forward_error
                        ),
                    },
                    "forward_plus_initial_state_backward": {
                        "primitive_work_efficient_scan": primitive_backward_timing,
                        "compiled_two_stage_scan": compiled_backward_timing,
                        "compiled_speedup_vs_primitive": _speedup(
                            primitive_backward_timing, compiled_backward_timing
                        ),
                        "forward_max_abs_error": backward_output_error,
                        "initial_state_gradient_max_abs_error": (
                            initial_gradient_error
                        ),
                    },
                    "float32_forward_backward_parity_passed": parity_passed,
                }
            )

    return {
        "view": view,
        "device": str(device),
        "matrix_dtype": str(prefix_table.dtype),
        "results": rows,
        "all_float32_forward_backward_parity_checks_passed": all(all_parity),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benchmark(
    devices: Sequence[str],
    batch_sizes: Sequence[int],
    chunk_counts: Sequence[int],
) -> dict[str, object]:
    """Run the bounded parallel forward/backward benchmark."""

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(20_260_817)
    forward_tolerance = 1e-4
    gradient_tolerance = 5e-5

    _, _, operator_generators = monomial._operator_group_certificate()
    _, _, automorphism_generators = monomial._automorphism_group_certificate()
    normalizer_generators = (*operator_generators, *automorphism_generators)
    monomial_steps = tuple(
        closure._field_matrix_from_monomial(step)
        for step in closure._symmetric_monomial_steps(normalizer_generators)
    )
    triality_matrices = golden._spin8_action_matrices()
    active_devices = [
        torch.device(name)
        for name in devices
        if name != "cuda" or torch.cuda.is_available()
    ]
    if not active_devices:
        raise RuntimeError("no requested benchmark device is available")

    reports = []
    for view, indices in closure.VIEW_GENERATOR_INDICES.items():
        pair = tuple(triality_matrices[index] for index in indices)
        golden_steps = closure._symmetric_golden_steps(pair)
        prefix_table = chunk_compiler.compile_prefix_table(
            monomial_steps, golden_steps
        )
        for device in active_devices:
            reports.append(
                _benchmark_view_device(
                    view,
                    prefix_table,
                    monomial_steps,
                    golden_steps,
                    device,
                    batch_sizes,
                    chunk_counts,
                    forward_tolerance,
                    gradient_tolerance,
                )
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    parity_passed = all(
        report["all_float32_forward_backward_parity_checks_passed"]
        for report in reports
    )
    properties = None
    if torch.cuda.is_available():
        value = torch.cuda.get_device_properties(0)
        properties = {
            "name": value.name,
            "compute_capability": [value.major, value.minor],
            "total_memory_bytes": value.total_memory,
        }
    return {
        "schema_version": 1,
        "experiment": "two-stage compiled parallel chunk scan benchmark",
        "recorded_at": datetime.now().astimezone().isoformat(),
        "exact_chunk_compiler_artifact": EXACT_CHUNK_ARTIFACT.name,
        "exact_chunk_compiler_artifact_sha256": _sha256(EXACT_CHUNK_ARTIFACT),
        "algorithm": {
            "primitive": (
                "work-efficient ordered scan over 3C primitive 8x8 matrices"
            ),
            "compiled": (
                "work-efficient ordered scan over C endpoint 8x8 matrices, "
                "then parallel selected 24x8 local-prefix application"
            ),
            "implementation": "eager PyTorch composition; no fused custom kernel",
        },
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": properties,
        },
        "settings": {
            "devices": [str(device) for device in active_devices],
            "batch_sizes": list(batch_sizes),
            "chunk_counts": list(chunk_counts),
            "primitive_lengths": [3 * value for value in chunk_counts],
            "torch_cpu_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "cuda_synchronized_each_complete_call": True,
            "dtype": "float32",
            "seed": 20_260_817,
            "forward_parity_tolerance_max_abs": forward_tolerance,
            "initial_gradient_parity_tolerance_max_abs": gradient_tolerance,
            "backward_scope": "gradient with respect to initial state",
        },
        "results": reports,
        "checks": {
            "all_float32_forward_backward_parity_checks_passed": parity_passed,
            "all_result_grids_are_complete": all(
                len(report["results"]) == len(batch_sizes) * len(chunk_counts)
                for report in reports
            ),
        },
        "claim_scope": {
            "validated_implementation": [
                "the two scan trees and sequential recurrence agree in float64 unit tests, including gradients with respect to L,H,R and initial state",
                "the recorded float32 forward and initial-state-gradient parity across the workstation grid",
            ],
            "empirical": [
                "the recorded eager-PyTorch forward and forward-plus-initial-state-backward timing distributions",
            ],
            "not_claimed": [
                "a fused Triton or custom CUDA scan kernel has been implemented",
                "gradients through discrete label selection or a learned continuous transition compiler",
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
    parser.add_argument("--devices", default="cpu,cuda")
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
    devices = tuple(
        value.strip() for value in arguments.devices.split(",") if value.strip()
    )
    batch_sizes = tuple(
        int(value) for value in arguments.batch_sizes.split(",") if value.strip()
    )
    chunk_counts = tuple(
        int(value) for value in arguments.chunk_counts.split(",") if value.strip()
    )
    report = benchmark(devices, batch_sizes, chunk_counts)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
