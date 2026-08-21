"""Benchmark exact compiled N-H-N chunks with every causal prefix emitted.

The exact chunk compiler stores one 24-by-8 operator per labelled triple.  Its
three 8-row blocks produce the right, middle-right, and left-middle-right
prefix states from one chunk input.  This empirical benchmark compares that
single stacked application with three primitive 8-by-8 applications over
multi-chunk recurrent sequences.

Endpoint-only and every-prefix paths are timed separately. CPU execution is
single-threaded; CUDA is synchronized around each complete timed call.
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

import numpy as np
import torch

import benchmark_mixed_monomial_golden_macro as macro_benchmark
import mixed_monomial_golden_chunk_compiler as chunk
import mixed_monomial_golden_closure as closure
import octonion_operator_groups as monomial
import spin8_triality_2a5_closure as golden

DEFAULT_BATCH_SIZES = (1, 64, 1024)
DEFAULT_CHUNK_COUNTS = (1, 4, 16, 64)
EXACT_CHUNK_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "mixed_monomial_golden_chunk_compiler_20260817.json"
)


def _float32_prefix_table(table: chunk.CompiledPrefixTable) -> np.ndarray:
    flat = np.stack(
        [
            np.asarray(
                [
                    [float(golden.FIELD.to_sympy(value)) for value in row]
                    for row in operator
                ],
                dtype=np.float32,
            )
            for operator in table.operators
        ]
    )
    return flat.reshape(
        table.monomial_count,
        table.golden_count,
        table.monomial_count,
        24,
        8,
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
    warmups = max(5, min(20, repeats // 5))
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


def _repeats(batch_size: int, chunk_count: int) -> int:
    work = batch_size * chunk_count
    if work <= 64:
        return 200
    if work <= 1024:
        return 50
    if work <= 16384:
        return 15
    return 5


def _speedup(baseline: dict[str, Any], candidate: dict[str, Any]) -> float:
    return float(
        baseline["median_microseconds"] / candidate["median_microseconds"]
    )


def _benchmark_view_device(
    view: str,
    prefix_table_exact: chunk.CompiledPrefixTable,
    monomial_steps: tuple[chunk.FieldMatrix, ...],
    golden_steps: tuple[chunk.FieldMatrix, ...],
    device: torch.device,
    batch_sizes: Sequence[int],
    chunk_counts: Sequence[int],
    parity_tolerance: float,
) -> dict[str, object]:
    monomial_table = torch.from_numpy(
        macro_benchmark._float32_matrices(monomial_steps)
    ).to(device)
    golden_table = torch.from_numpy(
        macro_benchmark._float32_matrices(golden_steps)
    ).to(device)
    prefix_table = torch.from_numpy(
        _float32_prefix_table(prefix_table_exact)
    ).to(device)
    endpoint_table = prefix_table[..., 16:24, :]
    _synchronize(device)

    rows = []
    all_parity = []
    for batch_size in batch_sizes:
        for chunk_count in chunk_counts:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                20_260_817 + 100_000 * batch_size + chunk_count
            )
            left = torch.randint(
                len(monomial_steps),
                (chunk_count, batch_size),
                generator=generator,
            ).to(device)
            middle = torch.randint(
                len(golden_steps),
                (chunk_count, batch_size),
                generator=generator,
            ).to(device)
            right = torch.randint(
                len(monomial_steps),
                (chunk_count, batch_size),
                generator=generator,
            ).to(device)
            initial = torch.randn(
                batch_size, 8, 1, generator=generator
            ).to(device)

            def online_endpoint(
                left: torch.Tensor = left,
                middle: torch.Tensor = middle,
                right: torch.Tensor = right,
                state: torch.Tensor = initial,
                chunk_count: int = chunk_count,
            ) -> torch.Tensor:
                value = state
                for offset in range(chunk_count):
                    value = torch.bmm(monomial_table[right[offset]], value)
                    value = torch.bmm(golden_table[middle[offset]], value)
                    value = torch.bmm(monomial_table[left[offset]], value)
                return value

            def compiled_endpoint(
                left: torch.Tensor = left,
                middle: torch.Tensor = middle,
                right: torch.Tensor = right,
                state: torch.Tensor = initial,
                chunk_count: int = chunk_count,
            ) -> torch.Tensor:
                value = state
                for offset in range(chunk_count):
                    matrix = endpoint_table[
                        left[offset], middle[offset], right[offset]
                    ]
                    value = torch.bmm(matrix, value)
                return value

            def online_every_prefix(
                left: torch.Tensor = left,
                middle: torch.Tensor = middle,
                right: torch.Tensor = right,
                state: torch.Tensor = initial,
                chunk_count: int = chunk_count,
            ) -> torch.Tensor:
                value = state
                outputs = []
                for offset in range(chunk_count):
                    value = torch.bmm(monomial_table[right[offset]], value)
                    outputs.append(value)
                    value = torch.bmm(golden_table[middle[offset]], value)
                    outputs.append(value)
                    value = torch.bmm(monomial_table[left[offset]], value)
                    outputs.append(value)
                return torch.stack(outputs, dim=1)

            def compiled_every_prefix(
                left: torch.Tensor = left,
                middle: torch.Tensor = middle,
                right: torch.Tensor = right,
                state: torch.Tensor = initial,
                chunk_count: int = chunk_count,
                batch_size: int = batch_size,
            ) -> torch.Tensor:
                value = state
                outputs = []
                for offset in range(chunk_count):
                    operator = prefix_table[
                        left[offset], middle[offset], right[offset]
                    ]
                    packed = torch.bmm(operator, value)
                    prefixes = packed.reshape(batch_size, 3, 8, 1)
                    outputs.append(prefixes)
                    value = prefixes[:, 2]
                return torch.cat(outputs, dim=1)

            repeats = _repeats(batch_size, chunk_count)
            with torch.inference_mode():
                endpoint_online_timing, endpoint_reference = _time_callable(
                    online_endpoint, device, repeats
                )
                endpoint_compiled_timing, endpoint_compiled_output = (
                    _time_callable(compiled_endpoint, device, repeats)
                )
                prefix_online_timing, prefix_reference = _time_callable(
                    online_every_prefix, device, repeats
                )
                prefix_compiled_timing, prefix_compiled_output = (
                    _time_callable(compiled_every_prefix, device, repeats)
                )
            endpoint_error = float(
                (endpoint_reference - endpoint_compiled_output).abs().max()
            )
            prefix_error = float(
                (prefix_reference - prefix_compiled_output).abs().max()
            )
            parity_passed = max(endpoint_error, prefix_error) <= parity_tolerance
            all_parity.append(parity_passed)
            rows.append(
                {
                    "batch_size": batch_size,
                    "chunk_count": chunk_count,
                    "primitive_sequence_length": 3 * chunk_count,
                    "endpoint_only": {
                        "online_three_bmm_per_chunk": endpoint_online_timing,
                        "compiled_one_bmm_per_chunk": endpoint_compiled_timing,
                        "compiled_speedup_vs_online": _speedup(
                            endpoint_online_timing, endpoint_compiled_timing
                        ),
                        "max_abs_error": endpoint_error,
                    },
                    "every_prefix": {
                        "online_three_bmm_per_chunk": prefix_online_timing,
                        "compiled_one_24x8_bmm_per_chunk": prefix_compiled_timing,
                        "compiled_speedup_vs_online": _speedup(
                            prefix_online_timing, prefix_compiled_timing
                        ),
                        "max_abs_error": prefix_error,
                    },
                    "float32_parity_passed": parity_passed,
                }
            )
    return {
        "view": view,
        "device": str(device),
        "matrix_dtype": str(prefix_table.dtype),
        "prefix_table_shape": list(prefix_table.shape),
        "results": rows,
        "all_float32_parity_checks_passed": all(all_parity),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benchmark(
    devices: Sequence[str],
    batch_sizes: Sequence[int],
    chunk_counts: Sequence[int],
) -> dict[str, object]:
    """Run the bounded every-prefix benchmark."""

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(20_260_817)
    parity_tolerance = 1e-4

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
        prefix_table = chunk.compile_prefix_table(monomial_steps, golden_steps)
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
                    parity_tolerance,
                )
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    parity_passed = all(
        report["all_float32_parity_checks_passed"] for report in reports
    )
    properties = None
    if torch.cuda.is_available():
        device_properties = torch.cuda.get_device_properties(0)
        properties = {
            "name": device_properties.name,
            "compute_capability": [
                device_properties.major,
                device_properties.minor,
            ],
            "total_memory_bytes": device_properties.total_memory,
        }
    return {
        "schema_version": 1,
        "experiment": "every-prefix compiled N-H-N chunk benchmark",
        "recorded_at": datetime.now().astimezone().isoformat(),
        "exact_chunk_compiler_artifact": EXACT_CHUNK_ARTIFACT.name,
        "exact_chunk_compiler_artifact_sha256": _sha256(EXACT_CHUNK_ARTIFACT),
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
            "parity_tolerance_max_abs": parity_tolerance,
        },
        "results": reports,
        "checks": {
            "all_float32_parity_checks_passed": parity_passed,
            "all_result_grids_are_complete": all(
                len(report["results"]) == len(batch_sizes) * len(chunk_counts)
                for report in reports
            ),
        },
        "claim_scope": {
            "empirical": [
                "the recorded endpoint-only and every-prefix timing distributions on this fixed workstation grid",
                "float32 recurrent outputs agree within the declared tolerance through the tested sequence lengths",
            ],
            "not_claimed": [
                "the timing ranking generalizes beyond the recorded hardware and software stack",
                "a parallel prefix-scan kernel has been implemented or benchmarked",
                "backward or training throughput improves",
                "the discrete N-H-N chunk table applies to continuous learned transitions",
                "an end-to-end SSM accuracy advantage",
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
