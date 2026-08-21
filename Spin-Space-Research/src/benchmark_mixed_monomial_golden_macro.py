"""Benchmark compiled N-H-N macro lookup against three primitive products.

This is an empirical workstation benchmark.  It consumes the exact dictionary
compiler, converts the maintained Q(sqrt(5)) matrices to float32, and compares
three execution paths:

* online: gather n1, h, n2 and apply or compose them at runtime;
* deduplicated: gather a uint16-deployable macro id, then one stored matrix;
* labelled table: store one matrix for every labelled triple and gather it
  directly, retaining duplicates for simpler addressing.

CPU runs use one PyTorch thread. CUDA runs synchronize around every timed call.
The artifact records medians and percentile spread but is not an exact theorem
or a claim about other hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

import mixed_monomial_golden_closure as closure
import mixed_monomial_golden_macro_compiler as compiler
import octonion_operator_groups as monomial
import spin8_triality_2a5_closure as golden

DEFAULT_BATCH_SIZES = (1, 64, 1024, 16384)
EXACT_COMPILER_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "mixed_monomial_golden_macro_compiler_20260817.json"
)


def _float32_matrix(matrix: compiler.FieldMatrix) -> np.ndarray:
    return np.asarray(
        [
            [float(golden.FIELD.to_sympy(value)) for value in row]
            for row in matrix
        ],
        dtype=np.float32,
    )


def _float32_matrices(
    matrices: Sequence[compiler.FieldMatrix],
) -> np.ndarray:
    return np.stack([_float32_matrix(matrix) for matrix in matrices])


def _lookup_array(dictionary: compiler.CompiledMacroDictionary) -> np.ndarray:
    return np.asarray(dictionary.lookup, dtype=np.int64)


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
    warmups = max(10, min(50, repeats // 5))
    output = function()
    for _ in range(warmups):
        output = function()
    _synchronize(device)

    elapsed_microseconds = []
    for _ in range(repeats):
        _synchronize(device)
        start = time.perf_counter_ns()
        output = function()
        _synchronize(device)
        elapsed_microseconds.append((time.perf_counter_ns() - start) / 1000)
    return (
        {
            "repeats": repeats,
            "warmups": warmups,
            "median_microseconds": statistics.median(elapsed_microseconds),
            "p10_microseconds": _percentile(elapsed_microseconds, 0.10),
            "p90_microseconds": _percentile(elapsed_microseconds, 0.90),
            "minimum_microseconds": min(elapsed_microseconds),
        },
        output,
    )


def _repeats(batch_size: int) -> int:
    if batch_size <= 64:
        return 500
    if batch_size <= 1024:
        return 150
    return 30


def _speedup(baseline: dict[str, Any], candidate: dict[str, Any]) -> float:
    return float(
        baseline["median_microseconds"] / candidate["median_microseconds"]
    )


def _benchmark_view_device(
    view: str,
    dictionary: compiler.CompiledMacroDictionary,
    monomial_steps: tuple[compiler.FieldMatrix, ...],
    golden_steps: tuple[compiler.FieldMatrix, ...],
    device: torch.device,
    batch_sizes: Sequence[int],
) -> dict[str, object]:
    monomial_table = torch.from_numpy(_float32_matrices(monomial_steps)).to(device)
    golden_table = torch.from_numpy(_float32_matrices(golden_steps)).to(device)
    macro_table = torch.from_numpy(_float32_matrices(dictionary.matrices)).to(device)
    lookup = torch.from_numpy(_lookup_array(dictionary)).to(device)
    labelled_table = macro_table[lookup]
    _synchronize(device)

    results = []
    all_checks = []
    for batch_size in batch_sizes:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20_260_817 + batch_size)
        left = torch.randint(
            len(monomial_steps), (batch_size,), generator=generator
        ).to(device)
        middle = torch.randint(
            len(golden_steps), (batch_size,), generator=generator
        ).to(device)
        right = torch.randint(
            len(monomial_steps), (batch_size,), generator=generator
        ).to(device)
        state = torch.randn(batch_size, 8, 1, generator=generator).to(device)

        def online_transition(
            left: torch.Tensor = left,
            middle: torch.Tensor = middle,
            right: torch.Tensor = right,
        ) -> torch.Tensor:
            return torch.bmm(
                torch.bmm(monomial_table[left], golden_table[middle]),
                monomial_table[right],
            )

        def deduplicated_transition(
            left: torch.Tensor = left,
            middle: torch.Tensor = middle,
            right: torch.Tensor = right,
        ) -> torch.Tensor:
            return macro_table[lookup[left, middle, right]]

        def labelled_transition(
            left: torch.Tensor = left,
            middle: torch.Tensor = middle,
            right: torch.Tensor = right,
        ) -> torch.Tensor:
            return labelled_table[left, middle, right]

        def online_apply(
            left: torch.Tensor = left,
            middle: torch.Tensor = middle,
            right: torch.Tensor = right,
            state: torch.Tensor = state,
        ) -> torch.Tensor:
            value = torch.bmm(monomial_table[right], state)
            value = torch.bmm(golden_table[middle], value)
            return torch.bmm(monomial_table[left], value)

        def deduplicated_apply(
            left: torch.Tensor = left,
            middle: torch.Tensor = middle,
            right: torch.Tensor = right,
            state: torch.Tensor = state,
        ) -> torch.Tensor:
            matrix = macro_table[lookup[left, middle, right]]
            return torch.bmm(matrix, state)

        def labelled_apply(
            left: torch.Tensor = left,
            middle: torch.Tensor = middle,
            right: torch.Tensor = right,
            state: torch.Tensor = state,
        ) -> torch.Tensor:
            return torch.bmm(labelled_table[left, middle, right], state)

        repeats = _repeats(batch_size)
        with torch.inference_mode():
            transition_online, transition_reference = _time_callable(
                online_transition, device, repeats
            )
            transition_deduplicated, transition_deduplicated_output = (
                _time_callable(deduplicated_transition, device, repeats)
            )
            transition_labelled, transition_labelled_output = _time_callable(
                labelled_transition, device, repeats
            )
            apply_online, apply_reference = _time_callable(
                online_apply, device, repeats
            )
            apply_deduplicated, apply_deduplicated_output = _time_callable(
                deduplicated_apply, device, repeats
            )
            apply_labelled, apply_labelled_output = _time_callable(
                labelled_apply, device, repeats
            )

        transition_deduplicated_error = float(
            (transition_reference - transition_deduplicated_output).abs().max()
        )
        transition_labelled_error = float(
            (transition_reference - transition_labelled_output).abs().max()
        )
        apply_deduplicated_error = float(
            (apply_reference - apply_deduplicated_output).abs().max()
        )
        apply_labelled_error = float(
            (apply_reference - apply_labelled_output).abs().max()
        )
        parity_passed = max(
            transition_deduplicated_error,
            transition_labelled_error,
            apply_deduplicated_error,
            apply_labelled_error,
        ) <= 2e-6
        all_checks.append(parity_passed)
        results.append(
            {
                "batch_size": batch_size,
                "transition_materialization": {
                    "online_two_bmm": transition_online,
                    "deduplicated_lookup": transition_deduplicated,
                    "labelled_table_lookup": transition_labelled,
                    "deduplicated_speedup_vs_online": _speedup(
                        transition_online, transition_deduplicated
                    ),
                    "labelled_speedup_vs_online": _speedup(
                        transition_online, transition_labelled
                    ),
                    "deduplicated_max_abs_error": transition_deduplicated_error,
                    "labelled_max_abs_error": transition_labelled_error,
                },
                "state_application": {
                    "online_three_bmm": apply_online,
                    "deduplicated_lookup_plus_one_bmm": apply_deduplicated,
                    "labelled_lookup_plus_one_bmm": apply_labelled,
                    "deduplicated_speedup_vs_online": _speedup(
                        apply_online, apply_deduplicated
                    ),
                    "labelled_speedup_vs_online": _speedup(
                        apply_online, apply_labelled
                    ),
                    "deduplicated_max_abs_error": apply_deduplicated_error,
                    "labelled_max_abs_error": apply_labelled_error,
                },
                "float32_parity_passed": parity_passed,
            }
        )

    return {
        "view": view,
        "device": str(device),
        "torch_lookup_index_dtype": str(lookup.dtype),
        "matrix_dtype": str(macro_table.dtype),
        "labelled_table_shape": list(labelled_table.shape),
        "deduplicated_table_shape": list(macro_table.shape),
        "batch_results": results,
        "all_float32_parity_checks_passed": all(all_checks),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benchmark(
    devices: Sequence[str], batch_sizes: Sequence[int]
) -> dict[str, object]:
    """Run the bounded empirical benchmark and return its JSON payload."""

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(20_260_817)

    _, _, operator_generators = monomial._operator_group_certificate()
    _, _, automorphism_generators = monomial._automorphism_group_certificate()
    normalizer_generators = (*operator_generators, *automorphism_generators)
    monomial_steps = tuple(
        closure._field_matrix_from_monomial(step)
        for step in closure._symmetric_monomial_steps(normalizer_generators)
    )
    triality_matrices = golden._spin8_action_matrices()

    requested_devices = []
    for name in devices:
        if name == "cuda" and not torch.cuda.is_available():
            continue
        requested_devices.append(torch.device(name))
    if not requested_devices:
        raise RuntimeError("no requested benchmark device is available")

    results = []
    for view, indices in closure.VIEW_GENERATOR_INDICES.items():
        pair = tuple(triality_matrices[index] for index in indices)
        golden_steps = closure._symmetric_golden_steps(pair)
        dictionary = compiler.compile_macro_dictionary(
            monomial_steps, golden_steps
        )
        for device in requested_devices:
            results.append(
                _benchmark_view_device(
                    view,
                    dictionary,
                    monomial_steps,
                    golden_steps,
                    device,
                    batch_sizes,
                )
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    parity_passed = all(
        report["all_float32_parity_checks_passed"] for report in results
    )
    cuda_properties = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        cuda_properties = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
    return {
        "schema_version": 1,
        "experiment": "compiled N-H-N macro lookup workstation benchmark",
        "recorded_at": datetime.now().astimezone().isoformat(),
        "exact_compiler_artifact": str(EXACT_COMPILER_ARTIFACT.name),
        "exact_compiler_artifact_sha256": _sha256(EXACT_COMPILER_ARTIFACT),
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": cuda_properties,
            "pid": os.getpid(),
        },
        "settings": {
            "devices": [str(device) for device in requested_devices],
            "batch_sizes": list(batch_sizes),
            "torch_cpu_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "dtype": "float32",
            "cuda_synchronized_each_timing": True,
            "seed": 20_260_817,
            "parity_tolerance_max_abs": 2e-6,
        },
        "results": results,
        "checks": {
            "all_float32_parity_checks_passed": parity_passed,
            "every_timing_is_positive": all(
                timing[key]["median_microseconds"] > 0
                for report in results
                for batch in report["batch_results"]
                for timing in (
                    batch["transition_materialization"],
                    batch["state_application"],
                )
                for key in (
                    "online_two_bmm",
                    "deduplicated_lookup",
                    "labelled_table_lookup",
                )
                if key in timing
            )
            and all(
                timing[key]["median_microseconds"] > 0
                for report in results
                for batch in report["batch_results"]
                for timing in (batch["state_application"],)
                for key in (
                    "online_three_bmm",
                    "deduplicated_lookup_plus_one_bmm",
                    "labelled_lookup_plus_one_bmm",
                )
            ),
        },
        "claim_scope": {
            "empirical": [
                "the recorded latency distributions on this workstation, software stack, dtype, and batch grid",
                "float32 output agreement within the displayed tolerance for the sampled deterministic inputs",
            ],
            "not_claimed": [
                "the timing ranking generalizes to other hardware, software, dtypes, or batch shapes",
                "the compiled path is faster per primitive token in a complete sequence scan",
                "float32 timing validates the exact Q(sqrt(5)) spectral theorem",
                "an end-to-end SSM accuracy or throughput advantage",
            ],
        },
        "passed": parity_passed,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--devices",
        default="cpu,cuda",
        help="comma-separated torch devices; unavailable CUDA is skipped",
    )
    parser.add_argument(
        "--batch-sizes",
        default=",".join(str(value) for value in DEFAULT_BATCH_SIZES),
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
    report = benchmark(devices, batch_sizes)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
