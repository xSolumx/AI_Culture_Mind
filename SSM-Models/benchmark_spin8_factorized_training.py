"""End-to-end training audit for the factorized Spin(8) compiler path."""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import statistics
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from pure_spin8_ssm.compiler import COMPILER_VERSION
from pure_spin8_ssm.torch_backend import PureSpin8SSMLayer, ScanMode

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "artifacts"
    / "spin8_factorized_training_rtx2070s_20260821.json"
)
GRID = (
    (4, 128, 4, 16),
    (8, 128, 8, 16),
    (4, 512, 4, 16),
)


def _active_parameters(
    layer: PureSpin8SSMLayer,
) -> tuple[torch.nn.Parameter, ...]:
    return tuple(
        parameter
        for name, parameter in layer.named_parameters()
        if name != "coupling_logits"
    )


def _timed(call: Callable[[], Any]) -> tuple[float, Any]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1_000.0, output


def _interleaved_samples(
    calls: dict[str, Callable[[], Any]], repeats: int
) -> dict[str, list[float]]:
    for call in calls.values():
        for _ in range(3):
            call()
    names = tuple(calls)
    samples = {name: [] for name in names}
    for repeat in range(repeats):
        shift = repeat % len(names)
        order = names[shift:] + names[:shift]
        for name in order:
            elapsed, _ = _timed(calls[name])
            samples[name].append(elapsed)
    return samples


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_microseconds": statistics.median(samples),
        "p10_microseconds": ordered[len(ordered) // 10],
        "p90_microseconds": ordered[9 * len(ordered) // 10],
    }


def _row(
    batch: int,
    length: int,
    channels: int,
    input_size: int,
) -> dict[str, Any]:
    torch.manual_seed(20_260_829 + batch * 1000 + length + channels)
    base = PureSpin8SSMLayer(
        input_size,
        channels=channels,
        action_mode="factorized",
        triality_coupling=False,
    ).cuda()
    with torch.no_grad():
        torch.nn.init.normal_(base.coefficient_controller.weight, std=0.02)
        torch.nn.init.normal_(base.coefficient_controller.bias, std=0.01)
    layers = {
        "eager_materialized_recurrent": base,
        "materialized_action_compiled_scan": copy.deepcopy(base),
        "direct_factor_compiled_scan": copy.deepcopy(base),
        "fused_controller_factor_scan": copy.deepcopy(base),
    }
    modes: dict[str, ScanMode] = {
        "eager_materialized_recurrent": "recurrent",
        "materialized_action_compiled_scan": "compiled_recurrent",
        "direct_factor_compiled_scan": "compiled_factorized",
        "fused_controller_factor_scan": "compiled_controller",
    }
    inputs = {
        name: torch.randn(
            batch,
            length,
            input_size,
            device="cuda",
            generator=torch.Generator(device="cuda").manual_seed(20_260_830),
            requires_grad=True,
        )
        for name in layers
    }

    def training_call(name: str) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        layer = layers[name]
        output, _ = layer(
            inputs[name],
            scan_mode=modes[name],
            return_raw_states=True,
        )
        differentiable = (inputs[name], *_active_parameters(layer))
        gradients = torch.autograd.grad(output.square().mean(), differentiable)
        return output, gradients

    reference_name = "eager_materialized_recurrent"
    reference_layer = layers[reference_name]
    reference_output, _ = reference_layer(
        inputs[reference_name],
        scan_mode=modes[reference_name],
        return_raw_states=True,
    )
    upstream = torch.randn_like(reference_output)
    reference_gradients = torch.autograd.grad(
        reference_output,
        (inputs[reference_name], *_active_parameters(reference_layer)),
        upstream,
    )
    parity: dict[str, dict[str, Any]] = {}
    for name in (
        "materialized_action_compiled_scan",
        "direct_factor_compiled_scan",
        "fused_controller_factor_scan",
    ):
        layer = layers[name]
        output, _ = layer(
            inputs[name], scan_mode=modes[name], return_raw_states=True
        )
        gradients = torch.autograd.grad(
            output,
            (inputs[name], *_active_parameters(layer)),
            upstream,
        )
        parity[name] = {
            "forward_max_abs": float((output - reference_output).abs().max()),
            "gradient_max_abs": [
                float((actual - expected).abs().max())
                for actual, expected in zip(gradients, reference_gradients)
            ],
            "coefficient_weight_gradient_norm": float(gradients[3].norm()),
            "coefficient_bias_gradient_norm": float(gradients[4].norm()),
        }

    calls = {name: lambda name=name: training_call(name) for name in layers}
    repeats = 10 if length <= 128 else 5
    samples = _interleaved_samples(calls, repeats)
    timings = {name: _summary(row) for name, row in samples.items()}
    peak_deltas = {}
    for name, call in calls.items():
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        call()
        torch.cuda.synchronize()
        peak_deltas[name] = torch.cuda.max_memory_allocated() - baseline

    eager_median = timings[reference_name]["median_microseconds"]
    materialized_median = timings["materialized_action_compiled_scan"][
        "median_microseconds"
    ]
    direct_median = timings["direct_factor_compiled_scan"]["median_microseconds"]
    fused_median = timings["fused_controller_factor_scan"]["median_microseconds"]
    return {
        "shape": {
            "batch_size": batch,
            "sequence_length": length,
            "channels": channels,
            "input_size": input_size,
            "representations": 3,
        },
        "timings": timings,
        "peak_allocation_delta_bytes": peak_deltas,
        "speedups": {
            "direct_vs_eager": eager_median / direct_median,
            "direct_vs_materialized_compiled": materialized_median / direct_median,
            "materialized_compiled_vs_eager": eager_median / materialized_median,
            "fused_controller_vs_eager": eager_median / fused_median,
            "fused_controller_vs_materialized_compiled": (
                materialized_median / fused_median
            ),
            "fused_controller_vs_staged_direct": direct_median / fused_median,
        },
        "parity": parity,
    }


def benchmark() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rows = [_row(*shape) for shape in GRID]
    checks = {
        "all_metrics_finite": all(
            math.isfinite(value)
            for row in rows
            for section in row["parity"].values()
            for value in (
                section["forward_max_abs"],
                *section["gradient_max_abs"],
                section["coefficient_weight_gradient_norm"],
                section["coefficient_bias_gradient_norm"],
            )
        ),
        "direct_forward_max_abs_below_5e_5": all(
            row["parity"]["direct_factor_compiled_scan"]["forward_max_abs"]
            < 5e-5
            for row in rows
        ),
        "direct_gradient_max_abs_below_2e_3": all(
            max(
                row["parity"]["direct_factor_compiled_scan"][
                    "gradient_max_abs"
                ]
            )
            < 2e-3
            for row in rows
        ),
        "controller_gradients_nonzero": all(
            row["parity"]["fused_controller_factor_scan"][
                "coefficient_weight_gradient_norm"
            ]
            > 0.0
            and row["parity"]["fused_controller_factor_scan"][
                "coefficient_bias_gradient_norm"
            ]
            > 0.0
            for row in rows
        ),
        "fused_controller_forward_max_abs_below_5e_5": all(
            row["parity"]["fused_controller_factor_scan"]["forward_max_abs"]
            < 5e-5
            for row in rows
        ),
        "fused_controller_gradient_max_abs_below_2e_3": all(
            max(
                row["parity"]["fused_controller_factor_scan"][
                    "gradient_max_abs"
                ]
            )
            < 2e-3
            for row in rows
        ),
    }
    properties = torch.cuda.get_device_properties(0)
    return {
        "schema_version": 1,
        "experiment": "trainable direct-factor Spin(8) compiler audit",
        "compiler_version": COMPILER_VERSION,
        "recorded_at": datetime.now().astimezone().isoformat(),
        "hardware": {
            "gpu": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "precision_contract": "FP32 forward and full backward",
        "rows": rows,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    report = benchmark()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
