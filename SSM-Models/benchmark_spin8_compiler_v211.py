"""Hardware audit for isotypic-to-silicon compiler v2.1.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import triton
from pure_spin8_ssm.compiler import COMPILER_VERSION
from pure_spin8_ssm.continuous_scan import (
    _continuous_forward_kernel,
    _continuous_tensor_core_kernel,
    continuous_ptx_evidence,
    eager_continuous_spin8_scan,
    triton_scalar_continuous_spin8_scan,
)
from pure_spin8_ssm.self_calibrating_ssm import SelfCalibratingSpin8SSMLayer
from pure_spin8_ssm.self_calibration import spin8_actions_from_seven_probes
from pure_spin8_ssm.torch_backend import spin8_factorized_actions
from spin8_triality import TRIALITY_REPRESENTATIONS, torch_triality_generators

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "artifacts"
    / "spin8_compiler_v211_rtx2070s_20260821.json"
)
GRID = (
    (1, 128, 1),
    (1, 128, 8),
    (1, 128, 16),
    (1, 128, 32),
    (1, 128, 64),
    (8, 128, 16),
    (32, 128, 16),
    (8, 1024, 16),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timed(call: Callable[[], Any]) -> tuple[float, Any]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1_000, output


def _paired(
    first: Callable[[], Any], second: Callable[[], Any], repeats: int
) -> tuple[dict[str, float], Any, Any]:
    first_output = first()
    second_output = second()
    for _ in range(10):
        first_output = first()
        second_output = second()
    first_samples = []
    second_samples = []
    for index in range(repeats):
        ordered = (
            (("first", first), ("second", second))
            if index % 2 == 0
            else (("second", second), ("first", first))
        )
        for name, call in ordered:
            elapsed, output = _timed(call)
            if name == "first":
                first_samples.append(elapsed)
                first_output = output
            else:
                second_samples.append(elapsed)
                second_output = output
    first_median = statistics.median(first_samples)
    second_median = statistics.median(second_samples)
    return (
        {
            "scalar_median_microseconds": first_median,
            "tensor_core_median_microseconds": second_median,
            "tensor_core_speedup_vs_scalar": first_median / second_median,
            "scalar_p10_microseconds": sorted(first_samples)[repeats // 10],
            "scalar_p90_microseconds": sorted(first_samples)[9 * repeats // 10],
            "tensor_core_p10_microseconds": sorted(second_samples)[repeats // 10],
            "tensor_core_p90_microseconds": sorted(second_samples)[9 * repeats // 10],
        },
        first_output,
        second_output,
    )


def _triality_actions(batch: int, length: int, *, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(
        20_260_821 + 10_000 * batch + length
    )
    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    coordinates = 0.03 * torch.randn(
        batch * length,
        28,
        generator=generator,
        device="cuda",
        dtype=compute_dtype,
    )
    actions = spin8_factorized_actions(
        coordinates,
        torch_triality_generators(dtype=compute_dtype, device="cuda"),
        TRIALITY_REPRESENTATIONS,
    ).reshape(batch, length, 3, 8, 8)
    return actions.to(dtype=dtype)


def _inference_row(batch: int, length: int, channels: int) -> dict[str, Any]:
    generator = torch.Generator(device="cuda").manual_seed(
        20_260_825 + 100_000 * batch + 100 * length + channels
    )
    action = _triality_actions(batch, length, dtype=torch.float16)
    scale = torch.full(
        (batch, length, channels), 0.98, dtype=torch.float16, device="cuda"
    )
    drive = 0.002 * torch.randn(
        batch,
        length,
        channels,
        3,
        8,
        generator=generator,
        dtype=torch.float16,
        device="cuda",
    )
    initial = torch.randn(
        batch,
        channels,
        3,
        8,
        generator=generator,
        dtype=torch.float16,
        device="cuda",
    )
    scalar_buffer = torch.empty_like(drive)
    tensor_buffer = torch.empty_like(drive)

    def scalar_call() -> torch.Tensor:
        _continuous_forward_kernel[(batch, channels, 3)](
            action,
            scale,
            drive,
            initial,
            scalar_buffer,
            length,
            channels,
            3,
            True,
            num_warps=1,
        )
        return scalar_buffer

    def tensor_call() -> torch.Tensor:
        _continuous_tensor_core_kernel[(batch, 3, triton.cdiv(channels, 16))](
            action,
            scale,
            drive,
            initial,
            tensor_buffer,
            length,
            channels,
            3,
            BLOCK_C=16,
            num_warps=4,
        )
        return tensor_buffer
    repeats = 100 if batch * length <= 1024 else 50
    with torch.inference_mode():
        timings, scalar, tensor = _paired(scalar_call, tensor_call, repeats)
        reference = triton_scalar_continuous_spin8_scan(
            action.float(), scale.float(), drive.float(), initial.float()
        )
    return {
        "batch_size": batch,
        "sequence_length": length,
        "isotypic_multiplicity": channels,
        **timings,
        "scalar_vs_float32_max_abs": float((scalar.float() - reference).abs().max()),
        "tensor_core_vs_float32_max_abs": float(
            (tensor.float() - reference).abs().max()
        ),
        "scalar_vs_tensor_core_max_abs": float((scalar - tensor).abs().max()),
        "scalar_ptx": continuous_ptx_evidence(
            action, scale, drive, initial, backend="triton_scalar"
        ),
        "tensor_core_ptx": continuous_ptx_evidence(
            action, scale, drive, initial, backend="triton_tensor_core"
        ),
    }


def _training_audit() -> dict[str, Any]:
    torch.manual_seed(20_260_826)
    batch, length, channels = 4, 128, 4
    action_base = _triality_actions(batch, length, dtype=torch.float32)
    scale_base = torch.full(
        (batch, length, channels), 0.97, dtype=torch.float32, device="cuda"
    )
    drive_base = 0.002 * torch.randn(
        batch, length, channels, 3, 8, device="cuda"
    )
    initial_base = torch.randn(batch, channels, 3, 8, device="cuda")

    def make_inputs() -> tuple[torch.Tensor, ...]:
        return tuple(
            tensor.detach().clone().requires_grad_()
            for tensor in (action_base, scale_base, drive_base, initial_base)
        )

    eager_inputs = make_inputs()
    compiled_inputs = make_inputs()
    gradient = torch.randn(batch, length, channels, 3, 8, device="cuda")
    eager_output = eager_continuous_spin8_scan(*eager_inputs)
    compiled_output = triton_scalar_continuous_spin8_scan(*compiled_inputs)
    eager_gradients = torch.autograd.grad(eager_output, eager_inputs, gradient)
    compiled_gradients = torch.autograd.grad(
        compiled_output, compiled_inputs, gradient
    )

    def eager_step() -> torch.Tensor:
        inputs = make_inputs()
        output = eager_continuous_spin8_scan(*inputs)
        torch.autograd.grad(output.square().mean(), inputs)
        return output

    def compiled_step() -> torch.Tensor:
        inputs = make_inputs()
        output = triton_scalar_continuous_spin8_scan(*inputs)
        torch.autograd.grad(output.square().mean(), inputs)
        return output

    timings, _, _ = _paired(eager_step, compiled_step, repeats=20)
    eager_time = timings["scalar_median_microseconds"]
    compiled_time = timings["tensor_core_median_microseconds"]
    return {
        "shape": {
            "batch_size": batch,
            "sequence_length": length,
            "isotypic_multiplicity": channels,
            "representations": 3,
        },
        "eager_forward_backward_median_microseconds": eager_time,
        "compiled_forward_backward_median_microseconds": compiled_time,
        "compiled_speedup_vs_eager": eager_time / compiled_time,
        "forward_max_abs": float((compiled_output - eager_output).abs().max()),
        "gradient_max_abs": [
            float((actual - expected).abs().max())
            for actual, expected in zip(compiled_gradients, eager_gradients)
        ],
    }


def _calibration_audit() -> dict[str, Any]:
    batch, length = 2, 7
    true_actions = _triality_actions(batch, length, dtype=torch.float64)
    probes = true_actions[..., 0, :, :7].transpose(-1, -2)
    generators = torch_triality_generators(dtype=torch.float64, device="cuda")
    canonical, _ = spin8_actions_from_seven_probes(
        probes, generators, project=False
    )
    plus = (canonical[..., 1, :, :] - true_actions[..., 1, :, :]).abs().amax(
        dim=(-1, -2)
    )
    minus = (canonical[..., 1, :, :] + true_actions[..., 1, :, :]).abs().amax(
        dim=(-1, -2)
    )
    lift_sign = torch.where(plus <= minus, 1.0, -1.0)
    recovered, triangular = spin8_actions_from_seven_probes(
        probes,
        generators,
        lift_sign=lift_sign,
        project=False,
    )
    return {
        "batch_size": batch,
        "sequence_length": length,
        "action_max_abs": float((recovered - true_actions).abs().max()),
        "triangular_identity_max_abs": float(
            (triangular - torch.eye(8, dtype=torch.float64, device="cuda"))
            .abs()
            .max()
        ),
        "lift_bits_positive": int((lift_sign > 0).sum()),
        "lift_bits_negative": int((lift_sign < 0).sum()),
    }


def _self_calibrating_runtime_audit() -> dict[str, Any]:
    batch, length, channels = 4, 16, 4
    true_actions = _triality_actions(batch, length, dtype=torch.float32)
    probes = true_actions[..., 0, :, :7].transpose(-1, -2).contiguous()
    generators = torch_triality_generators(device="cuda")
    canonical, _ = spin8_actions_from_seven_probes(
        probes, generators, project=False
    )
    plus = (canonical[..., 1, :, :] - true_actions[..., 1, :, :]).abs().amax(
        dim=(-1, -2)
    )
    minus = (canonical[..., 1, :, :] + true_actions[..., 1, :, :]).abs().amax(
        dim=(-1, -2)
    )
    lift_sign = torch.where(plus <= minus, 1.0, -1.0)
    layer = SelfCalibratingSpin8SSMLayer(
        channels=channels, projection="none", hardware_profile=None
    ).cuda().eval()

    with torch.inference_mode():
        timings, eager, compiled = _paired(
            lambda: layer(probes, lift_sign, backend="eager")[0],
            lambda: layer(probes, lift_sign, backend="triton_scalar")[0],
            repeats=30,
        )
    eager_time = timings["scalar_median_microseconds"]
    compiled_time = timings["tensor_core_median_microseconds"]

    gradient_probes = probes[:1, :4].detach().clone().requires_grad_()
    gradient_sign = lift_sign[:1, :4]
    gradient_output, _ = layer(
        gradient_probes, gradient_sign, backend="triton_scalar"
    )
    probe_gradient, initial_gradient = torch.autograd.grad(
        gradient_output.square().mean(),
        (gradient_probes, layer.initial_state),
    )
    return {
        "shape": {
            "batch_size": batch,
            "sequence_length": length,
            "isotypic_multiplicity": channels,
        },
        "eager_end_to_end_median_microseconds": eager_time,
        "compiled_end_to_end_median_microseconds": compiled_time,
        "compiled_speedup_vs_eager": eager_time / compiled_time,
        "forward_max_abs": float((compiled - eager).abs().max()),
        "probe_gradient_finite": bool(torch.isfinite(probe_gradient).all()),
        "initial_gradient_finite": bool(torch.isfinite(initial_gradient).all()),
        "probe_gradient_norm": float(probe_gradient.norm()),
    }


def benchmark() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rows = [_inference_row(*shape) for shape in GRID]
    training = _training_audit()
    calibration = _calibration_audit()
    self_calibrating_runtime = _self_calibrating_runtime_audit()
    checks = {
        "all_metrics_finite": all(
            math.isfinite(float(value))
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and key != "isotypic_multiplicity"
        ),
        "tensor_ptx_contains_mma_sync": all(
            row["tensor_core_ptx"]["mma_sync_occurrences"] > 0 for row in rows
        ),
        "scalar_ptx_contains_no_mma_sync": all(
            row["scalar_ptx"]["mma_sync_occurrences"] == 0 for row in rows
        ),
        "inference_max_abs_below_0p1": all(
            row["tensor_core_vs_float32_max_abs"] < 0.1 for row in rows
        ),
        "training_forward_max_abs_below_5e_5": training["forward_max_abs"] < 5e-5,
        "training_gradient_max_abs_below_1e_3": max(training["gradient_max_abs"])
        < 1e-3,
        "seven_probe_lift_max_abs_below_1e_12": calibration["action_max_abs"]
        < 1e-12,
        "self_calibrating_forward_max_abs_below_5e_5": self_calibrating_runtime[
            "forward_max_abs"
        ]
        < 5e-5,
        "self_calibrating_gradients_finite": self_calibrating_runtime[
            "probe_gradient_finite"
        ]
        and self_calibrating_runtime["initial_gradient_finite"],
    }
    properties = torch.cuda.get_device_properties(0)
    return {
        "schema_version": 1,
        "experiment": "isotypic-to-silicon compiler v2.1.1 hardware audit",
        "recorded_at": datetime.now().astimezone().isoformat(),
        "compiler_version": COMPILER_VERSION,
        "hardware": {
            "gpu": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "precision_contract": {
            "inference": "FP16 recurrent state with FP32 dot accumulation",
            "training": "FP32 full gradients for action, scale, drive, initial",
            "reference": "eager PyTorch for training and FP32 scalar Triton for long inference",
        },
        "rows": rows,
        "training": training,
        "calibration": calibration,
        "self_calibrating_runtime": self_calibrating_runtime,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
