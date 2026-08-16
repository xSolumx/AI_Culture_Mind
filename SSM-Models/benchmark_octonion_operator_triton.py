"""Frozen WSL/Triton benchmark for the fused octonion recurrence."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from pure_rotor_ssm.octonion_operator_scan import octonion_state_scan, unit_octonion
from pure_rotor_ssm.octonion_operator_triton import (
    fused_octonion_state_scan,
    triton_is_available,
)

try:
    import triton
except ImportError:  # pragma: no cover - native Windows provenance path
    triton = None


PROTOCOL_FROZEN_AT = "2026-08-16T20:56:25+02:00"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "experiments"
    / "artifacts"
    / "octonion_triton_scan_wsl_rtx2070s_20260816.json"
)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def summarize(samples: list[float]) -> dict[str, float | int]:
    ordered = sorted(samples)
    return {
        "repeats": len(samples),
        "minimum_ms": ordered[0],
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.mean(ordered),
        "standard_deviation_ms": statistics.pstdev(ordered),
        "p20_ms": ordered[int(0.2 * (len(ordered) - 1))],
        "p80_ms": ordered[int(0.8 * (len(ordered) - 1))],
    }


def benchmark(
    operation: Callable[[], Any], *, warmup: int, repeats: int
) -> dict[str, float | int]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        finish = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        finish.record()
        finish.synchronize()
        samples.append(float(start.elapsed_time(finish)))
    return summarize(samples)


def correctness(seed: int) -> dict[str, float | bool]:
    torch.manual_seed(seed)
    tokens = torch.randn(3, 127, 4, 8, device="cuda")
    initial = unit_octonion(torch.randn(3, 4, 8, device="cuda"))
    expected, _ = octonion_state_scan(tokens, initial, mode="recurrent")
    actual, _ = fused_octonion_state_scan(tokens, initial)
    forward_error = float((actual - expected).abs().max())

    gradient_tokens = torch.randn(2, 31, 3, 8, device="cuda")
    gradient_initial = unit_octonion(torch.randn(2, 3, 8, device="cuda"))
    cotangent = torch.randn_like(gradient_tokens)

    def gradients(fused: bool) -> tuple[torch.Tensor, torch.Tensor]:
        local_tokens = gradient_tokens.clone().requires_grad_(True)
        local_initial = gradient_initial.clone().requires_grad_(True)
        if fused:
            states, _ = fused_octonion_state_scan(local_tokens, local_initial)
        else:
            states, _ = octonion_state_scan(
                local_tokens, local_initial, mode="recurrent"
            )
        return torch.autograd.grad(
            (states * cotangent).sum(), (local_tokens, local_initial)
        )

    expected_gradients = gradients(False)
    actual_gradients = gradients(True)
    token_gradient_error = float(
        (actual_gradients[0] - expected_gradients[0]).abs().max()
    )
    initial_gradient_error = float(
        (actual_gradients[1] - expected_gradients[1]).abs().max()
    )

    long_tokens = torch.randn(8, 4096, 4, 8, device="cuda")
    long_states, _ = fused_octonion_state_scan(long_tokens)
    unit_norm_error = float(
        (torch.linalg.vector_norm(long_states, dim=-1) - 1).abs().max()
    )
    return {
        "length127_forward_max_abs_error": forward_error,
        "length31_token_gradient_max_abs_error": token_gradient_error,
        "length31_initial_gradient_max_abs_error": initial_gradient_error,
        "length4096_unit_norm_max_abs_error": unit_norm_error,
        "all_outputs_and_gradients_finite": bool(
            torch.isfinite(actual).all()
            and torch.isfinite(actual_gradients[0]).all()
            and torch.isfinite(actual_gradients[1]).all()
            and torch.isfinite(long_states).all()
        ),
    }


def forward_operation(
    backend: str, tokens: torch.Tensor, initial: torch.Tensor
) -> Callable[[], Any]:
    if backend == "triton_fused_recurrent":
        return lambda: fused_octonion_state_scan(tokens, initial)
    if backend == "pytorch_raw_recurrent":
        return lambda: octonion_state_scan(tokens, initial, mode="recurrent")
    if backend == "pytorch_work_efficient_operator":
        return lambda: octonion_state_scan(tokens, initial, mode="work_efficient")
    raise ValueError(f"unknown backend {backend}")


def backward_operation(
    backend: str,
    base_tokens: torch.Tensor,
    base_initial: torch.Tensor,
    cotangent: torch.Tensor,
) -> Callable[[], Any]:
    def operation() -> tuple[torch.Tensor, torch.Tensor]:
        tokens = base_tokens.detach().clone().requires_grad_(True)
        initial = base_initial.detach().clone().requires_grad_(True)
        states, _ = forward_operation(backend, tokens, initial)()
        return torch.autograd.grad((states * cotangent).sum(), (tokens, initial))

    return operation


def timed_sweep(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed + 1)
    forward: dict[str, dict[str, dict[str, float | int]]] = {}
    backward: dict[str, dict[str, dict[str, float | int]]] = {}
    forward_backends = (
        "triton_fused_recurrent",
        "pytorch_work_efficient_operator",
    )
    backward_backends = forward_backends
    for length in args.lengths:
        tokens = torch.randn(args.batch, length, args.lanes, 8, device="cuda")
        initial = unit_octonion(torch.randn(args.batch, args.lanes, 8, device="cuda"))
        length_forward: dict[str, dict[str, float | int]] = {}
        for backend in forward_backends:
            length_forward[backend] = benchmark(
                forward_operation(backend, tokens, initial),
                warmup=args.warmup,
                repeats=args.repeats,
            )
        if length <= args.raw_max_length:
            length_forward["pytorch_raw_recurrent"] = benchmark(
                forward_operation("pytorch_raw_recurrent", tokens, initial),
                warmup=max(1, args.warmup // 2),
                repeats=max(5, args.repeats // 2),
            )
        forward[str(length)] = length_forward

        if length <= args.backward_max_length:
            cotangent = torch.randn_like(tokens)
            length_backward: dict[str, dict[str, float | int]] = {}
            for backend in backward_backends:
                length_backward[backend] = benchmark(
                    backward_operation(backend, tokens, initial, cotangent),
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
            if length <= args.raw_max_length:
                length_backward["pytorch_raw_recurrent"] = benchmark(
                    backward_operation(
                        "pytorch_raw_recurrent", tokens, initial, cotangent
                    ),
                    warmup=max(1, args.warmup // 2),
                    repeats=max(5, args.repeats // 2),
                )
            backward[str(length)] = length_backward

    ratios = {}
    for length, rows in forward.items():
        fused = rows["triton_fused_recurrent"]["median_ms"]
        ratios[length] = {
            "work_efficient_over_triton_forward": rows[
                "pytorch_work_efficient_operator"
            ]["median_ms"]
            / fused
        }
        if "pytorch_raw_recurrent" in rows:
            ratios[length]["raw_recurrent_over_triton_forward"] = (
                rows["pytorch_raw_recurrent"]["median_ms"] / fused
            )
    return {"forward": forward, "forward_backward": backward, "speedup_ratios": ratios}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not triton_is_available():
        raise RuntimeError("this benchmark requires WSL/Linux Triton with CUDA")
    started_at = now()
    correctness_metrics = correctness(args.seed)
    checks = {
        "forward_parity": correctness_metrics["length127_forward_max_abs_error"] < 2e-5,
        "token_gradient_parity": correctness_metrics[
            "length31_token_gradient_max_abs_error"
        ]
        < 2e-4,
        "initial_gradient_parity": correctness_metrics[
            "length31_initial_gradient_max_abs_error"
        ]
        < 2e-4,
        "long_unit_norm_stability": correctness_metrics[
            "length4096_unit_norm_max_abs_error"
        ]
        < 2e-4,
        "finite_outputs_and_gradients": correctness_metrics[
            "all_outputs_and_gradients_finite"
        ],
    }
    timings = timed_sweep(args)
    return {
        "schema_version": 1,
        "experiment": "fused octonion recurrence WSL Triton benchmark",
        "protocol_frozen_at": PROTOCOL_FROZEN_AT,
        "started_at": started_at,
        "finished_at": now(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "triton": triton.__version__ if triton is not None else None,
            "device_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
        "configuration": {
            "seed": args.seed,
            "dtype": "float32",
            "batch": args.batch,
            "lanes": args.lanes,
            "lengths": args.lengths,
            "raw_max_length": args.raw_max_length,
            "backward_max_length": args.backward_max_length,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "correctness": correctness_metrics,
        "checks": checks,
        "all_required_checks_passed": all(checks.values()),
        "timings": timings,
        "claim_boundary": (
            "Fused differentiable chunk recurrence in the recorded WSL/Triton "
            "environment; not a production SSM or fused associative matrix scan."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[128, 512, 1024, 4096]
    )
    parser.add_argument("--raw-max-length", type=int, default=512)
    parser.add_argument("--backward-max-length", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if (
        min(args.batch, args.lanes, args.raw_max_length, args.backward_max_length) < 1
        or min(args.lengths) < 1
        or args.repeats < 1
        or args.warmup < 0
    ):
        parser.error("shape/repeat arguments must be positive and warmup nonnegative")
    return args


def main() -> None:
    args = parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_required_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
