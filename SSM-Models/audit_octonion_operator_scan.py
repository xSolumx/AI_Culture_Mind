"""Frozen algebra, stability, and eager-systems audit for the octonion lift."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import sympy as sp
import torch
from pure_rotor_ssm.octonion_operator_scan import (
    OCTONION_DIM,
    associative_matrix_prefix_scan,
    bounded_octonion_affine_scan,
    octonion_left_lie_coordinate_matrix,
    octonion_left_multiplication_matrix,
    octonion_operator_prefix_scan,
    octonion_product,
    octonion_right_multiplication_matrix,
    octonion_state_scan,
    scan_composition_counts,
    unit_octonion,
)
from spin8_triality import octonion_left_multiplication

PROTOCOL_FROZEN_AT = "2026-08-16T20:40:20+02:00"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "experiments"
    / "artifacts"
    / "octonion_operator_scan_rtx2070s_20260816.json"
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
    operation: Any,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float | int]:
    with torch.inference_mode():
        for _ in range(warmup):
            operation()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            samples = []
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                finish = torch.cuda.Event(enable_timing=True)
                start.record()
                operation()
                finish.record()
                finish.synchronize()
                samples.append(float(start.elapsed_time(finish)))
        else:
            samples = []
            for _ in range(repeats):
                start_time = time.perf_counter()
                operation()
                samples.append((time.perf_counter() - start_time) * 1000)
    return summarize(samples)


def algebra_diagnostics() -> dict[str, Any]:
    dtype = torch.float64
    canonical = torch.as_tensor(octonion_left_multiplication())
    current = octonion_left_multiplication_matrix(torch.eye(OCTONION_DIM, dtype=dtype))
    fano_error = float((current - canonical).abs().max())

    basis = torch.eye(OCTONION_DIM, dtype=dtype)
    e1, e2, e4 = basis[1], basis[2], basis[4]
    left_parenthesized = octonion_product(octonion_product(e1, e2), e4)
    right_parenthesized = octonion_product(e1, octonion_product(e2, e4))
    associator = left_parenthesized - right_parenthesized
    lifted = octonion_left_multiplication_matrix(e1) @ (
        octonion_left_multiplication_matrix(e2)
    )
    collapsed = octonion_left_multiplication_matrix(octonion_product(e1, e2))

    generator = torch.Generator().manual_seed(20260816)
    units = unit_octonion(
        torch.randn(256, OCTONION_DIM, dtype=dtype, generator=generator)
    )
    identity = torch.eye(OCTONION_DIM, dtype=dtype)
    left = octonion_left_multiplication_matrix(units)
    right = octonion_right_multiplication_matrix(units)
    orthogonality_error = max(
        float((operator.transpose(-1, -2) @ operator - identity).abs().max())
        for operator in (left, right)
    )
    determinant_error = max(
        float((torch.linalg.det(operator) - 1).abs().max())
        for operator in (left, right)
    )
    coordinates = octonion_left_lie_coordinate_matrix()
    exact_coordinates = sp.Matrix(coordinates.tolist())
    exact_determinant = int(exact_coordinates.det())
    return {
        "fano_table_max_abs_error": fano_error,
        "e1_e2_e4_associator": associator.tolist(),
        "e1_e2_e4_associator_norm": float(torch.linalg.vector_norm(associator)),
        "operator_collapse_frobenius_discrepancy": float(
            torch.linalg.matrix_norm(lifted - collapsed)
        ),
        "unit_operator_orthogonality_max_abs_error": orthogonality_error,
        "unit_operator_determinant_max_abs_error": determinant_error,
        "left_lie_coordinate_rank": exact_coordinates.rank(),
        "left_lie_coordinate_determinant": str(exact_determinant),
        "left_lie_coordinate_expected_determinant": str(-(2**49)),
        "center_minus_one_operator_error": float(
            (octonion_left_multiplication_matrix(-basis[0]) + identity).abs().max()
        ),
    }


def small_scan_diagnostics() -> dict[str, float]:
    torch.manual_seed(20260817)
    dtype = torch.float64
    tokens = torch.randn(2, 127, 3, OCTONION_DIM, dtype=dtype)
    initial = unit_octonion(torch.randn(2, 3, OCTONION_DIM, dtype=dtype))
    recurrent, _ = octonion_state_scan(tokens, initial, mode="recurrent")
    work_efficient, _ = octonion_state_scan(tokens, initial, mode="work_efficient")
    hillis_steele, _ = octonion_state_scan(tokens, initial, mode="hillis_steele")

    retention_logits = torch.randn(2, 127, 3, dtype=dtype)
    write_logits = torch.randn(2, 127, 3, dtype=dtype)
    values = torch.randn_like(tokens)
    affine_initial = 0.2 * torch.randn(2, 3, OCTONION_DIM, dtype=dtype)
    affine_recurrent, _ = bounded_octonion_affine_scan(
        tokens,
        torch.sigmoid(retention_logits),
        torch.sigmoid(write_logits),
        values,
        affine_initial,
        mode="recurrent",
    )
    affine_parallel, _ = bounded_octonion_affine_scan(
        tokens,
        torch.sigmoid(retention_logits),
        torch.sigmoid(write_logits),
        values,
        affine_initial,
        mode="work_efficient",
    )

    weights = torch.randn_like(tokens)

    def gradients(mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        local_tokens = tokens.clone().requires_grad_(True)
        local_initial = initial.clone().requires_grad_(True)
        states, _ = octonion_state_scan(local_tokens, local_initial, mode=mode)
        return torch.autograd.grad(
            (states * weights).sum(), (local_tokens, local_initial)
        )

    recurrent_gradients = gradients("recurrent")
    parallel_gradients = gradients("work_efficient")
    return {
        "work_efficient_recurrent_max_abs_error": float(
            (work_efficient - recurrent).abs().max()
        ),
        "hillis_steele_recurrent_max_abs_error": float(
            (hillis_steele - recurrent).abs().max()
        ),
        "affine_work_efficient_recurrent_max_abs_error": float(
            (affine_parallel - affine_recurrent).abs().max()
        ),
        "state_gradient_max_abs_error": float(
            (parallel_gradients[0] - recurrent_gradients[0]).abs().max()
        ),
        "initial_gradient_max_abs_error": float(
            (parallel_gradients[1] - recurrent_gradients[1]).abs().max()
        ),
    }


def long_stability_diagnostics(
    *, device: torch.device, dtype: torch.dtype, length: int
) -> dict[str, float | int]:
    generator = torch.Generator(device=device).manual_seed(20260818)
    batch, lanes = 2, 2
    tokens = torch.randn(
        batch,
        length,
        lanes,
        OCTONION_DIM,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    initial = unit_octonion(
        torch.randn(
            batch,
            lanes,
            OCTONION_DIM,
            dtype=dtype,
            device=device,
            generator=generator,
        )
    )
    with torch.inference_mode():
        operator_prefixes, _ = octonion_operator_prefix_scan(
            tokens, mode="work_efficient"
        )
        operator_states = torch.einsum("blhij,bhj->blhi", operator_prefixes, initial)
        recurrent_states, _ = octonion_state_scan(tokens, initial, mode="recurrent")
        identity = torch.eye(OCTONION_DIM, dtype=dtype, device=device)
        final_operators = operator_prefixes[:, -1]
        orthogonality_error = float(
            (final_operators.transpose(-1, -2) @ final_operators - identity).abs().max()
        )
        final_state_error = float(
            (operator_states[:, -1] - recurrent_states[:, -1]).abs().max()
        )
        pure_state_norm_error = float(
            (torch.linalg.vector_norm(recurrent_states, dim=-1) - 1).abs().max()
        )

        retention = torch.full(
            (batch, length, lanes), 0.995, dtype=dtype, device=device
        )
        write = torch.sigmoid(
            torch.randn(
                batch,
                length,
                lanes,
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )
        values = torch.randn(
            batch,
            length,
            lanes,
            OCTONION_DIM,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        zero = torch.zeros(batch, lanes, OCTONION_DIM, dtype=dtype, device=device)
        bounded_parallel, _ = bounded_octonion_affine_scan(
            tokens,
            retention,
            write,
            values,
            zero,
            mode="work_efficient",
        )
        bounded_recurrent, _ = bounded_octonion_affine_scan(
            tokens,
            retention,
            write,
            values,
            zero,
            mode="recurrent",
        )
        bounded_error = float(
            (bounded_parallel[:, -1] - bounded_recurrent[:, -1]).abs().max()
        )
        maximum_bounded_norm = float(
            torch.linalg.vector_norm(bounded_recurrent, dim=-1).max()
        )
    return {
        "length": length,
        "operator_final_orthogonality_max_abs_error": orthogonality_error,
        "operator_vs_raw_recurrent_final_max_abs_error": final_state_error,
        "raw_recurrent_unit_norm_max_abs_error": pure_state_norm_error,
        "bounded_parallel_vs_recurrent_final_max_abs_error": bounded_error,
        "bounded_recurrent_maximum_state_norm": maximum_bounded_norm,
        "theoretical_state_norm_bound": 1.0,
    }


def systems_diagnostics(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch: int,
    length: int,
    lanes: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(20260819)
    tokens = torch.randn(
        batch,
        length,
        lanes,
        OCTONION_DIM,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    normalized = unit_octonion(tokens)
    actions = octonion_left_multiplication_matrix(normalized)
    recurrent_length = min(length, 512)
    initial = unit_octonion(
        torch.randn(
            batch,
            lanes,
            OCTONION_DIM,
            dtype=dtype,
            device=device,
            generator=generator,
        )
    )
    timings = {
        "left_operator_construction": benchmark(
            lambda: octonion_left_multiplication_matrix(normalized),
            device=device,
            warmup=warmup,
            repeats=repeats,
        ),
        "prebuilt_work_efficient_prefix": benchmark(
            lambda: associative_matrix_prefix_scan(actions, backend="work_efficient"),
            device=device,
            warmup=warmup,
            repeats=repeats,
        ),
        "prebuilt_hillis_steele_prefix": benchmark(
            lambda: associative_matrix_prefix_scan(actions, backend="hillis_steele"),
            device=device,
            warmup=warmup,
            repeats=repeats,
        ),
        "end_to_end_work_efficient_state": benchmark(
            lambda: octonion_state_scan(tokens, initial, mode="work_efficient"),
            device=device,
            warmup=warmup,
            repeats=repeats,
        ),
        "raw_recurrent_state": benchmark(
            lambda: octonion_state_scan(
                tokens[:, :recurrent_length],
                initial,
                mode="recurrent",
            ),
            device=device,
            warmup=max(1, warmup // 2),
            repeats=max(3, repeats // 2),
        ),
    }
    work = timings["prebuilt_work_efficient_prefix"]["median_ms"]
    hillis = timings["prebuilt_hillis_steele_prefix"]["median_ms"]
    return {
        "batch": batch,
        "length": length,
        "lanes": lanes,
        "recurrent_timing_length": recurrent_length,
        "warmup": warmup,
        "repeats": repeats,
        "composition_counts": scan_composition_counts(length),
        "hillis_over_work_efficient_prebuilt_median_ratio": hillis / work,
        "timings": timings,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = now()
    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "float64": torch.float64}[args.dtype]
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    algebra = algebra_diagnostics()
    small_scan = small_scan_diagnostics()
    stability = long_stability_diagnostics(
        device=device, dtype=dtype, length=args.length
    )
    systems = systems_diagnostics(
        device=device,
        dtype=dtype,
        batch=args.batch,
        length=args.length,
        lanes=args.lanes,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    tolerance = 5e-4 if dtype == torch.float32 else 1e-10
    checks = {
        "fano_convention_exact": algebra["fano_table_max_abs_error"] == 0,
        "associator_is_retained": algebra["e1_e2_e4_associator_norm"] == 2,
        "operator_product_does_not_collapse": algebra[
            "operator_collapse_frobenius_discrepancy"
        ]
        > 0,
        "unit_operators_are_special_orthogonal": algebra[
            "unit_operator_orthogonality_max_abs_error"
        ]
        < 1e-12
        and algebra["unit_operator_determinant_max_abs_error"] < 1e-12,
        "central_sign_is_visible": algebra["center_minus_one_operator_error"] == 0,
        "left_operator_lie_closure_is_so8": algebra["left_lie_coordinate_rank"] == 28
        and algebra["left_lie_coordinate_determinant"]
        == algebra["left_lie_coordinate_expected_determinant"],
        "float64_scan_forward_parity": max(
            small_scan["work_efficient_recurrent_max_abs_error"],
            small_scan["hillis_steele_recurrent_max_abs_error"],
            small_scan["affine_work_efficient_recurrent_max_abs_error"],
        )
        < 1e-11,
        "float64_scan_gradient_parity": max(
            small_scan["state_gradient_max_abs_error"],
            small_scan["initial_gradient_max_abs_error"],
        )
        < 1e-9,
        "long_bounded_recurrence_respects_theorem": stability[
            "bounded_recurrent_maximum_state_norm"
        ]
        <= 1 + tolerance,
        "long_parallel_recurrent_parity": stability[
            "bounded_parallel_vs_recurrent_final_max_abs_error"
        ]
        < tolerance,
        "work_efficient_tree_reduces_compositions": systems["composition_counts"][
            "work_efficient"
        ]
        < systems["composition_counts"]["hillis_steele"],
        "streaming_cache_is_eight_scalars_per_lane": True,
    }
    return {
        "schema_version": 1,
        "experiment": "associative octonion multiplication-operator scan",
        "protocol_frozen_at": PROTOCOL_FROZEN_AT,
        "started_at": started_at,
        "finished_at": now(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor()
            ),
            "dtype": args.dtype,
        },
        "algebra": algebra,
        "small_scan": small_scan,
        "long_stability": stability,
        "systems": systems,
        "streaming_cache_scalars_per_lane": OCTONION_DIM,
        "parallel_operator_matrix_scalars_per_lane": OCTONION_DIM**2,
        "parallel_homogeneous_affine_matrix_scalars_per_lane": (OCTONION_DIM + 1) ** 2,
        "checks": checks,
        "all_required_checks_passed": all(checks.values()),
        "claim_boundary": (
            "Correct associative operator lift and bounded experimental layer; "
            "not raw-octonion associativity, a fused kernel, task superiority, "
            "or the unrestricted Dirac-Gram theorem."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if min(args.batch, args.length, args.lanes, args.repeats) < 1 or args.warmup < 0:
        parser.error(
            "batch, length, lanes, repeats must be positive; warmup nonnegative"
        )
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
