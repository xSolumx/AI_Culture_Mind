"""Falsify or validate Tensor-Core dispatch for repeated Spin(8) blocks.

An isotypic component ``V tensor R^m`` carries a common irreducible action on
each of its ``m`` copies.  In aligned coordinates this is a matrix-matrix
recurrence: the multiplicity axis supplies the tile width absent from a lone
eight-dimensional matrix-vector recurrence.  This hardware-only benchmark
compares a scalar Triton realization with an FP16 ``tl.dot`` realization and
records PTX evidence rather than inferring Tensor-Core use from dtype.

The benchmark is deliberately separate from the maintained float32 inference
path.  It does not train actions and does not claim that FP16 is acceptable
until the reported long-horizon error gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - hardware-only boundary
    triton = None
    tl = None


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "pure_spin8_compiled_token_scan"
    / "compiled_latent_pure_spin8_seed1.pt"
)
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "artifacts"
    / "spin8_isotypic_tensor_core_rtx2070s_20260821.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode())
    digest.update(str(value.dtype).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def isotypic_action(action: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Apply one common irrep action to every aligned multiplicity copy."""

    return state @ action.transpose(-1, -2)


if triton is not None:

    @triton.jit
    def _scalar_isotypic_recurrence(
        action_pointer,
        token_pointer,
        initial_pointer,
        output_pointer,
        length,
        multiplicity,
        BLOCK_M: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        copy_index = tl.program_id(1)
        rows = tl.arange(0, 8)
        columns = tl.arange(0, 8)
        state = tl.load(
            initial_pointer + (batch_index * multiplicity + copy_index) * 8 + columns
        ).to(tl.float16)

        for position in tl.range(0, length):
            token = tl.load(token_pointer + batch_index * length + position)
            action = tl.load(
                action_pointer + token * 64 + rows[:, None] * 8 + columns[None, :]
            )
            state = tl.sum(action * state[None, :], axis=1).to(tl.float16)
            output_offsets = (
                ((batch_index * length + position) * multiplicity + copy_index) * 8
                + rows
            )
            tl.store(output_pointer + output_offsets, state)

    @triton.jit
    def _tensor_core_isotypic_recurrence(
        action_pointer,
        token_pointer,
        initial_pointer,
        output_pointer,
        length,
        multiplicity,
        BLOCK_M: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        copy_block = tl.program_id(1)
        copies = copy_block * BLOCK_M + tl.arange(0, BLOCK_M)
        padded_rows = tl.arange(0, 16)
        padded_columns = tl.arange(0, 16)
        # X has shape (padded representation dimension, multiplicity tile).
        state = tl.load(
            initial_pointer
            + (batch_index * multiplicity + copies[None, :]) * 8
            + padded_columns[:, None],
            mask=(padded_columns[:, None] < 8) & (copies[None, :] < multiplicity),
            other=0.0,
        ).to(tl.float16)

        for position in tl.range(0, length):
            token = tl.load(token_pointer + batch_index * length + position)
            action = tl.load(
                action_pointer
                + token * 64
                + padded_rows[:, None] * 8
                + padded_columns[None, :],
                mask=(padded_rows[:, None] < 8) & (padded_columns[None, :] < 8),
                other=0.0,
            ).to(tl.float16)
            state = tl.dot(action, state).to(tl.float16)
            output_offsets = (
                ((batch_index * length + position) * multiplicity + copies[None, :]) * 8
                + padded_rows[:, None]
            )
            tl.store(
                output_pointer + output_offsets,
                state,
                mask=(padded_rows[:, None] < 8) & (copies[None, :] < multiplicity),
            )


def _run_kernel(
    kernel: Any,
    actions: torch.Tensor,
    tokens: torch.Tensor,
    initial: torch.Tensor,
    *,
    block_m: int,
    output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Any]:
    batch, length = tokens.shape
    multiplicity = initial.shape[1]
    if output is None:
        output = torch.empty(
            batch,
            length,
            multiplicity,
            8,
            device=initial.device,
            dtype=torch.float16,
        )
    grid = (
        (batch, multiplicity)
        if kernel is _scalar_isotypic_recurrence
        else (batch, triton.cdiv(multiplicity, block_m))
    )
    compiled = kernel[grid](
        actions,
        tokens,
        initial,
        output,
        length,
        multiplicity,
        BLOCK_M=block_m,
        num_warps=4 if kernel is _tensor_core_isotypic_recurrence else 1,
    )
    return output, compiled


def _timed_call(call: Any) -> tuple[float, torch.Tensor, Any]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output, compiled = call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1_000, output, compiled


def _paired_timings(
    scalar_call: Any, tensor_call: Any, repeats: int
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, Any, Any]:
    scalar_output, scalar_compiled = scalar_call()
    tensor_output, tensor_compiled = tensor_call()
    for _ in range(10):
        scalar_output, scalar_compiled = scalar_call()
        tensor_output, tensor_compiled = tensor_call()
    scalar_samples = []
    tensor_samples = []
    for index in range(repeats):
        ordered = (
            (("scalar", scalar_call), ("tensor", tensor_call))
            if index % 2 == 0
            else (("tensor", tensor_call), ("scalar", scalar_call))
        )
        for name, call in ordered:
            elapsed, output, compiled = _timed_call(call)
            if name == "scalar":
                scalar_samples.append(elapsed)
                scalar_output, scalar_compiled = output, compiled
            else:
                tensor_samples.append(elapsed)
                tensor_output, tensor_compiled = output, compiled
    scalar_median = statistics.median(scalar_samples)
    tensor_median = statistics.median(tensor_samples)
    return (
        {
            "scalar_median_microseconds": scalar_median,
            "tensor_core_median_microseconds": tensor_median,
            "tensor_core_speedup_vs_scalar": scalar_median / tensor_median,
            "scalar_p10_microseconds": sorted(scalar_samples)[repeats // 10],
            "scalar_p90_microseconds": sorted(scalar_samples)[9 * repeats // 10],
            "tensor_core_p10_microseconds": sorted(tensor_samples)[repeats // 10],
            "tensor_core_p90_microseconds": sorted(tensor_samples)[9 * repeats // 10],
        },
        scalar_output,
        tensor_output,
        scalar_compiled,
        tensor_compiled,
    )


def _reference(
    actions: torch.Tensor, tokens: torch.Tensor, initial: torch.Tensor
) -> torch.Tensor:
    actions64 = actions.double()
    state = initial.double()
    outputs = []
    for position in range(tokens.shape[1]):
        selected = actions64[tokens[:, position]]
        state = torch.einsum("bij,bmj->bmi", selected, state)
        outputs.append(state)
    return torch.stack(outputs, dim=1)


def _ptx_counts(compiled: Any) -> dict[str, int]:
    ptx = compiled.asm.get("ptx", "")
    return {
        "ptx_characters": len(ptx),
        "mma_sync_occurrences": ptx.count("mma.sync"),
        "fma_occurrences": ptx.count("fma."),
    }


def benchmark(checkpoint: Path) -> dict[str, Any]:
    if triton is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA and Triton are required")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # Use the vector action.  The theorem and dispatch mechanism are identical
    # for either half-spin representation; their numerical audits are separate.
    actions = payload["action_table"][:, 0].cuda().half().contiguous()
    rows = []
    seed = 20_260_821
    for batch, length, multiplicity in (
        (1, 128, 1),
        (1, 128, 8),
        (1, 128, 16),
        (1, 128, 32),
        (1, 128, 64),
        (8, 128, 16),
        (32, 128, 16),
        (8, 1024, 16),
    ):
        generator = torch.Generator(device="cpu").manual_seed(
            seed + 100_000 * batch + 100 * length + multiplicity
        )
        tokens = torch.randint(8, (batch, length), generator=generator).cuda()
        initial = torch.randn(
            (batch, multiplicity, 8), generator=generator, dtype=torch.float32
        ).cuda().half()
        # Keep one hardware-native 16-column tile per program.  Wider tiles
        # reduce occupancy on SM75 and are not the natural isotypic schedule.
        block_m = 16
        repeats = 100 if batch * length <= 1024 else 50
        scalar_buffer = torch.empty(
            batch, length, multiplicity, 8, device="cuda", dtype=torch.float16
        )
        tensor_buffer = torch.empty_like(scalar_buffer)
        scalar_call = lambda: _run_kernel(
                _scalar_isotypic_recurrence,
                actions,
                tokens,
                initial,
                block_m=block_m,
                output=scalar_buffer,
            )
        tensor_call = lambda: _run_kernel(
                _tensor_core_isotypic_recurrence,
                actions,
                tokens,
                initial,
                block_m=block_m,
                output=tensor_buffer,
            )
        timings, scalar, tensor, scalar_compiled, tensor_compiled = _paired_timings(
            scalar_call, tensor_call, repeats
        )
        reference = _reference(actions, tokens, initial)
        scalar_error = float((scalar.double() - reference).abs().max())
        tensor_error = float((tensor.double() - reference).abs().max())
        scalar_tensor_error = float((scalar - tensor).abs().max())
        rows.append(
            {
                "batch_size": batch,
                "sequence_length": length,
                "isotypic_multiplicity": multiplicity,
                "tile_multiplicity": block_m,
                **timings,
                "scalar_vs_float64_max_abs": scalar_error,
                "tensor_core_vs_float64_max_abs": tensor_error,
                "scalar_vs_tensor_core_max_abs": scalar_tensor_error,
                "scalar_ptx": _ptx_counts(scalar_compiled),
                "tensor_core_ptx": _ptx_counts(tensor_compiled),
            }
        )

    tensor_mma = {row["tensor_core_ptx"]["mma_sync_occurrences"] for row in rows}
    scalar_mma = {row["scalar_ptx"]["mma_sync_occurrences"] for row in rows}
    finite = all(
        math.isfinite(row[key])
        for row in rows
        for key in (
            "scalar_median_microseconds",
            "tensor_core_median_microseconds",
            "tensor_core_speedup_vs_scalar",
            "scalar_vs_float64_max_abs",
            "tensor_core_vs_float64_max_abs",
        )
    )
    properties = torch.cuda.get_device_properties(0)
    return {
        "schema_version": 1,
        "experiment": "Spin8 isotypic multiplicity Tensor-Core dispatch gate",
        "recorded_at": datetime.now().astimezone().isoformat(),
        "theorem_used": (
            "On V tensor R^m, a shared representation action rho(g) acts as "
            "X -> X rho(g)^T; aligned multiplicity is a GEMM tile axis."
        ),
        "source": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "representation": payload["representations"][0],
            "float16_action_table_sha256": tensor_sha256(actions),
        },
        "hardware": {
            "gpu": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "triton": triton.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "precision_contract": {
            "input_action_state": "float16",
            "dot_accumulator": "float32",
            "recurrent_state_rounding": "float16 after every token",
            "reference": "float64 recurrence from the same float16 inputs",
        },
        "checks": {
            "all_metrics_finite": finite,
            "tensor_kernel_contains_mma_sync": tensor_mma and min(tensor_mma) > 0,
            "scalar_kernel_contains_no_mma_sync": scalar_mma == {0},
            "all_tensor_core_errors_below_0p1": all(
                row["tensor_core_vs_float64_max_abs"] < 0.1 for row in rows
            ),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = benchmark(args.checkpoint)
    report["passed"] = all(report["checks"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
