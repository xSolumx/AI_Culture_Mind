"""Compile and benchmark a validated latent Pure Spin(8) token tracker.

The learned embedding/router is evaluated once for each vocabulary item. Its
eight faithful triality actions are stored as a frozen dictionary, then scanned
by the maintained eager oracle or the register-resident Triton recurrence.
This is model-specific inference compilation, not a replacement training path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from benchmark_pure_spin8_latent_increment import (
    LatentIncrementConfig,
    LatentPureSpin8Tracker,
    evaluate_relation_batches,
    make_relation_batches,
    token_action_table,
)
from pure_spin8_ssm import __version__ as PURE_SPIN8_VERSION
from pure_spin8_ssm.discrete_scan import (
    CompiledSpin8TokenTracker,
    triton_is_available,
)
from spin8_triality import TRIALITY_REPRESENTATIONS

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_ARTIFACT = (
    ROOT
    / "experiments"
    / "artifacts"
    / "pure_spin8_latent_increment_validation_seed1.json"
)
DEFAULT_COMPILED_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "pure_spin8_compiled_token_scan"
    / "compiled_latent_pure_spin8_seed1.pt"
)
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "artifacts"
    / "pure_spin8_compiled_token_scan_seed1.json"
)
DEFAULT_GRID = (
    (1, 16),
    (1, 128),
    (1, 1024),
    (8, 16),
    (8, 128),
    (8, 1024),
    (32, 16),
    (32, 128),
)


class _SourceOutput(nn.Module):
    def __init__(self, model: LatentPureSpin8Tracker) -> None:
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model(tokens)


class _CompiledOutput(nn.Module):
    def __init__(
        self, model: CompiledSpin8TokenTracker, *, backend: str
    ) -> None:
        super().__init__()
        self.model = model
        self.backend = backend

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        states, _ = self.model(
            tokens,
            backend=self.backend,
            validate_token_range=False,
        )
        return states


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha256(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def repeat_count(batch: int, length: int) -> int:
    work = batch * length
    if work <= 128:
        return 100
    if work <= 1024:
        return 50
    if work <= 8192:
        return 20
    return 10


@torch.inference_mode()
def time_callable(
    function: Callable[[], torch.Tensor], repeats: int
) -> tuple[dict[str, float | int], torch.Tensor]:
    warmups = max(5, min(10, repeats // 4))
    output = function()
    for _ in range(warmups):
        output = function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        output = function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1_000)
    return (
        {
            "repeats": repeats,
            "warmups": warmups,
            "median_microseconds": statistics.median(samples),
            "p10_microseconds": percentile(samples, 0.10),
            "p90_microseconds": percentile(samples, 0.90),
            "minimum_microseconds": min(samples),
        },
        output,
    )


@torch.inference_mode()
def peak_memory(function: Callable[[], torch.Tensor]) -> int:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    output = function()
    torch.cuda.synchronize()
    del output
    return int(torch.cuda.max_memory_allocated())


def load_source(
    artifact_path: Path, device: torch.device
) -> tuple[dict[str, Any], LatentPureSpin8Tracker, Path]:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not artifact["passed"] or artifact["config"]["seed"] != 1:
        raise ValueError("source must be the passing fresh seed-1 artifact")
    record = artifact["results"]["latent_pure_spin8"]
    checkpoint = Path(record["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if sha256(checkpoint) != record["checkpoint_sha256"]:
        raise ValueError("source checkpoint SHA-256 mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = LatentPureSpin8Tracker()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return artifact, model, checkpoint


@torch.inference_mode()
def compile_tracker(
    source: LatentPureSpin8Tracker,
    source_artifact: Path,
    source_checkpoint: Path,
) -> CompiledSpin8TokenTracker:
    actions = token_action_table(
        source.token_coordinates(), device=source.token_embedding.weight.device
    )
    initial = source.layer.initial_state[0]
    return CompiledSpin8TokenTracker(
        actions,
        initial,
        representations=TRIALITY_REPRESENTATIONS,
        metadata={
            "compiler": "finite latent-token action dictionary",
            "source_artifact": str(source_artifact),
            "source_artifact_sha256": sha256(source_artifact),
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": sha256(source_checkpoint),
            "source_seed": 1,
            "source_action_mode": "factorized",
            "source_package_version": "1.0.0",
        },
    )


@torch.inference_mode()
def relation_parity(
    source: nn.Module,
    compiled: nn.Module,
    config: LatentIncrementConfig,
    device: torch.device,
) -> dict[str, Any]:
    compiled_metrics = {}
    maximum_output_error = 0.0
    split_errors = {}
    for length in config.evaluation_lengths:
        for position in ("early", "late"):
            key = f"{position}_L{length}"
            batches = make_relation_batches(config, length, position, device)
            compiled_metrics[key] = evaluate_relation_batches(
                compiled,
                batches,
                device,
                config.evaluation_microbatch_size,
            )
            error = 0.0
            for batch in batches:
                for start in range(
                    0, batch.inputs.shape[0], config.evaluation_microbatch_size
                ):
                    tokens = batch.inputs[
                        start : start + config.evaluation_microbatch_size
                    ].to(device)
                    error = max(
                        error,
                        float((source(tokens) - compiled(tokens)).abs().max()),
                    )
            split_errors[key] = error
            maximum_output_error = max(maximum_output_error, error)
    return {
        "compiled_metrics": compiled_metrics,
        "source_vs_compiled_max_abs_by_split": split_errors,
        "source_vs_compiled_max_abs": maximum_output_error,
    }


def benchmark(
    source_artifact_path: Path,
    compiled_checkpoint_path: Path,
    grid: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    if not triton_is_available():
        raise RuntimeError("this benchmark requires Triton and CUDA")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device("cuda")
    source_artifact, source_model, source_checkpoint = load_source(
        source_artifact_path, device
    )
    compiled = compile_tracker(
        source_model, source_artifact_path, source_checkpoint
    ).to(device)
    compiled_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    compiled.save_checkpoint(compiled_checkpoint_path)
    compiled_checkpoint_sha = sha256(compiled_checkpoint_path)
    reloaded = CompiledSpin8TokenTracker.load_checkpoint(
        compiled_checkpoint_path, map_location=device
    ).eval()
    checkpoint_reload_exact = bool(
        torch.equal(reloaded.action_table, compiled.action_table)
        and torch.equal(reloaded.initial_state, compiled.initial_state)
        and reloaded.metadata == compiled.metadata
    )

    source_output = _SourceOutput(source_model).eval()
    eager_output = _CompiledOutput(reloaded, backend="eager").eval()
    triton_output = _CompiledOutput(reloaded, backend="triton").eval()
    config = LatentIncrementConfig(**source_artifact["config"])
    parity = relation_parity(source_output, triton_output, config, device)

    rows = []
    for batch, length in grid:
        generator = torch.Generator(device="cpu").manual_seed(
            20_260_817 + 10_000 * batch + length
        )
        tokens = torch.randint(8, (batch, length), generator=generator).to(device)
        calls = {
            "source_dynamic_factorized_work_efficient": (
                lambda tokens=tokens: source_output(tokens)
            ),
            "compiled_eager_recurrence": (
                lambda tokens=tokens: eager_output(tokens)
            ),
            "compiled_triton_recurrence": (
                lambda tokens=tokens: triton_output(tokens)
            ),
        }
        repeats = repeat_count(batch, length)
        timings = {}
        outputs = {}
        memories = {}
        for name, call in calls.items():
            memories[name] = peak_memory(call)
            timings[name], outputs[name] = time_callable(call, repeats)
        eager_error = float(
            (
                outputs["compiled_eager_recurrence"]
                - outputs["source_dynamic_factorized_work_efficient"]
            )
            .abs()
            .max()
        )
        triton_error = float(
            (
                outputs["compiled_triton_recurrence"]
                - outputs["source_dynamic_factorized_work_efficient"]
            )
            .abs()
            .max()
        )
        eager_triton_error = float(
            (
                outputs["compiled_triton_recurrence"]
                - outputs["compiled_eager_recurrence"]
            )
            .abs()
            .max()
        )
        source_time = timings["source_dynamic_factorized_work_efficient"][
            "median_microseconds"
        ]
        triton_time = timings["compiled_triton_recurrence"][
            "median_microseconds"
        ]
        rows.append(
            {
                "batch_size": batch,
                "sequence_length": length,
                "timings": timings,
                "peak_cuda_memory_bytes": memories,
                "source_vs_compiled_eager_max_abs": eager_error,
                "source_vs_compiled_triton_max_abs": triton_error,
                "compiled_eager_vs_triton_max_abs": eager_triton_error,
                "triton_speedup_vs_source_dynamic": source_time / triton_time,
            }
        )

    all_relation_signatures_pass = all(
        metrics["center_classification_accuracy"] == 1.0
        and metrics["center_rows_correct"] == 1.0
        and metrics["identity_rows_correct"] == 1.0
        for metrics in parity["compiled_metrics"].values()
    )
    maximum_grid_error = max(
        row["source_vs_compiled_triton_max_abs"] for row in rows
    )
    checks = {
        "source_artifact_passed": bool(source_artifact["passed"]),
        "compiled_checkpoint_rehashes": (
            sha256(compiled_checkpoint_path) == compiled_checkpoint_sha
        ),
        "compiled_checkpoint_reload_exact": checkpoint_reload_exact,
        "all_relation_signatures_preserved": all_relation_signatures_pass,
        "relation_output_parity_below_5e_5": (
            parity["source_vs_compiled_max_abs"] <= 5e-5
        ),
        "grid_output_parity_below_5e_5": maximum_grid_error <= 5e-5,
        "all_metrics_finite": all(
            math.isfinite(value)
            for row in rows
            for value in (
                row["source_vs_compiled_eager_max_abs"],
                row["source_vs_compiled_triton_max_abs"],
                row["compiled_eager_vs_triton_max_abs"],
                row["triton_speedup_vs_source_dynamic"],
            )
        ),
    }
    properties = torch.cuda.get_device_properties(device)
    speedups = [row["triton_speedup_vs_source_dynamic"] for row in rows]
    return {
        "schema_version": 1,
        "experiment": "compiled latent Pure Spin8 token-action recurrence",
        "recorded_at": datetime.now().astimezone().isoformat(),
        "pure_spin8_version": PURE_SPIN8_VERSION,
        "source": {
            "artifact": str(source_artifact_path),
            "artifact_sha256": sha256(source_artifact_path),
            "checkpoint": str(source_checkpoint),
            "checkpoint_sha256": sha256(source_checkpoint),
            "seed": source_artifact["config"]["seed"],
        },
        "compiled_checkpoint": {
            "path": str(compiled_checkpoint_path),
            "sha256": compiled_checkpoint_sha,
            "action_table_shape": list(reloaded.action_table.shape),
            "action_table_and_initial_sha256": tensor_sha256(
                reloaded.action_table, reloaded.initial_state
            ),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in reloaded.parameters()
                if parameter.requires_grad
            ),
            "recurrent_state_scalars": reloaded.recurrent_state_scalars,
        },
        "algorithm": {
            "source": (
                "embedding/router, 28 ordered factorized triality actions per token, "
                "then work-efficient affine prefix scan"
            ),
            "compiled": (
                "router evaluated once per vocabulary item; one Triton program per "
                "batch/representation walks frozen 8x8 actions in registers"
            ),
            "depth_tradeoff": (
                "compiled recurrence is serial in sequence length inside each program; "
                "it is a streaming recurrence, not a parallel prefix kernel"
            ),
        },
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "triton": __import__("triton").__version__,
            "cuda_device": {
                "name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "total_memory_bytes": properties.total_memory,
            },
            "torch_cpu_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "settings": {
            "dtype": "float32",
            "index_dtype": "int64",
            "grid": [list(cell) for cell in grid],
            "cuda_synchronized_each_complete_call": True,
            "timing_includes": "every-prefix model forward",
            "timing_excludes": "one-time checkpoint load and action compilation",
        },
        "relation_parity": parity,
        "benchmark_rows": rows,
        "aggregate": {
            "triton_speedup_vs_source_dynamic_range": [
                min(speedups),
                max(speedups),
            ],
            "triton_speedup_vs_source_dynamic_median": statistics.median(speedups),
            "maximum_grid_source_vs_triton_abs_error": maximum_grid_error,
        },
        "checks": checks,
        "claim_scope": {
            "validated_implementation": [
                "frozen learned token actions compiled into one faithful triality table",
                "one-kernel CUDA float32 every-prefix recurrence",
                "strict compiled-checkpoint roundtrip and initial-state backward",
            ],
            "empirical": [
                "output parity, center-relation preservation, latency, and allocation on the recorded workstation grid"
            ],
            "not_claimed": [
                "action-table gradients in the Triton path",
                "a parallel prefix implementation",
                "fused training or optimizer-step throughput",
                "a fused-Mamba comparison",
                "speedup generalization beyond the recorded GPU and grid",
            ],
        },
        "passed": all(checks.values()),
    }


def parse_grid(value: str) -> tuple[tuple[int, int], ...]:
    cells = []
    for item in value.split(","):
        batch, length = item.lower().split("x")
        cell = (int(batch), int(length))
        if cell[0] < 1 or cell[1] < 1:
            raise ValueError("grid dimensions must be positive")
        cells.append(cell)
    if not cells or len(set(cells)) != len(cells):
        raise ValueError("grid must contain distinct cells")
    return tuple(cells)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", type=Path, default=DEFAULT_SOURCE_ARTIFACT)
    parser.add_argument(
        "--compiled-checkpoint", type=Path, default=DEFAULT_COMPILED_CHECKPOINT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--grid",
        default=",".join(f"{batch}x{length}" for batch, length in DEFAULT_GRID),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = benchmark(
        args.source_artifact,
        args.compiled_checkpoint,
        parse_grid(args.grid),
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
