"""Fail-closed complete-step SM75 benchmark for sparse exceptional transport."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .benchmark_train import (
    VARIANTS,
    TrainingConfig,
    _build_model,
    _copy_shared_exceptional_initialization,
    _forward_logits,
    _seed_all,
)
from .model import ExceptionalDeltaLM, parameter_count

PRIMARY_VARIANTS = (
    "e6_primitive_dead",
    "e6_primitive_event",
    "e6_safe",
    "mamba2_official",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _percentile(samples: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(samples, dtype=np.float64), percentile))


def _timing_summary(samples_ms: list[float]) -> dict[str, object]:
    mean = statistics.fmean(samples_ms)
    deviation = statistics.pstdev(samples_ms)
    return {
        "samples_ms": samples_ms,
        "median_ms": statistics.median(samples_ms),
        "p10_ms": _percentile(samples_ms, 10),
        "p90_ms": _percentile(samples_ms, 90),
        "mean_ms": mean,
        "coefficient_of_variation": deviation / mean if mean else math.inf,
    }


def _complete_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    logits = _forward_logits(model, inputs)
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
        raise FloatingPointError("nonfinite loss or gradient in cost benchmark")
    optimizer.step()
    return float(loss.detach())


def _build_paired_model(
    variant: str, config: TrainingConfig, device: torch.device
) -> torch.nn.Module:
    _seed_all(config.seed)
    model = _build_model(variant, config).to(device)
    if isinstance(model, ExceptionalDeltaLM) and variant != "identity_safe":
        _seed_all(config.seed)
        reference = _build_model(
            "identity_safe", replace(config, d_model=model.config.d_model)
        )
        if not isinstance(reference, ExceptionalDeltaLM):
            raise AssertionError("identity reference must be exceptional model")
        _copy_shared_exceptional_initialization(model, reference.to(device))
    return model


def _saved_tensor_shapes(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> list[list[int]]:
    shapes: set[tuple[int, ...]] = set()

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        shapes.add(tuple(tensor.shape))
        return tensor

    model.zero_grad(set_to_none=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        logits = _forward_logits(model, inputs)
        F.cross_entropy(logits.flatten(0, 1), targets.flatten()).backward()
    model.zero_grad(set_to_none=True)
    return [list(shape) for shape in sorted(shapes)]


def _run_child(args: argparse.Namespace) -> dict[str, object]:
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the primary cost benchmark requires CUDA")
    if args.require_sm75 and torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError("the primary cost benchmark requires exact SM75")
    if args.child_variant not in VARIANTS:
        raise ValueError(f"unknown variant {args.child_variant!r}")

    device = torch.device(args.device)
    seed = args.seed + args.cycle
    config = TrainingConfig(
        steps=args.samples,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        validation_batches=0,
        d_model=args.d_model,
        layers=args.layers,
        memory_width=args.memory_width,
        update_rank=args.update_rank,
        d_conv=args.d_conv,
        activation_checkpointing=args.activation_checkpointing,
        primitive_sequence_backend=args.primitive_sequence_backend,
        channel_mixer=args.channel_mixer,
        readout_mode=args.readout_mode,
        learning_rate=args.learning_rate,
        seed=seed,
    )
    build_config = (
        replace(config, d_model=args.mamba_d_model)
        if args.child_variant == "mamba2_official"
        and args.mamba_d_model is not None
        else config
    )
    base_model = _build_paired_model(args.child_variant, build_config, device)
    model = (
        torch.compile(base_model, mode=args.compile_mode)
        if args.compile_exceptional and isinstance(base_model, ExceptionalDeltaLM)
        else base_model
    )
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )

    generator = torch.Generator(device=device).manual_seed(seed + 40_000)
    batch_count = 1 + args.warmups + args.samples
    batches = [
        (
            torch.randint(
                0,
                256,
                (args.batch_size, args.sequence_length),
                device=device,
                generator=generator,
            ),
            torch.randint(
                0,
                256,
                (args.batch_size, args.sequence_length),
                device=device,
                generator=generator,
            ),
        )
        for _ in range(batch_count)
    ]

    # Allocate optimizer state and compile/load any native extension before the
    # warm-up and measured windows.
    losses = [_complete_step(model, optimizer, *batches[0])]
    for index in range(1, 1 + args.warmups):
        losses.append(_complete_step(model, optimizer, *batches[index]))
    torch.cuda.synchronize()

    saved_shapes = _saved_tensor_shapes(model, *batches[-1])
    materialized_dense_sequence_action = any(
        len(shape) >= 4
        and shape[0] == args.batch_size
        and shape[1] == args.sequence_length
        and shape[-2:] == [27, 27]
        for shape in saved_shapes
    )

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = int(torch.cuda.memory_allocated())
    reserved_before = int(torch.cuda.memory_reserved())
    starts: list[torch.cuda.Event] = []
    ends: list[torch.cuda.Event] = []
    measured_losses: list[float] = []
    offset = 1 + args.warmups
    for index in range(args.samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        measured_losses.append(
            _complete_step(model, optimizer, *batches[offset + index])
        )
        end.record()
        starts.append(start)
        ends.append(end)
    torch.cuda.synchronize()
    samples_ms = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())

    optimizer_bytes = sum(
        _tensor_bytes(value)
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    action_buffer_bytes = 0
    if isinstance(base_model, ExceptionalDeltaLM):
        action_buffer_bytes = sum(
            _tensor_bytes(buffer)
            for block in base_model.blocks
            for buffer in block.action.buffers()
        )
    model_buffer_bytes = sum(_tensor_bytes(buffer) for buffer in base_model.buffers())
    event_stride = (
        base_model.config.action_event_stride
        if isinstance(base_model, ExceptionalDeltaLM)
        and base_model.config.action_geometry == "canonical_product"
        else None
    )
    transport_enabled = (
        base_model.config.primitive_transport_enabled
        if isinstance(base_model, ExceptionalDeltaLM)
        and base_model.config.action_geometry == "canonical_product"
        else None
    )
    events_per_layer = (
        args.sequence_length // event_stride if event_stride is not None else 0
    )
    mamba_runtime = None
    if args.child_variant == "mamba2_official":
        import mamba_ssm

        try:
            mamba_version = importlib.metadata.version("mamba-ssm")
        except importlib.metadata.PackageNotFoundError:
            mamba_version = getattr(mamba_ssm, "__version__", "unknown")
        mamba_runtime = {
            "version": mamba_version,
            "module_path": str(Path(mamba_ssm.__file__).resolve()),
        }
    return {
        "variant": args.child_variant,
        "cycle": args.cycle,
        "seed": seed,
        "model_d_model": build_config.d_model,
        "execution": (
            f"torch_compile_{args.compile_mode}"
            if args.compile_exceptional and isinstance(base_model, ExceptionalDeltaLM)
            else "eager"
        ),
        "mamba_ssm_runtime": mamba_runtime,
        "parameters": parameter_count(base_model),
        "parameter_bytes": sum(_tensor_bytes(p) for p in base_model.parameters()),
        "model_buffer_bytes": model_buffer_bytes,
        "action_buffer_bytes": action_buffer_bytes,
        "optimizer_state_bytes": optimizer_bytes,
        "cache_scalars": getattr(base_model, "cache_scalars", None),
        "event_stride": event_stride,
        "primitive_transport_enabled": transport_enabled,
        "primitive_sequence_backend_config": (
            base_model.config.primitive_sequence_backend
            if isinstance(base_model, ExceptionalDeltaLM)
            else None
        ),
        "primitive_sequence_backend_effective": (
            "chunked_parallel"
            if isinstance(base_model, ExceptionalDeltaLM)
            and base_model.config.action_geometry == "canonical_product"
            and (
                base_model.config.primitive_sequence_backend == "chunked_parallel"
                or (
                    base_model.config.primitive_sequence_backend == "auto"
                    and args.sequence_length >= 256
                )
            )
            and args.sequence_length % base_model.config.action_event_stride == 0
            else "recurrent"
            if isinstance(base_model, ExceptionalDeltaLM)
            and base_model.config.action_geometry == "canonical_product"
            else None
        ),
        "events_per_layer": events_per_layer,
        "event_density": (
            events_per_layer / args.sequence_length if args.sequence_length else 0.0
        ),
        "state_action_applications_per_step": (
            args.batch_size * args.layers * events_per_layer
            if transport_enabled
            else 0
        ),
        "timing": _timing_summary(samples_ms),
        "loss_first": losses[0],
        "loss_last": measured_losses[-1],
        "allocated_before_window": allocated_before,
        "reserved_before_window": reserved_before,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "incremental_peak_allocated_bytes": peak_allocated - allocated_before,
        "saved_tensor_shapes": saved_shapes,
        "materialized_dense_sequence_action": materialized_dense_sequence_action,
    }


def _child_command(args: argparse.Namespace, variant: str, cycle: int) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pure_f4_delta_ssm_v1_3.benchmark_sparse_action_cost",
        "--child-variant",
        variant,
        "--cycle",
        str(cycle),
        "--warmups",
        str(args.warmups),
        "--samples",
        str(args.samples),
        "--batch-size",
        str(args.batch_size),
        "--sequence-length",
        str(args.sequence_length),
        "--d-model",
        str(args.d_model),
        "--layers",
        str(args.layers),
        "--memory-width",
        str(args.memory_width),
        "--update-rank",
        str(args.update_rank),
        "--d-conv",
        str(args.d_conv),
        "--learning-rate",
        str(args.learning_rate),
        "--primitive-sequence-backend",
        args.primitive_sequence_backend,
        "--channel-mixer",
        args.channel_mixer,
        "--readout-mode",
        args.readout_mode,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.mamba_d_model is not None:
        command.extend(("--mamba-d-model", str(args.mamba_d_model)))
    if args.activation_checkpointing:
        command.append("--activation-checkpointing")
    if args.compile_exceptional:
        command.extend(("--compile-exceptional", "--compile-mode", args.compile_mode))
    if args.require_sm75:
        command.append("--require-sm75")
    return command


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    variants = sorted({str(row["variant"]) for row in rows})
    result: dict[str, object] = {}
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        samples = [
            float(sample)
            for row in selected
            for sample in row["timing"]["samples_ms"]  # type: ignore[index]
        ]
        peaks = [int(row["peak_allocated_bytes"]) for row in selected]
        result[variant] = {
            "cycles": len(selected),
            "timing": _timing_summary(samples),
            "maximum_peak_allocated_bytes": max(peaks),
            "parameters": sorted({int(row["parameters"]) for row in selected}),
            "model_d_model": sorted(
                {int(row["model_d_model"]) for row in selected}
            ),
            "any_dense_sequence_action": any(
                bool(row["materialized_dense_sequence_action"]) for row in selected
            ),
        }
    return result


def _verdict(summary: dict[str, object]) -> dict[str, object]:
    required = set(PRIMARY_VARIANTS)
    if not required.issubset(summary):
        return {
            "cheap_action_path_pass": False,
            "mamba_competitive_pass": False,
            "reason": "required benchmark arms are missing",
        }

    def median(name: str) -> float:
        return float(summary[name]["timing"]["median_ms"])  # type: ignore[index]

    def peak(name: str) -> int:
        return int(summary[name]["maximum_peak_allocated_bytes"])  # type: ignore[index]

    candidate = "e6_primitive_event"
    dead = "e6_primitive_dead"
    dense = "e6_safe"
    mamba = "mamba2_official"

    parameter_sets = {
        name: [int(value) for value in summary[name]["parameters"]]  # type: ignore[index]
        for name in required
    }
    if any(len(values) != 1 or values[0] <= 0 for values in parameter_sets.values()):
        return {
            "cheap_action_path_pass": False,
            "mamba_competitive_pass": False,
            "reason": "parameter counts are missing, nonpositive, or inconsistent across cycles",
            "parameter_sets": parameter_sets,
        }
    model_width_sets = {
        name: [int(value) for value in summary[name]["model_d_model"]]  # type: ignore[index]
        for name in required
    }
    if any(len(values) != 1 or values[0] <= 0 for values in model_width_sets.values()):
        return {
            "cheap_action_path_pass": False,
            "mamba_competitive_pass": False,
            "reason": "model widths are missing, nonpositive, or inconsistent across cycles",
            "model_width_sets": model_width_sets,
        }
    ratios = {
        "candidate_time_over_dead_budget": median(candidate) / median(dead),
        "candidate_peak_over_dead_budget": peak(candidate) / peak(dead),
        "candidate_time_over_dense_e6": median(candidate) / median(dense),
        "candidate_peak_over_dense_e6": peak(candidate) / peak(dense),
        "candidate_time_over_mamba2": median(candidate) / median(mamba),
        "candidate_peak_over_mamba2": peak(candidate) / peak(mamba),
    }
    candidate_parameters = parameter_sets[candidate][0]
    dead_parameters = parameter_sets[dead][0]
    dense_parameters = parameter_sets[dense][0]
    mamba_parameters = parameter_sets[mamba][0]
    dead_parameter_match = candidate_parameters == dead_parameters
    dense_parameter_match = candidate_parameters == dense_parameters
    mamba_parameter_residual = (
        candidate_parameters - mamba_parameters
    ) / mamba_parameters
    no_dense_candidate = not bool(
        summary[candidate]["any_dense_sequence_action"]  # type: ignore[index]
    )
    cheap = (
        ratios["candidate_time_over_dead_budget"] <= 1.25
        and ratios["candidate_peak_over_dead_budget"] <= 1.30
        and ratios["candidate_time_over_dense_e6"] <= 0.75
        and ratios["candidate_peak_over_dense_e6"] <= 0.60
        and no_dense_candidate
        and dead_parameter_match
        and dense_parameter_match
    )
    mamba_competitive = (
        ratios["candidate_time_over_mamba2"] <= 1.25
        and ratios["candidate_peak_over_mamba2"] <= 1.25
        and no_dense_candidate
        and abs(mamba_parameter_residual) <= 0.01
    )
    return {
        "cheap_action_path_pass": cheap,
        "mamba_competitive_pass": mamba_competitive,
        "ratios": ratios,
        "dead_budget_parameter_match": dead_parameter_match,
        "dense_e6_parameter_match": dense_parameter_match,
        "candidate_parameter_residual_fraction_vs_mamba2": mamba_parameter_residual,
        "candidate_has_no_dense_sequence_action": no_dense_candidate,
    }


def _run_parent(args: argparse.Namespace) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    variants = list(args.variants)
    for cycle in range(args.cycles):
        ordered = variants[cycle % len(variants) :] + variants[: cycle % len(variants)]
        for variant in ordered:
            completed = subprocess.run(
                _child_command(args, variant, cycle),
                check=False,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            if completed.returncode:
                raise RuntimeError(
                    f"cost child failed for {variant} cycle {cycle}:\n"
                    f"{completed.stderr}\n{completed.stdout}"
                )
            rows.append(json.loads(completed.stdout))

    summary = _aggregate(rows)
    root = Path(__file__).resolve().parent
    repository = root.parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    source_names = (
        "action.py",
        "albert.py",
        "benchmark_sparse_action_cost.py",
        "benchmark_train.py",
        "model.py",
        "primitive_action.py",
        "primitive_action_bindings.cpp",
        "primitive_action_cuda.cu",
        "scan.py",
    )
    mamba_adapter = root.parent / "pure_spin_ssm_v1_2" / "mamba2_baseline.py"
    report = {
        "schema_version": 1,
        "experiment": "complete-step sparse primitive exceptional transport cost",
        "status": "SM75 systems qualification; not model-quality evidence",
        "config": {
            "variants": args.variants,
            "cycles": args.cycles,
            "warmups_per_cycle": args.warmups,
            "samples_per_cycle": args.samples,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "d_model": args.d_model,
            "mamba_d_model": args.mamba_d_model,
            "layers": args.layers,
            "memory_width": args.memory_width,
            "update_rank": args.update_rank,
            "d_conv": args.d_conv,
            "activation_checkpointing": args.activation_checkpointing,
            "primitive_sequence_backend": args.primitive_sequence_backend,
            "channel_mixer": args.channel_mixer,
            "readout_mode": args.readout_mode,
            "compile_exceptional": args.compile_exceptional,
            "compile_mode": args.compile_mode,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "compute_capability": (
                list(torch.cuda.get_device_capability())
                if torch.cuda.is_available()
                else None
            ),
        },
        "git": {
            "revision": revision,
            "dirty": bool(status.strip()),
            "working_patch_sha256": hashlib.sha256(diff).hexdigest(),
        },
        "source_sha256": {
            **{name: _sha256(root / name) for name in source_names},
            "../pure_spin_ssm_v1_2/mamba2_baseline.py": _sha256(mamba_adapter),
        },
        "rows": rows,
        "summary": summary,
        "verdict": _verdict(summary),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=list(PRIMARY_VARIANTS))
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--mamba-d-model", type=int)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--memory-width", type=int, default=4)
    parser.add_argument("--update-rank", type=int, default=2)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument(
        "--primitive-sequence-backend",
        choices=("auto", "recurrent", "chunked_parallel"),
        default="auto",
    )
    parser.add_argument(
        "--channel-mixer", choices=("jordan", "swiglu", "none"), default="jordan"
    )
    parser.add_argument(
        "--readout-mode",
        choices=("albert_invariants", "vector"),
        default="albert_invariants",
    )
    parser.add_argument("--compile-exceptional", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--require-sm75", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child-variant", choices=tuple(VARIANTS), help=argparse.SUPPRESS)
    parser.add_argument("--cycle", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child_variant is not None:
        print(json.dumps(_run_child(args), sort_keys=True))
        return
    report = _run_parent(args)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(payload, end="")


if __name__ == "__main__":
    main()
