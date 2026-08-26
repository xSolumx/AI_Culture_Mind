"""Fail-closed SM75 prefill and cached-decode benchmark for sparse E6 memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch

from pure_spin_ssm_v1_2.mamba2_baseline import OfficialMamba2LM

from .benchmark_sparse_action_cost import (
    _build_paired_model,
    _sha256,
    _tensor_bytes,
    _timing_summary,
)
from .benchmark_train import TrainingConfig
from .model import ExceptionalDeltaLM, ExceptionalDeltaState, parameter_count


VARIANTS = (
    "e6_primitive_dead",
    "e6_primitive_event",
    "mamba2_official",
)


def _candidate_cache_bytes(states: list[ExceptionalDeltaState]) -> int:
    return sum(
        _tensor_bytes(state.memory) + _tensor_bytes(state.convolution)
        for state in states
    )


def _mamba_cache_bytes(inference_params) -> int:
    return sum(
        _tensor_bytes(tensor)
        for cache in inference_params.key_value_memory_dict.values()
        for tensor in cache
    )


def _run_child(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or args.device != "cuda":
        raise RuntimeError("the inference benchmark requires CUDA")
    if args.require_sm75 and torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError("the inference benchmark requires exact SM75")
    if args.child_variant not in VARIANTS:
        raise ValueError(f"unknown inference arm {args.child_variant!r}")
    if args.decode_warmups % 32 or args.decode_samples % 32:
        raise ValueError("decode warmups and samples must be multiples of 32")

    seed = args.seed + args.cycle
    config = TrainingConfig(
        batch_size=args.batch_size,
        sequence_length=args.prefix_length,
        d_model=args.d_model,
        layers=args.layers,
        memory_width=args.memory_width,
        update_rank=args.update_rank,
        d_conv=args.d_conv,
        primitive_sequence_backend=args.primitive_sequence_backend,
        channel_mixer=args.channel_mixer,
        readout_mode=args.readout_mode,
        seed=seed,
    )
    build_config = (
        replace(config, d_model=args.mamba_d_model)
        if args.child_variant == "mamba2_official"
        else config
    )
    device = torch.device("cuda")
    model = _build_paired_model(args.child_variant, build_config, device).eval()
    generator = torch.Generator(device=device).manual_seed(seed + 91_000)
    prefix = torch.randint(
        0,
        256,
        (args.batch_size, args.prefix_length),
        device=device,
        generator=generator,
    )
    decode_tokens = torch.randint(
        0,
        256,
        (args.batch_size, args.decode_warmups + args.decode_samples),
        device=device,
        generator=generator,
    )

    with torch.inference_mode():
        for _ in range(args.prefill_warmups):
            model(prefix)
        torch.cuda.synchronize()
        prefill_starts: list[torch.cuda.Event] = []
        prefill_ends: list[torch.cuda.Event] = []
        for _ in range(args.prefill_samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(prefix)
            end.record()
            prefill_starts.append(start)
            prefill_ends.append(end)
        torch.cuda.synchronize()
        prefill_samples_ms = [
            float(start.elapsed_time(end))
            for start, end in zip(prefill_starts, prefill_ends, strict=True)
        ]

        cache_prefill_start = torch.cuda.Event(enable_timing=True)
        cache_prefill_end = torch.cuda.Event(enable_timing=True)
        if isinstance(model, ExceptionalDeltaLM):
            cache_prefill_start.record()
            result = model(prefix)
            cache_prefill_end.record()
            states = result["states"]
            if not all(isinstance(state, ExceptionalDeltaState) for state in states):
                raise AssertionError("candidate did not return complete recurrent states")
            cache_bytes = _candidate_cache_bytes(states)
            inference_params = None
        elif isinstance(model, OfficialMamba2LM):
            from mamba_ssm.utils.generation import InferenceParams

            inference_params = InferenceParams(
                max_seqlen=(
                    args.prefix_length
                    + args.decode_warmups
                    + args.decode_samples
                ),
                max_batch_size=args.batch_size,
            )
            cache_prefill_start.record()
            model(prefix, inference_params=inference_params)
            cache_prefill_end.record()
            inference_params.seqlen_offset = args.prefix_length
            states = None
            cache_bytes = _mamba_cache_bytes(inference_params)
        else:
            raise TypeError("unsupported inference model")
        torch.cuda.synchronize()
        cache_prefill_ms = float(
            cache_prefill_start.elapsed_time(cache_prefill_end)
        )

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_before_decode = int(torch.cuda.memory_allocated())
        decode_starts: list[torch.cuda.Event] = []
        decode_ends: list[torch.cuda.Event] = []
        total_decode = args.decode_warmups + args.decode_samples
        for index in range(total_decode):
            token = decode_tokens[:, index : index + 1]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if isinstance(model, ExceptionalDeltaLM):
                result = model(token, states=states)
                states = result["states"]
            else:
                model(token, inference_params=inference_params)
                inference_params.seqlen_offset += 1
            end.record()
            if index >= args.decode_warmups:
                decode_starts.append(start)
                decode_ends.append(end)
        torch.cuda.synchronize()
        decode_samples_ms = [
            float(start.elapsed_time(end))
            for start, end in zip(decode_starts, decode_ends, strict=True)
        ]
        peak_decode = int(torch.cuda.max_memory_allocated())

    return {
        "variant": args.child_variant,
        "cycle": args.cycle,
        "seed": seed,
        "parameters": parameter_count(model),
        "model_d_model": build_config.d_model,
        "batch_size": args.batch_size,
        "prefix_length": args.prefix_length,
        "prefill": _timing_summary(prefill_samples_ms),
        "cache_building_prefill_ms": cache_prefill_ms,
        "decode": _timing_summary(decode_samples_ms),
        "decode_tokens_per_second": (
            args.batch_size * 1000.0 / _timing_summary(decode_samples_ms)["median_ms"]
        ),
        "cache_bytes": cache_bytes,
        "allocated_before_decode": allocated_before_decode,
        "peak_decode_allocated_bytes": peak_decode,
        "incremental_decode_peak_bytes": peak_decode - allocated_before_decode,
        "candidate_cache_scalars_per_stream": (
            model.cache_scalars if isinstance(model, ExceptionalDeltaLM) else None
        ),
        "event_stride": (
            model.config.action_event_stride
            if isinstance(model, ExceptionalDeltaLM)
            else None
        ),
    }


def _child_command(args: argparse.Namespace, variant: str, cycle: int) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pure_f4_delta_ssm_v1_3.benchmark_sparse_inference_cost",
        "--child-variant",
        variant,
        "--cycle",
        str(cycle),
        "--batch-size",
        str(args.batch_size),
        "--prefix-length",
        str(args.prefix_length),
        "--prefill-warmups",
        str(args.prefill_warmups),
        "--prefill-samples",
        str(args.prefill_samples),
        "--decode-warmups",
        str(args.decode_warmups),
        "--decode-samples",
        str(args.decode_samples),
        "--d-model",
        str(args.d_model),
        "--mamba-d-model",
        str(args.mamba_d_model),
        "--layers",
        str(args.layers),
        "--memory-width",
        str(args.memory_width),
        "--update-rank",
        str(args.update_rank),
        "--d-conv",
        str(args.d_conv),
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
    if args.require_sm75:
        command.append("--require-sm75")
    return command


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        prefill = [
            float(sample)
            for row in selected
            for sample in row["prefill"]["samples_ms"]  # type: ignore[index]
        ]
        decode = [
            float(sample)
            for row in selected
            for sample in row["decode"]["samples_ms"]  # type: ignore[index]
        ]
        result[variant] = {
            "cycles": len(selected),
            "parameters": sorted({int(row["parameters"]) for row in selected}),
            "model_d_model": sorted({int(row["model_d_model"]) for row in selected}),
            "prefill": _timing_summary(prefill),
            "cache_building_prefill": _timing_summary(
                [float(row["cache_building_prefill_ms"]) for row in selected]
            ),
            "decode": _timing_summary(decode),
            "maximum_cache_bytes": max(int(row["cache_bytes"]) for row in selected),
            "maximum_decode_peak_allocated_bytes": max(
                int(row["peak_decode_allocated_bytes"]) for row in selected
            ),
        }
    return result


def _verdict(summary: dict[str, object]) -> dict[str, object]:
    if not set(VARIANTS).issubset(summary):
        return {
            "streaming_decode_pass": False,
            "bulk_prefill_pass": False,
            "cache_building_prefill_pass": False,
            "reason": "required inference arms are missing",
        }
    candidate = summary["e6_primitive_event"]  # type: ignore[assignment]
    dead = summary["e6_primitive_dead"]  # type: ignore[assignment]
    mamba = summary["mamba2_official"]  # type: ignore[assignment]
    parameter_sets = {
        name: summary[name]["parameters"]  # type: ignore[index]
        for name in VARIANTS
    }
    parameter_match = all(
        len(values) == 1 and int(values[0]) > 0
        for values in parameter_sets.values()
    )
    if not parameter_match:
        return {
            "streaming_decode_pass": False,
            "bulk_prefill_pass": False,
            "cache_building_prefill_pass": False,
            "reason": "parameter counts are invalid or inconsistent across cycles",
        }
    candidate_parameters = int(parameter_sets["e6_primitive_event"][0])
    dead_parameters = int(parameter_sets["e6_primitive_dead"][0])
    mamba_parameters = int(parameter_sets["mamba2_official"][0])
    parameter_residual = (candidate_parameters - mamba_parameters) / mamba_parameters

    def median(arm: dict[str, object], field: str) -> float:
        return float(arm[field]["median_ms"])  # type: ignore[index]

    ratios = {
        "candidate_decode_time_over_dead": median(candidate, "decode")
        / median(dead, "decode"),
        "candidate_decode_time_over_mamba2": median(candidate, "decode")
        / median(mamba, "decode"),
        "candidate_cache_bytes_over_mamba2": int(candidate["maximum_cache_bytes"])
        / int(mamba["maximum_cache_bytes"]),
        "candidate_bulk_prefill_time_over_mamba2": median(candidate, "prefill")
        / median(mamba, "prefill"),
        "candidate_cache_prefill_time_over_mamba2": median(
            candidate, "cache_building_prefill"
        )
        / median(mamba, "cache_building_prefill"),
    }
    budgets_match = (
        candidate_parameters == dead_parameters and abs(parameter_residual) <= 0.01
    )
    streaming = (
        budgets_match
        and ratios["candidate_decode_time_over_dead"] <= 1.25
        and ratios["candidate_decode_time_over_mamba2"] <= 1.25
        and ratios["candidate_cache_bytes_over_mamba2"] <= 1.25
    )
    return {
        "streaming_decode_pass": streaming,
        "bulk_prefill_pass": budgets_match
        and ratios["candidate_bulk_prefill_time_over_mamba2"] <= 1.25,
        "cache_building_prefill_pass": budgets_match
        and ratios["candidate_cache_prefill_time_over_mamba2"] <= 1.25,
        "parameter_match_pass": budgets_match,
        "candidate_parameter_residual_fraction_vs_mamba2": parameter_residual,
        "ratios": ratios,
    }


def _run_parent(args: argparse.Namespace) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for cycle in range(args.cycles):
        ordered = list(VARIANTS[cycle % len(VARIANTS) :] + VARIANTS[: cycle % len(VARIANTS)])
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
                    f"inference child failed for {variant} cycle {cycle}:\n"
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
    mamba_adapter = root.parent / "pure_spin_ssm_v1_2" / "mamba2_baseline.py"
    try:
        import mamba_ssm

        mamba_version = getattr(mamba_ssm, "__version__", "unknown")
        mamba_path = str(Path(mamba_ssm.__file__).resolve())
    except Exception as error:  # pragma: no cover - fail-closed runtime metadata
        raise RuntimeError(
            "could not identify the official mamba_ssm package"
        ) from error
    return {
        "schema_version": 1,
        "experiment": "SM75 bulk prefill and true cached one-token decode",
        "status": "systems qualification; not model-quality evidence",
        "config": {
            "cycles": args.cycles,
            "batch_size": args.batch_size,
            "prefix_length": args.prefix_length,
            "prefill_warmups": args.prefill_warmups,
            "prefill_samples": args.prefill_samples,
            "decode_warmups": args.decode_warmups,
            "decode_samples": args.decode_samples,
            "d_model": args.d_model,
            "mamba_d_model": args.mamba_d_model,
            "layers": args.layers,
            "memory_width": args.memory_width,
            "update_rank": args.update_rank,
            "primitive_sequence_backend": args.primitive_sequence_backend,
            "channel_mixer": args.channel_mixer,
            "readout_mode": args.readout_mode,
            "seed": args.seed,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "mamba_ssm_version": mamba_version,
            "mamba_ssm_path": mamba_path,
        },
        "git": {"revision": revision, "dirty": bool(status.strip())},
        "source_sha256": {
            "benchmark_sparse_inference_cost.py": _sha256(Path(__file__)),
            "benchmark_sparse_action_cost.py": _sha256(
                root / "benchmark_sparse_action_cost.py"
            ),
            "model.py": _sha256(root / "model.py"),
            "primitive_action_cuda.cu": _sha256(root / "primitive_action_cuda.cu"),
            "scan.py": _sha256(root / "scan.py"),
            "../pure_spin_ssm_v1_2/mamba2_baseline.py": _sha256(mamba_adapter),
        },
        "rows": rows,
        "summary": summary,
        "verdict": _verdict(summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefix-length", type=int, default=4096)
    parser.add_argument("--prefill-warmups", type=int, default=3)
    parser.add_argument("--prefill-samples", type=int, default=10)
    parser.add_argument("--decode-warmups", type=int, default=32)
    parser.add_argument("--decode-samples", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=204)
    parser.add_argument("--mamba-d-model", type=int, default=224)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--memory-width", type=int, default=8)
    parser.add_argument("--update-rank", type=int, default=2)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument(
        "--primitive-sequence-backend",
        choices=("auto", "recurrent", "chunked_parallel"),
        default="auto",
    )
    parser.add_argument(
        "--channel-mixer", choices=("jordan", "swiglu", "none"), default="swiglu"
    )
    parser.add_argument(
        "--readout-mode",
        choices=("albert_invariants", "vector"),
        default="vector",
    )
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--require-sm75", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child-variant", choices=VARIANTS, help=argparse.SUPPRESS)
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
