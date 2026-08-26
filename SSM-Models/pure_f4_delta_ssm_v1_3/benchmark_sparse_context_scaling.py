"""Prospective fixed-token SM75 shape qualification for sparse E6 transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch


DEFAULT_LENGTHS = (128, 256, 512, 1024, 2048, 4096)
DEFAULT_VARIANTS = (
    "e6_primitive_dead",
    "e6_primitive_event",
    "mamba2_official",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _single_parameter_count(report: dict[str, object], variant: str) -> int:
    values = report["summary"][variant]["parameters"]  # type: ignore[index]
    if not isinstance(values, list) or len(values) != 1 or int(values[0]) <= 0:
        raise ValueError(f"invalid parameter counts for {variant}: {values!r}")
    return int(values[0])


def _single_model_width(report: dict[str, object], variant: str) -> int:
    values = report["summary"][variant]["model_d_model"]  # type: ignore[index]
    if not isinstance(values, list) or len(values) != 1 or int(values[0]) <= 0:
        raise ValueError(f"invalid model widths for {variant}: {values!r}")
    return int(values[0])


def _scaling_verdict(reports: list[dict[str, object]]) -> dict[str, object]:
    if not reports:
        return {
            "fixed_token_scaling_pass": False,
            "mamba_competitive_all_contexts_pass": False,
            "reason": "no context reports",
        }

    expected_variants = set(DEFAULT_VARIANTS)
    revisions = {str(report["git"]["revision"]) for report in reports}  # type: ignore[index]
    dirty = [bool(report["git"]["dirty"]) for report in reports]  # type: ignore[index]
    token_counts = {
        int(report["config"]["batch_size"])  # type: ignore[index]
        * int(report["config"]["sequence_length"])  # type: ignore[index]
        for report in reports
    }
    variants_match = all(
        set(report["config"]["variants"]) == expected_variants  # type: ignore[index]
        for report in reports
    )

    rows: list[dict[str, object]] = []
    parameter_match = True
    width_match = True
    no_dense_candidate = True
    for report in reports:
        summary = report["summary"]  # type: ignore[assignment]
        candidate_parameters = _single_parameter_count(
            report, "e6_primitive_event"
        )
        dead_parameters = _single_parameter_count(report, "e6_primitive_dead")
        mamba_parameters = _single_parameter_count(report, "mamba2_official")
        residual = (candidate_parameters - mamba_parameters) / mamba_parameters
        parameter_match = parameter_match and (
            candidate_parameters == dead_parameters and abs(residual) <= 0.01
        )
        candidate_width = _single_model_width(report, "e6_primitive_event")
        dead_width = _single_model_width(report, "e6_primitive_dead")
        mamba_width = _single_model_width(report, "mamba2_official")
        width_match = width_match and (
            candidate_width == dead_width
            and candidate_width == int(report["config"]["d_model"])  # type: ignore[index]
            and mamba_width == int(report["config"]["mamba_d_model"])  # type: ignore[index]
        )

        def median(variant: str) -> float:
            return float(summary[variant]["timing"]["median_ms"])  # type: ignore[index]

        def peak(variant: str) -> int:
            return int(summary[variant]["maximum_peak_allocated_bytes"])  # type: ignore[index]

        candidate_dense = bool(
            summary["e6_primitive_event"]["any_dense_sequence_action"]  # type: ignore[index]
        )
        no_dense_candidate = no_dense_candidate and not candidate_dense
        batch_size = int(report["config"]["batch_size"])  # type: ignore[index]
        sequence_length = int(report["config"]["sequence_length"])  # type: ignore[index]
        rows.append(
            {
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "tokens_per_step": batch_size * sequence_length,
                "candidate_parameters": candidate_parameters,
                "mamba2_parameters": mamba_parameters,
                "candidate_parameter_residual_fraction_vs_mamba2": residual,
                "candidate_median_ms": median("e6_primitive_event"),
                "dead_budget_median_ms": median("e6_primitive_dead"),
                "mamba2_median_ms": median("mamba2_official"),
                "candidate_peak_allocated_bytes": peak("e6_primitive_event"),
                "dead_budget_peak_allocated_bytes": peak("e6_primitive_dead"),
                "mamba2_peak_allocated_bytes": peak("mamba2_official"),
                "candidate_time_over_dead_budget": (
                    median("e6_primitive_event")
                    / median("e6_primitive_dead")
                ),
                "candidate_peak_over_dead_budget": (
                    peak("e6_primitive_event")
                    / peak("e6_primitive_dead")
                ),
                "candidate_time_over_mamba2": (
                    median("e6_primitive_event")
                    / median("mamba2_official")
                ),
                "candidate_peak_over_mamba2": (
                    peak("e6_primitive_event")
                    / peak("mamba2_official")
                ),
                "candidate_has_dense_sequence_action": candidate_dense,
            }
        )

    candidate_times = [float(row["candidate_median_ms"]) for row in rows]
    candidate_peaks = [int(row["candidate_peak_allocated_bytes"]) for row in rows]
    candidate_time_spread = max(candidate_times) / min(candidate_times)
    candidate_peak_spread = max(candidate_peaks) / min(candidate_peaks)
    marginal_all = all(
        float(row["candidate_time_over_dead_budget"]) <= 1.25
        and float(row["candidate_peak_over_dead_budget"]) <= 1.30
        for row in rows
    )
    mamba_all = all(
        float(row["candidate_time_over_mamba2"]) <= 1.25
        and float(row["candidate_peak_over_mamba2"]) <= 1.25
        for row in rows
    )
    provenance_pass = len(revisions) == 1 and not any(dirty)
    fixed_work_pass = len(token_counts) == 1
    scaling_pass = (
        provenance_pass
        and fixed_work_pass
        and variants_match
        and parameter_match
        and width_match
        and no_dense_candidate
        and marginal_all
        and candidate_time_spread <= 2.0
        and candidate_peak_spread <= 1.5
    )
    return {
        "fixed_token_scaling_pass": scaling_pass,
        "mamba_competitive_all_contexts_pass": scaling_pass and mamba_all,
        "provenance_pass": provenance_pass,
        "fixed_token_work_pass": fixed_work_pass,
        "variant_set_pass": variants_match,
        "parameter_match_pass": parameter_match,
        "model_width_match_pass": width_match,
        "candidate_has_no_dense_sequence_action": no_dense_candidate,
        "marginal_cost_all_contexts_pass": marginal_all,
        "candidate_time_spread_max_over_min": candidate_time_spread,
        "candidate_peak_spread_max_over_min": candidate_peak_spread,
        "thresholds": {
            "candidate_time_over_dead_budget_max": 1.25,
            "candidate_peak_over_dead_budget_max": 1.30,
            "candidate_time_spread_max_over_min": 2.0,
            "candidate_peak_spread_max_over_min": 1.5,
            "candidate_time_over_mamba2_max": 1.25,
            "candidate_peak_over_mamba2_max": 1.25,
            "candidate_parameter_residual_abs_max_vs_mamba2": 0.01,
        },
        "rows": rows,
    }


def _benchmark_command(args: argparse.Namespace, length: int) -> list[str]:
    if args.token_budget % length:
        raise ValueError(
            f"token budget {args.token_budget} is not divisible by length {length}"
        )
    batch_size = args.token_budget // length
    command = [
        sys.executable,
        "-m",
        "pure_f4_delta_ssm_v1_3.benchmark_sparse_action_cost",
        "--variants",
        *DEFAULT_VARIANTS,
        "--cycles",
        str(args.cycles),
        "--warmups",
        str(args.warmups),
        "--samples",
        str(args.samples),
        "--batch-size",
        str(batch_size),
        "--sequence-length",
        str(length),
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
        "--learning-rate",
        str(args.learning_rate),
        "--seed",
        str(args.seed),
        "--device",
        "cuda",
    ]
    if args.require_sm75:
        command.append("--require-sm75")
    if args.activation_checkpointing:
        command.append("--activation-checkpointing")
    return command


def _run(args: argparse.Namespace) -> dict[str, object]:
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the context scaling qualification requires CUDA")
    if args.require_sm75 and torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError("the context scaling qualification requires exact SM75")
    reports: list[dict[str, object]] = []
    for length in args.lengths:
        completed = subprocess.run(
            _benchmark_command(args, length),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"context benchmark failed at length {length}:\n"
                f"{completed.stderr}\n{completed.stdout}"
            )
        reports.append(json.loads(completed.stdout))

    root = Path(__file__).resolve().parent
    return {
        "schema_version": 1,
        "experiment": "fixed-token sparse exceptional shape scaling on SM75",
        "status": "prospective systems qualification; not model-quality evidence",
        "config": {
            "lengths": list(args.lengths),
            "token_budget": args.token_budget,
            "cycles": args.cycles,
            "warmups_per_cycle": args.warmups,
            "samples_per_cycle": args.samples,
            "d_model": args.d_model,
            "mamba_d_model": args.mamba_d_model,
            "layers": args.layers,
            "memory_width": args.memory_width,
            "update_rank": args.update_rank,
            "d_conv": args.d_conv,
            "learning_rate": args.learning_rate,
            "activation_checkpointing": args.activation_checkpointing,
            "seed": args.seed,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
        "source_sha256": {
            "benchmark_sparse_context_scaling.py": _sha256(Path(__file__)),
            "benchmark_sparse_action_cost.py": _sha256(
                root / "benchmark_sparse_action_cost.py"
            ),
        },
        "reports": reports,
        "verdict": _scaling_verdict(reports),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", nargs="+", type=int, default=DEFAULT_LENGTHS)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=126)
    parser.add_argument("--mamba-d-model", type=int, default=140)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--memory-width", type=int, default=8)
    parser.add_argument("--update-rank", type=int, default=2)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--require-sm75", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = _run(args)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(payload, end="")


if __name__ == "__main__":
    main()
