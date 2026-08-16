"""Replicate rigid motor identification across coordinates and schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from benchmark_spin_motor_rigid_2a5 import RigidSpinConfig
from identify_spin_motor_rigid_2a5 import run_identification

SCREENING_ARTIFACT_SHA256 = (
    "df132600d8be86505a4e5156b161a7e2ee33ce84dc4dba1dae26d3207653c62b"
)


def compact_run(report: dict[str, object]) -> dict[str, object]:
    """Retain falsifiers and provenance without duplicating full metric trees."""

    result = report["result"]
    metrics = result["final_relation_metrics"]
    audits = report["split_audit"]["evaluations"]
    identification = result["identification_audit"]
    return {
        "coordinate_label": report["coordinate_label"],
        "seed": report["config"]["seed"],
        "elapsed_wall_seconds": report["elapsed_wall_seconds"],
        "training_schedule_sha256": report["split_audit"]["training"][
            "input_group_pose_schedule_sha256"
        ],
        "training_audit_passed": report["split_audit"]["training"]["passed"],
        "evaluation_audits_passed": all(value["passed"] for value in audits.values()),
        "evaluation_schedule_sha256": {
            key: value["schedule_sha256"] for key, value in audits.items()
        },
        "evaluation_split_count": len(audits),
        "identification_gates": report["identification_gates"],
        "maximum_exact_token_quaternion_error_degrees": identification[
            "maximum_exact_token_quaternion_error_degrees"
        ],
        "maximum_exact_token_translation_error": identification[
            "maximum_exact_token_translation_error"
        ],
        "minimum_signed_hemisphere_accuracy": min(
            value["signed_hemisphere_accuracy"] for value in metrics.values()
        ),
        "minimum_signed_rotation_threshold_accuracy": min(
            value["signed_rotation_threshold_accuracy"] for value in metrics.values()
        ),
        "minimum_translation_threshold_accuracy": min(
            value["translation_threshold_accuracy"] for value in metrics.values()
        ),
        "minimum_joint_signed_pose_accuracy": min(
            value["joint_signed_pose_accuracy"] for value in metrics.values()
        ),
        "minimum_paired_double_cover_pose_accuracy": min(
            value["paired_double_cover_pose_accuracy"] for value in metrics.values()
        ),
        "maximum_mean_signed_rotation_degrees": max(
            value["mean_signed_rotation_degrees"] for value in metrics.values()
        ),
        "maximum_mean_translation_l2": max(
            value["mean_translation_l2"] for value in metrics.values()
        ),
        "maximum_paired_mean_translation_difference": max(
            value["paired_mean_translation_difference"] for value in metrics.values()
        ),
        "checkpoint": result["checkpoint"],
        "checkpoint_sha256": result["checkpoint_sha256"],
    }


def run_replication(
    *,
    coordinates: tuple[str, ...],
    seeds: tuple[int, ...],
    device: torch.device,
    checkpoint_directory: Path,
) -> dict[str, object]:
    started = datetime.now(ZoneInfo("Africa/Johannesburg"))
    runs = []
    for coordinate in coordinates:
        for seed in seeds:
            checkpoint = checkpoint_directory / (
                f"identified_motor_coord-{coordinate}_seed{seed}.pt"
            )
            report = run_identification(
                replace(RigidSpinConfig(), seed=seed),
                coordinate_label=coordinate,
                device=device,
                checkpoint_path=checkpoint,
            )
            compact = compact_run(report)
            runs.append(compact)
            print(
                f"coord={coordinate} seed={seed} "
                f"all_passed={compact['identification_gates']['all_passed']} "
                f"max_translation={compact['maximum_mean_translation_l2']:.3e}",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finished = datetime.now(ZoneInfo("Africa/Johannesburg"))
    schedule_hashes = [run["training_schedule_sha256"] for run in runs]
    all_passed = all(
        run["training_audit_passed"]
        and run["evaluation_audits_passed"]
        and run["identification_gates"]["all_passed"]
        for run in runs
    )
    return {
        "schema_version": 1,
        "benchmark": "spin_motor_identification_replication",
        "status": "replicated finite deterministic identification result",
        "claim_boundary": (
            "Nine leak-free finite deterministic identification runs with "
            "every-prefix pose supervision; not a continuous-data, final-only, "
            "or end-to-end sequence-model superiority result."
        ),
        "screening_artifact_sha256": SCREENING_ARTIFACT_SHA256,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_wall_seconds": (finished - started).total_seconds(),
        "coordinates": list(coordinates),
        "seeds": list(seeds),
        "run_count": len(runs),
        "unique_training_schedule_hashes": len(set(schedule_hashes)),
        "all_runs_passed": all_passed,
        "aggregate": {
            "maximum_token_quaternion_error_degrees": max(
                run["maximum_exact_token_quaternion_error_degrees"] for run in runs
            ),
            "maximum_token_translation_error": max(
                run["maximum_exact_token_translation_error"] for run in runs
            ),
            "minimum_joint_signed_pose_accuracy": min(
                run["minimum_joint_signed_pose_accuracy"] for run in runs
            ),
            "minimum_paired_double_cover_pose_accuracy": min(
                run["minimum_paired_double_cover_pose_accuracy"] for run in runs
            ),
            "maximum_mean_signed_rotation_degrees": max(
                run["maximum_mean_signed_rotation_degrees"] for run in runs
            ),
            "maximum_mean_translation_l2": max(
                run["maximum_mean_translation_l2"] for run in runs
            ),
            "maximum_paired_mean_translation_difference": max(
                run["maximum_paired_mean_translation_difference"] for run in runs
            ),
        },
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
            "cuda_version": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "runs": runs,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates", nargs="+", default=("e", "a", "b"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    report = run_replication(
        coordinates=tuple(args.coordinates),
        seeds=tuple(args.seeds),
        device=device,
        checkpoint_directory=args.checkpoint_directory,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(rendered + "\n", encoding="utf-8")
    print(f"artifact_sha256={hashlib.sha256(args.artifact.read_bytes()).hexdigest()}")
    print(
        json.dumps(
            {key: report[key] for key in ("run_count", "all_runs_passed", "aggregate")},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
