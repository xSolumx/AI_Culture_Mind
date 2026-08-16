"""Noise-robustness audit for local rigid-motor identification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from benchmark_pure_rotor_2a5 import parameter_count
from benchmark_spin_motor_rigid_2a5 import (
    RigidRelationPairBatch,
    RigidSpinConfig,
    RigidTrainingBatch,
    _aggregate_gate_summary,
    _evaluate_all,
    identify_direct_motor_from_prefixes,
    make_relation_pair_batches,
    make_rigid_spin_task,
    make_training_batches,
    relation_pair_audit,
    training_split_audit,
)
from pure_rotor_ssm.spin_scan import quaternion_product, unit_quaternion

EXACT_REPLICATION_ARTIFACT_SHA256 = (
    "97ffc994889278b21da7482ff49a597d9799f4ac38472e43b459465468a00aa5"
)
NOISE_TIERS = {
    "clean": (0.0, 0.0),
    "low": (1.0, 0.01),
    "medium": (5.0, 0.05),
    "high": (15.0, 0.15),
}


def noisy_training_batches(
    training: list[RigidTrainingBatch],
    *,
    rotation_std_degrees: float,
    translation_std: float,
    noise_seed: int,
) -> tuple[list[RigidTrainingBatch], dict[str, float | str]]:
    """Apply independent signed-pose noise without antipodal flips."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(8_675_309 + noise_seed)
    noisy_batches = []
    angle_squares = []
    translation_squares = []
    digest = hashlib.sha256()
    for batch in training:
        true_pose = batch.pose_targets.double()
        true_q = true_pose[..., :4]
        axes = torch.randn(
            *true_q.shape[:-1], 3, generator=generator, dtype=torch.double
        )
        axes = torch.nn.functional.normalize(axes, dim=-1)
        angles = math.radians(rotation_std_degrees) * torch.randn(
            *true_q.shape[:-1], generator=generator, dtype=torch.double
        )
        half_angles = 0.5 * angles
        perturbation = torch.cat(
            (
                torch.cos(half_angles)[..., None],
                axes * torch.sin(half_angles)[..., None],
            ),
            dim=-1,
        )
        noisy_q = unit_quaternion(quaternion_product(true_q, perturbation))
        translation_noise = translation_std * torch.randn(
            *true_pose[..., 4:].shape, generator=generator, dtype=torch.double
        )
        noisy_pose = torch.cat(
            (noisy_q, true_pose[..., 4:] + translation_noise), dim=-1
        ).float()
        noisy_batches.append(
            RigidTrainingBatch(batch.inputs, batch.group_targets, noisy_pose)
        )
        angle_squares.append(angles.square().flatten())
        translation_squares.append(translation_noise.square().flatten())
        contiguous = noisy_pose.contiguous()
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.numpy().tobytes())
    all_angles = torch.cat(angle_squares)
    all_translation = torch.cat(translation_squares)
    return noisy_batches, {
        "requested_rotation_std_degrees": rotation_std_degrees,
        "requested_translation_std": translation_std,
        "realized_rotation_rms_degrees": math.degrees(
            math.sqrt(float(all_angles.mean()))
        ),
        "realized_translation_per_axis_rms": math.sqrt(float(all_translation.mean())),
        "noisy_pose_schedule_sha256": digest.hexdigest(),
    }


def compact_metrics(metrics: dict[str, dict[str, float | int]]) -> dict[str, float]:
    return {
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
    }


def run_audit(
    *,
    noise_seeds: tuple[int, ...],
    device: torch.device,
    checkpoint_directory: Path,
) -> dict[str, object]:
    started = datetime.now(ZoneInfo("Africa/Johannesburg"))
    config = RigidSpinConfig(seed=0)
    task = make_rigid_spin_task("e", config.translation_step)
    clean_training = make_training_batches(task, config)
    clean_training_audit = training_split_audit(task, clean_training)
    if not clean_training_audit["passed"]:
        raise RuntimeError("clean training audit failed")

    evaluations: dict[str, list[RigidRelationPairBatch]] = {}
    evaluation_hashes = {}
    for relation in task.rotation.relations:
        for length in config.evaluation_lengths:
            for position in ("early", "late"):
                key = f"{relation.key}__{position}_L{length}"
                batches = make_relation_pair_batches(
                    task, relation, config, length, position
                )
                audit = relation_pair_audit(task, relation, batches)
                if not audit["passed"]:
                    raise RuntimeError(f"evaluation audit failed for {key}")
                evaluations[key] = batches
                evaluation_hashes[key] = audit["schedule_sha256"]

    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    runs = []
    for tier, (rotation_std, translation_std) in NOISE_TIERS.items():
        for noise_seed in noise_seeds:
            noisy_training, noise_audit = noisy_training_batches(
                clean_training,
                rotation_std_degrees=rotation_std,
                translation_std=translation_std,
                noise_seed=noise_seed,
            )
            model, identification = identify_direct_motor_from_prefixes(
                task, noisy_training
            )
            model = model.to(device)
            metrics = _evaluate_all(
                "direct_motor_pose_scan",
                model,
                evaluations,
                device,
                config,
                spin_scan_mode="parallel",
                motor_scan_mode="parallel",
                delta_scan_mode="parallel",
            )
            gate = _aggregate_gate_summary(
                [
                    {
                        "name": "identified_direct_motor",
                        "final_relation_metrics": metrics,
                    }
                ]
            )["identified_direct_motor"]
            all_pair_gate = all(
                value["paired_double_cover_pose_accuracy"] >= 0.80
                for value in metrics.values()
            )
            checkpoint = checkpoint_directory / f"tier-{tier}_seed{noise_seed}.pt"
            torch.save(
                {
                    "format_version": 1,
                    "candidate": "noise_identified_direct_motor",
                    "noise_tier": tier,
                    "noise_seed": noise_seed,
                    "noise_audit": noise_audit,
                    "identification_audit": identification,
                    "clean_training_schedule_sha256": clean_training_audit[
                        "input_group_pose_schedule_sha256"
                    ],
                    "config": asdict(config),
                    "metrics": metrics,
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                },
                checkpoint,
            )
            run = {
                "tier": tier,
                "noise_seed": noise_seed,
                "parameters": parameter_count(model),
                "noise_audit": noise_audit,
                "maximum_exact_token_quaternion_error_degrees": identification[
                    "maximum_exact_token_quaternion_error_degrees"
                ],
                "maximum_exact_token_translation_error": identification[
                    "maximum_exact_token_translation_error"
                ],
                "metrics": compact_metrics(metrics),
                "long_center_gate_passed": gate["all_long_splits_center_gate_90pct"],
                "long_joint_pose_gate_passed": gate[
                    "all_long_splits_joint_pose_gate_80pct"
                ],
                "all_split_pair_gate_passed": all_pair_gate,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
            }
            runs.append(run)
            print(
                f"tier={tier} seed={noise_seed} "
                f"joint={run['metrics']['minimum_joint_signed_pose_accuracy']:.3f} "
                f"max_t={run['metrics']['maximum_mean_translation_l2']:.3e}",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    tier_summaries = {}
    for tier in NOISE_TIERS:
        selected = [run for run in runs if run["tier"] == tier]
        tier_summaries[tier] = {
            "run_count": len(selected),
            "all_center_gates_passed": all(
                run["long_center_gate_passed"] for run in selected
            ),
            "all_joint_pose_gates_passed": all(
                run["long_joint_pose_gate_passed"] for run in selected
            ),
            "all_pair_gates_passed": all(
                run["all_split_pair_gate_passed"] for run in selected
            ),
            "minimum_joint_signed_pose_accuracy": min(
                run["metrics"]["minimum_joint_signed_pose_accuracy"] for run in selected
            ),
            "minimum_paired_double_cover_pose_accuracy": min(
                run["metrics"]["minimum_paired_double_cover_pose_accuracy"]
                for run in selected
            ),
            "maximum_mean_signed_rotation_degrees": max(
                run["metrics"]["maximum_mean_signed_rotation_degrees"]
                for run in selected
            ),
            "maximum_mean_translation_l2": max(
                run["metrics"]["maximum_mean_translation_l2"] for run in selected
            ),
            "maximum_token_quaternion_error_degrees": max(
                run["maximum_exact_token_quaternion_error_degrees"] for run in selected
            ),
            "maximum_token_translation_error": max(
                run["maximum_exact_token_translation_error"] for run in selected
            ),
        }
    required_tiers_passed = all(
        tier_summaries[tier]["all_center_gates_passed"]
        and tier_summaries[tier]["all_joint_pose_gates_passed"]
        and tier_summaries[tier]["all_pair_gates_passed"]
        for tier in ("clean", "low", "medium")
    )
    finished = datetime.now(ZoneInfo("Africa/Johannesburg"))
    return {
        "schema_version": 1,
        "benchmark": "spin_motor_noisy_local_identification",
        "status": "signed-pose metric-noise audit",
        "claim_boundary": (
            "Independent metric noise with signed quaternion supervision; "
            "excludes antipodal flips, correlated drift, outliers, continuous "
            "inputs, and sparse/final-only supervision."
        ),
        "exact_replication_artifact_sha256": EXACT_REPLICATION_ARTIFACT_SHA256,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_wall_seconds": (finished - started).total_seconds(),
        "noise_tiers": {
            key: {
                "rotation_std_degrees": value[0],
                "translation_std": value[1],
            }
            for key, value in NOISE_TIERS.items()
        },
        "noise_seeds": list(noise_seeds),
        "run_count": len(runs),
        "clean_training_audit": clean_training_audit,
        "evaluation_schedule_sha256": evaluation_hashes,
        "required_clean_low_medium_tiers_passed": required_tiers_passed,
        "tier_summaries": tier_summaries,
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
    parser.add_argument("--noise-seeds", nargs="+", type=int, default=range(5))
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
    report = run_audit(
        noise_seeds=tuple(args.noise_seeds),
        device=device,
        checkpoint_directory=args.checkpoint_directory,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(rendered + "\n", encoding="utf-8")
    print(f"artifact_sha256={hashlib.sha256(args.artifact.read_bytes()).hexdigest()}")
    print(
        json.dumps(
            {
                "required_tiers_passed": report[
                    "required_clean_low_medium_tiers_passed"
                ],
                "tier_summaries": report["tier_summaries"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
