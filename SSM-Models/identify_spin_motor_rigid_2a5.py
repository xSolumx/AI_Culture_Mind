"""Leak-free local transition identification for the rigid ``2.A5`` motor."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from benchmark_pure_rotor_2a5 import parameter_count
from benchmark_spin_motor_rigid_2a5 import (
    RigidSpinConfig,
    _aggregate_gate_summary,
    _evaluate_all,
    identify_direct_motor_from_prefixes,
    make_relation_pair_batches,
    make_rigid_spin_task,
    make_training_batches,
    recurrent_state_scalars,
    relation_pair_audit,
    training_split_audit,
)

PARENT_ARTIFACT_SHA256 = (
    "7a364b61ba51666db65f0ced909fc78d81855582fd14e9dd5e598d2d4d3ab1f2"
)


def run_identification(
    config: RigidSpinConfig,
    *,
    coordinate_label: str,
    device: torch.device,
    checkpoint_path: Path | None,
) -> dict[str, object]:
    """Identify from legal prefixes, evaluate, and optionally checkpoint."""

    started = datetime.now(ZoneInfo("Africa/Johannesburg"))
    start_clock = time.perf_counter()
    task = make_rigid_spin_task(coordinate_label, config.translation_step)
    training = make_training_batches(task, config)
    training_audit = training_split_audit(task, training)
    if not training_audit["passed"]:
        raise RuntimeError("training split audit failed")
    model, identification_audit = identify_direct_motor_from_prefixes(task, training)
    identification_seconds = time.perf_counter() - start_clock

    evaluations = {}
    evaluation_audits = {}
    for relation in task.rotation.relations:
        for length in config.evaluation_lengths:
            for position in ("early", "late"):
                key = f"{relation.key}__{position}_L{length}"
                batches = make_relation_pair_batches(
                    task, relation, config, length, position
                )
                audit = relation_pair_audit(task, relation, batches)
                if not audit["passed"]:
                    raise RuntimeError(f"evaluation split audit failed for {key}")
                evaluations[key] = batches
                evaluation_audits[key] = audit

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
    result: dict[str, object] = {
        "name": "identified_direct_motor",
        "parameters": parameter_count(model),
        "recurrent_state_scalars": recurrent_state_scalars(config)[
            "direct_motor_pose_scan"
        ],
        "identification_seconds": identification_seconds,
        "identification_audit": identification_audit,
        "final_relation_metrics": metrics,
    }
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "candidate": "identified_direct_motor",
                "benchmark": "spin_motor_rigid_2a5_local_identification",
                "coordinate_label": coordinate_label,
                "group_table_sha256": task.rotation.binary.group_table_sha256,
                "task_presentation": task.rotation.binary.presentation,
                "translation_tokens": task.token_translations,
                "benchmark_config": asdict(config),
                "training_schedule_sha256": training_audit[
                    "input_group_pose_schedule_sha256"
                ],
                "identification_audit": identification_audit,
                "metrics": metrics,
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
            },
            checkpoint_path,
        )
        result["checkpoint"] = str(checkpoint_path)
        result["checkpoint_sha256"] = hashlib.sha256(
            checkpoint_path.read_bytes()
        ).hexdigest()

    gate_summary = _aggregate_gate_summary([result])["identified_direct_motor"]
    all_split_pair_gate = all(
        value["paired_double_cover_pose_accuracy"] >= 0.80 for value in metrics.values()
    )
    identification_gates = {
        "all_tokens_observed": all(
            value["observations"] > 0
            for value in identification_audit["per_token"].values()
        ),
        "token_quaternion_error_below_0p1_degrees": identification_audit[
            "maximum_exact_token_quaternion_error_degrees"
        ]
        < 0.1,
        "token_translation_error_below_1e-6": identification_audit[
            "maximum_exact_token_translation_error"
        ]
        < 1e-6,
        "all_long_splits_center_gate_90pct": gate_summary[
            "all_long_splits_center_gate_90pct"
        ],
        "all_long_splits_joint_pose_gate_80pct": gate_summary[
            "all_long_splits_joint_pose_gate_80pct"
        ],
        "all_splits_paired_double_cover_pose_gate_80pct": all_split_pair_gate,
    }
    identification_gates["all_passed"] = all(identification_gates.values())
    finished = datetime.now(ZoneInfo("Africa/Johannesburg"))
    return {
        "schema_version": 1,
        "benchmark": "spin_motor_rigid_2a5_local_identification",
        "status": "single-coordinate deterministic identification audit",
        "claim_boundary": (
            "Leak-free realizability and identifiability result for a deterministic "
            "token task with every-prefix pose supervision; not an end-to-end "
            "learning comparison or general continuous-observation theorem."
        ),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_wall_seconds": (finished - started).total_seconds(),
        "parent_gradient_run_artifact_sha256": PARENT_ARTIFACT_SHA256,
        "config": asdict(config),
        "coordinate_label": coordinate_label,
        "task": {
            "input_symbols": list(task.input_symbols),
            "input_elements": list(task.input_elements),
            "token_translations": task.token_translations,
            "relations": [asdict(relation) for relation in task.rotation.relations],
            "group_table_sha256": task.rotation.binary.group_table_sha256,
        },
        "split_audit": {
            "training": training_audit,
            "evaluations": evaluation_audits,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_version": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "result": result,
        "gate_summary": gate_summary,
        "identification_gates": identification_gates,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--coordinate", choices=("e", "a", "b"), default="e")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    report = run_identification(
        replace(RigidSpinConfig(), steps=args.steps, seed=args.seed),
        coordinate_label=args.coordinate,
        device=device,
        checkpoint_path=args.checkpoint,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.artifact is not None:
        args.artifact.parent.mkdir(parents=True, exist_ok=True)
        args.artifact.write_text(rendered + "\n", encoding="utf-8")
        print(
            f"artifact_sha256={hashlib.sha256(args.artifact.read_bytes()).hexdigest()}"
        )
    print(rendered)


if __name__ == "__main__":
    main()
