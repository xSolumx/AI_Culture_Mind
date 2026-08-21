"""Run the frozen measured-wall continuation of endpoint-only Spin(8).

The update counts were fixed from the corrected seed-0 development walls before
fresh endpoint-only validation.  One maximal endpoint-only schedule is built per
seed and every candidate receives a deterministic prefix.  Schedule generation
is outside the measured model-update interval.

This hardware-specific continuation cannot rescue or replace the equal-update
cohort and does not claim FLOP, energy, fused-kernel, or hardware-independent
compute equality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

import benchmark_pure_spin8_continuous_observation as continuous
import benchmark_pure_spin8_endpoint_supervision as endpoint
from benchmark_pure_spin8_continuous_wall_matched import (
    _numeric_max_abs,
    wall_alignment,
)

PROTOCOL_FROZEN_AT = endpoint.PROTOCOL_FROZEN_AT
ALLOCATION_FROZEN_AT = endpoint.PROTOCOL_FROZEN_AT
REFERENCE_UPDATES = 2_000
FROZEN_WALL_UPDATES = {
    "shared_pure_spin8": 2_000,
    "independent_so8_triplet": 1_558,
    "mamba2_parameter_near": 2_811,
    "gru_parameter_near": 11_907,
    "observation_only_ablation": 15_482,
    "gru_state_matched": 11_911,
}

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_DIRECTORY = (
    ROOT / "checkpoints" / "pure_spin8_endpoint_supervision_wall_matched"
)


def default_primary_artifact(seed: int) -> Path:
    return (
        ROOT
        / "experiments"
        / "artifacts"
        / f"pure_spin8_endpoint_supervision_validation_seed{seed}.json"
    )


def default_output(seed: int) -> Path:
    return (
        ROOT
        / "experiments"
        / "artifacts"
        / f"pure_spin8_endpoint_supervision_wall_matched_seed{seed}.json"
    )


def _schedule_hash(schedule: list[endpoint.EndpointTrainingBatch]) -> str:
    return continuous.tensor_hash(
        [
            value
            for batch in schedule
            for value in (
                batch.observations,
                batch.endpoint_targets,
                batch.coordinates,
                batch.events,
            )
        ]
    )


def run_wall_matched(
    config: continuous.ContinuousObservationConfig,
    *,
    device: torch.device,
    primary_artifact_path: Path,
    checkpoint_directory: Path | None,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    source = json.loads(primary_artifact_path.read_text(encoding="utf-8"))
    if source["config"]["seed"] != config.seed:
        raise RuntimeError("primary endpoint artifact seed does not match")
    if source["status"] != "unadjudicated":
        raise RuntimeError("source must be an untouched primary validation artifact")
    if source["protocol_frozen_at"] != PROTOCOL_FROZEN_AT:
        raise RuntimeError("source primary protocol freeze does not match")

    contract = continuous.teacher_contract(device)
    if not contract["passed"]:
        raise RuntimeError("teacher relation contract failed")
    system = continuous.make_observation_system(config.seed)
    maximal_config = replace(config, steps=max(FROZEN_WALL_UPDATES.values()))
    schedule = endpoint.make_endpoint_training_schedule(
        maximal_config, system, device
    )
    full_split = endpoint.endpoint_training_split_audit(schedule)
    if not full_split["passed"]:
        raise RuntimeError("maximal endpoint-only training split audit failed")

    prefix_hash = _schedule_hash(schedule[:REFERENCE_UPDATES])
    expected_prefix_hash = source["task"]["training_split"]["schedule_sha256"]

    evaluations = {}
    evaluation_audits = {}
    evaluation_hashes = {}
    for length in config.evaluation_lengths:
        for position in ("early", "late"):
            key = f"{position}_L{length}"
            batch = continuous.make_relation_batch(
                config, system, length, position, device
            )
            evaluations[key] = [batch]
            evaluation_audits[key] = continuous.relation_batch_audit(batch)
            evaluation_hashes[key] = continuous.tensor_hash(
                (
                    batch.observations,
                    batch.targets,
                    batch.coordinates,
                    batch.post_relation_mask,
                )
            )
    if not all(audit["passed"] for audit in evaluation_audits.values()):
        raise RuntimeError("measured-wall evaluation split audit failed")

    shapes = continuous.build_models()
    counts = {
        name: continuous.parameter_count(model) for name, model in shapes.items()
    }
    states = {
        name: int(model.recurrent_state_scalars) for name, model in shapes.items()
    }
    del shapes

    results = {}
    for offset, name in enumerate(continuous.CANDIDATES):
        steps = FROZEN_WALL_UPDATES[name]
        candidate_config = replace(config, steps=steps)
        continuous.seed_everything(960_000 + 1_000 * config.seed + offset)
        results[name] = endpoint.train_endpoint_candidate(
            name,
            endpoint.FACTORIES[name],
            schedule[:steps],
            evaluations,
            candidate_config,
            device,
            checkpoint_directory,
        )

    source_shared = source["results"]["shared_pure_spin8"]
    shared_replay_max_abs = max(
        _numeric_max_abs(
            results["shared_pure_spin8"]["evaluation"],
            source_shared["evaluation"],
        ),
        _numeric_max_abs(
            results["shared_pure_spin8"]["action_identification"],
            source_shared["action_identification"],
        ),
        abs(
            results["shared_pure_spin8"]["final_training_loss"]
            - source_shared["final_training_loss"]
        ),
    )
    checks = {
        "candidate_set_exact": set(results) == set(continuous.CANDIDATES),
        "allocation_exact": {
            name: str(FROZEN_WALL_UPDATES[name])
            in results[name]["loss_samples"]
            for name in continuous.CANDIDATES
        },
        "prefix_hash_matches_primary": prefix_hash == expected_prefix_hash,
        "observation_system_matches_primary": continuous.tensor_hash(
            (system.projection, system.bias)
        )
        == source["task"]["observation_system_sha256"],
        "evaluation_hashes_match_primary": evaluation_hashes
        == source["task"]["evaluation_schedule_sha256"],
        "shared_2000_update_replay_exact": shared_replay_max_abs == 0.0,
        "full_training_split_passed": full_split["passed"],
        "no_intermediate_targets_retained": (
            full_split["retained_intermediate_target_count"] == 0
            and full_split["supervised_scalars_per_sequence"] == 24
            and full_split["checks"][
                "no_intermediate_targets_in_training_batch"
            ]
        ),
        "all_evaluation_splits_passed": all(
            audit["passed"] for audit in evaluation_audits.values()
        ),
        "all_metrics_finite": all(
            math.isfinite(value)
            for result in results.values()
            for metrics in result["evaluation"].values()
            for value in metrics.values()
        ),
    }
    integrity_passed = all(
        value
        for key, value in checks.items()
        if key != "allocation_exact"
    ) and all(checks["allocation_exact"].values())

    return {
        "schema_version": 1,
        "experiment": "Pure Spin8 endpoint-only measured-wall continuation",
        "status": "unadjudicated endpoint-only measured-wall continuation",
        "protocol_frozen_at": PROTOCOL_FROZEN_AT,
        "allocation_frozen_at": ALLOCATION_FROZEN_AT,
        "recorded_at": continuous.now(),
        "pure_spin8_version": continuous.PURE_SPIN8_VERSION,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor()
            ),
            "mamba2_backend": "huggingface_transformers_naive_fallback",
            "torch_cpu_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "config": asdict(config),
        "frozen_update_allocation": FROZEN_WALL_UPDATES,
        "allocation_basis": (
            "round(2000 * corrected_seed0_shared_wall / "
            "corrected_seed0_candidate_wall)"
        ),
        "source_primary_artifact": str(primary_artifact_path),
        "source_primary_artifact_sha256": hashlib.sha256(
            primary_artifact_path.read_bytes()
        ).hexdigest(),
        "task": {
            "training_supervision": "final signed 24-real triality state only",
            "intermediate_prefix_targets_retained_by_training_schedule": False,
            "maximal_training_split": full_split,
            "primary_2000_update_prefix_sha256": prefix_hash,
            "evaluation_audits": evaluation_audits,
            "evaluation_schedule_sha256": evaluation_hashes,
        },
        "integrity": {
            "checks": checks,
            "shared_replay_max_abs": shared_replay_max_abs,
            "parameter_counts": counts,
            "recurrent_state_scalars": states,
            "one_precomputed_maximal_schedule": True,
            "candidate_data_are_prefixes_of_same_schedule": True,
            "schedule_generation_excluded_from_training_wall": True,
            "passed": integrity_passed,
        },
        "wall_alignment": wall_alignment(results),
        "results": results,
        "claim_scope": {
            "empirical": [
                "hardware-specific endpoint-only continuation using a pre-frozen update allocation",
                "same-prefix model-update wall timing on an RTX 2070 SUPER",
            ],
            "not_claimed": [
                "a rescue or replacement of the primary equal-update cohort",
                "hardware-independent compute, FLOP, energy, or kernel equality",
                "fused-Mamba parity, natural-data utility, or language-model superiority",
            ],
        },
        "integrity_passed": integrity_passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-length", type=int, default=16)
    parser.add_argument("--evaluation-pairs", type=int, default=64)
    parser.add_argument("--evaluation-lengths", default="16,64,128")
    parser.add_argument("--evaluation-microbatch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--observation-noise-std", type=float, default=0.01)
    parser.add_argument("--half-center-probability", type=float, default=0.12)
    parser.add_argument("--regular-coordinate-std", type=float, default=0.40)
    parser.add_argument("--half-center-delta", type=float, default=0.25)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--primary-artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIRECTORY,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = continuous.ContinuousObservationConfig(
        steps=REFERENCE_UPDATES,
        batch_size=args.batch_size,
        training_length=args.training_length,
        evaluation_pairs=args.evaluation_pairs,
        evaluation_lengths=continuous.parse_lengths(args.evaluation_lengths),
        evaluation_microbatch_size=args.evaluation_microbatch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        observation_noise_std=args.observation_noise_std,
        half_center_probability=args.half_center_probability,
        regular_coordinate_std=args.regular_coordinate_std,
        half_center_delta=args.half_center_delta,
        seed=args.seed,
    )
    source = args.primary_artifact or default_primary_artifact(args.seed)
    output = args.output or default_output(args.seed)
    report = run_wall_matched(
        config,
        device=torch.device(args.device),
        primary_artifact_path=source,
        checkpoint_directory=args.checkpoint_directory,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")
    print(
        f"wrote {output} integrity_passed={report['integrity_passed']} "
        "max_wall_deviation="
        f"{report['wall_alignment']['maximum_nonreference_relative_deviation']:.6f}"
    )
    return 0 if report["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
