"""Run the separately frozen measured-wall continuation of the Spin(8) task.

The update counts in this file were fixed from the corrected seed-0 development
wall times before the fresh cohort was evaluated.  This runner precomputes one
maximal schedule and gives each candidate a deterministic prefix, so schedule
construction is outside the measured model-update interval and every shared
update index denotes byte-identical training data.

This is a hardware-specific follow-up.  It cannot rescue the same-update cohort
and does not claim hardware-independent compute matching or fused-Mamba parity.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

import benchmark_pure_spin8_continuous_observation as primary

PROTOCOL_FROZEN_AT = "2026-08-17T04:13:00+02:00"
ALLOCATION_FROZEN_AT = "2026-08-17T04:17:46+02:00"
REFERENCE_UPDATES = 800
FROZEN_WALL_UPDATES = {
    "shared_pure_spin8": 800,
    "independent_so8_triplet": 636,
    "mamba2_parameter_near": 1_134,
    "gru_parameter_near": 5_001,
    "observation_only_ablation": 6_604,
    "gru_state_matched": 5_005,
}

FACTORIES = {
    "shared_pure_spin8": primary.SharedPureSpin8Tracker,
    "independent_so8_triplet": primary.IndependentSO8TripletTracker,
    "mamba2_parameter_near": primary.ContinuousMamba2Tracker,
    "gru_parameter_near": primary.ParameterNearGRUTracker,
    "observation_only_ablation": primary.ObservationOnlyAblation,
    "gru_state_matched": primary.StateMatchedGRUTracker,
}

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_DIRECTORY = (
    ROOT / "checkpoints" / "pure_spin8_continuous_observation_wall_matched"
)


def default_primary_artifact(seed: int) -> Path:
    return (
        ROOT
        / "experiments"
        / "artifacts"
        / f"pure_spin8_continuous_observation_validation_seed{seed}.json"
    )


def default_output(seed: int) -> Path:
    return (
        ROOT
        / "experiments"
        / "artifacts"
        / f"pure_spin8_continuous_observation_wall_matched_seed{seed}.json"
    )


def _schedule_hash(schedule: list[primary.TrainingBatch]) -> str:
    return primary.tensor_hash(
        [
            value
            for batch in schedule
            for value in (
                batch.observations,
                batch.targets,
                batch.coordinates,
                batch.events,
            )
        ]
    )


def _numeric_max_abs(left: Any, right: Any) -> float:
    """Maximum absolute difference over matching numeric JSON leaves."""

    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return math.inf
        return max((_numeric_max_abs(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return math.inf
        return max((_numeric_max_abs(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else math.inf


def wall_alignment(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reference = float(results["shared_pure_spin8"]["training_wall_seconds"])
    rows = {}
    for name in primary.CANDIDATES:
        elapsed = float(results[name]["training_wall_seconds"])
        rows[name] = {
            "training_wall_seconds": elapsed,
            "ratio_to_shared": elapsed / reference,
            "absolute_relative_deviation_from_shared": abs(elapsed - reference)
            / reference,
        }
    nonreference = [
        rows[name]["absolute_relative_deviation_from_shared"]
        for name in primary.CANDIDATES
        if name != "shared_pure_spin8"
    ]
    return {
        "reference_candidate": "shared_pure_spin8",
        "reference_wall_seconds": reference,
        "rows": rows,
        "maximum_nonreference_relative_deviation": max(nonreference),
    }


def run_wall_matched(
    config: primary.ContinuousObservationConfig,
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
        raise RuntimeError("primary artifact seed does not match requested seed")
    if source["status"] != "unadjudicated":
        raise RuntimeError("source must be an untouched primary validation artifact")

    contract = primary.teacher_contract(device)
    if not contract["passed"]:
        raise RuntimeError("teacher relation contract failed")
    system = primary.make_observation_system(config.seed)
    maximal_config = replace(config, steps=max(FROZEN_WALL_UPDATES.values()))
    schedule = primary.make_training_schedule(maximal_config, system, device)
    full_split = primary.training_split_audit(schedule)
    if not full_split["passed"]:
        raise RuntimeError("maximal measured-wall training split audit failed")

    prefix = schedule[:REFERENCE_UPDATES]
    prefix_hash = _schedule_hash(prefix)
    expected_prefix_hash = source["task"]["training_split"]["schedule_sha256"]

    evaluations = {}
    evaluation_audits = {}
    evaluation_hashes = {}
    for length in config.evaluation_lengths:
        for position in ("early", "late"):
            key = f"{position}_L{length}"
            batch = primary.make_relation_batch(config, system, length, position, device)
            evaluations[key] = [batch]
            evaluation_audits[key] = primary.relation_batch_audit(batch)
            evaluation_hashes[key] = primary.tensor_hash(
                (
                    batch.observations,
                    batch.targets,
                    batch.coordinates,
                    batch.post_relation_mask,
                )
            )
    if not all(audit["passed"] for audit in evaluation_audits.values()):
        raise RuntimeError("measured-wall evaluation split audit failed")

    shapes = primary.build_models()
    counts = {name: primary.parameter_count(model) for name, model in shapes.items()}
    states = {name: int(model.recurrent_state_scalars) for name, model in shapes.items()}
    del shapes

    results = {}
    for offset, name in enumerate(primary.CANDIDATES):
        steps = FROZEN_WALL_UPDATES[name]
        candidate_config = replace(config, steps=steps)
        primary.seed_everything(860_000 + 1_000 * config.seed + offset)
        results[name] = primary.train_candidate(
            name,
            FACTORIES[name],
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
        "candidate_set_exact": set(results) == set(primary.CANDIDATES),
        "allocation_exact": {
            name: results[name]["loss_samples"].get(str(FROZEN_WALL_UPDATES[name]))
            is not None
            for name in primary.CANDIDATES
        },
        "prefix_hash_matches_primary": prefix_hash == expected_prefix_hash,
        "observation_system_matches_primary": primary.tensor_hash(
            (system.projection, system.bias)
        )
        == source["task"]["observation_system_sha256"],
        "evaluation_hashes_match_primary": evaluation_hashes
        == source["task"]["evaluation_schedule_sha256"],
        "shared_800_update_replay_exact": shared_replay_max_abs == 0.0,
        "full_training_split_passed": full_split["passed"],
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
    allocation_checks = checks["allocation_exact"]
    integrity_passed = all(
        value
        for key, value in checks.items()
        if key != "allocation_exact"
    ) and all(allocation_checks.values())

    return {
        "schema_version": 1,
        "experiment": "Pure Spin8 continuous-observation measured-wall continuation",
        "status": "unadjudicated measured-wall continuation",
        "protocol_frozen_at": PROTOCOL_FROZEN_AT,
        "protocol_split_correction_at": primary.PROTOCOL_SPLIT_CORRECTION_AT,
        "allocation_frozen_at": ALLOCATION_FROZEN_AT,
        "recorded_at": primary.now(),
        "pure_spin8_version": primary.PURE_SPIN8_VERSION,
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
            "round(800 * corrected_seed0_shared_wall / corrected_seed0_candidate_wall)"
        ),
        "source_primary_artifact": str(primary_artifact_path),
        "source_primary_artifact_sha256": primary.hashlib.sha256(
            primary_artifact_path.read_bytes()
        ).hexdigest(),
        "task": {
            "maximal_training_split": full_split,
            "primary_800_update_prefix_sha256": prefix_hash,
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
                "hardware-specific continuation using a pre-frozen update allocation",
                "same-prefix continuous-observation training with model-update wall timing",
            ],
            "not_claimed": [
                "a rescue of the primary same-update result",
                "hardware-independent compute matching",
                "FLOP, energy, kernel, or fused-training equality",
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
    config = primary.ContinuousObservationConfig(
        steps=REFERENCE_UPDATES,
        batch_size=args.batch_size,
        training_length=args.training_length,
        evaluation_pairs=args.evaluation_pairs,
        evaluation_lengths=primary.parse_lengths(args.evaluation_lengths),
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
        f"max_wall_deviation="
        f"{report['wall_alignment']['maximum_nonreference_relative_deviation']:.6f}"
    )
    return 0 if report["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
