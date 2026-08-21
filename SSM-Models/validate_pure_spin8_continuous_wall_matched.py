"""Adjudicate the three-seed measured-wall Spin(8) continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

import benchmark_pure_spin8_continuous_observation as primary
from benchmark_pure_spin8_continuous_wall_matched import (
    ALLOCATION_FROZEN_AT,
    FACTORIES,
    FROZEN_WALL_UPDATES,
    PROTOCOL_FROZEN_AT,
)

ROOT = Path(__file__).resolve().parent
EXPECTED_SEEDS = (1, 2, 3)
EXPECTED_CONFIG = {
    "batch_size": 32,
    "evaluation_lengths": [16, 64, 128],
    "evaluation_microbatch_size": 32,
    "evaluation_pairs": 64,
    "gradient_clip": 1.0,
    "half_center_delta": 0.25,
    "half_center_probability": 0.12,
    "learning_rate": 0.003,
    "observation_noise_std": 0.01,
    "regular_coordinate_std": 0.4,
    "steps": 800,
    "training_length": 16,
    "weight_decay": 0.0001,
}
EXPECTED_PARAMETER_COUNTS = {
    "shared_pure_spin8": 930,
    "independent_so8_triplet": 957,
    "mamba2_parameter_near": 931,
    "gru_parameter_near": 960,
    "observation_only_ablation": 949,
    "gru_state_matched": 3312,
}
EXPECTED_STATES = {
    "shared_pure_spin8": 24,
    "independent_so8_triplet": 24,
    "mamba2_parameter_near": 160,
    "gru_parameter_near": 10,
    "observation_only_ablation": 0,
    "gru_state_matched": 24,
}
MAXIMAL_OBSERVATION_COUNT = (
    max(FROZEN_WALL_UPDATES.values())
    * EXPECTED_CONFIG["batch_size"]
    * EXPECTED_CONFIG["training_length"]
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def all_numeric_values_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(all_numeric_values_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(all_numeric_values_finite(item) for item in value)
    return False


def reload_checkpoint(name: str, record: dict[str, Any], seed: int) -> bool:
    path = resolve_recorded_path(record["checkpoint"])
    if not path.is_file() or sha256(path) != record["checkpoint_sha256"]:
        return False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    if (
        payload.get("candidate") != name
        or config.get("seed") != seed
        or config.get("steps") != FROZEN_WALL_UPDATES[name]
    ):
        return False
    model = FACTORIES[name]()
    model.load_state_dict(payload["state_dict"], strict=True)
    return True


def validate_seed(path: Path, expected_seed: int) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    config = dict(report["config"])
    seed = config.pop("seed")
    results = report["results"]
    split = report["task"]["maximal_training_split"]
    source_path = resolve_recorded_path(report["source_primary_artifact"])
    source_exists = source_path.is_file()
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_exists else {}

    checkpoint_checks = {
        name: reload_checkpoint(name, record, expected_seed)
        for name, record in results.items()
    }
    allocation_result_checks = {
        name: (
            str(FROZEN_WALL_UPDATES[name]) in results[name]["loss_samples"]
            and results[name]["parameters"] == EXPECTED_PARAMETER_COUNTS[name]
            and results[name]["recurrent_state_scalars"] == EXPECTED_STATES[name]
        )
        for name in primary.CANDIDATES
    }
    wall_rows = report["wall_alignment"]["rows"]
    wall_consistency = all(
        math.isclose(
            wall_rows[name]["training_wall_seconds"],
            results[name]["training_wall_seconds"],
            rel_tol=0.0,
            abs_tol=0.0,
        )
        for name in primary.CANDIDATES
    )
    evaluation_source_match = bool(source_exists) and report["task"][
        "evaluation_schedule_sha256"
    ] == source["task"]["evaluation_schedule_sha256"]
    prefix_source_match = bool(source_exists) and report["task"][
        "primary_800_update_prefix_sha256"
    ] == source["task"]["training_split"]["schedule_sha256"]

    checks = {
        "seed_exact": seed == expected_seed,
        "config_exact": config == EXPECTED_CONFIG,
        "status_exact": report["status"]
        == "unadjudicated measured-wall continuation",
        "protocol_freeze_exact": report["protocol_frozen_at"]
        == PROTOCOL_FROZEN_AT,
        "split_correction_exact": report["protocol_split_correction_at"]
        == primary.PROTOCOL_SPLIT_CORRECTION_AT,
        "allocation_freeze_exact": report["allocation_frozen_at"]
        == ALLOCATION_FROZEN_AT,
        "allocation_exact": report["frozen_update_allocation"]
        == FROZEN_WALL_UPDATES,
        "candidate_set_exact": set(results) == set(primary.CANDIDATES),
        "parameter_counts_exact": report["integrity"]["parameter_counts"]
        == EXPECTED_PARAMETER_COUNTS,
        "state_counts_exact": report["integrity"]["recurrent_state_scalars"]
        == EXPECTED_STATES,
        "allocation_results_exact": all(allocation_result_checks.values()),
        "runner_integrity_passed": report["integrity_passed"]
        and report["integrity"]["passed"]
        and all(report["integrity"]["checks"]["allocation_exact"].values()),
        "shared_replay_exact": report["integrity"]["shared_replay_max_abs"] == 0.0,
        "maximal_split_passed": split["passed"],
        "maximal_observation_count_exact": split["observation_count"]
        == MAXIMAL_OBSERVATION_COUNT,
        "every_maximal_observation_unique": split["unique_observation_count"]
        == MAXIMAL_OBSERVATION_COUNT,
        "heldout_relation_absent_from_training": split[
            "held_out_adjacent_half_center_count"
        ]
        == 0,
        "source_primary_exists": source_exists,
        "source_primary_rehashes": source_exists
        and sha256(source_path) == report["source_primary_artifact_sha256"],
        "source_primary_seed_exact": source_exists
        and source.get("config", {}).get("seed") == expected_seed,
        "prefix_matches_source_primary": prefix_source_match,
        "evaluation_matches_source_primary": evaluation_source_match,
        "wall_rows_consistent": wall_consistency,
        "all_metrics_finite": all_numeric_values_finite(report),
        "all_checkpoints_rehash_and_reload": all(checkpoint_checks.values()),
    }
    return {
        "seed": expected_seed,
        "source": str(path),
        "source_sha256": sha256(path),
        "checks": checks,
        "checkpoint_checks": checkpoint_checks,
        "allocation_result_checks": allocation_result_checks,
        "passed": all(checks.values()),
        "maximal_schedule_sha256": split["schedule_sha256"],
        "evaluation_schedule_sha256": report["task"][
            "evaluation_schedule_sha256"
        ],
        "source_primary_artifact_sha256": report["source_primary_artifact_sha256"],
        "wall_alignment": report["wall_alignment"],
        "results": results,
    }


def aggregate(paths: list[Path]) -> dict[str, Any]:
    if len(paths) != len(EXPECTED_SEEDS):
        raise ValueError(f"expected {len(EXPECTED_SEEDS)} artifacts")
    seeds = [validate_seed(path, seed) for path, seed in zip(paths, EXPECTED_SEEDS)]
    schedule_hashes = {seed["maximal_schedule_sha256"] for seed in seeds}
    evaluation_hashes = {
        value
        for seed in seeds
        for value in seed["evaluation_schedule_sha256"].values()
    }
    primary_hashes = {seed["source_primary_artifact_sha256"] for seed in seeds}
    checkpoint_count = sum(len(seed["checkpoint_checks"]) for seed in seeds)

    l128_by_candidate = {
        name: [
            seed["results"][name]["evaluation"][key]["post_relation_mse"]
            for seed in seeds
            for key in ("early_L128", "late_L128")
        ]
        for name in primary.CANDIDATES
    }
    wall_deviations = [
        seed["wall_alignment"]["maximum_nonreference_relative_deviation"]
        for seed in seeds
    ]
    shared_beats_all = all(
        seed["results"]["shared_pure_spin8"]["evaluation"][key][
            "post_relation_mse"
        ]
        < seed["results"][name]["evaluation"][key]["post_relation_mse"]
        for seed in seeds
        for key in ("early_L128", "late_L128")
        for name in primary.CANDIDATES
        if name != "shared_pure_spin8"
    )
    cohort_checks = {
        "all_three_seed_integrity_reports_pass": all(seed["passed"] for seed in seeds),
        "all_18_checkpoints_rehash_and_reload": checkpoint_count == 18
        and all(
            value
            for seed in seeds
            for value in seed["checkpoint_checks"].values()
        ),
        "maximal_training_schedules_distinct": len(schedule_hashes) == 3,
        "all_evaluation_schedules_distinct": len(evaluation_hashes) == 18,
        "source_primary_artifacts_distinct": len(primary_hashes) == 3,
    }
    return {
        "schema_version": 1,
        "experiment": "Pure Spin8 continuous-observation measured-wall continuation",
        "status": "three-seed measured-wall adjudication",
        "protocol_frozen_at": PROTOCOL_FROZEN_AT,
        "protocol_split_correction_at": primary.PROTOCOL_SPLIT_CORRECTION_AT,
        "allocation_frozen_at": ALLOCATION_FROZEN_AT,
        "recorded_at": datetime.now().astimezone().isoformat(),
        "fresh_seeds": list(EXPECTED_SEEDS),
        "frozen_update_allocation": FROZEN_WALL_UPDATES,
        "cohort_checks": cohort_checks,
        "aggregate": {
            "maximum_wall_deviation_range": [
                min(wall_deviations),
                max(wall_deviations),
            ],
            "maximum_wall_deviation_median": statistics.median(wall_deviations),
            "l128_post_relation_mse": {
                name: {
                    "range": [min(values), max(values)],
                    "median": statistics.median(values),
                }
                for name, values in l128_by_candidate.items()
            },
            "independent_to_shared_median_mse_ratio": (
                statistics.median(l128_by_candidate["independent_so8_triplet"])
                / statistics.median(l128_by_candidate["shared_pure_spin8"])
            ),
            "observed_shared_beats_every_row_on_every_l128_split": shared_beats_all,
        },
        "seed_reports": seeds,
        "claim_scope": {
            "empirical": [
                "three fresh seeds under the separately frozen RTX 2070 SUPER update allocation",
                "model-update wall matching with identical data prefixes per seed",
            ],
            "not_claimed": [
                "a rescue or replacement of the primary same-update cohort",
                "hardware-independent compute, FLOP, energy, or kernel equality",
                "fused-Mamba parity, natural-data utility, or language-model superiority",
                "a theorem about Spin8, triality, Mamba-2, or global optimality",
            ],
        },
        "passed": all(cohort_checks.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = aggregate(args.artifacts)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(
        f"wrote {args.output} passed={report['passed']} "
        f"max_wall_deviation="
        f"{report['aggregate']['maximum_wall_deviation_range'][1]:.6f}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
