"""Adjudicate the frozen endpoint-only continuous Spin(8) cohort."""

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

import benchmark_pure_spin8_continuous_observation as continuous
from benchmark_pure_spin8_endpoint_supervision import (
    FACTORIES,
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
    "steps": 2000,
    "training_length": 16,
    "weight_decay": 0.0001,
}
EXPECTED_PARAMETERS = {
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
THRESHOLDS = {
    "shared_action_rmse": 0.03,
    "independent_action_rmse": 0.12,
    "shared_l128_post_relation_mse": 0.04,
    "shared_l128_signature_rmse": 0.05,
    "independent_l128_post_relation_mse": 0.15,
    "independent_l128_classification": 0.95,
    "independent_l128_row_correctness": 0.90,
    "shared_relation_action_rmse": 0.04,
    "independent_relation_action_rmse": 0.08,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_path(value: str) -> Path:
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
    path = checkpoint_path(record["checkpoint"])
    if not path.is_file() or sha256(path) != record["checkpoint_sha256"]:
        return False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    if (
        payload.get("candidate") != name
        or payload.get("training_supervision")
        != "endpoint_only_signed_triality_state"
        or config.get("seed") != seed
        or config.get("steps") != EXPECTED_CONFIG["steps"]
    ):
        return False
    model = FACTORIES[name]()
    model.load_state_dict(payload["state_dict"], strict=True)
    return True


def _maximum_relation_residual(result: dict[str, Any]) -> float:
    return max(
        value
        for residuals in result["action_identification"][
            "relation_action_residuals"
        ].values()
        for value in residuals.values()
    )


def seed_gate_checks(report: dict[str, Any]) -> dict[str, bool]:
    results = report["results"]
    shared = results["shared_pure_spin8"]
    independent = results["independent_so8_triplet"]
    split = report["task"]["training_split"]
    checks = {
        "runner_passed": report["passed"],
        "teacher_contract_passed": report["task"]["teacher_contract"]["passed"],
        "training_split_passed": split["passed"],
        "no_intermediate_targets_retained": (
            report["task"]["intermediate_prefix_targets_retained_by_training_schedule"]
            is False
            and split["retained_intermediate_target_count"] == 0
            and split["supervised_scalars_per_sequence"] == 24
            and split["checks"]["no_intermediate_targets_in_training_batch"]
        ),
        "all_evaluation_splits_passed": all(
            audit["passed"]
            for audit in report["task"]["evaluation_audits"].values()
        ),
        "candidate_set_exact": set(results) == set(continuous.CANDIDATES),
        "parameter_counts_exact": report["integrity"]["parameter_counts"]
        == EXPECTED_PARAMETERS,
        "state_counts_exact": report["integrity"]["recurrent_state_scalars"]
        == EXPECTED_STATES,
        "same_schedule_for_every_candidate": report["integrity"][
            "same_precomputed_schedule_for_every_candidate"
        ],
        "all_metrics_finite": report["integrity"]["all_metrics_finite"]
        and all_numeric_values_finite(report),
        "shared_action_rmse": shared["action_identification"]["action_rmse"]
        <= THRESHOLDS["shared_action_rmse"],
        "shared_action_better_than_independent": shared["action_identification"][
            "action_rmse"
        ]
        < independent["action_identification"]["action_rmse"],
        "independent_action_rmse": independent["action_identification"][
            "action_rmse"
        ]
        <= THRESHOLDS["independent_action_rmse"],
        "shared_relation_action_residuals": _maximum_relation_residual(shared)
        <= THRESHOLDS["shared_relation_action_rmse"],
        "independent_relation_action_residuals": _maximum_relation_residual(
            independent
        )
        <= THRESHOLDS["independent_relation_action_rmse"],
    }
    for key in ("early_L128", "late_L128"):
        shared_metrics = shared["evaluation"][key]
        independent_metrics = independent["evaluation"][key]
        prefix = key.lower()
        checks.update(
            {
                f"{prefix}_shared_post_mse": shared_metrics["post_relation_mse"]
                <= THRESHOLDS["shared_l128_post_relation_mse"],
                f"{prefix}_shared_classification": shared_metrics[
                    "center_classification_accuracy"
                ]
                == 1.0,
                f"{prefix}_shared_center_rows": shared_metrics[
                    "center_rows_correct"
                ]
                == 1.0,
                f"{prefix}_shared_identity_rows": shared_metrics[
                    "identity_rows_correct"
                ]
                == 1.0,
                f"{prefix}_shared_vector_signature": shared_metrics[
                    "predicted_vector_pair_rmse"
                ]
                <= THRESHOLDS["shared_l128_signature_rmse"],
                f"{prefix}_shared_spinor_signature": shared_metrics[
                    "predicted_spinor_negation_rmse"
                ]
                <= THRESHOLDS["shared_l128_signature_rmse"],
                f"{prefix}_independent_post_mse": independent_metrics[
                    "post_relation_mse"
                ]
                <= THRESHOLDS["independent_l128_post_relation_mse"],
                f"{prefix}_independent_classification": independent_metrics[
                    "center_classification_accuracy"
                ]
                >= THRESHOLDS["independent_l128_classification"],
                f"{prefix}_independent_center_rows": independent_metrics[
                    "center_rows_correct"
                ]
                >= THRESHOLDS["independent_l128_row_correctness"],
                f"{prefix}_independent_identity_rows": independent_metrics[
                    "identity_rows_correct"
                ]
                >= THRESHOLDS["independent_l128_row_correctness"],
            }
        )
        for name in continuous.CANDIDATES:
            if name == "shared_pure_spin8":
                continue
            checks[f"{prefix}_shared_beats_{name}"] = shared_metrics[
                "post_relation_mse"
            ] < results[name]["evaluation"][key]["post_relation_mse"]
    return checks


def validate_seed(path: Path, expected_seed: int) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    config = dict(report["config"])
    seed = config.pop("seed")
    checks = seed_gate_checks(report)
    checks.update(
        {
            "seed_exact": seed == expected_seed,
            "config_exact": config == EXPECTED_CONFIG,
            "status_unadjudicated": report["status"] == "unadjudicated",
            "protocol_freeze_exact": report["protocol_frozen_at"]
            == PROTOCOL_FROZEN_AT,
            "all_candidates_requested_in_canonical_order": tuple(
                report["candidates"]
            )
            == continuous.CANDIDATES,
        }
    )
    checkpoint_checks = {
        name: reload_checkpoint(name, record, expected_seed)
        for name, record in report["results"].items()
    }
    checks["all_checkpoints_rehash_and_reload"] = all(checkpoint_checks.values())
    return {
        "seed": expected_seed,
        "source": str(path),
        "source_sha256": sha256(path),
        "checks": checks,
        "checkpoint_checks": checkpoint_checks,
        "passed": all(checks.values()),
        "results": report["results"],
        "training_schedule_sha256": report["task"]["training_split"][
            "schedule_sha256"
        ],
        "evaluation_schedule_sha256": report["task"][
            "evaluation_schedule_sha256"
        ],
        "observation_system_sha256": report["task"][
            "observation_system_sha256"
        ],
    }


def aggregate(paths: list[Path]) -> dict[str, Any]:
    if len(paths) != len(EXPECTED_SEEDS):
        raise ValueError(f"expected {len(EXPECTED_SEEDS)} artifacts")
    seeds = [validate_seed(path, seed) for path, seed in zip(paths, EXPECTED_SEEDS)]
    l128 = {
        name: [
            seed["results"][name]["evaluation"][key]["post_relation_mse"]
            for seed in seeds
            for key in ("early_L128", "late_L128")
        ]
        for name in continuous.CANDIDATES
    }
    training_hashes = {seed["training_schedule_sha256"] for seed in seeds}
    evaluation_hashes = {
        value
        for seed in seeds
        for value in seed["evaluation_schedule_sha256"].values()
    }
    cohort_checks = {
        "all_three_seeds_pass": all(seed["passed"] for seed in seeds),
        "all_18_checkpoints_rehash_and_reload": all(
            value
            for seed in seeds
            for value in seed["checkpoint_checks"].values()
        ),
        "training_schedules_distinct": len(training_hashes) == 3,
        "all_evaluation_schedules_distinct": len(evaluation_hashes) == 18,
        "observation_systems_distinct": len(
            {seed["observation_system_sha256"] for seed in seeds}
        )
        == 3,
    }
    return {
        "schema_version": 1,
        "experiment": "endpoint-only noisy continuous-observation Pure Spin8 identification",
        "status": "fresh three-seed endpoint-only validation",
        "protocol_frozen_at": PROTOCOL_FROZEN_AT,
        "recorded_at": datetime.now().astimezone().isoformat(),
        "fresh_seeds": list(EXPECTED_SEEDS),
        "thresholds": THRESHOLDS,
        "cohort_checks": cohort_checks,
        "aggregate": {
            "l128_post_relation_mse": {
                name: {
                    "range": [min(values), max(values)],
                    "median": statistics.median(values),
                }
                for name, values in l128.items()
            },
            "independent_to_shared_median_mse_ratio": (
                statistics.median(l128["independent_so8_triplet"])
                / statistics.median(l128["shared_pure_spin8"])
            ),
        },
        "seed_reports": seeds,
        "claim_scope": {
            "empirical": [
                "fresh three-seed endpoint-only continuous action identification",
                "no intermediate prefix targets retained by training batches",
            ],
            "not_claimed": [
                "natural-data, unsigned-state, or input-sparse supervision",
                "all-28-coordinate Spin8 identification",
                "measured-compute equality before the separately frozen continuation",
                "fused-Mamba or language-model superiority",
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
        f"ratio={report['aggregate']['independent_to_shared_median_mse_ratio']:.6f}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
