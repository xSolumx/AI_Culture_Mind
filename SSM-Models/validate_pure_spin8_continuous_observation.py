"""Adjudicate the frozen fresh-seed continuous-observation Spin(8) cohort."""

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

from benchmark_pure_spin8_continuous_observation import (
    CANDIDATES,
    ContinuousMamba2Tracker,
    IndependentSO8TripletTracker,
    ObservationOnlyAblation,
    ParameterNearGRUTracker,
    SharedPureSpin8Tracker,
    StateMatchedGRUTracker,
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
THRESHOLDS = {
    "maximum_relative_parameter_gap": 0.04,
    "shared_action_rmse": 0.03,
    "independent_action_rmse": 0.06,
    "shared_l128_post_relation_mse": 0.05,
    "shared_l128_signature_rmse": 0.05,
    "independent_l128_post_relation_mse": 0.12,
    "independent_l128_center_classification": 0.95,
    "independent_l128_row_correctness": 0.90,
    "shared_relation_action_rmse": 0.04,
    "independent_relation_action_rmse": 0.08,
}
FACTORIES = {
    "shared_pure_spin8": SharedPureSpin8Tracker,
    "independent_so8_triplet": IndependentSO8TripletTracker,
    "mamba2_parameter_near": ContinuousMamba2Tracker,
    "gru_parameter_near": ParameterNearGRUTracker,
    "observation_only_ablation": ObservationOnlyAblation,
    "gru_state_matched": StateMatchedGRUTracker,
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
    if payload.get("candidate") != name or payload.get("config", {}).get("seed") != seed:
        return False
    model = FACTORIES[name]()
    model.load_state_dict(payload["state_dict"], strict=True)
    return True


def seed_gate_checks(report: dict[str, Any]) -> dict[str, bool]:
    results = report["results"]
    shared = results["shared_pure_spin8"]
    independent = results["independent_so8_triplet"]
    shared_action = shared["action_identification"]
    independent_action = independent["action_identification"]
    checks = {
        "runner_integrity_passed": bool(report["passed"]),
        "teacher_contract_passed": bool(report["task"]["teacher_contract"]["passed"]),
        "training_split_passed": bool(report["task"]["training_split"]["passed"]),
        "all_evaluation_splits_passed": all(
            audit["passed"] for audit in report["task"]["evaluation_audits"].values()
        ),
        "all_metrics_finite": all_numeric_values_finite(results),
        "parameter_counts_exact": report["integrity"]["parameter_counts"]
        == EXPECTED_PARAMETER_COUNTS,
        "state_counts_exact": report["integrity"]["recurrent_state_scalars"]
        == EXPECTED_STATES,
        "parameter_gap_below_4pct": (
            report["integrity"]["maximum_relative_parameter_gap_near_cohort"]
            <= THRESHOLDS["maximum_relative_parameter_gap"]
        ),
        "shared_independent_state_match": bool(
            report["integrity"]["shared_vs_independent_so8_state_matched"]
        ),
        "shared_gru_state_match": bool(
            report["integrity"]["shared_vs_state_gru_state_matched"]
        ),
        "shared_action_rmse": shared_action["action_rmse"]
        <= THRESHOLDS["shared_action_rmse"],
        "independent_action_rmse": independent_action["action_rmse"]
        <= THRESHOLDS["independent_action_rmse"],
        "shared_action_better_than_independent": shared_action["action_rmse"]
        < independent_action["action_rmse"],
    }
    for key in ("early_L128", "late_L128"):
        slug = key.lower()
        shared_metrics = shared["evaluation"][key]
        independent_metrics = independent["evaluation"][key]
        checks[f"{slug}_shared_post_mse"] = (
            shared_metrics["post_relation_mse"]
            <= THRESHOLDS["shared_l128_post_relation_mse"]
        )
        checks[f"{slug}_shared_center_classification"] = (
            shared_metrics["center_classification_accuracy"] == 1.0
        )
        checks[f"{slug}_shared_center_rows"] = (
            shared_metrics["center_rows_correct"] == 1.0
        )
        checks[f"{slug}_shared_identity_rows"] = (
            shared_metrics["identity_rows_correct"] == 1.0
        )
        checks[f"{slug}_shared_vector_signature"] = (
            shared_metrics["predicted_vector_pair_rmse"]
            <= THRESHOLDS["shared_l128_signature_rmse"]
        )
        checks[f"{slug}_shared_spinor_signature"] = (
            shared_metrics["predicted_spinor_negation_rmse"]
            <= THRESHOLDS["shared_l128_signature_rmse"]
        )
        for reference in (
            "independent_so8_triplet",
            "mamba2_parameter_near",
            "gru_parameter_near",
            "observation_only_ablation",
            "gru_state_matched",
        ):
            checks[f"{slug}_shared_beats_{reference}"] = (
                shared_metrics["post_relation_mse"]
                < results[reference]["evaluation"][key]["post_relation_mse"]
            )
        checks[f"{slug}_independent_post_mse"] = (
            independent_metrics["post_relation_mse"]
            <= THRESHOLDS["independent_l128_post_relation_mse"]
        )
        checks[f"{slug}_independent_center_classification"] = (
            independent_metrics["center_classification_accuracy"]
            >= THRESHOLDS["independent_l128_center_classification"]
        )
        checks[f"{slug}_independent_center_rows"] = (
            independent_metrics["center_rows_correct"]
            >= THRESHOLDS["independent_l128_row_correctness"]
        )
        checks[f"{slug}_independent_identity_rows"] = (
            independent_metrics["identity_rows_correct"]
            >= THRESHOLDS["independent_l128_row_correctness"]
        )
    for family, diagnostics, limit in (
        (
            "shared",
            shared_action,
            THRESHOLDS["shared_relation_action_rmse"],
        ),
        (
            "independent",
            independent_action,
            THRESHOLDS["independent_relation_action_rmse"],
        ),
    ):
        checks[f"{family}_all_relation_action_residuals"] = all(
            value <= limit
            for residuals in diagnostics["relation_action_residuals"].values()
            for value in residuals.values()
        )
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
            "split_correction_exact": report["protocol_split_correction_at"]
            == "2026-08-17T04:15:00+02:00",
            "candidate_set_exact": set(report["results"]) == set(CANDIDATES),
            "checkpoint_count_six": len(report["results"]) == 6,
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
        "teacher_initial_state_sha256": report["task"][
            "teacher_initial_state_sha256"
        ],
    }


def aggregate(paths: list[Path]) -> dict[str, Any]:
    if len(paths) != len(EXPECTED_SEEDS):
        raise ValueError(f"expected {len(EXPECTED_SEEDS)} artifacts")
    seeds = [validate_seed(path, seed) for path, seed in zip(paths, EXPECTED_SEEDS)]
    shared_l128 = [
        seed["results"]["shared_pure_spin8"]["evaluation"][key][
            "post_relation_mse"
        ]
        for seed in seeds
        for key in ("early_L128", "late_L128")
    ]
    independent_l128 = [
        seed["results"]["independent_so8_triplet"]["evaluation"][key][
            "post_relation_mse"
        ]
        for seed in seeds
        for key in ("early_L128", "late_L128")
    ]
    training_hashes = {seed["training_schedule_sha256"] for seed in seeds}
    evaluation_hashes = {
        value
        for seed in seeds
        for value in seed["evaluation_schedule_sha256"].values()
    }
    cohort_checks = {
        "all_three_seeds_pass": all(seed["passed"] for seed in seeds),
        "training_schedules_distinct": len(training_hashes) == 3,
        "all_evaluation_schedules_distinct": len(evaluation_hashes) == 18,
        "observation_systems_distinct": (
            len({seed["observation_system_sha256"] for seed in seeds}) == 3
        ),
        "teacher_initial_state_identical": (
            len({seed["teacher_initial_state_sha256"] for seed in seeds}) == 1
        ),
    }
    return {
        "schema_version": 1,
        "experiment": "noisy continuous-observation Pure Spin8 identification",
        "status": "fresh three-seed validation",
        "protocol_frozen_at": "2026-08-17T04:13:00+02:00",
        "protocol_split_correction_at": "2026-08-17T04:15:00+02:00",
        "recorded_at": datetime.now().astimezone().isoformat(),
        "fresh_seeds": list(EXPECTED_SEEDS),
        "thresholds": THRESHOLDS,
        "cohort_checks": cohort_checks,
        "aggregate": {
            "shared_l128_post_relation_mse_range": [
                min(shared_l128),
                max(shared_l128),
            ],
            "shared_l128_post_relation_mse_median": statistics.median(shared_l128),
            "independent_l128_post_relation_mse_range": [
                min(independent_l128),
                max(independent_l128),
            ],
            "independent_l128_post_relation_mse_median": statistics.median(
                independent_l128
            ),
            "independent_to_shared_median_mse_ratio": (
                statistics.median(independent_l128)
                / statistics.median(shared_l128)
            ),
        },
        "seed_reports": seeds,
        "claim_scope": {
            "empirical": [
                "fresh three-seed noisy continuous online identification",
                "shared-action comparison with a capable parameter/state-near independent orthogonal tracker",
            ],
            "not_claimed": [
                "natural-data or language-model utility",
                "generic triality necessity",
                "hardware-independent compute superiority",
                "fused Mamba or training throughput",
                "a theorem about Spin8, Mamba-2, Dirac-Gram, or global optimality",
            ],
        },
        "passed": all(cohort_checks.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs=3, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = aggregate(args.artifacts)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
