"""Validate the frozen fresh-seed latent-increment Pure Spin(8) cohort."""

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

from benchmark_pure_spin8_latent_increment import (
    CANDIDATES,
    GRUTrialityTracker,
    LatentPureSpin8Tracker,
    Mamba2TrialityTracker,
    TokenOnlyTrialityAblation,
)

ROOT = Path(__file__).resolve().parent
EXPECTED_SEEDS = (1, 2, 3)
EXPECTED_CONFIG = {
    "batch_size": 32,
    "evaluation_lengths": [16, 64, 128],
    "evaluation_microbatch_size": 32,
    "evaluation_pairs": 64,
    "gradient_clip": 1.0,
    "learning_rate": 0.003,
    "steps": 500,
    "training_length": 16,
    "weight_decay": 0.0001,
}
EXPECTED_PARAMETER_COUNTS = {
    "latent_pure_spin8": 892,
    "mamba2_transformers": 891,
    "gru_reference": 887,
    "token_only_ablation": 874,
}
THRESHOLDS = {
    "maximum_relative_parameter_gap": 0.03,
    "action_rmse": 0.02,
    "learned_relation_rmse": 0.01,
    "l128_post_relation_mse": 0.002,
    "l128_relation_signature_rmse": 0.02,
}
FACTORIES = {
    "latent_pure_spin8": LatentPureSpin8Tracker,
    "mamba2_transformers": Mamba2TrialityTracker,
    "gru_reference": GRUTrialityTracker,
    "token_only_ablation": TokenOnlyTrialityAblation,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _all_numeric_values_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_numeric_values_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_numeric_values_finite(item) for item in value)
    return False


def reload_checkpoint(name: str, record: dict[str, Any], seed: int) -> bool:
    path = checkpoint_path(record["checkpoint"])
    if not path.is_file() or sha256(path) != record["checkpoint_sha256"]:
        return False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("candidate") != name:
        return False
    if payload.get("config", {}).get("seed") != seed:
        return False
    model = FACTORIES[name]()
    model.load_state_dict(payload["state_dict"], strict=True)
    return True


def seed_gate_checks(report: dict[str, Any]) -> dict[str, bool]:
    pure = report["results"]["latent_pure_spin8"]
    action = pure["action_identification"]
    checks = {
        "runner_integrity_passed": bool(report["passed"]),
        "teacher_contract_passed": bool(report["task"]["teacher_contract"]["passed"]),
        "training_split_passed": bool(report["task"]["training_split"]["passed"]),
        "all_evaluation_splits_passed": all(
            audit["passed"]
            for audit in report["task"]["evaluation_audits"].values()
        ),
        "all_metrics_finite": _all_numeric_values_finite(report["results"]),
        "parameter_counts_exact": (
            report["integrity"]["parameter_counts"] == EXPECTED_PARAMETER_COUNTS
        ),
        "parameter_gap_below_3pct": (
            report["integrity"]["maximum_relative_parameter_gap"]
            <= THRESHOLDS["maximum_relative_parameter_gap"]
        ),
        "state_not_misreported_as_matched": not report["integrity"]["state_matched"],
        "pure_action_rmse": action["action_rmse"] <= THRESHOLDS["action_rmse"],
        "pure_a_square_vector_relation": (
            action["a_square_vector_identity_rmse"]
            <= THRESHOLDS["learned_relation_rmse"]
        ),
        "pure_a_square_spinor_relation": (
            action["a_square_spinor_minus_identity_rmse"]
            <= THRESHOLDS["learned_relation_rmse"]
        ),
        "pure_b_inverse_b_relation": (
            action["b_inverse_b_identity_rmse"]
            <= THRESHOLDS["learned_relation_rmse"]
        ),
    }
    for key in ("early_L128", "late_L128"):
        metrics = pure["evaluation"][key]
        slug = key.lower()
        checks[f"{slug}_post_relation_mse"] = (
            metrics["post_relation_mse"]
            <= THRESHOLDS["l128_post_relation_mse"]
        )
        checks[f"{slug}_center_classification"] = (
            metrics["center_classification_accuracy"] == 1.0
        )
        checks[f"{slug}_center_rows"] = metrics["center_rows_correct"] == 1.0
        checks[f"{slug}_identity_rows"] = metrics["identity_rows_correct"] == 1.0
        checks[f"{slug}_vector_signature"] = (
            metrics["predicted_vector_pair_rmse"]
            <= THRESHOLDS["l128_relation_signature_rmse"]
        )
        checks[f"{slug}_spinor_signature"] = (
            metrics["predicted_spinor_negation_rmse"]
            <= THRESHOLDS["l128_relation_signature_rmse"]
        )
        for reference in ("mamba2_transformers", "gru_reference"):
            checks[f"{slug}_beats_{reference}"] = (
                metrics["post_relation_mse"]
                < report["results"][reference]["evaluation"][key][
                    "post_relation_mse"
                ]
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
            # Benchmark JSON is intentionally serialized with sort_keys=True,
            # so mapping order is not the in-memory CANDIDATES order.
            "candidate_set_exact": set(report["results"]) == set(CANDIDATES),
            "checkpoint_count_four": len(report["results"]) == 4,
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
        "teacher_coordinates_sha256": report["task"][
            "teacher_coordinates_sha256"
        ],
        "teacher_initial_state_sha256": report["task"][
            "teacher_initial_state_sha256"
        ],
    }


def aggregate(paths: list[Path]) -> dict[str, Any]:
    if len(paths) != len(EXPECTED_SEEDS):
        raise ValueError(f"expected {len(EXPECTED_SEEDS)} artifacts")
    seeds = [validate_seed(path, seed) for path, seed in zip(paths, EXPECTED_SEEDS)]
    pure_l128 = [
        seed["results"]["latent_pure_spin8"]["evaluation"][position][
            "post_relation_mse"
        ]
        for seed in seeds
        for position in ("early_L128", "late_L128")
    ]
    mamba_l128 = [
        seed["results"]["mamba2_transformers"]["evaluation"][position][
            "post_relation_mse"
        ]
        for seed in seeds
        for position in ("early_L128", "late_L128")
    ]
    gru_l128 = [
        seed["results"]["gru_reference"]["evaluation"][position][
            "post_relation_mse"
        ]
        for seed in seeds
        for position in ("early_L128", "late_L128")
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
        "teacher_coordinates_identical": (
            len({seed["teacher_coordinates_sha256"] for seed in seeds}) == 1
        ),
        "teacher_initial_state_identical": (
            len({seed["teacher_initial_state_sha256"] for seed in seeds}) == 1
        ),
    }
    return {
        "schema_version": 1,
        "experiment": "latent-token Pure Spin8 center-relation identification",
        "status": "fresh three-seed validation",
        "protocol_frozen_at": "2026-08-17T03:34:00+02:00",
        "recorded_at": datetime.now().astimezone().isoformat(),
        "fresh_seeds": list(EXPECTED_SEEDS),
        "thresholds": THRESHOLDS,
        "cohort_checks": cohort_checks,
        "aggregate": {
            "pure_l128_post_relation_mse_range": [min(pure_l128), max(pure_l128)],
            "pure_l128_post_relation_mse_median": statistics.median(pure_l128),
            "mamba_l128_post_relation_mse_median": statistics.median(mamba_l128),
            "gru_l128_post_relation_mse_median": statistics.median(gru_l128),
            "mamba_to_pure_median_mse_ratio": (
                statistics.median(mamba_l128) / statistics.median(pure_l128)
            ),
            "gru_to_pure_median_mse_ratio": (
                statistics.median(gru_l128) / statistics.median(pure_l128)
            ),
        },
        "seed_reports": seeds,
        "claim_scope": {
            "empirical": [
                "fresh three-seed parameter-near synthetic relation validation",
                "strictly rehashed and reloaded checkpoints",
            ],
            "not_claimed": [
                "state- or compute-matched superiority",
                "natural-data or generic language-model superiority",
                "fused Mamba throughput comparison",
                "a theorem about Mamba-2, diagonal SSMs, or global optimality",
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
