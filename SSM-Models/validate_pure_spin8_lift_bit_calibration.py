"""Strict validator for minimal lift-bit calibration experiments."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
from dataclasses import fields
from pathlib import Path
from typing import Any

import benchmark_pure_spin8_continuous_observation as continuous
import benchmark_pure_spin8_endpoint_observability as observability_benchmark
import benchmark_pure_spin8_endpoint_supervision as endpoint
import benchmark_pure_spin8_lift_bit_calibration as benchmark
import torch

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCES = tuple(
    ROOT
    / "experiments"
    / "artifacts"
    / f"pure_spin8_lift_bit_calibration_validation_seed{seed}.json"
    for seed in (4, 5, 6)
)
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "artifacts"
    / "pure_spin8_lift_bit_calibration_validation_seeds4_6.json"
)
FLOAT_ATOL = 2e-7


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_from_payload(payload: dict[str, Any]) -> continuous.ContinuousObservationConfig:
    allowed = {field.name for field in fields(continuous.ContinuousObservationConfig)}
    values = {key: value for key, value in payload.items() if key in allowed}
    values["evaluation_lengths"] = tuple(values["evaluation_lengths"])
    return continuous.ContinuousObservationConfig(**values)


def resolve_checkpoint(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=2e-6, abs_tol=FLOAT_ATOL)


def compare_metric_tree(
    source: Any, rebuilt: Any, path: str = "metric"
) -> dict[str, bool]:
    if isinstance(rebuilt, dict):
        if not isinstance(source, dict):
            return {path: False}
        checks = {}
        for key, value in rebuilt.items():
            checks.update(compare_metric_tree(source.get(key), value, f"{path}.{key}"))
        return checks
    if rebuilt is None or isinstance(rebuilt, (bool, str)):
        return {path: source == rebuilt}
    if isinstance(rebuilt, (int, float)):
        return {
            path: isinstance(source, (int, float))
            and not isinstance(source, bool)
            and close(float(source), float(rebuilt))
        }
    return {path: source == rebuilt}


def rebuild_evaluations(
    config: continuous.ContinuousObservationConfig,
    system: continuous.ObservationSystem,
    device: torch.device,
) -> dict[str, list[continuous.ContinuousRelationBatch]]:
    return {
        f"{position}_L{length}": [
            continuous.make_relation_batch(config, system, length, position, device)
        ]
        for length in config.evaluation_lengths
        for position in ("early", "late")
    }


def validate_source(
    source_path: Path,
    *,
    device: torch.device,
    development: bool,
) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    config = config_from_payload(source["config"])
    system = continuous.make_observation_system(config.seed)
    schedule = endpoint.make_endpoint_training_schedule(config, system, device)
    split = endpoint.endpoint_training_split_audit(schedule)
    fixed_bit_audit = benchmark.lift_bit_training_audit(schedule)
    adaptive_bit_audit = benchmark.adaptive_lift_training_audit(schedule)
    del schedule
    gc.collect()
    evaluations = rebuild_evaluations(config, system, device)

    checks: dict[str, bool] = {
        "source_passed": bool(source["passed"]),
        "schema_version": source["schema_version"] == 1,
        "experiment_name": source["experiment"]
        == "Pure Spin8 minimal one-bit lift calibration",
        "seed_status": source["status"]
        == ("development" if development else "unadjudicated"),
        "exact_certificate_passed": benchmark.exact_one_bit_certificate()["passed"],
        "training_split_recomputed": split["passed"],
        "training_schedule_hash_reproduced": split["schedule_sha256"]
        == source["task"]["training_split"]["schedule_sha256"],
        "fixed_bit_hash_reproduced": fixed_bit_audit["bit_sha256"]
        == source["task"]["lift_bit_audit"]["bit_sha256"],
        "adaptive_address_hash_reproduced": adaptive_bit_audit["address_sha256"]
        == source["task"]["adaptive_lift_bit_audit"]["address_sha256"],
        "adaptive_bit_hash_reproduced": adaptive_bit_audit["bit_sha256"]
        == source["task"]["adaptive_lift_bit_audit"]["bit_sha256"],
        "adaptive_margin_bound_reproduced": adaptive_bit_audit["passed"],
        "observation_system_hash_reproduced": continuous.tensor_hash(
            (system.projection, system.bias)
        )
        == source["task"]["observation_system_sha256"],
        "only_one_external_lift_bit": source["one_bit_certificate"][
            "robust_adaptive_chart"
        ]["external_lift_bits_given_address"]
        == 1,
        "full_hidden_spinor_not_transferred": source["task"][
            "hidden_full_spinor_target_transferred_for_bit_mode"
        ]
        is False,
        "no_intermediate_targets": source["task"]["intermediate_targets_retained"]
        is False,
    }

    rebuilt_results = {}
    checkpoint_validation = {}
    for mode, rows in source["results"].items():
        rebuilt_results[mode] = {}
        for candidate, original in rows.items():
            prefix = f"{mode}.{candidate}"
            checkpoint_path = resolve_checkpoint(original["checkpoint"])
            checkpoint_hash = file_sha256(checkpoint_path)
            payload = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            local_checks = {
                "checkpoint_hash": checkpoint_hash
                == original["checkpoint_sha256"],
                "format_version": payload["format_version"] == 1,
                "candidate": payload["candidate"] == candidate,
                "mode": payload["mode"] == mode,
                "supervision": payload["training_supervision"]
                == "endpoint_vector_plus_optional_one_lift_bit",
                "probe_index": payload["positive_probe_index"]
                == benchmark.POSITIVE_PROBE_INDEX,
                "bit_logit_scale": payload["bit_logit_scale"]
                == benchmark.BIT_LOGIT_SCALE,
                "bit_loss_weight": payload["bit_loss_weight"]
                == benchmark.BIT_LOSS_WEIGHT,
                "config_seed": payload["config"]["seed"] == config.seed,
                "config_steps": payload["config"]["steps"] == config.steps,
            }
            model = benchmark.FACTORIES[candidate]().to(device)
            incompatible = model.load_state_dict(payload["state_dict"], strict=True)
            local_checks["strict_state_dict_reload"] = (
                not incompatible.missing_keys and not incompatible.unexpected_keys
            )
            rebuilt_evaluation = {
                key: observability_benchmark.evaluate_by_representation(
                    model, batches, device, config.evaluation_microbatch_size
                )
                for key, batches in evaluations.items()
            }
            rebuilt_fixed_bit = {
                key: benchmark.evaluate_lift_bit(
                    model, batches, device, config.evaluation_microbatch_size
                )
                for key, batches in evaluations.items()
            }
            rebuilt_adaptive_bit = {
                key: benchmark.evaluate_adaptive_lift_bit(
                    model, batches, device, config.evaluation_microbatch_size
                )
                for key, batches in evaluations.items()
            }
            rebuilt_action = (
                observability_benchmark.action_rmse_by_representation(
                    model, evaluations, device
                )
            )
            agreement = {}
            agreement.update(
                compare_metric_tree(
                    original["evaluation"], rebuilt_evaluation, f"{prefix}.evaluation"
                )
            )
            agreement.update(
                compare_metric_tree(
                    original["lift_bit_evaluation"],
                    rebuilt_fixed_bit,
                    f"{prefix}.fixed_bit",
                )
            )
            agreement.update(
                compare_metric_tree(
                    original["adaptive_lift_evaluation"],
                    rebuilt_adaptive_bit,
                    f"{prefix}.adaptive_bit",
                )
            )
            agreement.update(
                compare_metric_tree(
                    original["action_rmse_by_representation"],
                    rebuilt_action,
                    f"{prefix}.action",
                )
            )
            local_checks["all_metrics_reproduced"] = all(agreement.values())
            local_checks["all_rebuilt_metrics_finite"] = (
                observability_benchmark._all_numeric_values_finite(
                    rebuilt_evaluation
                )
                and observability_benchmark._all_numeric_values_finite(
                    rebuilt_fixed_bit
                )
                and observability_benchmark._all_numeric_values_finite(
                    rebuilt_adaptive_bit
                )
                and observability_benchmark._all_numeric_values_finite(
                    rebuilt_action
                )
            )
            checkpoint_validation[prefix] = {
                "path": str(checkpoint_path),
                "sha256": checkpoint_hash,
                "checks": local_checks,
                "agreement_checks": agreement,
                "passed": all(local_checks.values()),
            }
            rebuilt_results[mode][candidate] = {
                "action_rmse_by_representation": rebuilt_action,
                "evaluation": rebuilt_evaluation,
                "lift_bit_evaluation": rebuilt_fixed_bit,
                "adaptive_lift_evaluation": rebuilt_adaptive_bit,
                "final_training_loss": original["final_training_loss"],
                "final_components": original["component_samples"][str(config.steps)],
                "training_wall_seconds": original["training_wall_seconds"],
            }
            del model, payload
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    checks["all_checkpoints_strictly_reloaded"] = all(
        item["passed"] for item in checkpoint_validation.values()
    )
    return {
        "seed": config.seed,
        "source": str(source_path),
        "source_sha256": file_sha256(source_path),
        "training_schedule_sha256": split["schedule_sha256"],
        "observation_system_sha256": source["task"]["observation_system_sha256"],
        "evaluation_schedule_sha256": source["task"][
            "evaluation_schedule_sha256"
        ],
        "adaptive_address_sha256": adaptive_bit_audit["address_sha256"],
        "adaptive_bit_sha256": adaptive_bit_audit["bit_sha256"],
        "adaptive_minimum_training_margin": adaptive_bit_audit[
            "minimum_selected_absolute_probe"
        ],
        "checks": checks,
        "checkpoint_validation": checkpoint_validation,
        "corrected_results": rebuilt_results,
        "passed": all(checks.values()),
    }


def frozen_seed_gates(seed_result: dict[str, Any]) -> dict[str, bool]:
    results = seed_result["corrected_results"]
    adaptive = results["vector_plus_adaptive_lift_bit"]["shared_pure_spin8"]
    adaptive_independent = results["vector_plus_adaptive_lift_bit"][
        "independent_so8_triplet"
    ]
    vector = results["vector_only"]["shared_pure_spin8"]
    positive = results["positive_only"]["shared_pure_spin8"]
    full = results["full_triality"]["shared_pure_spin8"]
    l128 = ("early_L128", "late_L128")
    reps = continuous.TRIALITY_REPRESENTATIONS
    checks = {
        "adaptive_training_bit_exact": adaptive["final_components"][
            "adaptive_bit_accuracy"
        ]
        == 1.0,
        "adaptive_all_action_rmse_cap": all(
            value <= 0.04
            for value in adaptive["action_rmse_by_representation"].values()
        ),
        "adaptive_all_l128_mse_cap": all(
            adaptive["evaluation"][split][rep]["post_relation_mse"] <= 0.05
            for split in l128
            for rep in reps
        ),
        "adaptive_all_l128_bits_exact": all(
            adaptive["adaptive_lift_evaluation"][split][
                "adaptive_lift_bit_accuracy"
            ]
            == 1.0
            for split in l128
        ),
        "adaptive_all_l128_center_rows_exact": all(
            adaptive["evaluation"][split][rep]["center_classification_accuracy"]
            == 1.0
            for split in l128
            for rep in ("positive", "negative")
        ),
        "adaptive_addresses_stable_on_relation_pairs": all(
            adaptive["adaptive_lift_evaluation"][split][
                "target_center_identity_same_address_fraction"
            ]
            == 1.0
            and adaptive["adaptive_lift_evaluation"][split][
                "target_center_identity_opposite_bit_fraction"
            ]
            == 1.0
            for split in l128
        ),
        "adaptive_beats_vector_action_every_view": all(
            adaptive["action_rmse_by_representation"][rep]
            < vector["action_rmse_by_representation"][rep]
            for rep in reps
        ),
        "adaptive_beats_vector_every_l128_view": all(
            adaptive["evaluation"][split][rep]["post_relation_mse"]
            < vector["evaluation"][split][rep]["post_relation_mse"]
            for split in l128
            for rep in reps
        ),
        "adaptive_beats_independent_action_every_view": all(
            adaptive["action_rmse_by_representation"][rep]
            < adaptive_independent["action_rmse_by_representation"][rep]
            for rep in reps
        ),
        "adaptive_beats_independent_every_l128_view": all(
            adaptive["evaluation"][split][rep]["post_relation_mse"]
            < adaptive_independent["evaluation"][split][rep]["post_relation_mse"]
            for split in l128
            for rep in reps
        ),
        "independent_adaptive_vector_capable": (
            adaptive_independent["action_rmse_by_representation"]["vector"] <= 0.18
            and all(
                adaptive_independent["evaluation"][split]["vector"][
                    "post_relation_mse"
                ]
                <= 0.22
                for split in l128
            )
        ),
        "positive_reference_passes": all(
            positive["action_rmse_by_representation"][rep] <= 0.05
            for rep in reps
        )
        and all(
            positive["evaluation"][split][rep]["post_relation_mse"] <= 0.07
            for split in l128
            for rep in reps
        ),
        "full_reference_passes": all(
            full["action_rmse_by_representation"][rep] <= 0.05 for rep in reps
        )
        and all(
            full["evaluation"][split][rep]["post_relation_mse"] <= 0.07
            for split in l128
            for rep in reps
        ),
    }
    return checks


def aggregate_summary(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for mode in benchmark.MODES:
        summary[mode] = {}
        for candidate in benchmark.CANDIDATES:
            summary[mode][candidate] = {}
            for representation in continuous.TRIALITY_REPRESENTATIONS:
                action = [
                    seed["corrected_results"][mode][candidate][
                        "action_rmse_by_representation"
                    ][representation]
                    for seed in seed_results
                ]
                l128 = [
                    seed["corrected_results"][mode][candidate]["evaluation"][split]
                    [representation]["post_relation_mse"]
                    for seed in seed_results
                    for split in ("early_L128", "late_L128")
                ]
                summary[mode][candidate][representation] = {
                    "action_rmse_median": statistics.median(action),
                    "action_rmse_range": [min(action), max(action)],
                    "l128_mse_median": statistics.median(l128),
                    "l128_mse_range": [min(l128), max(l128)],
                }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--development", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.sources:
        raise SystemExit("at least one source artifact is required")
    device = torch.device(args.device)
    seed_results = [
        validate_source(path, device=device, development=args.development)
        for path in args.sources
    ]
    seeds = [item["seed"] for item in seed_results]
    global_checks = {
        "all_sources_pass_integrity": all(item["passed"] for item in seed_results),
        "seeds_unique": len(seeds) == len(set(seeds)),
    }
    gates = {}
    if args.development:
        global_checks["development_single_seed"] = seeds == [0]
    else:
        global_checks.update(
            {
                "fresh_seeds_exact": seeds == [4, 5, 6],
                "training_schedules_distinct": len(
                    {item["training_schedule_sha256"] for item in seed_results}
                )
                == 3,
                "observation_systems_distinct": len(
                    {item["observation_system_sha256"] for item in seed_results}
                )
                == 3,
                "adaptive_addresses_distinct": len(
                    {item["adaptive_address_sha256"] for item in seed_results}
                )
                == 3,
                "adaptive_bits_distinct": len(
                    {item["adaptive_bit_sha256"] for item in seed_results}
                )
                == 3,
                "evaluation_schedules_distinct": len(
                    {
                        value
                        for item in seed_results
                        for value in item["evaluation_schedule_sha256"].values()
                    }
                )
                == 18,
            }
        )
        gates = {str(item["seed"]): frozen_seed_gates(item) for item in seed_results}
        global_checks["every_frozen_gate_passes_without_median_rescue"] = all(
            all(values.values()) for values in gates.values()
        )
    result = {
        "schema_version": 1,
        "experiment": "Pure Spin8 one-bit lift calibration adjudication",
        "status": "development_replay" if args.development else "adjudicated",
        "recorded_at": continuous.now(),
        "seeds": seeds,
        "global_checks": global_checks,
        "frozen_seed_gates": gates,
        "per_seed": seed_results,
        "aggregate_summary": None if args.development else aggregate_summary(seed_results),
        "passed": all(global_checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"global_checks": global_checks, "passed": result["passed"]}, indent=2))
    print(f"wrote {args.output}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
