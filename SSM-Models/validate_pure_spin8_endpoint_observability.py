"""Strict adjudicator for the Pure Spin(8) endpoint observability cohort."""

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

import torch

import analyze_spin8_endpoint_observability as observability
import benchmark_pure_spin8_continuous_observation as continuous
import benchmark_pure_spin8_endpoint_observability as benchmark
import benchmark_pure_spin8_endpoint_supervision as endpoint

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCES = tuple(
    ROOT
    / "experiments"
    / "artifacts"
    / f"pure_spin8_endpoint_observability_validation_seed{seed}.json"
    for seed in (1, 2, 3)
)
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "artifacts"
    / "pure_spin8_endpoint_observability_validation_seeds1_3.json"
)
FLOAT_ATOL = 2e-7


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_from_payload(payload: dict[str, Any]) -> continuous.ContinuousObservationConfig:
    allowed = {field.name for field in fields(continuous.ContinuousObservationConfig)}
    values = {key: value for key, value in payload.items() if key in allowed}
    if "evaluation_lengths" in values:
        values["evaluation_lengths"] = tuple(values["evaluation_lengths"])
    return continuous.ContinuousObservationConfig(**values)


def resolve_checkpoint(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def close(left: float, right: float, *, atol: float = FLOAT_ATOL) -> bool:
    return math.isclose(left, right, rel_tol=2e-6, abs_tol=atol)


def rebuild_evaluations(
    config: continuous.ContinuousObservationConfig,
    system: continuous.ObservationSystem,
    device: torch.device,
) -> dict[str, list[continuous.ContinuousRelationBatch]]:
    evaluations = {}
    for length in config.evaluation_lengths:
        for position in ("early", "late"):
            key = f"{position}_L{length}"
            evaluations[key] = [
                continuous.make_relation_batch(config, system, length, position, device)
            ]
    return evaluations


def evaluation_agreement_checks(
    source: dict[str, Any],
    rebuilt: dict[str, Any],
    *,
    allow_development_vector_visibility_correction: bool,
) -> dict[str, bool]:
    checks = {}
    numeric_keys = (
        "all_prefix_mse",
        "post_relation_mse",
        "final_mse",
        "target_center_identity_rmse",
        "predicted_center_identity_rmse",
    )
    classification_keys = (
        "center_visible_in_target",
        "center_classification_accuracy",
        "center_rows_correct",
        "identity_rows_correct",
    )
    for split, split_metrics in rebuilt.items():
        for representation, metrics in split_metrics.items():
            original = source[split][representation]
            prefix = f"{split}.{representation}"
            for key in numeric_keys:
                checks[f"{prefix}.{key}"] = close(original[key], metrics[key])
            for key in classification_keys:
                if (
                    allow_development_vector_visibility_correction
                    and representation == "vector"
                ):
                    checks[f"{prefix}.{key}.known_development_correction"] = True
                else:
                    checks[f"{prefix}.{key}"] = original[key] == metrics[key]
    return checks


def validate_source(
    source_path: Path,
    *,
    device: torch.device,
    development: bool,
) -> dict[str, Any]:
    source_sha256 = file_sha256(source_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    config = config_from_payload(source["config"])
    seed = config.seed
    system = continuous.make_observation_system(seed)
    exact_certificate = observability.build_certificate()

    schedule = endpoint.make_endpoint_training_schedule(config, system, device)
    split = endpoint.endpoint_training_split_audit(schedule)
    del schedule
    gc.collect()
    evaluations = rebuild_evaluations(config, system, device)
    checks: dict[str, bool] = {
        "source_passed": bool(source["passed"]),
        "schema_version": source["schema_version"] == 1,
        "experiment_name": source["experiment"]
        == "Pure Spin8 endpoint partial-readout observability",
        "seed_status": source["status"]
        == ("development" if development else "unadjudicated"),
        "exact_certificate_passed": bool(exact_certificate["passed"]),
        "training_schedule_recomputed": split["passed"],
        "training_schedule_hash_reproduced": split["schedule_sha256"]
        == source["task"]["training_split"]["schedule_sha256"],
        "observation_system_hash_reproduced": continuous.tensor_hash(
            (system.projection, system.bias)
        )
        == source["task"]["observation_system_sha256"],
        "no_intermediate_targets": source["task"][
            "intermediate_prefix_targets_retained_by_training_schedule"
        ]
        is False,
        "hidden_blocks_sliced_before_device": source["integrity"].get(
            "hidden_endpoint_blocks_sliced_before_device_transfer", development
        ),
    }
    rebuilt_results = {}
    checkpoint_checks = {}
    for readout_name, rows in source["results"].items():
        rebuilt_results[readout_name] = {}
        expected_indices = tuple(source["readouts"][readout_name]["indices"])
        for candidate, original in rows.items():
            checkpoint_path = resolve_checkpoint(original["checkpoint"])
            checkpoint_hash = file_sha256(checkpoint_path)
            payload = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            prefix = f"{readout_name}.{candidate}"
            local_checks = {
                "checkpoint_hash": checkpoint_hash
                == original["checkpoint_sha256"],
                "format_version": payload["format_version"] == 1,
                "candidate": payload["candidate"] == candidate,
                "readout": payload["readout"] == readout_name,
                "supervised_indices": tuple(
                    payload["supervised_representation_indices"]
                )
                == expected_indices,
                "supervision_metadata": payload["training_supervision"]
                == "partial_endpoint_only_signed_state",
                "config_seed": payload["config"]["seed"] == seed,
                "config_steps": payload["config"]["steps"] == config.steps,
            }
            model = benchmark.FACTORIES[candidate]().to(device)
            incompatible = model.load_state_dict(payload["state_dict"], strict=True)
            local_checks["strict_state_dict_reload"] = (
                not incompatible.missing_keys and not incompatible.unexpected_keys
            )
            rebuilt_evaluation = {
                key: benchmark.evaluate_by_representation(
                    model,
                    batches,
                    device,
                    config.evaluation_microbatch_size,
                )
                for key, batches in evaluations.items()
            }
            rebuilt_action = benchmark.action_rmse_by_representation(
                model, evaluations, device
            )
            agreement = evaluation_agreement_checks(
                original["evaluation"],
                rebuilt_evaluation,
                allow_development_vector_visibility_correction=development,
            )
            local_checks["evaluation_reproduced"] = all(agreement.values())
            local_checks["action_rmse_reproduced"] = all(
                close(original["action_rmse_by_representation"][representation], value)
                for representation, value in rebuilt_action.items()
            )
            local_checks["all_rebuilt_metrics_finite"] = (
                benchmark._all_numeric_values_finite(rebuilt_evaluation)
                and benchmark._all_numeric_values_finite(rebuilt_action)
            )
            checkpoint_checks[prefix] = {
                "path": str(checkpoint_path),
                "sha256": checkpoint_hash,
                "checks": local_checks,
                "agreement_checks": agreement,
                "passed": all(local_checks.values()),
            }
            rebuilt_results[readout_name][candidate] = {
                "action_rmse_by_representation": rebuilt_action,
                "evaluation": rebuilt_evaluation,
                "final_training_loss": original["final_training_loss"],
                "training_wall_seconds": original["training_wall_seconds"],
            }
            del model, payload
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    checks["all_checkpoints_strictly_reloaded"] = all(
        item["passed"] for item in checkpoint_checks.values()
    )
    checks["all_vector_targets_certified_invisible"] = all(
        metrics["vector"]["center_visible_in_target"] is False
        and metrics["vector"]["center_classification_accuracy"] is None
        for rows in rebuilt_results.values()
        for result in rows.values()
        for metrics in result["evaluation"].values()
    )
    checks["all_spinor_targets_certified_visible"] = all(
        metrics[representation]["center_visible_in_target"] is True
        for rows in rebuilt_results.values()
        for result in rows.values()
        for metrics in result["evaluation"].values()
        for representation in ("positive", "negative")
    )
    return {
        "seed": seed,
        "source": str(source_path),
        "source_sha256": source_sha256,
        "training_schedule_sha256": split["schedule_sha256"],
        "observation_system_sha256": source["task"]["observation_system_sha256"],
        "evaluation_schedule_sha256": source["task"][
            "evaluation_schedule_sha256"
        ],
        "checks": checks,
        "checkpoint_validation": checkpoint_checks,
        "corrected_results": rebuilt_results,
        "passed": all(checks.values()),
    }


def frozen_seed_gates(seed_result: dict[str, Any]) -> dict[str, bool]:
    results = seed_result["corrected_results"]
    l128_splits = ("early_L128", "late_L128")
    checks: dict[str, bool] = {}
    for readout_name, rows in results.items():
        shared = rows["shared_pure_spin8"]
        independent = rows["independent_so8_triplet"]
        shared_action_cap = 0.09 if readout_name == "vector_only" else 0.05
        shared_l128_cap = 0.14 if readout_name == "vector_only" else 0.07
        checks[f"{readout_name}.shared_all_action_rmse_cap"] = all(
            value <= shared_action_cap
            for value in shared["action_rmse_by_representation"].values()
        )
        checks[f"{readout_name}.shared_all_l128_mse_cap"] = all(
            shared["evaluation"][split][representation]["post_relation_mse"]
            <= shared_l128_cap
            for split in l128_splits
            for representation in continuous.TRIALITY_REPRESENTATIONS
        )
        checks[f"{readout_name}.shared_all_spinor_center_rows_exact"] = all(
            shared["evaluation"][split][representation][
                "center_classification_accuracy"
            ]
            == 1.0
            for split in l128_splits
            for representation in ("positive", "negative")
        )
        checks[f"{readout_name}.shared_beats_independent_action_all_views"] = all(
            shared["action_rmse_by_representation"][representation]
            < independent["action_rmse_by_representation"][representation]
            for representation in continuous.TRIALITY_REPRESENTATIONS
        )
        checks[f"{readout_name}.shared_beats_independent_every_l128_view"] = all(
            shared["evaluation"][split][representation]["post_relation_mse"]
            < independent["evaluation"][split][representation][
                "post_relation_mse"
            ]
            for split in l128_splits
            for representation in continuous.TRIALITY_REPRESENTATIONS
        )
        supervised = benchmark.READOUTS[readout_name]
        checks[f"{readout_name}.independent_supervised_action_capable"] = all(
            independent["action_rmse_by_representation"][
                continuous.TRIALITY_REPRESENTATIONS[index]
            ]
            <= 0.16
            for index in supervised
        )
        checks[f"{readout_name}.independent_supervised_l128_capable"] = all(
            independent["evaluation"][split][
                continuous.TRIALITY_REPRESENTATIONS[index]
            ]["post_relation_mse"]
            <= 0.20
            for split in l128_splits
            for index in supervised
        )
        visible_supervised = [index for index in supervised if index != 0]
        checks[f"{readout_name}.independent_visible_supervised_center"] = all(
            independent["evaluation"][split][
                continuous.TRIALITY_REPRESENTATIONS[index]
            ]["center_classification_accuracy"]
            >= 0.95
            for split in l128_splits
            for index in visible_supervised
        )
    return checks


def aggregate_summaries(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for readout_name in benchmark.READOUTS:
        summary[readout_name] = {}
        for candidate in benchmark.CANDIDATES:
            summary[readout_name][candidate] = {}
            for representation in continuous.TRIALITY_REPRESENTATIONS:
                l128_values = [
                    seed["corrected_results"][readout_name][candidate]["evaluation"]
                    [split][representation]["post_relation_mse"]
                    for seed in seed_results
                    for split in ("early_L128", "late_L128")
                ]
                action_values = [
                    seed["corrected_results"][readout_name][candidate][
                        "action_rmse_by_representation"
                    ][representation]
                    for seed in seed_results
                ]
                summary[readout_name][candidate][representation] = {
                    "l128_post_relation_mse_median": statistics.median(l128_values),
                    "l128_post_relation_mse_range": [
                        min(l128_values),
                        max(l128_values),
                    ],
                    "action_rmse_median": statistics.median(action_values),
                    "action_rmse_range": [min(action_values), max(action_values)],
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
    gate_results = {}
    if args.development:
        global_checks["development_single_seed"] = seeds == [0]
    else:
        global_checks.update(
            {
                "fresh_seeds_exact": seeds == [1, 2, 3],
                "training_schedules_distinct": len(
                    {item["training_schedule_sha256"] for item in seed_results}
                )
                == 3,
                "observation_systems_distinct": len(
                    {item["observation_system_sha256"] for item in seed_results}
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
        gate_results = {
            str(item["seed"]): frozen_seed_gates(item) for item in seed_results
        }
        global_checks["every_frozen_gate_passes_without_median_rescue"] = all(
            all(checks.values()) for checks in gate_results.values()
        )
    result = {
        "schema_version": 1,
        "experiment": "Pure Spin8 endpoint observability adjudication",
        "status": "development_replay" if args.development else "adjudicated",
        "recorded_at": continuous.now(),
        "seeds": seeds,
        "global_checks": global_checks,
        "frozen_seed_gates": gate_results,
        "per_seed": seed_results,
        "aggregate_summary": (
            aggregate_summaries(seed_results) if not args.development else None
        ),
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
