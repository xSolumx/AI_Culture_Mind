"""Strict validator for the shared-latent scrambled-alignment control."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path
from typing import Any

import analyze_pure_spin8_lift_gradient_identifiability as gradient_analysis
import benchmark_pure_spin8_continuous_observation as continuous
import benchmark_pure_spin8_endpoint_observability as observability
import benchmark_pure_spin8_endpoint_supervision as endpoint
import benchmark_pure_spin8_lift_bit_calibration as calibration
import benchmark_pure_spin8_scrambled_alignment as benchmark
import torch
import validate_pure_spin8_lift_bit_calibration as lift_validator

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCES = tuple(
    ROOT
    / "experiments"
    / "artifacts"
    / f"pure_spin8_scrambled_alignment_validation_seed{seed}.json"
    for seed in (7, 8, 9)
)
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "artifacts"
    / "pure_spin8_scrambled_alignment_validation_seeds7_9.json"
)
FRESH_SEEDS = [7, 8, 9]
ALIGNMENT_DATA_UPDATE_MINIMUM = 1e-5
ALIGNMENT_DECAY_TOLERANCE = 2e-6


def _alignment_trace(
    *,
    payload: dict[str, Any],
    config: continuous.ContinuousObservationConfig,
    mode: str,
) -> dict[str, Any]:
    continuous.seed_everything(benchmark.SEED_BASE + 1_000 * config.seed)
    initial_model = benchmark.ScrambledSharedLatentSO8TripletTracker()
    initial = initial_model.spinor_alignment_coordinates.detach()
    final = payload["state_dict"]["spinor_alignment_coordinates"]
    decay_only = gradient_analysis._repeated_adamw_decay(
        initial,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        steps=config.steps,
    )
    residuals = {
        representation: float((final[index] - decay_only[index]).abs().max())
        for index, representation in enumerate(("positive", "negative"))
    }
    checks = {
        "positive_alignment_has_data_update": (
            residuals["positive"] > ALIGNMENT_DATA_UPDATE_MINIMUM
        ),
        "negative_alignment_behavior_matches_mode": (
            residuals["negative"] <= ALIGNMENT_DECAY_TOLERANCE
            if mode == "vector_plus_adaptive_lift_bit"
            else residuals["negative"] > ALIGNMENT_DATA_UPDATE_MINIMUM
        ),
    }
    return {"residuals": residuals, "checks": checks, "passed": all(checks.values())}


def _rebuild_evaluations(
    config: continuous.ContinuousObservationConfig,
    system: continuous.ObservationSystem,
    device: torch.device,
) -> dict[str, list[continuous.ContinuousRelationBatch]]:
    return lift_validator.rebuild_evaluations(config, system, device)


def validate_source(
    source_path: Path,
    *,
    device: torch.device,
    development: bool,
) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    config = lift_validator.config_from_payload(source["config"])
    system = continuous.make_observation_system(config.seed)
    schedule = endpoint.make_endpoint_training_schedule(config, system, device)
    split = endpoint.endpoint_training_split_audit(schedule)
    adaptive_audit = calibration.adaptive_lift_training_audit(schedule)
    del schedule
    gc.collect()
    evaluations = _rebuild_evaluations(config, system, device)
    evaluation_hashes = {
        key: continuous.tensor_hash(
            (
                batches[0].observations,
                batches[0].targets,
                batches[0].coordinates,
                batches[0].post_relation_mask,
            )
        )
        for key, batches in evaluations.items()
    }
    initialization = benchmark.matched_initialization_audit(config.seed)
    gradient_routes = benchmark.gradient_route_audit(config, device)

    checks: dict[str, bool] = {
        "source_passed": bool(source["passed"]),
        "schema_version": source["schema_version"] == 1,
        "experiment_name": source["experiment"]
        == "Pure Spin8 shared-latent scrambled-alignment control",
        "seed_status": source["status"]
        == ("development" if development else "unadjudicated"),
        "protocol_freeze_matches": (
            development
            or source["protocol_frozen_at"] == benchmark.PROTOCOL_FROZEN_AT
        ),
        "modes_exact": tuple(source["modes"]) == benchmark.MODES,
        "candidates_exact": tuple(source["candidates"]) == benchmark.CANDIDATES,
        "training_split_recomputed": split["passed"],
        "training_schedule_hash_reproduced": split["schedule_sha256"]
        == source["task"]["training_split"]["schedule_sha256"],
        "adaptive_address_hash_reproduced": adaptive_audit["address_sha256"]
        == source["task"]["adaptive_lift_bit_audit"]["address_sha256"],
        "adaptive_bit_hash_reproduced": adaptive_audit["bit_sha256"]
        == source["task"]["adaptive_lift_bit_audit"]["bit_sha256"],
        "adaptive_margin_reproduced": adaptive_audit["passed"],
        "observation_system_hash_reproduced": continuous.tensor_hash(
            (system.projection, system.bias)
        )
        == source["task"]["observation_system_sha256"],
        "evaluation_hashes_reproduced": evaluation_hashes
        == source["task"]["evaluation_schedule_sha256"],
        "matched_initialization_reproduced": initialization
        == source["architecture"]["initialization_audit"],
        "gradient_routes_reproduced": gradient_routes
        == source["architecture"]["gradient_route_audit"],
        "alignment_initialization_frozen": source["architecture"][
            "alignment_initialization_std"
        ]
        == benchmark.ALIGNMENT_INITIALIZATION_STD,
    }

    corrected_results = {}
    checkpoint_validation = {}
    for mode, rows in source["results"].items():
        corrected_results[mode] = {}
        for candidate, original in rows.items():
            prefix = f"{mode}.{candidate}"
            checkpoint_path = lift_validator.resolve_checkpoint(
                original["checkpoint"]
            )
            checkpoint_hash = lift_validator.file_sha256(checkpoint_path)
            payload = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            local_checks = {
                "checkpoint_hash": checkpoint_hash
                == original["checkpoint_sha256"],
                "format_version": payload["format_version"] == 1,
                "candidate": payload["candidate"] == candidate,
                "mode": payload["mode"] == mode,
                "config_seed": payload["config"]["seed"] == config.seed,
                "config_steps": payload["config"]["steps"] == config.steps,
            }
            model = benchmark.FACTORIES[candidate]().to(device)
            incompatible = model.load_state_dict(payload["state_dict"], strict=True)
            local_checks["strict_state_dict_reload"] = (
                not incompatible.missing_keys and not incompatible.unexpected_keys
            )
            rebuilt_evaluation = {
                key: observability.evaluate_by_representation(
                    model, batches, device, config.evaluation_microbatch_size
                )
                for key, batches in evaluations.items()
            }
            rebuilt_adaptive = {
                key: calibration.evaluate_adaptive_lift_bit(
                    model, batches, device, config.evaluation_microbatch_size
                )
                for key, batches in evaluations.items()
            }
            rebuilt_action = observability.action_rmse_by_representation(
                model, evaluations, device
            )
            agreement = {}
            agreement.update(
                lift_validator.compare_metric_tree(
                    original["evaluation"], rebuilt_evaluation, f"{prefix}.evaluation"
                )
            )
            agreement.update(
                lift_validator.compare_metric_tree(
                    original["adaptive_lift_evaluation"],
                    rebuilt_adaptive,
                    f"{prefix}.adaptive",
                )
            )
            agreement.update(
                lift_validator.compare_metric_tree(
                    original["action_rmse_by_representation"],
                    rebuilt_action,
                    f"{prefix}.action",
                )
            )
            local_checks["all_metrics_reproduced"] = all(agreement.values())
            local_checks["all_rebuilt_metrics_finite"] = observability._all_numeric_values_finite(
                (rebuilt_evaluation, rebuilt_adaptive, rebuilt_action)
            )
            alignment_trace = None
            if candidate == benchmark.SCRAMBLED:
                alignment_trace = _alignment_trace(
                    payload=payload, config=config, mode=mode
                )
                local_checks["alignment_trace_passes"] = alignment_trace["passed"]
            checkpoint_validation[prefix] = {
                "path": str(checkpoint_path),
                "sha256": checkpoint_hash,
                "checks": local_checks,
                "agreement_checks": agreement,
                "alignment_trace": alignment_trace,
                "passed": all(local_checks.values()),
            }
            corrected_results[mode][candidate] = {
                "action_rmse_by_representation": rebuilt_action,
                "evaluation": rebuilt_evaluation,
                "adaptive_lift_evaluation": rebuilt_adaptive,
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
        "source_sha256": lift_validator.file_sha256(source_path),
        "training_schedule_sha256": split["schedule_sha256"],
        "observation_system_sha256": source["task"][
            "observation_system_sha256"
        ],
        "evaluation_schedule_sha256": evaluation_hashes,
        "adaptive_address_sha256": adaptive_audit["address_sha256"],
        "adaptive_bit_sha256": adaptive_audit["bit_sha256"],
        "checks": checks,
        "checkpoint_validation": checkpoint_validation,
        "corrected_results": corrected_results,
        "passed": all(checks.values()),
    }


def frozen_seed_gates(seed_result: dict[str, Any]) -> dict[str, bool]:
    results = seed_result["corrected_results"]
    shared = results["vector_plus_adaptive_lift_bit"][benchmark.SHARED]
    scrambled = results["vector_plus_adaptive_lift_bit"][benchmark.SCRAMBLED]
    scrambled_full = results["full_triality"][benchmark.SCRAMBLED]
    l128 = ("early_L128", "late_L128")
    reps = continuous.TRIALITY_REPRESENTATIONS
    scrambled_adaptive_trace = seed_result["checkpoint_validation"][
        f"vector_plus_adaptive_lift_bit.{benchmark.SCRAMBLED}"
    ]["alignment_trace"]
    scrambled_full_trace = seed_result["checkpoint_validation"][
        f"full_triality.{benchmark.SCRAMBLED}"
    ]["alignment_trace"]
    return {
        "shared_adaptive_training_bit_exact": shared["final_components"][
            "adaptive_bit_accuracy"
        ]
        == 1.0,
        "shared_adaptive_action_cap": all(
            value <= 0.04 for value in shared["action_rmse_by_representation"].values()
        ),
        "shared_adaptive_l128_cap": all(
            shared["evaluation"][split][rep]["post_relation_mse"] <= 0.05
            for split in l128
            for rep in reps
        ),
        "shared_adaptive_l128_lift_and_center_exact": all(
            shared["adaptive_lift_evaluation"][split][
                "adaptive_lift_bit_accuracy"
            ]
            == 1.0
            and all(
                shared["evaluation"][split][rep][
                    "center_classification_accuracy"
                ]
                == 1.0
                for rep in ("positive", "negative")
            )
            for split in l128
        ),
        "scrambled_adaptive_training_bit_exact": scrambled["final_components"][
            "adaptive_bit_accuracy"
        ]
        == 1.0,
        "scrambled_adaptive_observed_channels_capable": (
            scrambled["action_rmse_by_representation"]["vector"] <= 0.15
            and scrambled["action_rmse_by_representation"]["positive"] <= 0.10
            and all(
                scrambled["evaluation"][split][rep]["post_relation_mse"] <= 0.13
                for split in l128
                for rep in ("vector", "positive")
            )
        ),
        "scrambled_full_capability": (
            all(
                value <= 0.10
                for value in scrambled_full["action_rmse_by_representation"].values()
            )
            and all(
                scrambled_full["evaluation"][split][rep]["post_relation_mse"]
                <= 0.10
                for split in l128
                for rep in reps
            )
            and all(
                scrambled_full["evaluation"][split][rep][
                    "center_classification_accuracy"
                ]
                == 1.0
                for split in l128
                for rep in ("positive", "negative")
            )
        ),
        "shared_beats_scrambled_adaptive_every_action_view": all(
            shared["action_rmse_by_representation"][rep]
            < scrambled["action_rmse_by_representation"][rep]
            for rep in reps
        ),
        "shared_beats_scrambled_adaptive_every_l128_view": all(
            shared["evaluation"][split][rep]["post_relation_mse"]
            < scrambled["evaluation"][split][rep]["post_relation_mse"]
            for split in l128
            for rep in reps
        ),
        "full_supervision_repairs_scrambled_negative": (
            scrambled_full["action_rmse_by_representation"]["negative"]
            < scrambled["action_rmse_by_representation"]["negative"]
            and all(
                scrambled_full["evaluation"][split]["negative"][
                    "post_relation_mse"
                ]
                < scrambled["evaluation"][split]["negative"]["post_relation_mse"]
                for split in l128
            )
        ),
        "adaptive_negative_alignment_is_decay_only": (
            scrambled_adaptive_trace["residuals"]["negative"]
            <= ALIGNMENT_DECAY_TOLERANCE
        ),
        "full_both_alignments_have_data_updates": (
            scrambled_full_trace["residuals"]["positive"]
            > ALIGNMENT_DATA_UPDATE_MINIMUM
            and scrambled_full_trace["residuals"]["negative"]
            > ALIGNMENT_DATA_UPDATE_MINIMUM
        ),
    }


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
                    seed["corrected_results"][mode][candidate]["evaluation"][split][
                        representation
                    ]["post_relation_mse"]
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
    seeds = [result["seed"] for result in seed_results]
    global_checks = {
        "all_sources_pass_integrity": all(result["passed"] for result in seed_results),
        "seeds_unique": len(seeds) == len(set(seeds)),
    }
    gates = {str(result["seed"]): frozen_seed_gates(result) for result in seed_results}
    if args.development:
        global_checks["development_single_seed"] = seeds == [0]
    else:
        global_checks.update(
            {
                "fresh_seeds_exact": seeds == FRESH_SEEDS,
                "training_schedules_distinct": len(
                    {result["training_schedule_sha256"] for result in seed_results}
                )
                == 3,
                "observation_systems_distinct": len(
                    {result["observation_system_sha256"] for result in seed_results}
                )
                == 3,
                "adaptive_addresses_distinct": len(
                    {result["adaptive_address_sha256"] for result in seed_results}
                )
                == 3,
                "adaptive_bits_distinct": len(
                    {result["adaptive_bit_sha256"] for result in seed_results}
                )
                == 3,
                "evaluation_schedules_distinct": len(
                    {
                        digest
                        for result in seed_results
                        for digest in result["evaluation_schedule_sha256"].values()
                    }
                )
                == 18,
            }
        )
    global_checks["every_frozen_gate_passes_without_median_rescue"] = all(
        all(values.values()) for values in gates.values()
    )
    output = {
        "schema_version": 1,
        "experiment": "Pure Spin8 scrambled-alignment adjudication",
        "status": "development_replay" if args.development else "adjudicated",
        "recorded_at": continuous.now(),
        "seeds": seeds,
        "global_checks": global_checks,
        "frozen_seed_gates": gates,
        "per_seed": seed_results,
        "aggregate_summary": aggregate_summary(seed_results),
        "passed": all(global_checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"global_checks": global_checks, "passed": output["passed"]}, indent=2))
    print(f"wrote {args.output}")
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
