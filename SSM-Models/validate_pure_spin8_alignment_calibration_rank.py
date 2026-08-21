"""Strict replay and prospective gates for the Spin(8) calibration-rank curve."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
from pathlib import Path
from typing import Any

import analyze_pure_spin8_alignment_calibration_rank as rank_analysis
import benchmark_pure_spin8_alignment_calibration_rank as benchmark
import benchmark_pure_spin8_continuous_observation as continuous
import benchmark_pure_spin8_endpoint_observability as observability
import benchmark_pure_spin8_endpoint_supervision as endpoint
import benchmark_pure_spin8_lift_bit_calibration as calibration
import torch
import validate_pure_spin8_lift_bit_calibration as lift_validator

ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent
FRESH_SEEDS = (10, 11, 12)
DEFAULT_SOURCES = tuple(
    ROOT
    / "experiments"
    / "artifacts"
    / f"pure_spin8_alignment_calibration_rank_validation_seed{seed}.json"
    for seed in FRESH_SEEDS
)
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "artifacts"
    / "pure_spin8_alignment_calibration_rank_validation_seeds10_12.json"
)
SHARED_ACTION_MAXIMUM = 0.03
SELECTED_PROBE_RMSE_MAXIMUM = 1e-4
FULL_ALIGNMENT_RMSE_MAXIMUM = 1e-4
PARTIAL_ALIGNMENT_RMSE_MINIMUM = 1e-3
FULL_RANK_METRIC_ATOL = 1e-5
FULL_RANK_VS_RANK27_RATIO_MAXIMUM = 0.5


def resolve_checkpoint(path_value: str) -> Path:
    path = Path(path_value)
    candidates = (
        path,
        REPOSITORY_ROOT / path,
        ROOT / path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(path_value)


def _evaluation_hashes(
    evaluations: dict[str, list[continuous.ContinuousRelationBatch]],
) -> dict[str, str]:
    return {
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


def _rebuilt_metrics(
    model: torch.nn.Module,
    evaluations: dict[str, list[continuous.ContinuousRelationBatch]],
    config: continuous.ContinuousObservationConfig,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "evaluation": {
            key: observability.evaluate_by_representation(
                model, batches, device, config.evaluation_microbatch_size
            )
            for key, batches in evaluations.items()
        },
        "adaptive_lift_evaluation": {
            key: calibration.evaluate_adaptive_lift_bit(
                model, batches, device, config.evaluation_microbatch_size
            )
            for key, batches in evaluations.items()
        },
        "action_rmse_by_representation": observability.action_rmse_by_representation(
            model, evaluations, device
        ),
    }


def _metric_agreement(
    original: dict[str, Any], rebuilt: dict[str, Any], prefix: str
) -> dict[str, bool]:
    checks = {}
    for key in (
        "evaluation",
        "adaptive_lift_evaluation",
        "action_rmse_by_representation",
    ):
        checks.update(
            lift_validator.compare_metric_tree(
                original[key], rebuilt[key], f"{prefix}.{key}"
            )
        )
    return checks


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
    evaluations, evaluation_audits, _ = benchmark._evaluation_batches(
        config, system, device
    )
    evaluation_hashes = _evaluation_hashes(evaluations)
    exact_certificate = rank_analysis.build_certificate(config.seed)
    initialization = benchmark.matched_initialization_audit(config.seed)
    gradient_audit = benchmark.calibration_gradient_audit(config.seed)

    checks: dict[str, bool] = {
        "source_passed": bool(source["passed"]),
        "schema_version": source["schema_version"] == 1,
        "experiment_name": source["experiment"]
        == "Pure Spin8 negative-alignment calibration-rank curve",
        "seed_status": source["status"]
        == ("development" if development else "unadjudicated"),
        "protocol_freeze_matches": development
        or source["protocol_frozen_at"] == benchmark.PROTOCOL_FROZEN_AT,
        "anchor_counts_exact": tuple(source["anchor_counts"])
        == benchmark.ANCHOR_COUNTS,
        "anchor_loss_weight_exact": source["anchor_loss_weight"]
        == benchmark.ANCHOR_LOSS_WEIGHT,
        "rank_profiles_recomputed": exact_certificate["passed"]
        and exact_certificate["exact_rational_profiles"]
        == source["rank_certificate"]["exact_rational_profiles"],
        "initialization_recomputed": initialization
        == source["architecture"]["initialization_audit"],
        "gradient_routes_recomputed": gradient_audit
        == source["task"]["gradient_audit"],
        "training_split_recomputed": split["passed"],
        "training_schedule_hash_reproduced": split["schedule_sha256"]
        == source["task"]["training_split"]["schedule_sha256"],
        "adaptive_address_hash_reproduced": adaptive_audit["address_sha256"]
        == source["task"]["adaptive_lift_bit_audit"]["address_sha256"],
        "adaptive_bit_hash_reproduced": adaptive_audit["bit_sha256"]
        == source["task"]["adaptive_lift_bit_audit"]["bit_sha256"],
        "evaluation_audits_recomputed": all(
            audit["passed"] for audit in evaluation_audits.values()
        ),
        "evaluation_hashes_reproduced": evaluation_hashes
        == source["task"]["evaluation_schedule_sha256"],
        "observation_system_hash_reproduced": continuous.tensor_hash(
            (system.projection, system.bias)
        )
        == source["task"]["observation_system_sha256"],
        "negative_endpoint_targets_never_transferred": source["task"][
            "negative_endpoint_targets_transferred"
        ]
        is False,
        "external_basis_targets_declared": source["task"][
            "negative_alignment_basis_targets_are_external"
        ]
        is True,
        "disjoint_optimizer_contract": source["architecture"][
            "disjoint_router_and_alignment_optimizers"
        ]
        is True,
    }

    checkpoint_validation = {}
    corrected_curve = {}

    shared_original = source["results"]["shared_aligned_reference"]
    shared_path = resolve_checkpoint(shared_original["checkpoint"])
    shared_hash = lift_validator.file_sha256(shared_path)
    shared_payload = torch.load(shared_path, map_location="cpu", weights_only=True)
    shared_model = continuous.SharedPureSpin8Tracker().to(device)
    shared_incompatible = shared_model.load_state_dict(
        shared_payload["state_dict"], strict=True
    )
    shared_rebuilt = _rebuilt_metrics(shared_model, evaluations, config, device)
    shared_agreement = _metric_agreement(
        shared_original, shared_rebuilt, "shared_aligned_reference"
    )
    shared_router_hash = benchmark._router_hash(shared_model)
    shared_checks = {
        "checkpoint_hash": shared_hash == shared_original["checkpoint_sha256"],
        "format_version": shared_payload["format_version"] == 1,
        "candidate": shared_payload["candidate"] == "shared_pure_spin8",
        "mode": shared_payload["mode"] == benchmark.MODE,
        "config_seed": shared_payload["config"]["seed"] == config.seed,
        "config_steps": shared_payload["config"]["steps"] == config.steps,
        "strict_state_dict_reload": not shared_incompatible.missing_keys
        and not shared_incompatible.unexpected_keys,
        "router_hash": shared_router_hash
        == shared_original["final_trainable_parameter_sha256"],
        "all_metrics_reproduced": all(shared_agreement.values()),
    }
    checkpoint_validation["shared_aligned_reference"] = {
        "path": str(shared_path),
        "sha256": shared_hash,
        "checks": shared_checks,
        "agreement_checks": shared_agreement,
        "passed": all(shared_checks.values()),
    }
    corrected_shared = {
        **shared_rebuilt,
        "final_components": shared_original["component_samples"][str(config.steps)],
        "router_sha256": shared_router_hash,
    }
    del shared_model, shared_payload
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for probe_count in benchmark.ANCHOR_COUNTS:
        key = str(probe_count)
        original = source["results"]["scrambled_anchor_curve"][key]
        checkpoint_path = resolve_checkpoint(original["checkpoint"])
        checkpoint_hash = lift_validator.file_sha256(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = benchmark.NegativeOnlyScrambledSpin8Tracker().to(device)
        incompatible = model.load_state_dict(payload["state_dict"], strict=True)
        rebuilt = _rebuilt_metrics(model, evaluations, config, device)
        rebuilt_alignment = benchmark.alignment_diagnostics(model, probe_count)
        agreement = _metric_agreement(original, rebuilt, f"anchor_{probe_count}")
        agreement.update(
            lift_validator.compare_metric_tree(
                original["final_alignment"],
                rebuilt_alignment,
                f"anchor_{probe_count}.alignment",
            )
        )
        router_hash = benchmark._router_hash(model)
        local_checks = {
            "checkpoint_hash": checkpoint_hash == original["checkpoint_sha256"],
            "format_version": payload["format_version"] == 1,
            "candidate": payload["candidate"]
            == "negative_only_scrambled_alignment",
            "mode": payload["mode"] == benchmark.MODE,
            "probe_count": payload["probe_count"] == probe_count,
            "rank": original["exact_independent_rank"]
            == rank_analysis.frame_orbit_rank(probe_count),
            "scalar_budget": original["transmitted_scalar_values"]
            == continuous.SPIN8_DIM * probe_count,
            "anchor_loss_weight": payload["anchor_loss_weight"]
            == benchmark.ANCHOR_LOSS_WEIGHT,
            "config_seed": payload["config"]["seed"] == config.seed,
            "config_steps": payload["config"]["steps"] == config.steps,
            "strict_state_dict_reload": not incompatible.missing_keys
            and not incompatible.unexpected_keys,
            "router_hash_matches_source": router_hash
            == original["final_router_sha256"],
            "router_hash_matches_shared": router_hash == shared_router_hash,
            "all_metrics_reproduced": all(agreement.values()),
            "all_rebuilt_values_finite": observability._all_numeric_values_finite(
                (rebuilt, rebuilt_alignment)
            ),
        }
        checkpoint_validation[f"anchor_{probe_count}"] = {
            "path": str(checkpoint_path),
            "sha256": checkpoint_hash,
            "checks": local_checks,
            "agreement_checks": agreement,
            "passed": all(local_checks.values()),
        }
        corrected_curve[key] = {
            **rebuilt,
            "alignment": rebuilt_alignment,
            "final_components": original["component_samples"][str(config.steps)],
            "router_sha256": router_hash,
        }
        del model, payload
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    checks["all_checkpoints_strictly_reloaded"] = all(
        item["passed"] for item in checkpoint_validation.values()
    )
    checks["all_router_hashes_bitwise_identical"] = all(
        row["router_sha256"] == shared_router_hash
        for row in corrected_curve.values()
    )
    return {
        "seed": config.seed,
        "source": str(source_path),
        "source_sha256": lift_validator.file_sha256(source_path),
        "training_schedule_sha256": split["schedule_sha256"],
        "observation_system_sha256": source["task"]["observation_system_sha256"],
        "evaluation_schedule_sha256": evaluation_hashes,
        "adaptive_address_sha256": adaptive_audit["address_sha256"],
        "adaptive_bit_sha256": adaptive_audit["bit_sha256"],
        "checks": checks,
        "checkpoint_validation": checkpoint_validation,
        "corrected_results": {
            "shared_aligned_reference": corrected_shared,
            "scrambled_anchor_curve": corrected_curve,
        },
        "passed": all(checks.values()),
    }


def frozen_seed_gates(seed_result: dict[str, Any]) -> dict[str, bool]:
    results = seed_result["corrected_results"]
    shared = results["shared_aligned_reference"]
    curve = results["scrambled_anchor_curve"]
    full7 = curve["7"]
    redundant8 = curve["8"]
    rank27 = curve["6"]
    l128 = ("early_L128", "late_L128")
    representations = continuous.TRIALITY_REPRESENTATIONS

    def close(left: float, right: float) -> bool:
        return math.isclose(left, right, rel_tol=0.0, abs_tol=FULL_RANK_METRIC_ATOL)

    return {
        "source_and_replay_integrity": seed_result["passed"],
        "shared_action_absolute_quality": all(
            shared["action_rmse_by_representation"][representation]
            <= SHARED_ACTION_MAXIMUM
            for representation in representations
        ),
        "shared_adaptive_bit_exact": shared["final_components"][
            "adaptive_bit_accuracy"
        ]
        == 1.0,
        "every_nonzero_anchor_frame_is_fitted": all(
            curve[str(count)]["alignment"]["selected_probe_rmse"]
            <= SELECTED_PROBE_RMSE_MAXIMUM
            for count in range(1, 9)
        ),
        "rank27_retains_unidentified_alignment": rank27["alignment"][
            "full_identity_rmse"
        ]
        >= PARTIAL_ALIGNMENT_RMSE_MINIMUM,
        "rank28_seven_probes_recovers_alignment": full7["alignment"][
            "full_identity_rmse"
        ]
        <= FULL_ALIGNMENT_RMSE_MAXIMUM,
        "rank28_eight_probes_recovers_alignment": redundant8["alignment"][
            "full_identity_rmse"
        ]
        <= FULL_ALIGNMENT_RMSE_MAXIMUM,
        "seven_probe_negative_action_matches_aligned": close(
            full7["action_rmse_by_representation"]["negative"],
            shared["action_rmse_by_representation"]["negative"],
        ),
        "eighth_probe_adds_no_negative_action_gain": close(
            redundant8["action_rmse_by_representation"]["negative"],
            full7["action_rmse_by_representation"]["negative"],
        ),
        "full_rank_beats_rank27_negative_action_by_factor": full7[
            "action_rmse_by_representation"
        ]["negative"]
        <= FULL_RANK_VS_RANK27_RATIO_MAXIMUM
        * rank27["action_rmse_by_representation"]["negative"],
        "full_rank_beats_rank27_every_negative_l128_split": all(
            full7["evaluation"][split]["negative"]["post_relation_mse"]
            < rank27["evaluation"][split]["negative"]["post_relation_mse"]
            for split in l128
        ),
        "full_rank_negative_l128_matches_aligned": all(
            close(
                full7["evaluation"][split]["negative"]["post_relation_mse"],
                shared["evaluation"][split]["negative"]["post_relation_mse"],
            )
            for split in l128
        ),
        "seven_and_eight_probe_l128_are_redundant": all(
            close(
                full7["evaluation"][split]["negative"]["post_relation_mse"],
                redundant8["evaluation"][split]["negative"]["post_relation_mse"],
            )
            for split in l128
        ),
        "router_trajectory_is_bitwise_matched": len(
            {
                shared["router_sha256"],
                *(row["router_sha256"] for row in curve.values()),
            }
        )
        == 1,
    }


def aggregate_summary(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = {}
    for count in benchmark.ANCHOR_COUNTS:
        key = str(count)
        action = [
            result["corrected_results"]["scrambled_anchor_curve"][key][
                "action_rmse_by_representation"
            ]["negative"]
            for result in seed_results
        ]
        alignment = [
            result["corrected_results"]["scrambled_anchor_curve"][key][
                "alignment"
            ]["full_identity_rmse"]
            for result in seed_results
        ]
        l128 = [
            result["corrected_results"]["scrambled_anchor_curve"][key][
                "evaluation"
            ][split]["negative"]["post_relation_mse"]
            for result in seed_results
            for split in ("early_L128", "late_L128")
        ]
        rows[key] = {
            "probe_count": count,
            "exact_rank": rank_analysis.frame_orbit_rank(count),
            "stabilizer_dimension": continuous.SPIN8_BIVECTOR_DIM
            - rank_analysis.frame_orbit_rank(count),
            "transmitted_scalar_values": continuous.SPIN8_DIM * count,
            "negative_action_rmse": {
                "median": statistics.median(action),
                "minimum": min(action),
                "maximum": max(action),
            },
            "full_alignment_rmse": {
                "median": statistics.median(alignment),
                "minimum": min(alignment),
                "maximum": max(alignment),
            },
            "negative_l128_post_relation_mse": {
                "median": statistics.median(l128),
                "minimum": min(l128),
                "maximum": max(l128),
            },
        }
    return rows


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
    device = torch.device(args.device)
    seed_results = [
        validate_source(path, device=device, development=args.development)
        for path in args.sources
    ]
    seeds = [result["seed"] for result in seed_results]
    gates = {
        str(result["seed"]): frozen_seed_gates(result) for result in seed_results
    }
    global_checks = {
        "all_sources_pass_integrity": all(result["passed"] for result in seed_results),
        "seeds_unique": len(seeds) == len(set(seeds)),
    }
    if args.development:
        global_checks["development_single_seed"] = seeds == [0]
    else:
        global_checks.update(
            {
                "fresh_seeds_exact": tuple(seeds) == FRESH_SEEDS,
                "training_schedules_distinct": len(
                    {result["training_schedule_sha256"] for result in seed_results}
                )
                == len(seed_results),
                "observation_systems_distinct": len(
                    {result["observation_system_sha256"] for result in seed_results}
                )
                == len(seed_results),
                "all_evaluation_schedules_distinct": len(
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
        all(seed_gates.values()) for seed_gates in gates.values()
    )
    output = {
        "schema_version": 1,
        "experiment": "Pure Spin8 alignment calibration-rank adjudication",
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
    print(json.dumps(output["global_checks"], indent=2))
    print(f"wrote {args.output}")
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
