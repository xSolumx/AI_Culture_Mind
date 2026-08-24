"""Executable matched MQAR campaign for the v1.4 candidate and control."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from .experiments import (
    ExperimentResult,
    HybridVariant,
    TrainingProtocol,
    delta_only_control,
    jsonable,
    run_matched_experiment,
)
from .long_context_screen import write_json_atomic
from .model import HybridMemoryConfig, HybridMemoryLM, parameter_count
from .tasks import DEFAULT_VOCABULARY


def candidate_config() -> HybridMemoryConfig:
    """Return the exact candidate configuration frozen in PREREGISTRATION.md."""

    return HybridMemoryConfig(
        vocab_size=DEFAULT_VOCABULARY.vocab_size,
        model_dim=64,
        layer_plan=("selected_block", "attention"),
        attention_heads=4,
        attention_window_size=128,
        delta_heads=4,
        delta_num_householder=1,
        selected_heads=2,
        selected_blocks=4,
        selected_slots_per_block=4,
        selected_value_dim=16,
        selected_update_rank=1,
        selected_controller_rank=None,
        use_local_conv=True,
        conv_kernel=3,
        expansion=2,
        dropout=0.0,
    )


def build_quality_cohort(
    *, routing_auxiliary_coefficient: float
) -> tuple[tuple[HybridVariant, HybridVariant], TrainingProtocol]:
    """Build the exact prospective G4a cohort without running it."""

    if routing_auxiliary_coefficient not in (0.0, 0.1):
        raise ValueError("quality routing auxiliary coefficient must be 0.0 or 0.1")
    config = candidate_config()
    candidate = HybridVariant("selected_attention_st", config)
    control = delta_only_control("delta_product_common_shell", config)
    protocol = TrainingProtocol(
        task="mqar",
        train_length=512,
        eval_lengths=(512, 2048, 8192),
        updates=600,
        train_batch_size=8,
        eval_batch_size=8,
        eval_batches=8,
        seeds=(41, 43, 47),
        learning_rate=3e-3,
        weight_decay=0.01,
        chunk_size=128,
        parameter_gap_threshold=0.05,
        routing_auxiliary_coefficient=routing_auxiliary_coefficient,
        selected_training_route_mode="straight_through",
    )
    return (candidate, control), protocol


def build_smoke_cohort() -> tuple[
    tuple[HybridVariant, HybridVariant], TrainingProtocol
]:
    """Build a deliberately non-evidentiary, short mechanical training smoke."""

    config = replace(
        candidate_config(),
        model_dim=16,
        attention_heads=2,
        attention_window_size=32,
        delta_heads=2,
        selected_heads=1,
        selected_blocks=2,
        selected_slots_per_block=2,
        selected_value_dim=4,
        conv_kernel=2,
        expansion=1,
    )
    candidate = HybridVariant("selected_attention_st_smoke", config)
    control = delta_only_control("delta_product_common_shell_smoke", config)
    protocol = TrainingProtocol(
        task="mqar",
        train_length=32,
        eval_lengths=(32, 64),
        updates=2,
        train_batch_size=2,
        eval_batch_size=2,
        eval_batches=1,
        seeds=(41,),
        learning_rate=3e-3,
        weight_decay=0.01,
        chunk_size=16,
        parameter_gap_threshold=0.20,
        routing_auxiliary_coefficient=0.0,
        selected_training_route_mode="straight_through",
    )
    return (candidate, control), protocol


def _evaluation_map(result: ExperimentResult, *, untrained: bool) -> dict:
    mapped = {}
    for run in result.runs:
        evaluations = run.untrained_evaluations if untrained else run.evaluations
        mapped[(run.variant_name, run.seed)] = {
            evaluation.length: evaluation for evaluation in evaluations
        }
    return mapped


def adjudicate_quality(result: ExperimentResult) -> dict[str, object]:
    """Apply the frozen G4a decisions to a completed evidentiary cohort."""

    if not isinstance(result, ExperimentResult):
        raise TypeError("result must be an ExperimentResult")
    if not result.evidentiary:
        raise ValueError("quality adjudication requires evidentiary=True")
    candidate_name = result.variants[0].name
    control_name = result.variants[1].name
    before = _evaluation_map(result, untrained=True)
    after = _evaluation_map(result, untrained=False)
    seeds = result.protocol.seeds
    untrained_passed = all(
        before[(candidate_name, seed)][512].exact_accuracy < 0.15 for seed in seeds
    )
    capability_passed = all(
        after[(candidate_name, seed)][512].exact_accuracy > 0.90 for seed in seeds
    )
    paired_regressions = {
        str(length): [
            after[(candidate_name, seed)][length].exact_accuracy
            - after[(control_name, seed)][length].exact_accuracy
            for seed in seeds
        ]
        for length in result.protocol.eval_lengths
    }
    no_large_regression = all(
        difference >= -0.02
        for differences in paired_regressions.values()
        for difference in differences
    )
    mean_2048 = sum(paired_regressions["2048"]) / len(seeds)
    quality_promotion_passed = bool(
        capability_passed and no_large_regression and mean_2048 > 0.0
    )
    return {
        "untrained_below_15_percent": untrained_passed,
        "capability_above_90_percent_every_seed": capability_passed,
        "paired_accuracy_differences_candidate_minus_control": paired_regressions,
        "no_paired_regression_below_minus_2pp": no_large_regression,
        "mean_length_2048_difference": mean_2048,
        "quality_promotion_passed": quality_promotion_passed,
        "label_free_route_passed": (
            quality_promotion_passed
            if result.protocol.routing_auxiliary_coefficient == 0.0
            else None
        ),
        "routing_supervision": (
            "label_free_straight_through"
            if result.protocol.routing_auxiliary_coefficient == 0.0
            else "explicit_task_label_auxiliary"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "quality"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--routing-auxiliary-coefficient", type=float, choices=(0.0, 0.1), default=0.0
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.mode == "quality":
        variants, protocol = build_quality_cohort(
            routing_auxiliary_coefficient=arguments.routing_auxiliary_coefficient
        )
        evidentiary = True
    else:
        variants, protocol = build_smoke_cohort()
        evidentiary = False
    result = run_matched_experiment(
        variants,
        protocol,
        device=arguments.device,
        dtype=torch.float32,
        evidentiary=evidentiary,
    )
    gate = adjudicate_quality(result) if evidentiary else None
    payload = jsonable(
        {
            "campaign": "hybrid_memory_v1_4_g4a",
            "mode": arguments.mode,
            "parameter_counts": {
                variant.name: parameter_count(HybridMemoryLM(variant.config))
                for variant in variants
            },
            "experiment": result,
            "gate": gate,
        }
    )
    write_json_atomic(arguments.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "adjudicate_quality",
    "build_quality_cohort",
    "build_smoke_cohort",
    "candidate_config",
    "main",
]
