"""Build a claim-audited synthesis of the frozen matched-retrieval campaign."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

VARIANTS = (
    "direct_slot_joint",
    "triality_slot_joint",
    "delta_chunk_joint",
    "delta_chunk_oracle",
    "fast_weight_joint",
)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _quality_means(
    rows: list[dict[str, object]],
    *,
    cell_key: str,
    metric_key: str,
    corrupted: bool,
) -> dict[str, float]:
    return {
        name: _mean(
            [
                float(cell[metric_key][name]["mean_query_cosine"])
                for row in rows
                for cell in row[cell_key]
                if (float(cell["requested_perturbation_norm"]) > 0) == corrupted
            ]
        )
        for name in VARIANTS
    }


def _paired_wins(
    rows: list[dict[str, object]], *, cell_key: str, metric_key: str
) -> dict[str, int]:
    mean_wins = 0
    every_cell_wins = 0
    for row in rows:
        cells = [
            cell
            for cell in row[cell_key]
            if float(cell["requested_perturbation_norm"]) > 0
        ]
        scores = {
            name: _mean(
                [float(cell[metric_key][name]["mean_query_cosine"]) for cell in cells]
            )
            for name in VARIANTS
        }
        mean_wins += int(
            scores["direct_slot_joint"] > scores["delta_chunk_joint"]
            and scores["triality_slot_joint"] > scores["delta_chunk_joint"]
        )
        every_cell_wins += int(
            all(
                float(cell[metric_key]["direct_slot_joint"]["mean_query_cosine"])
                > float(cell[metric_key]["delta_chunk_joint"]["mean_query_cosine"])
                and float(cell[metric_key]["triality_slot_joint"]["mean_query_cosine"])
                > float(cell[metric_key]["delta_chunk_joint"]["mean_query_cosine"])
                for cell in cells
            )
        )
    return {
        "paired_seed_mean_wins": mean_wins,
        "paired_seed_every_cell_wins": every_cell_wins,
    }


def _direct_triality_effects(
    rows: list[dict[str, object]], *, cell_key: str, metric_key: str
) -> dict[str, object]:
    result = {}
    for label, corrupted in (("clean", False), ("corrupted", True)):
        differences = [
            float(cell[metric_key]["triality_slot_joint"]["mean_query_cosine"])
            - float(cell[metric_key]["direct_slot_joint"]["mean_query_cosine"])
            for row in rows
            for cell in row[cell_key]
            if (float(cell["requested_perturbation_norm"]) > 0) == corrupted
        ]
        result[label] = {
            "cells": len(differences),
            "mean_triality_minus_direct": _mean(differences),
            "maximum_absolute_difference": max(map(abs, differences)),
            "triality_cell_wins": sum(value > 0 for value in differences),
            "direct_cell_wins": sum(value < 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
        }
    return result


def _cold_length_curves(rows: list[dict[str, object]]) -> dict[str, object]:
    lengths = sorted({int(cell["length"]) for cell in rows[0]["stream_cells"]})
    perturbations = sorted(
        {float(cell["requested_perturbation_norm"]) for cell in rows[0]["stream_cells"]}
    )
    return {
        str(perturbation): {
            str(length): {
                name: _mean(
                    [
                        float(cell["cold_metrics"][name]["mean_query_cosine"])
                        for row in rows
                        for cell in row["stream_cells"]
                        if int(cell["length"]) == length
                        and float(cell["requested_perturbation_norm"]) == perturbation
                    ]
                )
                for name in VARIANTS
            }
            for length in lengths
        }
        for perturbation in perturbations
    }


def _task_a(task_a: dict[str, object]) -> dict[str, object]:
    rows = task_a["results"]
    overwrite_wins = _paired_wins(
        rows, cell_key="overwrite_cells", metric_key="metrics"
    )
    stream_wins = _paired_wins(rows, cell_key="stream_cells", metric_key="metrics")
    hot_wins = _paired_wins(rows, cell_key="stream_cells", metric_key="hot_metrics")
    cold_wins = _paired_wins(rows, cell_key="stream_cells", metric_key="cold_metrics")
    query_count = sum(
        int(cell["query_count"])
        for row in rows
        for cell in row["overwrite_cells"] + row["stream_cells"]
    )
    maximum_hard_gauge_gap = max(
        float(cell["diagnostics"]["direct_triality_prediction_max_abs_gap"])
        for row in rows
        for cell in row["oracle_hard_route_capacity_cells"]
    )
    maximum_oracle_delta_error = max(
        float(cell["metrics"]["delta_chunk_oracle"]["maximum_relative_squared_error"])
        for row in rows
        for cell in row["overwrite_cells"] + row["stream_cells"]
    )
    return {
        "seeds": len(rows),
        "overwrite_cells_per_seed": len(rows[0]["overwrite_cells"]),
        "stream_cells_per_seed": len(rows[0]["stream_cells"]),
        "total_reported_queries": query_count,
        "implementation": {
            "passed": bool(task_a["summary"]["implementation_gate_passed"]),
            "oracle_delta_all_cell_passes": int(
                task_a["summary"]["oracle_delta_all_cell_passes"]
            ),
            "hard_route_direct_triality_gauge_passes": int(
                task_a["summary"]["hard_route_direct_triality_gauge_passes"]
            ),
            "maximum_chunk_recurrent_abs_error": float(
                task_a["summary"]["maximum_chunk_recurrent_abs_error"]
            ),
            "maximum_hard_route_direct_triality_gap": maximum_hard_gauge_gap,
            "maximum_oracle_delta_relative_squared_error": maximum_oracle_delta_error,
        },
        "mean_quality": {
            "overwrite_clean": _quality_means(
                rows,
                cell_key="overwrite_cells",
                metric_key="metrics",
                corrupted=False,
            ),
            "overwrite_corrupted": _quality_means(
                rows,
                cell_key="overwrite_cells",
                metric_key="metrics",
                corrupted=True,
            ),
            "stream_clean": _quality_means(
                rows,
                cell_key="stream_cells",
                metric_key="metrics",
                corrupted=False,
            ),
            "stream_corrupted": _quality_means(
                rows,
                cell_key="stream_cells",
                metric_key="metrics",
                corrupted=True,
            ),
            "stream_hot_corrupted": _quality_means(
                rows,
                cell_key="stream_cells",
                metric_key="hot_metrics",
                corrupted=True,
            ),
            "stream_cold_corrupted": _quality_means(
                rows,
                cell_key="stream_cells",
                metric_key="cold_metrics",
                corrupted=True,
            ),
        },
        "hard_routing_paired_verdict": {
            "overwrite": overwrite_wins,
            "stream_combined": stream_wins,
            "stream_hot": hot_wins,
            "stream_cold": cold_wins,
            "decision_rule_supported": (
                overwrite_wins["paired_seed_every_cell_wins"] >= 8
                and int(task_a["summary"]["oracle_delta_all_cell_passes"]) >= 8
            ),
        },
        "direct_triality_effects": {
            "overwrite": _direct_triality_effects(
                rows, cell_key="overwrite_cells", metric_key="metrics"
            ),
            "stream_combined": _direct_triality_effects(
                rows, cell_key="stream_cells", metric_key="metrics"
            ),
            "stream_hot": _direct_triality_effects(
                rows, cell_key="stream_cells", metric_key="hot_metrics"
            ),
            "stream_cold": _direct_triality_effects(
                rows, cell_key="stream_cells", metric_key="cold_metrics"
            ),
            "triality_memory_law_advantage_supported": False,
        },
        "cold_length_curves": _cold_length_curves(rows),
        "sample_efficiency": task_a["summary"]["sample_efficiency_by_steps_per_stage"],
        "claim_boundary": {
            "learned_delta_failure_is_update_capacity_failure": False,
            "reason": "oracle delta is exact in every cell",
            "soft_route_difference_is_hard_slot_gauge_violation": False,
            "language_or_production_superiority_established": False,
        },
    }


def _systems(performance: dict[str, object]) -> dict[str, object]:
    rows = []
    for row in performance["rows"]:
        direct_forward = float(row["forward_ms"]["direct_slot_hybrid"]["median_ms"])
        direct_backward = float(
            row["forward_backward_ms"]["direct_slot_hybrid"]["median_ms"]
        )
        rows.append(
            {
                "length": int(row["length"]),
                "slot_backend": row["slot_backend"],
                "forward_median_ms": {
                    name: float(row["forward_ms"][name]["median_ms"])
                    for name in row["forward_ms"]
                },
                "forward_backward_median_ms": {
                    name: float(row["forward_backward_ms"][name]["median_ms"])
                    for name in row["forward_backward_ms"]
                },
                "forward_ratio_to_direct": {
                    name: float(row["forward_ms"][name]["median_ms"]) / direct_forward
                    for name in row["forward_ms"]
                },
                "forward_backward_ratio_to_direct": {
                    name: float(row["forward_backward_ms"][name]["median_ms"])
                    / direct_backward
                    for name in row["forward_backward_ms"]
                },
                "forward_peak_bytes": row["forward_incremental_cuda_memory_bytes"],
                "forward_backward_peak_bytes": row[
                    "forward_backward_incremental_cuda_memory_bytes"
                ],
            }
        )
    return {
        "device_metadata": performance["device_metadata"],
        "protocol": performance["protocol"],
        "rows": rows,
        "direct_fastest_forward_all_lengths": all(
            min(
                row["forward_ms"],
                key=lambda name: float(row["forward_ms"][name]["median_ms"]),
            )
            == "direct_slot_hybrid"
            for row in performance["rows"]
        ),
        "claim_boundary": performance["claim_boundary"],
    }


def _task_b(
    blind: dict[str, object],
    identification: dict[str, object],
    replication: dict[str, object] | None = None,
) -> dict[str, object]:
    rows = blind["results"]
    variants = (
        "joint_triality",
        "joint_direct",
        "independent_binding",
        "independent_direct",
        "direct_negative_oracle",
    )
    long_quality = {
        name: {
            "mean": _mean(
                [
                    float(row["variants"][name]["evaluation"][-1]["mean_query_cosine"])
                    for row in rows
                ]
            ),
            "minimum_seed": min(
                float(row["variants"][name]["evaluation"][-1]["mean_query_cosine"])
                for row in rows
            ),
        }
        for name in variants
    }
    complement_quality = {
        name: {
            "mean": _mean(
                [
                    float(
                        row["variants"][name]["negative_subspaces"][
                            "complement_mean_cosine"
                        ]
                    )
                    for row in rows
                ]
            ),
            "minimum_seed": min(
                float(
                    row["variants"][name]["negative_subspaces"][
                        "complement_mean_cosine"
                    ]
                )
                for row in rows
            ),
        }
        for name in variants
    }
    family_summaries = {
        name: family["summary"] for name, family in identification["families"].items()
    }
    shared_family_wins = (
        int(blind["summary"]["joint_direct_length2048_wins"]) >= 8
        and int(blind["summary"]["joint_complement_wins"]) >= 8
    )
    replication_summary = replication["summary"] if replication is not None else None
    replication_closed = bool(
        replication_summary
        and replication_summary["task_b_decision_rule_fully_empirically_closed"]
    )
    return {
        "action_completion": {
            "seeds": len(rows),
            "source_summary": blind["summary"],
            "length_2048_quality": long_quality,
            "held_out_negative_complement_quality": complement_quality,
            "shared_family_beats_independent_direct": shared_family_wins,
            "binding_bypass_detected": int(
                blind["summary"]["independent_binding_retrieval_parity"]
            )
            >= 8,
            "reason": (
                "independent binding retrieves nearly perfectly despite its failed "
                "negative action, while independent direct exposes the failure"
            ),
        },
        "equivariant_identification": {
            "passed": bool(identification["passed"]),
            "families": family_summaries,
            "restricted_generic_interpolates_but_fails_orbit": True,
            "group_augmented_generic_recovers_orbit": True,
            "so3_control_reproduces_structured_prior_effect": True,
            "exceptional_triality_advantage_supported": False,
        },
        "delta_action_row": {
            "status": (
                "prospectively replicated with retained learned parameters"
                if replication_closed
                else "theorem-backed but not independently rerun in Task B"
            ),
            "hard_key_equivalence": (
                "with orthogonal one-hot keys, delta memory and direct slots obey "
                "the same overwrite, value-transport, and read equations"
            ),
            "prospective_replication": replication_summary,
            "task_b_decision_rule_fully_empirically_closed": replication_closed,
        },
        "claim_boundary": {
            "spin8_shared_representation_prior_supported": (
                shared_family_wins and replication_closed
            ),
            "triality_specific_memory_update_supported": False,
            "generic_equivariance_prior_ruled_out": False,
        },
    }


def analyze(
    task_a: dict[str, object],
    performance: dict[str, object],
    blind: dict[str, object],
    identification: dict[str, object],
    task_b_replication: dict[str, object] | None = None,
) -> dict[str, object]:
    task_b_closed = bool(
        task_b_replication
        and task_b_replication["summary"][
            "task_b_decision_rule_fully_empirically_closed"
        ]
    )
    return {
        "experiment": "matched learned retrieval campaign synthesis",
        "protocol": "MATCHED_LEARNED_RETRIEVAL_PREREGISTRATION.md",
        "task_a": _task_a(task_a),
        "measured_systems_tier": _systems(performance),
        "task_b": _task_b(blind, identification, task_b_replication),
        "programme_verdict": {
            "best_current_memory_scanning_direction": (
                "hierarchical hard or block routing over exact local memory, with "
                "co-moving fused delta retained as the global continuous-key path"
                if task_b_closed
                else "hard or discretized semantic routing with the optimized local "
                "slot scan; retain chunkwise delta as the continuous-key baseline"
            ),
            "triality_role": (
                "use triality as a shared cross-view representation prior when "
                "actions are partially observed, not as a generic overwrite-law claim"
            ),
            "dirac_gram_prerequisite": False,
            "next_blocker": (
                "overlapping-semantic large-slot routing with a measured gather kernel"
                if task_b_closed
                else "fused compact-WY delta comparison and an explicit Task B delta-action replay"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-a", type=Path, required=True)
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--blind-action", type=Path, required=True)
    parser.add_argument("--identification", type=Path, required=True)
    parser.add_argument("--task-b-replication", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        json.loads(args.task_a.read_text(encoding="utf-8")),
        json.loads(args.performance.read_text(encoding="utf-8")),
        json.loads(args.blind_action.read_text(encoding="utf-8")),
        json.loads(args.identification.read_text(encoding="utf-8")),
        (
            json.loads(args.task_b_replication.read_text(encoding="utf-8"))
            if args.task_b_replication is not None
            else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["programme_verdict"], indent=2))


if __name__ == "__main__":
    main()
