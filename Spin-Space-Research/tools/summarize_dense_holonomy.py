"""Summarize dense multi-scale holonomy reliability without hiding outliers."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_record(path: Path) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    result = report["results"][0]
    sequence_length = report["config"]["sequence_length"]
    accuracies = {
        int(length): 100.0 * metrics["final_position_accuracy"]
        for length, metrics in result["length_generalization"].items()
    }
    dense = {
        length: accuracy
        for length, accuracy in accuracies.items()
        if (
            sequence_length <= length <= 16 * sequence_length
            and length % sequence_length == 0
        )
    }
    minimum_length, minimum_accuracy = min(dense.items(), key=lambda item: item[1])
    scale_margins = [
        metrics["alternate_minimum_nearest_negative_margin"]
        for metrics in result["path_holonomy_diagnostics_by_multiplier"].values()
    ]
    diagnostics = result["mechanism_diagnostics"]
    ratio_diagnostics = diagnostics.get("generator_angle_ratio_diagnostics", [])
    rational_errors = [
        entry["nearest_positive_rational_with_terms_at_most_8"]["absolute_error"]
        for entry in ratio_diagnostics
        if entry["nearest_positive_rational_with_terms_at_most_8"] is not None
    ]
    return {
        "path": str(path),
        "seed": report["config"]["seed"],
        "accuracies": {str(length): value for length, value in sorted(accuracies.items())},
        "dense_minimum_accuracy": minimum_accuracy,
        "dense_minimum_length": minimum_length,
        "checkpoint_functional_pass": (
            accuracies[sequence_length] >= 95.0
            and accuracies[4 * sequence_length] > 90.0
            and accuracies[8 * sequence_length] > 90.0
        ),
        "positive_margin_state_contract": min(scale_margins) > 0.0,
        "dense_accuracy_floor_pass": minimum_accuracy >= 90.0,
        "minimum_holonomy_scale_margin": min(scale_margins),
        "mean_low_rational_ratio_error": (
            statistics.fmean(rational_errors) if rational_errors else None
        ),
        "minimum_low_rational_ratio_error": min(rational_errors, default=None),
        "generator_axis_dot_products": [
            entry["axis_dot_product"] for entry in ratio_diagnostics
        ],
        "per_channel_maximum_commutator": diagnostics.get(
            "per_channel_maximum_generator_commutator_separation", []
        ),
        "maximum_finite_order_relator_rms": max(
            (
                relator["maximum_channel_rms"]
                for relator in diagnostics.get("finite_order_relator_residuals", [])
            ),
            default=None,
        ),
        "raw_linear_homomorphism_rms": diagnostics["linear_homomorphism_rms"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    records = sorted((load_record(path) for path in args.reports), key=lambda x: x["seed"])
    lengths = sorted({int(length) for record in records for length in record["accuracies"]})
    aggregate = {
        "seeds": len(records),
        "checkpoint_functional_passes": sum(
            record["checkpoint_functional_pass"] for record in records
        ),
        "positive_margin_state_contract_passes": sum(
            record["positive_margin_state_contract"] for record in records
        ),
        "dense_accuracy_floor_passes": sum(
            record["dense_accuracy_floor_pass"] for record in records
        ),
        "per_length_accuracy": {
            str(length): {
                "mean": statistics.fmean(
                    record["accuracies"][str(length)]
                    for record in records
                    if str(length) in record["accuracies"]
                ),
                "median": statistics.median(
                    record["accuracies"][str(length)]
                    for record in records
                    if str(length) in record["accuracies"]
                ),
                "minimum": min(
                    record["accuracies"][str(length)]
                    for record in records
                    if str(length) in record["accuracies"]
                ),
            }
            for length in lengths
        },
    }
    print(json.dumps({"aggregate": aggregate, "per_seed": records}, indent=2))


if __name__ == "__main__":
    main()
