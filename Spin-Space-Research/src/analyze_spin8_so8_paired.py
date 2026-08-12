"""Summarize the preregistered positive-Spin(8)/generic-SO(8) paired cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median


FAMILIES = ("pure_spin8_positive", "pure_so8_exponential")


def _metrics(result: dict[str, object]) -> dict[str, float | bool | int]:
    diagnostics = result["representation_diagnostics"]
    central = result["central_pair_evaluation"]
    streaming = result["streaming_equivalence"]
    return {
        "trainable_parameters": int(result["trainable_parameters"]),
        "action_parameters": int(result["action_parameters"]),
        "final_training_loss": float(result["trajectory"]["2000"]["loss"]),
        "final_training_accuracy": float(result["trajectory"]["2000"]["accuracy"]),
        "dense_minimum_pair_member_accuracy": float(
            central["minimum_pair_member_accuracy"]
        ),
        "dense_minimum_both_members_accuracy": float(
            central["minimum_both_members_correct_accuracy"]
        ),
        "dense_gate_pass": bool(central["gate_pass"]),
        "linear_homomorphism_rms": float(diagnostics["linear_homomorphism_rms"]),
        "streaming_max_abs_error": max(float(value) for value in streaming.values()),
        "elapsed_seconds": float(result["elapsed_seconds"]),
    }


def analyze(source: dict[str, object]) -> dict[str, object]:
    by_seed: dict[int, dict[str, dict[str, object]]] = {}
    for result in source["results"]:
        seed = int(result["seed"])
        family = str(result["family"])
        if family not in FAMILIES:
            raise ValueError(f"unexpected family {family!r}")
        by_seed.setdefault(seed, {})[family] = result
    if any(set(pair) != set(FAMILIES) for pair in by_seed.values()):
        raise ValueError("every seed must contain both chart families")

    paired = []
    for seed in sorted(by_seed):
        positive = _metrics(by_seed[seed][FAMILIES[0]])
        generic = _metrics(by_seed[seed][FAMILIES[1]])
        paired.append(
            {
                "seed": seed,
                "positive_spin8": positive,
                "generic_so8": generic,
                "generic_minus_positive": {
                    key: float(generic[key]) - float(positive[key])
                    for key in (
                        "final_training_loss",
                        "dense_minimum_pair_member_accuracy",
                        "dense_minimum_both_members_accuracy",
                        "linear_homomorphism_rms",
                        "elapsed_seconds",
                    )
                },
            }
        )

    summaries = {}
    for family, label in zip(FAMILIES, ("positive_spin8", "generic_so8")):
        rows = [row[label] for row in paired]
        summaries[family] = {
            "dense_gate_pass_count": sum(bool(row["dense_gate_pass"]) for row in rows),
            "mean_dense_minimum_pair_member_accuracy": mean(
                float(row["dense_minimum_pair_member_accuracy"]) for row in rows
            ),
            "median_dense_minimum_pair_member_accuracy": median(
                float(row["dense_minimum_pair_member_accuracy"]) for row in rows
            ),
            "mean_linear_homomorphism_rms": mean(
                float(row["linear_homomorphism_rms"]) for row in rows
            ),
            "mean_elapsed_seconds": mean(float(row["elapsed_seconds"]) for row in rows),
        }
    checks = {
        "five_paired_fresh_seeds": sorted(by_seed) == [60, 61, 62, 63, 64],
        "parameter_counts_match_in_every_pair": all(
            row["positive_spin8"]["trainable_parameters"]
            == row["generic_so8"]["trainable_parameters"]
            and row["positive_spin8"]["action_parameters"]
            == row["generic_so8"]["action_parameters"]
            for row in paired
        ),
        "all_final_curriculum_batches_fit": all(
            row[label]["final_training_accuracy"] == 1.0
            for row in paired
            for label in ("positive_spin8", "generic_so8")
        ),
        "all_streaming_checks_pass": all(
            row[label]["streaming_max_abs_error"] <= 1e-5
            for row in paired
            for label in ("positive_spin8", "generic_so8")
        ),
    }
    return {
        "experiment": "paired positive-Spin8 versus generic-SO8 AdamW cohort",
        "source_experiment": source["experiment"],
        "paired_seed_count": len(paired),
        "checks": checks,
        "protocol_passed": all(checks.values()),
        "summaries": summaries,
        "paired_results": paired,
        "interpretation_boundary": (
            "the transition families are algebraically identical; paired "
            "differences measure optimizer-coordinate bias, not capacity"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    report = analyze(source)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
