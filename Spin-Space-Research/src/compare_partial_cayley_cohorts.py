"""Compare two partial-evidence cohorts for post-recovery bitwise invariance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def indexed_results(path: Path) -> dict[int, dict[str, object]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {result["training_seed"]: result for result in report["results"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left_report", type=Path)
    parser.add_argument("right_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left = indexed_results(args.left_report)
    right = indexed_results(args.right_report)
    if left.keys() != right.keys():
        raise ValueError("cohorts contain different seeds")

    results = []
    for seed in sorted(left):
        left_result = left[seed]
        right_result = right[seed]
        left_checkpoint = torch.load(
            left_result["checkpoint"], map_location="cpu", weights_only=False
        )
        right_checkpoint = torch.load(
            right_result["checkpoint"], map_location="cpu", weights_only=False
        )
        left_state = left_checkpoint["state_dict"]
        right_state = right_checkpoint["state_dict"]
        same_keys = left_state.keys() == right_state.keys()
        tensor_equal = same_keys and all(
            torch.equal(left_state[key], right_state[key]) for key in left_state
        )
        trajectory_equal = left_result["trajectory"] == right_result["trajectory"]
        structural_equal = all(
            left_result[key] == right_result[key]
            for key in (
                "trigger_step",
                "anchor_channel",
                "candidate_index",
                "trigger_alignment_rms",
                "trigger_runner_up_rms",
                "trigger_commutator",
                "compiler_invariance_rms",
                "compiler_homomorphism_rms",
                "mechanism_diagnostics",
                "dense_evaluation",
                "long_stress",
            )
        )
        results.append(
            {
                "seed": seed,
                "state_dict_bitwise_equal": tensor_equal,
                "trajectory_exactly_equal": trajectory_equal,
                "post_recovery_reports_exactly_equal": structural_equal,
            }
        )
    report = {
        "experiment": "partial-Cayley post-recovery invariance comparison",
        "left": str(args.left_report),
        "right": str(args.right_report),
        "all_state_dicts_bitwise_equal": all(
            result["state_dict_bitwise_equal"] for result in results
        ),
        "all_trajectories_exactly_equal": all(
            result["trajectory_exactly_equal"] for result in results
        ),
        "all_post_recovery_reports_exactly_equal": all(
            result["post_recovery_reports_exactly_equal"] for result in results
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
