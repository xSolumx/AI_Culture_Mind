"""Freeze Linux FLA benchmark backends from independent tuning processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

TUNED_VARIANTS = (
    "direct_slot",
    "triality_slot",
    "local_eager_delta",
    "fla_chunk_delta",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distinct_inputs(paths: list[Path]) -> list[dict[str, str]]:
    inputs = [{"path": str(path), "sha256": _sha256(path)} for path in paths]
    if len({item["sha256"] for item in inputs}) != len(inputs):
        raise ValueError("tuning process artifacts must have distinct contents")
    return inputs


def select(paths: list[Path]) -> dict[str, object]:
    if len(paths) < 2:
        raise ValueError("at least two tuning processes are required")
    inputs = _distinct_inputs(paths)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    lengths = reports[0]["protocol"]["lengths"]
    shared_protocol_keys = (
        "dtype",
        "batch",
        "lengths",
        "state_scalars",
        "problem_seed",
        "tuning_problem_seed",
    )
    reference_tuning_repeats = reports[0]["protocol"].get(
        "tuning_repeats_in_this_process",
        reports[0]["protocol"].get("tuning_repeats_on_disjoint_problem"),
    )
    for report in reports:
        if not report["passed"]:
            raise ValueError("tuning inputs must pass")
        for key in shared_protocol_keys:
            if report["protocol"].get(key) != reports[0]["protocol"].get(key):
                raise ValueError(f"tuning protocol mismatch for {key}")
        tuning_repeats = report["protocol"].get(
            "tuning_repeats_in_this_process",
            report["protocol"].get("tuning_repeats_on_disjoint_problem"),
        )
        if tuning_repeats != reference_tuning_repeats or not tuning_repeats:
            raise ValueError("tuning inputs must share a positive tuning repeat count")
        selection_mode = report["protocol"].get("selection_mode")
        if selection_mode not in (None, "in_process_disjoint_tuning"):
            raise ValueError("selection inputs must contain disjoint in-process tuning")
    if reports[0]["protocol"]["tuning_problem_seed"] == reports[0]["protocol"][
        "problem_seed"
    ]:
        raise ValueError("tuning and measurement problem seeds must differ")

    selections: dict[str, object] = {}
    for row_index, length in enumerate(lengths):
        if any(report["rows"][row_index]["length"] != length for report in reports):
            raise ValueError(f"row alignment mismatch at length {length}")
        diagnostics: dict[str, object] = {}
        chosen: dict[str, str] = {}
        for variant in TUNED_VARIANTS:
            candidate_names = reports[0]["rows"][row_index]["tuning"][variant][
                "candidate_ms"
            ].keys()
            candidate_process_medians = {
                candidate: [
                    float(
                        report["rows"][row_index]["tuning"][variant]["candidate_ms"][
                            candidate
                        ]["median_ms"]
                    )
                    for report in reports
                ]
                for candidate in candidate_names
            }
            aggregate_medians = {
                candidate: statistics.median(process_medians)
                for candidate, process_medians in candidate_process_medians.items()
            }
            ordered = sorted(aggregate_medians, key=aggregate_medians.get)
            winner, runner_up = ordered[:2]
            diagnostics[variant] = {
                "candidate_process_medians_ms": candidate_process_medians,
                "median_of_process_medians_ms": aggregate_medians,
                "selected": winner,
                "runner_up": runner_up,
                "relative_gap_to_runner_up": (
                    aggregate_medians[runner_up] / aggregate_medians[winner] - 1.0
                ),
            }
            chosen[variant] = winner
        selections[str(length)] = {
            "slot_backends": {
                "direct_slot": chosen["direct_slot"],
                "triality_slot": chosen["triality_slot"],
            },
            "local_delta_chunk_size": int(chosen["local_eager_delta"]),
            "fla_delta_chunk_size": int(chosen["fla_chunk_delta"]),
            "diagnostics": diagnostics,
        }
    return {
        "experiment": "frozen Linux FLA benchmark implementation selection",
        "selection_rule": (
            "minimum median of independent-process candidate medians on the "
            "disjoint frozen tuning problem"
        ),
        "inputs": inputs,
        "tuning_processes": len(paths),
        "problem_seed": reports[0]["protocol"]["problem_seed"],
        "tuning_problem_seed": reports[0]["protocol"]["tuning_problem_seed"],
        "selections": selections,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = select(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
