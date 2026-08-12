"""Freeze co-moving FLA implementations from independent tuning processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from benchmark_comoving_fla_delta_rule import TUNED_VARIANTS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(paths: list[Path]) -> dict[str, object]:
    if len(paths) < 3:
        raise ValueError("three independent tuning processes are required")
    inputs = [{"path": str(path), "sha256": _sha256(path)} for path in paths]
    if len({row["sha256"] for row in inputs}) != len(inputs):
        raise ValueError("tuning process artifacts must have distinct contents")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    protocol = reports[0]["protocol"]
    for report in reports:
        if not report["passed"]:
            raise ValueError("tuning input failed")
        for key in (
            "dtype",
            "batch",
            "lengths",
            "state_scalars",
            "problem_seed",
            "tuning_problem_seed",
            "tuning_repeats_in_this_process",
        ):
            if report["protocol"][key] != protocol[key]:
                raise ValueError(f"tuning protocol mismatch for {key}")
        if report["protocol"]["selection_mode"] != "in_process_disjoint_tuning":
            raise ValueError("selection requires in-process disjoint tuning artifacts")
    if protocol["problem_seed"] == protocol["tuning_problem_seed"]:
        raise ValueError("tuning and measurement seeds must differ")

    selections: dict[str, object] = {}
    for row_index, length in enumerate(protocol["lengths"]):
        diagnostics: dict[str, object] = {}
        implementations: dict[str, str] = {}
        for variant in TUNED_VARIANTS:
            candidate_names = reports[0]["rows"][row_index]["tuning"][variant][
                "candidate_ms"
            ].keys()
            process_medians = {
                candidate: [
                    float(
                        report["rows"][row_index]["tuning"][variant]["candidate_ms"]
                        [candidate]["median_ms"]
                    )
                    for report in reports
                ]
                for candidate in candidate_names
            }
            medians = {
                candidate: statistics.median(values)
                for candidate, values in process_medians.items()
            }
            ordered = sorted(medians, key=medians.get)
            winner = ordered[0]
            runner_up = ordered[1]
            implementations[variant] = winner
            diagnostics[variant] = {
                "candidate_process_medians_ms": process_medians,
                "median_of_process_medians_ms": medians,
                "selected": winner,
                "runner_up": runner_up,
                "relative_gap_to_runner_up": medians[runner_up] / medians[winner]
                - 1.0,
            }
        selections[str(length)] = {
            "implementations": implementations,
            "diagnostics": diagnostics,
        }
    return {
        "experiment": "frozen co-moving FLA implementation selection",
        "selection_rule": (
            "minimum median of three independent-process candidate medians on "
            "the disjoint tuning problem"
        ),
        "inputs": inputs,
        "tuning_processes": len(paths),
        "problem_seed": protocol["problem_seed"],
        "tuning_problem_seed": protocol["tuning_problem_seed"],
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
