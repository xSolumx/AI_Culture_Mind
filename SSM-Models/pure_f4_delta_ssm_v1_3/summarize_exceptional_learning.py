"""Fail-closed summary for F4/E6 hidden-coordinate learning cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED = {"f4": ("spin9", "f4"), "e6": ("f4", "e6")}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(paths: list[Path], expected_seeds: tuple[int, ...]) -> dict[str, object]:
    cells: dict[tuple[str, int], tuple[Path, dict[str, object]]] = {}
    sources = None
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        target = str(report["target"])
        if target not in EXPECTED:
            raise ValueError(f"unexpected target {target}")
        if tuple(report["candidates"]) != EXPECTED[target]:
            raise ValueError(f"candidate controls changed for {target}")
        if report["environment"].get("compute_capability") != [7, 5]:
            raise ValueError("learning cohort requires exact SM75")
        rows = report["rows"]
        seeds = {int(row["config"]["seed"]) for row in rows}
        if len(seeds) != 1:
            raise ValueError("rows in one artifact must share one seed")
        seed = next(iter(seeds))
        key = (target, seed)
        if key in cells:
            raise ValueError(f"duplicate target/seed cell {key}")
        if seed not in expected_seeds:
            raise ValueError(f"unexpected seed {seed}")
        if sources is None:
            sources = report["source_sha256"]
        elif report["source_sha256"] != sources:
            raise ValueError("source drift across learning artifacts")
        cells[key] = (path, report)
    expected_cells = {
        (target, seed) for target in EXPECTED for seed in expected_seeds
    }
    if set(cells) != expected_cells:
        raise ValueError("learning target/seed grid is incomplete")

    targets: dict[str, object] = {}
    for target, (predecessor, correct) in EXPECTED.items():
        correct_errors = []
        predecessor_errors = []
        cubic_errors = []
        pass_count = 0
        by_seed = {}
        for seed in expected_seeds:
            report = cells[(target, seed)][1]
            rows = {row["config"]["candidate_algebra"]: row for row in report["rows"]}
            if set(rows) != {predecessor, correct}:
                raise ValueError(f"row identity changed for {target}/{seed}")
            predecessor_eval = rows[predecessor]["evaluations"][-1]
            correct_eval = rows[correct]["evaluations"][-1]
            predecessor_error = float(predecessor_eval["relative_probe_error"])
            correct_error = float(correct_eval["relative_probe_error"])
            cubic = float(correct_eval["maximum_cubic_error"])
            if not all(math.isfinite(value) for value in (predecessor_error, correct_error, cubic)):
                raise ValueError("nonfinite learning metric")
            predecessor_errors.append(predecessor_error)
            correct_errors.append(correct_error)
            cubic_errors.append(cubic)
            pass_count += int(bool(rows[correct]["same_rung_gate_passed"]))
            by_seed[str(seed)] = {
                "predecessor_relative_probe_error_L16": predecessor_error,
                "correct_relative_probe_error_L16": correct_error,
                "correct_maximum_cubic_error_L16": cubic,
                "correct_gate_passed": bool(rows[correct]["same_rung_gate_passed"]),
            }
        targets[target] = {
            "predecessor": predecessor,
            "correct": correct,
            "by_seed": by_seed,
            "mean_predecessor_relative_probe_error_L16": sum(predecessor_errors)
            / len(predecessor_errors),
            "mean_correct_relative_probe_error_L16": sum(correct_errors)
            / len(correct_errors),
            "maximum_correct_cubic_error_L16": max(cubic_errors),
            "correct_pass_count": pass_count,
            "passed": bool(
                pass_count == len(expected_seeds)
                and max(correct_errors) <= 5e-3
                and min(predecessor_errors) >= 5e-2
            ),
        }
    return {
        "schema_version": 1,
        "experiment": "exceptional ladder hidden-coordinate learning summary",
        "expected_seeds": list(expected_seeds),
        "targets": targets,
        "passed": all(bool(row["passed"]) for row in targets.values()),
        "claim_boundary": (
            "learned primitive coordinates and held-out composition under oracle event timing; "
            "not autonomous routing or language quality"
        ),
        "input_artifacts": [
            {"path": str(path), "sha256": _sha256(path)} for path in paths
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--expected-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.inputs, tuple(args.expected_seeds))
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
