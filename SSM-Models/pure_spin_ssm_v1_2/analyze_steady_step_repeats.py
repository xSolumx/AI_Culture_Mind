"""Combine compatible steady-step artifacts without hiding run-to-run reversals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

MODEL_NAMES = ("pure_spin_v1_2", "mamba2_fused")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric means require positive observations")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _compatibility_key(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": report["config"],
        "environment": report["environment"],
        "parameter_match": report["parameter_match"],
        "implementation_sha256": report["implementation_sha256"],
    }


def analyze(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two repeat artifacts are required")
    reports = [json.loads(path.read_text()) for path in paths]
    expected = _compatibility_key(reports[0])
    for path, report in zip(paths[1:], reports[1:]):
        if _compatibility_key(report) != expected:
            raise ValueError(f"incompatible benchmark artifact: {path}")

    run_rows: list[dict[str, Any]] = []
    pooled: dict[str, list[float]] = {name: [] for name in MODEL_NAMES}
    paired_ratios: list[float] = []
    run_ratios: list[float] = []
    for path, report in zip(paths, reports):
        cycle_values = {
            name: report["aggregate"][name]["cycle_medians_tokens_per_second"]
            for name in MODEL_NAMES
        }
        if len(cycle_values[MODEL_NAMES[0]]) != len(cycle_values[MODEL_NAMES[1]]):
            raise ValueError(f"unpaired cycle counts in {path}")
        spin = report["aggregate"][MODEL_NAMES[0]][
            "median_of_cycle_medians_tokens_per_second"
        ]
        mamba = report["aggregate"][MODEL_NAMES[1]][
            "median_of_cycle_medians_tokens_per_second"
        ]
        ratio = spin / mamba
        run_ratios.append(ratio)
        paired_ratios.extend(
            spin_cycle / mamba_cycle
            for spin_cycle, mamba_cycle in zip(
                cycle_values[MODEL_NAMES[0]], cycle_values[MODEL_NAMES[1]]
            )
        )
        for name in MODEL_NAMES:
            pooled[name].extend(cycle_values[name])
        run_rows.append(
            {
                "artifact": path.as_posix(),
                "sha256": sha256(path),
                "spin_tokens_per_second": spin,
                "mamba2_tokens_per_second": mamba,
                "spin_over_mamba2": ratio,
            }
        )

    ordering_reversed = min(run_ratios) < 1.0 < max(run_ratios)
    pooled_medians = {
        name: statistics.median(values) for name, values in pooled.items()
    }
    return {
        "schema_version": 1,
        "analysis_implementation_sha256": sha256(Path(__file__)),
        "claim_scope": (
            "repeatability analysis of compatible order-balanced fixed-batch "
            "training-step artifacts; not a convergence or superiority claim"
        ),
        "compatibility": expected,
        "runs": run_rows,
        "analysis": {
            "run_count": len(run_rows),
            "cycle_pair_count": len(paired_ratios),
            "ordering_reversed_across_repeats": ordering_reversed,
            "run_spin_over_mamba2": run_ratios,
            "geometric_mean_run_spin_over_mamba2": geometric_mean(run_ratios),
            "paired_cycle_spin_over_mamba2": paired_ratios,
            "median_paired_cycle_spin_over_mamba2": statistics.median(paired_ratios),
            "geometric_mean_paired_cycle_spin_over_mamba2": geometric_mean(
                paired_ratios
            ),
            "pooled_cycle_medians_tokens_per_second": pooled_medians,
            "pooled_median_spin_over_mamba2": (
                pooled_medians[MODEL_NAMES[0]] / pooled_medians[MODEL_NAMES[1]]
            ),
            "verdict": (
                "throughput_ordering_unresolved_at_observed_repeatability"
                if ordering_reversed
                else "ordering_consistent_across_supplied_repeats"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(args.artifacts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["analysis"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
