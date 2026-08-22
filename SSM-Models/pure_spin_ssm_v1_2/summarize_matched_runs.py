"""Summarize compatible matched natural-data runs across distinct seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

MODEL_NAMES = ("pure_spin_v1_2", "mamba2_fused")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixed_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "seed"}


def _compatibility_key(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_excluding_seed": _fixed_config(report["config"]),
        "dataset": report["dataset"],
        "environment": report["environment"],
        "parameter_match": report["parameter_match"],
        "implementation_sha256": report["implementation_sha256"],
    }


def summarize(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two distinct-seed artifacts are required")
    reports = [json.loads(path.read_text()) for path in paths]
    expected = _compatibility_key(reports[0])
    seeds = [report["config"]["seed"] for report in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError("input artifacts must have distinct seeds")
    for path, report in zip(paths[1:], reports[1:]):
        if _compatibility_key(report) != expected:
            raise ValueError(f"incompatible matched-run artifact: {path}")

    rows: list[dict[str, Any]] = []
    bpb: dict[str, list[float]] = {name: [] for name in MODEL_NAMES}
    memory: dict[str, list[int]] = {name: [] for name in MODEL_NAMES}
    differences: list[float] = []
    for path, report, seed in zip(paths, reports, seeds):
        by_name = {result["name"]: result for result in report["results"]}
        if set(by_name) != set(MODEL_NAMES):
            raise ValueError(f"unexpected model set in {path}: {sorted(by_name)}")
        spin_bpb = by_name[MODEL_NAMES[0]]["final_bits_per_byte"]
        mamba_bpb = by_name[MODEL_NAMES[1]]["final_bits_per_byte"]
        difference = spin_bpb - mamba_bpb
        differences.append(difference)
        for name in MODEL_NAMES:
            bpb[name].append(by_name[name]["final_bits_per_byte"])
            memory[name].append(by_name[name]["peak_cuda_bytes"])
        rows.append(
            {
                "seed": seed,
                "artifact": path.as_posix(),
                "sha256": sha256(path),
                "pure_spin_v1_2_bits_per_byte": spin_bpb,
                "mamba2_fused_bits_per_byte": mamba_bpb,
                "spin_minus_mamba2_bits_per_byte": difference,
                "quality_winner": (
                    MODEL_NAMES[0]
                    if difference < 0.0
                    else MODEL_NAMES[1]
                    if difference > 0.0
                    else "tie"
                ),
            }
        )

    mean_bpb = {name: statistics.fmean(values) for name, values in bpb.items()}
    return {
        "schema_version": 1,
        "analysis_implementation_sha256": sha256(Path(__file__)),
        "claim_scope": (
            "matched multi-seed natural-data quality summary; empirical and "
            "specific to the recorded scale, budget, data, and environment"
        ),
        "compatibility": expected,
        "runs": rows,
        "analysis": {
            "seed_count": len(seeds),
            "seeds": seeds,
            "mean_bits_per_byte": mean_bpb,
            "sample_standard_deviation_bits_per_byte": {
                name: statistics.stdev(values) for name, values in bpb.items()
            },
            "spin_minus_mamba2_bits_per_byte": differences,
            "mean_spin_minus_mamba2_bits_per_byte": statistics.fmean(differences),
            "mamba2_quality_wins": sum(value > 0.0 for value in differences),
            "pure_spin_quality_wins": sum(value < 0.0 for value in differences),
            "peak_cuda_bytes": {
                name: sorted(set(values)) for name, values in memory.items()
            },
            "verdict": (
                "mamba2_wins_quality_at_all_recorded_seeds"
                if all(value > 0.0 for value in differences)
                else "quality_ordering_not_uniform_across_recorded_seeds"
            ),
        },
        "timing_boundary": (
            "sequential per-model training timers are retained in source artifacts "
            "but excluded here; use the order-balanced steady-step benchmark for speed"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.artifacts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["analysis"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
