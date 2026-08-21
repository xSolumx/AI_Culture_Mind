"""Aggregate compatible multi-seed natural-data benchmark artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_standard_deviation": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.inputs) < 2:
        raise ValueError("at least two seed artifacts are required")
    reports = [json.loads(path.read_text()) for path in args.inputs]
    reference = reports[0]
    compatibility_keys = ("dataset", "implementation_sha256", "parameter_match")
    for report in reports[1:]:
        for key in compatibility_keys:
            if report[key] != reference[key]:
                raise ValueError(f"incompatible {key} across seed artifacts")
        reference_config = {k: v for k, v in reference["config"].items() if k != "seed"}
        report_config = {k: v for k, v in report["config"].items() if k != "seed"}
        if report_config != reference_config:
            raise ValueError("incompatible benchmark configuration across seeds")

    by_seed = []
    for report in reports:
        results = {result["name"]: result for result in report["results"]}
        spin = results["pure_spin_v1_2"]
        mamba = results["mamba2_fused"]
        by_seed.append(
            {
                "seed": report["config"]["seed"],
                "pure_spin_bits_per_byte": spin["final_bits_per_byte"],
                "mamba2_bits_per_byte": mamba["final_bits_per_byte"],
                "mamba2_minus_spin_bits_per_byte": (
                    mamba["final_bits_per_byte"] - spin["final_bits_per_byte"]
                ),
                "pure_spin_tokens_per_second": spin["training_tokens_per_second"],
                "mamba2_tokens_per_second": mamba["training_tokens_per_second"],
                "mamba2_throughput_over_spin": (
                    mamba["training_tokens_per_second"]
                    / spin["training_tokens_per_second"]
                ),
            }
        )
    report = {
        "schema_version": 1,
        "claim_scope": "three-seed matched WikiText-2 byte-LM comparison",
        "seeds": [item["seed"] for item in by_seed],
        "dataset": reference["dataset"],
        "config_except_seed": {
            k: v for k, v in reference["config"].items() if k != "seed"
        },
        "parameter_match": reference["parameter_match"],
        "environment": reference["environment"],
        "implementation_sha256": reference["implementation_sha256"],
        "input_artifacts": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in args.inputs
        },
        "by_seed": by_seed,
        "aggregate": {
            key: summarize([float(item[key]) for item in by_seed])
            for key in (
                "pure_spin_bits_per_byte",
                "mamba2_bits_per_byte",
                "mamba2_minus_spin_bits_per_byte",
                "pure_spin_tokens_per_second",
                "mamba2_tokens_per_second",
                "mamba2_throughput_over_spin",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
