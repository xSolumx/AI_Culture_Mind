"""Strict, dependency-free validation for local quality benchmark artifacts.

This checks the claims that can be checked from the saved report itself:
schema closure, unique seeds, finite metrics, parameter/config consistency,
the reported bits-per-byte and throughput formulas, and dataset hash shape.
It intentionally does not turn a JSON report into a proof of generalization;
it only prevents malformed or internally inconsistent evidence from being
promoted as a result.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

HEX64 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_TOP = {
    "device",
    "torch_version",
    "config",
    "training_tokens_per_run",
    "integrity",
    "data",
    "results",
}
REQUIRED_RESULT = {
    "name",
    "parameters",
    "initial_loss",
    "final_loss",
    "perplexity",
    "bits_per_byte",
    "final_train_loss",
    "mean_last_20_train_loss",
    "elapsed_seconds",
    "tokens_per_second",
    "peak_cuda_memory_mib",
    "jit_trace",
    "seed",
}


def _finite(value: Any, label: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")


def validate_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    missing = REQUIRED_TOP - report.keys()
    if missing:
        raise ValueError(f"missing top-level fields: {sorted(missing)}")
    config = report["config"]
    for field in ("steps", "batch_size", "seq_len"):
        if not isinstance(config.get(field), int) or config[field] <= 0:
            raise ValueError(f"config.{field} must be a positive integer")
    expected_tokens = config["steps"] * config["batch_size"] * config["seq_len"]
    if report["training_tokens_per_run"] != expected_tokens:
        raise ValueError("training_tokens_per_run disagrees with config")

    integrity = report["integrity"]
    for field in ("model_initialized_after_seed", "python_numpy_torch_cuda_seeded"):
        if integrity.get(field) is not True:
            raise ValueError(f"integrity.{field} must be true")

    data = report["data"]
    for field in ("train_sha256", "validation_sha256"):
        if not isinstance(data.get(field), str) or not HEX64.fullmatch(data[field]):
            raise ValueError(f"data.{field} is not a lowercase SHA-256 digest")
    for field in ("train_bytes", "validation_bytes"):
        if not isinstance(data.get(field), int) or data[field] <= 0:
            raise ValueError(f"data.{field} must be a positive integer")

    rows = report["results"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("results must be a non-empty list")
    seeds: list[int] = []
    parameter_counts: set[int] = set()
    for index, row in enumerate(rows):
        missing = REQUIRED_RESULT - row.keys()
        if missing:
            raise ValueError(f"result {index} missing fields: {sorted(missing)}")
        if not isinstance(row["seed"], int):
            raise ValueError(f"result {index}.seed must be an integer")
        seeds.append(row["seed"])
        if not isinstance(row["parameters"], int) or row["parameters"] <= 0:
            raise ValueError(f"result {index}.parameters must be positive")
        parameter_counts.add(row["parameters"])
        for field in (
            "initial_loss",
            "final_loss",
            "perplexity",
            "bits_per_byte",
            "final_train_loss",
            "mean_last_20_train_loss",
            "elapsed_seconds",
            "tokens_per_second",
            "peak_cuda_memory_mib",
        ):
            _finite(row[field], f"result {index}.{field}")
        if row["elapsed_seconds"] <= 0 or row["tokens_per_second"] <= 0:
            raise ValueError(f"result {index} has non-positive timing")
        expected_bpb = row["final_loss"] / math.log(2.0)
        if not math.isclose(
            row["bits_per_byte"], expected_bpb, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError(f"result {index}.bits_per_byte disagrees with final_loss")
        expected_tps = expected_tokens / row["elapsed_seconds"]
        if not math.isclose(
            row["tokens_per_second"], expected_tps, rel_tol=1e-9, abs_tol=1e-6
        ):
            raise ValueError(f"result {index}.tokens_per_second disagrees with timing")
        expected_ppl = math.exp(row["final_loss"])
        if not math.isclose(
            row["perplexity"], expected_ppl, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError(f"result {index}.perplexity disagrees with final_loss")

    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seeds make the aggregate unreliable")
    mean_bpb = sum(row["bits_per_byte"] for row in rows) / len(rows)
    mean_tps = sum(row["tokens_per_second"] for row in rows) / len(rows)
    return {
        "path": str(path),
        "rows": len(rows),
        "seeds": seeds,
        "parameter_counts": sorted(parameter_counts),
        "mean_bits_per_byte": mean_bpb,
        "mean_tokens_per_second": mean_tps,
        "dataset_sha256": {
            "train": data["train_sha256"],
            "validation": data["validation_sha256"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    summary = validate_report(args.report)
    print(json.dumps({"passed": True, **summary}, indent=2))


if __name__ == "__main__":
    main()
