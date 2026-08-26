"""Fail-closed summarizer for the v1.3 matched SM75 quality cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_VARIANTS = (
    "identity_matched",
    "f4_matched",
    "e6_safe",
    "mamba2_official",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(paths: list[Path], expected_seeds: tuple[int, ...]) -> dict[str, object]:
    if len(paths) != len(expected_seeds):
        raise ValueError("one artifact is required for every expected seed")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    by_seed: dict[int, tuple[Path, dict[str, object]]] = {}
    reference_config = None
    reference_dataset = None
    reference_sources = None
    for path, report in zip(paths, reports, strict=True):
        if int(report.get("schema_version", 0)) < 2:
            raise ValueError(f"legacy or unknown schema in {path}")
        config = dict(report["config"])
        seed = int(config.pop("seed"))
        if seed in by_seed:
            raise ValueError(f"duplicate seed {seed}")
        if seed not in expected_seeds:
            raise ValueError(f"unexpected seed {seed}")
        variants = tuple(report["variants"])
        if variants != EXPECTED_VARIANTS:
            raise ValueError(f"unexpected variants in seed {seed}: {variants}")
        if report.get("execution") != "eager":
            raise ValueError("quality cohort requires eager semantic execution")
        environment = report["environment"]
        if environment.get("compute_capability") != [7, 5]:
            raise ValueError("quality cohort requires exact SM75")
        rows = report["rows"]
        if len(rows) != len(EXPECTED_VARIANTS):
            raise ValueError(f"seed {seed} has missing or duplicate rows")
        if tuple(row["name"] for row in rows) != EXPECTED_VARIANTS:
            raise ValueError(f"seed {seed} row ordering or identity changed")
        if any(not math.isfinite(float(row["final_bits_per_byte"])) for row in rows):
            raise ValueError(f"seed {seed} contains nonfinite quality")
        target_digests = {row["training_target_sha256"] for row in rows}
        if len(target_digests) != 1:
            raise ValueError(f"seed {seed} arms saw different target streams")
        if reference_config is None:
            reference_config = config
            reference_dataset = report["dataset"]
            reference_sources = report["source_sha256"]
        elif (
            config != reference_config
            or report["dataset"] != reference_dataset
            or report["source_sha256"] != reference_sources
        ):
            raise ValueError("configuration, data, or source drift across seeds")
        by_seed[seed] = (path, report)
    if set(by_seed) != set(expected_seeds):
        raise ValueError("expected seed set is incomplete")

    arms: dict[str, dict[str, object]] = {}
    parameter_target = None
    for name in EXPECTED_VARIANTS:
        rows = [
            next(row for row in by_seed[seed][1]["rows"] if row["name"] == name)
            for seed in expected_seeds
        ]
        parameters = {int(row["parameters"]) for row in rows}
        if len(parameters) != 1:
            raise ValueError(f"parameter count drift for {name}")
        quality = [float(row["final_bits_per_byte"]) for row in rows]
        throughput = [float(row["training_tokens_per_second"]) for row in rows]
        peak = [int(row["peak_cuda_bytes"]) for row in rows]
        arms[name] = {
            "parameters": next(iter(parameters)),
            "bits_per_byte_by_seed": dict(zip(map(str, expected_seeds), quality, strict=True)),
            "mean_bits_per_byte": sum(quality) / len(quality),
            "training_tokens_per_second_by_seed": dict(
                zip(map(str, expected_seeds), throughput, strict=True)
            ),
            "geometric_mean_training_tokens_per_second": math.exp(
                sum(math.log(value) for value in throughput) / len(throughput)
            ),
            "maximum_peak_cuda_bytes": max(peak),
        }
        if name == "mamba2_official":
            parameter_target = next(iter(parameters))
    if parameter_target is None:
        raise AssertionError("Mamba-2 parameter target missing")
    mamba_mean = float(arms["mamba2_official"]["mean_bits_per_byte"])
    for name, row in arms.items():
        row["parameter_residual_fraction_vs_mamba2"] = (
            int(row["parameters"]) - parameter_target
        ) / parameter_target
        row["mean_bpb_minus_mamba2"] = float(row["mean_bits_per_byte"]) - mamba_mean
        if name != "mamba2_official":
            candidate = row["bits_per_byte_by_seed"]
            baseline = arms["mamba2_official"]["bits_per_byte_by_seed"]
            row["seed_wins_vs_mamba2"] = sum(
                float(candidate[str(seed)]) < float(baseline[str(seed)])
                for seed in expected_seeds
            )
    best_exceptional = min(
        ("identity_matched", "f4_matched", "e6_safe"),
        key=lambda name: float(arms[name]["mean_bits_per_byte"]),
    )
    promoted = bool(
        float(arms[best_exceptional]["mean_bpb_minus_mamba2"]) <= -0.01
        and int(arms[best_exceptional]["seed_wins_vs_mamba2"])
        == len(expected_seeds)
        and all(
            abs(float(row["parameter_residual_fraction_vs_mamba2"])) <= 0.01
            for row in arms.values()
        )
    )
    return {
        "schema_version": 1,
        "experiment": "Pure Exceptional Delta SSM v1.3.1 matched SM75 quality summary",
        "expected_seeds": list(expected_seeds),
        "expected_variants": list(EXPECTED_VARIANTS),
        "arms": arms,
        "best_non_mamba_arm": best_exceptional,
        "promotion_gate": {
            "required_mean_bpb_advantage": 0.01,
            "required_seed_wins": len(expected_seeds),
            "maximum_parameter_residual_fraction": 0.01,
            "passed": promoted,
        },
        "input_artifacts": [
            {"path": str(path), "sha256": _sha256(path)} for path in paths
        ],
        "status": (
            "promoted matched quality result"
            if promoted
            else "completed matched cohort; no promotion"
        ),
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
