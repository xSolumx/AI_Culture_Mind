"""Fail-closed summary for the v1.3.2 sparse-transport text cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_VARIANTS = (
    "e6_primitive_dead",
    "e6_primitive_event",
    "e6_safe",
    "mamba2_official",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(
    paths: list[Path],
    expected_seeds: tuple[int, ...],
    cost_artifact: Path,
) -> dict[str, object]:
    if len(paths) != len(expected_seeds) or len(expected_seeds) < 3:
        raise ValueError("one quality artifact and at least three seeds are required")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    by_seed: dict[int, tuple[Path, dict[str, object]]] = {}
    reference_config = None
    reference_dataset = None
    reference_sources = None
    for path, report in zip(paths, reports, strict=True):
        if int(report.get("schema_version", 0)) < 2:
            raise ValueError(f"legacy or unknown quality schema in {path}")
        config = dict(report["config"])
        seed = int(config.pop("seed"))
        if seed in by_seed or seed not in expected_seeds:
            raise ValueError(f"duplicate or unexpected seed {seed}")
        if tuple(report["variants"]) != EXPECTED_VARIANTS:
            raise ValueError(f"unexpected variant cohort at seed {seed}")
        if report.get("execution") != "eager":
            raise ValueError("quality cohort requires eager semantic execution")
        if report["environment"].get("compute_capability") != [7, 5]:
            raise ValueError("quality cohort requires exact SM75")
        rows = report["rows"]
        if tuple(row["name"] for row in rows) != EXPECTED_VARIANTS:
            raise ValueError(f"missing, duplicated, or reordered row at seed {seed}")
        if any(not math.isfinite(float(row["final_bits_per_byte"])) for row in rows):
            raise ValueError(f"seed {seed} contains nonfinite quality")
        if len({row["training_target_sha256"] for row in rows}) != 1:
            raise ValueError(f"seed {seed} arms saw different target streams")
        if any(not row.get("checkpoint") or not row.get("checkpoint_sha256") for row in rows):
            raise ValueError(f"seed {seed} is missing a bound checkpoint")
        for row in rows:
            checkpoint = Path(str(row["checkpoint"]))
            if (
                not checkpoint.is_file()
                or _sha256(checkpoint) != row["checkpoint_sha256"]
            ):
                raise ValueError(
                    f"seed {seed} checkpoint is missing or has hash drift: {checkpoint}"
                )
        if reference_config is None:
            reference_config = config
            reference_dataset = report["dataset"]
            reference_sources = report["source_sha256"]
        elif (
            config != reference_config
            or report["dataset"] != reference_dataset
            or report["source_sha256"] != reference_sources
        ):
            raise ValueError("configuration, data, or source drift across quality seeds")
        by_seed[seed] = (path, report)
    if set(by_seed) != set(expected_seeds):
        raise ValueError("quality seed set is incomplete")

    arms: dict[str, dict[str, object]] = {}
    for name in EXPECTED_VARIANTS:
        rows = [
            next(row for row in by_seed[seed][1]["rows"] if row["name"] == name)
            for seed in expected_seeds
        ]
        parameter_counts = {int(row["parameters"]) for row in rows}
        if len(parameter_counts) != 1:
            raise ValueError(f"parameter drift for {name}")
        bpb = [float(row["final_bits_per_byte"]) for row in rows]
        throughput = [float(row["training_tokens_per_second"]) for row in rows]
        arms[name] = {
            "parameters": next(iter(parameter_counts)),
            "bits_per_byte_by_seed": dict(
                zip(map(str, expected_seeds), bpb, strict=True)
            ),
            "mean_bits_per_byte": sum(bpb) / len(bpb),
            "geometric_mean_training_tokens_per_second": math.exp(
                sum(math.log(value) for value in throughput) / len(throughput)
            ),
            "maximum_peak_cuda_bytes": max(
                int(row["peak_cuda_bytes"]) for row in rows
            ),
            "checkpoints": [
                {
                    "path": row["checkpoint"],
                    "sha256": row["checkpoint_sha256"],
                }
                for row in rows
            ],
        }

    candidate = arms["e6_primitive_event"]
    dead = arms["e6_primitive_dead"]
    mamba = arms["mamba2_official"]
    candidate_seed = candidate["bits_per_byte_by_seed"]
    dead_seed = dead["bits_per_byte_by_seed"]
    mamba_seed = mamba["bits_per_byte_by_seed"]
    wins_vs_dead = sum(
        float(candidate_seed[str(seed)]) < float(dead_seed[str(seed)])
        for seed in expected_seeds
    )
    wins_vs_mamba = sum(
        float(candidate_seed[str(seed)]) < float(mamba_seed[str(seed)])
        for seed in expected_seeds
    )
    mean_minus_dead = float(candidate["mean_bits_per_byte"]) - float(
        dead["mean_bits_per_byte"]
    )
    mean_minus_mamba = float(candidate["mean_bits_per_byte"]) - float(
        mamba["mean_bits_per_byte"]
    )
    parameter_residual = (
        int(candidate["parameters"]) - int(mamba["parameters"])
    ) / int(mamba["parameters"])

    cost = json.loads(cost_artifact.read_text(encoding="utf-8"))
    if cost.get("environment", {}).get("compute_capability") != [7, 5]:
        raise ValueError("cost artifact is not exact SM75")
    systems = cost.get("verdict", {})
    cheap_systems = bool(systems.get("cheap_action_path_pass", False))
    mamba_systems = bool(systems.get("mamba_competitive_pass", False))
    identity_quality = wins_vs_dead >= 2 and mean_minus_dead <= 0.02
    mamba_quality = (
        wins_vs_mamba == len(expected_seeds) and mean_minus_mamba <= -0.01
    )
    return {
        "schema_version": 1,
        "experiment": "Pure Exceptional Delta SSM v1.3.2 sparse SM75 quality summary",
        "expected_seeds": list(expected_seeds),
        "expected_variants": list(EXPECTED_VARIANTS),
        "arms": arms,
        "comparisons": {
            "candidate_seed_wins_vs_dead_budget": wins_vs_dead,
            "candidate_mean_bpb_minus_dead_budget": mean_minus_dead,
            "candidate_seed_wins_vs_mamba2": wins_vs_mamba,
            "candidate_mean_bpb_minus_mamba2": mean_minus_mamba,
            "candidate_parameter_residual_fraction_vs_mamba2": parameter_residual,
        },
        "gates": {
            "quality_noninferior_to_dead_budget": identity_quality,
            "quality_beats_mamba2": mamba_quality,
            "cheap_action_systems": cheap_systems,
            "mamba_competitive_systems": mamba_systems,
            "cheap_exceptional_transport_promoted": identity_quality and cheap_systems,
            "complete_model_promoted_over_mamba2": (
                mamba_quality
                and mamba_systems
                and abs(parameter_residual) <= 0.01
            ),
        },
        "input_artifacts": [
            {"path": str(path), "sha256": _sha256(path)} for path in paths
        ],
        "cost_artifact": {
            "path": str(cost_artifact),
            "sha256": _sha256(cost_artifact),
        },
        "status": "completed fail-closed sparse-transport cohort",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--expected-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--cost-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(
        args.inputs, tuple(args.expected_seeds), args.cost_artifact
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
