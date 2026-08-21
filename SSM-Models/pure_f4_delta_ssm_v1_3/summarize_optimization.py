"""Aggregate the frozen paired v1.3 eager-optimization experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(
    paths: list[Path],
    *,
    baseline: str,
    candidate: str,
    expected_seeds: set[int],
    maximum_mean_regression_bpb: float,
    maximum_seed_regression_bpb: float,
) -> dict[str, object]:
    """Validate paired artifacts and apply the preregistered quality gate."""
    if not paths:
        raise ValueError("at least one input artifact is required")

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    reference_dataset = reports[0]["dataset"]
    reference_config = dict(reports[0]["config"])
    reference_config.pop("seed")
    reference_source = reports[0]["source_sha256"]
    paired_rows: list[dict[str, object]] = []
    observed_seeds: set[int] = set()

    for path, report in zip(paths, reports, strict=True):
        config = dict(report["config"])
        seed = config.pop("seed")
        if seed in observed_seeds:
            raise ValueError(f"duplicate seed {seed} in {path}")
        observed_seeds.add(seed)
        if report["dataset"] != reference_dataset:
            raise ValueError(f"dataset mismatch in {path}")
        if config != reference_config:
            raise ValueError(f"configuration mismatch in {path}")
        if report["source_sha256"] != reference_source:
            raise ValueError(f"source mismatch in {path}")
        if report.get("execution") != "eager":
            raise ValueError(f"expected eager execution in {path}")

        rows = {row["name"]: row for row in report["rows"]}
        if set(rows) != {baseline, candidate}:
            raise ValueError(f"expected exactly {baseline} and {candidate} in {path}")
        baseline_row = rows[baseline]
        candidate_row = rows[candidate]
        baseline_bpb = baseline_row["final_bits_per_byte"]
        candidate_bpb = candidate_row["final_bits_per_byte"]
        regression = candidate_bpb - baseline_bpb
        baseline_throughput = baseline_row["training_tokens_per_second"]
        candidate_throughput = candidate_row["training_tokens_per_second"]
        paired_rows.append(
            {
                "seed": seed,
                "baseline_bpb": baseline_bpb,
                "candidate_bpb": candidate_bpb,
                "candidate_minus_baseline_bpb": regression,
                "baseline_training_tokens_per_second": baseline_throughput,
                "candidate_training_tokens_per_second": candidate_throughput,
                "candidate_throughput_speedup": (
                    candidate_throughput / baseline_throughput
                ),
                "baseline_peak_cuda_bytes": baseline_row["peak_cuda_bytes"],
                "candidate_peak_cuda_bytes": candidate_row["peak_cuda_bytes"],
                "input_artifact": path.name,
                "input_sha256": _sha256(path),
            }
        )

    if observed_seeds != expected_seeds:
        raise ValueError(
            "seed set mismatch: "
            f"expected {sorted(expected_seeds)}, observed {sorted(observed_seeds)}"
        )

    paired_rows.sort(key=lambda row: int(row["seed"]))
    regressions = [
        float(row["candidate_minus_baseline_bpb"]) for row in paired_rows
    ]
    mean_regression = statistics.mean(regressions)
    worst_regression = max(regressions)
    mean_passed = mean_regression <= maximum_mean_regression_bpb
    seed_passed = worst_regression <= maximum_seed_regression_bpb
    root = Path(__file__).resolve().parent

    return {
        "schema_version": 1,
        "experiment": "Tiny Shakespeare v1.3 eager-optimization fresh-seed gate",
        "status": "prospective development gate; not a language-model advantage claim",
        "baseline": baseline,
        "candidate": candidate,
        "dataset": reference_dataset,
        "config_without_seed": reference_config,
        "source_sha256": reference_source,
        "aggregator_sha256": _sha256(root / "summarize_optimization.py"),
        "decision_rule": {
            "expected_seeds": sorted(expected_seeds),
            "maximum_mean_regression_bpb": maximum_mean_regression_bpb,
            "maximum_seed_regression_bpb": maximum_seed_regression_bpb,
        },
        "summary": {
            "passed": mean_passed and seed_passed,
            "mean_condition_passed": mean_passed,
            "individual_seed_condition_passed": seed_passed,
            "seeds": len(paired_rows),
            "mean_candidate_minus_baseline_bpb": mean_regression,
            "worst_candidate_minus_baseline_bpb": worst_regression,
            "mean_baseline_bpb": statistics.mean(
                float(row["baseline_bpb"]) for row in paired_rows
            ),
            "mean_candidate_bpb": statistics.mean(
                float(row["candidate_bpb"]) for row in paired_rows
            ),
            "mean_baseline_training_tokens_per_second": statistics.mean(
                float(row["baseline_training_tokens_per_second"])
                for row in paired_rows
            ),
            "mean_candidate_training_tokens_per_second": statistics.mean(
                float(row["candidate_training_tokens_per_second"])
                for row in paired_rows
            ),
            "geometric_mean_candidate_throughput_speedup": statistics.geometric_mean(
                float(row["candidate_throughput_speedup"]) for row in paired_rows
            ),
            "mean_baseline_peak_cuda_bytes": statistics.mean(
                int(row["baseline_peak_cuda_bytes"]) for row in paired_rows
            ),
            "mean_candidate_peak_cuda_bytes": statistics.mean(
                int(row["candidate_peak_cuda_bytes"]) for row in paired_rows
            ),
        },
        "rows": paired_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--baseline", default="identity_legacy")
    parser.add_argument("--candidate", default="identity_delta")
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=range(102, 107))
    parser.add_argument("--maximum-mean-regression-bpb", type=float, default=0.01)
    parser.add_argument("--maximum-seed-regression-bpb", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(
        args.inputs,
        baseline=args.baseline,
        candidate=args.candidate,
        expected_seeds=set(args.expected_seeds),
        maximum_mean_regression_bpb=args.maximum_mean_regression_bpb,
        maximum_seed_regression_bpb=args.maximum_seed_regression_bpb,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
