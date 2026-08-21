"""Aggregate a frozen paired layer-localization experiment."""

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
    required_wins: int,
    required_mean_improvement: float,
) -> dict[str, object]:
    if not paths:
        raise ValueError("at least one input artifact is required")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    reference_dataset = reports[0]["dataset"]
    reference_config = dict(reports[0]["config"])
    reference_config.pop("seed")
    reference_source = reports[0]["source_sha256"]
    paired_rows = []
    for path, report in zip(paths, reports, strict=True):
        config = dict(report["config"])
        seed = config.pop("seed")
        if report["dataset"] != reference_dataset:
            raise ValueError(f"dataset mismatch in {path}")
        if config != reference_config:
            raise ValueError(f"configuration mismatch in {path}")
        if report["source_sha256"] != reference_source:
            raise ValueError(f"source mismatch in {path}")
        rows = {row["name"]: row for row in report["rows"]}
        if set(rows) != {baseline, candidate}:
            raise ValueError(f"expected exactly {baseline} and {candidate} in {path}")
        baseline_bpb = rows[baseline]["final_bits_per_byte"]
        candidate_bpb = rows[candidate]["final_bits_per_byte"]
        paired_rows.append(
            {
                "seed": seed,
                "baseline_bpb": baseline_bpb,
                "candidate_bpb": candidate_bpb,
                "candidate_improvement_bpb": baseline_bpb - candidate_bpb,
                "baseline_training_tokens_per_second": rows[baseline][
                    "training_tokens_per_second"
                ],
                "candidate_training_tokens_per_second": rows[candidate][
                    "training_tokens_per_second"
                ],
                "input_artifact": path.name,
                "input_sha256": _sha256(path),
            }
        )
    paired_rows.sort(key=lambda row: row["seed"])
    improvements = [row["candidate_improvement_bpb"] for row in paired_rows]
    wins = sum(value > 0 for value in improvements)
    mean_improvement = statistics.mean(improvements)
    passed = wins >= required_wins and mean_improvement >= required_mean_improvement
    root = Path(__file__).resolve().parent
    return {
        "schema_version": 1,
        "experiment": "Tiny Shakespeare E6 layer-localization fresh-seed gate",
        "status": "prospective development gate; not a promoted model claim",
        "baseline": baseline,
        "candidate": candidate,
        "dataset": reference_dataset,
        "config_without_seed": reference_config,
        "source_sha256": reference_source,
        "aggregator_sha256": _sha256(root / "summarize_localization.py"),
        "decision_rule": {
            "required_wins": required_wins,
            "required_mean_improvement_bpb": required_mean_improvement,
        },
        "summary": {
            "passed": passed,
            "wins": wins,
            "seeds": len(paired_rows),
            "mean_candidate_improvement_bpb": mean_improvement,
            "mean_baseline_bpb": statistics.mean(
                row["baseline_bpb"] for row in paired_rows
            ),
            "mean_candidate_bpb": statistics.mean(
                row["candidate_bpb"] for row in paired_rows
            ),
            "mean_baseline_training_tokens_per_second": statistics.mean(
                row["baseline_training_tokens_per_second"] for row in paired_rows
            ),
            "mean_candidate_training_tokens_per_second": statistics.mean(
                row["candidate_training_tokens_per_second"] for row in paired_rows
            ),
        },
        "rows": paired_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--baseline", default="identity_delta")
    parser.add_argument("--candidate", default="early_e6_delta")
    parser.add_argument("--required-wins", type=int, default=4)
    parser.add_argument("--required-mean-improvement", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(
        args.inputs,
        baseline=args.baseline,
        candidate=args.candidate,
        required_wins=args.required_wins,
        required_mean_improvement=args.required_mean_improvement,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
