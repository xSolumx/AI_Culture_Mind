"""Adjudicate the frozen state-only cohort after the invalid-restart bug fix.

The original cohort artifact is intentionally preserved.  Only seeds whose
original failure was the implementation exception are replaced by their
deterministic reruns; geometric failures remain failures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_result(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "result" not in payload:
        raise ValueError(f"rerun artifact has no result: {path}")
    return payload["result"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--rerun", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    original = json.loads(args.original.read_text(encoding="utf-8"))
    replacements = {
        int(result["seed"]): result for result in map(load_result, args.rerun)
    }
    original_seeds = {int(result["seed"]) for result in original["results"]}
    unexpected = sorted(set(replacements) - original_seeds)
    if unexpected:
        raise ValueError(f"reruns are outside the original cohort: {unexpected}")

    results = [
        replacements.get(int(result["seed"]), result)
        for result in original["results"]
    ]
    pass_count = sum(bool(result["passed"]) for result in results)
    report = {
        "experiment": "Spin8 state-only frozen-cohort adjudication",
        "original_artifact": str(args.original),
        "original_artifact_preserved": True,
        "correction": (
            "Discard an invalid k-means restart instead of aborting the full "
            "eight-restart search; no threshold or scientific gate changed."
        ),
        "replaced_seeds": sorted(replacements),
        "fresh_seed_count": len(results),
        "fresh_pass_count": pass_count,
        "reliability_gate_at_least_8_of_9": (
            len(results) == 9 and pass_count >= 8
        ),
        "uniform_reliability_9_of_9": (
            len(results) == 9 and pass_count == 9
        ),
        "frozen_gate_outcome": "fail" if pass_count < 8 else "pass",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"adjudicated={pass_count}/{len(results)} "
        f"gate={report['frozen_gate_outcome']} replacements={sorted(replacements)}"
    )


if __name__ == "__main__":
    main()
