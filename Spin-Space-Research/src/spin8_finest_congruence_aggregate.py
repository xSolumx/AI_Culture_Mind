"""Aggregate independently executed finest-congruence compiler reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for path in args.reports:
        payload = json.loads(path.read_text(encoding="utf-8"))
        results.append(payload["result"])
    results.sort(key=lambda result: int(result["seed"]))
    seeds = [int(result["seed"]) for result in results]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seed reports")
    passed = sum(bool(result["passed"]) for result in results)
    report = {
        "experiment": "Spin8 state-only finest-congruence fresh cohort",
        "development_seeds_38_43_46_excluded": True,
        "prospective_smoke_seed_48_excluded": True,
        "fresh_seed_count": len(results),
        "fresh_pass_count": passed,
        "reliability_gate_at_least_8_of_9": len(results) == 9 and passed >= 8,
        "uniform_reliability_9_of_9": len(results) == 9 and passed == 9,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"passes={passed}/{len(results)} seeds={seeds}")


if __name__ == "__main__":
    main()
