"""Run the standalone Collatz frontier scanner under a wall-clock budget.

The timed mode records only finite prefix evidence.  A final partial orbit is
never marked as a completed suffix, so the path-merging invariant remains
sound for every alpha value that is actually credited.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from collatz_inverse_frontier import (
    bounded_descent_path_merge,
    hard_records,
    inverse_chain,
    inverse_source,
    odd_orbit,
    scan_alpha_with_stats,
    sha256_json,
)


def build_timed_report(
    *,
    duration_seconds: float,
    target_limit: int,
    source_limit: int,
    max_steps: int,
    max_cached_states: int,
    verification_limit: int,
) -> dict[str, object]:
    alpha, scan_stats = scan_alpha_with_stats(
        target_limit,
        source_limit,
        max_steps=max_steps,
        duration_seconds=duration_seconds,
        max_cached_states=max_cached_states,
    )
    records = hard_records(alpha)
    report: dict[str, object] = {
        "programme": "07-collatz-inverse-frontier-dynamics",
        "status": "timed-bounded-control-only",
        "inputs": {
            "duration_seconds": duration_seconds,
            "target_limit": target_limit,
            "source_limit": source_limit,
            "max_steps": max_steps,
            "max_cached_states": max_cached_states,
            "verification_limit": verification_limit,
        },
        "early_control": {
            "alpha_19": alpha.get(19),
            "orbit_33": odd_orbit(33),
            "inverse_1_4_at_19": str(inverse_source((1, 4), 19)),
            "inverse_1_4_chain_at_19": inverse_chain((1, 4), 19),
        },
        "scan": {
            "algorithm": "ascending-source global path-merge with transition cache",
            "targets_reached": len(alpha),
            "hard_record_count": len(records),
            "hard_records": records,
            "largest_reached_target": max(alpha, default=None),
            "largest_minimal_source": max(alpha.values(), default=None),
            "optimization": scan_stats,
        },
        "bounded_verification": bounded_descent_path_merge(verification_limit),
        "claim_boundary": {
            "establishes": [
                "exact accelerated odd-step arithmetic",
                "exact alpha values for the finite source prefix actually traversed",
                "finite descent/path-merge certificates",
            ],
            "does_not_establish": [
                "classical Collatz convergence for all integers",
                "the universal 32/9 upper bound",
                "E(N)/N -> 32/9",
                "eventual (1,4) dominance",
                "coverage of targets not reached before the wall-clock deadline",
            ],
        },
    }
    report["content_sha256"] = sha256_json(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=float, default=300.0)
    parser.add_argument("--target-limit", type=int, default=10_000_000)
    parser.add_argument("--source-limit", type=int, default=10_000_000_000)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--max-cached-states", type=int, default=1_000_000)
    parser.add_argument("--verification-limit", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_timed_report(
        duration_seconds=args.duration_seconds,
        target_limit=args.target_limit,
        source_limit=args.source_limit,
        max_steps=args.max_steps,
        max_cached_states=args.max_cached_states,
        verification_limit=args.verification_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": report["status"],
                "scan": report["scan"]["optimization"],
                "targets_reached": report["scan"]["targets_reached"],
                "hard_record_count": report["scan"]["hard_record_count"],
                "content_sha256": report["content_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
