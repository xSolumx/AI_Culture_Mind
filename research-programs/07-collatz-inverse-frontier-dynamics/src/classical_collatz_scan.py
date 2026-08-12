"""Bounded forward verification controls for the classical Collatz map.

This script performs finite verification only: it confirms that each start in
``[1, limit]`` reaches 1 within a step budget and records a path-merge
certificate by memoizing known trajectories.  It is not a proof of the global
conjecture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def collatz_step(value: int) -> int:
    """Apply one classical Collatz step."""

    if value <= 0:
        raise ValueError("collatz_step requires positive integers")
    return value // 2 if value % 2 == 0 else 3 * value + 1


def bounded_forward_collatz_scan(
    limit: int, max_steps: int
) -> tuple[dict[int, int], list[dict[str, int | str]], dict[str, int]]:
    """Scan starts from 1..limit with a finite step budget and memoized certificate.

    Returns:
    - ``known_steps``: map from n to verified steps-to-1 (for all verified starts)
    - ``counter_examples``: bounded failures (if a start exceeds ``max_steps`` or loops)
    - ``stats``: raw counters and cache behavior for reproducibility
    """

    if limit < 1:
        raise ValueError("limit must be positive")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    known_steps: dict[int, int] = {1: 0}
    counter_examples: list[dict[str, int]] = []
    step_cache: dict[int, int] = {}
    stats = {
        "starts_scanned": 0,
        "transition_evaluations": 0,
        "transition_cache_hits": 0,
        "path_merges": 0,
        "cycle_terminations": 0,
        "max_steps_exhaustions": 0,
    }

    max_steps_observed = 0
    max_peak_observed = 1
    max_peak_start = 1

    for start in range(1, limit + 1):
        stats["starts_scanned"] += 1
        current = start
        path: list[int] = []
        seen_in_path: dict[int, int] = {}
        local_steps = 0
        exceeded = False
        terminated_by_cycle = False
        while current not in known_steps and current not in seen_in_path:
            if local_steps >= max_steps:
                exceeded = True
                break
            seen_in_path[current] = local_steps
            path.append(current)
            cached_next = step_cache.get(current)
            if cached_next is None:
                cached_next = collatz_step(current)
                step_cache[current] = cached_next
                stats["transition_evaluations"] += 1
            else:
                stats["transition_cache_hits"] += 1
            current = cached_next
            local_steps += 1
            if current > max_peak_observed:
                max_peak_observed = current
                max_peak_start = start

        if current in seen_in_path:
            terminated_by_cycle = True
            cycle_start_step = seen_in_path[current]
            cycle_start_value = path[cycle_start_step]
            stats["cycle_terminations"] += 1
            counter_examples.append(
                {
                    "start": start,
                    "reason": "cycle_detected",
                    "last_value": current,
                    "observed_steps": local_steps,
                    "cycle_start_value": cycle_start_value,
                }
            )
            continue

        if exceeded:
            stats["max_steps_exhaustions"] += 1
            counter_examples.append(
                {
                    "start": start,
                    "reason": "max_steps_exceeded",
                    "last_value": current,
                    "observed_steps": local_steps,
                    "cycle_start_value": current,
                }
            )
            continue

        base_steps = known_steps[current]
        stats["path_merges"] += 1
        for i, value in enumerate(path):
            known_steps[value] = (len(path) - i) + base_steps
            if i == 0 and (len(path) + base_steps) > max_steps_observed:
                max_steps_observed = len(path) + base_steps

    return known_steps, counter_examples, stats | {
        "max_known_steps_observed": max_steps_observed,
        "max_peak_observed": max_peak_observed,
        "max_peak_start": max_peak_start,
    }


def sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_report(limit: int, max_steps: int) -> dict[str, object]:
    known_steps, counter_examples, stats = bounded_forward_collatz_scan(limit, max_steps)
    all_verified = len(counter_examples) == 0
    bounded_set = all(v in known_steps for v in range(1, limit + 1))

    # ``known_steps`` is authoritative for every verified number in [1, limit].
    max_verified_start = max(known_steps) if known_steps else 0
    max_verified_steps = max(known_steps.values()) if known_steps else 0

    report: dict[str, object] = {
        "programme": "07-collatz-inverse-frontier-dynamics",
        "status": "bounded-classical-forward-verification-only",
        "inputs": {"limit": limit, "max_steps": max_steps},
        "early_control": {
            "known_steps_1": known_steps.get(1),
            "known_steps_3": known_steps.get(3),
            "known_steps_7": known_steps.get(7),
        },
        "scan": {
            "algorithm": "forward finite scan with local-path and global memoization",
            "starts_verified": len(known_steps),
            "starts_verified_in_range": sum(
                1 for value in range(1, limit + 1) if value in known_steps
            ),
            "max_start_verified_steps": max_verified_steps,
            "max_start_verified": max_verified_start,
            "counter_example_count": len(counter_examples),
            "hardest_verified_start": max((v for v in known_steps if v <= limit), default=0),
            "stats": stats,
        },
        "counter_examples": counter_examples,
        "bounded_verification": {
            "limit": limit,
            "all_verified": all_verified and bounded_set,
            "verified_count": len(known_steps),
            "counter_example_count": len(counter_examples),
            "bounded_by_steps": max_steps,
            "hardest_verified": max((v for v, steps in known_steps.items() if v <= limit), default=0),
            "hardest_verified_steps": max((steps for v, steps in known_steps.items() if v <= limit), default=0),
        },
        "claim_boundary": {
            "establishes": [
                "exact Collatz arithmetic for bounded forward iterations",
                "deterministic finite checks for all starts in a bounded range",
                "path-merge certificates from completed suffixes",
            ],
            "does_not_establish": [
                "global termination of all orbits",
                "unbounded conjectural constants",
                "any asymptotic claim about trajectory extrema",
            ],
        },
    }
    report["content_sha256"] = sha256_json(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.limit, args.max_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": report["status"],
                "all_verified": report["bounded_verification"]["all_verified"],
                "starts_verified_in_range": report["scan"]["starts_verified_in_range"],
                "counter_examples": report["scan"]["counter_example_count"],
                "content_sha256": report["content_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
