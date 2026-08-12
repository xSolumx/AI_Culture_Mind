"""Exact bounded controls for the inverse Collatz frontier programme.

This module deliberately separates finite computation from the asymptotic
claims in the programme README.  It implements the accelerated odd map, exact
inverse valuation words, an ascending-source alpha scan, and bounded descent /
path-merge certificates inspired by the 2026 verification-algorithm preprint.
Nothing here proves the classical Collatz conjecture or the 32/9 frontier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Iterable
from fractions import Fraction
from pathlib import Path


def v2(value: int) -> int:
    """Return the exact 2-adic valuation of a positive integer."""

    if value <= 0:
        raise ValueError("v2 requires a positive integer")
    return (value & -value).bit_length() - 1


def accelerated_odd_step(value: int) -> tuple[int, int]:
    """Return the next odd Collatz value and its exact valuation."""

    if value <= 0 or value % 2 == 0:
        raise ValueError("accelerated_odd_step requires a positive odd value")
    numerator = 3 * value + 1
    valuation = v2(numerator)
    return numerator >> valuation, valuation


def odd_orbit(source: int, *, max_steps: int = 100_000) -> list[tuple[int, int | None]]:
    """Return ``(odd_value, valuation_used_to_leave_it)`` pairs."""

    if source <= 0 or source % 2 == 0:
        raise ValueError("source must be positive and odd")
    values: list[tuple[int, int | None]] = []
    current = source
    seen: set[int] = set()
    for _ in range(max_steps):
        if current in seen:
            values.append((current, None))
            return values
        seen.add(current)
        if current == 1:
            values.append((current, None))
            return values
        nxt, valuation = accelerated_odd_step(current)
        values.append((current, valuation))
        current = nxt
    raise RuntimeError(f"odd orbit exceeded max_steps={max_steps} from {source}")


def inverse_step(target: Fraction | int, valuation: int) -> Fraction:
    """Apply one exact inverse odd step with a prescribed valuation."""

    if valuation <= 0:
        raise ValueError("valuation must be positive")
    return (2**valuation * Fraction(target) - 1) / 3


def inverse_source(word: Iterable[int], target: int) -> Fraction:
    """Compute the exact source for a forward valuation word.

    ``word=(a1,...,ar)`` means source -> ... -> target.  Inversion therefore
    applies the valuations in reverse order.
    """

    value = Fraction(target)
    for valuation in reversed(tuple(word)):
        value = inverse_step(value, valuation)
    return value


def inverse_chain(word: Iterable[int], target: int) -> tuple[int, ...] | None:
    """Return the integer odd chain or ``None`` when a word is inadmissible."""

    values = [Fraction(target)]
    for valuation in reversed(tuple(word)):
        values.append(inverse_step(values[-1], valuation))
    if any(value.denominator != 1 for value in values):
        return None
    integers = tuple(int(value) for value in reversed(values))
    if any(value <= 0 or value % 2 == 0 for value in integers):
        return None
    observed = [valuation for _, valuation in odd_orbit(integers[0])]
    expected = list(word)
    if observed[: len(expected)] != expected:
        return None
    if integers[len(expected)] != target:
        return None
    return integers


def scan_alpha(
    target_limit: int,
    source_limit: int,
    *,
    max_steps: int = 100_000,
) -> dict[int, int]:
    """Scan odd multiples of three in ascending order for finite alpha values.

    For every target found, the first source is globally minimal because all
    smaller odd multiples of three have already been scanned.  Targets not
    reached before ``source_limit`` are intentionally absent, rather than
    being treated as disproved.  The production path uses global path merging:
    once a smaller source has completely traversed a state, later sources stop
    at that state because its entire deterministic suffix has already been
    credited to an even smaller source.
    """

    alpha, _ = scan_alpha_with_stats(
        target_limit, source_limit, max_steps=max_steps
    )
    return alpha


def scan_alpha_reference(
    target_limit: int,
    source_limit: int,
    *,
    max_steps: int = 100_000,
) -> dict[int, int]:
    """Reference implementation that intentionally recomputes each orbit."""

    if target_limit < 1 or source_limit < 3:
        raise ValueError("limits are too small")
    alpha: dict[int, int] = {}
    for source in range(3, source_limit + 1, 6):
        for value, _ in odd_orbit(source, max_steps=max_steps):
            if value <= target_limit and value % 2:
                alpha.setdefault(value, source)
    return alpha


def scan_alpha_with_stats(
    target_limit: int,
    source_limit: int,
    *,
    max_steps: int = 100_000,
    duration_seconds: float | None = None,
    max_cached_states: int | None = None,
) -> tuple[dict[int, int], dict[str, int | bool]]:
    """Run the memoized ascending-source scan and return diagnostics.

    A deterministic orbit is a directed graph with out-degree one.  When an
    ascending source reaches a state whose complete suffix was already
    traversed by a smaller source, no target on that suffix can receive a
    smaller first source from the current orbit.  The global ``completed`` set
    therefore gives exact path merging, not a heuristic cutoff.  States from a
    path that exhausts ``max_steps`` are deliberately *not* marked complete.
    """

    if target_limit < 1 or source_limit < 3:
        raise ValueError("limits are too small")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if max_cached_states is not None and max_cached_states < 1:
        raise ValueError("max_cached_states must be positive")

    alpha: dict[int, int] = {}
    completed: set[int] = {1}
    next_cache: dict[int, int] = {}
    odd_target_count = (target_limit + 1) // 2
    merge_enabled = True
    deadline = (
        None
        if duration_seconds is None
        else time.monotonic() + duration_seconds
    )
    stats: dict[str, int | bool] = {
        "sources_scanned": 0,
        "transition_evaluations": 0,
        "transition_cache_hits": 0,
        "global_path_merges": 0,
        "cycle_terminations": 0,
        "step_budget_exhaustions": 0,
        "completed_path_states": 0,
        "stopped_when_complete": False,
        "stopped_by_deadline": False,
        "path_merge_disabled_by_cache_cap": False,
    }

    for source in range(3, source_limit + 1, 6):
        if deadline is not None and time.monotonic() >= deadline:
            stats["stopped_by_deadline"] = True
            break
        stats["sources_scanned"] = int(stats["sources_scanned"]) + 1
        current = source
        path: list[int] = []
        local_seen: set[int] = set()
        timed_out = False
        while (
            (not merge_enabled or current not in completed)
            and current not in local_seen
            and len(path) < max_steps
        ):
            if (
                deadline is not None
                and len(path) % 1024 == 0
                and time.monotonic() >= deadline
            ):
                timed_out = True
                break
            local_seen.add(current)
            path.append(current)
            cached_next = next_cache.get(current)
            if cached_next is None:
                cached_next, _ = accelerated_odd_step(current)
                next_cache[current] = cached_next
                stats["transition_evaluations"] = (
                    int(stats["transition_evaluations"]) + 1
                )
            else:
                stats["transition_cache_hits"] = (
                    int(stats["transition_cache_hits"]) + 1
                )
            current = cached_next

        if timed_out:
            # Values observed before the deadline are valid finite hits, but
            # this partial path cannot certify its suffix for later sources.
            for value in path:
                if value <= target_limit:
                    alpha.setdefault(value, source)
            stats["stopped_by_deadline"] = True
            break

        joined_completed = merge_enabled and current in completed
        closed_cycle = current in local_seen
        complete = joined_completed or closed_cycle
        if complete:
            if merge_enabled:
                completed.update(path)
            stats["completed_path_states"] = (
                int(stats["completed_path_states"]) + len(path)
            )
            if joined_completed:
                stats["global_path_merges"] = (
                    int(stats["global_path_merges"]) + 1
                )
            else:
                stats["cycle_terminations"] = (
                    int(stats["cycle_terminations"]) + 1
                )
        else:
            stats["step_budget_exhaustions"] = (
                int(stats["step_budget_exhaustions"]) + 1
            )
            raise RuntimeError(
                f"odd orbit exceeded max_steps={max_steps} from {source}"
            )

        credited_values = path
        if joined_completed:
            # The terminal/cached state is not part of ``path``.  It may be
            # the first bounded target (notably 1), so credit it explicitly;
            # all other cached suffix targets were already credited by the
            # smaller source that completed that suffix.
            credited_values = [*path, current]
        for value in credited_values:
            if value <= target_limit:
                alpha.setdefault(value, source)
        if len(alpha) == odd_target_count:
            stats["stopped_when_complete"] = True
            break

        if (
            merge_enabled
            and max_cached_states is not None
            and len(completed) >= max_cached_states
        ):
            # Alpha minimality depends only on ascending source order.  The
            # completed-state set is an acceleration structure, not evidence
            # needed to preserve already-recorded minima.  Drop it at the
            # configured cap and continue with local cycle detection so a long
            # timed run has bounded cache memory.
            merge_enabled = False
            completed.clear()
            next_cache.clear()
            stats["path_merge_disabled_by_cache_cap"] = True

    stats["cached_transitions"] = len(next_cache)
    stats["completed_states"] = len(completed)
    stats["alpha_targets_reached"] = len(alpha)
    return alpha, stats


def hard_records(alpha: dict[int, int]) -> list[dict[str, int]]:
    """Return targets that set a new alpha record in increasing target order."""

    records: list[dict[str, int]] = []
    best = -1
    for target in sorted(alpha):
        source = alpha[target]
        if source > best:
            records.append({"target": target, "alpha": source})
            best = source
    return records


def bounded_descent_path_merge(limit: int) -> dict[str, int | bool]:
    """Build a finite convergence certificate using descent/path merging.

    Each number is accepted once its accelerated orbit descends below its
    starting value or intersects a previously certified lower orbit.  This is
    a finite verification control, not an asymptotic proof.
    """

    if limit < 1:
        raise ValueError("limit must be positive")
    certified: dict[int, int] = {1: 0}
    next_cache: dict[int, int] = {}
    certificates = 0
    max_steps = 0
    transition_evaluations = 0
    transition_cache_hits = 0
    for start in range(2, limit + 1):
        current = start
        path: list[int] = []
        seen: set[int] = set()
        while current not in certified and current not in seen and current >= start:
            seen.add(current)
            path.append(current)
            cached_next = next_cache.get(current)
            if cached_next is None:
                cached_next = (
                    current // 2
                    if current % 2 == 0
                    else (3 * current + 1) // 2
                )
                next_cache[current] = cached_next
                transition_evaluations += 1
            else:
                transition_cache_hits += 1
            current = cached_next
        if current in certified or current < start:
            certificates += 1
            max_steps = max(max_steps, len(path))
            for value in path:
                certified[value] = len(path)
    certified_below_limit = sum(value in certified for value in range(1, limit + 1))
    return {
        "limit": limit,
        "certified_count": certified_below_limit,
        "certificate_count": certificates,
        "max_path_steps": max_steps,
        "transition_evaluations": transition_evaluations,
        "transition_cache_hits": transition_cache_hits,
        "cached_transitions": len(next_cache),
        "all_certified": certified_below_limit == limit,
    }


def sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_report(
    target_limit: int,
    source_limit: int,
    verification_limit: int,
    *,
    max_steps: int = 100_000,
) -> dict[str, object]:
    alpha, scan_stats = scan_alpha_with_stats(
        target_limit, source_limit, max_steps=max_steps
    )
    records = hard_records(alpha)
    word = (1, 4)
    admissible = [
        target
        for target in range(1, target_limit + 1, 2)
        if inverse_chain(word, target) is not None
    ]
    formula_matches = []
    for target in admissible:
        candidate = inverse_source(word, target)
        if candidate.denominator == 1 and alpha.get(target) == int(candidate):
            formula_matches.append(target)
    report: dict[str, object] = {
        "programme": "07-collatz-inverse-frontier-dynamics",
        "status": "bounded-control-only",
        "inputs": {
            "target_limit": target_limit,
            "source_limit": source_limit,
            "verification_limit": verification_limit,
            "max_steps": max_steps,
        },
        "early_control": {
            "alpha_19": alpha.get(19),
            "orbit_33": odd_orbit(33),
            "inverse_1_4_at_19": str(inverse_source(word, 19)),
            "inverse_1_4_chain_at_19": inverse_chain(word, 19),
        },
        "scan": {
            "algorithm": "ascending-source global path-merge with transition cache",
            "targets_reached": len(alpha),
            "hard_record_count": len(records),
            "hard_records": records,
            "admissible_1_4_targets": len(admissible),
            "alpha_matches_1_4": len(formula_matches),
            "alpha_match_targets": formula_matches,
            "optimization": scan_stats,
        },
        "bounded_verification": bounded_descent_path_merge(verification_limit),
        "claim_boundary": {
            "establishes": [
                "exact accelerated odd-step arithmetic",
                "exact inverse-word affine identities",
                "finite ascending-source alpha values for reached targets",
                "finite descent/path-merge certificates",
            ],
            "does_not_establish": [
                "classical Collatz convergence for all integers",
                "the universal 32/9 upper bound",
                "E(N)/N -> 32/9",
                "eventual (1,4) dominance",
            ],
        },
        "source": {
            "verification_algorithm": "https://arxiv.org/abs/2602.10466",
            "published_2_71_verification": "https://link.springer.com/article/10.1007/s11227-025-07337-0",
        },
    }
    report["content_sha256"] = sha256_json(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-limit", type=int, default=1_000)
    parser.add_argument("--source-limit", type=int, default=100_000)
    parser.add_argument("--verification-limit", type=int, default=10_000)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        args.target_limit,
        args.source_limit,
        args.verification_limit,
        max_steps=args.max_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
