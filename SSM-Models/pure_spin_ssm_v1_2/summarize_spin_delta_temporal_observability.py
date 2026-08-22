"""Apply frozen decisions to the Spin-Delta temporal observability audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from spin_delta_temporal_observability import (
    BATCH_SIZE,
    DEPTHS,
    MODES,
    PROTOCOL,
    SEEDS,
    WRITES,
    position_roles,
)


def summarize(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if (
        payload["stage"] != "spin_delta_temporal_observability"
        or payload["protocol"] != PROTOCOL
        or payload["seeds"] != list(SEEDS)
        or payload["depths"] != list(DEPTHS)
        or payload["modes"] != list(MODES)
        or payload["writes"] != WRITES
        or payload["batch_size"] != BATCH_SIZE
        or payload["position_roles"] != position_roles(3 * WRITES + 2)
    ):
        raise ValueError("audit does not match the frozen protocol")
    rows = {
        (row["seed"], row["depth"], path_row["mode"]): path_row
        for row in payload["rows"]
        for path_row in row["paths"]
    }
    expected = {
        (seed, depth, mode) for seed in SEEDS for depth in DEPTHS for mode in MODES
    }
    if set(rows) != expected:
        raise ValueError("audit rows do not match the frozen grid")
    finite = all(row["finite"] for row in rows.values())
    one_block_zero = finite and all(
        rows[(seed, 1, mode)]["nonfinal_event_gradient_maximum_absolute"] <= 1.0e-12
        and rows[(seed, 1, mode)]["nonfinal_slot_gradient_maximum_norm"] <= 1.0e-12
        for seed in SEEDS
        for mode in MODES
    )
    final_topology = finite and all(
        rows[(seed, depth, "hard_fallback")]["final_event_gradient_mean_absolute"]
        > 1.0e-8
        and rows[(seed, depth, "hard_fallback")]["final_slot_gradient_mean_norm"]
        <= 1.0e-12
        and rows[(seed, depth, "soft_query_event")]["final_slot_gradient_mean_norm"]
        > 1.0e-8
        and rows[(seed, depth, "authoritative_query")]["final_slot_gradient_mean_norm"]
        > 1.0e-8
        for seed in SEEDS
        for depth in DEPTHS
    )
    stack_induced = finite and all(
        rows[(seed, 2, "soft_query_event")]["nonfinal_event_gradient_maximum_absolute"]
        > 1.0e-10
        and rows[(seed, 2, "soft_query_event")]["nonfinal_slot_gradient_maximum_norm"]
        > 1.0e-10
        for seed in SEEDS
    )
    hard_slot_dead = finite and all(
        rows[(seed, depth, "hard_fallback")]["hard_query_event_maximum"] == 0.0
        and max(
            rows[(seed, depth, "hard_fallback")]["slot_gradient_mean_norm_by_position"]
        )
        <= 1.0e-12
        for seed in SEEDS
        for depth in DEPTHS
    )
    alignment_rows = []
    for seed in SEEDS:
        row = rows[(seed, 2, "soft_query_event")]
        alignment_rows.append(
            {
                "seed": seed,
                "nonfinal_event_off_aligned_mass": row[
                    "nonfinal_event_off_aligned_mass"
                ],
                "final_event_on_aligned_mass": row["final_event_on_aligned_mass"],
                "final_correct_slot_descent_margin": row[
                    "final_correct_slot_descent_margin"
                ],
            }
        )
    grammar_aligned = finite and all(
        row["nonfinal_event_off_aligned_mass"] >= 0.90
        and row["final_event_on_aligned_mass"] >= 0.90
        and row["final_correct_slot_descent_margin"] > 0.0
        for row in alignment_rows
    )
    compact_rows = []
    for key in sorted(rows):
        row = rows[key]
        compact_rows.append(
            {
                "seed": key[0],
                "depth": key[1],
                "mode": key[2],
                "nonfinal_event_gradient_maximum_absolute": row[
                    "nonfinal_event_gradient_maximum_absolute"
                ],
                "nonfinal_slot_gradient_maximum_norm": row[
                    "nonfinal_slot_gradient_maximum_norm"
                ],
                "final_event_gradient_mean_absolute": row[
                    "final_event_gradient_mean_absolute"
                ],
                "final_slot_gradient_mean_norm": row["final_slot_gradient_mean_norm"],
            }
        )
    return {
        "schema_version": 1,
        "finite_audit_pass": finite,
        "one_block_structural_identity_pass": one_block_zero,
        "final_path_topology_pass": final_topology,
        "stack_induced_observability_pass": stack_induced,
        "hard_slot_dead_path_pass": hard_slot_dead,
        "grammar_aligned_credit_pass": grammar_aligned,
        "alignment_rows": alignment_rows,
        "compact_rows": compact_rows,
        "role_rows": [
            {
                "seed": seed,
                "depth": depth,
                "mode": mode,
                "role_metrics": rows[(seed, depth, mode)]["role_metrics"],
            }
            for seed in SEEDS
            for depth in DEPTHS
            for mode in MODES
        ],
        "source": path.as_posix(),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "implementation_sha256": payload["implementation_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
