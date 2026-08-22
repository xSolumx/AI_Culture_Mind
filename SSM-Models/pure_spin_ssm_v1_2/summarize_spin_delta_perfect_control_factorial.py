"""Summarize the frozen 3x3 perfect-control factorial."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

INIT_SEEDS = (491, 499, 503)
DATA_SEEDS = (509, 521, 523)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(paths: list[Path]) -> dict[str, object]:
    if len(paths) != 9:
        raise ValueError("perfect-control factorial requires nine artifacts")
    rows = []
    hashes = None
    seen = set()
    for path in paths:
        payload = json.loads(path.read_text())
        if payload["stage"] != "spin_delta_perfect_control_factorial":
            raise ValueError(f"wrong stage: {path}")
        pair = (payload["config"]["init_seed"], payload["config"]["data_seed"])
        if pair in seen:
            raise ValueError(f"duplicate cell {pair}")
        seen.add(pair)
        if hashes is None:
            hashes = payload["implementation_sha256"]
        if payload["implementation_sha256"] != hashes:
            raise ValueError("implementation hashes differ")
        rows.append({
            "init_seed": pair[0],
            "data_seed": pair[1],
            "accuracy": {w: payload["final"][w]["accuracy"] for w in ("8", "16", "32")},
            "source": path.as_posix(),
            "source_sha256": _sha(path),
        })
    expected = {(i, d) for i in INIT_SEEDS for d in DATA_SEEDS}
    if seen != expected:
        raise ValueError("factorial cells do not match frozen grid")
    rows.sort(key=lambda row: (row["init_seed"], row["data_seed"]))
    init_ranges = {
        str(data): max(r["accuracy"]["16"] for r in rows if r["data_seed"] == data)
        - min(r["accuracy"]["16"] for r in rows if r["data_seed"] == data)
        for data in DATA_SEEDS
    }
    data_ranges = {
        str(init): max(r["accuracy"]["16"] for r in rows if r["init_seed"] == init)
        - min(r["accuracy"]["16"] for r in rows if r["init_seed"] == init)
        for init in INIT_SEEDS
    }
    robust = all(
        row["accuracy"]["8"] >= 0.95
        and row["accuracy"]["16"] >= 0.95
        and row["accuracy"]["32"] >= 0.93
        for row in rows
    )
    return {
        "schema_version": 1,
        "rows": rows,
        "perfect_control_robustness_pass": robust,
        "initialization_ranges_at_16": init_ranges,
        "data_order_ranges_at_16": data_ranges,
        "initialization_sensitivity_detected": max(init_ranges.values()) >= 0.05,
        "data_order_sensitivity_detected": max(data_ranges.values()) >= 0.05,
        "implementation_sha256": hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
