"""Validate and summarize the single-router-execution v2 transfer cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import summarize_spin_delta_router_curriculum_transfer as v1
from spin_delta_router_curriculum_transfer_v2 import DATA_SEEDS, INIT_SEEDS, PROTOCOL


def summarize(paths: list[Path]) -> dict[str, object]:
    execution_ids: dict[int, str] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("protocol") != PROTOCOL or payload.get("schema_version") != 2:
            raise ValueError(f"wrong v2 protocol: {path}")
        execution = payload.get("cohort_execution", {})
        config = payload["config"]
        if (
            execution.get("shared_router_single_execution") is not True
            or execution.get("init_seed") != config["init_seed"]
            or execution.get("data_seeds") != list(DATA_SEEDS)
        ):
            raise ValueError(f"invalid shared-router execution: {path}")
        prior = execution_ids.setdefault(
            config["init_seed"], execution.get("execution_id")
        )
        if execution.get("execution_id") != prior:
            raise ValueError("execution ID differs within an initialization cohort")

    old_init, old_data = v1.INIT_SEEDS, v1.DATA_SEEDS
    v1.INIT_SEEDS, v1.DATA_SEEDS = INIT_SEEDS, DATA_SEEDS
    try:
        report = v1.summarize(paths)
    finally:
        v1.INIT_SEEDS, v1.DATA_SEEDS = old_init, old_data
    report["schema_version"] = 2
    report["protocol"] = PROTOCOL
    report["shared_router_single_execution_pass"] = True
    report["cohort_execution_ids"] = {
        str(seed): execution_ids[seed] for seed in INIT_SEEDS
    }
    return report


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
