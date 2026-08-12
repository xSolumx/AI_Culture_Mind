"""Aggregate independent gathered-block CUDA benchmark processes."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

from benchmark_gathered_block_memory import UPDATE_LAWS, VARIANTS


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return (
        ordered[middle]
        if len(ordered) % 2
        else 0.5 * (ordered[middle - 1] + ordered[middle])
    )


def _validate_timing_orders(report: dict[str, object]) -> None:
    expected_blocks = int(report["grid"]["timing_blocks"])
    for row in report["rows"]:
        raw_orders = row.get("timing_block_orders")
        if not isinstance(raw_orders, list) or len(raw_orders) != expected_blocks:
            raise ValueError("timing-block order record is incomplete")
        orders = [tuple(order) for order in raw_orders]
        if tuple(row.get("timing_order", ())) != orders[0]:
            raise ValueError("first timing order does not match block record")
        if any(
            len(order) != len(VARIANTS) or set(order) != set(VARIANTS)
            for order in orders
        ):
            raise ValueError("timing block is not a permutation of variants")
        if any(
            current != previous[1:] + previous[:1]
            for previous, current in pairwise(orders)
        ):
            raise ValueError("timing variants were not cycled across blocks")


def aggregate(
    reports: list[dict[str, object]],
    *,
    input_hashes: list[str] | None = None,
) -> dict[str, object]:
    if len(reports) != 3:
        raise ValueError("exactly three independent measurement processes required")
    if input_hashes is not None and len(set(input_hashes)) != len(input_hashes):
        raise ValueError("measurement processes must have distinct contents")
    if any(
        report["experiment"] != "actual gathered-block recurrent memory benchmark"
        for report in reports
    ):
        raise ValueError("unexpected experiment identifier")
    if any(not bool(report["passed"]) for report in reports):
        raise ValueError("a source benchmark failed correctness")
    for report in reports:
        _validate_timing_orders(report)
    reference_grid = reports[0]["grid"]
    if any(report["grid"] != reference_grid for report in reports[1:]):
        raise ValueError("benchmark grids differ")
    if any(report["device"] != reports[0]["device"] for report in reports[1:]):
        raise ValueError("benchmark devices differ")
    indexed = [
        {(int(row["slots"]), int(row["batch"])): row for row in report["rows"]}
        for report in reports
    ]
    if any(set(rows) != set(indexed[0]) for rows in indexed[1:]):
        raise ValueError("benchmark cells differ")
    aggregate_rows = []
    maximum_state_error = 0.0
    maximum_prediction_error = 0.0
    for key in sorted(indexed[0]):
        source_rows = [rows[key] for rows in indexed]
        timing: dict[str, object] = {}
        for variant in VARIANTS:
            process_medians = [
                float(row["timing"][variant]["median_ms"]) for row in source_rows
            ]
            memory_values = [
                row["timing"][variant]["incremental_peak_bytes"] for row in source_rows
            ]
            timing[variant] = {
                "process_median_ms": process_medians,
                "median_of_process_medians_ms": _median(process_medians),
                "process_incremental_peak_bytes": memory_values,
                "median_incremental_peak_bytes": (
                    None
                    if any(value is None for value in memory_values)
                    else _median([float(value) for value in memory_values])
                ),
            }
        for row in source_rows:
            for law in UPDATE_LAWS:
                maximum_state_error = max(
                    maximum_state_error,
                    float(row["correctness"]["rows"][law]["maximum_state_error"]),
                )
                maximum_prediction_error = max(
                    maximum_prediction_error,
                    float(row["correctness"]["rows"][law]["maximum_prediction_error"]),
                )
        comparisons = {}
        for law in UPDATE_LAWS:
            masked = float(
                timing[f"{law}_block_masked_full"]["median_of_process_medians_ms"]
            )
            gathered = float(
                timing[f"{law}_block_gathered"]["median_of_process_medians_ms"]
            )
            dense = float(timing[f"{law}_dense_full"]["median_of_process_medians_ms"])
            comparisons[law] = {
                "gathered_speedup_over_masked_full": masked / gathered,
                "gathered_speedup_over_dense_full": dense / gathered,
            }
        aggregate_rows.append(
            {
                "slots": key[0],
                "batch": key[1],
                "logical_state_scalars": source_rows[0]["logical_state_scalars"],
                "timing": timing,
                "comparisons": comparisons,
            }
        )
    principal = [
        row
        for row in aggregate_rows
        if row["batch"] == 16 and row["slots"] in (1024, 4096)
    ]
    principal_passes = {
        law: all(
            float(row["comparisons"][law]["gathered_speedup_over_masked_full"]) > 1.0
            for row in principal
        )
        for law in UPDATE_LAWS
    }
    return {
        "experiment": "gathered-block recurrent memory benchmark aggregate",
        "protocol": "GATHERED_BLOCK_MEMORY_BENCHMARK_PREREGISTRATION.md",
        "processes": len(reports),
        "input_sha256": input_hashes,
        "device": reports[0]["device"],
        "dtype": reports[0]["dtype"],
        "hardware": reports[0]["hardware"],
        "grid": reference_grid,
        "router_parameters_at_max_slots": reports[0]["router_parameters_at_max_slots"],
        "rows": aggregate_rows,
        "summary": {
            "maximum_masked_gathered_state_error": maximum_state_error,
            "maximum_masked_gathered_prediction_error": maximum_prediction_error,
            "principal_direct_cells_pass": principal_passes["direct"],
            "principal_delta_cells_pass": principal_passes["delta"],
            "gathered_systems_advantage_supported": all(principal_passes.values()),
            "claim_boundary": reports[0]["claim_boundary"],
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs=3, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    report = aggregate(reports, input_hashes=[_sha256(path) for path in args.inputs])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
