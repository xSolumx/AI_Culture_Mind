"""Aggregate independent fused gathered-block benchmark processes."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

from benchmark_fused_gathered_block_memory import UPDATE_LAWS, VARIANTS


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


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
    reports: list[dict[str, object]], *, input_hashes: list[str]
) -> dict[str, object]:
    if len(reports) != 3:
        raise ValueError("exactly three independent measurement processes required")
    if len(set(input_hashes)) != 3:
        raise ValueError("measurement processes must have distinct contents")
    if any(
        report["experiment"] != "fused gathered-block recurrent memory benchmark"
        for report in reports
    ):
        raise ValueError("unexpected experiment identifier")
    if any(not bool(report["passed"]) for report in reports):
        raise ValueError("a source benchmark failed correctness")
    for report in reports:
        _validate_timing_orders(report)
    if any(report["grid"] != reports[0]["grid"] for report in reports[1:]):
        raise ValueError("benchmark grids differ")
    indexed = [
        {(int(row["slots"]), int(row["batch"])): row for row in report["rows"]}
        for report in reports
    ]
    if any(set(rows) != set(indexed[0]) for rows in indexed[1:]):
        raise ValueError("benchmark cells differ")
    rows = []
    maximum_state_error = 0.0
    maximum_prediction_error = 0.0
    for key in sorted(indexed[0]):
        source_rows = [index[key] for index in indexed]
        timing = {}
        for variant in VARIANTS:
            process_medians = [
                float(row["timing"][variant]["median_ms"]) for row in source_rows
            ]
            process_memory = [
                float(row["timing"][variant]["incremental_peak_bytes"])
                for row in source_rows
            ]
            timing[variant] = {
                "process_median_ms": process_medians,
                "median_of_process_medians_ms": _median(process_medians),
                "process_incremental_peak_bytes": process_memory,
                "median_incremental_peak_bytes": _median(process_memory),
            }
        comparisons = {}
        for law in UPDATE_LAWS:
            fused = float(
                timing[f"{law}_triton_fused_gathered"]["median_of_process_medians_ms"]
            )
            dense = float(timing[f"{law}_eager_dense"]["median_of_process_medians_ms"])
            gathered = float(
                timing[f"{law}_eager_gathered"]["median_of_process_medians_ms"]
            )
            comparisons[law] = {
                "fused_speedup_over_eager_dense": dense / fused,
                "fused_speedup_over_eager_gathered": gathered / fused,
            }
        for source in source_rows:
            for law in UPDATE_LAWS:
                maximum_state_error = max(
                    maximum_state_error,
                    float(source["correctness"]["rows"][law]["maximum_state_error"]),
                )
                maximum_prediction_error = max(
                    maximum_prediction_error,
                    float(
                        source["correctness"]["rows"][law]["maximum_prediction_error"]
                    ),
                )
        rows.append(
            {
                "slots": key[0],
                "batch": key[1],
                "logical_state_scalars": source_rows[0]["logical_state_scalars"],
                "timing": timing,
                "comparisons": comparisons,
            }
        )
    principal = [
        row for row in rows if row["batch"] == 16 and row["slots"] in (1024, 4096)
    ]
    decisions = {
        law: all(
            float(row["comparisons"][law]["fused_speedup_over_eager_dense"]) > 1.0
            and float(row["comparisons"][law]["fused_speedup_over_eager_gathered"])
            > 1.0
            for row in principal
        )
        for law in UPDATE_LAWS
    }
    return {
        "experiment": "fused gathered-block recurrent memory benchmark aggregate",
        "protocol": "FUSED_GATHERED_BLOCK_MEMORY_PREREGISTRATION.md",
        "processes": 3,
        "input_sha256": input_hashes,
        "device": reports[0]["device"],
        "dtype": reports[0]["dtype"],
        "hardware": reports[0]["hardware"],
        "grid": reports[0]["grid"],
        "rows": rows,
        "summary": {
            "maximum_eager_fused_state_error": maximum_state_error,
            "maximum_eager_fused_prediction_error": maximum_prediction_error,
            "principal_direct_cells_pass": decisions["direct"],
            "principal_delta_cells_pass": decisions["delta"],
            "fused_gathered_advantage_supported": all(decisions.values()),
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
    result = aggregate(reports, input_hashes=[_sha256(path) for path in args.inputs])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
