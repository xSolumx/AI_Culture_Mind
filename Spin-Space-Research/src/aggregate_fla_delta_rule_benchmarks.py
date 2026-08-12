"""Aggregate frozen external FLA benchmark processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

VARIANTS = (
    "direct_slot",
    "triality_slot",
    "local_eager_delta",
    "fla_chunk_delta",
    "fla_fused_recurrent_delta",
)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values)
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "p05": _percentile(values, 0.05),
        "p95": _percentile(values, 0.95),
        "maximum": max(values),
        "mean": mean,
        "stdev": stdev,
        "coefficient_of_variation": stdev / mean if mean else 0.0,
        "count": len(values),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distinct_inputs(paths: list[Path]) -> list[dict[str, str]]:
    inputs = [{"path": str(path), "sha256": _sha256(path)} for path in paths]
    if len({item["sha256"] for item in inputs}) != len(inputs):
        raise ValueError("independent process artifacts must have distinct contents")
    return inputs


def aggregate(paths: list[Path]) -> dict[str, object]:
    if len(paths) < 2:
        raise ValueError("at least two independent process artifacts are required")
    inputs = _distinct_inputs(paths)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    reference_protocol = reports[0]["protocol"]
    invariant_protocol_keys = (
        "dtype",
        "batch",
        "lengths",
        "state_scalars",
        "warmup",
        "forward_repeats",
        "forward_backward_repeats",
        "same_process_timing_blocks",
        "forward_samples_per_variant_and_length",
        "forward_backward_samples_per_variant_and_length",
        "selection_mode",
        "problem_seed",
        "tuning_problem_seed",
        "frozen_selection_inputs",
        "input_generation_timed",
        "address_encoding_timed",
        "transition_construction_timed_for_local_rows",
        "fla_operator_inputs_preconstructed",
        "noncommuting_value_transport",
        "fla_scale",
    )
    for report in reports:
        if not report["passed"]:
            raise ValueError("cannot aggregate a failed benchmark")
        for key in invariant_protocol_keys:
            if report["protocol"].get(key) != reference_protocol.get(key):
                raise ValueError(f"protocol mismatch for {key}")

    rows = []
    for row_index, length in enumerate(reference_protocol["lengths"]):
        process_rows = [report["rows"][row_index] for report in reports]
        selected = process_rows[0]["selected_implementations"]
        if any(row["length"] != length for row in process_rows):
            raise ValueError(f"row alignment mismatch at length {length}")
        if any(row["selected_implementations"] != selected for row in process_rows):
            raise ValueError(f"implementation selection mismatch at length {length}")

        directions: dict[str, object] = {}
        for direction, sample_key in (
            ("forward", "forward_samples_ms"),
            ("forward_backward", "forward_backward_samples_ms"),
        ):
            directions[direction] = {}
            for variant in VARIANTS:
                per_process_samples = [
                    [float(value) for value in row[sample_key][variant]]
                    for row in process_rows
                ]
                process_medians = [
                    statistics.median(samples) for samples in per_process_samples
                ]
                pooled = [
                    sample
                    for process_samples in per_process_samples
                    for sample in process_samples
                ]
                directions[direction][variant] = {
                    "process_medians_ms": process_medians,
                    "process_median_summary_ms": _summary(process_medians),
                    "pooled_raw_summary_ms": _summary(pooled),
                    "raw_samples_ms_by_process": per_process_samples,
                }

        memory: dict[str, object] = {}
        for memory_key in (
            "forward_cuda_memory",
            "forward_backward_cuda_memory",
        ):
            memory[memory_key] = {
                variant: {
                    field: _summary(
                        [float(row[memory_key][variant][field]) for row in process_rows]
                    )
                    for field in (
                        "incremental_peak_allocated_bytes",
                        "incremental_peak_reserved_bytes",
                        "absolute_peak_allocated_bytes",
                        "absolute_peak_reserved_bytes",
                    )
                }
                for variant in VARIANTS
            }
        rows.append(
            {
                "length": length,
                "selected_implementations": selected,
                "maximum_forward_relative_error_vs_direct": {
                    variant: max(
                        float(row["forward_relative_error_vs_direct"][variant])
                        for row in process_rows
                    )
                    for variant in (
                        "triality_slot",
                        "local_eager_delta",
                        "fla_chunk_delta",
                        "fla_fused_recurrent_delta",
                    )
                },
                "maximum_fla_chunk_vs_local_gradient_relative_error": {
                    name: max(
                        float(row["fla_chunk_vs_local_gradient_relative_error"][name])
                        for row in process_rows
                    )
                    for name in ("values", "write_keys", "query_keys")
                },
                **directions,
                "cuda_memory": memory,
            }
        )

    return {
        "experiment": "independent-process aggregate of external FLA benchmark",
        "inputs": inputs,
        "protocol": {
            **reference_protocol,
            "independent_process_runs": len(reports),
            "forward_samples_per_variant_and_length": sum(
                int(report["protocol"]["forward_samples_per_variant_and_length"])
                for report in reports
            ),
            "forward_backward_samples_per_variant_and_length": sum(
                int(
                    report["protocol"][
                        "forward_backward_samples_per_variant_and_length"
                    ]
                )
                for report in reports
            ),
            "primary_location_statistic": "median of independent process medians",
        },
        "device_metadata_by_process": [report["device_metadata"] for report in reports],
        "rows": rows,
        "claim_boundary": reports[0]["claim_boundary"],
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
