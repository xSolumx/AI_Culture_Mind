"""Validate and summarize a partial-Cayley joint-retraction cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(path: Path) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    results = report["results"]
    rows = []
    for result in results:
        recovery = result.get("transition_recovery") or {}
        mask = recovery.get("partial_inverse_cover") or {}
        endpoint = result.get("endpoint_recovery") or {}
        dense_values = [
            evaluation["minimum_accuracy"]
            for evaluation in result.get("dense_evaluation", {}).values()
        ]
        long_values = [
            evaluation["minimum_accuracy"]
            for evaluation in result.get("long_stress", {}).values()
        ]
        dense_minimum = min(dense_values) if dense_values else 0.0
        long_minimum = min(long_values) if long_values else 0.0
        diagnostics = result.get("mechanism_diagnostics", {})
        row = {
            "seed": result["training_seed"],
            "observed_edges": (
                endpoint.get("observed_transition_edges")
                if endpoint
                else mask.get("observed_edges")
            ),
            "completed_edges": (
                endpoint.get("completed_transition_edges")
                if endpoint
                else mask.get("completed_edges")
            ),
            "inferred_inverse_tokens": (
                endpoint.get("inferred_inverse_tokens")
                if endpoint
                else mask.get("inferred_inverse_tokens")
            ),
            "endpoint_labels": endpoint.get("total_endpoint_labels"),
            "samples_until_all_states": endpoint.get(
                "samples_until_all_states"
            ),
            "triggered": result.get("triggered", False),
            "trigger_step": result.get("trigger_step"),
            "compiler_invariance_rms": result.get("compiler_invariance_rms"),
            "compiler_homomorphism_rms": result.get("compiler_homomorphism_rms"),
            "float32_vector_homomorphism_rms": diagnostics.get(
                "vector_homomorphism_rms"
            ),
            "dense_minimum_accuracy": dense_minimum,
            "long_minimum_accuracy": long_minimum,
        }
        row["gate_pass"] = bool(
            row["triggered"]
            and row["inferred_inverse_tokens"] == [1, 0, 3, 2]
            and row["completed_edges"] == 240 - row["observed_edges"]
            and (not endpoint or row["endpoint_labels"] == 1_148)
            and row["compiler_invariance_rms"] < 1e-10
            and row["compiler_homomorphism_rms"] < 1e-10
            and dense_minimum >= 0.90
            and long_minimum >= 0.90
        )
        rows.append(row)

    numeric_keys = (
        "trigger_step",
        "compiler_invariance_rms",
        "compiler_homomorphism_rms",
        "float32_vector_homomorphism_rms",
        "dense_minimum_accuracy",
        "long_minimum_accuracy",
    )
    aggregates = {}
    for key in numeric_keys:
        available = [row[key] for row in rows if row[key] is not None]
        if available:
            values = np.asarray(available, dtype=np.float64)
            aggregates[key] = {
                "minimum": float(values.min()),
                "mean": float(values.mean()),
                "maximum": float(values.max()),
            }
        else:
            aggregates[key] = None
    return {
        "source": str(path),
        "seed_count": len(rows),
        "gate_passes": sum(row["gate_pass"] for row in rows),
        "all_seed_gate_pass": all(row["gate_pass"] for row in rows),
        "aggregates": aggregates,
        "seeds": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.input)
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
