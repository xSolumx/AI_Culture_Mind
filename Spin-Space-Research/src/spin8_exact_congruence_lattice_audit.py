"""Exhaustively audit the finite congruence lattice recovered by Spin(8)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from action_congruence_lattice import exact_congruence_lattice_audit


def audit_result(result: dict[str, object]) -> dict[str, object]:
    next_states = np.asarray(result["primary"]["recovered_next_states"], dtype=np.int64)
    exact = exact_congruence_lattice_audit(next_states)
    discovered = result["finest_congruence_discovery"]
    metric_cardinalities = tuple(int(value) for value in discovered["viable_cardinalities"])
    exact_nontrivial = {
        int(cardinality)
        for cardinality in exact["congruence_count_by_block_count"]
        if int(cardinality) > 1
    }
    metric_set = set(metric_cardinalities)
    return {
        "seed": int(result["seed"]),
        "metric_scan_viable_cardinalities": list(metric_cardinalities),
        "exact_lattice": exact,
        "metric_scan_covers_every_nontrivial_lattice_cardinality": (
            metric_set == exact_nontrivial
        ),
        "missing_exact_nontrivial_cardinalities": sorted(exact_nontrivial - metric_set),
        "metric_scan_is_not_a_complete_lattice_enumeration": metric_set != exact_nontrivial,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    results = [audit_result(result) for result in source["results"]]
    expected_q8_lattice = {"1": 1, "2": 3, "4": 1, "8": 1}
    checks = {
        "all_nine_inputs_audited": len(results) == 9,
        "all_actions_have_exact_q8_congruence_spectrum": all(
            result["exact_lattice"]["congruence_count_by_block_count"]
            == expected_q8_lattice
            for result in results
        ),
        "all_4140_partitions_exhausted": all(
            result["exact_lattice"]["enumerated_set_partitions"] == 4140
            for result in results
        ),
        "metric_scan_incomplete_for_every_seed": all(
            result["metric_scan_is_not_a_complete_lattice_enumeration"]
            for result in results
        ),
        "observation_free_identifiability_refused": all(
            not result["exact_lattice"][
                "observation_free_unique_nontrivial_quotient_identifiable"
            ]
            for result in results
        ),
    }
    report = {
        "experiment": "exhaustive Spin8 recovered-action congruence-lattice audit",
        "source": str(args.input),
        "expected_q8_congruence_count_by_block_count": expected_q8_lattice,
        "checks": checks,
        "passed": all(checks.values()),
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
