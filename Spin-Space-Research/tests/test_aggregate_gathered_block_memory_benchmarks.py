from __future__ import annotations

import unittest

from aggregate_gathered_block_memory_benchmarks import aggregate
from benchmark_gathered_block_memory import UPDATE_LAWS, VARIANTS


def source_report(scale: float) -> dict[str, object]:
    timing_orders = [VARIANTS[offset:] + VARIANTS[:offset] for offset in range(3)]
    rows = []
    for slots in (1024, 4096):
        timing = {}
        for variant in VARIANTS:
            value = scale
            if variant.endswith("block_masked_full"):
                value *= 1.2
            timing[variant] = {
                "median_ms": value,
                "incremental_peak_bytes": 100,
            }
        rows.append(
            {
                "slots": slots,
                "batch": 16,
                "logical_state_scalars": 16 * slots * 8,
                "timing_order": timing_orders[0],
                "timing_block_orders": timing_orders,
                "timing": timing,
                "correctness": {
                    "rows": {
                        law: {
                            "maximum_state_error": 0.0,
                            "maximum_prediction_error": 0.0,
                        }
                        for law in UPDATE_LAWS
                    }
                },
            }
        )
    return {
        "experiment": "actual gathered-block recurrent memory benchmark",
        "passed": True,
        "device": "cuda",
        "dtype": "torch.float32",
        "hardware": {},
        "grid": {
            "slots": [1024, 4096],
            "batches": [16],
            "timing_blocks": 3,
        },
        "router_parameters_at_max_slots": {},
        "claim_boundary": {},
        "rows": rows,
    }


class AggregateGatheredBlockMemoryBenchmarkTests(unittest.TestCase):
    def test_three_process_decision_and_duplicate_rejection(self) -> None:
        reports = [source_report(scale) for scale in (1.0, 1.1, 1.2)]
        result = aggregate(reports, input_hashes=["a", "b", "c"])
        self.assertTrue(result["summary"]["gathered_systems_advantage_supported"])
        with self.assertRaisesRegex(ValueError, "distinct contents"):
            aggregate(reports, input_hashes=["a", "a", "c"])
        reports[0]["rows"][0]["timing_block_orders"][1] = VARIANTS
        with self.assertRaisesRegex(ValueError, "not cycled"):
            aggregate(reports, input_hashes=["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
