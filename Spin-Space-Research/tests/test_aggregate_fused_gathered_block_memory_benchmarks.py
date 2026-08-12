from __future__ import annotations

import unittest

from aggregate_fused_gathered_block_memory_benchmarks import aggregate
from benchmark_fused_gathered_block_memory import UPDATE_LAWS, VARIANTS


def report(scale: float) -> dict[str, object]:
    timing_orders = [VARIANTS[offset:] + VARIANTS[:offset] for offset in range(3)]
    rows = []
    for slots in (1024, 4096):
        timing = {}
        for variant in VARIANTS:
            value = scale
            if variant.endswith("eager_dense"):
                value *= 4.0
            elif variant.endswith("eager_gathered"):
                value *= 8.0
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
        "experiment": "fused gathered-block recurrent memory benchmark",
        "passed": True,
        "device": "cuda",
        "dtype": "torch.float32",
        "hardware": {},
        "grid": {"timing_blocks": 3},
        "claim_boundary": {},
        "rows": rows,
    }


class AggregateFusedGatheredBlockMemoryTests(unittest.TestCase):
    def test_frozen_decision_and_duplicate_rejection(self) -> None:
        reports = [report(scale) for scale in (1.0, 1.1, 1.2)]
        result = aggregate(reports, input_hashes=["a", "b", "c"])
        self.assertTrue(result["summary"]["fused_gathered_advantage_supported"])
        with self.assertRaisesRegex(ValueError, "distinct contents"):
            aggregate(reports, input_hashes=["a", "a", "c"])
        reports[0]["rows"][0]["timing_block_orders"][1] = VARIANTS
        with self.assertRaisesRegex(ValueError, "not cycled"):
            aggregate(reports, input_hashes=["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
