from __future__ import annotations

import json
from pathlib import Path

from summarize_spin_delta_perfect_control_factorial import (
    DATA_SEEDS,
    INIT_SEEDS,
    summarize,
)


def test_factorial_summary_separates_initialization_and_data_order(
    tmp_path: Path,
) -> None:
    paths = []
    for init_index, init_seed in enumerate(INIT_SEEDS):
        for data_index, data_seed in enumerate(DATA_SEEDS):
            accuracy = 0.98 - 0.06 * init_index + 0.01 * data_index
            payload = {
                "stage": "spin_delta_perfect_control_factorial",
                "config": {"init_seed": init_seed, "data_seed": data_seed},
                "implementation_sha256": {"same": "hash"},
                "final": {
                    str(writes): {"accuracy": accuracy} for writes in (8, 16, 32)
                },
            }
            path = tmp_path / f"i{init_seed}_d{data_seed}.json"
            path.write_text(json.dumps(payload))
            paths.append(path)
    report = summarize(paths)
    assert report["perfect_control_robustness_pass"] is False
    assert report["initialization_sensitivity_detected"] is True
    assert report["data_order_sensitivity_detected"] is False
