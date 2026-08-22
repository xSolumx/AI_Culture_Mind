from __future__ import annotations

import json
from pathlib import Path

import pytest

from summarize_spin_delta_write_curriculum import (
    ARMS,
    DATA_SEEDS,
    EXPECTED_SCHEDULES,
    INIT_SEEDS,
    summarize,
)


def _artifacts(tmp_path: Path) -> list[Path]:
    paths = []
    for init_index, init_seed in enumerate(INIT_SEEDS):
        for data_index, data_seed in enumerate(DATA_SEEDS):
            for arm in ARMS:
                fixed_accuracy = 0.91 + 0.01 * init_index + 0.005 * data_index
                accuracy = 0.98 + 0.002 * init_index - 0.001 * data_index
                if arm == "fixed":
                    accuracy = fixed_accuracy
                payload = {
                    "stage": "spin_delta_write_curriculum",
                    "config": {
                        "init_seed": init_seed,
                        "data_seed": data_seed,
                        "arm": arm,
                        "steps": 800,
                        "batch_size": 128,
                        "evaluation_writes": [8, 16, 32],
                        "evaluation_batches": 16,
                        "learning_rate": 0.003,
                        "weight_decay": 0.01,
                        "gradient_clip": 1.0,
                        "d_model": 64,
                        "layers": 2,
                    },
                    "training_schedule": EXPECTED_SCHEDULES[arm],
                    "implementation_sha256": {"same": "hash"},
                    "initial_state_sha256": f"init-{init_seed}",
                    "initial": {"same": init_seed},
                    "training_examples": 102_400,
                    "training_tokens": 2_662_400 if arm == "fixed" else 2_009_600,
                    "final": {
                        str(writes): {"accuracy": accuracy}
                        for writes in (8, 16, 32)
                    },
                }
                path = tmp_path / f"i{init_seed}_d{data_seed}_{arm}.json"
                path.write_text(json.dumps(payload))
                paths.append(path)
    return paths


def test_summary_promotes_a_robust_curriculum(tmp_path: Path) -> None:
    report = summarize(_artifacts(tmp_path))
    assert report["arms"]["fixed"]["robustness_pass"] is False
    assert report["arms"]["curriculum"]["robustness_pass"] is True
    assert report["worst_cell_rescue_pass"] is True
    assert report["variance_contraction_pass"] is True
    assert report["no_large_paired_regression_pass"] is True
    assert report["robust_core_repair_pass"] is True


def test_summary_rejects_broken_pairing(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    payload = json.loads(paths[0].read_text())
    payload["initial_state_sha256"] = "tampered"
    paths[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="paired initial model states differ"):
        summarize(paths)
