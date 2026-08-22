from __future__ import annotations

import json
from pathlib import Path

import pytest

from summarize_spin_delta_router_curriculum_transfer import (
    ARMS,
    DATA_SEEDS,
    EXPECTED_SCHEDULES,
    INIT_SEEDS,
    READINESS_WRITES,
    SCORED_WRITES,
    summarize,
)


def _router_metrics(value: float) -> dict[str, float]:
    return {
        "write_event_f1": value,
        "query_event_f1": value,
        "write_slot_accuracy": value,
        "query_slot_accuracy": value,
    }


def _artifacts(tmp_path: Path) -> list[Path]:
    paths = []
    for init_index, init_seed in enumerate(INIT_SEEDS):
        for data_index, data_seed in enumerate(DATA_SEEDS):
            arms = []
            for arm in ARMS:
                fixed = 0.91 + 0.01 * init_index + 0.005 * data_index
                accuracy = 0.98 + 0.002 * init_index - 0.001 * data_index
                if arm == "fixed":
                    accuracy = fixed
                arms.append(
                    {
                        "arm": arm,
                        "training_schedule": EXPECTED_SCHEDULES[arm],
                        "initial_state_sha256": f"clone-{init_seed}-{data_seed}",
                        "router_frozen": True,
                        "training_router_metrics": _router_metrics(1.0),
                        "training_examples": 102_400,
                        "training_tokens": 2_662_400 if arm == "fixed" else 2_009_600,
                        "final": {
                            str(writes): {"accuracy": accuracy}
                            for writes in SCORED_WRITES
                        },
                    }
                )
            payload = {
                "stage": "spin_delta_router_curriculum_transfer",
                "config": {
                    "init_seed": init_seed,
                    "data_seed": data_seed,
                    "router_steps": 100,
                    "core_steps": 800,
                    "batch_size": 128,
                    "router_training_writes": 8,
                    "evaluation_writes": [2, 3, 5, 8, 16, 32],
                    "evaluation_batches": 16,
                    "learning_rate": 0.003,
                    "weight_decay": 0.01,
                    "gradient_clip": 1.0,
                    "d_model": 64,
                    "layers": 2,
                    "router_width": 32,
                    "router_kernel_size": 3,
                    "router_temperature": 1.0,
                },
                "autonomous_evaluation": True,
                "oracle_controls_supplied_to_model": False,
                "implementation_sha256": {"same": "hash"},
                "router_phase": {
                    "core_untouched": True,
                    "post_router_state_sha256": f"router-{init_seed}",
                    "readiness": {
                        str(writes): {"router": _router_metrics(1.0)}
                        for writes in READINESS_WRITES
                    },
                },
                "arms": arms,
            }
            path = tmp_path / f"i{init_seed}_d{data_seed}.json"
            path.write_text(json.dumps(payload))
            paths.append(path)
    return paths


def test_summary_promotes_autonomous_curriculum_transfer(tmp_path: Path) -> None:
    report = summarize(_artifacts(tmp_path))
    assert report["autonomous_router_validity_pass"] is True
    assert report["arms"]["fixed"]["robustness_pass"] is False
    assert report["arms"]["curriculum"]["robustness_pass"] is True
    assert report["curriculum_transfer_pass"] is True


def test_summary_rejects_inexact_training_controls(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    payload = json.loads(paths[0].read_text())
    payload["arms"][0]["training_router_metrics"]["write_event_f1"] = 0.999
    paths[0].write_text(json.dumps(payload))
    report = summarize(paths)
    assert report["training_controls_exact_pass"] is False
    assert report["curriculum_transfer_pass"] is False


def test_summary_rejects_nonidentical_clones(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    payload = json.loads(paths[0].read_text())
    payload["arms"][1]["initial_state_sha256"] = "tampered"
    paths[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="cloned arm states differ"):
        summarize(paths)
