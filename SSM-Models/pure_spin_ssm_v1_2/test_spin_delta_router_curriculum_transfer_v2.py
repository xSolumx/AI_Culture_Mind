from __future__ import annotations

import json
from pathlib import Path

import pytest

from spin_delta_router_curriculum_transfer_v2 import DATA_SEEDS, INIT_SEEDS, PROTOCOL
from summarize_spin_delta_router_curriculum_transfer import EXPECTED_SCHEDULES
from summarize_spin_delta_router_curriculum_transfer_v2 import summarize


def _metrics(value: float) -> dict[str, float]:
    return {
        "write_event_f1": value,
        "query_event_f1": value,
        "write_slot_accuracy": value,
        "query_slot_accuracy": value,
    }


def _artifacts(tmp_path: Path) -> list[Path]:
    paths = []
    for i, init in enumerate(INIT_SEEDS):
        for j, data in enumerate(DATA_SEEDS):
            arms = []
            for arm in ("fixed", "curriculum"):
                accuracy = 0.91 + 0.01 * i + 0.005 * j
                if arm == "curriculum":
                    accuracy = 0.98 + 0.002 * i - 0.001 * j
                arms.append(
                    {
                        "arm": arm,
                        "training_schedule": EXPECTED_SCHEDULES[arm],
                        "initial_state_sha256": f"clone-{init}-{data}",
                        "router_frozen": True,
                        "training_router_metrics": _metrics(1.0),
                        "training_examples": 102_400,
                        "training_tokens": 2_662_400 if arm == "fixed" else 2_009_600,
                        "final": {
                            str(writes): {"accuracy": accuracy}
                            for writes in (8, 16, 32)
                        },
                    }
                )
            execution_id = f"shared-{init}"
            payload = {
                "schema_version": 2,
                "stage": "spin_delta_router_curriculum_transfer",
                "protocol": PROTOCOL,
                "config": {
                    "init_seed": init,
                    "data_seed": data,
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
                "cohort_execution": {
                    "shared_router_single_execution": True,
                    "execution_id": execution_id,
                    "init_seed": init,
                    "data_seeds": list(DATA_SEEDS),
                },
                "router_phase": {
                    "core_untouched": True,
                    "post_router_state_sha256": execution_id,
                    "readiness": {
                        str(writes): {"router": _metrics(1.0)}
                        for writes in (2, 3, 5, 8, 16, 32)
                    },
                },
                "arms": arms,
            }
            path = tmp_path / f"i{init}_d{data}.json"
            path.write_text(json.dumps(payload))
            paths.append(path)
    return paths


def test_v2_summary_accepts_one_shared_router_per_init(tmp_path: Path) -> None:
    report = summarize(_artifacts(tmp_path))
    assert report["shared_router_single_execution_pass"] is True
    assert report["autonomous_router_validity_pass"] is True
    assert report["curriculum_transfer_pass"] is True


def test_v2_summary_rejects_split_execution(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    payload = json.loads(paths[1].read_text())
    payload["cohort_execution"]["execution_id"] = "different"
    paths[1].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="execution ID differs"):
        summarize(paths)
