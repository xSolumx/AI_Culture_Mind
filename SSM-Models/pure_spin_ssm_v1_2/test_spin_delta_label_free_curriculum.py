from __future__ import annotations

import json
from pathlib import Path

from spin_delta_label_free_curriculum import (
    ARMS,
    DATA_SEEDS,
    INIT_SEEDS,
    PROTOCOL,
    READINESS_WRITES,
    SCHEDULES,
)
from summarize_spin_delta_label_free_curriculum import summarize


def _router(gauge: str) -> dict[str, float | int]:
    correct = 128 if gauge == "identity" else 0
    return {
        "write_event_f1": 1.0,
        "query_event_f1": 1.0,
        "write_slot_accuracy": correct / 128,
        "query_slot_accuracy": correct / 128,
        "write_slot_correct": correct,
        "write_slot_total": 128,
        "query_slot_correct": correct,
        "query_slot_total": 128,
    }


def _artifacts(tmp_path: Path) -> list[Path]:
    paths = []
    for i, init in enumerate(INIT_SEEDS):
        source = f"source-{init}"
        for j, data in enumerate(DATA_SEEDS):
            arms = []
            for arm in ARMS:
                accuracy = 0.91 + 0.01 * i + 0.005 * j
                if arm == "curriculum":
                    accuracy = 0.98 + 0.002 * i - 0.001 * j
                gauge = "swap" if (i + j) % 2 else "identity"
                arms.append(
                    {
                        "arm": arm,
                        "training_schedule": [
                            {"writes": writes, "steps": steps}
                            for writes, steps in SCHEDULES[arm]
                        ],
                        "initial_state_sha256": source,
                        "training_uses_router_labels": False,
                        "router_auxiliary_loss_weight": 0.0,
                        "oracle_controls_supplied_to_model": False,
                        "audit_labels_detached_from_loss": True,
                        "training_examples": 102_400,
                        "training_tokens": 2_662_400 if arm == "fixed" else 2_009_600,
                        "final": {
                            str(writes): {"accuracy": accuracy}
                            for writes in (8, 16, 32)
                        },
                        "router_readiness": {
                            str(writes): {"router": _router(gauge)}
                            for writes in READINESS_WRITES
                        },
                    }
                )
            payload = {
                "stage": "spin_delta_label_free_curriculum",
                "protocol": PROTOCOL,
                "config": {
                    "init_seed": init,
                    "data_seed": data,
                    "steps": 800,
                    "batch_size": 128,
                    "evaluation_writes": list(READINESS_WRITES),
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
                "source_state_sha256": source,
                "implementation_sha256": {"same": "hash"},
                "contract": {
                    "training_uses_router_labels": False,
                    "router_auxiliary_loss_weight": 0.0,
                    "oracle_controls_supplied_to_model": False,
                    "audit_labels_detached_from_loss": True,
                    "router_and_core_jointly_trainable": True,
                },
                "cohort_execution": {
                    "shared_initial_single_execution": True,
                    "execution_id": source,
                    "init_seed": init,
                    "data_seeds": list(DATA_SEEDS),
                },
                "arms": arms,
            }
            path = tmp_path / f"i{init}_d{data}.json"
            path.write_text(json.dumps(payload))
            paths.append(path)
    return paths


def test_summary_accepts_label_free_learning_up_to_slot_gauge(tmp_path: Path) -> None:
    report = summarize(_artifacts(tmp_path))
    assert report["label_free_contract_pass"] is True
    assert report["retrieval_autonomy_pass"] is True
    assert report["router_identification_pass"] is True
    assert report["learning_autonomy_pass"] is True
    assert {row["slot_gauge"] for row in report["rows"]} == {"identity", "swap"}


def test_summary_separates_retrieval_from_router_identification(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    payload = json.loads(paths[0].read_text())
    payload["arms"][1]["router_readiness"]["16"]["router"]["write_event_f1"] = 0.9
    paths[0].write_text(json.dumps(payload))
    report = summarize(paths)
    assert report["retrieval_autonomy_pass"] is True
    assert report["router_identification_pass"] is False
    assert report["learning_autonomy_pass"] is False
