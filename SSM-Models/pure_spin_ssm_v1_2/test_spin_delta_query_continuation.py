from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from spin_delta_query_continuation import (
    ARMS,
    DATA_SEEDS,
    INIT_SEEDS,
    PROTOCOL,
    READINESS_WRITES,
    SCHEDULE,
    ContinuationConfig,
    continuation_alpha,
    query_event_controls,
)
from summarize_spin_delta_query_continuation import summarize


def test_linear_continuation_preserves_other_controls_and_reaches_hard() -> None:
    class Router:
        temperature = 1.0

    class Model:
        router = Router()

    class Routing:
        query_event_logits = torch.tensor([[-3.0, 1.0]])
        controls = torch.tensor(
            [[[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0, 1.0, 0.0]]]
        )

    soft, alpha_zero = query_event_controls(
        Model(), Routing(), "linear_continuation", 1, 800
    )
    hard, alpha_one = query_event_controls(
        Model(), Routing(), "linear_continuation", 800, 800
    )
    assert alpha_zero == 0.0
    assert alpha_one == 1.0
    assert torch.equal(soft[..., :3], Routing.controls[..., :3])
    assert torch.equal(soft[..., 4:], Routing.controls[..., 4:])
    assert torch.allclose(soft[..., 3], torch.sigmoid(Routing.query_event_logits))
    assert torch.equal(hard, Routing.controls)


def test_linear_continuation_preserves_event_surrogate_derivative() -> None:
    class Router:
        temperature = 1.0

    class Model:
        router = Router()

    derivatives = []
    event_values = []
    for step in (1, 400, 800):
        logit = torch.tensor([[-3.0]], dtype=torch.float64, requires_grad=True)
        probability = torch.sigmoid(logit)
        hard_value = (probability >= 0.5).to(probability)
        straight_through = hard_value.detach() - probability.detach() + probability

        class Routing:
            query_event_logits = logit
            controls = torch.cat(
                (
                    torch.zeros(1, 1, 3, dtype=torch.float64),
                    straight_through.unsqueeze(-1),
                    torch.zeros(1, 1, 2, dtype=torch.float64),
                ),
                dim=-1,
            )

        controls, _ = query_event_controls(
            Model(), Routing(), "linear_continuation", step, 800
        )
        derivatives.append(torch.autograd.grad(controls[..., 3].sum(), logit)[0])
        event_values.append(float(controls[..., 3].detach()))
    torch.testing.assert_close(derivatives[0], derivatives[1], rtol=1e-14, atol=0.0)
    torch.testing.assert_close(derivatives[1], derivatives[2], rtol=1e-14, atol=0.0)
    assert event_values[0] > event_values[1] > event_values[2] == 0.0


def _router_metrics(value: float) -> dict[str, float | int]:
    total = 100
    correct = round(value * total)
    return {
        "write_event_f1": value,
        "query_event_f1": value,
        "write_slot_accuracy": value,
        "query_slot_accuracy": value,
        "write_slot_correct": correct,
        "query_slot_correct": correct,
        "write_slot_total": total,
        "query_slot_total": total,
    }


def _artifact(
    path: Path, init_seed: int, data_seed: int, promoted: bool = False
) -> None:
    config = ContinuationConfig(init_seed, data_seed)
    source = f"state-{init_seed}"
    arms = []
    for arm in ARMS:
        continuation = arm == "linear_continuation"
        accuracy = (
            0.97 if promoted and continuation else (0.80 if continuation else 0.40)
        )
        router_value = (
            0.995 if promoted and continuation else (0.80 if continuation else 0.40)
        )
        end = 0
        start = 1
        ranges = {}
        for writes, steps in SCHEDULE:
            end += steps
            alpha_start = 1.0 if arm == "hard_event" else continuation_alpha(start, 800)
            alpha_end = 1.0 if arm == "hard_event" else continuation_alpha(end, 800)
            ranges[str(end)] = {
                "writes": writes,
                "alpha_start": alpha_start,
                "alpha_end": alpha_end,
            }
            start = end + 1
        readiness = {
            str(writes): {
                "accuracy": accuracy,
                "router": _router_metrics(router_value),
            }
            for writes in READINESS_WRITES
        }
        arms.append(
            {
                "arm": arm,
                "training_schedule": [
                    {"writes": writes, "steps": steps} for writes, steps in SCHEDULE
                ],
                "initial_state_sha256": source,
                "query_event_training_path": arm,
                "evaluation_uses_hard_router": True,
                "training_uses_router_labels": False,
                "router_auxiliary_loss_weight": 0.0,
                "oracle_controls_supplied_to_model": False,
                "audit_labels_detached_from_loss": True,
                "final": {
                    str(writes): readiness[str(writes)] for writes in (8, 16, 32)
                },
                "router_readiness": readiness,
                "event_forward_ranges": ranges,
                "training_examples": 102400,
                "training_tokens": 2009600,
            }
        )
    payload = {
        "stage": "spin_delta_query_continuation",
        "protocol": PROTOCOL,
        "config": asdict(config),
        "source_state_sha256": source,
        "arms": arms,
        "contract": {
            "only_query_event_forward_path_differs": True,
            "training_uses_router_labels": False,
            "router_auxiliary_loss_weight": 0.0,
            "oracle_controls_supplied_to_model": False,
            "audit_labels_detached_from_loss": True,
            "router_and_core_jointly_trainable": True,
            "evaluation_uses_hard_router": True,
        },
        "cohort_execution": {
            "shared_initial_single_execution": True,
            "identical_batch_generators_between_arms": True,
            "execution_id": source,
            "init_seed": init_seed,
            "data_seeds": list(DATA_SEEDS),
        },
        "implementation_sha256": {"same": "hash"},
    }
    path.write_text(json.dumps(payload))


def _cohort(tmp_path: Path, promoted: bool = False) -> list[Path]:
    paths = []
    for init_seed in INIT_SEEDS:
        for data_seed in DATA_SEEDS:
            path = tmp_path / f"i{init_seed}_d{data_seed}.json"
            _artifact(path, init_seed, data_seed, promoted)
            paths.append(path)
    return paths


def test_summary_separates_mechanism_repair_from_promotion(tmp_path: Path) -> None:
    report = summarize(_cohort(tmp_path))
    assert report["mechanism_repair_pass"] is True
    assert report["learning_autonomy_promotion_pass"] is False
    assert report["mean_paired_improvement_at_16"] == pytest.approx(0.40)


def test_summary_can_promote_only_at_strong_bar(tmp_path: Path) -> None:
    report = summarize(_cohort(tmp_path, promoted=True))
    assert report["mechanism_repair_pass"] is True
    assert report["learning_autonomy_promotion_pass"] is True


def test_summary_rejects_a_changed_continuation_schedule(tmp_path: Path) -> None:
    paths = _cohort(tmp_path)
    payload = json.loads(paths[0].read_text())
    payload["arms"][1]["event_forward_ranges"]["100"]["alpha_end"] = 0.9
    paths[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="continuation schedule differs"):
        summarize(paths)
