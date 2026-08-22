from __future__ import annotations

import json
from pathlib import Path

from spin_delta_query_gradient_topology import MODES, PROTOCOL, SEEDS
from summarize_spin_delta_query_gradient_topology import summarize


def _artifact(path: Path) -> None:
    local = [
        {
            "mode": "hard_fallback",
            "slot_gradient_norm": 0.0,
            "event_logit_gradient": 0.01,
        },
        {
            "mode": "soft_query_event",
            "slot_gradient_norm": 0.02,
            "event_logit_gradient": 0.01,
        },
        {
            "mode": "authoritative_query",
            "slot_gradient_norm": 0.2,
            "event_logit_gradient": 0.0,
        },
    ]
    rows = []
    for seed in SEEDS:
        paths = []
        for mode in MODES:
            paths.append(
                {
                    "mode": mode,
                    "loss": 1.0,
                    "final_query_event_mean": 0.0 if mode == "hard_fallback" else 0.1,
                    "query_event_gradient_norm": 0.01,
                    "query_slot_gradient_norm": 0.0
                    if mode == "hard_fallback"
                    else 0.02,
                    "maximum_absolute_logit_change_from_hard": {
                        "hard_fallback": 0.0,
                        "soft_query_event": 0.1,
                        "authoritative_query": 1.0,
                    }[mode],
                    "finite_logits": True,
                    "finite_router_gradients": True,
                }
            )
        rows.append({"seed": seed, "paths": paths})
    path.write_text(
        json.dumps(
            {
                "stage": "spin_delta_query_gradient_topology",
                "protocol": PROTOCOL,
                "seeds": list(SEEDS),
                "modes": list(MODES),
                "local_float64": local,
                "full_model": rows,
                "implementation_sha256": {"same": "hash"},
            }
        )
    )


def test_summary_selects_lower_perturbation_soft_restoration(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    _artifact(path)
    report = summarize(path)
    assert report["dead_path_certificate_pass"] is True
    assert report["soft_restoration_pass"] is True
    assert report["authoritative_restoration_pass"] is True
    assert report["selected_repair"] == "soft_query_event"


def test_summary_refuses_selection_when_hard_path_is_not_dead(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    _artifact(path)
    payload = json.loads(path.read_text())
    payload["full_model"][0]["paths"][0]["query_slot_gradient_norm"] = 0.1
    path.write_text(json.dumps(payload))
    report = summarize(path)
    assert report["dead_path_certificate_pass"] is False
    assert report["selected_repair"] is None
