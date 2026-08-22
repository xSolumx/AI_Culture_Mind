from __future__ import annotations

import json
from pathlib import Path

import pytest

from summarize_spin_delta_causal_router_gate import EXPECTED_SEEDS, summarize


def _artifact(seed: int, baseline: float, candidate: float, router: float) -> dict:
    baseline_rows = {
        str(writes): {"accuracy": baseline} for writes in (8, 16, 32)
    }
    candidate_rows = {
        str(writes): {
            "accuracy": candidate,
            "router": {
                "write_event_f1": router,
                "query_event_f1": router,
                "write_slot_accuracy": router,
                "query_slot_accuracy": router,
            },
        }
        for writes in (8, 16, 32)
    }
    return {
        "stage": "spin_delta_causal_router_gate",
        "variant_order": ["learned_continuous", "causal_discrete_aux"],
        "config": {"seed": seed},
        "pairing": {
            "all_core_tensors_bitwise_equal": True,
            "candidate_extra_parameters": 100,
        },
        "implementation_sha256": {"same": "hash"},
        "intervention": {"same": "intervention"},
        "results": [
            {"final": baseline_rows},
            {
                "autonomous_evaluation": True,
                "final": candidate_rows,
            },
        ],
    }


def _write(
    tmp_path: Path, baseline: float, candidate: float, router: float
) -> list[Path]:
    paths = []
    for seed in EXPECTED_SEEDS:
        path = tmp_path / f"seed_{seed}.json"
        path.write_text(json.dumps(_artifact(seed, baseline, candidate, router)))
        paths.append(path)
    return paths


def test_causal_router_summary_accepts_all_three_decisions(tmp_path: Path) -> None:
    report = summarize(_write(tmp_path, 0.84, 0.97, 0.995))
    assert report["autonomous_retrieval_capacity_pass"] is True
    assert report["router_identification_pass"] is True
    assert report["robust_rescue_pass"] is True


def test_causal_router_summary_keeps_decisions_independent(tmp_path: Path) -> None:
    report = summarize(_write(tmp_path, 0.96, 0.97, 0.98))
    assert report["autonomous_retrieval_capacity_pass"] is True
    assert report["router_identification_pass"] is False
    assert report["robust_rescue_pass"] is False


def test_causal_router_summary_rejects_core_mismatch(tmp_path: Path) -> None:
    paths = _write(tmp_path, 0.84, 0.97, 0.995)
    payload = json.loads(paths[0].read_text())
    payload["pairing"]["all_core_tensors_bitwise_equal"] = False
    paths[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="core parameter mismatch"):
        summarize(paths)
