from __future__ import annotations

import json
from pathlib import Path

import pytest

from summarize_spin_delta_phased_router_gate import EXPECTED_SEEDS, summarize


def _artifact(seed: int, joint: float, phased: float, router: float) -> dict:
    router_row = {
        "write_event_f1": router,
        "query_event_f1": router,
        "write_slot_accuracy": router,
        "query_slot_accuracy": router,
    }
    result_row = lambda accuracy: {
        str(writes): {"accuracy": accuracy, "router": router_row}
        for writes in (8, 16, 32)
    }
    return {
        "stage": "spin_delta_phased_router_gate",
        "variant_order": ["joint_schedule", "phase_separated_schedule"],
        "config": {"seed": seed},
        "pairing": {"all_initial_tensors_bitwise_equal": True},
        "implementation_sha256": {"same": "hash"},
        "intervention": {"same": "intervention"},
        "results": [
            {
                "final": result_row(joint),
                "autonomous_evaluation": True,
            },
            {
                "final": result_row(phased),
                "phase_a_router": result_row(0.0),
                "phase_a_core_untouched": True,
                "router_frozen_during_phase_b": True,
                "autonomous_evaluation": True,
            },
        ],
    }


def _write(tmp_path: Path, joint: float, phased: float, router: float) -> list[Path]:
    paths = []
    for seed in EXPECTED_SEEDS:
        path = tmp_path / f"seed_{seed}.json"
        path.write_text(json.dumps(_artifact(seed, joint, phased, router)))
        paths.append(path)
    return paths


def test_phased_summary_accepts_all_decisions(tmp_path: Path) -> None:
    report = summarize(_write(tmp_path, 0.86, 0.97, 0.995))
    assert report["phase_separated_capacity_pass"] is True
    assert report["router_readiness_pass"] is True
    assert report["coadaptation_bottleneck_pass"] is True


def test_phased_summary_separates_capacity_and_differential(tmp_path: Path) -> None:
    report = summarize(_write(tmp_path, 0.96, 0.97, 0.995))
    assert report["phase_separated_capacity_pass"] is True
    assert report["router_readiness_pass"] is True
    assert report["coadaptation_bottleneck_pass"] is False


def test_phased_summary_rejects_modified_core(tmp_path: Path) -> None:
    paths = _write(tmp_path, 0.86, 0.97, 0.995)
    payload = json.loads(paths[1].read_text())
    payload["results"][1]["phase_a_core_untouched"] = False
    paths[1].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="phase separation contract"):
        summarize(paths)
