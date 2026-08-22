from __future__ import annotations

import json
from pathlib import Path

from spin_delta_temporal_observability import (
    BATCH_SIZE,
    DEPTHS,
    MODES,
    PROTOCOL,
    SEEDS,
    WRITES,
    position_roles,
)
from summarize_spin_delta_temporal_observability import summarize


def _path(mode: str, depth: int, aligned: bool = True):
    hard = mode == "hard_fallback"
    soft = mode == "soft_query_event"
    nonfinal = 0.0 if depth == 1 else 1.0e-5
    return {
        "mode": mode,
        "hard_query_event_maximum": 0.0,
        "event_gradient_mean_absolute_by_position": [nonfinal] * 25 + [1.0e-4],
        "slot_gradient_mean_norm_by_position": [0.0 if hard else nonfinal] * 25
        + [0.0 if hard else 1.0e-4],
        "nonfinal_event_gradient_maximum_absolute": nonfinal,
        "nonfinal_slot_gradient_maximum_norm": 0.0 if hard else nonfinal,
        "final_event_gradient_mean_absolute": 1.0e-4
        if mode != "authoritative_query"
        else 0.0,
        "final_slot_gradient_mean_norm": 0.0 if hard else 1.0e-4,
        "nonfinal_event_off_aligned_mass": 0.95 if soft and aligned else 0.5,
        "final_event_on_aligned_mass": 0.95 if soft and aligned else 0.5,
        "final_correct_slot_descent_margin": 1.0e-4 if soft and aligned else -1.0e-4,
        "role_metrics": {},
        "finite": True,
    }


def _artifact(path: Path, aligned: bool = True):
    rows = []
    for seed in SEEDS:
        for depth in DEPTHS:
            rows.append(
                {
                    "seed": seed,
                    "depth": depth,
                    "paths": [_path(mode, depth, aligned) for mode in MODES],
                }
            )
    path.write_text(
        json.dumps(
            {
                "stage": "spin_delta_temporal_observability",
                "protocol": PROTOCOL,
                "seeds": list(SEEDS),
                "depths": list(DEPTHS),
                "modes": list(MODES),
                "writes": WRITES,
                "batch_size": BATCH_SIZE,
                "position_roles": position_roles(3 * WRITES + 2),
                "rows": rows,
                "implementation_sha256": {"same": "hash"},
            }
        )
    )


def test_summary_separates_observability_from_alignment(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    _artifact(path, aligned=False)
    report = summarize(path)
    assert report["one_block_structural_identity_pass"] is True
    assert report["stack_induced_observability_pass"] is True
    assert report["hard_slot_dead_path_pass"] is True
    assert report["grammar_aligned_credit_pass"] is False


def test_summary_accepts_fully_aligned_fixture(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    _artifact(path)
    report = summarize(path)
    assert report["final_path_topology_pass"] is True
    assert report["grammar_aligned_credit_pass"] is True
