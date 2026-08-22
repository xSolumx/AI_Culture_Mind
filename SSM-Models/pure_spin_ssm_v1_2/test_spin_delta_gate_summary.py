from __future__ import annotations

import json
from pathlib import Path

import pytest

from summarize_spin_delta_gate import EXPECTED_SEEDS, summarize


def _artifact(seed: int, baseline: float, candidate: float) -> dict[str, object]:
    return {
        "stage": "spin_delta",
        "variant_order": ["independent_v1_2", "spin_delta"],
        "config": {"seed": seed},
        "initial_pairing": {
            "common_parameters_bitwise_equal": True,
            "maximum_absolute_logit_difference": 8.0e-7,
        },
        "implementation_sha256": {"model.py": "same"},
        "dataset": {"name": "tiny_shakespeare", "sha256": "same"},
        "results": [
            {"final_bits_per_byte": baseline},
            {"final_bits_per_byte": candidate},
        ],
    }


def _write(tmp_path: Path, rows: list[dict[str, object]]) -> list[Path]:
    paths = []
    for index, row in enumerate(rows):
        path = tmp_path / f"row_{index}.json"
        path.write_text(json.dumps(row))
        paths.append(path)
    return paths


def test_spin_delta_summary_passes_only_the_frozen_rule(tmp_path: Path) -> None:
    rows = [
        _artifact(EXPECTED_SEEDS[0], 3.00, 2.98),
        _artifact(EXPECTED_SEEDS[1], 3.00, 2.989),
        _artifact(EXPECTED_SEEDS[2], 3.00, 3.00),
    ]
    report = summarize(_write(tmp_path, rows))
    assert report["quality_pass"] is True
    assert report["wins"] == 2


def test_spin_delta_summary_rejects_pairing_drift(tmp_path: Path) -> None:
    rows = [
        _artifact(seed, 3.00, 2.98) for seed in EXPECTED_SEEDS
    ]
    rows[1]["initial_pairing"]["maximum_absolute_logit_difference"] = 2.1e-6
    with pytest.raises(ValueError, match="pairing bound"):
        summarize(_write(tmp_path, rows))
