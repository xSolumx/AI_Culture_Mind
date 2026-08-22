from __future__ import annotations

import json
from pathlib import Path

import pytest
from summarize_isotypic_spectrum_gate import summarize


def artifact(path: Path, seed: int, baseline: float, candidate: float) -> Path:
    payload = {
        "stage": "isotypic_spectrum",
        "variant_order": ["shared_retention", "isotypic_spectrum"],
        "initial_pairing": {
            "common_parameters_bitwise_equal": True,
            "maximum_absolute_logit_difference": 0.0,
        },
        "implementation_sha256": {"model.py": "same"},
        "dataset": {"name": "tiny_shakespeare"},
        "config": {"seed": seed},
        "results": [
            {"final_bits_per_byte": baseline},
            {"final_bits_per_byte": candidate},
        ],
    }
    path.write_text(json.dumps(payload))
    return path


def test_isotypic_spectrum_summary_applies_frozen_rule(tmp_path: Path) -> None:
    paths = [
        artifact(tmp_path / "283.json", 283, 2.70, 2.67),
        artifact(tmp_path / "293.json", 293, 2.72, 2.70),
        artifact(tmp_path / "307.json", 307, 2.69, 2.70),
    ]
    report = summarize(paths)
    assert report["wins"] == 2
    assert report["quality_pass"]


def test_isotypic_spectrum_summary_rejects_wrong_seed(tmp_path: Path) -> None:
    paths = [
        artifact(tmp_path / "283.json", 283, 2.70, 2.67),
        artifact(tmp_path / "293.json", 293, 2.72, 2.70),
        artifact(tmp_path / "311.json", 311, 2.69, 2.70),
    ]
    with pytest.raises(ValueError, match="expected frozen seeds"):
        summarize(paths)
