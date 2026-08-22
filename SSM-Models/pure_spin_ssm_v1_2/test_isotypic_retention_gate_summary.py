from __future__ import annotations

import json
from pathlib import Path

import pytest
from summarize_isotypic_retention_gate import summarize


def artifact(path: Path, seed: int, baseline: float, candidate: float) -> Path:
    payload = {
        "stage": "isotypic_retention",
        "variant_order": ["shared_retention", "isotypic_retention"],
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


def test_isotypic_retention_summary_applies_frozen_rule(tmp_path: Path) -> None:
    paths = [
        artifact(tmp_path / "271.json", 271, 2.70, 2.67),
        artifact(tmp_path / "277.json", 277, 2.72, 2.70),
        artifact(tmp_path / "281.json", 281, 2.69, 2.70),
    ]
    report = summarize(paths)
    assert report["wins"] == 2
    assert report["quality_pass"]
    assert report["speed_gate_authorized"]


def test_isotypic_retention_summary_rejects_nonpaired_input(
    tmp_path: Path,
) -> None:
    paths = [
        artifact(tmp_path / "271.json", 271, 2.70, 2.68),
        artifact(tmp_path / "277.json", 277, 2.72, 2.71),
        artifact(tmp_path / "281.json", 281, 2.69, 2.70),
    ]
    payload = json.loads(paths[0].read_text())
    payload["initial_pairing"]["maximum_absolute_logit_difference"] = 1.0e-7
    paths[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="initial logit mismatch"):
        summarize(paths)
