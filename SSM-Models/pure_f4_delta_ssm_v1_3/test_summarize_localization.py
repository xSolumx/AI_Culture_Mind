from __future__ import annotations

import json
from pathlib import Path

import pytest

from .summarize_localization import summarize


def _write_report(
    path: Path, *, seed: int, identity_bpb: float, candidate_bpb: float
) -> None:
    report = {
        "config": {"seed": seed, "steps": 500},
        "dataset": {"name": "tiny_shakespeare", "split": "90/5/5"},
        "source_sha256": {"model.py": "same"},
        "rows": [
            {
                "name": "identity_delta",
                "final_bits_per_byte": identity_bpb,
                "training_tokens_per_second": 100.0,
            },
            {
                "name": "early_e6_delta",
                "final_bits_per_byte": candidate_bpb,
                "training_tokens_per_second": 80.0,
            },
        ],
    }
    path.write_text(json.dumps(report), encoding="utf-8")


def test_localization_summary_applies_paired_gate(tmp_path: Path) -> None:
    first = tmp_path / "seed1.json"
    second = tmp_path / "seed2.json"
    _write_report(first, seed=1, identity_bpb=3.0, candidate_bpb=2.98)
    _write_report(second, seed=2, identity_bpb=3.0, candidate_bpb=3.01)

    report = summarize(
        [first, second],
        baseline="identity_delta",
        candidate="early_e6_delta",
        required_wins=2,
        required_mean_improvement=0.01,
    )

    assert report["summary"]["wins"] == 1
    assert report["summary"]["mean_candidate_improvement_bpb"] == pytest.approx(0.005)
    assert report["summary"]["passed"] is False


def test_localization_summary_rejects_dataset_mismatch(tmp_path: Path) -> None:
    first = tmp_path / "seed1.json"
    second = tmp_path / "seed2.json"
    _write_report(first, seed=1, identity_bpb=3.0, candidate_bpb=2.9)
    _write_report(second, seed=2, identity_bpb=3.0, candidate_bpb=2.9)
    report = json.loads(second.read_text(encoding="utf-8"))
    report["dataset"]["name"] = "other"
    second.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset mismatch"):
        summarize(
            [first, second],
            baseline="identity_delta",
            candidate="early_e6_delta",
            required_wins=2,
            required_mean_improvement=0.01,
        )
