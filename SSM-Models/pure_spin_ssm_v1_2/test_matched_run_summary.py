import json
from pathlib import Path

import pytest

from pure_spin_ssm_v1_2.summarize_matched_runs import summarize


def _artifact(path: Path, seed: int, spin_bpb: float, mamba_bpb: float) -> Path:
    report = {
        "config": {"seed": seed, "steps": 300},
        "dataset": {"name": "tiny_shakespeare", "sha256": "fixed"},
        "environment": {"cuda": "12.6"},
        "parameter_match": {"relative_gap": 0.004},
        "implementation_sha256": {"model.py": "fixed"},
        "results": [
            {
                "name": "pure_spin_v1_2",
                "final_bits_per_byte": spin_bpb,
                "peak_cuda_bytes": 150,
            },
            {
                "name": "mamba2_fused",
                "final_bits_per_byte": mamba_bpb,
                "peak_cuda_bytes": 120,
            },
        ],
    }
    path.write_text(json.dumps(report))
    return path


def test_summary_reports_uniform_mamba_quality_win(tmp_path: Path) -> None:
    first = _artifact(tmp_path / "17.json", 17, 2.7, 2.5)
    second = _artifact(tmp_path / "29.json", 29, 2.8, 2.6)
    result = summarize([first, second])
    assert result["analysis"]["mamba2_quality_wins"] == 2
    assert result["analysis"]["mean_spin_minus_mamba2_bits_per_byte"] == pytest.approx(
        0.2
    )
    assert result["analysis"]["verdict"] == (
        "mamba2_wins_quality_at_all_recorded_seeds"
    )


def test_summary_rejects_duplicate_seed(tmp_path: Path) -> None:
    first = _artifact(tmp_path / "first.json", 17, 2.7, 2.5)
    second = _artifact(tmp_path / "second.json", 17, 2.8, 2.6)
    with pytest.raises(ValueError, match="distinct seeds"):
        summarize([first, second])
