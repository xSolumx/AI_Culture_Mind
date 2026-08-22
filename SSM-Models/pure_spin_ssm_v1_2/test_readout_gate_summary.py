import json
from pathlib import Path

import pytest

from pure_spin_ssm_v1_2.summarize_readout_gate import summarize


def _artifact(path: Path, seed: int, improvement: float) -> Path:
    baseline = 2.7
    report = {
        "config": {"seed": seed, "steps": 300},
        "readouts": ["direction", "triality_invariants"],
        "parameter_counts": {"direction": 10, "triality_invariants": 11},
        "dataset": {"sha256": "fixed"},
        "environment": {"cuda": "12.6"},
        "implementation_sha256": {"model.py": "fixed"},
        "results": [
            {"readout": "direction", "final_bits_per_byte": baseline},
            {
                "readout": "triality_invariants",
                "final_bits_per_byte": baseline - improvement,
            },
        ],
    }
    path.write_text(json.dumps(report))
    return path


def test_gate_enforces_mean_effect_even_with_three_wins(tmp_path: Path) -> None:
    paths = [
        _artifact(tmp_path / f"{seed}.json", seed, improvement)
        for seed, improvement in zip((71, 73, 79), (0.015, 0.002, 0.003))
    ]
    result = summarize(paths)
    assert result["decision"]["wins"] == 3
    assert result["decision"]["passed"] is False
    assert result["decision"]["verdict"] == "retain_direction_default"


def test_gate_rejects_wrong_seed_set(tmp_path: Path) -> None:
    paths = [_artifact(tmp_path / f"{seed}.json", seed, 0.02) for seed in (71, 73, 80)]
    with pytest.raises(ValueError, match="expected exactly seeds"):
        summarize(paths)
