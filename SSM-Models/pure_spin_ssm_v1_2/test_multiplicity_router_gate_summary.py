import json
from pathlib import Path

from pure_spin_ssm_v1_2.summarize_multiplicity_router_gate import summarize


def _artifact(path: Path, seed: int, improvement: float) -> Path:
    baseline = 2.7
    report = {
        "config": {"seed": seed, "steps": 300},
        "routers": ["none", "orthogonal_query"],
        "parameter_counts": {"none": 10, "orthogonal_query": 11},
        "dataset": {"sha256": "fixed"},
        "environment": {"cuda": "12.6"},
        "implementation_sha256": {"model.py": "fixed"},
        "results": [
            {"multiplicity_router": "none", "final_bits_per_byte": baseline},
            {
                "multiplicity_router": "orthogonal_query",
                "final_bits_per_byte": baseline - improvement,
            },
        ],
    }
    path.write_text(json.dumps(report))
    return path


def test_gate_rejects_one_of_three_router_wins(tmp_path: Path) -> None:
    paths = [
        _artifact(tmp_path / f"{seed}.json", seed, improvement)
        for seed, improvement in zip((83, 89, 97), (-0.003, -0.009, 0.015))
    ]
    result = summarize(paths)
    assert result["decision"]["wins"] == 1
    assert result["decision"]["passed"] is False
    assert result["decision"]["verdict"] == "retain_no_router_default"
