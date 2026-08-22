import copy
import json
from pathlib import Path

import pytest

from pure_spin_ssm_v1_2.analyze_steady_step_repeats import analyze


def _artifact(path: Path, spin: list[float], mamba: list[float]) -> Path:
    report = {
        "config": {"cycles": len(spin)},
        "environment": {"cuda": "12.6"},
        "parameter_match": {"relative_gap": 0.004},
        "implementation_sha256": {"model.py": "fixed"},
        "aggregate": {
            "pure_spin_v1_2": {
                "cycle_medians_tokens_per_second": spin,
                "median_of_cycle_medians_tokens_per_second": sorted(spin)[
                    len(spin) // 2
                ],
            },
            "mamba2_fused": {
                "cycle_medians_tokens_per_second": mamba,
                "median_of_cycle_medians_tokens_per_second": sorted(mamba)[
                    len(mamba) // 2
                ],
            },
        },
    }
    path.write_text(json.dumps(report))
    return path


def test_repeat_analysis_exposes_ordering_reversal(tmp_path: Path) -> None:
    first = _artifact(tmp_path / "first.json", [90.0], [100.0])
    second = _artifact(tmp_path / "second.json", [110.0], [100.0])
    result = analyze([first, second])
    assert result["analysis"]["ordering_reversed_across_repeats"] is True
    assert result["analysis"]["verdict"] == (
        "throughput_ordering_unresolved_at_observed_repeatability"
    )
    assert result["analysis"]["cycle_pair_count"] == 2


def test_repeat_analysis_rejects_incompatible_artifacts(tmp_path: Path) -> None:
    first = _artifact(tmp_path / "first.json", [90.0], [100.0])
    second = _artifact(tmp_path / "second.json", [90.0], [100.0])
    changed = json.loads(second.read_text())
    changed["implementation_sha256"] = copy.deepcopy(changed["implementation_sha256"])
    changed["implementation_sha256"]["model.py"] = "changed"
    second.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="incompatible benchmark artifact"):
        analyze([first, second])
