from __future__ import annotations

import json
from pathlib import Path

import pytest

from .summarize_quality_cohort import EXPECTED_VARIANTS, summarize


def _artifact(path: Path, seed: int) -> None:
    rows = [
        {
            "name": name,
            "parameters": 100 + index,
            "final_bits_per_byte": 2.0 + 0.01 * index,
            "training_tokens_per_second": 1000.0 - index,
            "peak_cuda_bytes": 1000 + index,
            "training_target_sha256": f"target-{seed}",
        }
        for index, name in enumerate(EXPECTED_VARIANTS)
    ]
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "config": {"seed": seed, "steps": 1},
                "variants": list(EXPECTED_VARIANTS),
                "execution": "eager",
                "environment": {"compute_capability": [7, 5]},
                "dataset": {"sha": "fixed"},
                "source_sha256": {"model.py": "fixed"},
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )


def test_quality_summarizer_requires_exact_seed_set_and_targets(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _artifact(first, 11)
    _artifact(second, 13)
    result = summarize([first, second], (11, 13))
    assert result["status"] == "completed matched cohort; no promotion"
    assert result["best_non_mamba_arm"] == "identity_matched"
    with pytest.raises(ValueError, match="duplicate seed"):
        summarize([first, first], (11, 13))
