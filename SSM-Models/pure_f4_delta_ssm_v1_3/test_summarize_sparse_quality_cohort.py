from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from .summarize_sparse_quality_cohort import EXPECTED_VARIANTS, summarize


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[list[Path], Path, tuple[int, ...]]:
    seeds = (11, 13, 17)
    reports = []
    for seed in seeds:
        rows = []
        for index, name in enumerate(EXPECTED_VARIANTS):
            checkpoint = tmp_path / f"{name}_{seed}.pt"
            checkpoint.write_bytes(f"{name}:{seed}".encode())
            bpb = {
                "e6_primitive_dead": 3.00,
                "e6_primitive_event": 2.98,
                "e6_safe": 2.99,
                "mamba2_official": 2.80,
            }[name]
            rows.append(
                {
                    "name": name,
                    "parameters": 40_848 if name == "mamba2_official" else 40_858,
                    "final_bits_per_byte": bpb + index * 1e-6,
                    "training_tokens_per_second": 10_000.0 + index,
                    "peak_cuda_bytes": 50_000_000 + index,
                    "training_target_sha256": f"targets-{seed}",
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _sha256(checkpoint),
                }
            )
        report = {
            "schema_version": 2,
            "config": {"seed": seed, "steps": 1000},
            "variants": list(EXPECTED_VARIANTS),
            "execution": "eager",
            "environment": {"compute_capability": [7, 5]},
            "dataset": {"name": "fixture"},
            "source_sha256": {"model.py": "same"},
            "rows": rows,
        }
        path = tmp_path / f"quality_{seed}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        reports.append(path)
    cost = tmp_path / "cost.json"
    cost.write_text(
        json.dumps(
            {
                "environment": {"compute_capability": [7, 5]},
                "verdict": {
                    "cheap_action_path_pass": True,
                    "mamba_competitive_pass": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return reports, cost, seeds


def test_sparse_quality_summary_separates_transport_and_mamba_promotion(
    tmp_path: Path,
) -> None:
    reports, cost, seeds = _fixture(tmp_path)
    result = summarize(reports, seeds, cost)
    assert result["gates"]["quality_noninferior_to_dead_budget"] is True
    assert result["gates"]["cheap_exceptional_transport_promoted"] is True
    assert result["gates"]["quality_beats_mamba2"] is False
    assert result["gates"]["complete_model_promoted_over_mamba2"] is False


def test_sparse_quality_summary_rejects_checkpoint_drift(tmp_path: Path) -> None:
    reports, cost, seeds = _fixture(tmp_path)
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    Path(report["rows"][0]["checkpoint"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint"):
        summarize(reports, seeds, cost)
