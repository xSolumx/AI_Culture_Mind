from __future__ import annotations

import json
from pathlib import Path

import pytest

from .summarize_optimization import summarize


def _write_report(
    path: Path,
    *,
    seed: int,
    baseline_bpb: float,
    candidate_bpb: float,
    execution: str = "eager",
) -> None:
    report = {
        "config": {"seed": seed, "steps": 300},
        "dataset": {"name": "tiny_shakespeare", "split": "90/5/5"},
        "source_sha256": {"model.py": "same"},
        "execution": execution,
        "rows": [
            {
                "name": "identity_legacy",
                "final_bits_per_byte": baseline_bpb,
                "training_tokens_per_second": 100.0,
                "peak_cuda_bytes": 1000,
            },
            {
                "name": "identity_delta",
                "final_bits_per_byte": candidate_bpb,
                "training_tokens_per_second": 125.0,
                "peak_cuda_bytes": 800,
            },
        ],
    }
    path.write_text(json.dumps(report), encoding="utf-8")


def test_optimization_summary_applies_both_noninferiority_conditions(
    tmp_path: Path,
) -> None:
    first = tmp_path / "seed102.json"
    second = tmp_path / "seed103.json"
    _write_report(first, seed=102, baseline_bpb=3.0, candidate_bpb=3.04)
    _write_report(second, seed=103, baseline_bpb=3.0, candidate_bpb=2.97)

    report = summarize(
        [first, second],
        baseline="identity_legacy",
        candidate="identity_delta",
        expected_seeds={102, 103},
        maximum_mean_regression_bpb=0.01,
        maximum_seed_regression_bpb=0.05,
    )

    assert report["summary"]["passed"] is True
    assert report["summary"]["mean_candidate_minus_baseline_bpb"] == pytest.approx(
        0.005
    )
    assert report["summary"]["worst_candidate_minus_baseline_bpb"] == pytest.approx(
        0.04
    )
    assert report["summary"]["geometric_mean_candidate_throughput_speedup"] == 1.25


def test_optimization_summary_rejects_one_bad_seed(tmp_path: Path) -> None:
    first = tmp_path / "seed102.json"
    second = tmp_path / "seed103.json"
    _write_report(first, seed=102, baseline_bpb=3.0, candidate_bpb=3.06)
    _write_report(second, seed=103, baseline_bpb=3.0, candidate_bpb=2.94)

    report = summarize(
        [first, second],
        baseline="identity_legacy",
        candidate="identity_delta",
        expected_seeds={102, 103},
        maximum_mean_regression_bpb=0.01,
        maximum_seed_regression_bpb=0.05,
    )

    assert report["summary"]["mean_condition_passed"] is True
    assert report["summary"]["individual_seed_condition_passed"] is False
    assert report["summary"]["passed"] is False


def test_optimization_summary_rejects_missing_or_duplicate_seeds(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    duplicate = tmp_path / "duplicate.json"
    _write_report(first, seed=102, baseline_bpb=3.0, candidate_bpb=3.0)
    _write_report(duplicate, seed=102, baseline_bpb=3.0, candidate_bpb=3.0)

    with pytest.raises(ValueError, match="duplicate seed"):
        summarize(
            [first, duplicate],
            baseline="identity_legacy",
            candidate="identity_delta",
            expected_seeds={102, 103},
            maximum_mean_regression_bpb=0.01,
            maximum_seed_regression_bpb=0.05,
        )

    with pytest.raises(ValueError, match="seed set mismatch"):
        summarize(
            [first],
            baseline="identity_legacy",
            candidate="identity_delta",
            expected_seeds={102, 103},
            maximum_mean_regression_bpb=0.01,
            maximum_seed_regression_bpb=0.05,
        )


def test_optimization_summary_rejects_compiled_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "compiled.json"
    _write_report(
        artifact,
        seed=102,
        baseline_bpb=3.0,
        candidate_bpb=3.0,
        execution="reduce-overhead",
    )

    with pytest.raises(ValueError, match="expected eager execution"):
        summarize(
            [artifact],
            baseline="identity_legacy",
            candidate="identity_delta",
            expected_seeds={102},
            maximum_mean_regression_bpb=0.01,
            maximum_seed_regression_bpb=0.05,
        )
