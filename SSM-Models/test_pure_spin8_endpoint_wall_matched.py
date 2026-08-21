"""Contracts for the frozen endpoint-only measured-wall continuation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import benchmark_pure_spin8_continuous_observation as continuous
from benchmark_pure_spin8_continuous_wall_matched import (
    _numeric_max_abs,
    wall_alignment,
)
from benchmark_pure_spin8_endpoint_wall_matched import (
    FROZEN_WALL_UPDATES,
    REFERENCE_UPDATES,
)

ROOT = Path(__file__).resolve().parent
DEVELOPMENT = (
    ROOT
    / "experiments"
    / "artifacts"
    / "pure_spin8_endpoint_supervision_development_seed0.json"
)
WALL_AGGREGATE = (
    ROOT
    / "experiments"
    / "artifacts"
    / "pure_spin8_endpoint_supervision_wall_matched_seeds1_3.json"
)
WALL_AGGREGATE_SHA256 = (
    "538a3bdbddfd76863a5bef5507a6d0019a114b35021d7b5b9d1223d31983ac64"
)


def test_frozen_allocation_reconstructs_development_formula() -> None:
    report = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
    shared_wall = report["results"]["shared_pure_spin8"][
        "training_wall_seconds"
    ]
    reconstructed = {
        name: round(
            REFERENCE_UPDATES
            * shared_wall
            / report["results"][name]["training_wall_seconds"]
        )
        for name in continuous.CANDIDATES
    }
    assert reconstructed == FROZEN_WALL_UPDATES


def test_frozen_allocation_and_candidate_order_are_exact() -> None:
    assert tuple(FROZEN_WALL_UPDATES) == continuous.CANDIDATES
    assert FROZEN_WALL_UPDATES == {
        "shared_pure_spin8": 2_000,
        "independent_so8_triplet": 1_558,
        "mamba2_parameter_near": 2_811,
        "gru_parameter_near": 11_907,
        "observation_only_ablation": 15_482,
        "gru_state_matched": 11_911,
    }


def test_wall_alignment_is_descriptive_not_a_hidden_gate() -> None:
    results = {
        name: {"training_wall_seconds": value}
        for name, value in zip(
            continuous.CANDIDATES, (10.0, 9.0, 11.0, 10.5, 8.0, 12.0)
        )
    }
    result = wall_alignment(results)
    assert result["reference_wall_seconds"] == 10.0
    assert result["rows"]["independent_so8_triplet"]["ratio_to_shared"] == 0.9
    assert result["maximum_nonreference_relative_deviation"] == pytest.approx(
        0.2
    )


def test_numeric_replay_comparison_requires_matching_structure() -> None:
    assert _numeric_max_abs({"x": [1.0, 2]}, {"x": [1.0, 2]}) == 0.0
    assert _numeric_max_abs({"x": 1.0}, {"x": 1.25}) == 0.25
    assert _numeric_max_abs({"x": 1.0}, {"y": 1.0}) == float("inf")


def test_wall_aggregate_is_content_locked_and_integrity_passed() -> None:
    payload = WALL_AGGREGATE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == WALL_AGGREGATE_SHA256
    report = json.loads(payload)
    assert report["passed"]
    assert all(report["cohort_checks"].values())
    assert report["aggregate"][
        "observed_shared_beats_every_row_on_every_l128_split"
    ]
