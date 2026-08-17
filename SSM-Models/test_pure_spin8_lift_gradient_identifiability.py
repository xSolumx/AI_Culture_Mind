from __future__ import annotations

import hashlib
import json
from pathlib import Path

import analyze_pure_spin8_lift_gradient_identifiability as analysis
import torch

ROOT = Path(__file__).resolve().parent


def _seed4_source() -> Path:
    return (
        ROOT
        / "experiments"
        / "artifacts"
        / "pure_spin8_lift_bit_calibration_validation_seed4.json"
    )


def test_independent_negative_specific_head_is_decay_only_in_seed4_checkpoint() -> None:
    audit = analysis.checkpoint_decay_audit(_seed4_source())
    assert audit["passed"]
    assert audit["checks"]["negative_weight_is_decay_only"]
    assert audit["checks"]["negative_bias_is_decay_only"]
    assert audit["checks"]["vector_weight_has_data_update"]
    assert audit["checks"]["positive_weight_has_data_update"]


def test_first_frozen_batch_has_the_predicted_gradient_support() -> None:
    source = json.loads(_seed4_source().read_text(encoding="utf-8"))
    config = analysis._config_from_mapping(source["config"])
    audit = analysis.first_batch_gradient_audit(
        seed=4,
        config=config,
        device=torch.device("cpu"),
    )
    assert audit["passed"]
    assert audit["checks"]["independent_negative_weight_gradient_exactly_zero"]
    assert audit["checks"]["independent_shared_trunk_has_data_gradient"]
    assert audit["checks"]["shared_all_28_coordinate_rows_have_data_gradient"]


def test_three_seed_gradient_certificate_is_content_locked_and_passed() -> None:
    path = (
        ROOT
        / "experiments"
        / "artifacts"
        / "pure_spin8_lift_gradient_identifiability_certificate.json"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "dee7c22e94bd627704609b7cad58939532e11c583a483e0e73987149f9339ab5"
    )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact["passed"]
    assert artifact["seeds"] == [4, 5, 6]
