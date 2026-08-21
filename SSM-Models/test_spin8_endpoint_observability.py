from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch

import analyze_spin8_endpoint_observability as observability

ROOT = Path(__file__).resolve().parent


def test_every_single_representation_reaches_full_lie_rank() -> None:
    profiles = observability.probe_rank_profiles()
    assert set(profiles) == {"vector", "positive", "negative"}
    assert all(
        tuple(profile) == observability.EXPECTED_PROBE_RANK_PROFILE
        for profile in profiles.values()
    )


def test_tested_center_visibility_is_representation_dependent() -> None:
    signature = observability.center_signature()
    assert not signature["readouts"]["vector_only"]["center_visible"]
    assert signature["readouts"]["positive_only"]["center_visible"]
    assert signature["readouts"]["negative_only"]["center_visible"]
    assert signature["readouts"]["spinor_pair"]["center_visible"]
    assert signature["readouts"]["full_triality"]["center_visible"]
    for representation in ("vector", "positive", "negative"):
        assert (
            signature["per_representation"][representation][
                "action_signature_max_abs"
            ]
            <= 1e-12
        )


def test_vector_quotient_has_an_exact_hidden_lift_collision() -> None:
    certificate = observability.quotient_collision_certificate()
    assert certificate["vector_input_collision_max_abs"] <= 1e-12
    assert certificate["positive_spinor_target_negation_max_abs"] <= 1e-12
    assert certificate["balanced_conditional_mean_max_abs"] <= 1e-12
    assert math.isclose(
        certificate["balanced_spinor_state_bayes_mse"],
        0.125,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert certificate["balanced_hidden_lift_bayes_accuracy"] == 0.5


def test_full_certificate_passes_and_is_finite() -> None:
    certificate = observability.build_certificate()
    assert certificate["passed"]
    values = torch.tensor(
        [
            certificate["quotient_collision"][
                "vector_input_collision_max_abs"
            ],
            certificate["quotient_collision"][
                "balanced_spinor_state_bayes_mse"
            ],
        ]
    )
    assert torch.isfinite(values).all()


def test_exact_certificate_artifact_is_content_locked() -> None:
    path = (
        ROOT
        / "experiments"
        / "artifacts"
        / "pure_spin8_endpoint_observability_certificate.json"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "fa29a9d74a927993c17328b7dffb5f96c7f42f308b2e30450d4f714a9ce89a53"
    )
