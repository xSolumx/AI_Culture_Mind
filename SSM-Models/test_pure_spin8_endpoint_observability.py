from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

import benchmark_pure_spin8_endpoint_observability as benchmark

ROOT = Path(__file__).resolve().parent


def test_partial_endpoint_loss_has_exact_readout_gradient_support() -> None:
    predictions = torch.randn(3, 5, 3, 8, requires_grad=True)
    targets = torch.randn(3, 1, 8)
    loss = benchmark.partial_endpoint_loss(predictions, targets, (1,))
    loss.backward()
    assert predictions.grad is not None
    assert torch.count_nonzero(predictions.grad[:, :-1]) == 0
    assert torch.count_nonzero(predictions.grad[:, -1, 0]) == 0
    assert torch.count_nonzero(predictions.grad[:, -1, 2]) == 0
    assert torch.count_nonzero(predictions.grad[:, -1, 1]) > 0


def test_readout_contracts_are_nonempty_unique_and_named() -> None:
    assert benchmark.READOUTS == {
        "vector_only": (0,),
        "positive_only": (1,),
        "negative_only": (2,),
        "spinor_pair": (1, 2),
        "full_triality": (0, 1, 2),
    }
    for indices in benchmark.READOUTS.values():
        assert indices
        assert len(indices) == len(set(indices))


def test_partial_endpoint_loss_rejects_invalid_masks() -> None:
    predictions = torch.randn(2, 3, 3, 8)
    targets = torch.randn(2, 1, 8)
    for invalid in ((), (1, 1), (-1,), (3,)):
        try:
            benchmark.partial_endpoint_loss(predictions, targets, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid mask was accepted: {invalid}")


def test_partial_endpoint_loss_rejects_unsliced_hidden_targets() -> None:
    predictions = torch.randn(2, 3, 3, 8)
    full_targets = torch.randn(2, 3, 8)
    try:
        benchmark.partial_endpoint_loss(predictions, full_targets, (1,))
    except ValueError:
        pass
    else:
        raise AssertionError("full hidden-target tensor reached the partial loss")


def test_recursive_finiteness_allows_structural_nulls_only() -> None:
    assert benchmark._all_numeric_values_finite(
        {"metric": 1.0, "classification": None, "nested": [True, 4]}
    )
    assert not benchmark._all_numeric_values_finite({"metric": float("nan")})


def test_center_visibility_threshold_separates_roundoff_from_spinor_gap() -> None:
    assert benchmark.CENTER_VISIBILITY_RMSE_THRESHOLD > 3.9e-8
    assert benchmark.CENTER_VISIBILITY_RMSE_THRESHOLD < 0.7


def test_development_evidence_is_content_locked() -> None:
    artifacts = ROOT / "experiments" / "artifacts"
    expected = {
        "pure_spin8_endpoint_observability_development_seed0.json": (
            "37e8d28eb9a23d5ea831c2530caa9e00d7320dd91db39313d815c534c840f7d2"
        ),
        "pure_spin8_endpoint_observability_development_seed0_validated.json": (
            "4f9933c4aa4e748bf1c1a1ddfd257db4b4e6e9f38abbe1e441efc294d5982b95"
        ),
    }
    for name, digest in expected.items():
        assert hashlib.sha256((artifacts / name).read_bytes()).hexdigest() == digest


def test_protocol_freeze_is_recorded_in_runner() -> None:
    assert benchmark.PROTOCOL_FROZEN_AT == "2026-08-17T06:14:38.9940486+02:00"


def test_fresh_observability_artifacts_are_content_locked() -> None:
    artifacts = ROOT / "experiments" / "artifacts"
    expected = {
        "pure_spin8_endpoint_observability_validation_seed1.json": (
            "46ca4c9a3fb66657f99312d4b4adc4ee7ab40bb89bdb40ca20796eb1aa960bca"
        ),
        "pure_spin8_endpoint_observability_validation_seed2.json": (
            "c3591e2a7c98eef45d6dfd053e0d19e900b71bdf3d63149588504444867410e2"
        ),
        "pure_spin8_endpoint_observability_validation_seed3.json": (
            "a1ddab19a2048e2a7b864ce467307be1b93c836d2bb968b3b144ab20018b51cf"
        ),
        "pure_spin8_endpoint_observability_validation_seeds1_3.json": (
            "baed378d569391e86c46df731cfc72db4f0c0a24d21883bb17a4604db9e5c987"
        ),
    }
    for name, digest in expected.items():
        assert hashlib.sha256((artifacts / name).read_bytes()).hexdigest() == digest


def test_failed_frozen_cohort_is_preserved_without_median_rescue() -> None:
    path = (
        ROOT
        / "experiments"
        / "artifacts"
        / "pure_spin8_endpoint_observability_validation_seeds1_3.json"
    )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert not artifact["passed"]
    assert not artifact["global_checks"][
        "every_frozen_gate_passes_without_median_rescue"
    ]
    expected_failures = {
        "1": {
            "vector_only.shared_all_spinor_center_rows_exact",
            "positive_only.independent_supervised_l128_capable",
            "positive_only.independent_visible_supervised_center",
        },
        "2": {"vector_only.shared_all_spinor_center_rows_exact"},
        "3": {"vector_only.shared_all_spinor_center_rows_exact"},
    }
    for seed, gates in artifact["frozen_seed_gates"].items():
        failures = {name for name, passed in gates.items() if not passed}
        assert failures == expected_failures[seed]
    assert all(
        seed["checks"]["all_checkpoints_strictly_reloaded"]
        for seed in artifact["per_seed"]
    )
