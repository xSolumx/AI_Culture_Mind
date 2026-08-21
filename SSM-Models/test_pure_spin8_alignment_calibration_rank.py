from __future__ import annotations

import hashlib
import json
from pathlib import Path

import analyze_pure_spin8_alignment_calibration_rank as analysis
import benchmark_pure_spin8_alignment_calibration_rank as benchmark
import benchmark_pure_spin8_continuous_observation as continuous
import torch

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "experiments" / "artifacts"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frame_orbit_rank_profile_is_exact() -> None:
    actual = tuple(analysis.frame_orbit_rank(count) for count in range(9))
    assert actual == analysis.EXPECTED_PROFILE
    assert actual[7] == actual[8] == 28


def test_exact_stabilizer_witnesses_make_the_threshold_global() -> None:
    for probe_count in range(7):
        witness = analysis.exact_stabilizer_witness(probe_count)
        assert witness is not None
        assert witness["passed"]
        assert witness["entrywise_rmse_from_identity"] == "1/4"
    assert analysis.exact_stabilizer_witness(7) is None
    assert analysis.exact_stabilizer_witness(8) is None


def test_exact_and_implemented_chart_profiles_agree() -> None:
    certificate = analysis.build_certificate(seed=0)
    assert certificate["passed"]
    assert all(
        tuple(profile) == analysis.EXPECTED_PROFILE
        for profile in certificate["exact_rational_profiles"].values()
    )
    assert (
        tuple(certificate["implemented_frozen_initial_chart"]["ranks"])
        == analysis.EXPECTED_PROFILE
    )


def test_zero_alignment_has_zero_probe_loss() -> None:
    model = benchmark.NegativeOnlyScrambledSpin8Tracker(alignment_std=0.0)
    for probe_count in range(9):
        loss, components = benchmark.negative_alignment_calibration_loss(
            model, probe_count
        )
        assert float(loss.detach()) == 0.0
        assert float(components["negative_anchor_mse"].detach()) == 0.0


def test_calibration_gradient_is_negative_specific() -> None:
    audit = benchmark.calibration_gradient_audit(seed=0)
    assert audit["passed"]
    assert audit["rows"]["0"]["negative_alignment_gradient_l2"] == 0.0
    assert audit["rows"]["1"]["negative_alignment_gradient_l2"] > 0.0
    assert audit["rows"]["7"]["expected_jacobian_rank"] == 28


def test_observed_views_and_router_initialization_are_exactly_matched() -> None:
    audit = benchmark.matched_initialization_audit(seed=0)
    assert audit["passed"]
    assert audit["parameter_counts"]["negative_only_scrambled"] == 958


def test_tiny_rank_curve_runs_without_negative_endpoint_targets() -> None:
    config = continuous.ContinuousObservationConfig(
        seed=0,
        steps=8,
        batch_size=32,
        training_length=16,
        evaluation_pairs=2,
        evaluation_lengths=(16,),
        evaluation_microbatch_size=2,
    )
    result = benchmark.run_benchmark(
        config,
        device=torch.device("cpu"),
        anchor_counts=(0, 7),
        checkpoint_directory=None,
    )
    assert result["passed"]
    assert not result["task"]["negative_endpoint_targets_transferred"]
    assert result["results"]["scrambled_anchor_curve"]["0"][
        "exact_independent_rank"
    ] == 0
    assert result["results"]["scrambled_anchor_curve"]["7"][
        "exact_independent_rank"
    ] == 28
    assert result["integrity"]["router_trajectory_bitwise_identical_at_every_rank"]


def test_frozen_protocol_and_adjudicated_artifacts_are_content_locked() -> None:
    assert benchmark.PROTOCOL_FROZEN_AT == "2026-08-17T09:52:34.5014289+02:00"
    expected_hashes = {
        ROOT / "experiments" / "PURE_SPIN8_ALIGNMENT_CALIBRATION_RANK_PROTOCOL.md":
            "ba49990666941399e81fba32848e8ff08906c2448480aebb75959a6a2632d2eb",
        ARTIFACTS / "pure_spin8_alignment_calibration_rank_certificate.json":
            "0a1c6ea0107aa732a0656bbb739b1e1eab650eabf40d64ad6229f42810255124",
        ARTIFACTS / "pure_spin8_alignment_calibration_rank_development_seed0.json":
            "681e75597acfe62f8e0458b308a89b3b4e481274a1213c7410bbf274d8d76f66",
        ARTIFACTS
        / "pure_spin8_alignment_calibration_rank_development_seed0_validated.json":
            "6b614d0dd019aefac2cf69f9618235395049e1b57f4eee9a66f14abb0b667f5f",
        ARTIFACTS / "pure_spin8_alignment_calibration_rank_validation_seed10.json":
            "81091d3cb8a0e1508e74b7136558131ae5efd8b699c25da4d978dc6d40871eb3",
        ARTIFACTS / "pure_spin8_alignment_calibration_rank_validation_seed11.json":
            "9b54e83706c445a36b144cea97727d04758d3dae2c389548a138c22b037db05c",
        ARTIFACTS / "pure_spin8_alignment_calibration_rank_validation_seed12.json":
            "d694e425828184f9ce57c6894f7daf953a3f3b4370e724e6cda31561e8d9928e",
        ARTIFACTS
        / "pure_spin8_alignment_calibration_rank_validation_seeds10_12.json":
            "1708d2932cce32f0b1715c1563af35686aa096e7020fe0cbd80ed7f67a2bad2a",
    }
    assert {path: _sha256(path) for path in expected_hashes} == expected_hashes

    aggregate_path = (
        ARTIFACTS
        / "pure_spin8_alignment_calibration_rank_validation_seeds10_12.json"
    )
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert aggregate["seeds"] == [10, 11, 12]
    assert not aggregate["passed"]
    assert aggregate["global_checks"]["all_sources_pass_integrity"]
    assert not aggregate["global_checks"][
        "every_frozen_gate_passes_without_median_rescue"
    ]
    assert not aggregate["frozen_seed_gates"]["10"][
        "full_rank_beats_rank27_negative_action_by_factor"
    ]
    assert not aggregate["frozen_seed_gates"]["10"][
        "full_rank_beats_rank27_every_negative_l128_split"
    ]
    assert all(aggregate["frozen_seed_gates"]["11"].values())
    assert all(aggregate["frozen_seed_gates"]["12"].values())
