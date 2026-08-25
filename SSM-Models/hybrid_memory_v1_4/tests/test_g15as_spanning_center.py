"""Structural contracts for the frozen G15A-S spanning/center cohort."""

from __future__ import annotations

import math

import torch
from spin8_triality import SPIN8_PAIRS

from hybrid_memory_v1_4.g15a_spin_dirac_cohort import _oracle_memory
from hybrid_memory_v1_4.g15as_spanning_center import (
    ACTION_VOCABULARY,
    PAIR_COUNT,
    STEP_ANGLE,
    SpanningCoordinateController,
    _coordinate_metrics,
    _frame_prediction,
    _oracle_certificate,
    _pool_certificate,
    _pool_hash,
    _probe_pool,
    _semantic_coordinate,
    _structured_batch,
    _structured_words,
    _teacher_target,
    _token_map,
    generate_batch,
)


def test_hidden_dictionary_covers_every_signed_plane_once() -> None:
    mapping = _token_map(2281)
    assert sorted(mapping.tolist()) == list(range(1, ACTION_VOCABULARY + 1))
    coordinates = torch.stack(
        [
            _semantic_coordinate(index, dtype=torch.float64)[0]
            for index in range(ACTION_VOCABULARY)
        ]
    )
    assert coordinates.shape == (2 * PAIR_COUNT, PAIR_COUNT)
    assert torch.equal((coordinates != 0).sum(-1), torch.ones(2 * PAIR_COUNT))
    torch.testing.assert_close(
        coordinates[:PAIR_COUNT],
        -coordinates[PAIR_COUNT:],
        rtol=0.0,
        atol=0.0,
    )
    assert math.isclose(float(coordinates.abs().max()), math.pi / 16)
    assert STEP_ANGLE < 0.25


def test_controller_starts_at_masked_identity() -> None:
    controller = SpanningCoordinateController()
    token_ids = torch.arange(ACTION_VOCABULARY + 1).unsqueeze(0)
    output = controller(token_ids)
    assert output.shape == (1, ACTION_VOCABULARY + 1, 1, PAIR_COUNT)
    assert torch.equal(output, torch.zeros_like(output))


def test_coordinate_assay_preserves_semantic_row_and_plane_axes() -> None:
    metrics = _coordinate_metrics(SpanningCoordinateController(), 2281)
    assert math.isclose(
        metrics["maximum_active_coordinate_abs_error"],
        STEP_ANGLE,
        rel_tol=1e-6,
        abs_tol=1e-8,
    )
    assert metrics["inactive_coordinate_rms"] == 0.0


def test_probe_pools_are_disjoint_and_structurally_observable() -> None:
    training = _probe_pool(2281, "training", 2)
    evaluation = _probe_pool(2281, "evaluation", 2)
    assert _pool_hash(training) != _pool_hash(evaluation)
    training_certificate = _pool_certificate(training)
    evaluation_certificate = _pool_certificate(evaluation)
    assert training_certificate["all_passed"]
    assert evaluation_certificate["all_passed"]
    assert training_certificate["minimum_rank"] == 56
    assert training_certificate["minimum_condition_ratio"] >= 0.10
    assert training_certificate["minimum_broken_projection_residual"] >= 0.05


def test_structured_words_cover_center_and_projective_witness() -> None:
    words = _structured_words()
    assert len(words) == 132
    centers = {word["center"] for word in words if word["center"] is not None}
    assert centers == {"identity", "minus_one", "omega", "minus_omega"}
    assert all(
        len(word["semantics"]) == 32
        for word in words
        if word["name"].startswith("two_pi")
    )
    assert all(
        len(word["semantics"]) == 64
        for word in words
        if word["name"].startswith("four_pi")
    )
    certificate = _oracle_certificate(2281, _probe_pool(2281, "evaluation", 4))
    assert certificate["passed"]
    assert certificate["maximum_analytic_center_sign_residual"] <= 1e-10
    assert certificate["projective_center_witnesses"]


def test_exact_controller_replays_teacher_on_unseen_frames() -> None:
    seed = 2281
    pool = _probe_pool(seed, "evaluation", 4)
    batch = generate_batch(
        3,
        16,
        seed=99,
        model_seed=seed,
        minimum_actions=2,
        maximum_actions=6,
        probe_pool=pool,
    ).to(torch.device("cpu"), torch.float64)
    teacher = _oracle_memory("S", dtype=torch.float64, device=torch.device("cpu"))
    expected = _teacher_target(teacher, batch, device=torch.device("cpu"))
    actual = _frame_prediction(
        teacher,
        batch,
        batch.exact_coordinates,
        device=torch.device("cpu"),
    )
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value, rtol=0.0, atol=0.0)


def test_structured_batch_uses_every_spin_plane() -> None:
    batch, words = _structured_batch(
        2281, _probe_pool(2281, "evaluation", 4), dtype=torch.float64
    )
    assert batch.token_ids.shape[0] == len(words)
    loop_pairs = {
        tuple(word["pair"])
        for word in words
        if word["name"].startswith("two_pi_positive")
    }
    assert loop_pairs == set(SPIN8_PAIRS)
