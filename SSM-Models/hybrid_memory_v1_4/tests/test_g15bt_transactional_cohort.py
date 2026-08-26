"""Frozen schedule, objective, intervention, and preflight tests for G15B-T."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hybrid_memory_v1_4.g15b_interleaved_cohort import _stable_seed
from hybrid_memory_v1_4.g15b_interleaved_tasks import generate_interleaved_batch
from hybrid_memory_v1_4.g15bt_transactional_cohort import (
    ARMS,
    _batch_for_update,
    _evaluate_cell,
    _evaluation_batch,
    _optimizer_partition,
    build_model,
    commissioned_losses,
    commit_event_mask,
    commit_positions,
    frozen_config,
    generate_boundary_batch,
    run_preflight,
    transactional_intervention_forward,
    valid_commit_mask,
)


def test_frozen_quality_and_smoke_schedules_are_exact() -> None:
    quality = frozen_config("quality")
    assert quality.seeds == (2381, 2383, 2389)
    assert [phase.length for phase in quality.phases] == [128, 256, 512, 1024]
    assert [phase.updates for phase in quality.phases] == [1000, 1200, 800, 400]
    assert sum(phase.updates for phase in quality.phases) == 3400
    assert quality.evaluation_decisions == 2048
    assert quality.intervention_decisions == 512

    smoke = frozen_config("smoke")
    assert smoke.seeds == (23,)
    assert [phase.updates for phase in smoke.phases] == [1, 1, 1, 1]
    assert smoke.evaluation_decisions == 16


def test_all_arms_are_exactly_matched_and_commit_is_control_optimized() -> None:
    models = {}
    for arm in ARMS:
        models[arm] = build_model(arm, 23, torch.device("cpu"))
    assert (
        len({sum(p.numel() for p in model.parameters()) for model in models.values()})
        == 1
    )
    assert (
        len({model.state_capacity_bytes(1, torch.float32) for model in models.values()})
        == 1
    )
    shapes = [
        {name: tuple(parameter.shape) for name, parameter in model.named_parameters()}
        for model in models.values()
    ]
    assert shapes[0] == shapes[1] == shapes[2]
    assert models["F"].config.transactional_controller_mode == "full"
    assert models["T"].config.transactional_controller_mode == "history"
    assert models["T-AUX"].config.transactional_controller_mode == "history"
    partition = _optimizer_partition(models["T"])
    assert partition["passed"] is True
    assert partition["commit_control_names"]


def test_commit_target_is_first_post_value_position_only() -> None:
    batch = generate_interleaved_batch("overwrite", 3, 128, 4, 8, 4, seed=2381)
    positions = commit_positions(batch)
    target = commit_event_mask(batch)
    assert torch.equal(positions, batch.write_positions + 1)
    assert torch.equal(
        target.gather(1, positions), torch.ones_like(positions, dtype=torch.bool)
    )
    assert torch.count_nonzero(target & batch.write_event_mask) == 0
    assert int(target.sum()) == batch.write_positions.numel()

    terminal = generate_interleaved_batch("mqar", 2, 128, 8, 24, 8, seed=68)
    terminal_positions = commit_positions(terminal)
    terminal_valid = valid_commit_mask(terminal)
    assert bool((terminal_positions == terminal.length).any())
    terminal_target = commit_event_mask(terminal)
    assert int(terminal_target.sum()) == int(terminal_valid.sum())
    assert int(terminal_target.sum()) < terminal.write_positions.numel()


def test_primary_and_auxiliary_losses_have_only_declared_components() -> None:
    torch.manual_seed(2382)
    batch = generate_interleaved_batch("overwrite", 2, 128, 4, 8, 4, seed=2382)
    model = build_model("T", 2382, torch.device("cpu"))
    output = model(batch.token_ids, return_diagnostics=True)
    primary, components = commissioned_losses(output, batch, arm="T")
    assert set(components) == {
        "retrieval",
        "reverse_binding",
        "query_to_commit_address",
    }
    expected = (
        components["retrieval"]
        + 0.25 * components["reverse_binding"]
        + 0.25 * components["query_to_commit_address"]
    )
    torch.testing.assert_close(primary, expected)

    auxiliary, auxiliary_components = commissioned_losses(output, batch, arm="T-AUX")
    assert set(auxiliary_components) == set(components) | {"balanced_commit"}
    torch.testing.assert_close(
        auxiliary, primary + 0.25 * auxiliary_components["balanced_commit"]
    )


def test_reconstructed_forward_and_each_intervention_are_structurally_distinct() -> (
    None
):
    torch.manual_seed(2383)
    model = build_model("T", 2383, torch.device("cpu")).eval()
    batch = generate_interleaved_batch("overwrite", 2, 128, 4, 8, 4, seed=2383)
    ordinary = model(batch.token_ids)["logits"]
    learned = transactional_intervention_forward(
        model, batch.token_ids, "learned_reconstruction"
    )
    torch.testing.assert_close(learned["logits"], ordinary, rtol=1e-5, atol=1e-6)
    assert torch.equal(learned["logits"].argmax(-1), ordinary.argmax(-1))

    commit_zero = transactional_intervention_forward(
        model, batch.token_ids, "commit_zero"
    )
    erase_zero = transactional_intervention_forward(
        model, batch.token_ids, "erase_zero"
    )
    minus = transactional_intervention_forward(
        model, batch.token_ids, "commit_shift_minus_one"
    )
    plus = transactional_intervention_forward(
        model, batch.token_ids, "commit_shift_plus_one"
    )
    permuted = transactional_intervention_forward(
        model, batch.token_ids, "permuted_history"
    )
    assert torch.count_nonzero(commit_zero["controls"]["commit"]) == 0
    assert torch.count_nonzero(erase_zero["controls"]["erase"]) == 0
    assert torch.equal(
        minus["controls"]["commit"][:, :-1], learned["controls"]["commit"][:, 1:]
    )
    assert torch.equal(
        plus["controls"]["commit"][:, 1:], learned["controls"]["commit"][:, :-1]
    )
    assert (
        torch.count_nonzero(permuted["controls"]["key"] - learned["controls"]["key"])
        > 0
    )


def test_paired_batch_schedule_is_arm_independent_and_fresh_evaluation_differs() -> (
    None
):
    phase = frozen_config("smoke").phases[0]
    first = _batch_for_update(phase, seed=23, global_update=0)
    replay = _batch_for_update(phase, seed=23, global_update=0)
    evaluation = _evaluation_batch("mqar", batch_size=2, length=128, seed=2384)
    assert first.fingerprint() == replay.fingerprint()
    assert first.fingerprint() != evaluation.fingerprint()


def test_boundary_batch_has_zero_gaps_without_local_answer_leakage() -> None:
    batch = generate_boundary_batch(3, 128, seed=2384)
    commits = commit_positions(batch)
    # The same-key overwrite starts exactly at the prior write's commit.
    assert torch.equal(commits[:, 1], batch.write_positions[:, 2] - 2)
    # The immediate post-write query asks an unrelated, older key.
    immediate_query = batch.query_positions[:, 1]
    latest_write = batch.write_positions[:, 3]
    assert torch.equal(immediate_query, latest_write + 2)
    assert torch.equal(batch.query_keys[:, 1], batch.write_keys[:, 0])
    assert torch.all(batch.targets[:, 1] != batch.write_values[:, 3])
    # Same-key answers are queried outside the four-token local receptive field.
    assert torch.all(batch.query_positions[:, 0] - batch.write_positions[:, 2] > 4)
    assert torch.all(batch.query_positions[:, 2] - batch.write_positions[:, 3] > 4)


def test_boundary_batch_excludes_random_filler_answer_leaks_across_seeds() -> None:
    seeds = list(range(256)) + [
        _stable_seed("g15bt-boundary", seed) for seed in (2381, 2383, 2389)
    ]
    for seed in seeds:
        batch = generate_boundary_batch(8, 128, seed=seed)
        for row in range(batch.batch_size):
            for query_index in range(batch.queries):
                position = int(batch.query_positions[row, query_index])
                target = batch.targets[row, query_index]
                assert not bool(
                    (batch.token_ids[row, position - 3 : position + 1] == target).any()
                )


def test_cpu_preflight_and_small_evaluation_cell_execute() -> None:
    preflight = run_preflight(torch.device("cpu"))
    assert preflight["passed"] is True
    assert preflight["checks"]["sealed_phase0"] is True
    assert preflight["checks"]["matched_arms"] is True
    assert preflight["checks"]["optimizer_partition"] is True
    model = build_model("T", 2385, torch.device("cpu"))
    cell, hashes = _evaluate_cell(
        model,
        "overwrite",
        128,
        seed=2385,
        decisions=16,
        batch_cap=2,
        namespace="g15bt-test",
        interventions=True,
    )
    assert cell["query_decisions"] == 16
    assert len(hashes) == 1
    assert cell["reconstruction_maximum_absolute_logit_residual"] <= 5e-4
    assert cell["reconstruction_query_predictions_equal"] is True
    assert set(cell["interventions"]) == {
        "learned_reconstruction",
        "commit_zero",
        "memory_zero",
        "permuted_history",
        "commit_shift_minus_one",
        "commit_shift_plus_one",
        "erase_zero",
        "bias_only_history",
    }
