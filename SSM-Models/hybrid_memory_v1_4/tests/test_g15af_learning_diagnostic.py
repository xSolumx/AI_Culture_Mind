"""Contracts for the bound G15A-F chart-error diagnostic."""

from __future__ import annotations

import torch

from hybrid_memory_v1_4.g15a_spin_dirac_cohort import _oracle_memory
from hybrid_memory_v1_4.g15af_full_frame_cohort import (
    _frame_prediction,
    _teacher_target,
    generate_frame_batch,
)
from hybrid_memory_v1_4.g15af_learning_diagnostic import (
    _chart_diagnostics,
    _expected_table,
    _table_variants,
)


def test_expected_table_contains_each_signed_primitive_once() -> None:
    table = _expected_table(2203)
    assert table.shape == (17, 28)
    assert torch.equal(table[0], torch.zeros(28))
    assert int((table != 0).sum()) == 16
    assert torch.allclose(
        table.abs().sum(dim=1)[1:], torch.full((16,), 0.12)
    )


def test_table_variants_separate_support_amplitude_and_leakage() -> None:
    expected = _expected_table(2203)
    raw = torch.zeros_like(expected)
    raw[1:, 0] = 0.1
    variants = _table_variants(raw.tolist(), 2203)
    support = expected != 0
    assert torch.equal(
        variants["active_only_learned_amplitude"][~support],
        torch.zeros_like(expected[~support]),
    )
    assert torch.equal(
        variants["exact_amplitude_with_learned_leakage"][support],
        expected[support],
    )
    assert torch.equal(variants["oracle_exact"], expected)
    assert _chart_diagnostics(variants)["inactive_coordinate_rms"] > 0


def test_oracle_table_exactly_reconstructs_teacher_coordinates_and_frames() -> None:
    device = torch.device("cpu")
    batch = generate_frame_batch(
        4,
        16,
        seed=29,
        model_seed=2203,
        minimum_actions=2,
        maximum_actions=6,
    ).to(device)
    table = _expected_table(2203).to(device)
    reconstructed = table[batch.token_ids].unsqueeze(2)
    assert torch.equal(reconstructed, batch.exact_coordinates)
    memory = _oracle_memory("S", dtype=torch.float32, device=device)
    prediction = _frame_prediction(memory, batch, reconstructed, device=device)
    target = _teacher_target(memory, batch, device=device)
    assert torch.equal(prediction, target)
