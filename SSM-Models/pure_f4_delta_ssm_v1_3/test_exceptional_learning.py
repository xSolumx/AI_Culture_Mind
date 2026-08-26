from __future__ import annotations

import torch

from .benchmark_exceptional_learning import (
    LearningConfig,
    _novel_generator_indices,
    run,
)


def test_novel_generator_selection_targets_each_new_subspace() -> None:
    for target, predecessor in (
        ("g2", "identity"),
        ("spin7", "g2"),
        ("spin8", "spin7"),
        ("spin9", "spin8"),
        ("f4", "spin9"),
        ("e6", "f4"),
    ):
        indices, actual_predecessor = _novel_generator_indices(target, 3)
        assert actual_predecessor == predecessor
        assert len(indices) == 3
        assert len(set(indices)) == 3


def test_small_same_rung_learning_smoke_is_finite_and_reaches_gradients() -> None:
    report = run(
        LearningConfig(
            target_algebra="g2",
            candidate_algebra="g2",
            primitive_count=2,
            train_word_length=2,
            train_probes=2,
            batch_size=4,
            steps=4,
            learning_rate=0.03,
            seed=17,
        ),
        torch.device("cpu"),
    )
    losses = torch.tensor(list(report["sampled_training_losses"].values()))
    assert torch.isfinite(losses).all()
    assert float(report["maximum_preclip_gradient_norm"]) > 0.0
    assert len(report["evaluations"]) == 4
