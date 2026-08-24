"""Fail-closed gates for causal retrieval task generation and scoring."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasks import (
    DEFAULT_VOCABULARY,
    RetrievalBatch,
    RetrievalTaskSchema,
    RetrievalVocabulary,
    extrapolation_lengths,
    gather_query_logits,
    generate_mqar_batch,
    generate_needle_batch,
    generate_overwrite_batch,
    generate_selective_copy_batch,
    generate_train_test_cohort,
    retrieval_accuracy,
    retrieval_loss,
    train_short_test_long_lengths,
)


def _assert_no_scored_answer_leakage(batch: RetrievalBatch) -> None:
    scored_inputs = batch.inputs.gather(1, batch.query_positions)
    assert not bool((scored_inputs == batch.targets).any())


def _assert_batch_equal(left: RetrievalBatch, right: RetrievalBatch) -> None:
    assert left.schema == right.schema
    assert torch.equal(left.inputs, right.inputs)
    assert torch.equal(left.targets, right.targets)
    assert torch.equal(left.query_positions, right.query_positions)


def test_vocabulary_and_schema_are_disjoint_and_auditable() -> None:
    vocab = DEFAULT_VOCABULARY
    marker_tokens = {
        vocab.pad_token,
        vocab.write_token,
        vocab.query_token,
        vocab.select_token,
        vocab.item_token,
    }
    ranged_tokens = set(range(vocab.key_start, vocab.key_stop))
    ranged_tokens.update(range(vocab.value_start, vocab.value_stop))
    ranged_tokens.update(range(vocab.filler_start, vocab.filler_stop))
    assert marker_tokens.isdisjoint(ranged_tokens)
    schema = RetrievalTaskSchema("task", "layout", "target", "query", vocab)
    assert schema.vocabulary.vocab_size == vocab.vocab_size


@pytest.mark.parametrize(
    "kwargs",
    [
        {"write_token": 0},
        {"key_start": 4},
        {"value_start": 60},
        {"filler_count": 0},
    ],
)
def test_invalid_vocabulary_is_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        RetrievalVocabulary(**kwargs)


def test_mqar_layout_unique_keys_queries_and_no_leakage() -> None:
    batch = generate_mqar_batch(8, num_pairs=7, num_queries=4, length=48, seed=11)
    vocab = batch.schema.vocabulary
    keys = batch.metadata["stored_keys"]
    values = batch.metadata["stored_values"]
    query_indices = batch.metadata["query_pair_indices"]
    assert isinstance(keys, torch.Tensor)
    assert isinstance(values, torch.Tensor)
    assert isinstance(query_indices, torch.Tensor)
    assert batch.inputs.shape == (8, 48)
    assert batch.targets.shape == (8, 4)
    assert batch.query_positions.shape == (8, 4)
    assert bool(
        (keys.sort(dim=1).values[:, 1:] != keys.sort(dim=1).values[:, :-1]).all()
    )
    assert bool(
        (
            query_indices.sort(dim=1).values[:, 1:]
            != query_indices.sort(dim=1).values[:, :-1]
        ).all()
    )
    assert bool((batch.inputs[:, 0:21:3] == vocab.write_token).all())
    assert torch.equal(batch.inputs[:, 1:21:3], keys)
    assert torch.equal(batch.inputs[:, 2:21:3], values)
    assert torch.equal(batch.targets, values.gather(1, query_indices))
    assert torch.equal(
        batch.inputs.gather(1, batch.query_positions), keys.gather(1, query_indices)
    )
    filler = batch.inputs[:, 21:40]
    assert bool((filler >= vocab.filler_start).all())
    assert bool((filler < vocab.filler_stop).all())
    _assert_no_scored_answer_leakage(batch)


def test_mqar_seed_and_generator_are_deterministic_and_local() -> None:
    kwargs = {"batch_size": 4, "num_pairs": 5, "num_queries": 3, "length": 32}
    first = generate_mqar_batch(**kwargs, seed=37)
    second = generate_mqar_batch(**kwargs, seed=37)
    _assert_batch_equal(first, second)
    seeded_generator = generate_mqar_batch(
        **kwargs, generator=torch.Generator().manual_seed(37)
    )
    _assert_batch_equal(first, seeded_generator)

    torch.manual_seed(812)
    state = torch.random.get_rng_state().clone()
    generate_mqar_batch(**kwargs, seed=9)
    assert torch.equal(torch.random.get_rng_state(), state)


def test_overwrite_targets_are_the_latest_writes_for_multiple_keys() -> None:
    batch = generate_overwrite_batch(
        12, writes=9, length=42, num_keys=3, num_queries=2, seed=19
    )
    keys = batch.metadata["write_keys"]
    values = batch.metadata["write_values"]
    query_keys = batch.metadata["query_keys"]
    latest = batch.metadata["latest_write_indices"]
    assert isinstance(keys, torch.Tensor)
    assert isinstance(values, torch.Tensor)
    assert isinstance(query_keys, torch.Tensor)
    assert isinstance(latest, torch.Tensor)
    rows = torch.arange(batch.inputs.shape[0])
    for row in range(batch.inputs.shape[0]):
        assert len(set(keys[row, :3].tolist())) == 3
        for query in range(2):
            matches = (keys[row] == query_keys[row, query]).nonzero(as_tuple=True)[0]
            assert matches.numel() >= 2
            assert int(latest[row, query]) == int(matches[-1])
            assert batch.targets[row, query] == values[row, matches[-1]]
    assert torch.equal(batch.targets, values[rows[:, None], latest])
    _assert_no_scored_answer_leakage(batch)


@pytest.mark.parametrize("distance", [2, 7, 23, 37])
def test_needle_metadata_distance_is_exact(distance: int) -> None:
    batch = generate_needle_batch(
        6, length=40, needle_distance=distance, seed=100 + distance
    )
    vocab = batch.schema.vocabulary
    stored_values = batch.metadata["stored_value_positions"]
    stored_keys = batch.metadata["stored_key_positions"]
    write_positions = batch.metadata["write_positions"]
    needle_keys = batch.metadata["needle_keys"]
    assert isinstance(stored_values, torch.Tensor)
    assert isinstance(stored_keys, torch.Tensor)
    assert isinstance(write_positions, torch.Tensor)
    assert isinstance(needle_keys, torch.Tensor)
    assert batch.metadata["distance"] == distance
    assert bool((batch.query_positions - stored_values == distance).all())
    assert torch.equal(stored_keys + 1, stored_values)
    assert torch.equal(write_positions + 2, stored_values)
    assert bool((batch.inputs.gather(1, write_positions) == vocab.write_token).all())
    assert torch.equal(batch.inputs.gather(1, stored_keys), needle_keys)
    assert torch.equal(batch.inputs.gather(1, stored_values), batch.targets)
    assert torch.equal(batch.inputs.gather(1, batch.query_positions), needle_keys)
    assert bool((batch.inputs == needle_keys).sum(dim=1).eq(2).all())
    _assert_no_scored_answer_leakage(batch)


def test_selective_copy_reproduces_selected_items_in_source_order() -> None:
    batch = generate_selective_copy_batch(
        10, num_items=9, num_selected=4, length=34, seed=29
    )
    vocab = batch.schema.vocabulary
    items = batch.metadata["items"]
    selected = batch.metadata["selected_indices"]
    selection_mask = batch.metadata["selection_mask"]
    assert isinstance(items, torch.Tensor)
    assert isinstance(selected, torch.Tensor)
    assert isinstance(selection_mask, torch.Tensor)
    assert bool((selected[:, 1:] > selected[:, :-1]).all())
    assert bool(selection_mask.sum(dim=1).eq(4).all())
    assert torch.equal(batch.targets, items.gather(1, selected))
    assert bool((batch.inputs.gather(1, 2 * selected) == vocab.select_token).all())
    assert bool(
        (batch.inputs.gather(1, batch.query_positions) == vocab.query_token).all()
    )
    assert batch.targets.shape == (10, 4)
    _assert_no_scored_answer_leakage(batch)


def test_all_tasks_keep_answers_separate_from_scored_positions() -> None:
    batches = (
        generate_mqar_batch(3, 4, 2, 28, seed=1),
        generate_overwrite_batch(3, writes=6, length=28, seed=2),
        generate_needle_batch(3, length=28, needle_distance=10, seed=3),
        generate_selective_copy_batch(3, 6, 3, 28, seed=4),
    )
    for batch in batches:
        assert batch.targets.ndim == 2
        assert batch.targets.shape == batch.query_positions.shape
        _assert_no_scored_answer_leakage(batch)


def test_train_short_test_long_lengths_and_cohort_are_reproducible() -> None:
    assert extrapolation_lengths(32, (2, 3, 5)) == (64, 96, 160)
    assert train_short_test_long_lengths(32, (2, 4)) == (32, 64, 128)
    kwargs = {"batch_size": 2, "num_pairs": 4, "num_queries": 2}
    first = generate_train_test_cohort(
        generate_mqar_batch, 32, (2, 3), seed=71, **kwargs
    )
    second = generate_train_test_cohort(
        generate_mqar_batch, 32, (2, 3), seed=71, **kwargs
    )
    assert tuple(first) == (32, 64, 96)
    for length in first:
        assert first[length].inputs.shape == (2, length)
        _assert_batch_equal(first[length], second[length])


def test_query_logit_gather_loss_accuracy_and_gradients() -> None:
    batch = generate_mqar_batch(3, 4, 2, 30, seed=43)
    vocab_size = batch.schema.vocabulary.vocab_size + 3
    logits = torch.zeros(3, 30, vocab_size)
    rows = torch.arange(3)[:, None]
    logits[rows, batch.query_positions, batch.targets] = 9.0
    logits.requires_grad_()

    gathered = gather_query_logits(logits, batch)
    expected = logits.gather(
        1, batch.query_positions.unsqueeze(-1).expand(-1, -1, vocab_size)
    )
    assert gathered.shape == (3, 2, vocab_size)
    assert torch.equal(gathered, expected)
    losses = retrieval_loss(logits, batch, reduction="none")
    direct = F.cross_entropy(
        gathered.flatten(0, 1), batch.targets.flatten(), reduction="none"
    ).view_as(batch.targets)
    assert torch.equal(losses, direct)
    loss = retrieval_loss(logits, batch)
    loss.backward()
    assert logits.grad is not None
    assert bool(
        (logits.grad.abs().sum(dim=-1).gather(1, batch.query_positions) > 0).all()
    )
    assert retrieval_accuracy(logits.detach(), batch) == 1.0

    wrong = logits.detach().clone()
    wrong[0, batch.query_positions[0, 0]] = 0.0
    wrong[0, batch.query_positions[0, 0], 0] = 10.0
    assert retrieval_accuracy(wrong, batch) == pytest.approx(5 / 6)
    assert retrieval_loss(logits.detach(), batch, reduction="sum") == pytest.approx(
        float(losses.detach().sum())
    )


@pytest.mark.parametrize(
    "logits",
    [
        torch.zeros(2, 12),
        torch.zeros(1, 12, 256),
        torch.zeros(2, 11, 256),
        torch.zeros(2, 12, 2),
        torch.zeros(2, 12, 256, dtype=torch.long),
    ],
)
def test_logit_shape_and_dtype_guards(logits: torch.Tensor) -> None:
    batch = generate_mqar_batch(2, 2, 1, 12, seed=5)
    with pytest.raises(ValueError):
        gather_query_logits(logits, batch)


def test_nonfinite_logits_and_invalid_reduction_are_rejected() -> None:
    batch = generate_mqar_batch(2, 2, 1, 12, seed=5)
    logits = torch.zeros(2, 12, 256)
    logits[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        retrieval_accuracy(logits, batch)
    with pytest.raises(ValueError, match="reduction"):
        retrieval_loss(torch.zeros_like(logits), batch, reduction="median")  # type: ignore[arg-type]


def test_retrieval_batch_shape_position_device_and_leakage_guards() -> None:
    batch = generate_mqar_batch(2, 3, 2, 20, seed=17)
    with pytest.raises(ValueError, match="same shape"):
        replace(batch, query_positions=batch.query_positions[:, :1])
    with pytest.raises(ValueError, match="out-of-range"):
        replace(batch, query_positions=torch.full_like(batch.query_positions, 20))
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(batch, query_positions=batch.query_positions.flip(1))
    leaking_targets = batch.inputs.gather(1, batch.query_positions)
    with pytest.raises(ValueError, match="answer token"):
        replace(batch, targets=leaking_targets)
    with pytest.raises(ValueError, match="dtype"):
        replace(batch, targets=batch.targets.float())

    moved = batch.to("cpu")
    assert moved.inputs.device.type == "cpu"
    assert all(
        value.device.type == "cpu"
        for value in moved.metadata.values()
        if isinstance(value, torch.Tensor)
    )
    if torch.cuda.is_available():
        cuda_batch = batch.to("cuda")
        with pytest.raises(ValueError, match="same device"):
            gather_query_logits(torch.zeros(2, 20, 256), cuda_batch)


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (
            generate_mqar_batch,
            {"batch_size": 0, "num_pairs": 2, "num_queries": 1, "length": 12},
        ),
        (
            generate_mqar_batch,
            {"batch_size": 2, "num_pairs": 2, "num_queries": 3, "length": 20},
        ),
        (
            generate_mqar_batch,
            {"batch_size": 2, "num_pairs": 3, "num_queries": 2, "length": 12},
        ),
        (generate_overwrite_batch, {"batch_size": 2, "writes": 4, "num_keys": 1}),
        (generate_overwrite_batch, {"batch_size": 2, "writes": 2, "num_keys": 2}),
        (
            generate_overwrite_batch,
            {
                "batch_size": 2,
                "writes": 5,
                "num_keys": 3,
                "num_queries": 2,
                "length": 10,
            },
        ),
        (generate_needle_batch, {"batch_size": 2, "length": 8, "needle_distance": 1}),
        (generate_needle_batch, {"batch_size": 2, "length": 8, "needle_distance": 6}),
        (
            generate_selective_copy_batch,
            {"batch_size": 2, "num_items": 3, "num_selected": 4, "length": 20},
        ),
        (
            generate_selective_copy_batch,
            {"batch_size": 2, "num_items": 4, "num_selected": 2, "length": 9},
        ),
    ],
)
def test_invalid_task_arguments_are_rejected(
    factory: object, kwargs: dict[str, int]
) -> None:
    assert callable(factory)
    with pytest.raises(ValueError):
        factory(**kwargs)  # type: ignore[operator]


def test_seed_generator_and_capacity_guards() -> None:
    generator = torch.Generator().manual_seed(1)
    with pytest.raises(ValueError, match="either seed or generator"):
        generate_mqar_batch(2, 2, 1, 12, seed=1, generator=generator)
    with pytest.raises(ValueError, match="unique key"):
        generate_mqar_batch(2, 5, 1, 20, key_vocab=4)
    with pytest.raises(ValueError, match="unique item"):
        generate_selective_copy_batch(2, 5, 2, 20, value_vocab=4)
    with pytest.raises(ValueError, match="strictly increasing"):
        extrapolation_lengths(16, (4, 2))
    with pytest.raises(ValueError):
        extrapolation_lengths(16, (1,))
    with pytest.raises(ValueError, match="controlled"):
        generate_train_test_cohort(
            generate_mqar_batch,
            16,
            (2,),
            seed=1,
            batch_size=2,
            num_pairs=2,
            num_queries=1,
            generator=generator,
        )
