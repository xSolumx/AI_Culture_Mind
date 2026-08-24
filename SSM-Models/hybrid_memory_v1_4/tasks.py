"""Causal synthetic retrieval tasks with explicit, leakage-free scoring.

Every target is scored from the model output at a query token already present
in ``inputs``. Answers are never inserted at those positions. This makes the
task contract valid for causal sequence models without shifting or masking
ambiguity.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Literal

import torch
from torch.nn import functional as F


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(name: str, value: int) -> int:
    if not _is_int(value) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_seed(seed: int) -> int:
    if not _is_int(seed) or not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must be an integer in [0, 2**63 - 1]")
    return seed


@dataclass(frozen=True)
class RetrievalVocabulary:
    """Disjoint marker, key, value, and filler token ranges."""

    pad_token: int = 0
    write_token: int = 1
    query_token: int = 2
    select_token: int = 3
    item_token: int = 4
    key_start: int = 5
    key_count: int = 64
    value_start: int = 69
    value_count: int = 64
    filler_start: int = 133
    filler_count: int = 64

    def __post_init__(self) -> None:
        marker_values = (
            self.pad_token,
            self.write_token,
            self.query_token,
            self.select_token,
            self.item_token,
        )
        if any(not _is_int(token) or token < 0 for token in marker_values):
            raise ValueError("marker tokens must be nonnegative integers")
        if len(set(marker_values)) != len(marker_values):
            raise ValueError("marker tokens must be unique")
        for name, start, count in self.ranges:
            if not _is_int(start) or start < 0:
                raise ValueError(f"{name}_start must be a nonnegative integer")
            _positive_int(f"{name}_count", count)
        for marker in marker_values:
            if any(start <= marker < stop for _, start, stop in self.intervals):
                raise ValueError("marker tokens must be disjoint from token ranges")
        for index, (left_name, left_start, left_stop) in enumerate(self.intervals):
            for right_name, right_start, right_stop in self.intervals[index + 1 :]:
                if max(left_start, right_start) < min(left_stop, right_stop):
                    raise ValueError(
                        f"{left_name} and {right_name} token ranges must be disjoint"
                    )

    @property
    def ranges(self) -> tuple[tuple[str, int, int], ...]:
        return (
            ("key", self.key_start, self.key_count),
            ("value", self.value_start, self.value_count),
            ("filler", self.filler_start, self.filler_count),
        )

    @property
    def intervals(self) -> tuple[tuple[str, int, int], ...]:
        return tuple((name, start, start + count) for name, start, count in self.ranges)

    @property
    def key_stop(self) -> int:
        return self.key_start + self.key_count

    @property
    def value_stop(self) -> int:
        return self.value_start + self.value_count

    @property
    def filler_stop(self) -> int:
        return self.filler_start + self.filler_count

    @property
    def vocab_size(self) -> int:
        markers = (
            self.pad_token,
            self.write_token,
            self.query_token,
            self.select_token,
            self.item_token,
        )
        return (
            max(*markers, self.key_stop - 1, self.value_stop - 1, self.filler_stop - 1)
            + 1
        )


DEFAULT_VOCABULARY = RetrievalVocabulary()


@dataclass(frozen=True)
class RetrievalTaskSchema:
    """Auditable task semantics attached to every generated batch."""

    name: str
    layout: str
    target_semantics: str
    scored_position: str
    vocabulary: RetrievalVocabulary = DEFAULT_VOCABULARY

    def __post_init__(self) -> None:
        for name in ("name", "layout", "target_semantics", "scored_position"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(self.vocabulary, RetrievalVocabulary):
            raise TypeError("vocabulary must be a RetrievalVocabulary")


MQAR_SCHEMA = RetrievalTaskSchema(
    name="mqar",
    layout="[WRITE,key,value]*pairs + filler + [QUERY,key]*queries",
    target_semantics="value associated with each queried unique key",
    scored_position="query key",
)
OVERWRITE_SCHEMA = RetrievalTaskSchema(
    name="overwrite_retrieval",
    layout="[WRITE,key,value]*writes + filler + [QUERY,key]*queries",
    target_semantics="latest value written to each queried key",
    scored_position="query key",
)
EXACT_DISTANCE_NEEDLE_SCHEMA = RetrievalTaskSchema(
    name="exact_distance_needle",
    layout="filler with one [WRITE,key,value] needle and trailing [QUERY,key]",
    target_semantics="stored needle value",
    scored_position="query key",
)
SELECTIVE_COPY_SCHEMA = RetrievalTaskSchema(
    name="selective_copy",
    layout="[ITEM|SELECT,item]*items + filler + [QUERY]*selected",
    target_semantics="selected items in source order",
    scored_position="query marker",
)


@dataclass(frozen=True)
class RetrievalBatch:
    """A validated causal retrieval batch.

    ``inputs`` has shape ``(B, L)``. ``targets`` and ``query_positions`` both
    have shape ``(B, Q)``. A logit at ``query_positions[b, q]`` is scored
    against ``targets[b, q]``.
    """

    inputs: torch.Tensor
    targets: torch.Tensor
    query_positions: torch.Tensor
    schema: RetrievalTaskSchema
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.schema, RetrievalTaskSchema):
            raise TypeError("schema must be a RetrievalTaskSchema")
        for name, tensor in (
            ("inputs", self.inputs),
            ("targets", self.targets),
            ("query_positions", self.query_positions),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            if tensor.dtype != torch.long:
                raise ValueError(f"{name} must have dtype torch.long")
        if self.inputs.ndim != 2:
            raise ValueError("inputs must have shape (B,L)")
        if self.targets.ndim != 2:
            raise ValueError("targets must have shape (B,Q)")
        if self.query_positions.shape != self.targets.shape:
            raise ValueError("query_positions must have the same shape as targets")
        batch_size, length = self.inputs.shape
        if batch_size < 1 or length < 1:
            raise ValueError("inputs must have positive batch and sequence dimensions")
        if self.targets.shape[0] != batch_size or self.targets.shape[1] < 1:
            raise ValueError("targets must contain at least one query per input row")
        if not (
            self.inputs.device == self.targets.device == self.query_positions.device
        ):
            raise ValueError("batch tensors must be on the same device")
        if (
            int(self.inputs.min()) < 0
            or int(self.inputs.max()) >= self.schema.vocabulary.vocab_size
        ):
            raise ValueError("inputs contain token ids outside the task vocabulary")
        if (
            int(self.targets.min()) < 0
            or int(self.targets.max()) >= self.schema.vocabulary.vocab_size
        ):
            raise ValueError("targets contain token ids outside the task vocabulary")
        if (
            int(self.query_positions.min()) < 0
            or int(self.query_positions.max()) >= length
        ):
            raise ValueError("query_positions contain an out-of-range position")
        if self.query_positions.shape[1] > 1 and bool(
            (self.query_positions[:, 1:] <= self.query_positions[:, :-1]).any()
        ):
            raise ValueError(
                "query positions must be strictly increasing within each row"
            )
        scored_tokens = self.inputs.gather(1, self.query_positions)
        if bool((scored_tokens == self.targets).any()):
            raise ValueError("an answer token is present at a scored query position")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        copied_metadata = dict(self.metadata)
        for name, value in copied_metadata.items():
            if not isinstance(name, str):
                raise TypeError("metadata keys must be strings")
            if isinstance(value, torch.Tensor) and value.device != self.inputs.device:
                raise ValueError("metadata tensors must share the batch device")
        object.__setattr__(self, "metadata", copied_metadata)

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> RetrievalBatch:
        destination = torch.device(device)
        metadata = {
            name: value.to(destination, non_blocking=non_blocking)
            if isinstance(value, torch.Tensor)
            else value
            for name, value in self.metadata.items()
        }
        return RetrievalBatch(
            inputs=self.inputs.to(destination, non_blocking=non_blocking),
            targets=self.targets.to(destination, non_blocking=non_blocking),
            query_positions=self.query_positions.to(
                destination, non_blocking=non_blocking
            ),
            schema=self.schema,
            metadata=metadata,
        )

    def cpu(self) -> RetrievalBatch:
        return self.to("cpu")


def _schema_with_vocabulary(
    schema: RetrievalTaskSchema, vocabulary: RetrievalVocabulary
) -> RetrievalTaskSchema:
    if schema.vocabulary == vocabulary:
        return schema
    return RetrievalTaskSchema(
        name=schema.name,
        layout=schema.layout,
        target_semantics=schema.target_semantics,
        scored_position=schema.scored_position,
        vocabulary=vocabulary,
    )


def _resolve_vocabulary(
    vocabulary: RetrievalVocabulary | None,
    *,
    key_count: int | None = None,
    value_count: int | None = None,
    filler_count: int | None = None,
) -> RetrievalVocabulary:
    if vocabulary is not None:
        if not isinstance(vocabulary, RetrievalVocabulary):
            raise TypeError("vocabulary must be a RetrievalVocabulary")
        requested = (
            ("key_vocab", key_count, vocabulary.key_count),
            ("value_vocab", value_count, vocabulary.value_count),
            ("filler_vocab", filler_count, vocabulary.filler_count),
        )
        for name, count, actual in requested:
            if count is not None and count != actual:
                raise ValueError(f"{name} conflicts with the supplied vocabulary")
        return vocabulary
    keys = (
        DEFAULT_VOCABULARY.key_count
        if key_count is None
        else _positive_int("key_vocab", key_count)
    )
    values = (
        DEFAULT_VOCABULARY.value_count
        if value_count is None
        else _positive_int("value_vocab", value_count)
    )
    fillers = (
        DEFAULT_VOCABULARY.filler_count
        if filler_count is None
        else _positive_int("filler_vocab", filler_count)
    )
    if (keys, values, fillers) == (
        DEFAULT_VOCABULARY.key_count,
        DEFAULT_VOCABULARY.value_count,
        DEFAULT_VOCABULARY.filler_count,
    ):
        return DEFAULT_VOCABULARY
    key_start = DEFAULT_VOCABULARY.key_start
    value_start = key_start + keys
    filler_start = value_start + values
    return RetrievalVocabulary(
        key_start=key_start,
        key_count=keys,
        value_start=value_start,
        value_count=values,
        filler_start=filler_start,
        filler_count=fillers,
    )


def _resolve_generator(
    seed: int | None, generator: torch.Generator | None
) -> tuple[torch.Generator, int | None]:
    if generator is not None:
        if not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator")
        if seed is not None:
            raise ValueError("pass either seed or generator, not both")
        if torch.device(generator.device).type != "cpu":
            raise ValueError("task generation requires a CPU generator")
        return generator, None
    resolved_seed = 0 if seed is None else _nonnegative_seed(seed)
    return torch.Generator(device="cpu").manual_seed(resolved_seed), resolved_seed


def _sample_without_replacement(
    batch_size: int,
    population: int,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if count > population:
        raise ValueError("sample count exceeds the available unique tokens")
    return torch.stack(
        [
            torch.randperm(population, generator=generator)[:count]
            for _ in range(batch_size)
        ]
    )


def _filler_inputs(
    batch_size: int,
    length: int,
    vocabulary: RetrievalVocabulary,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.randint(
        vocabulary.filler_start,
        vocabulary.filler_stop,
        (batch_size, length),
        generator=generator,
        dtype=torch.long,
    )


def generate_mqar_batch(
    batch_size: int,
    num_pairs: int,
    num_queries: int,
    length: int,
    key_vocab: int | None = None,
    value_vocab: int | None = None,
    seed: int | None = None,
    *,
    generator: torch.Generator | None = None,
    vocabulary: RetrievalVocabulary | None = None,
) -> RetrievalBatch:
    """Generate leakage-free multi-query associative recall.

    Keys and query choices are sampled without replacement independently in
    every row. Filler tokens are from a range disjoint from keys and values.
    """

    batch_size = _positive_int("batch_size", batch_size)
    num_pairs = _positive_int("num_pairs", num_pairs)
    num_queries = _positive_int("num_queries", num_queries)
    length = _positive_int("length", length)
    if num_queries > num_pairs:
        raise ValueError("num_queries cannot exceed num_pairs")
    minimum_length = 3 * num_pairs + 2 * num_queries
    if length < minimum_length:
        raise ValueError("length is too short for the MQAR layout")
    vocab = _resolve_vocabulary(
        vocabulary, key_count=key_vocab, value_count=value_vocab
    )
    if num_pairs > vocab.key_count:
        raise ValueError("num_pairs exceeds the unique key vocabulary")
    rng, recorded_seed = _resolve_generator(seed, generator)

    key_offsets = _sample_without_replacement(
        batch_size, vocab.key_count, num_pairs, rng
    )
    keys = key_offsets + vocab.key_start
    values = torch.randint(
        vocab.value_start,
        vocab.value_stop,
        (batch_size, num_pairs),
        generator=rng,
    )
    query_pair_indices = _sample_without_replacement(
        batch_size, num_pairs, num_queries, rng
    )
    targets = values.gather(1, query_pair_indices)

    inputs = _filler_inputs(batch_size, length, vocab, rng)
    inputs[:, 0 : 3 * num_pairs : 3] = vocab.write_token
    inputs[:, 1 : 3 * num_pairs : 3] = keys
    inputs[:, 2 : 3 * num_pairs : 3] = values
    query_start = length - 2 * num_queries
    query_positions = (
        (query_start + 1 + 2 * torch.arange(num_queries, dtype=torch.long))
        .expand(batch_size, -1)
        .clone()
    )
    inputs[:, query_start:length:2] = vocab.query_token
    inputs.scatter_(1, query_positions, keys.gather(1, query_pair_indices))

    return RetrievalBatch(
        inputs=inputs,
        targets=targets,
        query_positions=query_positions,
        schema=_schema_with_vocabulary(MQAR_SCHEMA, vocab),
        metadata={
            "seed": recorded_seed,
            "num_pairs": num_pairs,
            "num_queries": num_queries,
            "stored_keys": keys,
            "stored_values": values,
            "query_pair_indices": query_pair_indices,
            "stored_key_positions": (1 + 3 * torch.arange(num_pairs, dtype=torch.long))
            .expand(batch_size, -1)
            .clone(),
            "stored_value_positions": (
                2 + 3 * torch.arange(num_pairs, dtype=torch.long)
            )
            .expand(batch_size, -1)
            .clone(),
            "filler_start_position": 3 * num_pairs,
            "filler_stop_position": query_start,
        },
    )


def generate_overwrite_batch(
    batch_size: int,
    writes: int,
    length: int | None = None,
    num_keys: int = 2,
    num_queries: int = 1,
    key_vocab: int | None = None,
    value_vocab: int | None = None,
    filler_vocab: int | None = None,
    seed: int | None = None,
    *,
    generator: torch.Generator | None = None,
    vocabulary: RetrievalVocabulary | None = None,
) -> RetrievalBatch:
    """Generate retrieval after guaranteed overwrites of queried keys."""

    batch_size = _positive_int("batch_size", batch_size)
    writes = _positive_int("writes", writes)
    num_keys = _positive_int("num_keys", num_keys)
    num_queries = _positive_int("num_queries", num_queries)
    if num_keys < 2:
        raise ValueError("overwrite retrieval requires at least two keys")
    if num_queries > num_keys:
        raise ValueError("num_queries cannot exceed num_keys")
    if writes < num_keys + num_queries:
        raise ValueError("writes must cover every key and overwrite every queried key")
    minimum_length = 3 * writes + 2 * num_queries
    if length is None:
        length = minimum_length
    length = _positive_int("length", length)
    if length < minimum_length:
        raise ValueError("length is too short for the overwrite layout")
    vocab = _resolve_vocabulary(
        vocabulary,
        key_count=key_vocab,
        value_count=value_vocab,
        filler_count=filler_vocab,
    )
    if num_keys > vocab.key_count:
        raise ValueError("num_keys exceeds the key vocabulary")
    if vocab.value_count < 2:
        raise ValueError("overwrite retrieval requires at least two values")
    rng, recorded_seed = _resolve_generator(seed, generator)

    key_permutations = _sample_without_replacement(batch_size, num_keys, num_keys, rng)
    key_offsets = torch.randint(num_keys, (batch_size, writes), generator=rng)
    key_offsets[:, :num_keys] = key_permutations
    query_key_offsets = key_permutations[:, :num_queries]
    key_offsets[:, num_keys : num_keys + num_queries] = query_key_offsets
    keys = key_offsets + vocab.key_start
    query_keys = query_key_offsets + vocab.key_start
    values = torch.randint(
        vocab.value_start,
        vocab.value_stop,
        (batch_size, writes),
        generator=rng,
    )
    row_positions = torch.arange(batch_size)
    for position in range(num_keys, writes):
        previous_positions = (
            torch.where(
                keys[:, :position] == keys[:, position : position + 1],
                torch.arange(position),
                -1,
            )
            .max(dim=1)
            .values
        )
        previous_values = values[row_positions, previous_positions]
        collision = values[:, position] == previous_values
        values[collision, position] = vocab.value_start + (
            (values[collision, position] - vocab.value_start + 1) % vocab.value_count
        )

    write_indices = torch.arange(writes).expand(batch_size, -1)
    latest_write_indices = (
        torch.where(
            keys[:, None, :] == query_keys[:, :, None],
            write_indices[:, None, :],
            -1,
        )
        .max(dim=2)
        .values
    )
    targets = values.gather(1, latest_write_indices)

    inputs = _filler_inputs(batch_size, length, vocab, rng)
    inputs[:, 0 : 3 * writes : 3] = vocab.write_token
    inputs[:, 1 : 3 * writes : 3] = keys
    inputs[:, 2 : 3 * writes : 3] = values
    query_start = length - 2 * num_queries
    query_positions = (
        (query_start + 1 + 2 * torch.arange(num_queries, dtype=torch.long))
        .expand(batch_size, -1)
        .clone()
    )
    inputs[:, query_start:length:2] = vocab.query_token
    inputs.scatter_(1, query_positions, query_keys)

    return RetrievalBatch(
        inputs=inputs,
        targets=targets,
        query_positions=query_positions,
        schema=_schema_with_vocabulary(OVERWRITE_SCHEMA, vocab),
        metadata={
            "seed": recorded_seed,
            "writes": writes,
            "num_keys": num_keys,
            "num_queries": num_queries,
            "write_keys": keys,
            "write_values": values,
            "query_keys": query_keys,
            "latest_write_indices": latest_write_indices,
            "stored_key_positions": (1 + 3 * torch.arange(writes, dtype=torch.long))
            .expand(batch_size, -1)
            .clone(),
            "stored_value_positions": (2 + 3 * torch.arange(writes, dtype=torch.long))
            .expand(batch_size, -1)
            .clone(),
            "filler_start_position": 3 * writes,
            "filler_stop_position": query_start,
        },
    )


def generate_needle_batch(
    batch_size: int,
    length: int,
    needle_distance: int,
    noise_vocab: int | None = None,
    key_vocab: int | None = None,
    value_vocab: int | None = None,
    seed: int | None = None,
    *,
    generator: torch.Generator | None = None,
    vocabulary: RetrievalVocabulary | None = None,
) -> RetrievalBatch:
    """Generate a needle whose declared distance is exact by construction.

    Distance is ``query key position - stored value position``. The query key
    is the final input token, and the answer remains separate in ``targets``.
    """

    batch_size = _positive_int("batch_size", batch_size)
    length = _positive_int("length", length)
    needle_distance = _positive_int("needle_distance", needle_distance)
    if not 2 <= needle_distance <= length - 3:
        raise ValueError("needle_distance must lie in [2, length - 3]")
    vocab = _resolve_vocabulary(
        vocabulary,
        key_count=key_vocab,
        value_count=value_vocab,
        filler_count=noise_vocab,
    )
    rng, recorded_seed = _resolve_generator(seed, generator)
    inputs = _filler_inputs(batch_size, length, vocab, rng)
    keys = torch.randint(
        vocab.key_start, vocab.key_stop, (batch_size, 1), generator=rng
    )
    values = torch.randint(
        vocab.value_start, vocab.value_stop, (batch_size, 1), generator=rng
    )
    query_position = length - 1
    stored_value_position = query_position - needle_distance
    stored_key_position = stored_value_position - 1
    write_position = stored_value_position - 2
    inputs[:, write_position] = vocab.write_token
    inputs[:, stored_key_position] = keys[:, 0]
    inputs[:, stored_value_position] = values[:, 0]
    inputs[:, query_position - 1] = vocab.query_token
    inputs[:, query_position] = keys[:, 0]
    query_positions = torch.full((batch_size, 1), query_position, dtype=torch.long)

    return RetrievalBatch(
        inputs=inputs,
        targets=values,
        query_positions=query_positions,
        schema=_schema_with_vocabulary(EXACT_DISTANCE_NEEDLE_SCHEMA, vocab),
        metadata={
            "seed": recorded_seed,
            "distance": needle_distance,
            "needle_keys": keys,
            "stored_value_positions": torch.full(
                (batch_size, 1), stored_value_position, dtype=torch.long
            ),
            "stored_key_positions": torch.full(
                (batch_size, 1), stored_key_position, dtype=torch.long
            ),
            "write_positions": torch.full(
                (batch_size, 1), write_position, dtype=torch.long
            ),
        },
    )


generate_exact_distance_needle_batch = generate_needle_batch
generate_overwrite_retrieval_batch = generate_overwrite_batch


def generate_selective_copy_batch(
    batch_size: int,
    num_items: int,
    num_selected: int,
    length: int,
    value_vocab: int | None = None,
    filler_vocab: int | None = None,
    seed: int | None = None,
    *,
    generator: torch.Generator | None = None,
    vocabulary: RetrievalVocabulary | None = None,
) -> RetrievalBatch:
    """Generate ordered selective copy with targets scored at query markers."""

    batch_size = _positive_int("batch_size", batch_size)
    num_items = _positive_int("num_items", num_items)
    num_selected = _positive_int("num_selected", num_selected)
    length = _positive_int("length", length)
    if num_selected > num_items:
        raise ValueError("num_selected cannot exceed num_items")
    minimum_length = 2 * num_items + num_selected
    if length < minimum_length:
        raise ValueError("length is too short for the selective-copy layout")
    vocab = _resolve_vocabulary(
        vocabulary, value_count=value_vocab, filler_count=filler_vocab
    )
    if num_items > vocab.value_count:
        raise ValueError("num_items exceeds the unique item vocabulary")
    rng, recorded_seed = _resolve_generator(seed, generator)
    item_offsets = _sample_without_replacement(
        batch_size, vocab.value_count, num_items, rng
    )
    items = item_offsets + vocab.value_start
    selected_indices = (
        _sample_without_replacement(batch_size, num_items, num_selected, rng)
        .sort(dim=1)
        .values
    )
    targets = items.gather(1, selected_indices)

    inputs = _filler_inputs(batch_size, length, vocab, rng)
    inputs[:, 0 : 2 * num_items : 2] = vocab.item_token
    inputs[:, 1 : 2 * num_items : 2] = items
    select_marker_positions = 2 * selected_indices
    inputs.scatter_(
        1,
        select_marker_positions,
        torch.full_like(select_marker_positions, vocab.select_token),
    )
    query_start = length - num_selected
    query_positions = (
        torch.arange(query_start, length, dtype=torch.long)
        .expand(batch_size, -1)
        .clone()
    )
    inputs[:, query_start:] = vocab.query_token
    selection_mask = torch.zeros(batch_size, num_items, dtype=torch.bool)
    selection_mask.scatter_(1, selected_indices, True)

    return RetrievalBatch(
        inputs=inputs,
        targets=targets,
        query_positions=query_positions,
        schema=_schema_with_vocabulary(SELECTIVE_COPY_SCHEMA, vocab),
        metadata={
            "seed": recorded_seed,
            "num_items": num_items,
            "num_selected": num_selected,
            "items": items,
            "selected_indices": selected_indices,
            "selection_mask": selection_mask,
            "source_item_positions": (1 + 2 * torch.arange(num_items, dtype=torch.long))
            .expand(batch_size, -1)
            .clone(),
            "filler_start_position": 2 * num_items,
            "filler_stop_position": query_start,
        },
    )


def extrapolation_lengths(
    train_length: int, eval_multiples: Sequence[int]
) -> tuple[int, ...]:
    """Return a strictly increasing test-long sequence from integer multiples."""

    train_length = _positive_int("train_length", train_length)
    if isinstance(eval_multiples, (str, bytes)) or not isinstance(
        eval_multiples, Sequence
    ):
        raise TypeError("eval_multiples must be a sequence of integers")
    multiples = tuple(eval_multiples)
    if not multiples:
        raise ValueError("eval_multiples cannot be empty")
    if any(not _is_int(multiple) or multiple <= 1 for multiple in multiples):
        raise ValueError(
            "every evaluation multiple must be an integer greater than one"
        )
    if any(right <= left for left, right in pairwise(multiples)):
        raise ValueError("evaluation multiples must be strictly increasing")
    return tuple(train_length * multiple for multiple in multiples)


def train_short_test_long_lengths(
    train_length: int, eval_multiples: Sequence[int]
) -> tuple[int, ...]:
    """Return ``(train_length, *test_lengths)`` for an extrapolation cohort."""

    return (train_length, *extrapolation_lengths(train_length, eval_multiples))


def generate_train_test_cohort(
    batch_generator: Callable[..., RetrievalBatch],
    train_length: int,
    eval_multiples: Sequence[int],
    *,
    seed: int = 0,
    **batch_kwargs: object,
) -> dict[int, RetrievalBatch]:
    """Generate a reproducible train-short/test-long batch at every length."""

    if not callable(batch_generator):
        raise TypeError("batch_generator must be callable")
    seed = _nonnegative_seed(seed)
    forbidden = {"length", "seed", "generator"}.intersection(batch_kwargs)
    if forbidden:
        raise ValueError("length, seed, and generator are controlled by the cohort")
    lengths = train_short_test_long_lengths(train_length, eval_multiples)
    if seed + len(lengths) - 1 > 2**63 - 1:
        raise ValueError("cohort seeds exceed the supported seed range")
    cohort: dict[int, RetrievalBatch] = {}
    for index, length in enumerate(lengths):
        batch = batch_generator(length=length, seed=seed + index, **batch_kwargs)
        if not isinstance(batch, RetrievalBatch):
            raise TypeError("batch_generator must return RetrievalBatch instances")
        if batch.inputs.shape[1] != length:
            raise ValueError("batch_generator returned an unexpected sequence length")
        cohort[length] = batch
    return cohort


generate_length_cohort = generate_train_test_cohort


def _validate_logits(logits: torch.Tensor, batch: RetrievalBatch) -> None:
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a tensor")
    if not torch.is_floating_point(logits):
        raise ValueError("logits must have a floating-point dtype")
    if logits.ndim != 3:
        raise ValueError("logits must have shape (B,L,V)")
    if logits.shape[:2] != batch.inputs.shape:
        raise ValueError("logit batch and sequence dimensions must match batch.inputs")
    if logits.device != batch.inputs.device:
        raise ValueError("logits and batch tensors must be on the same device")
    if logits.shape[2] <= int(batch.targets.max()):
        raise ValueError("logit vocabulary does not cover every target token")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")


def gather_query_logits(logits: torch.Tensor, batch: RetrievalBatch) -> torch.Tensor:
    """Gather ``(B,Q,V)`` logits at the batch's explicit query positions."""

    _validate_logits(logits, batch)
    indices = batch.query_positions.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
    return logits.gather(1, indices)


def retrieval_loss(
    logits: torch.Tensor,
    batch: RetrievalBatch,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> torch.Tensor:
    """Cross-entropy over query positions only."""

    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    query_logits = gather_query_logits(logits, batch)
    losses = F.cross_entropy(
        query_logits.flatten(0, 1), batch.targets.flatten(), reduction="none"
    ).view_as(batch.targets)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()


def retrieval_accuracy(logits: torch.Tensor, batch: RetrievalBatch) -> float:
    """Exact token accuracy over all explicit query positions."""

    predictions = gather_query_logits(logits, batch).argmax(dim=-1)
    return float((predictions == batch.targets).float().mean().item())


__all__ = [
    "DEFAULT_VOCABULARY",
    "EXACT_DISTANCE_NEEDLE_SCHEMA",
    "MQAR_SCHEMA",
    "OVERWRITE_SCHEMA",
    "SELECTIVE_COPY_SCHEMA",
    "RetrievalBatch",
    "RetrievalTaskSchema",
    "RetrievalVocabulary",
    "extrapolation_lengths",
    "gather_query_logits",
    "generate_exact_distance_needle_batch",
    "generate_length_cohort",
    "generate_mqar_batch",
    "generate_needle_batch",
    "generate_overwrite_batch",
    "generate_overwrite_retrieval_batch",
    "generate_selective_copy_batch",
    "generate_train_test_cohort",
    "retrieval_accuracy",
    "retrieval_loss",
    "train_short_test_long_lengths",
]
