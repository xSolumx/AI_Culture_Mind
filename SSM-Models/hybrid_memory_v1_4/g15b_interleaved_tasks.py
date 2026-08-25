"""Interleaved shared-payload tasks for the frozen G15B controller gate."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Literal

import torch
from torch.nn import functional as F

TaskName = Literal["mqar", "overwrite", "selective", "needle"]

PAD_TOKEN = 0
WRITE_TOKEN = 1
QUERY_TOKEN = 2
SELECT_TOKEN = 3
ITEM_TOKEN = 4
PAYLOAD_START = 5
PAYLOAD_COUNT = 64
VOCAB_SIZE = PAYLOAD_START + PAYLOAD_COUNT

ROLE_FILLER = 0
ROLE_WRITE_MARKER = 1
ROLE_WRITE_KEY = 2
ROLE_WRITE_VALUE = 3
ROLE_QUERY_MARKER = 4
ROLE_QUERY_KEY = 5
ROLE_ITEM_MARKER = 6
ROLE_ITEM_KEY = 7
ROLE_ITEM_VALUE = 8


def _stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def _generator(seed: int) -> torch.Generator:
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    return torch.Generator(device="cpu").manual_seed(seed)


def _randint(high: int, generator: torch.Generator) -> int:
    return int(torch.randint(high, (), generator=generator))


def _payload(offset: int) -> int:
    return PAYLOAD_START + offset


@dataclass(frozen=True)
class _Event:
    kind: Literal["write", "select", "item", "query"]
    key_index: int
    value_token: int | None = None

    @property
    def width(self) -> int:
        return 2 if self.kind == "query" else 3

    @property
    def valid_write(self) -> bool:
        return self.kind in ("write", "select")


@dataclass(frozen=True)
class InterleavedBatch:
    """Validated G15B episode tensors with explicit audit-only event labels."""

    task: TaskName
    token_ids: torch.Tensor
    targets: torch.Tensor
    query_positions: torch.Tensor
    query_keys: torch.Tensor
    query_key_indices: torch.Tensor
    live_keys: torch.Tensor
    write_positions: torch.Tensor
    write_keys: torch.Tensor
    write_values: torch.Tensor
    overwrite_mask: torch.Tensor
    write_event_mask: torch.Tensor
    erase_event_mask: torch.Tensor
    roles: torch.Tensor
    needle_distances: torch.Tensor
    seed: int

    def __post_init__(self) -> None:
        self.validate()

    @property
    def batch_size(self) -> int:
        return self.token_ids.shape[0]

    @property
    def length(self) -> int:
        return self.token_ids.shape[1]

    @property
    def queries(self) -> int:
        return self.targets.shape[1]

    @property
    def writes(self) -> int:
        return self.write_positions.shape[1]

    def to(self, device: torch.device | str) -> InterleavedBatch:
        destination = torch.device(device)
        updates = {
            name: value.to(destination)
            for name, value in self.__dict__.items()
            if isinstance(value, torch.Tensor)
        }
        return replace(self, **updates)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.task.encode())
        for tensor in (
            self.token_ids,
            self.targets,
            self.query_positions,
            self.query_keys,
            self.live_keys,
            self.write_positions,
            self.write_keys,
            self.write_values,
            self.overwrite_mask,
        ):
            contiguous = tensor.detach().cpu().contiguous()
            digest.update(str(tuple(contiguous.shape)).encode())
            digest.update(contiguous.numpy().tobytes())
        return digest.hexdigest()

    def validate(self) -> None:
        integer_names = (
            "token_ids",
            "targets",
            "query_positions",
            "query_keys",
            "query_key_indices",
            "live_keys",
            "write_positions",
            "write_keys",
            "write_values",
            "roles",
            "needle_distances",
        )
        boolean_names = ("overwrite_mask", "write_event_mask", "erase_event_mask")
        for name in integer_names:
            tensor = getattr(self, name)
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.long:
                raise TypeError(f"{name} must be a long tensor")
        for name in boolean_names:
            tensor = getattr(self, name)
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.bool:
                raise TypeError(f"{name} must be a bool tensor")
        batch, length = self.token_ids.shape
        if batch < 1 or length < 8:
            raise ValueError("token_ids must have shape (B,L) with B >= 1 and L >= 8")
        if self.targets.ndim != 2 or self.targets.shape[0] != batch:
            raise ValueError("targets must have shape (B,Q)")
        if self.query_positions.shape != self.targets.shape:
            raise ValueError("query_positions must match targets")
        if self.query_keys.shape != self.targets.shape:
            raise ValueError("query_keys must match targets")
        if self.query_key_indices.shape != self.targets.shape:
            raise ValueError("query_key_indices must match targets")
        if self.live_keys.ndim != 2 or self.live_keys.shape[0] != batch:
            raise ValueError("live_keys must have shape (B,K)")
        write_shape = self.write_positions.shape
        if len(write_shape) != 2 or write_shape[0] != batch:
            raise ValueError("write_positions must have shape (B,W)")
        for name in ("write_keys", "write_values", "overwrite_mask"):
            if getattr(self, name).shape != write_shape:
                raise ValueError(f"{name} must match write_positions")
        for name in ("write_event_mask", "erase_event_mask", "roles"):
            if getattr(self, name).shape != (batch, length):
                raise ValueError(f"{name} must have shape (B,L)")
        if self.needle_distances.shape != (batch, self.queries):
            raise ValueError("needle_distances must have shape (B,Q)")
        if not (
            self.token_ids.device
            == self.targets.device
            == self.query_positions.device
            == self.write_positions.device
        ):
            raise ValueError("batch tensors must share one device")
        if int(self.token_ids.min()) < 0 or int(self.token_ids.max()) >= VOCAB_SIZE:
            raise ValueError("token id outside the frozen G15B vocabulary")
        for name in (
            "targets",
            "query_keys",
            "live_keys",
            "write_keys",
            "write_values",
        ):
            tensor = getattr(self, name)
            if int(tensor.min()) < PAYLOAD_START or int(tensor.max()) >= VOCAB_SIZE:
                raise ValueError(f"{name} must contain shared-payload tokens")
        if (
            int(self.query_positions.min()) < 0
            or int(self.query_positions.max()) >= length
        ):
            raise ValueError("query position outside the sequence")
        if (
            int(self.write_positions.min()) < 0
            or int(self.write_positions.max()) >= length
        ):
            raise ValueError("write position outside the sequence")
        if bool((self.query_key_indices < 0).any()) or bool(
            (self.query_key_indices >= self.live_keys.shape[1]).any()
        ):
            raise ValueError("query key index outside live_keys")
        if not torch.equal(
            self.query_keys,
            self.live_keys.gather(1, self.query_key_indices),
        ):
            raise ValueError("query keys and live-key indices disagree")
        if not torch.equal(
            self.token_ids.gather(1, self.query_positions), self.query_keys
        ):
            raise ValueError("query positions do not point to query keys")
        if not torch.equal(
            self.token_ids.gather(1, self.write_positions), self.write_values
        ):
            raise ValueError("write positions do not point to written values")
        expected_write_events = torch.zeros_like(self.write_event_mask)
        expected_write_events.scatter_(1, self.write_positions, True)
        if not torch.equal(expected_write_events, self.write_event_mask):
            raise ValueError("write event mask disagrees with write positions")
        expected_erases = torch.zeros_like(self.erase_event_mask)
        expected_erases.scatter_(1, self.write_positions, self.overwrite_mask)
        if not torch.equal(expected_erases, self.erase_event_mask):
            raise ValueError("erase event mask disagrees with overwrites")
        if bool((self.query_keys == self.targets).any()):
            raise ValueError("a query key equals its answer")

        for row in range(batch):
            state: dict[int, int] = {}
            write_cursor = 0
            query_cursor = 0
            for position in range(length):
                if (
                    write_cursor < self.writes
                    and int(self.write_positions[row, write_cursor]) == position
                ):
                    key = int(self.write_keys[row, write_cursor])
                    value = int(self.write_values[row, write_cursor])
                    expected_overwrite = key in state
                    if (
                        bool(self.overwrite_mask[row, write_cursor])
                        != expected_overwrite
                    ):
                        raise ValueError("overwrite label disagrees with causal replay")
                    state[key] = value
                    write_cursor += 1
                if (
                    query_cursor < self.queries
                    and int(self.query_positions[row, query_cursor]) == position
                ):
                    key = int(self.query_keys[row, query_cursor])
                    if key not in state or state[key] != int(
                        self.targets[row, query_cursor]
                    ):
                        raise ValueError(
                            "query target is not the latest preceding write"
                        )
                    start = max(0, position - 3)
                    if bool(
                        (
                            self.token_ids[row, start : position + 1]
                            == self.targets[row, query_cursor]
                        ).any()
                    ):
                        raise ValueError("answer leaks into the query receptive field")
                    query_cursor += 1
            if write_cursor != self.writes or query_cursor != self.queries:
                raise ValueError("event positions are not strictly chronological")


def _sample_keys(count: int, generator: torch.Generator) -> list[int]:
    return [
        _payload(int(offset))
        for offset in torch.randperm(PAYLOAD_COUNT, generator=generator)[:count]
    ]


def _sample_value(
    key: int,
    generator: torch.Generator,
    *,
    forbidden: tuple[int, ...] = (),
) -> int:
    forbidden_set = {key, *forbidden}
    candidates = [
        token
        for token in range(PAYLOAD_START, VOCAB_SIZE)
        if token not in forbidden_set
    ]
    return candidates[_randint(len(candidates), generator)]


def _mqar_events(
    keys: list[int], queries: int, generator: torch.Generator
) -> list[_Event]:
    events: list[_Event] = []
    written: list[int] = []
    remaining_writes = list(range(len(keys)))
    remaining_queries = queries
    while remaining_writes or remaining_queries:
        can_query = bool(written) and remaining_queries > 0
        choose_query = can_query and (
            not remaining_writes or bool(_randint(2, generator))
        )
        if choose_query:
            events.append(_Event("query", written[_randint(len(written), generator)]))
            remaining_queries -= 1
        else:
            selected = _randint(len(remaining_writes), generator)
            key_index = remaining_writes.pop(selected)
            value = _sample_value(keys[key_index], generator)
            events.append(_Event("write", key_index, value))
            written.append(key_index)
    return events


def _overwrite_events(
    keys: list[int], max_writes: int, queries: int, generator: torch.Generator
) -> list[_Event]:
    key_count = len(keys)
    writes = max(2 * key_count, max_writes)
    if queries < 2:
        raise ValueError("overwrite episodes require at least two queries")
    values: dict[int, int] = {}
    events: list[_Event] = []
    for key_index, key in enumerate(keys):
        value = _sample_value(key, generator)
        values[key_index] = value
        events.append(_Event("write", key_index, value))
    anchor = _randint(key_count, generator)
    events.append(_Event("query", anchor))
    for key_index, key in enumerate(keys):
        value = _sample_value(key, generator, forbidden=(values[key_index],))
        values[key_index] = value
        events.append(_Event("write", key_index, value))
    events.append(_Event("query", anchor))
    extra: list[_Event] = []
    for _ in range(writes - 2 * key_count):
        key_index = _randint(key_count, generator)
        value = _sample_value(
            keys[key_index], generator, forbidden=(values[key_index],)
        )
        values[key_index] = value
        extra.append(_Event("write", key_index, value))
    for _ in range(queries - 2):
        extra.append(_Event("query", _randint(key_count, generator)))
    if extra:
        order = torch.randperm(len(extra), generator=generator).tolist()
        events.extend(extra[index] for index in order)
    return events


def _selective_events(
    keys: list[int], queries: int, generator: torch.Generator
) -> list[_Event]:
    events: list[_Event] = []
    selected: list[int] = []
    for key_index, key in enumerate(keys):
        ignored = _sample_value(key, generator)
        kept = _sample_value(key, generator, forbidden=(ignored,))
        if bool(_randint(2, generator)):
            events.extend(
                (_Event("item", key_index, ignored), _Event("select", key_index, kept))
            )
        else:
            events.extend(
                (_Event("select", key_index, kept), _Event("item", key_index, ignored))
            )
        selected.append(key_index)
        if len(events) > 2 and bool(_randint(2, generator)):
            events.append(_Event("query", key_index))
    existing_queries = sum(event.kind == "query" for event in events)
    while existing_queries < queries:
        events.append(_Event("query", selected[_randint(len(selected), generator)]))
        existing_queries += 1
    return events


def _schedule_events(
    events: list[_Event], length: int, generator: torch.Generator
) -> list[int]:
    mandatory = [2 if event.kind == "query" else 0 for event in events]
    used = sum(event.width for event in events) + sum(mandatory)
    if used > length:
        raise ValueError(
            f"event schedule requires {used} tokens but length is {length}"
        )
    extra = length - used
    gaps = [0] * (len(events) + 1)
    if extra:
        assignments = torch.randint(
            len(gaps), (extra,), generator=generator, dtype=torch.long
        )
        counts = torch.bincount(assignments, minlength=len(gaps))
        gaps = [int(value) for value in counts]
    positions: list[int] = []
    cursor = gaps[0]
    for index, event in enumerate(events):
        cursor += mandatory[index]
        positions.append(cursor)
        cursor += event.width + gaps[index + 1]
    if cursor != length:
        raise RuntimeError("internal G15B gap accounting error")
    return positions


def _needle_row(
    *,
    length: int,
    key: int,
    distance: int,
    generator: torch.Generator,
) -> tuple[list[_Event], list[int]]:
    if not 4 <= distance <= length - 3:
        raise ValueError("needle distance must lie in [4, length - 3]")
    write_start = _randint(length - distance - 2, generator)
    query_start = write_start + distance + 1
    value = _sample_value(key, generator)
    return [_Event("write", 0, value), _Event("query", 0)], [write_start, query_start]


def generate_interleaved_batch(
    task: TaskName,
    batch_size: int,
    length: int,
    live_keys: int,
    max_writes: int,
    queries: int,
    *,
    seed: int,
    needle_distance: int | None = None,
) -> InterleavedBatch:
    """Generate a deterministic batch from the frozen G15B grammar."""

    if task not in ("mqar", "overwrite", "selective", "needle"):
        raise ValueError(f"unknown G15B task {task!r}")
    for name, value in (
        ("batch_size", batch_size),
        ("length", length),
        ("live_keys", live_keys),
        ("max_writes", max_writes),
        ("queries", queries),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if live_keys > 8:
        raise ValueError("G15B freezes at most eight simultaneously live keys")
    if task == "needle":
        live_keys = 1
        queries = 1

    root = _generator(seed)
    rows_tokens: list[torch.Tensor] = []
    rows_roles: list[torch.Tensor] = []
    rows_targets: list[list[int]] = []
    rows_query_positions: list[list[int]] = []
    rows_query_keys: list[list[int]] = []
    rows_query_indices: list[list[int]] = []
    rows_live_keys: list[list[int]] = []
    rows_write_positions: list[list[int]] = []
    rows_write_keys: list[list[int]] = []
    rows_write_values: list[list[int]] = []
    rows_overwrite: list[list[bool]] = []
    rows_distances: list[list[int]] = []

    for row in range(batch_size):
        generator = _generator(
            _stable_seed("g15b-row", seed, row, int(root.initial_seed()))
        )
        keys = _sample_keys(live_keys, generator)
        if task == "mqar":
            events = _mqar_events(keys, queries, generator)
            starts = _schedule_events(events, length, generator)
        elif task == "overwrite":
            events = _overwrite_events(keys, max_writes, queries, generator)
            starts = _schedule_events(events, length, generator)
        elif task == "selective":
            events = _selective_events(keys, queries, generator)
            starts = _schedule_events(events, length, generator)
        else:
            distance = (
                needle_distance
                if needle_distance is not None
                else 4 + _randint(length - 6, generator)
            )
            events, starts = _needle_row(
                length=length, key=keys[0], distance=distance, generator=generator
            )

        tokens = torch.randint(
            PAYLOAD_START,
            VOCAB_SIZE,
            (length,),
            generator=generator,
            dtype=torch.long,
        )
        roles = torch.full((length,), ROLE_FILLER, dtype=torch.long)
        state: dict[int, int] = {}
        targets: list[int] = []
        query_positions: list[int] = []
        query_keys: list[int] = []
        query_indices: list[int] = []
        write_positions: list[int] = []
        write_keys: list[int] = []
        write_values: list[int] = []
        overwrite: list[bool] = []
        distances: list[int] = []

        for event, start in zip(events, starts, strict=True):
            key = keys[event.key_index]
            if event.kind == "query":
                if key not in state:
                    raise RuntimeError(
                        "internal G15B event order queried an unwritten key"
                    )
                tokens[start] = QUERY_TOKEN
                tokens[start + 1] = key
                roles[start] = ROLE_QUERY_MARKER
                roles[start + 1] = ROLE_QUERY_KEY
                target = state[key]
                targets.append(target)
                query_positions.append(start + 1)
                query_keys.append(key)
                query_indices.append(event.key_index)
                latest_write = max(
                    position
                    for position, written_key in zip(
                        write_positions, write_keys, strict=True
                    )
                    if written_key == key
                )
                distances.append(start + 1 - latest_write)
                for local in range(max(0, start - 2), start):
                    if roles[local] == ROLE_FILLER and int(tokens[local]) == target:
                        tokens[local] = _payload(
                            (target - PAYLOAD_START + 1) % PAYLOAD_COUNT
                        )
                continue

            assert event.value_token is not None
            marker = {
                "write": WRITE_TOKEN,
                "select": SELECT_TOKEN,
                "item": ITEM_TOKEN,
            }[event.kind]
            marker_role = {
                "write": ROLE_WRITE_MARKER,
                "select": ROLE_WRITE_MARKER,
                "item": ROLE_ITEM_MARKER,
            }[event.kind]
            key_role = ROLE_ITEM_KEY if event.kind == "item" else ROLE_WRITE_KEY
            value_role = ROLE_ITEM_VALUE if event.kind == "item" else ROLE_WRITE_VALUE
            tokens[start : start + 3] = torch.tensor(
                (marker, key, event.value_token), dtype=torch.long
            )
            roles[start : start + 3] = torch.tensor(
                (marker_role, key_role, value_role), dtype=torch.long
            )
            if event.valid_write:
                is_overwrite = key in state
                state[key] = event.value_token
                write_positions.append(start + 2)
                write_keys.append(key)
                write_values.append(event.value_token)
                overwrite.append(is_overwrite)

        if len(targets) != queries:
            raise RuntimeError(
                f"internal G15B generator produced {len(targets)} queries, expected {queries}"
            )
        rows_tokens.append(tokens)
        rows_roles.append(roles)
        rows_targets.append(targets)
        rows_query_positions.append(query_positions)
        rows_query_keys.append(query_keys)
        rows_query_indices.append(query_indices)
        rows_live_keys.append(keys)
        rows_write_positions.append(write_positions)
        rows_write_keys.append(write_keys)
        rows_write_values.append(write_values)
        rows_overwrite.append(overwrite)
        rows_distances.append(distances)

    write_counts = {len(row) for row in rows_write_positions}
    if len(write_counts) != 1:
        raise RuntimeError("G15B batches require a fixed number of writes per row")
    token_ids = torch.stack(rows_tokens)
    roles = torch.stack(rows_roles)
    write_positions_tensor = torch.tensor(rows_write_positions, dtype=torch.long)
    overwrite_tensor = torch.tensor(rows_overwrite, dtype=torch.bool)
    write_event_mask = torch.zeros(batch_size, length, dtype=torch.bool)
    write_event_mask.scatter_(1, write_positions_tensor, True)
    erase_event_mask = torch.zeros_like(write_event_mask)
    erase_event_mask.scatter_(1, write_positions_tensor, overwrite_tensor)
    return InterleavedBatch(
        task=task,
        token_ids=token_ids,
        targets=torch.tensor(rows_targets, dtype=torch.long),
        query_positions=torch.tensor(rows_query_positions, dtype=torch.long),
        query_keys=torch.tensor(rows_query_keys, dtype=torch.long),
        query_key_indices=torch.tensor(rows_query_indices, dtype=torch.long),
        live_keys=torch.tensor(rows_live_keys, dtype=torch.long),
        write_positions=write_positions_tensor,
        write_keys=torch.tensor(rows_write_keys, dtype=torch.long),
        write_values=torch.tensor(rows_write_values, dtype=torch.long),
        overwrite_mask=overwrite_tensor,
        write_event_mask=write_event_mask,
        erase_event_mask=erase_event_mask,
        roles=roles,
        needle_distances=torch.tensor(rows_distances, dtype=torch.long),
        seed=seed,
    )


def oracle_direct_read_accuracy(
    batch: InterleavedBatch,
    memory: object,
) -> tuple[float, float]:
    """Run the actual memory law under exact addresses and edit timing.

    The supplied ``memory`` must be a float64 ``SpinDiracMemory`` with at least
    one head. Accuracy is decoded from its positive read against a frozen 64
    vector codebook. This is a direct-state ceiling, not an LM/controller result.
    """

    if not hasattr(memory, "forward_controls") or not hasattr(memory, "config"):
        raise TypeError("memory must expose SpinDirac forward_controls and config")
    config = memory.config
    device = next(memory.parameters()).device
    dtype = next(memory.parameters()).dtype
    if dtype != torch.float64:
        raise TypeError("oracle direct-read ceiling requires float64")
    heads = config.heads
    batch_device = batch.to(device)
    shape = (batch.batch_size, batch.length, heads, 8)
    query = torch.zeros(shape, dtype=dtype, device=device)
    key = torch.zeros_like(query)
    value = torch.zeros_like(query)
    erase = torch.zeros(*shape[:-1], 1, dtype=dtype, device=device)
    write = torch.zeros_like(erase)
    retention = torch.full_like(erase, 1.0 - 1e-14)
    coordinates = torch.zeros(
        batch.batch_size,
        batch.length,
        heads,
        28,
        dtype=dtype,
        device=device,
    )
    code_generator = _generator(150_815)
    codebook = F.normalize(
        torch.randn(PAYLOAD_COUNT, 8, generator=code_generator, dtype=dtype), dim=-1
    ).to(device)
    rows = torch.arange(batch.batch_size, device=device)
    for write_index in range(batch.writes):
        positions = batch_device.write_positions[:, write_index]
        key_indices = (
            (
                batch_device.live_keys
                == batch_device.write_keys[:, write_index : write_index + 1]
            )
            .long()
            .argmax(dim=1)
        )
        key[rows, positions, 0, key_indices] = 1.0
        offsets = batch_device.write_values[:, write_index] - PAYLOAD_START
        value[rows, positions, 0] = codebook[offsets]
        write[rows, positions, 0, 0] = 1.0
        erase[rows, positions, 0, 0] = batch_device.overwrite_mask[:, write_index].to(
            dtype
        )
    for query_index in range(batch.queries):
        positions = batch_device.query_positions[:, query_index]
        key_indices = batch_device.query_key_indices[:, query_index]
        query[rows, positions, 0, key_indices] = 1.0
    read, _ = memory.forward_controls(
        query,
        key,
        value,
        erase,
        write,
        retention,
        coordinates,
        scan_mode="parallel",
    )
    selected = read[:, :, 0, :8].gather(
        1,
        batch_device.query_positions.unsqueeze(-1).expand(-1, -1, 8),
    )
    scores = F.normalize(selected, dim=-1) @ codebook.T
    predictions = scores.argmax(dim=-1) + PAYLOAD_START
    accuracy = float((predictions == batch_device.targets).double().mean())
    target_vectors = codebook[batch_device.targets - PAYLOAD_START]
    residual = float((selected - target_vectors).abs().max())
    return accuracy, residual


__all__ = [
    "ITEM_TOKEN",
    "PAD_TOKEN",
    "PAYLOAD_COUNT",
    "PAYLOAD_START",
    "QUERY_TOKEN",
    "SELECT_TOKEN",
    "VOCAB_SIZE",
    "WRITE_TOKEN",
    "InterleavedBatch",
    "TaskName",
    "generate_interleaved_batch",
    "oracle_direct_read_accuracy",
]
