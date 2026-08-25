"""Matched synthetic-retrieval experiments for :mod:`hybrid_memory_v1_4`.

The harness deliberately keeps data generation outside model RNG streams.  A
data seed identifies every training episode and evaluation cohort, so paired
variants can be audited without retaining large batches in memory.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from . import baselines as _baselines
    from . import tasks as _tasks
    from .model import HybridMemoryConfig, HybridMemoryLM, parameter_count
else:  # Support ``python experiments.py`` and the repository's direct imports.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4 import baselines as _baselines  # type: ignore[no-redef]
    from hybrid_memory_v1_4 import tasks as _tasks  # type: ignore[no-redef]
    from hybrid_memory_v1_4.model import (  # type: ignore[no-redef]
        HybridMemoryConfig,
        HybridMemoryLM,
        parameter_count,
    )


TaskName = Literal["mqar", "overwrite", "exact_distance_needle", "selective_copy"]
_TASK_ALIASES: dict[str, TaskName] = {
    "mqar": "mqar",
    "overwrite": "overwrite",
    "overwrite_retrieval": "overwrite",
    "overwrite-retrieval": "overwrite",
    "exact_distance_needle": "exact_distance_needle",
    "exact-distance-needle": "exact_distance_needle",
    "exact-distance needle": "exact_distance_needle",
    "needle": "exact_distance_needle",
    "selective_copy": "selective_copy",
    "selective-copy": "selective_copy",
}
_MAX_SEED = 2**63 - 1


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class HybridVariant:
    """One named model configuration in a matched cohort."""

    name: str
    config: HybridMemoryConfig

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("variant name must be a nonempty string")
        if not isinstance(self.config, HybridMemoryConfig):
            raise TypeError("config must be a HybridMemoryConfig")


@dataclass(frozen=True)
class TrainingProtocol:
    """All quantities that must remain fixed across compared variants."""

    task: TaskName | str
    train_length: int
    eval_lengths: tuple[int, ...]
    updates: int
    train_batch_size: int
    eval_batch_size: int
    eval_batches: int
    seeds: tuple[int, ...]
    learning_rate: float
    weight_decay: float
    chunk_size: int
    parameter_gap_threshold: float
    routing_auxiliary_coefficient: float
    selected_training_route_mode: str

    def __post_init__(self) -> None:
        if not isinstance(self.task, str):
            raise TypeError("task must be a string")
        try:
            canonical_task = _TASK_ALIASES[self.task.strip().lower()]
        except KeyError as error:
            raise ValueError(f"unsupported retrieval task {self.task!r}") from error
        object.__setattr__(self, "task", canonical_task)
        for name in (
            "train_length",
            "updates",
            "train_batch_size",
            "eval_batch_size",
            "eval_batches",
            "chunk_size",
        ):
            _positive_integer(name, getattr(self, name))
        if type(self.eval_lengths) is not tuple or not self.eval_lengths:
            raise TypeError("eval_lengths must be a nonempty tuple")
        for length in self.eval_lengths:
            _positive_integer("eval length", length)
        if tuple(sorted(set(self.eval_lengths))) != self.eval_lengths:
            raise ValueError("eval_lengths must be unique and strictly increasing")
        if self.train_length not in self.eval_lengths:
            raise ValueError("eval_lengths must include train_length")
        if type(self.seeds) is not tuple or not self.seeds:
            raise TypeError("seeds must be a nonempty tuple")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        for seed in self.seeds:
            if (
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or not 0 <= seed <= _MAX_SEED
            ):
                raise ValueError("every seed must be an integer in [0, 2**63 - 1]")
        learning_rate = _nonnegative_finite("learning_rate", self.learning_rate)
        if learning_rate == 0.0:
            raise ValueError("learning_rate must be positive")
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(
            self, "weight_decay", _nonnegative_finite("weight_decay", self.weight_decay)
        )
        object.__setattr__(
            self,
            "parameter_gap_threshold",
            _nonnegative_finite(
                "parameter_gap_threshold", self.parameter_gap_threshold
            ),
        )
        object.__setattr__(
            self,
            "routing_auxiliary_coefficient",
            _nonnegative_finite(
                "routing_auxiliary_coefficient",
                self.routing_auxiliary_coefficient,
            ),
        )
        if self.selected_training_route_mode not in (
            "hard",
            "soft",
            "straight_through",
        ):
            raise ValueError(
                "selected_training_route_mode must be hard, soft, or straight_through"
            )


@dataclass(frozen=True)
class ParameterComparison:
    variant_name: str
    reference_name: str
    parameter_count: int
    reference_parameter_count: int
    absolute_gap: int
    relative_gap: float
    passed: bool


@dataclass(frozen=True)
class EvaluationResult:
    length: int
    exact_accuracy: float
    exact_sequence_accuracy: float
    bits_per_query: float
    query_count: int
    data_seeds: tuple[int, ...]
    actual_cache_bytes: int
    capacity_cache_bytes: int
    finite: bool


@dataclass(frozen=True)
class VariantSeedResult:
    variant_name: str
    seed: int
    parameter_count: int
    steps: int
    presented_tokens: int
    training_data_seeds: tuple[int, ...]
    training_batch_fingerprints: tuple[str, ...]
    mean_retrieval_loss: float
    mean_routing_auxiliary_loss: float
    untrained_evaluations: tuple[EvaluationResult, ...]
    evaluations: tuple[EvaluationResult, ...]
    chunk_replay_confirmed: bool
    elapsed_wall_seconds: float


@dataclass(frozen=True)
class SourceFileDigest:
    path: str
    sha256: str


@dataclass(frozen=True)
class EnvironmentReport:
    platform: str
    python: str
    torch: str
    device: str
    dtype: str
    cuda_available: bool
    cuda_version: str | None
    cuda_device: str | None


@dataclass(frozen=True)
class ExperimentResult:
    schema_version: int
    protocol: TrainingProtocol
    variants: tuple[HybridVariant, ...]
    parameter_comparisons: tuple[ParameterComparison, ...]
    runs: tuple[VariantSeedResult, ...]
    optimizer: str
    environment: EnvironmentReport
    source_files: tuple[SourceFileDigest, ...]
    git_commit: str | None
    git_status: tuple[str, ...]
    elapsed_wall_seconds: float
    evidentiary: bool


@dataclass(frozen=True)
class StateSize:
    blocks: int
    slots_per_block: int
    value_dim: int

    def __post_init__(self) -> None:
        for name in ("blocks", "slots_per_block", "value_dim"):
            _positive_integer(name, getattr(self, name))


@dataclass(frozen=True)
class SweepQuality:
    length: int
    exact_accuracy: float
    bits_per_query: float
    actual_cache_bytes: int
    capacity_cache_bytes: int


@dataclass(frozen=True)
class StateSweepRow:
    variant_name: str
    paired_control: bool
    blocks: int
    slots_per_block: int
    value_dim: int
    parameter_count: int
    parameter_gap_from_control: int
    relative_parameter_gap_from_control: float
    quality: tuple[SweepQuality, ...]


@dataclass(frozen=True)
class StateSizeSweepResult:
    control_name: str
    rows: tuple[StateSweepRow, ...]
    experiment: ExperimentResult


def _canonical_device(device: torch.device | str) -> torch.device:
    try:
        result = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise TypeError("device must identify a torch device") from error
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def environment_report(
    device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
) -> EnvironmentReport:
    resolved = _canonical_device(device)
    if (
        not isinstance(dtype, torch.dtype)
        or not torch.empty((), dtype=dtype).is_floating_point()
    ):
        raise TypeError("dtype must be a floating-point torch dtype")
    cuda_device = None
    if resolved.type == "cuda":
        cuda_device = torch.cuda.get_device_name(resolved)
    return EnvironmentReport(
        platform=platform.platform(),
        python=sys.version.split()[0],
        torch=torch.__version__,
        device=str(resolved),
        dtype=str(dtype).removeprefix("torch."),
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda,
        cuda_device=cuda_device,
    )


def source_file_digests() -> tuple[SourceFileDigest, ...]:
    root = Path(__file__).resolve().parent
    paths = (
        root / "experiments.py",
        root / "long_context_screen.py",
        root / "temporal_observability_screen.py",
        root / "retrieval_screen.py",
        root / "precision_screen.py",
        root / "learnability_screen.py",
        root / "long_context_continuation.py",
        root / "validation_screen.py",
        root / "continuation_validation.py",
        root / "optimization_diagnostic.py",
        root / "successor_screen.py",
        root / "successor_validation.py",
        root / "identity_validation.py",
        root / "tied_validation.py",
        root / "reverse_binding_validation.py",
        root / "competence_validation.py",
        root / "distance_consolidation.py",
        root / "combined_validation.py",
        root / "retention_successor_screen.py",
        root / "retention_validation.py",
        root / "natural_text_data.py",
        root / "natural_text_screen.py",
        root / "natural_text_diagnostic.py",
        root / "tokenization.py",
        root / "tokenizer_audit.py",
        root / "optimizers.py",
        root / "natural_text_frontier.py",
        root / "long_context_recall.py",
        root / "compute_matched_frontier.py",
        root / "long_context_curriculum.py",
        root / "long_context_diagnostic.py",
        root / "upstream_learning_comparison.py",
        root / "upstream_probe.py",
        root / "model.py",
        root / "gated_delta.py",
        root / "gated_delta_v2.py",
        root / "spin_dirac_memory.py",
        root / "g14_gate_law_screen.py",
        root / "g15_integrity_screen.py",
        root / "g15a_tasks.py",
        root / "g15a_spin_dirac_cohort.py",
        root / "g15a_conditional_controls.py",
        root / "g15al_learned_coordinate_cohort.py",
        root / "g15al_observability_diagnostic.py",
        root / "g15af_full_frame_cohort.py",
        root / "modern_ssm_probe.py",
        root / "native_sm75_probe.py",
        root / "pretrained_sm75_probe.py",
        root / "tasks.py",
        root / "baselines.py",
        root / "selected_block.py",
        root / "attention.py",
        root / "structured_memory.py",
        root / "structured_tier.py",
        root / "fla_adapter.py",
        root / "audits.py",
        root / "PREREGISTRATION.md",
        root / "G4B_PREREGISTRATION.md",
        root / "G4C_PREREGISTRATION.md",
        root / "G4D_PREREGISTRATION.md",
        root / "G4E_PREREGISTRATION.md",
        root / "G4F_PREREGISTRATION.md",
        root / "G5_PREREGISTRATION.md",
        root / "G5B_PREREGISTRATION.md",
        root / "G6_PREREGISTRATION.md",
        root / "G7_PREREGISTRATION.md",
        root / "G8_PREREGISTRATION.md",
        root / "G9_PREREGISTRATION.md",
        root / "G10_PREREGISTRATION.md",
        root / "G11_PREREGISTRATION.md",
        root / "G12_PREREGISTRATION.md",
        root / "G12E_PREREGISTRATION.md",
        root / "G13_PREREGISTRATION.md",
        root / "G14_PREREGISTRATION.md",
        root / "G15_SPIN_DIRAC_PREREGISTRATION.md",
        root / "G15_SPIN_DIRAC_AMENDMENT_2026-08-25.md",
        root / "G15_SPIN_DIRAC_EDIT_LAW_AMENDMENT_2026-08-25.md",
        root / "G15A_EXECUTION_PROTOCOL_2026-08-25.md",
        root / "G15A_CONDITIONAL_CONTROLS_PROTOCOL_2026-08-25.md",
        root / "G15AL_LEARNED_COORDINATE_PROTOCOL_2026-08-25.md",
        root / "G15AL_EXECUTION_AMENDMENT_2026-08-25.md",
        root / "G15AF_FULL_FRAME_PROTOCOL_2026-08-25.md",
        root.parent / "delta_product_reference.py",
    )
    reports = []
    for path in paths:
        if path.is_file():
            reports.append(
                SourceFileDigest(
                    path=path.relative_to(root.parent).as_posix(),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
    return tuple(reports)


def git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def git_status() -> tuple[str, ...]:
    """Return exact tracked/untracked provenance for the v1.4 source surface."""

    try:
        completed = subprocess.run(
            [
                "git",
                "status",
                "--short",
                "--untracked-files=all",
                "--",
                str(Path(__file__).resolve().parent),
                str(
                    Path(__file__).resolve().parent.parent
                    / "delta_product_reference.py"
                ),
            ],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ("git status unavailable",)
    return tuple(line for line in completed.stdout.splitlines() if line)


def jsonable(value: object) -> object:
    """Recursively convert immutable result schemas to JSON-native values."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: jsonable(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, (torch.dtype, torch.device, Path)):
        return str(value).removeprefix("torch.")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def result_json(result: object, *, indent: int | None = 2) -> str:
    return json.dumps(jsonable(result), indent=indent, sort_keys=True, allow_nan=False)


def _derive_data_seed(model_seed: int, phase: str, index: int, length: int) -> int:
    payload = f"hybrid-memory-v1.4:{model_seed}:{phase}:{index}:{length}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & _MAX_SEED


def generate_task_batch(
    task: TaskName | str,
    *,
    batch_size: int,
    length: int,
    seed: int,
) -> _tasks.RetrievalBatch:
    """Generate one deterministic episode with task dimensions scaled to length."""

    canonical = _TASK_ALIASES.get(str(task).strip().lower())
    if canonical is None:
        raise ValueError(f"unsupported retrieval task {task!r}")
    _positive_integer("batch_size", batch_size)
    _positive_integer("length", length)
    vocab = _tasks.DEFAULT_VOCABULARY
    if canonical == "mqar":
        pairs = min(16, vocab.key_count, (length - 2) // 3)
        if pairs < 1:
            raise ValueError("mqar requires length >= 5")
        queries = min(4, pairs, (length - 3 * pairs) // 2)
        while queries < 1 and pairs > 1:
            pairs -= 1
            queries = min(4, pairs, (length - 3 * pairs) // 2)
        return _tasks.generate_mqar_batch(batch_size, pairs, queries, length, seed=seed)
    if canonical == "overwrite":
        writes = min(16, (length - 2) // 3)
        if writes < 3:
            raise ValueError("overwrite requires length >= 11")
        keys = min(4, max(2, writes // 2))
        queries = min(4, keys, writes - keys, (length - 3 * writes) // 2)
        while queries < 1 and writes > 3:
            writes -= 1
            keys = min(4, max(2, writes // 2))
            queries = min(4, keys, writes - keys, (length - 3 * writes) // 2)
        return _tasks.generate_overwrite_batch(
            batch_size,
            writes,
            length,
            num_keys=keys,
            num_queries=queries,
            seed=seed,
        )
    if canonical == "exact_distance_needle":
        if length < 5:
            raise ValueError("exact-distance needle requires length >= 5")
        return _tasks.generate_needle_batch(
            batch_size, length, needle_distance=length - 3, seed=seed
        )
    items = min(16, vocab.value_count, (length - 1) // 2)
    if items < 1:
        raise ValueError("selective-copy requires length >= 3")
    selected = min(4, items, length - 2 * items)
    return _tasks.generate_selective_copy_batch(
        batch_size, items, selected, length, seed=seed
    )


def _batch_fingerprint(batch: _tasks.RetrievalBatch) -> str:
    digest = hashlib.sha256()
    for tensor in (batch.inputs, batch.targets, batch.query_positions):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(contiguous.shape)).encode())
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _write_keys_and_positions(
    batch: _tasks.RetrievalBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    metadata = batch.metadata
    positions = metadata.get("stored_value_positions")
    if isinstance(positions, torch.Tensor):
        for name in ("stored_keys", "write_keys", "needle_keys"):
            keys = metadata.get(name)
            if isinstance(keys, torch.Tensor):
                return keys, positions
    source_positions = metadata.get("source_item_positions")
    items = metadata.get("items")
    if isinstance(source_positions, torch.Tensor) and isinstance(items, torch.Tensor):
        vocab = batch.schema.vocabulary
        synthetic_keys = vocab.key_start + (items - vocab.value_start)
        return synthetic_keys, source_positions
    raise ValueError(
        f"task {batch.schema.name!r} lacks routing-supervision write metadata"
    )


def _read_keys(batch: _tasks.RetrievalBatch) -> torch.Tensor:
    if batch.schema.name == "selective_copy":
        vocab = batch.schema.vocabulary
        return vocab.key_start + (batch.targets - vocab.value_start)
    return batch.inputs.gather(1, batch.query_positions)


def _time_gather(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if values.ndim < 3 or positions.ndim != 2 or positions.shape[0] != values.shape[0]:
        raise ValueError("routing values or positions have incompatible shapes")
    index = positions.reshape(*positions.shape, *([1] * (values.ndim - 2)))
    index = index.expand(positions.shape[0], positions.shape[1], *values.shape[2:])
    return values.gather(1, index)


def routing_auxiliary_loss(
    output: Mapping[str, Any], batch: _tasks.RetrievalBatch
) -> torch.Tensor:
    """Supervise selected-block routes from explicit task labels.

    This is intentionally label-supervised routing, not a label-free objective.
    Each selected layer receives write/erase labels at stored value positions,
    read labels at query positions, and a write-event BCE on its gate.
    """

    logits = output.get("logits")
    diagnostics = output.get("diagnostics")
    if not isinstance(logits, torch.Tensor):
        raise TypeError("output must contain tensor logits")
    if not isinstance(diagnostics, Sequence):
        raise TypeError("output must come from return_diagnostics=True")
    write_keys, write_positions = _write_keys_and_positions(batch)
    read_keys = _read_keys(batch)

    losses: list[torch.Tensor] = []
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, Mapping) or diagnostic.get("kind") in (
            "attention",
            "delta_product",
        ):
            continue
        write_logits = diagnostic.get("write_block_logits")
        erase_logits = diagnostic.get("erase_block_logits")
        read_logits = diagnostic.get("read_block_logits")
        if not all(
            isinstance(item, torch.Tensor)
            for item in (write_logits, erase_logits, read_logits)
        ):
            continue
        assert isinstance(write_logits, torch.Tensor)
        assert isinstance(erase_logits, torch.Tensor)
        assert isinstance(read_logits, torch.Tensor)
        blocks = write_logits.shape[-1]
        if (
            blocks < 1
            or erase_logits.shape[-1] != blocks
            or read_logits.shape[-1] != blocks
        ):
            raise ValueError("selected-layer block diagnostics are inconsistent")
        vocab = batch.schema.vocabulary
        write_labels = (write_keys - vocab.key_start).remainder(blocks)
        read_labels = (read_keys - vocab.key_start).remainder(blocks)

        def route_loss(
            values: torch.Tensor, positions: torch.Tensor, labels: torch.Tensor
        ) -> torch.Tensor:
            selected = _time_gather(values, positions)
            heads = selected.shape[2]
            targets = labels[:, :, None].expand(-1, -1, heads)
            return F.cross_entropy(selected.flatten(0, 2), targets.flatten())

        layer_losses = [
            route_loss(write_logits, write_positions, write_labels),
            route_loss(erase_logits, write_positions, write_labels),
            route_loss(read_logits, batch.query_positions, read_labels),
        ]
        write_gate = diagnostic.get("write_gate")
        if isinstance(write_gate, torch.Tensor):
            if write_gate.shape[:2] != batch.inputs.shape:
                raise ValueError("write_gate has incompatible batch or time dimensions")
            events = torch.zeros_like(write_gate)
            event_index = write_positions[:, :, None].expand(
                -1, -1, write_gate.shape[-1]
            )
            events.scatter_(1, event_index, 1.0)
            epsilon = torch.finfo(write_gate.dtype).eps
            layer_losses.append(
                F.binary_cross_entropy(write_gate.clamp(epsilon, 1.0 - epsilon), events)
            )
        losses.append(torch.stack(layer_losses).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


selected_block_router_auxiliary_loss = routing_auxiliary_loss


def gated_delta_association_auxiliary_loss(
    output: Mapping[str, Any],
    batch: _tasks.RetrievalBatch,
    *,
    temperature: float = 0.10,
    write_gate_weight: float = 1.0,
    retention_weight: float = 0.05,
) -> torch.Tensor:
    """Commission content addressing with explicit synthetic-task labels.

    The loss aligns query vectors with the key vector emitted when the matching
    value was presented, supervises write strength only at value events, and
    discourages filler decay.  It is an intentionally label-supervised
    commissioning objective, not evidence of label-free learning.
    """

    logits = output.get("logits")
    diagnostics = output.get("diagnostics")
    if not isinstance(logits, torch.Tensor):
        raise TypeError("output must contain tensor logits")
    if not isinstance(diagnostics, Sequence):
        raise TypeError("output must come from return_diagnostics=True")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    write_gate_weight = _nonnegative_finite("write_gate_weight", write_gate_weight)
    retention_weight = _nonnegative_finite("retention_weight", retention_weight)
    write_keys, write_positions = _write_keys_and_positions(batch)
    read_keys = _read_keys(batch)

    # Select the last matching write. This also gives overwrite tasks the
    # correct target association while MQAR's unique keys remain unchanged.
    matches = read_keys[:, :, None] == write_keys[:, None, :]
    if not bool(matches.any(dim=-1).all()):
        raise ValueError("every query key must match at least one stored key")
    indices = torch.arange(write_keys.shape[1], device=write_keys.device)
    target_write = torch.where(matches, indices, -1).amax(dim=-1)

    losses: list[torch.Tensor] = []
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, Mapping):
            continue
        kind = diagnostic.get("kind")
        if kind not in ("gated_delta", "gated_delta_v2", "spin_dirac"):
            continue
        query = diagnostic.get("query_vector" if kind == "spin_dirac" else "query")
        key = diagnostic.get("key_vector" if kind == "spin_dirac" else "key")
        write = diagnostic.get("write_strength")
        retention = diagnostic.get("retention")
        if not all(isinstance(item, torch.Tensor) for item in (query, key, write)):
            raise TypeError("gated-delta diagnostics are incomplete")
        assert isinstance(query, torch.Tensor)
        assert isinstance(key, torch.Tensor)
        assert isinstance(write, torch.Tensor)
        query_at_reads = _time_gather(query, batch.query_positions)
        key_at_writes = _time_gather(key, write_positions)
        similarities = (
            torch.einsum("bqhk,bphk->bhqp", query_at_reads, key_at_writes) / temperature
        )
        targets = target_write[:, None, :].expand(-1, similarities.shape[1], -1)
        address_loss = F.cross_entropy(similarities.flatten(0, 2), targets.flatten())

        events = torch.zeros_like(write)
        event_index = write_positions.view(
            write_positions.shape[0],
            write_positions.shape[1],
            *((1,) * (write.ndim - 2)),
        ).expand(write_positions.shape[0], write_positions.shape[1], *write.shape[2:])
        events.scatter_(1, event_index, 1.0)
        write_max = 2.0 if float(write.detach().max()) > 1.0 else 1.0
        probabilities = (write / write_max).clamp(
            torch.finfo(write.dtype).eps, 1.0 - torch.finfo(write.dtype).eps
        )
        write_loss = F.binary_cross_entropy(probabilities, events)
        layer_loss = address_loss + write_gate_weight * write_loss
        if isinstance(retention, torch.Tensor):
            event_by_token = events.amax(dim=-1, keepdim=True)
            nonwrite = (1.0 - event_by_token).expand_as(retention)
            retention_loss = (
                (1.0 - retention).square() * nonwrite
            ).sum() / nonwrite.sum().clamp_min(1.0)
            layer_loss = layer_loss + retention_weight * retention_loss
        losses.append(layer_loss)
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def intermediate_retrieval_auxiliary_loss(
    output: Mapping[str, Any], batch: _tasks.RetrievalBatch
) -> torch.Tensor:
    """Apply the task loss directly after each Gated Delta memory block.

    This is layerwise deep supervision for commissioning the memory core. It
    prevents a later attention/FFN block from hiding a weak recurrent readout.
    """

    logits = output.get("logits")
    diagnostics = output.get("diagnostics")
    intermediate = output.get("intermediate_logits")
    if not isinstance(logits, torch.Tensor):
        raise TypeError("output must contain tensor logits")
    if not isinstance(diagnostics, Sequence) or not isinstance(intermediate, Sequence):
        raise TypeError("output must come from return_diagnostics=True")
    if len(diagnostics) != len(intermediate):
        raise ValueError("diagnostics and intermediate logits must align by layer")
    losses = []
    for diagnostic, layer_logits in zip(diagnostics, intermediate, strict=True):
        if not isinstance(diagnostic, Mapping):
            continue
        if diagnostic.get("kind") not in (
            "gated_delta",
            "gated_delta_v2",
            "spin_dirac",
        ):
            continue
        if not isinstance(layer_logits, torch.Tensor):
            raise TypeError("intermediate logits must be tensors")
        losses.append(_tasks.retrieval_loss(layer_logits, batch))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def stream_model(
    model: HybridMemoryLM, token_ids: torch.Tensor, chunk_size: int
) -> tuple[torch.Tensor, tuple[Any, ...]]:
    """Evaluate arbitrary chunks while carrying every complete layer state."""

    if not isinstance(model, HybridMemoryLM):
        raise TypeError("model must be a HybridMemoryLM")
    _positive_integer("chunk_size", chunk_size)
    if token_ids.ndim != 2 or token_ids.shape[1] < 1:
        raise ValueError("token_ids must have nonempty shape (batch, length)")
    pieces: list[torch.Tensor] = []
    states = None
    for start in range(0, token_ids.shape[1], chunk_size):
        result = model(
            token_ids[:, start : start + chunk_size],
            states,
            delta_scan_mode="recurrent",
            selected_scan_mode="physical_gather",
            structured_scan_mode="recurrent",
        )
        pieces.append(result["logits"])
        states = result["states"]
    assert states is not None
    return torch.cat(pieces, dim=1), states


def _states_close(left: object, right: object, *, rtol: float, atol: float) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.allclose(left, right, rtol=rtol, atol=atol))
    if is_dataclass(left) and is_dataclass(right) and type(left) is type(right):
        return all(
            _states_close(
                getattr(left, field.name),
                getattr(right, field.name),
                rtol=rtol,
                atol=atol,
            )
            for field in fields(left)
        )
    if isinstance(left, Sequence) and isinstance(right, Sequence):
        return len(left) == len(right) and all(
            _states_close(a, b, rtol=rtol, atol=atol)
            for a, b in zip(left, right, strict=True)
        )
    return left == right


def confirm_chunk_replay(
    model: HybridMemoryLM, token_ids: torch.Tensor, chunk_size: int
) -> bool:
    """Fail closed unless a short full recurrent control equals chunk replay."""

    replay_chunk = min(
        _positive_integer("chunk_size", chunk_size), max(1, token_ids.shape[1] // 2)
    )
    was_training = model.training
    model.eval()
    with torch.no_grad():
        full = model(
            token_ids,
            delta_scan_mode="recurrent",
            selected_scan_mode="physical_gather",
            structured_scan_mode="recurrent",
        )
        chunk_logits, chunk_states = stream_model(model, token_ids, replay_chunk)
    model.train(was_training)
    dtype = full["logits"].dtype
    tolerance = 1e-10 if dtype == torch.float64 else 2e-5
    matched = bool(
        torch.allclose(full["logits"], chunk_logits, rtol=tolerance, atol=tolerance)
        and _states_close(full["states"], chunk_states, rtol=tolerance, atol=tolerance)
    )
    if not matched:
        raise RuntimeError("short-control full and arbitrary-chunk replay disagree")
    return True


def _state_is_finite(states: Sequence[Any]) -> bool:
    def finite(value: object) -> bool:
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if is_dataclass(value):
            return all(finite(getattr(value, field.name)) for field in fields(value))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return all(finite(item) for item in value)
        return True

    return finite(states)


def _evaluate(
    model: HybridMemoryLM,
    protocol: TrainingProtocol,
    *,
    model_seed: int,
    device: torch.device,
) -> tuple[EvaluationResult, ...]:
    model.eval()
    reports: list[EvaluationResult] = []
    with torch.no_grad():
        for length_index, length in enumerate(protocol.eval_lengths):
            correct = 0
            exact_sequences = 0
            queries = 0
            sequences = 0
            nll = 0.0
            actual_cache = 0
            capacity_cache = 0
            finite = True
            data_seeds: list[int] = []
            for batch_index in range(protocol.eval_batches):
                data_seed = _derive_data_seed(
                    model_seed,
                    "evaluation",
                    batch_index + length_index * protocol.eval_batches,
                    length,
                )
                data_seeds.append(data_seed)
                batch = generate_task_batch(
                    protocol.task,
                    batch_size=protocol.eval_batch_size,
                    length=length,
                    seed=data_seed,
                ).to(device)
                logits, states = stream_model(model, batch.inputs, protocol.chunk_size)
                query_logits = _tasks.gather_query_logits(logits, batch)
                predictions = query_logits.argmax(dim=-1)
                matches = predictions == batch.targets
                correct += int(matches.sum().item())
                exact_sequences += int(matches.all(dim=1).sum().item())
                queries += batch.targets.numel()
                sequences += batch.targets.shape[0]
                nll += float(
                    _tasks.retrieval_loss(logits, batch, reduction="sum").item()
                )
                byte_report = model.state_byte_report(states)
                actual_cache = max(actual_cache, int(byte_report["actual_bytes"]))
                capacity_cache = max(capacity_cache, int(byte_report["capacity_bytes"]))
                finite = (
                    finite
                    and bool(torch.isfinite(logits).all())
                    and _state_is_finite(states)
                )
            reports.append(
                EvaluationResult(
                    length=length,
                    exact_accuracy=correct / queries,
                    exact_sequence_accuracy=exact_sequences / sequences,
                    bits_per_query=nll / (queries * math.log(2.0)),
                    query_count=queries,
                    data_seeds=tuple(data_seeds),
                    actual_cache_bytes=actual_cache,
                    capacity_cache_bytes=capacity_cache,
                    finite=finite,
                )
            )
    return tuple(reports)


def _fork_rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [torch.cuda.current_device() if device.index is None else device.index]


def _run_variant_seed(
    variant: HybridVariant,
    protocol: TrainingProtocol,
    *,
    model_seed: int,
    expected_parameter_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> VariantSeedResult:
    started = time.perf_counter()
    with torch.random.fork_rng(devices=_fork_rng_devices(device)):
        torch.manual_seed(model_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(model_seed)
        model = HybridMemoryLM(variant.config).to(device=device, dtype=dtype)
        actual_parameters = parameter_count(model)
        if actual_parameters != expected_parameter_count:
            raise RuntimeError("parameter count changed between gate and training")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=protocol.learning_rate,
            weight_decay=protocol.weight_decay,
        )
        untrained_evaluations = _evaluate(
            model, protocol, model_seed=model_seed, device=device
        )
        train_seeds: list[int] = []
        fingerprints: list[str] = []
        retrieval_total = 0.0
        routing_total = 0.0
        model.train()
        for step in range(protocol.updates):
            data_seed = _derive_data_seed(
                model_seed, "training", step, protocol.train_length
            )
            train_seeds.append(data_seed)
            cpu_batch = generate_task_batch(
                protocol.task,
                batch_size=protocol.train_batch_size,
                length=protocol.train_length,
                seed=data_seed,
            )
            fingerprints.append(_batch_fingerprint(cpu_batch))
            batch = cpu_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            detailed = protocol.routing_auxiliary_coefficient > 0.0
            selected_scan_mode = (
                "physical_gather"
                if protocol.selected_training_route_mode == "hard"
                else "dense_recurrent"
            )
            output = model(
                batch.inputs,
                selected_scan_mode=selected_scan_mode,
                selected_route_mode=protocol.selected_training_route_mode,
                return_diagnostics=detailed,
            )
            retrieval = _tasks.retrieval_loss(output["logits"], batch)
            if detailed:
                routing = routing_auxiliary_loss(output, batch)
            else:
                routing = retrieval.new_zeros(())
            loss = retrieval + protocol.routing_auxiliary_coefficient * routing
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            optimizer.step()
            retrieval_total += float(retrieval.detach().item())
            routing_total += float(routing.detach().item())
        if len(set(train_seeds)) != len(train_seeds):
            raise RuntimeError(
                "training data seed schedule contains a repeated episode"
            )
        control_seed = _derive_data_seed(
            model_seed, "chunk-control", 0, protocol.train_length
        )
        control = generate_task_batch(
            protocol.task,
            batch_size=protocol.eval_batch_size,
            length=protocol.train_length,
            seed=control_seed,
        ).to(device)
        replay = confirm_chunk_replay(model, control.inputs, protocol.chunk_size)
        evaluations = _evaluate(model, protocol, model_seed=model_seed, device=device)
    return VariantSeedResult(
        variant_name=variant.name,
        seed=model_seed,
        parameter_count=actual_parameters,
        steps=protocol.updates,
        presented_tokens=protocol.updates
        * protocol.train_batch_size
        * protocol.train_length,
        training_data_seeds=tuple(train_seeds),
        training_batch_fingerprints=tuple(fingerprints),
        mean_retrieval_loss=retrieval_total / protocol.updates,
        mean_routing_auxiliary_loss=routing_total / protocol.updates,
        untrained_evaluations=untrained_evaluations,
        evaluations=evaluations,
        chunk_replay_confirmed=replay,
        elapsed_wall_seconds=time.perf_counter() - started,
    )


def _parameter_gate(
    variants: tuple[HybridVariant, ...], protocol: TrainingProtocol
) -> tuple[ParameterComparison, ...]:
    counts: list[int] = []
    for variant in variants:
        with torch.random.fork_rng():
            torch.manual_seed(protocol.seeds[0])
            counts.append(parameter_count(HybridMemoryLM(variant.config)))
    reference = counts[0]
    reports = []
    for variant, count in zip(variants, counts, strict=True):
        absolute = abs(count - reference)
        relative = absolute / max(reference, 1)
        reports.append(
            ParameterComparison(
                variant_name=variant.name,
                reference_name=variants[0].name,
                parameter_count=count,
                reference_parameter_count=reference,
                absolute_gap=absolute,
                relative_gap=relative,
                passed=relative <= protocol.parameter_gap_threshold,
            )
        )
    failed = [report for report in reports if not report.passed]
    if failed:
        details = ", ".join(
            f"{report.variant_name}={report.relative_gap:.6f}" for report in failed
        )
        raise ValueError(
            "parameter gap threshold exceeded relative to "
            f"{variants[0].name!r}: {details}; threshold="
            f"{protocol.parameter_gap_threshold:.6f}"
        )
    return tuple(reports)


def _verify_pairing(
    runs: Sequence[VariantSeedResult], variants: Sequence[HybridVariant]
) -> None:
    for seed in sorted({run.seed for run in runs}):
        cohort = [run for run in runs if run.seed == seed]
        if len(cohort) != len(variants):
            raise RuntimeError("a seed is missing a paired variant run")
        reference = cohort[0]
        for run in cohort[1:]:
            if (
                run.training_data_seeds != reference.training_data_seeds
                or run.training_batch_fingerprints
                != reference.training_batch_fingerprints
                or tuple(item.data_seeds for item in run.evaluations)
                != tuple(item.data_seeds for item in reference.evaluations)
                or tuple(item.data_seeds for item in run.untrained_evaluations)
                != tuple(item.data_seeds for item in reference.untrained_evaluations)
            ):
                raise RuntimeError("paired variants received different data cohorts")


def run_matched_experiment(
    variants: Sequence[HybridVariant],
    protocol: TrainingProtocol,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    evidentiary: bool = False,
) -> ExperimentResult:
    """Train and evaluate a seed-by-variant matched retrieval cohort."""

    if isinstance(variants, (str, bytes)) or not isinstance(variants, Sequence):
        raise TypeError("variants must be a sequence of HybridVariant instances")
    cohort = tuple(variants)
    if len(cohort) < 2:
        raise ValueError("a matched experiment requires at least two variants")
    if any(not isinstance(variant, HybridVariant) for variant in cohort):
        raise TypeError("every variant must be a HybridVariant")
    if len({variant.name for variant in cohort}) != len(cohort):
        raise ValueError("variant names must be unique")
    if not isinstance(protocol, TrainingProtocol):
        raise TypeError("protocol must be a TrainingProtocol")
    if type(evidentiary) is not bool:
        raise TypeError("evidentiary must be a bool")
    resolved_device = _canonical_device(device)
    if (
        not isinstance(dtype, torch.dtype)
        or not torch.empty((), dtype=dtype).is_floating_point()
    ):
        raise TypeError("dtype must be a floating-point torch dtype")
    required_vocab = _tasks.DEFAULT_VOCABULARY.vocab_size
    for variant in cohort:
        if variant.config.vocab_size < required_vocab:
            raise ValueError(
                f"variant {variant.name!r} vocab_size does not cover the task vocabulary"
            )

    started = time.perf_counter()
    comparisons = _parameter_gate(cohort, protocol)
    expected = {item.variant_name: item.parameter_count for item in comparisons}
    runs = []
    # Seed-major ordering makes every adjacent cohort an explicit paired block.
    for model_seed in protocol.seeds:
        for variant in cohort:
            runs.append(
                _run_variant_seed(
                    variant,
                    protocol,
                    model_seed=model_seed,
                    expected_parameter_count=expected[variant.name],
                    device=resolved_device,
                    dtype=dtype,
                )
            )
    _verify_pairing(runs, cohort)
    return ExperimentResult(
        schema_version=1,
        protocol=protocol,
        variants=cohort,
        parameter_comparisons=comparisons,
        runs=tuple(runs),
        optimizer="torch.optim.AdamW",
        environment=environment_report(resolved_device, dtype),
        source_files=source_file_digests(),
        git_commit=git_commit(),
        git_status=git_status(),
        elapsed_wall_seconds=time.perf_counter() - started,
        evidentiary=evidentiary,
    )


def delta_only_control(name: str, base_config: HybridMemoryConfig) -> HybridVariant:
    """Create the normal matched-shell DeltaProduct control."""

    if not isinstance(base_config, HybridMemoryConfig):
        raise TypeError("base_config must be a HybridMemoryConfig")
    plan = tuple("delta_product" for _ in base_config.layer_plan)
    return HybridVariant(name, replace(base_config, layer_plan=plan))


def run_state_size_sweep(
    protocol: TrainingProtocol,
    *,
    control: HybridVariant | None,
    base_config: HybridMemoryConfig,
    sizes: Sequence[StateSize | tuple[int, int, int]],
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    evidentiary: bool = False,
) -> StateSizeSweepResult:
    """Compare selected-memory quality to actual state bytes with one control row."""

    if control is None:
        raise ValueError("state-size sweeps require an explicit paired control row")
    if not isinstance(control, HybridVariant):
        raise TypeError("control must be a HybridVariant")
    if not isinstance(base_config, HybridMemoryConfig):
        raise TypeError("base_config must be a HybridMemoryConfig")
    if "selected_block" not in base_config.layer_plan:
        raise ValueError("base_config must contain at least one selected_block layer")
    if isinstance(sizes, (str, bytes)) or not isinstance(sizes, Sequence) or not sizes:
        raise ValueError("sizes must be a nonempty sequence")
    normalized: list[StateSize] = []
    for size in sizes:
        if isinstance(size, StateSize):
            normalized.append(size)
        elif isinstance(size, tuple) and len(size) == 3:
            normalized.append(StateSize(*size))
        else:
            raise TypeError("each size must be StateSize or a three-integer tuple")
    if len(set(normalized)) != len(normalized):
        raise ValueError("state sizes must be unique")
    variants = [control]
    for size in normalized:
        config = replace(
            base_config,
            selected_blocks=size.blocks,
            selected_slots_per_block=size.slots_per_block,
            selected_value_dim=size.value_dim,
            selected_update_rank=min(
                base_config.selected_update_rank, size.slots_per_block
            ),
        )
        variants.append(
            HybridVariant(
                f"selected_b{size.blocks}_s{size.slots_per_block}_v{size.value_dim}",
                config,
            )
        )
    experiment = run_matched_experiment(
        variants,
        protocol,
        device=device,
        dtype=dtype,
        evidentiary=evidentiary,
    )
    comparison_by_name = {
        item.variant_name: item for item in experiment.parameter_comparisons
    }
    rows = []
    for variant in variants:
        variant_runs = [
            run for run in experiment.runs if run.variant_name == variant.name
        ]
        quality = []
        for length in protocol.eval_lengths:
            evaluations = [
                evaluation
                for run in variant_runs
                for evaluation in run.evaluations
                if evaluation.length == length
            ]
            quality.append(
                SweepQuality(
                    length=length,
                    exact_accuracy=sum(item.exact_accuracy for item in evaluations)
                    / len(evaluations),
                    bits_per_query=sum(item.bits_per_query for item in evaluations)
                    / len(evaluations),
                    actual_cache_bytes=max(
                        item.actual_cache_bytes for item in evaluations
                    ),
                    capacity_cache_bytes=max(
                        item.capacity_cache_bytes for item in evaluations
                    ),
                )
            )
        comparison = comparison_by_name[variant.name]
        rows.append(
            StateSweepRow(
                variant_name=variant.name,
                paired_control=variant.name == control.name,
                blocks=variant.config.selected_blocks,
                slots_per_block=variant.config.selected_slots_per_block,
                value_dim=variant.config.selected_value_dim,
                parameter_count=comparison.parameter_count,
                parameter_gap_from_control=comparison.absolute_gap,
                relative_parameter_gap_from_control=comparison.relative_gap,
                quality=tuple(quality),
            )
        )
    if sum(row.paired_control for row in rows) != 1:
        raise RuntimeError("state-size sweep did not produce exactly one control row")
    return StateSizeSweepResult(control.name, tuple(rows), experiment)


def build_official_mamba2(
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    skip_unavailable: bool = False,
    **kwargs: Any,
) -> tuple[nn.Module | None, _baselines.BaselineAvailability]:
    """Build only official Mamba-2, or explicitly raise/return a skipped row."""

    if type(skip_unavailable) is not bool:
        raise TypeError("skip_unavailable must be a bool")
    availability = _baselines.baseline_availability(
        "mamba2_official", device=device, dtype=dtype
    )
    if not availability:
        if skip_unavailable:
            return None, availability
        raise _baselines.BaselineUnavailableError(availability)
    model = _baselines.build_baseline(
        "mamba2_official", device=device, dtype=dtype, **kwargs
    )
    return model, availability


__all__ = [
    "EnvironmentReport",
    "EvaluationResult",
    "ExperimentResult",
    "HybridVariant",
    "ParameterComparison",
    "SourceFileDigest",
    "StateSize",
    "StateSizeSweepResult",
    "StateSweepRow",
    "SweepQuality",
    "TrainingProtocol",
    "VariantSeedResult",
    "build_official_mamba2",
    "confirm_chunk_replay",
    "delta_only_control",
    "environment_report",
    "gated_delta_association_auxiliary_loss",
    "generate_task_batch",
    "git_commit",
    "git_status",
    "intermediate_retrieval_auxiliary_loss",
    "jsonable",
    "result_json",
    "routing_auxiliary_loss",
    "run_matched_experiment",
    "run_state_size_sweep",
    "selected_block_router_auxiliary_loss",
    "source_file_digests",
    "stream_model",
]
