"""Preregistered multi-relation ``2.A5`` state-tracking benchmark.

This task withholds all three central presentation words
``a^2``, ``b^3``, and ``(ab)^5`` from training while retaining exact reachability
of every binary-icosahedral state.  Each held-out word is paired with an
equal-length identity-token block.  The benchmark repeats the symbolic task
under conjugated generating sets and compares the maintained Pure Rotor model,
its identity-transport ablation, an explicit Spin quaternion scan, Transformers
Mamba-2, and an unfused equation-faithful DeltaProduct reference.

The result is a controlled empirical mechanism test, not a theorem about any
model family and not a fused-kernel systems benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from functools import cache
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from benchmark_pure_rotor_2a5 import (
    BinaryA5Task,
    CounterMetrics,
    SpinQuaternionScanModel,
    binary_icosahedral_task,
    mamba2_model,
    oracle_logits,
    parameter_count,
    pure_rotor_model,
    seed_everything,
)
from compare_recurrences import state_and_pair_coverage_audit
from delta_product_reference import DeltaProductReferenceModel
from pdssm_group_actions import ExactRegularPD
from pure_rotor_ssm import __version__ as PURE_ROTOR_VERSION
from torch import nn
from torch.nn import functional as F

TRAINED_CANDIDATES = (
    "pure_rotor",
    "identity_rotation_ablation",
    "spin_quaternion_scan",
    "mamba2_transformers",
    "delta_product_reference",
)
ANALYTIC_ORACLES = (
    "exact_table_oracle",
    "projective_a5_oracle",
    "float64_quaternion_oracle",
)
ALL_ORACLES = (*ANALYTIC_ORACLES, "exact_regular_pd_oracle")
CONJUGATOR_LABELS = {"e": 0, "a": 1, "b": 2}
DELTA_PRODUCT_SOURCE_COMMIT = DeltaProductReferenceModel.source_commit
PD_SSM_SOURCE_COMMIT = "8682e78101be84f67ceb64702855e5d9e820f7d2"
_WITNESS_CACHE: dict[str, dict[int, tuple[int, ...]]] = {}


@dataclass(frozen=True)
class MultiRelationConfig:
    """Parameter-near pilot configuration."""

    steps: int = 300
    batch_size: int = 16
    training_length: int = 16
    validation_batches: int = 2
    validation_pairs_per_batch: int = 32
    evaluation_microbatch_size: int = 16
    evaluation_lengths: tuple[int, ...] = (16, 64, 128)
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    rotor_channels: int = 9
    rotor_layers: int = 3
    rotor_expansion: int = 2
    quaternion_lanes: int = 8
    quaternion_decoder_hidden: int = 192
    mamba_hidden_size: int = 32
    mamba_layers: int = 3
    mamba_heads: int = 4
    mamba_head_dim: int = 16
    mamba_state_size: int = 24
    delta_hidden_size: int = 32
    delta_heads: int = 4
    delta_householder_updates: int = 4
    delta_intermediate_size: int = 112
    seed: int = 0


@dataclass(frozen=True)
class RelationSpec:
    """One central presentation word and its equal-length identity control."""

    key: str
    display: str
    tokens: tuple[int, ...]
    identity_tokens: tuple[int, ...]


@dataclass(frozen=True)
class MultiRelationTask:
    """A binary-icosahedral task in one conjugated generator coordinate."""

    binary: BinaryA5Task
    coordinate_label: str
    conjugator_index: int
    conjugator_inverse_index: int
    relations: tuple[RelationSpec, ...]


@dataclass(frozen=True)
class RelationPairBatch:
    """Paired central/identity words and their post-block scoring mask."""

    inputs: torch.Tensor
    targets: torch.Tensor
    post_relation_mask: torch.Tensor
    relation_key: str
    relation_position: str
    block_start: int


def _inverse_index(task: BinaryA5Task, element: int) -> int:
    table = task.group.table
    for candidate in range(task.group.order):
        if int(table[element, candidate]) == 0 and int(table[candidate, element]) == 0:
            return candidate
    raise RuntimeError(f"element {element} has no two-sided inverse")


def _conjugate(task: BinaryA5Task, element: int, conjugator: int) -> int:
    inverse = _inverse_index(task, conjugator)
    table = task.group.table
    return int(table[int(table[inverse, element]), conjugator])


def _word_product(task: BinaryA5Task, tokens: Sequence[int]) -> int:
    state = 0
    for token in tokens:
        state = int(task.group.table[state, task.input_elements[token]])
    return state


@cache
def _base_binary_task() -> BinaryA5Task:
    return binary_icosahedral_task()


@cache
def make_multirelation_task(coordinate_label: str) -> MultiRelationTask:
    """Conjugate the standard generators and add an explicit identity token."""

    if coordinate_label not in CONJUGATOR_LABELS:
        raise ValueError(f"unknown coordinate label {coordinate_label!r}")
    base = _base_binary_task()
    conjugator = CONJUGATOR_LABELS[coordinate_label]
    inverse = _inverse_index(base, conjugator)
    conjugated = tuple(
        _conjugate(base, element, conjugator) for element in base.input_elements
    )
    inputs = (*conjugated, 0)
    presentation = {
        **base.presentation,
        "coordinate_label": coordinate_label,
        "conjugator_index": conjugator,
        "conjugator_inverse_index": inverse,
        "a_index": inputs[0],
        "b_index": inputs[1],
        "b_inverse_index": inputs[2],
        "identity_input_index": inputs[3],
    }
    binary = replace(
        base,
        input_elements=inputs,
        input_symbols=("a", "b", "b_inverse", "e"),
        presentation=presentation,
    )
    relations = (
        RelationSpec("a_squared", "a^2=z", (0, 0), (3, 3)),
        RelationSpec("b_cubed", "b^3=z", (1, 1, 1), (3, 3, 3)),
        RelationSpec("ab_fifth", "(ab)^5=z", (0, 1) * 5, (3,) * 10),
    )
    for relation in relations:
        if _word_product(binary, relation.tokens) != binary.center_index:
            raise RuntimeError(f"{relation.display} failed in {coordinate_label}")
        if _word_product(binary, relation.identity_tokens) != 0:
            raise RuntimeError(f"identity control failed for {relation.key}")
    if _word_product(binary, (1, 2)) != 0:
        raise RuntimeError("conjugated b inverse relation failed")
    return MultiRelationTask(binary, coordinate_label, conjugator, inverse, relations)


def _prefix_targets(task: BinaryA5Task, tokens: np.ndarray) -> np.ndarray:
    states = np.zeros(tokens.shape[0], dtype=np.int64)
    targets = np.empty_like(tokens)
    input_elements = np.asarray(task.input_elements, dtype=np.int64)
    for position in range(tokens.shape[1]):
        states = task.group.table[states, input_elements[tokens[:, position]]]
        targets[:, position] = states
    return targets


def _occurrence_starts(word: Sequence[int], pattern: Sequence[int]) -> tuple[int, ...]:
    width = len(pattern)
    return tuple(
        start
        for start in range(len(word) - width + 1)
        if tuple(word[start : start + width]) == tuple(pattern)
    )


def relation_occurrences(
    word: Sequence[int], relations: Sequence[RelationSpec]
) -> dict[str, tuple[int, ...]]:
    return {
        relation.key: _occurrence_starts(word, relation.tokens)
        for relation in relations
    }


def _is_legal_extension(
    prefix: Sequence[int], token: int, relations: Sequence[RelationSpec]
) -> bool:
    candidate = (*prefix, token)
    return not any(
        len(candidate) >= len(relation.tokens)
        and tuple(candidate[-len(relation.tokens) :]) == relation.tokens
        for relation in relations
    )


def _sample_legal_word(
    length: int, generator: np.random.Generator, relations: Sequence[RelationSpec]
) -> tuple[int, ...]:
    word: list[int] = []
    for _ in range(length):
        choices = generator.permutation(4)
        for value in choices:
            token = int(value)
            if _is_legal_extension(word, token, relations):
                word.append(token)
                break
        else:  # pragma: no cover - identity is always a legal fallback here
            raise RuntimeError("the forbidden-word automaton has no legal extension")
    return tuple(word)


def shortest_legal_state_witnesses(
    task: MultiRelationTask,
) -> dict[int, tuple[int, ...]]:
    """Breadth-first witnesses in the language excluding all three relations."""

    cached = _WITNESS_CACHE.get(task.coordinate_label)
    if cached is not None:
        return dict(cached)
    max_suffix = max(len(relation.tokens) for relation in task.relations) - 1
    queue: deque[tuple[int, tuple[int, ...], tuple[int, ...]]] = deque([(0, (), ())])
    seen = {(0, ())}
    witnesses: dict[int, tuple[int, ...]] = {}
    while queue and len(witnesses) < task.binary.group.order:
        state, suffix, word = queue.popleft()
        witnesses.setdefault(state, word)
        for token in range(4):
            if not _is_legal_extension(suffix, token, task.relations):
                continue
            next_state = int(
                task.binary.group.table[state, task.binary.input_elements[token]]
            )
            next_suffix = (*suffix, token)[-max_suffix:]
            key = (next_state, next_suffix)
            if key not in seen:
                seen.add(key)
                queue.append((next_state, next_suffix, (*word, token)))
    _WITNESS_CACHE[task.coordinate_label] = dict(witnesses)
    return witnesses


def make_training_batches(
    task: MultiRelationTask, config: MultiRelationConfig
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Create an immutable relation-free schedule and inject state witnesses."""

    generator = np.random.default_rng(31_415 + config.seed)
    batches = []
    for _ in range(config.steps):
        inputs = np.asarray(
            [
                _sample_legal_word(config.training_length, generator, task.relations)
                for _ in range(config.batch_size)
            ],
            dtype=np.int64,
        )
        targets = _prefix_targets(task.binary, inputs)
        batches.append((torch.from_numpy(inputs), torch.from_numpy(targets)))

    coordinate_witnesses = shortest_legal_state_witnesses(task)
    if len(coordinate_witnesses) != task.binary.group.order:
        raise RuntimeError("the relation-free language does not reach all 120 states")
    # Inject one canonical word schedule in every conjugated coordinate.  Group
    # conjugation bijectively relabels the reached states, so this preserves full
    # coverage without changing token IDs between coordinate controls.
    injection_witnesses = shortest_legal_state_witnesses(make_multirelation_task("e"))
    maximum_length = max(map(len, injection_witnesses.values()))
    if maximum_length > config.training_length:
        raise ValueError(
            f"training length {config.training_length} is shorter than witness "
            f"length {maximum_length}"
        )
    if config.steps * config.batch_size < len(injection_witnesses):
        raise ValueError("the schedule has too few rows for state-witness injection")
    for row_index, (_, word) in enumerate(sorted(injection_witnesses.items())):
        batch_index, row = divmod(row_index, config.batch_size)
        inputs = batches[batch_index][0].numpy()
        inputs[row] = 3
        inputs[row, : len(word)] = word
        batches[batch_index] = (
            batches[batch_index][0],
            torch.from_numpy(_prefix_targets(task.binary, inputs)),
        )
    return batches


def _schedule_sha256(
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]], *, inputs_only: bool
) -> str:
    digest = hashlib.sha256()
    for inputs, targets in batches:
        tensors = (inputs,) if inputs_only else (inputs, targets)
        for tensor in tensors:
            values = tensor.detach().cpu().contiguous()
            digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
            digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def training_split_audit(
    task: MultiRelationTask,
    training: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, object]:
    occurrence_counts = Counter({relation.key: 0 for relation in task.relations})
    for inputs, _ in training:
        for row in inputs.tolist():
            for key, starts in relation_occurrences(row, task.relations).items():
                occurrence_counts[key] += len(starts)
    if any(occurrence_counts.values()):
        raise RuntimeError(
            f"central relation leaked into training: {occurrence_counts}"
        )
    witnesses = shortest_legal_state_witnesses(task)
    lengths = Counter(map(len, witnesses.values()))
    targets = torch.cat([target.flatten() for _, target in training])
    target_states = {int(value) for value in targets.tolist()}
    projective_states = {task.binary.projective_label[state] for state in target_states}
    center_bits = {task.binary.center_bit[state] for state in target_states}
    return {
        "forbidden_relation_occurrences": dict(occurrence_counts),
        "exact_training_language_coverage": {
            "reachable_binary_states": len(witnesses),
            "all_binary_states_reachable": len(witnesses) == 120,
            "maximum_shortest_witness_length": max(map(len, witnesses.values())),
            "shortest_witness_length_distribution": {
                str(length): count for length, count in sorted(lengths.items())
            },
        },
        "realized_training_coverage": state_and_pair_coverage_audit(
            list(training), input_order=4, group_order=120
        ),
        "observed_binary_target_states": len(target_states),
        "observed_projective_target_states": len(projective_states),
        "observed_center_bits": sorted(center_bits),
        "input_schedule_sha256": _schedule_sha256(training, inputs_only=True),
        "input_and_target_schedule_sha256": _schedule_sha256(
            training, inputs_only=False
        ),
    }


def make_relation_pair_batches(
    task: MultiRelationTask,
    relation: RelationSpec,
    config: MultiRelationConfig,
    length: int,
    relation_position: str,
) -> list[RelationPairBatch]:
    """Create pairs with exactly one forced central relation and no leakage."""

    width = len(relation.tokens)
    if length < width:
        raise ValueError(f"length {length} is shorter than {relation.key}")
    if relation_position not in {"early", "late"}:
        raise ValueError("relation_position must be early or late")
    block_start = 0 if relation_position == "early" else length - width
    batches = []
    relation_offset = next(
        index for index, candidate in enumerate(task.relations) if candidate == relation
    )
    for batch_index in range(config.validation_batches):
        generator = np.random.default_rng(
            271_828
            + 10_000 * config.seed
            + 1_000 * length
            + 100 * relation_offset
            + 10 * (relation_position == "late")
            + batch_index
        )
        center_words = []
        identity_words = []
        attempts = 0
        while len(center_words) < config.validation_pairs_per_batch:
            attempts += 1
            if attempts > 100_000:
                raise RuntimeError("could not sample audited relation contexts")
            left = _sample_legal_word(block_start, generator, task.relations)
            right = _sample_legal_word(
                length - block_start - width, generator, task.relations
            )
            center = (*left, *relation.tokens, *right)
            identity = (*left, *relation.identity_tokens, *right)
            center_counts = relation_occurrences(center, task.relations)
            identity_counts = relation_occurrences(identity, task.relations)
            if any(identity_counts[key] for key in identity_counts):
                continue
            if any(
                len(starts) != (1 if key == relation.key else 0)
                for key, starts in center_counts.items()
            ):
                continue
            if center_counts[relation.key][0] != block_start:
                continue
            center_words.append(center)
            identity_words.append(identity)
        interleaved = np.empty(
            (2 * config.validation_pairs_per_batch, length), dtype=np.int64
        )
        interleaved[0::2] = np.asarray(center_words)
        interleaved[1::2] = np.asarray(identity_words)
        targets = _prefix_targets(task.binary, interleaved)
        mask = np.zeros_like(interleaved, dtype=bool)
        mask[:, block_start + width - 1 :] = True
        batches.append(
            RelationPairBatch(
                torch.from_numpy(interleaved),
                torch.from_numpy(targets),
                torch.from_numpy(mask),
                relation.key,
                relation_position,
                block_start,
            )
        )
    return batches


def relation_pair_evaluation_audit(
    task: MultiRelationTask,
    relation: RelationSpec,
    batches: Sequence[RelationPairBatch],
) -> dict[str, object]:
    counts = Counter({candidate.key: 0 for candidate in task.relations})
    identity_counts = Counter({candidate.key: 0 for candidate in task.relations})
    total_pairs = post_positions = central_checks = projective_checks = 0
    shared_context_checks = forced_block_checks = 0
    digest = hashlib.sha256()
    for batch in batches:
        for tensor in (batch.inputs, batch.targets, batch.post_relation_mask):
            values = tensor.detach().cpu().contiguous()
            digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
            digest.update(values.numpy().tobytes())
        inputs = batch.inputs.numpy()
        targets = batch.targets.numpy()
        masks = batch.post_relation_mask.numpy()
        width = len(relation.tokens)
        total_pairs += len(inputs) // 2
        forced_block_checks += int(
            np.all(
                inputs[0::2, batch.block_start : batch.block_start + width]
                == np.asarray(relation.tokens),
                axis=1,
            ).sum()
        )
        forced_block_checks += int(
            np.all(
                inputs[1::2, batch.block_start : batch.block_start + width]
                == np.asarray(relation.identity_tokens),
                axis=1,
            ).sum()
        )
        for row_index, row in enumerate(inputs):
            row_counts = relation_occurrences(row.tolist(), task.relations)
            destination = counts if row_index % 2 == 0 else identity_counts
            for key, starts in row_counts.items():
                destination[key] += len(starts)
        for center_row in range(0, len(inputs), 2):
            identity_row = center_row + 1
            center_context = np.delete(
                inputs[center_row],
                np.s_[batch.block_start : batch.block_start + width],
            )
            identity_context = np.delete(
                inputs[identity_row],
                np.s_[batch.block_start : batch.block_start + width],
            )
            shared_context_checks += int(
                np.array_equal(center_context, identity_context)
            )
            for position in np.flatnonzero(masks[center_row]):
                post_positions += 1
                center_target = int(targets[center_row, position])
                identity_target = int(targets[identity_row, position])
                central_checks += int(
                    task.binary.central_partner[identity_target] == center_target
                )
                projective_checks += int(
                    task.binary.projective_label[identity_target]
                    == task.binary.projective_label[center_target]
                )
    expected_relation_counts = {
        candidate.key: total_pairs if candidate.key == relation.key else 0
        for candidate in task.relations
    }
    passed = (
        dict(counts) == expected_relation_counts
        and not any(identity_counts.values())
        and central_checks == post_positions
        and projective_checks == post_positions
        and shared_context_checks == total_pairs
        and forced_block_checks == 2 * total_pairs
    )
    return {
        "relation": relation.display,
        "paired_sequences": total_pairs,
        "post_relation_pair_positions": post_positions,
        "center_branch_relation_occurrences": dict(counts),
        "identity_branch_relation_occurrences": dict(identity_counts),
        "exact_central_partner_checks": central_checks,
        "projective_match_checks": projective_checks,
        "shared_context_checks": shared_context_checks,
        "forced_block_checks": forced_block_checks,
        "input_schedule_sha256": _pair_schedule_sha256(batches, inputs_only=True),
        "input_target_mask_schedule_sha256": digest.hexdigest(),
        "passed": passed,
    }


def _pair_schedule_sha256(
    batches: Sequence[RelationPairBatch], *, inputs_only: bool
) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        tensors = (
            (batch.inputs,)
            if inputs_only
            else (batch.inputs, batch.targets, batch.post_relation_mask)
        )
        for tensor in tensors:
            values = tensor.detach().cpu().contiguous()
            digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
            digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def _spin_model(
    task: BinaryA5Task, config: MultiRelationConfig
) -> SpinQuaternionScanModel:
    return SpinQuaternionScanModel(
        input_order=len(task.input_elements),
        output_order=task.group.order,
        lanes=config.quaternion_lanes,
        decoder_hidden=config.quaternion_decoder_hidden,
    )


def _delta_model(
    task: BinaryA5Task, config: MultiRelationConfig
) -> DeltaProductReferenceModel:
    return DeltaProductReferenceModel(
        input_vocab_size=task.group.order,
        output_size=task.group.order,
        hidden_size=config.delta_hidden_size,
        num_heads=config.delta_heads,
        num_householder=config.delta_householder_updates,
        intermediate_size=config.delta_intermediate_size,
    )


def build_models(
    task: BinaryA5Task, config: MultiRelationConfig
) -> dict[str, nn.Module]:
    """Build all trained candidates and enforce the parameter-near contract."""

    models: dict[str, nn.Module] = {
        "pure_rotor": pure_rotor_model(task, config, max_rotor_angle=math.pi),
        "identity_rotation_ablation": pure_rotor_model(
            task, config, max_rotor_angle=0.0
        ),
        "spin_quaternion_scan": _spin_model(task, config),
        "mamba2_transformers": mamba2_model(task, config),
        "delta_product_reference": _delta_model(task, config),
    }
    counts = {name: parameter_count(model) for name, model in models.items()}
    relative_gap = (max(counts.values()) - min(counts.values())) / max(counts.values())
    if relative_gap > 0.02:
        raise RuntimeError(f"candidate parameter gap exceeds 2%: {counts}")
    return models


def logits_for(
    name: str,
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    rotor_scan_mode: str,
    quaternion_scan_mode: str,
    delta_scan_mode: str,
) -> torch.Tensor:
    if name == "mamba2_transformers":
        return model(input_ids=inputs, use_cache=False).logits
    if name == "spin_quaternion_scan":
        return model(inputs, scan_mode=quaternion_scan_mode)
    if name == "delta_product_reference":
        return model(inputs, scan_mode=delta_scan_mode)
    if name == "exact_regular_pd_oracle":
        return model(inputs)
    return model(inputs, scan_mode=rotor_scan_mode)


@torch.no_grad()
def evaluate_relation_pairs(
    name: str,
    model: nn.Module | None,
    batches: Sequence[RelationPairBatch],
    task: BinaryA5Task,
    device: torch.device,
    config: MultiRelationConfig,
    *,
    rotor_scan_mode: str,
    quaternion_scan_mode: str,
    delta_scan_mode: str,
) -> dict[str, float]:
    if model is not None:
        model.eval()
    projective = torch.tensor(task.projective_label, dtype=torch.long)
    center_bit = torch.tensor(task.center_bit, dtype=torch.long)
    partner = torch.tensor(task.central_partner, dtype=torch.long)
    totals = CounterMetrics()
    for batch in batches:
        pieces = []
        for start in range(0, len(batch.inputs), config.evaluation_microbatch_size):
            inputs = batch.inputs[start : start + config.evaluation_microbatch_size]
            if model is None:
                logits = oracle_logits(name, inputs, task)
            else:
                logits = logits_for(
                    name,
                    model,
                    inputs.to(device),
                    rotor_scan_mode=rotor_scan_mode,
                    quaternion_scan_mode=quaternion_scan_mode,
                    delta_scan_mode=delta_scan_mode,
                ).cpu()
            pieces.append(logits)
        logits = torch.cat(pieces)
        predictions = logits.argmax(dim=-1)
        mask = batch.post_relation_mask
        targets = batch.targets
        selected_logits = logits[mask]
        selected_targets = targets[mask]
        selected_predictions = predictions[mask]
        selected_partners = partner[selected_targets]
        target_logits = selected_logits.gather(1, selected_targets[:, None]).squeeze(1)
        partner_logits = selected_logits.gather(1, selected_partners[:, None]).squeeze(
            1
        )
        margins = target_logits - partner_logits
        totals.nll_sum += float(
            F.cross_entropy(selected_logits, selected_targets, reduction="sum")
        )
        totals.positions += selected_targets.numel()
        totals.exact += int((selected_predictions == selected_targets).sum())
        totals.projective += int(
            (projective[selected_predictions] == projective[selected_targets]).sum()
        )
        totals.center_bit += int(
            (center_bit[selected_predictions] == center_bit[selected_targets]).sum()
        )
        totals.margin_score += float(
            (margins > 0).float().sum() + 0.5 * (margins == 0).float().sum()
        )
        totals.central_probability += float(torch.sigmoid(margins).sum())

        final_predictions = predictions[:, -1]
        final_targets = targets[:, -1]
        totals.final_rows += len(final_targets)
        totals.final_exact += int((final_predictions == final_targets).sum())
        totals.final_projective += int(
            (projective[final_predictions] == projective[final_targets]).sum()
        )
        totals.final_center_bit += int(
            (center_bit[final_predictions] == center_bit[final_targets]).sum()
        )
        center_predictions = final_predictions[0::2]
        identity_predictions = final_predictions[1::2]
        center_targets = final_targets[0::2]
        identity_targets = final_targets[1::2]
        totals.final_pairs += len(center_targets)
        totals.paired_exact += int(
            (
                (center_predictions == center_targets)
                & (identity_predictions == identity_targets)
            ).sum()
        )
        totals.paired_structural_separation += int(
            (
                (projective[center_predictions] == projective[identity_predictions])
                & (center_bit[center_predictions] != center_bit[identity_predictions])
            ).sum()
        )
    return totals.as_metrics()


def _evaluate_all(
    name: str,
    model: nn.Module | None,
    evaluations: dict[str, list[RelationPairBatch]],
    task: BinaryA5Task,
    config: MultiRelationConfig,
    device: torch.device,
    *,
    rotor_scan_mode: str,
    quaternion_scan_mode: str,
    delta_scan_mode: str,
) -> dict[str, dict[str, float]]:
    return {
        key: evaluate_relation_pairs(
            name,
            model,
            batches,
            task,
            device,
            config,
            rotor_scan_mode=rotor_scan_mode,
            quaternion_scan_mode=quaternion_scan_mode,
            delta_scan_mode=delta_scan_mode,
        )
        for key, batches in evaluations.items()
    }


def _checkpoint_payload(
    name: str,
    model: nn.Module,
    task: MultiRelationTask,
    config: MultiRelationConfig,
    result: dict[str, object],
) -> dict[str, object]:
    return {
        "format_version": 1,
        "candidate": name,
        "pure_rotor_model_version": PURE_ROTOR_VERSION,
        "coordinate_label": task.coordinate_label,
        "conjugator_index": task.conjugator_index,
        "task_presentation": task.binary.presentation,
        "group_table_sha256": task.binary.group_table_sha256,
        "benchmark_config": asdict(config),
        "metrics": result,
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }


def train_one(
    name: str,
    make_model: Callable[[], nn.Module],
    task: MultiRelationTask,
    training: Sequence[tuple[torch.Tensor, torch.Tensor]],
    evaluations: dict[str, list[RelationPairBatch]],
    config: MultiRelationConfig,
    device: torch.device,
    *,
    rotor_scan_mode: str,
    quaternion_scan_mode: str,
    delta_scan_mode: str,
    checkpoint_directory: Path | None,
) -> dict[str, object]:
    seed_everything(config.seed)
    model = make_model().to(device)
    initial = _evaluate_all(
        name,
        model,
        evaluations,
        task.binary,
        config,
        device,
        rotor_scan_mode=rotor_scan_mode,
        quaternion_scan_mode=quaternion_scan_mode,
        delta_scan_mode=delta_scan_mode,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    loss_samples: dict[str, float] = {}
    start = time.perf_counter()
    gradient_norm = torch.tensor(0.0)
    for step, (inputs, targets) in enumerate(training, start=1):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = logits_for(
            name,
            model,
            inputs,
            rotor_scan_mode=rotor_scan_mode,
            quaternion_scan_mode=quaternion_scan_mode,
            delta_scan_mode=delta_scan_mode,
        )
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == config.steps:
            loss_samples[str(step)] = float(loss.detach())
            print(
                f"{task.coordinate_label}/{name} seed={config.seed} "
                f"step={step}/{config.steps} loss={loss_samples[str(step)]:.5f}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    final = _evaluate_all(
        name,
        model,
        evaluations,
        task.binary,
        config,
        device,
        rotor_scan_mode=rotor_scan_mode,
        quaternion_scan_mode=quaternion_scan_mode,
        delta_scan_mode=delta_scan_mode,
    )
    result: dict[str, object] = {
        "name": name,
        "seed": config.seed,
        "coordinate_label": task.coordinate_label,
        "conjugator_index": task.conjugator_index,
        "parameters": parameter_count(model),
        "initial_relation_metrics": initial,
        "final_relation_metrics": final,
        "final_train_loss": loss_samples[str(config.steps)],
        "loss_samples": loss_samples,
        "last_preclip_gradient_norm": float(gradient_norm),
        "elapsed_seconds": elapsed,
        "tokens_per_second": (
            config.steps * config.batch_size * config.training_length / elapsed
        ),
        "peak_cuda_memory_mib": (
            float(torch.cuda.max_memory_allocated(device) / 2**20)
            if device.type == "cuda"
            else 0.0
        ),
    }
    if checkpoint_directory is not None:
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        path = checkpoint_directory / (
            f"{name}_coord-{task.coordinate_label}_seed{config.seed}_"
            f"step{config.steps}.pt"
        )
        torch.save(_checkpoint_payload(name, model, task, config, result), path)
        result["checkpoint"] = str(path)
        result["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def run_benchmark(
    config: MultiRelationConfig,
    *,
    seeds: Sequence[int],
    coordinate_labels: Sequence[str],
    candidates: Sequence[str],
    device: torch.device,
    rotor_scan_mode: str,
    quaternion_scan_mode: str,
    delta_scan_mode: str,
    checkpoint_directory: Path | None = None,
) -> dict[str, object]:
    unknown = set(candidates) - set(TRAINED_CANDIDATES)
    if unknown:
        raise ValueError(f"unknown candidates: {sorted(unknown)}")
    started = datetime.now(ZoneInfo("Africa/Johannesburg"))
    results = []
    oracles: dict[str, object] = {}
    split_audits: dict[str, object] = {}
    parameter_counts: dict[str, int] | None = None
    input_schedule_hashes: dict[int, set[str]] = {seed: set() for seed in seeds}
    evaluation_input_hashes: dict[tuple[int, str], set[str]] = {}
    coordinate_metadata = {}
    for coordinate_label in coordinate_labels:
        task = make_multirelation_task(coordinate_label)
        coordinate_metadata[coordinate_label] = {
            "conjugator_index": task.conjugator_index,
            "conjugator_inverse_index": task.conjugator_inverse_index,
            "input_elements": list(task.binary.input_elements),
            "presentation": task.binary.presentation,
        }
        shapes = build_models(task.binary, config)
        counts = {name: parameter_count(model) for name, model in shapes.items()}
        if parameter_counts is None:
            parameter_counts = counts
        elif counts != parameter_counts:
            raise RuntimeError("model parameter counts changed across coordinates")
        del shapes
        for seed in seeds:
            run_config = replace(config, seed=seed)
            training = make_training_batches(task, run_config)
            training_audit = training_split_audit(task, training)
            input_schedule_hashes[seed].add(training_audit["input_schedule_sha256"])
            evaluations: dict[str, list[RelationPairBatch]] = {}
            evaluation_audits: dict[str, object] = {}
            for relation in task.relations:
                for length in run_config.evaluation_lengths:
                    for position in ("early", "late"):
                        key = f"{relation.key}__{position}_L{length}"
                        batches = make_relation_pair_batches(
                            task, relation, run_config, length, position
                        )
                        audit = relation_pair_evaluation_audit(task, relation, batches)
                        if not audit["passed"]:
                            raise RuntimeError(f"evaluation audit failed for {key}")
                        evaluations[key] = batches
                        evaluation_audits[key] = audit
                        hash_key = (seed, key)
                        evaluation_input_hashes.setdefault(hash_key, set()).add(
                            audit["input_schedule_sha256"]
                        )
            split_key = f"{coordinate_label}/seed{seed}"
            split_audits[split_key] = {
                "training": training_audit,
                "evaluations": evaluation_audits,
            }
            exact_pd = ExactRegularPD(task.binary.group, task.binary.input_elements).to(
                device
            )
            oracles[split_key] = {
                **{
                    name: _evaluate_all(
                        name,
                        None,
                        evaluations,
                        task.binary,
                        run_config,
                        device,
                        rotor_scan_mode=rotor_scan_mode,
                        quaternion_scan_mode=quaternion_scan_mode,
                        delta_scan_mode=delta_scan_mode,
                    )
                    for name in ANALYTIC_ORACLES
                },
                "exact_regular_pd_oracle": _evaluate_all(
                    "exact_regular_pd_oracle",
                    exact_pd,
                    evaluations,
                    task.binary,
                    run_config,
                    device,
                    rotor_scan_mode=rotor_scan_mode,
                    quaternion_scan_mode=quaternion_scan_mode,
                    delta_scan_mode=delta_scan_mode,
                ),
            }
            factories: dict[str, Callable[[], nn.Module]] = {
                "pure_rotor": lambda task=task, config=run_config: pure_rotor_model(
                    task.binary, config, max_rotor_angle=math.pi
                ),
                "identity_rotation_ablation": (
                    lambda task=task, config=run_config: pure_rotor_model(
                        task.binary, config, max_rotor_angle=0.0
                    )
                ),
                "spin_quaternion_scan": (
                    lambda task=task, config=run_config: _spin_model(
                        task.binary, config
                    )
                ),
                "mamba2_transformers": (
                    lambda task=task, config=run_config: mamba2_model(
                        task.binary, config
                    )
                ),
                "delta_product_reference": (
                    lambda task=task, config=run_config: _delta_model(
                        task.binary, config
                    )
                ),
            }
            for name in candidates:
                results.append(
                    train_one(
                        name,
                        factories[name],
                        task,
                        training,
                        evaluations,
                        run_config,
                        device,
                        rotor_scan_mode=rotor_scan_mode,
                        quaternion_scan_mode=quaternion_scan_mode,
                        delta_scan_mode=delta_scan_mode,
                        checkpoint_directory=checkpoint_directory,
                    )
                )
    assert parameter_counts is not None
    maximum_count = max(parameter_counts.values())
    finished = datetime.now(ZoneInfo("Africa/Johannesburg"))
    return {
        "experiment": "multi-relation center-sensitive 2.A5 prefix tracking",
        "status": "completed empirical pilot; exact controls are not trained models",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "pure_rotor_model_version": PURE_ROTOR_VERSION,
        "device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
        ),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "config": asdict(config),
        "seeds": list(seeds),
        "coordinates": list(coordinate_labels),
        "candidates": list(candidates),
        "scan_modes": {
            "pure_rotor": rotor_scan_mode,
            "spin_quaternion": quaternion_scan_mode,
            "delta_product": delta_scan_mode,
        },
        "task": {
            "group": {"key": "2a5", "order": 120, "projective_order": 60},
            "group_table_sha256": make_multirelation_task(
                coordinate_labels[0]
            ).binary.group_table_sha256,
            "input_symbols": ["a", "b", "b_inverse", "e"],
            "withheld_relations": [
                asdict(relation)
                for relation in make_multirelation_task(coordinate_labels[0]).relations
            ],
            "coordinate_metadata": coordinate_metadata,
            "splits": split_audits,
        },
        "integrity": {
            "same_input_training_schedule_across_coordinates": all(
                len(hashes) == 1 for hashes in input_schedule_hashes.values()
            ),
            "same_input_evaluation_schedule_across_coordinates": all(
                len(hashes) == 1 for hashes in evaluation_input_hashes.values()
            ),
            "candidate_initialized_after_seed": True,
            "parameter_counts": parameter_counts,
            "maximum_relative_parameter_gap": (
                maximum_count - min(parameter_counts.values())
            )
            / maximum_count,
            "recurrent_state_scalars": {
                "pure_rotor_and_identity": (
                    config.rotor_layers * config.rotor_channels * 8
                ),
                "spin_quaternion_scan": config.quaternion_lanes * 4,
                "delta_product_reference": (
                    config.delta_heads
                    * (config.delta_hidden_size // config.delta_heads) ** 2
                ),
                "exact_regular_pd_oracle_real_equivalent": 240,
                "mamba2_transformers": "architecture-specific cache; not state matched",
            },
            "state_size_matched": False,
            "delta_product_reference": {
                "source_commit": DELTA_PRODUCT_SOURCE_COMMIT,
                "official_fla_or_triton_kernel_used": False,
                "equation_faithful_unfused_reference": True,
                "fla_importable": importlib.util.find_spec("fla") is not None,
            },
            "pd_ssm_exact_regular_oracle": {
                "source_commit_reviewed": PD_SSM_SOURCE_COMMIT,
                "complex_state_size": 120,
                "trained": False,
            },
        },
        "oracle_results": oracles,
        "results": results,
        "claim_scope": {
            "empirical": [
                "acquisition and retention of three held-out central relations",
                "robustness under three conjugated generating sets",
                "parameter-near learned-model comparison on identical token schedules",
            ],
            "not_claimed": [
                "a theorem about Mamba-2, DeltaProduct, PD-SSM, or diagonal SSMs",
                "official DeltaProduct fused-kernel throughput",
                "a state-matched comparison",
                "language-model quality",
            ],
        },
    }


def _parse_csv(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed or len(set(parsed)) != len(parsed):
        raise ValueError("CSV arguments must be nonempty and distinct")
    return parsed


def _parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _parse_csv(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--training-length", type=int, default=16)
    parser.add_argument("--validation-batches", type=int, default=2)
    parser.add_argument("--validation-pairs-per-batch", type=int, default=32)
    parser.add_argument("--evaluation-microbatch-size", type=int, default=16)
    parser.add_argument("--evaluation-lengths", default="16,64,128")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--coordinates", default="e,a,b")
    parser.add_argument("--candidates", default=",".join(TRAINED_CANDIDATES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--rotor-scan-mode",
        choices=("parallel", "schur_parallel", "recurrent"),
        default="schur_parallel",
    )
    parser.add_argument(
        "--quaternion-scan-mode",
        choices=("parallel", "recurrent"),
        default="parallel",
    )
    parser.add_argument(
        "--delta-scan-mode", choices=("parallel", "recurrent"), default="parallel"
    )
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quiet-report", action="store_true")
    args = parser.parse_args()
    positive = (
        args.steps,
        args.batch_size,
        args.training_length,
        args.validation_batches,
        args.validation_pairs_per_batch,
        args.evaluation_microbatch_size,
    )
    if min(positive) < 1:
        raise ValueError("steps, batch sizes, and lengths must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    lengths = _parse_int_csv(args.evaluation_lengths)
    if any(length < 10 for length in lengths):
        raise ValueError("all evaluation lengths must fit the length-10 relation")
    coordinates = _parse_csv(args.coordinates)
    if set(coordinates) - set(CONJUGATOR_LABELS):
        raise ValueError("coordinates must be selected from e,a,b")
    config = MultiRelationConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        training_length=args.training_length,
        validation_batches=args.validation_batches,
        validation_pairs_per_batch=args.validation_pairs_per_batch,
        evaluation_microbatch_size=args.evaluation_microbatch_size,
        evaluation_lengths=lengths,
    )
    report = run_benchmark(
        config,
        seeds=_parse_int_csv(args.seeds),
        coordinate_labels=coordinates,
        candidates=_parse_csv(args.candidates),
        device=torch.device(args.device),
        rotor_scan_mode=args.rotor_scan_mode,
        quaternion_scan_mode=args.quaternion_scan_mode,
        delta_scan_mode=args.delta_scan_mode,
        checkpoint_directory=args.checkpoint_directory,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not args.quiet_report:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
