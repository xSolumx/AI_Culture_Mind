"""Rigid-motion plus central-sign benchmark for Spin and motor scans.

The task crosses the binary icosahedral group ``2.A5`` with local-frame
translations.  Rotation tokens are ``a``, ``b``, and ``b_inverse``; ``tx``,
``ty``, and ``tz`` translate in the *current body frame*.  Consequently the
translation recurrence is non-commutative with the rotations.  Training omits
all three central presentation words ``a^2``, ``b^3``, and ``(ab)^5`` while an
audited legal language still reaches all 120 binary rotation states.

At evaluation a held-out central word is paired with an equal-width identity
block in an otherwise identical context.  The two branches have identical
physical ``SE(3)`` poses but antipodal signed quaternions.  This simultaneously
tests physical pose tracking and retention of the Spin double-cover bit.

This is a controlled pilot, not a theorem about model families and not a
fused-kernel systems comparison.  The Transformers Mamba-2 path and the local
DeltaProduct reference both use unfused PyTorch execution on this Windows
checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from benchmark_pure_rotor_2a5 import parameter_count, seed_everything
from benchmark_spin_multirelation_2a5 import (
    DELTA_PRODUCT_SOURCE_COMMIT,
    MultiRelationTask,
    RelationSpec,
    _is_legal_extension,
    make_multirelation_task,
    relation_occurrences,
    shortest_legal_state_witnesses,
)
from delta_product_reference import DeltaProductReferenceModel
from pure_rotor_ssm.motor_scan import (
    DirectMotorPoseTracker,
    DirectProductPoseTracker,
    MotorCompositionClassifier,
    quaternion_conjugate,
)
from pure_rotor_ssm.spin_scan import (
    SpinCompositionClassifier,
    quaternion_product,
    unit_quaternion,
)
from torch import nn
from transformers import Mamba2Config, Mamba2ForCausalLM

DIRECT_CANDIDATES = (
    "direct_product_pose_scan",
    "direct_motor_pose_scan",
)
READOUT_CANDIDATES = (
    "spin_quaternion_scan",
    "spin_motor_scan",
    "mamba2_transformers",
    "delta_product_reference",
)
TRAINED_CANDIDATES = (*DIRECT_CANDIDATES, *READOUT_CANDIDATES)
ORACLE_CANDIDATES = (
    "exact_spin_motor_oracle",
    "se3_quotient_oracle",
    "spin_only_oracle",
)
INPUT_SYMBOLS = ("a", "b", "b_inverse", "e", "tx", "ty", "tz")
TRANSLATION_TOKEN_INDICES = (4, 5, 6)
FP_RNN_SOURCE_COMMIT = "0cc1e3c520423e02674c20333fcf9dfa46b7d204"
MAMBA2_PAPER = "https://arxiv.org/abs/2405.21060"
FP_RNN_PAPER = "https://arxiv.org/abs/2503.10799"
PATH_DEVELOPMENT_PAPER = "https://arxiv.org/abs/2204.00740"


@dataclass(frozen=True)
class RigidSpinConfig:
    """Frozen parameter-near pilot configuration."""

    steps: int = 300
    batch_size: int = 16
    training_length: int = 16
    validation_batches: int = 2
    validation_pairs_per_batch: int = 16
    evaluation_microbatch_size: int = 16
    evaluation_lengths: tuple[int, ...] = (16, 64, 128)
    translation_step: float = 0.25
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    translation_loss_weight: float = 1.0
    quaternion_norm_weight: float = 0.01
    signed_rotation_threshold_degrees: float = 15.0
    physical_rotation_threshold_degrees: float = 15.0
    translation_threshold: float = 0.10
    spin_lanes: int = 8
    spin_decoder_hidden: int = 547
    motor_lanes: int = 4
    motor_decoder_hidden: int = 548
    mamba_hidden_size: int = 32
    mamba_layers: int = 3
    mamba_heads: int = 4
    mamba_head_dim: int = 16
    mamba_state_size: int = 8
    delta_hidden_size: int = 32
    delta_heads: int = 4
    delta_householder_updates: int = 4
    delta_intermediate_size: int = 112
    seed: int = 0


@dataclass(frozen=True)
class RigidSpinTask:
    """Exact binary rotation task augmented by body-frame translations."""

    rotation: MultiRelationTask
    input_symbols: tuple[str, ...]
    input_elements: tuple[int, ...]
    token_translations: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class RigidTrainingBatch:
    inputs: torch.Tensor
    group_targets: torch.Tensor
    pose_targets: torch.Tensor


@dataclass(frozen=True)
class RigidRelationPairBatch:
    inputs: torch.Tensor
    group_targets: torch.Tensor
    pose_targets: torch.Tensor
    post_relation_mask: torch.Tensor
    relation_key: str
    relation_position: str
    block_start: int


def make_rigid_spin_task(
    coordinate_label: str, translation_step: float
) -> RigidSpinTask:
    """Create a coordinate-controlled ``2.A5``-by-translation task."""

    if not math.isfinite(translation_step) or translation_step <= 0:
        raise ValueError("translation_step must be finite and positive")
    rotation = make_multirelation_task(coordinate_label)
    input_elements = (*rotation.binary.input_elements, 0, 0, 0)
    if len(input_elements) != len(INPUT_SYMBOLS):
        raise AssertionError("rigid task vocabulary construction failed")
    zero = (0.0, 0.0, 0.0)
    token_translations = (
        zero,
        zero,
        zero,
        zero,
        (translation_step, 0.0, 0.0),
        (0.0, translation_step, 0.0),
        (0.0, 0.0, translation_step),
    )
    return RigidSpinTask(
        rotation=rotation,
        input_symbols=INPUT_SYMBOLS,
        input_elements=input_elements,
        token_translations=token_translations,
    )


def _rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector by a scalar-first unit quaternion."""

    imaginary = quaternion[1:]
    twice_cross = 2.0 * np.cross(imaginary, vector)
    return vector + quaternion[0] * twice_cross + np.cross(imaginary, twice_cross)


def rigid_prefix_targets(
    task: RigidSpinTask, inputs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact rotation labels and float64 signed-pose prefix targets."""

    if inputs.ndim != 2 or inputs.shape[1] == 0:
        raise ValueError("inputs must have nonempty shape (batch,length)")
    if inputs.min() < 0 or inputs.max() >= len(task.input_symbols):
        raise ValueError("inputs contain an out-of-vocabulary token")
    binary = task.rotation.binary
    quaternions = np.asarray(binary.quaternions, dtype=np.float64)
    translations = np.asarray(task.token_translations, dtype=np.float64)
    input_elements = np.asarray(task.input_elements, dtype=np.int64)
    group_targets = np.empty_like(inputs, dtype=np.int64)
    pose_targets = np.empty((*inputs.shape, 7), dtype=np.float64)
    states = np.zeros(inputs.shape[0], dtype=np.int64)
    positions = np.zeros((inputs.shape[0], 3), dtype=np.float64)
    for position in range(inputs.shape[1]):
        token_ids = inputs[:, position]
        local_steps = translations[token_ids]
        for row in range(inputs.shape[0]):
            positions[row] += _rotate_vector(quaternions[states[row]], local_steps[row])
        states = binary.group.table[states, input_elements[token_ids]]
        group_targets[:, position] = states
        pose_targets[:, position, :4] = quaternions[states]
        pose_targets[:, position, 4:] = positions
    return group_targets, pose_targets


def _sample_legal_rigid_word(
    length: int,
    generator: np.random.Generator,
    relations: Sequence[RelationSpec],
) -> tuple[int, ...]:
    word: list[int] = []
    for _ in range(length):
        for value in generator.permutation(len(INPUT_SYMBOLS)):
            token = int(value)
            if _is_legal_extension(word, token, relations):
                word.append(token)
                break
        else:  # pragma: no cover - identity and translations remain legal
            raise RuntimeError("the forbidden-word automaton has no legal extension")
    return tuple(word)


def make_training_batches(
    task: RigidSpinTask, config: RigidSpinConfig
) -> list[RigidTrainingBatch]:
    """Create one immutable legal schedule and inject all rotation witnesses."""

    if config.steps * config.batch_size < task.rotation.binary.group.order:
        raise ValueError("training schedule is too small for 120 witness rows")
    generator = np.random.default_rng(73_939 + config.seed)
    input_batches = [
        np.asarray(
            [
                _sample_legal_rigid_word(
                    config.training_length, generator, task.rotation.relations
                )
                for _ in range(config.batch_size)
            ],
            dtype=np.int64,
        )
        for _ in range(config.steps)
    ]
    witnesses = shortest_legal_state_witnesses(task.rotation)
    if len(witnesses) != task.rotation.binary.group.order:
        raise RuntimeError("legal rotation language does not reach all 120 states")
    if max(map(len, witnesses.values())) > config.training_length:
        raise ValueError("training_length is shorter than a legal state witness")
    for row_index, (_, word) in enumerate(sorted(witnesses.items())):
        batch_index, row = divmod(row_index, config.batch_size)
        input_batches[batch_index][row] = 3
        input_batches[batch_index][row, : len(word)] = word
    batches = []
    for inputs in input_batches:
        groups, poses = rigid_prefix_targets(task, inputs)
        batches.append(
            RigidTrainingBatch(
                torch.from_numpy(inputs),
                torch.from_numpy(groups),
                torch.from_numpy(poses).float(),
            )
        )
    return batches


def _tensor_schedule_sha256(tensors: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def training_split_audit(
    task: RigidSpinTask, batches: Sequence[RigidTrainingBatch]
) -> dict[str, object]:
    """Prove the finite-language and realized-data parts of the training split."""

    occurrence_counts = Counter(
        {relation.key: 0 for relation in task.rotation.relations}
    )
    token_counts = Counter({symbol: 0 for symbol in task.input_symbols})
    all_groups = []
    all_translations = []
    schedule_tensors: list[torch.Tensor] = []
    for batch in batches:
        schedule_tensors.extend((batch.inputs, batch.group_targets, batch.pose_targets))
        all_groups.append(batch.group_targets.flatten())
        all_translations.append(batch.pose_targets[..., 4:].flatten(0, 1))
        for token in batch.inputs.flatten().tolist():
            token_counts[task.input_symbols[int(token)]] += 1
        for row in batch.inputs.tolist():
            for key, starts in relation_occurrences(
                row, task.rotation.relations
            ).items():
                occurrence_counts[key] += len(starts)
    witnesses = shortest_legal_state_witnesses(task.rotation)
    group_values = torch.cat(all_groups)
    translations = torch.cat(all_translations)
    observed_states = set(map(int, group_values.tolist()))
    binary = task.rotation.binary
    projective = {binary.projective_label[state] for state in observed_states}
    center_bits = {binary.center_bit[state] for state in observed_states}
    translation_norms = torch.linalg.vector_norm(translations.double(), dim=-1)
    passed = (
        not any(occurrence_counts.values())
        and len(witnesses) == 120
        and len(observed_states) == 120
        and len(projective) == 60
        and center_bits == {0, 1}
        and all(token_counts[symbol] > 0 for symbol in task.input_symbols)
        and float(translation_norms.max()) > 0
    )
    return {
        "forbidden_relation_occurrences": dict(occurrence_counts),
        "exact_legal_rotation_state_coverage": len(witnesses),
        "maximum_shortest_witness_length": max(map(len, witnesses.values())),
        "observed_binary_rotation_states": len(observed_states),
        "observed_projective_rotation_states": len(projective),
        "observed_center_bits": sorted(center_bits),
        "token_counts": dict(token_counts),
        "maximum_target_translation_norm": float(translation_norms.max()),
        "input_group_pose_schedule_sha256": _tensor_schedule_sha256(schedule_tensors),
        "passed": passed,
    }


def make_relation_pair_batches(
    task: RigidSpinTask,
    relation: RelationSpec,
    config: RigidSpinConfig,
    length: int,
    relation_position: str,
) -> list[RigidRelationPairBatch]:
    """Create central/identity pairs in shared rigid-motion contexts."""

    width = len(relation.tokens)
    if length < width:
        raise ValueError(f"length {length} is shorter than {relation.key}")
    if relation_position not in {"early", "late"}:
        raise ValueError("relation_position must be early or late")
    block_start = 0 if relation_position == "early" else length - width
    relation_offset = task.rotation.relations.index(relation)
    batches = []
    for batch_index in range(config.validation_batches):
        generator = np.random.default_rng(
            314_159
            + 10_000 * config.seed
            + 1_000 * length
            + 100 * relation_offset
            + 10 * (relation_position == "late")
            + batch_index
        )
        centers: list[tuple[int, ...]] = []
        identities: list[tuple[int, ...]] = []
        attempts = 0
        while len(centers) < config.validation_pairs_per_batch:
            attempts += 1
            if attempts > 100_000:
                raise RuntimeError("could not sample audited rigid relation contexts")
            left = _sample_legal_rigid_word(
                block_start, generator, task.rotation.relations
            )
            right = _sample_legal_rigid_word(
                length - block_start - width, generator, task.rotation.relations
            )
            center = (*left, *relation.tokens, *right)
            identity = (*left, *relation.identity_tokens, *right)
            center_counts = relation_occurrences(center, task.rotation.relations)
            identity_counts = relation_occurrences(identity, task.rotation.relations)
            if any(identity_counts[key] for key in identity_counts):
                continue
            if any(
                len(starts) != (1 if key == relation.key else 0)
                for key, starts in center_counts.items()
            ):
                continue
            if center_counts[relation.key][0] != block_start:
                continue
            centers.append(center)
            identities.append(identity)
        interleaved = np.empty(
            (2 * config.validation_pairs_per_batch, length), dtype=np.int64
        )
        interleaved[0::2] = np.asarray(centers)
        interleaved[1::2] = np.asarray(identities)
        groups, poses = rigid_prefix_targets(task, interleaved)
        mask = np.zeros_like(interleaved, dtype=bool)
        mask[:, block_start + width - 1 :] = True
        batches.append(
            RigidRelationPairBatch(
                torch.from_numpy(interleaved),
                torch.from_numpy(groups),
                torch.from_numpy(poses).float(),
                torch.from_numpy(mask),
                relation.key,
                relation_position,
                block_start,
            )
        )
    return batches


def relation_pair_audit(
    task: RigidSpinTask,
    relation: RelationSpec,
    batches: Sequence[RigidRelationPairBatch],
) -> dict[str, object]:
    """Audit central sign separation and exact physical-pose equality."""

    relation_counts = Counter(
        {candidate.key: 0 for candidate in task.rotation.relations}
    )
    identity_counts = Counter(
        {candidate.key: 0 for candidate in task.rotation.relations}
    )
    total_pairs = 0
    shared_context_checks = 0
    group_partner_checks = 0
    projective_checks = 0
    antipodal_checks = 0
    translation_equality_checks = 0
    scored_pair_positions = 0
    nonzero_translation_positions = 0
    translation_tokens = 0
    schedule_tensors: list[torch.Tensor] = []
    binary = task.rotation.binary
    width = len(relation.tokens)
    for batch in batches:
        schedule_tensors.extend(
            (
                batch.inputs,
                batch.group_targets,
                batch.pose_targets,
                batch.post_relation_mask,
            )
        )
        inputs = batch.inputs.numpy()
        groups = batch.group_targets.numpy()
        poses = batch.pose_targets.double().numpy()
        masks = batch.post_relation_mask.numpy()
        total_pairs += len(inputs) // 2
        translation_tokens += int(np.isin(inputs, TRANSLATION_TOKEN_INDICES).sum())
        for row_index, row in enumerate(inputs):
            counts = relation_occurrences(row.tolist(), task.rotation.relations)
            destination = relation_counts if row_index % 2 == 0 else identity_counts
            for key, starts in counts.items():
                destination[key] += len(starts)
        for center_row in range(0, len(inputs), 2):
            identity_row = center_row + 1
            shared_context_checks += int(
                np.array_equal(
                    np.delete(
                        inputs[center_row],
                        np.s_[batch.block_start : batch.block_start + width],
                    ),
                    np.delete(
                        inputs[identity_row],
                        np.s_[batch.block_start : batch.block_start + width],
                    ),
                )
            )
            for position in np.flatnonzero(masks[center_row]):
                scored_pair_positions += 1
                center_state = int(groups[center_row, position])
                identity_state = int(groups[identity_row, position])
                group_partner_checks += int(
                    binary.central_partner[identity_state] == center_state
                )
                projective_checks += int(
                    binary.projective_label[center_state]
                    == binary.projective_label[identity_state]
                )
                center_pose = poses[center_row, position]
                identity_pose = poses[identity_row, position]
                antipodal_checks += int(
                    np.max(np.abs(center_pose[:4] + identity_pose[:4])) < 2e-6
                )
                translation_equality_checks += int(
                    np.max(np.abs(center_pose[4:] - identity_pose[4:])) < 2e-6
                )
                nonzero_translation_positions += int(
                    np.linalg.norm(center_pose[4:]) > 1e-8
                )
    expected_relation_counts = {
        candidate.key: total_pairs if candidate.key == relation.key else 0
        for candidate in task.rotation.relations
    }
    passed = (
        dict(relation_counts) == expected_relation_counts
        and not any(identity_counts.values())
        and shared_context_checks == total_pairs
        and group_partner_checks == scored_pair_positions
        and projective_checks == scored_pair_positions
        and antipodal_checks == scored_pair_positions
        and translation_equality_checks == scored_pair_positions
        and nonzero_translation_positions > 0
        and translation_tokens > 0
    )
    return {
        "relation": relation.display,
        "paired_sequences": total_pairs,
        "scored_pair_positions": scored_pair_positions,
        "center_branch_relation_occurrences": dict(relation_counts),
        "identity_branch_relation_occurrences": dict(identity_counts),
        "shared_context_checks": shared_context_checks,
        "exact_central_partner_checks": group_partner_checks,
        "projective_match_checks": projective_checks,
        "antipodal_quaternion_checks": antipodal_checks,
        "translation_equality_checks": translation_equality_checks,
        "nonzero_translation_scored_positions": nonzero_translation_positions,
        "translation_token_occurrences": translation_tokens,
        "schedule_sha256": _tensor_schedule_sha256(schedule_tensors),
        "passed": passed,
    }


def mamba2_model(config: RigidSpinConfig) -> Mamba2ForCausalLM:
    if config.mamba_hidden_size * 2 != config.mamba_heads * config.mamba_head_dim:
        raise ValueError("Mamba-2 expand=2 requires 2*hidden_size=heads*head_dim")
    return Mamba2ForCausalLM(
        Mamba2Config(
            vocab_size=len(INPUT_SYMBOLS),
            hidden_size=config.mamba_hidden_size,
            state_size=config.mamba_state_size,
            num_hidden_layers=config.mamba_layers,
            num_heads=config.mamba_heads,
            head_dim=config.mamba_head_dim,
            expand=2,
            conv_kernel=4,
            n_groups=1,
            tie_word_embeddings=True,
            use_cache=False,
        )
    )


def build_models(config: RigidSpinConfig) -> dict[str, nn.Module]:
    """Build all candidates and enforce a two-percent parameter envelope."""

    models: dict[str, nn.Module] = {
        "direct_product_pose_scan": DirectProductPoseTracker(len(INPUT_SYMBOLS)),
        "direct_motor_pose_scan": DirectMotorPoseTracker(len(INPUT_SYMBOLS)),
        "spin_quaternion_scan": SpinCompositionClassifier(
            input_vocab_size=len(INPUT_SYMBOLS),
            output_size=7,
            lanes=config.spin_lanes,
            decoder_hidden=config.spin_decoder_hidden,
        ),
        "spin_motor_scan": MotorCompositionClassifier(
            input_vocab_size=len(INPUT_SYMBOLS),
            output_size=7,
            lanes=config.motor_lanes,
            decoder_hidden=config.motor_decoder_hidden,
        ),
        "mamba2_transformers": mamba2_model(config),
        "delta_product_reference": DeltaProductReferenceModel(
            input_vocab_size=len(INPUT_SYMBOLS),
            output_size=7,
            hidden_size=config.delta_hidden_size,
            num_heads=config.delta_heads,
            num_householder=config.delta_householder_updates,
            intermediate_size=config.delta_intermediate_size,
        ),
    }
    counts = {name: parameter_count(model) for name, model in models.items()}
    groups = (
        ("direct_product_pose_scan", "direct_motor_pose_scan"),
        (
            "spin_quaternion_scan",
            "spin_motor_scan",
            "mamba2_transformers",
            "delta_product_reference",
        ),
    )
    for group in groups:
        group_counts = [counts[name] for name in group]
        relative_gap = (max(group_counts) - min(group_counts)) / max(group_counts)
        if relative_gap > 0.02:
            raise RuntimeError(
                f"candidate parameter group exceeds two percent: {group_counts}"
            )
    return models


def recurrent_state_scalars(config: RigidSpinConfig) -> dict[str, int]:
    """Report per-sequence recurrent cache sizes, excluding batch dimension."""

    mamba_inner = 2 * config.mamba_hidden_size
    mamba_conv_kernel = 4
    return {
        "direct_product_pose_scan": 7,
        "direct_motor_pose_scan": 8,
        "spin_quaternion_scan": 4 * config.spin_lanes,
        "spin_motor_scan": 8 * config.motor_lanes,
        "mamba2_transformers": config.mamba_layers
        * (
            mamba_inner * mamba_conv_kernel
            + config.mamba_heads * config.mamba_head_dim * config.mamba_state_size
        ),
        "delta_product_reference": config.delta_heads
        * (config.delta_hidden_size // config.delta_heads) ** 2,
    }


def identify_direct_motor_from_prefixes(
    task: RigidSpinTask, training: Sequence[RigidTrainingBatch]
) -> tuple[DirectMotorPoseTracker, dict[str, object]]:
    """Identify token motors from legal supervised prefix differences.

    For signed poses ``M_t`` the right-composed token increment is exactly
    ``M_{t-1}^{-1} M_t``.  In quaternion/translation coordinates this is

    ``q_delta = conjugate(q_prev) * q_t`` and
    ``t_delta = R(q_prev)^T (t_t - t_prev)``.

    The estimator averages repeated legal observations of each token.  It never
    reads an evaluation sequence or a held-out relation occurrence.
    """

    vocabulary = len(task.input_symbols)
    quaternion_observations: list[list[torch.Tensor]] = [[] for _ in range(vocabulary)]
    translation_observations: list[list[torch.Tensor]] = [[] for _ in range(vocabulary)]
    for batch in training:
        targets = batch.pose_targets.double()
        batch_size = targets.shape[0]
        identity_pose = torch.zeros(batch_size, 1, 7, dtype=torch.double)
        identity_pose[..., 0] = 1
        previous = torch.cat((identity_pose, targets[:, :-1]), dim=1)
        previous_q = previous[..., :4]
        current_q = targets[..., :4]
        inverse_previous_q = quaternion_conjugate(previous_q)
        increment_q = unit_quaternion(quaternion_product(inverse_previous_q, current_q))
        global_translation_delta = targets[..., 4:] - previous[..., 4:]
        pure_global_delta = torch.cat(
            (
                torch.zeros_like(global_translation_delta[..., :1]),
                global_translation_delta,
            ),
            dim=-1,
        )
        local_translation_delta = quaternion_product(
            quaternion_product(inverse_previous_q, pure_global_delta), previous_q
        )[..., 1:]
        flat_tokens = batch.inputs.flatten()
        flat_q = increment_q.flatten(0, 1)
        flat_t = local_translation_delta.flatten(0, 1)
        for token in range(vocabulary):
            selected = flat_tokens == token
            if selected.any():
                quaternion_observations[token].append(flat_q[selected])
                translation_observations[token].append(flat_t[selected])

    mean_quaternions = []
    mean_translations = []
    counts = []
    per_token_audit: dict[str, object] = {}
    for token, symbol in enumerate(task.input_symbols):
        if not quaternion_observations[token]:
            raise RuntimeError(f"training contains no observations of token {symbol}")
        observed_q = torch.cat(quaternion_observations[token])
        observed_t = torch.cat(translation_observations[token])
        mean_q = unit_quaternion(observed_q.mean(dim=0))
        mean_t = observed_t.mean(dim=0)
        signed_dot = (observed_q * mean_q).sum(dim=-1).clamp(-1.0, 1.0)
        angle_residual = torch.rad2deg(2 * torch.acos(signed_dot))
        translation_residual = torch.linalg.vector_norm(observed_t - mean_t, dim=-1)
        mean_quaternions.append(mean_q)
        mean_translations.append(mean_t)
        counts.append(len(observed_q))
        per_token_audit[symbol] = {
            "observations": len(observed_q),
            "maximum_signed_quaternion_residual_degrees": float(angle_residual.max()),
            "maximum_local_translation_residual": float(translation_residual.max()),
        }

    identified_q = torch.stack(mean_quaternions)
    identified_t = torch.stack(mean_translations)
    model = DirectMotorPoseTracker(vocabulary)
    with torch.no_grad():
        model.composition.token_rotations.copy_(
            identified_q[:, None].to(model.composition.token_rotations)
        )
        model.composition.token_translations.copy_(
            identified_t[:, None].to(model.composition.token_translations)
        )
    exact_q = torch.tensor(
        [task.rotation.binary.quaternions[element] for element in task.input_elements],
        dtype=torch.double,
    )
    exact_t = torch.tensor(task.token_translations, dtype=torch.double)
    exact_dot = (identified_q * exact_q).sum(dim=-1).clamp(-1.0, 1.0)
    exact_angle_error = torch.rad2deg(2 * torch.acos(exact_dot))
    exact_translation_error = torch.linalg.vector_norm(identified_t - exact_t, dim=-1)
    audit = {
        "method": "right_local_prefix_difference_and_per_token_mean",
        "uses_evaluation_data": False,
        "uses_forbidden_relation_occurrences": False,
        "total_prefix_observations": sum(counts),
        "per_token": per_token_audit,
        "maximum_exact_token_quaternion_error_degrees": float(exact_angle_error.max()),
        "maximum_exact_token_translation_error": float(exact_translation_error.max()),
        "identified_token_quaternions": identified_q.tolist(),
        "identified_token_translations": identified_t.tolist(),
    }
    return model, audit


def pose_outputs_for(
    name: str,
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    spin_scan_mode: str,
    motor_scan_mode: str,
    delta_scan_mode: str,
) -> torch.Tensor:
    if name == "mamba2_transformers":
        return model(input_ids=inputs, use_cache=False).logits
    if name == "direct_product_pose_scan":
        return model(inputs, scan_mode=spin_scan_mode)
    if name == "direct_motor_pose_scan":
        return model(inputs, scan_mode=motor_scan_mode)
    if name == "spin_quaternion_scan":
        return model(inputs, scan_mode=spin_scan_mode)
    if name == "spin_motor_scan":
        return model(inputs, scan_mode=motor_scan_mode)
    if name == "delta_product_reference":
        return model(inputs, scan_mode=delta_scan_mode)
    raise ValueError(f"unknown trained candidate {name!r}")


def canonicalize_quaternion_sign(quaternion: torch.Tensor) -> torch.Tensor:
    """Choose a deterministic representative of an ``SO(3)`` quaternion."""

    absolute = quaternion.abs()
    first_nonzero = absolute > 1e-7
    indices = first_nonzero.to(torch.int64).argmax(dim=-1, keepdim=True)
    pivot = quaternion.gather(-1, indices)
    sign = torch.where(pivot < 0, -torch.ones_like(pivot), torch.ones_like(pivot))
    return quaternion * sign


def oracle_pose_outputs(name: str, targets: torch.Tensor) -> torch.Tensor:
    if name == "exact_spin_motor_oracle":
        return targets.clone()
    if name == "se3_quotient_oracle":
        output = targets.clone()
        output[..., :4] = canonicalize_quaternion_sign(output[..., :4])
        return output
    if name == "spin_only_oracle":
        output = targets.clone()
        output[..., 4:] = 0
        return output
    raise ValueError(f"unknown oracle {name!r}")


def pose_loss(
    raw_outputs: torch.Tensor,
    targets: torch.Tensor,
    config: RigidSpinConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    raw_quaternion = raw_outputs[..., :4]
    predicted_quaternion = unit_quaternion(raw_quaternion)
    target_quaternion = targets[..., :4]
    alignment = (predicted_quaternion * target_quaternion).sum(dim=-1)
    rotation_loss = (1.0 - alignment).mean()
    translation_loss = torch.mean((raw_outputs[..., 4:] - targets[..., 4:]) ** 2)
    norm_loss = torch.mean(
        (torch.linalg.vector_norm(raw_quaternion, dim=-1) - 1.0) ** 2
    )
    total = (
        rotation_loss
        + config.translation_loss_weight * translation_loss
        + config.quaternion_norm_weight * norm_loss
    )
    return total, {
        "rotation": float(rotation_loss.detach()),
        "translation": float(translation_loss.detach()),
        "quaternion_norm": float(norm_loss.detach()),
    }


@torch.no_grad()
def evaluate_relation_pairs(
    name: str,
    model: nn.Module | None,
    batches: Sequence[RigidRelationPairBatch],
    device: torch.device,
    config: RigidSpinConfig,
    *,
    spin_scan_mode: str,
    motor_scan_mode: str,
    delta_scan_mode: str,
) -> dict[str, float | int]:
    """Evaluate signed, quotient-pose, and paired double-cover criteria."""

    if model is not None:
        model.eval()
    totals = Counter()
    signed_angle_sum = 0.0
    physical_angle_sum = 0.0
    translation_error_sum = 0.0
    paired_physical_angle_sum = 0.0
    paired_translation_delta_sum = 0.0
    signed_cos = math.cos(math.radians(config.signed_rotation_threshold_degrees / 2))
    physical_cos = math.cos(
        math.radians(config.physical_rotation_threshold_degrees / 2)
    )
    for batch in batches:
        output_pieces = []
        for start in range(0, len(batch.inputs), config.evaluation_microbatch_size):
            inputs = batch.inputs[start : start + config.evaluation_microbatch_size]
            targets = batch.pose_targets[
                start : start + config.evaluation_microbatch_size
            ]
            if model is None:
                raw = oracle_pose_outputs(name, targets)
            else:
                raw = pose_outputs_for(
                    name,
                    model,
                    inputs.to(device),
                    spin_scan_mode=spin_scan_mode,
                    motor_scan_mode=motor_scan_mode,
                    delta_scan_mode=delta_scan_mode,
                ).cpu()
            output_pieces.append(raw)
        outputs = torch.cat(output_pieces)
        predicted_q = unit_quaternion(outputs[..., :4])
        target_q = batch.pose_targets[..., :4]
        dot = (predicted_q * target_q).sum(dim=-1).clamp(-1.0, 1.0)
        absolute_dot = dot.abs()
        signed_degrees = torch.rad2deg(2 * torch.acos(dot))
        physical_degrees = torch.rad2deg(2 * torch.acos(absolute_dot))
        translation_error = torch.linalg.vector_norm(
            outputs[..., 4:] - batch.pose_targets[..., 4:], dim=-1
        )
        mask = batch.post_relation_mask
        selected_dot = dot[mask]
        selected_absolute_dot = absolute_dot[mask]
        selected_translation = translation_error[mask]
        totals["positions"] += selected_dot.numel()
        totals["signed_hemisphere"] += int((selected_dot > 0).sum())
        totals["signed_rotation"] += int((selected_dot >= signed_cos).sum())
        totals["physical_rotation"] += int(
            (selected_absolute_dot >= physical_cos).sum()
        )
        totals["translation"] += int(
            (selected_translation <= config.translation_threshold).sum()
        )
        totals["joint"] += int(
            (
                (selected_dot >= signed_cos)
                & (selected_translation <= config.translation_threshold)
            ).sum()
        )
        signed_angle_sum += float(signed_degrees[mask].sum())
        physical_angle_sum += float(physical_degrees[mask].sum())
        translation_error_sum += float(selected_translation.sum())

        pair_mask = batch.post_relation_mask[0::2]
        center_q = predicted_q[0::2][pair_mask]
        identity_q = predicted_q[1::2][pair_mask]
        center_target_q = target_q[0::2][pair_mask]
        identity_target_q = target_q[1::2][pair_mask]
        center_translation = outputs[0::2, :, 4:][pair_mask]
        identity_translation = outputs[1::2, :, 4:][pair_mask]
        target_translation = batch.pose_targets[0::2, :, 4:][pair_mask]
        center_dot = (center_q * center_target_q).sum(dim=-1).clamp(-1.0, 1.0)
        identity_dot = (identity_q * identity_target_q).sum(dim=-1).clamp(-1.0, 1.0)
        pair_dot = (center_q * identity_q).sum(dim=-1).clamp(-1.0, 1.0)
        pair_physical_degrees = torch.rad2deg(2 * torch.acos(pair_dot.abs()))
        pair_translation_delta = torch.linalg.vector_norm(
            center_translation - identity_translation, dim=-1
        )
        center_translation_error = torch.linalg.vector_norm(
            center_translation - target_translation, dim=-1
        )
        identity_translation_error = torch.linalg.vector_norm(
            identity_translation - target_translation, dim=-1
        )
        totals["pair_positions"] += pair_dot.numel()
        totals["paired_antipodal"] += int((pair_dot <= -signed_cos).sum())
        totals["paired_physical_agreement"] += int(
            (
                (pair_dot.abs() >= physical_cos)
                & (pair_translation_delta <= config.translation_threshold)
            ).sum()
        )
        totals["paired_double_cover"] += int(
            (
                (center_dot >= signed_cos)
                & (identity_dot >= signed_cos)
                & (center_translation_error <= config.translation_threshold)
                & (identity_translation_error <= config.translation_threshold)
            ).sum()
        )
        paired_physical_angle_sum += float(pair_physical_degrees.sum())
        paired_translation_delta_sum += float(pair_translation_delta.sum())
    positions = totals["positions"]
    pairs = totals["pair_positions"]
    return {
        "scored_positions": positions,
        "signed_hemisphere_accuracy": totals["signed_hemisphere"] / positions,
        "signed_rotation_threshold_accuracy": totals["signed_rotation"] / positions,
        "physical_rotation_threshold_accuracy": totals["physical_rotation"] / positions,
        "translation_threshold_accuracy": totals["translation"] / positions,
        "joint_signed_pose_accuracy": totals["joint"] / positions,
        "mean_signed_rotation_degrees": signed_angle_sum / positions,
        "mean_physical_rotation_degrees": physical_angle_sum / positions,
        "mean_translation_l2": translation_error_sum / positions,
        "paired_scored_positions": pairs,
        "paired_antipodal_threshold_accuracy": totals["paired_antipodal"] / pairs,
        "paired_physical_pose_agreement_accuracy": totals["paired_physical_agreement"]
        / pairs,
        "paired_double_cover_pose_accuracy": totals["paired_double_cover"] / pairs,
        "paired_mean_physical_rotation_difference_degrees": (
            paired_physical_angle_sum / pairs
        ),
        "paired_mean_translation_difference": paired_translation_delta_sum / pairs,
    }


def _evaluate_all(
    name: str,
    model: nn.Module | None,
    evaluations: dict[str, list[RigidRelationPairBatch]],
    device: torch.device,
    config: RigidSpinConfig,
    *,
    spin_scan_mode: str,
    motor_scan_mode: str,
    delta_scan_mode: str,
) -> dict[str, dict[str, float | int]]:
    return {
        key: evaluate_relation_pairs(
            name,
            model,
            batches,
            device,
            config,
            spin_scan_mode=spin_scan_mode,
            motor_scan_mode=motor_scan_mode,
            delta_scan_mode=delta_scan_mode,
        )
        for key, batches in evaluations.items()
    }


def train_one(
    name: str,
    make_model: Callable[[], nn.Module],
    task: RigidSpinTask,
    training: Sequence[RigidTrainingBatch],
    evaluations: dict[str, list[RigidRelationPairBatch]],
    config: RigidSpinConfig,
    device: torch.device,
    *,
    spin_scan_mode: str,
    motor_scan_mode: str,
    delta_scan_mode: str,
    checkpoint_directory: Path | None,
) -> dict[str, object]:
    seed_everything(config.seed)
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    loss_samples: dict[str, object] = {}
    start = time.perf_counter()
    gradient_norm = torch.tensor(0.0)
    for step, batch in enumerate(training, start=1):
        inputs = batch.inputs.to(device)
        targets = batch.pose_targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        raw_outputs = pose_outputs_for(
            name,
            model,
            inputs,
            spin_scan_mode=spin_scan_mode,
            motor_scan_mode=motor_scan_mode,
            delta_scan_mode=delta_scan_mode,
        )
        loss, components = pose_loss(raw_outputs, targets, config)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == config.steps:
            loss_samples[str(step)] = {
                "total": float(loss.detach()),
                **components,
            }
            print(
                f"{task.rotation.coordinate_label}/{name} seed={config.seed} "
                f"step={step}/{config.steps} loss={float(loss.detach()):.6f}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    final = _evaluate_all(
        name,
        model,
        evaluations,
        device,
        config,
        spin_scan_mode=spin_scan_mode,
        motor_scan_mode=motor_scan_mode,
        delta_scan_mode=delta_scan_mode,
    )
    result: dict[str, object] = {
        "name": name,
        "seed": config.seed,
        "coordinate_label": task.rotation.coordinate_label,
        "parameters": parameter_count(model),
        "recurrent_state_scalars": recurrent_state_scalars(config)[name],
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
            f"{name}_coord-{task.rotation.coordinate_label}_seed{config.seed}_"
            f"step{config.steps}.pt"
        )
        torch.save(
            {
                "format_version": 1,
                "candidate": name,
                "benchmark": "spin_motor_rigid_2a5",
                "coordinate_label": task.rotation.coordinate_label,
                "task_presentation": task.rotation.binary.presentation,
                "translation_tokens": task.token_translations,
                "group_table_sha256": task.rotation.binary.group_table_sha256,
                "benchmark_config": asdict(config),
                "metrics": result,
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
            },
            path,
        )
        result["checkpoint"] = str(path)
        result["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _aggregate_gate_summary(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Summarize strict long-context gates without hiding split failures."""

    summary: dict[str, object] = {}
    for result in results:
        metrics = result["final_relation_metrics"]
        long_metrics = {
            key: value
            for key, value in metrics.items()
            if "_L64" in key or "_L128" in key
        }
        selected_metrics = long_metrics or metrics
        summary[str(result["name"])] = {
            "long_split_count": len(long_metrics),
            "minimum_signed_hemisphere_accuracy": min(
                value["signed_hemisphere_accuracy"]
                for value in selected_metrics.values()
            ),
            "minimum_joint_signed_pose_accuracy": min(
                value["joint_signed_pose_accuracy"]
                for value in selected_metrics.values()
            ),
            "minimum_paired_double_cover_pose_accuracy": min(
                value["paired_double_cover_pose_accuracy"]
                for value in selected_metrics.values()
            ),
            "maximum_mean_translation_l2": max(
                value["mean_translation_l2"] for value in selected_metrics.values()
            ),
            "all_long_splits_center_gate_90pct": all(
                value["signed_hemisphere_accuracy"] >= 0.90
                for value in long_metrics.values()
            )
            if long_metrics
            else None,
            "all_long_splits_joint_pose_gate_80pct": all(
                value["joint_signed_pose_accuracy"] >= 0.80
                for value in long_metrics.values()
            )
            if long_metrics
            else None,
        }
    return summary


def run_benchmark(
    config: RigidSpinConfig,
    *,
    coordinate_label: str,
    candidates: Sequence[str],
    device: torch.device,
    spin_scan_mode: str,
    motor_scan_mode: str,
    delta_scan_mode: str,
    checkpoint_directory: Path | None = None,
) -> dict[str, object]:
    unknown = set(candidates) - set(TRAINED_CANDIDATES)
    if unknown:
        raise ValueError(f"unknown candidates: {sorted(unknown)}")
    started = datetime.now(ZoneInfo("Africa/Johannesburg"))
    task = make_rigid_spin_task(coordinate_label, config.translation_step)
    training = make_training_batches(task, config)
    split_audit = training_split_audit(task, training)
    if not split_audit["passed"]:
        raise RuntimeError("training split audit failed")
    evaluations: dict[str, list[RigidRelationPairBatch]] = {}
    evaluation_audits: dict[str, object] = {}
    for relation in task.rotation.relations:
        for length in config.evaluation_lengths:
            for position in ("early", "late"):
                key = f"{relation.key}__{position}_L{length}"
                batches = make_relation_pair_batches(
                    task, relation, config, length, position
                )
                audit = relation_pair_audit(task, relation, batches)
                if not audit["passed"]:
                    raise RuntimeError(f"evaluation split audit failed for {key}")
                evaluations[key] = batches
                evaluation_audits[key] = audit

    shapes = build_models(config)
    parameter_counts = {name: parameter_count(model) for name, model in shapes.items()}
    del shapes
    oracle_metrics = {
        name: _evaluate_all(
            name,
            None,
            evaluations,
            device,
            config,
            spin_scan_mode=spin_scan_mode,
            motor_scan_mode=motor_scan_mode,
            delta_scan_mode=delta_scan_mode,
        )
        for name in ORACLE_CANDIDATES
    }
    results = []
    for name in candidates:
        results.append(
            train_one(
                name,
                lambda name=name: build_models(config)[name],
                task,
                training,
                evaluations,
                config,
                device,
                spin_scan_mode=spin_scan_mode,
                motor_scan_mode=motor_scan_mode,
                delta_scan_mode=delta_scan_mode,
                checkpoint_directory=checkpoint_directory,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    finished = datetime.now(ZoneInfo("Africa/Johannesburg"))
    return {
        "schema_version": 1,
        "benchmark": "spin_motor_rigid_2a5",
        "status": "bounded single-seed pilot",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_wall_seconds": (finished - started).total_seconds(),
        "claim_boundary": (
            "Controlled empirical comparison of one initialization in one generator "
            "coordinate; not a family theorem, multi-seed result, or fused timing claim."
        ),
        "config": asdict(config),
        "coordinate_label": coordinate_label,
        "task": {
            "input_symbols": list(task.input_symbols),
            "input_elements": list(task.input_elements),
            "token_translations": task.token_translations,
            "relations": [asdict(relation) for relation in task.rotation.relations],
            "presentation": task.rotation.binary.presentation,
            "group_table_sha256": task.rotation.binary.group_table_sha256,
            "signed_pose_target": "[w,x,y,z,tx,ty,tz]",
            "translation_convention": "body/local-frame right composition",
        },
        "split_audit": {
            "training": split_audit,
            "evaluations": evaluation_audits,
        },
        "parameter_counts": parameter_counts,
        "recurrent_state_scalars": recurrent_state_scalars(config),
        "scan_modes": {
            "spin": spin_scan_mode,
            "motor": motor_scan_mode,
            "delta_product": delta_scan_mode,
            "mamba2": "Transformers unfused fallback on Windows",
        },
        "source_provenance": {
            "path_development_paper": PATH_DEVELOPMENT_PAPER,
            "mamba2_paper": MAMBA2_PAPER,
            "delta_product_reference_source_commit": DELTA_PRODUCT_SOURCE_COMMIT,
            "fixed_point_rnn": {
                "paper": FP_RNN_PAPER,
                "official_source_commit_reviewed": FP_RNN_SOURCE_COMMIT,
                "benchmark_status": "excluded rather than approximated",
                "reason": (
                    "Official source pins Python <3.12, torch 2.4.1, Triton 3.0, "
                    "mamba-ssm, and custom scan/causal-conv paths; this environment "
                    "uses Python 3.12 and torch 2.12 on Windows."
                ),
            },
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device": str(device),
            "cuda_version": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "oracles": oracle_metrics,
        "trained_results": results,
        "gate_summary": _aggregate_gate_summary(results),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--coordinate", choices=("e", "a", "b"), default="e")
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=TRAINED_CANDIDATES,
        default=READOUT_CANDIDATES,
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--spin-scan-mode", choices=("parallel", "recurrent"), default="parallel"
    )
    parser.add_argument(
        "--motor-scan-mode", choices=("parallel", "recurrent"), default="parallel"
    )
    parser.add_argument(
        "--delta-scan-mode", choices=("parallel", "recurrent"), default="parallel"
    )
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--checkpoint-directory", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    config = replace(RigidSpinConfig(), steps=args.steps, seed=args.seed)
    report = run_benchmark(
        config,
        coordinate_label=args.coordinate,
        candidates=args.candidates,
        device=device,
        spin_scan_mode=args.spin_scan_mode,
        motor_scan_mode=args.motor_scan_mode,
        delta_scan_mode=args.delta_scan_mode,
        checkpoint_directory=args.checkpoint_directory,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.artifact is not None:
        args.artifact.parent.mkdir(parents=True, exist_ok=True)
        args.artifact.write_text(rendered + "\n", encoding="utf-8")
        print(
            f"artifact_sha256={hashlib.sha256(args.artifact.read_bytes()).hexdigest()}"
        )
    print(rendered)


if __name__ == "__main__":
    main()
