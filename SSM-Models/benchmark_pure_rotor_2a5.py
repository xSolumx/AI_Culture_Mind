"""Center-sensitive ``2.A5`` prefix benchmark and Spin composition scan.

The earlier A5 task cannot distinguish a rotor from its negative because both
project to the same SO(3) action.  This benchmark keeps all 120 binary states
and evaluates paired words whose projected A5 trajectories agree after a
relation block while their binary products differ by the central element.

The trained comparison contains the maintained Pure Rotor v2.1 model, its
identity-transport ablation, a parameter-near quaternion composition scan, and
Transformers Mamba-2.  Exact table, projective-quotient, and float64 quaternion
oracles establish the task ceiling and the information lost by quotienting.

No outcome from this runner is a theorem about broad SSM families.  In
particular, the quaternion scan is an explicit Spin-sensitive inductive bias,
whereas the maintained Pure Rotor transport acts by rotor sandwiching and is
therefore center-blind at the transport operation itself.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from benchmark_pure_rotor_a5 import (
    batch_schedule_sha256,
    generated_subgroup_order,
    group_power,
    inject_training_state_coverage,
    parameter_count,
    seed_everything,
    shortest_allowed_prefix_words,
)
from compare_recurrences import (
    FiniteGroup,
    make_group_batches,
    pair_split_audit,
    state_and_pair_coverage_audit,
)
from pure_rotor_ssm import __version__ as PURE_ROTOR_VERSION
from pure_rotor_ssm.torch_backend import GASSMLanguageModel
from spin_dirac_a5_cohomology import build_exact_group_table
from spin_dirac_a5_rigidity import (
    FIELD,
    exact_icosahedral_quaternion_generators,
)
from torch import nn
from torch.nn import functional as F
from transformers import Mamba2Config, Mamba2ForCausalLM

TRAINED_CANDIDATES = (
    "pure_rotor",
    "identity_rotation_ablation",
    "spin_quaternion_scan",
    "mamba2_transformers",
)
ORACLE_CANDIDATES = (
    "exact_table_oracle",
    "projective_a5_oracle",
    "float64_quaternion_oracle",
)


@dataclass(frozen=True)
class BinaryA5BenchmarkConfig:
    """Parameter-near screening configuration for center-sensitive tracking."""

    steps: int = 300
    batch_size: int = 16
    training_length: int = 16
    validation_batches: int = 2
    validation_pairs_per_batch: int = 32
    evaluation_microbatch_size: int = 16
    evaluation_lengths: tuple[int, ...] = (2, 16, 64, 128)
    enforce_training_state_coverage: bool = True
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
    seed: int = 0


@dataclass(frozen=True)
class BinaryA5Task:
    """Exact binary-icosahedral task and its projective quotient labels."""

    group: FiniteGroup
    input_elements: tuple[int, ...]
    input_symbols: tuple[str, ...]
    held_out_pair: tuple[int, int]
    identity_relation_pair: tuple[int, int]
    center_index: int
    central_partner: tuple[int, ...]
    projective_label: tuple[int, ...]
    center_bit: tuple[int, ...]
    quaternions: tuple[tuple[float, float, float, float], ...]
    presentation: dict[str, int | str | bool]
    group_table_sha256: str


@dataclass(frozen=True)
class CentralPairBatch:
    """Paired identity/central-relation words with a post-relation mask."""

    inputs: torch.Tensor
    targets: torch.Tensor
    post_relation_mask: torch.Tensor
    relation_position: str


def _field_float(value: object) -> float:
    return float(FIELD.to_sympy(value).evalf(30))


def binary_icosahedral_task() -> BinaryA5Task:
    """Build the exact 120-state task and audit its central presentation."""

    exact_group, exact_table = build_exact_group_table()
    table = np.asarray(exact_table, dtype=np.int64)
    group = FiniteGroup(
        key="2a5",
        name="binary icosahedral group 2.A5",
        elements=tuple(f"q{index:03d}" for index in range(len(exact_group))),
        table=table,
    )
    lookup = {quaternion: index for index, quaternion in enumerate(exact_group)}
    a_quaternion, b_quaternion = exact_icosahedral_quaternion_generators()
    a = lookup[a_quaternion]
    b = lookup[b_quaternion]
    b_inverse_quaternion = (
        b_quaternion[0],
        -b_quaternion[1],
        -b_quaternion[2],
        -b_quaternion[3],
    )
    b_inverse = lookup[b_inverse_quaternion]
    center_quaternion = (-FIELD.one, FIELD.zero, FIELD.zero, FIELD.zero)
    center = lookup[center_quaternion]
    ab = int(table[a, b])
    central_partner = tuple(int(table[state, center]) for state in range(group.order))

    projective_label = [-1] * group.order
    center_bit = [-1] * group.order
    projective_pairs = []
    for state in range(group.order):
        partner = central_partner[state]
        if state < partner:
            projective_pairs.append((state, partner))
    projective_pairs.sort()
    for label, (positive, negative) in enumerate(projective_pairs):
        projective_label[positive] = label
        projective_label[negative] = label
        center_bit[positive] = 0
        center_bit[negative] = 1
    if any(value < 0 for value in (*projective_label, *center_bit)):
        raise AssertionError("projective quotient labelling is incomplete")

    table_sha256 = hashlib.sha256(
        json.dumps(exact_table, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    presentation = {
        "a_index": a,
        "b_index": b,
        "b_inverse_index": b_inverse,
        "center_index": center,
        "a_squared_is_center": group_power(group, a, 2) == center,
        "b_cubed_is_center": group_power(group, b, 3) == center,
        "ab_to_fifth_is_center": group_power(group, ab, 5) == center,
        "center_squared_is_identity": group_power(group, center, 2) == 0,
        "b_times_b_inverse_is_identity": int(table[b, b_inverse]) == 0,
        "generated_subgroup_order": generated_subgroup_order(group, (a, b)),
        "binary_group_order": group.order,
        "projective_group_order": len(projective_pairs),
    }
    if not (
        presentation["a_squared_is_center"]
        and presentation["b_cubed_is_center"]
        and presentation["ab_to_fifth_is_center"]
        and presentation["center_squared_is_identity"]
        and presentation["b_times_b_inverse_is_identity"]
        and presentation["generated_subgroup_order"] == 120
        and presentation["projective_group_order"] == 60
    ):
        raise RuntimeError("configured 2.A5 generators failed the central presentation")
    quaternions = tuple(
        tuple(_field_float(coordinate) for coordinate in quaternion)
        for quaternion in exact_group
    )
    return BinaryA5Task(
        group=group,
        input_elements=(a, b, b_inverse),
        input_symbols=("a", "b", "b_inverse"),
        held_out_pair=(0, 0),
        identity_relation_pair=(1, 2),
        center_index=center,
        central_partner=central_partner,
        projective_label=tuple(projective_label),
        center_bit=tuple(center_bit),
        quaternions=quaternions,
        presentation=presentation,
        group_table_sha256=table_sha256,
    )


def _prefix_targets(task: BinaryA5Task, tokens: np.ndarray) -> np.ndarray:
    states = np.zeros(tokens.shape[0], dtype=np.int64)
    targets = np.empty_like(tokens)
    input_elements = np.asarray(task.input_elements, dtype=np.int64)
    for position in range(tokens.shape[1]):
        states = task.group.table[states, input_elements[tokens[:, position]]]
        targets[:, position] = states
    return targets


def make_training_batches(
    task: BinaryA5Task, config: BinaryA5BenchmarkConfig
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Create one immutable schedule with the central ``a,a`` relation absent."""

    batches = make_group_batches(
        task.group,
        config.steps,
        config.batch_size,
        config.training_length,
        seed=10_000 + config.seed,
        input_elements=task.input_elements,
        held_out_pairs=(task.held_out_pair,),
    )
    if config.enforce_training_state_coverage:
        inject_training_state_coverage(task, batches, seed=config.seed)
    return batches


def _sample_pair_free_contexts(
    count: int,
    length: int,
    generator: np.random.Generator,
    *,
    forbid_first_a: bool,
    forbid_last_a: bool,
) -> np.ndarray:
    if length == 0:
        return np.empty((count, 0), dtype=np.int64)
    contexts = generator.integers(0, 3, size=(count, length), dtype=np.int64)
    if forbid_first_a:
        contexts[:, 0] = generator.integers(1, 3, size=count, dtype=np.int64)
    for position in range(1, length):
        invalid = (contexts[:, position - 1] == 0) & (contexts[:, position] == 0)
        while np.any(invalid):
            contexts[invalid, position] = generator.integers(
                1, 3, size=int(invalid.sum()), dtype=np.int64
            )
            invalid = (contexts[:, position - 1] == 0) & (contexts[:, position] == 0)
    if forbid_last_a:
        contexts[:, -1] = generator.integers(1, 3, size=count, dtype=np.int64)
    return contexts


def make_central_pair_evaluation_batches(
    task: BinaryA5Task,
    config: BinaryA5BenchmarkConfig,
    length: int,
    relation_position: str,
) -> list[CentralPairBatch]:
    """Pair ``a a = z`` against ``b b^-1 = e`` with shared context."""

    if length < 2:
        raise ValueError("central relation evaluation requires length at least two")
    if relation_position not in {"early", "late"}:
        raise ValueError("relation_position must be early or late")
    batches = []
    for batch_index in range(config.validation_batches):
        generator = np.random.default_rng(
            20_000
            + 10_000 * config.seed
            + 1_000 * length
            + 100 * (relation_position == "late")
            + batch_index
        )
        context = _sample_pair_free_contexts(
            config.validation_pairs_per_batch,
            length - 2,
            generator,
            forbid_first_a=relation_position == "early",
            forbid_last_a=relation_position == "late",
        )
        center_relation = np.tile(np.asarray(task.held_out_pair), (len(context), 1))
        identity_relation = np.tile(
            np.asarray(task.identity_relation_pair), (len(context), 1)
        )
        if relation_position == "early":
            center_words = np.concatenate((center_relation, context), axis=1)
            identity_words = np.concatenate((identity_relation, context), axis=1)
            post_start = 1
        else:
            center_words = np.concatenate((context, center_relation), axis=1)
            identity_words = np.concatenate((context, identity_relation), axis=1)
            post_start = length - 1
        interleaved = np.empty((2 * len(context), length), dtype=np.int64)
        interleaved[0::2] = center_words
        interleaved[1::2] = identity_words
        targets = _prefix_targets(task, interleaved)
        mask = np.zeros_like(interleaved, dtype=bool)
        mask[:, post_start:] = True
        batches.append(
            CentralPairBatch(
                inputs=torch.from_numpy(interleaved),
                targets=torch.from_numpy(targets),
                post_relation_mask=torch.from_numpy(mask),
                relation_position=relation_position,
            )
        )
    return batches


def training_split_audit(
    task: BinaryA5Task,
    training: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, object]:
    pair_audit = pair_split_audit(training, (task.held_out_pair,))
    if pair_audit["pair_occurrences"]:
        raise RuntimeError("the central a,a relation leaked into training")
    coverage = state_and_pair_coverage_audit(
        training,
        input_order=len(task.input_elements),
        group_order=task.group.order,
    )
    witnesses = shortest_allowed_prefix_words(task, training[0][0].shape[1])
    targets = torch.cat([target.flatten() for _, target in training])
    target_states = {int(value) for value in targets.tolist()}
    projective_states = {task.projective_label[state] for state in target_states}
    center_bits = {task.center_bit[state] for state in target_states}
    return {
        "held_out_relation": "a*a=z",
        "identity_control_relation": "b*b_inverse=e",
        "training_pair_audit": pair_audit,
        "training_coverage": coverage,
        "exact_training_language_coverage": {
            "reachable_binary_states": len(witnesses),
            "maximum_shortest_witness_length": max(map(len, witnesses.values())),
            "all_binary_states_reachable": len(witnesses) == 120,
        },
        "observed_binary_target_states": len(target_states),
        "observed_projective_target_states": len(projective_states),
        "observed_center_bits": sorted(center_bits),
        "training_schedule_sha256": batch_schedule_sha256(training),
    }


def central_pair_evaluation_audit(
    task: BinaryA5Task, batches: list[CentralPairBatch]
) -> dict[str, object]:
    total_pairs = 0
    post_positions = 0
    exact_central_partner_checks = 0
    projective_match_checks = 0
    center_relation_occurrences = 0
    identity_relation_occurrences = 0
    forced_center_relation_checks = 0
    forced_identity_relation_checks = 0
    schedule_digest = hashlib.sha256()
    for batch in batches:
        for tensor in (batch.inputs, batch.targets, batch.post_relation_mask):
            values = tensor.detach().cpu().contiguous()
            schedule_digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
            schedule_digest.update(values.numpy().tobytes())
        inputs = batch.inputs.numpy()
        targets = batch.targets.numpy()
        mask = batch.post_relation_mask.numpy()
        total_pairs += len(inputs) // 2
        center_relation_occurrences += int(
            np.sum((inputs[:, :-1] == 0) & (inputs[:, 1:] == 0))
        )
        identity_relation_occurrences += int(
            np.sum((inputs[:, :-1] == 1) & (inputs[:, 1:] == 2))
        )
        post_start = int(np.flatnonzero(mask[0])[0])
        relation_start = post_start - 1
        forced_center_relation_checks += int(
            np.all(
                inputs[0::2, relation_start : relation_start + 2]
                == np.asarray(task.held_out_pair),
                axis=1,
            ).sum()
        )
        forced_identity_relation_checks += int(
            np.all(
                inputs[1::2, relation_start : relation_start + 2]
                == np.asarray(task.identity_relation_pair),
                axis=1,
            ).sum()
        )
        for center_row in range(0, len(inputs), 2):
            identity_row = center_row + 1
            pair_mask = mask[center_row]
            for position in np.flatnonzero(pair_mask):
                center_target = int(targets[center_row, position])
                identity_target = int(targets[identity_row, position])
                post_positions += 1
                exact_central_partner_checks += int(
                    task.central_partner[identity_target] == center_target
                )
                projective_match_checks += int(
                    task.projective_label[identity_target]
                    == task.projective_label[center_target]
                )
    passed = (
        exact_central_partner_checks == post_positions
        and projective_match_checks == post_positions
        and forced_center_relation_checks == total_pairs
        and forced_identity_relation_checks == total_pairs
    )
    return {
        "paired_sequences": total_pairs,
        "post_relation_pair_positions": post_positions,
        "exact_central_partner_checks": exact_central_partner_checks,
        "projective_match_checks": projective_match_checks,
        "center_relation_occurrences": center_relation_occurrences,
        "identity_relation_occurrences": identity_relation_occurrences,
        "forced_center_relation_checks": forced_center_relation_checks,
        "forced_identity_relation_checks": forced_identity_relation_checks,
        "evaluation_schedule_sha256": schedule_digest.hexdigest(),
        "passed": passed,
    }


def quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Batched Hamilton product on a final quaternion axis."""

    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quaternion_prefix_scan(
    token_quaternions: torch.Tensor, *, mode: str
) -> torch.Tensor:
    """Inclusive sequential or Hillis--Steele quaternion prefix products."""

    if token_quaternions.ndim != 4 or token_quaternions.shape[-1] != 4:
        raise ValueError("token quaternions must have shape [batch,time,lanes,4]")
    if mode == "recurrent":
        state = torch.zeros_like(token_quaternions[:, 0])
        state[..., 0] = 1.0
        outputs = []
        for position in range(token_quaternions.shape[1]):
            state = quaternion_multiply(state, token_quaternions[:, position])
            state = F.normalize(state, dim=-1)
            outputs.append(state)
        return torch.stack(outputs, dim=1)
    if mode != "parallel":
        raise ValueError("quaternion scan mode must be recurrent or parallel")
    outputs = token_quaternions
    offset = 1
    while offset < outputs.shape[1]:
        previous = outputs
        updated = previous.clone()
        updated[:, offset:] = quaternion_multiply(
            previous[:, :-offset], previous[:, offset:]
        )
        outputs = F.normalize(updated, dim=-1)
        offset *= 2
    return outputs


class SpinQuaternionScanModel(nn.Module):
    """Sign-sensitive associative quaternion state with a parameter-near decoder."""

    def __init__(
        self,
        *,
        input_order: int,
        output_order: int,
        lanes: int,
        decoder_hidden: int,
    ) -> None:
        super().__init__()
        self.lanes = lanes
        self.token_rotors = nn.Parameter(torch.empty(input_order, lanes, 4))
        nn.init.normal_(self.token_rotors, mean=0.0, std=0.5)
        with torch.no_grad():
            self.token_rotors[..., 0].add_(1.0)
        self.decoder = nn.Sequential(
            nn.Linear(4 * lanes, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, output_order),
        )

    def forward(
        self, tokens: torch.Tensor, *, scan_mode: str = "parallel"
    ) -> torch.Tensor:
        rotors = F.normalize(self.token_rotors, dim=-1)
        token_rotors = rotors[tokens]
        states = quaternion_prefix_scan(token_rotors, mode=scan_mode)
        return self.decoder(states.flatten(start_dim=-2))


def pure_rotor_model(
    task: BinaryA5Task,
    config: BinaryA5BenchmarkConfig,
    *,
    max_rotor_angle: float,
) -> GASSMLanguageModel:
    return GASSMLanguageModel(
        vocab_size=task.group.order,
        channels=config.rotor_channels,
        num_layers=config.rotor_layers,
        expansion=config.rotor_expansion,
        dropout_rate=0.0,
        max_rotor_angle=max_rotor_angle,
    )


def spin_quaternion_model(
    task: BinaryA5Task, config: BinaryA5BenchmarkConfig
) -> SpinQuaternionScanModel:
    return SpinQuaternionScanModel(
        input_order=len(task.input_elements),
        output_order=task.group.order,
        lanes=config.quaternion_lanes,
        decoder_hidden=config.quaternion_decoder_hidden,
    )


def mamba2_model(
    task: BinaryA5Task, config: BinaryA5BenchmarkConfig
) -> Mamba2ForCausalLM:
    if config.mamba_hidden_size * 2 != config.mamba_heads * config.mamba_head_dim:
        raise ValueError("Mamba-2 expand=2 requires 2*hidden_size = heads*head_dim")
    return Mamba2ForCausalLM(
        Mamba2Config(
            vocab_size=task.group.order,
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


def build_models(
    task: BinaryA5Task, config: BinaryA5BenchmarkConfig
) -> dict[str, nn.Module]:
    models: dict[str, nn.Module] = {
        "pure_rotor": pure_rotor_model(task, config, max_rotor_angle=math.pi),
        "identity_rotation_ablation": pure_rotor_model(
            task, config, max_rotor_angle=0.0
        ),
        "spin_quaternion_scan": spin_quaternion_model(task, config),
        "mamba2_transformers": mamba2_model(task, config),
    }
    counts = {name: parameter_count(model) for name, model in models.items()}
    maximum = max(counts.values())
    minimum = min(counts.values())
    if (maximum - minimum) / maximum > 0.02:
        raise ValueError(
            f"trained candidate parameter counts differ by more than 2%: {counts}"
        )
    return models


def logits_for(
    name: str,
    model: nn.Module,
    inputs: torch.Tensor,
    rotor_scan_mode: str,
    quaternion_scan_mode: str,
) -> torch.Tensor:
    if name == "mamba2_transformers":
        return model(input_ids=inputs, use_cache=False).logits
    if name == "spin_quaternion_scan":
        return model(inputs, scan_mode=quaternion_scan_mode)
    return model(inputs, scan_mode=rotor_scan_mode)


def oracle_logits(name: str, inputs: torch.Tensor, task: BinaryA5Task) -> torch.Tensor:
    """Return exact/projective/quaternion analytic baseline logits."""

    input_array = inputs.detach().cpu().numpy()
    targets = _prefix_targets(task, input_array)
    batch, length = input_array.shape
    if name in {"exact_table_oracle", "projective_a5_oracle"}:
        logits = torch.full((batch, length, task.group.order), -20.0)
        for row in range(batch):
            for position in range(length):
                target = int(targets[row, position])
                logits[row, position, target] = 20.0
                if name == "projective_a5_oracle":
                    logits[row, position, task.central_partner[target]] = 20.0
        return logits
    if name != "float64_quaternion_oracle":
        raise KeyError(name)
    prototypes = torch.tensor(task.quaternions, dtype=torch.float64)
    token_rotors = prototypes[torch.tensor(task.input_elements)][inputs.cpu()]
    state = torch.zeros(batch, 4, dtype=torch.float64)
    state[:, 0] = 1.0
    states = []
    for position in range(length):
        state = quaternion_multiply(state, token_rotors[:, position])
        state = F.normalize(state, dim=-1)
        states.append(state)
    stacked = torch.stack(states, dim=1)
    return (50.0 * torch.einsum("btd,gd->btg", stacked, prototypes)).float()


@torch.no_grad()
def evaluate_central_pairs(
    name: str,
    model: nn.Module | None,
    batches: list[CentralPairBatch],
    task: BinaryA5Task,
    device: torch.device,
    rotor_scan_mode: str,
    quaternion_scan_mode: str,
    microbatch_size: int,
) -> dict[str, float]:
    """Evaluate exact, projective, center-bit, margin, and paired metrics."""

    if model is not None:
        model.eval()
    projective = torch.tensor(task.projective_label, dtype=torch.long)
    center_bit = torch.tensor(task.center_bit, dtype=torch.long)
    partner = torch.tensor(task.central_partner, dtype=torch.long)
    totals = CounterMetrics()
    for batch in batches:
        batch_predictions = []
        batch_targets = batch.targets
        batch_logits = []
        for start in range(0, batch.inputs.shape[0], microbatch_size):
            inputs = batch.inputs[start : start + microbatch_size]
            if model is None:
                logits = oracle_logits(name, inputs, task)
            else:
                logits = logits_for(
                    name,
                    model,
                    inputs.to(device),
                    rotor_scan_mode,
                    quaternion_scan_mode,
                ).cpu()
            batch_logits.append(logits)
            batch_predictions.append(logits.argmax(dim=-1))
        logits = torch.cat(batch_logits)
        predictions = torch.cat(batch_predictions)
        mask = batch.post_relation_mask
        targets = batch_targets
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


@dataclass
class CounterMetrics:
    nll_sum: float = 0.0
    positions: int = 0
    exact: int = 0
    projective: int = 0
    center_bit: int = 0
    margin_score: float = 0.0
    central_probability: float = 0.0
    final_rows: int = 0
    final_exact: int = 0
    final_projective: int = 0
    final_center_bit: int = 0
    final_pairs: int = 0
    paired_exact: int = 0
    paired_structural_separation: int = 0

    def as_metrics(self) -> dict[str, float]:
        return {
            "post_relation_nll": self.nll_sum / self.positions,
            "post_relation_exact_accuracy": self.exact / self.positions,
            "post_relation_projective_accuracy": self.projective / self.positions,
            "post_relation_center_bit_accuracy": self.center_bit / self.positions,
            "post_relation_central_margin_accuracy": self.margin_score / self.positions,
            "post_relation_target_probability_within_central_pair": (
                self.central_probability / self.positions
            ),
            "final_exact_accuracy": self.final_exact / self.final_rows,
            "final_projective_accuracy": self.final_projective / self.final_rows,
            "final_center_bit_accuracy": self.final_center_bit / self.final_rows,
            "paired_final_exact_accuracy": self.paired_exact / self.final_pairs,
            "paired_final_structural_center_separation": (
                self.paired_structural_separation / self.final_pairs
            ),
        }


def _evaluate_all_splits(
    name: str,
    model: nn.Module | None,
    evaluations: dict[str, list[CentralPairBatch]],
    task: BinaryA5Task,
    config: BinaryA5BenchmarkConfig,
    device: torch.device,
    rotor_scan_mode: str,
    quaternion_scan_mode: str,
) -> dict[str, dict[str, float]]:
    return {
        key: evaluate_central_pairs(
            name,
            model,
            batches,
            task,
            device,
            rotor_scan_mode,
            quaternion_scan_mode,
            config.evaluation_microbatch_size,
        )
        for key, batches in evaluations.items()
    }


def checkpoint_payload(
    name: str,
    model: nn.Module,
    task: BinaryA5Task,
    config: BinaryA5BenchmarkConfig,
    result: dict[str, object],
) -> dict[str, object]:
    return {
        "format_version": 1,
        "candidate": name,
        "pure_rotor_model_version": PURE_ROTOR_VERSION,
        "task_presentation": task.presentation,
        "group_table_sha256": task.group_table_sha256,
        "benchmark_config": asdict(config),
        "metrics": result,
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }


def train_one(
    name: str,
    make_model: Callable[[], nn.Module],
    task: BinaryA5Task,
    training: list[tuple[torch.Tensor, torch.Tensor]],
    evaluations: dict[str, list[CentralPairBatch]],
    config: BinaryA5BenchmarkConfig,
    device: torch.device,
    rotor_scan_mode: str,
    quaternion_scan_mode: str,
    checkpoint_directory: Path | None,
) -> dict[str, object]:
    seed_everything(config.seed)
    model = make_model().to(device)
    initial = _evaluate_all_splits(
        name,
        model,
        evaluations,
        task,
        config,
        device,
        rotor_scan_mode,
        quaternion_scan_mode,
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
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = logits_for(name, model, inputs, rotor_scan_mode, quaternion_scan_mode)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == config.steps:
            loss_samples[str(step)] = float(loss.detach())
            print(
                f"{name} seed={config.seed} step={step}/{config.steps} "
                f"loss={loss_samples[str(step)]:.5f}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    final = _evaluate_all_splits(
        name,
        model,
        evaluations,
        task,
        config,
        device,
        rotor_scan_mode,
        quaternion_scan_mode,
    )
    result: dict[str, object] = {
        "name": name,
        "seed": config.seed,
        "parameters": parameter_count(model),
        "initial_center_pair_metrics": initial,
        "final_center_pair_metrics": final,
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
        path = checkpoint_directory / f"{name}_seed{config.seed}_step{config.steps}.pt"
        torch.save(checkpoint_payload(name, model, task, config, result), path)
        result["checkpoint"] = str(path)
        result["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def parse_lengths(value: str) -> tuple[int, ...]:
    lengths = tuple(int(item) for item in value.split(",") if item.strip())
    if not lengths or any(length < 2 for length in lengths):
        raise ValueError("evaluation lengths must be distinct integers >= 2")
    if len(set(lengths)) != len(lengths):
        raise ValueError("evaluation lengths must be distinct")
    return lengths


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in value.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list of distinct integers")
    return seeds


def run_benchmark(
    config: BinaryA5BenchmarkConfig,
    *,
    seeds: Sequence[int],
    device: torch.device,
    rotor_scan_mode: str,
    quaternion_scan_mode: str,
    checkpoint_directory: Path | None = None,
) -> dict[str, object]:
    task = binary_icosahedral_task()
    model_shapes = build_models(task, config)
    counts = {name: parameter_count(model) for name, model in model_shapes.items()}
    results = []
    oracle_results: dict[str, object] = {}
    split_audits: dict[str, object] = {}
    for seed in seeds:
        run_config = BinaryA5BenchmarkConfig(**{**asdict(config), "seed": seed})
        training = make_training_batches(task, run_config)
        evaluations: dict[str, list[CentralPairBatch]] = {}
        evaluation_audits: dict[str, object] = {}
        for length in run_config.evaluation_lengths:
            positions = ("early",) if length == 2 else ("early", "late")
            for position in positions:
                key = f"{position}_L{length}"
                evaluations[key] = make_central_pair_evaluation_batches(
                    task, run_config, length, position
                )
                evaluation_audits[key] = central_pair_evaluation_audit(
                    task, evaluations[key]
                )
                if not evaluation_audits[key]["passed"]:
                    raise RuntimeError(f"central pair audit failed for {key}")
        split_audits[str(seed)] = {
            "training": training_split_audit(task, training),
            "evaluations": evaluation_audits,
        }
        oracle_results[str(seed)] = {
            name: _evaluate_all_splits(
                name,
                None,
                evaluations,
                task,
                run_config,
                device,
                rotor_scan_mode,
                quaternion_scan_mode,
            )
            for name in ORACLE_CANDIDATES
        }
        factories: dict[str, Callable[[], nn.Module]] = {
            "pure_rotor": lambda config=run_config: pure_rotor_model(
                task, config, max_rotor_angle=math.pi
            ),
            "identity_rotation_ablation": lambda config=run_config: pure_rotor_model(
                task, config, max_rotor_angle=0.0
            ),
            "spin_quaternion_scan": lambda config=run_config: spin_quaternion_model(
                task, config
            ),
            "mamba2_transformers": lambda config=run_config: mamba2_model(task, config),
        }
        for name in TRAINED_CANDIDATES:
            results.append(
                train_one(
                    name,
                    factories[name],
                    task,
                    training,
                    evaluations,
                    run_config,
                    device,
                    rotor_scan_mode,
                    quaternion_scan_mode,
                    checkpoint_directory,
                )
            )
    return {
        "experiment": "center-sensitive 2.A5 prefix tracking",
        "status": "completed empirical screen; exact oracles are controls, not trained models",
        "pure_rotor_model_version": PURE_ROTOR_VERSION,
        "device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
        ),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "config": asdict(config),
        "seeds": list(seeds),
        "rotor_scan_mode": rotor_scan_mode,
        "quaternion_scan_mode": quaternion_scan_mode,
        "task": {
            "group": {"key": task.group.key, "name": task.group.name, "order": 120},
            "projective_order": 60,
            "presentation": task.presentation,
            "group_table_sha256": task.group_table_sha256,
            "target": "every ordered binary-icosahedral prefix product",
            "paired_relation": "a*a=z versus b*b_inverse=e",
            "split": split_audits,
        },
        "integrity": {
            "same_precomputed_training_batches_per_seed": True,
            "same_paired_evaluation_batches_per_seed": True,
            "candidate_initialized_after_seed": True,
            "pure_rotor_vs_identity_initial_function_identical": True,
            "parameter_counts": counts,
            "maximum_relative_parameter_gap": (
                (max(counts.values()) - min(counts.values())) / max(counts.values())
            ),
            "mamba2_backend": "huggingface_transformers",
            "mamba_ssm_extension_importable": importlib.util.find_spec("mamba_ssm")
            is not None,
            "mamba2_fused_kernel_claimed": False,
            "state_size_matched": False,
            "state_matching_note": (
                "Candidates are parameter-near but retain architecture-specific states; "
                "the Spin quaternion scan uses 8 persistent quaternion lanes."
            ),
            "transport_center_action": {
                "pure_rotor_sandwich": "q and -q induce the same conjugation",
                "spin_quaternion_scan": "q and -q are distinct persistent states",
            },
        },
        "oracle_results": oracle_results,
        "results": results,
        "claim_scope": {
            "empirical": [
                "paired center acquisition and retention for the configured models",
                "parameter-near comparison on identical symbolic schedules",
            ],
            "not_claimed": [
                "a theorem about Mamba-2 or diagonal SSMs",
                "a language-model quality result",
                "a state-matched systems comparison",
                "that the projective transport alone can represent the binary center",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--training-length", type=int, default=16)
    parser.add_argument("--validation-batches", type=int, default=2)
    parser.add_argument("--validation-pairs-per-batch", type=int, default=32)
    parser.add_argument("--evaluation-microbatch-size", type=int, default=16)
    parser.add_argument("--evaluation-lengths", default="2,16,64,128")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--rotor-scan-mode",
        choices=("parallel", "schur_parallel", "recurrent"),
        default="parallel",
    )
    parser.add_argument(
        "--quaternion-scan-mode",
        choices=("parallel", "recurrent"),
        default="parallel",
    )
    parser.add_argument("--allow-incomplete-training-coverage", action="store_true")
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quiet-report",
        action="store_true",
        help="write the JSON artifact without echoing the full report to stdout",
    )
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
    config = BinaryA5BenchmarkConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        training_length=args.training_length,
        validation_batches=args.validation_batches,
        validation_pairs_per_batch=args.validation_pairs_per_batch,
        evaluation_microbatch_size=args.evaluation_microbatch_size,
        evaluation_lengths=parse_lengths(args.evaluation_lengths),
        enforce_training_state_coverage=not args.allow_incomplete_training_coverage,
    )
    report = run_benchmark(
        config,
        seeds=parse_seeds(args.seeds),
        device=torch.device(args.device),
        rotor_scan_mode=args.rotor_scan_mode,
        quaternion_scan_mode=args.quaternion_scan_mode,
        checkpoint_directory=args.checkpoint_directory,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not args.quiet_report:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
