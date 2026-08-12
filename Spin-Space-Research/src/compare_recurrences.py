"""Controlled recurrence-family experiments on finite noncommutative groups.

The six variants have identical parameter tensor shapes, initial parameters,
state size, training batches, optimizer, and surrounding network.  At step
zero every transition is the same positive decay because all rotation
controllers start at identity. Every position is supervised with the ordered
prefix product, while final-position accuracy measures full-sequence tracking.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from recurrence_families_torch import (
    FAMILY_NAMES,
    STATE_WIDTH,
    RecurrenceSequenceModel,
)


@dataclass(frozen=True)
class FiniteGroup:
    key: str
    name: str
    elements: tuple[str, ...]
    table: np.ndarray

    @property
    def order(self) -> int:
        return len(self.elements)


@dataclass(frozen=True)
class HarnessConfig:
    steps: int = 300
    batch_size: int = 256
    sequence_length: int = 16
    validation_batches: int = 8
    validation_batch_size: int = 512
    channels: int = 4
    layers: int = 2
    expansion: int = 2
    learning_rate: float = 3e-3
    seed: int = 0
    diagnostic_interval: int = 0


def q8_multiplication_table() -> np.ndarray:
    """Return the Cayley table for ``[1,i,j,k,-1,-i,-j,-k]``."""
    # Entries are (sign bit, unsigned basis) for 1, i, j, k.
    unsigned = (
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 1), (1, 0), (0, 3), (1, 2)),
        ((0, 2), (1, 3), (1, 0), (0, 1)),
        ((0, 3), (0, 2), (1, 1), (1, 0)),
    )
    table = np.empty((8, 8), dtype=np.int64)
    for left in range(8):
        for right in range(8):
            left_sign, left_basis = divmod(left, 4)
            right_sign, right_basis = divmod(right, 4)
            product_sign, product_basis = unsigned[left_basis][right_basis]
            sign = left_sign ^ right_sign ^ product_sign
            table[left, right] = sign * 4 + product_basis
    return table


Q8_TABLE = q8_multiplication_table()


def d4_multiplication_table() -> np.ndarray:
    """Dihedral group D4 in element order ``r^a s^b``."""
    table = np.empty((8, 8), dtype=np.int64)
    for left in range(8):
        for right in range(8):
            left_flip, left_rotation = divmod(left, 4)
            right_flip, right_rotation = divmod(right, 4)
            rotation = (
                left_rotation
                + (-1 if left_flip else 1) * right_rotation
            ) % 4
            flip = left_flip ^ right_flip
            table[left, right] = flip * 4 + rotation
    return table


def s3_multiplication_table() -> tuple[np.ndarray, tuple[str, ...]]:
    """Symmetric group S3 using permutation-function composition."""
    permutations = tuple(itertools.permutations(range(3)))
    lookup = {permutation: index for index, permutation in enumerate(permutations)}
    table = np.empty((6, 6), dtype=np.int64)
    for left_index, left in enumerate(permutations):
        for right_index, right in enumerate(permutations):
            composition = tuple(left[right[index]] for index in range(3))
            table[left_index, right_index] = lookup[composition]
    names = tuple("".join(str(value + 1) for value in item) for item in permutations)
    return table, names


def permutation_is_even(permutation: tuple[int, ...]) -> bool:
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return inversions % 2 == 0


def a5_multiplication_table() -> tuple[np.ndarray, tuple[str, ...]]:
    """Alternating group A5 using even permutation-function composition."""
    permutations = tuple(
        permutation
        for permutation in itertools.permutations(range(5))
        if permutation_is_even(permutation)
    )
    lookup = {permutation: index for index, permutation in enumerate(permutations)}
    table = np.empty((60, 60), dtype=np.int64)
    for left_index, left in enumerate(permutations):
        for right_index, right in enumerate(permutations):
            composition = tuple(left[right[index]] for index in range(5))
            table[left_index, right_index] = lookup[composition]
    names = tuple("".join(str(value + 1) for value in item) for item in permutations)
    return table, names


S3_TABLE, S3_ELEMENTS = s3_multiplication_table()
A5_TABLE, A5_ELEMENTS = a5_multiplication_table()
GROUPS = {
    "q8": FiniteGroup(
        "q8",
        "quaternion group Q8",
        ("1", "i", "j", "k", "-1", "-i", "-j", "-k"),
        Q8_TABLE,
    ),
    "d4": FiniteGroup(
        "d4",
        "dihedral group D4",
        ("1", "r", "r2", "r3", "s", "rs", "r2s", "r3s"),
        d4_multiplication_table(),
    ),
    "s3": FiniteGroup("s3", "symmetric group S3", S3_ELEMENTS, S3_TABLE),
    "a5": FiniteGroup("a5", "alternating group A5", A5_ELEMENTS, A5_TABLE),
}


def q8_products(tokens: np.ndarray) -> np.ndarray:
    return q8_prefix_products(tokens)[:, -1]


def q8_prefix_products(tokens: np.ndarray) -> np.ndarray:
    return group_prefix_products(tokens, GROUPS["q8"])


def group_prefix_products(tokens: np.ndarray, group: FiniteGroup) -> np.ndarray:
    products = np.zeros(tokens.shape[0], dtype=np.int64)
    prefixes = np.empty_like(tokens)
    for position in range(tokens.shape[1]):
        products = group.table[products, tokens[:, position]]
        prefixes[:, position] = products
    return prefixes


def make_q8_batches(
    count: int,
    batch_size: int,
    sequence_length: int,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return make_group_batches(
        GROUPS["q8"], count, batch_size, sequence_length, seed
    )


def make_group_batches(
    group: FiniteGroup,
    count: int,
    batch_size: int,
    sequence_length: int,
    seed: int,
    *,
    input_elements: tuple[int, ...] | None = None,
    held_out_pairs: tuple[tuple[int, int], ...] = (),
    require_held_out_pair: bool = False,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if require_held_out_pair and not held_out_pairs:
        raise ValueError("held-out pairs are required for the requested evaluation split")
    if held_out_pairs and sequence_length < 2:
        raise ValueError("held-out pair splits require sequence length >= 2")
    input_elements = input_elements or tuple(range(group.order))
    if not input_elements or len(set(input_elements)) != len(input_elements):
        raise ValueError("input elements must be non-empty and unique")
    if any(not 0 <= element < group.order for element in input_elements):
        raise ValueError(f"input element is outside {group.key}")
    input_order = len(input_elements)
    for left, right in held_out_pairs:
        if not (0 <= left < input_order and 0 <= right < input_order):
            raise ValueError(f"invalid held-out input pair {(left, right)}")
    generator = np.random.default_rng(seed)
    batches = []
    for _ in range(count):
        tokens = generator.integers(
            0, input_order, size=(batch_size, sequence_length), dtype=np.int64
        )
        if held_out_pairs and require_held_out_pair:
            positions = generator.integers(0, sequence_length - 1, size=batch_size)
            pair_choices = generator.integers(0, len(held_out_pairs), size=batch_size)
            for row, (position, pair_choice) in enumerate(
                zip(positions, pair_choices)
            ):
                tokens[row, position : position + 2] = held_out_pairs[pair_choice]
        elif held_out_pairs:
            forbidden = set(held_out_pairs)
            for position in range(1, sequence_length):
                previous = tokens[:, position - 1]
                current = tokens[:, position]
                invalid = np.fromiter(
                    (
                        (int(left), int(right)) in forbidden
                        for left, right in zip(previous, current)
                    ),
                    dtype=bool,
                    count=batch_size,
                )
                while np.any(invalid):
                    current[invalid] = generator.integers(
                        0, input_order, size=int(invalid.sum()), dtype=np.int64
                    )
                    invalid = np.fromiter(
                        (
                            (int(left), int(right)) in forbidden
                            for left, right in zip(previous, current)
                        ),
                        dtype=bool,
                        count=batch_size,
                    )
        group_tokens = np.asarray(input_elements, dtype=np.int64)[tokens]
        targets = group_prefix_products(group_tokens, group)
        batches.append((torch.from_numpy(tokens), torch.from_numpy(targets)))
    return batches


def parse_held_out_pairs(
    specifications: list[str], group: FiniteGroup
) -> tuple[tuple[int, int], ...]:
    lookup = {element: index for index, element in enumerate(group.elements)}

    def parse_element(value: str) -> int:
        if value in lookup:
            return lookup[value]
        try:
            index = int(value)
        except ValueError as error:
            raise ValueError(f"unknown {group.key} element {value!r}") from error
        if not 0 <= index < group.order:
            raise ValueError(f"element index {index} is outside {group.key}")
        return index

    pairs = []
    for specification in specifications:
        parts = specification.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"held-out pair {specification!r} must have LEFT:RIGHT form"
            )
        pairs.append((parse_element(parts[0]), parse_element(parts[1])))
    if len(set(pairs)) != len(pairs):
        raise ValueError("held-out transition pairs must be unique")
    return tuple(pairs)


def parse_input_elements(
    specifications: list[str], group: FiniteGroup
) -> tuple[int, ...]:
    if not specifications:
        return tuple(range(group.order))
    lookup = {element: index for index, element in enumerate(group.elements)}
    parsed = []
    for specification in specifications:
        if specification in lookup:
            parsed.append(lookup[specification])
            continue
        try:
            index = int(specification)
        except ValueError as error:
            raise ValueError(
                f"unknown {group.key} input element {specification!r}"
            ) from error
        if not 0 <= index < group.order:
            raise ValueError(f"input element index {index} is outside {group.key}")
        parsed.append(index)
    if len(set(parsed)) != len(parsed):
        raise ValueError("input elements must be unique")
    return tuple(parsed)


def pair_split_audit(
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    held_out_pairs: tuple[tuple[int, int], ...],
) -> dict[str, int]:
    occurrences = 0
    sequences_with_pair = 0
    total_sequences = 0
    for tokens, _ in batches:
        adjacent = torch.stack((tokens[:, :-1], tokens[:, 1:]), dim=-1)
        matches = torch.zeros(adjacent.shape[:2], dtype=torch.bool)
        for left, right in held_out_pairs:
            matches |= (adjacent[..., 0] == left) & (adjacent[..., 1] == right)
        occurrences += int(matches.sum())
        sequences_with_pair += int(matches.any(dim=1).sum())
        total_sequences += tokens.shape[0]
    return {
        "pair_occurrences": occurrences,
        "sequences_with_pair": sequences_with_pair,
        "total_sequences": total_sequences,
    }


def state_and_pair_coverage_audit(
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    input_order: int,
    group_order: int,
) -> dict[str, int | float | list[int]]:
    """Audit whether a pair split accidentally collapses the training language."""
    seen_targets: set[int] = set()
    seen_pairs: set[tuple[int, int]] = set()
    for tokens, targets in batches:
        seen_targets.update(int(value) for value in torch.unique(targets))
        if tokens.shape[1] >= 2:
            adjacent = torch.stack((tokens[:, :-1], tokens[:, 1:]), dim=-1)
            seen_pairs.update(
                (int(left), int(right))
                for left, right in adjacent.reshape(-1, 2)
            )
    possible_pairs = input_order * input_order
    return {
        "observed_group_states": len(seen_targets),
        "group_state_fraction": len(seen_targets) / group_order,
        "missing_group_state_indices": sorted(set(range(group_order)) - seen_targets),
        "observed_input_pairs": len(seen_pairs),
        "possible_input_pairs": possible_pairs,
        "input_pair_fraction": len(seen_pairs) / possible_pairs,
    }


@torch.no_grad()
def evaluate(
    model: RecurrenceSequenceModel,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    prefix_correct = 0
    prefix_examples = 0
    final_correct = 0
    final_examples = 0
    for tokens, targets in batches:
        tokens, targets = tokens.to(device), targets.to(device)
        logits = model(tokens)
        total_loss += float(
            torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), reduction="sum"
            )
        )
        predictions = logits.argmax(dim=-1)
        prefix_correct += int((predictions == targets).sum())
        prefix_examples += targets.numel()
        final_correct += int((predictions[:, -1] == targets[:, -1]).sum())
        final_examples += targets.shape[0]
    return (
        total_loss / prefix_examples,
        prefix_correct / prefix_examples,
        final_correct / final_examples,
    )


@torch.no_grad()
def streaming_equivalence(
    model: RecurrenceSequenceModel, tokens: torch.Tensor
) -> dict[str, float]:
    model.eval()
    full_logits, full_states = model(tokens, return_recurrent_states=True)
    split = max(1, tokens.shape[1] // 2)
    first_logits, states = model(
        tokens[:, :split], return_recurrent_states=True
    )
    second_logits, states = model(
        tokens[:, split:], states, return_recurrent_states=True
    )
    chunked_logits = torch.cat((first_logits, second_logits), dim=1)

    stream_logits = []
    stream_states = None
    for position in range(tokens.shape[1]):
        logits, stream_states = model(
            tokens[:, position : position + 1],
            stream_states,
            return_recurrent_states=True,
        )
        stream_logits.append(logits)
    streamed_logits = torch.cat(stream_logits, dim=1)
    state_error = max(
        float((expected - actual).abs().max())
        for expected, actual in zip(full_states, stream_states)
    )
    return {
        "chunked_logit_max_abs_error": float(
            (full_logits - chunked_logits).abs().max()
        ),
        "streaming_logit_max_abs_error": float(
            (full_logits - streamed_logits).abs().max()
        ),
        "streaming_state_max_abs_error": state_error,
    }


@torch.no_grad()
def transition_diagnostics(
    model: RecurrenceSequenceModel, tokens: torch.Tensor
) -> list[dict[str, float | str]]:
    outputs = model.token_embedding(tokens).reshape(
        tokens.shape[0], tokens.shape[1], model.channels, STATE_WIDTH
    )
    diagnostics = []
    for block in model.blocks:
        normalized = block.input_norm(outputs.flatten(-2)).reshape_as(outputs)
        diagnostics.append(block.recurrence.diagnostics(normalized))
        outputs, _ = block(outputs)
    return diagnostics


@torch.no_grad()
def recurrent_state_diagnostics(
    model: RecurrenceSequenceModel, tokens: torch.Tensor
) -> list[dict[str, float]]:
    _, states = model(tokens, return_recurrent_states=True)
    diagnostics = []
    for state in states:
        flattened = state.flatten(1).float()
        centered = flattened - flattened.mean(dim=0, keepdim=True)
        singular_values = torch.linalg.svdvals(centered)
        squared = singular_values.square()
        participation = squared.sum().square() / squared.square().sum().clamp_min(1e-12)
        diagnostics.append(
            {
                "state_rms": float(state.float().square().mean().sqrt()),
                "mean_state_norm": float(state.float().flatten(-2).norm(dim=-1).mean()),
                "max_state_norm": float(state.float().flatten(-2).norm(dim=-1).max()),
                "spectral_participation_ratio": float(participation),
            }
        )
    return diagnostics


def controller_gradient_norms(model: RecurrenceSequenceModel) -> list[float]:
    norms = []
    for block in model.blocks:
        gradients = [
            parameter.grad.detach().float().norm().square()
            for parameter in block.recurrence.control_projection.parameters()
            if parameter.grad is not None
        ]
        norms.append(float(torch.stack(gradients).sum().sqrt()) if gradients else 0.0)
    return norms


def run_variant(
    family: str,
    group: FiniteGroup,
    input_order: int,
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    validation_batches: list[tuple[torch.Tensor, torch.Tensor]],
    generalization_batches: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
    config: HarnessConfig,
    device: torch.device,
) -> dict:
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model = RecurrenceSequenceModel(
        vocab_size=input_order,
        output_size=group.order,
        family=family,
        channels=config.channels,
        num_layers=config.layers,
        expansion=config.expansion,
    ).to(device)
    shape_payload = json.dumps(
        [(name, list(parameter.shape)) for name, parameter in model.named_parameters()],
        separators=(",", ":"),
    ).encode("utf-8")
    parameter_shape_sha256 = hashlib.sha256(shape_payload).hexdigest()
    parameter_hash = hashlib.sha256()
    for name, parameter in model.named_parameters():
        parameter_hash.update(name.encode("utf-8"))
        parameter_hash.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    initial_parameter_sha256 = parameter_hash.hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    initial_loss, initial_prefix_accuracy, initial_final_accuracy = evaluate(
        model, validation_batches, device
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loss_samples = {}
    training_trajectory = {}
    start = time.perf_counter()
    model.train()
    for step, (tokens, targets) in enumerate(train_batches, start=1):
        tokens, targets = tokens.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.flatten()
        )
        loss.backward()
        preclip_gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        control_gradient_norms = controller_gradient_norms(model)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == config.steps:
            loss_samples[str(step)] = float(loss.detach())
            print(
                f"{family} step={step}/{config.steps} "
                f"loss={loss_samples[str(step)]:.4f}"
            )
        if config.diagnostic_interval and (
            step == 1 or step % config.diagnostic_interval == 0 or step == config.steps
        ):
            diagnostic_tokens = tokens[: min(64, tokens.shape[0])]
            actions = transition_diagnostics(model, diagnostic_tokens)
            states = recurrent_state_diagnostics(model, diagnostic_tokens)
            training_trajectory[str(step)] = {
                "loss": float(loss.detach()),
                "preclip_gradient_norm": float(preclip_gradient_norm),
                "layers": [
                    {
                        **action,
                        **state,
                        "controller_gradient_norm": control_gradient_norm,
                    }
                    for action, state, control_gradient_norm in zip(
                        actions, states, control_gradient_norms
                    )
                ],
            }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    final_loss, final_prefix_accuracy, final_position_accuracy = evaluate(
        model, validation_batches, device
    )
    length_generalization = {}
    for length, batches in generalization_batches.items():
        loss, prefix_accuracy, last_accuracy = evaluate(model, batches, device)
        length_generalization[str(length)] = {
            "validation_loss": loss,
            "prefix_accuracy": prefix_accuracy,
            "final_position_accuracy": last_accuracy,
        }
    probe_tokens = validation_batches[0][0][:4].to(device)

    return {
        "family": family,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_shape_sha256": parameter_shape_sha256,
        "initial_parameter_sha256": initial_parameter_sha256,
        "state_scalars_per_sequence": config.layers * config.channels * STATE_WIDTH,
        "initial_validation_loss": initial_loss,
        "initial_validation_prefix_accuracy": initial_prefix_accuracy,
        "initial_validation_final_position_accuracy": initial_final_accuracy,
        "final_validation_loss": final_loss,
        "final_validation_prefix_accuracy": final_prefix_accuracy,
        "final_validation_final_position_accuracy": final_position_accuracy,
        "length_generalization": length_generalization,
        "elapsed_seconds": elapsed,
        "steps_per_second": config.steps / elapsed,
        "peak_cuda_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda"
            else 0.0
        ),
        "loss_samples": loss_samples,
        "training_trajectory": training_trajectory,
        "streaming_equivalence": streaming_equivalence(model, probe_tokens),
        "transition_diagnostics": transition_diagnostics(model, probe_tokens),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=512)
    parser.add_argument(
        "--eval-lengths",
        nargs="*",
        type=int,
        help="Held-out lengths; defaults to 2, 4, 8, train length, and 2x train length",
    )
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--diagnostic-interval",
        type=int,
        default=0,
        help="Record gradient, transition, and state-spectrum diagnostics every N steps",
    )
    parser.add_argument("--group", choices=tuple(GROUPS), default="q8")
    parser.add_argument(
        "--input-elements",
        nargs="*",
        default=[],
        metavar="ELEMENT",
        help="Restrict inputs to a generating subset while retaining all group outputs",
    )
    parser.add_argument(
        "--held-out-pairs",
        nargs="*",
        default=[],
        metavar="LEFT:RIGHT",
        help=(
            "Exclude ordered adjacent pairs from training and require one in every "
            "validation sequence; elements may be names or zero-based indices"
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--families", nargs="+", choices=FAMILY_NAMES, default=list(FAMILY_NAMES)
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.seq_len < 1 or args.batch_size < 1:
        raise ValueError("steps, sequence length, and batch size must be positive")
    if args.diagnostic_interval < 0:
        raise ValueError("diagnostic interval cannot be negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    config = HarnessConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.seq_len,
        validation_batches=args.validation_batches,
        validation_batch_size=args.validation_batch_size,
        channels=args.channels,
        layers=args.layers,
        learning_rate=args.learning_rate,
        seed=args.seed,
        diagnostic_interval=args.diagnostic_interval,
    )
    group = GROUPS[args.group]
    input_elements = parse_input_elements(args.input_elements, group)
    held_out_group_pairs = parse_held_out_pairs(args.held_out_pairs, group)
    input_lookup = {
        group_element: token for token, group_element in enumerate(input_elements)
    }
    try:
        held_out_pairs = tuple(
            (input_lookup[left], input_lookup[right])
            for left, right in held_out_group_pairs
        )
    except KeyError as error:
        raise ValueError("every held-out pair element must be in the input alphabet") from error
    train_batches = make_group_batches(
        group,
        config.steps,
        config.batch_size,
        config.sequence_length,
        config.seed + 1000,
        input_elements=input_elements,
        held_out_pairs=held_out_pairs,
    )
    validation_batches = make_group_batches(
        group,
        config.validation_batches,
        config.validation_batch_size,
        config.sequence_length,
        91_337,
        input_elements=input_elements,
        held_out_pairs=held_out_pairs,
        require_held_out_pair=bool(held_out_pairs),
    )
    evaluation_lengths = args.eval_lengths or sorted(
        {2, 4, 8, config.sequence_length, 2 * config.sequence_length}
    )
    if any(length < 1 for length in evaluation_lengths):
        raise ValueError("evaluation lengths must be positive")
    generalization_batches = {
        length: make_group_batches(
            group,
            config.validation_batches,
            config.validation_batch_size,
            length,
            91_337 + length,
            input_elements=input_elements,
            held_out_pairs=held_out_pairs,
            require_held_out_pair=bool(held_out_pairs),
        )
        for length in evaluation_lengths
    }
    results = [
        run_variant(
            family,
            group,
            len(input_elements),
            train_batches,
            validation_batches,
            generalization_batches,
            config,
            device,
        )
        for family in args.families
    ]
    report = {
        "experiment": f"ordered {group.key.upper()} prefix-product state tracking",
        "device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
        ),
        "torch_version": torch.__version__,
        "config": asdict(config),
        "group": {
            "key": group.key,
            "name": group.name,
            "order": group.order,
            "element_order": list(group.elements),
            "multiplication_table": group.table.tolist(),
        },
        "input_alphabet": [
            {
                "token_index": token,
                "group_index": element,
                "element": group.elements[element],
            }
            for token, element in enumerate(input_elements)
        ],
        "evaluation_lengths": evaluation_lengths,
        "held_out_transition_pairs": [
            {
                "left_token_index": left,
                "right_token_index": right,
                "left_group_index": group_left,
                "right_group_index": group_right,
                "left_element": group.elements[group_left],
                "right_element": group.elements[group_right],
            }
            for (left, right), (group_left, group_right) in zip(
                held_out_pairs, held_out_group_pairs
            )
        ],
        "data_split_audit": (
            {
                "training": pair_split_audit(train_batches, held_out_pairs),
                "validation": pair_split_audit(validation_batches, held_out_pairs),
                "generalization": {
                    str(length): pair_split_audit(batches, held_out_pairs)
                    for length, batches in generalization_batches.items()
                },
                "acceptance": {
                    "zero_training_pair_occurrences": (
                        pair_split_audit(train_batches, held_out_pairs)[
                            "pair_occurrences"
                        ]
                        == 0
                    ),
                    "every_evaluation_sequence_contains_pair": all(
                        audit["sequences_with_pair"] == audit["total_sequences"]
                        for audit in (
                            pair_split_audit(validation_batches, held_out_pairs),
                            *(
                                pair_split_audit(batches, held_out_pairs)
                                for batches in generalization_batches.values()
                            ),
                        )
                    ),
                },
            }
            if held_out_pairs
            else None
        ),
        "language_coverage_audit": {
            "training": state_and_pair_coverage_audit(
                train_batches,
                input_order=len(input_elements),
                group_order=group.order,
            ),
            "validation": state_and_pair_coverage_audit(
                validation_batches,
                input_order=len(input_elements),
                group_order=group.order,
            ),
        },
        "fairness_contract": {
            "identical_parameter_count": len(
                {result["parameters"] for result in results}
            )
            == 1,
            "identical_parameter_shapes": len(
                {result["parameter_shape_sha256"] for result in results}
            )
            == 1,
            "identical_initial_parameters": len(
                {result["initial_parameter_sha256"] for result in results}
            )
            == 1,
            "identical_initial_function": (
                max(result["initial_validation_loss"] for result in results)
                - min(result["initial_validation_loss"] for result in results)
                < 1e-7
            ),
            "same_training_batches": True,
            "same_validation_batches": True,
            "same_real_state_width_per_channel": STATE_WIDTH,
        },
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
