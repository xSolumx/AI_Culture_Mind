"""Hard one-hot/diagonal baselines for the finite-group mechanism gate.

PD-SSM uses a column one-hot transition P and a complex diagonal D.  This
mechanistic specialization assigns one hard transition to each fixed token,
removes writes/residuals, and uses a unit-modulus D so long-horizon state
tracking is not confounded by decay.  The learned model uses the same
straight-through hard-one-hot idea as the reference implementation.  The exact
regular action is reported separately as an oracle ceiling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.nn import functional as F

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - only incomplete optional runtimes
    linear_sum_assignment = None

from compare_recurrences import (
    GROUPS,
    FiniteGroup,
    make_group_batches,
    pair_split_audit,
    parse_held_out_pairs,
    parse_input_elements,
    state_and_pair_coverage_audit,
)


@dataclass(frozen=True)
class PDConfig:
    steps: int = 1500
    batch_size: int = 256
    sequence_length: int = 16
    validation_batches: int = 2
    validation_batch_size: int = 512
    state_size: int = 60
    learning_rate: float = 3e-3
    warmup_fraction: float = 0.0
    final_learning_rate: float = 1e-5
    seed: int = 0


def _hard_column_one_hot(scores: torch.Tensor) -> torch.Tensor:
    """Return STE matrices with one hard nonzero in every input column."""
    soft = scores.softmax(dim=-2)
    indices = soft.argmax(dim=-2)
    hard = F.one_hot(indices, num_classes=scores.shape[-2]).movedim(-1, -2)
    return hard.to(soft.dtype) + soft - soft.detach()


def _sinkhorn(scores: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    log_probabilities = scores
    for _ in range(iterations):
        log_probabilities = log_probabilities - torch.logsumexp(
            log_probabilities, dim=-1, keepdim=True
        )
        log_probabilities = log_probabilities - torch.logsumexp(
            log_probabilities, dim=-2, keepdim=True
        )
    return log_probabilities.exp()


def _hard_projected_permutation(scores: torch.Tensor) -> torch.Tensor:
    """Hungarian hard forward projection with a Sinkhorn gradient surrogate."""
    if linear_sum_assignment is None:
        raise RuntimeError("projected permutations require scipy")
    soft = _sinkhorn(scores)
    hard = torch.zeros_like(soft)
    for token in range(scores.shape[0]):
        rows, columns = linear_sum_assignment(
            -scores[token].detach().cpu().double().numpy()
        )
        hard[token, rows, columns] = 1.0
    return hard + soft - soft.detach()


class LearnedHardPD(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        output_size: int,
        state_size: int,
        *,
        project_permutation: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.state_size = state_size
        self.project_permutation = project_permutation
        self.transition_scores = nn.Parameter(
            torch.randn(vocab_size, state_size, state_size) / math.sqrt(state_size)
        )
        self.diagonal_phases = nn.Parameter(torch.zeros(vocab_size, state_size))
        self.initial_index_logits = nn.Parameter(torch.zeros(state_size))
        self.output_head = nn.Linear(2 * state_size, output_size)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def transition_matrices(self) -> torch.Tensor:
        p = (
            _hard_projected_permutation(self.transition_scores)
            if self.project_permutation
            else _hard_column_one_hot(self.transition_scores)
        )
        phase = torch.polar(torch.ones_like(self.diagonal_phases), self.diagonal_phases)
        # P @ D: each input column is phase-scaled before its hard transition.
        return p.to(phase.dtype) * phase.unsqueeze(-2)

    def initial_state(self, batch_size: int) -> torch.Tensor:
        soft = self.initial_index_logits.softmax(dim=0)
        hard = F.one_hot(soft.argmax(dim=0), self.state_size).to(soft.dtype)
        initial = hard + soft - soft.detach()
        return torch.complex(initial, torch.zeros_like(initial)).expand(batch_size, -1)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        features = torch.cat((state.real, state.imag), dim=-1)
        return self.logit_scale.exp().clamp(max=100.0) * self.output_head(features)

    def forward(
        self,
        tokens: torch.Tensor,
        recurrent_state: torch.Tensor | None = None,
        *,
        return_recurrent_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        state = (
            self.initial_state(tokens.shape[0])
            if recurrent_state is None
            else recurrent_state
        )
        transitions = self.transition_matrices()
        states = []
        for position in range(tokens.shape[1]):
            selected = transitions[tokens[:, position]]
            state = torch.einsum("bij,bj->bi", selected, state)
            states.append(state)
        logits = self.decode(torch.stack(states, dim=1))
        return (logits, state) if return_recurrent_state else logits


class ExactRegularPD(nn.Module):
    """Oracle N-state regular action with D=I and an exact linear decoder."""

    def __init__(self, group: FiniteGroup, input_elements: tuple[int, ...]) -> None:
        super().__init__()
        transitions = torch.zeros(len(input_elements), group.order, group.order)
        for token, element in enumerate(input_elements):
            for source in range(group.order):
                target = int(group.table[source, element])
                transitions[token, target, source] = 1.0
        self.register_buffer("transitions", transitions.to(torch.complex64))
        self.state_size = group.order
        self.output_size = group.order

    def initial_state(self, batch_size: int) -> torch.Tensor:
        state = torch.zeros(batch_size, self.state_size, dtype=torch.complex64)
        state[:, 0] = 1.0
        return state

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return 100.0 * state.real

    def forward(
        self,
        tokens: torch.Tensor,
        recurrent_state: torch.Tensor | None = None,
        *,
        return_recurrent_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        state = (
            self.initial_state(tokens.shape[0]).to(tokens.device)
            if recurrent_state is None
            else recurrent_state
        )
        states = []
        for position in range(tokens.shape[1]):
            selected = self.transitions[tokens[:, position]]
            state = torch.einsum("bij,bj->bi", selected, state)
            states.append(state)
        logits = self.decode(torch.stack(states, dim=1))
        return (logits, state) if return_recurrent_state else logits


@torch.no_grad()
def evaluate(
    model: nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    prefix_correct = final_correct = examples = prefix_examples = 0
    maximum_norm_error = 0.0
    for tokens, targets in batches:
        tokens, targets = tokens.to(device), targets.to(device)
        logits, final_state = model(tokens, return_recurrent_state=True)
        total_loss += float(
            F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="sum")
        )
        predictions = logits.argmax(dim=-1)
        prefix_correct += int((predictions == targets).sum())
        final_correct += int((predictions[:, -1] == targets[:, -1]).sum())
        prefix_examples += targets.numel()
        examples += len(tokens)
        maximum_norm_error = max(
            maximum_norm_error,
            float((final_state.norm(dim=-1) - 1.0).abs().max()),
        )
    return {
        "validation_loss": total_loss / prefix_examples,
        "prefix_accuracy": prefix_correct / prefix_examples,
        "final_position_accuracy": final_correct / examples,
        "maximum_final_state_norm_error": maximum_norm_error,
    }


@torch.no_grad()
def transition_diagnostics(model: LearnedHardPD) -> dict[str, object]:
    hard = (
        _hard_projected_permutation(model.transition_scores)
        if model.project_permutation
        else _hard_column_one_hot(model.transition_scores)
    ).detach()
    column_sums = hard.sum(dim=-2)
    row_sums = hard.sum(dim=-1)
    unique_targets = (row_sums > 0).sum(dim=-1)
    permutation = torch.isclose(
        row_sums, torch.ones_like(row_sums), atol=1e-6, rtol=0
    ).all(dim=-1)
    return {
        "hard_column_one_hot_error": float((column_sums - 1.0).abs().max()),
        "unique_targets_per_token": [int(value) for value in unique_targets],
        "is_bijection_per_token": [bool(value) for value in permutation],
        "collision_count_per_token": [
            model.state_size - int(value) for value in unique_targets
        ],
        "mean_absolute_diagonal_phase": float(model.diagonal_phases.abs().mean()),
        "hard_projection": (
            "hungarian_permutation" if model.project_permutation else "column_one_hot"
        ),
    }


@torch.no_grad()
def streaming_equivalence(model: nn.Module, tokens: torch.Tensor) -> dict[str, float]:
    full_logits, full_state = model(tokens, return_recurrent_state=True)
    state = None
    pieces = []
    for position in range(tokens.shape[1]):
        logits, state = model(
            tokens[:, position : position + 1],
            recurrent_state=state,
            return_recurrent_state=True,
        )
        pieces.append(logits)
    streamed = torch.cat(pieces, dim=1)
    return {
        "logit_max_abs_error": float((streamed - full_logits).abs().max()),
        "state_max_abs_error": float((state - full_state).abs().max()),
    }


def parameter_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run_learned(
    group: FiniteGroup,
    input_elements: tuple[int, ...],
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    validation_batches: list[tuple[torch.Tensor, torch.Tensor]],
    generalization_batches: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
    config: PDConfig,
    device: torch.device,
    project_permutation: bool = False,
) -> dict[str, object]:
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = LearnedHardPD(
        len(input_elements),
        group.order,
        config.state_size,
        project_permutation=project_permutation,
    ).to(device)
    initial_hash = parameter_fingerprint(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    warmup_steps = int(config.warmup_fraction * config.steps)

    def learning_rate_multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        decay_steps = max(1, config.steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        final_ratio = config.final_learning_rate / config.learning_rate
        return final_ratio + (1.0 - final_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=learning_rate_multiplier
    )
    start = time.perf_counter()
    model.train()
    for step, (tokens, targets) in enumerate(train_batches, start=1):
        tokens, targets = tokens.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 50 == 0 or step == config.steps:
            print(
                f"learned_hard_pd step={step}/{config.steps} "
                f"loss={float(loss.detach()):.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.8f}"
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    probe = validation_batches[0][0][:8].to(device)
    return {
        "family": (
            "learned_hard_projected_permutation_unit_pd"
            if project_permutation
            else "learned_hard_column_one_hot_unit_pd"
        ),
        "oracle": False,
        "parameters": sum(p.numel() for p in model.parameters()),
        "complex_state_size": config.state_size,
        "real_state_scalars": 2 * config.state_size,
        "initial_parameter_sha256": initial_hash,
        "validation": evaluate(model, validation_batches, device),
        "length_generalization": {
            str(length): evaluate(model, batches, device)
            for length, batches in generalization_batches.items()
        },
        "transition_diagnostics": transition_diagnostics(model),
        "streaming_equivalence": streaming_equivalence(model, probe),
        "elapsed_seconds": elapsed,
        "steps_per_second": config.steps / elapsed,
        "final_optimizer_learning_rate": optimizer.param_groups[0]["lr"],
    }


def run_oracle(
    group: FiniteGroup,
    input_elements: tuple[int, ...],
    validation_batches: list[tuple[torch.Tensor, torch.Tensor]],
    generalization_batches: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
    device: torch.device,
) -> dict[str, object]:
    model = ExactRegularPD(group, input_elements).to(device)
    probe = validation_batches[0][0][:8].to(device)
    return {
        "family": "exact_regular_action_pd_ceiling",
        "oracle": True,
        "parameters": 0,
        "complex_state_size": group.order,
        "real_state_scalars": 2 * group.order,
        "validation": evaluate(model, validation_batches, device),
        "length_generalization": {
            str(length): evaluate(model, batches, device)
            for length, batches in generalization_batches.items()
        },
        "transition_diagnostics": {
            "hard_column_one_hot_error": 0.0,
            "unique_targets_per_token": [group.order] * len(input_elements),
            "is_bijection_per_token": [True] * len(input_elements),
            "collision_count_per_token": [0] * len(input_elements),
        },
        "streaming_equivalence": streaming_equivalence(model, probe),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--validation-batches", type=int, default=2)
    parser.add_argument("--validation-batch-size", type=int, default=512)
    parser.add_argument("--state-size", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--warmup-fraction", type=float, default=0.0)
    parser.add_argument("--final-learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-lengths", nargs="*", type=int)
    parser.add_argument("--group", choices=tuple(GROUPS), default="a5")
    parser.add_argument("--input-elements", nargs="*", default=[])
    parser.add_argument("--held-out-pairs", nargs="*", default=[])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--projected-permutation", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.use_deterministic_algorithms(True)
    if min(args.steps, args.batch_size, args.seq_len, args.state_size) < 1:
        raise ValueError("steps, batch size, sequence length, and state size must be positive")
    if not 0.0 <= args.warmup_fraction < 1.0:
        raise ValueError("warmup fraction must lie in [0, 1)")
    if not 0.0 < args.final_learning_rate <= args.learning_rate:
        raise ValueError("final learning rate must lie in (0, learning rate]")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    group = GROUPS[args.group]
    input_elements = parse_input_elements(args.input_elements, group)
    held_out_group_pairs = parse_held_out_pairs(args.held_out_pairs, group)
    token_for_element = {element: token for token, element in enumerate(input_elements)}
    held_out_pairs = tuple(
        (token_for_element[left], token_for_element[right])
        for left, right in held_out_group_pairs
    )
    config = PDConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.seq_len,
        validation_batches=args.validation_batches,
        validation_batch_size=args.validation_batch_size,
        state_size=args.state_size,
        learning_rate=args.learning_rate,
        warmup_fraction=args.warmup_fraction,
        final_learning_rate=args.final_learning_rate,
        seed=args.seed,
    )
    train_batches = make_group_batches(
        group, config.steps, config.batch_size, config.sequence_length,
        config.seed + 1000, input_elements=input_elements, held_out_pairs=held_out_pairs,
    )
    validation_batches = make_group_batches(
        group, config.validation_batches, config.validation_batch_size,
        config.sequence_length, 91_337, input_elements=input_elements,
        held_out_pairs=held_out_pairs, require_held_out_pair=bool(held_out_pairs),
    )
    evaluation_lengths = args.eval_lengths or sorted(
        {2, 4, 8, *range(config.sequence_length, 16 * config.sequence_length + 1,
                         config.sequence_length)}
    )
    generalization_batches = {
        length: make_group_batches(
            group, config.validation_batches, config.validation_batch_size, length,
            91_337 + length, input_elements=input_elements,
            held_out_pairs=held_out_pairs, require_held_out_pair=bool(held_out_pairs),
        )
        for length in evaluation_lengths
    }
    results = [
        run_oracle(
            group, input_elements, validation_batches, generalization_batches, device
        )
    ]
    if not args.oracle_only:
        results.append(
            run_learned(
                group, input_elements, train_batches, validation_batches,
                generalization_batches, config, device,
                project_permutation=args.projected_permutation,
            )
        )
    report = {
        "experiment": "hard column-one-hot unit-diagonal PD A5 baseline",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "torch_version": torch.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "config": asdict(config),
        "group": {"key": group.key, "name": group.name, "order": group.order},
        "input_elements": list(input_elements),
        "evaluation_lengths": evaluation_lengths,
        "data_split_audit": {
            "training": pair_split_audit(train_batches, held_out_pairs),
            "validation": pair_split_audit(validation_batches, held_out_pairs),
        },
        "language_coverage_audit": {
            "training": state_and_pair_coverage_audit(
                train_batches,
                group_order=group.order,
                input_order=len(input_elements),
            ),
            "validation": state_and_pair_coverage_audit(
                validation_batches,
                group_order=group.order,
                input_order=len(input_elements),
            ),
        },
        "comparison_contract": {
            "same_split_and_dense_lengths_as_rotor": True,
            "no_affine_write": True,
            "no_residual_or_feed_forward_path": True,
            "learned_forward_transition_is_hard_column_one_hot": True,
            "learned_transition_is_bijective_by_construction": (
                args.projected_permutation
            ),
            "unit_complex_diagonal": True,
            "state_width_is_not_parameter_matched": True,
            "oracle_and_learned_results_are_separate": True,
        },
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
