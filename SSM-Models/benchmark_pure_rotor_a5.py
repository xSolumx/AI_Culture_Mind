"""Controlled A5 prefix-tracking comparison for Pure Rotor and Mamba-2.

This runner is deliberately a *mechanism-oriented empirical benchmark*, not a
language-model benchmark and not a proof about any SSM family.  It compares the
maintained Pure Rotor v2.1.0 model, its fixed-identity-transport ablation, and
Hugging Face's Mamba2ForCausalLM on exactly the same finite-group strings.

The task uses an explicit two-generator presentation of A5,
``<a,b | a^2 = b^3 = (ab)^5 = e>``.  Training never contains the ordered
symbol pair ``a -> b``; every evaluation sequence does.  The input alphabet
also contains ``b^-1`` so this restriction retains all A5 target states rather
than collapsing the language to a nearly one-dimensional grammar.

No result from this file establishes a diagonal-SSM expressivity theorem for
Mamba-2: Mamba-2 is a richer selective/convolutional model than the diagonal
SSM class addressed by that theorem.  Likewise, a length result cannot by
itself establish that a candidate learned an exact group representation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from compare_recurrences import (
    GROUPS,
    FiniteGroup,
    make_group_batches,
    pair_split_audit,
    state_and_pair_coverage_audit,
)
from pure_rotor_ssm import __version__ as PURE_ROTOR_VERSION
from pure_rotor_ssm.torch_backend import GASSMLanguageModel
from torch import nn
from torch.nn import functional as F
from transformers import Mamba2Config, Mamba2ForCausalLM

TRAINED_CANDIDATES = (
    "pure_rotor",
    "identity_rotation_ablation",
    "mamba2_transformers",
)
PRESENTATION_A = "13254"
PRESENTATION_B = "24315"


@dataclass(frozen=True)
class A5BenchmarkConfig:
    """Parameter-near screening configuration for the A5 task."""

    steps: int = 1_000
    batch_size: int = 256
    training_length: int = 16
    validation_batches: int = 8
    validation_batch_size: int = 256
    evaluation_microbatch_size: int = 16
    evaluation_lengths: tuple[int, ...] = (2, 16, 64, 128)
    enforce_training_state_coverage: bool = True
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    rotor_channels: int = 9
    rotor_layers: int = 3
    rotor_expansion: int = 2
    mamba_hidden_size: int = 32
    mamba_layers: int = 3
    mamba_heads: int = 4
    mamba_head_dim: int = 16
    mamba_state_size: int = 16
    seed: int = 0


@dataclass(frozen=True)
class A5Task:
    """Token-level task representation and the underlying A5 elements."""

    group: FiniteGroup
    input_elements: tuple[int, ...]
    input_symbols: tuple[str, ...]
    held_out_pair: tuple[int, int]
    presentation: dict[str, int | str | bool]


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def group_power(group: FiniteGroup, element: int, exponent: int) -> int:
    """Return a nonnegative group power under the harness's right convention."""

    if exponent < 0:
        raise ValueError("this presentation audit only needs nonnegative powers")
    value = 0
    for _ in range(exponent):
        value = int(group.table[value, element])
    return value


def generated_subgroup_order(group: FiniteGroup, generators: Sequence[int]) -> int:
    """Enumerate the finite right-generated subgroup without assuming A5."""

    reached = {0}
    frontier = [0]
    while frontier:
        source = frontier.pop()
        for generator in generators:
            target = int(group.table[source, generator])
            if target not in reached:
                reached.add(target)
                frontier.append(target)
    return len(reached)


def a5_presentation_task() -> A5Task:
    """Build and independently audit the requested ``(2,3,5)`` presentation."""

    group = GROUPS["a5"]
    lookup = {name: index for index, name in enumerate(group.elements)}
    try:
        a = lookup[PRESENTATION_A]
        b = lookup[PRESENTATION_B]
    except KeyError as error:
        raise RuntimeError(
            "the canonical A5 presentation generators are absent"
        ) from error
    b_inverse = group_power(group, b, 2)
    ab = int(group.table[a, b])
    presentation = {
        "a_symbol": PRESENTATION_A,
        "b_symbol": PRESENTATION_B,
        "a_index": a,
        "b_index": b,
        "b_inverse_index": b_inverse,
        "a_squared_is_identity": group_power(group, a, 2) == 0,
        "b_cubed_is_identity": group_power(group, b, 3) == 0,
        "ab_to_fifth_is_identity": group_power(group, ab, 5) == 0,
        "generated_subgroup_order": generated_subgroup_order(group, (a, b)),
    }
    if not (
        presentation["a_squared_is_identity"]
        and presentation["b_cubed_is_identity"]
        and presentation["ab_to_fifth_is_identity"]
        and presentation["generated_subgroup_order"] == group.order
    ):
        raise RuntimeError(
            "configured A5 generators do not satisfy the (2,3,5) presentation"
        )
    return A5Task(
        group=group,
        input_elements=(a, b, b_inverse),
        input_symbols=("a", "b", "b_inverse"),
        held_out_pair=(0, 1),
        presentation=presentation,
    )


def shortest_allowed_prefix_words(
    task: A5Task, maximum_length: int
) -> dict[int, tuple[int, ...]]:
    """Find one shortest nonempty training-language word for every A5 state."""

    if maximum_length < 1:
        raise ValueError("maximum length must be positive")
    queue: list[tuple[int, tuple[int, ...], int | None]] = [(0, (), None)]
    visited = {(0, None)}
    words: dict[int, tuple[int, ...]] = {}
    offset = 0
    while offset < len(queue):
        state, word, previous = queue[offset]
        offset += 1
        if word and state not in words:
            words[state] = word
        if len(word) == maximum_length:
            continue
        for token, element in enumerate(task.input_elements):
            if previous is not None and (previous, token) == task.held_out_pair:
                continue
            next_state = int(task.group.table[state, element])
            node = (next_state, token)
            if node in visited:
                continue
            visited.add(node)
            queue.append((next_state, (*word, token), token))
    missing = sorted(set(range(task.group.order)) - set(words))
    if missing:
        raise RuntimeError(
            f"the held-out-pair language cannot reach A5 states {missing} "
            f"within length {maximum_length}"
        )
    return words


def _held_out_safe_completion(
    prefix: tuple[int, ...],
    *,
    length: int,
    input_order: int,
    held_out_pair: tuple[int, int],
    generator: np.random.Generator,
) -> tuple[int, ...]:
    """Extend a valid witness to a valid fixed-length training word."""

    if len(prefix) > length:
        raise ValueError("witness exceeds requested training length")
    word = list(prefix)
    while len(word) < length:
        token = int(generator.integers(0, input_order))
        if word and (word[-1], token) == held_out_pair:
            continue
        word.append(token)
    return tuple(word)


def _prefix_targets(task: A5Task, tokens: Sequence[int]) -> tuple[int, ...]:
    state = 0
    targets = []
    for token in tokens:
        state = int(task.group.table[state, task.input_elements[token]])
        targets.append(state)
    return tuple(targets)


def inject_training_state_coverage(
    task: A5Task,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    seed: int,
) -> dict[str, int]:
    """Embed one legal shortest witness per state in a shared train schedule."""

    if not batches:
        raise ValueError("at least one training batch is required")
    sequence_length = int(batches[0][0].shape[1])
    available_rows = sum(inputs.shape[0] for inputs, _ in batches)
    if available_rows < task.group.order:
        raise ValueError(
            "enforced A5 state coverage needs at least one training row per state"
        )
    if any(inputs.shape[1] != sequence_length for inputs, _ in batches):
        raise ValueError("all training batches must have the same sequence length")
    witnesses = shortest_allowed_prefix_words(task, sequence_length)
    rows = [
        (batch_index, row)
        for batch_index, (inputs, _) in enumerate(batches)
        for row in range(inputs.shape[0])
    ]
    generator = np.random.default_rng(30_000 + seed)
    for state, (batch_index, row) in zip(sorted(witnesses), rows):
        word = _held_out_safe_completion(
            witnesses[state],
            length=sequence_length,
            input_order=len(task.input_elements),
            held_out_pair=task.held_out_pair,
            generator=generator,
        )
        inputs, targets = batches[batch_index]
        inputs[row] = torch.tensor(word, dtype=inputs.dtype)
        targets[row] = torch.tensor(_prefix_targets(task, word), dtype=targets.dtype)
    return {
        "injected_witness_count": len(witnesses),
        "maximum_witness_length": max(len(word) for word in witnesses.values()),
    }


def batch_schedule_sha256(batches: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> str:
    """Hash the exact symbolic schedule shared by every candidate in one seed."""

    digest = hashlib.sha256()
    for inputs, targets in batches:
        for tensor in (inputs, targets):
            values = tensor.detach().cpu().contiguous()
            digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
            digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def make_training_batches(
    task: A5Task, config: A5BenchmarkConfig
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Generate one immutable, held-out-pair-free schedule for every model."""

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


def make_evaluation_batches(
    task: A5Task, config: A5BenchmarkConfig, length: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Generate evaluation strings which each contain the unseen ordered pair."""

    return make_group_batches(
        task.group,
        config.validation_batches,
        config.validation_batch_size,
        length,
        seed=20_000 + 1_000 * length + config.seed,
        input_elements=task.input_elements,
        held_out_pairs=(task.held_out_pair,),
        require_held_out_pair=True,
    )


def task_split_audit(
    task: A5Task,
    training: list[tuple[torch.Tensor, torch.Tensor]],
    evaluations: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
) -> dict[str, object]:
    """Record the split instead of assuming a missing pair is a valid test."""

    train_pairs = pair_split_audit(training, (task.held_out_pair,))
    if train_pairs["pair_occurrences"] != 0:
        raise RuntimeError("the held-out pair leaked into the training schedule")
    evaluation_audits = {
        str(length): pair_split_audit(batches, (task.held_out_pair,))
        for length, batches in evaluations.items()
    }
    if any(
        audit["sequences_with_pair"] != audit["total_sequences"]
        for audit in evaluation_audits.values()
    ):
        raise RuntimeError("an evaluation sequence is missing the forced held-out pair")
    coverage = state_and_pair_coverage_audit(
        training,
        input_order=len(task.input_elements),
        group_order=task.group.order,
    )
    exact_words = shortest_allowed_prefix_words(task, training[0][0].shape[1])
    return {
        "input_symbols": list(task.input_symbols),
        "input_group_element_indices": list(task.input_elements),
        "held_out_input_pair": list(task.held_out_pair),
        "held_out_symbol_pair": [
            task.input_symbols[task.held_out_pair[0]],
            task.input_symbols[task.held_out_pair[1]],
        ],
        "training_pair_audit": train_pairs,
        "training_coverage": coverage,
        "exact_training_language_coverage": {
            "reachable_group_states": len(exact_words),
            "maximum_shortest_witness_length": max(
                len(word) for word in exact_words.values()
            ),
            "all_group_states_reachable_within_training_length": (
                len(exact_words) == task.group.order
            ),
        },
        "training_schedule_sha256": batch_schedule_sha256(training),
        "evaluation_pair_audits": evaluation_audits,
    }


def pure_rotor_model(
    task: A5Task, config: A5BenchmarkConfig, *, max_rotor_angle: float
) -> GASSMLanguageModel:
    return GASSMLanguageModel(
        vocab_size=task.group.order,
        channels=config.rotor_channels,
        num_layers=config.rotor_layers,
        expansion=config.rotor_expansion,
        dropout_rate=0.0,
        max_rotor_angle=max_rotor_angle,
    )


def mamba2_model(task: A5Task, config: A5BenchmarkConfig) -> Mamba2ForCausalLM:
    """Construct a small, public Mamba-2 architecture without pretrained weights."""

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


def build_models(task: A5Task, config: A5BenchmarkConfig) -> dict[str, nn.Module]:
    """Build parameter-near candidates and reject an accidental size mismatch."""

    models: dict[str, nn.Module] = {
        "pure_rotor": pure_rotor_model(task, config, max_rotor_angle=math.pi),
        "identity_rotation_ablation": pure_rotor_model(
            task, config, max_rotor_angle=0.0
        ),
        "mamba2_transformers": mamba2_model(task, config),
    }
    rotor_parameters = parameter_count(models["pure_rotor"])
    mamba_parameters = parameter_count(models["mamba2_transformers"])
    relative_gap = abs(rotor_parameters - mamba_parameters) / max(
        rotor_parameters, mamba_parameters
    )
    if relative_gap > 0.05:
        raise ValueError(
            "configured Pure Rotor and Mamba-2 parameter counts differ by more than 5%"
        )
    return models


def identity_ablation_metadata(model: GASSMLanguageModel) -> dict[str, int | bool]:
    disabled = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".ssm.rotor_control." in name
    )
    raw = parameter_count(model)
    return {
        "raw_parameter_count": raw,
        "disabled_rotor_controller_parameter_count": disabled,
        "effective_parameter_count_if_rotation_is_fixed_identity": raw - disabled,
        "raw_parameter_match_is_not_effective_capacity_match": True,
    }


def decoder_tying_metadata(
    pure_rotor: GASSMLanguageModel, mamba2: Mamba2ForCausalLM
) -> dict[str, bool]:
    mamba_input = mamba2.get_input_embeddings()
    mamba_output = mamba2.get_output_embeddings()
    return {
        "pure_rotor_input_output_embeddings_tied": True,
        "mamba2_input_output_embeddings_tied": (
            mamba_input.weight.data_ptr() == mamba_output.weight.data_ptr()
        ),
    }


def logits_for(
    name: str, model: nn.Module, inputs: torch.Tensor, rotor_scan_mode: str
) -> torch.Tensor:
    if name == "mamba2_transformers":
        return model(input_ids=inputs, use_cache=False).logits
    return model(inputs, scan_mode=rotor_scan_mode)


@torch.no_grad()
def evaluate(
    name: str,
    model: nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    rotor_scan_mode: str,
    microbatch_size: int,
) -> dict[str, float]:
    """Return both every-prefix and final-prefix quality for one fixed split."""

    if microbatch_size < 1:
        raise ValueError("evaluation microbatch size must be positive")
    model.eval()
    total_loss = 0.0
    total_positions = 0
    all_correct = 0
    final_correct = 0
    total_sequences = 0
    for inputs, targets in batches:
        for start in range(0, inputs.shape[0], microbatch_size):
            input_microbatch = inputs[start : start + microbatch_size].to(device)
            target_microbatch = targets[start : start + microbatch_size].to(device)
            logits = logits_for(name, model, input_microbatch, rotor_scan_mode)
            total_loss += float(
                F.cross_entropy(
                    logits.flatten(0, 1), target_microbatch.flatten(), reduction="sum"
                )
            )
            predictions = logits.argmax(dim=-1)
            all_correct += int((predictions == target_microbatch).sum())
            final_correct += int((predictions[:, -1] == target_microbatch[:, -1]).sum())
            total_positions += target_microbatch.numel()
            total_sequences += target_microbatch.shape[0]
    return {
        "prefix_nll": total_loss / total_positions,
        "all_prefix_accuracy": all_correct / total_positions,
        "final_position_accuracy": final_correct / total_sequences,
    }


def checkpoint_payload(
    name: str,
    model: nn.Module,
    task: A5Task,
    config: A5BenchmarkConfig,
    result: dict[str, object],
) -> dict[str, object]:
    return {
        "format_version": 1,
        "candidate": name,
        "pure_rotor_model_version": PURE_ROTOR_VERSION,
        "task_presentation": task.presentation,
        "benchmark_config": asdict(config),
        "metrics": result,
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }


def train_one(
    name: str,
    make_model: Callable[[], nn.Module],
    task: A5Task,
    training: list[tuple[torch.Tensor, torch.Tensor]],
    evaluations: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
    config: A5BenchmarkConfig,
    device: torch.device,
    rotor_scan_mode: str,
    checkpoint_directory: Path | None,
) -> dict[str, object]:
    """Train one candidate on the already-fixed symbolic schedule."""

    seed_everything(config.seed)
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    initial = {
        str(length): evaluate(
            name,
            model,
            batches,
            device,
            rotor_scan_mode,
            config.evaluation_microbatch_size,
        )
        for length, batches in evaluations.items()
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    loss_samples: dict[str, float] = {}
    start = time.perf_counter()
    for step, (inputs, targets) in enumerate(training, start=1):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = logits_for(name, model, inputs, rotor_scan_mode)
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
    final = {
        str(length): evaluate(
            name,
            model,
            batches,
            device,
            rotor_scan_mode,
            config.evaluation_microbatch_size,
        )
        for length, batches in evaluations.items()
    }
    result: dict[str, object] = {
        "name": name,
        "parameters": parameter_count(model),
        "initial_held_out_pair_metrics": initial,
        "final_held_out_pair_metrics": final,
        "final_train_loss": loss_samples[str(config.steps)],
        "loss_samples": loss_samples,
        "last_preclip_gradient_norm": float(gradient_norm),
        "elapsed_seconds": elapsed,
        "tokens_per_second": config.steps
        * config.batch_size
        * config.training_length
        / elapsed,
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
        raise ValueError(
            "evaluation lengths must be a nonempty comma list of integers >= 2"
        )
    if len(set(lengths)) != len(lengths):
        raise ValueError("evaluation lengths must be distinct")
    return lengths


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in value.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty comma list of distinct integers")
    return seeds


def run_benchmark(
    config: A5BenchmarkConfig,
    *,
    seeds: Sequence[int],
    device: torch.device,
    rotor_scan_mode: str,
    checkpoint_directory: Path | None = None,
) -> dict[str, object]:
    """Run a complete fixed-schedule cohort and return a provenance report."""

    task = a5_presentation_task()
    model_shapes = build_models(task, config)
    counts = {name: parameter_count(model) for name, model in model_shapes.items()}
    identity_metadata = identity_ablation_metadata(
        model_shapes["identity_rotation_ablation"]
    )
    decoder_metadata = decoder_tying_metadata(
        model_shapes["pure_rotor"], model_shapes["mamba2_transformers"]
    )
    results = []
    split_audits: dict[str, object] = {}
    for seed in seeds:
        run_config = A5BenchmarkConfig(**{**asdict(config), "seed": seed})
        training = make_training_batches(task, run_config)
        evaluations = {
            length: make_evaluation_batches(task, run_config, length)
            for length in run_config.evaluation_lengths
        }
        split_audits[str(seed)] = task_split_audit(task, training, evaluations)
        factories: dict[str, Callable[[], nn.Module]] = {
            "pure_rotor": lambda config=run_config: pure_rotor_model(
                task, config, max_rotor_angle=math.pi
            ),
            "identity_rotation_ablation": lambda config=run_config: pure_rotor_model(
                task, config, max_rotor_angle=0.0
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
                    checkpoint_directory,
                )
            )
    return {
        "experiment": "Pure Rotor v2.1 versus Mamba-2 A5 held-out-pair tracking",
        "status": (
            "completed empirical screen; no exact-representation, diagonal-SSM, "
            "or language-model superiority claim"
        ),
        "pure_rotor_model_version": PURE_ROTOR_VERSION,
        "device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else str(device),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "config": asdict(config),
        "seeds": list(seeds),
        "rotor_scan_mode": rotor_scan_mode,
        "task": {
            "group": {
                "key": task.group.key,
                "name": task.group.name,
                "order": task.group.order,
            },
            "presentation": task.presentation,
            "target": "every ordered A5 prefix product",
            "split": split_audits,
        },
        "integrity": {
            "same_precomputed_training_batches_per_seed": True,
            "same_precomputed_evaluation_batches_per_seed": True,
            "candidate_initialized_after_seed": True,
            "pure_rotor_vs_identity_initial_function_identical": True,
            "cross_architecture_initial_function_identical": False,
            "mamba2_backend": "huggingface_transformers",
            "mamba_ssm_extension_importable": importlib.util.find_spec("mamba_ssm")
            is not None,
            "mamba2_fused_kernel_claimed": False,
            "parameter_matching": {
                "parameter_counts": counts,
                "relative_pure_rotor_mamba2_gap": abs(
                    counts["pure_rotor"] - counts["mamba2_transformers"]
                )
                / max(counts["pure_rotor"], counts["mamba2_transformers"]),
                "state_size_matched": False,
                "state_matching_note": (
                    "Pure Rotor has fixed Cl(3,0) layer state; Transformers Mamba-2 "
                    "has architecture-specific SSM and convolution caches. This is a "
                    "parameter-near comparison, not a matched-state comparison."
                ),
            },
            "identity_rotation_ablation": identity_metadata,
            "decoder_tying": decoder_metadata,
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--training-length", type=int, default=16)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=256)
    parser.add_argument("--evaluation-microbatch-size", type=int, default=16)
    parser.add_argument(
        "--allow-incomplete-training-coverage",
        action="store_true",
        help="disable the deterministic one-witness-per-A5-state schedule injection",
    )
    parser.add_argument("--evaluation-lengths", default="2,16,64,128")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--rotor-scan-mode",
        choices=("parallel", "schur_parallel", "recurrent"),
        default="parallel",
    )
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    positive = (
        args.steps,
        args.batch_size,
        args.training_length,
        args.validation_batches,
        args.validation_batch_size,
        args.evaluation_microbatch_size,
    )
    if min(positive) < 1:
        raise ValueError("steps, batch sizes, and training length must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config = A5BenchmarkConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        training_length=args.training_length,
        validation_batches=args.validation_batches,
        validation_batch_size=args.validation_batch_size,
        evaluation_microbatch_size=args.evaluation_microbatch_size,
        evaluation_lengths=parse_lengths(args.evaluation_lengths),
        enforce_training_state_coverage=not args.allow_incomplete_training_coverage,
    )
    report = run_benchmark(
        config,
        seeds=parse_seeds(args.seeds),
        device=torch.device(args.device),
        rotor_scan_mode=args.rotor_scan_mode,
        checkpoint_directory=args.checkpoint_directory,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
