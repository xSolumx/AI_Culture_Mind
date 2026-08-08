"""Run the preregistered pure v2.1 transport-ablation ladder on CUDA.

The runner is intentionally resumable.  Every trained family/seed/view is
written as an atomic JSON shard, while calibration, systems measurements, and
the assembled report remain separate artifacts.  The frozen protocol is in
``experiments/PURE_V2_1_TRANSPORT_ABLATION_PREREGISTRATION.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pure_rotor_ssm import __version__ as MODEL_VERSION
from torch import nn
from torch.nn import functional as F
from transport_ablation_v2 import (
    FAMILY_NAMES,
    MatchedTransportClassifier,
    MatchedTransportLanguageModel,
)

ROOT = Path(__file__).resolve().parent
PREREGISTRATION = (
    ROOT / "experiments" / "PURE_V2_1_TRANSPORT_ABLATION_PREREGISTRATION.md"
)
DEFAULT_RUN_DIR = ROOT / "experiments" / "pure_v2.1.0_transport_runs"
DEFAULT_REPORT = ROOT / "experiments" / "pure_v2.1.0_transport_ablation.json"
ACTION_KEYS = (
    ".ssm.action_control.",
    ".ssm.rotor_control.",
    ".ssm.rotor_source.",
    ".ssm.fixed_bivector",
)


@dataclass(frozen=True)
class PredictionConfig:
    steps: int = 300
    batch_size: int = 64
    sequence_length: int = 128
    channels: int = 8
    layers: int = 2
    expansion: int = 2
    learning_rate: float = 3e-3
    development_batches: int = 20
    confirmation_batches: int = 40


@dataclass(frozen=True)
class MemoryConfig:
    steps: int = 800
    batch_size: int = 128
    channels: int = 4
    layers: int = 2
    expansion: int = 2
    learning_rate: float = 3e-3
    evaluation_batches: int = 32


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def protocol_sha256() -> str:
    return sha256_bytes(PREREGISTRATION.read_bytes())


def stable_seed(*parts: object) -> int:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "little")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def common_state(reference: nn.Module, target: nn.Module) -> int:
    """Copy same-shaped non-transport tensors and return the copied scalar count."""

    source = reference.state_dict()
    destination = target.state_dict()
    copied = 0
    for key, value in source.items():
        if any(marker in key for marker in ACTION_KEYS):
            continue
        if key in destination and destination[key].shape == value.shape:
            destination[key] = value.detach().clone()
            copied += value.numel()
    target.load_state_dict(destination, strict=True)
    return copied


def make_language_model(
    family: str, channels: int, seed: int, config: PredictionConfig
) -> tuple[MatchedTransportLanguageModel, int]:
    set_seed(seed)
    reference = MatchedTransportLanguageModel(
        256, channels, config.layers, "identity", expansion=config.expansion
    )
    if family == "identity":
        return reference, parameter_count(reference)
    set_seed(stable_seed("action", family, channels, seed))
    model = MatchedTransportLanguageModel(
        256, channels, config.layers, family, expansion=config.expansion
    )
    return model, common_state(reference, model)


def make_classifier(
    task: str, family: str, seed: int, config: MemoryConfig
) -> tuple[MatchedTransportClassifier, int]:
    vocab_size, classes = (273, 16) if task == "associative_recall" else (4, 8)
    set_seed(seed)
    reference = MatchedTransportClassifier(
        vocab_size,
        classes,
        config.channels,
        config.layers,
        "identity",
        expansion=config.expansion,
    )
    if family == "identity":
        return reference, parameter_count(reference)
    set_seed(stable_seed("action", task, family, seed))
    model = MatchedTransportClassifier(
        vocab_size,
        classes,
        config.channels,
        config.layers,
        family,
        expansion=config.expansion,
    )
    return model, common_state(reference, model)


def wiki_bytes(split: str) -> np.ndarray:
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)


def wiki_provenance(train: np.ndarray, validation: np.ndarray) -> dict[str, Any]:
    return {
        "dataset": "wikitext/wikitext-2-raw-v1",
        "configuration": "wikitext-2-raw-v1",
        "encoding": "UTF-8 bytes",
        "train_bytes": int(train.size),
        "validation_bytes": int(validation.size),
        "train_sha256": sha256_bytes(train.tobytes()),
        "validation_sha256": sha256_bytes(validation.tobytes()),
    }


def fixed_language_batches(
    tokens: np.ndarray, config: PredictionConfig
) -> tuple[
    list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]]
]:
    count = config.development_batches + config.confirmation_batches
    width = config.sequence_length + 1
    required = count * config.batch_size * width
    if tokens.size < required:
        raise ValueError("validation split is too short for the frozen batches")
    data = tokens[:required].reshape(count, config.batch_size, width)
    batches = [
        (torch.from_numpy(batch[:, :-1]), torch.from_numpy(batch[:, 1:]))
        for batch in data
    ]
    return batches[: config.development_batches], batches[config.development_batches :]


def random_language_batch(
    tokens: torch.Tensor, config: PredictionConfig, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(
        0,
        tokens.numel() - config.sequence_length - 1,
        (config.batch_size,),
        generator=generator,
    )
    offsets = torch.arange(config.sequence_length + 1)
    sequences = tokens[starts[:, None] + offsets]
    return sequences[:, :-1], sequences[:, 1:]


@torch.no_grad()
def evaluate_language(
    model: MatchedTransportLanguageModel,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    force_identity: bool = False,
    permutation: torch.Tensor | None = None,
    scan_mode: str = "parallel",
) -> float:
    model.eval()
    total = 0.0
    for inputs, targets in batches:
        logits = model(
            inputs.to(device),
            scan_mode=scan_mode,
            force_identity=force_identity,
            action_permutation=permutation,
        )
        total += float(
            F.cross_entropy(logits.flatten(0, 1), targets.to(device).flatten())
        )
    return total / len(batches)


@torch.no_grad()
def language_parity(
    model: MatchedTransportLanguageModel,
    inputs: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    parallel = model(inputs, scan_mode="parallel")
    recurrent = model(inputs, scan_mode="recurrent")
    states = None
    pieces = []
    split = max(1, inputs.shape[1] // 3)
    for chunk in inputs.split(split, dim=1):
        logits, states = model(
            chunk,
            states,
            scan_mode="parallel",
            return_recurrent_states=True,
        )
        pieces.append(logits)
    chunked = torch.cat(pieces, dim=1)
    return {
        "parallel_vs_recurrent_max_abs": float((parallel - recurrent).abs().max()),
        "parallel_vs_chunked_max_abs": float((parallel - chunked).abs().max()),
    }


def layer_diagnostics(
    model: MatchedTransportLanguageModel, inputs: torch.Tensor
) -> list[dict[str, float | str]]:
    model.eval()
    outputs = model.token_embeddings[inputs]
    diagnostics = []
    with torch.no_grad():
        for block in model.blocks:
            normalized = block.norm1(outputs)
            diagnostics.append(block.ssm.diagnostics(normalized))
            outputs, _ = block(outputs)
    return diagnostics


def shard_path(run_dir: Path, category: str, *parts: object) -> Path:
    name = "_".join(str(part) for part in parts)
    return run_dir / category / f"{name}.json"


def valid_shard(path: Path, expected_protocol: str) -> bool:
    if not path.exists():
        return False
    try:
        return (
            json.loads(path.read_text(encoding="utf-8"))["protocol_sha256"]
            == expected_protocol
        )
    except (OSError, KeyError, json.JSONDecodeError):
        return False


def run_prediction(
    family: str,
    view: str,
    channels: int,
    seed: int,
    config: PredictionConfig,
    train_tokens: torch.Tensor,
    development: list[tuple[torch.Tensor, torch.Tensor]],
    confirmation: list[tuple[torch.Tensor, torch.Tensor]],
    provenance: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    protocol = protocol_sha256()
    path = shard_path(run_dir, "prediction", view, family, f"seed{seed}")
    if resume and valid_shard(path, protocol):
        print(f"resume prediction {view} {family} seed={seed}", flush=True)
        return json.loads(path.read_text(encoding="utf-8"))

    model, common_scalars = make_language_model(family, channels, seed, config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(seed)
    initial_development = evaluate_language(model, development, device)
    loss_samples: dict[str, float] = {}
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    final_loss = math.nan
    for step in range(1, config.steps + 1):
        inputs, targets = random_language_batch(train_tokens, config, generator)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
        if step == 1 or step % 50 == 0 or step == config.steps:
            loss_samples[str(step)] = final_loss
            print(
                f"prediction {view}/{family} seed={seed} step={step}/{config.steps} "
                f"loss={final_loss:.4f}",
                flush=True,
            )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / 2**20
    development_loss = evaluate_language(model, development, device)
    confirmation_loss = evaluate_language(model, confirmation, device)
    diagnostic_inputs = confirmation[0][0][: min(8, config.batch_size)].to(device)
    interventions = None
    if family == "rotor":
        permutation = torch.randperm(
            config.sequence_length,
            generator=torch.Generator().manual_seed(stable_seed("shuffle", seed)),
        ).to(device)
        interventions = {
            "identity_clamp_confirmation_loss": evaluate_language(
                model, confirmation, device, force_identity=True
            ),
            "time_shuffled_confirmation_loss": evaluate_language(
                model, confirmation, device, permutation=permutation
            ),
            "permutation_sha256": sha256_bytes(permutation.cpu().numpy().tobytes()),
        }
    result = {
        "kind": "prediction",
        "model_version": MODEL_VERSION,
        "protocol_sha256": protocol,
        "view": view,
        "family": family,
        "seed": seed,
        "channels": channels,
        "recurrent_state_scalars": config.layers * channels * 8,
        "effective_parameters": parameter_count(model),
        "common_initialized_scalars": common_scalars,
        "config": asdict(config),
        "data": provenance,
        "initial_development_loss": initial_development,
        "development_loss": development_loss,
        "confirmation_loss": confirmation_loss,
        "confirmation_bits_per_byte": confirmation_loss / math.log(2),
        "final_train_loss": final_loss,
        "loss_samples": loss_samples,
        "elapsed_seconds": elapsed,
        "training_tokens_per_second": config.steps
        * config.batch_size
        * config.sequence_length
        / elapsed,
        "peak_cuda_memory_mib": peak,
        "transition_diagnostics": layer_diagnostics(model, diagnostic_inputs),
        "numerical_parity": language_parity(model, diagnostic_inputs),
        "interventions": interventions,
    }
    atomic_json(path, result)
    del optimizer, model
    torch.cuda.empty_cache()
    return result


def associative_recall_batch(
    batch_size: int, length: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    if length < 4 or length > 512 or length % 2:
        raise ValueError("associative-recall length must be even and in [4,512]")
    pairs = (length - 2) // 2
    scores = torch.rand(batch_size, 256, generator=generator)
    keys = scores.topk(pairs, dim=1).indices
    values = torch.randint(0, 16, (batch_size, pairs), generator=generator)
    query_position = torch.randint(0, pairs, (batch_size,), generator=generator)
    row = torch.arange(batch_size)
    query_key = keys[row, query_position]
    target = values[row, query_position]
    sequence = torch.empty(batch_size, length, dtype=torch.long)
    sequence[:, 0 : 2 * pairs : 2] = keys
    sequence[:, 1 : 2 * pairs : 2] = 256 + values
    sequence[:, -2] = 272
    sequence[:, -1] = query_key
    return sequence, target


def _q8_table() -> torch.Tensor:
    # Elements are +1,-1,+i,-i,+j,-j,+k,-k.
    elements = ((1, 0), (-1, 0), (1, 1), (-1, 1), (1, 2), (-1, 2), (1, 3), (-1, 3))
    lookup = {element: index for index, element in enumerate(elements)}
    basis = {
        (0, 0): (1, 0),
        (0, 1): (1, 1),
        (0, 2): (1, 2),
        (0, 3): (1, 3),
        (1, 0): (1, 1),
        (2, 0): (1, 2),
        (3, 0): (1, 3),
        (1, 1): (-1, 0),
        (2, 2): (-1, 0),
        (3, 3): (-1, 0),
        (1, 2): (1, 3),
        (2, 3): (1, 1),
        (3, 1): (1, 2),
        (2, 1): (-1, 3),
        (3, 2): (-1, 1),
        (1, 3): (-1, 2),
    }
    table = torch.empty(8, 8, dtype=torch.long)
    for left, (left_sign, left_basis) in enumerate(elements):
        for right, (right_sign, right_basis) in enumerate(elements):
            sign, product_basis = basis[left_basis, right_basis]
            table[left, right] = lookup[left_sign * right_sign * sign, product_basis]
    return table


Q8_TABLE = _q8_table()
Q8_TOKENS = torch.tensor((2, 4, 3, 5), dtype=torch.long)


def q8_product_batch(
    batch_size: int, length: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    if length < 1:
        raise ValueError("Q8 length must be positive")
    token_ids = torch.randint(0, 4, (batch_size, length), generator=generator)
    elements = Q8_TOKENS[token_ids]
    product = torch.zeros(batch_size, dtype=torch.long)
    for position in range(length):
        product = Q8_TABLE[product, elements[:, position]]
    return token_ids, product


MEMORY_LENGTHS = {
    "associative_recall": (64, 128, 256, 512),
    "q8_ordered_product": (32, 64, 128, 256),
}


def memory_batch(
    task: str, batch_size: int, length: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    if task == "associative_recall":
        return associative_recall_batch(batch_size, length, generator)
    if task == "q8_ordered_product":
        return q8_product_batch(batch_size, length, generator)
    raise ValueError(f"unknown task: {task}")


@torch.no_grad()
def evaluate_memory(
    model: MatchedTransportClassifier,
    task: str,
    length: int,
    seed: int,
    config: MemoryConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    generator = torch.Generator().manual_seed(
        stable_seed("memory-eval", task, length, seed)
    )
    correct = 0
    total = 0
    loss = 0.0
    for _ in range(config.evaluation_batches):
        inputs, targets = memory_batch(task, config.batch_size, length, generator)
        logits = model(inputs.to(device))
        targets = targets.to(device)
        loss += float(F.cross_entropy(logits, targets))
        correct += int((logits.argmax(dim=-1) == targets).sum())
        total += targets.numel()
    return {
        "accuracy": correct / total,
        "cross_entropy": loss / config.evaluation_batches,
        "examples": total,
    }


def run_memory(
    task: str,
    family: str,
    seed: int,
    config: MemoryConfig,
    device: torch.device,
    run_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    protocol = protocol_sha256()
    path = shard_path(run_dir, "memory", task, family, f"seed{seed}")
    if resume and valid_shard(path, protocol):
        print(f"resume memory {task} {family} seed={seed}", flush=True)
        return json.loads(path.read_text(encoding="utf-8"))

    model, common_scalars = make_classifier(task, family, seed, config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    loss_samples: dict[str, float] = {}
    final_loss = math.nan
    for step in range(1, config.steps + 1):
        if task == "associative_recall":
            length = 4 + 2 * int(torch.randint(0, 31, (), generator=generator))
        else:
            length = 4 + int(torch.randint(0, 29, (), generator=generator))
        inputs, targets = memory_batch(task, config.batch_size, length, generator)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
        if step == 1 or step % 100 == 0 or step == config.steps:
            loss_samples[str(step)] = final_loss
            print(
                f"memory {task}/{family} seed={seed} step={step}/{config.steps} "
                f"loss={final_loss:.4f}",
                flush=True,
            )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    evaluations = {
        str(length): evaluate_memory(model, task, length, seed, config, device)
        for length in MEMORY_LENGTHS[task]
    }
    result = {
        "kind": "memory",
        "model_version": MODEL_VERSION,
        "protocol_sha256": protocol,
        "task": task,
        "family": family,
        "seed": seed,
        "config": asdict(config),
        "recurrent_state_scalars": config.layers * config.channels * 8,
        "effective_parameters": parameter_count(model),
        "common_initialized_scalars": common_scalars,
        "final_train_loss": final_loss,
        "loss_samples": loss_samples,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "evaluations": evaluations,
    }
    atomic_json(path, result)
    del optimizer, model
    torch.cuda.empty_cache()
    return result


def timed_cuda(
    callable_: Callable[[], Any], warmup: int, iterations: int
) -> list[float]:
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        callable_()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return times


def timing_summary(times: list[float], tokens: int) -> dict[str, float]:
    median = statistics.median(times)
    ordered = sorted(times)
    return {
        "median_ms": median,
        "p95_ms": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
        "tokens_per_second": 1000 * tokens / median,
    }


def benchmark_model(
    family: str,
    channels: int,
    length: int,
    batch_size: int,
    device: torch.device,
    config: PredictionConfig,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    model, _ = make_language_model(family, channels, 200, config)
    model.to(device)
    tokens = torch.randint(0, 256, (batch_size, length), device=device)
    model.eval()

    def inference() -> None:
        with torch.no_grad():
            model(tokens)

    torch.cuda.reset_peak_memory_stats(device)
    inference_times = timed_cuda(inference, warmup, iterations)
    inference_peak = torch.cuda.max_memory_allocated(device) / 2**20
    model.train()

    def forward_backward() -> None:
        model.zero_grad(set_to_none=True)
        logits = model(tokens)
        logits.square().mean().backward()

    torch.cuda.reset_peak_memory_stats(device)
    training_times = timed_cuda(forward_backward, warmup, iterations)
    training_peak = torch.cuda.max_memory_allocated(device) / 2**20
    result = {
        "family": family,
        "channels": channels,
        "parameters": parameter_count(model),
        "batch_size": batch_size,
        "sequence_length": length,
        "inference": timing_summary(inference_times, batch_size * length)
        | {"peak_memory_mib": inference_peak},
        "forward_backward": timing_summary(training_times, batch_size * length)
        | {"peak_memory_mib": training_peak},
    }
    torch.cuda.empty_cache()
    return result


def nearest_parameter_widths(
    config: PredictionConfig, maximum: int = 24
) -> dict[str, int]:
    target = parameter_count(
        make_language_model("rotor", config.channels, 0, config)[0]
    )
    result = {}
    for family in FAMILY_NAMES:
        choices = []
        for channels in range(1, maximum + 1):
            count = parameter_count(make_language_model(family, channels, 0, config)[0])
            choices.append((abs(count - target), channels, count))
        result[family] = min(choices)[1]
    return result


def run_calibration(
    config: PredictionConfig,
    device: torch.device,
    run_dir: Path,
    resume: bool,
    *,
    maximum_channels: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    path = run_dir / "calibration.json"
    protocol = protocol_sha256()
    if resume and valid_shard(path, protocol):
        print("resume calibration", flush=True)
        return json.loads(path.read_text(encoding="utf-8"))
    parameter_widths = nearest_parameter_widths(config, maximum_channels)
    rotor = benchmark_model(
        "rotor",
        config.channels,
        config.sequence_length,
        config.batch_size,
        device,
        config,
        warmup=warmup,
        iterations=iterations,
    )
    target_ms = rotor["forward_backward"]["median_ms"]
    candidates: dict[str, list[dict[str, Any]]] = {}
    compute_widths = {}
    truncated_after: dict[str, int | None] = {}
    for family in FAMILY_NAMES:
        if family == "rotor":
            candidates[family] = [rotor]
            compute_widths[family] = config.channels
            truncated_after[family] = config.channels
            continue
        family_candidates = []
        at_or_above = 0
        stopped_at = None
        for channels in range(1, maximum_channels + 1):
            try:
                measurement = benchmark_model(
                    family,
                    channels,
                    config.sequence_length,
                    config.batch_size,
                    device,
                    config,
                    warmup=warmup,
                    iterations=iterations,
                )
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                stopped_at = channels
                print(f"calibrate {family} C={channels} OOM", flush=True)
                break
            family_candidates.append(measurement)
            print(
                f"calibrate {family} C={channels} "
                f"{measurement['forward_backward']['median_ms']:.3f}ms",
                flush=True,
            )
            if measurement["forward_backward"]["median_ms"] >= target_ms:
                at_or_above += 1
                if at_or_above >= 2 and channels >= config.channels:
                    stopped_at = channels
                    break
        candidates[family] = family_candidates
        truncated_after[family] = stopped_at
        compute_widths[family] = min(
            family_candidates,
            key=lambda item: (
                abs(item["forward_backward"]["median_ms"] - target_ms),
                item["channels"],
            ),
        )["channels"]
    result = {
        "kind": "calibration",
        "model_version": MODEL_VERSION,
        "protocol_sha256": protocol,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "config": asdict(config),
        "warmup": warmup,
        "iterations": iterations,
        "rotor_target": rotor,
        "parameter_matched_widths": parameter_widths,
        "cuda_matched_widths": compute_widths,
        "cuda_candidates": candidates,
        "cuda_search_stopped_at": truncated_after,
    }
    atomic_json(path, result)
    return result


def run_system_benchmarks(
    calibration: dict[str, Any],
    config: PredictionConfig,
    device: torch.device,
    run_dir: Path,
    resume: bool,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    path = run_dir / "systems_benchmark.json"
    protocol = protocol_sha256()
    if resume and valid_shard(path, protocol):
        print("resume systems benchmark", flush=True)
        return json.loads(path.read_text(encoding="utf-8"))
    results = []
    for view, widths in (
        ("state_matched", {family: config.channels for family in FAMILY_NAMES}),
        ("parameter_matched", calibration["parameter_matched_widths"]),
        ("cuda_matched", calibration["cuda_matched_widths"]),
    ):
        for family in FAMILY_NAMES:
            for length in (64, 128, 256, 512):
                measurement = benchmark_model(
                    family,
                    int(widths[family]),
                    length,
                    8,
                    device,
                    config,
                    warmup=warmup,
                    iterations=iterations,
                )
                measurement["view"] = view
                results.append(measurement)
                print(
                    f"benchmark {view}/{family} L={length} "
                    f"train={measurement['forward_backward']['median_ms']:.3f}ms",
                    flush=True,
                )
    result = {
        "kind": "systems_benchmark",
        "model_version": MODEL_VERSION,
        "protocol_sha256": protocol,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "warmup": warmup,
        "iterations": iterations,
        "results": results,
    }
    atomic_json(path, result)
    return result


def mean_ci95(values: list[float]) -> dict[str, float | int]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean, "n": len(values)}
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    critical = 2.776 if len(values) == 5 else 1.96
    return {
        "mean": mean,
        "ci95_low": mean - critical * standard_error,
        "ci95_high": mean + critical * standard_error,
        "n": len(values),
    }


def read_shards(directory: Path) -> list[dict[str, Any]]:
    if not directory.exists():
        return []
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.rglob("*.json"))
    ]


def assemble_report(run_dir: Path, output: Path) -> dict[str, Any]:
    protocol = protocol_sha256()
    predictions = [
        item
        for item in read_shards(run_dir / "prediction")
        if item.get("protocol_sha256") == protocol
    ]
    memory = [
        item
        for item in read_shards(run_dir / "memory")
        if item.get("protocol_sha256") == protocol
    ]
    calibration = (
        json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
        if (run_dir / "calibration.json").exists()
        else None
    )
    systems = (
        json.loads((run_dir / "systems_benchmark.json").read_text(encoding="utf-8"))
        if (run_dir / "systems_benchmark.json").exists()
        else None
    )
    prediction_summary: dict[str, Any] = {}
    for view in ("state_matched", "parameter_matched", "cuda_matched"):
        view_items = [item for item in predictions if item["view"] == view]
        identity = {
            item["seed"]: item for item in view_items if item["family"] == "identity"
        }
        families = {}
        for family in FAMILY_NAMES:
            items = sorted(
                (item for item in view_items if item["family"] == family),
                key=lambda item: item["seed"],
            )
            losses = [item["confirmation_loss"] for item in items]
            paired = [
                item["confirmation_loss"] - identity[item["seed"]]["confirmation_loss"]
                for item in items
                if item["seed"] in identity
            ]
            families[family] = {
                "confirmation_loss": mean_ci95(losses) if losses else None,
                "paired_loss_minus_identity": mean_ci95(paired) if paired else None,
                "wins_vs_identity": sum(delta < 0 for delta in paired),
                "seeds": [item["seed"] for item in items],
            }
        prediction_summary[view] = families
    memory_summary: dict[str, Any] = {}
    for task, lengths in MEMORY_LENGTHS.items():
        task_items = [item for item in memory if item["task"] == task]
        identity = {
            item["seed"]: item for item in task_items if item["family"] == "identity"
        }
        families = {}
        for family in FAMILY_NAMES:
            items = sorted(
                (item for item in task_items if item["family"] == family),
                key=lambda item: item["seed"],
            )
            by_length = {}
            for length in lengths:
                accuracies = [
                    item["evaluations"][str(length)]["accuracy"] for item in items
                ]
                paired = [
                    item["evaluations"][str(length)]["accuracy"]
                    - identity[item["seed"]]["evaluations"][str(length)]["accuracy"]
                    for item in items
                    if item["seed"] in identity
                ]
                by_length[str(length)] = {
                    "accuracy": mean_ci95(accuracies) if accuracies else None,
                    "paired_accuracy_minus_identity": mean_ci95(paired)
                    if paired
                    else None,
                }
            families[family] = by_length
        memory_summary[task] = families
    result = {
        "model_version": MODEL_VERSION,
        "protocol_sha256": protocol,
        "preregistration": str(PREREGISTRATION),
        "prediction_run_count": len(predictions),
        "memory_run_count": len(memory),
        "calibration": calibration,
        "systems_benchmark": systems,
        "prediction_runs": predictions,
        "memory_runs": memory,
        "summary": {"prediction": prediction_summary, "memory": memory_summary},
    }
    atomic_json(output, result)
    return result


def parse_csv(value: str, allowed: tuple[str, ...] | None = None) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts or (
        allowed is not None and any(part not in allowed for part in parts)
    ):
        raise argparse.ArgumentTypeError(f"invalid comma-separated selection: {value}")
    return parts


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from error
    if not seeds or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be nonnegative")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("calibrate", "prediction", "memory", "benchmark", "all", "assemble"),
        default="all",
    )
    parser.add_argument("--families", default=",".join(FAMILY_NAMES))
    parser.add_argument("--seeds", type=parse_seeds, default=(0, 1, 2, 3, 4))
    parser.add_argument(
        "--views", default="state_matched,parameter_matched,cuda_matched"
    )
    parser.add_argument("--tasks", default="associative_recall,q8_ordered_product")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--maximum-channels", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--prediction-steps", type=int, default=300)
    parser.add_argument("--memory-steps", type=int, default=800)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen experiment")
    if args.maximum_channels < 1 or args.warmup < 1 or args.iterations < 1:
        raise ValueError("channel limit, warmup, and iterations must be positive")
    families = parse_csv(args.families, FAMILY_NAMES)
    views = parse_csv(
        args.views, ("state_matched", "parameter_matched", "cuda_matched")
    )
    tasks = parse_csv(args.tasks, tuple(MEMORY_LENGTHS))
    device = torch.device("cuda")
    prediction_config = PredictionConfig(steps=args.prediction_steps)
    memory_config = MemoryConfig(steps=args.memory_steps)
    resume = not args.no_resume

    calibration = None
    if args.phase in ("calibrate", "all", "prediction", "benchmark"):
        calibration = run_calibration(
            prediction_config,
            device,
            args.run_dir,
            resume,
            maximum_channels=args.maximum_channels,
            warmup=args.warmup,
            iterations=args.iterations,
        )
    if args.phase in ("prediction", "all"):
        train_array = wiki_bytes("train")
        validation_array = wiki_bytes("validation")
        provenance = wiki_provenance(train_array, validation_array)
        train_tokens = torch.from_numpy(train_array)
        development, confirmation = fixed_language_batches(
            validation_array, prediction_config
        )
        assert calibration is not None
        widths_by_view = {
            "state_matched": {
                family: prediction_config.channels for family in FAMILY_NAMES
            },
            "parameter_matched": calibration["parameter_matched_widths"],
            "cuda_matched": calibration["cuda_matched_widths"],
        }
        for view in views:
            for seed in args.seeds:
                for family in families:
                    run_prediction(
                        family,
                        view,
                        int(widths_by_view[view][family]),
                        seed,
                        prediction_config,
                        train_tokens,
                        development,
                        confirmation,
                        provenance,
                        device,
                        args.run_dir,
                        resume,
                    )
    if args.phase in ("memory", "all"):
        for task in tasks:
            for seed in args.seeds:
                for family in families:
                    run_memory(
                        task, family, seed, memory_config, device, args.run_dir, resume
                    )
    if args.phase in ("benchmark", "all"):
        assert calibration is not None
        run_system_benchmarks(
            calibration,
            prediction_config,
            device,
            args.run_dir,
            resume,
            warmup=args.warmup,
            iterations=args.iterations,
        )
    if args.phase in ("all", "assemble"):
        assembled = assemble_report(args.run_dir, args.output)
        print(
            f"assembled prediction_runs={assembled['prediction_run_count']} "
            f"memory_runs={assembled['memory_run_count']} output={args.output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
