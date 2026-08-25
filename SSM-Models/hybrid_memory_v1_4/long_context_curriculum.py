"""Run the frozen G13 exact-target long-context curriculum comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

if __package__:
    from .long_context_recall import (
        RECALL_EXAMPLES,
        _ascii_filler,
        _recall_pair,
    )
    from .model import HybridMemoryConfig, HybridMemoryLM
    from .natural_text_frontier import (
        DATA_SEED,
        GRADIENT_CLIP,
        LEARNING_RATE,
        WEIGHT_DECAY,
        _offset,
        _optimizer_report,
        _parameter_matched_config,
        _sha256,
        _snapshot_text,
    )
    from .optimizers import build_optimizer
    from .tokenization import (
        ByteLevelBPETokenizer,
        EncodedText,
        LosslessTextTokenizer,
        tokenizer_fingerprint,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4.long_context_recall import (  # type: ignore[no-redef]
        RECALL_EXAMPLES,
        _ascii_filler,
        _recall_pair,
    )
    from hybrid_memory_v1_4.model import (  # type: ignore[no-redef]
        HybridMemoryConfig,
        HybridMemoryLM,
    )
    from hybrid_memory_v1_4.natural_text_frontier import (  # type: ignore[no-redef]
        DATA_SEED,
        GRADIENT_CLIP,
        LEARNING_RATE,
        WEIGHT_DECAY,
        _offset,
        _optimizer_report,
        _parameter_matched_config,
        _sha256,
        _snapshot_text,
    )
    from hybrid_memory_v1_4.optimizers import (  # type: ignore[no-redef]
        build_optimizer,
    )
    from hybrid_memory_v1_4.tokenization import (  # type: ignore[no-redef]
        ByteLevelBPETokenizer,
        EncodedText,
        LosslessTextTokenizer,
        tokenizer_fingerprint,
    )


PREREGISTRATION = Path(__file__).with_name("G13_PREREGISTRATION.md")
MODEL_SEEDS = (2011, 2017, 2027)
CONTEXT_LENGTHS = (256, 512, 1024, 2048, 4096)
CURRICULUM_LENGTHS = CONTEXT_LENGTHS
FIXED_LENGTHS = (256,) * len(CURRICULUM_LENGTHS)
PHASE_UPDATES = 200
TOTAL_UPDATES = PHASE_UPDATES * len(CURRICULUM_LENGTHS)
TARGETS_PER_UPDATE = 4096
FORWARD_CHUNK_SIZE = 1024
MONITOR_CONTEXT_LENGTHS = (256, 4096)
EVAL_MACRO_BATCHES = 8
RECALL_RAW_BYTE_DISTANCES = (128, 256, 512, 1024, 2048, 4096, 8192)
RECALL_MODES = ("full", "gated_delta_off", "attention_off")
EXPECTED_MODEL_DIM = 48
EXPECTED_EXPANSION = 5
EXPECTED_PARAMETER_COUNT = 124_534


def _git() -> tuple[str, list[str]]:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return commit, status


def _load_bpe(tokenizer_audit_path: Path) -> ByteLevelBPETokenizer:
    audit = json.loads(tokenizer_audit_path.read_text(encoding="utf-8"))
    selection = audit["selection_rule"]
    tokenizer_path = Path(selection["selected_path"])
    tokenizer = ByteLevelBPETokenizer.from_serialized(
        tokenizer_path.read_text(encoding="utf-8")
    )
    if tokenizer.vocab_size != 512:
        raise ValueError("G13 requires the frozen 512-token G12A tokenizer")
    if tokenizer_fingerprint(tokenizer) != selection["selected_sha256"]:
        raise ValueError("G13 tokenizer fingerprint mismatch")
    return tokenizer


def _frozen_config(vocab_size: int) -> HybridMemoryConfig:
    config, report = _parameter_matched_config(vocab_size)
    if (
        config.model_dim != EXPECTED_MODEL_DIM
        or config.expansion != EXPECTED_EXPANSION
        or report["actual"] != EXPECTED_PARAMETER_COUNT
    ):
        raise RuntimeError("G13 frozen parameter-matched shape changed")
    return config


def _macro_batch(
    stream: EncodedText,
    *,
    namespace: str,
    macro_index: int,
    sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Partition one deterministic macro-window into equal causal rows."""

    if sequence_length not in CONTEXT_LENGTHS:
        raise ValueError(f"sequence_length must be one of {CONTEXT_LENGTHS}")
    if TARGETS_PER_UPDATE % sequence_length:
        raise ValueError("sequence_length must divide TARGETS_PER_UPDATE")
    maximum = stream.token_count - TARGETS_PER_UPDATE
    start = _offset(namespace, macro_index, maximum)
    macro_tokens = stream.token_ids[start : start + TARGETS_PER_UPDATE + 1]
    macro_bytes = stream.token_byte_lengths[start : start + TARGETS_PER_UPDATE + 1]
    batch_size = TARGETS_PER_UPDATE // sequence_length
    token_rows = []
    byte_rows = []
    for row in range(batch_size):
        row_start = row * sequence_length
        row_stop = row_start + sequence_length + 1
        token_rows.append(macro_tokens[row_start:row_stop])
        byte_rows.append(macro_bytes[row_start:row_stop])
    tokens = torch.stack(token_rows).to(device)
    byte_lengths = torch.stack(byte_rows).to(device)
    return tokens[:, :-1], tokens[:, 1:], byte_lengths[:, 1:], start


def _forward_logits(
    model: HybridMemoryLM,
    inputs: torch.Tensor,
    *,
    chunk_size: int = FORWARD_CHUNK_SIZE,
) -> torch.Tensor:
    """Execute long rows in live-state chunks without detaching state."""

    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    states = None
    logits = []
    for start in range(0, inputs.shape[1], chunk_size):
        output = model(
            inputs[:, start : start + chunk_size],
            states,
            delta_scan_mode="parallel",
        )
        logits.append(output["logits"])
        states = output["states"]
    return torch.cat(logits, dim=1)


def _ordinary_evaluation(
    model: HybridMemoryLM,
    validation: EncodedText,
    *,
    sequence_length: int,
    device: torch.device,
    macro_batches: int = EVAL_MACRO_BATCHES,
) -> dict[str, float | int | bool]:
    nll_sum = 0.0
    raw_bytes = 0
    correct = 0
    token_count = 0
    model.eval()
    with torch.inference_mode():
        for macro_index in range(macro_batches):
            inputs, targets, byte_lengths, _ = _macro_batch(
                validation,
                namespace="g13:validation",
                macro_index=macro_index,
                sequence_length=sequence_length,
                device=device,
            )
            logits = _forward_logits(model, inputs)
            losses = F.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), reduction="none"
            )
            nll_sum += float(losses.sum())
            raw_bytes += int(byte_lengths.sum())
            correct += int((logits.argmax(-1) == targets).sum())
            token_count += targets.numel()
    nats_per_token = nll_sum / token_count
    nats_per_raw_byte = nll_sum / raw_bytes
    return {
        "sequence_length_tokens": sequence_length,
        "batch_size": TARGETS_PER_UPDATE // sequence_length,
        "macro_batches": macro_batches,
        "scored_tokens": token_count,
        "scored_raw_bytes": raw_bytes,
        "mean_sequence_span_raw_bytes": raw_bytes
        / (macro_batches * (TARGETS_PER_UPDATE // sequence_length)),
        "mean_nats_per_token": nats_per_token,
        "mean_nats_per_raw_byte": nats_per_raw_byte,
        "bits_per_raw_byte": nats_per_raw_byte / math.log(2.0),
        "next_token_accuracy": correct / token_count,
        "finite": math.isfinite(nll_sum),
    }


def _evaluations(
    model: HybridMemoryLM,
    validation: EncodedText,
    lengths: tuple[int, ...],
    device: torch.device,
    *,
    macro_batches: int = EVAL_MACRO_BATCHES,
) -> list[dict[str, float | int | bool]]:
    return [
        _ordinary_evaluation(
            model,
            validation,
            sequence_length=length,
            device=device,
            macro_batches=macro_batches,
        )
        for length in lengths
    ]


def _phase_lengths(arm_name: str) -> tuple[int, ...]:
    if arm_name == "fixed_256_control":
        return FIXED_LENGTHS
    if arm_name == "long_context_curriculum":
        return CURRICULUM_LENGTHS
    raise ValueError(f"unknown G13 arm {arm_name!r}")


def _train_arm(
    *,
    arm_name: str,
    seed: int,
    config: HybridMemoryConfig,
    train: EncodedText,
    validation: EncodedText,
    tokenizer: ByteLevelBPETokenizer,
    checkpoint_dir: Path,
    device: torch.device,
    phase_updates: int = PHASE_UPDATES,
    eval_macro_batches: int = EVAL_MACRO_BATCHES,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = HybridMemoryLM(config).to(device)
    optimizer = build_optimizer(
        model,
        "harmonic_muon_adamw",
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    optimizer_metadata = _optimizer_report(optimizer)
    curve = [
        {
            "update": 0,
            "completed_phase": 0,
            "cumulative_training_tokens": 0,
            "cumulative_training_raw_bytes": 0,
            "ordinary_evaluation": _evaluations(
                model,
                validation,
                MONITOR_CONTEXT_LENGTHS,
                device,
                macro_batches=eval_macro_batches,
            ),
        }
    ]
    cumulative_tokens = 0
    cumulative_raw_bytes = 0
    target_digest = hashlib.sha256()
    update_times: list[float] = []
    phase_systems = []
    training_peak = 0
    evaluation_peak = 0
    phase_lengths = _phase_lengths(arm_name)
    training_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    last_loss = float("nan")
    last_gradient_norm = float("nan")
    for phase_index, sequence_length in enumerate(phase_lengths, start=1):
        phase_times = []
        phase_started = time.perf_counter()
        phase_first_update = (phase_index - 1) * phase_updates + 1
        phase_last_update = phase_index * phase_updates
        for update in range(phase_first_update, phase_last_update + 1):
            step_started = time.perf_counter()
            inputs, targets, byte_lengths, macro_start = _macro_batch(
                train,
                namespace=f"g13:training:{seed}",
                macro_index=update - 1,
                sequence_length=sequence_length,
                device=device,
            )
            expected_targets = train.token_ids[
                macro_start + 1 : macro_start + TARGETS_PER_UPDATE + 1
            ]
            target_digest.update(expected_targets.numpy().tobytes())
            optimizer.zero_grad(set_to_none=True)
            logits = _forward_logits(model, inputs)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite G13 loss for {arm_name} seed {seed} update {update}"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRADIENT_CLIP
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError(
                    f"non-finite G13 gradient for {arm_name} seed {seed} update {update}"
                )
            optimizer.step()
            cumulative_tokens += targets.numel()
            cumulative_raw_bytes += int(byte_lengths.sum())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                training_peak = max(
                    training_peak, torch.cuda.max_memory_allocated(device)
                )
            elapsed = time.perf_counter() - step_started
            phase_times.append(elapsed)
            update_times.append(elapsed)
            last_loss = float(loss.detach())
            last_gradient_norm = float(gradient_norm)
        phase_systems.append(
            {
                "phase": phase_index,
                "sequence_length_tokens": sequence_length,
                "batch_size": TARGETS_PER_UPDATE // sequence_length,
                "updates": phase_updates,
                "wall_seconds": time.perf_counter() - phase_started,
                "median_update_wall_seconds_after_10_warmups": statistics.median(
                    phase_times[min(10, max(0, len(phase_times) - 1)) :]
                ),
            }
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        evaluation_lengths = (
            CONTEXT_LENGTHS
            if phase_index == len(phase_lengths)
            else MONITOR_CONTEXT_LENGTHS
        )
        ordinary = _evaluations(
            model,
            validation,
            evaluation_lengths,
            device,
            macro_batches=eval_macro_batches,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            evaluation_peak = max(
                evaluation_peak, torch.cuda.max_memory_allocated(device)
            )
            torch.cuda.reset_peak_memory_stats(device)
        curve.append(
            {
                "update": phase_last_update,
                "completed_phase": phase_index,
                "phase_sequence_length_tokens": sequence_length,
                "training_loss_nats_per_token": last_loss,
                "pre_clip_gradient_norm": last_gradient_norm,
                "cumulative_training_tokens": cumulative_tokens,
                "cumulative_training_raw_bytes": cumulative_raw_bytes,
                "ordinary_evaluation": ordinary,
            }
        )
        model.train()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"g13_{arm_name}_seed{seed}.pt"
    fingerprint = tokenizer_fingerprint(tokenizer)
    torch.save(
        {
            "schema_version": 1,
            "stage": "G13",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(config),
            "seed": seed,
            "arm_name": arm_name,
            "optimizer_name": "harmonic_muon_adamw",
            "tokenizer": {
                "name": tokenizer.name,
                "vocab_size": tokenizer.vocab_size,
                "sha256": fingerprint,
            },
            "phase_lengths": phase_lengths,
            "targets_per_update": TARGETS_PER_UPDATE,
            "training_target_ids_sha256": target_digest.hexdigest(),
        },
        checkpoint,
    )
    run = {
        "seed": seed,
        "arm_name": arm_name,
        "optimizer_name": "harmonic_muon_adamw",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "config": asdict(config),
        "tokenizer": {
            "name": tokenizer.name,
            "vocab_size": tokenizer.vocab_size,
            "sha256": fingerprint,
        },
        "phase_lengths": list(phase_lengths),
        "phase_systems": phase_systems,
        "learning_curve": curve,
        "training_target_ids_sha256": target_digest.hexdigest(),
        "optimizer": optimizer_metadata,
        "systems": {
            "training_wall_seconds_excluding_evaluation": sum(update_times),
            "elapsed_wall_seconds_including_evaluation": time.perf_counter()
            - training_started,
            "median_update_wall_seconds_after_50_warmups": statistics.median(
                update_times[min(50, max(0, len(update_times) - 1)) :]
            ),
            "peak_training_cuda_allocated_bytes": (
                training_peak if device.type == "cuda" else None
            ),
            "peak_evaluation_cuda_allocated_bytes": (
                evaluation_peak if device.type == "cuda" else None
            ),
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
    }
    return {"run": run, "model": model, "optimizer": optimizer}


@contextmanager
def _mixer_mode(model: HybridMemoryLM, mode: str) -> Iterator[None]:
    if mode not in RECALL_MODES:
        raise ValueError(f"mode must be one of {RECALL_MODES}")
    if mode == "full":
        yield
        return
    suppressed_kind = mode.removesuffix("_off")
    saved = []
    with torch.no_grad():
        for block in model.blocks:
            if block.kind == suppressed_kind:
                saved.append(
                    (block.residual_scale, block.residual_scale.detach().clone())
                )
                block.residual_scale.fill_(-30.0)
    if not saved:
        raise RuntimeError(f"model has no {suppressed_kind!r} mixer to suppress")
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, value in saved:
                parameter.copy_(value)


def _continuation_score(
    model: HybridMemoryLM,
    tokenizer: LosslessTextTokenizer,
    prompt: str,
    continuation: str,
    device: torch.device,
) -> tuple[float, int, int]:
    prompt_ids = tokenizer.encode(prompt).token_ids
    full_ids = tokenizer.encode(prompt + continuation).token_ids
    if full_ids[: prompt_ids.numel()].tolist() != prompt_ids.tolist():
        raise RuntimeError("recall continuation does not preserve the prompt prefix")
    continuation_ids = full_ids[prompt_ids.numel() :]
    if not continuation_ids.numel():
        raise RuntimeError("recall continuation encoded to no tokens")
    inputs = full_ids[:-1].unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        logits = _forward_logits(model, inputs)[0]
        start = prompt_ids.numel() - 1
        continuation_logits = logits[start : start + continuation_ids.numel()]
        log_probabilities = F.log_softmax(continuation_logits.float(), dim=-1)
        selected = log_probabilities.gather(
            -1, continuation_ids.to(device).unsqueeze(-1)
        )
    return float(selected.sum()), prompt_ids.numel(), continuation_ids.numel()


def _recall_evaluation(
    model: HybridMemoryLM,
    tokenizer: ByteLevelBPETokenizer,
    filler_source: str,
    *,
    mode: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    with _mixer_mode(model, mode):
        for distance in RECALL_RAW_BYTE_DISTANCES:
            for example in range(RECALL_EXAMPLES):
                pair = _recall_pair(filler_source, distance=distance, example=example)
                matching, matching_tokens, continuation_tokens = _continuation_score(
                    model,
                    tokenizer,
                    str(pair["matching_prompt"]),
                    str(pair["continuation"]),
                    device,
                )
                mismatched, mismatched_tokens, mismatch_continuation_tokens = (
                    _continuation_score(
                        model,
                        tokenizer,
                        str(pair["mismatched_prompt"]),
                        str(pair["continuation"]),
                        device,
                    )
                )
                if continuation_tokens != mismatch_continuation_tokens:
                    raise RuntimeError(
                        "counterfactual continuation token counts differ"
                    )
                rows.append(
                    {
                        "mode": mode,
                        "distance_raw_bytes": distance,
                        "example": example,
                        "name": pair["name"],
                        "supported_value": pair["supported_value"],
                        "counterfactual_value": pair["counterfactual_value"],
                        "filler_offset": pair["filler_offset"],
                        "matching_prompt_tokens": matching_tokens,
                        "mismatched_prompt_tokens": mismatched_tokens,
                        "prompt_exceeds_attention_window": (
                            matching_tokens > model.config.attention_window_size
                        ),
                        "attention_window_tokens": model.config.attention_window_size,
                        "continuation_tokens": continuation_tokens,
                        "continuation_raw_bytes": len(
                            str(pair["continuation"]).encode("utf-8")
                        ),
                        "matching_log_probability_nats": matching,
                        "mismatched_log_probability_nats": mismatched,
                        "matching_minus_mismatched_nats": matching - mismatched,
                    }
                )
    return rows


def _curve_evaluation(run: dict[str, Any], update: int, length: int) -> float:
    point = next(row for row in run["learning_curve"] if row["update"] == update)
    evaluation = next(
        row
        for row in point["ordinary_evaluation"]
        if row["sequence_length_tokens"] == length
    )
    return float(evaluation["bits_per_raw_byte"])


def _summaries(runs: list[dict[str, Any]]) -> dict[str, Any]:
    ordinary = []
    by_arm_seed = {(run["arm_name"], run["seed"]): run for run in runs}
    for length in CONTEXT_LENGTHS:
        control_values = [
            _curve_evaluation(by_arm_seed[("fixed_256_control", seed)], 1000, length)
            for seed in MODEL_SEEDS
        ]
        curriculum_values = [
            _curve_evaluation(
                by_arm_seed[("long_context_curriculum", seed)], 1000, length
            )
            for seed in MODEL_SEEDS
        ]
        deltas = [
            candidate - control
            for candidate, control in zip(
                curriculum_values, control_values, strict=True
            )
        ]
        ordinary.append(
            {
                "sequence_length_tokens": length,
                "fixed_control_mean_bprb": statistics.mean(control_values),
                "fixed_control_worst_bprb": max(control_values),
                "curriculum_mean_bprb": statistics.mean(curriculum_values),
                "curriculum_worst_bprb": max(curriculum_values),
                "curriculum_minus_control_by_seed_bprb": dict(
                    zip((str(seed) for seed in MODEL_SEEDS), deltas, strict=True)
                ),
                "curriculum_minus_control_mean_bprb": statistics.mean(deltas),
                "curriculum_wins": sum(delta < 0.0 for delta in deltas),
            }
        )

    recall_groups: dict[tuple[str, str, int, int], list[float]] = defaultdict(list)
    for run in runs:
        for row in run["recall"]:
            recall_groups[
                (
                    run["arm_name"],
                    row["mode"],
                    run["seed"],
                    row["distance_raw_bytes"],
                )
            ].append(row["matching_minus_mismatched_nats"])
    recall = []
    for arm_name in ("fixed_256_control", "long_context_curriculum"):
        for mode in RECALL_MODES:
            for distance in RECALL_RAW_BYTE_DISTANCES:
                seed_means = {
                    str(seed): statistics.mean(
                        recall_groups[(arm_name, mode, seed, distance)]
                    )
                    for seed in MODEL_SEEDS
                }
                recall.append(
                    {
                        "arm_name": arm_name,
                        "mode": mode,
                        "distance_raw_bytes": distance,
                        "seed_mean_gains_nats": seed_means,
                        "mean_gain_nats_across_seeds": statistics.mean(
                            seed_means.values()
                        ),
                        "minimum_seed_mean_gain_nats": min(seed_means.values()),
                        "positive_seed_means": sum(
                            value > 0.0 for value in seed_means.values()
                        ),
                    }
                )
    return {"ordinary_final": ordinary, "recall": recall}


def _decision(runs: list[dict[str, Any]], summaries: dict[str, Any]) -> dict[str, Any]:
    by_arm_seed = {(run["arm_name"], run["seed"]): run for run in runs}
    target_pairs = []
    phase1_deltas = []
    finite = True
    for seed in MODEL_SEEDS:
        control = by_arm_seed[("fixed_256_control", seed)]
        curriculum = by_arm_seed[("long_context_curriculum", seed)]
        target_pairs.append(
            {
                "seed": seed,
                "target_hash_equal": control["training_target_ids_sha256"]
                == curriculum["training_target_ids_sha256"],
                "raw_bytes_equal": control["learning_curve"][-1][
                    "cumulative_training_raw_bytes"
                ]
                == curriculum["learning_curve"][-1]["cumulative_training_raw_bytes"],
            }
        )
        phase1_deltas.append(
            _curve_evaluation(curriculum, 200, 256)
            - _curve_evaluation(control, 200, 256)
        )
    for run in runs:
        for point in run["learning_curve"]:
            finite = finite and all(
                bool(row["finite"]) for row in point["ordinary_evaluation"]
            )
        finite = finite and all(
            math.isfinite(row["matching_minus_mismatched_nats"])
            for row in run["recall"]
        )
    integrity = (
        finite
        and all(
            row["target_hash_equal"] and row["raw_bytes_equal"] for row in target_pairs
        )
        and max(abs(delta) for delta in phase1_deltas) <= 1e-6
    )

    ordinary_256 = next(
        row
        for row in summaries["ordinary_final"]
        if row["sequence_length_tokens"] == 256
    )
    ordinary_4096 = next(
        row
        for row in summaries["ordinary_final"]
        if row["sequence_length_tokens"] == 4096
    )
    delta_256 = ordinary_256["curriculum_minus_control_mean_bprb"]
    seed_deltas_256 = ordinary_256["curriculum_minus_control_by_seed_bprb"].values()
    ordinary_pass = (
        ordinary_4096["curriculum_minus_control_mean_bprb"] <= -0.02
        and ordinary_4096["curriculum_wins"] >= 2
        and delta_256 <= 0.02
        and max(seed_deltas_256) <= 0.05
    )

    def recall_row(arm: str, mode: str) -> dict[str, Any]:
        return next(
            row
            for row in summaries["recall"]
            if row["arm_name"] == arm
            and row["mode"] == mode
            and row["distance_raw_bytes"] == 8192
        )

    curriculum_full = recall_row("long_context_curriculum", "full")
    control_full = recall_row("fixed_256_control", "full")
    curriculum_gdn_off = recall_row("long_context_curriculum", "gated_delta_off")
    recall_control_delta = (
        curriculum_full["mean_gain_nats_across_seeds"]
        - control_full["mean_gain_nats_across_seeds"]
    )
    recall_gdn_contribution = (
        curriculum_full["mean_gain_nats_across_seeds"]
        - curriculum_gdn_off["mean_gain_nats_across_seeds"]
    )
    recall_pass = (
        curriculum_full["positive_seed_means"] == len(MODEL_SEEDS)
        and curriculum_full["mean_gain_nats_across_seeds"] >= 0.02
        and recall_control_delta >= 0.01
        and recall_gdn_contribution >= 0.01
    )
    return {
        "passed_integrity_gate": integrity,
        "passed_ordinary_long_context_gate": ordinary_pass,
        "passed_learned_long_range_recall_gate": recall_pass,
        "passed_full_g13_promotion": integrity and ordinary_pass and recall_pass,
        "finite": finite,
        "paired_target_integrity": target_pairs,
        "phase1_curriculum_minus_control_bprb": dict(
            zip((str(seed) for seed in MODEL_SEEDS), phase1_deltas, strict=True)
        ),
        "ordinary_4096_curriculum_minus_control_mean_bprb": ordinary_4096[
            "curriculum_minus_control_mean_bprb"
        ],
        "ordinary_4096_curriculum_wins": ordinary_4096["curriculum_wins"],
        "ordinary_256_curriculum_minus_control_mean_bprb": delta_256,
        "ordinary_256_maximum_seed_regression_bprb": max(seed_deltas_256),
        "recall_8192_curriculum_full_mean_gain_nats": curriculum_full[
            "mean_gain_nats_across_seeds"
        ],
        "recall_8192_curriculum_positive_seed_means": curriculum_full[
            "positive_seed_means"
        ],
        "recall_8192_curriculum_minus_control_mean_gain_nats": recall_control_delta,
        "recall_8192_gated_delta_contribution_nats": recall_gdn_contribution,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--tokenizer-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(MODEL_SEEDS),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--phase-updates", type=int, default=PHASE_UPDATES, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--eval-macro-batches",
        type=int,
        default=EVAL_MACRO_BATCHES,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--skip-recall", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if tuple(args.seeds) != MODEL_SEEDS and args.phase_updates == PHASE_UPDATES:
        parser.error("non-smoke G13 runs must use the frozen seed cohort")
    if args.phase_updates != PHASE_UPDATES and len(args.seeds) != 1:
        parser.error("G13 smoke overrides require exactly one seed")
    if args.eval_macro_batches < 1:
        parser.error("--eval-macro-batches must be positive")
    device = torch.device(args.device)
    git_commit, git_status = _git()
    train_text, validation_text, snapshot = _snapshot_text(args.snapshot)
    tokenizer = _load_bpe(args.tokenizer_audit)
    train = tokenizer.encode(train_text)
    validation = tokenizer.encode(validation_text)
    config = _frozen_config(tokenizer.vocab_size)
    validation_ascii = _ascii_filler(validation_text)
    started = time.perf_counter()
    runs = []
    for seed in args.seeds:
        for arm_name in ("fixed_256_control", "long_context_curriculum"):
            result = _train_arm(
                arm_name=arm_name,
                seed=seed,
                config=config,
                train=train,
                validation=validation,
                tokenizer=tokenizer,
                checkpoint_dir=args.checkpoint_dir,
                device=device,
                phase_updates=args.phase_updates,
                eval_macro_batches=args.eval_macro_batches,
            )
            run = result["run"]
            model = result["model"]
            if not args.skip_recall:
                run["recall"] = [
                    row
                    for mode in RECALL_MODES
                    for row in _recall_evaluation(
                        model,
                        tokenizer,
                        validation_ascii,
                        mode=mode,
                        device=device,
                    )
                ]
            else:
                run["recall"] = []
            runs.append(run)
            del model, result["optimizer"]
            if device.type == "cuda":
                torch.cuda.empty_cache()
    frozen_run = (
        tuple(args.seeds) == MODEL_SEEDS
        and args.phase_updates == PHASE_UPDATES
        and args.eval_macro_batches == EVAL_MACRO_BATCHES
        and not args.skip_recall
    )
    if frozen_run:
        summaries = _summaries(runs)
        decision = _decision(runs, summaries)
    else:
        summaries = {}
        decision = {"smoke_only": True, "not_a_g13_result": True}
    output = {
        "schema_version": 1,
        "stage": "G13" if frozen_run else "G13_SMOKE",
        "claim_status": (
            (
                "passed full G13 promotion"
                if decision["passed_full_g13_promotion"]
                else "failed full G13 promotion; inspect separated gates"
            )
            if frozen_run
            else "smoke only; not a G13 result"
        ),
        "decision": decision,
        "summaries": summaries,
        "runs": runs,
        "protocol": {
            "model_seeds": list(args.seeds),
            "data_seed": DATA_SEED,
            "context_lengths_tokens": list(CONTEXT_LENGTHS),
            "curriculum_lengths_tokens": list(CURRICULUM_LENGTHS),
            "fixed_control_lengths_tokens": list(FIXED_LENGTHS),
            "phase_updates": args.phase_updates,
            "targets_per_update": TARGETS_PER_UPDATE,
            "forward_chunk_size": FORWARD_CHUNK_SIZE,
            "eval_macro_batches": args.eval_macro_batches,
            "recall_raw_byte_distances": list(RECALL_RAW_BYTE_DISTANCES),
            "recall_examples_per_distance": RECALL_EXAMPLES,
            "recall_modes": list(RECALL_MODES),
        },
        "dataset": {
            "snapshot": str(args.snapshot),
            "snapshot_sha256": _sha256(args.snapshot),
            "hub_sha": snapshot["hub_sha_at_snapshot"],
            "train_tokens": train.token_count,
            "validation_tokens": validation.token_count,
        },
        "tokenizer": {
            "name": tokenizer.name,
            "vocab_size": tokenizer.vocab_size,
            "sha256": tokenizer_fingerprint(tokenizer),
            "audit": str(args.tokenizer_audit),
            "audit_sha256": _sha256(args.tokenizer_audit),
        },
        "model_config": asdict(config),
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "preregistration": str(PREREGISTRATION),
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "git_commit_at_start": git_commit,
        "git_status_at_start": git_status,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "elapsed_wall_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "exact target-token and raw-byte exposure match; not compute matched, "
            "not a scaling law, and recall promotion requires the frozen causal gate"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(json.dumps({"decision": decision, "summaries": summaries}, sort_keys=True))


if __name__ == "__main__":
    main()
