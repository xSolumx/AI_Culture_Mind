"""Run frozen G12 optimizer development and natural-text robustness stages."""

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
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from .model import HybridMemoryConfig, HybridMemoryLM
    from .natural_text_data import SEPARATOR
    from .optimizers import HarmonicMuonAdamW, build_optimizer
    from .successor_screen import _retention_safe_config
    from .tokenization import (
        ByteLevelBPETokenizer,
        EncodedText,
        LosslessTextTokenizer,
        RawByteTokenizer,
        tokenizer_fingerprint,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4.model import (  # type: ignore[no-redef]
        HybridMemoryConfig,
        HybridMemoryLM,
    )
    from hybrid_memory_v1_4.natural_text_data import (  # type: ignore[no-redef]
        SEPARATOR,
    )
    from hybrid_memory_v1_4.optimizers import (  # type: ignore[no-redef]
        HarmonicMuonAdamW,
        build_optimizer,
    )
    from hybrid_memory_v1_4.successor_screen import (  # type: ignore[no-redef]
        _retention_safe_config,
    )
    from hybrid_memory_v1_4.tokenization import (  # type: ignore[no-redef]
        ByteLevelBPETokenizer,
        EncodedText,
        LosslessTextTokenizer,
        RawByteTokenizer,
        tokenizer_fingerprint,
    )


PREREGISTRATION = Path(__file__).with_name("G12_PREREGISTRATION.md")
SNAPSHOT_SHA256 = "ebe7c7c948f3e59781097ffa64e214da15364d61b38622d33dd076d40471adc6"
TRAIN_BYTES_SHA256 = "e25563b202a9f669b5d479d0cf94bd9a2c48a0050fcb569d63f1f6fc8580b4c1"
VALIDATION_BYTES_SHA256 = (
    "4b7aadf02f91e1dce13b699bd428bda3984ac4ee9ec17167db1813e0a34b1b53"
)
DATA_SEED = 1817
DEVELOPMENT_SEEDS = (1823, 1829)
VALIDATION_SEEDS = (1871, 1873, 1877)
SEQUENCE_LENGTH = 256
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
EVAL_BATCHES = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
DEVELOPMENT_UPDATES = 500
VALIDATION_UPDATES = 1000
DEVELOPMENT_EVAL_UPDATES = (0, 250, 500)
VALIDATION_EVAL_UPDATES = (0, 250, 500, 750, 1000)
TARGET_PARAMETER_COUNT = 119_962


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _snapshot_text(snapshot: Path) -> tuple[str, str, dict[str, Any]]:
    if _sha256(snapshot) != SNAPSHOT_SHA256:
        raise ValueError("TinyStories snapshot hash does not match frozen G12 input")
    report = json.loads(snapshot.read_text(encoding="utf-8"))
    train_text = SEPARATOR.join(row["text"] for row in report["train"]["rows"])
    validation_text = SEPARATOR.join(
        row["text"] for row in report["validation"]["rows"]
    )
    if hashlib.sha256(train_text.encode("utf-8")).hexdigest() != TRAIN_BYTES_SHA256:
        raise ValueError("G12 training text hash mismatch")
    if (
        hashlib.sha256(validation_text.encode("utf-8")).hexdigest()
        != VALIDATION_BYTES_SHA256
    ):
        raise ValueError("G12 validation text hash mismatch")
    return train_text, validation_text, report


def _offset(namespace: str, index: int, maximum: int) -> int:
    if maximum < 1:
        raise ValueError("token stream is too short for the frozen sequence length")
    payload = f"{namespace}:{DATA_SEED}:{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % maximum


def _batch(
    stream: EncodedText,
    *,
    namespace: str,
    batch_index: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum = stream.token_count - SEQUENCE_LENGTH
    token_windows = []
    byte_windows = []
    for row in range(batch_size):
        index = batch_index * batch_size + row
        start = _offset(namespace, index, maximum)
        token_windows.append(stream.token_ids[start : start + SEQUENCE_LENGTH + 1])
        byte_windows.append(
            stream.token_byte_lengths[start : start + SEQUENCE_LENGTH + 1]
        )
    tokens = torch.stack(token_windows).to(device)
    byte_lengths = torch.stack(byte_windows).to(device)
    return tokens[:, :-1], tokens[:, 1:], byte_lengths[:, 1:]


def _parameter_matched_config(
    vocab_size: int,
) -> tuple[HybridMemoryConfig, dict[str, int]]:
    candidates = []
    for width in range(24, 97, 4):
        for expansion in range(1, 7):
            try:
                config = replace(
                    _retention_safe_config(),
                    vocab_size=vocab_size,
                    model_dim=width,
                    expansion=expansion,
                    gated_delta_key_dim=width // 2,
                    gated_delta_value_dim=width // 4,
                )
                model = HybridMemoryLM(config)
            except ValueError:
                continue
            count = sum(parameter.numel() for parameter in model.parameters())
            candidates.append(
                (abs(count - TARGET_PARAMETER_COUNT), -width, expansion, count, config)
            )
    if not candidates:
        raise RuntimeError("no valid parameter-matched configuration")
    difference, negative_width, expansion, count, config = min(candidates)
    return config, {
        "target": TARGET_PARAMETER_COUNT,
        "actual": count,
        "absolute_difference": difference,
        "relative_difference": difference / TARGET_PARAMETER_COUNT,
        "model_dim": -negative_width,
        "expansion": expansion,
    }


def _build_model(
    config: HybridMemoryConfig, seed: int, device: torch.device
) -> nn.Module:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return HybridMemoryLM(config).to(device)


def _evaluate(
    model: nn.Module,
    validation: EncodedText,
    *,
    namespace: str,
    device: torch.device,
) -> dict[str, float | int | bool]:
    nll_sum = 0.0
    raw_bytes = 0
    correct = 0
    token_count = 0
    model.eval()
    with torch.inference_mode():
        for batch_index in range(EVAL_BATCHES):
            inputs, targets, byte_lengths = _batch(
                validation,
                namespace=namespace,
                batch_index=batch_index,
                batch_size=EVAL_BATCH_SIZE,
                device=device,
            )
            logits = model(inputs, delta_scan_mode="parallel")["logits"]
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
        "scored_tokens": token_count,
        "scored_raw_bytes": raw_bytes,
        "mean_nats_per_token": nats_per_token,
        "mean_nats_per_raw_byte": nats_per_raw_byte,
        "bits_per_raw_byte": nats_per_raw_byte / math.log(2.0),
        "next_token_accuracy": correct / token_count,
        "finite": math.isfinite(nll_sum),
    }


def _optimizer_report(
    optimizer: torch.optim.Optimizer | HarmonicMuonAdamW,
) -> dict[str, Any]:
    if isinstance(optimizer, HarmonicMuonAdamW):
        return {
            "name": "harmonic_muon_adamw",
            "runtime_class": f"{type(optimizer).__module__}.{type(optimizer).__name__}",
            "partition": optimizer.partition_report(),
        }
    return {
        "name": "adamw",
        "runtime_class": f"{type(optimizer).__module__}.{type(optimizer).__name__}",
        "partition": [
            {
                "optimizer": "torch.optim.AdamW",
                "role": "all_parameters",
                "parameters": sum(
                    parameter.numel()
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                ),
                "second_moment": "coordinatewise",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            }
        ],
    }


def _train(
    model: nn.Module,
    train: EncodedText,
    validation: EncodedText,
    *,
    optimizer_name: str,
    namespace: str,
    updates: int,
    eval_updates: tuple[int, ...],
    device: torch.device,
) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
    optimizer = build_optimizer(
        model,
        optimizer_name,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    optimizer_metadata = _optimizer_report(optimizer)
    curve = [
        {
            "update": 0,
            "cumulative_training_tokens": 0,
            "cumulative_training_raw_bytes": 0,
            "cumulative_training_wall_seconds": 0.0,
            **_evaluate(
                model,
                validation,
                namespace=f"{namespace}:validation",
                device=device,
            ),
        }
    ]
    cumulative_raw_bytes = 0
    cumulative_tokens = 0
    training_started = time.perf_counter()
    update_times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for update in range(1, updates + 1):
        step_started = time.perf_counter()
        inputs, targets, byte_lengths = _batch(
            train,
            namespace=f"{namespace}:training",
            batch_index=update - 1,
            batch_size=BATCH_SIZE,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs, delta_scan_mode="parallel")["logits"]
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite G12 loss for {optimizer_name}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRADIENT_CLIP
        )
        optimizer.step()
        cumulative_raw_bytes += int(byte_lengths.sum())
        cumulative_tokens += targets.numel()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        update_times.append(time.perf_counter() - step_started)
        if update in eval_updates:
            training_elapsed = time.perf_counter() - training_started
            evaluation = _evaluate(
                model,
                validation,
                namespace=f"{namespace}:validation",
                device=device,
            )
            curve.append(
                {
                    "update": update,
                    "training_loss_nats_per_token": float(loss.detach()),
                    "pre_clip_gradient_norm": float(gradient_norm),
                    "cumulative_training_tokens": cumulative_tokens,
                    "cumulative_training_raw_bytes": cumulative_raw_bytes,
                    "cumulative_training_wall_seconds": training_elapsed,
                    **evaluation,
                }
            )
            model.train()
    timing_slice = update_times[min(50, len(update_times)) :]
    systems = {
        "training_wall_seconds": time.perf_counter() - training_started,
        "median_update_wall_seconds_after_50_warmups": (
            statistics.median(timing_slice) if timing_slice else None
        ),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }
    return curve, optimizer, {"optimizer": optimizer_metadata, "systems": systems}


def _load_bpe_from_audit(
    audit_path: Path,
) -> tuple[ByteLevelBPETokenizer, dict[str, Any]]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    selection = audit["selection_rule"]
    path = Path(selection["selected_path"])
    tokenizer = ByteLevelBPETokenizer.from_serialized(path.read_text(encoding="utf-8"))
    if tokenizer_fingerprint(tokenizer) != selection["selected_sha256"]:
        raise ValueError("selected G12 tokenizer fingerprint mismatch")
    return tokenizer, audit


def _run_development(
    train_text: str,
    validation_text: str,
    *,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = RawByteTokenizer()
    train = tokenizer.encode(train_text)
    validation = tokenizer.encode(validation_text)
    config = replace(_retention_safe_config(), vocab_size=tokenizer.vocab_size)
    runs = []
    for seed in DEVELOPMENT_SEEDS:
        for optimizer_name in ("adamw", "harmonic_muon_adamw"):
            model = _build_model(config, seed, device)
            started = time.perf_counter()
            curve, optimizer, metadata = _train(
                model,
                train,
                validation,
                optimizer_name=optimizer_name,
                namespace=f"g12b:{seed}:raw_utf8_bytes",
                updates=DEVELOPMENT_UPDATES,
                eval_updates=DEVELOPMENT_EVAL_UPDATES,
                device=device,
            )
            runs.append(
                {
                    "seed": seed,
                    "optimizer_name": optimizer_name,
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "config": asdict(config),
                    "tokenizer": {
                        "name": tokenizer.name,
                        "vocab_size": tokenizer.vocab_size,
                        "sha256": tokenizer_fingerprint(tokenizer),
                    },
                    "learning_curve": curve,
                    **metadata,
                    "elapsed_wall_seconds_including_evaluation": time.perf_counter()
                    - started,
                }
            )
            del model, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
    paired = []
    for seed in DEVELOPMENT_SEEDS:
        adamw = next(
            run
            for run in runs
            if run["seed"] == seed and run["optimizer_name"] == "adamw"
        )
        harmonic = next(
            run
            for run in runs
            if run["seed"] == seed and run["optimizer_name"] == "harmonic_muon_adamw"
        )
        delta = (
            harmonic["learning_curve"][-1]["bits_per_raw_byte"]
            - adamw["learning_curve"][-1]["bits_per_raw_byte"]
        )
        paired.append({"seed": seed, "harmonic_minus_adamw_bprb": delta})
    deltas = [row["harmonic_minus_adamw_bprb"] for row in paired]
    harmonic_advances = (
        all(math.isfinite(delta) for delta in deltas)
        and statistics.mean(deltas) <= -0.02
        and max(deltas) <= 0.02
    )
    decision = {
        "selected_optimizer": ("harmonic_muon_adamw" if harmonic_advances else "adamw"),
        "harmonic_advances": harmonic_advances,
        "paired": paired,
        "mean_harmonic_minus_adamw_bprb": statistics.mean(deltas),
        "required_mean_delta_at_most": -0.02,
        "maximum_allowed_single_seed_regression": 0.02,
    }
    return runs, decision


def _run_validation(
    train_text: str,
    validation_text: str,
    *,
    optimizer_audit: Path,
    tokenizer_audit: Path,
    checkpoint_dir: Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    optimizer_result = json.loads(optimizer_audit.read_text(encoding="utf-8"))
    selected_optimizer = optimizer_result["decision"]["selected_optimizer"]
    bpe, tokenizer_result = _load_bpe_from_audit(tokenizer_audit)
    raw = RawByteTokenizer()
    bpe_config, matching = _parameter_matched_config(bpe.vocab_size)
    arms: list[tuple[str, LosslessTextTokenizer, HybridMemoryConfig, str]] = [
        (
            "raw_adamw_control",
            raw,
            replace(_retention_safe_config(), vocab_size=raw.vocab_size),
            "adamw",
        )
    ]
    if selected_optimizer != "adamw":
        arms.append(
            (
                "raw_selected_optimizer",
                raw,
                replace(_retention_safe_config(), vocab_size=raw.vocab_size),
                selected_optimizer,
            )
        )
    arms.append(("bpe_parameter_matched", bpe, bpe_config, selected_optimizer))
    encoded: dict[str, tuple[EncodedText, EncodedText]] = {}
    for _, tokenizer, _, _ in arms:
        fingerprint = tokenizer_fingerprint(tokenizer)
        if fingerprint not in encoded:
            encoded[fingerprint] = (
                tokenizer.encode(train_text),
                tokenizer.encode(validation_text),
            )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in VALIDATION_SEEDS:
        for arm_name, tokenizer, config, optimizer_name in arms:
            fingerprint = tokenizer_fingerprint(tokenizer)
            train, validation = encoded[fingerprint]
            model = _build_model(config, seed, device)
            started = time.perf_counter()
            curve, optimizer, metadata = _train(
                model,
                train,
                validation,
                optimizer_name=optimizer_name,
                namespace=f"g12c:{seed}:{fingerprint}",
                updates=VALIDATION_UPDATES,
                eval_updates=VALIDATION_EVAL_UPDATES,
                device=device,
            )
            checkpoint = checkpoint_dir / f"g12c_{arm_name}_seed{seed}.pt"
            torch.save(
                {
                    "schema_version": 1,
                    "stage": "G12C",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": asdict(config),
                    "seed": seed,
                    "arm_name": arm_name,
                    "optimizer_name": optimizer_name,
                    "tokenizer": {
                        "name": tokenizer.name,
                        "vocab_size": tokenizer.vocab_size,
                        "sha256": fingerprint,
                    },
                },
                checkpoint,
            )
            runs.append(
                {
                    "seed": seed,
                    "arm_name": arm_name,
                    "optimizer_name": optimizer_name,
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "config": asdict(config),
                    "tokenizer": {
                        "name": tokenizer.name,
                        "vocab_size": tokenizer.vocab_size,
                        "sha256": fingerprint,
                        "train_tokens": train.token_count,
                        "train_raw_bytes": train.raw_byte_count,
                        "validation_tokens": validation.token_count,
                        "validation_raw_bytes": validation.raw_byte_count,
                    },
                    "learning_curve": curve,
                    **metadata,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _sha256(checkpoint),
                    "elapsed_wall_seconds_including_evaluation": time.perf_counter()
                    - started,
                }
            )
            del model, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
    control = [run for run in runs if run["arm_name"] == "raw_adamw_control"]
    selected_arm = "bpe_parameter_matched"
    selected = [run for run in runs if run["arm_name"] == selected_arm]
    control_final = [run["learning_curve"][-1]["bits_per_raw_byte"] for run in control]
    selected_final = [
        run["learning_curve"][-1]["bits_per_raw_byte"] for run in selected
    ]
    selected_improvements = [
        run["learning_curve"][0]["bits_per_raw_byte"]
        - run["learning_curve"][-1]["bits_per_raw_byte"]
        for run in selected
    ]
    passed = (
        all(point["finite"] for run in selected for point in run["learning_curve"])
        and min(selected_improvements) >= 2.0
        and statistics.mean(selected_final) <= statistics.mean(control_final) + 0.02
        and max(selected_final) <= max(control_final) + 0.05
    )
    decision = {
        "selected_optimizer": selected_optimizer,
        "selected_tokenizer_vocab_size": bpe.vocab_size,
        "selected_arm": selected_arm,
        "passed_robustness_gate": passed,
        "control_mean_final_bprb": statistics.mean(control_final),
        "control_worst_final_bprb": max(control_final),
        "selected_mean_final_bprb": statistics.mean(selected_final),
        "selected_worst_final_bprb": max(selected_final),
        "selected_minimum_improvement_bprb": min(selected_improvements),
        "parameter_matching": matching,
        "optimizer_audit_sha256": _sha256(optimizer_audit),
        "tokenizer_audit_sha256": _sha256(tokenizer_audit),
        "tokenizer_selection": tokenizer_result["selection_rule"],
    }
    return runs, decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("development", "validation"), required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimizer-audit", type=Path)
    parser.add_argument("--tokenizer-audit", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.stage == "validation" and (
        args.optimizer_audit is None
        or args.tokenizer_audit is None
        or args.checkpoint_dir is None
    ):
        parser.error(
            "validation requires --optimizer-audit, --tokenizer-audit, and --checkpoint-dir"
        )
    device = torch.device(args.device)
    git_commit, git_status = _git()
    train_text, validation_text, snapshot = _snapshot_text(args.snapshot)
    started = time.perf_counter()
    if args.stage == "development":
        runs, decision = _run_development(train_text, validation_text, device=device)
        stage = "G12B"
        claim_status = "completed optimizer development screen"
    else:
        runs, decision = _run_validation(
            train_text,
            validation_text,
            optimizer_audit=args.optimizer_audit,
            tokenizer_audit=args.tokenizer_audit,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
        )
        stage = "G12C"
        claim_status = (
            "passed fresh multi-seed natural-text robustness gate"
            if decision["passed_robustness_gate"]
            else "failed fresh multi-seed natural-text robustness gate"
        )
    output = {
        "schema_version": 1,
        "stage": stage,
        "claim_status": claim_status,
        "decision": decision,
        "runs": runs,
        "training": {
            "sequence_length_tokens": SEQUENCE_LENGTH,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "updates": (
                DEVELOPMENT_UPDATES
                if args.stage == "development"
                else VALIDATION_UPDATES
            ),
        },
        "evaluation": {
            "batch_size": EVAL_BATCH_SIZE,
            "batches": EVAL_BATCHES,
            "primary_metric": "bits per original UTF-8 byte",
        },
        "dataset": {
            "snapshot": str(args.snapshot),
            "snapshot_sha256": _sha256(args.snapshot),
            "hub_sha": snapshot["hub_sha_at_snapshot"],
            "train_bytes_sha256": TRAIN_BYTES_SHA256,
            "validation_bytes_sha256": VALIDATION_BYTES_SHA256,
        },
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
            "bounded TinyStories experiment; optimizer, tokenizer, parameter, "
            "and hardware effects remain separately reported"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
