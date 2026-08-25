"""Calibrate and train the frozen G12E measured-compute-matched BPE model."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

if __package__:
    from .model import HybridMemoryConfig, HybridMemoryLM
    from .natural_text_frontier import (
        LEARNING_RATE,
        VALIDATION_EVAL_UPDATES,
        VALIDATION_SEEDS,
        VALIDATION_UPDATES,
        WEIGHT_DECAY,
        _sha256,
        _snapshot_text,
        _train,
    )
    from .optimizers import build_optimizer
    from .successor_screen import _retention_safe_config
    from .tokenization import (
        ByteLevelBPETokenizer,
        tokenizer_fingerprint,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4.model import (  # type: ignore[no-redef]
        HybridMemoryConfig,
        HybridMemoryLM,
    )
    from hybrid_memory_v1_4.natural_text_frontier import (  # type: ignore[no-redef]
        LEARNING_RATE,
        VALIDATION_EVAL_UPDATES,
        VALIDATION_SEEDS,
        VALIDATION_UPDATES,
        WEIGHT_DECAY,
        _sha256,
        _snapshot_text,
        _train,
    )
    from hybrid_memory_v1_4.optimizers import (  # type: ignore[no-redef]
        build_optimizer,
    )
    from hybrid_memory_v1_4.successor_screen import (  # type: ignore[no-redef]
        _retention_safe_config,
    )
    from hybrid_memory_v1_4.tokenization import (  # type: ignore[no-redef]
        ByteLevelBPETokenizer,
        tokenizer_fingerprint,
    )


PREREGISTRATION = Path(__file__).with_name("G12E_PREREGISTRATION.md")
CALIBRATION_SEED = 1951
CALIBRATION_WARMUPS = 5
CALIBRATION_REPEATS = 15
CALIBRATION_BATCH_SIZE = 16
CALIBRATION_SEQUENCE_LENGTH = 256
CALIBRATION_WIDTHS = tuple(range(24, 97, 4))
CALIBRATION_EXPANSIONS = tuple(range(1, 7))
MAXIMUM_COMPUTE_RESIDUAL = 0.10


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


def _config(vocab_size: int, width: int, expansion: int) -> HybridMemoryConfig:
    return replace(
        _retention_safe_config(),
        vocab_size=vocab_size,
        model_dim=width,
        expansion=expansion,
        gated_delta_key_dim=width // 2,
        gated_delta_value_dim=width // 4,
    )


def _measure(
    config: HybridMemoryConfig,
    optimizer_name: str,
    *,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(CALIBRATION_SEED)
    torch.cuda.manual_seed_all(CALIBRATION_SEED)
    model = HybridMemoryLM(config).to(device)
    optimizer = build_optimizer(
        model,
        optimizer_name,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    generator = torch.Generator(device="cpu").manual_seed(CALIBRATION_SEED + 1)
    tokens = torch.randint(
        0,
        config.vocab_size,
        (CALIBRATION_BATCH_SIZE, CALIBRATION_SEQUENCE_LENGTH + 1),
        generator=generator,
    ).to(device)
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    times = []
    torch.cuda.reset_peak_memory_stats(device)
    for iteration in range(CALIBRATION_WARMUPS + CALIBRATION_REPEATS):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        logits = model(inputs, delta_scan_mode="parallel")["logits"]
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if iteration >= CALIBRATION_WARMUPS:
            times.append(elapsed)
    result = {
        "status": "measured",
        "vocab_size": config.vocab_size,
        "model_dim": config.model_dim,
        "expansion": config.expansion,
        "optimizer": optimizer_name,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "median_update_seconds": statistics.median(times),
        "minimum_update_seconds": min(times),
        "maximum_update_seconds": max(times),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }
    del model, optimizer, tokens, inputs, targets
    torch.cuda.empty_cache()
    return result


def _calibrate(device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_config = _config(256, 64, 2)
    target = _measure(target_config, "adamw", device=device)
    candidates = []
    for width in CALIBRATION_WIDTHS:
        for expansion in CALIBRATION_EXPANSIONS:
            try:
                config = _config(512, width, expansion)
            except ValueError as error:
                candidates.append(
                    {
                        "status": "invalid",
                        "model_dim": width,
                        "expansion": expansion,
                        "reason": str(error),
                    }
                )
                continue
            try:
                measurement = _measure(
                    config, "harmonic_muon_adamw", device=device
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                measurement = {
                    "status": "out_of_memory",
                    "model_dim": width,
                    "expansion": expansion,
                }
            candidates.append(measurement)
    measured = [row for row in candidates if row["status"] == "measured"]
    target_seconds = target["median_update_seconds"]
    selected = min(
        measured,
        key=lambda row: (
            abs(row["median_update_seconds"] - target_seconds),
            row["parameter_count"],
            -row["model_dim"],
            row["expansion"],
        ),
    )
    selected = {
        **selected,
        "target_median_update_seconds": target_seconds,
        "relative_compute_residual": (
            (selected["median_update_seconds"] - target_seconds) / target_seconds
        ),
    }
    return {"target": target, "selected": selected}, candidates


def _load_bpe(tokenizer_audit_path: Path) -> ByteLevelBPETokenizer:
    audit = json.loads(tokenizer_audit_path.read_text(encoding="utf-8"))
    selection = audit["selection_rule"]
    tokenizer = ByteLevelBPETokenizer.from_serialized(
        Path(selection["selected_path"]).read_text(encoding="utf-8")
    )
    if tokenizer_fingerprint(tokenizer) != selection["selected_sha256"]:
        raise ValueError("G12E selected tokenizer hash mismatch")
    return tokenizer


def _train_selected(
    selection: dict[str, Any],
    *,
    train_text: str,
    validation_text: str,
    tokenizer: ByteLevelBPETokenizer,
    checkpoint_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    config = _config(512, selection["model_dim"], selection["expansion"])
    train = tokenizer.encode(train_text)
    validation = tokenizer.encode(validation_text)
    fingerprint = tokenizer_fingerprint(tokenizer)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in VALIDATION_SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = HybridMemoryLM(config).to(device)
        started = time.perf_counter()
        curve, optimizer, metadata = _train(
            model,
            train,
            validation,
            optimizer_name="harmonic_muon_adamw",
            namespace=f"g12c:{seed}:{fingerprint}",
            updates=VALIDATION_UPDATES,
            eval_updates=VALIDATION_EVAL_UPDATES,
            device=device,
        )
        checkpoint = checkpoint_dir / f"g12e_compute_matched_bpe_seed{seed}.pt"
        torch.save(
            {
                "schema_version": 1,
                "stage": "G12E",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": asdict(config),
                "seed": seed,
                "arm_name": "bpe_compute_matched",
                "optimizer_name": "harmonic_muon_adamw",
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
                "arm_name": "bpe_compute_matched",
                "optimizer_name": "harmonic_muon_adamw",
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "config": asdict(config),
                "tokenizer": {
                    "name": tokenizer.name,
                    "vocab_size": tokenizer.vocab_size,
                    "sha256": fingerprint,
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
        torch.cuda.empty_cache()
    return runs


def _summary(
    calibration: dict[str, Any],
    runs: list[dict[str, Any]],
    g12c: dict[str, Any],
) -> dict[str, Any]:
    control = [
        run for run in g12c["runs"] if run["arm_name"] == "raw_adamw_control"
    ]
    candidate_final = [
        run["learning_curve"][-1]["bits_per_raw_byte"] for run in runs
    ]
    control_final = [
        run["learning_curve"][-1]["bits_per_raw_byte"] for run in control
    ]
    candidate_times = [
        run["systems"]["median_update_wall_seconds_after_50_warmups"] for run in runs
    ]
    control_times = [
        run["systems"]["median_update_wall_seconds_after_50_warmups"]
        for run in control
    ]
    residual = calibration["selected"]["relative_compute_residual"]
    outcome_time_ratio = statistics.mean(candidate_times) / statistics.mean(
        control_times
    )
    finite = all(
        point["finite"] for run in runs for point in run["learning_curve"]
    )
    passed = (
        abs(residual) <= MAXIMUM_COMPUTE_RESIDUAL
        and finite
        and statistics.mean(candidate_final) < statistics.mean(control_final)
        and max(candidate_final) < max(control_final)
        and outcome_time_ratio <= 1.10
    )
    return {
        "passed_compute_pareto_gate": passed,
        "calibration_relative_compute_residual": residual,
        "maximum_absolute_calibration_residual": MAXIMUM_COMPUTE_RESIDUAL,
        "candidate_mean_final_bprb": statistics.mean(candidate_final),
        "candidate_worst_final_bprb": max(candidate_final),
        "control_mean_final_bprb": statistics.mean(control_final),
        "control_worst_final_bprb": max(control_final),
        "candidate_mean_median_update_seconds": statistics.mean(candidate_times),
        "control_mean_median_update_seconds": statistics.mean(control_times),
        "outcome_update_time_ratio": outcome_time_ratio,
        "candidate_mean_presented_raw_bytes": statistics.mean(
            run["learning_curve"][-1]["cumulative_training_raw_bytes"] for run in runs
        ),
        "control_mean_presented_raw_bytes": statistics.mean(
            run["learning_curve"][-1]["cumulative_training_raw_bytes"]
            for run in control
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--tokenizer-audit", type=Path, required=True)
    parser.add_argument("--g12c-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("G12E requires CUDA for measured-compute calibration")
    git_commit, git_status = _git()
    train_text, validation_text, snapshot = _snapshot_text(args.snapshot)
    tokenizer = _load_bpe(args.tokenizer_audit)
    g12c = json.loads(args.g12c_report.read_text(encoding="utf-8"))
    started = time.perf_counter()
    calibration, candidates = _calibrate(device)
    runs = _train_selected(
        calibration["selected"],
        train_text=train_text,
        validation_text=validation_text,
        tokenizer=tokenizer,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
    )
    summary = _summary(calibration, runs, g12c)
    output = {
        "schema_version": 1,
        "stage": "G12E",
        "claim_status": (
            "passed measured-compute-matched BPE Pareto gate"
            if summary["passed_compute_pareto_gate"]
            else "failed measured-compute-matched BPE Pareto gate"
        ),
        "summary": summary,
        "calibration": calibration,
        "calibration_candidates": candidates,
        "runs": runs,
        "tokenizer": {
            "name": tokenizer.name,
            "vocab_size": tokenizer.vocab_size,
            "sha256": tokenizer_fingerprint(tokenizer),
        },
        "dataset": {
            "snapshot": str(args.snapshot),
            "snapshot_sha256": _sha256(args.snapshot),
            "hub_sha": snapshot["hub_sha_at_snapshot"],
        },
        "inputs": {
            "tokenizer_audit": str(args.tokenizer_audit),
            "tokenizer_audit_sha256": _sha256(args.tokenizer_audit),
            "g12c_report": str(args.g12c_report),
            "g12c_report_sha256": _sha256(args.g12c_report),
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
            "device_name": torch.cuda.get_device_name(device),
        },
        "elapsed_wall_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "native RTX 2070 SUPER measured-compute allocation point; not a "
            "scaling exponent, hardware-general result, or architecture-only win"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(json.dumps({"calibration": calibration, "summary": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
