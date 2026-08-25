"""Train the prospectively frozen G16 SM75 small-model frontier cohort."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import torch
from pure_spin_ssm_v1_2.mamba2_baseline import OfficialMamba2LM
from torch import nn
from torch.nn import functional as F

from .baselines import build_baseline
from .long_context_curriculum import (
    CONTEXT_LENGTHS,
    FORWARD_CHUNK_SIZE,
    PHASE_UPDATES,
    TARGETS_PER_UPDATE,
    _macro_batch,
)
from .long_context_recall import _ascii_filler, _recall_pair
from .model import HybridMemoryConfig, HybridMemoryLM
from .natural_text_frontier import (
    GRADIENT_CLIP,
    LEARNING_RATE,
    WEIGHT_DECAY,
    _optimizer_report,
    _sha256,
    _snapshot_text,
)
from .optimizers import build_optimizer
from .tokenization import ByteLevelBPETokenizer, EncodedText, tokenizer_fingerprint

PREREGISTRATION = Path(__file__).with_name(
    "G16_SM75_FRONTIER_SHOOTOUT_PROTOCOL_2026-08-25.md"
)
MODEL_SEED = 2333
ARMS = (
    "hybrid_v1_4_5",
    "hybrid_gdn2",
    "mamba2",
    "olmo_hybrid",
)
EXPECTED_PARAMETERS = {
    "hybrid_v1_4_5": 124_534,
    "hybrid_gdn2": 124_414,
    "mamba2": 124_172,
    "olmo_hybrid": 124_376,
}
TARGET_PARAMETERS = 124_534
MAX_PARAMETER_RESIDUAL = 0.01
EVAL_MACRO_BATCHES = 4
MONITOR_LENGTHS = (256, 4096)
RECALL_DISTANCES = (512, 2048, 8192)
RECALL_EXAMPLES = 8
TOKENIZER_SHA256 = "6b8031ebb2c899eaa780ee9216b2feddd00fac0a04265a6f3a5111a8dbda8ee1"
MAMBA_VERSION = "2.3.2.post1"


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


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


def _resolve_recorded_path(recorded: str) -> Path:
    """Resolve a repository-relative path recorded by Windows or POSIX."""

    normalized = Path(recorded.replace("\\", "/"))
    if normalized.is_absolute():
        return normalized
    return Path(__file__).resolve().parents[2] / normalized


def _load_tokenizer(audit_path: Path) -> ByteLevelBPETokenizer:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    selection = audit["selection_rule"]
    path = _resolve_recorded_path(selection["selected_path"])
    tokenizer = ByteLevelBPETokenizer.from_serialized(path.read_text(encoding="utf-8"))
    fingerprint = tokenizer_fingerprint(tokenizer)
    if fingerprint != TOKENIZER_SHA256 or fingerprint != selection["selected_sha256"]:
        raise RuntimeError("G16 tokenizer fingerprint mismatch")
    if tokenizer.vocab_size != 512:
        raise RuntimeError("G16 requires the frozen 512-token BPE")
    return tokenizer


def _mamba_source_provenance(source_root: Path) -> dict[str, Any]:
    distribution = importlib.metadata.distribution("mamba-ssm")
    if distribution.version != MAMBA_VERSION:
        raise RuntimeError(
            f"G16 requires mamba-ssm {MAMBA_VERSION}, got {distribution.version}"
        )
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError("mamba-ssm has no direct_url.json provenance")
    direct_url = json.loads(direct_url_text)
    parsed = urlparse(direct_url.get("url", ""))
    if parsed.scheme != "file":
        raise RuntimeError("mamba-ssm is not installed from a local source tree")
    installed_from = Path(unquote(parsed.path)).resolve()
    expected = source_root.resolve()
    if installed_from != expected:
        raise RuntimeError(
            f"mamba-ssm was installed from {installed_from}, not {expected}"
        )
    revision = subprocess.run(
        ["git", "-C", str(expected), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "distribution": "mamba-ssm",
        "version": distribution.version,
        "direct_url": direct_url,
        "source_root": str(expected),
        "source_revision": revision,
    }


def _hybrid_config(arm: str) -> HybridMemoryConfig:
    common: dict[str, Any] = {
        "vocab_size": 512,
        "model_dim": 48,
        "attention_heads": 4,
        "attention_window_size": 1024,
        "gated_delta_heads": 4,
        "gated_delta_value_dim": 12,
        "gated_delta_allow_negative_eigenvalues": False,
        "gated_delta_normalize_values": True,
        "gated_delta_identity_value_path": True,
        "gated_delta_identity_output_gate": True,
        "gated_delta_tie_query_key": True,
        "gated_delta_residual_scale_init": 0.0,
        "gated_delta_minimum_retention": 0.999,
        "gated_delta_initial_retention": 0.9995,
        "gated_delta_initial_erase_strength": 0.1,
        "gated_delta_initial_write_strength": 0.1,
        "use_local_conv": True,
        "conv_kernel": 4,
        "dropout": 0.0,
        "tie_embeddings": True,
    }
    if arm == "hybrid_v1_4_5":
        return HybridMemoryConfig(
            expansion=5,
            gated_delta_key_dim=24,
            **common,
        )
    if arm == "hybrid_gdn2":
        return HybridMemoryConfig(
            expansion=4,
            layer_plan=("gated_delta_v2", "attention"),
            gated_delta_key_dim=28,
            **common,
        )
    raise ValueError(f"unknown hybrid arm {arm!r}")


def _mamba2_kwargs() -> dict[str, int]:
    """Return the exact official fused Mamba-2 shape qualified on SM75."""

    return {
        "vocab_size": 512,
        "d_model": 56,
        "num_layers": 4,
        "d_state": 32,
        "expand": 2,
        "headdim": 16,
    }


def _olmo_kwargs() -> dict[str, Any]:
    """Return the parameter-matched actual Transformers OLMo Hybrid shape."""

    return {
        "vocab_size": 512,
        "hidden_size": 64,
        "intermediate_size": 138,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": True,
        "use_cache": False,
        "pad_token_id": 0,
        "eos_token_id": 1,
    }


def build_model(arm: str, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    if arm.startswith("hybrid_"):
        config = _hybrid_config(arm)
        model: nn.Module = HybridMemoryLM(config).to(device)
        metadata = {"family": "hybrid_memory_v1_4", "config": asdict(config)}
    elif arm == "mamba2":
        config = _mamba2_kwargs()
        model = OfficialMamba2LM(**config).to(device=device, dtype=torch.float32)
        metadata = {
            "family": "mamba_ssm",
            "runtime_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "config": config,
        }
    elif arm == "olmo_hybrid":
        config = _olmo_kwargs()
        model = build_baseline(
            "transformers_olmo_hybrid",
            device=device,
            dtype=torch.float32,
            **config,
        )
        metadata = {
            "family": "transformers_olmo_hybrid",
            "runtime_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "transformers_version": _installed_version("transformers"),
            "fla_version": _installed_version("flash-linear-attention"),
            "config": config,
        }
    else:
        raise ValueError(f"unknown G16 arm {arm!r}")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != EXPECTED_PARAMETERS[arm]:
        raise RuntimeError(
            f"{arm} parameter count changed: {parameters} != {EXPECTED_PARAMETERS[arm]}"
        )
    metadata["parameter_count"] = parameters
    metadata["parameter_residual"] = (
        parameters - TARGET_PARAMETERS
    ) / TARGET_PARAMETERS
    return model, metadata


def _forward_logits(arm: str, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    if arm.startswith("hybrid_"):
        assert isinstance(model, HybridMemoryLM)
        states = None
        pieces = []
        for start in range(0, inputs.shape[1], FORWARD_CHUNK_SIZE):
            output = model(
                inputs[:, start : start + FORWARD_CHUNK_SIZE],
                states,
                delta_scan_mode="parallel",
            )
            pieces.append(output["logits"])
            states = output["states"]
        return torch.cat(pieces, dim=1)
    if arm == "mamba2":
        return model(inputs)["logits"]  # type: ignore[index,no-any-return]
    if arm == "olmo_hybrid":
        return model(inputs, use_cache=False).logits  # type: ignore[operator,no-any-return]
    raise ValueError(f"unknown G16 arm {arm!r}")


def _build_optimizer(
    arm: str, model: nn.Module
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    if arm.startswith("hybrid_"):
        optimizer = build_optimizer(
            model,
            "harmonic_muon_adamw",
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        return optimizer, _optimizer_report(optimizer)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    return optimizer, {
        "name": "adamw",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "parameter_tensors": sum(1 for _ in model.parameters()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


@torch.inference_mode()
def evaluate_ordinary(
    arm: str,
    model: nn.Module,
    validation: EncodedText,
    *,
    lengths: tuple[int, ...],
    device: torch.device,
    macro_batches: int = EVAL_MACRO_BATCHES,
) -> list[dict[str, Any]]:
    model.eval()
    rows = []
    for length in lengths:
        nll_sum = 0.0
        raw_bytes = 0
        correct = 0
        token_count = 0
        for macro_index in range(macro_batches):
            inputs, targets, byte_lengths, _ = _macro_batch(
                validation,
                namespace="g16:validation",
                macro_index=macro_index,
                sequence_length=length,
                device=device,
            )
            logits = _forward_logits(arm, model, inputs)
            losses = F.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), reduction="sum"
            )
            nll_sum += float(losses)
            raw_bytes += int(byte_lengths.sum())
            correct += int((logits.argmax(-1) == targets).sum())
            token_count += targets.numel()
        rows.append(
            {
                "sequence_length_tokens": length,
                "batch_size": TARGETS_PER_UPDATE // length,
                "macro_batches": macro_batches,
                "scored_tokens": token_count,
                "scored_raw_bytes": raw_bytes,
                "bits_per_raw_byte": nll_sum / raw_bytes / math.log(2.0),
                "mean_nats_per_token": nll_sum / token_count,
                "next_token_accuracy": correct / token_count,
                "finite": math.isfinite(nll_sum),
            }
        )
    return rows


def _continuation_score(
    arm: str,
    model: nn.Module,
    tokenizer: ByteLevelBPETokenizer,
    prompt: str,
    continuation: str,
    device: torch.device,
) -> tuple[float, int, int]:
    prompt_ids = tokenizer.encode(prompt).token_ids
    full_ids = tokenizer.encode(prompt + continuation).token_ids
    if full_ids[: prompt_ids.numel()].tolist() != prompt_ids.tolist():
        raise RuntimeError("tokenizer continuation does not preserve prompt prefix")
    continuation_ids = full_ids[prompt_ids.numel() :]
    if not continuation_ids.numel():
        raise RuntimeError("recall continuation encoded to no tokens")
    inputs = full_ids[:-1].unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        logits = _forward_logits(arm, model, inputs)[0]
        start = prompt_ids.numel() - 1
        continuation_logits = logits[start : start + continuation_ids.numel()]
        selected = F.log_softmax(continuation_logits.float(), dim=-1).gather(
            -1, continuation_ids.to(device).unsqueeze(-1)
        )
    return float(selected.sum()), prompt_ids.numel(), continuation_ids.numel()


def evaluate_recall(
    arm: str,
    model: nn.Module,
    tokenizer: ByteLevelBPETokenizer,
    filler_source: str,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    for distance in RECALL_DISTANCES:
        for example in range(RECALL_EXAMPLES):
            pair = _recall_pair(filler_source, distance=distance, example=example)
            matching, matching_tokens, continuation_tokens = _continuation_score(
                arm,
                model,
                tokenizer,
                str(pair["matching_prompt"]),
                str(pair["continuation"]),
                device,
            )
            mismatched, mismatched_tokens, mismatch_tokens = _continuation_score(
                arm,
                model,
                tokenizer,
                str(pair["mismatched_prompt"]),
                str(pair["continuation"]),
                device,
            )
            if continuation_tokens != mismatch_tokens:
                raise RuntimeError("counterfactual continuation token counts differ")
            rows.append(
                {
                    "distance_raw_bytes": distance,
                    "example": example,
                    "name": pair["name"],
                    "supported_value": pair["supported_value"],
                    "counterfactual_value": pair["counterfactual_value"],
                    "filler_offset": pair["filler_offset"],
                    "matching_prompt_tokens": matching_tokens,
                    "mismatched_prompt_tokens": mismatched_tokens,
                    "continuation_tokens": continuation_tokens,
                    "matching_log_probability_nats": matching,
                    "mismatched_log_probability_nats": mismatched,
                    "matching_minus_mismatched_nats": matching - mismatched,
                }
            )
    return rows


def _train_arm(
    arm: str,
    *,
    train: EncodedText,
    validation: EncodedText,
    tokenizer: ByteLevelBPETokenizer,
    filler_source: str,
    checkpoint_dir: Path,
    device: torch.device,
    phase_updates: int,
    eval_macro_batches: int,
) -> dict[str, Any]:
    torch.manual_seed(MODEL_SEED)
    torch.cuda.manual_seed_all(MODEL_SEED)
    model, model_metadata = build_model(arm, device)
    optimizer, optimizer_metadata = _build_optimizer(arm, model)
    curve = [
        {
            "update": 0,
            "completed_phase": 0,
            "cumulative_training_tokens": 0,
            "cumulative_training_raw_bytes": 0,
            "ordinary_evaluation": evaluate_ordinary(
                arm,
                model,
                validation,
                lengths=MONITOR_LENGTHS,
                device=device,
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
    last_loss = float("nan")
    last_gradient_norm = float("nan")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for phase_index, length in enumerate(CONTEXT_LENGTHS, start=1):
        phase_times = []
        phase_started = time.perf_counter()
        first_update = (phase_index - 1) * phase_updates + 1
        last_update = phase_index * phase_updates
        for update in range(first_update, last_update + 1):
            step_started = time.perf_counter()
            inputs, targets, byte_lengths, macro_start = _macro_batch(
                train,
                namespace=f"g16:training:{MODEL_SEED}",
                macro_index=update - 1,
                sequence_length=length,
                device=device,
            )
            expected_targets = train.token_ids[
                macro_start + 1 : macro_start + TARGETS_PER_UPDATE + 1
            ]
            target_digest.update(expected_targets.numpy().tobytes())
            optimizer.zero_grad(set_to_none=True)
            logits = _forward_logits(arm, model, inputs)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite G16 loss for {arm} update {update}"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRADIENT_CLIP
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError(
                    f"non-finite G16 gradient for {arm} update {update}"
                )
            optimizer.step()
            torch.cuda.synchronize(device)
            training_peak = max(training_peak, torch.cuda.max_memory_allocated(device))
            elapsed = time.perf_counter() - step_started
            phase_times.append(elapsed)
            update_times.append(elapsed)
            cumulative_tokens += targets.numel()
            cumulative_raw_bytes += int(byte_lengths.sum())
            last_loss = float(loss.detach())
            last_gradient_norm = float(gradient_norm)
        phase_systems.append(
            {
                "phase": phase_index,
                "sequence_length_tokens": length,
                "batch_size": TARGETS_PER_UPDATE // length,
                "updates": phase_updates,
                "wall_seconds": time.perf_counter() - phase_started,
                "median_update_wall_seconds_after_10_warmups": statistics.median(
                    phase_times[min(10, max(0, len(phase_times) - 1)) :]
                ),
            }
        )
        torch.cuda.reset_peak_memory_stats(device)
        evaluation_lengths = (
            CONTEXT_LENGTHS if phase_index == len(CONTEXT_LENGTHS) else MONITOR_LENGTHS
        )
        ordinary = evaluate_ordinary(
            arm,
            model,
            validation,
            lengths=evaluation_lengths,
            device=device,
            macro_batches=eval_macro_batches,
        )
        torch.cuda.synchronize(device)
        evaluation_peak = max(evaluation_peak, torch.cuda.max_memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
        curve.append(
            {
                "update": last_update,
                "completed_phase": phase_index,
                "phase_sequence_length_tokens": length,
                "training_loss_nats_per_token": last_loss,
                "pre_clip_gradient_norm": last_gradient_norm,
                "cumulative_training_tokens": cumulative_tokens,
                "cumulative_training_raw_bytes": cumulative_raw_bytes,
                "ordinary_evaluation": ordinary,
            }
        )
        model.train()
    recall = evaluate_recall(arm, model, tokenizer, filler_source, device=device)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"g16_{arm}_seed{MODEL_SEED}.pt"
    torch.save(
        {
            "schema_version": 1,
            "stage": "G16",
            "arm_name": arm,
            "seed": MODEL_SEED,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model": model_metadata,
            "optimizer": optimizer_metadata,
            "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
            "phase_lengths": list(CONTEXT_LENGTHS),
            "phase_updates": phase_updates,
            "targets_per_update": TARGETS_PER_UPDATE,
            "training_target_ids_sha256": target_digest.hexdigest(),
        },
        checkpoint,
    )
    return {
        "arm_name": arm,
        "seed": MODEL_SEED,
        "model": model_metadata,
        "optimizer": optimizer_metadata,
        "phase_lengths": list(CONTEXT_LENGTHS),
        "phase_updates": phase_updates,
        "learning_curve": curve,
        "recall": recall,
        "training_target_ids_sha256": target_digest.hexdigest(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "systems": {
            "phase_systems": phase_systems,
            "training_wall_seconds_excluding_evaluation": sum(update_times),
            "elapsed_wall_seconds_including_evaluation": time.perf_counter() - started,
            "median_update_wall_seconds_after_50_warmups": statistics.median(
                update_times[min(50, max(0, len(update_times) - 1)) :]
            ),
            "peak_training_cuda_allocated_bytes": training_peak,
            "peak_evaluation_cuda_allocated_bytes": evaluation_peak,
        },
    }


def _evaluation(run: dict[str, Any], length: int) -> dict[str, Any]:
    return next(
        row
        for row in run["learning_curve"][-1]["ordinary_evaluation"]
        if row["sequence_length_tokens"] == length
    )


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    reference = next(run for run in runs if run["arm_name"] == "hybrid_v1_4_5")
    reference_4096 = _evaluation(reference, 4096)["bits_per_raw_byte"]
    digests = {run["training_target_ids_sha256"] for run in runs}
    rows = []
    finite = True
    parameter_gate = True
    for run in runs:
        final = {length: _evaluation(run, length) for length in CONTEXT_LENGTHS}
        recall_8192 = [
            row["matching_minus_mismatched_nats"]
            for row in run["recall"]
            if row["distance_raw_bytes"] == 8192
        ]
        recall_mean = statistics.mean(recall_8192)
        residual = abs(run["model"]["parameter_residual"])
        run_finite = all(row["finite"] for row in final.values()) and all(
            math.isfinite(row["matching_minus_mismatched_nats"])
            for row in run["recall"]
        )
        qualified = (
            run_finite
            and final[256]["bits_per_raw_byte"] <= 2.0
            and final[4096]["bits_per_raw_byte"] <= 2.0
            and final[4096]["bits_per_raw_byte"] - reference_4096 <= 0.10
            and math.isfinite(recall_mean)
        )
        rows.append(
            {
                "arm_name": run["arm_name"],
                "parameter_count": run["model"]["parameter_count"],
                "absolute_parameter_residual": residual,
                "final_bprb": {
                    str(length): final[length]["bits_per_raw_byte"]
                    for length in CONTEXT_LENGTHS
                },
                "bprb_4096_minus_hybrid_v1_4_5": (
                    final[4096]["bits_per_raw_byte"] - reference_4096
                ),
                "mean_recall_gain_nats_8192": recall_mean,
                "learned_recall_gate": recall_mean >= 0.02,
                "development_qualified": qualified,
                "finite": run_finite,
                "median_update_wall_seconds": run["systems"][
                    "median_update_wall_seconds_after_50_warmups"
                ],
                "peak_training_cuda_allocated_bytes": run["systems"][
                    "peak_training_cuda_allocated_bytes"
                ],
            }
        )
        finite = finite and run_finite
        parameter_gate = parameter_gate and residual <= MAX_PARAMETER_RESIDUAL
    return {
        "integrity": {
            "finite": finite,
            "parameter_gate": parameter_gate,
            "paired_training_target_digests": len(digests) == 1,
            "passed": finite and parameter_gate and len(digests) == 1,
        },
        "models": rows,
        "qualified_arms": [
            row["arm_name"] for row in rows if row["development_qualified"]
        ],
        "best_4096_bprb_arm": min(rows, key=lambda row: row["final_bprb"]["4096"])[
            "arm_name"
        ],
        "best_8192_recall_arm": max(
            rows, key=lambda row: row["mean_recall_gain_nats_8192"]
        )["arm_name"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--tokenizer-audit", type=Path, required=True)
    parser.add_argument("--mamba-source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase-updates", type=int, default=PHASE_UPDATES)
    parser.add_argument("--eval-macro-batches", type=int, default=EVAL_MACRO_BATCHES)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("G16 requires CUDA")
    if torch.cuda.get_device_capability(device) != (7, 5):
        parser.error("G16 is frozen to compute capability 7.5")
    if args.phase_updates < 1 or args.eval_macro_batches < 1:
        parser.error("phase/evaluation counts must be positive")
    if tuple(args.arms) != ARMS and args.phase_updates == PHASE_UPDATES:
        parser.error("the full G16 budget requires every frozen arm")
    git_commit, git_status = _git()
    mamba_provenance = _mamba_source_provenance(args.mamba_source_root)
    train_text, validation_text, snapshot = _snapshot_text(args.snapshot)
    tokenizer = _load_tokenizer(args.tokenizer_audit)
    train = tokenizer.encode(train_text)
    validation = tokenizer.encode(validation_text)
    filler_source = _ascii_filler(validation_text)
    started = time.perf_counter()
    runs = []
    for arm in args.arms:
        runs.append(
            _train_arm(
                arm,
                train=train,
                validation=validation,
                tokenizer=tokenizer,
                filler_source=filler_source,
                checkpoint_dir=args.checkpoint_dir,
                device=device,
                phase_updates=args.phase_updates,
                eval_macro_batches=args.eval_macro_batches,
            )
        )
        torch.cuda.empty_cache()
    full_protocol = (
        tuple(args.arms) == ARMS
        and args.phase_updates == PHASE_UPDATES
        and args.eval_macro_batches == EVAL_MACRO_BATCHES
    )
    report = {
        "schema_version": 1,
        "stage": "G16",
        "claim_status": (
            "completed prospectively frozen single-seed trained frontier shootout"
            if full_protocol
            else "non-adjudicating G16 smoke run"
        ),
        "adjudicating": full_protocol,
        "summary": _summary(runs),
        "runs": runs,
        "training": {
            "model_seed": MODEL_SEED,
            "context_lengths_tokens": list(CONTEXT_LENGTHS),
            "phase_updates": args.phase_updates,
            "total_updates": args.phase_updates * len(CONTEXT_LENGTHS),
            "targets_per_update": TARGETS_PER_UPDATE,
            "target_tokens_per_arm": (
                args.phase_updates * len(CONTEXT_LENGTHS) * TARGETS_PER_UPDATE
            ),
            "objective": "ordinary causal next-token cross entropy only",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
        },
        "tokenizer": {
            "name": tokenizer.name,
            "vocab_size": tokenizer.vocab_size,
            "sha256": tokenizer_fingerprint(tokenizer),
            "audit": str(args.tokenizer_audit),
            "audit_sha256": _sha256(args.tokenizer_audit),
        },
        "dataset": {
            "snapshot": str(args.snapshot),
            "snapshot_sha256": _sha256(args.snapshot),
            "hub_sha": snapshot["hub_sha_at_snapshot"],
        },
        "mamba_source": mamba_provenance,
        "preregistration": str(PREREGISTRATION),
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "git_commit_at_start": git_commit,
        "git_status_at_start": git_status,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        },
        "elapsed_wall_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "single-seed small-model TinyStories development cohort; parameter- and "
            "target-matched but model-specific-optimizer and not step-time matched"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if full_protocol and not report["summary"]["integrity"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
