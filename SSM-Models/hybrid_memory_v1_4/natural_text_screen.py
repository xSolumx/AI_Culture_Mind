"""Run the prospectively frozen G11 TinyStories next-byte learning screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from .baselines import build_baseline
    from .model import HybridMemoryLM
    from .natural_text_data import rows_to_bytes
    from .successor_screen import _retention_safe_config
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4.baselines import build_baseline  # type: ignore[no-redef]
    from hybrid_memory_v1_4.model import (  # type: ignore[no-redef]
        HybridMemoryLM,
    )
    from hybrid_memory_v1_4.natural_text_data import (  # type: ignore[no-redef]
        rows_to_bytes,
    )
    from hybrid_memory_v1_4.successor_screen import (  # type: ignore[no-redef]
        _retention_safe_config,
    )

PREREGISTRATION = Path(__file__).with_name("G11_PREREGISTRATION.md")
SNAPSHOT_SHA256 = "ebe7c7c948f3e59781097ffa64e214da15364d61b38622d33dd076d40471adc6"
TRAIN_BYTES_SHA256 = "e25563b202a9f669b5d479d0cf94bd9a2c48a0050fcb569d63f1f6fc8580b4c1"
VALIDATION_BYTES_SHA256 = (
    "4b7aadf02f91e1dce13b699bd428bda3984ac4ee9ec17167db1813e0a34b1b53"
)
MODEL_NAMES = ("hybrid_v1_4_5", "transformers_mamba2", "transformers_olmo_hybrid")
MODEL_SEED = 1811
DATA_SEED = 1817
SEQUENCE_LENGTH = 256
TRAIN_BATCH_SIZE = 16
TRAIN_UPDATES = 2000
EVAL_BATCH_SIZE = 16
EVAL_BATCHES = 32
EVAL_UPDATES = (0, 500, 1000, 2000)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0


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


def _load_streams(snapshot: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if _sha256(snapshot) != SNAPSHOT_SHA256:
        raise ValueError("TinyStories snapshot hash does not match frozen G11 input")
    report = json.loads(snapshot.read_text(encoding="utf-8"))
    train_bytes = rows_to_bytes(report["train"]["rows"])
    validation_bytes = rows_to_bytes(report["validation"]["rows"])
    if hashlib.sha256(train_bytes).hexdigest() != TRAIN_BYTES_SHA256:
        raise ValueError("G11 training byte stream hash mismatch")
    if hashlib.sha256(validation_bytes).hexdigest() != VALIDATION_BYTES_SHA256:
        raise ValueError("G11 validation byte stream hash mismatch")
    train = torch.tensor(list(train_bytes), dtype=torch.long)
    validation = torch.tensor(list(validation_bytes), dtype=torch.long)
    return train, validation, report


def _offset(namespace: str, index: int, maximum: int) -> int:
    if maximum < 1:
        raise ValueError("byte stream is too short for the frozen sequence length")
    payload = f"{namespace}:{DATA_SEED}:{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % maximum


def _batch(
    stream: torch.Tensor,
    *,
    namespace: str,
    batch_index: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = stream.numel() - SEQUENCE_LENGTH
    windows = []
    for row in range(batch_size):
        index = batch_index * batch_size + row
        start = _offset(namespace, index, maximum)
        windows.append(stream[start : start + SEQUENCE_LENGTH + 1])
    tokens = torch.stack(windows).to(device)
    return tokens[:, :-1], tokens[:, 1:]


def _build_model(name: str, device: torch.device) -> nn.Module:
    if name == "hybrid_v1_4_5":
        config = replace(_retention_safe_config(), vocab_size=256)
        return HybridMemoryLM(config).to(device)
    common = {
        "vocab_size": 256,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "tie_word_embeddings": True,
        "use_cache": False,
        "pad_token_id": 0,
        "eos_token_id": 1,
    }
    if name == "transformers_mamba2":
        return build_baseline(
            name,
            device=device,
            dtype=torch.float32,
            **common,
            state_size=16,
            expand=2,
            head_dim=16,
            num_heads=8,
            n_groups=4,
            conv_kernel=4,
            chunk_size=64,
        )
    if name == "transformers_olmo_hybrid":
        return build_baseline(
            name,
            device=device,
            dtype=torch.float32,
            **common,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            layer_types=["linear_attention", "full_attention"],
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            max_position_embeddings=1024,
        )
    raise ValueError(f"unknown G11 model {name!r}")


def _logits(name: str, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    if name == "hybrid_v1_4_5":
        return model(inputs, delta_scan_mode="parallel")["logits"]
    return model(input_ids=inputs, use_cache=False).logits


def _evaluate(
    name: str,
    model: nn.Module,
    validation: torch.Tensor,
    device: torch.device,
) -> dict[str, float | int | bool]:
    loss_sum = 0.0
    correct = 0
    count = 0
    model.eval()
    with torch.inference_mode():
        for batch_index in range(EVAL_BATCHES):
            inputs, targets = _batch(
                validation,
                namespace="g11-validation",
                batch_index=batch_index,
                batch_size=EVAL_BATCH_SIZE,
                device=device,
            )
            logits = _logits(name, model, inputs)
            loss_sum += float(
                F.cross_entropy(
                    logits.flatten(0, 1),
                    targets.flatten(),
                    reduction="sum",
                )
            )
            correct += int((logits.argmax(-1) == targets).sum())
            count += targets.numel()
    nats = loss_sum / count
    return {
        "scored_bytes": count,
        "mean_nats_per_byte": nats,
        "bits_per_byte": nats / math.log(2.0),
        "next_byte_accuracy": correct / count,
        "finite": math.isfinite(nats),
    }


def _train_model(
    name: str,
    model: nn.Module,
    train: torch.Tensor,
    validation: torch.Tensor,
    device: torch.device,
) -> tuple[list[dict[str, Any]], torch.optim.Optimizer]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    curve = [{"update": 0, **_evaluate(name, model, validation, device)}]
    train_loss_since_eval = 0.0
    train_steps_since_eval = 0
    model.train()
    for update in range(1, TRAIN_UPDATES + 1):
        inputs, targets = _batch(
            train,
            namespace="g11-training",
            batch_index=update - 1,
            batch_size=TRAIN_BATCH_SIZE,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = _logits(name, model, inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite G11 loss for {name}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        train_loss_since_eval += float(loss.detach())
        train_steps_since_eval += 1
        if update in EVAL_UPDATES:
            evaluation = _evaluate(name, model, validation, device)
            curve.append(
                {
                    "update": update,
                    "mean_train_nats_since_previous_evaluation": (
                        train_loss_since_eval / train_steps_since_eval
                    ),
                    **evaluation,
                }
            )
            train_loss_since_eval = 0.0
            train_steps_since_eval = 0
            model.train()
    return curve, optimizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    git_commit, git_status_start = _git()
    train, validation, snapshot_report = _load_streams(args.snapshot)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runs = []
    for name in MODEL_NAMES:
        torch.manual_seed(MODEL_SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(MODEL_SEED)
        model_started = time.perf_counter()
        model = _build_model(name, device)
        curve, optimizer = _train_model(name, model, train, validation, device)
        checkpoint = args.checkpoint_dir / f"g11_{name}_seed{MODEL_SEED}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_name": name,
                "model_seed": MODEL_SEED,
                "training_protocol": {
                    "sequence_length": SEQUENCE_LENGTH,
                    "batch_size": TRAIN_BATCH_SIZE,
                    "updates": TRAIN_UPDATES,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                },
                "snapshot_sha256": SNAPSHOT_SHA256,
            },
            checkpoint,
        )
        runs.append(
            {
                "model_name": name,
                "runtime_class": f"{type(model).__module__}.{type(model).__name__}",
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "learning_curve": curve,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "elapsed_wall_seconds": time.perf_counter() - model_started,
            }
        )
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    candidate = next(run for run in runs if run["model_name"] == "hybrid_v1_4_5")
    initial_bpb = candidate["learning_curve"][0]["bits_per_byte"]
    final_bpb = candidate["learning_curve"][-1]["bits_per_byte"]
    finite = all(
        point["finite"]
        and math.isfinite(point["bits_per_byte"])
        and math.isfinite(point["mean_nats_per_byte"])
        for run in runs
        for point in run["learning_curve"]
    )
    improvement = initial_bpb - final_bpb
    passed = finite and improvement >= 2.0 and final_bpb <= 4.0
    try:
        import transformers

        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = None
    report = {
        "schema_version": 1,
        "claim_status": (
            "passed bounded real-text ordinary next-token learning screen"
            if passed
            else "failed bounded real-text ordinary next-token learning screen"
        ),
        "passed": passed,
        "gate": {
            "all_metrics_finite": finite,
            "candidate_bits_per_byte_improvement": improvement,
            "required_improvement": 2.0,
            "candidate_final_bits_per_byte": final_bpb,
            "maximum_final_bits_per_byte": 4.0,
        },
        "models": list(MODEL_NAMES),
        "model_seed": MODEL_SEED,
        "data_seed": DATA_SEED,
        "objective": "ordinary causal UTF-8 next-byte cross entropy only",
        "training": {
            "sequence_length": SEQUENCE_LENGTH,
            "batch_size": TRAIN_BATCH_SIZE,
            "updates": TRAIN_UPDATES,
            "scored_training_bytes_per_model": (
                SEQUENCE_LENGTH * TRAIN_BATCH_SIZE * TRAIN_UPDATES
            ),
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
        },
        "evaluation": {
            "updates": list(EVAL_UPDATES),
            "batch_size": EVAL_BATCH_SIZE,
            "batches": EVAL_BATCHES,
            "scored_bytes_per_evaluation": (
                SEQUENCE_LENGTH * EVAL_BATCH_SIZE * EVAL_BATCHES
            ),
        },
        "dataset": {
            "name": snapshot_report["dataset"],
            "hub_sha": snapshot_report["hub_sha_at_snapshot"],
            "license": snapshot_report["license"],
            "snapshot": str(args.snapshot),
            "snapshot_sha256": SNAPSHOT_SHA256,
            "train_bytes_sha256": TRAIN_BYTES_SHA256,
            "validation_bytes_sha256": VALIDATION_BYTES_SHA256,
            "exact_story_overlap_count": snapshot_report["exact_story_overlap_count"],
        },
        "runs": runs,
        "preregistration": str(PREREGISTRATION),
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "git_commit_at_start": git_commit,
        "git_status_at_start": git_status_start,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers_version,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
        "elapsed_wall_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "single-seed bounded TinyStories next-byte learning screen; not "
            "general language quality, scaling, superiority, or speed"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(
        json.dumps(
            {
                run["model_name"]: {
                    "initial_bpb": run["learning_curve"][0]["bits_per_byte"],
                    "final_bpb": run["learning_curve"][-1]["bits_per_byte"],
                }
                for run in runs
            },
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
