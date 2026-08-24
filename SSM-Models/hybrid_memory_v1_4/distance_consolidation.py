"""Run the frozen G8 length-512 continuation on every G7 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

if __package__:
    from . import tasks
    from .competence_validation import MODEL_NAME, VALIDATION_SEEDS
    from .learnability_screen import CurriculumPhase, _batch, _seed
    from .model import HybridMemoryConfig, HybridMemoryLM
    from .upstream_learning_comparison import (
        _evaluate,
        _forward_logits,
        externally_observable_losses,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4 import tasks  # type: ignore[no-redef]
    from hybrid_memory_v1_4.competence_validation import (  # type: ignore[no-redef]
        MODEL_NAME,
        VALIDATION_SEEDS,
    )
    from hybrid_memory_v1_4.learnability_screen import (  # type: ignore[no-redef]
        CurriculumPhase,
        _batch,
        _seed,
    )
    from hybrid_memory_v1_4.model import (  # type: ignore[no-redef]
        HybridMemoryConfig,
        HybridMemoryLM,
    )
    from hybrid_memory_v1_4.upstream_learning_comparison import (  # type: ignore[no-redef]
        _evaluate,
        _forward_logits,
        externally_observable_losses,
    )

PREREGISTRATION = Path(__file__).with_name("G8_PREREGISTRATION.md")
PHASE = CurriculumPhase(16, 4, 512, 600)


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


def _continue(
    model: HybridMemoryLM,
    optimizer: torch.optim.Optimizer,
    *,
    model_seed: int,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    retrieval_sum = 0.0
    association_sum = 0.0
    last_batch_accuracy = 0.0
    for step in range(PHASE.updates):
        batch = _batch(
            PHASE,
            16,
            seed=_seed("g8-distance-training", step, model_seed + 700_000),
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = _forward_logits(MODEL_NAME, model, batch.inputs)
        retrieval, association = externally_observable_losses(
            logits,
            batch,
            "reverse_key",
        )
        loss = retrieval + 0.25 * association
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite G8 distance-consolidation loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        predictions = tasks.gather_query_logits(logits, batch).argmax(-1)
        last_batch_accuracy = float((predictions == batch.targets).float().mean())
        retrieval_sum += float(retrieval.detach())
        association_sum += float(association.detach())
    return {
        "mean_retrieval_loss": retrieval_sum / PHASE.updates,
        "mean_association_reconstruction_loss": association_sum / PHASE.updates,
        "last_batch_accuracy": last_batch_accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    git_commit, git_status_start = _git()
    args.output_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runs = []
    for model_seed in VALIDATION_SEEDS:
        source = args.checkpoint_dir / f"g7_hybrid_v1_4_4_seed{model_seed}.pt"
        payload = torch.load(source, map_location="cpu", weights_only=False)
        if payload.get("model_seed") != model_seed:
            raise ValueError(f"checkpoint seed mismatch: {source}")
        config_payload = payload.get("model_config")
        optimizer_payload = payload.get("optimizer_state_dict")
        if not isinstance(config_payload, dict) or not isinstance(
            optimizer_payload, dict
        ):
            raise TypeError("G7 checkpoint must contain model and optimizer mappings")
        model = HybridMemoryLM(HybridMemoryConfig(**config_payload)).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-3,
            weight_decay=0.01,
        )
        optimizer.load_state_dict(optimizer_payload)
        for group in optimizer.param_groups:
            group["lr"] = 1e-3
        model_started = time.perf_counter()
        trace = _continue(
            model,
            optimizer,
            model_seed=model_seed,
            device=device,
        )
        evaluations = [
            _evaluate(
                MODEL_NAME,
                model,
                CurriculumPhase(16, 16, 96, 0),
                seed_base=model_seed + 2_200_000,
                device=device,
            ),
            _evaluate(
                MODEL_NAME,
                model,
                CurriculumPhase(16, 4, 512, 0),
                seed_base=model_seed + 2_300_000,
                device=device,
            ),
        ]
        checkpoint = (
            args.output_checkpoint_dir / f"g8_hybrid_v1_4_4_seed{model_seed}.pt"
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_name": MODEL_NAME,
                "model_config": config_payload,
                "model_seed": model_seed,
                "data_seed": model_seed + 700_000,
                "association_target": "reverse_key",
                "source_checkpoint_sha256": _sha256(source),
                "preregistration_sha256": _sha256(PREREGISTRATION),
            },
            checkpoint,
        )
        runs.append(
            {
                "model_seed": model_seed,
                "source_checkpoint": str(source),
                "source_checkpoint_sha256": _sha256(source),
                "training_seed": model_seed + 700_000,
                "training_trace": trace,
                "evaluations": evaluations,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "elapsed_wall_seconds": time.perf_counter() - model_started,
            }
        )
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    passed = all(
        evaluation["exact_accuracy"] >= 0.90
        for run in runs
        for evaluation in run["evaluations"]
    )
    report = {
        "schema_version": 1,
        "claim_status": (
            "validated target-distance external learning"
            if passed
            else "failed target-distance consolidation"
        ),
        "passed": passed,
        "gate": "every seed has exact query accuracy >= 0.90 at L96 and L512",
        "validation_seeds": list(VALIDATION_SEEDS),
        "phase": asdict(PHASE),
        "batch_size": 16,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "optimizer_state_restored": True,
        "association_target": "reverse_key",
        "association_coefficient": 0.25,
        "internal_memory_labels": False,
        "retrieval_labels_per_seed": PHASE.updates * 16 * PHASE.queries,
        "association_labels_per_seed": PHASE.updates * 16 * PHASE.pairs,
        "runs": runs,
        "aggregate": {
            str(length): {
                "mean_exact_accuracy": sum(
                    evaluation["exact_accuracy"]
                    for run in runs
                    for evaluation in run["evaluations"]
                    if evaluation["length"] == length
                )
                / len(runs),
                "minimum_exact_accuracy": min(
                    evaluation["exact_accuracy"]
                    for run in runs
                    for evaluation in run["evaluations"]
                    if evaluation["length"] == length
                ),
            }
            for length in (96, 512)
        },
        "preregistration": str(PREREGISTRATION),
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "git_commit_at_start": git_commit,
        "git_status_at_start": git_status_start,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
        "elapsed_wall_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "continuation of failed G7 checkpoints with external synthetic labels; "
            "not a fresh-from-scratch validation or natural-language claim"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(json.dumps(report["aggregate"], sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
