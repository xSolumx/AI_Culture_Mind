"""Run the prospectively frozen G10 retention-safe fresh validation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

if __package__:
    from .combined_validation import (
        COMBINED_SCHEDULE,
        _git,
        _sha256,
        _train_schedule,
        schedule_label_counts,
    )
    from .learnability_screen import CurriculumPhase
    from .successor_screen import _retention_safe_config
    from .upstream_learning_comparison import _build_model, _evaluate
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4.combined_validation import (  # type: ignore[no-redef]
        COMBINED_SCHEDULE,
        _git,
        _sha256,
        _train_schedule,
        schedule_label_counts,
    )
    from hybrid_memory_v1_4.learnability_screen import (  # type: ignore[no-redef]
        CurriculumPhase,
    )
    from hybrid_memory_v1_4.successor_screen import (  # type: ignore[no-redef]
        _retention_safe_config,
    )
    from hybrid_memory_v1_4.upstream_learning_comparison import (  # type: ignore[no-redef]
        _build_model,
        _evaluate,
    )

PREREGISTRATION = Path(__file__).with_name("G10_PREREGISTRATION.md")
VALIDATION_SEEDS = (1753, 1759, 1777)
MODEL_NAME = "hybrid_v1_4_5"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    git_commit, git_status_start = _git()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runs = []
    retrieval_labels, association_labels = schedule_label_counts()
    for model_seed in VALIDATION_SEEDS:
        torch.manual_seed(model_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(model_seed)
        model_started = time.perf_counter()
        model = _build_model(MODEL_NAME, device)
        traces, optimizer = _train_schedule(
            model,
            model_seed=model_seed,
            device=device,
            model_name=MODEL_NAME,
        )
        evaluations = [
            _evaluate(
                MODEL_NAME,
                model,
                CurriculumPhase(16, 16, 96, 0),
                seed_base=model_seed + 2_400_000,
                device=device,
            ),
            _evaluate(
                MODEL_NAME,
                model,
                CurriculumPhase(16, 4, 512, 0),
                seed_base=model_seed + 2_500_000,
                device=device,
            ),
        ]
        checkpoint = args.checkpoint_dir / f"g10_hybrid_v1_4_5_seed{model_seed}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_name": MODEL_NAME,
                "model_config": asdict(_retention_safe_config()),
                "model_seed": model_seed,
                "schedule": [asdict(phase) for phase in COMBINED_SCHEDULE],
                "association_target": "reverse_key",
                "preregistration_sha256": _sha256(PREREGISTRATION),
            },
            checkpoint,
        )
        runs.append(
            {
                "model_seed": model_seed,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "phase_traces": traces,
                "total_updates": sum(p.updates for p in COMBINED_SCHEDULE),
                "retrieval_labels": retrieval_labels,
                "association_labels": association_labels,
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
            "validated fresh retention-safe external-learning schedule"
            if passed
            else "failed fresh retention-safe external-learning validation"
        ),
        "passed": passed,
        "gate": "every fresh seed has exact query accuracy >= 0.90 at L96 and L512",
        "model_name": MODEL_NAME,
        "validation_seeds": list(VALIDATION_SEEDS),
        "exposed_development_seed_excluded": 1723,
        "config": asdict(_retention_safe_config()),
        "minimum_retention_survival_at_512": 0.999**512,
        "schedule": [asdict(phase) for phase in COMBINED_SCHEDULE],
        "total_updates_per_seed": sum(p.updates for p in COMBINED_SCHEDULE),
        "retrieval_labels_per_seed": retrieval_labels,
        "association_labels_per_seed": association_labels,
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "association_target": "reverse_key",
        "association_coefficient": 0.25,
        "internal_memory_labels": False,
        "optimizer_state_continuous": True,
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
            "fresh synthetic external causal-label learning; not ordinary "
            "next-token learning, natural-language quality, or speed"
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
