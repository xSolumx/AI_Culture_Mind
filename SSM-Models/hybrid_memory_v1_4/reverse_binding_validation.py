"""Run the frozen G6 v1.4.4 external reverse-binding validation."""

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
    from .identity_validation import CURRICULUM
    from .learnability_screen import CurriculumPhase
    from .successor_screen import _tied_identity_config
    from .upstream_learning_comparison import _build_model, _evaluate, _train_model
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4.identity_validation import (  # type: ignore[no-redef]
        CURRICULUM,
    )
    from hybrid_memory_v1_4.learnability_screen import (  # type: ignore[no-redef]
        CurriculumPhase,
    )
    from hybrid_memory_v1_4.successor_screen import (  # type: ignore[no-redef]
        _tied_identity_config,
    )
    from hybrid_memory_v1_4.upstream_learning_comparison import (  # type: ignore[no-redef]
        _build_model,
        _evaluate,
        _train_model,
    )

PREREGISTRATION = Path(__file__).with_name("G6_PREREGISTRATION.md")
VALIDATION_SEEDS = (1643, 1657, 1663)
MODEL_NAME = "hybrid_v1_4_4"


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
    for model_seed in VALIDATION_SEEDS:
        torch.manual_seed(model_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(model_seed)
        model_started = time.perf_counter()
        model = _build_model(MODEL_NAME, device)
        traces = _train_model(
            MODEL_NAME,
            model,
            device=device,
            data_seed=model_seed + 100_000,
            association_target="reverse_key",
        )
        evaluations = [
            _evaluate(
                MODEL_NAME,
                model,
                CurriculumPhase(16, 16, 96, 0),
                seed_base=model_seed + 2_000_000,
                device=device,
            ),
            _evaluate(
                MODEL_NAME,
                model,
                CurriculumPhase(16, 4, 512, 0),
                seed_base=model_seed + 2_100_000,
                device=device,
            ),
        ]
        checkpoint = args.checkpoint_dir / f"g6_hybrid_v1_4_4_seed{model_seed}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_name": MODEL_NAME,
                "model_config": asdict(_tied_identity_config()),
                "model_seed": model_seed,
                "data_seed": model_seed + 100_000,
                "association_target": "reverse_key",
                "preregistration_sha256": _sha256(PREREGISTRATION),
            },
            checkpoint,
        )
        runs.append(
            {
                "model_seed": model_seed,
                "data_seed": model_seed + 100_000,
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "phase_traces": traces,
                "evaluations": evaluations,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "elapsed_wall_seconds": time.perf_counter() - model_started,
            }
        )
        del model
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
            "validated external reverse-binding learning"
            if passed
            else "failed external reverse-binding validation"
        ),
        "passed": passed,
        "gate": "every fresh seed has exact query accuracy >= 0.90 at L96 and L512",
        "model_name": MODEL_NAME,
        "validation_seeds": list(VALIDATION_SEEDS),
        "config": asdict(_tied_identity_config()),
        "curriculum": [asdict(phase) for phase in CURRICULUM],
        "batch_size": 32,
        "learning_rate": 3e-3,
        "weight_decay": 0.01,
        "association_target": "reverse_key",
        "association_coefficient": 0.25,
        "internal_memory_labels": False,
        "retrieval_labels_per_seed": sum(
            phase.updates * 32 * phase.queries for phase in CURRICULUM
        ),
        "association_labels_per_seed": sum(
            phase.updates * 32 * phase.pairs for phase in CURRICULUM
        ),
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
            "external causal synthetic retrieval and reverse-binding labels; not "
            "ordinary label-free next-token learning or natural-language quality"
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
