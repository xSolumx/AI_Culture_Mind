"""Run the prospectively frozen v1.4.3 G4e identity-path validation."""

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
    from .learnability_screen import CurriculumPhase, _train_steps, evaluate
    from .model import HybridMemoryLM, parameter_count
    from .successor_screen import _identity_config
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4.learnability_screen import (  # type: ignore[no-redef]
        CurriculumPhase,
        _train_steps,
        evaluate,
    )
    from hybrid_memory_v1_4.model import (  # type: ignore[no-redef]
        HybridMemoryLM,
        parameter_count,
    )
    from hybrid_memory_v1_4.successor_screen import (  # type: ignore[no-redef]
        _identity_config,
    )

PREREGISTRATION = Path(__file__).with_name("G4E_PREREGISTRATION.md")
VALIDATION_SEEDS = (1481, 1483, 1487)
CURRICULUM = (
    CurriculumPhase(2, 2, 16, 300),
    CurriculumPhase(4, 4, 24, 300),
    CurriculumPhase(8, 8, 48, 400),
    CurriculumPhase(16, 16, 96, 1200),
)


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
    started = time.perf_counter()
    runs = []
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for model_seed in VALIDATION_SEEDS:
        torch.manual_seed(model_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(model_seed)
        config = _identity_config()
        model = HybridMemoryLM(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
        traces = []
        step_offset = 0
        for phase in CURRICULUM:
            trace = _train_steps(
                model,
                optimizer,
                phase,
                batch_size=32,
                association_coefficient=0.25,
                intermediate_coefficient=0.50,
                seed_base=model_seed,
                step_offset=step_offset,
                device=device,
            )
            traces.append({"phase": asdict(phase), **trace})
            step_offset += phase.updates
        evaluations = [
            evaluate(
                model,
                length=96,
                pairs=16,
                queries=16,
                batch_size=32,
                batches=16,
                seed_base=model_seed + 1_000_000,
                device=device,
            ),
            evaluate(
                model,
                length=512,
                pairs=16,
                queries=4,
                batch_size=32,
                batches=16,
                seed_base=model_seed + 1_100_000,
                device=device,
            ),
        ]
        checkpoint = args.checkpoint_dir / f"hybrid_v1_4_3_g4e_seed{model_seed}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": asdict(config),
                "model_seed": model_seed,
                "preregistration_sha256": _sha256(PREREGISTRATION),
            },
            checkpoint,
        )
        runs.append(
            {
                "model_seed": model_seed,
                "parameter_count": parameter_count(model),
                "phase_traces": traces,
                "evaluations": [asdict(item) for item in evaluations],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
            }
        )
    passed = all(
        evaluation["exact_accuracy"] >= 0.90
        for run in runs
        for evaluation in run["evaluations"]
    )
    report = {
        "schema_version": 1,
        "claim_status": "validated label-supervised v1.4.3 identity path"
        if passed
        else "failed v1.4.3 identity-path validation",
        "passed": passed,
        "gate": "every fresh seed has exact query accuracy >= 0.90 at L96 and L512",
        "validation_seeds": list(VALIDATION_SEEDS),
        "config": asdict(_identity_config()),
        "curriculum": [asdict(item) for item in CURRICULUM],
        "batch_size": 32,
        "learning_rate": 3e-3,
        "weight_decay": 0.01,
        "association_coefficient": 0.25,
        "intermediate_retrieval_coefficient": 0.50,
        "label_supervised": True,
        "useful_query_labels_per_seed": sum(
            phase.updates * 32 * phase.queries for phase in CURRICULUM
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
