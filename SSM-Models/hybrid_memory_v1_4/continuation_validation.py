"""Run the prospectively frozen G4c consolidation on all G4b checkpoints."""

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
    from .model import HybridMemoryConfig, HybridMemoryLM
    from .validation_screen import VALIDATION_SEEDS
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4.learnability_screen import (  # type: ignore[no-redef]
        CurriculumPhase,
        _train_steps,
        evaluate,
    )
    from hybrid_memory_v1_4.model import (  # type: ignore[no-redef]
        HybridMemoryConfig,
        HybridMemoryLM,
    )
    from hybrid_memory_v1_4.validation_screen import (  # type: ignore[no-redef]
        VALIDATION_SEEDS,
    )

PREREGISTRATION = Path(__file__).with_name("G4C_PREREGISTRATION.md")


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
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if not PREREGISTRATION.is_file():
        raise FileNotFoundError(PREREGISTRATION)
    device = torch.device(args.device)
    git_commit, git_status_start = _git()
    phase = CurriculumPhase(16, 4, 512, 300)
    runs = []
    started = time.perf_counter()
    args.output_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for model_seed in VALIDATION_SEEDS:
        source = (
            args.checkpoint_dir / f"hybrid_v1_4_1_g4b_validation_seed{model_seed}.pt"
        )
        payload = torch.load(source, map_location="cpu", weights_only=False)
        if payload.get("model_seed") != model_seed:
            raise ValueError(f"checkpoint seed mismatch: {source}")
        model = HybridMemoryLM(HybridMemoryConfig(**payload["config"])).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        continuation_seed = model_seed + 300_000
        trace = _train_steps(
            model,
            optimizer,
            phase,
            batch_size=16,
            association_coefficient=0.25,
            seed_base=continuation_seed,
            step_offset=0,
            device=device,
        )
        evaluations = [
            evaluate(
                model,
                length=96,
                pairs=16,
                queries=16,
                batch_size=32,
                batches=16,
                seed_base=model_seed + 400_000,
                device=device,
            ),
            evaluate(
                model,
                length=512,
                pairs=16,
                queries=4,
                batch_size=32,
                batches=16,
                seed_base=model_seed + 500_000,
                device=device,
            ),
        ]
        checkpoint = (
            args.output_checkpoint_dir
            / f"hybrid_v1_4_1_g4c_validation_seed{model_seed}.pt"
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": payload["config"],
                "model_seed": model_seed,
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
                "continuation_seed_base": continuation_seed,
                "training_trace": trace,
                "evaluations": [asdict(item) for item in evaluations],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
            }
        )
    passed = all(
        evaluation["exact_accuracy"] >= 0.90
        for run in runs
        for evaluation in run["evaluations"]
        if evaluation["length"] in (96, 512)
    )
    report = {
        "schema_version": 1,
        "claim_status": "validated label-supervised consolidation"
        if passed
        else "failed consolidation validation",
        "passed": passed,
        "gate": "every seed has exact query accuracy >= 0.90 at L96 and L512",
        "validation_seeds": list(VALIDATION_SEEDS),
        "phase": asdict(phase),
        "batch_size": 16,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "association_coefficient": 0.25,
        "label_supervised_association": True,
        "useful_query_labels_per_seed": phase.updates * 16 * phase.queries,
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
