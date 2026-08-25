"""Fresh-seed validation for the preregistered v1.4.1 G4b learning gate."""

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
    from .learnability_screen import DEFAULT_CURRICULUM, _train_steps, evaluate
    from .model import HybridMemoryConfig, HybridMemoryLM, parameter_count
    from .tasks import DEFAULT_VOCABULARY
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4.learnability_screen import (  # type: ignore[no-redef]
        DEFAULT_CURRICULUM,
        _train_steps,
        evaluate,
    )
    from hybrid_memory_v1_4.model import (  # type: ignore[no-redef]
        HybridMemoryConfig,
        HybridMemoryLM,
        parameter_count,
    )
    from hybrid_memory_v1_4.tasks import DEFAULT_VOCABULARY  # type: ignore[no-redef]

PREREGISTRATION = Path(__file__).with_name("G4B_PREREGISTRATION.md")
VALIDATION_SEEDS = (1423, 1427, 1429)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _config() -> HybridMemoryConfig:
    return HybridMemoryConfig(
        vocab_size=DEFAULT_VOCABULARY.vocab_size,
        model_dim=64,
        layer_plan=("gated_delta", "attention"),
        attention_heads=4,
        attention_window_size=1024,
        gated_delta_heads=4,
        gated_delta_key_dim=16,
        gated_delta_value_dim=16,
        gated_delta_minimum_retention=0.90,
        gated_delta_initial_retention=0.995,
        use_local_conv=True,
        conv_kernel=4,
        expansion=2,
        dropout=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--association-coefficient", type=float, default=0.25)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(VALIDATION_SEEDS))
    args = parser.parse_args()
    if tuple(args.seeds) != VALIDATION_SEEDS:
        parser.error(f"evidentiary validation seeds are frozen as {VALIDATION_SEEDS}")
    if not PREREGISTRATION.is_file():
        raise FileNotFoundError(PREREGISTRATION)
    device = torch.device(args.device)
    git_commit, git_status_start = _git()
    started = time.perf_counter()
    runs = []
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for model_seed in VALIDATION_SEEDS:
        torch.manual_seed(model_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(model_seed)
        config = _config()
        model = HybridMemoryLM(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=0.01
        )
        step_offset = 0
        phase_traces = []
        for phase in DEFAULT_CURRICULUM:
            trace = _train_steps(
                model,
                optimizer,
                phase,
                batch_size=args.batch_size,
                association_coefficient=args.association_coefficient,
                seed_base=model_seed,
                step_offset=step_offset,
                device=device,
            )
            phase_traces.append({"phase": asdict(phase), **trace})
            step_offset += phase.updates
        evaluations = [
            evaluate(
                model,
                length=96,
                pairs=16,
                queries=16,
                batch_size=args.eval_batch_size,
                batches=args.eval_batches,
                seed_base=model_seed + 100_000,
                device=device,
            ),
            evaluate(
                model,
                length=512,
                pairs=16,
                queries=4,
                batch_size=args.eval_batch_size,
                batches=args.eval_batches,
                seed_base=model_seed + 200_000,
                device=device,
            ),
        ]
        checkpoint = (
            args.checkpoint_dir / f"hybrid_v1_4_1_g4b_validation_seed{model_seed}.pt"
        )
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
                "phase_traces": phase_traces,
                "evaluations": [asdict(item) for item in evaluations],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
            }
        )
    length_512 = [
        evaluation
        for run in runs
        for evaluation in run["evaluations"]
        if evaluation["length"] == 512
    ]
    passed = all(item["exact_accuracy"] >= 0.90 for item in length_512)
    report = {
        "schema_version": 1,
        "claim_status": "validated label-supervised commissioning"
        if passed
        else "failed validation",
        "passed": passed,
        "gate": "every fresh model seed has length-512 exact query accuracy >= 0.90",
        "validation_seeds": list(VALIDATION_SEEDS),
        "config": asdict(_config()),
        "curriculum": [asdict(item) for item in DEFAULT_CURRICULUM],
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "association_coefficient": args.association_coefficient,
        "label_supervised_association": True,
        "useful_query_labels_per_seed": sum(
            phase.updates * args.batch_size * phase.queries
            for phase in DEFAULT_CURRICULUM
        ),
        "eval_query_count_per_length_per_seed": args.eval_batch_size
        * args.eval_batches
        * 4,
        "runs": runs,
        "aggregate_length_512": {
            "mean_exact_accuracy": sum(item["exact_accuracy"] for item in length_512)
            / len(length_512),
            "minimum_exact_accuracy": min(
                item["exact_accuracy"] for item in length_512
            ),
            "mean_bits_per_query": sum(item["bits_per_query"] for item in length_512)
            / len(length_512),
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
    print(json.dumps(report["aggregate_length_512"], sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
