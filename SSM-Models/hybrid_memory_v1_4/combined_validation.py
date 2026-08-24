"""Run the frozen G9 complete external-learning schedule from scratch."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

if __package__:
    from . import tasks
    from .learnability_screen import CurriculumPhase, _batch, _seed
    from .successor_screen import _tied_identity_config
    from .upstream_learning_comparison import (
        _build_model,
        _evaluate,
        _forward_logits,
        externally_observable_losses,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4 import tasks  # type: ignore[no-redef]
    from hybrid_memory_v1_4.learnability_screen import (  # type: ignore[no-redef]
        CurriculumPhase,
        _batch,
        _seed,
    )
    from hybrid_memory_v1_4.successor_screen import (  # type: ignore[no-redef]
        _tied_identity_config,
    )
    from hybrid_memory_v1_4.upstream_learning_comparison import (  # type: ignore[no-redef]
        _build_model,
        _evaluate,
        _forward_logits,
        externally_observable_losses,
    )

PREREGISTRATION = Path(__file__).with_name("G9_PREREGISTRATION.md")
VALIDATION_SEEDS = (1721, 1723, 1733)
MODEL_NAME = "hybrid_v1_4_4"


@dataclass(frozen=True)
class FixedPhase:
    pairs: int
    queries: int
    length: int
    batch_size: int
    updates: int
    learning_rate: float
    seed_offset: int
    seed_label: str

    @property
    def curriculum_phase(self) -> CurriculumPhase:
        return CurriculumPhase(self.pairs, self.queries, self.length, self.updates)


COMBINED_SCHEDULE = (
    FixedPhase(2, 2, 16, 32, 1200, 3e-3, 100_000, "g9-base-training"),
    FixedPhase(4, 4, 24, 32, 1200, 3e-3, 100_000, "g9-base-training"),
    FixedPhase(8, 8, 48, 32, 1400, 3e-3, 100_000, "g9-base-training"),
    FixedPhase(16, 16, 96, 32, 1300, 3e-3, 100_000, "g9-base-training"),
    FixedPhase(16, 4, 512, 16, 600, 1e-3, 700_000, "g9-distance-training"),
)


def schedule_label_counts() -> tuple[int, int]:
    """Return retrieval and reverse-binding label counts for one G9 seed."""

    retrieval = sum(
        phase.updates * phase.batch_size * phase.queries for phase in COMBINED_SCHEDULE
    )
    association = sum(
        phase.updates * phase.batch_size * phase.pairs for phase in COMBINED_SCHEDULE
    )
    return retrieval, association


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


def _train_schedule(
    model: torch.nn.Module,
    *,
    model_seed: int,
    device: torch.device,
    model_name: str = MODEL_NAME,
) -> tuple[list[dict[str, Any]], torch.optim.Optimizer]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    traces = []
    base_step = 0
    model.train()
    for phase_index, phase in enumerate(COMBINED_SCHEDULE):
        for group in optimizer.param_groups:
            group["lr"] = phase.learning_rate
        retrieval_sum = 0.0
        association_sum = 0.0
        last_batch_accuracy = 0.0
        phase_started = time.perf_counter()
        for local_step in range(phase.updates):
            step_index = base_step + local_step if phase_index < 4 else local_step
            batch = _batch(
                phase.curriculum_phase,
                phase.batch_size,
                seed=_seed(
                    phase.seed_label,
                    step_index,
                    model_seed + phase.seed_offset,
                ),
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = _forward_logits(model_name, model, batch.inputs)
            retrieval, association = externally_observable_losses(
                logits,
                batch,
                "reverse_key",
            )
            loss = retrieval + 0.25 * association
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite G9 combined-schedule loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            predictions = tasks.gather_query_logits(logits, batch).argmax(-1)
            last_batch_accuracy = float((predictions == batch.targets).float().mean())
            retrieval_sum += float(retrieval.detach())
            association_sum += float(association.detach())
        traces.append(
            {
                "phase": asdict(phase),
                "mean_retrieval_loss": retrieval_sum / phase.updates,
                "mean_association_reconstruction_loss": association_sum / phase.updates,
                "last_batch_accuracy": last_batch_accuracy,
                "elapsed_wall_seconds": time.perf_counter() - phase_started,
            }
        )
        if phase_index < 4:
            base_step += phase.updates
    return traces, optimizer


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
        checkpoint = args.checkpoint_dir / f"g9_hybrid_v1_4_4_seed{model_seed}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_name": MODEL_NAME,
                "model_config": asdict(_tied_identity_config()),
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
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "phase_traces": traces,
                "total_updates": sum(phase.updates for phase in COMBINED_SCHEDULE),
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
            "validated fresh combined external-learning schedule"
            if passed
            else "failed fresh combined external-learning validation"
        ),
        "passed": passed,
        "gate": "every fresh seed has exact query accuracy >= 0.90 at L96 and L512",
        "model_name": MODEL_NAME,
        "validation_seeds": list(VALIDATION_SEEDS),
        "config": asdict(_tied_identity_config()),
        "schedule": [asdict(phase) for phase in COMBINED_SCHEDULE],
        "total_updates_per_seed": sum(phase.updates for phase in COMBINED_SCHEDULE),
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
            "fresh-from-scratch external causal synthetic learning; not ordinary "
            "label-free next-token learning or natural-language quality"
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
