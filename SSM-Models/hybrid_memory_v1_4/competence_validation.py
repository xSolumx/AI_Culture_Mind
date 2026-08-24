"""Run the frozen G7 competence-paced external-learning validation."""

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

PREREGISTRATION = Path(__file__).with_name("G7_PREREGISTRATION.md")
VALIDATION_SEEDS = (1693, 1697, 1699)
MODEL_NAME = "hybrid_v1_4_4"


@dataclass(frozen=True)
class PacedPhase:
    pairs: int
    queries: int
    length: int
    minimum_updates: int
    maximum_updates: int

    @property
    def curriculum_phase(self) -> CurriculumPhase:
        return CurriculumPhase(self.pairs, self.queries, self.length, 0)


PACED_CURRICULUM = (
    PacedPhase(2, 2, 16, 300, 1200),
    PacedPhase(4, 4, 24, 300, 1200),
    PacedPhase(8, 8, 48, 400, 1600),
    PacedPhase(16, 16, 96, 1200, 2400),
)


def two_consecutive_mastery(
    probe_accuracies: list[float], threshold: float = 0.90
) -> bool:
    """Return whether the latest two competence probes clear the threshold."""

    return (
        len(probe_accuracies) >= 2
        and probe_accuracies[-2] >= threshold
        and probe_accuracies[-1] >= threshold
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


@torch.no_grad()
def _competence_probe(
    model: torch.nn.Module,
    phase: PacedPhase,
    *,
    model_seed: int,
    phase_index: int,
    probe_index: int,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    count = 0
    for batch_index in range(4):
        batch = _batch(
            phase.curriculum_phase,
            32,
            seed=_seed(
                f"g7-competence-{phase_index}-{probe_index}",
                batch_index,
                model_seed + 500_000,
            ),
            device=device,
        )
        logits = _forward_logits(MODEL_NAME, model, batch.inputs)
        predictions = tasks.gather_query_logits(logits, batch).argmax(-1)
        correct += int((predictions == batch.targets).sum())
        count += batch.targets.numel()
    model.train()
    return correct / count


def _train_paced(
    model: torch.nn.Module,
    *,
    model_seed: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], torch.optim.Optimizer]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    traces = []
    global_step = 0
    model.train()
    for phase_index, phase in enumerate(PACED_CURRICULUM):
        retrieval_sum = 0.0
        association_sum = 0.0
        last_batch_accuracy = 0.0
        probes = []
        phase_started = time.perf_counter()
        updates_used = 0
        for local_step in range(phase.maximum_updates):
            batch = _batch(
                phase.curriculum_phase,
                32,
                seed=_seed("training", global_step, model_seed + 100_000),
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
                raise FloatingPointError("non-finite G7 competence-paced loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            predictions = tasks.gather_query_logits(logits, batch).argmax(-1)
            last_batch_accuracy = float((predictions == batch.targets).float().mean())
            retrieval_sum += float(retrieval.detach())
            association_sum += float(association.detach())
            global_step += 1
            updates_used = local_step + 1
            if updates_used >= phase.minimum_updates and updates_used % 100 == 0:
                accuracy = _competence_probe(
                    model,
                    phase,
                    model_seed=model_seed,
                    phase_index=phase_index,
                    probe_index=len(probes),
                    device=device,
                )
                probes.append(
                    {
                        "updates_used": updates_used,
                        "exact_accuracy": accuracy,
                    }
                )
                if two_consecutive_mastery(
                    [probe["exact_accuracy"] for probe in probes]
                ):
                    break
        traces.append(
            {
                "phase": asdict(phase),
                "updates_used": updates_used,
                "competence_achieved": two_consecutive_mastery(
                    [probe["exact_accuracy"] for probe in probes]
                ),
                "competence_probes": probes,
                "mean_retrieval_loss": retrieval_sum / updates_used,
                "mean_association_reconstruction_loss": association_sum / updates_used,
                "last_batch_accuracy": last_batch_accuracy,
                "elapsed_wall_seconds": time.perf_counter() - phase_started,
            }
        )
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
    for model_seed in VALIDATION_SEEDS:
        torch.manual_seed(model_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(model_seed)
        model_started = time.perf_counter()
        model = _build_model(MODEL_NAME, device)
        traces, optimizer = _train_paced(
            model,
            model_seed=model_seed,
            device=device,
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
        checkpoint = args.checkpoint_dir / f"g7_hybrid_v1_4_4_seed{model_seed}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_name": MODEL_NAME,
                "model_config": asdict(_tied_identity_config()),
                "model_seed": model_seed,
                "data_seed": model_seed + 100_000,
                "association_target": "reverse_key",
                "preregistration_sha256": _sha256(PREREGISTRATION),
            },
            checkpoint,
        )
        updates_used = [trace["updates_used"] for trace in traces]
        runs.append(
            {
                "model_seed": model_seed,
                "data_seed": model_seed + 100_000,
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "phase_traces": traces,
                "total_updates": sum(updates_used),
                "retrieval_labels": sum(
                    updates * 32 * phase.queries
                    for updates, phase in zip(
                        updates_used, PACED_CURRICULUM, strict=True
                    )
                ),
                "association_labels": sum(
                    updates * 32 * phase.pairs
                    for updates, phase in zip(
                        updates_used, PACED_CURRICULUM, strict=True
                    )
                ),
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
        all(trace["competence_achieved"] for trace in run["phase_traces"])
        and all(
            evaluation["exact_accuracy"] >= 0.90 for evaluation in run["evaluations"]
        )
        for run in runs
    )
    report = {
        "schema_version": 1,
        "claim_status": (
            "validated competence-paced external learning"
            if passed
            else "failed competence-paced external-learning validation"
        ),
        "passed": passed,
        "gate": (
            "every phase reaches two consecutive 0.90 competence probes and every "
            "fresh seed has exact query accuracy >= 0.90 at L96 and L512"
        ),
        "model_name": MODEL_NAME,
        "validation_seeds": list(VALIDATION_SEEDS),
        "config": asdict(_tied_identity_config()),
        "paced_curriculum": [asdict(phase) for phase in PACED_CURRICULUM],
        "competence_probe_batches": 4,
        "competence_check_interval": 100,
        "competence_threshold": 0.90,
        "consecutive_probes_required": 2,
        "batch_size": 32,
        "learning_rate": 3e-3,
        "weight_decay": 0.01,
        "association_target": "reverse_key",
        "association_coefficient": 0.25,
        "internal_memory_labels": False,
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
            "external causal synthetic labels under a capped variable budget; not "
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
