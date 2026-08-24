"""Run the frozen G5 actual-upstream MQAR learning comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from . import tasks
    from .baselines import build_baseline
    from .identity_validation import CURRICULUM
    from .learnability_screen import CurriculumPhase, _batch, _seed
    from .model import HybridMemoryLM
    from .successor_screen import _tied_identity_config
    from .tasks import RetrievalBatch
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4 import tasks  # type: ignore[no-redef]
    from hybrid_memory_v1_4.baselines import (  # type: ignore[no-redef]
        build_baseline,
    )
    from hybrid_memory_v1_4.identity_validation import (  # type: ignore[no-redef]
        CURRICULUM,
    )
    from hybrid_memory_v1_4.learnability_screen import (  # type: ignore[no-redef]
        CurriculumPhase,
        _batch,
        _seed,
    )
    from hybrid_memory_v1_4.model import (  # type: ignore[no-redef]
        HybridMemoryLM,
    )
    from hybrid_memory_v1_4.successor_screen import (  # type: ignore[no-redef]
        _tied_identity_config,
    )
    from hybrid_memory_v1_4.tasks import (  # type: ignore[no-redef]
        RetrievalBatch,
    )

PREREGISTRATION = Path(__file__).with_name("G5_PREREGISTRATION.md")
MODEL_SEED = 1601
DATA_SEED = 1661
WRITE_COEFFICIENT = 0.25
MODEL_NAMES = ("hybrid_v1_4_4", "transformers_mamba2", "transformers_olmo_hybrid")
AssociationTarget = Literal["unseen_value", "reverse_key"]


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


def _build_model(name: str, device: torch.device) -> nn.Module:
    if name == "hybrid_v1_4_4":
        return HybridMemoryLM(_tied_identity_config()).to(device)
    common = {
        "vocab_size": 197,
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
    raise ValueError(f"unknown model {name!r}")


def _forward_logits(name: str, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    if name == "hybrid_v1_4_4":
        output = model(inputs, delta_scan_mode="parallel")
        return output["logits"]
    output = model(input_ids=inputs, use_cache=False)
    return output.logits


def _gather_positions(logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return logits.gather(
        1,
        positions.unsqueeze(-1).expand(-1, -1, logits.shape[-1]),
    )


def externally_observable_losses(
    logits: torch.Tensor,
    batch: RetrievalBatch,
    association_target: AssociationTarget = "unseen_value",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return query retrieval and an externally observable association loss."""

    retrieval_logits = tasks.gather_query_logits(logits, batch)
    retrieval = F.cross_entropy(
        retrieval_logits.flatten(0, 1),
        batch.targets.flatten(),
    )
    if association_target == "unseen_value":
        positions = batch.metadata["stored_key_positions"]
        association_targets = batch.metadata["stored_values"]
    elif association_target == "reverse_key":
        positions = batch.metadata["stored_value_positions"]
        association_targets = batch.metadata["stored_keys"]
    else:
        raise ValueError(f"unknown association target {association_target!r}")
    if not isinstance(positions, torch.Tensor) or not isinstance(
        association_targets, torch.Tensor
    ):
        raise TypeError("MQAR metadata must expose tensor positions and targets")
    association_logits = _gather_positions(logits, positions)
    reconstruction = F.cross_entropy(
        association_logits.flatten(0, 1),
        association_targets.flatten(),
    )
    return retrieval, reconstruction


def _train_model(
    name: str,
    model: nn.Module,
    *,
    device: torch.device,
    data_seed: int,
    association_target: AssociationTarget,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    traces = []
    step_offset = 0
    model.train()
    for phase in CURRICULUM:
        retrieval_sum = 0.0
        reconstruction_sum = 0.0
        final_accuracy = 0.0
        phase_started = time.perf_counter()
        for local_step in range(phase.updates):
            step = step_offset + local_step
            batch = _batch(
                phase,
                32,
                seed=_seed("training", step, data_seed),
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = _forward_logits(name, model, batch.inputs)
            retrieval, reconstruction = externally_observable_losses(
                logits,
                batch,
                association_target,
            )
            loss = retrieval + WRITE_COEFFICIENT * reconstruction
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite {name} G5 loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            predictions = tasks.gather_query_logits(logits, batch).argmax(-1)
            final_accuracy = float((predictions == batch.targets).float().mean())
            retrieval_sum += float(retrieval.detach())
            reconstruction_sum += float(reconstruction.detach())
        traces.append(
            {
                "phase": asdict(phase),
                "mean_retrieval_loss": retrieval_sum / phase.updates,
                "mean_association_reconstruction_loss": reconstruction_sum
                / phase.updates,
                "last_batch_accuracy": final_accuracy,
                "elapsed_wall_seconds": time.perf_counter() - phase_started,
            }
        )
        step_offset += phase.updates
    return traces


@torch.no_grad()
def _evaluate(
    name: str,
    model: nn.Module,
    phase: CurriculumPhase,
    *,
    seed_base: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    correct = 0
    exact_sequences = 0
    query_count = 0
    sequence_count = 0
    nll = 0.0
    for index in range(16):
        batch = _batch(
            phase,
            32,
            seed=_seed(f"evaluation-{phase.length}", index, seed_base),
            device=device,
        )
        logits = _forward_logits(name, model, batch.inputs)
        query_logits = tasks.gather_query_logits(logits, batch)
        matches = query_logits.argmax(-1) == batch.targets
        correct += int(matches.sum())
        exact_sequences += int(matches.all(-1).sum())
        query_count += batch.targets.numel()
        sequence_count += batch.targets.shape[0]
        nll += float(
            F.cross_entropy(
                query_logits.flatten(0, 1),
                batch.targets.flatten(),
                reduction="sum",
            )
        )
    return {
        "length": phase.length,
        "pairs": phase.pairs,
        "queries_per_sequence": phase.queries,
        "query_count": query_count,
        "exact_accuracy": correct / query_count,
        "exact_sequence_accuracy": exact_sequences / sequence_count,
        "bits_per_query": nll / (query_count * math.log(2.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, default=MODEL_SEED)
    parser.add_argument("--data-seed", type=int, default=DATA_SEED)
    parser.add_argument("--eval-seed-l96", type=int, default=1_601_601)
    parser.add_argument("--eval-seed-l512", type=int, default=1_701_601)
    parser.add_argument(
        "--association-target",
        choices=("unseen_value", "reverse_key"),
        default="unseen_value",
    )
    parser.add_argument("--run-label", default="g5")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    git_commit, git_status_start = _git()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    started = time.perf_counter()
    for name in MODEL_NAMES:
        torch.manual_seed(args.model_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.model_seed)
        model_started = time.perf_counter()
        model = _build_model(name, device)
        traces = _train_model(
            name,
            model,
            device=device,
            data_seed=args.data_seed,
            association_target=args.association_target,
        )
        evaluations = [
            _evaluate(
                name,
                model,
                CurriculumPhase(16, 16, 96, 0),
                seed_base=args.eval_seed_l96,
                device=device,
            ),
            _evaluate(
                name,
                model,
                CurriculumPhase(16, 4, 512, 0),
                seed_base=args.eval_seed_l512,
                device=device,
            ),
        ]
        checkpoint = (
            args.checkpoint_dir / f"{args.run_label}_{name}_seed{args.model_seed}.pt"
        )
        model_config = (
            asdict(_tied_identity_config())
            if name == "hybrid_v1_4_4"
            else model.config.to_dict()
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_name": name,
                "model_config": model_config,
                "model_seed": args.model_seed,
                "data_seed": args.data_seed,
                "association_target": args.association_target,
                "preregistration_sha256": _sha256(PREREGISTRATION),
            },
            checkpoint,
        )
        runs.append(
            {
                "model_name": name,
                "model_type": type(model).__name__,
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
    report = {
        "schema_version": 1,
        "claim_status": "single-seed paired externally supervised comparison",
        "validation_claim": False,
        "model_seed": args.model_seed,
        "data_seed": args.data_seed,
        "evaluation_seed_l96": args.eval_seed_l96,
        "evaluation_seed_l512": args.eval_seed_l512,
        "association_target": args.association_target,
        "models": list(MODEL_NAMES),
        "curriculum": [asdict(phase) for phase in CURRICULUM],
        "batch_size": 32,
        "learning_rate": 3e-3,
        "weight_decay": 0.01,
        "association_reconstruction_coefficient": WRITE_COEFFICIENT,
        "retrieval_labels_per_model": sum(
            phase.updates * 32 * phase.queries for phase in CURRICULUM
        ),
        "association_labels_per_model": sum(
            phase.updates * 32 * phase.pairs for phase in CURRICULUM
        ),
        "runs": runs,
        "preregistration": str(PREREGISTRATION),
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "git_commit_at_start": git_commit,
        "git_status_at_start": git_status_start,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
        "elapsed_wall_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "one paired seed with externally observable synthetic labels; actual "
            "randomly initialized library architectures, not pretrained quality or "
            "multi-seed superiority"
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
                    str(row["length"]): row["exact_accuracy"]
                    for row in run["evaluations"]
                }
                for run in runs
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
