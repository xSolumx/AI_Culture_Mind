"""Commission the v1.4.1 learning problem before another large quality cohort.

The screen separates four questions that the original G4 run conflated:

1. can a model memorize one fixed batch (optimizer/gradient-shell control)?
2. can it learn fresh short associations (rule-learning control)?
3. does content-address supervision repair the address bottleneck?
4. does the learned rule transfer to the frozen 16-pair, length-512 MQAR task?

Association supervision is always reported explicitly. It is a commissioning
route, not a label-free result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

if __package__:
    from . import tasks
    from .experiments import (
        gated_delta_association_auxiliary_loss,
        intermediate_retrieval_auxiliary_loss,
    )
    from .model import HybridMemoryConfig, HybridMemoryLM, parameter_count
    from .tasks import DEFAULT_VOCABULARY, RetrievalBatch, generate_mqar_batch
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4 import tasks  # type: ignore[no-redef]
    from hybrid_memory_v1_4.experiments import (  # type: ignore[no-redef]
        gated_delta_association_auxiliary_loss,
        intermediate_retrieval_auxiliary_loss,
    )
    from hybrid_memory_v1_4.model import (  # type: ignore[no-redef]
        HybridMemoryConfig,
        HybridMemoryLM,
        parameter_count,
    )
    from hybrid_memory_v1_4.tasks import (  # type: ignore[no-redef]
        DEFAULT_VOCABULARY,
        RetrievalBatch,
        generate_mqar_batch,
    )


@dataclass(frozen=True)
class CurriculumPhase:
    pairs: int
    queries: int
    length: int
    updates: int


@dataclass(frozen=True)
class Evaluation:
    length: int
    pairs: int
    queries_per_sequence: int
    query_count: int
    exact_accuracy: float
    exact_sequence_accuracy: float
    bits_per_query: float


DEFAULT_CURRICULUM = (
    CurriculumPhase(2, 2, 16, 300),
    CurriculumPhase(4, 4, 24, 300),
    CurriculumPhase(8, 8, 48, 400),
    CurriculumPhase(16, 16, 96, 600),
)


def _seed(label: str, index: int, base: int) -> int:
    payload = f"hybrid-v1.4.1:{base}:{label}:{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def _batch(
    phase: CurriculumPhase, batch_size: int, *, seed: int, device: torch.device
) -> RetrievalBatch:
    return generate_mqar_batch(
        batch_size,
        phase.pairs,
        phase.queries,
        phase.length,
        seed=seed,
    ).to(device)


def _loss_and_accuracy(
    model: HybridMemoryLM,
    batch: RetrievalBatch,
    *,
    association_coefficient: float,
    intermediate_coefficient: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    output = model(
        batch.inputs,
        delta_scan_mode="parallel",
        return_diagnostics=(
            association_coefficient > 0.0 or intermediate_coefficient > 0.0
        ),
    )
    retrieval = tasks.retrieval_loss(output["logits"], batch)
    association = retrieval.new_zeros(())
    if association_coefficient > 0.0:
        association = gated_delta_association_auxiliary_loss(output, batch)
    intermediate = retrieval.new_zeros(())
    if intermediate_coefficient > 0.0:
        intermediate = intermediate_retrieval_auxiliary_loss(output, batch)
    predictions = tasks.gather_query_logits(output["logits"], batch).argmax(-1)
    accuracy = float((predictions == batch.targets).float().mean().item())
    return retrieval, association, intermediate, accuracy


@torch.no_grad()
def evaluate(
    model: HybridMemoryLM,
    *,
    length: int,
    pairs: int,
    queries: int,
    batch_size: int,
    batches: int,
    seed_base: int,
    device: torch.device,
) -> Evaluation:
    model.eval()
    correct = 0
    exact_sequences = 0
    query_count = 0
    sequence_count = 0
    nll = 0.0
    phase = CurriculumPhase(pairs, queries, length, 0)
    for index in range(batches):
        batch = _batch(
            phase,
            batch_size,
            seed=_seed(f"evaluation-{length}", index, seed_base),
            device=device,
        )
        output = model(batch.inputs, delta_scan_mode="parallel")
        logits = tasks.gather_query_logits(output["logits"], batch)
        matches = logits.argmax(-1) == batch.targets
        correct += int(matches.sum())
        exact_sequences += int(matches.all(-1).sum())
        query_count += batch.targets.numel()
        sequence_count += batch.targets.shape[0]
        nll += float(tasks.retrieval_loss(output["logits"], batch, reduction="sum"))
    return Evaluation(
        length=length,
        pairs=pairs,
        queries_per_sequence=queries,
        query_count=query_count,
        exact_accuracy=correct / query_count,
        exact_sequence_accuracy=exact_sequences / sequence_count,
        bits_per_query=nll / (query_count * math.log(2.0)),
    )


def _train_steps(
    model: HybridMemoryLM,
    optimizer: torch.optim.Optimizer,
    phase: CurriculumPhase,
    *,
    batch_size: int,
    association_coefficient: float,
    seed_base: int,
    step_offset: int,
    device: torch.device,
    fixed_batch: RetrievalBatch | None = None,
    intermediate_coefficient: float = 0.0,
) -> dict[str, float]:
    model.train()
    retrieval_sum = 0.0
    association_sum = 0.0
    intermediate_sum = 0.0
    final_accuracy = 0.0
    for local_step in range(phase.updates):
        step = step_offset + local_step
        batch = fixed_batch or _batch(
            phase,
            batch_size,
            seed=_seed("training", step, seed_base),
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        retrieval, association, intermediate, final_accuracy = _loss_and_accuracy(
            model,
            batch,
            association_coefficient=association_coefficient,
            intermediate_coefficient=intermediate_coefficient,
        )
        loss = (
            retrieval
            + association_coefficient * association
            + intermediate_coefficient * intermediate
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite learnability loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        retrieval_sum += float(retrieval.detach())
        association_sum += float(association.detach())
        intermediate_sum += float(intermediate.detach())
    return {
        "mean_retrieval_loss": retrieval_sum / phase.updates,
        "mean_association_loss": association_sum / phase.updates,
        "mean_intermediate_retrieval_loss": intermediate_sum / phase.updates,
        "last_batch_accuracy": final_accuracy,
    }


def oracle_hash_ceiling(
    *, pairs: int, slots: int, batches: int, batch_size: int, seed_base: int
) -> float:
    """Exact last-write-wins ceiling for the static key-mod-slot strategy."""

    correct = 0
    count = 0
    phase = CurriculumPhase(pairs, min(4, pairs), max(5 * pairs, 16), 0)
    for index in range(batches):
        batch = _batch(
            phase,
            batch_size,
            seed=_seed(f"oracle-{slots}", index, seed_base),
            device=torch.device("cpu"),
        )
        keys = batch.metadata["stored_keys"]
        query_indices = batch.metadata["query_pair_indices"]
        assert isinstance(keys, torch.Tensor)
        assert isinstance(query_indices, torch.Tensor)
        addresses = (keys - DEFAULT_VOCABULARY.key_start).remainder(slots)
        queried_addresses = addresses.gather(1, query_indices)
        write_indices = torch.arange(pairs).expand(batch_size, -1)
        for query in range(phase.queries):
            address = queried_addresses[:, query, None]
            last = torch.where(addresses == address, write_indices, -1).amax(-1)
            correct += int((last == query_indices[:, query]).sum())
            count += batch_size
    return correct / count


def _git_state() -> dict[str, Any]:
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
    return {"commit": commit, "status": status}


def run_screen(
    *,
    seed_base: int,
    batch_size: int,
    eval_batch_size: int,
    eval_batches: int,
    learning_rate: float,
    association_coefficient: float,
    device: torch.device,
    quick: bool,
) -> tuple[dict[str, Any], HybridMemoryLM]:
    started = time.perf_counter()
    torch.manual_seed(seed_base)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed_base)
    curriculum = (
        tuple(
            CurriculumPhase(p.pairs, p.queries, p.length, max(20, p.updates // 10))
            for p in DEFAULT_CURRICULUM
        )
        if quick
        else DEFAULT_CURRICULUM
    )
    config = HybridMemoryConfig(
        vocab_size=DEFAULT_VOCABULARY.vocab_size,
        model_dim=64,
        layer_plan=("gated_delta", "attention"),
        attention_heads=4,
        attention_window_size=1024,
        gated_delta_heads=4,
        gated_delta_key_dim=16,
        gated_delta_value_dim=16,
        use_local_conv=True,
        conv_kernel=4,
        expansion=2,
        dropout=0.0,
    )

    # Fixed-batch control uses a separate initialization and retrieval-only loss.
    fixed_model = HybridMemoryLM(config).to(device)
    fixed_optimizer = torch.optim.AdamW(fixed_model.parameters(), lr=learning_rate)
    fixed_phase = CurriculumPhase(16, 16, 96, 80 if quick else 250)
    fixed_batch = _batch(
        fixed_phase,
        batch_size,
        seed=_seed("fixed", 0, seed_base),
        device=device,
    )
    fixed_trace = _train_steps(
        fixed_model,
        fixed_optimizer,
        fixed_phase,
        batch_size=batch_size,
        association_coefficient=0.0,
        seed_base=seed_base,
        step_offset=0,
        device=device,
        fixed_batch=fixed_batch,
    )

    torch.manual_seed(seed_base)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed_base)
    model = HybridMemoryLM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    phases = []
    step_offset = 0
    useful_labels = 0
    presented_tokens = 0
    for phase in curriculum:
        trace = _train_steps(
            model,
            optimizer,
            phase,
            batch_size=batch_size,
            association_coefficient=association_coefficient,
            seed_base=seed_base,
            step_offset=step_offset,
            device=device,
        )
        useful_labels += phase.updates * batch_size * phase.queries
        presented_tokens += phase.updates * batch_size * phase.length
        phases.append({"phase": asdict(phase), **trace})
        step_offset += phase.updates

    evaluations = [
        evaluate(
            model,
            length=96,
            pairs=16,
            queries=16,
            batch_size=eval_batch_size,
            batches=eval_batches,
            seed_base=seed_base,
            device=device,
        ),
        evaluate(
            model,
            length=512,
            pairs=16,
            queries=4,
            batch_size=eval_batch_size,
            batches=eval_batches,
            seed_base=seed_base,
            device=device,
        ),
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "claim_status": "commissioning",
        "label_supervised_association": association_coefficient > 0.0,
        "association_coefficient": association_coefficient,
        "seed_base": seed_base,
        "config": asdict(config),
        "parameter_count": parameter_count(model),
        "curriculum": [asdict(item) for item in curriculum],
        "useful_query_labels": useful_labels,
        "presented_tokens": presented_tokens,
        "fixed_batch_control": fixed_trace,
        "phase_traces": phases,
        "evaluations": [asdict(item) for item in evaluations],
        "selected_static_hash_oracle_ceiling": {
            str(slots): oracle_hash_ceiling(
                pairs=16,
                slots=slots,
                batches=max(4, eval_batches),
                batch_size=eval_batch_size,
                seed_base=seed_base,
            )
            for slots in (4, 8, 16, 32, 64)
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "git": _git_state(),
        "elapsed_wall_seconds": time.perf_counter() - started,
    }
    return report, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=1401)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--association-coefficient", type=float, default=0.25)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.eval_batch_size < 1 or args.eval_batches < 1:
        parser.error("batch sizes and eval batches must be positive")
    if args.learning_rate <= 0.0 or args.association_coefficient < 0.0:
        parser.error("learning rate must be positive and coefficient nonnegative")
    report, model = run_screen(
        seed_base=args.seed,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        eval_batches=args.eval_batches,
        learning_rate=args.learning_rate,
        association_coefficient=args.association_coefficient,
        device=torch.device(args.device),
        quick=args.quick,
    )
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(args.output)
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": report["config"],
                "report_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            },
            args.checkpoint,
        )
        print(args.checkpoint)


if __name__ == "__main__":
    main()
