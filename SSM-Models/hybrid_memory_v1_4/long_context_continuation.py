"""Continue a commissioned v1.4.1 checkpoint on fresh length-512 MQAR episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

if __package__:
    from .learnability_screen import CurriculumPhase, _train_steps, evaluate
    from .model import HybridMemoryConfig, HybridMemoryLM
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=2401)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--association-coefficient", type=float, default=0.25)
    args = parser.parse_args()
    if min(args.updates, args.batch_size, args.eval_batch_size, args.eval_batches) < 1:
        parser.error("updates and batch counts must be positive")
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = HybridMemoryConfig(**payload["config"])
    model = HybridMemoryLM(config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    started = time.perf_counter()
    evaluation_kwargs = {
        "length": 512,
        "pairs": 16,
        "queries": 4,
        "batch_size": args.eval_batch_size,
        "batches": args.eval_batches,
        "seed_base": args.seed,
        "device": device,
    }
    before = evaluate(model, **evaluation_kwargs)
    phase = CurriculumPhase(16, 4, 512, args.updates)
    trace = _train_steps(
        model,
        optimizer,
        phase,
        batch_size=args.batch_size,
        association_coefficient=args.association_coefficient,
        seed_base=args.seed,
        step_offset=0,
        device=device,
    )
    after = evaluate(model, **evaluation_kwargs)
    extrapolation = [
        evaluate(
            model,
            length=length,
            pairs=16,
            queries=4,
            batch_size=max(4, args.eval_batch_size // 4),
            batches=max(4, args.eval_batches // 4),
            seed_base=args.seed + length,
            device=device,
        )
        for length in (1024, 2048)
    ]
    report = {
        "schema_version": 1,
        "claim_status": "commissioning continuation",
        "source_checkpoint": str(args.checkpoint),
        "source_report_sha256": payload.get("report_sha256"),
        "seed": args.seed,
        "phase": asdict(phase),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "association_coefficient": args.association_coefficient,
        "label_supervised_association": args.association_coefficient > 0.0,
        "useful_query_labels": args.updates * args.batch_size * 4,
        "before": asdict(before),
        "training_trace": trace,
        "after": asdict(after),
        "extrapolation": [asdict(item) for item in extrapolation],
        "elapsed_wall_seconds": time.perf_counter() - started,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "report_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        },
        args.output_checkpoint,
    )
    print(args.output)
    print(args.output_checkpoint)


if __name__ == "__main__":
    main()
