"""Run a post-preregistered recurrent correctness diagnostic for the fused kernel."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import torch

from benchmark_fused_gathered_block_memory import (
    TRITON_AVAILABLE,
    _nvidia_smi,
    recurrent_correctness,
)
from benchmark_gathered_block_memory import make_problem


def run_diagnostic(
    *,
    slots: tuple[int, ...],
    batch: int,
    steps: int,
    seed: int,
) -> dict[str, object]:
    rows = []
    passed = True
    for slot_count in slots:
        problem = make_problem(
            batch=batch,
            slots=slot_count,
            device=torch.device("cuda"),
            dtype=torch.float32,
            seed=seed + slot_count,
        )
        report = recurrent_correctness(problem, steps=steps)
        passed = passed and bool(report["passed"])
        rows.append(
            {
                "slots": slot_count,
                "batch": batch,
                "correctness": report,
            }
        )
    return {
        "experiment": "fused gathered-block recurrent trajectory diagnostic",
        "status": "post-preregistered strengthening check",
        "passed": passed,
        "device": "cuda",
        "dtype": "torch.float32",
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton_windows": importlib.metadata.version("triton-windows"),
            "nvidia_smi": _nvidia_smi(),
        },
        "grid": {
            "slots": slots,
            "batch": batch,
            "steps": steps,
            "seed": seed,
        },
        "rows": rows,
        "thresholds": {
            "maximum_state_error": 1e-5,
            "maximum_prediction_error": 1e-5,
            "all_finite": True,
        },
        "claim_boundary": {
            "not_a_frozen_decision_gate": True,
            "tests_repeated_identical_inputs": True,
            "tests_training_or_gradients": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", nargs="+", type=int, default=(64, 1024, 4096))
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=257)
    parser.add_argument("--seed", type=int, default=1_090_257)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if not TRITON_AVAILABLE:
        parser.error("triton-windows is required")
    report = run_diagnostic(
        slots=tuple(args.slots),
        batch=args.batch,
        steps=args.steps,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "rows": len(report["rows"])}, indent=2))


if __name__ == "__main__":
    main()
