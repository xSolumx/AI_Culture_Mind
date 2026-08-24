"""Matched development replay of the G9 weak seed with safe retention."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

if __package__:
    from .combined_validation import (
        COMBINED_SCHEDULE,
        _git,
        _sha256,
        _train_schedule,
        schedule_label_counts,
    )
    from .learnability_screen import CurriculumPhase
    from .successor_screen import _retention_safe_config, _tied_identity_config
    from .upstream_learning_comparison import _build_model, _evaluate
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4.combined_validation import (  # type: ignore[no-redef]
        COMBINED_SCHEDULE,
        _git,
        _sha256,
        _train_schedule,
        schedule_label_counts,
    )
    from hybrid_memory_v1_4.learnability_screen import (  # type: ignore[no-redef]
        CurriculumPhase,
    )
    from hybrid_memory_v1_4.successor_screen import (  # type: ignore[no-redef]
        _retention_safe_config,
        _tied_identity_config,
    )
    from hybrid_memory_v1_4.upstream_learning_comparison import (  # type: ignore[no-redef]
        _build_model,
        _evaluate,
    )

EXPOSED_FAILURE_SEED = 1723
CANDIDATE_NAME = "hybrid_v1_4_5"
REFERENCE_NAME = "hybrid_v1_4_4"


def _reference_run(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("model_name") != REFERENCE_NAME:
        raise ValueError("reference report is not the v1.4.4 G9 cohort")
    frozen_config = json.loads(json.dumps(asdict(_tied_identity_config())))
    if report.get("config") != frozen_config:
        raise ValueError("reference report config does not match frozen v1.4.4")
    if report.get("schedule") != [asdict(phase) for phase in COMBINED_SCHEDULE]:
        raise ValueError("reference report schedule does not match frozen G9")
    runs = [
        run
        for run in report.get("runs", [])
        if run.get("model_seed") == EXPOSED_FAILURE_SEED
    ]
    if len(runs) != 1:
        raise ValueError("reference report must contain the exposed seed once")
    return runs[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    reference = _reference_run(args.reference_report)
    git_commit, git_status_start = _git()
    started = time.perf_counter()
    torch.manual_seed(EXPOSED_FAILURE_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(EXPOSED_FAILURE_SEED)
    model = _build_model(CANDIDATE_NAME, device)
    traces, optimizer = _train_schedule(
        model,
        model_seed=EXPOSED_FAILURE_SEED,
        device=device,
        model_name=CANDIDATE_NAME,
    )
    evaluations = [
        _evaluate(
            CANDIDATE_NAME,
            model,
            CurriculumPhase(16, 16, 96, 0),
            seed_base=EXPOSED_FAILURE_SEED + 2_400_000,
            device=device,
        ),
        _evaluate(
            CANDIDATE_NAME,
            model,
            CurriculumPhase(16, 4, 512, 0),
            seed_base=EXPOSED_FAILURE_SEED + 2_500_000,
            device=device,
        ),
    ]
    passed = all(item["exact_accuracy"] >= 0.90 for item in evaluations)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_name": CANDIDATE_NAME,
            "model_config": asdict(_retention_safe_config()),
            "model_seed": EXPOSED_FAILURE_SEED,
            "schedule": [asdict(phase) for phase in COMBINED_SCHEDULE],
            "association_target": "reverse_key",
            "claim_status": "exposed-seed causal development intervention",
        },
        args.checkpoint,
    )
    retrieval_labels, association_labels = schedule_label_counts()
    report = {
        "schema_version": 1,
        "claim_status": (
            "exposed failure repaired by retention-only intervention"
            if passed
            else "retention-only intervention did not repair exposed failure"
        ),
        "passed_development_gate": passed,
        "gate": "exposed seed exact query accuracy >= 0.90 at L96 and L512",
        "causal_intervention": {
            "changed_fields": {
                "gated_delta_minimum_retention": [0.90, 0.999],
                "gated_delta_initial_retention": [0.995, 0.9995],
            },
            "unchanged": (
                "model seed, tensor shapes, address geometry, data namespaces, "
                "optimizer, schedule, objective, and evaluation cohorts"
            ),
            "minimum_retention_half_life_tokens": math.log(0.5) / math.log(0.999),
        },
        "model_name": CANDIDATE_NAME,
        "model_config": asdict(_retention_safe_config()),
        "exposed_failure_seed": EXPOSED_FAILURE_SEED,
        "schedule": [asdict(phase) for phase in COMBINED_SCHEDULE],
        "retrieval_labels": retrieval_labels,
        "association_labels": association_labels,
        "phase_traces": traces,
        "evaluations": evaluations,
        "reference": {
            "report": str(args.reference_report),
            "report_sha256": _sha256(args.reference_report),
            "evaluations": reference["evaluations"],
            "checkpoint_sha256": reference["checkpoint_sha256"],
        },
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
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
            "exposed-seed causal development result; not fresh validation, "
            "label-free learning, or natural-language quality"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(json.dumps({str(e["length"]): e["exact_accuracy"] for e in evaluations}))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
