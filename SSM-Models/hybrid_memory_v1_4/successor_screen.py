"""Development screen for the v1.4.2 normalized/deep-supervised successor."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, replace
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

DEVELOPMENT_SEEDS = (1401, 1429)
RETENTION_SAFE_MINIMUM = 0.999
RETENTION_SAFE_INITIAL = 0.9995


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> HybridMemoryConfig:
    return HybridMemoryConfig(
        vocab_size=DEFAULT_VOCABULARY.vocab_size,
        model_dim=64,
        layer_plan=("gated_delta", "attention"),
        attention_heads=4,
        attention_window_size=1024,
        gated_delta_heads=4,
        gated_delta_key_dim=32,
        gated_delta_value_dim=16,
        gated_delta_normalize_values=True,
        use_local_conv=True,
        conv_kernel=4,
        expansion=2,
        dropout=0.0,
    )


def _identity_config() -> HybridMemoryConfig:
    return HybridMemoryConfig(
        vocab_size=DEFAULT_VOCABULARY.vocab_size,
        model_dim=64,
        layer_plan=("gated_delta", "attention"),
        attention_heads=4,
        attention_window_size=1024,
        gated_delta_heads=4,
        gated_delta_key_dim=32,
        gated_delta_value_dim=16,
        gated_delta_normalize_values=True,
        gated_delta_identity_value_path=True,
        gated_delta_identity_output_gate=True,
        gated_delta_residual_scale_init=0.0,
        use_local_conv=True,
        conv_kernel=4,
        expansion=2,
        dropout=0.0,
    )


def _tied_identity_config() -> HybridMemoryConfig:
    return HybridMemoryConfig(
        **{
            **asdict(_identity_config()),
            "gated_delta_tie_query_key": True,
        }
    )


def _retention_safe_config() -> HybridMemoryConfig:
    """Return the v1.4.5 candidate with a 512-token-safe decay floor.

    This preserves the validated v1.4.4 tensor shapes and address geometry.
    Only the range and initialization of the learned global retention control
    change.  Content-selective DeltaNet overwrites remain unconstrained.
    """

    return HybridMemoryConfig(
        **{
            **asdict(_tied_identity_config()),
            "gated_delta_minimum_retention": RETENTION_SAFE_MINIMUM,
            "gated_delta_initial_retention": RETENTION_SAFE_INITIAL,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--identity", action="store_true")
    parser.add_argument("--tied-query-key", action="store_true")
    parser.add_argument("--final-updates", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEVELOPMENT_SEEDS))
    args = parser.parse_args()
    device = torch.device(args.device)
    root = Path(__file__).resolve().parents[2]
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    started = time.perf_counter()
    runs = []
    development_seeds = tuple(args.seeds)
    if args.tied_query_key and not args.identity:
        parser.error("--tied-query-key requires --identity")
    config_factory = (
        _tied_identity_config
        if args.tied_query_key
        else _identity_config
        if args.identity
        else _config
    )
    version_label = (
        "v1_4_4" if args.tied_query_key else "v1_4_3" if args.identity else "v1_4_2"
    )
    curriculum = (
        *DEFAULT_CURRICULUM[:-1],
        replace(DEFAULT_CURRICULUM[-1], updates=args.final_updates),
    )
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for model_seed in development_seeds:
        torch.manual_seed(model_seed)
        torch.cuda.manual_seed_all(model_seed)
        config = config_factory()
        model = HybridMemoryLM(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
        traces = []
        step_offset = 0
        for phase in curriculum:
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
                seed_base=model_seed + 600_000,
                device=device,
            ),
            evaluate(
                model,
                length=512,
                pairs=16,
                queries=4,
                batch_size=32,
                batches=16,
                seed_base=model_seed + 700_000,
                device=device,
            ),
        ]
        checkpoint = (
            args.checkpoint_dir / f"hybrid_{version_label}_dev_seed{model_seed}.pt"
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": asdict(config),
                "model_seed": model_seed,
                "status": "development",
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
    length_512 = [
        evaluation
        for run in runs
        for evaluation in run["evaluations"]
        if evaluation["length"] == 512
    ]
    report = {
        "schema_version": 1,
        "claim_status": "development objective and capacity screen",
        "development_seeds_reused_from_prior_work": list(development_seeds),
        "identity_preserving_value_path": args.identity,
        "config": asdict(config_factory()),
        "curriculum": [asdict(item) for item in curriculum],
        "batch_size": 32,
        "learning_rate": 3e-3,
        "association_coefficient": 0.25,
        "intermediate_retrieval_coefficient": 0.50,
        "label_supervised": True,
        "runs": runs,
        "aggregate_length_512": {
            "mean_exact_accuracy": sum(item["exact_accuracy"] for item in length_512)
            / len(length_512),
            "minimum_exact_accuracy": min(
                item["exact_accuracy"] for item in length_512
            ),
        },
        "git_commit": git_commit,
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


if __name__ == "__main__":
    main()
