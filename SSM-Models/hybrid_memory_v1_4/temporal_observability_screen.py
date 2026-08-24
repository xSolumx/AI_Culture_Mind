"""Deterministic replay of the v1.4 temporal-gradient topology gate."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .audits import temporal_query_observability_audit
from .experiments import (
    environment_report,
    git_commit,
    git_status,
    jsonable,
    source_file_digests,
)
from .long_context_screen import write_json_atomic
from .model import HybridMemoryConfig, HybridMemoryLM


def _model(layer_plan: tuple[str, ...], *, seed: int) -> HybridMemoryLM:
    torch.manual_seed(seed)
    return HybridMemoryLM(
        HybridMemoryConfig(
            vocab_size=29,
            model_dim=8,
            layer_plan=layer_plan,  # type: ignore[arg-type]
            attention_heads=1,
            attention_window_size=8,
            delta_heads=1,
            delta_num_householder=1,
            selected_heads=1,
            selected_blocks=2,
            selected_slots_per_block=2,
            selected_value_dim=4,
            selected_update_rank=1,
            use_local_conv=False,
            expansion=1,
            dropout=0.0,
        )
    ).double()


def run_temporal_observability_screen(*, seed: int = 42) -> dict[str, object]:
    """Replay hard, straight-through, and hybrid final-only credit paths."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    started = time.perf_counter()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    hard_one_block = temporal_query_observability_audit(
        _model(("selected_block",), seed=seed),
        tokens,
        selected_layer_index=0,
        route_mode="hard",
    )
    straight_through_one_block = temporal_query_observability_audit(
        _model(("selected_block",), seed=seed),
        tokens,
        selected_layer_index=0,
        route_mode="straight_through",
    )
    straight_through_hybrid = temporal_query_observability_audit(
        _model(("selected_block", "attention"), seed=seed),
        tokens,
        selected_layer_index=0,
        route_mode="straight_through",
    )
    original_g3_passed = bool(
        hard_one_block["coarse_routes_connected"]
        and hard_one_block["nonfinal_read_path_present"]
    )
    successor_passed = bool(
        straight_through_one_block["coarse_routes_connected"]
        and not straight_through_one_block["nonfinal_read_path_present"]
        and straight_through_hybrid["coarse_routes_connected"]
        and straight_through_hybrid["nonfinal_read_path_present"]
    )
    report = {
        "schema_version": 1,
        "audit": "v1_4_temporal_observability_screen",
        "seed": seed,
        "tokens": tokens.tolist(),
        "cases": {
            "hard_one_block": hard_one_block,
            "straight_through_one_block": straight_through_one_block,
            "straight_through_selected_attention": straight_through_hybrid,
        },
        "original_g3_passed": original_g3_passed,
        "successor_topology_acceptance_passed": successor_passed,
        "evidentiary": True,
        "claim_boundary": (
            "Deterministic float64 gradient-topology evidence only; not retrieval, "
            "language-model, scaling, or speed evidence."
        ),
        "environment": environment_report("cpu", torch.float64),
        "source_files": source_file_digests(),
        "git_commit": git_commit(),
        "git_status": git_status(),
        "elapsed_wall_seconds": time.perf_counter() - started,
    }
    serialized = jsonable(report)
    if not isinstance(serialized, dict):
        raise TypeError("temporal observability report must serialize to a mapping")
    return serialized


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = run_temporal_observability_screen(seed=arguments.seed)
    write_json_atomic(arguments.output, report)
    return 0 if report["successor_topology_acceptance_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_temporal_observability_screen"]
