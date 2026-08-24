"""Publishable fp16/fp32 precision-horizon screen against direct fp64 sources."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from delta_product_reference import DeltaProductReferenceLayer

from .audits import precision_horizon_audit
from .experiments import (
    environment_report,
    git_commit,
    git_status,
    jsonable,
    source_file_digests,
)
from .long_context_screen import write_json_atomic
from .selected_block import SelectedBlockConfig, SelectedBlockMemory


def run_precision_screen(
    *,
    horizon: int = 65_536,
    seed: int = 20_260_824,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Run generic, selected-block, and DeltaProduct paths at one horizon."""

    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    resolved_device = torch.device(device)
    started = time.perf_counter()
    with torch.random.fork_rng(
        devices=[]
        if resolved_device.type != "cuda"
        else [
            torch.cuda.current_device()
            if resolved_device.index is None
            else resolved_device.index
        ]
    ):
        torch.manual_seed(seed)
        selected = (
            SelectedBlockMemory(
                SelectedBlockConfig(
                    model_dim=4,
                    heads=1,
                    blocks=2,
                    slots_per_block=2,
                    value_dim=2,
                    update_rank=1,
                    retention_min=0.99,
                    retention_max=0.9999,
                )
            )
            .double()
            .to(resolved_device)
            .eval()
        )
        delta = (
            DeltaProductReferenceLayer(
                hidden_size=4,
                num_heads=1,
                num_householder=2,
            )
            .double()
            .to(resolved_device)
            .eval()
        )
        generator = torch.Generator(device=resolved_device).manual_seed(seed + 1)
        selected_inputs = torch.randn(
            1,
            horizon,
            4,
            dtype=torch.float64,
            device=resolved_device,
            generator=generator,
        )
        delta_inputs = torch.randn(
            1,
            horizon,
            4,
            dtype=torch.float64,
            device=resolved_device,
            generator=generator,
        )
        audit = precision_horizon_audit(
            horizon=horizon,
            retention=0.9999,
            state_size=4,
            value_size=2,
            seed=seed,
            parallel_chunk_size=min(1024, horizon),
            device=resolved_device,
            selected_block=selected,
            selected_inputs=selected_inputs,
            delta_product=delta,
            delta_inputs=delta_inputs,
        )
    completed_without_nonfinite = all(
        path["nonfinite_count"] == 0
        for case in audit["cases"]
        for path in case["paths"]
    )
    report = jsonable(
        {
            "schema_version": 1,
            "screen": "hybrid_memory_v1_4_precision_horizon",
            "horizon": horizon,
            "seed": seed,
            "audit": audit,
            "completed_without_nonfinite": completed_without_nonfinite,
            "evidentiary": True,
            "claim_boundary": (
                "Numerical error-growth evidence only; not model quality or speed."
            ),
            "environment": environment_report(resolved_device, torch.float64),
            "source_files": source_file_digests(),
            "git_commit": git_commit(),
            "git_status": git_status(),
            "elapsed_wall_seconds": time.perf_counter() - started,
        }
    )
    if not isinstance(report, dict):
        raise TypeError("precision report must serialize to a mapping")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=65_536)
    parser.add_argument("--seed", type=int, default=20_260_824)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = run_precision_screen(
        horizon=arguments.horizon,
        seed=arguments.seed,
        device=arguments.device,
    )
    write_json_atomic(arguments.output, report)
    return 0 if report["completed_without_nonfinite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_precision_screen"]
