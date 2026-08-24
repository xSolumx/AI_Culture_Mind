"""Non-training long-context mechanical screen with atomic JSON output."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

if __package__:
    from .experiments import (
        EnvironmentReport,
        SourceFileDigest,
        confirm_chunk_replay,
        environment_report,
        git_commit,
        git_status,
        jsonable,
        source_file_digests,
    )
    from .model import HybridMemoryConfig, HybridMemoryLM
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4.experiments import (  # type: ignore[no-redef]
        EnvironmentReport,
        SourceFileDigest,
        confirm_chunk_replay,
        environment_report,
        git_commit,
        git_status,
        jsonable,
        source_file_digests,
    )
    from hybrid_memory_v1_4.model import (  # type: ignore[no-redef]
        HybridMemoryConfig,
        HybridMemoryLM,
    )


@dataclass(frozen=True)
class MechanicalLengthResult:
    length: int
    chunks: int
    elapsed_wall_seconds: float
    tokens_per_second: float
    finite_logits: bool
    finite_state: bool
    actual_cache_bytes: int
    capacity_cache_bytes: int
    bounded_cache: bool


@dataclass(frozen=True)
class MechanicalScreenResult:
    schema_version: int
    lengths: tuple[int, ...]
    chunk_size: int
    batch_size: int
    seed: int
    layer_plan: tuple[str, ...]
    parameter_count: int
    chunk_replay_confirmed: bool
    rows: tuple[MechanicalLengthResult, ...]
    environment: EnvironmentReport
    source_files: tuple[SourceFileDigest, ...]
    git_commit: str | None
    git_status: tuple[str, ...]
    elapsed_wall_seconds: float
    evidentiary: bool
    passed: bool


def _positive(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _state_finite(states: tuple[object, ...]) -> bool:
    for state in states:
        for value in vars(state).values():
            if isinstance(value, torch.Tensor) and not bool(
                torch.isfinite(value).all()
            ):
                return False
    return True


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_mechanical_screen(
    *,
    lengths: tuple[int, ...] = (512, 2048, 8192),
    chunk_size: int = 128,
    batch_size: int = 1,
    seed: int = 0,
    device: torch.device | str = "cpu",
    model_dim: int = 32,
    layer_plan: tuple[str, ...] = ("selected_block", "attention"),
    attention_heads: int = 4,
    attention_window_size: int = 128,
    selected_blocks: int = 4,
    slots_per_block: int = 2,
    value_dim: int = 8,
) -> MechanicalScreenResult:
    """Stream random-token cohorts and audit finite, bounded complete states."""

    if type(lengths) is not tuple or not lengths:
        raise TypeError("lengths must be a nonempty tuple")
    for length in lengths:
        _positive("length", length)
    if tuple(sorted(set(lengths))) != lengths:
        raise ValueError("lengths must be unique and strictly increasing")
    for name, value in (
        ("chunk_size", chunk_size),
        ("batch_size", batch_size),
        ("model_dim", model_dim),
        ("attention_heads", attention_heads),
        ("attention_window_size", attention_window_size),
        ("selected_blocks", selected_blocks),
        ("slots_per_block", slots_per_block),
        ("value_dim", value_dim),
    ):
        _positive(name, value)
    if type(layer_plan) is not tuple or not layer_plan:
        raise TypeError("layer_plan must be a nonempty tuple")
    allowed_layers = {
        "attention",
        "delta_product",
        "selected_block",
        "structured_spin8",
    }
    if any(kind not in allowed_layers for kind in layer_plan):
        raise ValueError(f"layer_plan entries must be in {sorted(allowed_layers)}")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 2**63 - 1
    ):
        raise ValueError("seed must be an integer in [0, 2**63 - 1]")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

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
        if resolved_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        config = HybridMemoryConfig(
            vocab_size=256,
            model_dim=model_dim,
            layer_plan=layer_plan,  # type: ignore[arg-type]
            attention_heads=attention_heads,
            attention_window_size=attention_window_size,
            selected_heads=1,
            selected_blocks=selected_blocks,
            selected_slots_per_block=slots_per_block,
            selected_value_dim=value_dim,
            selected_update_rank=1,
            use_local_conv=True,
            conv_kernel=3,
            expansion=2,
            dropout=0.0,
        )
        model = HybridMemoryLM(config).to(resolved_device).eval()
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        short_tokens = torch.randint(
            config.vocab_size,
            (batch_size, lengths[0]),
            generator=generator,
            dtype=torch.long,
        ).to(resolved_device)
        replay = confirm_chunk_replay(model, short_tokens, chunk_size)

        rows = []
        with torch.inference_mode():
            for length in lengths:
                tokens = torch.randint(
                    config.vocab_size,
                    (batch_size, length),
                    generator=generator,
                    dtype=torch.long,
                ).to(resolved_device)
                states = None
                finite_logits = True
                chunks = 0
                _synchronize(resolved_device)
                row_started = time.perf_counter()
                for start in range(0, length, chunk_size):
                    output = model(
                        tokens[:, start : start + chunk_size],
                        states,
                        delta_scan_mode="recurrent",
                        selected_scan_mode="physical_gather",
                        structured_scan_mode="recurrent",
                    )
                    finite_logits = finite_logits and bool(
                        torch.isfinite(output["logits"]).all()
                    )
                    states = output["states"]
                    chunks += 1
                _synchronize(resolved_device)
                elapsed = time.perf_counter() - row_started
                assert states is not None
                byte_report = model.state_byte_report(states)
                actual = int(byte_report["actual_bytes"])
                capacity = int(byte_report["capacity_bytes"])
                rows.append(
                    MechanicalLengthResult(
                        length=length,
                        chunks=chunks,
                        elapsed_wall_seconds=elapsed,
                        tokens_per_second=batch_size * length / elapsed,
                        finite_logits=finite_logits,
                        finite_state=_state_finite(states),
                        actual_cache_bytes=actual,
                        capacity_cache_bytes=capacity,
                        bounded_cache=actual <= capacity,
                    )
                )
        parameter_total = sum(parameter.numel() for parameter in model.parameters())
    passed = replay and all(
        row.finite_logits and row.finite_state and row.bounded_cache for row in rows
    )
    return MechanicalScreenResult(
        schema_version=1,
        lengths=lengths,
        chunk_size=chunk_size,
        batch_size=batch_size,
        seed=seed,
        layer_plan=layer_plan,
        parameter_count=parameter_total,
        chunk_replay_confirmed=replay,
        rows=tuple(rows),
        environment=environment_report(resolved_device),
        source_files=source_file_digests(),
        git_commit=git_commit(),
        git_status=git_status(),
        elapsed_wall_seconds=time.perf_counter() - started,
        evidentiary=False,
        passed=passed,
    )


def write_json_atomic(path: str | Path, payload: object) -> None:
    """Durably replace one JSON file without exposing a partial document."""

    destination = Path(path).resolve()
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {destination.parent}")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(
                jsonable(payload), handle, indent=2, sort_keys=True, allow_nan=False
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", nargs="+", type=int, default=[512, 2048, 8192])
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-dim", type=int, default=32)
    parser.add_argument(
        "--layer-plan",
        nargs="+",
        choices=("attention", "delta_product", "selected_block", "structured_spin8"),
        default=["selected_block", "attention"],
    )
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-window-size", type=int, default=128)
    parser.add_argument("--selected-blocks", type=int, default=4)
    parser.add_argument("--slots-per-block", type=int, default=2)
    parser.add_argument("--value-dim", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_mechanical_screen(
        lengths=tuple(arguments.lengths),
        chunk_size=arguments.chunk_size,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
        device=arguments.device,
        model_dim=arguments.model_dim,
        layer_plan=tuple(arguments.layer_plan),
        attention_heads=arguments.attention_heads,
        attention_window_size=arguments.attention_window_size,
        selected_blocks=arguments.selected_blocks,
        slots_per_block=arguments.slots_per_block,
        value_dim=arguments.value_dim,
    )
    write_json_atomic(arguments.output, result)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MechanicalLengthResult",
    "MechanicalScreenResult",
    "main",
    "run_mechanical_screen",
    "write_json_atomic",
]
