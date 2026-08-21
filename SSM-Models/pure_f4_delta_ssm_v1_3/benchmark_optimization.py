"""Matched eager/compiled benchmark for semantics-preserving v1.3 optimizations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import random
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from .model import ExceptionalDeltaConfig, ExceptionalDeltaLM, parameter_count


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(samples: list[float]) -> dict[str, object]:
    ordered = sorted(samples)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "p20_ms": percentile(0.2),
        "p80_ms": percentile(0.8),
        "mean_ms": statistics.fmean(samples),
        "stdev_ms": statistics.pstdev(samples),
    }


def _cuda_time(operation: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop))


def _interleaved_timings(
    operations: dict[str, Callable[[], None]], *, repeats: int, seed: int
) -> dict[str, dict[str, object]]:
    samples = {name: [] for name in operations}
    generator = random.Random(seed)
    for _ in range(repeats):
        order = list(operations)
        generator.shuffle(order)
        for name in order:
            samples[name].append(_cuda_time(operations[name]))
    return {name: _summary(values) for name, values in samples.items()}


def _build(
    config: ExceptionalDeltaConfig,
    *,
    fast: bool,
    product_backend: str,
    determinant_backend: str,
) -> ExceptionalDeltaLM:
    values = dict(config.__dict__)
    values["identity_fast_path"] = fast
    values["albert_product_backend"] = product_backend
    values["albert_determinant_backend"] = determinant_backend
    return ExceptionalDeltaLM(ExceptionalDeltaConfig(**values))


def _loss(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    return model(tokens, scan_mode="auto")["logits"].square().mean()


def _output_and_gradients(
    model: nn.Module, tokens: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    output = model(tokens, scan_mode="auto")["logits"]
    output.square().mean().backward()
    gradients = [
        parameter.grad.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    return output.detach().clone(), gradients


def _difference(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = actual.double() - expected.double()
    denominator = max(float(expected.double().norm()), 1e-30)
    return {
        "maximum_absolute_error": float(delta.abs().max()),
        "relative_l2_error": float(delta.norm()) / denominator,
    }


def _gradient_difference(
    actual: list[torch.Tensor], expected: list[torch.Tensor]
) -> dict[str, float]:
    if len(actual) != len(expected):
        raise AssertionError("gradient lists differ in length")
    actual_flat = torch.cat([tensor.double().flatten() for tensor in actual])
    expected_flat = torch.cat([tensor.double().flatten() for tensor in expected])
    return _difference(actual_flat, expected_flat)


def _peak_incremental_cuda_bytes(operation: Callable[[], None]) -> int:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    operation()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated() - baseline)


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    config = ExceptionalDeltaConfig(
        d_model=args.d_model,
        num_layers=args.layers,
        memory_width=args.memory_width,
        update_rank=args.update_rank,
        action_algebra="identity",
        action_geometry="direct",
        d_conv=args.d_conv,
        channel_mixer="jordan",
        readout_mode="albert_invariants",
    )
    generic_legacy = _build(
        config,
        fast=False,
        product_backend="dense",
        determinant_backend="jordan",
    ).to(device)
    generic_explicit = _build(
        config,
        fast=False,
        product_backend="dense",
        determinant_backend="explicit",
    ).to(device)
    fast_dense_safe = _build(
        config,
        fast=True,
        product_backend="dense",
        determinant_backend="jordan",
    ).to(device)
    fast_dense_explicit = _build(
        config,
        fast=True,
        product_backend="dense",
        determinant_backend="explicit",
    ).to(device)
    fast_sparse = _build(
        config,
        fast=True,
        product_backend="sparse",
        determinant_backend="jordan",
    ).to(device)
    generic_explicit.load_state_dict(generic_legacy.state_dict())
    fast_dense_safe.load_state_dict(generic_legacy.state_dict())
    fast_dense_explicit.load_state_dict(generic_legacy.state_dict())
    fast_sparse.load_state_dict(generic_legacy.state_dict())
    compiled_dense_source = copy.deepcopy(fast_dense_safe)
    compiled_sparse_source = copy.deepcopy(fast_sparse)
    compiled_dense = torch.compile(
        compiled_dense_source,
        fullgraph=True,
        mode=args.compile_mode,
    )
    compiled_sparse = torch.compile(
        compiled_sparse_source,
        fullgraph=True,
        mode=args.compile_mode,
    )
    tokens = torch.randint(0, config.vocab_size, (args.batch, args.length), device=device)

    compilation_seconds = {}
    compiled_results = {}
    for name, model in (
        ("compiled_one_sided_dense", compiled_dense),
        ("compiled_one_sided_sparse", compiled_sparse),
    ):
        compile_start = time.perf_counter()
        compiled_results[name] = _output_and_gradients(model, tokens)
        torch.cuda.synchronize()
        compilation_seconds[name] = time.perf_counter() - compile_start

    generic_output, generic_gradients = _output_and_gradients(generic_legacy, tokens)
    explicit_output, explicit_gradients = _output_and_gradients(
        generic_explicit, tokens
    )
    fast_dense_safe_output, fast_dense_safe_gradients = _output_and_gradients(
        fast_dense_safe, tokens
    )
    fast_dense_explicit_output, fast_dense_explicit_gradients = _output_and_gradients(
        fast_dense_explicit, tokens
    )
    fast_sparse_output, fast_sparse_gradients = _output_and_gradients(
        fast_sparse, tokens
    )
    parity = {
        "explicit_determinant_vs_legacy_output": _difference(
            explicit_output, generic_output
        ),
        "explicit_determinant_vs_legacy_gradients": _gradient_difference(
            explicit_gradients, generic_gradients
        ),
        "safe_one_sided_dense_vs_legacy_output": _difference(
            fast_dense_safe_output, generic_output
        ),
        "safe_one_sided_dense_vs_legacy_gradients": _gradient_difference(
            fast_dense_safe_gradients, generic_gradients
        ),
        "rejected_explicit_one_sided_vs_explicit_generic_output": _difference(
            fast_dense_explicit_output, explicit_output
        ),
        "rejected_explicit_one_sided_vs_explicit_generic_gradients": _gradient_difference(
            fast_dense_explicit_gradients, explicit_gradients
        ),
        "one_sided_sparse_vs_generic_output": _difference(
            fast_sparse_output, generic_output
        ),
        "one_sided_sparse_vs_generic_gradients": _gradient_difference(
            fast_sparse_gradients, generic_gradients
        ),
    }
    for name, (output, gradients) in compiled_results.items():
        parity[f"{name}_vs_generic_output"] = _difference(output, generic_output)
        parity[f"{name}_vs_generic_gradients"] = _gradient_difference(
            gradients, generic_gradients
        )

    def inference(model: nn.Module, *, mark_step: bool = False) -> Callable[[], None]:
        def operation() -> None:
            if mark_step:
                torch.compiler.cudagraph_mark_step_begin()
            with torch.inference_mode():
                model(tokens, scan_mode="auto")["logits"]

        return operation

    def forward_backward(
        model: nn.Module, *, mark_step: bool = False
    ) -> Callable[[], None]:
        def operation() -> None:
            if mark_step:
                torch.compiler.cudagraph_mark_step_begin()
            model.zero_grad(set_to_none=True)
            _loss(model, tokens).backward()

        return operation

    inference_operations = {
        "eager_generic_legacy": inference(generic_legacy),
        "eager_generic_explicit": inference(generic_explicit),
        "eager_one_sided_dense_safe": inference(fast_dense_safe),
        "eager_one_sided_dense_explicit_rejected": inference(fast_dense_explicit),
        "eager_one_sided_sparse": inference(fast_sparse),
        "compiled_one_sided_dense": inference(compiled_dense, mark_step=True),
        "compiled_one_sided_sparse": inference(compiled_sparse, mark_step=True),
    }
    training_operations = {
        "eager_generic_legacy": forward_backward(generic_legacy),
        "eager_generic_explicit": forward_backward(generic_explicit),
        "eager_one_sided_dense_safe": forward_backward(fast_dense_safe),
        "eager_one_sided_dense_explicit_rejected": forward_backward(
            fast_dense_explicit
        ),
        "eager_one_sided_sparse": forward_backward(fast_sparse),
        "compiled_one_sided_dense": forward_backward(compiled_dense, mark_step=True),
        "compiled_one_sided_sparse": forward_backward(compiled_sparse, mark_step=True),
    }
    for operation in (*inference_operations.values(), *training_operations.values()):
        for _ in range(args.warmups):
            operation()
    torch.cuda.synchronize()
    inference_timings = _interleaved_timings(
        inference_operations, repeats=args.repeats, seed=args.seed + 1
    )
    training_timings = _interleaved_timings(
        training_operations, repeats=args.repeats, seed=args.seed + 2
    )
    peak_memory = {
        name: _peak_incremental_cuda_bytes(operation)
        for name, operation in training_operations.items()
    }
    root = Path(__file__).resolve().parent
    return {
        "schema_version": 1,
        "experiment": "v1.3 matched identity fast-path and compile optimization",
        "status": "development systems benchmark; no hardware-general claim",
        "config": config.__dict__,
        "protocol": {
            "batch": args.batch,
            "length": args.length,
            "tokens_per_call": args.batch * args.length,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "interleaved_order": True,
            "compile_mode": args.compile_mode,
            "fullgraph": True,
            "seed": args.seed,
            "objective": "mean squared logits for matched forward-backward timing",
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "parameter_count": parameter_count(generic_legacy),
        "compilation_and_first_training_seconds": compilation_seconds,
        "parity": parity,
        "inference": inference_timings,
        "forward_backward": training_timings,
        "peak_incremental_cuda_bytes": peak_memory,
        "source_sha256": {
            name: _sha256(root / name)
            for name in (
                "action.py",
                "albert.py",
                "benchmark_optimization.py",
                "model.py",
                "scan.py",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--memory-width", type=int, default=4)
    parser.add_argument("--update-rank", type=int, default=2)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark(args)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
