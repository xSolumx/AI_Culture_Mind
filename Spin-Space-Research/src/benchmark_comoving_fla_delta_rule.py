"""Full-path CUDA benchmark for exact co-moving transported DeltaRule memory."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from benchmark_matched_memory_cores import (
    DELTA_CHUNK_SIZES,
    SLOT_BACKENDS,
    MatchedProblem,
    _clear_tensor_grads,
    _gradient_diagnostic,
    _metadata,
    _nvidia_smi_snapshot,
    _peak_memory,
    _relative_error,
    _time_interleaved,
    make_problem,
)
from intertwiner_schurscan import ScanBackend
from schurscan_comoving_delta import (
    cumulative_actions,
    reads_to_physical_frame,
    values_to_comoving_frame,
)
from schurscan_delta_memory import (
    compose_delta,
    delta_read,
    delta_write_transitions,
    scanned_delta_states,
    value_transport_transitions,
)
from spin8_learned_address import DIMENSION, KEYS, SLOTS
from spin8_triality_memory import SlotTransition, packed_homogeneous_slot_scan

try:
    from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule
except ImportError as error:  # pragma: no cover - Linux external tier only
    raise SystemExit(
        "FLA is unavailable. Use the isolated supported Linux environment."
    ) from error


VARIANTS = (
    "direct_slot",
    "native_local_delta",
    "comoving_local_delta",
    "comoving_fla_chunk",
    "comoving_fla_recurrent",
)
TUNED_VARIANTS = VARIANTS
STATE_SCALARS = {
    "direct_slot": 64,
    "native_local_delta": 64,
    "comoving_local_delta": 128,
    "comoving_fla_chunk": 128,
    "comoving_fla_recurrent": 128,
}


def direct_generic(
    values: torch.Tensor,
    routes: torch.Tensor,
    query_routes: torch.Tensor,
    actions: torch.Tensor,
    *,
    backend: str,
) -> torch.Tensor:
    transition = SlotTransition(
        retention=1.0 - routes,
        action=actions,
        drive=routes[..., None] * values[:, :, None],
    )
    initial = torch.zeros(
        values.shape[0], SLOTS, DIMENSION, dtype=values.dtype, device=values.device
    )
    states = packed_homogeneous_slot_scan(
        transition, initial, backend=backend
    )
    return (query_routes[..., None] * states).sum(dim=2)


def native_delta_generic(
    values: torch.Tensor,
    keys: torch.Tensor,
    query_keys: torch.Tensor,
    actions: torch.Tensor,
    *,
    chunk_size: int,
    beta: torch.Tensor | None = None,
) -> torch.Tensor:
    if beta is None:
        beta = torch.ones(values.shape[:2], dtype=values.dtype, device=values.device)
    write = delta_write_transitions(keys, values, beta)
    transport = value_transport_transitions(actions, key_dimension=KEYS)
    transition = compose_delta(write, transport)
    initial = torch.zeros(
        values.shape[0], KEYS, DIMENSION, dtype=values.dtype, device=values.device
    )
    states = scanned_delta_states(
        transition, initial, backend="chunkwise", chunk_size=chunk_size
    )
    return delta_read(states, query_keys)


def comoving_inputs(
    values: torch.Tensor,
    actions: torch.Tensor,
    *,
    action_backend: ScanBackend,
) -> tuple[torch.Tensor, torch.Tensor]:
    # FLA requires fp16/bfloat16, but accumulating nominally orthogonal action
    # words in fp16 makes transpose cease to approximate inverse at long
    # lengths.  The general inverse-frame compiler is evaluated in fp32, then
    # only the DeltaRule payload is cast back to the external kernel dtype.
    accumulation_dtype = (
        torch.float32
        if values.dtype in (torch.float16, torch.bfloat16)
        else values.dtype
    )
    prefixes = cumulative_actions(
        actions.to(dtype=accumulation_dtype), backend=action_backend
    )
    transformed = values_to_comoving_frame(
        values.to(dtype=accumulation_dtype), prefixes
    )
    return transformed.to(dtype=values.dtype), prefixes


def comoving_local_generic(
    values: torch.Tensor,
    keys: torch.Tensor,
    query_keys: torch.Tensor,
    actions: torch.Tensor,
    *,
    action_backend: ScanBackend,
    chunk_size: int,
    beta: torch.Tensor | None = None,
) -> torch.Tensor:
    transformed, prefixes = comoving_inputs(
        values, actions, action_backend=action_backend
    )
    if beta is None:
        beta = torch.ones(values.shape[:2], dtype=values.dtype, device=values.device)
    transition = delta_write_transitions(keys, transformed, beta)
    initial = torch.zeros(
        values.shape[0], KEYS, DIMENSION, dtype=values.dtype, device=values.device
    )
    states = scanned_delta_states(
        transition, initial, backend="chunkwise", chunk_size=chunk_size
    )
    return reads_to_physical_frame(
        delta_read(states, query_keys).to(dtype=prefixes.dtype), prefixes
    )


def comoving_fla_chunk_generic(
    values: torch.Tensor,
    keys: torch.Tensor,
    query_keys: torch.Tensor,
    actions: torch.Tensor,
    *,
    action_backend: ScanBackend,
    chunk_size: int,
) -> torch.Tensor:
    transformed, prefixes = comoving_inputs(
        values, actions, action_backend=action_backend
    )
    beta = torch.ones(*values.shape[:2], 1, dtype=values.dtype, device=values.device)
    output, _ = chunk_delta_rule(
        query_keys[:, :, None],
        keys[:, :, None],
        transformed[:, :, None],
        beta,
        scale=1.0,
        chunk_size=chunk_size,
    )
    return reads_to_physical_frame(
        output[:, :, 0].to(dtype=prefixes.dtype), prefixes
    )


def comoving_fla_recurrent_generic(
    values: torch.Tensor,
    keys: torch.Tensor,
    query_keys: torch.Tensor,
    actions: torch.Tensor,
    *,
    action_backend: ScanBackend,
) -> torch.Tensor:
    transformed, prefixes = comoving_inputs(
        values, actions, action_backend=action_backend
    )
    beta = torch.ones(*values.shape[:2], 1, dtype=values.dtype, device=values.device)
    output, _ = fused_recurrent_delta_rule(
        query_keys[:, :, None],
        keys[:, :, None],
        transformed[:, :, None],
        beta,
        scale=1.0,
    )
    return reads_to_physical_frame(
        output[:, :, 0].to(dtype=prefixes.dtype), prefixes
    )


def functions(
    problem: MatchedProblem,
    *,
    selection: dict[str, str],
) -> dict[
    str,
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
]:
    local_action, local_chunk = selection["comoving_local_delta"].split(":")
    fla_action, fla_chunk = selection["comoving_fla_chunk"].split(":")
    return {
        "direct_slot": lambda values, keys, query, actions: direct_generic(
            values, keys, query, actions, backend=selection["direct_slot"]
        ),
        "native_local_delta": lambda values, keys, query, actions: (
            native_delta_generic(
                values,
                keys,
                query,
                actions,
                chunk_size=int(selection["native_local_delta"]),
            )
        ),
        "comoving_local_delta": lambda values, keys, query, actions: (
            comoving_local_generic(
                values,
                keys,
                query,
                actions,
                action_backend=local_action,
                chunk_size=int(local_chunk),
            )
        ),
        "comoving_fla_chunk": lambda values, keys, query, actions: (
            comoving_fla_chunk_generic(
                values,
                keys,
                query,
                actions,
                action_backend=fla_action,
                chunk_size=int(fla_chunk),
            )
        ),
        "comoving_fla_recurrent": lambda values, keys, query, actions: (
            comoving_fla_recurrent_generic(
                values,
                keys,
                query,
                actions,
                action_backend=selection["comoving_fla_recurrent"],
            )
        ),
    }


def _candidate_functions(
    problem: MatchedProblem,
) -> dict[str, dict[str, Callable[[], torch.Tensor]]]:
    values, keys, query, actions = (
        problem.values,
        problem.keys,
        problem.query_keys,
        problem.negative_actions,
    )
    action_backends = ("hillis_steele", "work_efficient")
    local_chunks = tuple(size for size in DELTA_CHUNK_SIZES if size <= 128)
    return {
        "direct_slot": {
            backend: (
                lambda backend=backend: direct_generic(
                    values, keys, query, actions, backend=backend
                )
            )
            for backend in SLOT_BACKENDS
        },
        "native_local_delta": {
            str(chunk): (
                lambda chunk=chunk: native_delta_generic(
                    values, keys, query, actions, chunk_size=chunk
                )
            )
            for chunk in local_chunks
        },
        "comoving_local_delta": {
            f"{backend}:{chunk}": (
                lambda backend=backend, chunk=chunk: comoving_local_generic(
                    values,
                    keys,
                    query,
                    actions,
                    action_backend=backend,
                    chunk_size=chunk,
                )
            )
            for backend in action_backends
            for chunk in local_chunks
        },
        "comoving_fla_chunk": {
            f"{backend}:{chunk}": (
                lambda backend=backend, chunk=chunk: comoving_fla_chunk_generic(
                    values,
                    keys,
                    query,
                    actions,
                    action_backend=backend,
                    chunk_size=chunk,
                )
            )
            for backend in action_backends
            for chunk in (16, 32, 64)
        },
        "comoving_fla_recurrent": {
            backend: (
                lambda backend=backend: comoving_fla_recurrent_generic(
                    values,
                    keys,
                    query,
                    actions,
                    action_backend=backend,
                )
            )
            for backend in action_backends
        },
    }


def tune(
    problem: MatchedProblem,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, str], dict[str, object]]:
    selection: dict[str, str] = {}
    report: dict[str, object] = {}
    for name, candidates in _candidate_functions(problem).items():
        timings, orders, samples = _time_interleaved(
            candidates,
            device=device,
            warmup=warmup,
            repeats=repeats,
            replications=1,
        )
        winner = min(timings, key=lambda candidate: timings[candidate].median_ms)
        selection[name] = winner
        report[name] = {
            "selected": winner,
            "candidate_ms": {
                candidate: asdict(summary) for candidate, summary in timings.items()
            },
            "candidate_samples_ms": samples,
            "timing_orders": orders,
        }
    return selection, report


def constrained_local_exact_gate(seed: int = 20260810) -> dict[str, object]:
    dtype = torch.float64
    generator = torch.Generator().manual_seed(seed)
    batch, length = 2, 19
    values = torch.randn(batch, length, DIMENSION, generator=generator, dtype=dtype)
    keys = torch.randn(batch, length, KEYS, generator=generator, dtype=dtype)
    query = torch.randn(batch, length, KEYS, generator=generator, dtype=dtype)
    beta = torch.sigmoid(
        torch.randn(batch, length, generator=generator, dtype=dtype)
    )
    raw = 0.03 * torch.randn(
        batch, length, DIMENSION, DIMENSION, generator=generator, dtype=dtype
    )
    tensors = (values, keys, query, beta, raw)
    gradients = []
    outputs = []
    for comoving in (False, True):
        leaves = tuple(tensor.detach().clone().requires_grad_(True) for tensor in tensors)
        v, k, q, b, coordinates = leaves
        actions = torch.matrix_exp(coordinates - coordinates.transpose(-1, -2))
        if comoving:
            output = comoving_local_generic(
                v,
                k,
                q,
                actions,
                action_backend="work_efficient",
                chunk_size=8,
                beta=b,
            )
        else:
            output = native_delta_generic(
                v, k, q, actions, chunk_size=8, beta=b
            )
        weights = torch.linspace(0.3, 1.3, output.numel(), dtype=dtype).reshape_as(
            output
        )
        outputs.append(output.detach())
        gradients.append(torch.autograd.grad((output * weights).sum(), leaves))
    gradient_errors = {
        name: _relative_error(actual, expected)
        for name, actual, expected in zip(
            ("values", "keys", "queries", "beta", "skew_coordinates"),
            gradients[1],
            gradients[0],
        )
    }
    output_error = _relative_error(outputs[1], outputs[0])
    passed = output_error < 2e-10 and max(gradient_errors.values()) < 2e-10
    return {
        "forward_relative_error": output_error,
        "gradient_relative_error": gradient_errors,
        "orthogonal_action_parameterization": "matrix_exp(raw - raw.T)",
        "passed": passed,
    }


def gradient_parity(
    left: Callable[..., torch.Tensor],
    right: Callable[..., torch.Tensor],
    problem: MatchedProblem,
) -> dict[str, float]:
    gradients = []
    source = (
        problem.values,
        problem.keys,
        problem.query_keys,
        problem.negative_actions,
    )
    for function in (left, right):
        tensors = tuple(tensor.detach().clone().requires_grad_(True) for tensor in source)
        output = function(*tensors)
        weights = torch.linspace(
            0.25, 1.25, output.numel(), dtype=output.dtype, device=output.device
        ).reshape_as(output)
        gradients.append(torch.autograd.grad((output * weights).sum(), tensors))
    return {
        name: _relative_error(actual.float(), expected.float())
        for name, actual, expected in zip(
            ("values", "keys", "queries", "action_matrices"),
            gradients[1],
            gradients[0],
        )
    }


def measure(
    forward_functions: dict[str, Callable[..., torch.Tensor]],
    *,
    problem: MatchedProblem,
    device: torch.device,
    warmup: int,
    repeats: int,
    backward_repeats: int,
    timing_blocks: int,
) -> dict[str, object]:
    source = (
        problem.values,
        problem.keys,
        problem.query_keys,
        problem.negative_actions,
    )
    forward_calls: dict[str, Callable[[], torch.Tensor]] = {}
    backward_calls: dict[str, Callable[[], None]] = {}
    gradient_tensors: dict[str, tuple[torch.Tensor, ...]] = {}
    for name, function in forward_functions.items():

        def forward_call(function: Callable[..., torch.Tensor] = function) -> torch.Tensor:
            with torch.no_grad():
                return function(*source)

        tensors = tuple(tensor.detach().clone().requires_grad_(True) for tensor in source)

        def backward_call(
            function: Callable[..., torch.Tensor] = function,
            tensors: tuple[torch.Tensor, ...] = tensors,
        ) -> None:
            _clear_tensor_grads(tensors)
            function(*tensors).float().square().mean().backward()

        forward_calls[name] = forward_call
        backward_calls[name] = backward_call
        gradient_tensors[name] = tensors

    forward_timings, forward_orders, forward_samples = _time_interleaved(
        forward_calls,
        device=device,
        warmup=warmup,
        repeats=repeats,
        replications=timing_blocks,
    )
    backward_timings, backward_orders, backward_samples = _time_interleaved(
        backward_calls,
        device=device,
        warmup=max(1, warmup // 2),
        repeats=backward_repeats,
        replications=timing_blocks,
    )
    forward_memory = {
        name: _peak_memory(call, device=device) for name, call in forward_calls.items()
    }
    backward_memory = {
        name: _peak_memory(
            backward_calls[name],
            device=device,
            clear_before=lambda tensors=gradient_tensors[name]: _clear_tensor_grads(
                tensors
            ),
        )
        for name in VARIANTS
    }
    gradient_diagnostics = {
        name: _gradient_diagnostic(
            backward_calls[name], gradient_tensors[name], device=device
        )
        for name in VARIANTS
    }
    tokens = problem.values.shape[0] * problem.values.shape[1]
    return {
        "forward_ms": {
            name: asdict(summary) for name, summary in forward_timings.items()
        },
        "forward_samples_ms": forward_samples,
        "forward_backward_ms": {
            name: asdict(summary) for name, summary in backward_timings.items()
        },
        "forward_backward_samples_ms": backward_samples,
        "forward_tokens_per_second": {
            name: 1000.0 * tokens / summary.median_ms
            for name, summary in forward_timings.items()
        },
        "forward_cuda_memory": forward_memory,
        "forward_backward_cuda_memory": backward_memory,
        "gradient_diagnostics": gradient_diagnostics,
        "timing_orders": {
            "forward_block_base_orders": forward_orders,
            "forward_backward_block_base_orders": backward_orders,
            "within_block": "cyclic rotation each repeat",
        },
    }


def benchmark(
    *,
    device: torch.device,
    batch: int,
    lengths: Sequence[int],
    warmup: int,
    repeats: int,
    backward_repeats: int,
    timing_blocks: int,
    tuning_repeats: int,
    seed: int = 20260810,
    frozen_selection: dict[str, object] | None = None,
) -> dict[str, object]:
    dtype = torch.float16
    metadata = _metadata(device)
    metadata.update(
        {
            "platform": platform.platform(),
            "fla_core": importlib.metadata.version("fla-core"),
            "flash_linear_attention": importlib.metadata.version(
                "flash-linear-attention"
            ),
            "triton": importlib.metadata.version("triton"),
            "nvidia_smi_before": _nvidia_smi_snapshot(),
        }
    )
    local_gate = constrained_local_exact_gate(seed)
    if not local_gate["passed"]:
        raise AssertionError(f"local exact gate failed: {local_gate}")
    rows = []
    forward_tolerance = 5e-2
    gradient_tolerance = 1.2e-1
    for length in lengths:
        if frozen_selection is None:
            tuning_problem = make_problem(
                batch=batch,
                length=length,
                dtype=dtype,
                device=device,
                seed=seed + 1_000_000,
            )
            selection, tuning = tune(
                tuning_problem,
                device=device,
                warmup=max(1, warmup // 2),
                repeats=tuning_repeats,
            )
            del tuning_problem
        else:
            frozen_row = frozen_selection["selections"][str(length)]
            selection = {
                name: str(value) for name, value in frozen_row["implementations"].items()
            }
            tuning = {
                "mode": "externally_frozen",
                "selection_rule": frozen_selection["selection_rule"],
                "selection_diagnostics": frozen_row["diagnostics"],
            }
        problem = make_problem(
            batch=batch, length=length, dtype=dtype, device=device, seed=seed
        )
        forward_functions = functions(problem, selection=selection)
        with torch.no_grad():
            outputs = {name: function(
                problem.values,
                problem.keys,
                problem.query_keys,
                problem.negative_actions,
            ) for name, function in forward_functions.items()}
        differences = {
            name: _relative_error(output.float(), outputs["direct_slot"].float())
            for name, output in outputs.items()
            if name != "direct_slot"
        }
        gradients = gradient_parity(
            forward_functions["comoving_local_delta"],
            forward_functions["comoving_fla_chunk"],
            problem,
        )
        if max(differences.values()) >= forward_tolerance:
            raise AssertionError(
                f"co-moving forward parity failed at length {length}: {differences}"
            )
        if max(gradients.values()) >= gradient_tolerance:
            raise AssertionError(
                f"co-moving FLA gradient parity failed at length {length}: {gradients}"
            )
        measurements = measure(
            forward_functions,
            problem=problem,
            device=device,
            warmup=warmup,
            repeats=repeats,
            backward_repeats=backward_repeats,
            timing_blocks=timing_blocks,
        )
        gradients_finite = all(
            bool(diagnostic[gate])
            for diagnostic in measurements["gradient_diagnostics"].values()
            for gate in ("all_present", "all_finite")
        )
        rows.append(
            {
                "length": length,
                "batch": batch,
                "selected_implementations": selection,
                "tuning": tuning,
                "forward_relative_error_vs_direct": differences,
                "fla_chunk_vs_comoving_local_gradient_relative_error": gradients,
                **measurements,
                "correctness_passed": gradients_finite,
            }
        )
    metadata["nvidia_smi_after"] = _nvidia_smi_snapshot()
    return {
        "experiment": "full-path co-moving fused transported DeltaRule benchmark",
        "device_metadata": metadata,
        "local_float64_exact_gate": local_gate,
        "protocol": {
            "dtype": "float16",
            "batch": batch,
            "lengths": list(lengths),
            "state_scalars": STATE_SCALARS,
            "input_generation_timed": False,
            "action_prefixes_precomputed": False,
            "cumulative_action_scan_timed": True,
            "value_frame_transforms_timed": True,
            "physical_read_frame_transform_timed": True,
            "noncommuting_value_transport": True,
            "warmup": warmup,
            "forward_repeats": repeats,
            "forward_backward_repeats": backward_repeats,
            "same_process_timing_blocks": timing_blocks,
            "forward_samples_per_variant_and_length": repeats * timing_blocks,
            "forward_backward_samples_per_variant_and_length": (
                backward_repeats * timing_blocks
            ),
            "selection_mode": (
                "in_process_disjoint_tuning"
                if frozen_selection is None
                else "externally_frozen_disjoint_tuning"
            ),
            "tuning_repeats_in_this_process": (
                tuning_repeats if frozen_selection is None else 0
            ),
            "problem_seed": seed,
            "tuning_problem_seed": seed + 1_000_000,
            "frozen_selection_inputs": (
                None if frozen_selection is None else frozen_selection["inputs"]
            ),
        },
        "rows": rows,
        "claim_boundary": {
            "official_fla_ops_compared": True,
            "noncommuting_transport_compiled_exactly": True,
            "state_matched_comparison": False,
            "spin8_specific_capacity_advantage_established": False,
            "absolute_architecture_winner_established": False,
        },
        "passed": all(bool(row["correctness_passed"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lengths", type=int, nargs="+", default=(256, 1024, 4096))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--backward-repeats", type=int, default=100)
    parser.add_argument("--timing-blocks", type=int, default=1)
    parser.add_argument("--tuning-repeats", type=int, default=5)
    parser.add_argument("--selection-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    frozen_selection = (
        None
        if args.selection_config is None
        else json.loads(args.selection_config.read_text(encoding="utf-8"))
    )
    report = benchmark(
        device=torch.device(args.device),
        batch=args.batch,
        lengths=tuple(args.lengths),
        warmup=args.warmup,
        repeats=args.repeats,
        backward_repeats=args.backward_repeats,
        timing_blocks=args.timing_blocks,
        tuning_repeats=args.tuning_repeats,
        frozen_selection=frozen_selection,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
