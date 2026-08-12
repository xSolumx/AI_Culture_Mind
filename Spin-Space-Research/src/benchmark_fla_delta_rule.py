"""Linux FLA benchmark for a state-matched, transport-free delta recurrence.

This is the external fused systems tier.  It compares direct and
triality-bound slots, the local eager two-level delta scan, FLA's compact-WY
chunk kernel, and FLA's fused recurrent kernel on the identical hard-key
overwrite recurrence.  All rows have 64 recurrent scalars.  Noncommuting
value transport is disabled because the standard FLA delta-rule operator has
no per-token value-axis action argument.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.nn import functional as F

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
    direct_forward,
    make_problem,
    triality_forward,
)
from schurscan_delta_memory import (
    delta_read,
    delta_write_transitions,
    scanned_delta_states,
)
from spin8_learned_address import DIMENSION, KEYS

try:
    from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule
except ImportError as error:  # pragma: no cover - exercised only without FLA
    raise SystemExit(
        "FLA is unavailable. Install flash-linear-attention[cuda] in a "
        "supported Linux environment."
    ) from error


VARIANTS = (
    "direct_slot",
    "triality_slot",
    "local_eager_delta",
    "fla_chunk_delta",
    "fla_fused_recurrent_delta",
)


def transport_free_problem(problem: MatchedProblem, *, seed: int) -> MatchedProblem:
    batch, length = problem.values.shape[:2]
    identity = torch.eye(
        DIMENSION, dtype=problem.values.dtype, device=problem.values.device
    ).reshape(1, 1, DIMENSION, DIMENSION)
    identity = identity.expand(batch, length, -1, -1)

    generator = torch.Generator().manual_seed(700_000 + seed + length)
    geometric_table = F.normalize(
        torch.randn(batch, KEYS, DIMENSION, generator=generator, dtype=torch.float64),
        dim=-1,
    ).to(dtype=problem.values.dtype, device=problem.values.device)
    write_labels = problem.routes.argmax(dim=-1)
    query_labels = problem.query_routes.argmax(dim=-1)

    def select(labels: torch.Tensor) -> torch.Tensor:
        return geometric_table.gather(1, labels[..., None].expand(-1, -1, DIMENSION))

    return replace(
        problem,
        vector_actions=identity,
        negative_actions=identity,
        write_geometric_keys=select(write_labels),
        query_geometric_keys=select(query_labels),
    )


def local_delta_forward(
    values: torch.Tensor,
    keys: torch.Tensor,
    query_keys: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    beta = torch.ones(values.shape[:2], dtype=values.dtype, device=values.device)
    transition = delta_write_transitions(keys, values, beta)
    initial = torch.zeros(
        values.shape[0], KEYS, DIMENSION, dtype=values.dtype, device=values.device
    )
    states = scanned_delta_states(
        transition, initial, backend="chunkwise", chunk_size=chunk_size
    )
    return delta_read(states, query_keys)


def fla_chunk_forward(
    values: torch.Tensor,
    keys: torch.Tensor,
    query_keys: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    beta = torch.ones(*values.shape[:2], 1, dtype=values.dtype, device=values.device)
    output, _ = chunk_delta_rule(
        query_keys[:, :, None],
        keys[:, :, None],
        values[:, :, None],
        beta,
        scale=1.0,
        chunk_size=chunk_size,
    )
    return output[:, :, 0]


def fla_recurrent_forward(
    values: torch.Tensor,
    keys: torch.Tensor,
    query_keys: torch.Tensor,
) -> torch.Tensor:
    beta = torch.ones(*values.shape[:2], 1, dtype=values.dtype, device=values.device)
    output, _ = fused_recurrent_delta_rule(
        query_keys[:, :, None],
        keys[:, :, None],
        values[:, :, None],
        beta,
        scale=1.0,
    )
    return output[:, :, 0]


def functions(
    problem: MatchedProblem,
    *,
    slot_backends: dict[str, str],
    local_chunk_size: int,
    fla_chunk_size: int,
) -> dict[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]:
    return {
        "direct_slot": lambda values, keys, query_keys: direct_forward(
            problem,
            values,
            routes=keys,
            query_routes=query_keys,
            backend=slot_backends["direct_slot"],
        ),
        "triality_slot": lambda values, keys, query_keys: triality_forward(
            problem,
            values,
            routes=keys,
            query_routes=query_keys,
            backend=slot_backends["triality_slot"],
        ),
        "local_eager_delta": lambda values, keys, query_keys: local_delta_forward(
            values, keys, query_keys, chunk_size=local_chunk_size
        ),
        "fla_chunk_delta": lambda values, keys, query_keys: fla_chunk_forward(
            values, keys, query_keys, chunk_size=fla_chunk_size
        ),
        "fla_fused_recurrent_delta": fla_recurrent_forward,
    }


def tune(
    problem: MatchedProblem,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, str], int, int, dict[str, object]]:
    report: dict[str, object] = {}
    slot_backends: dict[str, str] = {}
    for name, implementation in (
        ("direct_slot", direct_forward),
        ("triality_slot", triality_forward),
    ):
        candidates = {}
        for backend in SLOT_BACKENDS:

            def call(
                implementation: Callable[..., torch.Tensor] = implementation,
                backend: str = backend,
            ) -> torch.Tensor:
                with torch.no_grad():
                    return implementation(
                        problem,
                        problem.values,
                        routes=problem.keys,
                        query_routes=problem.query_keys,
                        backend=backend,
                    )

            candidates[backend] = call
        timings, orders, samples = _time_interleaved(
            candidates,
            device=device,
            warmup=warmup,
            repeats=repeats,
            replications=1,
        )
        winner = min(timings, key=lambda candidate: timings[candidate].median_ms)
        slot_backends[name] = winner
        report[name] = {
            "selected": winner,
            "candidate_ms": {
                candidate: asdict(summary) for candidate, summary in timings.items()
            },
            "candidate_samples_ms": samples,
            "timing_orders": orders,
        }

    local_candidates = {
        str(chunk_size): (
            lambda chunk_size=chunk_size: local_delta_forward(
                problem.values,
                problem.keys,
                problem.query_keys,
                chunk_size=chunk_size,
            )
        )
        for chunk_size in DELTA_CHUNK_SIZES
    }
    local_timings, local_orders, local_samples = _time_interleaved(
        local_candidates,
        device=device,
        warmup=warmup,
        repeats=repeats,
        replications=1,
    )
    local_winner = min(
        local_timings, key=lambda candidate: local_timings[candidate].median_ms
    )
    report["local_eager_delta"] = {
        "selected": int(local_winner),
        "candidate_ms": {
            candidate: asdict(summary) for candidate, summary in local_timings.items()
        },
        "candidate_samples_ms": local_samples,
        "timing_orders": local_orders,
    }

    fla_candidates = {
        str(chunk_size): (
            lambda chunk_size=chunk_size: fla_chunk_forward(
                problem.values,
                problem.keys,
                problem.query_keys,
                chunk_size=chunk_size,
            )
        )
        for chunk_size in (16, 32, 64)
    }
    fla_timings, fla_orders, fla_samples = _time_interleaved(
        fla_candidates,
        device=device,
        warmup=warmup,
        repeats=repeats,
        replications=1,
    )
    fla_winner = min(
        fla_timings, key=lambda candidate: fla_timings[candidate].median_ms
    )
    report["fla_chunk_delta"] = {
        "selected": int(fla_winner),
        "candidate_ms": {
            candidate: asdict(summary) for candidate, summary in fla_timings.items()
        },
        "candidate_samples_ms": fla_samples,
        "timing_orders": fla_orders,
    }
    return slot_backends, int(local_winner), int(fla_winner), report


def measure(
    forward_functions: dict[
        str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    ],
    *,
    problem: MatchedProblem,
    device: torch.device,
    warmup: int,
    repeats: int,
    backward_repeats: int,
    timing_blocks: int,
) -> dict[str, object]:
    forward_calls: dict[str, Callable[[], torch.Tensor | None]] = {}
    backward_calls: dict[str, Callable[[], torch.Tensor | None]] = {}
    gradient_tensors: dict[str, tuple[torch.Tensor, ...]] = {}
    for name, function in forward_functions.items():

        def forward_call(
            function: Callable[..., torch.Tensor] = function,
        ) -> torch.Tensor:
            with torch.no_grad():
                return function(problem.values, problem.keys, problem.query_keys)

        tensors = tuple(
            tensor.detach().clone().requires_grad_(True)
            for tensor in (problem.values, problem.keys, problem.query_keys)
        )

        def backward_call(
            function: Callable[..., torch.Tensor] = function,
            tensors: tuple[torch.Tensor, ...] = tensors,
        ) -> None:
            _clear_tensor_grads(tensors)
            output = function(*tensors)
            output.float().square().mean().backward()

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


def gradient_parity(
    left: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    right: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    problem: MatchedProblem,
) -> dict[str, float]:
    gradients = []
    for function in (left, right):
        tensors = tuple(
            tensor.detach().clone().requires_grad_(True)
            for tensor in (problem.values, problem.keys, problem.query_keys)
        )
        output = function(*tensors)
        weights = torch.linspace(
            0.25,
            1.25,
            output.numel(),
            dtype=output.dtype,
            device=output.device,
        ).reshape_as(output)
        gradients.append(torch.autograd.grad((output * weights).sum(), tensors))
    return {
        name: _relative_error(actual.float(), expected.float())
        for name, actual, expected in zip(
            ("values", "write_keys", "query_keys"), gradients[1], gradients[0]
        )
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
    rows = []
    tolerance = 3e-2
    gradient_tolerance = 8e-2
    for length in lengths:
        if frozen_selection is None:
            tuning_problem = transport_free_problem(
                make_problem(
                    batch=batch,
                    length=length,
                    dtype=dtype,
                    device=device,
                    seed=seed + 1_000_000,
                ),
                seed=seed + 1_000_000,
            )
            slot_backends, local_chunk_size, fla_chunk_size, tuning = tune(
                tuning_problem,
                device=device,
                warmup=max(1, warmup // 2),
                repeats=tuning_repeats,
            )
            del tuning_problem
        else:
            selection = frozen_selection["selections"][str(length)]
            slot_backends = {
                name: str(value) for name, value in selection["slot_backends"].items()
            }
            local_chunk_size = int(selection["local_delta_chunk_size"])
            fla_chunk_size = int(selection["fla_delta_chunk_size"])
            tuning = {
                "mode": "externally_frozen",
                "selection_rule": frozen_selection["selection_rule"],
                "selection_diagnostics": selection["diagnostics"],
            }
        problem = transport_free_problem(
            make_problem(
                batch=batch,
                length=length,
                dtype=dtype,
                device=device,
                seed=seed,
            ),
            seed=seed,
        )
        forward_functions = functions(
            problem,
            slot_backends=slot_backends,
            local_chunk_size=local_chunk_size,
            fla_chunk_size=fla_chunk_size,
        )
        with torch.no_grad():
            outputs = {
                name: function(problem.values, problem.keys, problem.query_keys)
                for name, function in forward_functions.items()
            }
        differences = {
            name: _relative_error(output.float(), outputs["direct_slot"].float())
            for name, output in outputs.items()
            if name != "direct_slot"
        }
        gradients = gradient_parity(
            forward_functions["local_eager_delta"],
            forward_functions["fla_chunk_delta"],
            problem,
        )
        if max(differences.values()) >= tolerance:
            raise AssertionError(
                f"FLA forward parity failed at length {length}: {differences}"
            )
        if max(gradients.values()) >= gradient_tolerance:
            raise AssertionError(
                f"FLA gradient parity failed at length {length}: {gradients}"
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
                "selected_implementations": {
                    "slot_backends": slot_backends,
                    "local_delta_chunk_size": local_chunk_size,
                    "fla_delta_chunk_size": fla_chunk_size,
                },
                "tuning": tuning,
                "forward_relative_error_vs_direct": differences,
                "fla_chunk_vs_local_gradient_relative_error": gradients,
                **measurements,
                "correctness_passed": gradients_finite,
            }
        )
    metadata["nvidia_smi_after"] = _nvidia_smi_snapshot()
    return {
        "experiment": "state-matched external FLA delta-rule benchmark",
        "device_metadata": metadata,
        "protocol": {
            "dtype": "float16",
            "batch": batch,
            "lengths": list(lengths),
            "state_scalars": {name: 64 for name in VARIANTS},
            "input_generation_timed": False,
            "address_encoding_timed": False,
            "transition_construction_timed_for_local_rows": True,
            "fla_operator_inputs_preconstructed": True,
            "noncommuting_value_transport": False,
            "fla_scale": 1.0,
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
            "compact_wy_chunk_compared": True,
            "fused_recurrent_compared": True,
            "gated_delta_net_2_layer_compared": False,
            "noncommuting_value_transport_supported_by_fla_row": False,
            "encoder_inclusive_comparison": False,
            "absolute_architecture_winner_established": False,
        },
        "passed": all(bool(row["correctness_passed"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=(64, 256, 1024, 2048, 4096)
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--backward-repeats", type=int, default=5)
    parser.add_argument("--timing-blocks", type=int, default=5)
    parser.add_argument("--tuning-repeats", type=int, default=7)
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
