"""Reusable numerical and streaming audits for the maintained v1.4 modules.

Every public audit consumes either explicit float64 transition data or a
supplied, already constructed model/tier.  Reports contain only JSON-native
values; the audit code never substitutes a newly randomized model for a
checkpoint.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

try:
    from .model import (
        AttentionState,
        DeltaProductState,
        HybridMemoryLM,
        SelectedBlockState,
    )
    from .selected_block import SelectedBlockMemory
    from .structured_tier import StructuredSpin8Tier, StructuredTierConfig
except ImportError:  # Support loading this file directly under a distinct name.
    from hybrid_memory_v1_4.model import (
        AttentionState,
        DeltaProductState,
        HybridMemoryLM,
        SelectedBlockState,
    )
    from hybrid_memory_v1_4.selected_block import SelectedBlockMemory
    from hybrid_memory_v1_4.structured_tier import (
        StructuredSpin8Tier,
        StructuredTierConfig,
    )


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _finite_float(value: torch.Tensor | float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _max_abs_error(actual: torch.Tensor, reference: torch.Tensor) -> float | None:
    if actual.shape != reference.shape:
        raise ValueError("compared tensors must have identical shapes")
    if not bool(torch.isfinite(actual).all()) or not bool(
        torch.isfinite(reference).all()
    ):
        return None
    return float((actual.double() - reference.double()).abs().max())


def _nonfinite_count(tensor: torch.Tensor) -> int:
    return int((~torch.isfinite(tensor)).sum())


def _default_checkpoints(horizon: int) -> list[int]:
    checkpoints = {1, horizon}
    position = 2
    while position < horizon:
        checkpoints.add(position)
        position *= 2
    checkpoints.update(
        position
        for position in (horizon // 4, horizon // 2, 3 * horizon // 4)
        if position > 0
    )
    return sorted(checkpoints)


def _validate_checkpoints(checkpoints: Sequence[int] | None, horizon: int) -> list[int]:
    if checkpoints is None:
        return _default_checkpoints(horizon)
    if isinstance(checkpoints, (str, bytes)):
        raise TypeError("checkpoints must be a sequence of integer positions")
    values = list(checkpoints)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("checkpoints must contain integers")
    if any(value < 1 or value > horizon for value in values):
        raise ValueError("checkpoints must lie in [1, horizon]")
    values.append(horizon)
    return sorted(set(values))


def _validate_affine_source(
    linear: torch.Tensor,
    drive: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    diagonal: bool,
) -> None:
    for name, tensor in (
        ("linear", linear),
        ("drive", drive),
        ("initial_state", initial_state),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.dtype != torch.float64:
            raise TypeError(f"{name} must be created in float64")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite")
    if linear.device != drive.device or linear.device != initial_state.device:
        raise ValueError("linear, drive, and initial_state must share a device")

    if diagonal:
        if linear.ndim != 4 or drive.ndim != 5:
            raise ValueError(
                "diagonal transitions require linear (B,L,H,N) and drive (B,L,H,N,V)"
            )
        batch, length, heads, slots = linear.shape
        if drive.shape[:4] != (batch, length, heads, slots):
            raise ValueError("diagonal linear and drive shapes are incompatible")
    else:
        if linear.ndim != 5 or drive.ndim != 5:
            raise ValueError(
                "matrix transitions require linear (B,L,H,N,N) and drive (B,L,H,N,V)"
            )
        batch, length, heads, slots, again = linear.shape
        if slots != again or drive.shape[:4] != (batch, length, heads, slots):
            raise ValueError("matrix linear and drive shapes are incompatible")
    if min(batch, length, heads, slots, drive.shape[-1]) < 1:
        raise ValueError("transition dimensions must be nonempty")
    if initial_state.shape != (batch, heads, slots, drive.shape[-1]):
        raise ValueError("initial_state must have shape (B,H,N,V)")


def _apply_affine(
    linear: torch.Tensor,
    drive: torch.Tensor,
    state: torch.Tensor,
    *,
    diagonal: bool,
) -> torch.Tensor:
    if diagonal:
        return linear[..., None] * state + drive
    return linear @ state + drive


def _compose_affine(
    later_linear: torch.Tensor,
    later_drive: torch.Tensor,
    earlier_linear: torch.Tensor,
    earlier_drive: torch.Tensor,
    *,
    diagonal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if diagonal:
        return (
            later_linear * earlier_linear,
            later_drive + later_linear[..., None] * earlier_drive,
        )
    return (
        later_linear @ earlier_linear,
        later_drive + later_linear @ earlier_drive,
    )


def _recurrent_affine_scan(
    linear: torch.Tensor,
    drive: torch.Tensor,
    initial_state: torch.Tensor,
    checkpoint_positions: set[int],
    *,
    diagonal: bool,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, int]:
    state = initial_state
    checkpoints: dict[int, torch.Tensor] = {}
    nonfinite = 0
    for index in range(linear.shape[0]):
        state = _apply_affine(linear[index], drive[index], state, diagonal=diagonal)
        nonfinite += _nonfinite_count(state)
        position = index + 1
        if position in checkpoint_positions:
            checkpoints[position] = state.clone()
    return checkpoints, state, nonfinite


def _blocked_parallel_affine_scan(
    linear: torch.Tensor,
    drive: torch.Tensor,
    initial_state: torch.Tensor,
    checkpoint_positions: set[int],
    *,
    diagonal: bool,
    chunk_size: int,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, int, int]:
    """Run bounded-memory Hillis-Steele scans over consecutive chunks."""

    state = initial_state
    checkpoints: dict[int, torch.Tensor] = {}
    state_nonfinite = 0
    composition_nonfinite = 0
    for start in range(0, linear.shape[0], chunk_size):
        stop = min(start + chunk_size, linear.shape[0])
        cumulative_linear = linear[start:stop].clone()
        cumulative_drive = drive[start:stop].clone()
        offset = 1
        while offset < stop - start:
            composed_linear, composed_drive = _compose_affine(
                cumulative_linear[offset:],
                cumulative_drive[offset:],
                cumulative_linear[:-offset],
                cumulative_drive[:-offset],
                diagonal=diagonal,
            )
            cumulative_linear = torch.cat(
                (cumulative_linear[:offset], composed_linear), dim=0
            )
            cumulative_drive = torch.cat(
                (cumulative_drive[:offset], composed_drive), dim=0
            )
            composition_nonfinite += _nonfinite_count(composed_linear)
            composition_nonfinite += _nonfinite_count(composed_drive)
            offset *= 2

        chunk_states = _apply_affine(
            cumulative_linear,
            cumulative_drive,
            state.unsqueeze(0),
            diagonal=diagonal,
        )
        state_nonfinite += _nonfinite_count(chunk_states)
        for position in checkpoint_positions:
            if start < position <= stop:
                checkpoints[position] = chunk_states[position - start - 1].clone()
        state = chunk_states[-1]
    return checkpoints, state, state_nonfinite, composition_nonfinite


def audit_affine_precision(
    linear: torch.Tensor,
    drive: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    diagonal: bool,
    checkpoints: Sequence[int] | None = None,
    parallel_chunk_size: int = 1024,
    name: str = "associative_affine",
    source_kind: str = "supplied_float64_affine",
    source_created_directly_in_float64: bool = False,
) -> dict[str, Any]:
    """Compare bounded-memory fp16/fp32 scans with an fp64 recurrence.

    The sequence axis is axis one in the supplied tensors.  The parallel path
    uses a Hillis-Steele scan within bounded chunks, so a long diagonal audit
    never allocates a full matrix-state trajectory.
    """

    if type(diagonal) is not bool:
        raise TypeError("diagonal must be a bool")
    if isinstance(parallel_chunk_size, bool) or not isinstance(
        parallel_chunk_size, int
    ):
        raise TypeError("parallel_chunk_size must be an integer")
    if parallel_chunk_size < 1:
        raise ValueError("parallel_chunk_size must be positive")
    _validate_affine_source(linear, drive, initial_state, diagonal=diagonal)
    horizon = linear.shape[1]
    positions = _validate_checkpoints(checkpoints, horizon)
    position_set = set(positions)
    sequence_linear = linear.movedim(1, 0)
    sequence_drive = drive.movedim(1, 0)

    with torch.no_grad():
        reference_checkpoints, reference_final, reference_nonfinite = (
            _recurrent_affine_scan(
                sequence_linear,
                sequence_drive,
                initial_state,
                position_set,
                diagonal=diagonal,
            )
        )
        paths: list[dict[str, Any]] = []
        for dtype in (torch.float32, torch.float16):
            candidate_linear = sequence_linear.to(dtype=dtype)
            candidate_drive = sequence_drive.to(dtype=dtype)
            candidate_initial = initial_state.to(dtype=dtype)
            for mode in ("recurrent", "parallel"):
                if mode == "recurrent":
                    candidate_checkpoints, candidate_final, state_nonfinite = (
                        _recurrent_affine_scan(
                            candidate_linear,
                            candidate_drive,
                            candidate_initial,
                            position_set,
                            diagonal=diagonal,
                        )
                    )
                    composition_nonfinite = 0
                else:
                    (
                        candidate_checkpoints,
                        candidate_final,
                        state_nonfinite,
                        composition_nonfinite,
                    ) = _blocked_parallel_affine_scan(
                        candidate_linear,
                        candidate_drive,
                        candidate_initial,
                        position_set,
                        diagonal=diagonal,
                        chunk_size=parallel_chunk_size,
                    )
                checkpoint_errors = [
                    _max_abs_error(
                        candidate_checkpoints[position],
                        reference_checkpoints[position],
                    )
                    for position in positions
                ]
                max_checkpoint_error = (
                    None
                    if any(error is None for error in checkpoint_errors)
                    else max(error for error in checkpoint_errors if error is not None)
                )
                paths.append(
                    {
                        "dtype": _dtype_name(dtype),
                        "mode": mode,
                        "max_checkpoint_error": max_checkpoint_error,
                        "final_error": _max_abs_error(candidate_final, reference_final),
                        "state_nonfinite_count": state_nonfinite,
                        "composition_nonfinite_count": composition_nonfinite,
                        "nonfinite_count": state_nonfinite + composition_nonfinite,
                    }
                )

    if diagonal:
        retention_min: float | None = float(linear.min())
        retention_max: float | None = float(linear.max())
    else:
        retention_min = None
        retention_max = None
    return {
        "name": name,
        "source_kind": source_kind,
        "source_dtype": "float64",
        "source_created_directly_in_float64": bool(source_created_directly_in_float64),
        "transition_representation": "diagonal" if diagonal else "matrix",
        "horizon": horizon,
        "checkpoint_positions": positions,
        "parallel_chunk_size": parallel_chunk_size,
        "retention_min": retention_min,
        "retention_max": retention_max,
        "reference": {
            "dtype": "float64",
            "mode": "recurrent",
            "nonfinite_count": reference_nonfinite,
            "final_l2_norm": _finite_float(torch.linalg.vector_norm(reference_final)),
        },
        "paths": paths,
    }


def audit_selected_block_precision(
    mixer: SelectedBlockMemory,
    inputs: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    checkpoints: Sequence[int] | None = None,
    parallel_chunk_size: int = 1024,
) -> dict[str, Any]:
    """Audit affine scans compiled by a supplied selected-block mixer."""

    if not isinstance(mixer, SelectedBlockMemory):
        raise TypeError("mixer must be a supplied SelectedBlockMemory")
    if not isinstance(inputs, torch.Tensor) or inputs.dtype != torch.float64:
        raise TypeError("selected-block inputs must be supplied in float64")
    parameter = next(mixer.parameters())
    if parameter.dtype != torch.float64:
        raise TypeError("selected-block mixer parameters must be float64")
    if initial_state is None:
        initial_state = mixer.initial_state(
            inputs.shape[0], device=inputs.device, dtype=torch.float64
        )
    if initial_state.dtype != torch.float64:
        raise TypeError("selected-block initial_state must be float64")
    with torch.no_grad():
        transition, _ = mixer.compile_semantic_transition(inputs)
        diagonal = transition.linear.diagonal(dim1=-2, dim2=-1)
        off_diagonal = transition.linear - torch.diag_embed(diagonal)
        off_diagonal_residual = float(off_diagonal.abs().max())
        if off_diagonal_residual != 0.0:
            raise ValueError("selected-block transition is not diagonal affine")
        flat_initial = initial_state.flatten(2, 3)
        report = audit_affine_precision(
            diagonal,
            transition.drive,
            flat_initial,
            diagonal=True,
            checkpoints=checkpoints,
            parallel_chunk_size=parallel_chunk_size,
            name="selected_block_diagonal_affine",
            source_kind="compiled_selected_block_transition",
        )
    report["off_diagonal_residual"] = off_diagonal_residual
    return report


def audit_delta_product_precision(
    mixer: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    checkpoints: Sequence[int] | None = None,
    parallel_chunk_size: int = 1024,
) -> dict[str, Any]:
    """Audit matrix-affine transitions from a supplied maintained DeltaProduct."""

    if not isinstance(mixer, torch.nn.Module) or not callable(
        getattr(mixer, "transitions", None)
    ):
        raise TypeError("mixer must expose the maintained DeltaProduct transitions API")
    if not isinstance(inputs, torch.Tensor) or inputs.dtype != torch.float64:
        raise TypeError("DeltaProduct inputs must be supplied in float64")
    parameter = next(mixer.parameters())
    if parameter.dtype != torch.float64:
        raise TypeError("DeltaProduct mixer parameters must be float64")
    with torch.no_grad():
        transition_linear, transition_drive, _ = mixer.transitions(inputs)
    if initial_state is None:
        initial_state = torch.zeros_like(transition_drive[:, 0])
    return audit_affine_precision(
        transition_linear,
        transition_drive,
        initial_state,
        diagonal=False,
        checkpoints=checkpoints,
        parallel_chunk_size=parallel_chunk_size,
        name="maintained_delta_product_matrix_affine",
        source_kind="compiled_maintained_delta_product_transition",
    )


def precision_horizon_audit(
    *,
    horizon: int = 65_536,
    retention: float = 0.9999,
    state_size: int = 8,
    value_size: int = 4,
    batch_size: int = 1,
    heads: int = 1,
    drive_scale: float = 1e-3,
    seed: int = 0,
    checkpoints: Sequence[int] | None = None,
    parallel_chunk_size: int = 1024,
    device: torch.device | str = "cpu",
    selected_block: SelectedBlockMemory | None = None,
    selected_inputs: torch.Tensor | None = None,
    delta_product: torch.nn.Module | None = None,
    delta_inputs: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Run a high-retention affine horizon audit plus supplied real mixers."""

    for name, value in (
        ("horizon", horizon),
        ("state_size", state_size),
        ("value_size", value_size),
        ("batch_size", batch_size),
        ("heads", heads),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if not isinstance(retention, (int, float)) or isinstance(retention, bool):
        raise TypeError("retention must be a real number")
    if not math.isfinite(float(retention)) or not 0.0 < retention < 1.0:
        raise ValueError(
            "retention must be finite and lie strictly between zero and one"
        )
    if not isinstance(drive_scale, (int, float)) or isinstance(drive_scale, bool):
        raise TypeError("drive_scale must be a real number")
    if not math.isfinite(float(drive_scale)) or drive_scale < 0.0:
        raise ValueError("drive_scale must be finite and nonnegative")
    if (selected_block is None) != (selected_inputs is None):
        raise ValueError("selected_block and selected_inputs must be supplied together")
    if (delta_product is None) != (delta_inputs is None):
        raise ValueError("delta_product and delta_inputs must be supplied together")

    target_device = torch.device(device)
    generator = torch.Generator(device=target_device)
    generator.manual_seed(seed)
    # These are allocated directly in float64; no lower-precision source is cast up.
    linear = torch.full(
        (batch_size, horizon, heads, state_size),
        float(retention),
        dtype=torch.float64,
        device=target_device,
    )
    drive = float(drive_scale) * torch.randn(
        batch_size,
        horizon,
        heads,
        state_size,
        value_size,
        dtype=torch.float64,
        device=target_device,
        generator=generator,
    )
    initial_state = torch.randn(
        batch_size,
        heads,
        state_size,
        value_size,
        dtype=torch.float64,
        device=target_device,
        generator=generator,
    )
    cases = [
        audit_affine_precision(
            linear,
            drive,
            initial_state,
            diagonal=True,
            checkpoints=checkpoints,
            parallel_chunk_size=parallel_chunk_size,
            name="high_retention_diagonal_affine",
            source_kind="generated_direct_float64_diagonal_affine",
            source_created_directly_in_float64=True,
        )
    ]
    if selected_block is not None and selected_inputs is not None:
        cases.append(
            audit_selected_block_precision(
                selected_block,
                selected_inputs,
                checkpoints=checkpoints,
                parallel_chunk_size=parallel_chunk_size,
            )
        )
    if delta_product is not None and delta_inputs is not None:
        cases.append(
            audit_delta_product_precision(
                delta_product,
                delta_inputs,
                checkpoints=checkpoints,
                parallel_chunk_size=parallel_chunk_size,
            )
        )
    return {
        "audit": "precision_horizon",
        "requested_horizon": horizon,
        "requested_retention": float(retention),
        "cases": cases,
    }


def _state_tensors(state: object) -> tuple[torch.Tensor, ...]:
    if isinstance(state, AttentionState):
        return (state.key_cache, state.value_cache)
    if isinstance(state, (DeltaProductState, SelectedBlockState)):
        return (state.memory, state.convolution)
    raise TypeError("unknown model layer-state type")


def _state_sequence_metrics(states: Sequence[object]) -> dict[str, Any]:
    nonfinite = 0
    sum_squares = 0.0
    maximum = 0.0
    for state in states:
        for tensor in _state_tensors(state):
            nonfinite += _nonfinite_count(tensor)
            if tensor.numel():
                values = tensor.detach().double()
                sum_squares += float(values.square().sum())
                maximum = max(maximum, float(values.abs().max()))
    return {
        "l2_norm": math.sqrt(sum_squares) if math.isfinite(sum_squares) else None,
        "max_abs": maximum if math.isfinite(maximum) else None,
        "nonfinite_count": nonfinite,
    }


def _compare_state_sequences(
    actual: Sequence[object], reference: Sequence[object]
) -> dict[str, Any]:
    if len(actual) != len(reference):
        return {
            "max_abs_error": None,
            "metadata_mismatch_count": abs(len(actual) - len(reference)) + 1,
            "nonfinite_count": sum(
                _state_sequence_metrics((state,))["nonfinite_count"] for state in actual
            ),
        }
    errors: list[float | None] = []
    metadata_mismatches = 0
    nonfinite = 0
    for left, right in zip(actual, reference, strict=True):
        if type(left) is not type(right):
            metadata_mismatches += 1
            continue
        left_tensors = _state_tensors(left)
        right_tensors = _state_tensors(right)
        for left_tensor, right_tensor in zip(left_tensors, right_tensors, strict=True):
            nonfinite += _nonfinite_count(left_tensor)
            if left_tensor.shape != right_tensor.shape:
                metadata_mismatches += 1
            else:
                errors.append(_max_abs_error(left_tensor, right_tensor))
        if isinstance(left, AttentionState) and isinstance(right, AttentionState):
            metadata_mismatches += int(left.position != right.position)
    maximum = (
        None
        if metadata_mismatches or any(error is None for error in errors)
        else max((error for error in errors if error is not None), default=0.0)
    )
    return {
        "max_abs_error": maximum,
        "metadata_mismatch_count": metadata_mismatches,
        "nonfinite_count": nonfinite,
    }


def _partition_sizes(partition: Sequence[int], sequence_length: int) -> list[int]:
    values = list(partition)
    if not values:
        raise ValueError("chunk partitions must be nonempty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("chunk partitions must contain integers")
    if any(value < 1 for value in values):
        raise ValueError("chunk sizes and boundaries must be positive")
    if sum(values) == sequence_length:
        return values
    if values[-1] == sequence_length and all(
        left < right for left, right in pairwise(values)
    ):
        return [values[0], *[right - left for left, right in pairwise(values)]]
    raise ValueError(
        "each partition must be sizes summing to length or final boundaries"
    )


def _normalize_partitions(
    partitions: Sequence[Sequence[int]] | Sequence[int] | None,
    sequence_length: int,
) -> list[list[int]]:
    if partitions is None:
        return [[sequence_length], [1] * sequence_length]
    values = list(partitions)
    if not values:
        raise ValueError("chunk_partitions must be nonempty")
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return [_partition_sizes(values, sequence_length)]  # type: ignore[arg-type]
    normalized = []
    for partition in values:
        if isinstance(partition, (str, bytes)) or not isinstance(partition, Sequence):
            raise TypeError("each chunk partition must be a sequence")
        normalized.append(_partition_sizes(partition, sequence_length))
    return normalized


def hybrid_chunk_replay_audit(
    model: HybridMemoryLM,
    token_ids: torch.Tensor,
    chunk_partitions: Sequence[Sequence[int]] | Sequence[int] | None = None,
    *,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compare full, arbitrary-chunk, and token-step execution of one model."""

    if not isinstance(model, HybridMemoryLM):
        raise TypeError("model must be a supplied HybridMemoryLM")
    if not isinstance(token_ids, torch.Tensor):
        raise TypeError("token_ids must be a tensor")
    if token_ids.ndim != 2 or min(token_ids.shape) < 1:
        raise ValueError("token_ids must have nonempty shape (batch,length)")
    partitions = _normalize_partitions(chunk_partitions, token_ids.shape[1])
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            full = model(
                token_ids,
                valid_mask=valid_mask,
                delta_scan_mode="recurrent",
                selected_scan_mode="physical_gather",
            )
            full_logits = full["logits"]
            full_states = full["states"]
            full_bytes = int(model.state_byte_report(full_states)["actual_bytes"])
            partition_reports = []
            for sizes in partitions:
                outputs = []
                states = None
                start = 0
                for size in sizes:
                    stop = start + size
                    chunk_mask = (
                        None if valid_mask is None else valid_mask[:, start:stop]
                    )
                    result = model(
                        token_ids[:, start:stop],
                        states,
                        valid_mask=chunk_mask,
                        delta_scan_mode="recurrent",
                        selected_scan_mode="physical_gather",
                    )
                    outputs.append(result["logits"])
                    states = result["states"]
                    start = stop
                assert states is not None
                logits = torch.cat(outputs, dim=1)
                partition_reports.append(
                    {
                        "chunk_sizes": sizes,
                        "output_max_abs_error": _max_abs_error(logits, full_logits),
                        "output_nonfinite_count": _nonfinite_count(logits),
                        "state": _compare_state_sequences(states, full_states),
                        "actual_bytes": int(
                            model.state_byte_report(states)["actual_bytes"]
                        ),
                        "actual_bytes_match_full": int(
                            model.state_byte_report(states)["actual_bytes"]
                        )
                        == full_bytes,
                    }
                )

            step_outputs = []
            step_states = None
            for position in range(token_ids.shape[1]):
                step_mask = None if valid_mask is None else valid_mask[:, position]
                logits, step_states = model.step(
                    token_ids[:, position], step_states, valid_mask=step_mask
                )
                step_outputs.append(logits)
            assert step_states is not None
            step_logits = torch.stack(step_outputs, dim=1)
            step_report = {
                "output_max_abs_error": _max_abs_error(step_logits, full_logits),
                "output_nonfinite_count": _nonfinite_count(step_logits),
                "state": _compare_state_sequences(step_states, full_states),
                "actual_bytes": int(
                    model.state_byte_report(step_states)["actual_bytes"]
                ),
                "actual_bytes_match_full": int(
                    model.state_byte_report(step_states)["actual_bytes"]
                )
                == full_bytes,
            }
    finally:
        model.train(was_training)
    return {
        "audit": "hybrid_chunk_replay",
        "batch_size": token_ids.shape[0],
        "sequence_length": token_ids.shape[1],
        "full": {
            "output_nonfinite_count": _nonfinite_count(full_logits),
            "state": _state_sequence_metrics(full_states),
            "actual_bytes": full_bytes,
        },
        "partitions": partition_reports,
        "token_step": step_report,
    }


def _load_structured_checkpoint(
    checkpoint: str | Path | Mapping[str, Any],
    *,
    map_location: str | torch.device | None,
) -> StructuredSpin8Tier:
    if isinstance(checkpoint, (str, Path)):
        payload = torch.load(
            Path(checkpoint), map_location=map_location, weights_only=False
        )
    elif isinstance(checkpoint, Mapping):
        payload = checkpoint
    else:
        raise TypeError("checkpoint must be a path or checkpoint mapping")
    if isinstance(payload, StructuredSpin8Tier):
        return payload
    if not isinstance(payload, Mapping):
        raise TypeError("structured-tier checkpoint payload must be a mapping")
    config_payload = payload.get("structured_tier_config", payload.get("config"))
    state_dict = payload.get("structured_tier_state_dict", payload.get("state_dict"))
    if isinstance(config_payload, StructuredTierConfig):
        config = config_payload
    elif isinstance(config_payload, Mapping):
        config = StructuredTierConfig(**dict(config_payload))
    else:
        raise TypeError("checkpoint is missing a StructuredTierConfig")
    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint is missing a structured-tier state_dict")
    tier = StructuredSpin8Tier(config)
    floating_tensors = [
        value
        for value in state_dict.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    ]
    if not floating_tensors:
        raise ValueError("structured-tier state_dict has no floating-point tensors")
    checkpoint_dtype = floating_tensors[0].dtype
    checkpoint_device = floating_tensors[0].device
    if any(value.dtype != checkpoint_dtype for value in floating_tensors):
        raise ValueError("structured-tier checkpoint mixes floating-point dtypes")
    tier.to(device=checkpoint_device, dtype=checkpoint_dtype)
    tier.load_state_dict(dict(state_dict), strict=True)
    return tier


def structured_rung_gauge_audit(
    tier: StructuredSpin8Tier | None = None,
    recurrent_states: torch.Tensor | None = None,
    *,
    mixer: StructuredSpin8Tier | None = None,
    checkpoint: str | Path | Mapping[str, Any] | None = None,
    composition_horizon: int | None = None,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Measure learned rung routing and numerical action drift on real states."""

    supplied = sum(value is not None for value in (tier, mixer, checkpoint))
    if supplied != 1:
        raise ValueError("supply exactly one trained tier/mixer or checkpoint")
    if mixer is not None:
        tier = mixer
    checkpoint_loaded = checkpoint is not None
    if checkpoint is not None:
        tier = _load_structured_checkpoint(checkpoint, map_location=map_location)
    if not isinstance(tier, StructuredSpin8Tier):
        raise TypeError("tier/mixer must be a StructuredSpin8Tier")
    if not isinstance(recurrent_states, torch.Tensor):
        raise TypeError("recurrent_states must be a supplied tensor")
    if (
        recurrent_states.ndim != 3
        or recurrent_states.shape[-1] != tier.config.model_dim
        or min(recurrent_states.shape[:2]) < 1
    ):
        raise ValueError("recurrent_states must have shape (B,L,model_dim)")
    if not recurrent_states.is_floating_point():
        raise TypeError("recurrent_states must be floating point")
    if not bool(torch.isfinite(recurrent_states).all()):
        raise ValueError("recurrent_states must be finite")
    parameter = next(tier.parameters())
    if (
        recurrent_states.dtype != parameter.dtype
        or recurrent_states.device != parameter.device
    ):
        raise ValueError("recurrent_states must match the tier dtype and device")
    length = recurrent_states.shape[1]
    if composition_horizon is None:
        composition_horizon = length
    if isinstance(composition_horizon, bool) or not isinstance(
        composition_horizon, int
    ):
        raise TypeError("composition_horizon must be an integer")
    if not 1 <= composition_horizon <= length:
        raise ValueError("composition_horizon must lie within the supplied trajectory")

    was_training = tier.training
    tier.eval()
    try:
        with torch.no_grad():
            output = tier(recurrent_states)
    finally:
        tier.train(was_training)
    diagnostics = output.diagnostics
    probabilities = diagnostics["rung_probabilities"]
    soft_probabilities = diagnostics["soft_rung_probabilities"]
    selected = diagnostics["selected_rung"]
    if not all(
        isinstance(value, torch.Tensor)
        for value in (probabilities, soft_probabilities, selected)
    ):
        raise TypeError("structured-tier diagnostics are missing tensor routing data")
    assert isinstance(probabilities, torch.Tensor)
    assert isinstance(soft_probabilities, torch.Tensor)
    assert isinstance(selected, torch.Tensor)

    tiny = torch.finfo(soft_probabilities.dtype).tiny
    item_entropy = -(soft_probabilities * soft_probabilities.clamp_min(tiny).log()).sum(
        dim=-1
    )
    total_assignments = selected.numel()
    occupancy = []
    occupancy_probabilities = []
    for rung in tier.config.rungs:
        count = int((selected == rung).sum())
        fraction = count / total_assignments
        occupancy.append({"rung": rung, "count": count, "fraction": fraction})
        occupancy_probabilities.append(fraction)
    occupancy_entropy = -sum(
        probability * math.log(probability)
        for probability in occupancy_probabilities
        if probability > 0.0
    )
    switches = int((selected[:, 1:] != selected[:, :-1]).sum())
    switch_opportunities = (
        recurrent_states.shape[0] * tier.config.channels * max(length - 1, 0)
    )

    actions = output.actions
    identity = torch.eye(8, dtype=actions.dtype, device=actions.device)
    residuals = (
        (actions.transpose(-1, -2) @ actions - identity).abs().amax(dim=(-1, -2))
    )
    representation_reports = []
    for representation in range(actions.shape[-3]):
        values = residuals[..., representation]
        representation_reports.append(
            {
                "representation_index": representation,
                "evaluated_action_count": values.numel(),
                "max_orthogonality_residual": _finite_float(values.max()),
                "mean_orthogonality_residual": _finite_float(values.mean()),
            }
        )

    candidate_composed = identity.expand(
        actions.shape[0], actions.shape[2], actions.shape[3], 8, 8
    ).clone()
    reference_identity = torch.eye(8, dtype=torch.float64, device=actions.device)
    reference_composed = reference_identity.expand_as(
        candidate_composed.double()
    ).clone()
    composed_drifts: list[float | None] = []
    composed_orthogonality: list[float | None] = []
    for position in range(composition_horizon):
        token_actions = actions[:, position]
        candidate_composed = token_actions @ candidate_composed
        reference_composed = token_actions.double() @ reference_composed
        composed_drifts.append(_max_abs_error(candidate_composed, reference_composed))
        composed_residual = (
            (candidate_composed.transpose(-1, -2) @ candidate_composed - identity)
            .abs()
            .max()
        )
        composed_orthogonality.append(_finite_float(composed_residual))

    return {
        "audit": "structured_rung_gauge",
        "source": "supplied_checkpoint"
        if checkpoint_loaded
        else "supplied_tier_or_mixer",
        "state_source": "supplied_recurrent_states",
        "batch_size": recurrent_states.shape[0],
        "trajectory_length": length,
        "composition_horizon": composition_horizon,
        "parameterization_guarantee": {
            "ordered_plane_rotation_parameterization": True,
            "single_actions_are_orthogonal_in_exact_arithmetic": True,
            "guarantees_learned_rung_use": False,
            "guarantees_chart_stability": False,
            "statement": (
                "The parameterization guarantees exact-arithmetic orthogonality; "
                "rung occupancy and chart switching are measured learned behavior."
            ),
        },
        "learned_behavior": {
            "rung_regime": str(diagnostics["rung_regime"]),
            "rung_occupancy": occupancy,
            "mean_routing_entropy": _finite_float(item_entropy.mean()),
            "max_routing_entropy": _finite_float(item_entropy.max()),
            "occupancy_entropy": occupancy_entropy,
            "chart_switch_count": switches,
            "chart_switch_opportunities": switch_opportunities,
            "chart_switch_frequency": (
                switches / switch_opportunities if switch_opportunities else 0.0
            ),
        },
        "numerical_behavior": {
            "action_nonfinite_count": _nonfinite_count(actions),
            "per_representation_orthogonality": representation_reports,
            "max_per_action_orthogonality_residual": _finite_float(residuals.max()),
            "max_composed_action_drift_from_float64": (
                None
                if any(value is None for value in composed_drifts)
                else max(value for value in composed_drifts if value is not None)
            ),
            "final_composed_action_drift_from_float64": composed_drifts[-1],
            "max_composed_orthogonality_residual": (
                None
                if any(value is None for value in composed_orthogonality)
                else max(value for value in composed_orthogonality if value is not None)
            ),
            "final_composed_orthogonality_residual": composed_orthogonality[-1],
        },
    }


def _json_byte_report(report: Mapping[str, Any], position: int) -> dict[str, Any]:
    layers = []
    for layer in report["layers"]:
        layers.append(
            {
                "index": int(layer["index"]),
                "kind": str(layer["kind"]),
                "actual_bytes": int(layer["actual_bytes"]),
                "capacity_bytes": int(layer["capacity_bytes"]),
                "actual_components": {
                    str(name): int(value)
                    for name, value in layer["actual_components"].items()
                },
            }
        )
    return {
        "position": position,
        "actual_bytes": int(report["actual_bytes"]),
        "capacity_bytes": int(report["capacity_bytes"]),
        "layers": layers,
    }


def cache_state_drift_audit(
    model: HybridMemoryLM,
    token_ids: torch.Tensor,
    *,
    chunk_size: int = 256,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Stream real tokens and summarize actual state norms, finiteness, and bytes."""

    if not isinstance(model, HybridMemoryLM):
        raise TypeError("model must be a supplied HybridMemoryLM")
    if not isinstance(token_ids, torch.Tensor):
        raise TypeError("token_ids must be a tensor")
    if token_ids.ndim != 2 or min(token_ids.shape) < 1:
        raise ValueError("token_ids must have nonempty shape (batch,length)")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    layer_summaries = [
        {
            "index": index,
            "kind": block.kind,
            "sample_count": 0,
            "min_l2_norm": None,
            "max_l2_norm": None,
            "final_l2_norm": None,
            "max_abs": None,
            "nonfinite_count": 0,
        }
        for index, block in enumerate(model.blocks)
    ]
    byte_reports: list[dict[str, Any]] = []
    states = None
    processed_length = 0
    output_nonfinite = 0
    state_nonfinite = 0
    byte_report_unavailable_due_to_nonfinite_state = False
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, token_ids.shape[1], chunk_size):
                stop = min(start + chunk_size, token_ids.shape[1])
                chunk_mask = None if valid_mask is None else valid_mask[:, start:stop]
                result = model(
                    token_ids[:, start:stop],
                    states,
                    valid_mask=chunk_mask,
                    delta_scan_mode="recurrent",
                    selected_scan_mode="physical_gather",
                )
                output_nonfinite += _nonfinite_count(result["logits"])
                states = result["states"]
                processed_length = stop
                for summary, state in zip(layer_summaries, states, strict=True):
                    metrics = _state_sequence_metrics((state,))
                    summary["sample_count"] += 1
                    summary["nonfinite_count"] += metrics["nonfinite_count"]
                    state_nonfinite += metrics["nonfinite_count"]
                    norm = metrics["l2_norm"]
                    maximum = metrics["max_abs"]
                    summary["final_l2_norm"] = norm
                    if norm is not None:
                        summary["min_l2_norm"] = (
                            norm
                            if summary["min_l2_norm"] is None
                            else min(summary["min_l2_norm"], norm)
                        )
                        summary["max_l2_norm"] = (
                            norm
                            if summary["max_l2_norm"] is None
                            else max(summary["max_l2_norm"], norm)
                        )
                    if maximum is not None:
                        summary["max_abs"] = (
                            maximum
                            if summary["max_abs"] is None
                            else max(summary["max_abs"], maximum)
                        )
                if state_nonfinite:
                    byte_report_unavailable_due_to_nonfinite_state = True
                    break
                byte_reports.append(
                    _json_byte_report(model.state_byte_report(states), stop)
                )
    finally:
        model.train(was_training)

    final_actual_bytes = byte_reports[-1]["actual_bytes"] if byte_reports else None
    final_capacity_bytes = byte_reports[-1]["capacity_bytes"] if byte_reports else None
    return {
        "audit": "cache_state_drift",
        "batch_size": token_ids.shape[0],
        "requested_length": token_ids.shape[1],
        "processed_length": processed_length,
        "chunk_size": chunk_size,
        "terminated_early": processed_length < token_ids.shape[1],
        "output_nonfinite_count": output_nonfinite,
        "state_nonfinite_count": state_nonfinite,
        "layers": layer_summaries,
        "byte_report_samples": byte_reports,
        "final_actual_bytes": final_actual_bytes,
        "final_capacity_bytes": final_capacity_bytes,
        "byte_report_unavailable_due_to_nonfinite_state": (
            byte_report_unavailable_due_to_nonfinite_state
        ),
    }


def temporal_query_observability_audit(
    model: HybridMemoryLM,
    token_ids: torch.Tensor,
    *,
    selected_layer_index: int,
    route_mode: str,
    threshold: float = 0.0,
) -> dict[str, Any]:
    """Measure per-position route activation credit under final-only loss.

    Activation gradients are reported separately for coarse and fine
    write/erase/read controls. This deliberately avoids using a shared
    controller-parameter norm, which cannot distinguish final-position credit
    from non-final temporal observability.
    """

    if not isinstance(model, HybridMemoryLM):
        raise TypeError("model must be a HybridMemoryLM")
    if not isinstance(token_ids, torch.Tensor):
        raise TypeError("token_ids must be a tensor")
    if token_ids.ndim != 2 or min(token_ids.shape) < 1 or token_ids.shape[1] < 2:
        raise ValueError("token_ids must have shape (batch, length>=2)")
    if token_ids.dtype != torch.long:
        raise TypeError("token_ids must have dtype torch.long")
    if token_ids.device != model.embedding.weight.device:
        raise ValueError("token_ids must share the model device")
    if model.embedding.weight.dtype != torch.float64:
        raise TypeError("observability audit requires a float64 model")
    if (
        isinstance(selected_layer_index, bool)
        or not isinstance(selected_layer_index, int)
        or not 0 <= selected_layer_index < len(model.blocks)
    ):
        raise ValueError("selected_layer_index must identify a model layer")
    block = model.blocks[selected_layer_index]
    if block.kind != "selected_block" or not isinstance(
        block.mixer, SelectedBlockMemory
    ):
        raise ValueError("selected_layer_index must identify a selected_block layer")
    if route_mode not in block.mixer.route_modes:
        raise ValueError(f"route_mode must be one of {block.mixer.route_modes}")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
        raise TypeError("threshold must be finite")
    if threshold < 0.0:
        raise ValueError("threshold must be nonnegative")

    names = (
        "write_block_logits",
        "erase_block_logits",
        "read_block_logits",
        "write_fine_logits",
        "erase_fine_logits",
        "read_fine_logits",
    )
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        output = model(
            token_ids,
            selected_scan_mode="dense_recurrent",
            selected_route_mode=route_mode,
            return_diagnostics=True,
        )
        diagnostics = output["diagnostics"][selected_layer_index]
        if not isinstance(diagnostics, Mapping):
            raise TypeError("selected layer did not return diagnostics")
        activations = tuple(diagnostics.get(name) for name in names)
        if any(not isinstance(value, torch.Tensor) for value in activations):
            raise TypeError("selected diagnostics are missing route logits")
        tensor_activations = tuple(
            value for value in activations if isinstance(value, torch.Tensor)
        )
        objective_weights = torch.linspace(
            0.5,
            1.5,
            model.config.vocab_size,
            dtype=model.embedding.weight.dtype,
            device=token_ids.device,
        )
        objective = (output["logits"][:, -1] * objective_weights).sum()
        gradients = torch.autograd.grad(
            objective,
            tensor_activations,
            allow_unused=True,
        )
    model.train(was_training)

    gradient_reports: dict[str, Any] = {}
    sequence_length = token_ids.shape[1]
    for name, gradient in zip(names, gradients, strict=True):
        if gradient is None:
            norms = [0.0] * sequence_length
            connected = False
        else:
            per_batch = gradient.flatten(start_dim=2).norm(dim=-1)
            norms = [float(value) for value in per_batch.amax(dim=0)]
            connected = True
        nonfinal_max = max(norms[:-1], default=0.0)
        gradient_reports[name] = {
            "connected": connected,
            "per_position_norm": norms,
            "nonfinal_max_norm": nonfinal_max,
            "final_norm": norms[-1],
            "nonfinal_present": nonfinal_max > threshold,
        }

    coarse_names = (
        "write_block_logits",
        "erase_block_logits",
        "read_block_logits",
    )
    read_coarse = gradient_reports["read_block_logits"]
    read_fine = gradient_reports["read_fine_logits"]
    return {
        "audit": "temporal_query_observability",
        "layer_plan": list(model.layer_plan),
        "selected_layer_index": selected_layer_index,
        "route_mode": route_mode,
        "objective": "weighted_sum_of_final_position_logits",
        "threshold": float(threshold),
        "gradients": gradient_reports,
        "coarse_routes_connected": all(
            gradient_reports[name]["connected"] for name in coarse_names
        ),
        "nonfinal_read_path_present": bool(
            read_coarse["nonfinal_present"] and read_fine["nonfinal_present"]
        ),
        "one_block_final_only_expected_zero": len(model.blocks) == 1,
    }


# Verb-first aliases make the public entry points easy to discover.
audit_precision_horizon = precision_horizon_audit
audit_hybrid_chunk_replay = hybrid_chunk_replay_audit
audit_structured_rung_gauge = structured_rung_gauge_audit
audit_cache_state_drift = cache_state_drift_audit
audit_temporal_query_observability = temporal_query_observability_audit


__all__ = [
    "audit_affine_precision",
    "audit_cache_state_drift",
    "audit_delta_product_precision",
    "audit_hybrid_chunk_replay",
    "audit_precision_horizon",
    "audit_selected_block_precision",
    "audit_structured_rung_gauge",
    "audit_temporal_query_observability",
    "cache_state_drift_audit",
    "hybrid_chunk_replay_audit",
    "precision_horizon_audit",
    "structured_rung_gauge_audit",
    "temporal_query_observability_audit",
]
