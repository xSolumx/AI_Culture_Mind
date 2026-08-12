"""Capacity laws and associative scan harness for coded Spin(8) triality memory."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from intertwiner_schurscan import matrix_scan
from spin8_triality import (
    SPIN8_BIVECTOR_DIM,
    AffineTransition,
    apply_affine,
    associative_prefix_scan,
    spin8_actions,
    torch_triality_generators,
)
from spin8_triality_lift import (
    triality_bind,
    triality_tensor,
    triality_unbind_negative,
)


@dataclass(frozen=True)
class SlotTransition:
    retention: torch.Tensor
    action: torch.Tensor
    drive: torch.Tensor


def compose_slot(after: SlotTransition, before: SlotTransition) -> SlotTransition:
    rotated_drive = before.drive @ after.action.transpose(-1, -2)
    return SlotTransition(
        retention=after.retention * before.retention,
        action=after.action @ before.action,
        drive=after.drive + after.retention[..., :, None] * rotated_drive,
    )


def apply_slot(transition: SlotTransition, state: torch.Tensor) -> torch.Tensor:
    rotated = state @ transition.action.transpose(-1, -2)
    return transition.retention[..., :, None] * rotated + transition.drive


def associative_slot_scan(transition: SlotTransition) -> SlotTransition:
    """Inclusive Hillis--Steele scan of structured slot transitions."""

    retention, action, drive = (
        transition.retention,
        transition.action,
        transition.drive,
    )
    offset = 1
    while offset < retention.shape[1]:
        composed = compose_slot(
            SlotTransition(
                retention[:, offset:], action[:, offset:], drive[:, offset:]
            ),
            SlotTransition(
                retention[:, :-offset], action[:, :-offset], drive[:, :-offset]
            ),
        )
        retention = torch.cat((retention[:, :offset], composed.retention), dim=1)
        action = torch.cat((action[:, :offset], composed.action), dim=1)
        drive = torch.cat((drive[:, :offset], composed.drive), dim=1)
        offset *= 2
    return SlotTransition(retention, action, drive)


def work_efficient_slot_scan(transition: SlotTransition) -> SlotTransition:
    """Inclusive ordered Blelloch scan of structured slot transitions.

    The state has shape ``slots x dimension``, but every slot shares one
    ``dimension x dimension`` action and has only a scalar retention.  This
    scan preserves that structure rather than materializing a dense
    ``(slots * dimension)`` affine operator.
    """

    retention, action, drive = (
        transition.retention,
        transition.action,
        transition.drive,
    )
    if (
        retention.ndim != 3
        or action.ndim != 4
        or drive.ndim != 4
        or retention.shape[:2] != action.shape[:2]
        or retention.shape != drive.shape[:3]
        or action.shape[-1] != action.shape[-2]
        or action.shape[-1] != drive.shape[-1]
    ):
        raise ValueError(
            "slot transition must have retention (batch, length, slots), "
            "action (batch, length, D, D), and drive (batch, length, slots, D)"
        )
    batch, length, slots = retention.shape
    dimension = action.shape[-1]
    if length < 1:
        raise ValueError("scan length must be positive")
    if length == 1:
        return transition

    padded_length = 1 << (length - 1).bit_length()
    if padded_length == length:
        leaves = transition
    else:
        padding_length = padded_length - length
        identity = torch.eye(
            dimension, dtype=action.dtype, device=action.device
        ).reshape(1, 1, dimension, dimension)
        leaves = SlotTransition(
            torch.cat(
                (
                    retention,
                    torch.ones(
                        batch,
                        padding_length,
                        slots,
                        dtype=retention.dtype,
                        device=retention.device,
                    ),
                ),
                dim=1,
            ),
            torch.cat(
                (action, identity.expand(batch, padding_length, -1, -1)),
                dim=1,
            ),
            torch.cat(
                (
                    drive,
                    torch.zeros(
                        batch,
                        padding_length,
                        slots,
                        dimension,
                        dtype=drive.dtype,
                        device=drive.device,
                    ),
                ),
                dim=1,
            ),
        )

    levels = [leaves]
    nodes = leaves
    while nodes.retention.shape[1] > 1:
        nodes = compose_slot(
            SlotTransition(
                nodes.retention[:, 1::2],
                nodes.action[:, 1::2],
                nodes.drive[:, 1::2],
            ),
            SlotTransition(
                nodes.retention[:, 0::2],
                nodes.action[:, 0::2],
                nodes.drive[:, 0::2],
            ),
        )
        levels.append(nodes)

    prefix = SlotTransition(
        torch.ones(batch, 1, slots, dtype=retention.dtype, device=retention.device),
        torch.eye(dimension, dtype=action.dtype, device=action.device)
        .reshape(1, 1, dimension, dimension)
        .expand(batch, 1, -1, -1),
        torch.zeros(
            batch,
            1,
            slots,
            dimension,
            dtype=drive.dtype,
            device=drive.device,
        ),
    )
    for children in reversed(levels[:-1]):
        left = SlotTransition(
            children.retention[:, 0::2],
            children.action[:, 0::2],
            children.drive[:, 0::2],
        )
        right_prefix = compose_slot(left, prefix)
        prefix = SlotTransition(
            torch.stack((prefix.retention, right_prefix.retention), dim=2).reshape(
                batch, -1, slots
            ),
            torch.stack((prefix.action, right_prefix.action), dim=2).reshape(
                batch, -1, dimension, dimension
            ),
            torch.stack((prefix.drive, right_prefix.drive), dim=2).reshape(
                batch, -1, slots, dimension
            ),
        )

    inclusive = compose_slot(leaves, prefix)
    return SlotTransition(
        inclusive.retention[:, :length],
        inclusive.action[:, :length],
        inclusive.drive[:, :length],
    )


def pack_slot_homogeneous_matrices(transition: SlotTransition) -> torch.Tensor:
    """Pack each structured slot as a local homogeneous affine matrix.

    The result has shape ``(batch, slots, length, D + 1, D + 1)``.  This keeps
    slot block structure while allowing one batched matrix multiplication per
    composition, avoiding both the multi-operation structured composition and
    the dense ``(slots * D + 1)`` homogeneous lift.
    """

    retention, action, drive = (
        transition.retention,
        transition.action,
        transition.drive,
    )
    if (
        retention.ndim != 3
        or action.ndim != 4
        or drive.ndim != 4
        or retention.shape[:2] != action.shape[:2]
        or retention.shape != drive.shape[:3]
        or action.shape[-1] != action.shape[-2]
        or action.shape[-1] != drive.shape[-1]
    ):
        raise ValueError("incompatible structured slot transition")
    batch, length, slots = retention.shape
    dimension = action.shape[-1]
    matrices = action.new_zeros(batch, length, slots, dimension + 1, dimension + 1)
    matrices[..., 0, 0] = 1
    matrices[..., 1:, 0] = drive
    matrices[..., 1:, 1:] = retention[..., None, None] * action[..., None, :, :]
    return matrices.permute(0, 2, 1, 3, 4)


def packed_homogeneous_slot_states(
    matrices: torch.Tensor,
    initial: torch.Tensor,
    *,
    backend: str = "work_efficient",
) -> torch.Tensor:
    """Apply all prefixes of prepacked local homogeneous slot matrices."""

    if (
        matrices.ndim != 5
        or matrices.shape[-1] != matrices.shape[-2]
        or initial.ndim != 3
        or matrices.shape[:2] != initial.shape[:2]
        or matrices.shape[-1] != initial.shape[-1] + 1
    ):
        raise ValueError(
            "matrices must be (batch, slots, length, D+1, D+1) and initial "
            "must be (batch, slots, D)"
        )
    batch, slots, length, homogeneous_dimension, _ = matrices.shape
    flattened = matrices.reshape(
        batch * slots, length, homogeneous_dimension, homogeneous_dimension
    )
    prefixes = matrix_scan(flattened, backend=backend)
    one = torch.ones(batch, slots, 1, dtype=initial.dtype, device=initial.device)
    initial_h = torch.cat((one, initial), dim=-1).reshape(
        batch * slots, homogeneous_dimension
    )
    states = torch.einsum("blij,bj->bli", prefixes, initial_h)[..., 1:]
    return states.reshape(batch, slots, length, homogeneous_dimension - 1).permute(
        0, 2, 1, 3
    )


def packed_homogeneous_slot_scan(
    transition: SlotTransition,
    initial: torch.Tensor,
    *,
    backend: str = "work_efficient",
) -> torch.Tensor:
    """Pack and scan local homogeneous slot matrices end to end."""

    return packed_homogeneous_slot_states(
        pack_slot_homogeneous_matrices(transition), initial, backend=backend
    )


def walsh_matrix(width: int, *, dtype: torch.dtype) -> torch.Tensor:
    if width < 1 or width & (width - 1):
        raise ValueError("Walsh width must be a positive power of two")
    matrix = torch.ones(1, 1, dtype=dtype)
    while matrix.shape[0] < width:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix


def walsh_codes(width: int, count: int, *, dtype: torch.dtype) -> torch.Tensor:
    """Return count orthonormal Walsh columns in R^width."""

    if not 1 <= count <= width:
        raise ValueError("count must lie in [1, width]")
    return walsh_matrix(width, dtype=dtype)[:, :count] / width**0.5


def tight_frame_codes(width: int, count: int, *, dtype: torch.dtype) -> torch.Tensor:
    """Return unit columns attaining the frame-potential lower bound."""

    if count <= width:
        return walsh_codes(width, count, dtype=dtype)
    if count & (count - 1) == 0:
        return walsh_matrix(count, dtype=dtype)[:width] / width**0.5
    if count == width + 1:
        projector = (
            torch.eye(count, dtype=dtype)
            - torch.ones(count, count, dtype=dtype) / count
        )
        basis, _ = torch.linalg.qr(projector[:, :width], mode="reduced")
        return (count / width) ** 0.5 * basis.T
    raise ValueError("implemented tight frames require K<=H, power-of-two K, or K=H+1")


def build_memory(
    codes: torch.Tensor,
    positive_keys: torch.Tensor,
    negative_values: torch.Tensor,
    rho: torch.Tensor,
) -> torch.Tensor:
    """Superpose triality-bound pairs across multiplicity channels."""

    bound = triality_bind(positive_keys, negative_values, rho)
    return torch.einsum("...hk,...kv->...hv", codes, bound)


def retrieve_all(
    codes: torch.Tensor,
    positive_keys: torch.Tensor,
    memory: torch.Tensor,
    rho: torch.Tensor,
) -> torch.Tensor:
    """Retrieve every code/key column from a coded vector memory."""

    channel_unbindings = torch.einsum(
        "...ki,vji,...hv->...khj", positive_keys, rho, memory
    )
    return torch.einsum("...hk,...khj->...kj", codes, channel_unbindings)


def relative_squared_errors(
    recovered: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    numerator = (recovered - targets).square().sum(dim=-1)
    denominator = (
        targets.square().sum(dim=-1).clamp_min(torch.finfo(targets.dtype).tiny)
    )
    return numerator / denominator


def random_unit(
    shape: tuple[int, ...], *, generator: torch.Generator, dtype: torch.dtype
) -> torch.Tensor:
    return F.normalize(torch.randn(*shape, generator=generator, dtype=dtype), dim=-1)


def capacity_audit(
    *,
    trials: int = 256,
    seed: int = 20260803,
) -> dict[str, object]:
    dtype = torch.float64
    generator = torch.Generator().manual_seed(seed)
    rho = triality_tensor(dtype=dtype)
    widths = (1, 2, 4, 8, 16, 32)
    counts = (1, 2, 4, 8, 16, 32)
    random_cells: list[dict[str, float | int]] = []
    exact_cells: list[dict[str, float | int]] = []
    tight_cells: list[dict[str, float | int]] = []

    for width in widths:
        width_counts = tuple(sorted(set(counts + ((width + 1,) if width > 1 else ()))))
        for count in width_counts:
            keys = random_unit((trials, count, 8), generator=generator, dtype=dtype)
            values = random_unit((trials, count, 8), generator=generator, dtype=dtype)
            random_codes = random_unit(
                (trials, count, width), generator=generator, dtype=dtype
            ).transpose(-1, -2)
            random_memory = build_memory(random_codes, keys, values, rho)
            random_recovered = retrieve_all(random_codes, keys, random_memory, rho)
            random_mse = float(relative_squared_errors(random_recovered, values).mean())
            prediction = (count - 1) / width
            relative_prediction_error = (
                abs(random_mse - prediction) / prediction if prediction else 0.0
            )
            random_cells.append(
                {
                    "channels": width,
                    "associations": count,
                    "mean_relative_squared_error": random_mse,
                    "prediction": prediction,
                    "relative_prediction_error": relative_prediction_error,
                }
            )

            if count <= width or count & (count - 1) == 0 or count == width + 1:
                codes = tight_frame_codes(width, count, dtype=dtype)
                tight_memory = build_memory(codes, keys, values, rho)
                tight_recovered = retrieve_all(codes, keys, tight_memory, rho)
                tight_errors = relative_squared_errors(tight_recovered, values)
                tight_prediction = max(0.0, (count - width) / width)
                tight_mse = float(tight_errors.mean())
                gram = codes.T @ codes
                off_diagonal = gram - torch.eye(count, dtype=dtype)
                tight_cells.append(
                    {
                        "channels": width,
                        "associations": count,
                        "mean_relative_squared_error": tight_mse,
                        "prediction": tight_prediction,
                        "relative_prediction_error": (
                            abs(tight_mse - tight_prediction) / tight_prediction
                            if tight_prediction
                            else 0.0
                        ),
                        "matched_random_error": random_mse,
                        "maximum_absolute_code_inner_product": float(
                            off_diagonal.abs().max()
                        ),
                        "maximum_relative_error": float(tight_errors.sqrt().max()),
                    }
                )
            if count <= width:
                exact_cells.append(
                    {
                        "channels": width,
                        "associations": count,
                        "maximum_relative_error": float(tight_errors.sqrt().max()),
                        "mean_relative_squared_error": float(tight_errors.mean()),
                    }
                )

    nonzero_random = [cell for cell in random_cells if float(cell["prediction"]) > 0.0]
    return {
        "trials_per_cell": trials,
        "random_code_cells": random_cells,
        "orthogonal_code_cells": exact_cells,
        "tight_frame_cells": tight_cells,
        "maximum_exact_relative_error": max(
            float(cell["maximum_relative_error"]) for cell in exact_cells
        ),
        "maximum_random_law_relative_error": max(
            float(cell["relative_prediction_error"]) for cell in nonzero_random
        ),
        "maximum_tight_frame_law_relative_error": max(
            float(cell["relative_prediction_error"])
            for cell in tight_cells
            if float(cell["prediction"]) > 0.0
        ),
        "tight_frame_beats_random_all_overcomplete_cells": all(
            float(cell["mean_relative_squared_error"])
            < float(cell["matched_random_error"])
            for cell in tight_cells
            if int(cell["associations"]) > int(cell["channels"])
            and int(cell["channels"]) > 1
        ),
    }


def transport_audit(seed: int = 20260804) -> dict[str, float]:
    dtype = torch.float64
    generator = torch.Generator().manual_seed(seed)
    rho = triality_tensor(dtype=dtype)
    keys = random_unit((8, 8), generator=generator, dtype=dtype)
    values = random_unit((8, 8), generator=generator, dtype=dtype)
    codes = walsh_codes(8, 8, dtype=dtype)
    memory = build_memory(codes, keys, values, rho)

    coefficients = 0.35 * torch.randn(
        SPIN8_BIVECTOR_DIM, generator=generator, dtype=dtype
    )
    action = spin8_actions(coefficients, torch_triality_generators(dtype=dtype))
    vector_action, positive_action, negative_action = action
    transported_keys = torch.einsum("ij,kj->ki", positive_action, keys)
    transported_values = torch.einsum("ij,kj->ki", negative_action, values)
    transported_memory = torch.einsum("ij,hj->hi", vector_action, memory)
    rebuilt_memory = build_memory(codes, transported_keys, transported_values, rho)
    recovered = retrieve_all(codes, transported_keys, transported_memory, rho)
    return {
        "memory_equivariance_max_error": float(
            (transported_memory - rebuilt_memory).abs().max()
        ),
        "transported_retrieval_max_error": float(
            (recovered - transported_values).abs().max()
        ),
    }


def scan_audit(seed: int = 20260805) -> dict[str, float | int]:
    dtype = torch.float64
    generator = torch.Generator().manual_seed(seed)
    batch, length, channels = 2, 64, 8
    coefficients = 0.12 * torch.randn(
        batch,
        length,
        SPIN8_BIVECTOR_DIM,
        generator=generator,
        dtype=dtype,
    )
    vector_generators = torch_triality_generators(("vector",), dtype=dtype)
    vector_action = spin8_actions(coefficients, vector_generators)[..., 0, :, :]
    action = vector_action[..., None, :, :].expand(batch, length, channels, 8, 8)
    scale = 0.95 + 0.04 * torch.rand(batch, length, generator=generator, dtype=dtype)
    drive = 0.05 * torch.randn(
        batch, length, channels, 8, generator=generator, dtype=dtype
    )
    transition = AffineTransition(scale=scale, action=action, drive=drive)
    initial = torch.randn(batch, channels, 8, generator=generator, dtype=dtype)

    prefixes = associative_prefix_scan(transition)
    parallel = apply_affine(prefixes, initial[:, None])
    state = initial
    recurrent = []
    for position in range(length):
        step = AffineTransition(
            scale=scale[:, position],
            action=action[:, position],
            drive=drive[:, position],
        )
        state = apply_affine(step, state)
        recurrent.append(state)
    recurrent_states = torch.stack(recurrent, dim=1)
    return {
        "length": length,
        "channels": channels,
        "parallel_recurrent_max_error": float(
            (parallel - recurrent_states).abs().max()
        ),
    }


def dynamic_slot_audit(seed: int = 20260806) -> dict[str, float | int]:
    dtype = torch.float64
    generator = torch.Generator().manual_seed(seed)
    batch, length, channels = 2, 128, 8
    rho = triality_tensor(dtype=dtype)
    generators = torch_triality_generators(dtype=dtype)
    coefficients = 0.10 * torch.randn(
        batch,
        length,
        SPIN8_BIVECTOR_DIM,
        generator=generator,
        dtype=dtype,
    )
    actions = spin8_actions(coefficients, generators)
    vector_action = actions[..., 0, :, :]
    positive_action = actions[..., 1, :, :]
    negative_action = actions[..., 2, :, :]

    initial_keys = random_unit((batch, channels, 8), generator=generator, dtype=dtype)
    initial_values = random_unit((batch, channels, 8), generator=generator, dtype=dtype)
    initial_memory = triality_bind(initial_keys, initial_values, rho)
    new_keys = random_unit((batch, length, 8), generator=generator, dtype=dtype)
    new_values = random_unit((batch, length, 8), generator=generator, dtype=dtype)
    addresses = torch.randint(0, channels, (batch, length), generator=generator)
    retention = torch.ones(batch, length, channels, dtype=dtype)
    retention.scatter_(2, addresses[..., None], 0.0)
    drive = torch.zeros(batch, length, channels, 8, dtype=dtype)
    bound_writes = triality_bind(new_keys, new_values, rho)
    drive.scatter_(
        2,
        addresses[..., None, None].expand(batch, length, 1, 8),
        bound_writes[..., None, :],
    )
    transition = SlotTransition(retention, vector_action, drive)

    left = compose_slot(
        SlotTransition(retention[:, 2], vector_action[:, 2], drive[:, 2]),
        compose_slot(
            SlotTransition(retention[:, 1], vector_action[:, 1], drive[:, 1]),
            SlotTransition(retention[:, 0], vector_action[:, 0], drive[:, 0]),
        ),
    )
    right = compose_slot(
        compose_slot(
            SlotTransition(retention[:, 2], vector_action[:, 2], drive[:, 2]),
            SlotTransition(retention[:, 1], vector_action[:, 1], drive[:, 1]),
        ),
        SlotTransition(retention[:, 0], vector_action[:, 0], drive[:, 0]),
    )
    associativity_error = max(
        float((left.retention - right.retention).abs().max()),
        float((left.action - right.action).abs().max()),
        float((left.drive - right.drive).abs().max()),
    )

    prefixes = associative_slot_scan(transition)
    parallel = apply_slot(prefixes, initial_memory[:, None])
    state = initial_memory
    keys = initial_keys
    values = initial_values
    recurrent_states = []
    oracle_states = []
    for position in range(length):
        step = SlotTransition(
            retention[:, position],
            vector_action[:, position],
            drive[:, position],
        )
        state = apply_slot(step, state)
        recurrent_states.append(state)

        keys = torch.einsum("bij,bhj->bhi", positive_action[:, position], keys)
        values = torch.einsum("bij,bhj->bhi", negative_action[:, position], values)
        batch_index = torch.arange(batch)
        keys[batch_index, addresses[:, position]] = new_keys[:, position]
        values[batch_index, addresses[:, position]] = new_values[:, position]
        oracle_states.append(triality_bind(keys, values, rho))
    recurrent = torch.stack(recurrent_states, dim=1)
    oracle = torch.stack(oracle_states, dim=1)
    recovered = triality_unbind_negative(keys, state, rho)
    return {
        "length": length,
        "channels": channels,
        "associativity_max_error": associativity_error,
        "parallel_recurrent_max_error": float((parallel - recurrent).abs().max()),
        "oracle_state_max_error": float((recurrent - oracle).abs().max()),
        "final_retrieval_max_error": float((recovered - values).abs().max()),
    }


def run() -> dict[str, object]:
    capacity = capacity_audit()
    transport = transport_audit()
    scan = scan_audit()
    dynamic_slot = dynamic_slot_audit()
    gates = {
        "orthogonal_exact": capacity["maximum_exact_relative_error"] < 1e-10,
        "random_capacity_law": (capacity["maximum_random_law_relative_error"] < 0.20),
        "tight_frame_capacity_law": (
            capacity["maximum_tight_frame_law_relative_error"] < 0.15
            and capacity["tight_frame_beats_random_all_overcomplete_cells"]
        ),
        "transport_equivariance": max(transport.values()) < 1e-10,
        "parallel_recurrent": scan["parallel_recurrent_max_error"] < 1e-10,
        "dynamic_slot": max(
            float(value)
            for key, value in dynamic_slot.items()
            if key.endswith("_error")
        )
        < 1e-10,
    }
    return {
        "experiment": "Spin(8) triality-coded associative memory",
        "capacity": capacity,
        "transport": transport,
        "scan": scan,
        "dynamic_slot": dynamic_slot,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/spin8_triality_coded_memory.json"),
    )
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
