"""Slotwise equivalence of triality-bound and direct Spin(8) memory.

For a unit positive-spinor key p, triality binding is an orthogonal map from
the negative-spinor value space to the vector memory space.  Addressed direct
and triality-bound slots are therefore related by a time-dependent orthogonal
change of coordinates.  Equivariance makes that gauge commute with shared
Spin(8) transport, including interleaved overwrite sequences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from spin8_triality import (
    SPIN8_BIVECTOR_DIM,
    spin8_actions,
    torch_triality_generators,
)
from spin8_triality_lift import (
    triality_bind,
    triality_tensor,
    triality_unbind_negative,
)
from spin8_triality_memory import (
    SlotTransition,
    apply_slot,
    associative_slot_scan,
    compose_slot,
)


def random_unit(
    shape: tuple[int, ...], *, generator: torch.Generator, dtype: torch.dtype
) -> torch.Tensor:
    return F.normalize(torch.randn(*shape, generator=generator, dtype=dtype), dim=-1)


def binding_operator(keys: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Return M(p) with triality_bind(p, n) = M(p) n."""

    if keys.shape[-1] != rho.shape[-1]:
        raise ValueError("keys and triality tensor have incompatible dimensions")
    return torch.einsum("vni,...i->...vn", rho, keys)


def gauge_direct_memory(
    keys: torch.Tensor, direct_memory: torch.Tensor, rho: torch.Tensor
) -> torch.Tensor:
    if keys.shape != direct_memory.shape:
        raise ValueError("keys and direct memory must have equal shapes")
    return torch.einsum("...i,vni,...n->...v", keys, rho, direct_memory)


def _associativity_error(transition: SlotTransition) -> float:
    steps = [
        SlotTransition(
            transition.retention[:, position],
            transition.action[:, position],
            transition.drive[:, position],
        )
        for position in range(3)
    ]
    left = compose_slot(steps[2], compose_slot(steps[1], steps[0]))
    right = compose_slot(compose_slot(steps[2], steps[1]), steps[0])
    return max(
        float((left.retention - right.retention).abs().max()),
        float((left.action - right.action).abs().max()),
        float((left.drive - right.drive).abs().max()),
    )


def diagnostics(seed: int = 20260810) -> dict[str, object]:
    dtype = torch.float64
    generator = torch.Generator().manual_seed(seed)
    batch, length, slots, dimension = 3, 127, 8, 8
    rho = triality_tensor(dtype=dtype)
    generators = torch_triality_generators(dtype=dtype)

    initial_keys = random_unit(
        (batch, slots, dimension), generator=generator, dtype=dtype
    )
    initial_values = random_unit(
        (batch, slots, dimension), generator=generator, dtype=dtype
    )
    initial_triality = triality_bind(initial_keys, initial_values, rho)

    coefficients = 0.18 * torch.randn(
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

    addresses = torch.randint(slots, (batch, length), generator=generator)
    new_keys = random_unit((batch, length, dimension), generator=generator, dtype=dtype)
    new_values = random_unit(
        (batch, length, dimension), generator=generator, dtype=dtype
    )
    retention = torch.ones(batch, length, slots, dtype=dtype)
    retention.scatter_(2, addresses[..., None], 0.0)

    direct_drive = torch.zeros(batch, length, slots, dimension, dtype=dtype)
    direct_drive.scatter_(
        2,
        addresses[..., None, None].expand(batch, length, 1, dimension),
        new_values[..., None, :],
    )
    bound_writes = triality_bind(new_keys, new_values, rho)
    triality_drive = torch.zeros_like(direct_drive)
    triality_drive.scatter_(
        2,
        addresses[..., None, None].expand(batch, length, 1, dimension),
        bound_writes[..., None, :],
    )

    direct_transition = SlotTransition(retention, negative_action, direct_drive)
    triality_transition = SlotTransition(retention, vector_action, triality_drive)
    direct_parallel = apply_slot(
        associative_slot_scan(direct_transition), initial_values[:, None]
    )
    triality_parallel = apply_slot(
        associative_slot_scan(triality_transition), initial_triality[:, None]
    )

    keys = initial_keys
    direct_state = initial_values
    triality_state = initial_triality
    direct_recurrent = []
    triality_recurrent = []
    gauged_states = []
    recovered_states = []
    batch_index = torch.arange(batch)
    for position in range(length):
        direct_state = apply_slot(
            SlotTransition(
                retention[:, position],
                negative_action[:, position],
                direct_drive[:, position],
            ),
            direct_state,
        )
        triality_state = apply_slot(
            SlotTransition(
                retention[:, position],
                vector_action[:, position],
                triality_drive[:, position],
            ),
            triality_state,
        )
        keys = torch.einsum("bij,bhj->bhi", positive_action[:, position], keys)
        keys = keys.clone()
        keys[batch_index, addresses[:, position]] = new_keys[:, position]
        direct_recurrent.append(direct_state)
        triality_recurrent.append(triality_state)
        gauged_states.append(triality_bind(keys, direct_state, rho))
        recovered_states.append(triality_unbind_negative(keys, triality_state, rho))

    direct_recurrent_tensor = torch.stack(direct_recurrent, dim=1)
    triality_recurrent_tensor = torch.stack(triality_recurrent, dim=1)
    gauged_tensor = torch.stack(gauged_states, dim=1)
    recovered_tensor = torch.stack(recovered_states, dim=1)

    operators = binding_operator(initial_keys, rho)
    identity = torch.eye(dimension, dtype=dtype)
    gauge_orthogonality = float(
        (operators.transpose(-1, -2) @ operators - identity).abs().max()
    )
    transformed_keys = torch.einsum("bij,bhj->bhi", positive_action[:, 0], initial_keys)
    before = binding_operator(initial_keys, rho)
    after = binding_operator(transformed_keys, rho)
    commuting_error = float(
        (
            torch.einsum("bij,bhjk->bhik", vector_action[:, 0], before)
            - torch.einsum("bhij,bjk->bhik", after, negative_action[:, 0])
        )
        .abs()
        .max()
    )

    metrics = {
        "binding_gauge_orthogonality_max_abs_error": gauge_orthogonality,
        "transport_gauge_commuting_max_abs_error": commuting_error,
        "direct_parallel_recurrent_max_abs_error": float(
            (direct_parallel - direct_recurrent_tensor).abs().max()
        ),
        "triality_parallel_recurrent_max_abs_error": float(
            (triality_parallel - triality_recurrent_tensor).abs().max()
        ),
        "triality_state_vs_gauged_direct_max_abs_error": float(
            (triality_recurrent_tensor - gauged_tensor).abs().max()
        ),
        "triality_retrieval_vs_direct_max_abs_error": float(
            (recovered_tensor - direct_recurrent_tensor).abs().max()
        ),
        "direct_transition_associativity_max_abs_error": _associativity_error(
            direct_transition
        ),
        "triality_transition_associativity_max_abs_error": _associativity_error(
            triality_transition
        ),
    }
    checks = {
        name.removesuffix("_max_abs_error"): value < 1e-10
        for name, value in metrics.items()
    }
    return {
        "experiment": "Spin(8) triality/direct addressed-memory equivalence",
        "seed": seed,
        "batch": batch,
        "length": length,
        "slots": slots,
        "coordinates_per_slot": dimension,
        "streaming_state_scalars_each": slots * dimension,
        "metrics": metrics,
        "checks": checks,
        "claims": {
            "supplied_unit_key_triality_slots_are_orthogonal_gauges_of_direct_slots": all(
                checks.values()
            ),
            "triality_specific_capacity_or_retrieval_advantage_established": False,
        },
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/spin8_triality_direct_memory_equivalence_20260810.json"
        ),
    )
    args = parser.parse_args()
    report = diagnostics()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
