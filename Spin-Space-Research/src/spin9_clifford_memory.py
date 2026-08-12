"""Spin(9) Clifford binding and its exact addressed-memory boundary.

A unit vector ``a in R^9`` defines the symmetric orthogonal involution
``D(a)`` on the real 16-dimensional spin module.  Thus ``D(a) psi`` is an
exactly invertible single-pair binding.  Slotwise binding is nevertheless an
orthogonal gauge of same-width direct memory, and raw superposition has
norm-preserving cross terms.  Spin(9) therefore supplies a structured
cross-chiral prior, not free associative-memory capacity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from spin8_triality_memory import (
    SlotTransition,
    apply_slot,
    work_efficient_slot_scan,
)
from spin9_dirac_clifford import (
    SPIN9_LIE_DIM,
    SPIN9_SPINOR_DIM,
    SPIN9_VECTOR_DIM,
    build_spin9_clifford_system,
)


def spin9_involutions(
    *, dtype: torch.dtype = torch.float64, device: torch.device | None = None
) -> torch.Tensor:
    system = build_spin9_clifford_system()
    return torch.as_tensor(system.involutions, dtype=dtype, device=device)


def spin9_bind(
    address: torch.Tensor,
    value: torch.Tensor,
    involutions: torch.Tensor,
) -> torch.Tensor:
    """Apply ``D(address)`` to a Spin(9) spinor value."""

    if address.shape[:-1] != value.shape[:-1]:
        raise ValueError("address and value must have equal leading shapes")
    if address.shape[-1] != SPIN9_VECTOR_DIM:
        raise ValueError("address must have final dimension 9")
    if value.shape[-1] != SPIN9_SPINOR_DIM:
        raise ValueError("value must have final dimension 16")
    if involutions.shape != (
        SPIN9_VECTOR_DIM,
        SPIN9_SPINOR_DIM,
        SPIN9_SPINOR_DIM,
    ):
        raise ValueError("involutions must have shape (9, 16, 16)")
    return torch.einsum("...i,iov,...v->...o", address, involutions, value)


def spin9_unbind(
    address: torch.Tensor,
    memory: torch.Tensor,
    involutions: torch.Tensor,
) -> torch.Tensor:
    """Invert binding for unit addresses; ``D(a)^2 = I``."""

    return spin9_bind(address, memory, involutions)


def spin9_hopf(spinor: torch.Tensor, involutions: torch.Tensor) -> torch.Tensor:
    """Return the nine quadratic Clifford expectations ``s^T P_i s``."""

    if spinor.shape[-1] != SPIN9_SPINOR_DIM:
        raise ValueError("spinor must have final dimension 16")
    return torch.einsum("...u,iuv,...v->...i", spinor, involutions, spinor)


def spin9_actions(
    coefficients: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exponentiate matched Spin(9) spinor and vector infinitesimal actions."""

    if coefficients.shape[-1] != SPIN9_LIE_DIM:
        raise ValueError("coefficients must have final dimension 36")
    dtype = coefficients.dtype if dtype is None else dtype
    device = coefficients.device if device is None else device
    system = build_spin9_clifford_system()
    spin_generators = 0.5 * torch.as_tensor(
        system.doubled_spin_generators, dtype=dtype, device=device
    )
    vector_generators = torch.as_tensor(
        system.vector_generators, dtype=dtype, device=device
    )
    coefficients = coefficients.to(dtype=dtype, device=device)
    spin = torch.matrix_exp(torch.einsum("...a,aij->...ij", coefficients, spin_generators))
    vector = torch.matrix_exp(
        torch.einsum("...a,aij->...ij", coefficients, vector_generators)
    )
    return vector, spin


def diagnostics(seed: int = 20260810) -> dict[str, object]:
    dtype = torch.float64
    generator = torch.Generator().manual_seed(seed)
    involutions = spin9_involutions(dtype=dtype)

    addresses = F.normalize(
        torch.randn(32, SPIN9_VECTOR_DIM, generator=generator, dtype=dtype), dim=-1
    )
    values = F.normalize(
        torch.randn(32, SPIN9_SPINOR_DIM, generator=generator, dtype=dtype), dim=-1
    )
    recovered = spin9_unbind(
        addresses, spin9_bind(addresses, values, involutions), involutions
    )

    first_address, second_address = addresses[:2]
    first_value, second_value = values[:2]
    superposed = spin9_bind(first_address, first_value, involutions) + spin9_bind(
        second_address, second_value, involutions
    )
    cross_term = spin9_unbind(first_address, superposed, involutions) - first_value

    coefficients = 0.08 * torch.randn(
        16, SPIN9_LIE_DIM, generator=generator, dtype=dtype
    )
    vector_action, spin_action = spin9_actions(coefficients)
    equivariant_left = spin9_bind(
        torch.einsum("bij,bj->bi", vector_action, addresses[:16]),
        torch.einsum("bij,bj->bi", spin_action, values[:16]),
        involutions,
    )
    equivariant_right = torch.einsum(
        "bij,bj->bi",
        spin_action,
        spin9_bind(addresses[:16], values[:16], involutions),
    )

    batch, length, slots = 2, 19, 4
    initial_addresses = F.normalize(
        torch.randn(
            batch, slots, SPIN9_VECTOR_DIM, generator=generator, dtype=dtype
        ),
        dim=-1,
    )
    initial_values = F.normalize(
        torch.randn(
            batch, slots, SPIN9_SPINOR_DIM, generator=generator, dtype=dtype
        ),
        dim=-1,
    )
    action_coefficients = 0.04 * torch.randn(
        batch, length, SPIN9_LIE_DIM, generator=generator, dtype=dtype
    )
    vector_actions, spin_actions = spin9_actions(action_coefficients)
    labels = torch.randint(slots, (batch, length), generator=generator)
    routes = F.one_hot(labels, slots).to(dtype=dtype)
    write_addresses = F.normalize(
        torch.randn(
            batch, length, SPIN9_VECTOR_DIM, generator=generator, dtype=dtype
        ),
        dim=-1,
    )
    write_values = F.normalize(
        torch.randn(
            batch, length, SPIN9_SPINOR_DIM, generator=generator, dtype=dtype
        ),
        dim=-1,
    )
    direct_transition = SlotTransition(
        1.0 - routes,
        spin_actions,
        routes[..., None] * write_values[:, :, None],
    )
    bound_writes = spin9_bind(write_addresses, write_values, involutions)
    bound_transition = SlotTransition(
        1.0 - routes,
        spin_actions,
        routes[..., None] * bound_writes[:, :, None],
    )
    direct_states = apply_slot(
        work_efficient_slot_scan(direct_transition), initial_values[:, None]
    )
    initial_bound = spin9_bind(initial_addresses, initial_values, involutions)
    bound_states = apply_slot(
        work_efficient_slot_scan(bound_transition), initial_bound[:, None]
    )

    current_addresses = initial_addresses
    address_prefixes = []
    for position in range(length):
        transported = torch.einsum(
            "bij,bhj->bhi", vector_actions[:, position], current_addresses
        )
        route = routes[:, position, :, None]
        current_addresses = (1.0 - route) * transported + route * write_addresses[
            :, position, None
        ]
        address_prefixes.append(current_addresses)
    address_states = torch.stack(address_prefixes, dim=1)
    gauged_direct = spin9_bind(address_states, direct_states, involutions)
    unbound = spin9_unbind(address_states, bound_states, involutions)

    hopf = spin9_hopf(values, involutions)
    result = {
        "experiment": "Spin(9) Clifford addressed-memory boundary",
        "seed": seed,
        "dimensions": {"address": 9, "value": 16, "memory_per_slot": 16},
        "single_pair_bind_unbind_max_abs_error": float((recovered - values).abs().max()),
        "equivariance_max_abs_error": float(
            (equivariant_left - equivariant_right).abs().max()
        ),
        "hopf_norm_identity_max_abs_error": float(
            (hopf.norm(dim=-1) - values.square().sum(dim=-1)).abs().max()
        ),
        "wrong_key_cross_term_norm_error": float(
            (cross_term.norm() - second_value.norm()).abs()
        ),
        "dynamic_bound_vs_gauged_direct_max_abs_error": float(
            (bound_states - gauged_direct).abs().max()
        ),
        "dynamic_unbound_vs_direct_max_abs_error": float(
            (unbound - direct_states).abs().max()
        ),
        "state_scalars": {
            "direct_slots": slots * SPIN9_SPINOR_DIM,
            "spin9_bound_slots": slots * SPIN9_SPINOR_DIM,
        },
        "claim_boundary": {
            "single_pair_binding_exact": True,
            "same_width_direct_gauge_equivalence_tested": True,
            "raw_superposition_cross_terms_are_norm_preserving": True,
            "spin9_specific_capacity_advantage_established": False,
            "cross_chiral_representation_structure_available": True,
            "spin9_specific_learned_prior_advantage_established": False,
            "learned_retrieval_advantage_established": False,
        },
    }
    tolerance = 2e-11
    result["passed"] = all(
        float(result[key]) < tolerance
        for key in (
            "single_pair_bind_unbind_max_abs_error",
            "equivariance_max_abs_error",
            "hopf_norm_identity_max_abs_error",
            "wrong_key_cross_term_norm_error",
            "dynamic_bound_vs_gauged_direct_max_abs_error",
            "dynamic_unbound_vs_direct_max_abs_error",
        )
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = diagnostics(args.seed)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
