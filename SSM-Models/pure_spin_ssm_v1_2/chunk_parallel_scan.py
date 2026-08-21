"""Chunk-parallel associative affine prefix compiler for continuous Spin(8).

Each token transition is an affine pair ``(aR, d)``. Within each chunk, local
prefixes are compiled in parallel across chunks. Chunk endpoints are then
scanned with the maintained ordered work-efficient tree, and every local
prefix is applied in parallel to its chunk's incoming state.
"""

from __future__ import annotations

import torch
from pure_spin8_ssm.torch_backend import (
    Spin8AffineTransition,
    apply_spin8_affine,
    compose_spin8_affine,
    work_efficient_spin8_scan,
)

REPRESENTATIONS = 3
STATE_DIMENSION = 8


def factorized_triality_actions(
    coordinates: torch.Tensor, generators: torch.Tensor
) -> torch.Tensor:
    """Materialize an arbitrary ordered generator subset in all triality views."""

    if coordinates.ndim != 4:
        raise ValueError("coordinates must have shape (B,L,C,F)")
    factors = coordinates.shape[-1]
    if generators.shape != (REPRESENTATIONS, factors, STATE_DIMENSION, STATE_DIMENSION):
        raise ValueError("generators must have shape (3,F,8,8)")
    if coordinates.device != generators.device or coordinates.dtype != generators.dtype:
        raise ValueError("coordinates and generators must share device and dtype")
    identity = torch.eye(
        STATE_DIMENSION, dtype=coordinates.dtype, device=coordinates.device
    )
    theta = coordinates[..., :, None, None]
    representation_factors = []
    vector_generators = generators[0]
    representation_factors.append(
        identity
        + torch.sin(theta) * vector_generators
        + (1.0 - torch.cos(theta)) * (vector_generators @ vector_generators)
    )
    for representation in (1, 2):
        representation_factors.append(
            torch.cos(0.5 * theta) * identity
            + 2.0 * torch.sin(0.5 * theta) * generators[representation]
        )
    # B,L,C,R,F,8,8 -> collapse B,L,C,R and reduce F chronologically.
    nodes = torch.stack(representation_factors, dim=3).reshape(
        -1, factors, STATE_DIMENSION, STATE_DIMENSION
    )
    while nodes.shape[1] > 1:
        if nodes.shape[1] % 2:
            padding = identity.expand(nodes.shape[0], 1, -1, -1)
            nodes = torch.cat((nodes, padding), dim=1)
        nodes = nodes[:, 1::2] @ nodes[:, 0::2]
    return nodes[:, 0].reshape(
        *coordinates.shape[:-1], REPRESENTATIONS, STATE_DIMENSION, STATE_DIMENSION
    )


def _slice_local(
    transition: Spin8AffineTransition, position: int
) -> Spin8AffineTransition:
    return Spin8AffineTransition(
        scale=transition.scale[:, :, position],
        action=transition.action[:, :, position],
        drive=transition.drive[:, :, position],
    )


def _stack_local(
    transitions: list[Spin8AffineTransition],
) -> Spin8AffineTransition:
    return Spin8AffineTransition(
        scale=torch.stack([item.scale for item in transitions], dim=2),
        action=torch.stack([item.action for item in transitions], dim=2),
        drive=torch.stack([item.drive for item in transitions], dim=2),
    )


def _pad_transition(
    transition: Spin8AffineTransition, padding: int
) -> Spin8AffineTransition:
    if padding == 0:
        return transition
    batch, _, channels = transition.scale.shape
    representations = transition.action.shape[-3]
    identity = torch.eye(
        STATE_DIMENSION,
        dtype=transition.action.dtype,
        device=transition.action.device,
    ).expand(batch, padding, channels, representations, -1, -1)
    return Spin8AffineTransition(
        scale=torch.cat(
            (transition.scale, transition.scale.new_ones(batch, padding, channels)),
            dim=1,
        ),
        action=torch.cat((transition.action, identity), dim=1),
        drive=torch.cat(
            (
                transition.drive,
                transition.drive.new_zeros(
                    batch,
                    padding,
                    channels,
                    representations,
                    STATE_DIMENSION,
                ),
            ),
            dim=1,
        ),
    )


def chunk_parallel_spin8_scan(
    transition: Spin8AffineTransition,
    initial_state: torch.Tensor,
    *,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Emit every causal state with local chunk compilation and endpoint scan."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if transition.scale.ndim != 3:
        raise ValueError("scale must have shape (B,L,C)")
    batch, length, channels = transition.scale.shape
    if length < 1:
        raise ValueError("sequence length must be positive")
    representations = transition.action.shape[-3]
    if transition.action.shape != (
        batch,
        length,
        channels,
        representations,
        STATE_DIMENSION,
        STATE_DIMENSION,
    ):
        raise ValueError("action has incompatible shape")
    if transition.drive.shape != (
        batch,
        length,
        channels,
        representations,
        STATE_DIMENSION,
    ):
        raise ValueError("drive has incompatible shape")
    if initial_state.shape != (
        batch,
        channels,
        representations,
        STATE_DIMENSION,
    ):
        raise ValueError("initial_state has incompatible shape")

    chunks = (length + chunk_size - 1) // chunk_size
    padded_length = chunks * chunk_size
    padded = _pad_transition(transition, padded_length - length)
    chunked = Spin8AffineTransition(
        scale=padded.scale.reshape(batch, chunks, chunk_size, channels),
        action=padded.action.reshape(
            batch,
            chunks,
            chunk_size,
            channels,
            representations,
            STATE_DIMENSION,
            STATE_DIMENSION,
        ),
        drive=padded.drive.reshape(
            batch,
            chunks,
            chunk_size,
            channels,
            representations,
            STATE_DIMENSION,
        ),
    )
    prefix = _slice_local(chunked, 0)
    local_prefixes = [prefix]
    for position in range(1, chunk_size):
        prefix = compose_spin8_affine(_slice_local(chunked, position), prefix)
        local_prefixes.append(prefix)
    local = _stack_local(local_prefixes)
    endpoints = Spin8AffineTransition(
        scale=local.scale[:, :, -1],
        action=local.action[:, :, -1],
        drive=local.drive[:, :, -1],
    )
    endpoint_prefixes = work_efficient_spin8_scan(endpoints)
    if chunks == 1:
        incoming = initial_state[:, None]
    else:
        earlier = Spin8AffineTransition(
            scale=endpoint_prefixes.scale[:, :-1],
            action=endpoint_prefixes.action[:, :-1],
            drive=endpoint_prefixes.drive[:, :-1],
        )
        earlier_states = apply_spin8_affine(earlier, initial_state[:, None])
        incoming = torch.cat((initial_state[:, None], earlier_states), dim=1)
    rotated = torch.einsum(
        "bqkcrij,bqcrj->bqkcri", local.action, incoming
    )
    states = local.scale[..., None, None] * rotated + local.drive
    states = states.reshape(
        batch, padded_length, channels, representations, STATE_DIMENSION
    )[:, :length]
    return states, states[:, -1]


__all__ = [
    "chunk_parallel_spin8_scan",
    "factorized_triality_actions",
]
