"""Experimental sign-sensitive Spin(3) composition scan for PyTorch.

The maintained Pure Rotor v2.1 recurrence transports multivectors through
``Ad(q)``.  That action is intentionally blind to the central sign because
``Ad(q) == Ad(-q)``.  This module supplies the complementary operation: keep a
unit quaternion itself as recurrent state and compose token-dependent Spin(3)
elements without quotienting by the center.

The scan is a multiplicative group-state layer, not a replacement for the
bounded affine memory in :mod:`pure_rotor_ssm.torch_backend`.  It is kept in a
separate experimental module until broader tasks establish when the extra
sign-sensitive state helps outside finite-group tracking.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from . import GA_DIM


def unit_quaternion(parameters: torch.Tensor) -> torch.Tensor:
    """Normalize ``[..., 4]`` parameters with identity as the zero fallback.

    As with every radial normalization, there is no continuous unit-sphere
    extension at the zero vector.  Trainable token rotors are initialized away
    from zero; the fallback makes malformed external inputs deterministic.
    """

    if parameters.shape[-1] != 4:
        raise ValueError("quaternion parameters must end in four components")
    norm = torch.linalg.vector_norm(parameters, dim=-1, keepdim=True)
    threshold = torch.as_tensor(1e-6, dtype=parameters.dtype, device=parameters.device)
    normalized = parameters / norm.clamp_min(threshold)
    identity = torch.zeros_like(parameters)
    identity[..., 0] = 1
    return torch.where(norm > threshold, normalized, identity)


def quaternion_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product in scalar-first ``[w,x,y,z]`` coordinates."""

    if left.shape[-1] != 4 or right.shape[-1] != 4:
        raise ValueError("quaternions must end in four components")
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quaternion_to_rotor(quaternion: torch.Tensor) -> torch.Tensor:
    """Embed compact quaternions in the repository's Cl(3,0) basis order."""

    if quaternion.shape[-1] != 4:
        raise ValueError("quaternions must end in four components")
    scalar, x, y, z = quaternion.unbind(dim=-1)
    zeros = torch.zeros_like(scalar)
    return torch.stack((scalar, zeros, zeros, zeros, -z, y, -x, zeros), dim=-1)


def rotor_to_quaternion(rotor: torch.Tensor) -> torch.Tensor:
    """Extract scalar-first quaternion coordinates from an even Cl(3,0) rotor."""

    if rotor.shape[-1] != GA_DIM:
        raise ValueError(f"rotors must end in {GA_DIM} components")
    return torch.stack(
        (rotor[..., 0], -rotor[..., 6], rotor[..., 5], -rotor[..., 4]),
        dim=-1,
    )


def quaternion_prefix_scan(
    token_quaternions: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    valid_mask: torch.Tensor | None = None,
    mode: str = "parallel",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive unit-quaternion prefix products with a fixed-size cache.

    The update convention is ``state_next = state * token_rotor``.  ``parallel``
    uses a differentiable Hillis--Steele tree; ``recurrent`` is the sequential
    oracle and inference path.  Inputs and intermediate products are normalized
    to control floating-point norm drift.  Exact associativity therefore becomes
    tolerance-based parity between floating-point tree orders.
    """

    if (
        token_quaternions.ndim != 4
        or token_quaternions.shape[1] == 0
        or token_quaternions.shape[-1] != 4
    ):
        raise ValueError(
            "token_quaternions must have nonempty shape (batch,length,lanes,4)"
        )
    batch, length, lanes, _ = token_quaternions.shape
    if valid_mask is not None and valid_mask.shape != (batch, length):
        raise ValueError("valid_mask must have shape (batch,length)")

    tokens = unit_quaternion(token_quaternions)
    if valid_mask is not None:
        identity_tokens = torch.zeros_like(tokens)
        identity_tokens[..., 0] = 1
        tokens = torch.where(
            valid_mask.bool()[..., None, None], tokens, identity_tokens
        )

    if initial_state is None:
        initial_state = torch.zeros_like(tokens[:, 0])
        initial_state[..., 0] = 1
    elif initial_state.shape != (batch, lanes, 4):
        raise ValueError("initial_state must have shape (batch,lanes,4)")
    initial_state = unit_quaternion(initial_state)

    if mode == "recurrent":
        state = initial_state
        states = []
        for position in range(length):
            state = unit_quaternion(quaternion_product(state, tokens[:, position]))
            states.append(state)
        sequence = torch.stack(states, dim=1)
        return sequence, state
    if mode != "parallel":
        raise ValueError("mode must be 'parallel' or 'recurrent'")

    prefixes = tokens
    offset = 1
    while offset < length:
        products = unit_quaternion(
            quaternion_product(prefixes[:, :-offset], prefixes[:, offset:])
        )
        prefixes = torch.cat((prefixes[:, :offset], products), dim=1)
        offset *= 2
    sequence = unit_quaternion(quaternion_product(initial_state[:, None], prefixes))
    return sequence, sequence[:, -1]


class SpinTokenComposition(nn.Module):
    """Learn token-conditioned Spin(3) increments and scan their products."""

    def __init__(
        self, input_vocab_size: int, lanes: int, initialization_std: float = 0.5
    ) -> None:
        super().__init__()
        if input_vocab_size < 2 or lanes < 1 or initialization_std <= 0:
            raise ValueError("vocabulary, lanes, and initialization_std are invalid")
        self.input_vocab_size = input_vocab_size
        self.lanes = lanes
        self.token_rotors = nn.Parameter(torch.empty(input_vocab_size, lanes, 4))
        nn.init.normal_(self.token_rotors, mean=0.0, std=initialization_std)
        with torch.no_grad():
            self.token_rotors[..., 0].add_(1.0)

    @property
    def recurrent_state_scalars(self) -> int:
        return 4 * self.lanes

    def normalized_token_rotors(self) -> torch.Tensor:
        return unit_quaternion(self.token_rotors)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        state = torch.zeros(
            batch_size,
            self.lanes,
            4,
            device=device or self.token_rotors.device,
            dtype=dtype or self.token_rotors.dtype,
        )
        state[..., 0] = 1
        return state

    def forward(
        self,
        token_ids: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        scan_mode: str = "parallel",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_ids.ndim != 2 or token_ids.shape[1] == 0:
            raise ValueError("token_ids must have nonempty shape (batch,length)")
        if attention_mask is not None and attention_mask.shape != token_ids.shape:
            raise ValueError("attention_mask must match token_ids")
        token_rotors = self.normalized_token_rotors()[token_ids]
        return quaternion_prefix_scan(
            token_rotors,
            initial_state,
            valid_mask=attention_mask,
            mode=scan_mode,
        )


class SpinCompositionClassifier(nn.Module):
    """Minimal sign-sensitive sequence model used by structural benchmarks.

    This decoder is intentionally conventional.  The research object is the
    fixed-size multiplicative Spin state, not a claim that this small classifier
    is a general-purpose language architecture.
    """

    def __init__(
        self,
        *,
        input_vocab_size: int,
        output_size: int,
        lanes: int,
        decoder_hidden: int,
    ) -> None:
        super().__init__()
        if output_size < 2 or decoder_hidden < 1:
            raise ValueError("output_size and decoder_hidden are invalid")
        self.composition = SpinTokenComposition(input_vocab_size, lanes)
        self.output_size = output_size
        self.decoder = nn.Sequential(
            nn.Linear(4 * lanes, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, output_size),
        )

    @property
    def recurrent_state_scalars(self) -> int:
        return self.composition.recurrent_state_scalars

    def forward(
        self,
        token_ids: torch.Tensor,
        recurrent_state: torch.Tensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        return_recurrent_state: bool = False,
        scan_mode: str = "parallel",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        states, final_state = self.composition(
            token_ids,
            recurrent_state,
            attention_mask=attention_mask,
            scan_mode=scan_mode,
        )
        logits = self.decoder(states.flatten(start_dim=-2))
        if return_recurrent_state:
            return logits, final_state
        return logits

    def step(
        self, token_ids: torch.Tensor, recurrent_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_ids.ndim != 1:
            raise ValueError("step token_ids must have shape (batch,)")
        logits, state = self(
            token_ids[:, None],
            recurrent_state,
            return_recurrent_state=True,
            scan_mode="recurrent",
        )
        return logits[:, 0], state


__all__: Sequence[str] = (
    "SpinCompositionClassifier",
    "SpinTokenComposition",
    "quaternion_prefix_scan",
    "quaternion_product",
    "quaternion_to_rotor",
    "rotor_to_quaternion",
    "unit_quaternion",
)
