"""Maintained triality-faithful Spin(8) state-space layers for PyTorch.

The recurrent state is a tuple of vector, positive-chiral, and
negative-chiral eight-vectors. One shared 28-coordinate controller produces a
single Spin(8) element in all three representations. The tuple distinguishes
all four central signatures; one vector or one chiral stream alone does not.

The scan composes ordinary affine maps and is therefore associative. A bounded
drive parameterization gives a sequence-length-independent state-norm theorem.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

from pure_spin8_ssm import __version__
from pure_spin8_ssm.continuous_scan import continuous_spin8_scan
from spin8_triality import (
    SPIN8_BIVECTOR_DIM,
    SPIN8_DIM,
    TRIALITY_REPRESENTATIONS,
    spin8_actions,
    torch_triality_generators,
)
from spin8_triality_lift import triality_tensor

ActionMode = Literal["factorized", "exponential"]
ScanMode = Literal[
    "work_efficient", "hillis_steele", "recurrent", "compiled_recurrent"
]


@dataclass(frozen=True)
class Spin8AffineTransition:
    """Channel-wise `state -> scale * action(state) + drive` transition."""

    scale: torch.Tensor
    action: torch.Tensor
    drive: torch.Tensor


@dataclass(frozen=True)
class PureSpin8Config:
    """Serializable configuration for the maintained causal model."""

    vocab_size: int
    d_model: int = 64
    num_layers: int = 2
    channels: int = 2
    representations: tuple[str, ...] = TRIALITY_REPRESENTATIONS
    action_mode: ActionMode = "exponential"
    min_retention_logit: float = 2.0
    triality_coupling: bool = True
    transport_only: bool = False
    normalize_inputs: bool = True
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least two")
        if self.d_model < 1 or self.num_layers < 1 or self.channels < 1:
            raise ValueError("model dimensions must be positive")
        if self.action_mode not in ("factorized", "exponential"):
            raise ValueError("unknown Spin(8) action mode")
        if not self.representations or len(set(self.representations)) != len(
            self.representations
        ):
            raise ValueError("representations must be nonempty and unique")
        if any(name not in TRIALITY_REPRESENTATIONS for name in self.representations):
            raise ValueError("unknown triality representation")


def _identity(
    reference: torch.Tensor, *leading: int, dimension: int = SPIN8_DIM
) -> torch.Tensor:
    return torch.eye(dimension, dtype=reference.dtype, device=reference.device).expand(
        *leading, dimension, dimension
    )


def spin8_factorized_actions(
    coordinates: torch.Tensor,
    generators: torch.Tensor,
    representations: Sequence[str],
) -> torch.Tensor:
    """Ordered 28-plane product in the selected triality representations.

    Vector generators use the exact Givens polynomial. Half-spin generators
    square to `-I/4`, giving the exact two-term spin exponential. A `2*pi`
    coordinate is therefore identity in `8v` and minus identity in both
    spinors, retaining the central sign without a Cayley singularity.
    """

    representations = tuple(representations)
    if coordinates.shape[-1] != SPIN8_BIVECTOR_DIM:
        raise ValueError("coordinates must end in 28")
    if generators.shape != (
        len(representations),
        SPIN8_BIVECTOR_DIM,
        SPIN8_DIM,
        SPIN8_DIM,
    ):
        raise ValueError("generator shape does not match representations")

    leading = coordinates.shape[:-1]
    identity = torch.eye(SPIN8_DIM, dtype=coordinates.dtype, device=coordinates.device)
    theta = coordinates[..., :, None, None]
    factors = []
    for representation_index, representation in enumerate(representations):
        generator = generators[representation_index]
        if representation == "vector":
            factor = (
                identity
                + torch.sin(theta) * generator
                + (1.0 - torch.cos(theta)) * (generator @ generator)
            )
        else:
            factor = (
                torch.cos(0.5 * theta) * identity
                + 2.0 * torch.sin(0.5 * theta) * generator
            )
        factors.append(factor)

    # (..., coordinates, representations, 8, 8), then fold every leading
    # item and representation into one batched ordered reduction.
    factors = torch.stack(factors, dim=-3)
    coordinate_axis = len(leading)
    representation_axis = coordinate_axis + 1
    permutation = (
        *range(len(leading)),
        representation_axis,
        coordinate_axis,
        coordinate_axis + 2,
        coordinate_axis + 3,
    )
    nodes = factors.permute(permutation).reshape(
        -1, SPIN8_BIVECTOR_DIM, SPIN8_DIM, SPIN8_DIM
    )
    while nodes.shape[1] > 1:
        if nodes.shape[1] % 2:
            padding = identity.expand(nodes.shape[0], 1, SPIN8_DIM, SPIN8_DIM)
            nodes = torch.cat((nodes, padding), dim=1)
        nodes = nodes[:, 1::2] @ nodes[:, 0::2]
    return nodes[:, 0].reshape(*leading, len(representations), SPIN8_DIM, SPIN8_DIM)


def spin8_group_actions(
    coordinates: torch.Tensor,
    generators: torch.Tensor,
    representations: Sequence[str],
    *,
    mode: ActionMode,
) -> torch.Tensor:
    """Map shared bivector coordinates to a faithful triality action tuple."""

    if mode == "factorized":
        return spin8_factorized_actions(coordinates, generators, representations)
    if mode == "exponential":
        return spin8_actions(coordinates, generators)
    raise ValueError("mode must be 'factorized' or 'exponential'")


def compose_spin8_affine(
    after: Spin8AffineTransition, before: Spin8AffineTransition
) -> Spin8AffineTransition:
    """Compose `after(before(state))` with chronological orientation."""

    action = after.action @ before.action
    rotated_drive = torch.einsum("...crij,...crj->...cri", after.action, before.drive)
    return Spin8AffineTransition(
        scale=after.scale * before.scale,
        action=action,
        drive=after.drive + after.scale[..., None, None] * rotated_drive,
    )


def apply_spin8_affine(
    transition: Spin8AffineTransition, state: torch.Tensor
) -> torch.Tensor:
    rotated = torch.einsum("...crij,...crj->...cri", transition.action, state)
    return transition.scale[..., None, None] * rotated + transition.drive


def _identity_transition(
    reference: Spin8AffineTransition, length: int
) -> Spin8AffineTransition:
    batch, _, channels = reference.scale.shape[:3]
    representations = reference.action.shape[-3]
    scale = reference.scale.new_ones(batch, length, channels)
    action = _identity(
        reference.action,
        batch,
        length,
        channels,
        representations,
        dimension=SPIN8_DIM,
    ).clone()
    drive = reference.drive.new_zeros(
        batch, length, channels, representations, SPIN8_DIM
    )
    return Spin8AffineTransition(scale=scale, action=action, drive=drive)


def _cat_transition(
    first: Spin8AffineTransition, second: Spin8AffineTransition
) -> Spin8AffineTransition:
    return Spin8AffineTransition(
        scale=torch.cat((first.scale, second.scale), dim=1),
        action=torch.cat((first.action, second.action), dim=1),
        drive=torch.cat((first.drive, second.drive), dim=1),
    )


def _even(transition: Spin8AffineTransition) -> Spin8AffineTransition:
    return Spin8AffineTransition(
        scale=transition.scale[:, 0::2],
        action=transition.action[:, 0::2],
        drive=transition.drive[:, 0::2],
    )


def _odd(transition: Spin8AffineTransition) -> Spin8AffineTransition:
    return Spin8AffineTransition(
        scale=transition.scale[:, 1::2],
        action=transition.action[:, 1::2],
        drive=transition.drive[:, 1::2],
    )


def _interleave(
    left: Spin8AffineTransition, right: Spin8AffineTransition
) -> Spin8AffineTransition:
    batch, nodes, channels = left.scale.shape
    representations = left.action.shape[-3]
    return Spin8AffineTransition(
        scale=torch.stack((left.scale, right.scale), dim=2).reshape(
            batch, 2 * nodes, channels
        ),
        action=torch.stack((left.action, right.action), dim=2).reshape(
            batch,
            2 * nodes,
            channels,
            representations,
            SPIN8_DIM,
            SPIN8_DIM,
        ),
        drive=torch.stack((left.drive, right.drive), dim=2).reshape(
            batch, 2 * nodes, channels, representations, SPIN8_DIM
        ),
    )


def work_efficient_spin8_scan(
    transition: Spin8AffineTransition,
) -> Spin8AffineTransition:
    """Ordered inclusive Blelloch-style scan with linear composition work."""

    length = transition.scale.shape[1]
    if length < 1:
        raise ValueError("scan length must be positive")
    if length == 1:
        return transition
    padded_length = 1 << (length - 1).bit_length()
    if padded_length == length:
        leaves = transition
    else:
        leaves = _cat_transition(
            transition, _identity_transition(transition, padded_length - length)
        )

    levels = [leaves]
    nodes = leaves
    while nodes.scale.shape[1] > 1:
        nodes = compose_spin8_affine(_odd(nodes), _even(nodes))
        levels.append(nodes)

    exclusive = _identity_transition(transition, 1)
    for children in reversed(levels[:-1]):
        left_totals = _even(children)
        right_exclusive = compose_spin8_affine(left_totals, exclusive)
        exclusive = _interleave(exclusive, right_exclusive)
    inclusive = compose_spin8_affine(leaves, exclusive)
    return Spin8AffineTransition(
        scale=inclusive.scale[:, :length],
        action=inclusive.action[:, :length],
        drive=inclusive.drive[:, :length],
    )


def hillis_steele_spin8_scan(
    transition: Spin8AffineTransition,
) -> Spin8AffineTransition:
    length = transition.scale.shape[1]
    if length < 1:
        raise ValueError("scan length must be positive")
    prefixes = transition
    offset = 1
    while offset < length:
        after = Spin8AffineTransition(
            prefixes.scale[:, offset:],
            prefixes.action[:, offset:],
            prefixes.drive[:, offset:],
        )
        before = Spin8AffineTransition(
            prefixes.scale[:, :-offset],
            prefixes.action[:, :-offset],
            prefixes.drive[:, :-offset],
        )
        composed = compose_spin8_affine(after, before)
        prefixes = _cat_transition(
            Spin8AffineTransition(
                prefixes.scale[:, :offset],
                prefixes.action[:, :offset],
                prefixes.drive[:, :offset],
            ),
            composed,
        )
        offset *= 2
    return prefixes


def recurrent_spin8_scan(
    transition: Spin8AffineTransition, initial_state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial_state
    rows = []
    for position in range(transition.scale.shape[1]):
        state = apply_spin8_affine(
            Spin8AffineTransition(
                transition.scale[:, position],
                transition.action[:, position],
                transition.drive[:, position],
            ),
            state,
        )
        rows.append(state)
    return torch.stack(rows, dim=1), state


def mask_spin8_transition(
    transition: Spin8AffineTransition, valid_mask: torch.Tensor | None
) -> Spin8AffineTransition:
    if valid_mask is None:
        return transition
    if valid_mask.shape != transition.scale.shape[:2]:
        raise ValueError("valid_mask must have shape (batch,length)")
    valid = valid_mask.bool()
    identity = _identity_transition(transition, transition.scale.shape[1])
    return Spin8AffineTransition(
        scale=torch.where(valid[..., None], transition.scale, identity.scale),
        action=torch.where(
            valid[..., None, None, None, None], transition.action, identity.action
        ),
        drive=torch.where(
            valid[..., None, None, None], transition.drive, identity.drive
        ),
    )


def unit_ball(raw: torch.Tensor) -> torch.Tensor:
    """Smooth radial map into the open Euclidean unit ball."""

    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    return raw / (1.0 + norm)


class PureSpin8SSMLayer(nn.Module):
    """Bounded selective Spin(8) recurrence with a faithful triality cache."""

    def __init__(
        self,
        input_size: int,
        *,
        channels: int = 1,
        representations: Sequence[str] = TRIALITY_REPRESENTATIONS,
        action_mode: ActionMode = "exponential",
        min_retention_logit: float = 2.0,
        triality_coupling: bool = True,
        transport_only: bool = False,
        normalize_inputs: bool = True,
    ) -> None:
        super().__init__()
        if input_size < 1 or channels < 1:
            raise ValueError("input_size and channels must be positive")
        representations = tuple(representations)
        if not representations or len(set(representations)) != len(representations):
            raise ValueError("representations must be nonempty and unique")
        if any(name not in TRIALITY_REPRESENTATIONS for name in representations):
            raise ValueError("unknown triality representation")
        if action_mode not in ("factorized", "exponential"):
            raise ValueError("unknown action mode")

        self.input_size = input_size
        self.channels = channels
        self.representations = representations
        self.action_mode = action_mode
        self.min_retention_logit = min_retention_logit
        self.triality_coupling = triality_coupling
        self.transport_only = transport_only
        self.normalize_inputs = normalize_inputs
        self.input_norm = nn.RMSNorm(input_size) if normalize_inputs else nn.Identity()
        self.coefficient_controller = nn.Linear(
            input_size, channels * SPIN8_BIVECTOR_DIM
        )
        nn.init.zeros_(self.coefficient_controller.weight)
        nn.init.zeros_(self.coefficient_controller.bias)
        if transport_only:
            self.retention_controller = None
            self.write_controller = None
            self.drive_controller = None
        else:
            self.retention_controller = nn.Linear(input_size, channels)
            self.write_controller = nn.Linear(input_size, channels)
            self.drive_controller = nn.Linear(
                input_size, channels * len(representations) * SPIN8_DIM
            )
            nn.init.zeros_(self.retention_controller.weight)
            nn.init.zeros_(self.retention_controller.bias)
            nn.init.zeros_(self.write_controller.weight)
            nn.init.zeros_(self.write_controller.bias)
            nn.init.zeros_(self.drive_controller.bias)
        initial = torch.randn(channels, len(representations), SPIN8_DIM)
        self.initial_state = nn.Parameter(F.normalize(initial, dim=-1))
        self.register_buffer(
            "generators",
            torch_triality_generators(representations),
            persistent=True,
        )
        self.register_buffer("rho", triality_tensor(), persistent=True)
        self.coupling_logits = nn.Parameter(
            torch.full((channels, len(representations)), -4.0)
        )

    @property
    def cache_scalars(self) -> int:
        return self.channels * len(self.representations) * SPIN8_DIM

    @property
    def output_size(self) -> int:
        return self.cache_scalars

    def initial_cache(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        return (
            self.initial_state.to(reference).unsqueeze(0).expand(batch_size, -1, -1, -1)
        )

    def transitions(
        self, inputs: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> Spin8AffineTransition:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_size:
            raise ValueError("inputs must have shape (batch,length,input_size)")
        batch, length, _ = inputs.shape
        normalized = self.input_norm(inputs)
        coordinates = self.coefficient_controller(normalized).reshape(
            batch, length, self.channels, SPIN8_BIVECTOR_DIM
        )
        generators = self.generators.to(dtype=inputs.dtype, device=inputs.device)
        action = spin8_group_actions(
            coordinates,
            generators,
            self.representations,
            mode=self.action_mode,
        )
        if self.transport_only:
            scale = inputs.new_ones(batch, length, self.channels)
            drive = inputs.new_zeros(
                batch,
                length,
                self.channels,
                len(self.representations),
                SPIN8_DIM,
            )
        else:
            assert self.retention_controller is not None
            assert self.write_controller is not None
            assert self.drive_controller is not None
            scale = torch.sigmoid(
                self.min_retention_logit + self.retention_controller(normalized)
            )
            write = torch.sigmoid(self.write_controller(normalized))
            raw_drive = self.drive_controller(normalized).reshape(
                batch,
                length,
                self.channels,
                len(self.representations),
                SPIN8_DIM,
            )
            drive = (
                (1.0 - scale)[..., None, None]
                * write[..., None, None]
                * unit_ball(raw_drive)
            )
        return mask_spin8_transition(
            Spin8AffineTransition(scale=scale, action=action, drive=drive),
            valid_mask,
        )

    def triality_readout(self, states: torch.Tensor) -> torch.Tensor:
        """Add a gated equivariant triality interaction to readout states."""

        required = set(TRIALITY_REPRESENTATIONS)
        if not self.triality_coupling or set(self.representations) != required:
            return states
        indices = {name: self.representations.index(name) for name in required}
        vector = states[..., indices["vector"], :]
        positive = states[..., indices["positive"], :]
        negative = states[..., indices["negative"], :]
        rho = self.rho.to(states)
        bound_vector = torch.einsum("...j,vji,...i->...v", negative, rho, positive)
        bound_positive = torch.einsum("...j,vji,...v->...i", negative, rho, vector)
        bound_negative = torch.einsum("...i,vji,...v->...j", positive, rho, vector)
        derived_by_name = {
            "vector": bound_vector,
            "positive": bound_positive,
            "negative": bound_negative,
        }
        derived = torch.stack(
            [derived_by_name[name] for name in self.representations], dim=-2
        )
        gate = torch.sigmoid(self.coupling_logits).to(states)
        leading_ones = (1,) * (states.ndim - 3)
        gate = gate.reshape(*leading_ones, self.channels, len(self.representations), 1)
        return states + gate * derived

    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        scan_mode: ScanMode = "work_efficient",
        return_raw_states: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transition = self.transitions(inputs, valid_mask)
        if state is None:
            state = self.initial_cache(inputs.shape[0], inputs)
        expected = (
            inputs.shape[0],
            self.channels,
            len(self.representations),
            SPIN8_DIM,
        )
        if state.shape != expected:
            raise ValueError(f"state must have shape {expected}")
        if scan_mode == "compiled_recurrent":
            raw = continuous_spin8_scan(
                transition.action,
                transition.scale,
                transition.drive,
                state,
                backend="auto",
            )
            final_state = raw[:, -1]
        elif scan_mode == "recurrent":
            raw, final_state = recurrent_spin8_scan(transition, state)
        else:
            if scan_mode == "work_efficient":
                prefixes = work_efficient_spin8_scan(transition)
            elif scan_mode == "hillis_steele":
                prefixes = hillis_steele_spin8_scan(transition)
            else:
                raise ValueError("unknown scan mode")
            raw = apply_spin8_affine(prefixes, state[:, None])
            final_state = raw[:, -1]
        return (raw if return_raw_states else self.triality_readout(raw)), final_state

    def step(
        self, inputs: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 2:
            raise ValueError("step inputs must have shape (batch,input_size)")
        outputs, state = self.forward(inputs[:, None], state, scan_mode="recurrent")
        return outputs[:, 0], state


class PureSpin8Block(nn.Module):
    """Residual model block around one maintained Spin(8) SSM layer."""

    def __init__(self, config: PureSpin8Config) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(config.d_model)
        self.ssm = PureSpin8SSMLayer(
            config.d_model,
            channels=config.channels,
            representations=config.representations,
            action_mode=config.action_mode,
            min_retention_logit=config.min_retention_logit,
            triality_coupling=config.triality_coupling,
            transport_only=config.transport_only,
            normalize_inputs=config.normalize_inputs,
        )
        self.state_norm = nn.RMSNorm(self.ssm.output_size)
        self.output_projection = nn.Linear(self.ssm.output_size, config.d_model)
        self.residual_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        hidden: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        scan_mode: ScanMode = "work_efficient",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states, state = self.ssm(
            self.norm(hidden),
            state,
            valid_mask=valid_mask,
            scan_mode=scan_mode,
        )
        update = self.output_projection(self.state_norm(states.flatten(start_dim=-3)))
        return hidden + torch.sigmoid(self.residual_gate) * update, state


class PureSpin8Model(nn.Module):
    """Stack of triality-faithful Spin(8) blocks for continuous embeddings."""

    def __init__(self, config: PureSpin8Config) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            [PureSpin8Block(config) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.RMSNorm(config.d_model)

    @property
    def cache_scalars(self) -> int:
        return sum(block.ssm.cache_scalars for block in self.blocks)

    def forward(
        self,
        hidden: torch.Tensor,
        states: Sequence[torch.Tensor | None] | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        scan_mode: ScanMode = "work_efficient",
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if hidden.ndim != 3 or hidden.shape[-1] != self.config.d_model:
            raise ValueError("hidden must have shape (batch,length,d_model)")
        if states is None:
            states = [None] * len(self.blocks)
        if len(states) != len(self.blocks):
            raise ValueError("one recurrent state is required per block")
        next_states = []
        for block, state in zip(self.blocks, states):
            hidden, next_state = block(
                hidden,
                state,
                valid_mask=valid_mask,
                scan_mode=scan_mode,
            )
            next_states.append(next_state)
        return self.final_norm(hidden), next_states


class PureSpin8CausalLM(nn.Module):
    """Checkpointed causal language-model shell around Pure Spin(8) blocks."""

    def __init__(self, config: PureSpin8Config) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.backbone = PureSpin8Model(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    @property
    def cache_scalars(self) -> int:
        return self.backbone.cache_scalars

    def forward(
        self,
        token_ids: torch.Tensor,
        states: Sequence[torch.Tensor | None] | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        scan_mode: ScanMode = "work_efficient",
    ) -> dict[str, Any]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch,length)")
        hidden, states = self.backbone(
            self.embedding(token_ids),
            states,
            valid_mask=valid_mask,
            scan_mode=scan_mode,
        )
        logits = self.lm_head(hidden)
        result: dict[str, Any] = {"logits": logits, "states": states}
        if labels is not None:
            if labels.shape != token_ids.shape:
                raise ValueError("labels must match token_ids")
            result["loss"] = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.config.vocab_size),
                labels[:, 1:].reshape(-1),
            )
        return result

    def save_checkpoint(
        self, path: str | Path, *, metadata: dict[str, Any] | None = None
    ) -> None:
        torch.save(
            {
                "format_version": 1,
                "model_type": "pure_spin8_causal_lm",
                "model_version": __version__,
                "config": asdict(self.config),
                "state_dict": {
                    k: v.detach().cpu() for k, v in self.state_dict().items()
                },
                "metadata": metadata or {},
            },
            Path(path),
        )

    @classmethod
    def load_checkpoint(
        cls, path: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> PureSpin8CausalLM:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if payload.get("model_type") != "pure_spin8_causal_lm":
            raise ValueError("not a Pure Spin(8) causal LM checkpoint")
        if payload.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format")
        config_payload = dict(payload["config"])
        config_payload["representations"] = tuple(config_payload["representations"])
        model = cls(PureSpin8Config(**config_payload))
        model.load_state_dict(payload["state_dict"])
        return model


__all__ = [
    "ActionMode",
    "PureSpin8Block",
    "PureSpin8CausalLM",
    "PureSpin8Config",
    "PureSpin8Model",
    "PureSpin8SSMLayer",
    "ScanMode",
    "Spin8AffineTransition",
    "apply_spin8_affine",
    "compose_spin8_affine",
    "hillis_steele_spin8_scan",
    "mask_spin8_transition",
    "recurrent_spin8_scan",
    "spin8_factorized_actions",
    "spin8_group_actions",
    "unit_ball",
    "work_efficient_spin8_scan",
]
