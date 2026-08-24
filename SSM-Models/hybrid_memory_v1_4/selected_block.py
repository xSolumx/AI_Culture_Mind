"""Semantic selected-block memory with sparse and full-state reference paths.

This module is a correctness reference, not a fused-kernel or performance
claim. ``physical_gather`` gathers and scatters only hard-selected blocks. The
two dense semantic modes materialize the exact full-state affine operator and
therefore scale quadratically in the total slot count.

Dense modes expose three explicit coarse-routing semantics: ``hard`` keeps the
non-differentiable argmax reference, ``soft`` uses the full block simplex, and
``straight_through`` has the hard forward value with a softmax surrogate
gradient. Physical gather is intentionally restricted to ``hard`` routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

ScanMode = Literal["physical_gather", "dense_recurrent", "dense_parallel"]
RouteMode = Literal["hard", "soft", "straight_through"]
DiagnosticValue = torch.Tensor | bool | str


def _dtype_bytes(dtype: torch.dtype) -> int:
    try:
        return torch.empty((), dtype=dtype).element_size()
    except (RuntimeError, TypeError) as error:
        raise ValueError(
            "dtype must be a torch dtype with fixed-width elements"
        ) from error


@dataclass(frozen=True)
class SelectedBlockConfig:
    """Configuration for :class:`SelectedBlockMemory`.

    ``retention_min`` and ``retention_max`` bound the learned local retention
    target. ``retention_max`` is strictly below one, which supplies a finite
    coordinate-wise state bound for bounded drives whenever a block is
    selected for writing.
    """

    model_dim: int
    heads: int
    blocks: int
    slots_per_block: int
    value_dim: int
    update_rank: int = 1
    controller_rank: int | None = None
    retention_min: float = 0.0
    retention_max: float = 0.99

    def __post_init__(self) -> None:
        dimensions = (
            self.model_dim,
            self.heads,
            self.blocks,
            self.slots_per_block,
            self.value_dim,
            self.update_rank,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in dimensions
        ):
            raise TypeError("all dimensions and update_rank must be integers")
        if min(dimensions) < 1:
            raise ValueError("all dimensions and update_rank must be positive")
        if self.update_rank > self.slots_per_block:
            raise ValueError("update_rank cannot exceed slots_per_block")
        if self.controller_rank is not None:
            if isinstance(self.controller_rank, bool) or not isinstance(
                self.controller_rank, int
            ):
                raise TypeError("controller_rank must be an integer or None")
            if self.controller_rank < 1:
                raise ValueError("controller_rank must be positive")
        if not 0.0 <= self.retention_min <= self.retention_max < 1.0:
            raise ValueError(
                "retention bounds must satisfy 0 <= retention_min <= retention_max < 1"
            )

    @property
    def slots(self) -> int:
        return self.blocks * self.slots_per_block

    @property
    def state_shape(self) -> tuple[int, int, int, int]:
        return (self.heads, self.blocks, self.slots_per_block, self.value_dim)

    @property
    def state_scalars(self) -> int:
        return self.heads * self.blocks * self.slots_per_block * self.value_dim

    @property
    def retention_bounds(self) -> tuple[float, float]:
        return (self.retention_min, self.retention_max)

    def state_bytes(
        self, dtype: torch.dtype = torch.float32, *, batch_size: int = 1
    ) -> int:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return batch_size * self.state_scalars * _dtype_bytes(dtype)


class LowRankLinear(nn.Module):
    """Linear map whose effective weight always has rank at most ``rank``.

    The weight is represented only by trainable down/up factors. There is no
    independently trainable dense weight that could leave the rank-constrained
    parameterization during optimization.
    """

    def __init__(
        self, in_features: int, out_features: int, rank: int, *, bias: bool = True
    ) -> None:
        super().__init__()
        if min(in_features, out_features, rank) < 1:
            raise ValueError("in_features, out_features, and rank must be positive")
        if rank > min(in_features, out_features):
            raise ValueError("rank cannot exceed min(in_features, out_features)")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(inputs))

    def effective_weight(self) -> torch.Tensor:
        """Return the derived dense weight for inspection, not as a parameter."""

        return self.up.weight @ self.down.weight


@dataclass(frozen=True)
class FullStateAffineTransition:
    """Full-state map ``S -> linear @ S + drive``.

    Scan transitions have shapes ``linear (B,L,H,N,N)`` and
    ``drive (B,L,H,N,V)``, where ``N = blocks * slots_per_block``.
    """

    linear: torch.Tensor
    drive: torch.Tensor

    @property
    def bias(self) -> torch.Tensor:
        return self.drive


def apply_full_state_transition(
    transition: FullStateAffineTransition, state: torch.Tensor
) -> torch.Tensor:
    """Apply one affine transition or a broadcast family of transitions."""

    if (
        transition.linear.ndim < 2
        or transition.linear.shape[-1] != transition.linear.shape[-2]
    ):
        raise ValueError("linear must end in square state axes")
    if transition.drive.ndim < 2:
        raise ValueError("drive must end in state and value axes")
    if transition.drive.shape[-2] != transition.linear.shape[-1]:
        raise ValueError("linear and drive state dimensions must agree")
    if state.shape[-2:] != transition.drive.shape[-2:]:
        raise ValueError("state must end in the transition state and value dimensions")
    return transition.linear @ state + transition.drive


def compose_full_state_transitions(
    later: FullStateAffineTransition, earlier: FullStateAffineTransition
) -> FullStateAffineTransition:
    """Compose chronological transitions, returning ``later o earlier``."""

    if later.linear.shape != earlier.linear.shape:
        raise ValueError("composed linear tensors must have identical shapes")
    if later.drive.shape != earlier.drive.shape:
        raise ValueError("composed drive tensors must have identical shapes")
    if later.linear.shape[:-2] != later.drive.shape[:-2]:
        raise ValueError("linear and drive leading dimensions must agree")
    return FullStateAffineTransition(
        linear=later.linear @ earlier.linear,
        drive=later.drive + later.linear @ earlier.drive,
    )


def full_state_prefix_scan(
    transition: FullStateAffineTransition,
) -> FullStateAffineTransition:
    """Inclusive Hillis-Steele scan along sequence axis one."""

    _validate_scan_transition(transition)
    current = transition
    offset = 1
    length = transition.linear.shape[1]
    while offset < length:
        later = FullStateAffineTransition(
            current.linear[:, offset:], current.drive[:, offset:]
        )
        earlier = FullStateAffineTransition(
            current.linear[:, :-offset], current.drive[:, :-offset]
        )
        composed = compose_full_state_transitions(later, earlier)
        current = FullStateAffineTransition(
            linear=torch.cat((current.linear[:, :offset], composed.linear), dim=1),
            drive=torch.cat((current.drive[:, :offset], composed.drive), dim=1),
        )
        offset *= 2
    return current


def recurrent_full_state_scan(
    transition: FullStateAffineTransition, initial_state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential full-state semantic reference."""

    _validate_scan_inputs(transition, initial_state)
    state = initial_state
    states = []
    for position in range(transition.linear.shape[1]):
        token = FullStateAffineTransition(
            transition.linear[:, position], transition.drive[:, position]
        )
        state = apply_full_state_transition(token, state)
        states.append(state)
    stacked = torch.stack(states, dim=1)
    return stacked, state


def parallel_full_state_scan(
    transition: FullStateAffineTransition, initial_state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-depth full-state scan with the same chronological convention."""

    _validate_scan_inputs(transition, initial_state)
    prefix = full_state_prefix_scan(transition)
    states = apply_full_state_transition(prefix, initial_state[:, None])
    return states, states[:, -1]


def _validate_scan_transition(transition: FullStateAffineTransition) -> None:
    if transition.linear.ndim != 5:
        raise ValueError("linear must have shape (B,L,H,N,N)")
    batch, length, heads, slots, slots_again = transition.linear.shape
    if min(batch, length, heads, slots) < 1 or slots != slots_again:
        raise ValueError("linear must be a nonempty sequence of square state maps")
    if transition.drive.ndim != 5:
        raise ValueError("drive must have shape (B,L,H,N,V)")
    if transition.drive.shape[:4] != (batch, length, heads, slots):
        raise ValueError("drive has incompatible batch, sequence, head, or state axes")
    if transition.drive.shape[-1] < 1:
        raise ValueError("drive value dimension must be nonempty")
    if transition.linear.dtype != transition.drive.dtype:
        raise ValueError("linear and drive must have the same dtype")
    if transition.linear.device != transition.drive.device:
        raise ValueError("linear and drive must be on the same device")


def _validate_scan_inputs(
    transition: FullStateAffineTransition, initial_state: torch.Tensor
) -> None:
    _validate_scan_transition(transition)
    batch, _, heads, slots, _ = transition.linear.shape
    value_dim = transition.drive.shape[-1]
    if initial_state.shape != (batch, heads, slots, value_dim):
        raise ValueError("initial_state must have shape (B,H,N,V)")
    if initial_state.dtype != transition.linear.dtype:
        raise ValueError("initial_state and transition must have the same dtype")
    if initial_state.device != transition.linear.device:
        raise ValueError("initial_state and transition must be on the same device")


@dataclass(frozen=True)
class _ControllerOutput:
    write_block_logits: torch.Tensor
    erase_block_logits: torch.Tensor
    read_block_logits: torch.Tensor
    write_fine_logits: torch.Tensor
    erase_fine_logits: torch.Tensor
    read_fine_logits: torch.Tensor
    write_block: torch.Tensor
    erase_block: torch.Tensor
    read_block: torch.Tensor
    write_local: torch.Tensor
    erase_local: torch.Tensor
    read_local: torch.Tensor
    write_gate: torch.Tensor
    retention: torch.Tensor
    erase_strength: torch.Tensor
    values: torch.Tensor


class SelectedBlockMemory(nn.Module):
    """Hierarchical semantic memory with explicit sparse and dense modes.

    State shape is ``(B,H,blocks,slots_per_block,value_dim)``. At each valid
    position, the mutation is the affine law

    ``S' = R_write R_erase S + D_write``.

    Both retention maps are diagonal contractions. Erase and write addresses
    are independently parameterized rank-r simplex families, and the value
    vectors are bounded by ``tanh``. The valid mask and learned write gate
    affect mutation only. Reads are evaluated after the update at every input
    position, including positions whose mutation mask is false.
    """

    scan_modes = ("physical_gather", "dense_recurrent", "dense_parallel")
    route_modes = ("hard", "soft", "straight_through")

    def __init__(self, config: SelectedBlockConfig) -> None:
        super().__init__()
        if not isinstance(config, SelectedBlockConfig):
            raise TypeError("config must be a SelectedBlockConfig")
        self.config = config
        self.state_scalars = config.state_scalars
        self._controller_width = self._controller_output_width()
        if config.controller_rank is None:
            self.controller: nn.Module = nn.Linear(
                config.model_dim, self._controller_width, bias=True
            )
        else:
            if config.controller_rank > min(config.model_dim, self._controller_width):
                raise ValueError(
                    "controller_rank cannot exceed the controller input or output width"
                )
            self.controller = LowRankLinear(
                config.model_dim,
                self._controller_width,
                config.controller_rank,
                bias=True,
            )
        feature_width = config.heads * (config.value_dim + 1)
        self.output = nn.Linear(feature_width, config.model_dim, bias=False)

    def _controller_output_width(self) -> int:
        c = self.config
        per_head = (
            3 * c.blocks
            + 2 * c.blocks * c.update_rank * c.slots_per_block
            + c.blocks * c.slots_per_block
            + 2
            + c.update_rank
            + c.update_rank * c.value_dim
        )
        return c.heads * per_head

    def state_bytes(
        self, dtype: torch.dtype | None = None, *, batch_size: int = 1
    ) -> int:
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return self.config.state_bytes(dtype, batch_size=batch_size)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        parameter = next(self.parameters())
        if device is None:
            device = parameter.device
        if dtype is None:
            dtype = parameter.dtype
        return torch.zeros(
            batch_size, *self.config.state_shape, device=device, dtype=dtype
        )

    def compile_semantic_transition(
        self,
        inputs: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        route_mode: RouteMode = "hard",
    ) -> tuple[FullStateAffineTransition, dict[str, DiagnosticValue]]:
        """Materialize the exact dense affine transition and routing diagnostics."""

        self._validate_route_mode(route_mode)
        mask = self._validate_inputs(inputs, None, valid_mask)[1]
        controls = self._compute_controls(inputs)
        transition = self._compile_full_transition(controls, mask, route_mode)
        return transition, self._diagnostics(controls, route_mode)

    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        scan_mode: ScanMode = "physical_gather",
        route_mode: RouteMode = "hard",
        return_diagnostics: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, dict[str, DiagnosticValue]]
    ):
        """Run one of the three explicitly named implementations.

        ``physical_gather`` is the sparse recurrent implementation.
        ``dense_recurrent`` and ``dense_parallel`` are semantic oracles and
        intentionally materialize full-state operators. Other mode names fail
        closed rather than silently changing implementation semantics.
        """

        if scan_mode not in self.scan_modes:
            raise ValueError(f"scan_mode must be one of {self.scan_modes}")
        self._validate_route_mode(route_mode)
        if scan_mode == "physical_gather" and route_mode != "hard":
            raise ValueError("physical_gather requires route_mode='hard'")
        state, mask = self._validate_inputs(inputs, state, valid_mask)
        controls = self._compute_controls(inputs)
        if scan_mode == "physical_gather":
            raw_reads, final_state = self._physical_gather_scan(controls, state, mask)
        else:
            transition = self._compile_full_transition(controls, mask, route_mode)
            flat_state = state.flatten(2, 3)
            if scan_mode == "dense_recurrent":
                states, flat_final = recurrent_full_state_scan(transition, flat_state)
            else:
                states, flat_final = parallel_full_state_scan(transition, flat_state)
            raw_reads = self._read_dense_states(states, controls, route_mode)
            final_state = flat_final.reshape(
                inputs.shape[0],
                self.config.heads,
                self.config.blocks,
                self.config.slots_per_block,
                self.config.value_dim,
            )
        outputs = self._project_reads(raw_reads)
        if return_diagnostics:
            return outputs, final_state, self._diagnostics(controls, route_mode)
        return outputs, final_state

    def step(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, dict[str, DiagnosticValue]]
    ):
        """Run one sparse recurrent step without allocating a state trajectory."""

        if inputs.ndim != 2:
            raise ValueError("step inputs must have shape (B,model_dim)")
        sequence_mask = None
        if valid_mask is not None:
            if valid_mask.ndim != 1:
                raise ValueError("step valid_mask must have shape (B,)")
            sequence_mask = valid_mask[:, None]
        result = self.forward(
            inputs[:, None],
            state,
            valid_mask=sequence_mask,
            scan_mode="physical_gather",
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            outputs, final_state, diagnostics = result
            squeezed = {
                key: value[:, 0] if isinstance(value, torch.Tensor) else value
                for key, value in diagnostics.items()
            }
            return outputs[:, 0], final_state, squeezed
        outputs, final_state = result
        return outputs[:, 0], final_state

    def _validate_inputs(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if inputs.ndim != 3 or inputs.shape[1] < 1:
            raise ValueError("inputs must have nonempty shape (B,L,model_dim)")
        if inputs.shape[0] < 1 or inputs.shape[-1] != self.config.model_dim:
            raise ValueError("inputs have incompatible batch or model dimensions")
        if not inputs.is_floating_point():
            raise TypeError("inputs must have a floating-point dtype")
        parameter = next(self.parameters())
        if inputs.dtype != parameter.dtype or inputs.device != parameter.device:
            raise ValueError("inputs must match the module parameter dtype and device")
        if not bool(torch.isfinite(inputs).all()):
            raise ValueError("inputs must be finite")
        expected_state = (inputs.shape[0], *self.config.state_shape)
        if state is None:
            state = self.initial_state(inputs.shape[0])
        elif not isinstance(state, torch.Tensor):
            raise TypeError("state must be a torch.Tensor or None")
        if state.shape != expected_state:
            raise ValueError(
                "state must have shape (B,H,blocks,slots_per_block,value_dim)"
            )
        if not state.is_floating_point():
            raise TypeError("state must have a floating-point dtype")
        if state.dtype != inputs.dtype or state.device != inputs.device:
            raise ValueError("state must match the input dtype and device")
        if not bool(torch.isfinite(state).all()):
            raise ValueError("state must be finite")
        if valid_mask is None:
            mask = torch.ones(inputs.shape[:2], dtype=torch.bool, device=inputs.device)
        else:
            if not isinstance(valid_mask, torch.Tensor):
                raise TypeError("valid_mask must be a torch.Tensor or None")
            if valid_mask.shape != inputs.shape[:2]:
                raise ValueError("valid_mask must have shape (B,L)")
            if valid_mask.dtype != torch.bool:
                raise TypeError("valid_mask must have dtype torch.bool")
            if valid_mask.device != inputs.device:
                raise ValueError("valid_mask must be on the input device")
            mask = valid_mask
        return state, mask

    def _compute_controls(self, inputs: torch.Tensor) -> _ControllerOutput:
        c = self.config
        batch, length, _ = inputs.shape
        raw = self.controller(inputs)
        cursor = 0

        def take(*shape: int) -> torch.Tensor:
            nonlocal cursor
            width = 1
            for dimension in shape:
                width *= dimension
            result = raw[..., cursor : cursor + width].reshape(batch, length, *shape)
            cursor += width
            return result

        write_block_logits = take(c.heads, c.blocks)
        erase_block_logits = take(c.heads, c.blocks)
        read_block_logits = take(c.heads, c.blocks)
        write_fine_logits = take(c.heads, c.blocks, c.update_rank, c.slots_per_block)
        erase_fine_logits = take(c.heads, c.blocks, c.update_rank, c.slots_per_block)
        read_fine_logits = take(c.heads, c.blocks, c.slots_per_block)
        write_gate = torch.sigmoid(take(c.heads))
        retention_logit = take(c.heads)
        erase_strength = torch.sigmoid(take(c.heads, c.update_rank))
        values = torch.tanh(take(c.heads, c.update_rank, c.value_dim))
        if cursor != self._controller_width:
            raise RuntimeError("internal controller layout is inconsistent")

        write_block = write_block_logits.argmax(dim=-1)
        erase_block = erase_block_logits.argmax(dim=-1)
        read_block = read_block_logits.argmax(dim=-1)
        write_local_logits = self._gather_rank_fine(write_fine_logits, write_block)
        erase_local_logits = self._gather_rank_fine(erase_fine_logits, erase_block)
        read_local_logits = self._gather_read_fine(read_fine_logits, read_block)
        retention = c.retention_min + (
            c.retention_max - c.retention_min
        ) * torch.sigmoid(retention_logit)
        return _ControllerOutput(
            write_block_logits=write_block_logits,
            erase_block_logits=erase_block_logits,
            read_block_logits=read_block_logits,
            write_fine_logits=write_fine_logits,
            erase_fine_logits=erase_fine_logits,
            read_fine_logits=read_fine_logits,
            write_block=write_block,
            erase_block=erase_block,
            read_block=read_block,
            write_local=F.softmax(write_local_logits, dim=-1),
            erase_local=F.softmax(erase_local_logits, dim=-1),
            read_local=F.softmax(read_local_logits, dim=-1),
            write_gate=write_gate,
            retention=retention,
            erase_strength=erase_strength,
            values=values,
        )

    def _validate_route_mode(self, route_mode: str) -> None:
        if route_mode not in self.route_modes:
            raise ValueError(f"route_mode must be one of {self.route_modes}")

    def _block_weights(
        self,
        logits: torch.Tensor,
        indices: torch.Tensor,
        route_mode: RouteMode,
    ) -> torch.Tensor:
        hard = F.one_hot(indices, self.config.blocks).to(logits.dtype)
        if route_mode == "hard":
            return hard
        soft = F.softmax(logits, dim=-1)
        if route_mode == "soft":
            return soft
        return hard + (soft - soft.detach())

    @staticmethod
    def _gather_rank_fine(logits: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
        rank, slots = logits.shape[-2:]
        index = block[..., None, None, None].expand(*block.shape, 1, rank, slots)
        return torch.gather(logits, 3, index).squeeze(3)

    @staticmethod
    def _gather_read_fine(logits: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
        slots = logits.shape[-1]
        index = block[..., None, None].expand(*block.shape, 1, slots)
        return torch.gather(logits, 3, index).squeeze(3)

    def _compile_full_transition(
        self,
        controls: _ControllerOutput,
        valid_mask: torch.Tensor,
        route_mode: RouteMode = "hard",
    ) -> FullStateAffineTransition:
        c = self.config
        dtype = controls.write_gate.dtype
        write_block_mask = self._block_weights(
            controls.write_block_logits, controls.write_block, route_mode
        )
        erase_block_mask = self._block_weights(
            controls.erase_block_logits, controls.erase_block, route_mode
        )
        write_fine = F.softmax(controls.write_fine_logits, dim=-1)
        erase_fine = F.softmax(controls.erase_fine_logits, dim=-1)
        write_address = (write_block_mask[..., :, None, None] * write_fine).permute(
            0, 1, 2, 4, 3, 5
        )
        erase_address = (erase_block_mask[..., :, None, None] * erase_fine).permute(
            0, 1, 2, 4, 3, 5
        )
        write_address = write_address.reshape(*write_address.shape[:4], c.slots)
        erase_address = erase_address.reshape(*erase_address.shape[:4], c.slots)
        erase_profile = (erase_address * controls.erase_strength[..., None]).mean(
            dim=-2
        )
        write_slot_mask = (
            write_block_mask[..., :, None]
            .expand(*write_block_mask.shape, c.slots_per_block)
            .reshape(*write_block_mask.shape[:-1], c.slots)
        )
        gate = controls.write_gate * valid_mask[..., None].to(dtype)
        erase_retention = (
            1.0 - gate[..., None] * controls.retention[..., None] * erase_profile
        )
        write_retention = (
            1.0
            - gate[..., None] * (1.0 - controls.retention[..., None]) * write_slot_mask
        )
        retention = erase_retention * write_retention
        drive = (
            torch.einsum("blhrn,blhrv->blhnv", write_address, controls.values)
            / c.update_rank
        )
        drive = drive * gate[..., None, None]
        return FullStateAffineTransition(torch.diag_embed(retention), drive)

    def _physical_gather_scan(
        self,
        controls: _ControllerOutput,
        initial_state: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.config
        state = initial_state
        reads = []
        for position in range(valid_mask.shape[1]):
            gate = controls.write_gate[:, position] * valid_mask[:, position, None].to(
                controls.write_gate.dtype
            )
            retention = controls.retention[:, position]

            erase_index = controls.erase_block[:, position]
            erase_state = self._gather_state_block(state, erase_index)
            erase_profile = (
                controls.erase_strength[:, position, :, :, None]
                * controls.erase_local[:, position]
            ).mean(dim=-2)
            erase_factor = 1.0 - gate[..., None] * retention[..., None] * erase_profile
            state = self._scatter_state_block(
                state, erase_index, erase_state * erase_factor[..., None]
            )

            write_index = controls.write_block[:, position]
            write_state = self._gather_state_block(state, write_index)
            write_factor = 1.0 - gate * (1.0 - retention)
            local_drive = (
                torch.einsum(
                    "bhrs,bhrv->bhsv",
                    controls.write_local[:, position],
                    controls.values[:, position],
                )
                / c.update_rank
            )
            write_state = (
                write_state * write_factor[..., None, None]
                + gate[..., None, None] * local_drive
            )
            state = self._scatter_state_block(state, write_index, write_state)

            read_state = self._gather_state_block(
                state, controls.read_block[:, position]
            )
            reads.append(
                torch.einsum(
                    "bhs,bhsv->bhv", controls.read_local[:, position], read_state
                )
            )
        return torch.stack(reads, dim=1), state

    @staticmethod
    def _gather_state_block(state: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
        batch, heads, _, slots, value_dim = state.shape
        if block.shape != (batch, heads):
            raise ValueError("block indices must have shape (B,H)")
        index = block[..., None, None, None].expand(batch, heads, 1, slots, value_dim)
        return torch.gather(state, 2, index).squeeze(2)

    @staticmethod
    def _scatter_state_block(
        state: torch.Tensor, block: torch.Tensor, selected: torch.Tensor
    ) -> torch.Tensor:
        batch, heads, _, slots, value_dim = state.shape
        if selected.shape != (batch, heads, slots, value_dim):
            raise ValueError("selected block has incompatible shape")
        index = block[..., None, None, None].expand(batch, heads, 1, slots, value_dim)
        return state.scatter(2, index, selected.unsqueeze(2))

    def _read_dense_states(
        self,
        states: torch.Tensor,
        controls: _ControllerOutput,
        route_mode: RouteMode = "hard",
    ) -> torch.Tensor:
        c = self.config
        read_block_mask = self._block_weights(
            controls.read_block_logits, controls.read_block, route_mode
        )
        read_fine = F.softmax(controls.read_fine_logits, dim=-1)
        read_address = (read_block_mask[..., :, None] * read_fine).reshape(
            *read_block_mask.shape[:-1], c.slots
        )
        return torch.einsum("blhn,blhnv->blhv", read_address, states)

    def _project_reads(self, raw_reads: torch.Tensor) -> torch.Tensor:
        energy = raw_reads.square().sum(dim=-1, keepdim=True)
        direction = raw_reads * torch.rsqrt(1.0 + energy)
        log_energy = torch.log1p(energy)
        features = torch.cat((direction, log_energy), dim=-1).flatten(-2)
        return self.output(features)

    @staticmethod
    def _diagnostics(
        controls: _ControllerOutput,
        route_mode: RouteMode = "hard",
    ) -> dict[str, DiagnosticValue]:
        return {
            "write_block_logits": controls.write_block_logits,
            "erase_block_logits": controls.erase_block_logits,
            "read_block_logits": controls.read_block_logits,
            "write_fine_logits": controls.write_fine_logits,
            "erase_fine_logits": controls.erase_fine_logits,
            "read_fine_logits": controls.read_fine_logits,
            "write_block_indices": controls.write_block,
            "erase_block_indices": controls.erase_block,
            "read_block_indices": controls.read_block,
            "write_gate": controls.write_gate,
            "retention": controls.retention,
            "route_mode": route_mode,
            "hard_block_selection_differentiable": route_mode == "straight_through",
        }


SelectedBlockMemoryConfig = SelectedBlockConfig
SemanticAffineTransition = FullStateAffineTransition
apply_transition = apply_full_state_transition
compose_transitions = compose_full_state_transitions
parallel_semantic_scan = parallel_full_state_scan
recurrent_semantic_scan = recurrent_full_state_scan


__all__ = [
    "FullStateAffineTransition",
    "LowRankLinear",
    "RouteMode",
    "SelectedBlockConfig",
    "SelectedBlockMemory",
    "SelectedBlockMemoryConfig",
    "SemanticAffineTransition",
    "apply_full_state_transition",
    "apply_transition",
    "compose_full_state_transitions",
    "compose_transitions",
    "full_state_prefix_scan",
    "parallel_full_state_scan",
    "parallel_semantic_scan",
    "recurrent_full_state_scan",
    "recurrent_semantic_scan",
]
