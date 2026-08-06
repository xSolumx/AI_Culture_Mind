"""Tensor-only selective rotor/delta SSM for the isolated benchmark."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

GA_DIM = 8
BASIS_MASKS = (0, 1, 2, 4, 3, 5, 6, 7)
GRADE_SLICES = ((0, 1), (1, 4), (4, 7), (7, 8))
REVERSION_SIGNS = torch.tensor((1, 1, 1, 1, -1, -1, -1, -1))


def _multiplication_table() -> torch.Tensor:
    table = torch.zeros(GA_DIM, GA_DIM, GA_DIM)
    lookup = {mask: index for index, mask in enumerate(BASIS_MASKS)}
    for left_index, left_mask in enumerate(BASIS_MASKS):
        for right_index, right_mask in enumerate(BASIS_MASKS):
            swaps = sum(
                (right_mask & ((1 << bit) - 1)).bit_count()
                for bit in range(3)
                if left_mask & (1 << bit)
            )
            table[lookup[left_mask ^ right_mask], left_index, right_index] = (
                -1 if swaps % 2 else 1
            )
    return table


MULTIPLICATION_TABLE = _multiplication_table()
ROTOR_INDICES = (0, 4, 5, 6)
ROTOR_PRODUCT_TABLE = MULTIPLICATION_TABLE[list(ROTOR_INDICES)][
    :, list(ROTOR_INDICES), :
][:, :, list(ROTOR_INDICES)]


def _rotor_action_table() -> torch.Tensor:
    """Return Q[k,j,a,b] for (R x reverse(R))_k = Q r_a r_b x_j."""
    table = torch.zeros(GA_DIM, GA_DIM, 4, 4)
    basis = torch.eye(GA_DIM)
    for a, rotor_index in enumerate(ROTOR_INDICES):
        for b, reverse_index in enumerate(ROTOR_INDICES):
            left = basis[rotor_index].expand(GA_DIM, -1)
            right = basis[reverse_index].expand(GA_DIM, -1)
            right = reversion(right)
            transformed = geometric_product(geometric_product(left, basis), right)
            table[:, :, a, b] = transformed.T
    return table


def geometric_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape[-1] != GA_DIM or right.shape[-1] != GA_DIM:
        raise ValueError("multivectors must end in eight components")
    table = MULTIPLICATION_TABLE.to(
        device=left.device, dtype=torch.result_type(left, right)
    )
    return torch.einsum("...i,...j,kij->...k", left, right, table)


def reversion(multivector: torch.Tensor) -> torch.Tensor:
    signs = REVERSION_SIGNS.to(device=multivector.device, dtype=multivector.dtype)
    return multivector * signs


ROTOR_ACTION_TABLE = _rotor_action_table()


def rotor_sandwich(rotor: torch.Tensor, multivector: torch.Tensor) -> torch.Tensor:
    rotor, multivector = torch.broadcast_tensors(rotor, multivector)
    return geometric_product(geometric_product(rotor, multivector), reversion(rotor))


def rotor_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Fast product for compact four- or full eight-coefficient rotors."""
    if left.shape[-1] not in (4, GA_DIM) or right.shape[-1] not in (4, GA_DIM):
        raise ValueError("rotors must use four even or eight multivector coefficients")
    compact = left.shape[-1] == 4 and right.shape[-1] == 4
    left_even = left if left.shape[-1] == 4 else left[..., list(ROTOR_INDICES)]
    right_even = right if right.shape[-1] == 4 else right[..., list(ROTOR_INDICES)]
    table = ROTOR_PRODUCT_TABLE.to(
        device=left.device, dtype=torch.result_type(left, right)
    )
    product = torch.einsum("...i,...j,kij->...k", left_even, right_even, table)
    if compact:
        return product
    output = torch.zeros_like(left)
    output[..., list(ROTOR_INDICES)] = product
    return output


def rotor_sandwich_fast(rotor: torch.Tensor, multivector: torch.Tensor) -> torch.Tensor:
    """Fast rotor action using a precomputed quadratic coefficient table."""
    if rotor.shape[-1] not in (4, GA_DIM) or multivector.shape[-1] != GA_DIM:
        raise ValueError("rotor must use four even or eight multivector coefficients")
    coefficients = rotor if rotor.shape[-1] == 4 else rotor[..., list(ROTOR_INDICES)]
    table = ROTOR_ACTION_TABLE.to(device=rotor.device, dtype=rotor.dtype)
    # The two GEMM-like contractions benchmark faster on the target RTX 2070
    # than the single four-operand contraction.  The intermediate is bounded
    # to the fixed 8x8 blade action and remains fully differentiable.
    action = torch.einsum("...a,...b,kjab->...kj", coefficients, coefficients, table)
    return torch.einsum("...kj,...j->...k", action, multivector)


def rotor_affine_scan(
    decay: torch.Tensor,
    rotors: torch.Tensor,
    drive: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable logarithmic-depth scan of rotor-affine maps.

    Each step is the affine map ``h -> d R h reverse(R) + b``. Composition is
    associative, so the scan computes all prefixes in O(log L) composition
    rounds instead of launching one tiny kernel per token. It is exact up to
    floating-point reassociation.
    """
    if rotors.shape[:-1] != drive.shape[:-1] or rotors.shape[:-1] != decay.shape:
        raise ValueError("decay, rotors, and drive have incompatible shapes")
    prefix_decay, prefix_rotor, prefix_drive = decay, rotors, drive
    length = decay.shape[1]
    offset = 1
    while offset < length:
        current_decay = prefix_decay[:, offset:]
        previous_decay = prefix_decay[:, :-offset]
        current_rotor = prefix_rotor[:, offset:]
        previous_rotor = prefix_rotor[:, :-offset]
        current_drive = prefix_drive[:, offset:]
        previous_drive = prefix_drive[:, :-offset]
        composed_rotor = rotor_product(current_rotor, previous_rotor)
        composed_drive = current_drive + current_decay[..., None] * rotor_sandwich_fast(
            current_rotor, previous_drive
        )
        composed_decay = current_decay * previous_decay
        prefix_decay = torch.cat((prefix_decay[:, :offset], composed_decay), dim=1)
        prefix_rotor = torch.cat((prefix_rotor[:, :offset], composed_rotor), dim=1)
        prefix_drive = torch.cat((prefix_drive[:, :offset], composed_drive), dim=1)
        offset *= 2
    if initial_state is not None:
        carried = rotor_sandwich_fast(prefix_rotor, initial_state[:, None])
        states = prefix_decay[..., None] * carried + prefix_drive
    else:
        states = prefix_drive
    return states, states[:, -1]


def grade_invariants(multivector: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            multivector[..., 0],
            multivector[..., 1:4].norm(dim=-1),
            multivector[..., 4:7].norm(dim=-1),
            multivector[..., 7],
        ),
        dim=-1,
    )


def pack_cl3_isotypic(multivector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack Cl(3) into its complete ``1+3+3+1`` isotypic coordinates."""
    if multivector.ndim < 2 or multivector.shape[-1] != GA_DIM:
        raise ValueError("multivectors must have shape (..., channels, 8)")
    channels = multivector.shape[-2]
    trivial = torch.stack((multivector[..., 0], multivector[..., 7]), dim=-1)
    trivial = trivial.reshape(*multivector.shape[:-2], 2 * channels)
    dual_bivector = torch.stack(
        (multivector[..., 6], -multivector[..., 5], multivector[..., 4]), dim=-1
    )
    active = torch.stack((multivector[..., 1:4], dual_bivector), dim=-2)
    active = active.reshape(*multivector.shape[:-2], 2 * channels, 3)
    return trivial, active


def unpack_cl3_isotypic(trivial: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Invert :func:`pack_cl3_isotypic`."""
    if trivial.shape[:-1] != active.shape[:-2] or active.shape[-1] != 3:
        raise ValueError("incompatible isotypic shapes")
    if trivial.shape[-1] != active.shape[-2] or trivial.shape[-1] % 2:
        raise ValueError("isotypic multiplicities must be even and agree")
    channels = trivial.shape[-1] // 2
    trivial = trivial.reshape(*trivial.shape[:-1], channels, 2)
    active = active.reshape(*active.shape[:-2], channels, 2, 3)
    vector, dual_bivector = active.unbind(dim=-2)
    output = trivial.new_zeros(*trivial.shape[:-1], GA_DIM)
    output[..., 0] = trivial[..., 0]
    output[..., 1:4] = vector
    output[..., 4] = dual_bivector[..., 2]
    output[..., 5] = -dual_bivector[..., 1]
    output[..., 6] = dual_bivector[..., 0]
    output[..., 7] = trivial[..., 1]
    return output


class Spin3IsotypicLinear(nn.Module):
    """Complete real Spin(3)-equivariant channel mixing.

    GradeLinear is a strict four-kernel subfamily.  The two trivial copies
    (scalar/pseudoscalar) and two vector copies (vector/Hodge-bivector) may
    each mix arbitrarily, giving the eight-kernel Schur commutant.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.trivial_kernel = nn.Parameter(torch.empty(out_channels, 2, in_channels, 2))
        self.active_kernel = nn.Parameter(torch.empty(out_channels, 2, in_channels, 2))
        nn.init.kaiming_uniform_(self.trivial_kernel, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.active_kernel, a=math.sqrt(5))
        self.trivial_bias = nn.Parameter(torch.zeros(out_channels, 2)) if bias else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-2:] != (self.in_channels, GA_DIM):
            raise ValueError("unexpected isotypic projection shape")
        trivial, active = pack_cl3_isotypic(inputs)
        # Flatten multiplicity copies and use GEMM.  The same map is shared
        # across the three active coordinates, as required by equivariance.
        trivial_kernel = self.trivial_kernel.reshape(
            self.out_channels * 2, self.in_channels * 2
        )
        trivial_output = torch.matmul(trivial, trivial_kernel.t())
        active = active.transpose(-1, -2)
        active_kernel = self.active_kernel.reshape(
            self.out_channels * 2, self.in_channels * 2
        )
        active_output = torch.matmul(active, active_kernel.t()).transpose(-1, -2)
        if self.trivial_bias is not None:
            trivial_output = trivial_output + self.trivial_bias.reshape(-1)
        return unpack_cl3_isotypic(trivial_output, active_output)


def normalized_rotor(parameters: torch.Tensor) -> torch.Tensor:
    parameters = parameters / parameters.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    scalar, e12, e13, e23 = parameters.chunk(4, dim=-1)
    zeros = torch.zeros_like(scalar)
    return torch.cat((scalar, zeros, zeros, zeros, e12, e13, e23, zeros), dim=-1)


def rotor_from_bivector(
    bivector: torch.Tensor, max_angle: float = math.pi / 2
) -> torch.Tensor:
    if bivector.shape[-1] != 3:
        raise ValueError("bivectors must end in three components")
    magnitude = bivector.norm(dim=-1, keepdim=True)
    angle = max_angle * torch.tanh(magnitude)
    scale = torch.sin(angle / 2) / magnitude.clamp_min(1e-7)
    scale = torch.where(
        magnitude > 1e-7,
        scale,
        torch.as_tensor(max_angle / 2, device=bivector.device, dtype=bivector.dtype),
    )
    return normalized_rotor(
        torch.cat((torch.cos(angle / 2), -scale * bivector), dim=-1)
    )


def rotor_coefficients_from_bivector(
    bivector: torch.Tensor, max_angle: float = math.pi / 2
) -> torch.Tensor:
    """Return the four nonzero even-rotor coefficients used by the scan."""
    if bivector.shape[-1] != 3:
        raise ValueError("bivectors must end in three components")
    magnitude = bivector.norm(dim=-1, keepdim=True)
    angle = max_angle * torch.tanh(magnitude)
    scale = torch.sin(angle / 2) / magnitude.clamp_min(1e-7)
    scale = torch.where(
        magnitude > 1e-7,
        scale,
        torch.as_tensor(max_angle / 2, device=bivector.device, dtype=bivector.dtype),
    )
    return torch.cat((torch.cos(angle / 2), -scale * bivector), dim=-1)


class GradeLinear(nn.Module):
    """A grade-preserving tensor projection over multivector channels."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(4, out_channels, in_channels))
        nn.init.kaiming_uniform_(self.kernel, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-2:] != (self.kernel.shape[-1], GA_DIM):
            raise ValueError("unexpected multivector projection shape")
        pieces = [
            torch.einsum("oi,...ic->...oc", self.kernel[g], inputs[..., start:stop])
            for g, (start, stop) in enumerate(GRADE_SLICES)
        ]
        outputs = torch.cat(pieces, dim=-1)
        if self.bias is not None:
            scalar = outputs[..., 0] + self.bias
            outputs = torch.cat((scalar.unsqueeze(-1), outputs[..., 1:]), dim=-1)
        return outputs


class GeometricRMSNorm(nn.Module):
    def __init__(self, channels: int, epsilon: float = 1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(channels, 1))
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        rms = inputs.square().mean(dim=(-2, -1), keepdim=True)
        return inputs * torch.rsqrt(rms + self.epsilon) * self.gain


class SelectiveRotorDeltaSSM(nn.Module):
    """Stable selective rotor transport plus an independently gated write.

    The base transition is
    ``h_t = d_t R_t h_(t-1) reverse(R_t) + b_t``.
    The drive uses ``(1-decay) * sigmoid(write) * candidate``.  Erase and
    write are independent, while the transition remains an exact affine map
    for the differentiable associative scan.
    """

    def __init__(
        self,
        channels: int,
        min_half_life: float = 4.0,
        max_half_life: float = 2048.0,
        max_rotor_angle: float = math.pi / 2,
    ):
        super().__init__()
        self.channels = channels
        self.max_rotor_angle = max_rotor_angle
        self.minimum_step_size = 1e-2
        self.minimum_decay_rate = 1e-4
        invariant_dim = channels * 4
        self.step_control = nn.Linear(invariant_dim, channels)
        self.rotor_control = nn.Linear(invariant_dim, channels)
        self.write_control = nn.Linear(invariant_dim, channels)
        nn.init.zeros_(self.step_control.weight)
        nn.init.zeros_(self.step_control.bias)
        nn.init.zeros_(self.rotor_control.weight)
        nn.init.zeros_(self.rotor_control.bias)
        nn.init.zeros_(self.write_control.weight)
        # Start with a neutral 50% write gate; the convex coefficient remains
        # bounded by ``1 - decay`` and the controller can learn to close it.
        nn.init.zeros_(self.write_control.bias)
        self.rotor_source = Spin3IsotypicLinear(channels, channels, bias=False)
        self.input_projection = Spin3IsotypicLinear(channels, channels)
        half_lives = torch.logspace(
            math.log10(min_half_life), math.log10(max_half_life), channels
        )
        expected_step = self.minimum_step_size + math.log(2.0)
        target_rates = math.log(2.0) / (half_lives * expected_step)
        free_rates = target_rates - self.minimum_decay_rate
        if bool(torch.any(free_rates <= 0)):
            raise ValueError("half-life range is incompatible with decay floor")
        self.log_rates = nn.Parameter(torch.log(torch.expm1(free_rates)))

    def transitions(self, inputs: torch.Tensor):
        invariants = grade_invariants(inputs).flatten(-2)
        step = self.minimum_step_size + F.softplus(self.step_control(invariants))
        rates = self.minimum_decay_rate + F.softplus(self.log_rates)
        decay = torch.exp(-step * rates)
        strength = torch.tanh(self.rotor_control(invariants))
        source = self.rotor_source(inputs)[..., 4:7]
        rotors = rotor_coefficients_from_bivector(
            source * strength.unsqueeze(-1), self.max_rotor_angle
        )
        candidate = self.input_projection(inputs)
        write_gate = torch.sigmoid(self.write_control(invariants))
        # Independent bounded write: old state is erased by decay, while the
        # innovation is separately controlled and remains BIBO/convex stable.
        drive = (1.0 - decay).unsqueeze(-1) * write_gate.unsqueeze(-1) * candidate
        return decay, rotors, drive

    def forward(self, inputs: torch.Tensor, initial_state: torch.Tensor | None = None):
        decay, rotors, drive = self.transitions(inputs)
        return rotor_affine_scan(decay, rotors, drive, initial_state)


class GeometricGatedFFN(nn.Module):
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.input = Spin3IsotypicLinear(channels, hidden)
        self.gate = nn.Linear(hidden * 4, hidden)
        self.output = Spin3IsotypicLinear(hidden, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input(inputs)
        gates = torch.sigmoid(self.gate(grade_invariants(hidden).flatten(-2)))
        return self.output(hidden * gates.unsqueeze(-1))


class SpinorDeltaBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm1 = GeometricRMSNorm(channels)
        self.ssm = SelectiveRotorDeltaSSM(channels)
        self.norm2 = GeometricRMSNorm(channels)
        self.ffn = GeometricGatedFFN(channels, expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, initial_state: torch.Tensor | None = None):
        sequence, state = self.ssm(self.norm1(inputs), initial_state)
        outputs = inputs + self.dropout(sequence)
        outputs = outputs + self.dropout(self.ffn(self.norm2(outputs)))
        return outputs, state


class SpinorDeltaLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        channels: int = 48,
        layers: int = 4,
        expansion: int = 2,
        dropout: float = 0.0,
        decoder_channels: int | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.channels = channels
        self.layers = layers
        self.decoder_channels = decoder_channels
        self.token_embeddings = nn.Parameter(torch.empty(vocab_size, channels, GA_DIM))
        nn.init.normal_(self.token_embeddings, std=0.02)
        self.blocks = nn.ModuleList(
            SpinorDeltaBlock(channels, expansion, dropout) for _ in range(layers)
        )
        self.final_norm = GeometricRMSNorm(channels)
        self.vocabulary_bias = nn.Parameter(torch.zeros(vocab_size))
        self.initial_state = nn.Parameter(torch.zeros(layers, channels, GA_DIM))
        if decoder_channels is None:
            self.decoder_projection = None
            self.output_projection = None
        else:
            if decoder_channels < 1:
                raise ValueError("decoder_channels must be positive")
            self.decoder_projection = Spin3IsotypicLinear(channels, decoder_channels)
            self.output_projection = nn.Linear(
                decoder_channels * GA_DIM, vocab_size, bias=False
            )

    def forward(
        self,
        token_ids: torch.Tensor,
        recurrent_states: Sequence[torch.Tensor] | None = None,
        return_states: bool = False,
    ):
        if token_ids.ndim != 2 or token_ids.shape[1] < 1:
            raise ValueError("token_ids must have shape (batch, nonempty sequence)")
        if recurrent_states is None:
            recurrent_states = tuple(
                self.initial_state[layer].expand(token_ids.shape[0], -1, -1)
                for layer in range(self.layers)
            )
        if len(recurrent_states) != self.layers:
            raise ValueError("one recurrent state is required per layer")
        outputs = self.token_embeddings[token_ids]
        states = []
        for layer, block in enumerate(self.blocks):
            outputs, state = block(outputs, recurrent_states[layer])
            states.append(state)
        outputs = self.final_norm(outputs)
        if self.decoder_projection is None:
            logits = torch.einsum("blci,vci->blv", outputs, self.token_embeddings)
            logits = logits / math.sqrt(self.channels * GA_DIM)
        else:
            decoded = self.decoder_projection(outputs).flatten(-2)
            logits = self.output_projection(decoded) / math.sqrt(decoded.shape[-1])
        logits = logits + self.vocabulary_bias
        if return_states:
            return logits, tuple(states)
        return logits


__all__ = [
    "SpinorDeltaLM",
    "SpinorDeltaBlock",
    "SelectiveRotorDeltaSSM",
    "Spin3IsotypicLinear",
    "GradeLinear",
    "pack_cl3_isotypic",
    "unpack_cl3_isotypic",
    "rotor_coefficients_from_bivector",
]
