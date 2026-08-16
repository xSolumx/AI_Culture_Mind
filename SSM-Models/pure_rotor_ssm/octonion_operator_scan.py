"""Associative multiplication-operator scans for octonionic recurrences.

Octonion multiplication is nonassociative, so raw octonions cannot be the
binary operator of a parallel prefix tree.  Linear maps *do* compose
associatively.  This module therefore lifts a unit octonion ``u`` to its real
left- or right-multiplication matrix and scans those operators in chronological
order.

For the left recurrence ``h_t = u_t h_(t-1)``, the prefix action is

``L(u_t) @ ... @ L(u_1)``.

In general this is not ``L(u_t ... u_1)``.  The difference is the octonion
associator and is intentionally preserved.  Parallel training materializes
operator prefixes, while streaming inference evaluates the parenthesized raw
product and keeps only eight state scalars per lane.

The trainable layer adds the contractive affine update

``h_t = d_t L(u_t) h_(t-1) + (1-d_t) w_t z_t``

with unit ``u_t``, gates in ``[0,1]``, and ``z_t`` in the closed unit ball.
Consequently ``||h_t|| <= max(||h_0||, 1)`` at every length.  This experimental
module is separate from maintained Pure Rotor v2.1.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache
from typing import Literal

import torch
from torch import nn

OCTONION_DIM = 8
FANO_TRIPLES = (
    (1, 2, 3),
    (1, 4, 5),
    (1, 7, 6),
    (2, 4, 6),
    (2, 5, 7),
    (3, 4, 7),
    (3, 6, 5),
)
ScanMode = Literal["work_efficient", "hillis_steele", "recurrent"]
OperatorSide = Literal["left", "right"]


def _build_integer_structure_constants() -> torch.Tensor:
    """Return ``C[k,i,j]`` for ``e_i e_j = sum_k C[k,i,j] e_k``."""

    structure = torch.zeros(OCTONION_DIM, OCTONION_DIM, OCTONION_DIM, dtype=torch.int8)
    for index in range(OCTONION_DIM):
        structure[index, 0, index] = 1
        structure[index, index, 0] = 1
    for index in range(1, OCTONION_DIM):
        structure[0, index, index] = -1
    for first, second, third in FANO_TRIPLES:
        for left, right, product in (
            (first, second, third),
            (second, third, first),
            (third, first, second),
        ):
            structure[product, left, right] = 1
            structure[product, right, left] = -1
    if int(torch.count_nonzero(structure)) != OCTONION_DIM * OCTONION_DIM:
        raise AssertionError("the fixed Fano plane did not define every basis product")
    return structure


_INTEGER_STRUCTURE_CONSTANTS = _build_integer_structure_constants()


@lru_cache(maxsize=32)
def _cached_structure_constants(
    dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return _INTEGER_STRUCTURE_CONSTANTS.to(dtype=dtype, device=device)


def octonion_structure_constants(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the fixed repository Fano tensor in a requested dtype/device."""

    normalized_device = torch.device("cpu" if device is None else device)
    return _cached_structure_constants(dtype, normalized_device)


def _validate_octonion_pair(left: torch.Tensor, right: torch.Tensor) -> None:
    if left.shape[-1] != OCTONION_DIM or right.shape[-1] != OCTONION_DIM:
        raise ValueError("octonions must end in eight components")
    if left.dtype != right.dtype or left.device != right.device:
        raise ValueError("octonion operands must have the same dtype and device")


def octonion_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Evaluate one explicitly parenthesized octonion product."""

    _validate_octonion_pair(left, right)
    structure = octonion_structure_constants(dtype=left.dtype, device=left.device)
    return torch.einsum("...i,kij,...j->...k", left, structure, right)


def unit_octonion(parameters: torch.Tensor) -> torch.Tensor:
    """Normalize final-axis octonions, using ``1`` as deterministic zero fallback."""

    if parameters.shape[-1] != OCTONION_DIM:
        raise ValueError("octonion parameters must end in eight components")
    if not parameters.is_floating_point():
        raise ValueError("unit octonions require a floating-point dtype")
    norm = torch.linalg.vector_norm(parameters, dim=-1, keepdim=True)
    threshold = torch.as_tensor(1e-6, dtype=parameters.dtype, device=parameters.device)
    normalized = parameters / norm.clamp_min(threshold)
    identity = torch.zeros_like(parameters)
    identity[..., 0] = 1
    return torch.where(norm > threshold, normalized, identity)


def project_to_unit_ball(values: torch.Tensor) -> torch.Tensor:
    """Radially project final-axis eight-vectors onto the closed unit ball."""

    if values.shape[-1] != OCTONION_DIM:
        raise ValueError("values must end in eight components")
    if not values.is_floating_point():
        raise ValueError("unit-ball projection requires a floating-point dtype")
    norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    one = torch.ones((), dtype=values.dtype, device=values.device)
    return values / torch.maximum(norm, one)


def octonion_left_multiplication_matrix(octonion: torch.Tensor) -> torch.Tensor:
    """Return ``L_u`` satisfying ``L_u x = u x``."""

    if octonion.shape[-1] != OCTONION_DIM:
        raise ValueError("octonions must end in eight components")
    structure = octonion_structure_constants(
        dtype=octonion.dtype, device=octonion.device
    )
    return torch.einsum("...i,kij->...kj", octonion, structure)


def octonion_right_multiplication_matrix(octonion: torch.Tensor) -> torch.Tensor:
    """Return ``R_u`` satisfying ``R_u x = x u``."""

    if octonion.shape[-1] != OCTONION_DIM:
        raise ValueError("octonions must end in eight components")
    structure = octonion_structure_constants(
        dtype=octonion.dtype, device=octonion.device
    )
    return torch.einsum("...j,kij->...ki", octonion, structure)


def octonion_multiplication_matrix(
    octonion: torch.Tensor, *, side: OperatorSide
) -> torch.Tensor:
    if side == "left":
        return octonion_left_multiplication_matrix(octonion)
    if side == "right":
        return octonion_right_multiplication_matrix(octonion)
    raise ValueError("side must be 'left' or 'right'")


def octonion_left_lie_coordinate_matrix() -> torch.Tensor:
    """Exact 28 by 28 coordinates for ``L(e_i)`` and their commutators.

    Rows use the seven imaginary left multiplications followed by the 21
    commutators ``[L(e_i),L(e_j)]``.  Columns are the strict upper triangle of
    an 8 by 8 skew matrix.  Its exact determinant is ``-2**49``.
    """

    structure = _INTEGER_STRUCTURE_CONSTANTS.to(torch.int64)
    basis_left = structure.permute(1, 0, 2)[1:]
    rows = [matrix for matrix in basis_left]
    rows.extend(
        basis_left[left] @ basis_left[right] - basis_left[right] @ basis_left[left]
        for left in range(7)
        for right in range(left + 1, 7)
    )
    upper = torch.triu_indices(OCTONION_DIM, OCTONION_DIM, offset=1)
    return torch.stack([matrix[upper[0], upper[1]] for matrix in rows])


def _validate_matrix_sequence(matrices: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if matrices.ndim not in (4, 5) or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError(
            "matrices must have shape (batch,length,D,D) or (batch,length,lanes,D,D)"
        )
    if matrices.shape[1] < 1:
        raise ValueError("scan length must be positive")
    if matrices.ndim == 4:
        return matrices, False
    batch, length, lanes, dimension, _ = matrices.shape
    flattened = matrices.permute(0, 2, 1, 3, 4).reshape(
        batch * lanes, length, dimension, dimension
    )
    return flattened, True


def _restore_matrix_lanes(
    matrices: torch.Tensor, reference: torch.Tensor, had_lanes: bool
) -> torch.Tensor:
    if not had_lanes:
        return matrices
    batch, length, lanes, dimension, _ = reference.shape
    return matrices.reshape(batch, lanes, length, dimension, dimension).permute(
        0, 2, 1, 3, 4
    )


def _hillis_steele_matrix_scan(matrices: torch.Tensor) -> torch.Tensor:
    prefixes = matrices
    offset = 1
    while offset < matrices.shape[1]:
        composed = prefixes[:, offset:] @ prefixes[:, :-offset]
        prefixes = torch.cat((prefixes[:, :offset], composed), dim=1)
        offset *= 2
    return prefixes


def _work_efficient_matrix_scan(matrices: torch.Tensor) -> torch.Tensor:
    """Ordered inclusive Blelloch-style scan with fewer than ``3P`` products."""

    batch, length, dimension, _ = matrices.shape
    if length == 1:
        return matrices
    padded_length = 1 << (length - 1).bit_length()
    identity = torch.eye(
        dimension, dtype=matrices.dtype, device=matrices.device
    ).reshape(1, 1, dimension, dimension)
    if padded_length == length:
        leaves = matrices
    else:
        padding = identity.expand(batch, padded_length - length, -1, -1)
        leaves = torch.cat((matrices, padding), dim=1)

    levels = [leaves]
    nodes = leaves
    while nodes.shape[1] > 1:
        nodes = nodes[:, 1::2] @ nodes[:, 0::2]
        levels.append(nodes)

    exclusive = identity.expand(batch, 1, -1, -1)
    for children in reversed(levels[:-1]):
        left_totals = children[:, 0::2]
        right_exclusive = left_totals @ exclusive
        exclusive = torch.stack((exclusive, right_exclusive), dim=2).reshape(
            batch, -1, dimension, dimension
        )
    inclusive = leaves @ exclusive
    return inclusive[:, :length]


def associative_matrix_prefix_scan(
    matrices: torch.Tensor, *, backend: Literal["work_efficient", "hillis_steele"]
) -> torch.Tensor:
    """Inclusive chronological matrix prefixes over sequence axis one."""

    flattened, had_lanes = _validate_matrix_sequence(matrices)
    if backend == "work_efficient":
        prefixes = _work_efficient_matrix_scan(flattened)
    elif backend == "hillis_steele":
        prefixes = _hillis_steele_matrix_scan(flattened)
    else:
        raise ValueError("backend must be 'work_efficient' or 'hillis_steele'")
    return _restore_matrix_lanes(prefixes, matrices, had_lanes)


def scan_composition_counts(length: int) -> dict[str, int]:
    """Return exact matrix-product counts for the two parallel trees."""

    if length < 1:
        raise ValueError("scan length must be positive")
    hillis_steele = 0
    offset = 1
    while offset < length:
        hillis_steele += length - offset
        offset *= 2
    work_efficient = 0 if length == 1 else 3 * (1 << (length - 1).bit_length()) - 2
    return {
        "hillis_steele": hillis_steele,
        "work_efficient": work_efficient,
    }


def _masked_unit_octonions(
    token_octonions: torch.Tensor, valid_mask: torch.Tensor | None
) -> torch.Tensor:
    tokens = unit_octonion(token_octonions)
    if valid_mask is None:
        return tokens
    batch, length = token_octonions.shape[:2]
    if valid_mask.shape != (batch, length):
        raise ValueError("valid_mask must have shape (batch,length)")
    identities = torch.zeros_like(tokens)
    identities[..., 0] = 1
    return torch.where(valid_mask.bool()[..., None, None], tokens, identities)


def octonion_operator_prefix_scan(
    token_octonions: torch.Tensor,
    initial_operator: torch.Tensor | None = None,
    *,
    side: OperatorSide = "left",
    valid_mask: torch.Tensor | None = None,
    mode: ScanMode = "work_efficient",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan multiplication operators, retaining nonassociative information."""

    if (
        token_octonions.ndim != 4
        or token_octonions.shape[1] < 1
        or token_octonions.shape[-1] != OCTONION_DIM
    ):
        raise ValueError(
            "token_octonions must have nonempty shape (batch,length,lanes,8)"
        )
    batch, length, lanes, _ = token_octonions.shape
    tokens = _masked_unit_octonions(token_octonions, valid_mask)
    actions = octonion_multiplication_matrix(tokens, side=side)
    if initial_operator is None:
        initial_operator = torch.eye(
            OCTONION_DIM, dtype=tokens.dtype, device=tokens.device
        ).expand(batch, lanes, -1, -1)
    elif initial_operator.shape != (batch, lanes, OCTONION_DIM, OCTONION_DIM):
        raise ValueError("initial_operator must have shape (batch,lanes,8,8)")

    if mode == "recurrent":
        operator = initial_operator
        rows = []
        for position in range(length):
            operator = actions[:, position] @ operator
            rows.append(operator)
        sequence = torch.stack(rows, dim=1)
        return sequence, operator
    if mode not in ("work_efficient", "hillis_steele"):
        raise ValueError("unknown scan mode")
    prefixes = associative_matrix_prefix_scan(actions, backend=mode)
    sequence = prefixes @ initial_operator[:, None]
    return sequence, sequence[:, -1]


def octonion_state_scan(
    token_octonions: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    side: OperatorSide = "left",
    valid_mask: torch.Tensor | None = None,
    mode: ScanMode = "work_efficient",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply lifted prefixes, or stream raw products with an eight-scalar cache."""

    if (
        token_octonions.ndim != 4
        or token_octonions.shape[1] < 1
        or token_octonions.shape[-1] != OCTONION_DIM
    ):
        raise ValueError(
            "token_octonions must have nonempty shape (batch,length,lanes,8)"
        )
    batch, length, lanes, _ = token_octonions.shape
    tokens = _masked_unit_octonions(token_octonions, valid_mask)
    if initial_state is None:
        initial_state = torch.zeros_like(tokens[:, 0])
        initial_state[..., 0] = 1
    elif initial_state.shape != (batch, lanes, OCTONION_DIM):
        raise ValueError("initial_state must have shape (batch,lanes,8)")

    if mode == "recurrent":
        state = initial_state
        rows = []
        for position in range(length):
            if side == "left":
                state = octonion_product(tokens[:, position], state)
            elif side == "right":
                state = octonion_product(state, tokens[:, position])
            else:
                raise ValueError("side must be 'left' or 'right'")
            rows.append(state)
        sequence = torch.stack(rows, dim=1)
        return sequence, state
    operators, _ = octonion_operator_prefix_scan(
        tokens,
        side=side,
        mode=mode,
    )
    sequence = torch.einsum("blhij,bhj->blhi", operators, initial_state)
    return sequence, sequence[:, -1]


def _validate_gates(gate: torch.Tensor, expected: tuple[int, int, int]) -> None:
    if gate.shape != expected:
        raise ValueError("retention/write gates must have shape (batch,length,lanes)")
    if not gate.is_floating_point() or not bool(torch.isfinite(gate).all()):
        raise ValueError("gates must be finite floating-point tensors")
    if bool(((gate < 0) | (gate > 1)).any()):
        raise ValueError("gates must lie in [0,1]")


def homogeneous_affine_matrix(
    action: torch.Tensor, drive: torch.Tensor
) -> torch.Tensor:
    """Pack lane-wise ``x -> action x + drive`` maps into 9 by 9 matrices."""

    if (
        action.ndim != 5
        or drive.ndim != 4
        or action.shape[:-2] != drive.shape[:-1]
        or action.shape[-2:] != (OCTONION_DIM, OCTONION_DIM)
        or drive.shape[-1] != OCTONION_DIM
    ):
        raise ValueError("incompatible lane-wise affine action and drive")
    matrix = action.new_zeros(*action.shape[:-2], OCTONION_DIM + 1, OCTONION_DIM + 1)
    matrix[..., 0, 0] = 1
    matrix[..., 1:, 0] = drive
    matrix[..., 1:, 1:] = action
    return matrix


def bounded_octonion_affine_scan(
    token_octonions: torch.Tensor,
    retention: torch.Tensor,
    write_gate: torch.Tensor,
    values: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    side: OperatorSide = "left",
    valid_mask: torch.Tensor | None = None,
    mode: ScanMode = "work_efficient",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the norm-bounded octonion-operator affine recurrence."""

    if (
        token_octonions.ndim != 4
        or token_octonions.shape[1] < 1
        or token_octonions.shape[-1] != OCTONION_DIM
        or values.shape != token_octonions.shape
    ):
        raise ValueError(
            "token_octonions and values must have equal nonempty shape "
            "(batch,length,lanes,8)"
        )
    batch, length, lanes, _ = token_octonions.shape
    expected_gates = (batch, length, lanes)
    _validate_gates(retention, expected_gates)
    _validate_gates(write_gate, expected_gates)
    tokens = unit_octonion(token_octonions)
    bounded_values = project_to_unit_ball(values)
    if initial_state is None:
        initial_state = torch.zeros_like(tokens[:, 0])
    elif initial_state.shape != (batch, lanes, OCTONION_DIM):
        raise ValueError("initial_state must have shape (batch,lanes,8)")

    if valid_mask is not None:
        if valid_mask.shape != (batch, length):
            raise ValueError("valid_mask must have shape (batch,length)")
        valid = valid_mask.bool()[..., None]
        identities = torch.zeros_like(tokens)
        identities[..., 0] = 1
        tokens = torch.where(valid[..., None], tokens, identities)
        retention = torch.where(valid, retention, torch.ones_like(retention))
        write_gate = torch.where(valid, write_gate, torch.zeros_like(write_gate))
        bounded_values = torch.where(
            valid[..., None], bounded_values, torch.zeros_like(bounded_values)
        )

    drive = (1 - retention)[..., None] * write_gate[..., None] * bounded_values
    if mode == "recurrent":
        state = initial_state
        rows = []
        for position in range(length):
            if side == "left":
                rotated = octonion_product(tokens[:, position], state)
            elif side == "right":
                rotated = octonion_product(state, tokens[:, position])
            else:
                raise ValueError("side must be 'left' or 'right'")
            state = retention[:, position, :, None] * rotated + drive[:, position]
            rows.append(state)
        sequence = torch.stack(rows, dim=1)
        return sequence, state
    if mode not in ("work_efficient", "hillis_steele"):
        raise ValueError("unknown scan mode")

    orthogonal = octonion_multiplication_matrix(tokens, side=side)
    action = retention[..., None, None] * orthogonal
    homogeneous = homogeneous_affine_matrix(action, drive)
    prefixes = associative_matrix_prefix_scan(homogeneous, backend=mode)
    one = torch.ones(
        batch, lanes, 1, dtype=initial_state.dtype, device=initial_state.device
    )
    initial_h = torch.cat((one, initial_state), dim=-1)
    sequence = torch.einsum("blhij,bhj->blhi", prefixes, initial_h)[..., 1:]
    return sequence, sequence[:, -1]


class OctonionOperatorSSMLayer(nn.Module):
    """Input-selective bounded octonion operator layer with compact streaming."""

    def __init__(
        self,
        input_dim: int,
        state_lanes: int,
        *,
        output_dim: int | None = None,
        operator_initialization_scale: float = 0.1,
        initial_retention: float = 0.95,
        residual: bool = True,
    ) -> None:
        super().__init__()
        output_dim = input_dim if output_dim is None else output_dim
        if (
            input_dim < 1
            or state_lanes < 1
            or output_dim < 1
            or operator_initialization_scale <= 0
            or not 0 < initial_retention < 1
            or (residual and output_dim != input_dim)
        ):
            raise ValueError("invalid octonion operator layer configuration")
        self.input_dim = input_dim
        self.state_lanes = state_lanes
        self.output_dim = output_dim
        self.operator_initialization_scale = operator_initialization_scale
        self.residual = residual

        self.operator_projection = nn.Linear(input_dim, state_lanes * OCTONION_DIM)
        self.retention_projection = nn.Linear(input_dim, state_lanes)
        self.write_projection = nn.Linear(input_dim, state_lanes)
        self.value_projection = nn.Linear(input_dim, state_lanes * OCTONION_DIM)
        self.output_projection = nn.Linear(state_lanes * OCTONION_DIM, output_dim)

        nn.init.normal_(self.operator_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.operator_projection.bias)
        nn.init.zeros_(self.retention_projection.weight)
        nn.init.constant_(
            self.retention_projection.bias,
            math.log(initial_retention / (1 - initial_retention)),
        )
        nn.init.zeros_(self.write_projection.weight)
        nn.init.zeros_(self.write_projection.bias)

    @property
    def recurrent_state_scalars(self) -> int:
        return self.state_lanes * OCTONION_DIM

    def initial_state(
        self,
        batch_size: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return torch.zeros(
            batch_size,
            self.state_lanes,
            OCTONION_DIM,
            dtype=dtype or self.operator_projection.weight.dtype,
            device=device or self.operator_projection.weight.device,
        )

    def transition_parameters(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError("inputs must have shape (batch,length,input_dim)")
        batch, length, _ = inputs.shape
        operator_delta = self.operator_projection(inputs).reshape(
            batch, length, self.state_lanes, OCTONION_DIM
        )
        identity = torch.zeros_like(operator_delta)
        identity[..., 0] = 1
        operators = identity + self.operator_initialization_scale * operator_delta
        retention = torch.sigmoid(self.retention_projection(inputs))
        write_gate = torch.sigmoid(self.write_projection(inputs))
        values = self.value_projection(inputs).reshape(
            batch, length, self.state_lanes, OCTONION_DIM
        )
        return operators, retention, write_gate, values

    def forward(
        self,
        inputs: torch.Tensor,
        recurrent_state: torch.Tensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        return_recurrent_state: bool = False,
        scan_mode: ScanMode = "work_efficient",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        operators, retention, write_gate, values = self.transition_parameters(inputs)
        states, final_state = bounded_octonion_affine_scan(
            operators,
            retention,
            write_gate,
            values,
            recurrent_state,
            valid_mask=attention_mask,
            mode=scan_mode,
        )
        outputs = self.output_projection(states.flatten(start_dim=-2))
        if self.residual:
            outputs = outputs + inputs
        if return_recurrent_state:
            return outputs, final_state
        return outputs

    def step(
        self, inputs: torch.Tensor, recurrent_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 2 or recurrent_state.ndim != 3:
            raise ValueError(
                "step inputs/state must have shapes (batch,input_dim) and "
                "(batch,lanes,8)"
            )
        outputs, final_state = self(
            inputs[:, None],
            recurrent_state,
            return_recurrent_state=True,
            scan_mode="recurrent",
        )
        return outputs[:, 0], final_state


__all__: Sequence[str] = (
    "FANO_TRIPLES",
    "OCTONION_DIM",
    "OctonionOperatorSSMLayer",
    "associative_matrix_prefix_scan",
    "bounded_octonion_affine_scan",
    "homogeneous_affine_matrix",
    "octonion_left_lie_coordinate_matrix",
    "octonion_left_multiplication_matrix",
    "octonion_multiplication_matrix",
    "octonion_operator_prefix_scan",
    "octonion_product",
    "octonion_right_multiplication_matrix",
    "octonion_state_scan",
    "octonion_structure_constants",
    "project_to_unit_ball",
    "scan_composition_counts",
    "unit_octonion",
)
