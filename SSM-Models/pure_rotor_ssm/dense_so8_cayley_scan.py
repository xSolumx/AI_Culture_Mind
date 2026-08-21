"""Experimental bounded affine scan with dense SO(8) Cayley transitions.

The fixed monomial--golden generators certified in ``Spin-Space-Research``
have dense closure in SO(8).  Their seven Clifford grade-one matrices and 21
grade-two products give a concrete 28-dimensional basis of ``so(8)``.  This
module turns that basis into trainable local Cayley increments:

``Q(B) = (I - B / 2)^(-1) (I + B / 2)``, with ``B`` skew-symmetric.

The recurrent state is only the acted-on eight-vector, not the accumulated
64-scalar operator.  The affine state update is contractive and therefore has
the same finite-input norm bound as the experimental octonion operator scan.
This is deliberately an experimental companion: dense solves and a full
28-dimensional local chart do not establish a speed or task-quality benefit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache
from typing import Literal

import torch
from torch import nn

from .octonion_operator_scan import (
    OCTONION_DIM,
    associative_matrix_prefix_scan,
    homogeneous_affine_matrix,
    octonion_left_multiplication_matrix,
    project_to_unit_ball,
)

SO8_LIE_DIM = 28
ScanMode = Literal["work_efficient", "hillis_steele", "recurrent"]


@lru_cache(maxsize=32)
def _cached_clifford_lie_basis(
    dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    units = torch.eye(OCTONION_DIM, dtype=dtype, device=device)
    grade_one = octonion_left_multiplication_matrix(units)[1:]
    grade_two = torch.stack(
        [
            grade_one[left] @ grade_one[right]
            for left in range(7)
            for right in range(left + 1, 7)
        ]
    )
    basis = torch.cat((grade_one, grade_two), dim=0)
    if basis.shape != (SO8_LIE_DIM, OCTONION_DIM, OCTONION_DIM):
        raise AssertionError("Clifford basis has the wrong SO(8) shape")
    if not torch.equal(basis.transpose(-1, -2), -basis):
        raise AssertionError("Clifford Lie basis is not exactly skew-symmetric")
    norms = torch.linalg.matrix_norm(basis, ord="fro", dim=(-2, -1))
    if not torch.allclose(norms, norms[0].expand_as(norms), rtol=0, atol=0):
        raise AssertionError("Clifford Lie basis does not have a shared norm")
    return basis / norms[0]


def so8_clifford_lie_basis(
    *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None
) -> torch.Tensor:
    """Return a Frobenius-normalized 28-generator ``so(8)`` basis.

    The order is the seven imaginary left actions followed by 21 products
    ``L(e_i)L(e_j)`` for increasing ``i,j``.  It is the basis used by the exact
    dense-closure theorem; normalizing it changes only controller coordinates.
    """

    if not dtype.is_floating_point:
        raise ValueError("SO(8) Cayley coordinates require a floating dtype")
    normalized_device = torch.device("cpu" if device is None else device)
    return _cached_clifford_lie_basis(dtype, normalized_device)


def so8_tangent_matrix(coordinates: torch.Tensor) -> torch.Tensor:
    """Map final-axis 28D coordinates to an exactly skew 8-by-8 tangent."""

    if coordinates.shape[-1] != SO8_LIE_DIM or not coordinates.is_floating_point():
        raise ValueError("coordinates must be floating tensors ending in 28")
    basis = so8_clifford_lie_basis(
        dtype=coordinates.dtype, device=coordinates.device
    )
    tangent = torch.einsum("...r,rij->...ij", coordinates, basis)
    # Preserve skewness under the contraction's floating-point rounding.
    return 0.5 * (tangent - tangent.transpose(-1, -2))


def cayley_so8(tangent: torch.Tensor) -> torch.Tensor:
    """Return a special-orthogonal Cayley transform of a skew tangent.

    ``I - tangent/2`` is nonsingular for every real skew-symmetric tangent.
    The result is orthogonal with determinant +1 in exact arithmetic.
    """

    if (
        tangent.ndim < 2
        or tangent.shape[-2:] != (OCTONION_DIM, OCTONION_DIM)
        or not tangent.is_floating_point()
    ):
        raise ValueError("tangent must be a floating tensor ending in (8,8)")
    identity = torch.eye(
        OCTONION_DIM, dtype=tangent.dtype, device=tangent.device
    ).expand_as(tangent)
    return torch.linalg.solve(identity - 0.5 * tangent, identity + 0.5 * tangent)


def _validate_gates(gate: torch.Tensor, expected: tuple[int, int, int]) -> None:
    if gate.shape != expected or not gate.is_floating_point():
        raise ValueError("gates must be floating tensors shaped (batch,length,lanes)")
    if not bool(torch.isfinite(gate).all()) or bool(((gate < 0) | (gate > 1)).any()):
        raise ValueError("gates must be finite and lie in [0,1]")


def bounded_so8_cayley_affine_scan(
    tangent_coordinates: torch.Tensor,
    retention: torch.Tensor,
    write_gate: torch.Tensor,
    values: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    valid_mask: torch.Tensor | None = None,
    mode: ScanMode = "work_efficient",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan ``h -> d Q(B) h + (1-d) w z`` with a hard state-norm bound."""

    if (
        tangent_coordinates.ndim != 4
        or tangent_coordinates.shape[1] < 1
        or tangent_coordinates.shape[-1] != SO8_LIE_DIM
        or values.shape != (*tangent_coordinates.shape[:-1], OCTONION_DIM)
    ):
        raise ValueError(
            "coordinates must be (batch,length,lanes,28), values must end in 8"
        )
    batch, length, lanes, _ = tangent_coordinates.shape
    expected_gates = (batch, length, lanes)
    _validate_gates(retention, expected_gates)
    _validate_gates(write_gate, expected_gates)
    if initial_state is None:
        initial_state = torch.zeros_like(values[:, 0])
    elif initial_state.shape != (batch, lanes, OCTONION_DIM):
        raise ValueError("initial_state must have shape (batch,lanes,8)")

    coordinates = tangent_coordinates
    bounded_values = project_to_unit_ball(values)
    if valid_mask is not None:
        if valid_mask.shape != (batch, length):
            raise ValueError("valid_mask must have shape (batch,length)")
        valid = valid_mask.bool()[..., None]
        coordinates = torch.where(valid[..., None], coordinates, torch.zeros_like(coordinates))
        retention = torch.where(valid, retention, torch.ones_like(retention))
        write_gate = torch.where(valid, write_gate, torch.zeros_like(write_gate))
        bounded_values = torch.where(valid[..., None], bounded_values, torch.zeros_like(bounded_values))

    tangent = so8_tangent_matrix(coordinates)
    rotations = cayley_so8(tangent)
    drive = (1 - retention)[..., None] * write_gate[..., None] * bounded_values
    if mode == "recurrent":
        state = initial_state
        states = []
        for position in range(length):
            transported = torch.einsum("bhij,bhj->bhi", rotations[:, position], state)
            state = retention[:, position, :, None] * transported + drive[:, position]
            states.append(state)
        sequence = torch.stack(states, dim=1)
        return sequence, state
    if mode not in ("work_efficient", "hillis_steele"):
        raise ValueError("unknown scan mode")

    action = retention[..., None, None] * rotations
    homogeneous = homogeneous_affine_matrix(action, drive)
    prefixes = associative_matrix_prefix_scan(homogeneous, backend=mode)
    one = torch.ones(
        batch, lanes, 1, dtype=initial_state.dtype, device=initial_state.device
    )
    initial_homogeneous = torch.cat((one, initial_state), dim=-1)
    sequence = torch.einsum("blhij,bhj->blhi", prefixes, initial_homogeneous)[..., 1:]
    return sequence, sequence[:, -1]


class DenseSO8CayleySSMLayer(nn.Module):
    """Input-selective dense-SO(8) bounded affine layer.

    A lane has an eight-scalar streaming state and 28 local tangent controls.
    It intentionally does not cache an accumulated matrix, so direct group
    readout requires an explicit 64-scalar operator probe outside this layer.
    """

    def __init__(
        self,
        input_dim: int,
        state_lanes: int,
        *,
        output_dim: int | None = None,
        tangent_initialization_scale: float = 0.1,
        initial_retention: float = 0.95,
        residual: bool = True,
    ) -> None:
        super().__init__()
        output_dim = input_dim if output_dim is None else output_dim
        if (
            input_dim < 1
            or state_lanes < 1
            or output_dim < 1
            or tangent_initialization_scale <= 0
            or not 0 < initial_retention < 1
            or (residual and output_dim != input_dim)
        ):
            raise ValueError("invalid dense SO(8) Cayley layer configuration")
        self.input_dim = input_dim
        self.state_lanes = state_lanes
        self.output_dim = output_dim
        self.tangent_initialization_scale = tangent_initialization_scale
        self.residual = residual
        self.tangent_projection = nn.Linear(input_dim, state_lanes * SO8_LIE_DIM)
        self.retention_projection = nn.Linear(input_dim, state_lanes)
        self.write_projection = nn.Linear(input_dim, state_lanes)
        self.value_projection = nn.Linear(input_dim, state_lanes * OCTONION_DIM)
        self.output_projection = nn.Linear(state_lanes * OCTONION_DIM, output_dim)
        nn.init.normal_(self.tangent_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.tangent_projection.bias)
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

    @property
    def transition_control_scalars(self) -> int:
        return self.state_lanes * SO8_LIE_DIM

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
            dtype=dtype or self.tangent_projection.weight.dtype,
            device=device or self.tangent_projection.weight.device,
        )

    def transition_parameters(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError("inputs must have shape (batch,length,input_dim)")
        batch, length, _ = inputs.shape
        coordinates = self.tangent_initialization_scale * self.tangent_projection(inputs)
        coordinates = coordinates.reshape(batch, length, self.state_lanes, SO8_LIE_DIM)
        retention = torch.sigmoid(self.retention_projection(inputs))
        write_gate = torch.sigmoid(self.write_projection(inputs))
        values = self.value_projection(inputs).reshape(
            batch, length, self.state_lanes, OCTONION_DIM
        )
        return coordinates, retention, write_gate, values

    def forward(
        self,
        inputs: torch.Tensor,
        recurrent_state: torch.Tensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        return_recurrent_state: bool = False,
        scan_mode: ScanMode = "work_efficient",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        coordinates, retention, write_gate, values = self.transition_parameters(inputs)
        states, final_state = bounded_so8_cayley_affine_scan(
            coordinates,
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
            raise ValueError("step expects (batch,input_dim) and (batch,lanes,8)")
        outputs, final_state = self(
            inputs[:, None],
            recurrent_state,
            return_recurrent_state=True,
            scan_mode="recurrent",
        )
        return outputs[:, 0], final_state


__all__: Sequence[str] = (
    "DenseSO8CayleySSMLayer",
    "SO8_LIE_DIM",
    "bounded_so8_cayley_affine_scan",
    "cayley_so8",
    "so8_clifford_lie_basis",
    "so8_tangent_matrix",
)
