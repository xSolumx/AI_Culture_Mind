"""Complex- and quaternionic-type Schur affine scan blocks.

For the canonical complex-type real irrep, identify the state with ``C`` and
let the group act by left multiplication.  Its real commutant is right
multiplication by ``C``.  For the canonical quaternionic-type irrep, identify
the state with ``H``; left unit-quaternion action has the complete right ``H``
commutant.  With ``m`` isotypic copies, the multiplicity transition is thus a
right matrix in ``Mat_m(C)`` or ``Mat_m(H)``.

States are stored as row vectors of division-algebra elements.  A transition
acts by

    x -> (g * x) M + b.

Left and right multiplication commute by associativity.  Ordered composition
therefore remains factored, including the noncommutative quaternionic order
``M_before M_after``.  This supplies the division-algebra blocks missing from
the real-type scanner; automatic isotypic decomposition of arbitrary supplied
representations remains outside this module.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import sympy as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "division_schur_scan_20260811.json"
ALGEBRA_DIMENSIONS = {"complex": 2, "quaternion": 4}


def division_product(
    left: torch.Tensor,
    right: torch.Tensor,
    algebra: str,
) -> torch.Tensor:
    """Multiply tensors of complex pairs or Hamilton quaternions."""

    dimension = ALGEBRA_DIMENSIONS.get(algebra)
    if dimension is None:
        raise ValueError("algebra must be 'complex' or 'quaternion'")
    if left.shape[-1] != dimension or right.shape[-1] != dimension:
        raise ValueError(f"{algebra} values must have {dimension} components")
    if algebra == "complex":
        left_real, left_imaginary = left.unbind(dim=-1)
        right_real, right_imaginary = right.unbind(dim=-1)
        return torch.stack(
            (
                left_real * right_real - left_imaginary * right_imaginary,
                left_real * right_imaginary + left_imaginary * right_real,
            ),
            dim=-1,
        )

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


def right_matrix_apply(
    state: torch.Tensor,
    matrix: torch.Tensor,
    algebra: str,
) -> torch.Tensor:
    """Apply a division-algebra matrix to a row state on the right."""

    if matrix.shape[-3] != matrix.shape[-2]:
        raise ValueError("right multiplicity matrices must be square")
    if state.shape[-2] != matrix.shape[-3]:
        raise ValueError("state and multiplicity dimensions do not agree")
    products = division_product(state.unsqueeze(-2), matrix, algebra)
    return products.sum(dim=-3)


def right_matrix_product(
    before: torch.Tensor,
    after: torch.Tensor,
    algebra: str,
) -> torch.Tensor:
    """Return ``before @ after`` with division-algebra scalar products."""

    if before.shape[-2] != after.shape[-3]:
        raise ValueError("multiplicity matrix dimensions do not compose")
    products = division_product(
        before.unsqueeze(-2),
        after.unsqueeze(-4),
        algebra,
    )
    return products.sum(dim=-3)


@dataclass(frozen=True)
class DivisionSchurTransition:
    """One factored affine transition over ``C`` or ``H``."""

    algebra: str
    left_action: torch.Tensor
    right_multiplicity: torch.Tensor
    drive: torch.Tensor


def apply_division_schur(
    transition: DivisionSchurTransition,
    state: torch.Tensor,
) -> torch.Tensor:
    """Apply ``x -> (g*x)M+b`` without a dense real matrix."""

    moved = division_product(
        transition.left_action.unsqueeze(-2),
        state,
        transition.algebra,
    )
    return right_matrix_apply(
        moved,
        transition.right_multiplicity,
        transition.algebra,
    ) + transition.drive


def compose_division_schur(
    after: DivisionSchurTransition,
    before: DivisionSchurTransition,
) -> DivisionSchurTransition:
    """Compose ``after(before(x))`` in chronological quaternionic order."""

    if after.algebra != before.algebra:
        raise ValueError("division-algebra types must agree")
    algebra = after.algebra
    transported_drive = right_matrix_apply(
        division_product(
            after.left_action.unsqueeze(-2),
            before.drive,
            algebra,
        ),
        after.right_multiplicity,
        algebra,
    )
    return DivisionSchurTransition(
        algebra=algebra,
        left_action=division_product(
            after.left_action,
            before.left_action,
            algebra,
        ),
        right_multiplicity=right_matrix_product(
            before.right_multiplicity,
            after.right_multiplicity,
            algebra,
        ),
        drive=after.drive + transported_drive,
    )


def associative_division_schur_scan(
    transition: DivisionSchurTransition,
) -> DivisionSchurTransition:
    """Inclusive Hillis--Steele scan over batch and sequence tensors."""

    if transition.left_action.ndim < 3:
        raise ValueError("transitions need batch and sequence dimensions")
    current = transition
    offset = 1
    length = transition.left_action.shape[1]
    while offset < length:
        after = DivisionSchurTransition(
            algebra=current.algebra,
            left_action=current.left_action[:, offset:],
            right_multiplicity=current.right_multiplicity[:, offset:],
            drive=current.drive[:, offset:],
        )
        before = DivisionSchurTransition(
            algebra=current.algebra,
            left_action=current.left_action[:, :-offset],
            right_multiplicity=current.right_multiplicity[:, :-offset],
            drive=current.drive[:, :-offset],
        )
        composed = compose_division_schur(after, before)
        current = DivisionSchurTransition(
            algebra=current.algebra,
            left_action=torch.cat(
                (current.left_action[:, :offset], composed.left_action),
                dim=1,
            ),
            right_multiplicity=torch.cat(
                (
                    current.right_multiplicity[:, :offset],
                    composed.right_multiplicity,
                ),
                dim=1,
            ),
            drive=torch.cat(
                (current.drive[:, :offset], composed.drive),
                dim=1,
            ),
        )
        offset *= 2
    return current


def _basis_products(algebra: str) -> list[list[list[int]]]:
    dimension = ALGEBRA_DIMENSIONS[algebra]
    basis = torch.eye(dimension, dtype=torch.int64)
    table: list[list[list[int]]] = []
    for left in basis:
        table.append(
            [
                [int(value) for value in division_product(left, right, algebra)]
                for right in basis
            ]
        )
    return table


def _multiplication_matrix(
    table: list[list[list[int]]],
    basis_index: int,
    *,
    side: str,
) -> sp.Matrix:
    dimension = len(table)
    if side == "left":
        columns = [table[basis_index][index] for index in range(dimension)]
    elif side == "right":
        columns = [table[index][basis_index] for index in range(dimension)]
    else:
        raise ValueError("side must be 'left' or 'right'")
    return sp.Matrix.hstack(*(sp.Matrix(column) for column in columns))


def exact_commutant_audit(algebra: str) -> dict[str, object]:
    """Compute the exact real centralizer of the canonical left action."""

    table = _basis_products(algebra)
    dimension = ALGEBRA_DIMENSIONS[algebra]
    identity = sp.eye(dimension)
    left_generators = [
        _multiplication_matrix(table, index, side="left")
        for index in range(1, dimension)
    ]
    right_basis = [
        _multiplication_matrix(table, index, side="right")
        for index in range(dimension)
    ]
    constraints = sp.Matrix.vstack(
        *(
            sp.kronecker_product(generator.T, identity)
            - sp.kronecker_product(identity, generator)
            for generator in left_generators
        )
    )
    commutant_dimension = dimension**2 - constraints.rank()
    basis_rank = sp.Matrix.hstack(
        *(sp.Matrix(matrix).reshape(dimension**2, 1) for matrix in right_basis)
    ).rank()
    return {
        "real_irrep_dimension": dimension,
        "exact_commutant_dimension": commutant_dimension,
        "right_multiplication_basis_rank": basis_rank,
        "right_basis_commutes_with_left_generators": all(
            right * left == left * right
            for right in right_basis
            for left in left_generators
        ),
        "complete_right_division_algebra_basis": bool(
            commutant_dimension == basis_rank == dimension
        ),
    }


def _maximum_difference(
    left: DivisionSchurTransition,
    right: DivisionSchurTransition,
) -> float:
    return max(
        float((left.left_action - right.left_action).abs().max()),
        float((left.right_multiplicity - right.right_multiplicity).abs().max()),
        float((left.drive - right.drive).abs().max()),
    )


def _numerical_audit(algebra: str) -> dict[str, object]:
    torch.manual_seed(20_260_811 + ALGEBRA_DIMENSIONS[algebra])
    dtype = torch.float64
    batch, length, multiplicity = 2, 17, 3
    dimension = ALGEBRA_DIMENSIONS[algebra]

    left_action = torch.randn(batch, length, dimension, dtype=dtype)
    left_action = left_action / left_action.norm(dim=-1, keepdim=True)
    right_multiplicity = 0.015 * torch.randn(
        batch,
        length,
        multiplicity,
        multiplicity,
        dimension,
        dtype=dtype,
    )
    diagonal = torch.arange(multiplicity)
    right_multiplicity[:, :, diagonal, diagonal, 0] += 0.88
    drive = 0.02 * torch.randn(
        batch,
        length,
        multiplicity,
        dimension,
        dtype=dtype,
    )
    transition = DivisionSchurTransition(
        algebra=algebra,
        left_action=left_action,
        right_multiplicity=right_multiplicity,
        drive=drive,
    )
    initial = torch.randn(batch, multiplicity, dimension, dtype=dtype)

    prefixes = associative_division_schur_scan(transition)
    parallel = apply_division_schur(prefixes, initial[:, None])
    recurrent_state = initial
    recurrent = []
    steps = []
    for position in range(length):
        step = DivisionSchurTransition(
            algebra=algebra,
            left_action=left_action[:, position],
            right_multiplicity=right_multiplicity[:, position],
            drive=drive[:, position],
        )
        steps.append(step)
        recurrent_state = apply_division_schur(step, recurrent_state)
        recurrent.append(recurrent_state)
    recurrent_tensor = torch.stack(recurrent, dim=1)

    left_associative = compose_division_schur(
        steps[2],
        compose_division_schur(steps[1], steps[0]),
    )
    right_associative = compose_division_schur(
        compose_division_schur(steps[2], steps[1]),
        steps[0],
    )

    gradient_tensors = tuple(
        value.detach().clone().requires_grad_(True)
        for value in (left_action, right_multiplicity, drive)
    )
    gradient_transition = DivisionSchurTransition(
        algebra=algebra,
        left_action=gradient_tensors[0],
        right_multiplicity=gradient_tensors[1],
        drive=gradient_tensors[2],
    )
    gradient_prefixes = associative_division_schur_scan(gradient_transition)
    parallel_final = apply_division_schur(
        gradient_prefixes,
        initial[:, None],
    )[:, -1]
    parallel_loss = parallel_final.square().mean()
    parallel_gradients = torch.autograd.grad(
        parallel_loss,
        gradient_tensors,
        retain_graph=True,
    )
    gradient_state = initial
    for position in range(length):
        gradient_state = apply_division_schur(
            DivisionSchurTransition(
                algebra=algebra,
                left_action=gradient_tensors[0][:, position],
                right_multiplicity=gradient_tensors[1][:, position],
                drive=gradient_tensors[2][:, position],
            ),
            gradient_state,
        )
    recurrent_loss = gradient_state.square().mean()
    recurrent_gradients = torch.autograd.grad(recurrent_loss, gradient_tensors)

    state = torch.randn(batch, multiplicity, dimension, dtype=dtype)
    scalar = torch.randn(batch, multiplicity, multiplicity, dimension, dtype=dtype)
    scalar[..., 0] += torch.eye(multiplicity, dtype=dtype)
    group = torch.randn(batch, dimension, dtype=dtype)
    commuting_error = float(
        (
            right_matrix_apply(
                division_product(group[:, None], state, algebra),
                scalar,
                algebra,
            )
            - division_product(
                group[:, None],
                right_matrix_apply(state, scalar, algebra),
                algebra,
            )
        )
        .abs()
        .max()
    )

    return {
        "batch": batch,
        "length": length,
        "multiplicity": multiplicity,
        "associativity_max_error": _maximum_difference(
            left_associative,
            right_associative,
        ),
        "scan_recurrent_max_error": float(
            (parallel - recurrent_tensor).abs().max()
        ),
        "gradient_max_error": max(
            float((left - right).abs().max())
            for left, right in zip(
                parallel_gradients,
                recurrent_gradients,
                strict=True,
            )
        ),
        "left_action_right_multiplicity_commutation_error": commuting_error,
    }


def diagnostics() -> dict[str, object]:
    rows = {}
    for algebra in ALGEBRA_DIMENSIONS:
        exact = exact_commutant_audit(algebra)
        numerical = _numerical_audit(algebra)
        rows[algebra] = {
            "exact_commutant": exact,
            "numerical_scan": numerical,
            "passed": bool(
                exact["complete_right_division_algebra_basis"]
                and exact["right_basis_commutes_with_left_generators"]
                and numerical["associativity_max_error"] < 1e-12
                and numerical["scan_recurrent_max_error"] < 1e-11
                and numerical["gradient_max_error"] < 1e-10
                and numerical["left_action_right_multiplicity_commutation_error"]
                < 1e-12
            ),
        }
    return {
        "schema_version": 1,
        "claim_scope": (
            "canonical complex- and quaternionic-type Schur affine scan blocks"
        ),
        "state_convention": "row state, left group action, right multiplicity matrix",
        "composition_order": "M_composed = M_before M_after",
        "algebras": rows,
        "automatic_arbitrary_representation_decomposition_implemented": False,
        "sequence_model_superiority_claimed": False,
        "passed": all(row["passed"] for row in rows.values()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = diagnostics()
    encoded = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALGEBRA_DIMENSIONS",
    "DivisionSchurTransition",
    "apply_division_schur",
    "associative_division_schur_scan",
    "compose_division_schur",
    "diagnostics",
    "division_product",
    "exact_commutant_audit",
    "right_matrix_apply",
    "right_matrix_product",
]
