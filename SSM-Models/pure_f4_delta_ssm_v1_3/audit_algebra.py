"""Emit a deterministic algebra/representation audit for v1.3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .action import exponential_action
from .albert import albert_determinant, build_albert_algebra, jordan_structure_constants


def _rank_mod_prime(matrix: np.ndarray, prime: int) -> int:
    values = np.asarray(matrix, dtype=np.int64) % prime
    row = 0
    for column in range(values.shape[1]):
        pivots = np.flatnonzero(values[row:, column])
        if not len(pivots):
            continue
        pivot = row + int(pivots[0])
        values[[row, pivot]] = values[[pivot, row]]
        values[row] = values[row] * pow(int(values[row, column]), -1, prime) % prime
        for target in range(values.shape[0]):
            if target != row and values[target, column]:
                values[target] = (
                    values[target] - values[target, column] * values[row]
                ) % prime
        row += 1
        if row == values.shape[0]:
            break
    return row


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        canonical = np.ascontiguousarray(array, dtype="<f8")
        digest.update(str(canonical.shape).encode("ascii"))
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def build_report() -> dict[str, object]:
    algebra = build_albert_algebra()
    structure = jordan_structure_constants()
    scaled_structure = 2.0 * structure
    scaled_f4_raw = 4.0 * algebra.f4_raw
    if not np.array_equal(scaled_structure, np.rint(scaled_structure)):
        raise AssertionError("Jordan structure is not half-integral")
    if not np.array_equal(scaled_f4_raw, np.rint(scaled_f4_raw)):
        raise AssertionError("selected derivations are not quarter-integral")
    integer_f4 = np.rint(scaled_f4_raw).astype(np.int64).reshape(52, -1)
    modular_ranks = {str(prime): _rank_mod_prime(integer_f4, prime) for prime in (101, 1009, 10007)}

    torch.manual_seed(20260821)
    value = torch.randn(8, 27, dtype=torch.float64)
    determinant_errors: dict[str, float] = {}
    orthogonality_errors: dict[str, float] = {}
    for name, generators in (
        ("g2", algebra.g2),
        ("spin7", algebra.spin7),
        ("spin8", algebra.spin8),
        ("spin9", algebra.spin9),
        ("f4", algebra.f4),
        ("e6", algebra.e6),
    ):
        coordinates = 0.02 * torch.randn(len(generators), dtype=torch.float64)
        action = exponential_action(
            coordinates, torch.tensor(generators, dtype=torch.float64)
        )
        transformed = value @ action.T
        determinant_errors[name] = float(
            (albert_determinant(transformed) - albert_determinant(value)).abs().max()
        )
        identity = torch.eye(27, dtype=torch.float64)
        orthogonality_errors[name] = float((action.T @ action - identity).abs().max())

    diagonal = np.eye(27)[:3]
    report = {
        "schema_version": 1,
        "experiment": "Pure Exceptional Delta SSM v1.3.2 algebra audit",
        "construction_hash_sha256": _hash_arrays(
            structure,
            algebra.f4_raw,
            algebra.g2,
            algebra.spin7,
            algebra.spin8,
            algebra.spin9,
            algebra.e6,
        ),
        "dimensions": {
            "albert": 27,
            "tracefree_albert": 26,
            "g2": int(algebra.g2.shape[0]),
            "spin7": int(algebra.spin7.shape[0]),
            "spin8": int(algebra.spin8.shape[0]),
            "spin9": int(algebra.spin9.shape[0]),
            "f4": int(algebra.f4.shape[0]),
            "e6_minus_26": int(algebra.e6.shape[0]),
        },
        "exact_data": {
            "jordan_twice_is_integer": True,
            "f4_derivations_times_four_are_integer": True,
            "f4_modular_ranks": modular_ranks,
        },
        "numerical_residuals": {
            "f4_skew_max_abs": float(np.max(np.abs(algebra.f4 + algebra.f4.swapaxes(1, 2)))),
            "e6_complement_symmetric_max_abs": float(
                np.max(np.abs(algebra.e6[52:] - algebra.e6[52:].swapaxes(1, 2)))
            ),
            "spin7_first_triality_unit_fix_max_abs": float(
                np.max(np.abs(algebra.spin7 @ np.eye(27)[3]))
            ),
            "g2_two_triality_units_fix_max_abs": float(
                np.max(
                    np.abs(
                        np.einsum(
                            "fij,pj->fpi", algebra.g2, np.eye(27)[[3, 11]]
                        )
                    )
                )
            ),
            "spin8_diagonal_frame_fix_max_abs": float(
                np.max(np.abs(np.einsum("fij,pj->fpi", algebra.spin8, diagonal)))
            ),
            "spin9_idempotent_fix_max_abs": float(
                np.max(np.abs(np.einsum("fij,j->fi", algebra.spin9, diagonal[0])))
            ),
            "finite_exponential_cubic_errors": determinant_errors,
            "finite_exponential_orthogonality_errors": orthogonality_errors,
        },
        "passed": bool(
            all(rank == 52 for rank in modular_ranks.values())
            and max(determinant_errors.values()) < 1e-10
            and orthogonality_errors["f4"] < 1e-10
            and orthogonality_errors["e6"] > 1e-6
        ),
        "claim_boundary": {
            "exact": "dyadic Albert product and modular rank-52 derivation witness",
            "numerical": "G2/Spin(7)/Spin(8)/Spin(9) stabilizer alignment and floating exponential invariants",
            "open": "custom fused kernels, general task transfer, scaling, and E7/Freudenthal continuation",
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
