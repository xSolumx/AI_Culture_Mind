"""Exact Cayley-spectrum certificates and global-optimum falsifiers.

This module explains the recurring five-query Spin(8) information spectrum.
It deliberately separates an exact theorem on the orthonormal balanced orbit
from numerical attacks on the still-open global D-optimality claim.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
import sympy as sp
import torch
from torch import nn
from torch.nn import functional as F

from spin8_active_sensing import (
    QUERY_COUNT,
    SensorDesign,
    information_matrix,
    information_metrics,
    views_from_allocation,
)
from spin8_triality import (
    SPIN8_BIVECTOR_DIM,
    SPIN8_PAIRS,
    build_spin8_triality_algebra,
    torch_triality_generators,
)

CAYLEY_TERMS: dict[tuple[int, int, int, int], int] = {
    (0, 1, 2, 3): 1,
    (0, 1, 4, 5): 1,
    (0, 1, 6, 7): -1,
    (0, 2, 4, 6): 1,
    (0, 2, 5, 7): 1,
    (0, 3, 4, 7): 1,
    (0, 3, 5, 6): -1,
    (1, 2, 4, 7): 1,
    (1, 2, 5, 6): -1,
    (1, 3, 4, 6): -1,
    (1, 3, 5, 7): -1,
    (2, 3, 4, 5): 1,
    (2, 3, 6, 7): -1,
    (4, 5, 6, 7): -1,
}

ALLOCATION_PARTITIONS = (
    (5, 0, 0),
    (4, 1, 0),
    (3, 2, 0),
    (3, 1, 1),
    (2, 2, 1),
)

ALLOCATION_TARGETS = {
    (5, 0, 0): {"rank": 25, "determinant": None, "trace_inverse": None},
    (4, 1, 0): {
        "rank": 28,
        "determinant": Fraction(1, 32),
        "trace_inverse": Fraction(115, 2),
    },
    (3, 2, 0): {
        "rank": 28,
        "determinant": Fraction(1, 16),
        "trace_inverse": Fraction(91, 2),
    },
    (3, 1, 1): {
        "rank": 28,
        "determinant": Fraction(135, 2048),
        "trace_inverse": Fraction(227, 5),
    },
    (2, 2, 1): {
        "rank": 28,
        "determinant": Fraction(81, 1024),
        "trace_inverse": Fraction(43, 1),
    },
}


@dataclass(frozen=True)
class FrameAudit:
    maximum_determinant_advantage: float
    maximum_logdet_advantage: float
    violating_count: int
    sample_count: int


def permutation_sign(values: tuple[int, ...] | list[int]) -> int:
    inversions = sum(
        values[left] > values[right]
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )
    return -1 if inversions % 2 else 1


def cayley_value(frame: torch.Tensor) -> torch.Tensor:
    """Evaluate the maintained Spin(7)-invariant Cayley form on 4-frames."""

    if frame.shape[-2:] != (4, 8):
        raise ValueError("Cayley frames must have shape (..., 4, 8)")
    result = frame.new_zeros(frame.shape[:-2])
    for indices, coefficient in CAYLEY_TERMS.items():
        result = result + coefficient * torch.linalg.det(frame[..., list(indices)])
    return result


def cayley_invariance_audit() -> dict[str, object]:
    """Check the Cayley form against all 21 vector-stabilizer generators."""

    algebra = build_spin8_triality_algebra()
    wedge_basis = list(itertools.combinations(range(8), 4))
    wedge_index = {indices: index for index, indices in enumerate(wedge_basis)}
    form = np.zeros(len(wedge_basis), dtype=np.float64)
    for indices, coefficient in CAYLEY_TERMS.items():
        form[wedge_index[indices]] = coefficient

    residual = 0.0
    checked = 0
    for generator_index, (left, right) in enumerate(SPIN8_PAIRS):
        if left == 0 or right == 0:
            continue
        generator = algebra.positive_generators[generator_index]
        induced = np.zeros((len(wedge_basis), len(wedge_basis)), dtype=np.float64)
        for column, basis_vector in enumerate(wedge_basis):
            for position, old_index in enumerate(basis_vector):
                remaining = basis_vector[:position] + basis_vector[position + 1 :]
                for new_index in range(8):
                    coefficient = generator[new_index, old_index]
                    if coefficient == 0.0 or new_index in remaining:
                        continue
                    replaced = list(basis_vector)
                    replaced[position] = new_index
                    sign = permutation_sign(replaced)
                    row = wedge_index[tuple(sorted(replaced))]
                    induced[row, column] += sign * coefficient
        residual = max(residual, float(np.max(np.abs(induced @ form))))
        checked += 1

    return {
        "stabilizer_generator_count": checked,
        "maximum_infinitesimal_invariance_error": residual,
        "passed": checked == 21 and residual == 0.0,
    }


def _rational(value: float) -> sp.Rational:
    return sp.Rational(str(float(value))).limit_denominator(4)


@lru_cache(maxsize=1)
def symbolic_triality_generators() -> list[list[list[list[sp.Rational]]]]:
    algebra = build_spin8_triality_algebra()
    arrays = np.stack(
        (
            algebra.vector_generators,
            algebra.positive_generators,
            algebra.negative_generators,
        )
    )
    return [
        [
            [
                [_rational(arrays[view, plane, row, column]) for column in range(8)]
                for row in range(8)
            ]
            for plane in range(SPIN8_BIVECTOR_DIM)
        ]
        for view in range(3)
    ]


def symbolic_query_projector(
    view: int,
    vector: list[sp.Expr],
    generators: list[list[list[list[sp.Rational]]]],
) -> sp.Matrix:
    jacobian = sp.Matrix(
        8,
        SPIN8_BIVECTOR_DIM,
        lambda row, plane: sum(
            generators[view][plane][row][column] * vector[column] for column in range(8)
        ),
    )
    return jacobian.T * jacobian


def exact_cayley_spectrum_certificate() -> dict[str, object]:
    """Prove the one-parameter characteristic law in exact arithmetic."""

    generators = symbolic_triality_generators()
    basis = [[sp.Integer(row == column) for column in range(8)] for row in range(8)]
    cayley, sine, eigenvalue = sp.symbols("c s lambda")
    final_negative = [
        cayley * basis[3][column] + sine * basis[4][column] for column in range(8)
    ]
    information = (
        symbolic_query_projector(0, basis[0], generators)
        + symbolic_query_projector(1, basis[0], generators)
        + symbolic_query_projector(1, basis[1], generators)
        + symbolic_query_projector(2, basis[2], generators)
        + symbolic_query_projector(2, final_negative, generators)
    )
    characteristic = information.charpoly(eigenvalue).as_expr()
    reduced = sp.rem(
        sp.Poly(characteristic, sine),
        sp.Poly(sine**2 - (1 - cayley**2), sine),
    ).as_expr()
    expected = (
        -sp.Rational(1, 1024)
        * (eigenvalue - 1) ** 4
        * (eigenvalue**2 - 3 * eigenvalue + 1)
        * (cayley - 2 * eigenvalue**2 + 4 * eigenvalue - 1) ** 2
        * (cayley - 2 * eigenvalue**2 + 6 * eigenvalue - 3) ** 2
        * (cayley + 2 * eigenvalue**2 - 6 * eigenvalue + 3) ** 2
        * (cayley + 2 * eigenvalue**2 - 4 * eigenvalue + 1) ** 2
        * (
            2 * cayley * eigenvalue
            - cayley
            - 2 * eigenvalue**3
            + 8 * eigenvalue**2
            - 6 * eigenvalue
            + 1
        )
        * (
            2 * cayley * eigenvalue
            - cayley
            + 2 * eigenvalue**3
            - 8 * eigenvalue**2
            + 6 * eigenvalue
            - 1
        )
    )
    difference = sp.expand(reduced - expected)
    determinant = sp.factor(expected.subs(eigenvalue, 0))
    target_determinant = (1 - cayley**2) ** 3 * (9 - cayley**2) ** 2 / 1024
    balanced = information.subs({cayley: 0, sine: 1})
    calibrated = information.subs({cayley: 1, sine: 0})
    z = sp.symbols("z", real=True)
    determinant_derivative = sp.factor(sp.diff((1 - z) ** 3 * (9 - z) ** 2 / 1024, z))
    expected_derivative = -(z - 9) * (z - 1) ** 2 * (5 * z - 29) / sp.Integer(1024)
    decreasing_identity = sp.simplify(determinant_derivative - expected_derivative) == 0
    return {
        "exact_characteristic_identity": difference == 0,
        "exact_determinant_identity": sp.expand(determinant - target_determinant) == 0,
        "balanced_determinant": str(sp.factor(balanced.det())),
        "balanced_rank": int(balanced.rank()),
        "calibrated_rank": int(calibrated.rank()),
        "determinant_derivative_in_c_squared": str(determinant_derivative),
        "strictly_decreasing_for_zero_to_one": decreasing_identity,
        "passed": (
            difference == 0
            and sp.expand(determinant - target_determinant) == 0
            and balanced.det() == sp.Rational(81, 1024)
            and balanced.rank() == 28
            and calibrated.rank() == 25
            and decreasing_identity
        ),
    }


def exact_restricted_orthogonalization_certificate() -> dict[str, object]:
    """Prove two load-bearing one-correlation slices of the QR conjecture."""

    generators = symbolic_triality_generators()
    basis = [[sp.Integer(row == column) for column in range(8)] for row in range(8)]
    correlation, complement, cayley, sine = sp.symbols("a b c s")
    same_view = [
        correlation * basis[0][column] + complement * basis[1][column]
        for column in range(8)
    ]
    cross_view = [
        correlation * basis[0][column] + complement * basis[2][column]
        for column in range(8)
    ]
    final_negative = [
        cayley * basis[3][column] + sine * basis[4][column] for column in range(8)
    ]

    fixed = symbolic_query_projector(0, basis[0], generators)
    same_information = (
        fixed
        + symbolic_query_projector(1, basis[0], generators)
        + symbolic_query_projector(1, same_view, generators)
        + symbolic_query_projector(2, basis[2], generators)
        + symbolic_query_projector(2, final_negative, generators)
    )
    cross_information = (
        fixed
        + symbolic_query_projector(1, basis[0], generators)
        + symbolic_query_projector(1, basis[1], generators)
        + symbolic_query_projector(2, cross_view, generators)
        + symbolic_query_projector(2, final_negative, generators)
    )

    def reduce_unit_circles(expression: sp.Expr) -> sp.Expr:
        reduced = sp.rem(
            sp.Poly(expression, complement),
            sp.Poly(complement**2 - (1 - correlation**2), complement),
        ).as_expr()
        return sp.factor(
            sp.rem(
                sp.Poly(reduced, sine),
                sp.Poly(sine**2 - (1 - cayley**2), sine),
            ).as_expr()
        )

    same_determinant = reduce_unit_circles(same_information.det(method="domain-ge"))
    cross_determinant = reduce_unit_circles(cross_information.det(method="domain-ge"))
    same_target = (
        (1 - correlation**2) ** 3
        * (2 - correlation**2)
        * (1 - cayley**2) ** 3
        * (9 - cayley**2 - 4 * correlation**2 + correlation**2 * cayley**2) ** 2
        / 2048
    )
    cross_target = (
        (1 - correlation**2) ** 3
        * (1 - cayley**2) ** 3
        * (9 - cayley**2 - 3 * correlation**2 + correlation**2 * cayley**2)
        * (
            correlation**4
            + 4 * correlation**2 * cayley**2
            - 12 * correlation**2
            - 4 * cayley**2
            + 36
        )
        / 4096
    )
    same_identity = sp.simplify(same_determinant - same_target) == 0
    cross_identity = sp.simplify(cross_determinant - cross_target) == 0
    return {
        "same_view_correlation_identity": same_identity,
        "cross_view_correlation_identity": cross_identity,
        "same_view_formula": str(sp.factor(same_target)),
        "cross_view_formula": str(sp.factor(cross_target)),
        "interpretation": (
            "Both normalized one-correlation slices are maximized at zero "
            "correlation for every Cayley square in [0,1]."
        ),
        "passed": same_identity and cross_identity,
    }


def exact_partition_representatives() -> dict[str, object]:
    """Construct exact coordinate representatives for all allocation targets."""

    generators = symbolic_triality_generators()
    basis = [[sp.Integer(row == column) for column in range(8)] for row in range(8)]
    representatives = {
        (5, 0, 0): ((0, 1, 2, 3, 4), (), ()),
        (4, 1, 0): ((1, 2, 6, 7), (0,), ()),
        (3, 2, 0): ((0, 2, 5), (0, 4), ()),
        (3, 1, 1): ((0, 1, 5), (0,), (3,)),
        (2, 2, 1): ((1, 6), (0, 2), (0,)),
    }
    rows = []
    for allocation, coordinate_sets in representatives.items():
        information = sp.zeros(SPIN8_BIVECTOR_DIM)
        for view, coordinates in enumerate(coordinate_sets):
            for coordinate in coordinates:
                information += symbolic_query_projector(
                    view, basis[coordinate], generators
                )
        rank = int(information.rank())
        determinant = None if rank < SPIN8_BIVECTOR_DIM else information.det()
        trace_inverse = (
            None if rank < SPIN8_BIVECTOR_DIM else sp.trace(information.inv())
        )
        target = ALLOCATION_TARGETS[allocation]
        passed = (
            rank == target["rank"]
            and determinant == target["determinant"]
            and trace_inverse == target["trace_inverse"]
        )
        rows.append(
            {
                "allocation": list(allocation),
                "coordinate_sets": [list(values) for values in coordinate_sets],
                "rank": rank,
                "determinant": None if determinant is None else str(determinant),
                "trace_inverse": (
                    None if trace_inverse is None else str(trace_inverse)
                ),
                "characteristic_polynomial": str(
                    sp.factor(information.charpoly().as_expr())
                ),
                "passed": passed,
            }
        )
    return {"rows": rows, "passed": all(bool(row["passed"]) for row in rows)}


def balanced_frame_information(
    frames: torch.Tensor, generators: torch.Tensor
) -> torch.Tensor:
    """Information for `(e0; X0, X1; X2, X3)` balanced sensors."""

    frames = F.normalize(frames, dim=-1)
    batch = frames.shape[0]
    views = torch.tensor([1, 1, 2, 2], device=generators.device)
    selected = generators[views]
    moving = torch.einsum("qpij,bqj->bqip", selected, frames)
    singleton = generators[0, :, :, 0].T.unsqueeze(0).expand(batch, -1, -1)
    jacobian = torch.cat((singleton[:, None], moving), dim=1).reshape(
        batch, QUERY_COUNT * 8, SPIN8_BIVECTOR_DIM
    )
    return jacobian.transpose(-1, -2) @ jacobian


def row_orthonormal_completion(frames: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(frames.transpose(-1, -2), mode="reduced")
    return q.transpose(-1, -2)


def compare_frames_to_qr(frames: torch.Tensor, generators: torch.Tensor) -> FrameAudit:
    raw_information = balanced_frame_information(frames, generators)
    orthogonal = row_orthonormal_completion(frames)
    orthogonal_information = balanced_frame_information(orthogonal, generators)
    raw_sign, raw_logdet = torch.linalg.slogdet(raw_information)
    orthogonal_sign, orthogonal_logdet = torch.linalg.slogdet(orthogonal_information)
    raw_determinant = torch.where(raw_sign > 0, torch.exp(raw_logdet), 0.0)
    orthogonal_determinant = torch.where(
        orthogonal_sign > 0, torch.exp(orthogonal_logdet), 0.0
    )
    determinant_advantage = raw_determinant - orthogonal_determinant
    logdet_advantage = raw_logdet - orthogonal_logdet
    violating = (determinant_advantage > 1e-10) | (logdet_advantage > 1e-8)
    return FrameAudit(
        maximum_determinant_advantage=float(determinant_advantage.max()),
        maximum_logdet_advantage=float(logdet_advantage.max()),
        violating_count=int(violating.sum()),
        sample_count=int(frames.shape[0]),
    )


def random_qr_falsifier(
    generators: torch.Tensor, *, seed: int, samples: int
) -> dict[str, object]:
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
    frames = torch.randn(samples, 4, 8, generator=cpu_generator, dtype=torch.float64)
    audit = compare_frames_to_qr(frames.to(generators.device), generators)
    orthogonal = row_orthonormal_completion(frames.to(generators.device))
    cayley = cayley_value(orthogonal)
    return {
        **audit.__dict__,
        "maximum_absolute_cayley_value": float(cayley.abs().max()),
        "cayley_comass_audit_passed": float(cayley.abs().max()) <= 1.0 + 1e-12,
        "passed": audit.violating_count == 0,
    }


def adversarial_qr_falsifier(
    generators: torch.Tensor,
    *,
    seed: int,
    restarts: int,
    steps: int,
    learning_rate: float,
) -> dict[str, object]:
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
    parameters = nn.Parameter(
        torch.randn(restarts, 4, 8, generator=cpu_generator, dtype=torch.float64).to(
            generators.device
        )
    )
    optimizer = torch.optim.Adam((parameters,), lr=learning_rate)
    ridge = 1e-10 * torch.eye(
        SPIN8_BIVECTOR_DIM, dtype=generators.dtype, device=generators.device
    )
    trajectory = []
    log_steps = {0, 49, 199, steps - 1}
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        frames = F.normalize(parameters, dim=-1)
        raw = balanced_frame_information(frames, generators)
        orthogonal = balanced_frame_information(
            row_orthonormal_completion(frames), generators
        )
        _, raw_logdet = torch.linalg.slogdet(raw + ridge)
        _, orthogonal_logdet = torch.linalg.slogdet(orthogonal + ridge)
        advantage = raw_logdet - orthogonal_logdet
        (-advantage.sum()).backward()
        optimizer.step()
        if step in log_steps:
            trajectory.append(
                {
                    "step": step + 1,
                    "maximum_regularized_logdet_advantage": float(
                        advantage.detach().max()
                    ),
                }
            )
    frames = F.normalize(parameters.detach(), dim=-1)
    audit = compare_frames_to_qr(frames, generators)
    return {
        **audit.__dict__,
        "restarts": restarts,
        "steps": steps,
        "learning_rate": learning_rate,
        "trajectory": trajectory,
        "passed": audit.violating_count == 0,
    }


def optimize_allocation_partitions(
    seed: int,
    generators: torch.Tensor,
    *,
    steps: int,
    restarts: int,
) -> list[dict[str, object]]:
    candidate_views = []
    candidate_vectors = []
    for allocation_index, allocation in enumerate(ALLOCATION_PARTITIONS):
        views = views_from_allocation(allocation)
        for restart in range(restarts):
            cpu_generator = torch.Generator(device="cpu").manual_seed(
                101000 + 10000 * seed + 100 * allocation_index + restart
            )
            candidate_views.append(views)
            candidate_vectors.append(
                torch.randn(
                    QUERY_COUNT, 8, generator=cpu_generator, dtype=torch.float64
                )
            )
    views = torch.stack(candidate_views).to(generators.device)
    vectors = nn.Parameter(torch.stack(candidate_vectors).to(generators.device))
    selected = generators[views]
    ridge = 1e-7 * torch.eye(
        SPIN8_BIVECTOR_DIM, dtype=generators.dtype, device=generators.device
    )
    optimizer = torch.optim.Adam((vectors,), lr=4e-2)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        normalized = F.normalize(vectors, dim=-1)
        blocks = torch.einsum("cqpij,cqj->cqip", selected, normalized)
        jacobian = blocks.reshape(-1, QUERY_COUNT * 8, SPIN8_BIVECTOR_DIM)
        information = jacobian.transpose(-1, -2) @ jacobian
        _, logdet = torch.linalg.slogdet(information + ridge)
        (-logdet.sum()).backward()
        optimizer.step()

    normalized = F.normalize(vectors.detach(), dim=-1)
    rows = []
    for allocation_index, allocation in enumerate(ALLOCATION_PARTITIONS):
        candidates = []
        for restart in range(restarts):
            index = allocation_index * restarts + restart
            design = SensorDesign(views[index], normalized[index], "fresh_partition")
            candidates.append(information_metrics(design, generators))
        best = max(
            candidates,
            key=lambda row: (
                -math.inf
                if row["log_determinant"] is None
                else float(row["log_determinant"])
            ),
        )
        target = ALLOCATION_TARGETS[allocation]
        determinant = (
            None
            if best["log_determinant"] is None
            else math.exp(float(best["log_determinant"]))
        )
        determinant_error = (
            None
            if target["determinant"] is None or determinant is None
            else abs(determinant - float(target["determinant"]))
        )
        inverse_error = (
            None
            if target["trace_inverse"] is None or best["trace_inverse"] is None
            else abs(float(best["trace_inverse"]) - float(target["trace_inverse"]))
        )
        passed = (
            int(best["rank"]) == int(target["rank"])
            and (determinant_error is None or determinant_error <= 1e-8)
            and (inverse_error is None or inverse_error <= 1e-8)
        )
        rows.append(
            {
                "allocation": list(allocation),
                "target_determinant": (
                    None
                    if target["determinant"] is None
                    else str(target["determinant"])
                ),
                "target_trace_inverse": (
                    None
                    if target["trace_inverse"] is None
                    else str(target["trace_inverse"])
                ),
                "observed_determinant": determinant,
                "determinant_error": determinant_error,
                "trace_inverse_error": inverse_error,
                "passed": passed,
                "best": best,
            }
        )
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    allocation_passes = sum(
        all(bool(item["passed"]) for item in row["allocation_partitions"])
        for row in rows
    )
    maximum_determinant = max(
        float(item["observed_determinant"])
        for row in rows
        for item in row["allocation_partitions"]
        if item["observed_determinant"] is not None
    )
    return {
        "seed_count": len(rows),
        "allocation_replication_pass_count": allocation_passes,
        "maximum_observed_partition_determinant": maximum_determinant,
        "global_counterexample_found": maximum_determinant > 81 / 1024 + 1e-10,
    }


def run(
    *,
    device: torch.device,
    seeds: tuple[int, ...],
    partition_steps: int,
    partition_restarts: int,
    random_frames: int,
    adversarial_restarts: int,
    adversarial_steps: int,
) -> dict[str, object]:
    generators = torch_triality_generators(dtype=torch.float64, device=device)
    exact = exact_cayley_spectrum_certificate()
    restricted = exact_restricted_orthogonalization_certificate()
    partition_exact = exact_partition_representatives()
    invariance = cayley_invariance_audit()
    random_audit = random_qr_falsifier(generators, seed=20260803, samples=random_frames)
    adversarial_audit = adversarial_qr_falsifier(
        generators,
        seed=20260804,
        restarts=adversarial_restarts,
        steps=adversarial_steps,
        learning_rate=2e-2,
    )
    rows = [
        {
            "seed": seed,
            "allocation_partitions": optimize_allocation_partitions(
                seed,
                generators,
                steps=partition_steps,
                restarts=partition_restarts,
            ),
        }
        for seed in seeds
    ]
    summary = summarize(rows)
    summary.update(
        {
            "exact_cayley_theorem_passed": bool(exact["passed"]),
            "restricted_orthogonalization_certificate_passed": bool(
                restricted["passed"]
            ),
            "exact_partition_representatives_passed": bool(partition_exact["passed"]),
            "cayley_invariance_passed": bool(invariance["passed"]),
            "random_qr_falsifier_passed": bool(random_audit["passed"]),
            "adversarial_qr_falsifier_passed": bool(adversarial_audit["passed"]),
            "qr_lemma_falsified": not (
                bool(random_audit["passed"]) and bool(adversarial_audit["passed"])
            ),
        }
    )
    return {
        "experiment": "Spin8 Cayley-spectrum theorem and global falsifiers",
        "device": str(device),
        "dtype": str(generators.dtype),
        "seeds": list(seeds),
        "partition_steps": partition_steps,
        "partition_restarts": partition_restarts,
        "random_frame_count": random_frames,
        "adversarial_restarts": adversarial_restarts,
        "adversarial_steps": adversarial_steps,
        "exact_cayley_spectrum": exact,
        "exact_restricted_orthogonalization": restricted,
        "exact_partition_representatives": partition_exact,
        "cayley_invariance": invariance,
        "random_qr_falsifier": random_audit,
        "adversarial_qr_falsifier": adversarial_audit,
        "results": rows,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", default="30,31,32,33,34,35,36,37,38,39")
    parser.add_argument("--partition-steps", type=int, default=1200)
    parser.add_argument("--partition-restarts", type=int, default=12)
    parser.add_argument("--random-frames", type=int, default=10000)
    parser.add_argument("--adversarial-restarts", type=int, default=32)
    parser.add_argument("--adversarial-steps", type=int, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        device=torch.device(args.device),
        seeds=tuple(int(value) for value in args.seeds.split(",") if value),
        partition_steps=args.partition_steps,
        partition_restarts=args.partition_restarts,
        random_frames=args.random_frames,
        adversarial_restarts=args.adversarial_restarts,
        adversarial_steps=args.adversarial_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
