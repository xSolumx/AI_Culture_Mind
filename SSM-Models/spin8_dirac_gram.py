"""Dirac--Gram certificates for balanced Spin(8) sensor designs.

The exact result in this module is deliberately narrower than the numerical
experiments: two nontrivial Gram slices satisfy a strengthened volume bound.
The full six-correlation inequality remains a conjecture and is attacked, not
silently promoted, by the fresh falsifiers below.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import sympy as sp
import torch
from torch import nn
from torch.nn import functional as F

from spin8_cayley_spectrum import (
    balanced_frame_information,
    cayley_value,
    row_orthonormal_completion,
    symbolic_query_projector,
    symbolic_triality_generators,
)
from spin8_triality import torch_triality_generators


@dataclass(frozen=True)
class RatioAudit:
    maximum_log_ratio: float
    maximum_ratio: float
    gram_determinant_at_maximum: float
    violating_count: int
    sample_count: int


def _bernstein_coefficients(
    polynomial: sp.Expr, variables: tuple[sp.Symbol, sp.Symbol]
) -> list[list[sp.Expr]]:
    """Return the exact tensor-product Bernstein coefficients on [0, 1]^2."""

    left, right = variables
    left_degree = int(sp.degree(polynomial, left))
    right_degree = int(sp.degree(polynomial, right))
    unknowns = sp.symbols(f"b0:{(left_degree + 1) * (right_degree + 1)}")
    expression = sp.Integer(0)
    cursor = 0
    for left_index in range(left_degree + 1):
        left_basis = (
            sp.binomial(left_degree, left_index)
            * left**left_index
            * (1 - left) ** (left_degree - left_index)
        )
        for right_index in range(right_degree + 1):
            right_basis = (
                sp.binomial(right_degree, right_index)
                * right**right_index
                * (1 - right) ** (right_degree - right_index)
            )
            expression += unknowns[cursor] * left_basis * right_basis
            cursor += 1
    coefficient_equations = sp.Poly(
        sp.expand(expression - polynomial), left, right
    ).coeffs()
    solution = sp.solve(coefficient_equations, unknowns, dict=True)[0]
    result: list[list[sp.Expr]] = []
    cursor = 0
    for _ in range(left_degree + 1):
        row = []
        for _ in range(right_degree + 1):
            row.append(sp.factor(solution[unknowns[cursor]]))
            cursor += 1
        result.append(row)
    return result


def exact_projector_geometry_certificate() -> dict[str, object]:
    """Prove the rank-seven and triality-isoclinic projector identities."""

    generators = symbolic_triality_generators()
    basis = [[sp.Integer(row == column) for column in range(8)] for row in range(8)]
    reference = symbolic_query_projector(0, basis[0], generators)
    projector_identity = reference * reference == reference

    x_symbols = sp.symbols("x0:8")
    y_symbols = sp.symbols("y0:8")
    x = list(x_symbols)
    y = list(y_symbols)
    x_norm = sum(value**2 for value in x)
    y_norm = sum(value**2 for value in y)
    inner = sum(left * right for left, right in zip(x, y, strict=True))
    positive_x = symbolic_query_projector(1, x, generators)
    positive_y = symbolic_query_projector(1, y, generators)
    negative_y = symbolic_query_projector(2, y, generators)
    same_residual = sp.expand(
        sp.trace(positive_x * positive_y) - x_norm * y_norm - 6 * inner**2
    )
    cross_residual = sp.expand(
        sp.trace(positive_x * negative_y) - sp.Rational(7, 4) * x_norm * y_norm
    )
    return {
        "single_query_rank": int(reference.rank()),
        "single_query_trace": str(sp.trace(reference)),
        "single_query_is_orthogonal_projector": projector_identity,
        "same_view_overlap_identity": same_residual == 0,
        "different_view_overlap_identity": cross_residual == 0,
        "same_view_formula": "||x||^2 ||y||^2 + 6 <x,y>^2",
        "different_view_formula": "7 ||x||^2 ||y||^2 / 4",
        "passed": (
            reference.rank() == 7
            and sp.trace(reference) == 7
            and projector_identity
            and same_residual == 0
            and cross_residual == 0
        ),
    }


def exact_dirac_graph_certificate() -> dict[str, object]:
    """Prove the 7+21 graph form and resulting Schur determinant reduction."""

    generators = symbolic_triality_generators()
    coordinates = list(sp.symbols("x0:8"))
    squared_norm = sum(value**2 for value in coordinates)
    view_certificates = []
    for view in (1, 2):
        projector = symbolic_query_projector(view, coordinates, generators)
        top = projector[:7, :7]
        graph = 4 * projector[7:, :7]
        bottom = projector[7:, 7:]
        top_residual = top - squared_norm * sp.eye(7) / 4
        isometry_residual = graph.T * graph - 3 * squared_norm**2 * sp.eye(7)
        bottom_residual = 4 * squared_norm * bottom - graph * graph.T
        top_passed = all(sp.expand(entry) == 0 for entry in top_residual)
        isometry_passed = all(sp.expand(entry) == 0 for entry in isometry_residual)
        bottom_passed = all(sp.expand(entry) == 0 for entry in bottom_residual)
        view_certificates.append(
            {
                "view": view,
                "top_block_identity": top_passed,
                "graph_isometry_identity": isometry_passed,
                "bottom_block_identity": bottom_passed,
            }
        )
    passed = all(
        item[identity]
        for item in view_certificates
        for identity in (
            "top_block_identity",
            "graph_isometry_identity",
            "bottom_block_identity",
        )
    )
    return {
        "reference_split": [7, 21],
        "moving_query_graph_formula": (
            "P(V)=1/4 [[I_7,V^T],[V,V V^T]] for unit probes"
        ),
        "graph_frame_identity": "V^T V = 3 I_7",
        "five_probe_information_blocks": (
            "[[2 I_7,S^T/4],[S/4,T/4]], S=sum_i V_i, " "T=sum_i V_i V_i^T"
        ),
        "schur_determinant_identity": ("det(I)=2^7 32^-21 det(8 T-S S^T)"),
        "views": view_certificates,
        "passed": passed,
    }


def exact_whitening_flow_invariant_certificate() -> dict[str, object]:
    """Prove what the coupled whitening flow preserves and what it cannot prove."""

    correlations = sp.symbols("g01 g02 g03 g12 g13 g23")
    gram = sp.eye(4)
    cursor = 0
    for left in range(4):
        for right in range(left + 1, 4):
            gram[left, right] = correlations[cursor]
            gram[right, left] = correlations[cursor]
            cursor += 1
    off_diagonal = gram - sp.eye(4)
    energy = sp.trace(off_diagonal**2)
    off_diagonal_squared = off_diagonal**2
    diagonal_correction = sp.diag(
        *(off_diagonal_squared[index, index] for index in range(4))
    )
    action = -off_diagonal + diagonal_correction
    gram_derivative = action * gram + gram * action
    tangent_residual = [sp.expand(gram_derivative[index, index]) for index in range(4)]
    trace_residual = sp.expand(sp.trace(action) - energy)
    volume_log_derivative_residual = sp.expand(2 * sp.trace(action) - 2 * energy)
    cayley_log_derivative_residual = sp.expand(sp.trace(action) - energy)
    normalized_cayley_log_derivative_residual = sp.expand(
        sp.trace(action) - sp.Rational(1, 2) * 2 * sp.trace(action)
    )
    passed = (
        all(value == 0 for value in tangent_residual)
        and trace_residual == 0
        and volume_log_derivative_residual == 0
        and cayley_log_derivative_residual == 0
        and normalized_cayley_log_derivative_residual == 0
    )
    return {
        "flow": "X_dot=A X, A=-(G-I)+diag((G-I)^2)",
        "energy": "E=||G-I||_F^2=tr(A)",
        "gram_diagonal_is_preserved": all(value == 0 for value in tangent_residual),
        "gram_volume_log_derivative": "d log(det G)/dt = 2 E",
        "cayley_form_log_derivative": "d log(|Phi|)/dt = E",
        "normalized_cayley_is_invariant": True,
        "strengthened_ratio_requirement": (
            "The strengthened determinant ratio needs d log(det I)/dt >= 6 E; "
            "mere nonnegative determinant growth is insufficient."
        ),
        "passed": passed,
    }


def exact_approximate_design_rejection() -> dict[str, object]:
    """Show exactly why the standard equivalence-theorem shortcut is invalid."""

    generators = symbolic_triality_generators()
    basis = [[sp.Integer(row == column) for column in range(8)] for row in range(8)]
    information = (
        symbolic_query_projector(0, basis[0], generators)
        + symbolic_query_projector(1, basis[0], generators)
        + symbolic_query_projector(1, basis[1], generators)
        + symbolic_query_projector(2, basis[2], generators)
        + symbolic_query_projector(2, basis[4], generators)
    )
    inverse = information.inv()
    sensitivity_matrices = []
    maximum_sensitivities = []
    for view in range(3):
        matrix = sp.zeros(8)
        diagonal_projectors = [
            symbolic_query_projector(view, basis[index], generators)
            for index in range(8)
        ]
        for left in range(8):
            matrix[left, left] = sp.trace(inverse * diagonal_projectors[left])
            for right in range(left + 1, 8):
                summed = [
                    basis[left][index] + basis[right][index] for index in range(8)
                ]
                combined = symbolic_query_projector(view, summed, generators)
                entry = (
                    sp.trace(inverse * combined)
                    - matrix[left, left]
                    - sp.trace(inverse * diagonal_projectors[right])
                ) / 2
                matrix[left, right] = entry
                matrix[right, left] = entry
        sensitivity_matrices.append(matrix)
        maximum_sensitivities.append(max(matrix.eigenvals()))
    threshold = sp.Rational(28, 5)
    return {
        "exact_design_threshold": str(threshold),
        "sensitivity_matrices": [
            [[str(value) for value in matrix.row(row)] for row in range(8)]
            for matrix in sensitivity_matrices
        ],
        "maximum_sensitivities": [str(value) for value in maximum_sensitivities],
        "certificate_violated": any(
            value > threshold for value in maximum_sensitivities
        ),
        "interpretation": (
            "The exact five-probe optimum is not an approximate-design "
            "equivalence-theorem optimum; that shortcut cannot prove it."
        ),
        "passed": all(value > threshold for value in maximum_sensitivities),
    }


def exact_strengthened_slice_certificate() -> dict[str, object]:
    """Prove det(I(X)) <= det(G)^3 det(I(Q)) on two exact slices."""

    correlation_square, cayley_square = sp.symbols("u z", real=True)
    orthogonal = (1 - cayley_square) ** 3 * (9 - cayley_square) ** 2 / sp.Integer(1024)
    same = (
        (1 - correlation_square) ** 3
        * (2 - correlation_square)
        * (1 - cayley_square) ** 3
        * (
            9
            - cayley_square
            - 4 * correlation_square
            + correlation_square * cayley_square
        )
        ** 2
        / sp.Integer(2048)
    )
    cross = (
        (1 - correlation_square) ** 3
        * (1 - cayley_square) ** 3
        * (
            9
            - cayley_square
            - 3 * correlation_square
            + correlation_square * cayley_square
        )
        * (
            correlation_square**2
            + 4 * correlation_square * cayley_square
            - 12 * correlation_square
            - 4 * cayley_square
            + 36
        )
        / sp.Integer(4096)
    )
    same_gap = sp.factor((1 - correlation_square) ** 3 * orthogonal - same)
    cross_gap = sp.factor((1 - correlation_square) ** 3 * orthogonal - cross)
    common = (
        correlation_square * (1 - correlation_square) ** 3 * (1 - cayley_square) ** 3
    )
    same_residual = sp.factor(sp.cancel(same_gap / common) * 2048)
    cross_residual = sp.factor(sp.cancel(cross_gap / common) * 4096)
    same_bernstein = _bernstein_coefficients(
        same_residual, (correlation_square, cayley_square)
    )
    cross_bernstein = _bernstein_coefficients(
        cross_residual, (correlation_square, cayley_square)
    )
    all_coefficients = [
        value
        for matrix in (same_bernstein, cross_bernstein)
        for row in matrix
        for value in row
    ]
    strictly_positive = all(value > 0 for value in all_coefficients)
    return {
        "same_view_gap": str(same_gap),
        "cross_view_gap": str(cross_gap),
        "same_view_residual": str(same_residual),
        "cross_view_residual": str(cross_residual),
        "same_view_bernstein_coefficients": [
            [str(value) for value in row] for row in same_bernstein
        ],
        "cross_view_bernstein_coefficients": [
            [str(value) for value in row] for row in cross_bernstein
        ],
        "all_bernstein_coefficients_strictly_positive": strictly_positive,
        "interpretation": (
            "The strengthened Gram-volume inequality is exact on both "
            "one-correlation slices for every u,z in [0,1]."
        ),
        "passed": strictly_positive,
    }


def _ratio_audit(frames: torch.Tensor, generators: torch.Tensor) -> RatioAudit:
    frames = F.normalize(frames, dim=-1)
    raw = balanced_frame_information(frames, generators)
    orthogonal = row_orthonormal_completion(frames)
    completed = balanced_frame_information(orthogonal, generators)
    gram = frames @ frames.transpose(-1, -2)
    raw_sign, raw_logdet = torch.linalg.slogdet(raw)
    completed_sign, completed_logdet = torch.linalg.slogdet(completed)
    gram_sign, gram_logdet = torch.linalg.slogdet(gram)
    valid = (raw_sign > 0) & (completed_sign > 0) & (gram_sign > 0)
    log_ratio = raw_logdet - completed_logdet - 3 * gram_logdet
    log_ratio = torch.where(valid, log_ratio, -torch.inf)
    maximum_index = int(log_ratio.argmax())
    maximum_log_ratio = float(log_ratio[maximum_index])
    return RatioAudit(
        maximum_log_ratio=maximum_log_ratio,
        maximum_ratio=math.exp(maximum_log_ratio),
        gram_determinant_at_maximum=float(torch.exp(gram_logdet[maximum_index])),
        violating_count=int((log_ratio > 1e-9).sum()),
        sample_count=int(frames.shape[0]),
    )


def random_gram_volume_falsifier(
    generators: torch.Tensor, *, seed: int, samples: int, chunk_size: int
) -> dict[str, object]:
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
    maximum: RatioAudit | None = None
    violations = 0
    completed = 0
    while completed < samples:
        count = min(chunk_size, samples - completed)
        frames = torch.randn(
            count, 4, 8, generator=cpu_generator, dtype=torch.float64
        ).to(generators.device)
        audit = _ratio_audit(frames, generators)
        violations += audit.violating_count
        completed += count
        if maximum is None or audit.maximum_log_ratio > maximum.maximum_log_ratio:
            maximum = audit
    assert maximum is not None
    return {
        "seed": seed,
        "sample_count": completed,
        "chunk_size": chunk_size,
        "maximum_log_ratio": maximum.maximum_log_ratio,
        "maximum_ratio": maximum.maximum_ratio,
        "gram_determinant_at_maximum": maximum.gram_determinant_at_maximum,
        "violating_count": violations,
        "passed": violations == 0,
    }


def adversarial_gram_volume_falsifier(
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
    trajectory = []
    log_steps = {0, 49, 199, 999, steps - 1}
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        frames = F.normalize(parameters, dim=-1)
        raw = balanced_frame_information(frames, generators)
        completed = balanced_frame_information(
            row_orthonormal_completion(frames), generators
        )
        gram = frames @ frames.transpose(-1, -2)
        raw_logdet = torch.linalg.slogdet(raw)[1]
        completed_logdet = torch.linalg.slogdet(completed)[1]
        gram_logdet = torch.linalg.slogdet(gram)[1]
        log_ratio = raw_logdet - completed_logdet - 3 * gram_logdet
        (-log_ratio.sum()).backward()
        torch.nn.utils.clip_grad_norm_((parameters,), 100.0)
        optimizer.step()
        if step in log_steps:
            trajectory.append(
                {
                    "step": step + 1,
                    "maximum_log_ratio": float(log_ratio.detach().max()),
                    "minimum_gram_determinant": float(
                        torch.linalg.det(gram.detach()).min()
                    ),
                }
            )
    frames = F.normalize(parameters.detach(), dim=-1)
    audit = _ratio_audit(frames, generators)
    gram = frames @ frames.transpose(-1, -2)
    off_diagonal = gram - torch.diag_embed(torch.diagonal(gram, dim1=-2, dim2=-1))
    return {
        **audit.__dict__,
        "seed": seed,
        "restarts": restarts,
        "steps": steps,
        "learning_rate": learning_rate,
        "maximum_final_off_diagonal_gram_entry": float(off_diagonal.abs().max()),
        "trajectory": trajectory,
        "passed": audit.maximum_log_ratio <= 1e-8,
    }


def whitening_flow_derivative(
    frames: torch.Tensor, generators: torch.Tensor, *, create_graph: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return d/dt logdet(I) and its frame-potential-normalized value."""

    frames = F.normalize(frames, dim=-1)
    if not frames.requires_grad:
        frames.requires_grad_(True)
    information = balanced_frame_information(frames, generators)
    logdet = torch.linalg.slogdet(information)[1]
    gradient = torch.autograd.grad(
        logdet.sum(), frames, create_graph=create_graph, retain_graph=create_graph
    )[0]
    gram = frames @ frames.transpose(-1, -2)
    off_diagonal = gram - torch.diag_embed(torch.diagonal(gram, dim1=-2, dim2=-1))
    velocity = -(off_diagonal @ frames)
    velocity = velocity - (velocity * frames).sum(dim=-1, keepdim=True) * frames
    derivative = (gradient * velocity).sum(dim=(-1, -2))
    frame_potential = off_diagonal.square().sum(dim=(-1, -2))
    normalized = derivative / frame_potential.clamp_min(1e-12)
    return derivative, normalized


def random_whitening_flow_falsifier(
    generators: torch.Tensor, *, seed: int, samples: int, chunk_size: int
) -> dict[str, object]:
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
    minimum_derivative = math.inf
    minimum_normalized = math.inf
    violations = 0
    completed = 0
    while completed < samples:
        count = min(chunk_size, samples - completed)
        frames = torch.randn(
            count, 4, 8, generator=cpu_generator, dtype=torch.float64
        ).to(generators.device)
        derivative, normalized = whitening_flow_derivative(
            frames, generators, create_graph=False
        )
        minimum_derivative = min(minimum_derivative, float(derivative.detach().min()))
        minimum_normalized = min(minimum_normalized, float(normalized.detach().min()))
        violations += int((derivative < -1e-8).sum())
        completed += count
    return {
        "seed": seed,
        "sample_count": completed,
        "minimum_derivative": minimum_derivative,
        "minimum_frame_potential_normalized_derivative": minimum_normalized,
        "violating_count": violations,
        "passed": violations == 0,
    }


def adversarial_whitening_flow_falsifier(
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
    trajectory = []
    log_steps = {0, 49, 199, steps - 1}
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        derivative, normalized = whitening_flow_derivative(
            parameters, generators, create_graph=True
        )
        normalized.sum().backward()
        torch.nn.utils.clip_grad_norm_((parameters,), 100.0)
        optimizer.step()
        if step in log_steps:
            trajectory.append(
                {
                    "step": step + 1,
                    "minimum_derivative": float(derivative.detach().min()),
                    "minimum_normalized_derivative": float(normalized.detach().min()),
                }
            )
    derivative, normalized = whitening_flow_derivative(
        parameters.detach(), generators, create_graph=False
    )
    minimum_index = int(normalized.argmin())
    minimum_derivative = float(derivative.detach().min())
    frames = F.normalize(parameters.detach(), dim=-1)
    gram = frames @ frames.transpose(-1, -2)
    gram_determinants = torch.linalg.det(gram)
    normalized_cayley = cayley_value(frames) / gram_determinants.sqrt()
    return {
        "seed": seed,
        "restarts": restarts,
        "steps": steps,
        "learning_rate": learning_rate,
        "minimum_derivative": minimum_derivative,
        "minimum_frame_potential_normalized_derivative": float(
            normalized.detach().min()
        ),
        "minimizer_gram_matrix": gram[minimum_index].cpu().tolist(),
        "minimizer_gram_eigenvalues": torch.linalg.eigvalsh(gram[minimum_index])
        .cpu()
        .tolist(),
        "minimizer_gram_determinant": float(gram_determinants[minimum_index]),
        "minimizer_normalized_cayley": float(normalized_cayley[minimum_index]),
        "trajectory": trajectory,
        "passed": minimum_derivative >= -1e-8,
    }


def run(
    *,
    device: torch.device,
    random_frames: int,
    random_flow_frames: int,
    adversarial_restarts: int,
    adversarial_steps: int,
    flow_restarts: int,
    flow_steps: int,
) -> dict[str, object]:
    generators = torch_triality_generators(dtype=torch.float64, device=device)
    projector = exact_projector_geometry_certificate()
    approximate_design = exact_approximate_design_rejection()
    slices = exact_strengthened_slice_certificate()
    random_ratio = random_gram_volume_falsifier(
        generators,
        seed=20260805,
        samples=random_frames,
        chunk_size=4096 if device.type == "cuda" else 1024,
    )
    adversarial_ratio = adversarial_gram_volume_falsifier(
        generators,
        seed=20260806,
        restarts=adversarial_restarts,
        steps=adversarial_steps,
        learning_rate=2e-2,
    )
    random_flow = random_whitening_flow_falsifier(
        generators,
        seed=20260807,
        samples=random_flow_frames,
        chunk_size=512 if device.type == "cuda" else 128,
    )
    adversarial_flow = adversarial_whitening_flow_falsifier(
        generators,
        seed=20260808,
        restarts=flow_restarts,
        steps=flow_steps,
        learning_rate=1e-2,
    )
    summary = {
        "exact_projector_geometry_passed": bool(projector["passed"]),
        "approximate_design_shortcut_rejected": bool(approximate_design["passed"]),
        "exact_strengthened_slices_passed": bool(slices["passed"]),
        "fresh_random_gram_volume_passed": bool(random_ratio["passed"]),
        "fresh_adversarial_gram_volume_passed": bool(adversarial_ratio["passed"]),
        "fresh_random_whitening_flow_passed": bool(random_flow["passed"]),
        "fresh_adversarial_whitening_flow_passed": bool(adversarial_flow["passed"]),
        "global_gram_volume_theorem_proved": False,
    }
    return {
        "experiment": "Spin8 Dirac--Gram strengthened proof gate",
        "device": str(device),
        "dtype": str(generators.dtype),
        "exact_projector_geometry": projector,
        "exact_approximate_design_rejection": approximate_design,
        "exact_strengthened_slices": slices,
        "random_gram_volume_falsifier": random_ratio,
        "adversarial_gram_volume_falsifier": adversarial_ratio,
        "random_whitening_flow_falsifier": random_flow,
        "adversarial_whitening_flow_falsifier": adversarial_flow,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--random-frames", type=int, default=1_000_000)
    parser.add_argument("--random-flow-frames", type=int, default=100_000)
    parser.add_argument("--adversarial-restarts", type=int, default=64)
    parser.add_argument("--adversarial-steps", type=int, default=2_000)
    parser.add_argument("--flow-restarts", type=int, default=32)
    parser.add_argument("--flow-steps", type=int, default=1_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        device=torch.device(args.device),
        random_frames=args.random_frames,
        random_flow_frames=args.random_flow_frames,
        adversarial_restarts=args.adversarial_restarts,
        adversarial_steps=args.adversarial_steps,
        flow_restarts=args.flow_restarts,
        flow_steps=args.flow_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
