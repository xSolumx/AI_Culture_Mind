"""Independent numerical audit of the Spin(9) rank-three local theorem.

This module deliberately does not import :mod:`spin9_local_hessian`.  The
exact certificate works in a 16-dimensional quadratic tower and constructs
an explicit quotient basis.  Here the same conclusion is attacked through a
different route: PyTorch automatic differentiation on a direct 44-coordinate
chart of the rank-three, trace-three positive-semidefinite stratum.

The result is a floating-point regression and falsifier, not part of the
exact proof.  It is useful precisely because it can expose a shared sign,
normalization, gauge-projection, or dimension-count error in that proof.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from spin9_dirac_clifford import build_spin9_clifford_system


def _candidate_frame(*, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    c = (math.sqrt(241.0) - 17.0) / 24.0
    d = math.sqrt((1.0 + c) / 2.0)
    b = math.sqrt((1.0 - c) / 2.0)
    y = c * (1.0 - c) / (2.0 * b * (1.0 + c))
    z = math.sqrt((1.0 - c) * (1.0 + 2.0 * c) / (2.0 * (1.0 + c) ** 2))
    frame = torch.zeros(16, 3, dtype=dtype, device=device)
    frame[0, 0] = 1.0
    frame[1, 1], frame[8, 1] = d, b
    frame[2, 2], frame[11, 2], frame[12, 2] = -d, y, z
    return frame


def _traceless_symmetric_basis(
    *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    basis = [
        torch.diag(torch.tensor([1.0, -1.0, 0.0], dtype=dtype, device=device))
        / math.sqrt(2.0),
        torch.diag(torch.tensor([1.0, 1.0, -2.0], dtype=dtype, device=device))
        / math.sqrt(6.0),
    ]
    for row, column in ((0, 1), (0, 2), (1, 2)):
        value = torch.zeros(3, 3, dtype=dtype, device=device)
        value[row, column] = value[column, row] = 1.0 / math.sqrt(2.0)
        basis.append(value)
    return torch.stack(basis)


def diagnostics() -> dict[str, object]:
    """Return the direct-autodiff quotient-sign audit."""

    dtype = torch.float64
    device = torch.device("cpu")
    frame = _candidate_frame(dtype=dtype, device=device)
    _, _, right = torch.linalg.svd(frame.T, full_matrices=True)
    complement = right[3:].T.contiguous()
    spectral_basis = _traceless_symmetric_basis(dtype=dtype, device=device)
    generators = (
        torch.tensor(
            np.asarray(build_spin9_clifford_system().doubled_spin_generators),
            dtype=dtype,
            device=device,
        )
        / 2.0
    )

    def objective(coordinates: torch.Tensor) -> torch.Tensor:
        spectral = torch.einsum("k,kij->ij", coordinates[:5], spectral_basis)
        grassmann = coordinates[5:].reshape(13, 3)
        # The factors 1/2 and 1/sqrt(2) make the first-order frame-operator
        # tangents orthonormal in the ambient Frobenius metric.
        factor = frame @ (torch.eye(3, dtype=dtype) + 0.5 * spectral)
        factor = factor + complement @ grassmann / math.sqrt(2.0)
        frame_operator = factor @ factor.T
        frame_operator = 3.0 * frame_operator / torch.trace(frame_operator)
        generator_times_frame = torch.einsum("aij,jk->aik", generators, frame_operator)
        information = torch.einsum("aij,bij->ab", generator_times_frame, generators)
        sign, log_determinant = torch.linalg.slogdet(information)
        if bool(sign.detach() <= 0):
            raise ArithmeticError("candidate chart left the positive information cone")
        return log_determinant

    origin = torch.zeros(44, dtype=dtype, device=device, requires_grad=True)
    value = objective(origin)
    gradient = torch.autograd.grad(value, origin)[0]
    hessian = torch.autograd.functional.hessian(objective, origin, vectorize=True)
    hessian = (hessian + hessian.T) / 2.0
    eigenvalues = torch.linalg.eigvalsh(hessian).detach().cpu().numpy()

    c = (math.sqrt(241.0) - 17.0) / 24.0
    expected_value = (
        10.0 * math.log(1.0 - c)
        + 5.0 * math.log(c + 2.0)
        + 3.0 * math.log(2.0 * c + 1.0)
        - 43.0 * math.log(2.0)
    )
    zero_tolerance = 1e-9
    negative = eigenvalues[eigenvalues < -zero_tolerance]
    zero = eigenvalues[np.abs(eigenvalues) <= zero_tolerance]
    positive = eigenvalues[eigenvalues > zero_tolerance]
    clusters = (
        [negative[:1], negative[1:6], negative[6:11]] if negative.size == 11 else []
    )
    fivefold_cluster_spreads = [float(np.ptp(cluster)) for cluster in clusters[1:]]

    report: dict[str, object] = {
        "schema_version": 1,
        "evidence_class": "independent float64 autodiff falsifier",
        "imports_exact_hessian_certificate": False,
        "chart_dimension": 44,
        "spectral_dimension": 5,
        "grassmann_dimension": 39,
        "candidate_frame_orthonormal_error": float(
            torch.max(torch.abs(frame.T @ frame - torch.eye(3, dtype=dtype)))
        ),
        "candidate_log_determinant": float(value.detach()),
        "closed_form_log_determinant_error": abs(
            float(value.detach()) - expected_value
        ),
        "gradient_max_abs": float(torch.max(torch.abs(gradient))),
        "negative_eigenvalue_count": int(negative.size),
        "numerical_nullity": int(zero.size),
        "positive_eigenvalue_count": int(positive.size),
        "negative_eigenvalues": [float(value) for value in negative],
        "largest_null_abs": float(np.max(np.abs(zero))) if zero.size else None,
        "fivefold_cluster_spreads": fivefold_cluster_spreads,
        "expected_quotient_signature": "11 negative, 33 zero, 0 positive",
        "exact_theorem_claimed_from_this_audit": False,
    }
    report["passed"] = bool(
        report["candidate_frame_orthonormal_error"] < 1e-12
        and report["closed_form_log_determinant_error"] < 1e-12
        and report["gradient_max_abs"] < 1e-11
        and report["negative_eigenvalue_count"] == 11
        and report["numerical_nullity"] == 33
        and report["positive_eigenvalue_count"] == 0
        and len(fivefold_cluster_spreads) == 2
        and max(fivefold_cluster_spreads) < 1e-10
        and not report["exact_theorem_claimed_from_this_audit"]
    )
    return report


if __name__ == "__main__":
    import json

    print(json.dumps(diagnostics(), indent=2, sort_keys=True))
