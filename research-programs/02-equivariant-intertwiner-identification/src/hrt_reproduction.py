"""Finite-sampling controls for recent HRT configuration results.

The HRT conjecture is an arbitrary-function theorem in continuous
L^2(R).  This script only reproduces its concrete configuration families on
deterministic sampled windows.  Full column rank here is a sanity check, not a
proof of HRT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

Window = Callable[[np.ndarray], np.ndarray]


def gaussian(t: np.ndarray) -> np.ndarray:
    return np.exp(-np.pi * t**2)


def ultimately_positive(t: np.ndarray) -> np.ndarray:
    return np.exp(-np.abs(t))


def real_signed_gaussian(t: np.ndarray) -> np.ndarray:
    return np.exp(-np.pi * t**2) * np.sin(1.7 * t)


def shift_columns(
    points: list[tuple[float, float]],
    window: Window,
    *,
    grid_size: int = 16_384,
    grid_radius: float = 24.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample normalized Weyl shifts on a fixed real-line quadrature grid."""

    grid = np.linspace(-grid_radius, grid_radius, grid_size)
    columns = []
    for tau, omega in points:
        values = np.exp(2j * np.pi * omega * grid) * window(grid - tau)
        norm = np.sqrt(np.trapezoid(np.abs(values) ** 2, grid))
        if norm == 0:
            raise ValueError("sampled window has zero norm")
        columns.append(values / norm)
    return grid, np.column_stack(columns)


def rank_diagnostic(columns: np.ndarray) -> dict[str, float | int | bool]:
    singular_values = np.linalg.svd(columns, compute_uv=False)
    threshold = float(singular_values[0] * 1e-10)
    numerical_rank = int(np.count_nonzero(singular_values > threshold))
    return {
        "columns": int(columns.shape[1]),
        "rows": int(columns.shape[0]),
        "numerical_rank": numerical_rank,
        "full_column_rank": numerical_rank == columns.shape[1],
        "smallest_singular_value": float(singular_values[-1]),
        "condition_number": float(singular_values[0] / singular_values[-1]),
    }


def finite_weyl_matrix(dimension: int, a: int, b: int) -> np.ndarray:
    """Return a finite Weyl operator on ``C^dimension``.

    The convention is ``(W[a,b]x)[n] = omega**(b*n) x[n-a]``.  It obeys
    ``W[a,b] W[c,e] = omega**(-e*a) W[a+c,b+e]`` modulo ``dimension``.
    """

    if dimension <= 1:
        raise ValueError("dimension must exceed one")
    omega = np.exp(2j * np.pi / dimension)
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for n in range(dimension):
        matrix[n, (n - a) % dimension] = omega ** (b * n)
    return matrix


def finite_weyl_composition_residual(
    dimension: int, left: tuple[int, int], right: tuple[int, int]
) -> float:
    """Check the finite projective Heisenberg composition identity."""

    a, b = left
    c, e = right
    omega = np.exp(2j * np.pi / dimension)
    lhs = finite_weyl_matrix(dimension, a, b) @ finite_weyl_matrix(dimension, c, e)
    rhs = omega ** (-e * a) * finite_weyl_matrix(dimension, a + c, b + e)
    return float(np.max(np.abs(lhs - rhs)))


def finite_weyl_orbit_rank(
    dimension: int, points: list[tuple[int, int]], seed: np.ndarray
) -> dict[str, float | int | bool]:
    """Measure rank of a finite Weyl orbit; this is not continuous HRT."""

    if seed.shape != (dimension,):
        raise ValueError("seed has the wrong dimension")
    columns = np.column_stack([finite_weyl_matrix(dimension, a, b) @ seed for a, b in points])
    return rank_diagnostic(columns)


def symmetric_configuration(n: int, a: float = np.sqrt(2), b: float = np.pi / 7) -> list[tuple[float, float]]:
    """The paper's symmetric (2n+1,2) family: collinear points plus a pair."""

    if n < 1:
        raise ValueError("n must be positive")
    return [(0.0, float(k)) for k in range(-n, n + 1)] + [(a, b), (a, -b)]


def build_report(grid_size: int = 16_384) -> dict[str, object]:
    general_four = [(0.0, 0.0), (0.0, 1.0), (0.73, -0.41), (1.17, 0.22)]
    lattice_four = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    cases: list[tuple[str, list[tuple[float, float]], Window]] = [
        ("symmetric_3_2_gaussian", symmetric_configuration(1), gaussian),
        ("symmetric_5_2_gaussian", symmetric_configuration(2), gaussian),
        ("symmetric_7_2_gaussian", symmetric_configuration(3), gaussian),
        ("general_four_real_signed", general_four, real_signed_gaussian),
        ("general_four_ultimately_positive", general_four, ultimately_positive),
        ("four_point_lattice_gaussian", lattice_four, gaussian),
    ]
    results = []
    for name, points, window in cases:
        _, columns = shift_columns(points, window, grid_size=grid_size)
        diagnostic = rank_diagnostic(columns)
        results.append({"name": name, "points": points, **diagnostic})
    finite_dimension = 32
    finite_seed = np.array(
        [np.exp(-0.08 * (k - 7) ** 2) * np.exp(0.31j * k) for k in range(finite_dimension)],
        dtype=np.complex128,
    )
    finite_points = [(0, 0), (0, 1), (1, 0), (3, 5)]
    report: dict[str, object] = {
        "programme": "02-equivariant-intertwiner-identification",
        "experiment": "finite-sampling-HRT-configuration-controls",
        "status": "numerical-sanity-check-only",
        "inputs": {"grid_size": grid_size, "grid_radius": 24.0, "rank_threshold": "1e-10 * sigma_max"},
        "results": results,
        "finite_heisenberg_bridge": {
            "dimension": finite_dimension,
            "points": finite_points,
            "projective_composition_residual": finite_weyl_composition_residual(
                finite_dimension, finite_points[1], finite_points[2]
            ),
            "orbit_rank": finite_weyl_orbit_rank(finite_dimension, finite_points, finite_seed),
            "status": "finite analogue and algebraic bridge only",
        },
        "published_claims_targeted": [
            {
                "claim": "symmetric (2n+1,2) configurations for arbitrary L2 windows",
                "source": "https://arxiv.org/abs/2607.26878",
            },
            {
                "claim": "every four-point configuration for real-valued windows",
                "source": "https://arxiv.org/abs/2607.26878",
            },
            {
                "claim": "four-point configurations for ultimately positive windows",
                "source": "https://arxiv.org/abs/2509.04281",
            },
        ],
        "claim_boundary": {
            "establishes": [
                "sampled shifted-window columns are numerically independent for the listed controls",
                "the exact configuration families and window classes can be encoded reproducibly",
            ],
            "does_not_establish": [
                "the arbitrary-L2 HRT theorem",
                "a proof for all continuous configurations",
                "a proof for an arbitrary complex-valued window",
                "the analytic product/orbit estimates used by the papers",
            ],
        },
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=16_384)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(grid_size=args.grid_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
