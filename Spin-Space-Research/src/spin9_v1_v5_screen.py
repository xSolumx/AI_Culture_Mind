"""Float64 falsification screen for the coupled Spin(9) V1+V5 slice.

This screen searches the complete three-coordinate Cartan graph chart and its
projective infinity face.  It is designed to find counterexamples and reject
bad monotonicity conjectures.  Passing it is not an exact determinant theorem.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.optimize import differential_evolution

from spin9_dirac_clifford import build_spin9_clifford_system
from spin9_v1_v5_reconstruction import ROOT

DEFAULT_OUTPUT = ROOT / "artifacts" / "spin9_v1_v5_screen_20260811.json"
OPTIMIZATION_SEEDS = tuple(range(6))
RANDOM_SEED = 20_260_811
RANDOM_POINT_COUNT = 30_000
ANGLE_EPSILON = 1e-5


def _frames_and_generators() -> tuple[np.ndarray, ...]:
    sqrt2 = math.sqrt(2.0)
    base = np.zeros((16, 3), dtype=np.float64)
    base[0, 0] = 1
    base[1, 1] = 1 / sqrt2
    base[8, 1] = 1 / sqrt2
    base[2, 2] = -1 / sqrt2
    base[12, 2] = 1 / sqrt2

    scalar = np.zeros((16, 3), dtype=np.float64)
    for row, column, value in (
        (1, 1, 1),
        (2, 2, -1),
        (5, 2, 1),
        (6, 1, 1),
        (8, 1, -1),
        (9, 0, -sqrt2),
        (11, 2, 1),
        (12, 2, -1),
        (14, 0, -sqrt2),
        (15, 1, -1),
    ):
        scalar[row, column] = value

    axis = np.zeros((16, 3), dtype=np.float64)
    for row, column, value in (
        (1, 1, 2),
        (2, 2, 1),
        (5, 2, -1),
        (6, 1, -1),
        (8, 1, -2),
        (9, 0, -2 * sqrt2),
        (11, 2, -1),
        (12, 2, 1),
        (14, 0, sqrt2),
        (15, 1, 1),
    ):
        axis[row, column] = value

    transverse = np.zeros((16, 3), dtype=np.float64)
    for row, column, value in (
        (4, 2, -sqrt2),
        (6, 0, -1),
        (10, 2, sqrt2),
        (14, 1, sqrt2),
        (15, 0, 1),
    ):
        transverse[row, column] = value

    generators = (
        np.asarray(build_spin9_clifford_system().doubled_spin_generators).astype(
            np.float64
        )
        / 2
    )
    return base, scalar, axis, transverse, generators


def _log_determinant(frame: np.ndarray, generators: np.ndarray) -> float:
    singular_values = np.linalg.svd(frame, compute_uv=False)
    if singular_values[-1] <= 1e-12 * singular_values[0]:
        return -math.inf
    orthonormal, _ = np.linalg.qr(frame)
    information = np.zeros((36, 36), dtype=np.float64)
    for column in range(3):
        observations = np.einsum(
            "gij,j->gi",
            generators,
            orthonormal[:, column],
            optimize=True,
        )
        information += observations @ observations.T
    sign, log_determinant = np.linalg.slogdet(information)
    return float(log_determinant) if sign > 0 else -math.inf


def _candidate_data() -> tuple[float, float]:
    c_star = (-17 + sp.sqrt(241)) / 24
    ratio = sp.factor(
        (1 - c_star) ** 10 * (c_star + 2) ** 5 * (2 * c_star + 1) ** 3 / 32
    )
    return float(sp.N(c_star, 17)), float(sp.N(ratio, 17))


def screen(
    *,
    random_point_count: int = RANDOM_POINT_COUNT,
) -> dict[str, object]:
    base, scalar, axis, transverse, generators = _frames_and_generators()
    base_log_determinant = _log_determinant(base, generators)
    sqrt2 = math.sqrt(2.0)

    def graph_log_ratio(theta: float, phi: float, shape: float) -> float:
        x_value = math.tan(theta)
        radial = math.tan(phi)
        frame = (
            base
            + x_value / sqrt2 * scalar
            + radial / sqrt2 * axis
            + 3 * shape * radial / sqrt2 * transverse
        )
        return _log_determinant(frame, generators) - base_log_determinant

    bounds = (
        (-math.pi / 2 + ANGLE_EPSILON, math.pi / 2 - ANGLE_EPSILON),
        (0, math.pi / 2 - ANGLE_EPSILON),
        (0, 1),
    )
    optimization_rows = []
    for seed in OPTIMIZATION_SEEDS:
        optimum = differential_evolution(
            lambda values: -graph_log_ratio(*values),
            bounds,
            seed=seed,
            maxiter=180,
            popsize=18,
            tol=1e-10,
            polish=True,
        )
        theta, phi, shape = optimum.x
        optimization_rows.append(
            {
                "seed": seed,
                "determinant_ratio": math.exp(-float(optimum.fun)),
                "x": math.tan(float(theta)),
                "v5_radial": math.tan(float(phi)),
                "shape_z": float(shape),
            }
        )

    rng = np.random.default_rng(RANDOM_SEED)
    random_best = {
        "determinant_ratio": -math.inf,
        "x": 0.0,
        "v5_radial": 0.0,
        "shape_z": 0.0,
    }
    for theta, phi, shape in zip(
        rng.uniform(bounds[0][0], bounds[0][1], random_point_count),
        rng.uniform(bounds[1][0], bounds[1][1], random_point_count),
        rng.uniform(0, 1, random_point_count),
        strict=True,
    ):
        ratio = math.exp(graph_log_ratio(theta, phi, shape))
        if ratio > random_best["determinant_ratio"]:
            random_best = {
                "determinant_ratio": ratio,
                "x": math.tan(theta),
                "v5_radial": math.tan(phi),
                "shape_z": shape,
            }

    boundary_rows = []
    for scalar_sign in (1, -1):

        def boundary_log_ratio(
            values: np.ndarray,
            sign: int = scalar_sign,
        ) -> float:
            chamber, shape = values
            variation = (
                sign * chamber / sqrt2 * scalar
                + (1 - chamber) / sqrt2 * axis
                + 3 * shape * (1 - chamber) / sqrt2 * transverse
            )
            return _log_determinant(variation, generators) - base_log_determinant

        optimum = differential_evolution(
            lambda values: -boundary_log_ratio(values),
            ((1e-7, 1 - 1e-7), (0, 1)),
            seed=900 + scalar_sign,
            maxiter=240,
            popsize=20,
            tol=1e-11,
            polish=True,
        )
        boundary_rows.append(
            {
                "scalar_sign": scalar_sign,
                "determinant_ratio": math.exp(-float(optimum.fun)),
                "chamber_a": float(optimum.x[0]),
                "shape_z": float(optimum.x[1]),
            }
        )

    pure_at_zero = math.exp(graph_log_ratio(0, 0, 0))
    restored = differential_evolution(
        lambda values: -graph_log_ratio(0, values[0], values[1]),
        (bounds[1], bounds[2]),
        seed=241,
        maxiter=140,
        popsize=16,
        tol=1e-10,
        polish=True,
    )
    restored_ratio = math.exp(-float(restored.fun))

    candidate_c, candidate_ratio = _candidate_data()
    best_optimization_ratio = max(row["determinant_ratio"] for row in optimization_rows)
    best_boundary_ratio = max(row["determinant_ratio"] for row in boundary_rows)
    tolerance = 2e-8
    return {
        "schema_version": 1,
        "claim_scope": "float64 falsification screen on the coupled V1+V5 Cartan graph and projective boundary",
        "candidate_c": candidate_c,
        "candidate_determinant_ratio": candidate_ratio,
        "optimization": {
            "seeds": list(OPTIMIZATION_SEEDS),
            "rows": optimization_rows,
            "best_determinant_ratio": best_optimization_ratio,
            "maximum_candidate_excess": best_optimization_ratio - candidate_ratio,
        },
        "random_screen": {
            "seed": RANDOM_SEED,
            "point_count": random_point_count,
            "best": random_best,
            "candidate_excess": random_best["determinant_ratio"] - candidate_ratio,
        },
        "projective_boundary": {
            "rows": boundary_rows,
            "best_determinant_ratio": best_boundary_ratio,
            "candidate_excess": best_boundary_ratio - candidate_ratio,
        },
        "pointwise_v5_monotonicity_falsifier": {
            "x": 0,
            "pure_v1_ratio": pure_at_zero,
            "optimized_coupled_ratio": restored_ratio,
            "v5_radial": math.tan(float(restored.x[0])),
            "shape_z": float(restored.x[1]),
            "strict_improvement": restored_ratio > pure_at_zero + 1e-4,
        },
        "no_candidate_counterexample_found": bool(
            best_optimization_ratio <= candidate_ratio + tolerance
            and random_best["determinant_ratio"] <= candidate_ratio + tolerance
            and best_boundary_ratio <= candidate_ratio + tolerance
        ),
        "global_optimality_proved": False,
        "passed": bool(
            restored_ratio > pure_at_zero + 1e-4
            and best_optimization_ratio <= candidate_ratio + tolerance
            and random_best["determinant_ratio"] <= candidate_ratio + tolerance
            and best_boundary_ratio <= candidate_ratio + tolerance
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--random-points", type=int, default=RANDOM_POINT_COUNT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = screen(random_point_count=args.random_points)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
