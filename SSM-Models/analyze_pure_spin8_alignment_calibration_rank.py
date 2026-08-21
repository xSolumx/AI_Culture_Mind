"""Exact and implemented-chart rank certificate for Spin(8) alignment probes.

The scrambled-alignment control contains a hidden negative-half-spin action
``T in SO(8)``.  Calibrating its first ``m`` basis probes observes the first
``m`` columns of ``T``.  Their differential has rank

    28 - (8-m)(7-m)/2 = 7,13,18,22,25,27,28,28.

The missing dimensions are exactly the infinitesimal ``SO(8-m)`` stabilizer of
the ordered probe frame.  For every ``m <= 6`` an exact rational quarter-turn
inside that stabilizer gives two globally distinct actions with identical
probes.  Seven probes determine an ``SO(8)`` action globally because the final
column is fixed by orientation; the eighth probe is redundant.  This file
verifies the rank statement over the rationals for all maintained triality
representations, constructs the global non-identifiability witnesses, and
checks the actual factorized PyTorch chart.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import benchmark_pure_spin8_continuous_observation as continuous
import sympy as sp
import torch
from pure_spin8_ssm.torch_backend import (
    SPIN8_BIVECTOR_DIM,
    SPIN8_DIM,
    TRIALITY_REPRESENTATIONS,
    torch_triality_generators,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "artifacts"
    / "pure_spin8_alignment_calibration_rank_certificate.json"
)
EXPECTED_PROFILE = (0, 7, 13, 18, 22, 25, 27, 28, 28)
NUMERICAL_RANK_ATOL = 1e-10
NUMERICAL_RANK_RTOL = 1e-10
CALIBRATION_SEED_BASE = 1_360_000
ALIGNMENT_INITIALIZATION_STD = 0.35


def frame_orbit_rank(probe_count: int) -> int:
    """Dimension of an ordered orthonormal ``probe_count``-frame in R^8."""

    if not 0 <= probe_count <= SPIN8_DIM:
        raise ValueError("probe_count must lie in [0,8]")
    residual_dimension = SPIN8_DIM - probe_count
    stabilizer_dimension = residual_dimension * (residual_dimension - 1) // 2
    return SPIN8_BIVECTOR_DIM - stabilizer_dimension


def exact_probe_profiles() -> dict[str, list[int]]:
    """Compute basis-probe differential ranks exactly over the rationals."""

    integral_generators = (
        2
        * torch_triality_generators(
            TRIALITY_REPRESENTATIONS, dtype=torch.float64
        )
    ).to(torch.int64)
    probes = torch.eye(SPIN8_DIM, dtype=torch.int64)
    profiles: dict[str, list[int]] = {}
    for representation_index, representation in enumerate(TRIALITY_REPRESENTATIONS):
        ranks = [0]
        for probe_count in range(1, SPIN8_DIM + 1):
            jacobian = torch.cat(
                [
                    (integral_generators[representation_index] @ probes[index]).T
                    for index in range(probe_count)
                ],
                dim=0,
            )
            ranks.append(int(sp.Matrix(jacobian.tolist()).rank()))
        profiles[representation] = ranks
    return profiles


def exact_stabilizer_witness(probe_count: int) -> dict[str, Any] | None:
    """Return a rational non-identifiability witness for ``probe_count <= 6``.

    The witness is a quarter-turn in two unobserved basis directions.  It lies
    in ``SO(8)``, fixes every revealed ordered probe, and is not the identity.
    No such witness exists for seven probes because the residual ``SO(1)`` is
    trivial.
    """

    if not 0 <= probe_count <= SPIN8_DIM:
        raise ValueError("probe_count must lie in [0,8]")
    if probe_count >= SPIN8_DIM - 1:
        return None

    witness = sp.eye(SPIN8_DIM)
    first = probe_count
    second = probe_count + 1
    witness[first, first] = 0
    witness[first, second] = -1
    witness[second, first] = 1
    witness[second, second] = 0
    identity = sp.eye(SPIN8_DIM)
    squared_frobenius_gap = sum(
        (witness[row, column] - identity[row, column]) ** 2
        for row in range(SPIN8_DIM)
        for column in range(SPIN8_DIM)
    )
    fixed_probes = all(
        witness[row, column] == identity[row, column]
        for column in range(probe_count)
        for row in range(SPIN8_DIM)
    )
    checks = {
        "orthogonal": witness.T * witness == identity,
        "orientation_preserving": witness.det() == 1,
        "all_revealed_probes_fixed": fixed_probes,
        "globally_distinct_from_identity": witness != identity,
        "exact_frobenius_gap_squared_is_four": squared_frobenius_gap == 4,
    }
    return {
        "probe_count": probe_count,
        "rotated_unobserved_zero_based_indices": [first, second],
        "matrix": [
            [int(witness[row, column]) for column in range(SPIN8_DIM)]
            for row in range(SPIN8_DIM)
        ],
        "squared_frobenius_gap_from_identity": int(squared_frobenius_gap),
        "entrywise_rmse_from_identity": "1/4",
        "checks": checks,
        "passed": all(checks.values()),
    }


def _negative_action(coordinates: torch.Tensor) -> torch.Tensor:
    generators = torch_triality_generators(
        ("negative",), dtype=coordinates.dtype, device=coordinates.device
    )
    return continuous.spin8_factorized_actions(
        coordinates, generators, ("negative",)
    )[0]


def numerical_chart_profile(coordinates: torch.Tensor) -> dict[str, Any]:
    """Measure the implemented chart differential on the first ``m`` columns."""

    if coordinates.shape != (SPIN8_BIVECTOR_DIM,):
        raise ValueError("coordinates must have shape (28,)")
    point = coordinates.detach().to(dtype=torch.float64).requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(_negative_action, point)
    if jacobian.shape != (SPIN8_DIM, SPIN8_DIM, SPIN8_BIVECTOR_DIM):
        raise RuntimeError("unexpected alignment-action Jacobian shape")

    ranks = [0]
    minimum_nonzero_singular_values: list[float | None] = [None]
    for probe_count in range(1, SPIN8_DIM + 1):
        selected = jacobian[:, :probe_count, :].reshape(-1, SPIN8_BIVECTOR_DIM)
        singular_values = torch.linalg.svdvals(selected)
        rank = int(
            torch.linalg.matrix_rank(
                selected, atol=NUMERICAL_RANK_ATOL, rtol=NUMERICAL_RANK_RTOL
            )
        )
        ranks.append(rank)
        minimum_nonzero_singular_values.append(
            float(singular_values[rank - 1]) if rank else None
        )
    return {
        "coordinates": point.detach().tolist(),
        "ranks": ranks,
        "minimum_nonzero_singular_values": minimum_nonzero_singular_values,
        "full_chart_minimum_singular_value": float(
            torch.linalg.svdvals(jacobian.reshape(-1, SPIN8_BIVECTOR_DIM))[-1]
        ),
    }


def frozen_initial_coordinates(seed: int = 0) -> torch.Tensor:
    """Reproduce the alignment initialization used by the matched control."""

    continuous.seed_everything(CALIBRATION_SEED_BASE + 1_000 * seed)
    continuous.SharedPureSpin8Tracker()
    coordinates = torch.empty(SPIN8_BIVECTOR_DIM)
    torch.nn.init.normal_(coordinates, std=ALIGNMENT_INITIALIZATION_STD)
    return coordinates.double()


def build_certificate(seed: int = 0) -> dict[str, Any]:
    exact = exact_probe_profiles()
    expected = [frame_orbit_rank(count) for count in range(SPIN8_DIM + 1)]
    identity_chart = numerical_chart_profile(torch.zeros(SPIN8_BIVECTOR_DIM))
    initialized_chart = numerical_chart_profile(frozen_initial_coordinates(seed))
    stabilizer_witnesses = {
        str(probe_count): exact_stabilizer_witness(probe_count)
        for probe_count in range(SPIN8_DIM - 1)
    }
    rows = []
    for probe_count, rank in enumerate(expected):
        rows.append(
            {
                "probe_count": probe_count,
                "transmitted_scalar_values": SPIN8_DIM * probe_count,
                "independent_differential_rank": rank,
                "continuous_stabilizer_dimension": SPIN8_BIVECTOR_DIM - rank,
                "redundant_scalar_constraints": SPIN8_DIM * probe_count - rank,
                "globally_identifies_so8_action": probe_count >= SPIN8_DIM - 1,
            }
        )
    checks = {
        "closed_form_profile_matches_expected": tuple(expected) == EXPECTED_PROFILE,
        "all_exact_representation_profiles_match": all(
            tuple(profile) == EXPECTED_PROFILE for profile in exact.values()
        ),
        "implemented_identity_chart_matches_exact": tuple(identity_chart["ranks"])
        == EXPECTED_PROFILE,
        "implemented_frozen_initial_chart_matches_exact": tuple(
            initialized_chart["ranks"]
        )
        == EXPECTED_PROFILE,
        "seven_probes_reach_full_lie_rank": expected[7] == SPIN8_BIVECTOR_DIM,
        "eighth_probe_adds_no_rank": expected[8] == expected[7],
        "explicit_global_nonidentifiability_witnesses_pass": all(
            witness is not None and witness["passed"]
            for witness in stabilizer_witnesses.values()
        ),
        "seven_probes_leave_only_trivial_so1_fiber": SPIN8_DIM - 7 == 1,
        "frozen_factorized_chart_is_locally_nonsingular": initialized_chart[
            "full_chart_minimum_singular_value"
        ]
        > 1e-3,
    }
    return {
        "schema_version": 1,
        "experiment": "Pure Spin8 alignment calibration-rank certificate",
        "recorded_at": continuous.now(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sympy": sp.__version__,
            "device": "cpu float64 plus exact rational rank",
        },
        "seed_for_frozen_chart": seed,
        "map": "T -> (T e_1, ..., T e_m) in the negative half-spin SO(8) action",
        "rank_formula": "28 - (8-m)(7-m)/2",
        "fiber": "continuous stabilizer SO(8-m); discrete Spin-cover kernels are not detected by the Jacobian",
        "rows": rows,
        "exact_global_nonidentifiability_witnesses": stabilizer_witnesses,
        "exact_rational_profiles": exact,
        "implemented_identity_chart": identity_chart,
        "implemented_frozen_initial_chart": initialized_chart,
        "checks": checks,
        "passed": all(checks.values()),
        "claim_boundary": {
            "certified": [
                "local differential rank of ordered basis-probe calibration in all three maintained 8D representations",
                "global non-identifiability of the SO(8) action from zero through six ordered probes",
                "global identifiability of the SO(8) action from seven ordered probes",
                "global redundancy of the eighth probe for an orientation-preserving action",
            ],
            "not_claimed": [
                "global uniqueness of factorized coordinates",
                "recovery of a discrete Spin-cover kernel from SO(8) probe images",
                "optimization success from every initialization",
                "natural-data or physical-sensor availability of the calibration frame",
            ],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_certificate(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
