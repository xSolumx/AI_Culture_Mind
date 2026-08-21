"""Exact/numerical observability audit for endpoint-only Pure Spin(8) tasks.

The audit separates two questions that are easy to conflate:

1. Can one representation locally identify the 28-dimensional Lie algebra when
   the same action is observed on sufficiently many independent probe states?
2. Can a quotient observation recover a discrete lift that the quotient erases?

The first answer is yes for each of ``8v``, ``8s+``, and ``8s-``.  The second
answer is no: angles separated by ``2*pi`` have identical vector actions and
opposite half-spin actions, yielding an exact balanced-label collision.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any

import sympy as sp
import torch

import benchmark_pure_spin8_continuous_observation as continuous
from benchmark_pure_spin8_latent_increment import teacher_initial_state
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
    / "pure_spin8_endpoint_observability_certificate.json"
)
EXPECTED_PROBE_RANK_PROFILE = (7, 13, 18, 22, 25, 27, 28, 28)
READOUTS = {
    "vector_only": (0,),
    "positive_only": (1,),
    "negative_only": (2,),
    "spinor_pair": (1, 2),
    "full_triality": (0, 1, 2),
}


def probe_rank_profiles() -> dict[str, list[int]]:
    """Exact rational ranks on the first ``m`` basis probes.

    Twice every maintained generator is integral.  Rank is therefore computed
    by SymPy over the rationals, not inferred from a floating singular-value
    threshold.
    """

    generators = (
        2
        * torch_triality_generators(
            TRIALITY_REPRESENTATIONS, dtype=torch.float64
        )
    ).to(torch.int64)
    probes = torch.eye(SPIN8_DIM, dtype=torch.int64)
    result = {}
    for representation_index, representation in enumerate(TRIALITY_REPRESENTATIONS):
        ranks = []
        for probe_count in range(1, SPIN8_DIM + 1):
            jacobian = torch.cat(
                [
                    (generators[representation_index] @ probes[index]).T
                    for index in range(probe_count)
                ],
                dim=0,
            )
            ranks.append(int(sp.Matrix(jacobian.tolist()).rank()))
        result[representation] = ranks
    return result


def center_signature(delta: float = 0.17) -> dict[str, Any]:
    """Action and endpoint signatures of the tested central element ``-1``."""

    coordinates = torch.zeros(4, SPIN8_BIVECTOR_DIM, dtype=torch.float64)
    coordinates[0, 0] = math.pi + delta
    coordinates[1, 0] = math.pi - delta
    coordinates[2, 0] = 0.83
    coordinates[3, 0] = -0.83
    actions = continuous.action_table_from_coordinates(
        coordinates, device=torch.device("cpu")
    ).double()
    center = actions[1] @ actions[0]
    identity_control = actions[3] @ actions[2]
    identity = torch.eye(SPIN8_DIM, dtype=torch.float64)
    initial = teacher_initial_state().double()[0]
    center_endpoint = torch.einsum("rij,rj->ri", center, initial)
    identity_endpoint = torch.einsum("rij,rj->ri", identity_control, initial)
    per_representation = {}
    for index, representation in enumerate(TRIALITY_REPRESENTATIONS):
        expected_sign = 1.0 if representation == "vector" else -1.0
        per_representation[representation] = {
            "expected_center_sign": expected_sign,
            "action_signature_max_abs": float(
                (center[index] - expected_sign * identity).abs().max()
            ),
            "identity_control_max_abs": float(
                (identity_control[index] - identity).abs().max()
            ),
            "endpoint_center_identity_rmse": float(
                (center_endpoint[index] - identity_endpoint[index])
                .square()
                .mean()
                .sqrt()
            ),
            "endpoint_signed_relation_max_abs": float(
                (
                    center_endpoint[index]
                    - expected_sign * identity_endpoint[index]
                )
                .abs()
                .max()
            ),
        }
    readout_gaps = {}
    for name, indices in READOUTS.items():
        selected_center = center_endpoint[list(indices)]
        selected_identity = identity_endpoint[list(indices)]
        readout_gaps[name] = {
            "representations": [TRIALITY_REPRESENTATIONS[index] for index in indices],
            "center_identity_rmse": float(
                (selected_center - selected_identity).square().mean().sqrt()
            ),
            "center_visible": bool(
                not torch.allclose(
                    selected_center, selected_identity, atol=1e-10, rtol=1e-10
                )
            ),
        }
    return {
        "central_element": "-1 in Spin(8)",
        "per_representation": per_representation,
        "readouts": readout_gaps,
    }


def quotient_collision_certificate() -> dict[str, Any]:
    """Certify an identical-vector-input/opposite-spinor-target collision."""

    coordinates = torch.zeros(2, SPIN8_BIVECTOR_DIM, dtype=torch.float64)
    coordinates[1, 0] = 2.0 * math.pi
    actions = continuous.action_table_from_coordinates(
        coordinates, device=torch.device("cpu")
    ).double()
    vector_collision = float((actions[1, 0] - actions[0, 0]).abs().max())
    spinor_negation = float((actions[1, 1:] + actions[0, 1:]).abs().max())

    probe = torch.zeros(SPIN8_DIM, dtype=torch.float64)
    probe[0] = 1.0
    positive_targets = actions[:, 1] @ probe
    conditional_mean = positive_targets.mean(dim=0)
    balanced_bayes_mse = float(
        (positive_targets - conditional_mean).square().mean()
    )
    return {
        "lift_coordinates_plane0": [0.0, 2.0 * math.pi],
        "quotient_observation": "flattened 8v action matrix",
        "vector_input_collision_max_abs": vector_collision,
        "positive_spinor_target_negation_max_abs": spinor_negation,
        "balanced_conditional_mean_max_abs": float(conditional_mean.abs().max()),
        "balanced_spinor_state_bayes_mse": balanced_bayes_mse,
        "balanced_hidden_lift_bayes_accuracy": 0.5,
        "proof": (
            "The two equally likely labels have identical quotient inputs and "
            "opposite unit-norm 8s+ targets. Their conditional mean is zero, so "
            "squared-loss Bayes risk is ||y||^2/8 = 1/8 and lift accuracy is 1/2."
        ),
    }


def build_certificate() -> dict[str, Any]:
    ranks = probe_rank_profiles()
    center = center_signature()
    collision = quotient_collision_certificate()
    checks = {
        "all_single_representation_probe_profiles_exact": all(
            tuple(profile) == EXPECTED_PROBE_RANK_PROFILE
            for profile in ranks.values()
        ),
        "all_single_representations_reach_lie_rank_28": all(
            profile[-2] == SPIN8_BIVECTOR_DIM for profile in ranks.values()
        ),
        "tested_center_invisible_in_vector_endpoint": not center["readouts"][
            "vector_only"
        ]["center_visible"],
        "tested_center_visible_in_each_half_spin_endpoint": center["readouts"][
            "positive_only"
        ]["center_visible"]
        and center["readouts"]["negative_only"]["center_visible"],
        "quotient_inputs_collide": collision["vector_input_collision_max_abs"]
        <= 1e-12,
        "hidden_spinor_targets_negate": collision[
            "positive_spinor_target_negation_max_abs"
        ]
        <= 1e-12,
        "balanced_bayes_mse_is_one_eighth": abs(
            collision["balanced_spinor_state_bayes_mse"] - 0.125
        )
        <= 1e-12,
    }
    return {
        "schema_version": 1,
        "experiment": "Pure Spin8 endpoint observability certificate",
        "recorded_at": continuous.now(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu float64",
        },
        "probe_rank_profile_theorem": {
            "arithmetic": "exact rank over Q after multiplying generators by 2",
            "expected": list(EXPECTED_PROBE_RANK_PROFILE),
            "measured": ranks,
            "interpretation": (
                "One state leaves a 21-dimensional stabilizer; seven independent "
                "probes remove the full infinitesimal stabilizer in every 8D view."
            ),
        },
        "center_signature": center,
        "quotient_collision": collision,
        "checks": checks,
        "passed": all(checks.values()),
        "claim_boundary": {
            "proved_or_certified": [
                "center visibility for the tested -1 element in the maintained representations",
                "full-rank local Lie-algebra probing from seven basis states",
                "an exact balanced collision and Bayes lower bound for vector-quotient inputs",
            ],
            "not_claimed": [
                "global group-element identifiability from one representation",
                "trainability from any partial readout",
                "natural-observation identifiability",
            ],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_certificate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
