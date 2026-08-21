"""Exact ordered-probe stabilizer tower and cross-representation atlas.

For the defining action of ``SO(8)``, fixing ``m`` ordered orthonormal probe
images leaves exactly ``SO(8-m)`` acting on the orthogonal complement.  The
orbit is the Stiefel manifold ``V_{8,m}=SO(8)/SO(8-m)`` and has dimension

``28 - binomial(8-m, 2) = m(15-m)/2``.

The odd tower ``m=3,5,7`` therefore splits the 28 tangent coordinates as
``18 + 7 + 3``.  Forgetting the last two probes reverses the refinement as an
exact quotient map, so the tower can be used as a multiresolution gauge chart.

Spinorial and mixed-triality sensors have different stabilizers.  This module
cross-checks, but does not conflate, the maintained Spin(8)
``SU(3)->SU(2)->1`` certificate and Spin(9) ``Spin(7)->SU(3)->1`` probe ladder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPIN8_GEOMETRY = ROOT / "artifacts" / "spin8_coordinate_geometry_20260806.json"
DEFAULT_SPIN9_GATE = ROOT / "artifacts" / "spin9_dirac_clifford_gate_20260807.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "spin8_probe_stabilizer_tower_20260821.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _so8_generators() -> tuple[sp.Matrix, ...]:
    generators = []
    for first in range(8):
        for second in range(first + 1, 8):
            generator = sp.zeros(8)
            generator[first, second] = 1
            generator[second, first] = -1
            generators.append(generator)
    return tuple(generators)


def ordered_probe_action_matrix(probe_count: int) -> sp.Matrix:
    """Return the exact infinitesimal action on the first ``m`` basis probes."""

    if not 0 <= probe_count <= 8:
        raise ValueError("probe_count must lie in 0,...,8")
    generators = _so8_generators()
    if probe_count == 0:
        return sp.zeros(0, len(generators))
    columns = []
    for generator in generators:
        columns.append(
            sp.Matrix.vstack(
                *(generator[:, probe] for probe in range(probe_count))
            )
        )
    return sp.Matrix.hstack(*columns)


def certificate(
    spin8_geometry_path: Path = DEFAULT_SPIN8_GEOMETRY,
    spin9_gate_path: Path = DEFAULT_SPIN9_GATE,
) -> dict[str, object]:
    generators = _so8_generators()
    rows = []
    previous_rank = 0
    for probe_count in range(9):
        observed_rank = int(ordered_probe_action_matrix(probe_count).rank())
        residual_dimension = math.comb(8 - probe_count, 2)
        expected_rank = 28 - residual_dimension
        rows.append(
            {
                "probe_count": probe_count,
                "observed_exact_rank": observed_rank,
                "expected_stiefel_rank": expected_rank,
                "new_coordinates": observed_rank - previous_rank,
                "residual_group": f"SO({8 - probe_count})",
                "residual_lie_dimension": residual_dimension,
                "rank_identity_passed": observed_rank == expected_rank,
            }
        )
        previous_rank = observed_rank

    odd_rows = {row["probe_count"]: row for row in rows}
    split = [
        odd_rows[3]["observed_exact_rank"],
        odd_rows[5]["observed_exact_rank"] - odd_rows[3]["observed_exact_rank"],
        odd_rows[7]["observed_exact_rank"] - odd_rows[5]["observed_exact_rank"],
    ]
    spin8_geometry = json.loads(spin8_geometry_path.read_text(encoding="utf-8"))
    spin8_chain = spin8_geometry["representative_stabilizer_chain"]
    spin9_gate = json.loads(spin9_gate_path.read_text(encoding="utf-8"))
    spin9_ranks = {
        row["probe_count"]: set(row["ranks_mod_prime"].values())
        for row in spin9_gate["spinor_probe_ranks"]
    }
    spin8_mixed_chain = {
        "binary_rank_3": spin8_chain["binary_rank_3"][
            "exact_lie_type_certificate"
        ]["classification"],
        "binary_rank_4": spin8_chain["binary_rank_4"][
            "exact_lie_type_certificate"
        ]["classification"],
        "binary_rank_5": spin8_chain["binary_rank_5"][
            "exact_lie_type_certificate"
        ]["classification"],
    }
    spin9_dimensions = {
        count: 36 - next(iter(ranks)) for count, ranks in spin9_ranks.items()
    }
    passed = (
        len(generators) == 28
        and all(row["rank_identity_passed"] for row in rows)
        and split == [18, 7, 3]
        and [odd_rows[count]["residual_lie_dimension"] for count in (1, 3, 5, 7)]
        == [21, 10, 3, 0]
        and "su(3)" in spin8_mixed_chain["binary_rank_3"]
        and "su(2)" in spin8_mixed_chain["binary_rank_4"]
        and spin8_mixed_chain["binary_rank_5"] == "trivial Lie stabilizer"
        and spin9_dimensions == {1: 21, 2: 8, 3: 0}
        and spin8_geometry["passed"]
        and spin9_gate["passed"]
    )
    return {
        "schema_version": 1,
        "claim_scope": "exact ordered vector-probe tower in SO(8), with separately sourced spinorial ladders",
        "so8_dimension": len(generators),
        "ordered_vector_probe_rows": rows,
        "odd_probe_tower": {
            "forward": "V_8,3 -> V_8,5 -> V_8,7",
            "residual_groups": ["SO(5)", "SO(3)", "SO(1)"],
            "cumulative_ranks": [18, 25, 28],
            "coordinate_split": split,
            "reverse": "forget probes 6,7 and then 4,5; exact quotient maps 28 -> 25 -> 18",
            "fiber_dimensions": [7, 3],
        },
        "one_three_five_seven_residual_dimensions": [21, 10, 3, 0],
        "cross_representation_atlas": {
            "warning": "sensor representation changes the stabilizer; these ladders are not the defining-vector SO chain",
            "spin8_mixed_triality": spin8_mixed_chain,
            "spin8_geometry_artifact": spin8_geometry_path.name,
            "spin8_geometry_sha256": _sha256(spin8_geometry_path),
            "spin9_spinor_orbit_ranks": {
                str(count): next(iter(ranks))
                for count, ranks in spin9_ranks.items()
            },
            "spin9_spinor_stabilizer_dimensions": {
                str(count): dimension
                for count, dimension in spin9_dimensions.items()
            },
            "spin9_interpretation": ["Spin(7)", "SU(3)", "trivial"],
            "spin9_gate_artifact": spin9_gate_path.name,
            "spin9_gate_sha256": _sha256(spin9_gate_path),
        },
        "compiler_interpretation": (
            "18 coarse coordinates, then 7 and 3 gauge refinements; reverse "
            "passes may discard refinements without changing earlier probes"
        ),
        "nonclaims": [
            "the 18+7+3 chart is not an isotypic decomposition",
            "the SU ladders do not follow from the defining-vector tower",
            "no global continuous frame selector is constructed",
            "no task-quality advantage follows from the stabilizer dimensions",
        ],
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    report = certificate()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
