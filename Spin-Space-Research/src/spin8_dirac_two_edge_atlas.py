"""Certified triangular Bernstein atlas for the finite second Dirac edge.

The maintained two-edge coefficient map gives eight physical orientation
margin numerators in the rational half-angle cube.  The paired blow-ups

    (x_i, x_j) = (u, u*v),       (x_i, x_j) = (u*v, v)

cover the complete ``(x_i,x_j)`` square.  The
finite trees below retain both children of every split.  At each leaf:

* outward binary64 error bounds certify every control whose lower endpoint is
  positive;
* unreachable controls are structurally and therefore exactly zero;
* any cancellation controls left by the enclosure are replayed with positively
  scaled exact integer Bernstein rows.

Consequently a passing report is a domain-wide sign certificate for the eight
rationalized orientation margins.  It is not a certificate for the final
Cholesky residual outside the frozen ``h=0`` two-edge family.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import TypeAlias

import numpy as np

from spin8_dirac_two_edge_rational import (
    float_bernstein_enclosure,
    orientation_power_tensor,
    rationalized_power_tensors,
    selected_scaled_integer_bernstein,
    triangle_blowup_power_tensor,
)
from spin8_resource_limits import constrain_current_process

Tree: TypeAlias = None | tuple[int, int, "Tree", "Tree"]


def split(first_axis: int, second_axis: int, lower: Tree, upper: Tree) -> Tree:
    """Make one cover-tree node with explicit lower and upper children."""

    return (first_axis, second_axis, lower, upper)


LEAF: Tree = None

# The trees were discovered by a tolerance-free chart search and are frozen as
# proof data here.  Every split contains both triangular children.  Channels
# follow the Hadamard order in spin8_dirac_two_edge_kernel.py.
ATLAS_TREES: dict[tuple[int, int], Tree] = {
    (0, 1): split(0, 1, LEAF, LEAF),
    (0, -1): split(
        1,
        3,
        split(0, 1, LEAF, LEAF),
        split(0, 2, LEAF, LEAF),
    ),
    (1, 1): split(
        2,
        3,
        split(0, 1, LEAF, LEAF),
        split(0, 1, LEAF, LEAF),
    ),
    (1, -1): split(
        2,
        3,
        split(
            1,
            2,
            split(0, 1, LEAF, LEAF),
            split(
                2,
                4,
                split(
                    0,
                    2,
                    LEAF,
                    split(1, 2, LEAF, LEAF),
                ),
                LEAF,
            ),
        ),
        split(
            1,
            3,
            split(0, 1, LEAF, LEAF),
            split(
                3,
                4,
                split(
                    0,
                    3,
                    LEAF,
                    split(1, 3, LEAF, LEAF),
                ),
                LEAF,
            ),
        ),
    ),
    (2, 1): split(
        2,
        3,
        split(0, 4, LEAF, LEAF),
        split(0, 4, LEAF, LEAF),
    ),
    (2, -1): split(
        2,
        3,
        split(1, 4, LEAF, LEAF),
        split(1, 4, LEAF, LEAF),
    ),
    (3, 1): split(0, 2, LEAF, LEAF),
    (3, -1): split(1, 3, LEAF, LEAF),
}


def _leaf_paths(tree: Tree, prefix: tuple[tuple[int, int, bool], ...] = ()):
    if tree is None:
        yield prefix
        return
    first_axis, second_axis, lower, upper = tree
    yield from _leaf_paths(lower, prefix + ((first_axis, second_axis, False),))
    yield from _leaf_paths(upper, prefix + ((first_axis, second_axis, True),))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_fallback(
    power: np.ndarray, unresolved: np.ndarray
) -> tuple[dict[str, object], bool]:
    indices = np.nonzero(unresolved)
    if not indices[0].size:
        return {
            "count": 0,
            "cartesian_control_count": 0,
            "negative_count": 0,
            "zero_count": 0,
            "positive_count": 0,
        }, True

    selected = tuple(
        tuple(int(value) for value in np.unique(axis_indices))
        for axis_indices in indices
    )
    cartesian_count = math.prod(len(rows) for rows in selected)
    exact, metadata = selected_scaled_integer_bernstein(power, selected)
    positions = [
        {value: position for position, value in enumerate(rows)} for rows in selected
    ]
    values = []
    for coordinate in zip(*indices, strict=True):
        selected_coordinate = tuple(
            positions[axis][int(value)] for axis, value in enumerate(coordinate)
        )
        values.append(int(exact[selected_coordinate]))
    negative = sum(value < 0 for value in values)
    zero = sum(value == 0 for value in values)
    positive = sum(value > 0 for value in values)
    return {
        "count": len(values),
        "selected_rows": [list(rows) for rows in selected],
        "cartesian_control_count": cartesian_count,
        "negative_count": negative,
        "zero_count": zero,
        "positive_count": positive,
        "integer_replay": metadata,
    }, negative == 0


def certify_leaf(
    base_power: np.ndarray,
    path: tuple[tuple[int, int, bool], ...],
) -> dict[str, object]:
    """Certify one chart leaf and release its large arrays before returning."""

    power = base_power
    for first_axis, second_axis, upper in path:
        power = triangle_blowup_power_tensor(
            power, first_axis, second_axis, upper=upper
        )

    centre, radius, possible_nonzero = float_bernstein_enclosure(power)
    lower = centre - radius
    structural_zero = ~possible_nonzero
    unresolved = possible_nonzero & (lower <= 0)
    certified_positive = possible_nonzero & (lower > 0)
    fallback, fallback_passed = _exact_fallback(power, unresolved)

    row = {
        "path": [
            {
                "first_axis": first_axis,
                "second_axis": second_axis,
                "triangle": "upper" if upper else "lower",
            }
            for first_axis, second_axis, upper in path
        ],
        "multidegree": [size - 1 for size in power.shape],
        "control_count": int(power.size),
        "structural_zero_count": int(np.count_nonzero(structural_zero)),
        "interval_positive_count": int(np.count_nonzero(certified_positive)),
        "minimum_interval_lower_bound": float(lower[certified_positive].min()),
        "maximum_interval_radius": float(radius.max()),
        "exact_fallback": fallback,
        "passed": bool(
            fallback_passed
            and np.count_nonzero(structural_zero)
            + np.count_nonzero(certified_positive)
            + fallback["count"]
            == power.size
        ),
    }
    del power, centre, radius, possible_nonzero, lower, structural_zero
    del unresolved, certified_positive
    gc.collect()
    return row


def run(
    coefficients: Path,
    output: Path,
    *,
    workers: int = 6,
) -> dict[str, object]:
    resource = constrain_current_process(workers=workers)
    target, amplitudes, rationalization = rationalized_power_tensors(coefficients)
    margins = []
    for channel in range(4):
        for odd_sign in (1, -1):
            base = orientation_power_tensor(
                target, amplitudes, channel, odd_sign
            )
            leaves = [
                certify_leaf(base, path)
                for path in _leaf_paths(ATLAS_TREES[(channel, odd_sign)])
            ]
            margins.append(
                {
                    "channel": channel,
                    "odd_sign": odd_sign,
                    "leaf_count": len(leaves),
                    "maximum_depth": max(len(row["path"]) for row in leaves),
                    "leaves": leaves,
                    "passed": all(row["passed"] for row in leaves),
                }
            )

    report = {
        "experiment": "certified finite two-edge triangular Bernstein atlas",
        "coefficient_artifact": str(coefficients),
        "coefficient_artifact_sha256": _sha256(coefficients),
        "rationalization": rationalization,
        "margin_count": len(margins),
        "leaf_count": sum(row["leaf_count"] for row in margins),
        "margins": margins,
        "certificate_layers": [
            "exact integer rational-circle numerator construction",
            "complete lower/upper triangular cover tree",
            "outward IEEE-754 Bernstein error enclosure",
            "exact selected-row integer replay of unresolved controls",
        ],
        "scope_boundary": (
            "A pass proves nonnegativity of all eight orientation margins on "
            "the frozen h=0 finite two-edge family. It does not cover the final "
            "Cholesky residual or other five-query allocations."
        ),
        "resource_contract": resource,
        "passed": all(row["passed"] for row in margins),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def verify_report(report: dict[str, object], coefficients: Path) -> bool:
    """Check artifact integrity and cover-tree bookkeeping.

    This lightweight verifier is deliberately not described as a proof replay:
    recomputing every enclosure requires running :func:`run`.  It does ensure
    that a stored report refers to the current coefficient artifact, contains
    exactly the frozen cover leaves, and accounts for every control at each
    leaf without a negative exact fallback.
    """

    if report.get("coefficient_artifact_sha256") != _sha256(coefficients):
        return False
    margins = report.get("margins")
    if not isinstance(margins, list) or len(margins) != 8:
        return False
    by_key = {(row.get("channel"), row.get("odd_sign")): row for row in margins}
    if set(by_key) != set(ATLAS_TREES):
        return False
    for key, tree in ATLAS_TREES.items():
        margin = by_key[key]
        expected_paths = {
            tuple(path) for path in _leaf_paths(tree)
        }
        actual_paths = set()
        for leaf in margin.get("leaves", []):
            path = tuple(
                (
                    int(step["first_axis"]),
                    int(step["second_axis"]),
                    step["triangle"] == "upper",
                )
                for step in leaf.get("path", [])
            )
            actual_paths.add(path)
            fallback = leaf.get("exact_fallback", {})
            covered = (
                int(leaf.get("structural_zero_count", -1))
                + int(leaf.get("interval_positive_count", -1))
                + int(fallback.get("count", -1))
            )
            if (
                covered != int(leaf.get("control_count", -2))
                or int(fallback.get("negative_count", 1)) != 0
                or not leaf.get("passed")
            ):
                return False
        if actual_paths != expected_paths or not margin.get("passed"):
            return False
    return bool(report.get("passed"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coefficients",
        type=Path,
        default=Path(
            "artifacts/spin8_dirac_two_edge_all_sectors_coefficients_20260806.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/spin8_dirac_two_edge_atlas_20260807.json"),
    )
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    report = run(
        arguments.coefficients,
        arguments.output,
        workers=arguments.workers,
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "passed": report["passed"],
                "leaf_count": report["leaf_count"],
                "margin_passes": [row["passed"] for row in report["margins"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
