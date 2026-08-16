"""Adaptive exact diagonal-dominance atlas on ``[1/8, 7/8]^5``.

This extends the small 32-box central-core theorem.  Thirty-two coarse affine
boxes cover the larger core.  Any box whose exact Bernstein/radical bound does
not prove strict diagonal dominance is split into all 32 five-axis children.
The frozen depth-four atlas closes every branch.

The source harness recomputes every Bernstein transform.  The compact verifier
checks hashes, exact rational gaps, and the complete prefix-tree cover while
explicitly retaining the stored-transform trust boundary.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

from flint import ctx, fmpz

from spin8_dirac_endpoint_octet import H0
from spin8_dirac_endpoint_octet_core_dominance import (
    SQRT_BITS,
    _ceil_sqrt_fraction,
    _endpoint_amplitudes,
    _fraction_text,
)
from spin8_dirac_endpoint_octet_determinant import (
    DEFAULT_COEFFICIENT_DIR,
    _coefficient_hashes,
)
from spin8_dirac_endpoint_octet_quadratic import _atomic_json, _sha256
from spin8_dirac_unrestricted_core import _bernstein_matrix, _transform_axis
from spin8_resource_limits import constrain_current_process

MAX_REFINEMENT_DEPTH = 4
ROOT_DENOMINATOR = 8
ROOT_WIDTH_NUMERATOR = 3


def _path_segments(path: str) -> tuple[str, ...]:
    segments = tuple(path.split("/"))
    if not segments or any(
        len(segment) != 5 or set(segment) - {"0", "1"} for segment in segments
    ):
        raise ValueError("atlas paths must contain slash-separated five-bit rows")
    return segments


def _box_from_path(path: str) -> tuple[tuple[int, int, int], ...]:
    """Return ``(left, width, denominator)`` for every coordinate."""

    segments = _path_segments(path)
    box = tuple(
        (
            1 if bit == "0" else 4,
            ROOT_WIDTH_NUMERATOR,
            ROOT_DENOMINATOR,
        )
        for bit in segments[0]
    )
    for segment in segments[1:]:
        box = tuple(
            (2 * left + int(bit) * width, width, 2 * denominator)
            for (left, width, denominator), bit in zip(box, segment, strict=True)
        )
    return box


def _box_intervals(path: str) -> list[list[str]]:
    return [
        [
            _fraction_text(Fraction(left, denominator)),
            _fraction_text(Fraction(left + width, denominator)),
        ]
        for left, width, denominator in _box_from_path(path)
    ]


def _children(path: str) -> tuple[str, ...]:
    return tuple(
        f"{path}/{''.join(map(str, bits))}"
        for bits in itertools.product((0, 1), repeat=5)
    )


def _affine_axis(polynomial, axis: int, left: int, width: int, denominator: int):
    """Map one variable to ``(left + width*x)/denominator`` exactly."""

    degree = int(polynomial.degrees()[axis])
    coefficients: dict[tuple[int, ...], int] = {}
    for powers, coefficient in polynomial.to_dict().items():
        power = powers[axis]
        base = int(coefficient) * denominator ** (degree - power)
        for target_power in range(power + 1):
            target = list(powers)
            target[axis] = target_power
            key = tuple(target)
            value = (
                base
                * math.comb(power, target_power)
                * left ** (power - target_power)
                * width**target_power
            )
            coefficients[key] = coefficients.get(key, 0) + value
    result = polynomial.context().from_dict(
        {powers: value for powers, value in coefficients.items() if value}
    )
    return result, denominator**degree


def _affine_bernstein_bounds(polynomial, path: str) -> tuple[Fraction, Fraction]:
    """Return exact convex-hull bounds on an atlas box."""

    result = polynomial
    positive_scale = 1
    for axis, (left, width, denominator) in enumerate(_box_from_path(path)):
        result, scale = _affine_axis(result, axis, left, width, denominator)
        positive_scale *= scale
    degrees = tuple(map(int, result.degrees()))
    shape = tuple(degree + 1 for degree in degrees)
    strides = tuple(math.prod(shape[axis + 1 :]) for axis in range(5))
    values = [fmpz(0)] * math.prod(shape)
    for powers, coefficient in result.to_dict().items():
        flat = sum(
            power * stride for power, stride in zip(powers, strides, strict=True)
        )
        values[flat] = fmpz(coefficient)
    for axis, degree in enumerate(degrees):
        matrix, scale = _bernstein_matrix(degree)
        values = _transform_axis(values, axis=axis, shape=shape, matrix=matrix)
        positive_scale *= int(scale)
    return (
        Fraction(int(min(values)), positive_scale),
        Fraction(int(max(values)), positive_scale),
    )


def _box_certificate(path: str, surviving, residuals, forced_squares):
    trivial_low, _trivial_high = _affine_bernstein_bounds(residuals[H0[0]], path)
    nontrivial_sum = Fraction(0)
    sqrt_bounds_verified = True
    for mask in surviving[1:]:
        residual_low, residual_high = _affine_bernstein_bounds(residuals[mask], path)
        _square_low, square_high = _affine_bernstein_bounds(forced_squares[mask], path)
        if square_high < 0:
            raise AssertionError(f"negative forced-square upper bound on {path}")
        radical_upper = _ceil_sqrt_fraction(square_high)
        sqrt_bounds_verified &= radical_upper**2 >= square_high
        nontrivial_sum += max(abs(residual_low), abs(residual_high)) * radical_upper
    gap = trivial_low - nontrivial_sum
    return {
        "path": path,
        "refinement_depth": len(_path_segments(path)) - 1,
        "trivial_amplitude_bernstein_lower": _fraction_text(trivial_low),
        "nontrivial_absolute_sum_upper": _fraction_text(nontrivial_sum),
        "integer_scaled_dominance_gap_lower": _fraction_text(gap),
        "physical_dominance_gap_lower": _fraction_text(gap / 4),
        "all_outward_sqrt_bounds_verified_exactly": sqrt_bounds_verified,
        "strictly_positive": gap > 0,
    }


def _tree_verification(
    leaves: list[dict[str, object]], rejected: list[dict[str, object]]
) -> dict[str, object]:
    leaf_paths = {row["path"] for row in leaves}
    rejected_paths = {row["path"] for row in rejected}
    all_paths = leaf_paths | rejected_paths
    duplicates = len(all_paths) != len(leaves) + len(rejected)
    expected_roots = {
        "".join(map(str, bits)) for bits in itertools.product((0, 1), repeat=5)
    }
    actual_roots = {path for path in all_paths if "/" not in path}
    missing_children = []
    unexpected_parent = []
    for path in all_paths:
        segments = _path_segments(path)
        if len(segments) > 1:
            parent = "/".join(segments[:-1])
            if parent not in rejected_paths:
                unexpected_parent.append(path)
    for path in rejected_paths:
        missing = sorted(set(_children(path)) - all_paths)
        if missing:
            missing_children.extend(missing)
    leaf_prefix_overlap = []
    for path in leaf_paths:
        prefix = f"{path}/"
        if any(other.startswith(prefix) for other in all_paths):
            leaf_prefix_overlap.append(path)
    complete = bool(
        not duplicates
        and actual_roots == expected_roots
        and not missing_children
        and not unexpected_parent
        and not leaf_prefix_overlap
    )
    return {
        "expected_root_count": len(expected_roots),
        "actual_root_count": len(actual_roots),
        "duplicate_path_found": duplicates,
        "missing_child_count": len(missing_children),
        "unexpected_parent_count": len(unexpected_parent),
        "leaf_prefix_overlap_count": len(leaf_prefix_overlap),
        "complete_prefix_tree_cover": complete,
    }


def build_full_certificate(
    coefficient_dir: Path = DEFAULT_COEFFICIENT_DIR,
    *,
    max_depth: int = MAX_REFINEMENT_DEPTH,
):
    """Recompute every exact transform in the adaptive atlas."""

    if max_depth != MAX_REFINEMENT_DEPTH:
        raise ValueError(f"the frozen atlas depth is {MAX_REFINEMENT_DEPTH}")
    surviving, residuals, forced_squares = _endpoint_amplitudes(coefficient_dir)
    active = sorted(
        "".join(map(str, bits)) for bits in itertools.product((0, 1), repeat=5)
    )
    leaves: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    level_rows = []
    unresolved: list[dict[str, object]] = []
    for depth in range(max_depth + 1):
        passed_level = []
        rejected_level = []
        for path in active:
            row = _box_certificate(path, surviving, residuals, forced_squares)
            if row["strictly_positive"]:
                passed_level.append(row)
            elif depth < max_depth:
                rejected_level.append(row)
            else:
                unresolved.append(row)
        leaves.extend(passed_level)
        rejected.extend(rejected_level)
        level_rows.append(
            {
                "refinement_depth": depth,
                "tested_box_count": len(active),
                "certified_leaf_count": len(passed_level),
                "rejected_basis_box_count": len(rejected_level),
                "unresolved_box_count": len(unresolved) if depth == max_depth else 0,
            }
        )
        if unresolved or not rejected_level:
            break
        active = [child for row in rejected_level for child in _children(row["path"])]

    tree = _tree_verification(leaves, rejected)
    minimum = min(
        leaves,
        key=lambda row: Fraction(row["integer_scaled_dominance_gap_lower"]),
    )
    passed = bool(
        not unresolved
        and len(leaves) == 2140
        and len(rejected) == 68
        and tree["complete_prefix_tree_cover"]
        and all(row["strictly_positive"] for row in leaves)
        and all(row["all_outward_sqrt_bounds_verified_exactly"] for row in leaves)
    )
    return {
        "experiment": "adjacent endpoint-octet extended-core dominance atlas",
        "evidence_class": "exact adaptive Bernstein and outward-radical certificate",
        "domain": "(ud,ue,ug,ui,y) in [1/8,7/8]^5",
        "coefficient_source_sha256": _coefficient_hashes(coefficient_dir),
        "residual_common_integer_scale": 4,
        "outward_square_root_precision_bits": SQRT_BITS,
        "maximum_refinement_depth": max_depth,
        "level_summary": level_rows,
        "certified_leaves": leaves,
        "rejected_basis_nodes": rejected,
        "certified_leaf_count": len(leaves),
        "rejected_basis_node_count": len(rejected),
        "unresolved_boxes": unresolved,
        "tree_verification": tree,
        "minimum_gap_path": minimum["path"],
        "minimum_integer_scaled_gap_lower": minimum[
            "integer_scaled_dominance_gap_lower"
        ],
        "minimum_physical_gap_lower": minimum["physical_dominance_gap_lower"],
        "strict_diagonal_dominance_proved": passed,
        "all_eight_physical_margins_strictly_positive": passed,
        "schur_matrix_positive_definite_on_extended_core": passed,
        "determinant_strictly_positive_on_extended_core": passed,
        "complete_adjacent_octet_proved": False,
        "global_dirac_gram_theorem_proved": False,
        "passed": passed,
        "compact_trust_boundary": (
            "The compact verifier checks source hashes, exact stored gaps, and "
            "the complete prefix-tree cover. Recomputing every Bernstein and "
            "outward-radical bound requires this source harness."
        ),
        "scope_boundary": (
            "The theorem covers exactly [1/8,7/8]^5. It does not cover the "
            "remaining boundary collars, complete adjacent octet, or "
            "unrestricted seven-variable Dirac--Gram domain."
        ),
        "next_exact_gate": (
            "Extend the atlas into the width-1/8 coordinate collars and "
            "delegate equality-corner branches to the order-eight nested blow-up."
        ),
    }


def verify_report(
    report_or_path, *, coefficient_dir: Path = DEFAULT_COEFFICIENT_DIR
) -> dict[str, object]:
    """Compactly verify hashes, exact gaps, nonclaims, and tree coverage."""

    if isinstance(report_or_path, (str, Path)):
        report = json.loads(Path(report_or_path).read_text(encoding="utf-8"))
    else:
        report = report_or_path
    failures: list[str] = []
    if report.get("coefficient_source_sha256") != _coefficient_hashes(coefficient_dir):
        failures.append("coefficient-source hashes disagree")
    leaves = report.get("certified_leaves", [])
    rejected = report.get("rejected_basis_nodes", [])
    tree = _tree_verification(leaves, rejected)
    if (
        tree != report.get("tree_verification")
        or not tree["complete_prefix_tree_cover"]
    ):
        failures.append("stored adaptive tree is not a complete exact cover")
    if len(leaves) != 2140 or report.get("certified_leaf_count") != len(leaves):
        failures.append("certified leaf count is not 2140")
    if len(rejected) != 68 or report.get("rejected_basis_node_count") != len(rejected):
        failures.append("rejected basis-node count is not 68")
    if report.get("unresolved_boxes") != []:
        failures.append("stored atlas contains unresolved boxes")
    if any(
        Fraction(row["integer_scaled_dominance_gap_lower"]) <= 0
        or Fraction(row["physical_dominance_gap_lower"])
        != Fraction(row["integer_scaled_dominance_gap_lower"]) / 4
        or row.get("all_outward_sqrt_bounds_verified_exactly") is not True
        or row.get("strictly_positive") is not True
        for row in leaves
    ):
        failures.append("a certified leaf has an invalid exact gap or radical flag")
    if any(
        Fraction(row["integer_scaled_dominance_gap_lower"]) > 0
        or row.get("strictly_positive") is not False
        for row in rejected
    ):
        failures.append("a refined parent was not rejected by its stored bound")
    if leaves:
        minimum = min(
            leaves,
            key=lambda row: Fraction(row["integer_scaled_dominance_gap_lower"]),
        )
        if (
            report.get("minimum_gap_path") != minimum["path"]
            or Fraction(report.get("minimum_integer_scaled_gap_lower", "0"))
            != Fraction(minimum["integer_scaled_dominance_gap_lower"])
            or Fraction(report.get("minimum_physical_gap_lower", "0"))
            != Fraction(minimum["physical_dominance_gap_lower"])
        ):
            failures.append("stored minimum exact gap disagrees with leaf table")
    for claim in (
        "strict_diagonal_dominance_proved",
        "all_eight_physical_margins_strictly_positive",
        "schur_matrix_positive_definite_on_extended_core",
        "determinant_strictly_positive_on_extended_core",
        "passed",
    ):
        if report.get(claim) is not True:
            failures.append(f"stored report does not establish {claim}")
    for nonclaim in (
        "complete_adjacent_octet_proved",
        "global_dirac_gram_theorem_proved",
    ):
        if report.get(nonclaim) is not False:
            failures.append(f"stored report overclaims {nonclaim}")
    return {
        "verified": not failures,
        "failures": failures,
        "tree_verification": tree,
        "certified_leaf_count": len(leaves),
        "rejected_basis_node_count": len(rejected),
        "minimum_gap_path": report.get("minimum_gap_path"),
        "minimum_physical_gap_lower": report.get("minimum_physical_gap_lower"),
        "full_transform_replay_required_for_independent_sign_check": True,
    }


def run(
    coefficient_dir: Path,
    *,
    output: Path,
    flint_threads: int = 6,
) -> dict[str, object]:
    if not 1 <= flint_threads <= 6:
        raise ValueError("FLINT thread count must be between one and six")
    resource = constrain_current_process(workers=flint_threads)
    ctx.threads = flint_threads
    report = build_full_certificate(coefficient_dir)
    report["resource_contract"] = resource
    _atomic_json(output, report)
    report["artifact_sha256"] = _sha256(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coefficient-dir", type=Path, default=DEFAULT_COEFFICIENT_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flint-threads", type=int, default=6)
    arguments = parser.parse_args()
    report = run(
        arguments.coefficient_dir,
        output=arguments.output,
        flint_threads=arguments.flint_threads,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
