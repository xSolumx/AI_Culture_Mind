"""Exact multi-slice degree audit for the variable-Cayley two-edge bridge.

This is a reconnaissance certificate, not a full tensor reconstruction.  The
rank-seven Cauchy--Binet bound permits degree at most 14 after squaring a Walsh
sector and removing its forced odd-coordinate factors.  Fifteen discovery
nodes reconstruct each univariate slice; four disjoint nodes then test it.
Three unrelated base points are used so a leading coefficient vanishing on one
slice cannot silently become the global degree claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import sympy as sp

from spin8_dirac_edge import _character
from spin8_dirac_star import rational_circle
from spin8_dirac_two_edge import (
    PARAMETER_NAMES,
    SIGNS,
    exact_normalized_determinant,
    exact_sign_symmetry_certificate,
)

MAX_SQUARED_SECTOR_DEGREE = 14
BASE_POINTS = (
    tuple(
        sp.Rational(k, d) for k, d in zip((1, 2, 3, 2, 3, 4), (7, 9, 11, 13, 14, 15))
    ),
    tuple(
        sp.Rational(k, d) for k, d in zip((2, 1, 4, 3, 5, 2), (7, 9, 11, 13, 14, 15))
    ),
    tuple(
        sp.Rational(k, d) for k, d in zip((3, 4, 2, 5, 1, 6), (10, 11, 13, 17, 9, 19))
    ),
)
DISCOVERY_NODES = tuple(sp.Rational(index, 19) for index in range(1, 16))
CONFIRMATION_NODES = (
    sp.Rational(1, 23),
    sp.Rational(5, 23),
    sp.Rational(11, 23),
    sp.Rational(17, 23),
)


def _sector_setup():
    symmetry = exact_sign_symmetry_certificate()
    masks = tuple(tuple(row) for row in symmetry["walsh_annihilator"])
    representatives = {}
    for signs in SIGNS:
        pattern = tuple(_character(signs, mask) for mask in masks)
        representatives.setdefault(pattern, signs)
    if len(representatives) != len(masks):
        raise AssertionError("sign quotient did not produce eight characters")
    patterns = sorted(representatives)
    rows = tuple(representatives[pattern] for pattern in patterns)
    hadamard = sp.Matrix(
        [[_character(signs, mask) for mask in masks] for signs in rows]
    )
    if hadamard.T * hadamard != len(masks) * sp.eye(len(masks)):
        raise AssertionError("quotient character table is not Hadamard")
    return masks, rows, hadamard.T / len(masks)


def _worker(job):
    point_index, parameters, signs = job
    return point_index, signs, str(exact_normalized_determinant(parameters, signs))


def _sector_measure(
    sector: sp.Expr,
    mask: tuple[int, ...],
    parameters: tuple[sp.Rational, ...],
) -> sp.Expr:
    if not any(mask):
        return sector
    squared_coordinates = tuple(rational_circle(value)[0] ** 2 for value in parameters)
    forced = sp.prod(
        squared_coordinates[index] for index, bit in enumerate(mask) if bit
    )
    return sp.factor(sector**2 / forced)


def _root_multiplicity(polynomial: sp.Poly, root: int) -> int:
    variable = polynomial.gens[0]
    factor = sp.Poly(variable - root, variable)
    current = polynomial
    multiplicity = 0
    while current.eval(root) == 0:
        current = current.exquo(factor)
        multiplicity += 1
    return multiplicity


def _coefficient_digest(polynomial: sp.Poly) -> str:
    payload = "\n".join(
        f"{powers[0]}:{coefficient}" for powers, coefficient in polynomial.terms()
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _rational_square_root(value: sp.Rational) -> sp.Rational | None:
    if value < 0:
        return None
    numerator_root = math.isqrt(int(value.p))
    denominator_root = math.isqrt(int(value.q))
    if numerator_root**2 != value.p or denominator_root**2 != value.q:
        return None
    return sp.Rational(numerator_root, denominator_root)


def _residual_square_certificate(
    polynomial: sp.Poly, multiplicity_at_one: int
) -> tuple[bool, int | None, str | None]:
    """Check whether removing ``(1-u)^k`` leaves an exact rational square."""

    variable = polynomial.gens[0]
    residual = sp.Poly(
        sp.cancel(polynomial.as_expr() / (1 - variable) ** multiplicity_at_one),
        variable,
    )
    coefficient, factors = sp.factor_list(residual.as_expr(), variable)
    coefficient_root = _rational_square_root(sp.Rational(coefficient))
    if coefficient_root is None or any(exponent % 2 for _, exponent in factors):
        return False, None, None
    root = coefficient_root * sp.prod(
        factor ** (exponent // 2) for factor, exponent in factors
    )
    root_polynomial = sp.Poly(sp.expand(root), variable)
    if sp.Poly(root_polynomial.as_expr() ** 2, variable) != residual:
        return False, None, None
    return True, int(root_polynomial.degree()), _coefficient_digest(root_polynomial)


def _rows_digest(rows: list[dict[str, object]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _degree_atlas_from_rows(
    rows: list[dict[str, object]], masks: tuple[tuple[int, ...], ...]
) -> list[dict[str, object]]:
    atlas = []
    for axis_name in PARAMETER_NAMES:
        for mask in masks:
            relevant = [
                row
                for row in rows
                if row["axis"] == axis_name and tuple(row["mask"]) == mask
            ]
            if len(relevant) != len(BASE_POINTS):
                raise ValueError("incomplete base-slice family")
            atlas.append(
                {
                    "axis": axis_name,
                    "mask": list(mask),
                    "maximum_observed_degree": max(row["degree"] for row in relevant),
                    "minimum_zero_multiplicity": min(
                        row["multiplicity_at_zero"] for row in relevant
                    ),
                    "minimum_one_multiplicity": min(
                        row["multiplicity_at_one"] for row in relevant
                    ),
                    "base_slice_degrees": [row["degree"] for row in relevant],
                }
            )
    return atlas


def exact_degree_audit(*, workers: int = 1) -> dict[str, object]:
    masks, representative_signs, inverse_character = _sector_setup()
    points = []
    point_lookup = {}
    for base_index, base in enumerate(BASE_POINTS):
        for axis in range(len(PARAMETER_NAMES)):
            for node_set, nodes in (
                ("discovery", DISCOVERY_NODES),
                ("confirmation", CONFIRMATION_NODES),
            ):
                for node_index, node in enumerate(nodes):
                    parameters = list(base)
                    parameters[axis] = node
                    key = (base_index, axis, node_set, node_index)
                    point_lookup[key] = len(points)
                    points.append(tuple(parameters))

    jobs = [
        (point_index, parameters, signs)
        for point_index, parameters in enumerate(points)
        for signs in representative_signs
    ]
    if workers == 1:
        evaluated = list(map(_worker, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            evaluated = list(pool.map(_worker, jobs, chunksize=4))
    direct = {
        (point_index, signs): sp.Rational(value)
        for point_index, signs, value in evaluated
    }

    sectors = {}
    for point_index in range(len(points)):
        values = sp.Matrix(
            [direct[point_index, signs] for signs in representative_signs]
        )
        coefficients = inverse_character * values
        sectors[point_index] = tuple(sp.factor(value) for value in coefficients)

    variable = sp.symbols("u")
    slice_rows = []
    for base_index, _base in enumerate(BASE_POINTS):
        for axis, axis_name in enumerate(PARAMETER_NAMES):
            discovery_coordinates = tuple(
                rational_circle(node)[0] ** 2 for node in DISCOVERY_NODES
            )
            confirmation_coordinates = tuple(
                rational_circle(node)[0] ** 2 for node in CONFIRMATION_NODES
            )
            for sector_index, mask in enumerate(masks):
                discovery_values = []
                for node_index in range(len(DISCOVERY_NODES)):
                    point_index = point_lookup[
                        (base_index, axis, "discovery", node_index)
                    ]
                    discovery_values.append(
                        _sector_measure(
                            sectors[point_index][sector_index],
                            mask,
                            points[point_index],
                        )
                    )
                expression = sp.interpolate(
                    list(zip(discovery_coordinates, discovery_values, strict=True)),
                    variable,
                )
                polynomial = sp.Poly(sp.factor(expression), variable)
                multiplicity_at_one = _root_multiplicity(polynomial, 1)
                square_passed, square_root_degree, square_root_digest = (
                    _residual_square_certificate(polynomial, multiplicity_at_one)
                    if any(mask)
                    else (False, None, None)
                )
                confirmations = []
                for node_index, coordinate in enumerate(confirmation_coordinates):
                    point_index = point_lookup[
                        (base_index, axis, "confirmation", node_index)
                    ]
                    observed = _sector_measure(
                        sectors[point_index][sector_index], mask, points[point_index]
                    )
                    confirmations.append(polynomial.eval(coordinate) == observed)
                slice_rows.append(
                    {
                        "base_index": base_index,
                        "axis": axis_name,
                        "mask": list(mask),
                        "measure": (
                            "sector" if not any(mask) else "sector^2/forced_odd_square"
                        ),
                        "degree": int(polynomial.degree()),
                        "multiplicity_at_zero": _root_multiplicity(polynomial, 0),
                        "multiplicity_at_one": multiplicity_at_one,
                        "coefficient_count": len(polynomial.terms()),
                        "coefficients_sha256": _coefficient_digest(polynomial),
                        "post_degree_discovery_residual_is_exact_square": (
                            square_passed
                        ),
                        "residual_square_root_degree": square_root_degree,
                        "residual_square_root_coefficients_sha256": (
                            square_root_digest
                        ),
                        "confirmation_nodes_passed": sum(confirmations),
                        "confirmation_nodes_total": len(confirmations),
                    }
                )

    degree_atlas = _degree_atlas_from_rows(slice_rows, masks)

    passed = len(points) == len(BASE_POINTS) * len(PARAMETER_NAMES) * (
        len(DISCOVERY_NODES) + len(CONFIRMATION_NODES)
    ) and all(
        row["degree"] <= MAX_SQUARED_SECTOR_DEGREE
        and row["confirmation_nodes_passed"] == row["confirmation_nodes_total"]
        for row in slice_rows
    )
    return {
        "experiment": "variable-Cayley two-edge exact multi-slice degree audit",
        "structural_degree_bound": {
            "maximum_squared_sector_degree_per_squared_coordinate": (
                MAX_SQUARED_SECTOR_DEGREE
            ),
            "reason": (
                "A query contributes a rank-seven projector, so Cauchy-Binet "
                "uses its coordinate pair at most seven times; squaring and "
                "passing to the squared coordinate gives degree at most 14."
            ),
        },
        "base_points": [[str(value) for value in row] for row in BASE_POINTS],
        "discovery_nodes": [str(value) for value in DISCOVERY_NODES],
        "confirmation_nodes": [str(value) for value in CONFIRMATION_NODES],
        "quotient_representative_signs": [list(row) for row in representative_signs],
        "exact_determinant_count": len(jobs),
        "slice_count": len(slice_rows),
        "slice_rows_sha256": _rows_digest(slice_rows),
        "slice_rows": slice_rows,
        "degree_atlas": degree_atlas,
        "post_degree_square_discovery": {
            "status": "exact on every audited slice; global factorization open",
            "nontrivial_slice_count": sum(any(row["mask"]) for row in slice_rows),
            "exact_square_slice_count": sum(
                bool(row["post_degree_discovery_residual_is_exact_square"])
                for row in slice_rows
            ),
            "all_nontrivial_residuals_are_exact_squares": all(
                row["post_degree_discovery_residual_is_exact_square"]
                for row in slice_rows
                if any(row["mask"])
            ),
        },
        "interpretation": (
            "These are exact multi-slice degrees under a structural upper bound. "
            "They freeze conservative full-grid degrees only after radical "
            "amplitudes and cross-sector products are derived."
        ),
        "passed": passed,
    }


def verify_degree_report(report: dict[str, object]) -> bool:
    """Replay the report's integrity and acceptance predicate without determinants.

    This deliberately does not claim to replay the 2,736 exact determinant
    evaluations. Full numerical evidence is replayed by regenerating the report.
    """

    rows = report.get("slice_rows")
    if not isinstance(rows, list) or report.get("slice_rows_sha256") != _rows_digest(
        rows
    ):
        return False
    masks = tuple(
        tuple(row) for row in exact_sign_symmetry_certificate()["walsh_annihilator"]
    )
    expected_keys = {
        (base_index, axis, mask)
        for base_index in range(len(BASE_POINTS))
        for axis in PARAMETER_NAMES
        for mask in masks
    }
    observed_keys = {
        (int(row["base_index"]), row["axis"], tuple(row["mask"])) for row in rows
    }
    if observed_keys != expected_keys or len(rows) != len(expected_keys):
        return False
    try:
        atlas = _degree_atlas_from_rows(rows, masks)
    except (KeyError, TypeError, ValueError):
        return False
    accepted = (
        report.get("base_points")
        == [[str(value) for value in row] for row in BASE_POINTS]
        and report.get("discovery_nodes") == [str(value) for value in DISCOVERY_NODES]
        and report.get("confirmation_nodes")
        == [str(value) for value in CONFIRMATION_NODES]
        and report.get("exact_determinant_count")
        == len(BASE_POINTS)
        * len(PARAMETER_NAMES)
        * (len(DISCOVERY_NODES) + len(CONFIRMATION_NODES))
        * len(masks)
        and report.get("slice_count") == len(expected_keys)
        and report.get("degree_atlas") == atlas
        and report.get("post_degree_square_discovery")
        == {
            "status": "exact on every audited slice; global factorization open",
            "nontrivial_slice_count": sum(any(row["mask"]) for row in rows),
            "exact_square_slice_count": sum(
                bool(row["post_degree_discovery_residual_is_exact_square"])
                for row in rows
            ),
            "all_nontrivial_residuals_are_exact_squares": all(
                row["post_degree_discovery_residual_is_exact_square"]
                for row in rows
                if any(row["mask"])
            ),
        }
        and all(
            int(row["degree"]) <= MAX_SQUARED_SECTOR_DEGREE
            and int(row["confirmation_nodes_passed"])
            == int(row["confirmation_nodes_total"])
            == len(CONFIRMATION_NODES)
            for row in rows
        )
    )
    return bool(accepted and report.get("passed"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()
    report = exact_degree_audit(workers=arguments.workers)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "exact_determinant_count": report["exact_determinant_count"],
                "slice_count": report["slice_count"],
                "degree_atlas": report["degree_atlas"],
                "passed": report["passed"],
            },
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit("two-edge exact degree audit failed")


if __name__ == "__main__":
    main()
